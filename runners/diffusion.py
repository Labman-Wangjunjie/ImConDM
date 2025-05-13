import os
import logging
import time
import glob
from Unit.utils import get_from_one,metrics_calculate
import numpy as np
import tqdm
import torch
import datetime

import torch.utils.data as data
from torch.utils.data import DataLoader
from models.LSTMAE import LSTMAE
from models.diffusion import Model, CNN_DiffusionUnet
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry,noise_estimation_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import pandas as pd
from sklearn.preprocessing import StandardScaler


import torchvision.utils as tvu
import argparse


from early_stopping import EarlyStopping

# early_stopping = EarlyStopping('./earlysave')



parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')#0.001
# parser.add_argument('--input-size', type=int, default=25, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=5, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--datapath',default='./data/PSM/PSM/train.npy',help='datapath')
parser.add_argument('--data',default="PSM",help='data')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args2 = parser.parse_args(args=[])

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        #dataset, test_dataset = get_dataset(args, config)

        dataset = np.loadtxt('./ServerMachineDataset/train/machine-1-1.txt',delimiter=',')
        windowsize = 64
        stride = 1

        dataset = get_from_one(dataset,window_size=windowsize,stride=stride)
        dataset = torch.randn(128,64,38)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        #model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        #是否加载模型
        if self.args.resume_training:

            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        datafirst = time.time()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            print(epoch)
            data_start = time.time()
            data_time = 0
            for i, x  in enumerate(train_loader):

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.type(torch.FloatTensor)
                x = x.to(self.device)
                #x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                x,loss = noise_estimation_loss(model, x, t, e, b)
                loss = loss /1000


                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
        datalast = time.time()
        print(datafirst-datalast)

    def get_th_values_for_SMD(self, SMD_number):
        SMD_data_set_number = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]
        SMD_data_set_number += ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9"]
        SMD_data_set_number += ["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]


        th_mapping = {
            "1-1": (8.503830909729004 , 44.26939010620117 ),
            "1-2": (11.281362533569336 , 31.138912200927734 ),
            "1-3": (12.446269989013672 , 15.8826322555542),
            "1-4": (13.638179779052734 , 15.16768741607666 ),
            "1-5": (22.045700073242188 , 31.185688018798828 ),
            "1-6": (7.716503620147705 , 42.311649322509766 ),
            "1-7": (7.6291399002075195, 26.341829299926758),
            "1-8": (13.8518648147583 , 21.441537857055664),

            "2-1": (10.363418579101562 , 47.916263580322266 ),
            "2-2": (1.8867558240890503 , 28.26812744140625 ),
            "2-3": (16.09589385986328 , 33.64556884765625),
            "2-4": (10.423513412475586 , 29.5872859954834 ),
            "2-5": (12.042879104614258 , 31.65403175354004 ),
            "2-6": (14.556453704833984 , 28.258737564086914 ),
            "2-7": (13.614714622497559 , 52.30949783325195 ),
            "2-8": (15.7821044921875 , 104.42024993896484 ),
            "2-9": (9.444400787353516 , 40.44288635253906 ),

            "3-1": (15.680384635925293 , 43.08465576171875 ),
            "3-2": (3.997633695602417 , 26.72594451904297 ),
            "3-3": (16.00547981262207 , 23.936519622802734 ),
            "3-4": (10.853405952453613 , 25.020414352416992 ),
            "3-5": (13.972044944763184 , 24.33083152770996 ),
            "3-6": (13.977245330810547 , 34.14533996582031),
            "3-7": (15.321301460266113 , 19.650753021240234 ),
            "3-8": (11.186708450317383 , 20.008277893066406),
            "3-9": (15.984029769897461 , 51.09666442871094 ),
            "3-10": (11.311290740966797 , 19.707237243652344 ),
            "3-11": (17.17762565612793, 69.03707122802734 ),
        }


        values = th_mapping.get(SMD_number)

        if values[0] is None or values[1] is None:
            raise ValueError(f"th1 or th2 value not defined for SMD_number: {SMD_number}")

        return values

    def  complete(self):
        early_stopping = EarlyStopping('./earlysave')
        current_time = datetime.datetime.now()

        # 将时间转换为字符串格式
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # 使用 PyTorch 输出当前时间
        print("当前时间：" + time_string)
        args, config = self.args, self.config

        # 选择数据集
        if (args.dataset == 'SMAP'):
            print('Load SMAP')
            dataset = np.load('./data/SMAP/SMAP/SMAP_train.npy')


            length = int(dataset.shape[0] * 0.9)
            traindata = dataset[:length]
            testdata = dataset[length:]



        elif (args.dataset == 'MSL'):
            print('Load MSL')
            dataset = np.load('./data/MSL/MSL/MSL_train.npy')


            length = int(dataset.shape[0] * 0.90)
            testdata = dataset[length:]
            traindata = dataset[:length]


        elif (args.dataset == 'WADI'):
            print('Load WADI')
            dataset = np.load('./data/WADI/wadi_train.npy')

            length = int(dataset.shape[0] * 0.90)
            testdata = dataset[length:]
            traindata = dataset[:length]



        elif (args.dataset == 'SWAT'):
            print('Load SWAT')
            dataset = np.load('./data/SWAT/SWaT_train.npy')



            length = int(dataset.shape[0] * 0.90)
            testdata = dataset[length:]
            traindata = dataset[:length]

            #
        elif (args.dataset == 'SMD'):

            print('Load SMD')
            SMD_number = args.SMD_number
            dataset = np.loadtxt(f'./data/SMD/train/machine-{SMD_number}.txt', delimiter=',')
            length = int(dataset.shape[0] * 0.80)
            testdata = dataset[length:]
            traindata = dataset[:length]

        elif (args.dataset == 'PSM'):
            print('Load PSM')
            dataset = pd.read_csv('./data/PSM/PSM/train.csv')
            dataset = dataset.values[:, 1:]
            dataset = np.nan_to_num(dataset)

            length = int(dataset.shape[0] * 0.95)
            traindata = dataset[:length]

            label = pd.read_csv('./data/PSM/PSM/test_label.csv')
            label = label.values[:, 1:]
            label = label.astype(None)
            testdata = dataset[length:]

        if args.dataset == "SMD":
            feature_dim = 38
            self.th1, self.th2 = self.get_th_values_for_SMD(args.SMD_number)
        elif args.dataset == "SMAP":
            feature_dim = 25
            self.th1 = 0.02140190452337265
            self.th2 = 22.34332275390625
        elif args.dataset == "PSM":
            feature_dim = 25
            self.th1 = 2.9937024116516113
            self.th2 = 16.7478084564209
        elif args.dataset == "MSL":
            feature_dim = 55
            self.th1 = 1.8141059875488281
            self.th2 = 29.75084114074707
        elif args.dataset == "SWAT":
            feature_dim = 51
            self.th1 = 9.908591270446777
            self.th2 = 19.076608657836914
        elif args.dataset == "WADI":
            feature_dim = 127
            self.th1 = 16.872468948364258
            self.th2 = 27.527860641479492


        self.feature_dim = feature_dim


        # label = torch.tensor(label)

        scaler = StandardScaler()

        traindata = scaler.fit_transform(traindata)
        testdata = scaler.fit_transform(testdata)

        print(traindata.shape)
        print(testdata.shape)
        # lstm 加载


        tb_logger = self.config.tb_logger
        # dataset, test_dataset = get_dataset(args, config)

        windowsize = 64
        stride = 1

        traindata = get_from_one(traindata, window_size=windowsize, stride=stride)
        train_loader = data.DataLoader(
            traindata,
            batch_size=config.training.batch_size,
            shuffle=True,
            # num_workers=0,
            num_workers=0,
            drop_last=True
        )
        test_loader = data.DataLoader(
            testdata,
            batch_size=config.training.batch_size,
            shuffle=True,
            # num_workers=0,
            num_workers=0,
            drop_last=True
        )
        model = CNN_DiffusionUnet(config, args, self.feature_dim)
        # model = Model(config)

        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        # 是否加载模型
        if self.args.resume_training:

            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        datafirst = time.time()
        real_data = torch.Tensor(testdata)

        start_time = time.time()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            # break


            print(epoch)
            data_start = time.time()
            data_time = 0
            for i, x in enumerate(train_loader):

                x = x.to(self.device)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.type(torch.FloatTensor)
                x = x.to(self.device)

                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                #ALL Mask
                #ratio = 0.5
                #mask_1 = np.random.choice([0, 1], size=x.shape, p=[ratio, 1 - ratio])
                #mask_1 = torch.tensor(mask_1)
                #mask_1 = mask_1.to(self.device)
                #mask_2 = 1 - mask_1
                #x1 = x * mask_1
                #x2 = x * mask_2
                #x, loss_1 = noise_estimation_loss(model, x1, t, e, b)
                #x, loss_2 = noise_estimation_loss(model, x2, t, e, b)
                #loss = (loss_1 + loss_2) / 1000

                #ratio = 0.5
                #mask = np.random.choice([0, 1], size=x.shape, p=[ratio, 1 - ratio])
                #mask = torch.tensor(mask)
                #mask = mask.to(self.device)
                #x = x * mask
                #x, loss = noise_estimation_loss(model, x, t, e, b)
                # loss = (loss1 + loss_2) / 1000
                #loss = loss / 1000

                #NO Mask
                x, loss = noise_estimation_loss(model, x, t, e, b)
                loss = loss / 1000


                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    print("I am {} step".format(step))
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            if epoch % 2 == 1:

                re_datas = []
                count = 0
                all_loss = 0
                for tdata in test_loader:
                    count += 1
                    print('validing...')
                    model.eval()
                    with torch.no_grad():
                        tdata = torch.reshape(tdata, (2, 64, self.feature_dim))
                        tdata = tdata.type(torch.FloatTensor)
                        tdata = tdata.to(self.device)

                        z = tdata
                        n = z.size(0)

                        e = torch.randn_like(z)
                        b = self.betas
                        # antithetic sampling
                        t = torch.randint(
                            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)

                        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                        ratio = 0.5
                        mask = np.random.choice([0, 1], size=z.shape, p=[ratio, 1 - ratio])
                        mask = torch.tensor(mask)
                        mask = mask.to(self.device)

                        a1 = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                        x = z * a1.sqrt() + e * (1.0 - a1).sqrt()
                        z_t = x

                        # z_t, loss = noise_estimation_loss(model, z, t, e, b)

                        re_z = self.sample_image(z, z_t, mask, config.diffusion.num_diffusion_timesteps, model, last=True)
                        # data = [inverse_data_transform(config, y) for y in data]
                        re_z = torch.tensor([item.cpu().detach().numpy() for item in re_z])

                        re_z = re_z.to(self.device)

                        tdata = re_z

                        tdata = torch.reshape(tdata, (128, self.feature_dim))
                        re_datas.extend(tdata)

                re_datas = torch.tensor([item.cpu().detach().numpy() for item in re_datas])
                # label = label[:int(len(re_datas))]

                real_data = real_data[:int(len(re_datas))]
                print(real_data.shape)
                print(re_datas.shape)
                f1 = torch.nn.functional.mse_loss(real_data, re_datas)
                # f1 = torch.nn.MSELoss(real_data, re_datas)
                print(f1)

                earlyloss = f1
                print('earlyloss={}'.format(earlyloss))
                if args.dataset == 'SMD' :
                    dataset_name = args.dataset + args.SMD_number
                else:
                    dataset_name = args.dataset
                early_stopping(earlyloss, model, states, 'ddim', dataset_name)

                # 记录训练时间
                save_dir = './TrainLogs_WJJ'

                end_time = time.time()
                duration = end_time - start_time
                with open(os.path.join(save_dir, 'training_time_one_epoch.txt'), 'a') as f:
                    f.write(f"\n{args.dataset} Training time: {duration:.2f} seconds\n")


                if early_stopping.early_stop:
                    print("*******************early stop*********************")
                    break





        datalast = time.time()
        print((datalast - datafirst) / 60)


        #
        save_dir = './TrainLogs_WJJ'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        end_time = time.time()
        duration = end_time - start_time
        with open(os.path.join(save_dir, 'training_time.txt'), 'a') as f:
            f.write(f"\n{args.dataset} Training time: {duration:.2f} seconds\n")

        model = CNN_DiffusionUnet(config, args, self.feature_dim)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                print("1")
                print('hhhhhh')
                if args.dataset == 'SMD' :
                    dataset_name = args.dataset + args.SMD_number
                else:
                    dataset_name = args.dataset
                states = torch.load(
                    f'./earlysave/TEST_{dataset_name}_DMnetwork.pth',
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)
            # model = torch.nn.DataParallel(model)

            model.load_state_dict(states[0])

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        f1 = []
        pre = []
        re = []

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            with open(os.path.join(save_dir, 'training_time.txt'), 'a') as f:
                f.write(f"\n Scale: 4.0\n")
            # ts_list = [[1000],[100,1000],[100,200,500,1000],[100,200,300,400,500,1000],[100,200,300,400,500,600,800,1000],[100,200,300,400,500,600,700,800,900,1000]]
            ts_list = [[config.diffusion.num_diffusion_timesteps]]

            mind_list = [[1]]
            # mind_list = [[1.0],[0.5,0.5],[0.25,0.25,0.25,0.25],[0.167,0.167,0.167,0.167,0.167,0.167],[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
            for tt in range(len(ts_list)):
                f1_, pre_, re_ = self.sample_sequence(model, ts_list[tt], mind_list[tt])
                f1.append(f1_)
                pre.append(pre_)
                re.append(re_)
            average_f1 = sum(f1) / len(f1)
            average_pre = sum(pre) / len(pre)
            average_re = sum(re) / len(re)
            return average_f1, average_pre, average_re

        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample(self):
        config = self.config
        args = self.args
        model = CNN_DiffusionUnet(config, args, self.feature_dim)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                print("1")
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt_1.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)
           # model = torch.nn.DataParallel(model)

            model.load_state_dict(states[0])

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model, ts, mind):
        start_time = time.time()
        args, config = self.args, self.config
        start_time = time.time()

        with torch.no_grad():
            print("here")
           
            print('sequence')
            if (args.dataset == 'SMAP'):
                testdata = np.load('./data/SMAP/SMAP/SMAP_test.npy')


                label = np.load('./data/SMAP/SMAP/SMAP_test_label.npy')





            elif (args.dataset == 'SMD'):

                SMD_number = args.SMD_number

                testdata = np.loadtxt(f'./data/SMD/test/machine-{SMD_number}.txt', delimiter=',')

                label = np.loadtxt(f'./data/SMD/test_label/machine-{SMD_number}.txt', delimiter=',')


            elif (args.dataset == 'WADI'):
                testdata = np.load('./data/WADI/wadi_test.npy')
                scaler = StandardScaler()

                testdata = scaler.fit_transform(testdata)
                label = np.load('./data/WADI/wadi_labels.npy')

            elif (args.dataset == 'MSL'):
                testdata = np.load('./data/MSL/MSL/MSL_test.npy')

                label = np.load('./data/MSL/MSL/MSL_test_label.npy')
            elif (args.dataset == 'SWAT'):
                testdata = np.load('./data/SWAT/SWaT_test.npy')


                label = np.load('./data/SWAT/SWaT_labels.npy').astype(float)
            elif (args.dataset == 'PSM'):
                testdata = pd.read_csv('./data/PSM/PSM/test.csv')
                testdata = testdata.values[:, 1:]
                testdata = np.nan_to_num(testdata)

                label = pd.read_csv('./data/PSM/PSM/test_label.csv')
                label = label.values[:, 1:]


            elif (args.dataset == 'GCP'):

                testdata = np.load('./data/GCP/test.npy')

                label = np.load('./data/GCP/test_label.npy')

            scaler = StandardScaler()
            testdata = scaler.fit_transform(testdata)
            label = label.astype(None)
            label = torch.Tensor(label)

            #testdata = testdata[0:128]
            #label = label[0:128]
            print(testdata.shape)




            # dataloader = DataLoader(
            #     testdata, batch_size=128, shuffle=True, num_workers=4, drop_last=True,
            #     pin_memory=True)
            dataloader = DataLoader(
                testdata, batch_size=config.training.batch_size, shuffle=True, num_workers=0, drop_last=True,
                pin_memory=True)

            # 测试数据

            real_data = torch.Tensor(testdata)
            # 新生成的数据
            masks = []
            re_datas = []
            i = 0
            for tt in range(len(ts)):
                for data in dataloader:


                    data = torch.reshape(data, (config.training.batch_size//64, config.sampling.batch_size, self.feature_dim))
                    data = data.type(torch.FloatTensor)
                    data = data.to(self.device)

                    z = data
                    ratio = 0.5
                    mask = np.random.choice([0, 1], size=z.shape, p=[ratio, 1 - ratio])
                    mask = torch.tensor(mask)
                    mask = mask.to(self.device)

                    #mask_2 = 1-mask
                    #mask_2 = torch.tensor(mask_2)
                    #mask_2 = mask_2.to(self.device)


                    n = z.size(0)
                    print(n)
                    # e = torch.randn_like(z)
                    e = torch.randn_like(z)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=ts[tt], size=(n // 2 + 1,)
                    ).to(self.device)

                    print('there is test')
                    print(t.shape)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    print(t.shape)


                    a1 = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                    x = z * a1.sqrt() + e * (1.0 - a1).sqrt()
                    z_t = x

                    # z_t, loss = noise_estimation_loss(model, z, t, e, b)
                    print(ts[tt])
                    re_z = self.sample_image(z, z_t ,mask,ts[tt],model,last=True)
                    # data = [inverse_data_transform(config, y) for y in data]
                    re_z = torch.tensor([item.cpu().detach().numpy() for item in re_z])

                    re_z = re_z.to(self.device)
                    #re_z2 = self.sample_image(z_t * mask_2, ts[tt], model, last=True)
                    # data = [inverse_data_transform(config, y) for y in data]
                    #re_z2 = torch.tensor([item.cpu().detach().numpy() for item in re_z2])

                    #re_z2 = re_z2.to(self.device)

                    data = re_z
                    print(data.shape)

                    data = torch.reshape(data, (config.training.batch_size, self.feature_dim))
                    mask = torch.reshape(mask, (config.training.batch_size, self.feature_dim))
                    masks.extend(mask)
                    re_datas.extend(data)

            re_datas = torch.tensor([item.cpu().detach().numpy() for item in re_datas])
            masks = torch.tensor([item.cpu().detach().numpy() for item in masks])
            print(len(re_datas))
            print(len(masks))
            label = label[:int(len(re_datas)/len(ts))]
            real_data = real_data[:int(len(re_datas)/len(ts))]
            save_dir = './TrainLogs_WJJ'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            end_time = time.time()
            duration = end_time - start_time
            with open(os.path.join(save_dir, 'training_time.txt'), 'a') as f:
                f.write(f"{args.dataset} Testing time: {duration:.2f} seconds\n")


            metrics_calculate(real_data, re_datas, label,masks, mind, ts, self.th1, self.th2)

        '''x = torch.randn(
            2,
            config.data.channels,
            config.data.image_size,
            device=self.device,
        )
        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.

        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )'''


    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x_0,x,mask,t_1, model, last=True):

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":

                skip = self.num_timesteps // self.args.timesteps


                seq = range(0, t_1, skip)
                #seq = range(0, 100, skip)


            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs= generalized_steps(x_0,x, mask,seq, model, self.betas, eta=self.args.eta)

            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:

            x = x[0][-1]

        return x

    def test(self):
        pass
