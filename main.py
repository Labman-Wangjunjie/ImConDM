import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="test.yml", help="Path to the config file (default: test.yml)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="./onlyDDIM",
        help="A string for documentation purpose. Will be the name of the log folder. (default: ./onlyDDIM)"
    )

    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='SMD',
        help='data choice'
    )
    parser.add_argument(
        "--timesteps", type=int, default=50, help="number of steps in test,PSM 25, SMAP 25, MSL 50, WADI 50"
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--sequence",
        dest="sequence",
        action="store_true",
        help="Enable sequence mode (default: True)",
    )
    parser.add_argument(
        "--no-sequence",
        dest="sequence",
        action="store_false",
        help="Disable sequence mode",
    )
    parser.set_defaults(sequence=True)














    # basic config
    parser.add_argument('--ii', type=int, default=0)
    parser.add_argument('--use_window_normalization', type=bool, default=True)

    parser.add_argument('--stage_mode', type=str, default="TWO", help="ONE, TWO")
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--out_figures', type=int, default=1)
    parser.add_argument('--vis_ar_part', type=int, default=0, help='status')
    parser.add_argument('--vis_MTS_analysis', type=int, default=1, help='status')

    parser.add_argument('--model', type=str, default='DDPM',
                        help='model name, options: [DDPM]')

    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='train epochs')

    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--beta_dist_alpha', type=float, default=-1)  # -1
    parser.add_argument('--our_ddpm_clip', type=float, default=100)  # 100

    # data loader
    parser.add_argument('--seq_len', type=int, default=192, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=7, help='start token length')
    parser.add_argument('--pred_len', type=int, default=14, help='prediction sequence length')

    parser.add_argument('--dataset_name', type=str, default='Exchange')
    parser.add_argument('--weather_type', type=str, default='mintemp', help="['rain' 'mintemp' 'maxtemp' 'solar']")

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_vars', type=int, default=8, help='encoder input size')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Diffusion Models
    parser.add_argument('--interval', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
    parser.add_argument("--beta-max", type=float, default=0.3, help="max diffusion for the diffusion model")
    parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
    parser.add_argument("--T", type=float, default=1., help="sigma end time in network parametrization")
    parser.add_argument('--model_channels', type=int, default=256)  # 256
    parser.add_argument('--nfe', type=int, default=100)
    parser.add_argument('--dim_LSTM', type=int, default=64)

    parser.add_argument('--diff_steps', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--UNet_Type', type=str, default='CNN', help=['CNN'])
    parser.add_argument('--D3PM_kernel_size', type=int, default=5)
    parser.add_argument('--use_freq_enhance', type=int, default=0)
    parser.add_argument('--type_sampler', type=str, default='none', help=["none", "dpm"])
    parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])

    parser.add_argument('--ddpm_inp_embed', type=int, default=256)  # 256
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)  # 256
    parser.add_argument('--ddpm_channels_conv', type=int, default=256)  # 256
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)  # 256
    parser.add_argument('--ddpm_layers_inp', type=int, default=5)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=5)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

    parser.add_argument('--ablation_study_case', type=str, default="none",
                        help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")
    parser.add_argument('--weight_pred_loss', type=float, default=0.0)
    parser.add_argument('--ablation_study_F_type', type=str, default="CNN", help="Linear, CNN")
    parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
    parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)

    # forecasting task

    parser.add_argument('--learning_rate', type=float, default=0.0006, help='optimizer learning rate')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=10, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=64, help='32 batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=64, help='32 batch size of train input data')  # 32

    # parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--tag', type=str, default='')









    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)


    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    SMD_data_set_number = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]
    SMD_data_set_number += ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9"]
    SMD_data_set_number += ["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    f1 = []
    pre = []
    re = []



    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        # elif args.dataset == 'SMD':
        #     for number in SMD_data_set_number:
        #         args.SMD_number = number
        #         f1_, pre_, re_ = runner.complete()
        #         f1.append(f1_)
        #         pre.append(pre_)
        #         re.append(re_)
        #     average_f1 = sum(f1) / len(f1)
        #     average_pre = sum(pre) / len(pre)
        #     average_re = sum(re) / len(re)
        #     print(f"f1 分数：{average_f1:.4f},pre 分数：{average_pre:.4f},re 分数：{average_re:.4f}")
        else:
            print("1111111")
            runner.complete()
    except Exception:
        logging.error(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
