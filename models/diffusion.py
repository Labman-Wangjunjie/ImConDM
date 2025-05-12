import math
import torch
import torch.nn as nn
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    # 先将维度分一半
    half_dim = embedding_dim // 2
    # 然后emb等于e为底log10000除以(维度的一半减1)
    emb = math.log(10000) / (half_dim - 1)
    # 生成从0到维度一半减1的序列,然后全部减掉刚刚的emb,最后把这些数全部变为e的指数,算出结果给到emb
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    emb = emb.to(device=timesteps.device)
    # 增加维度，将timesteps和emb分别转换为列向量和行向量，然后进行乘法变为矩阵
    emb = timesteps.float()[:, None] * emb[None, :]
    # 合并sin和cos的emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果维度是奇数进行0填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1
                                        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=1.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=1
                                     )
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1)

    def forward(self, x, temb):
        h = x

        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.reshape(b, c, h)
        q = q.permute(0, 2, 1)  # b,h,c
        k = k.reshape(b, c, h)  # b,c,h
        w_ = torch.bmm(q, k)  # b,h,h   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h)

        h_ = self.proj_out(h_)

        return x + h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # tuple是数据结构，作用是让其中元素不可修改，靠索引访问
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)

        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling

        self.conv_in = nn.Conv1d(out_ch, ch, 1)

        curr_res = resolution

        # 创建一个新的元组，第一个元素是1，后面是ch_mult
        in_ch_mult = (1,) + ch_mult

        self.down = nn.ModuleList()
        block_in = None
        # 遍历元组的每个元素（1，2，2，2），0-3
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            # self.num_res_blocks为2
            for i_block in range(self.num_res_blocks):
                # 构建resnet
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # 构建注意力模块
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            # 把block和attn全部变为down的属性中
            down = nn.Module()
            down.block = block
            down.attn = attn
            # 到最后一个num_resolutions的时候进行，加入一个卷积层
            if i_level != self.num_resolutions - 1:
                # resamp_with_conv：true
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]

                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=1)

    def forward(self, x, t):

        # timestep embedding
        # ch:128
        temb = get_timestep_embedding(t, self.ch)
        # 128*128

        # self.temb.dense是两线性层，第一层是输入128维度输出128*4
        temb = self.temb.dense[0](temb)
        # 让temb*sigmoid（temb）即加入激活函数
        temb = nonlinearity(temb)
        # 第二层是输入128*4维度输出128*4
        temb = self.temb.dense[1](temb)

        # downsampling
        # 构造卷积层，输入数据x，得到hs
        hs = [self.conv_in(x)]

        # 用卷积的数据和时间步嵌入数据进行
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # self.down，有八个注意力模块和resnet残差块，以及一个卷积，将数据和时间步嵌入信息给他得到h，最后添加到hs
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # h在经过一个resnet一个注意力和另一个resnet
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):

                hspop = hs.pop()

                h = self.up[i_level].block[i_block](
                    torch.cat([h, hspop], dim=1), temb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class CNN_DiffusionUnet(nn.Module):

    def __init__(self, config ,args, num_vars, seq_len = 0, pred_len= 0, diff_steps= 0):
        super(CNN_DiffusionUnet, self).__init__()

        self.args = args

        self.num_vars = num_vars  # 变量数
        self.ori_seq_len = 64  # 原始序列长度
        # self.seq_len = seq_len #
        # self.label_len = args.label_len #
        self.pred_len = 64  # 预测的长度

        diff_steps = config.diffusion.num_diffusion_timesteps



        kernel_size = 3
        padding = 1
        # self.channels = 128  # args.ddpm_inp_embed
        self.channels = 64
        if self.args.features in ['MS']:
            self.input_projection = InputConvNetwork(args, 1, self.channels)
        else:
            self.input_projection = InputConvNetwork(args, self.num_vars, self.channels,
                                                     num_layers=args.ddpm_layers_inp)

        self.dim_diff_steps = 100  # args.ddpm_dim_diff_steps

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=diff_steps,
            embedding_dim=self.dim_diff_steps,
        )

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        kernel_size = 3
        padding = 1
        # self.dim_intermediate_enc = 128  # args.ddpm_channels_fusion_I
        self.dim_intermediate_enc = 256

        self.enc_conv = InputConvNetwork(args, self.channels + self.dim_diff_steps, self.dim_intermediate_enc,
                                         num_layers=args.ddpm_layers_I)

        self.cond_projections = nn.ModuleList()

        if args.ablation_study_F_type == "Linear":
            for i in range(self.num_vars):
                self.cond_projections.append(nn.Linear(self.ori_seq_len, self.pred_len))
                self.cond_projections[i].weight = nn.Parameter(
                    (1 / self.ori_seq_len) * torch.ones([self.pred_len, self.ori_seq_len]))
        elif args.ablation_study_F_type == "CNN":
            for i in range(self.num_vars):
                self.cond_projections.append(nn.Linear(self.ori_seq_len, self.pred_len))
                self.cond_projections[i].weight = nn.Parameter(
                    (1 / self.ori_seq_len) * torch.ones([self.pred_len, self.ori_seq_len]))

            self.cnn_cond_projections = InputConvNetwork(args, self.num_vars, self.pred_len,
                                                         num_layers=args.cond_ddpm_num_layers,
                                                         ddpm_channels_conv=args.cond_ddpm_channels_conv)
            self.cnn_linear = nn.Linear(self.ori_seq_len, self.num_vars)

        if self.args.ablation_study_case in ["mix_1", "mix_ar_0"]:
            self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc + self.num_vars, self.num_vars,
                                                 num_layers=args.ddpm_layers_II)
        else:
            self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc + self.num_vars + self.num_vars,
                                                 self.num_vars, num_layers=args.ddpm_layers_II)

    def forward(self, yn=None, diffusion_step=None, cond_info=None, y_clean=None):

        x = yn.permute(0,2,1)
        x = self.input_projection(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step.long())
        # diffusion_emb = self.act(self.diffusion_embedding(diffusion_step))
        diffusion_emb = self.act(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1).repeat(1, 1, np.shape(x)[-1])

        # print(">>>>", np.shape(diffusion_emb), np.shape(x))
        # torch.Size([64, 1024, 168]) torch.Size([64, 1024, 168])

        # print(np.shape(diffusion_emb), np.shape(x))
        # torch.Size([64, 128, 168]) torch.Size([64, 256, 168])

        # print(">>>", np.shape(diffusion_emb))
        x = self.enc_conv(torch.cat([diffusion_emb, x], dim=1))
        # print(np.shape(x))

        pred_out = torch.zeros([yn.size(0), self.num_vars, self.pred_len], dtype=yn.dtype).to(yn.device)
        cond_info = cond_info.permute(0,2,1)
        for i in range(self.num_vars):
            pred_out[:, i, :] = self.cond_projections[i](cond_info[:, i, :self.ori_seq_len])
            # pred_out[:, i, :] = self.cond_projections[i](cond_info[:, i, :])

        if self.args.ablation_study_F_type == "CNN":
            # cnn with residual links
            temp_out = self.cnn_cond_projections(cond_info[:, :, :self.ori_seq_len])
            pred_out += self.cnn_linear(temp_out).permute(0, 2, 1)

        return_pred_out = pred_out

        if y_clean is not None:
            # y_clean = y_clean[:, :, -self.pred_len:]
            y_clean = y_clean[:, :, :]

            rand_for_mask = torch.rand_like(y_clean).to(x.device)

        # ==================================================================================
        pred_out = pred_out.permute(0,2,1)

        if y_clean is not None:
            pred_out = rand_for_mask * pred_out + (1 - rand_for_mask) * y_clean
            pred_out = pred_out.permute(0,2,1)
        inp = torch.cat([x, pred_out, cond_info[:, :, :]], dim=1)

        out = self.combine_conv(inp).permute(0,2,1) #decoder

        # if y_clean is not None:
        #     return out, return_pred_out
        return out


def noise_mask(X, masking_ratio=0.15, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
                1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        # print("1", np.shape(x))
        x = self.projection1(x)
        # print("2", np.shape(x))
        x = F.silu(x)
        x = self.projection2(x)
        # print("3", np.shape(x))
        x = F.silu(x)
        # 1 torch.Size([64, 128])
        # 2 torch.Size([64, 128])
        # 3 torch.Size([64, 128])
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class InputConvNetwork(nn.Module):

    def __init__(self, args, inp_num_channel, out_num_channel, num_layers=3, ddpm_channels_conv=None):
        super(InputConvNetwork, self).__init__()

        self.args = args

        self.inp_num_channel = inp_num_channel
        self.out_num_channel = out_num_channel

        kernel_size = 3
        padding = 1
        if ddpm_channels_conv is None:
            self.channels = args.ddpm_channels_conv
        else:
            self.channels = ddpm_channels_conv
        self.num_layers = num_layers

        self.net = nn.ModuleList()

        if num_layers == 1:
            self.net.append(Conv1dWithInitialization(
                in_channels=self.inp_num_channel,
                out_channels=self.out_num_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding, bias=True
            )
            )
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    dim_inp = self.inp_num_channel
                else:
                    dim_inp = self.channels
                self.net.append(Conv1dWithInitialization(
                    in_channels=dim_inp,
                    out_channels=self.channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding, bias=True
                ))
                self.net.append(torch.nn.BatchNorm1d(self.channels)),
                self.net.append(torch.nn.LeakyReLU(0.1)),
                self.net.append(torch.nn.Dropout(0.1, inplace=True))

            self.net.append(Conv1dWithInitialization(
                in_channels=self.channels,
                out_channels=self.out_num_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding, bias=True
            )
            )

    def forward(self, x=None):

        out = x
        for m in self.net:
            out = m(out)

        return out


if __name__ == '__main__':
    with open('../configs/test.yml', "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device("cpu")

    new_config.device = device

    model = Model(new_config)
    model = model.to(device)

    # data = torch.randn(128,3,32,32)
    for i in range(10):
        data = torch.randn(128, 64, 38)
        t = torch.randint(1000, size=(data.shape[0],))
        data = model(data, t)
        print(data.shape)

    print('test')


