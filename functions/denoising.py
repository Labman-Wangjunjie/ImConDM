import torch
import torch.nn.functional as F

def extract(a, t, x_shape):

    batch_size = t.shape[0]
    # t = t.view(2, 1, 1) / 1000
    t = t.view(2, 1, 1) / 1000
    out = torch.gather(a,1, t.to(torch.int64).to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def scale_func(posterior_variance, y, t, base_scale):
    return extract(posterior_variance, t, y.shape) * base_scale


def fast_denoise(xt, t, at,model, observation,noise=None):
    if noise is None:
        noise = model(xt, t, observation, observation)
    sqrt_alphas_cumprod = torch.sqrt(at)
    sqrt_one_minus_alphas_cumprod =torch.sqrt(
            1.0 - at
        )
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, xt.shape
    )
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, xt.shape)

    return (
                   xt - sqrt_one_minus_alphas_cumprod_t * noise
           ) / sqrt_alphas_cumprod_t


def energy_func(y, t, observation, observation_mask, at, model):
    return F.mse_loss(
        fast_denoise(y, t,at, model,observation),
        observation,
        reduction="none",
    )[observation_mask == 1].sum()


def score_func(y, t, observation, observation_mask, at, model):
    with torch.enable_grad():
        y.requires_grad_(True)
        Ey = energy_func(
            y, t, observation, observation_mask, at, model
        )
        return -torch.autograd.grad(Ey, y)[0]


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x_0, x, mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        base_scale =4.0
        obs = x_0
        x = x * mask
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)

            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            posterior_variance = b * (1.0 - at_next) / (1.0 - at)
            scale = scale_func(posterior_variance, x, t, base_scale)

            xt = xs[-1].to('cuda')
            et = model(xt, t, obs, obs)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xt_next = xt_next + scale * score_func(xt_next, t, obs, mask, at,model)

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                               (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                       ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
