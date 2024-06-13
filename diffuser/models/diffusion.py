from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion model for denoising and sampling from a probability distribution.

    Args:
        model (nn.Module): The diffusion model to be used.
        horizon (int): The length of the diffusion chain.
        observation_dim (int): The dimension of the observation space.
        action_dim (int): The dimension of the action space.
        n_timesteps (int): The number of diffusion steps.
        loss_type (str): The type of loss function to be used.
        clip_denoised (bool): Whether to clip denoised outputs.
        predict_epsilon (bool): Whether the model predicts epsilon directly.
        action_weight (float): Coefficient on first action loss.
        loss_discount (float): Multiplier for the loss at each step.
        loss_weights (dict): Loss coefficients for observation dimensions.

    Forward Input:
        cond (tuple): Conditions for diffusion sampling.

    Forward Output:
        Sample: Sampled trajectories and values.
    """
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        # alphas 是 betas 的补集,决定了每一步扩散中保留原始状态的比例
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        # 当 predict_epsilon 设置为 True 时，模型被训练来直接预测每一步扩散中添加的噪声
        # 当 predict_epsilon 设置为 False 时，模型被训练来预测去噪后的状态，即直接预测原始数据
        self.predict_epsilon = predict_epsilon

        # betas表示扩散过程中的方差增加项，用于控制噪声的引入速度
        self.register_buffer('betas', betas)
        # alphas_cumprod 是 alphas 的累积乘积，反映了在扩散过程中保留了多少原始信息
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # alphas_cumprod_prev 是前一步的alphas_cumprod，用于计算后验分布q(x_{t-1} | x_t, x_0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            # transition_dim = observation_dim + action_dim
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        # 将结果缩并成一个形状为 (horizon, transition_dim) 的张量
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            Predicts the starting point from noise based on the predict_epsilon flag.
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly.

            Args:
                x_t (torch.Tensor): The current noisy state.
                t (torch.Tensor): The current time step.
                noise (torch.Tensor): The predicted noise.

            Returns:
                torch.Tensor: The predicted starting point.
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        Computes the posterior mean, variance, and log variance.

        Args:
            x_start (torch.Tensor): The initial state.
            x_t (torch.Tensor): The current state.
            t (torch.Tensor): The current time step.

        Returns:
            tuple: A tuple containing the posterior mean, variance, and log variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        Computes the mean and variance of the predicted distribution.

        Args:
            x (torch.Tensor): The current state.
            cond (tuple): The conditions for the diffusion.
            t (torch.Tensor): The current time step.

        Returns:
            tuple: A tuple containing the model mean, posterior variance, and log variance.
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, 
                      sample_fn=default_sample_fn, **sample_kwargs):
        """
        Perform the sampling loop to generate samples from the diffusion model.

        Args:
            shape (tuple): The shape of the samples to be generated.
            cond (tuple): The conditions for the diffusion sampling.
            verbose (bool): Whether to display the sampling progress.
            return_chain (bool): Whether to return the entire sampling chain.
            sample_fn (function): The function to use for sampling at each step.
            **sample_kwargs: Additional keyword arguments for the sample_fn.

        Returns:
            Sample: A namedtuple containing the final sampled data, associated values, 
            and optionally the sampling chain.
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process(add noise).

        Args:
            x_start (torch.Tensor): The initial state.
            t (torch.Tensor): The current time step.
            noise (torch.Tensor): The noise to be added.

        Returns:
            torch.Tensor: The sampled state at time t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        """
        Computes the losses for the diffusion model.

        Args:
            x_start (torch.Tensor): The initial state.
            cond (tuple): The conditions for the diffusion.
            t (torch.Tensor): The current time step.

        Returns:
            tuple: A tuple containing the loss and additional information.
        """
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        """
        Computes the loss for a batch of data.

        Args:
            x (torch.Tensor): The batch of data.
            *args: Additional arguments for the loss computation.

        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        """
        Performs the forward denoise pass of the diffusion model.

        Args:
            cond (tuple): The conditions for the diffusion sampling.
            *args: Additional arguments for the forward pass.
            **kwargs: Additional keyword arguments for the forward pass.

        Returns:
            Sample: A namedtuple containing the sampled data and associated values.
        """
        return self.conditional_sample(cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):
    """
    Value diffusion model for denoising and sampling from a probability distribution.

    Args:
        (inherits arguments from GaussianDiffusion)
    """

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

