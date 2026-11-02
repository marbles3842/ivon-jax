import jax
import jax.numpy as jnp

import optax

from typing import NamedTuple, Callable
from jax import tree
from jax.lax import rsqrt

import jax.random as jr


class IVONState(NamedTuple):
    hessian: optax.Params
    std: optax.Params
    acc_grads: optax.Updates
    acc_hess: optax.Updates
    step: int


def sample_parameters(
    rng_key: jr.key, mean: optax.Params, std: optax.Params
) -> optax.Params:
    noise = optax.tree.random_like(rng_key=rng_key, target_tree=mean)
    return noise, tree.map(lambda m, s, eps: m + eps * s, mean, std, noise)


def make_monte_carlo_grad(loss_fn: Callable, num_samples: int = 1):

    def _single_sample(key: jr.PRNGKey, mean: optax.Params, std: optax.Updates, data):
        noise, psample = sample_parameters(rng_key=key, mean=mean, std=std)        
        x, y = data

        loss, grad = jax.value_and_grad(loss_fn)(psample, x, y)
        hess_upd = optax.tree.mul(noise, grad)
        return loss, grad, hess_upd

    def grad_fn(key: jr.PRNGKey, params: optax.Params, state: IVONState, data):
        keys = jr.split(key, num_samples)
        
        losses, grads, hess = jax.vmap(_single_sample, in_axes=(0, None, None, None))(
            keys, params, state.std, data
        )

        avg_grad = tree.map(lambda g: jnp.mean(g, axis=0), grads)
        avg_hess = tree.map(lambda h: jnp.mean(h, axis=0), hess) #TODO: change naming
        return jnp.mean(losses), avg_grad, state._replace(acc_hess = avg_hess)

    return grad_fn


def ivon(
    learning_rate: float,
    scaling_parameter: int,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    hessian_init: float = 1.0,
    weight_decay: float = 1e-4,
) -> optax.GradientTransformation:
    # TODO: add lr scaling

    def ivon_update(
        avg_grads: optax.Updates, state: IVONState, params: optax.Params = None
    ):
        current_step = state.step + 1
                
        g = optax.tree.add(
            optax.tree.scale(beta1, state.acc_grads),
            optax.tree.scale(1 - beta1, avg_grads),
        )
        hess = tree.map(
            lambda h, hh: beta2 * h
            + (1 - beta2) * hh
            + 0.5 * (1 - beta2) ** 2 * (h - hh) ** 2 / (h + weight_decay),
            state.hessian,
            state.acc_hess,
        )

        
        updates = tree.map(
            lambda m, h, g: -learning_rate
            * (g + weight_decay * m)
            / (h + weight_decay),
            params,
            hess,
            optax.tree.scale(1 / (1 - beta1**current_step), g),
        )
        std = _compute_std(
            hessian=hess, weight_decay=weight_decay, scaling_param=scaling_parameter
        )
        return updates, state._replace(step=current_step, std=std, hessian=hess, acc_grads = g)

    return optax.GradientTransformation(init=lambda p: _ivon_init(p, hessian_init, weight_decay, scaling_parameter), update=ivon_update)


def _ivon_init(
    params: optax.Param, hessian_init: float, weight_decay: float, scaling_param: float
) -> IVONState:
    hessian = optax.tree.full_like(tree=params, fill_value=hessian_init)
    grads = optax.tree.zeros_like(params)
    acc_hess = optax.tree.zeros_like(params)
    std = _compute_std(
        hessian=hessian, weight_decay=weight_decay, scaling_param=scaling_param
    )
    return IVONState(
        hessian=hessian,
        std=std,
        acc_grads=grads,
        acc_hess=acc_hess,
        step=0,
    )


def _compute_std(hessian: optax.Params, weight_decay: float, scaling_param: float):
    return tree.map(lambda h: rsqrt(scaling_param * (h + weight_decay)), hessian)

