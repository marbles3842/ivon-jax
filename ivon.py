import jax
import jax.numpy as jnp

import optax

from typing import NamedTuple, Callable, Optional
from jax import tree
from jax.lax import rsqrt, scan

import jax.random as jr


class IVONState(NamedTuple):
    hessian: optax.Params
    acc_grads: optax.Updates
    step: jax.Array
    weight_decay: float
    scaling_parameter: float


def sample_parameters(
    rng_key: jr.key, mean: optax.Params, std: optax.Params
) -> optax.Params:
    noise = optax.tree.random_like(rng_key=rng_key, target_tree=mean)
    return noise, tree.map(lambda m, s, eps: m + eps * s, mean, std, noise)


def ivon(
    learning_rate: float,
    scaling_parameter: int,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    hessian_init: float = 1.0,
    weight_decay: float = 1e-4,
) -> optax.GradientTransformation:
    return optax.chain(
        scale_by_ivon(
            scaling_parameter=scaling_parameter,
            beta1=beta1,
            beta2=beta2,
            hessian_init=hessian_init,
            weight_decay=weight_decay,
        ),
        optax.scale_by_learning_rate(learning_rate=learning_rate),
    )


def scale_by_ivon(
    scaling_parameter: int,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    hessian_init: float = 1.0,
    weight_decay: float = 1e-4,
) -> optax.GradientTransformation:

    def ivon_update(
        avg_grads: Optional[optax.Updates],
        state: IVONState,
        params: Optional[optax.Params],
        *,
        nxg: optax.Updates
    ):
        current_step = optax.safe_int32_increment(state.step)

        g = optax.update_moment(avg_grads, state.acc_grads, decay=beta1, order=1)
        g = optax.bias_correction(g, beta1, current_step)

        hess = _update_hessian(
            hessian=state.hessian, avg_hess=nxg, beta2=beta2, weight_decay=weight_decay
        )

        updates = tree.map(
            lambda m, h, g: (g + weight_decay * m) / (h + weight_decay),
            params,
            hess,
            g,
        )
        return updates, state._replace(step=current_step, hessian=hess, acc_grads=g)

    return optax.GradientTransformationExtraArgs(
        init=lambda p: _ivon_init(p, hessian_init, weight_decay, scaling_parameter),
        update=ivon_update,
    )


def _ivon_init(
    params: optax.Params, hessian_init: float, weight_decay: float, scaling_param: float
) -> IVONState:
    hessian = optax.tree.full_like(tree=params, fill_value=hessian_init)
    grads = optax.tree.zeros_like(params)
    return IVONState(
        hessian=hessian,
        acc_grads=grads,
        step=jnp.zeros([], dtype=jnp.int32),
        scaling_parameter=scaling_param,
        weight_decay=weight_decay,
    )


# TODO: check if it should be called average hessian as per the original paper
def _update_hessian(
    hessian: optax.Updates, avg_hess: optax.Updates, beta2: float, weight_decay: float
) -> optax.Updates:
    return tree.map(
        lambda h, hh: beta2 * h
        + (1 - beta2) * hh
        + 0.5 * (1 - beta2) ** 2 * (h - hh) ** 2 / (h + weight_decay),
        hessian,
        avg_hess,
    )


def _compute_std(hessian: optax.Params, weight_decay: float, scaling_param: float):
    return tree.map(lambda h: rsqrt(scaling_param * (h + weight_decay)), hessian)


def ivon_value_and_grad(
    fun: Callable,
    key: jax.Array,
    opt_state: optax.OptState,
    *,
    argnums: int = 0,
    has_aux: bool = False,
    mc_samples: int = 1
):
    ivon_state = optax.tree.get(opt_state, IVONState.__name__)
    if ivon_state is None:
        raise ValueError("Could not find IVONState in optimizer state.")

    keys = jr.split(key, mc_samples)

    std = _compute_std(
        ivon_state.hessian,
        weight_decay=ivon_state.weight_decay,
        scaling_param=ivon_state.scaling_parameter,
    )

    def wrapper(*args, **kwargs):
        def step(_, key: jax.Array):
            params = args[argnums]
            noise, psample = sample_parameters(rng_key=key, mean=params, std=std)
            perturbed_args = args[:argnums] + (psample,) + args[argnums + 1 :]
            value, updates = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)(
                *perturbed_args, **kwargs
            )
            nxg = tree.map(lambda g, eps, s: g * eps / s, updates, noise, std)
            return _, (value, updates, nxg)

        _, (values, grads, nxgs) = scan(step, init=(), xs=keys)

        avg_grad = tree.map(lambda g: jnp.mean(g, axis=0), grads)
        avg_hess = tree.map(lambda h: jnp.mean(h, axis=0), nxgs)

        return jnp.mean(values), (avg_grad, avg_hess)

    return wrapper
