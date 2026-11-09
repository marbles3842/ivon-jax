import jax
import jax.numpy as jnp

import optax

from typing import NamedTuple, Callable, Optional
from jax import tree
from jax.lax import rsqrt, scan

# TODO
# Fix has_aux=True unpacking in ivon_value_and_grad
# Support Distributed Training (pmean) ? Verify that 


class IVONState(NamedTuple):
    precision: optax.Params
    momentum: optax.Updates
    step: jax.Array


def sample_parameters(
    rng_key: jax.Array, mean: optax.Params, std: optax.Params
) -> optax.Params:
    noise = optax.tree.random_like(rng_key=rng_key, target_tree=mean)
    return noise, tree.map(lambda m, s, eps: m + eps * s, mean, std, noise)


def ivon(
    learning_rate: optax.ScalarOrSchedule,
    scaling_parameter: int,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    hessian_init: float = 1.0,
    weight_decay: float = 1e-4,
    rescale_lr: bool = True,
    clip_radius: float = float("inf"),
) -> optax.GradientTransformation:
    lr_scale = (
        (
            optax.scale_by_learning_rate(learning_rate),
            optax.scale(hessian_init + weight_decay),
        )
        if rescale_lr
        else (optax.scale_by_learning_rate(learning_rate),)
    )

    cr_transform = (optax.clip(clip_radius),) if clip_radius < float("inf") else ()

    return optax.chain(
        scale_by_ivon(
            scaling_parameter=scaling_parameter,
            beta1=beta1,
            beta2=beta2,
            hessian_init=hessian_init,
            weight_decay=weight_decay,
        ),
        *cr_transform,
        *lr_scale
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
        avg_hess_estimate: optax.Updates
    ):
        current_step = optax.safe_int32_increment(state.step)

        momentum = optax.update_moment(
            updates=avg_grads, moments=state.momentum, decay=beta1, order=1
        )

        precision = _update_precision(
            precision=state.precision,
            avg_hess=avg_hess_estimate,
            beta2=beta2,
            weight_decay=weight_decay,
            scaling_param=scaling_parameter,
        )

        updates = tree.map(
            lambda p, s, m: scaling_parameter * (m + weight_decay * p) / s,
            params,
            precision,
            optax.bias_correction(momentum, beta1, current_step),
        )
        return updates, state._replace(
            step=current_step, precision=precision, momentum=momentum
        )

    return optax.GradientTransformationExtraArgs(
        init=lambda p: _ivon_init(p, hessian_init, weight_decay, scaling_parameter),
        update=ivon_update,
    )


def _ivon_init(
    params: optax.Params, hessian_init: float, weight_decay: float, scaling_param: float
) -> IVONState:
    hessian = optax.tree.full_like(tree=params, fill_value=hessian_init)
    momentum = optax.tree.zeros_like(params)
    return IVONState(
        precision=tree.map(lambda h: scaling_param * (h + weight_decay), hessian),
        momentum=momentum,
        step=jnp.zeros([], dtype=jnp.int32),
    )


def _update_precision(
    precision: optax.Updates,
    avg_hess: optax.Updates,
    beta2: float,
    weight_decay: float,
    scaling_param: float,
) -> optax.Updates:
    update_weight = 1 - beta2
    correction_weight = 0.5 * update_weight**2

    def update_leaf(current_precision, hessian):
        estimated_precision = scaling_param * (hessian + weight_decay)
        correction = (
            correction_weight
            * (current_precision - estimated_precision) ** 2
            / current_precision
        )
        return (
            beta2 * current_precision + update_weight * estimated_precision + correction
        )

    return tree.map(update_leaf, precision, avg_hess)


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

    keys = jax.random.split(key, mc_samples)

    std = tree.map(rsqrt, ivon_state.precision)

    def wrapper(*args, **kwargs):
        def step(carry, key: jax.Array):
            updates, hess = carry
            params = args[argnums]
            noise, psample = sample_parameters(rng_key=key, mean=params, std=std)
            perturbed_args = args[:argnums] + (psample,) + args[argnums + 1 :]

            out, new_grads = jax.value_and_grad(
                fun, argnums=argnums, has_aux=has_aux
            )(*perturbed_args, **kwargs)

            if has_aux:
                new_value, aux = out
            else:
                new_value, aux = out, None  # None is an empty pytree, scan handles it

            new_hess = tree.map(lambda g, eps, s: g * eps / s, new_grads, noise, std)

            updates = optax.tree.add_scale(updates, 1 / mc_samples, new_grads)
            hess = optax.tree.add_scale(hess, 1 / mc_samples, new_hess)

            return (updates, hess), (new_value, aux)

        init_state = (
            optax.tree.zeros_like(args[argnums]),
            optax.tree.zeros_like(args[argnums]),
        )
        (avg_grad, avg_hess), (values, auxs) = scan(step, init=init_state, xs=keys)

        value = jnp.mean(values)
        grads_and_hess = (avg_grad, avg_hess)

        if has_aux:
            aux = tree.map(lambda x: jnp.mean(x, axis=0), auxs)
            return (value, aux), grads_and_hess
        return value, grads_and_hess

    return wrapper
