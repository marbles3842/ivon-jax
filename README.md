# (Work in progress) IVON for Optax

A clean, lightweight JAX / [Optax](https://github.com/google-deepmind/optax) implementation of the **IVON** (Improved Variational Online Newton) optimizer.

## Usage

### Basic Training Step

Use `ivon` to initialize the optimizer and `ivon_value_and_grad` to compute the Monte Carlo averaged loss, gradients, and diagonal Hessian estimate over perturbed parameter samples:

```python
import jax
import optax
from ivon import ivon, ivon_value_and_grad

# Initialize optimizer (scaling_parameter is typically the dataset size N / effective sample size)
optimizer = ivon(
    learning_rate=0.1,
    scaling_parameter=60000,
    beta1=0.9,
    beta2=0.99999,
    hessian_init=1.0,
    weight_decay=1e-4,
)
opt_state = optimizer.init(params)

@jax.jit
def train_step(key, params, opt_state, batch):
    (loss, aux), (grad, hess) = ivon_value_and_grad(
        loss_fn,
        key=key,
        opt_state=opt_state,
        has_aux=True,
        mc_samples=1,
    )(params, batch)

    updates, opt_state = optimizer.update(
        grad, opt_state, params, avg_hess_estimate=hess
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux
```

### API Overview

- **`ivon(learning_rate, scaling_parameter, beta1=0.9, beta2=0.99999, hessian_init=1.0, weight_decay=1e-4, rescale_lr=True, clip_radius=float("inf"))`**: Full Optax gradient transformation chain for IVON.
- **`scale_by_ivon(scaling_parameter, beta1=0.9, beta2=0.99999, hessian_init=1.0, weight_decay=1e-4)`**: Core IVON scaling transformation (`GradientTransformationExtraArgs`), expecting `avg_hess_estimate` in `.update(...)`.
- **`ivon_value_and_grad(fun, key, opt_state, *, argnums=0, has_aux=False, mc_samples=1)`**: Drop-in wrapper around `jax.value_and_grad` that samples perturbed parameters from the variational posterior using `jax.lax.scan` and returns `(value, (avg_grad, avg_hess))` (or `((value, aux), (avg_grad, avg_hess))` when `has_aux=True`).

## Acknowledgements

This implementation is heavily inspired by the original Optax implementation by Yuesong Shen ([ysngshn/ivon-optax](https://github.com/ysngshn/ivon-optax)) and the original IVON paper, [*Variational Learning is Effective for Large Deep Networks*](https://arxiv.org/abs/2402.17641) (Shen et al., ICML 2024).

```bibtex
@inproceedings{shen2024variational,
      title={Variational Learning is Effective for Large Deep Networks}, 
      author={Yuesong Shen and Nico Daheim and Bai Cong and Peter Nickl and Gian Maria Marconi and Clement Bazan and Rio Yokota and Iryna Gurevych and Daniel Cremers and Mohammad Emtiyaz Khan and Thomas Möllenhoff},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2024},
      url={https://arxiv.org/abs/2402.17641}
}
```
