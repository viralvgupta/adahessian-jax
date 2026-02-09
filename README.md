# adahessian-jax

![tests](https://github.com/viralvgupta/adahessian-jax/actions/workflows/tests.yml/badge.svg)

Standalone JAX implementation of the AdaHessian optimizer with Hutchinson diagonal Hessian estimation.

## Why AdaHessian

Adam adapts step sizes using **gradient variance**. That works well, but it does not directly account for **curvature** of the loss landscape. AdaHessian adds curvature information by estimating the **diagonal of the Hessian**, which lets it shrink steps in directions with high curvature and take larger steps in flatter directions.

In short:
- **Adam** scales updates by the EMA of squared gradients.
- **AdaHessian** scales updates by the EMA of squared Hessian diagonals.

This can improve stability and convergence on some problems, especially when curvature varies widely across parameters.

## How it works (high level)

1. Compute gradients as usual.
2. Estimate the Hessian diagonal using Hutchinson’s method.
3. Maintain an EMA of the squared Hessian diagonal.
4. Scale the momentum update by the curvature estimate.

## Install

```bash
python -m pip install -e .
```

## Tests

```bash
python -m pip install pytest
python -m pytest tests
```

## Quick demo

```bash
python examples/hutchinson_demo.py
```

## Usage

```python
import jax
import jax.numpy as jnp
from adahessian_jax import adahessian

def loss_fn(params):
    return jnp.sum(params ** 2)

params = jnp.array([1.0, 2.0, 3.0])
opt = adahessian(learning_rate=1e-2)
state = opt.init(params)

grads = jax.grad(loss_fn)(params)
updates, state = opt.update(grads, state, params, obj_fn=loss_fn)
params = params + updates
```

## Notes

- `obj_fn` is required for the Hutchinson Hessian estimator.
- `update_interval` lets you compute Hessian diagonals less frequently.

## License

Apache-2.0. See `LICENSE`.
