import jax
import jax.numpy as jnp

from adahessian_jax import hutchinson_estimator_diag_hessian


def loss_fn(params):
  return jnp.sum(params ** 2)


def main():
  params = jnp.array([1.0, 2.0, 3.0])
  estimator = hutchinson_estimator_diag_hessian(n_samples=5)
  state = estimator.init(params)
  diag, _ = estimator.update(None, state, params=params, obj_fn=loss_fn)
  print("Estimated Hessian diagonal:", diag)


if __name__ == "__main__":
  main()
