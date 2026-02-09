import jax
import jax.numpy as jnp

from adahessian_jax import adahessian, hutchinson_estimator_diag_hessian


def test_hutchinson_estimator_quadratic_diag():
  def obj_fn(params):
    return jnp.sum(params ** 2)

  params = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
  estimator = hutchinson_estimator_diag_hessian(n_samples=4)
  state = estimator.init(params)
  diag, _ = estimator.update(None, state, params=params, obj_fn=obj_fn)

  expected = jnp.full_like(params, 2.0)
  assert jnp.allclose(diag, expected, rtol=0.0, atol=0.0)


def test_adahessian_update_runs():
  def obj_fn(params):
    return jnp.sum(params ** 2)

  params = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
  opt = adahessian(learning_rate=1e-2, update_interval=2)
  state = opt.init(params)

  grads = jax.grad(obj_fn)(params)
  updates, state = opt.update(grads, state, params, obj_fn=obj_fn)

  def _all_finite(x):
    return jnp.all(jnp.isfinite(x))

  assert jax.tree.reduce(lambda a, b: a & b,
                          jax.tree.map(_all_finite, updates))
  assert jax.tree.reduce(lambda a, b: a & b,
                          jax.tree.map(_all_finite, state))
