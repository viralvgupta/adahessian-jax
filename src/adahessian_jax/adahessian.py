"""AdaHessian optimizer in pure JAX."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

PyTree = Any


class GradientTransformation(NamedTuple):
  init: Callable[[PyTree], PyTree]
  update: Callable[..., Tuple[PyTree, PyTree]]


class GradientTransformationExtraArgs(GradientTransformation):
  pass


class HutchinsonState(NamedTuple):
  key: jax.Array


class AdaHessianState(NamedTuple):
  count: jax.Array
  mu: PyTree
  nu: PyTree
  hessian_diag: PyTree
  hessian_fn_state: Any


def _tree_leaves(tree: PyTree):
  return jax.tree_util.tree_leaves(tree)


def _tree_map(fn, *trees: PyTree):
  return jax.tree_util.tree_map(fn, *trees)


def _tree_zeros_like(tree: PyTree, dtype: Optional[Any] = None):
  if dtype is None:
    return _tree_map(lambda x: jnp.zeros_like(x), tree)
  return _tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def _tree_cast(tree: PyTree, dtype: Optional[Any]):
  if dtype is None:
    return tree
  return _tree_map(lambda x: x.astype(dtype), tree)


def _tree_dtype(tree: PyTree):
  leaves = _tree_leaves(tree)
  if not leaves:
    raise ValueError("tree has no leaves")
  return leaves[0].dtype


def _tree_size(tree: PyTree):
  return sum(x.size for x in _tree_leaves(tree))


def _tree_update_moment(updates, moments, decay, order):
  return _tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t,
      updates,
      moments,
  )


def _tree_update_moment_per_elem_norm(updates, moments, decay, order):
  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    half_order = order / 2
    if float(half_order).is_integer():
      half_order = int(half_order)
    return (jnp.real(g) ** 2 + jnp.imag(g) ** 2) ** half_order

  return _tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t,
      updates,
      moments,
  )


def _tree_bias_correction(moment, decay, count):
  bias_correction_ = 1 - decay**count
  return _tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)


def _tree_random_like(key: jax.Array, tree: PyTree, sampler, dtype):
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  keys = jax.random.split(key, len(leaves))
  new_leaves = [sampler(k, l.shape, dtype=dtype) for k, l in zip(keys, leaves)]
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


def hutchinson_estimator_diag_hessian(
    random_seed: Optional[jax.Array] = None,
    n_samples: int = 1,
) -> GradientTransformationExtraArgs:
  if n_samples < 1:
    raise ValueError("n_samples must be >= 1.")

  def init_fn(params):
    del params
    key = random_seed if random_seed is not None else jax.random.PRNGKey(0)
    return HutchinsonState(key=key)

  def update_fn(updates, state, params=None, obj_fn=None, **extra_args):
    del extra_args, updates
    if params is None:
      raise ValueError("params must be provided to hutchinson update function.")
    if obj_fn is None:
      raise ValueError("obj_fn must be provided to hutchinson update function.")

    key, *subkeys = jax.random.split(state.key, n_samples + 1)

    def one_sample(subkey):
      random_signs = _tree_random_like(
          subkey, params, jax.random.rademacher, dtype=jnp.float32
      )
      random_signs = _tree_cast(random_signs, _tree_dtype(params))
      hvp = jax.jvp(jax.grad(obj_fn), (params,), (random_signs,))[1]
      return _tree_map(lambda h, r: h * r, hvp, random_signs)

    samples = [one_sample(sk) for sk in subkeys]

    def sum_tree(x, y):
      return _tree_map(lambda a, b: a + b, x, y)

    hessian_diag = samples[0]
    for sample in samples[1:]:
      hessian_diag = sum_tree(hessian_diag, sample)

    hessian_diag = _tree_map(lambda x: x / n_samples, hessian_diag)
    return hessian_diag, HutchinsonState(key=key)

  return GradientTransformationExtraArgs(init_fn, update_fn)


def _average_conv_kernel_hessian(hessian_diag: PyTree) -> PyTree:
  def maybe_average(h):
    if hasattr(h, "ndim") and h.ndim == 4:
      mean = jnp.mean(jnp.abs(h), axis=(2, 3), keepdims=True)
      return jnp.ones_like(h) * mean
    return h

  return _tree_map(maybe_average, hessian_diag)


def scale_by_adahessian(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    hessian_power: float = 1.0,
    update_interval: int = 1,
    average_conv_kernel: bool = True,
    hessian_diagonal_fn: Union[
        GradientTransformation,
        GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
) -> GradientTransformationExtraArgs:
  def init_fn(params):
    return AdaHessianState(
        count=jnp.zeros([], jnp.int32),
        mu=_tree_zeros_like(params, dtype=mu_dtype),
        nu=_tree_zeros_like(params),
        hessian_diag=_tree_zeros_like(params),
        hessian_fn_state=hessian_diagonal_fn.init(params),
    )

  def update_fn(updates, state: AdaHessianState, params=None, **hess_fn_kwargs):
    if params is None:
      raise ValueError("params must be provided to AdaHessian update")

    count_inc = state.count + jnp.array(1, dtype=state.count.dtype)

    mu = _tree_update_moment(updates, state.mu, b1, 1)
    mu_hat = _tree_bias_correction(mu, b1, count_inc)

    def update_hessian_diag(hess_fn_state, hessian_diag):
      hessian_diag, hess_fn_state = hessian_diagonal_fn.update(
          updates, hess_fn_state, params=params, **hess_fn_kwargs
      )
      if average_conv_kernel:
        hessian_diag = _average_conv_kernel_hessian(hessian_diag)
      return hess_fn_state, hessian_diag

    hessian_fn_state, hessian_diag = jax.lax.cond(
        jnp.equal(state.count % update_interval, 0),
        update_hessian_diag,
        lambda h, d: (h, d),
        state.hessian_fn_state,
        state.hessian_diag,
    )

    nu = _tree_update_moment_per_elem_norm(hessian_diag, state.nu, b2, 2)
    nu_hat = _tree_bias_correction(nu, b2, count_inc)

    denom = _tree_map(
        lambda n: jnp.power(n, hessian_power / 2) + eps, nu_hat
    )
    updates = _tree_map(lambda m, d: m / d, mu_hat, denom)

    mu = _tree_cast(mu, mu_dtype)

    new_state = AdaHessianState(
        count=count_inc,
        mu=mu,
        nu=nu,
        hessian_diag=hessian_diag,
        hessian_fn_state=hessian_fn_state,
    )
    return updates, new_state

  return GradientTransformationExtraArgs(init_fn, update_fn)


def adahessian(
    learning_rate: Union[float, Callable[[jax.Array], float]],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    hessian_power: float = 1.0,
    update_interval: int = 1,
    weight_decay: float = 0.0,
    average_conv_kernel: bool = True,
    hessian_diagonal_fn: Union[
        GradientTransformation,
        GradientTransformationExtraArgs,
    ] = hutchinson_estimator_diag_hessian(),
    mu_dtype: Optional[Any] = None,
) -> GradientTransformationExtraArgs:
  base_opt = scale_by_adahessian(
      b1=b1,
      b2=b2,
      eps=eps,
      hessian_power=hessian_power,
      update_interval=update_interval,
      average_conv_kernel=average_conv_kernel,
      hessian_diagonal_fn=hessian_diagonal_fn,
      mu_dtype=mu_dtype,
  )

  def init_fn(params):
    return base_opt.init(params)

  def update_fn(updates, state, params=None, **kwargs):
    if params is None:
      raise ValueError("params must be provided to AdaHessian update")

    count = state.count
    lr = learning_rate(count) if callable(learning_rate) else learning_rate
    scaled_updates, new_state = base_opt.update(updates, state, params, **kwargs)
    if weight_decay != 0.0:
      scaled_updates = _tree_map(
          lambda u, p: u + weight_decay * p, scaled_updates, params
      )
    scaled_updates = _tree_map(lambda u: -lr * u, scaled_updates)
    return scaled_updates, new_state

  return GradientTransformationExtraArgs(init_fn, update_fn)
