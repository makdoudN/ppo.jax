"""PPO Core functions"""

import jax
import chex
import rlax
import tree
import optax
import jax.numpy as jnp
import typing

from jax import jit
from jax import vmap
from jax import partial
from jax.experimental.optimizers import clip_grads
from rlax import truncated_generalized_advantage_estimation

vmap_gae = vmap(truncated_generalized_advantage_estimation, (1, 1, None, 1), 1)


@chex.dataclass
class Data:
    """Inputs Data for computing PPO losses."""
    observation: chex.Array
    last_observation: chex.Array
    terminal: chex.Array
    reward: chex.Array
    action: chex.Array


@chex.dataclass
class Batch:
    observation: chex.Array
    action: chex.Array
    advantage: chex.Array
    returns: chex.Array
    old_logprob: chex.Array


@chex.dataclass
class State:
    """ Variable State needed for the algorithm."""
    key: chex.Array
    params: chex.ArrayTree
    opt_state: chex.ArrayTree


def process_data(
    params: typing.Dict[str, typing.Any],
    data: Data,
    policy_apply: typing.Callable,
    value_apply: typing.Callable,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    discount_factor: float = 0.99,
    **kwargs,
) -> Batch:

    H, E = data['reward'].shape
    value_params = params['value']
    policy_params = params['policy']

    # -- Compute A^{GAE}: (Trunctated) Generalized Advantage Function.
    vf = value_apply(value_params, data['observation'])
    vf_last = value_apply(value_params, data['last_observation'])[None]
    values = jnp.concatenate([vf, vf_last])
    discount = data['terminal'] * discount_factor
    advantage = vmap_gae(data['reward'] * reward_scaling, discount, lambda_, values)
    chex.assert_equal_shape([vf, advantage])
    returns = advantage + vf

    # -- Flatten the data
    # Flatten from (H, E, *) to (H*E, *) where
    # H: Horizon,
    # E: Number of the environments.

    vf = vf.reshape(H * E)
    returns = returns.reshape(H * E)
    advantage = advantage.reshape(H * E,)
    action = data['action'].reshape(H * E, -1)
    if data['action'].ndim == 2:       # is discrete [H, E]
        action = action.squeeze(-1)
    observation = data['observation']
    observation = observation.reshape(H * E, *observation.shape[2:])
    logprob = policy_apply(policy_params, observation).log_prob(action)

    return Batch(observation=observation,
                 action=action,
                 returns=returns,
                 old_logprob=logprob,
                 advantage=advantage)


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def loss_ppo_def(
    params: typing.Dict[str, typing.Any],
    batch: Batch,
    policy_apply: typing.Callable,
    value_apply: typing.Callable,
    value_cost: float = 0.5,
    entropy_cost: float = 1e-4,
    epsilon_ppo: float = 0.2
):
    """Compute the PPO Loss Function."""
    value_params = params['value']
    policy_params = params['policy']

    # -- Loss Policy
    policy_distrib = policy_apply(policy_params, batch.observation)
    entropy = policy_distrib.entropy()
    logprob = policy_distrib.log_prob(batch.action)
    policy_ratio = jnp.exp(logprob - batch.old_logprob)

    chex.assert_equal_shape([logprob, batch.old_logprob])
    chex.assert_equal_shape([policy_ratio, batch.advantage])

    loss_H = -entropy_cost * entropy.mean()
    loss_CPI = -policy_ratio * batch.advantage
    loss_CPI = loss_CPI.mean()
    loss_policy = loss_CPI + loss_H

    # -- Loss value function
    vf = value_apply(value_params, batch.observation)
    chex.assert_equal_shape([vf, batch.returns])
    loss_V = rlax.l2_loss(vf, batch.returns).mean()
    loss = loss_policy + value_cost * loss_V

    metrics = dict(ppo_loss=loss, policy_loss=loss_CPI, value_loss=loss_V, H_loss=loss_H)
    return loss, metrics


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def update_ppo(
    state: State,
    data: Batch,
    policy_opt: optax.TransformUpdateFn,
    value_opt: optax.TransformUpdateFn,
    loss_kwargs: dict,
    process_data_kwargs: dict,
    niters: int = 5,
    minibatch_size: int = 32,
    max_grad_norm: float = -1.
):
    loss_fn = partial(loss_ppo_def, **loss_kwargs)

    @jit
    def update_fn(state, batch):
        grad, metrics = jax.grad(loss_fn, has_aux=True)(state.params, batch)
        if max_grad_norm > 0.0:
            grad = clip_grads(grad, max_grad_norm)
        updates, value_opt_state = value_opt(grad['value'], state.opt_state['value'])
        value_params = jax.tree_multimap(lambda p, u: p + u, state.params['value'], updates)
        updates, policy_opt_state = policy_opt(grad['policy'], state.opt_state['policy'])
        policy_params = jax.tree_multimap(lambda p, u: p + u, state.params['policy'], updates)
        params = state.params
        params = dict(policy=policy_params, value=value_params)
        opt_state = state.opt_state
        opt_state = dict(policy=policy_opt_state, value=value_opt_state)
        state = state.replace(params=params, opt_state=opt_state)
        return state, metrics

    @jit
    def _step(carry, xs):
        state, batch = carry
        minibatch = tree.map_structure(lambda v: v[xs], batch)
        state, info = update_fn(state, minibatch)
        carry = (state, batch)
        return carry, info

    @jit
    def _train_one_epoch(carry, xs):
        state, batch      = carry
        key, subkey       = jax.random.split(state.key, 2)
        n                 = batch['observation'].shape[0]
        index             = jnp.arange(n)
        indexes           = jax.random.permutation(subkey, index)
        indexes           = jnp.stack(
            jnp.array_split(indexes, n / minibatch_size)
        )

        state  = state.replace(key=key)
        _carry = (state, batch)
        _xs    = indexes
        (state, batch), info = jax.lax.scan(_step, _carry, _xs)    
        carry = (state, batch)
        return carry, info

    batch = partial(process_data, **process_data_kwargs)(state.params, data)
    carry = (state, batch)
    xs    = jnp.arange(niters)
    (state, _), info = jax.lax.scan(_train_one_epoch, carry, xs)
    info = tree.map_structure(lambda v: jnp.mean(v), info)
    return state, info


