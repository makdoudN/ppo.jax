import jax
import tax
import gym
import tree
import yaml
import optax
import haiku as hk
import jax.numpy as jnp
from ppo.ppo import State
from ppo.ppo import loss_ppo_def
from ppo.ppo import process_data
from ppo.common.utils import evaluation
from gym.vector import AsyncVectorEnv
from functools import partial
from jax import jit
from omegaconf import OmegaConf


def save_conf(conf: OmegaConf, rec) -> None:
    yconf = OmegaConf.to_yaml(conf, resolve=True)
    print(yconf)
    rec.save(yaml.safe_load(yconf), "conf.yaml")


def setup_envs(conf):
    make_env = lambda: gym.make(conf.env)
    venv = AsyncVectorEnv([make_env for _ in range(conf.num_envs)])
    env_valid = make_env()
    action_size = env_valid.action_space.shape[0]
    observation_size = env_valid.observation_space.shape[0]
    return {
        "env": venv,
        "env_valid": env_valid,
        "observation_size": observation_size,
        "action_size": action_size,
    }


def setup(conf):
    rng = jax.random.PRNGKey(conf.seed)
    envs_info = setup_envs(conf)
    action_size = envs_info["action_size"]
    observation_size = envs_info["observation_size"]
    dummy_observation = jnp.zeros((observation_size,))

    value_def = lambda x: tax.mlp_deterministic(1, **conf.value_kwargs)(x).squeeze(-1)
    value_def = hk.transform(value_def)
    value_def = hk.without_apply_rng(value_def)
    value_opt = getattr(optax, conf.value_opt)(**conf.value_opt_kwargs)

    policy_def = lambda x: tax.mlp_multivariate_normal_diag(action_size, **conf.policy_kwargs)(x)
    policy_def = hk.transform(policy_def)
    policy_def = hk.without_apply_rng(policy_def)
    policy_opt = getattr(optax, conf.policy_opt)(**conf.policy_opt_kwargs)

    rng, rng_policy, rng_value = jax.random.split(rng, 3)
    value_params = value_def.init(rng_value, dummy_observation)
    policy_params = policy_def.init(rng_policy, dummy_observation)
    value_opt_state = value_opt.init(value_params)
    policy_opt_state = policy_opt.init(policy_params)

    params = {"policy": policy_params, "value": value_params}
    opt_state = {"policy": policy_opt_state, "value": value_opt_state}

    process_data_to_batch = partial(
        process_data,
        discount_factor=conf.discount_factor,
        reward_scaling=conf.reward_scaling,
        lambda_=conf.lambda_,
        policy_apply=policy_def.apply,
        value_apply=value_def.apply)

    loss = partial(
        loss_ppo_def,
        value_apply=value_def.apply,
        policy_apply=policy_def.apply,
        value_cost=conf.value_cost,
        entropy_cost=conf.entropy_cost,
        epsilon_ppo=conf.epsilon_ppo
    )

    @jit
    def update_fn(state, inputs):
        """ Generic Update function """
        g, metrics = jax.grad(loss, has_aux=True)(state.params, inputs)

        updates, value_opt_state = value_opt.update(g["value"], state.opt_state["value"])
        value_params = jax.tree_multimap(
            lambda p, u: p + u, state.params["value"], updates
        )

        updates, policy_opt_state = policy_opt.update(g["policy"], state.opt_state["policy"])
        policy_params = jax.tree_multimap(
            lambda p, u: p + u, state.params["policy"], updates
        )

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
    def _fit(carry, xs):
        state, batch, key = carry
        n                 = batch['observation'].shape[0]
        key, subkey       = jax.random.split(rng)
        index             = jnp.arange(n)
        indexes           = jax.random.permutation(subkey, index)
        indexes           = jnp.stack(
            jnp.array_split(indexes, n / conf.minibatch_ppo)
        )

        _carry = (state, batch)
        _xs    = indexes
        (state, batch), info = jax.lax.scan(_step, _carry, _xs)
        carry = (state, batch, key)
        return carry, info

    @jit
    def training_step(rng, state, data):
        batch = process_data_to_batch(state.params, data)
        rng, subrng = jax.random.split(rng)
        carry = (state, batch, subrng)
        xs = jnp.arange(conf.epoch_ppo)
        (state, batch, _), info = _fit(carry, xs)
        info = tree.map_structure(lambda v: jnp.mean(v), info)
        return state, info

    def interaction(env, horizon: int = 10, seed: int = 42):
        steps = 0
        rng = jax.random.PRNGKey(seed)
        observation, buf = env.reset(), []
        num_envs = observation.shape[0]
        params = yield

        while True:
            for _ in range(horizon):
                steps += num_envs
                rng, rng_action = jax.random.split(rng)
                action = jit(policy_def.apply)(params, observation).sample(
                    seed=rng_action
                )
                observation_next, reward, done, info = env.step(action)
                buf.append(
                    {
                        "observation": observation,
                        "reward": reward,
                        "terminal": 1.0 - done,
                        "action": action,
                    }
                )
                observation = observation_next.copy()

            data = jit(tax.reduce)(buf)
            data["last_observation"] = observation
            info = {'steps': steps}
            params = yield data, info
            buf = []

    interaction_step = interaction(envs_info["env"], horizon=conf.horizon, seed=conf.seed + 1)
    interaction_step.send(None)

    def make_policy(params):
        fn = lambda rng, x: policy_def.apply(params, x).sample(seed=rng)
        return jit(fn)

    return State(key=rng, params=params, opt_state=opt_state), {
        "interaction": interaction_step,
        "evaluation": lambda rng, params: evaluation(
            rng, envs_info["env_valid"], make_policy(params)
        ),
        "update": jit(
            lambda rng, state, data: training_step(rng, state, data)
        ),
    }


