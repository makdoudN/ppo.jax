import jax
import numpy as np
from jax import jit
from tax import reduce

reduce = jit(reduce)


def gym_evaluation(rng, env, policy_params, policy_apply, niters: int = 5):
    action_type = env.action_space.__class__.__name__
    all_scores = []
    for _ in range(niters):
        observation, score = env.reset(), 0
        for _ in range(env.spec.max_episode_steps):
            rng, rng_action = jax.random.split(rng)
            action = policy_apply(policy_params, observation).sample(seed=rng_action)
            if action_type == 'Discrete':
                action = int(action)
            else:
                action = np.asarray(action)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        all_scores.append(score)
    info = {}
    info['eval/score'] = np.mean(all_scores)
    info['eval/score_std'] = np.std(all_scores)
    return info


def gym_interaction(env, policy_apply, horizon: int = 10, seed: int = 42):
    action_type = env.action_space.__class__.__name__
    rng = jax.random.PRNGKey(seed)
    observation, buf = env.reset(), []
    params = yield
        
    while True:
        for _ in range(horizon):
            # Select the action
            rng, rng_action = jax.random.split(rng)
            action = policy_apply(params, observation).sample(seed=rng_action)
            if action_type == 'Discrete':
                action = int(action)
            else:
                action = np.asarray(action)
            
            # Interact with the environment and store the result.
            observation_next, reward, done, info = env.step(action)
            buf.append({
                'observation': observation,
                'reward': reward,
                'terminal': 1. - done,
                'action': action
            })
            observation = observation_next.copy()
            
        data = reduce(buf)  # stack all subtree.
        data['last_observation'] = observation
        params = yield data
        buf = []
        
        
        
# Deluca interaction. 
# Deluca evaluation.