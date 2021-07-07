import jax
import numpy as np


def evaluation(rng, env, policy, niters: int = 5):
    action_type = env.action_space.__class__.__name__
    all_scores = []
    for _ in range(niters):
        observation, score = env.reset(), 0
        for _ in range(env.spec.max_episode_steps):
            rng, rng_action = jax.random.split(rng)
            action = policy(rng, observation)
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
