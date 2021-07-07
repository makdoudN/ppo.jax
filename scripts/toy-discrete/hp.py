import os
import jax
import tax
import optax
import hydra
import tqdm
import optuna
import subprocess
import jax.numpy as jnp
import numpy as np
from common import setup


tax.set_platform("cpu")


@hydra.main(config_path=".", config_name="conf")
def hps(conf):
    def objective(trial):
        # Hyperparameter
        conf.lambda_ = trial.suggest_categorical("lambda_", [0.5, 0.75, 0.9, 0.93, 0.95, 0.99, 1.0])
        conf.value_cost = trial.suggest_categorical("value_cost", [1.0, 0.5, 0.25, 0.1, 0.01, 0.0])
        conf.entropy_cost = trial.suggest_categorical("entropy_cost", [1.0, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0])
        conf.policy_opt = trial.suggest_categorical("policy_opt", ['adabelief', 'adam', 'rmsprop'])
        conf.value_opt = trial.suggest_categorical("value_opt", ['adabelief', 'adam', 'rmsprop'])
        conf.value_opt_kwargs.learning_rate = \
            trial.suggest_categorical("value_lr", [1e-4, 1e-3, 1e-2, 5e-3, 5e-4])
        conf.policy_opt_kwargs.learning_rate = \
            trial.suggest_categorical("policy_lr", [1e-4, 1e-3, 1e-2, 5e-3, 5e-4])

        rng = jax.random.PRNGKey(conf.seed)
        state, alg = setup(conf)
        ncycles = conf.maxsteps // conf.horizon // conf.num_envs
        ncyles_per_epoch = ncycles // conf.nepochs
        for e in range(conf.nepochs):
            S = tax.Store()
            for _ in tqdm.trange(ncyles_per_epoch):
                data, info_interaction = alg['interaction'].send(state.params['policy'])
                rng, rng_update = jax.random.split(rng)
                state, info_update = alg['update'](rng_update, state, data)
                S.add(**info_update, **info_interaction)
            rng, rng_eval = jax.random.split(rng)
            info_eval = alg['evaluation'](rng_eval, state.params['policy'])
            info = S.get()
            info.update(info_eval)
            score = info['eval/score']
            trial.report(score, e)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    storage_name = f'sqlite:///../../../data/{conf.env}.db'
    cmd = f'optuna create-study --study-name {conf.env} --storage {storage_name} --direction maximize --skip-if-exists'
    sub = subprocess.call(cmd, shell=True)
    print(sub)

    pruner = optuna.pruners.SuccessiveHalvingPruner()
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.load_study(
        study_name=conf.env, pruner=pruner,
        sampler=sampler, storage=storage_name)

    study.optimize(objective, n_trials=1)


if __name__ == "__main__":
    hps()
