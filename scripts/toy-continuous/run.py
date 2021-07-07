import jax
import tax
import hydra
import tqdm
from mlrec.recorder import Recorder
from common import setup
from common import save_conf


tax.set_platform("cpu")


@hydra.main(config_path="conf", config_name="conf")
def main(conf):
    rng = jax.random.PRNGKey(conf.seed)
    rec = Recorder(output_dir=".")
    save_conf(conf, rec)

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
        info['epoch'] = e
        rec.write(info)


if __name__ == "__main__":
    main()
