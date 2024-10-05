import numpy as np
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT
#from ..videogpt.reward_models import LOAD_REWARD_MODEL_DICT
from viper_rl.dreamerv3 import embodied
from viper_rl.dreamerv3.embodied import wrappers
from flax.training import checkpoints


# noinspection DuplicatedCode
def load_model(argv=None):
    from viper_rl.dreamerv3 import agent as agt

    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    config.reward_model = 'dmc_anomaly_detection'
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    reward_model = LOAD_REWARD_MODEL_DICT[config.reward_model](
        task=config.task,
        compute_joint=config.reward_model_compute_joint,
        minibatch_size=config.reward_model_batch_size,
        encoding_minibatch_size=config.reward_model_batch_size,
        reward_model_device=config.jax.reward_model_device)
    print(reward_model([0, 1, 2]))


if __name__ == '__main__':
    load_model()

