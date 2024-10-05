import numpy as np
# from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT
# from ..videogpt.reward_models import LOAD_REWARD_MODEL_DICT
# from viper_rl.dreamerv3 import embodied
# from viper_rl.dreamerv3.embodied import wrappers
from flax.training import checkpoints
import os
import pathlib
import sys
import warnings
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.dreamerv3 import embodied
from viper_rl.dreamerv3.embodied import wrappers
from train_videogpt import collect_data
from flax.training import checkpoints


# noinspection DuplicatedCode
def load_model(argv=None):
    directory = pathlib.Path(__file__).resolve()
    directory = directory.parent
    sys.path.append(str(directory.parent))
    sys.path.insert(0, os.path.realpath(os.path.join(__file__, '..', '..')))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Add parent directory to sys.path
    sys.path.insert(0, parent_dir)

    from viper_rl.dreamerv3 import agent as agt
    from viper_rl.dreamerv3 import embodied
    from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

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

