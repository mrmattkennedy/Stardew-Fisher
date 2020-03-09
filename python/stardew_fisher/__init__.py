import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='StardewFisherEnv-v0',
    entry_point='stardew_fisher.envs:StardewFisherEnv',
    #timestep_limit=1000,
    reward_threshold=50000.0,
    nondeterministic = True,
)
