# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
My Anymal locomotion environment with velocity command visualization.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-MyAnymal-Flat-v0",
    entry_point="isaaclab_tasks.direct.my_anymal_quadruped.my_anymal_c_env:MyAnymalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.my_anymal_quadruped.my_anymal_c_env_cfg:MyAnymalFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)
