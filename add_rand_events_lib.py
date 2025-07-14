#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  add_rand_events_lib.py
#  Copyright © 2025   Pier Stanislao Paolucci   <pier.paolucci@roma1.infn.it>
#  Copyright © 2025   Elena Pastorelli          <elena.pastorelli@roma1.infn.it>
#
#  SPDX-License-Identifier: GPL-3.0-only
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import random

def add_rand_events(config_pms, additional_syn_N = 3, max_event_N = 5):
    rand_W_init = [random.uniform(config_pms["W_min"], config_pms["W_max"]) for _ in range(additional_syn_N)]
    rounded_rand_W_init = [round(_,1) for _ in rand_W_init]
    config_pms["W_init"] += rounded_rand_W_init

    rand_dendritic_delay_ms = [random.uniform(config_pms["min_dendritic_delay_ms"], config_pms["max_dendritic_delay_ms"]) for _ in range(additional_syn_N)]
    rounded_rand_dendritic_delay_ms = [round(_,1) for _ in rand_dendritic_delay_ms]
    config_pms["dendritic_delay_ms"] += rounded_rand_dendritic_delay_ms

    rand_axonal_delay_ms = [random.uniform(config_pms["min_axonal_delay_ms"], config_pms["max_axonal_delay_ms"]) for _ in range(additional_syn_N)]
    rounded_rand_axonal_delay_ms = [round(_,1) for _ in rand_axonal_delay_ms]
    config_pms["axonal_delay_ms"] += rounded_rand_axonal_delay_ms

    rounded_rand_spike_train_pre_ms = [
    sorted({
        round(random.uniform(0.1, config_pms["T_sim_ms"]), 1)
        for _ in range(random.randint(1, max_event_N))
    })
    for _ in range(additional_syn_N)
    ]
    config_pms["spike_train_pre_ms"] += rounded_rand_spike_train_pre_ms

    rounded_rand_spike_train_post_ms = [
    sorted({
        round(random.uniform(0.1, config_pms["T_sim_ms"]), 1)
        for _ in range(random.randint(1, max_event_N))
    })
    for _ in range(additional_syn_N)
    ]
    config_pms["spike_train_post_ms"] += rounded_rand_spike_train_post_ms

    config_pms["N"] += additional_syn_N
    
    return config_pms
