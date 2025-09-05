#!/usr/bin/env python
# coding: utf-8
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
from copy import deepcopy

def add_rand_events(config_pms, additional_syn_N=3, max_event_N=5):
    """Append `additional_syn_N` synthetic synapses without disturbing the
    ones that were already there.  The first synapse always uses seed
    base_seed+1, the second base_seed+2, … so results are prefix‑invariant."""
    
    cfg               = deepcopy(config_pms)
    T_sim_ms          = cfg["T_sim_ms"]
    W_min             = cfg["W_min"]
    W_max             = cfg["W_max"]
    dmin              = cfg["min_dendritic_delay_ms"]
    dmax              = cfg["max_dendritic_delay_ms"]
    amin              = cfg["min_axonal_delay_ms"]
    amax              = cfg["max_axonal_delay_ms"]
    base_seed         = cfg["random_seed"]
    additional_syn_N  = cfg["add_rand_syn"]
    max_event_N       = cfg["max_rand_events_per_syn"]
    
    
    for syn_idx in range(1, additional_syn_N+1):
        rng = random.Random(base_seed + syn_idx)

        cfg["W_init"].append(round(rng.uniform(W_min, W_max), 1))
        dd = round(rng.uniform(dmin, dmax), 1)
        cfg["dendritic_delay_ms"].append(dd)
        ad = round(rng.uniform(amin, amax), 1)
        cfg["axonal_delay_ms"].append(ad)
        
        guard_last_prespike_ms = 20 + amax         # last pre-spike at fixed time before simulation ends
        guard_ms = guard_last_prespike_ms + 10   # 10ms before last pre-spike
        assert (guard_ms > guard_last_prespike_ms and guard_ms < T_sim_ms), "Please, augment T_sim_ms in config file"

        def one_train():
            #assuming ~5ms the max reaction time of the neuron to the external stimulus
            n_ev = rng.randint(1, max_event_N)
            # use a set to avoid duplicates; round to 0.1 ms resolution
            return sorted({round(rng.uniform(0.1, (T_sim_ms-guard_ms)), 1) for _ in range(n_ev)})
        cfg["spike_train_pre_ms"].append(one_train()+[round(float(T_sim_ms-guard_last_prespike_ms),1)])
        cfg["spike_train_post_ms"].append(one_train())
        
    return cfg


