#!/usr/bin/env python
# coding: utf-8
#  dump_failed_tests_config.py
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


import yaml

def dump_failed_tests_config(config,failureList,filename):


    failed_config = {}

    W_init = []
    [W_init.append(config['W_init'][i]) for i in failureList]
    dendritic_delay = []
    [dendritic_delay.append(config['dendritic_delay_ms'][i]) for i in failureList]
    axonal_delay = []
    [axonal_delay.append(config['axonal_delay_ms'][i]) for i in failureList]
    spike_train_pre = []
    [spike_train_pre.append(config['spike_train_pre_ms'][i]) for i in failureList]
    spike_train_post = []
    [spike_train_post.append(config['spike_train_post_ms'][i]) for i in failureList]
    
    failed_config['W_init'] = W_init
    failed_config['dendritic_delay'] = dendritic_delay
    failed_config['axonal_delay'] = axonal_delay
    failed_config['spike_train_pre'] = spike_train_pre
    failed_config['spike_train_post'] = spike_train_post


    with open(filename, 'w') as file:
        yaml.dump(failed_config, file)

    return
