#!/usr/bin/env python
# coding: utf-8
#  read_config_lib.py
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

import os
import yaml 
from utils_lib import get_script_dir


def read_config(name_without_path):
    current_dir = get_script_dir()
    cfg_file = os.path.join(current_dir,name_without_path)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg_pms={}

        cfg_pms["described_syn"] = cfg["described_syn"]
        cfg_pms["add_rand_syn"] = int(cfg.get("add_rand_syn", 0))
        cfg_pms["N"] = cfg_pms["described_syn"] + cfg_pms["add_rand_syn"]
        assert cfg_pms["N"] > 0, ("Modify config.yaml: the sum of described_syn + add_rand_syn must be greater than zero") 
        cfg_pms["random_seed"] = int(cfg.get("random_seed", 123456))
        cfg_pms["max_rand_events_per_syn"] = int(cfg.get("max_rand_events_per_syn", 1))
            
        cfg_pms["verbose_sim"] = cfg.get("verbose_sim", False)
        cfg_pms["verbose_pred"] = cfg.get("verbose_pred", False)

        cfg_pms["sim_plot_save"] = cfg.get("sim_plot_save", False)
        cfg_pms["prediction_plot_save"] = cfg.get("prediction_plot_save", False)
        
        cfg_pms["plot_display"] = cfg.get("plot_display", False)
        cfg_pms["output_files_list"] = []
        
        cfg_pms["csv_file_pre"] = 'spikes_pre_neurons.csv'
        cfg_pms["csv_file_post"] = 'spikes_post_neurons.csv'
        
        cfg_pms["T_sim_ms"] = cfg["T_sim_ms"]
        cfg_pms["save_int_ms"] = cfg["save_int_ms"]
        cfg_pms["resolution"] = cfg["resolution"]
        
        # If user doesn't specify, default to [1..N]
        cfg_pms["start_syn"] = cfg.get("start_syn", 0)
        cfg_pms["end_syn"] = cfg.get("end_syn", cfg_pms["N"]-1)

        cfg_pms["W_init"] = []
        cfg_pms["dendritic_delay_ms"] = []
        cfg_pms["axonal_delay_ms"] = []
        cfg_pms["spike_train_pre_ms"] = []  
        cfg_pms["spike_train_post_ms"] = []  
          
        if cfg_pms["described_syn"] > 0:
            cfg_pms["W_init"] = cfg["W_init"]
            cfg_pms["dendritic_delay_ms"] = cfg["dendritic_delay_ms"]
            cfg_pms["axonal_delay_ms"] = cfg["axonal_delay_ms"]
            cfg_pms["spike_train_pre_ms"] = cfg["spike_train_pre_ms"]  
            cfg_pms["spike_train_post_ms"] = cfg["spike_train_post_ms"]  
            
        if cfg_pms["add_rand_syn"] > 0:      
            cfg_pms["W_min"] = cfg["W_min"]
            cfg_pms["W_max"] = cfg["W_max"]
            cfg_pms["min_dendritic_delay_ms"] = cfg["min_dendritic_delay_ms"]
            cfg_pms["max_dendritic_delay_ms"] = cfg["max_dendritic_delay_ms"]  
            cfg_pms["min_axonal_delay_ms"] = cfg["min_axonal_delay_ms"]
            cfg_pms["max_axonal_delay_ms"] = cfg["max_axonal_delay_ms"]
        
        cfg_pms["stdp_params"] = cfg.get("stdp_params", {"tau_plus": 20.0, "lambda": 0.9,
                                                        "alpha": 0.11, "mu": 0.4})
        cfg_pms["w_0"] = cfg["w_0"]

        cfg_pms["neu_params"] = cfg.get("neu_params", {"C_m": 250.0, "E_L": -70.0,
                                                       "t_ref": 2.0, "tau_m": 10,
                                                       "V_th": -10.0, "V_reset": -70.0})
        
        cfg_pms["forced_in_weight"] = cfg.get("forced_in_weight",  1000.0)
        cfg_pms["forced_out_weight"] = cfg.get("forced_out_weight", 1000.0)
        
        cfg_pms["plot_mm"] = cfg["plot_mm"]

        cfg_pms["compare_AxDsimVSnoAxDsim"] = cfg["compare_AxDsimVSnoAxDsim"]

        return cfg_pms
