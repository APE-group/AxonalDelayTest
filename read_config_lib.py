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

from get_script_dir import get_script_dir
def read_config(name_without_path):
    current_dir = get_script_dir()
    cfg_file = os.path.join(current_dir,name_without_path)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg_pms={}
        cfg_pms["random_seed"] = int(cfg.get("random_seed", 123456))
        cfg_pms["described_syn"] = cfg["Total_number_described_synapses_for_sim"]
        cfg_pms["add_rand_syn"] = int(cfg.get("add_rand_syn", 0))
        cfg_pms["N"] = cfg_pms["described_syn"] + cfg_pms["add_rand_syn"]
        cfg_pms["max_rand_events_per_syn"] = int(cfg.get("max_rand_events_per_syn", 1))
        
        cfg_pms["axonal_support"] = cfg["axonal_support"]
            
        cfg_pms["verbose_sim"] = cfg.get("verbose_sim", True)
        cfg_pms["sim_plot_save"] = cfg["sim_plot_save"]
        cfg_pms["plot_display"] = cfg["plot_display"]
        
        cfg_pms["csv_file_pre"] = cfg["csv_file_pre"]
        cfg_pms["csv_file_post"] = cfg["csv_file_post"]
        
        cfg_pms["T_sim_ms"] = cfg["T_sim_ms"]
        cfg_pms["save_int_ms"] = cfg["save_int_ms"]
        
        # If user doesn't specify, default to [1..N]
        cfg_pms["start_syn"] = cfg.get("start_syn", 0)
        cfg_pms["end_syn"] = cfg.get("end_syn", cfg_pms["N"]-1)
        
        cfg_pms["spike_train_pre_ms"] = cfg["spike_train_pre_ms"]  
        cfg_pms["spike_train_post_ms"] = cfg["spike_train_post_ms"]  
        cfg_pms["output_sim_csv"] = cfg.get("output_sim_csv","sim_summary.csv")
        cfg_pms["output_pred_csv"] = cfg.get("output_pred_csv","pred_summary.csv")
        
        cfg_pms["axonal_support"] = cfg["axonal_support"]
        cfg_pms["dendritic_delay_ms"]   = cfg["dendritic_delay_ms"]
        cfg_pms["min_dendritic_delay_ms"]   = cfg["min_dendritic_delay_ms"]
        cfg_pms["max_dendritic_delay_ms"]   = cfg["max_dendritic_delay_ms"]  
        
        cfg_pms["axonal_delay_ms"]     = cfg["axonal_delay_ms"]
        cfg_pms["min_axonal_delay_ms"]     = cfg["min_axonal_delay_ms"]
        cfg_pms["max_axonal_delay_ms"]     = cfg["max_axonal_delay_ms"]
        
        cfg_pms["W_init"]                = cfg["W_init"]
        cfg_pms["W_min"]                = cfg["W_min"]
        cfg_pms["W_max"]                = cfg["W_max"]
        
        cfg_pms["stdp_params"]           = cfg.get("stdp_params", {"tau_plus": 20.0, "lambda": 0.9,
                                                        "alpha": 0.11, "mu": 0.4})
        cfg_pms["w_0"] = cfg["w_0"]
        cfg_pms["forced_in_weight"]      = cfg.get("forced_in_weight",  1000.0)
        cfg_pms["forced_out_weight"]     = cfg.get("forced_out_weight", 1000.0)
        
        cfg_pms["plot_marker_ms"]        = cfg["plot_marker_ms"]
        cfg_pms["plot_major_ticks_ms"]   = cfg["plot_major_ticks_ms"]
        
        cfg_pms["plot_mm"]               = cfg["plot_mm"]
        cfg_pms["prediction_plot_save"]  = cfg["prediction_plot_save"]

        cfg_pms["csv_file_pre"]  = cfg["csv_file_pre"]
        cfg_pms["csv_file_post"] = cfg["csv_file_post"]
        cfg_pms["verbose_pred"]  = cfg.get("verbose_prediction_summary", True)

        cfg_pms["stdp_params"]  = cfg["stdp_params"]

        assert (cfg_pms["N"] + cfg_pms["add_rand_syn"] > 0)

        return cfg_pms
