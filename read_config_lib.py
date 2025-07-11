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
def read_config(name_without_path):
    current_dir = get_script_dir()
    config_file = os.path.join(current_dir,name_without_path)
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
        config_pms={}
        config_pms["axonal_support"] = cfg["axonal_support"]
            
        config_pms["verbose"] = cfg["verbose_sim"]
        config_pms["sim_plot_save"] = cfg["sim_plot_save"]
        config_pms["plot_display"] = cfg["plot_display"]
        
        config_pms["csv_file_pre"] = cfg["csv_file_pre"]
        config_pms["csv_file_post"] = cfg["csv_file_post"]
        
        config_pms["T_sim_ms"] = cfg["T_sim_ms"]
        config_pms["save_int_ms"] = cfg["save_int_ms"]
        config_pms["N"] = cfg["Total_number_described_synapses_for_sim"]
        
        # If user doesn't specify, default to [1..N]
        config_pms["start_syn"] = cfg.get("start_synapse", 1)
        config_pms["end_syn"] = cfg.get("end_synapse", N)
        
        config_pms["spike_train_pre_ms"] = cfg["spike_train_pre_ms"]  
        config_pms["spike_train_post_ms"] = cfg["spike_train_post_ms"]  
        
        config_pms["axonal_support"] = cfg["axonal_support"]
        if axonal_support:
            config_pms["dendritic_delay"]   = config_pms[dendritic_delay_ms"]
            config_pms["axonal_delay"]     = cfg["axonal_delay_ms"]
        else:
            config_pms["delay"]             = cfg["dendritic_delay_ms"]
        config_pms["W_init"]                = cfg["W_init"]
        
        config_pms["stdp_params"]           = cfg.get("stdp_params", {"tau_plus": 20.0, "lambda": 0.9,
                                                        "alpha": 0.11, "mu": 0.4})
        config_pms["forced_in_weight"]      = cfg.get("forced_in_weight",  1000.0)
        config_pms["forced_out_weight"]     = cfg.get("forced_out_weight", 1000.0)
        
        pconfig_pms["lot_marker_ms"]        = cfg["plot_marker_ms"]
        config_pms["plot_major_ticks_ms"]   = cfg["plot_major_ticks_ms"]
        
        config_pms["plot_mm"]               = cfg["plot_mm"]
        return config_pms