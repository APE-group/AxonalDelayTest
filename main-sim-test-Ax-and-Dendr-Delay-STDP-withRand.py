#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  main-sim-test-Ax-and-Dendr-Delay-STDP.py
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
import matplotlib as plt
import pandas as pd
#import numpy as np
#import random
from read_config_lib import read_config
from get_script_dir import *
from sim_stdp_alpha_forced_pl_lib import sim_stdp_alpha_forced_pl
from predict_stdp_alpha_forced_pl_lib import predict_stdp_alpha_forced_pl
from compare_sim_prediction_lib import compare_csv_files
from add_rand_events_lib import add_rand_events
    
    
if __name__ == "__main__":
    config_pms = read_config("config_sim_test_Ax_and_Dendr_Delay_STDP.yaml")
    config_pms = add_rand_events(config_pms, additional_syn_N = 1, max_event_N = 5)
    
    # sim 1)
    df_w, sim_summary, plot_display = sim_stdp_alpha_forced_pl(config_pms)
    #print("df_w",df_w)
    
    # sim 2)convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value"]
    df_sim_summary = pd.DataFrame.from_dict(sim_summary, orient='index')

    
    # sim 3) Save to CSV
    output_sim_csv = config_pms["output_sim_csv"]
    df_sim_summary.to_csv(output_sim_csv, index=False)
    print(f"Simulation summary saved to {output_sim_csv}.")

    # prediction 1) Run the STDP routine, which returns a dictionary

    prediction_summary = predict_stdp_alpha_forced_pl(config_pms)
    
    # prediction 2) Convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value",
    #                                     "axonal_delay", "dendritic_delay"]
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # prediction 3) Save to CSV
    output_pred_csv = config_pms["output_pred_csv"]
    df_summary.to_csv(output_pred_csv, index=False)
    print(f"Prediction summary saved to {output_pred_csv}.")

    
    # 4) Compare the two CSVs
    match_ok = compare_csv_files(output_sim_csv, output_pred_csv, threshold=1e-8)
    if match_ok:
        print("Both CSV files match within threshold.")
    else:
        print("Mismatch found above threshold.")
        assert("MISMATCH DETECTED")

    # Show all plots
    if plot_display:
        plt.show()
    

