#!/usr/bin/env python
# coding: utf-8
#  main.py
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
import matplotlib.pyplot as plt
import pandas as pd
import random
import yaml
from read_config_lib import read_config
from get_script_dir import *
from sim_stdp_alpha_forced_pl_lib import sim_stdp_alpha_forced_pl
from predict_stdp_alpha_forced_pl_lib import predict_stdp_alpha_forced_pl
from compare_sim_prediction_lib import compare_csv_files
from add_rand_events_lib import add_rand_events
from dump_failed_tests_config import dump_failed_tests_config    
    
if __name__ == "__main__":
    config_pms = read_config("config_sim_test_Ax_and_Dendr_Delay_STDP.yaml")
    config_pms = add_rand_events(config_pms)

    with open('current_config.yaml', 'w') as file:
        yaml.dump(config_pms, file)
        
    #################################
    # SIMULATION WITH AXONAL DELAY
    #################################

    config_pms["axonal_support"] = True
    
    # sim AxD 1)
    df_w, sim_summary, plot_display = sim_stdp_alpha_forced_pl(config_pms,prefix="AxD_")
    #print("df_w",df_w)
    
    # sim AxD 2)convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value"]
    df_sim_summary = pd.DataFrame.from_dict(sim_summary, orient='index')
    
    # sim AxD 3) Save to CSV
    output_AxD_sim_csv = "AxD_sim_summary.csv"
    df_sim_summary.to_csv(output_AxD_sim_csv, index=False)
    print(f"Simulation summary saved to {output_AxD_sim_csv}.")

    #################################
    # PREDICTION WITH AXONAL DELAY
    #################################

    # prediction 1) Run the STDP routine, which returns a dictionary
    prediction_summary = predict_stdp_alpha_forced_pl(config_pms,prefix="AxD_")
    
    # prediction 2) Convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value",
    #                                     "axonal_delay", "dendritic_delay"]
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # prediction 3) Save to CSV
    output_AxD_pred_csv = "AxD_pred_summary.csv"
    df_summary.to_csv(output_AxD_pred_csv, index=False)
    print(f"Prediction summary saved to {output_AxD_pred_csv}.")

    #################################
    # SIMULATION WITHOUT AXONAL DELAY
    #################################
    
    config_pms["axonal_support"] = False

    # sim noAxD 1)
    df_w, sim_summary, plot_display = sim_stdp_alpha_forced_pl(config_pms,prefix="noAxD_")
    #print("df_w",df_w)
    
    # sim noAxD 2)convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value"]
    df_sim_summary = pd.DataFrame.from_dict(sim_summary, orient='index')
    
    # sim noAxD 3) Save to CSV
    output_noAxD_sim_csv = "noAxD_sim_summary.csv"
    df_sim_summary.to_csv(output_noAxD_sim_csv, index=False)
    print(f"Simulation summary saved to {output_noAxD_sim_csv}.")    

    #################################
    # PREDICTION WITHOUT AXONAL DELAY
    #################################

    # prediction 1) Run the STDP routine, which returns a dictionary
    prediction_summary = predict_stdp_alpha_forced_pl(config_pms,prefix="noAxD_")
    
    # prediction 2) Convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value",
    #                                     "axonal_delay", "dendritic_delay"]
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # prediction 3) Save to CSV
    output_noAxD_pred_csv = "noAxD_pred_summary.csv"
    df_summary.to_csv(output_noAxD_pred_csv, index=False)
    print(f"Prediction summary saved to {output_noAxD_pred_csv}.")
    
    #################################
    # COMPARISONS
    #################################

    print("")
    print("-----------------------------")
    print("Final comparisons")
    print("-----------------------------")
    
    # Compare 1) AxD_sim vs AxD_pred
    print("")
    print("Comparison AxD_sim vs AxD_pred:")
    print("-------------------------------")
    match_ok, failureList = compare_csv_files(output_AxD_sim_csv, output_AxD_pred_csv, threshold=1e-8)
    if not match_ok:
        dump_failed_tests_config(config_pms,failureList,'AxDsimVSAxDpred_failed_config.yaml')
    
    # Compare 2) noAxD_sim vs noAxD_pred
    print("")   
    print("Comparison noAxD_sim vs noAxD_pred:")
    print("-----------------------------------")
    match_ok, failureList = compare_csv_files(output_noAxD_sim_csv, output_noAxD_pred_csv, threshold=1e-8)
    if not match_ok:
        dump_failed_tests_config(config_pms,failureList,'noAxDsimVSnoAxDpred_failed_config.yaml')
    
    # Compare 3) AxD_sim vs noAxD_sim
    print("")
    print("Comparison AxD_sim vs noAxD_sim:")
    print("--------------------------------")
    match_ok, failureList = compare_csv_files(output_AxD_sim_csv, output_noAxD_sim_csv, threshold=1e-8)
    if not match_ok:
        dump_failed_tests_config(config_pms,failureList,'AxDsimVSnoAxDsim_failed_config.yaml')

        
    # Show all plots
    if plot_display:
        plt.show()
    

