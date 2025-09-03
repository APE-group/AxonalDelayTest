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
from sim_stdp_alpha_forced_pl_lib import sim_stdp_alpha_forced_pl
from predict_stdp_alpha_forced_pl_lib import predict_stdp_alpha_forced_pl
from compare_sim_prediction_lib import compare_csv_files
from add_rand_events_lib import add_rand_events
from utils_lib import *


if __name__ == "__main__":
    config = read_config("config_sim_test_Ax_and_Dendr_Delay_STDP.yaml")
    config = add_rand_events(config)

    with open("current_config.yaml", "w") as file:
        yaml.dump(config, file)

    output_files_list = config["output_files_list"]
    output_files_list.append("current_config.yaml")

        
    #################################
    # SIMULATION WITH AXONAL DELAY
    #################################

    config["axonal_support"] = True

    # Simulation
    df_w, sim_summary = sim_stdp_alpha_forced_pl(config,prefix="AxD_")
    
    # Convert the dictionary to a Pandas DataFrame
    df_sim_summary = pd.DataFrame.from_dict(sim_summary, orient='index')
    
    # Save to CSV
    output_AxD_sim_csv = "AxD_sim_summary.csv"
    df_sim_summary.to_csv(output_AxD_sim_csv, index=False)
    print(f"Simulation summary saved to {output_AxD_sim_csv}.")
    output_files_list.append(output_AxD_sim_csv)

    
    #################################
    # PREDICTION WITH AXONAL DELAY
    #################################

    # Prediction
    prediction_summary = predict_stdp_alpha_forced_pl(config,prefix="AxD_")
    
    # Convert the dictionary to a Pandas DataFrame
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # Save to CSV
    output_AxD_pred_csv = "AxD_pred_summary.csv"
    df_summary.to_csv(output_AxD_pred_csv, index=False)
    print(f"Prediction summary saved to {output_AxD_pred_csv}.")
    output_files_list.append(output_AxD_pred_csv)


    #################################
    # SIMULATION WITHOUT AXONAL DELAY
    #################################
    
    config["axonal_support"] = False

    # Simulation
    df_w, sim_summary  = sim_stdp_alpha_forced_pl(config,prefix="noAxD_")
    
    # Convert the dictionary to a Pandas DataFrame
    df_sim_summary = pd.DataFrame.from_dict(sim_summary, orient='index')
    
    # Save to CSV
    output_noAxD_sim_csv = "noAxD_sim_summary.csv"
    df_sim_summary.to_csv(output_noAxD_sim_csv, index=False)
    print(f"Simulation summary saved to {output_noAxD_sim_csv}.")
    output_files_list.append(output_noAxD_sim_csv)

    
    #################################
    # PREDICTION WITHOUT AXONAL DELAY
    #################################

    # Prediction 
    prediction_summary = predict_stdp_alpha_forced_pl(config,prefix="noAxD_")
    
    # Convert the dictionary to a Pandas DataFrame
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # Save to CSV
    output_noAxD_pred_csv = "noAxD_pred_summary.csv"
    df_summary.to_csv(output_noAxD_pred_csv, index=False)
    print(f"Prediction summary saved to {output_noAxD_pred_csv}.")
    output_files_list.append(output_noAxD_pred_csv)

    
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
    match_ok, failureList = compare_csv_files(output_AxD_sim_csv,output_AxD_pred_csv,
                                              threshold=1e-8)
    if not match_ok:
        dump_failed_tests_config(config,failureList,'AxDsimVSAxDpred_failed_config.yaml')
    
    # Compare 2) noAxD_sim vs noAxD_pred
    print("")   
    print("Comparison noAxD_sim vs noAxD_pred:")
    print("-----------------------------------")
    match_ok, failureList = compare_csv_files(output_noAxD_sim_csv,output_noAxD_pred_csv,
                                              threshold=1e-8)
    if not match_ok:
        dump_failed_tests_config(config,failureList,'noAxDsimVSnoAxDpred_failed_config.yaml')

    if config["compare_AxDsimVSnoAxDsim"]:
        # Compare 3) AxD_sim vs noAxD_sim
        print("")
        print("Comparison AxD_sim vs noAxD_sim:")
        print("--------------------------------")
        match_ok, failureList = compare_csv_files(output_AxD_sim_csv,output_noAxD_sim_csv,
                                                  threshold=1e-8)
        if not match_ok:
            dump_failed_tests_config(config,failureList,'AxDsimVSnoAxDsim_failed_config.yaml')
    
    # Copy output files in folder
    if config["save_files_in_folder"]:
        foldername = copy_in_output_folder(output_files_list)
        print(f"All output files saved in {foldername}")

    # Show all plots
    if config["plot_display"]:
        plt.show()
    

