import os
import matplotlib as plt
import pandas as pd
from get_script_dir import *
from test_STDP_lib import *

if __name__ == "__main__":
    # prediction 1) Determine the path to your config file
    current_dir = get_script_dir()
    config_check_stdp_filename = "config_sim_test_Ax_and_Dendr_Delay_STDP.yaml"
    config_file = os.path.join(current_dir, config_check_stdp_filename)
    
    # prediction 2) Run the STDP routine, which returns a dictionary
    prediction_summary = test_stdp_main(config_file)
    
    # prediction 3) Convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value",
    #                                     "axonal_delay", "dendritic_delay"]
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # prediction 4) Save to CSV
    output_csv = "prediction_summary.csv"
    df_summary.to_csv(output_csv, index=False)
    print(f"Prediction summary saved to {output_csv}.")

    # Show all plots
    plt.show()
