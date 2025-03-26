import os
import matplotlib as plt
from get_script_dir import *
from run_stdp_alpha_forced_pl import *
from test_STDP_lib import *

if __name__ == "__main__":
    # sim 1)
    current_dir = get_script_dir()
    config_file = os.path.join(current_dir, "config_sim_STDP.yaml")
    df_w = run_stdp_alpha_forced_pl(config_file)
    #print("df_w",df_w)

    # prediction 1) Run the STDP routine, which returns a dictionary
    config_check_stdp_filename = "config_check_STDP.yaml"
    config_file = os.path.join(current_dir, config_check_stdp_filename)
    prediction_summary = test_stdp_main(config_file)
    
    # prediction 2) Convert the dictionary to a Pandas DataFrame
    #    Each dictionary entry has keys: ["syn_ID", "start_syn_value", "final_syn_value",
    #                                     "axonal_delay", "dendritic_delay"]
    df_summary = pd.DataFrame.from_dict(prediction_summary, orient='index')
    
    # prediction 3) Save to CSV
    output_csv = "prediction_summary.csv"
    df_summary.to_csv(output_csv, index=False)
    print(f"Prediction summary saved to {output_csv}.")

    # Show all plots
    plt.show()
    