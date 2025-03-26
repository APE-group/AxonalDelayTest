import os
from get_script_dir import *
from run_stdp_alpha_forced_pl import *

if __name__ == "__main__":
    current_dir = get_script_dir()
    config_file = os.path.join(current_dir, "config_sim_test_Ax_and_Dendr_Delay_STDP.yaml")
    df_w = run_stdp_alpha_forced_pl(config_file)
    print("df_w",df_w)
    plt.show()
