import os
import matplotlib as plt
from get_script_dir import *
from run_stdp_alpha_forced_pl import *
from test_STDP_lib import *

if __name__ == "__main__":
    current_dir = get_script_dir()
    config_file = os.path.join(current_dir, "config_sim_STDP.yaml")
    df_w = run_stdp_alpha_forced_pl(config_file)
    #print("df_w",df_w)
    config_check_stdp_filename = "config_check_STDP.yaml"
    config_file = os.path.join(current_dir, config_check_stdp_filename)
    test_stdp_main(config_check_stdp_filename)
    test_stdp_main()
    plt.show()
    