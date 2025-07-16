#  sim_stdp_alpha_forced_pl_lib_zero_based.py
#  Version: 2025‑07‑16  —  IDs in all saved artefacts run 0 … N‑1
#  (everything else identical to the previous 1‑based version)
#
#  Copyright © 2025   Pier Stanislao Paolucci <pier.paolucci@roma1.infn.it>
#  Copyright © 2025   Elena Pastorelli       <elena.pastorelli@roma1.infn.it>
#  SPDX‑License‑Identifier: GPL‑3.0‑or‑later
#
# ----------------------------------------------------------------------

import nest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Top‑level API
# ----------------------------------------------------------------------
def get_syn_color(k):      # cycle through C0…C9
    return f"C{k % 10}"

def plot_raster(spike_dict, offset, tmin, tmax, start_syn, end_syn, label, fname):
    plt.figure()
    plt.title(label)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")

    for i in range(start_syn, end_syn+1):
        times = spike_dict.get(i+offset, [])#[i]
        plt.scatter(times, [i+offset]*len(times),
                    color=get_syn_color(i), marker='.')
    plt.xlim(tmin, tmax)
    plt.yticks(range(start_syn+offset, end_syn+offset+1))
    if fname:
        plt.savefig(fname)
        
def df_to_spikedict(df):
    """
    Convert a DataFrame with columns ['senders', 'times']
    into a dict  {id: sorted list of times}.
    """
    return (
        df.groupby("senders")["times"]
          .apply(lambda s: sorted(s.tolist()))
          .to_dict()
    )

        

def sim_stdp_alpha_forced_pl(cfg):
    """
    Run the same forced‑spike STDP experiment as before **but save every
    external identifier (spike senders, diagnostic dictionaries, plot labels,
    etc.) with 0‑based numbering**:

        • PRE neuron IDs      0 … N‑1
        • POST neuron IDs     N … 2N‑1
        • Synapse indices     0 … N‑1

    Internally NEST still uses its global IDs starting at 1; we subtract 1
    only when writing out.
    """

    # ------------------------------------------------------------------
    # --- 1.  Read config ------------------------------------------------
    # ------------------------------------------------------------------
    verbose_sim   = cfg["verbose_sim"]
    sim_plot_save = cfg["sim_plot_save"]
    plot_display  = cfg["plot_display"]

    csv_file_pre  = cfg["csv_file_pre"]
    csv_file_post = cfg["csv_file_post"]

    T_sim_ms      = cfg["T_sim_ms"]
    save_int_ms   = cfg["save_int_ms"]
    N             = cfg["N"]

    # default selection expressed *in zero‑based indices*
    start_syn     = cfg["start_syn"]
    end_syn       = cfg["end_syn"]

    spike_train_pre_ms  = cfg["spike_train_pre_ms"]
    spike_train_post_ms = cfg["spike_train_post_ms"]

    axonal_support      = cfg["axonal_support"]
    if axonal_support:
        dendritic_delay = cfg["dendritic_delay_ms"]
        axonal_delay    = cfg["axonal_delay_ms"]
    else:
        delay           = cfg["dendritic_delay_ms"]

    W_init              = cfg["W_init"]

    stdp_params         = cfg["stdp_params"]
                                      
    forced_in_weight    = cfg["forced_in_weight"]
    forced_out_weight   = cfg["forced_out_weight"]

    plot_marker_ms      = cfg["plot_marker_ms"]
    plot_major_ticks_ms = cfg["plot_major_ticks_ms"]
    plot_mm             = cfg["plot_mm"]

    # validate selection
    if start_syn < 0 or end_syn >= N or start_syn > end_syn:
        raise ValueError(f"Invalid synapse range: {start_syn=} {end_syn=} for N={N}")

    # ------------------------------------------------------------------
    # --- 2.  NEST set‑up ----------------------------------------------
    # ------------------------------------------------------------------
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    nest.set_verbosity("M_ERROR")

    if axonal_support:
        nest.CopyModel("stdp_pl_synapse_hom_ax_delay", "my_stdp_pl_hom",
                       stdp_params)
    else:
        nest.CopyModel("stdp_pl_synapse_hom", "my_stdp_pl_hom",
                       stdp_params)

    pre_neurons  = nest.Create("iaf_psc_alpha", N)
    post_neurons = nest.Create("iaf_psc_alpha", N)

    nest.SetStatus(pre_neurons + post_neurons,
                   {"V_th": -10.0, "E_L": -70.0, "V_reset": -70.0})

    # -- spike generators ------------------------------------------------
    spike_generators_in  = []
    spike_generators_out = []

    for i in range(N):
        sg_in  = nest.Create("spike_generator",
                             params={"spike_times": spike_train_pre_ms[i]})
        sg_out = nest.Create("spike_generator",
                             params={"spike_times": spike_train_post_ms[i]})

        spike_generators_in.append(sg_in)
        spike_generators_out.append(sg_out)

        nest.Connect(sg_in,  pre_neurons[i],
                     "one_to_one",
                     {"synapse_model": "static_synapse",
                      "weight": forced_in_weight, "delay": 1.0})

        nest.Connect(sg_out, post_neurons[i],
                     "one_to_one",
                     {"synapse_model": "static_synapse",
                      "weight": forced_out_weight, "delay": 1.0})

    # -- plastic synapses -----------------------------------------------
    connection_handles = []
    for i in range(N):
        syn_kwargs = {"synapse_model": "my_stdp_pl_hom",
                      "weight": W_init[i]}
        if axonal_support:
            syn_kwargs.update({"dendritic_delay": dendritic_delay[i],
                               "axonal_delay":    axonal_delay[i]})
        else:
            syn_kwargs.update({"delay": delay[i]})

        nest.Connect(pre_neurons[i], post_neurons[i], "one_to_one", syn_kwargs)
        connection_handles.append(nest.GetConnections(pre_neurons[i],
                                                      post_neurons[i])[0])

    # -- spike recorders & multimeters ----------------------------------
    spike_rec_pre  = nest.Create("spike_recorder")
    spike_rec_post = nest.Create("spike_recorder")
    nest.Connect(pre_neurons,  spike_rec_pre,  "all_to_all")
    nest.Connect(post_neurons, spike_rec_post, "all_to_all")

    mm_indices = list(range(start_syn, end_syn+1))
    mm_pre = nest.Create("multimeter", len(mm_indices),
                         {"record_from": ["V_m"], "interval": .1})
    mm_post = nest.Create("multimeter", len(mm_indices),
                          {"record_from": ["V_m"], "interval": .1})
    for k, idx in enumerate(mm_indices):
        nest.Connect(mm_pre[k],  pre_neurons[idx])
        nest.Connect(mm_post[k], post_neurons[idx])

    # ------------------------------------------------------------------
    # --- 3.  Simulate in chunks & log weights --------------------------
    # ------------------------------------------------------------------
    weight_records = []
    current_time   = 0.0
    while current_time < T_sim_ms:
        next_time = min(current_time + save_int_ms, T_sim_ms)
        nest.Simulate(next_time - current_time)
        current_time = next_time

        weight_records.append({"time_ms": current_time,
                               **{f"w_{j}": conn.get("weight")
                                  for j, conn in enumerate(connection_handles)}})

    df_w = pd.DataFrame(weight_records)
    df_w.to_csv("sim_stdp_evolution_line_summary.csv", index=False)
    print("SIM: weights → sim_stdp_evolution_line_summary.csv")

    # ------------------------------------------------------------------
    # --- 4.  Save spikes — NOW 0‑based IDs -----------------------------
    # ------------------------------------------------------------------
    def save_spikes(recorder, fname, offset=1):
        ev  = recorder.get("events")
        ids = np.asarray(ev["senders"], dtype=int) - offset   # shift to 0‑based
        pd.DataFrame({"senders": ids, "times": ev["times"]}).to_csv(fname,
                                                                    index=False)
        print(f"SIM: spikes → {fname}")

    save_spikes(spike_rec_pre,  csv_file_pre,  offset=1)
    save_spikes(spike_rec_post, csv_file_post, offset=1)   # works for PRE+POST

    # ------------------------------------------------------------------
    # --- 5.  Diagnostics & plots (labels 0‑based) ----------------------
    # ------------------------------------------------------------------
    # weight evolution
    plt.figure(figsize=(8, 6))
    for i in range(start_syn,end_syn+1):
        plt.plot(df_w["time_ms"], df_w[f"w_{i}"], marker="o",
                 linestyle="none", label=f"Syn {i}")
    plt.xlim(0, T_sim_ms)
    plt.xticks(np.arange(0, T_sim_ms+1, plot_major_ticks_ms))
    plt.xlabel("Time (ms)"); plt.ylabel("Weight")
    plt.title("SIM: stdp_pl_synapse_hom — weight trace")
    plt.legend()
    plt.tight_layout()
    if sim_plot_save:
        plt.savefig("simulated_synaptic_evolution.png", dpi=150)
        
    df_pre  = pd.read_csv(csv_file_pre)
    df_post = pd.read_csv(csv_file_post)

    pre_dict  = df_to_spikedict(df_pre)
    post_dict = df_to_spikedict(df_post)

    tmin, tmax = 0, T_sim_ms                    

    plot_raster(pre_dict, 0, tmin, tmax,
                start_syn, end_syn, "SIM: PRE raster",
                "simulated_presyn_raster.png" if sim_plot_save else None)

    plot_raster(post_dict, N, tmin, tmax,
                start_syn, end_syn, "SIM: POST raster",
                "simulated_postsyn_raster.png" if sim_plot_save else None)

    plt.tight_layout()

    if sim_plot_save:
        plt.savefig("sim_raster_alpha_forced_pl.png", dpi=150)

        
    # ------------------------------------------------------------------
    # --- 6.  Pack minimal summary -------------------------------------
    # ------------------------------------------------------------------
    sim_summary = {
        i: {"syn_ID": i,
            "start_syn_value": df_w[f"w_{i}"].iloc[0],
            "final_syn_value": df_w[f"w_{i}"].iloc[-1]}
        for i in range(start_syn, end_syn+1)
    }

    return df_w, sim_summary, plot_display
