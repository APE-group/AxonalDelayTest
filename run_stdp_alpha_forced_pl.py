import nest
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def run_stdp_alpha_forced_pl(config_file):
    """
    For each synapse n:
      - PRE-neuron: iaf_psc_alpha(high threshold), forced by spike_generator_in.
      - POST-neuron: iaf_psc_alpha (high threshold), forced by spike_generator_out.
      - STDP synapse = stdp_pl_synapse_hom from pre -> post, with user-defined params.
      - Different spike trains are used for pre- and post-neurons.
      - We record spikes (pre & post), the evolving weight and the membrane potentials.
    """

    #--------------------------------------------------------------------------
    # Read YAML config
    #--------------------------------------------------------------------------
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    verbose               = cfg["verbose"]
    T_sim_ms              = cfg["T_sim_ms"]
    save_int_ms           = cfg["save_int_ms"]
    N                     = cfg["N"]

    spike_train_pre_ms    = cfg["spike_train_pre_ms"]  
    spike_train_post_ms   = cfg["spike_train_post_ms"]  

    axonal_support        = cfg["axonal_support"]
    if axonal_support:
        dendritic_delay   = cfg["dendritic_delay"]
        axonal_delay      = cfg["axonal_delay"]
    else:
        delay             = cfg["dendritic_delay"]
    W0                    = cfg["W0"]

    stdp_params           = cfg.get("stdp_params", {})
    forced_in_weight      = cfg.get("forced_in_weight",  1000.0)
    forced_out_weight     = cfg.get("forced_out_weight", 1000.0)

    plot_marker_ms        = cfg["plot_marker_ms"]
    plot_major_ticks_ms   = cfg["plot_major_ticks_ms"]

    plot_mm               = cfg["plot_mm"]
    mm_pre                = cfg["mm_pre"]
    mm_post               = cfg["mm_post"]
    
    #--------------------------------------------------------------------------
    # Reset and configure NEST kernel
    #--------------------------------------------------------------------------
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    nest.set_verbosity('M_ERROR')

    #--------------------------------------------------------------------------
    # Create or set stdp_pl_synapse_hom in NEST 3.7
    #--------------------------------------------------------------------------
    if axonal_support:
        nest.CopyModel("stdp_pl_synapse_hom_ax_delay", "my_stdp_pl_hom", stdp_params)
    else:
        nest.CopyModel("stdp_pl_synapse_hom", "my_stdp_pl_hom", stdp_params)

    #--------------------------------------------------------------------------
    # Build the PRE-neuron (iaf_psc_alpha) with high threshold
    #    so it won't spike unless forced by the output generator
    #--------------------------------------------------------------------------
    if verbose: print(" Build the PRE-neuron ------------------")
    pre_neurons = nest.Create("iaf_psc_alpha", N)
    nest.SetStatus(pre_neurons, {
        "V_th": -10.0,   # artificially high
        "E_L": -70.0,
        "V_reset": -70.0
    })

    #--------------------------------------------------------------------------
    # Build the POST-neuron (iaf_psc_alpha) with high threshold
    #    so it won't spike unless forced by the output generator
    #--------------------------------------------------------------------------
    if verbose: print(" Build the POST-neuron ------------------")
    post_neurons = nest.Create("iaf_psc_alpha", N)
    nest.SetStatus(post_neurons, {
        "V_th": -10.0,   # artificially high
        "E_L": -70.0,
        "V_reset": -70.0
    })
    
    #--------------------------------------------------------------------------
    # Create and connect spike generators to PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose: print(" Create and connect spike generators ------------------")

    spike_generators_in = []
    
    #loop over N synapses
    for i in range(N):

        sg_in = nest.Create("spike_generator", params={
            "spike_times": spike_train_pre_ms[i]
        })
        spike_generators_in.append(sg_in)

        # Connect input generator -> pre_neuron[i] with large weight
        nest.Connect(
            sg_in,
            pre_neurons[i],
            {"rule": "one_to_one"},
            {"synapse_model": "static_synapse", "weight": forced_in_weight, "delay": 1.0}
        )

    spike_generators_out = []

    #loop over N synapses
    for i in range(N):

        sg_out = nest.Create("spike_generator", params={
            "spike_times": spike_train_post_ms[i]
        })
        spike_generators_out.append(sg_out)


        # Connect output generator -> post_neuron[i] with large weight
        nest.Connect(
            sg_out,
            post_neurons[i],
            {"rule": "one_to_one"},
            {"synapse_model": "static_synapse", "weight": forced_out_weight, "delay": 1.0}
        )

    #--------------------------------------------------------------------------
    # Connect PRE-neuron -> POST-neuron with the custom "my_stdp_pl_hom" synapse
    #--------------------------------------------------------------------------
    if verbose: print("Connect pre_neuron -> post_neuron ------------------")
    connection_handles = []
    for i in range(N):
        if axonal_support:
            nest.Connect(
                pre_neurons[i],
                post_neurons[i],
                {"rule": "one_to_one"},
                {
                    "synapse_model": "my_stdp_pl_hom",  # The custom copy with user parameters
                    "weight": W0[i],
                    "dendritic_delay": dendritic_delay[i],
                    "axonal_delay": axonal_delay[i]
                }
            )
        else:
            nest.Connect(
                pre_neurons[i],
                post_neurons[i],
                {"rule": "one_to_one"},
                {
                    "synapse_model": "my_stdp_pl_hom",  # The custom copy with user parameters
                    "weight": W0[i],
                    "delay": delay[i]
                }
            )            
        # Grab the connection handle for weight logging
        conn_obj = nest.GetConnections(pre_neurons[i], post_neurons[i])[0]
        connection_handles.append(conn_obj)
    #--------------------------------------------------------------------------
    # Create and connect spike recorders for PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose: print("Create and connect spike recorders  ------------------")
    spike_rec_pre  = nest.Create("spike_recorder")
    spike_rec_post = nest.Create("spike_recorder")

    nest.Connect(pre_neurons,  spike_rec_pre,  {"rule": "all_to_all"})
    nest.Connect(post_neurons, spike_rec_post, {"rule": "all_to_all"})

    #--------------------------------------------------------------------------
    # Create and connect multimeters for PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose: print("Create and connect multimeters  ------------------")
    n_mm_pre = len(mm_pre)
    n_mm_post = len(mm_post)
    
    if n_mm_pre > 0:
        mm_pre = nest.Create('multimeter', n_mm_pre, {'record_from': ["V_m"], 'interval': .1})
        [nest.Connect(mm_pre[i], pre_neurons[i]) for i in range (n_mm_pre)]
    if n_mm_post > 0:
        mm_post = nest.Create('multimeter', n_mm_post, {'record_from': ["V_m"], 'interval': .1})
        [nest.Connect(mm_post[i], post_neurons[i]) for i in range (n_mm_post)]

    #--------------------------------------------------------------------------
    # Simulation in steps, log weight changes
    #--------------------------------------------------------------------------
    if verbose: print("Simulate in steps  ------------------")
    current_time = 0.0
    weight_records = []

    while current_time < T_sim_ms:
        next_time = min(current_time + save_int_ms, T_sim_ms)
        nest.Simulate(next_time - current_time)
        current_time = next_time

        record_now = {"time_ms": current_time}
        for j, conn_obj in enumerate(connection_handles):
            w_val = conn_obj.get("weight")
            record_now[f"w_{j}"] = w_val
        weight_records.append(record_now)
        
    #--------------------------------------------------------------------------
    # Save weight evolution to CSV
    #--------------------------------------------------------------------------
    df_w = pd.DataFrame(weight_records)
    df_w.to_csv("weights_alpha_forced_pl.csv", index=False)
    print("Saved synaptic weight evolution to 'weights_alpha_forced_pl.csv'")

    #--------------------------------------------------------------------------
    # Retrieve and save spike data
    #--------------------------------------------------------------------------
    events_pre = spike_rec_pre.get("events")
    df_pre = pd.DataFrame({
        "senders": events_pre["senders"],
        "times":   events_pre["times"]
    })
    df_pre.to_csv("spikes_pre_neurons.csv", index=False)
    print("Saved spikes of pre_neurons to 'spikes_pre_neurons.csv'")

    events_post = spike_rec_post.get("events")
    df_post = pd.DataFrame({
        "senders": events_post["senders"],
        "times":   events_post["times"]
    })
    df_post.to_csv("spikes_post_neurons.csv", index=False)
    print("Saved spikes of post_neurons to 'spikes_post_neurons.csv'")

    #--------------------------------------------------------------------------
    # Retrieve multimeter data
    #--------------------------------------------------------------------------
    if n_mm_pre > 0:
        res_pre = [nest.GetStatus(mm_pre[i], 'events')[0] for i in range(n_mm_pre)]
    if n_mm_post > 0:
        res_post = [nest.GetStatus(mm_post[i], 'events')[0] for i in range(n_mm_post)]


    #--------------------------------------------------------------------------
    # Plot weight evolution with points
    #--------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for i in range(N):
        # Plot each weight as markers (no connecting lines)
        plt.plot(
            df_w["time_ms"],
            df_w[f"w_{i}"],
            marker='o',
            markersize=4,
            linestyle='none',   # no line
            label=f"Syn {i}"
        )

    plt.legend()
    plt.xlim(0, T_sim_ms)
    plt.xticks(np.arange(0, T_sim_ms + 1, plot_major_ticks_ms))
    plt.minorticks_on()
    plt.xlabel("Time (ms)")
    plt.ylabel("Synaptic Weight")
    plt.title("STDP with stdp_pl_synapse_hom (iaf_psc_alpha) - Forced Pre & Post Spikes")

    plt.tight_layout()
    plt.savefig("weights_alpha_forced_pl.png", dpi=150)
    print("Saved synaptic weight plot to 'weights_alpha_forced_pl.png'")

    #--------------------------------------------------------------------------
    # Plot raster of PRE- and POST-neurons
    #--------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.scatter(df_pre["times"], df_pre["senders"], s=5, c='tab:blue')
    plt.xlim(0, T_sim_ms)
    plt.xticks(np.arange(0, T_sim_ms + 1, plot_major_ticks_ms))
    plt.minorticks_on()
    plt.ylabel('PRE-neuron IDs')
    plt.title('Raster: PRE (iaf_psc_alpha)')

    plt.subplot(212)
    plt.scatter(df_post["times"], df_post["senders"], s=5, c='tab:red')
    plt.xlim(0, T_sim_ms)
    plt.xticks(np.arange(0, T_sim_ms + 1, plot_major_ticks_ms))
    plt.minorticks_on()
    plt.xlabel('Time (ms)')
    plt.ylabel('POST-neuron IDs')
    plt.title('Raster: POST (iaf_psc_alpha)')

    plt.tight_layout()
    plt.savefig("raster_alpha_forced_pl.png", dpi=150)
    print("Saved spike raster to 'raster_alpha_forced_pl.png'")

    #--------------------------------------------------------------------------
    # Plot membrane potentials
    #--------------------------------------------------------------------------
    if plot_mm:
        if n_mm_pre > 0:
            plt.figure('Membrane Voltage PRE (iaf_psc_alpha)')
            for i in range(n_mm_pre):
                plt.subplot(n_mm_pre,1,i+1)
                plt.plot(res_pre[i]['times'], res_pre[i]['V_m'], label='neu '+str(i))
                plt.legend()
                plt.xlim(0, T_sim_ms)
                plt.xticks(np.arange(0, T_sim_ms + 1, 10))
                plt.ylabel('Vm [mV]')
                if i==0:
                    plt.title('PRE-neurons')
            plt.xlabel('Time [ms]')
    
        if n_mm_post > 0:
            plt.figure('Membrane Voltage POST (iaf_psc_alpha)')
            for i in range(n_mm_post):
                plt.subplot(n_mm_post,1,i+1)
                plt.plot(res_post[i]['times'], res_post[i]['V_m'], label='neu '+str(i))
                plt.legend()
                plt.xlim(0, T_sim_ms)
                plt.xticks(np.arange(0, T_sim_ms + 1, 10))
                plt.ylabel('Vm [mV]')
                if i==0:
                    plt.title('POST-neurons')
            plt.xlabel('Time [ms]')
   
    return df_w
