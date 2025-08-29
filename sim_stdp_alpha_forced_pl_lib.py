#!/usr/bin/env python
# coding: utf-8
#  sim_stdp_alpha_forced_pl_lib.py
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
#

import nest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                    color=get_syn_color(i), marker='.',label=f"Neu {i}")
    plt.xlim(tmin, tmax)
    plt.yticks(range(start_syn+offset, end_syn+offset+1))
    plt.tight_layout()
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

        
def sim_stdp_alpha_forced_pl(cfg,prefix=""):
    """
    For each synapse n:
      - PRE-neuron: iaf_psc_alpha(high threshold), forced by spike_generator_in.
      - POST-neuron: iaf_psc_alpha (high threshold), forced by spike_generator_out.
      - STDP synapse = stdp_pl_synapse_hom from pre -> post, with user-defined params.
      - Different spike trains are used for pre- and post-neurons.
      - We record spikes (pre & post), the evolving weight and the membrane potentials.
    """
    
    axonal_support        = cfg["axonal_support"]
        
    verbose_sim           = cfg["verbose_sim"]
    sim_plot_save         = cfg["sim_plot_save"]
    plot_display          = cfg["plot_display"]

    csv_file_pre          = cfg["csv_file_pre"]
    csv_file_post         = cfg["csv_file_post"]
    
    T_sim_ms              = cfg["T_sim_ms"]
    save_int_ms           = cfg["save_int_ms"]
    resolution            = cfg["resolution"]
    N                     = cfg["N"]

    # If user doesn't specify, default to [0..N-1]
    start_syn             = cfg["start_syn"]
    end_syn               = cfg["end_syn"]
    
    spike_train_pre_ms    = cfg["spike_train_pre_ms"]  
    spike_train_post_ms   = cfg["spike_train_post_ms"]  

    axonal_support        = cfg["axonal_support"]
    
    if axonal_support:
        dendritic_delay   = cfg["dendritic_delay_ms"]
        axonal_delay      = cfg["axonal_delay_ms"]
    else:
        delay             = cfg["dendritic_delay_ms"]
        axonal_delay      = cfg["axonal_delay_ms"]
        spike_train_pre_ms= [np.array(spike_train_pre_ms[i])+axonal_delay[i] for i in range(len(axonal_delay))]
        T_sim_ms          = T_sim_ms + 2*max(axonal_delay)
    
    W_init                = cfg["W_init"]

    stdp_params           = cfg["stdp_params"]

    forced_in_weight      = cfg["forced_in_weight"]
    forced_out_weight     = cfg["forced_out_weight"]

    plot_mm               = cfg["plot_mm"]

    # Validate range
    if not isinstance(start_syn, int) or not isinstance(end_syn, int):
        raise ValueError("start_synapse and end_synapse must be integers.")
    if start_syn < 0 or end_syn >= N or start_syn > end_syn:
        raise ValueError(f"Invalid synapse range: start={start_syn}, end={end_syn}, must be in [0..{N-1}] and start<=end.")

    
    #--------------------------------------------------------------------------
    # Reset and configure NEST kernel
    #--------------------------------------------------------------------------
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution})
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
    if verbose_sim: print(" Build the PRE-neuron ------------------")
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
    if verbose_sim: print(" Build the POST-neuron ------------------")
    post_neurons = nest.Create("iaf_psc_alpha", N)
    nest.SetStatus(post_neurons, {
        "V_th": -10.0,   # artificially high
        "E_L": -70.0,
        "V_reset": -70.0
    })

    
    #--------------------------------------------------------------------------
    # Create and connect spike generators to PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose_sim: print(" Create and connect spike generators ------------------")

    spike_generators_in = []
    
    #loop over N synapses
    for i in range(N):
        if verbose_sim: print("spike_train_pre_ms[",i,"]:",spike_train_pre_ms[i])
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
        if verbose_sim: print("spike_train_post_ms[",i,"]:",spike_train_post_ms[i])
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
    if verbose_sim: print("Connect pre_neuron -> post_neuron ------------------")
    connection_handles = []
    for i in range(N):
        if axonal_support:
            if verbose_sim: print("syn number:",i,"ad:",axonal_delay[i],"dd:",dendritic_delay[i],"W:",W_init[i])
            nest.Connect(
                pre_neurons[i],
                post_neurons[i],
                {"rule": "one_to_one"},
                {
                    "synapse_model": "my_stdp_pl_hom",  # The custom copy with user parameters
                    "weight": W_init[i],
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
                    "weight": W_init[i],
                    "delay": delay[i]
                }
            )            
        # Grab the connection handle for weight logging
        conn_obj = nest.GetConnections(pre_neurons[i], post_neurons[i])[0]
        connection_handles.append(conn_obj)

        
    #--------------------------------------------------------------------------
    # Create and connect spike recorders for PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose_sim: print("Create and connect spike recorders  ------------------")
    spike_rec_pre  = nest.Create("spike_recorder")
    spike_rec_post = nest.Create("spike_recorder")

    nest.Connect(pre_neurons,  spike_rec_pre,  {"rule": "all_to_all"})
    nest.Connect(post_neurons, spike_rec_post, {"rule": "all_to_all"})

    
    #--------------------------------------------------------------------------
    # Create and connect multimeters for PRE- and POST-neurons
    #--------------------------------------------------------------------------
    if verbose_sim: print("Create and connect multimeters  ------------------")

    mm_pre = list(range(start_syn-1, end_syn))
    mm_post = list(range(start_syn-1, end_syn))

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
    if verbose_sim: print("Simulate in steps  ------------------")
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
    df_w.to_csv(prefix+"simulated_synaptic_evolution.csv", index=False)
    print("SIM: Saved synaptic weight evolution to 'simulated_synaptic_evolution.csv'")

    
    #--------------------------------------------------------------------------
    # Retrieve and save spike data
    #--------------------------------------------------------------------------
    offset = 1
    events_pre = spike_rec_pre.get("events")
    df_pre = pd.DataFrame({
        #"senders": events_pre["senders"],
        "senders": np.asarray(events_pre["senders"], dtype=int) - offset,
        "times":   np.around(events_pre["times"],decimals=int(np.log(1/resolution)))
    })
    df_pre.to_csv(prefix+csv_file_pre, index=False)
    print("SIM: Saved spikes of pre_neurons to", csv_file_pre)

    events_post = spike_rec_post.get("events")
    df_post = pd.DataFrame({
        #"senders": events_post["senders"],
        "senders": np.asarray(events_post["senders"], dtype=int) - offset,
        "times":   np.around(events_post["times"],decimals=int(np.log(1/resolution)))
    })
    df_post.to_csv(prefix+csv_file_post, index=False)  
    print("SIM: Saved spikes of post_neurons to", csv_file_post)

    
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
    plt.figure()
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
    plt.xlabel("Time (ms)")
    plt.ylabel("Synaptic Weight")
    plt.title(f"{prefix}SIM: Synaptic evolution (Syn {start_syn}…{end_syn})")
    
    if sim_plot_save:
        plt.savefig(prefix+"simulated_synaptic_evolution.png", dpi=150)
    print("SIM: Saved synaptic weight plot to 'simulated_synaptic_evolution.png'")

    
    #--------------------------------------------------------------------------
    # Plot raster of PRE- and POST-neurons
    #--------------------------------------------------------------------------
    pre_dict  = df_to_spikedict(df_pre)
    post_dict = df_to_spikedict(df_post)
    tmin, tmax = 0, T_sim_ms                    

    plot_raster(pre_dict, 0, tmin, tmax,
                start_syn, end_syn, prefix+"SIM: PRE-neurons",
                prefix+"simulated_presynneu_raster.png" if sim_plot_save else None)

    plot_raster(post_dict, N, tmin, tmax,
                start_syn, end_syn, prefix+"SIM: POST-neurons",
                prefix+"simulated_postsynneu_raster.png" if sim_plot_save else None)

    
    #--------------------------------------------------------------------------
    # Plot membrane potentials
    #--------------------------------------------------------------------------
    if plot_mm:
        if n_mm_pre > 0:
            plt.figure(prefix+'Membrane Voltage PRE (iaf_psc_alpha)')
            for i in reversed(range(n_mm_pre)):
                plt.subplot(n_mm_pre,1,n_mm_pre-i)
                plt.plot(res_pre[i]['times'], res_pre[i]['V_m'], label='neu '+str(start_syn+i))
                plt.legend()
                plt.xlim(0, T_sim_ms)
                plt.xticks(np.arange(0, T_sim_ms + 1, 10))
                plt.ylabel('Vm [mV]')
                if i==n_mm_pre-1:
                    plt.title(prefix+'SIM: PRE-neurons')
            plt.xlabel('Time [ms]')
    
        if n_mm_post > 0:
            plt.figure(prefix+'Membrane Voltage POST (iaf_psc_alpha)')
            for i in reversed(range(n_mm_post)):
                plt.subplot(n_mm_post,1,n_mm_post-i)
                plt.plot(res_post[i]['times'], res_post[i]['V_m'], label='neu '+str(start_syn+i))
                plt.legend()
                plt.xlim(0, T_sim_ms)
                plt.xticks(np.arange(0, T_sim_ms + 1, 10))
                plt.ylabel('Vm [mV]')
                if i==n_mm_post-1:
                    plt.title(prefix+'SIM: POST-neurons')
            plt.xlabel('Time [ms]')
    
    
    # ------------------------------------------------------------------
    # Pack minimal summary
    # ------------------------------------------------------------------
    sim_summary = {
        i: {"syn_ID": i,
            "start_syn_value": W_init[i],#df_w[f"w_{i}"].iloc[0],
            "final_syn_value": df_w[f"w_{i}"].iloc[-1]}
        for i in range(start_syn, end_syn+1)
    }
    return df_w, sim_summary, plot_display
