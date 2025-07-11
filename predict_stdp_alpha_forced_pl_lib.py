#  predict_stdp_alpha_forced_pl_lib.py
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

import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Helper Functions
###############################################################################

def safe_get_list_item(lst, index, default_val):
    """
    Returns lst[index] if it exists, otherwise returns default_val.
    For partial lists or missing config.
    """
    if lst is None or index >= len(lst):
        return default_val
    return lst[index]


def load_spikes_pre(csv_file, N):
    """
    Reads a CSV file for pre-synaptic spikes with columns:
      senders, times
    Pre neurons: 1..N

    Returns:
      pre_spikes_dict : { i: list of float spike_time_ms }
      all_spike_times_ms : global list of times
      all_neuron_ids : global list of IDs
    """
    pre_spikes_dict = {i: [] for i in range(1, N+1)}
    all_spike_times_ms = []
    all_neuron_ids = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neuron_id = int(row["senders"])
            spike_time_ms = float(row["times"])
            if not (1 <= neuron_id <= N):
                raise ValueError(f"Pre neuron_id {neuron_id} out of range 1..{N}")

            pre_spikes_dict[neuron_id].append(spike_time_ms)
            all_spike_times_ms.append(spike_time_ms)
            all_neuron_ids.append(neuron_id)

    # Sort times
    for i in range(1, N+1):
        pre_spikes_dict[i].sort()

    return pre_spikes_dict, all_spike_times_ms, all_neuron_ids


def load_spikes_post(csv_file, N):
    """
    Reads a CSV file for post-synaptic spikes with columns:
      senders, times
    Post neurons: N+1..2N

    Returns:
      post_spikes_dict : { i: list of float spike_time_ms } 
                         (i in 1..N => post neuron = i+N)
      all_spike_times_ms : global list of times
      all_neuron_ids : global list of IDs
    """
    post_spikes_dict = {i: [] for i in range(1, N+1)}
    all_spike_times_ms = []
    all_neuron_ids = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neuron_id = int(row["senders"])
            spike_time_ms = float(row["times"])
            if not (N+1 <= neuron_id <= 2*N):
                raise ValueError(f"Post neuron_id {neuron_id} out of range {N+1}..{2*N}")

            i = neuron_id - N
            post_spikes_dict[i].append(spike_time_ms)

            all_spike_times_ms.append(spike_time_ms)
            all_neuron_ids.append(neuron_id)

    # Sort times
    for i in range(1, N+1):
        post_spikes_dict[i].sort()

    return post_spikes_dict, all_spike_times_ms, all_neuron_ids


###############################################################################
# STDP Function (Lumpsum LTP & LTD)
###############################################################################

def stdp_pl_synapse_hom_causal(
    syn_id,
    pre_spikes_dict, post_spikes_dict,
    arrived_pre_dict, arrived_post_dict,
    tau_plus_ms, lambda_pmt, alpha, mu, w_0,
    w_init,
    writer,
    initial_time_for_plot_ms
):
    """
    Lumpsum version of stdp_pl_synapse_hom for both LTD(dt<0) and LTP(dt>0).

    For each synapse 'syn_id':
      - We gather all (pre_spike, post_spike) pairs => each event has dt_ms= post_arr - pre_arr
      - If dt_ms>0 => LTP event => lumpsum with all events sharing the same (post arrival time, j_post)
      - If dt_ms<0 => LTD event => lumpsum with all events sharing the same (pre arrival time, i_pre)
      - Mark partial lines (w_after=None) for each partial event, then one lumpsum line with the real new weight.
    
    We store an 8-tuple for each event:
      (pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, w_before, dW, w_after)
    in the final 'trajectory'.

    We also log to CSV with columns:
      syn_id, event_idx,
      pre_spike_ms, post_spike_ms,
      pre_arr_ms, post_arr_ms,
      dt_ms, w_before, dW, w_after
    """
    # 1) Retrieve original raw and arrival times
    pre_times_raw = pre_spikes_dict[syn_id]
    post_times_raw = post_spikes_dict[syn_id]
    pre_times_arr = arrived_pre_dict[syn_id]
    post_times_arr = arrived_post_dict[syn_id]

    w = w_init
    trajectory = []

    # 2) "Fake" initial row => no real spikes yet
    writer.writerow([syn_id, 0, None, None, None, None, None, 0.0, w])
    trajectory.append((None, None, None, None, 0.0, 0.0, 0.0, w))

    # 3) Build global event list
    #   For each pre index => pre_t_raw, pre_t_arr
    #   For each post index => post_t_raw, post_t_arr => dt_ms
    all_events = []
    for i_pre, pre_t_raw in enumerate(pre_times_raw):
        pre_t_arr = pre_times_arr[i_pre]
        for j_post, post_t_raw in enumerate(post_times_raw):
            post_t_arr = post_times_arr[j_post]
            dt_ms = post_t_arr - pre_t_arr

            # event_time for lumpsum grouping:
            #  LTP => post arrival => lumpsum if (dt>0, same post_arr, same j_post)
            #  LTD => pre arrival  => lumpsum if (dt<0, same pre_arr, same i_pre)
            if dt_ms > 0:
                # lumpsum key => (post_t_arr, j_post)
                event_time = post_t_arr
            else:
                # dt_ms <= 0 => lumpsum key => (pre_t_arr, i_pre)
                event_time = pre_t_arr

            all_events.append((
                event_time,   # for sorting
                i_pre,
                j_post,
                dt_ms,
                pre_t_raw, 
                post_t_raw,
                pre_t_arr,
                post_t_arr
            ))

    # 4) Sort by event_time ascending
    all_events.sort(key=lambda x: x[0])

    # 5) Lumpsum logic
    event_idx = 1
    i = 0
    n_ev = len(all_events)

    while i < n_ev:
        (current_time, i_pre, j_post, dt_ms,
         pre_t_raw, post_t_raw,
         pre_t_arr, post_t_arr) = all_events[i]

        if dt_ms == 0:
            # skip or handle corner case
            i += 1
            continue

        if dt_ms > 0:
            # LTP lumpsum => group all consecutive events with the same (post_t_arr, j_post)
            w_before = w
            lumpsum = 0.0
            batch_time  = current_time  # post arrival
            batch_jpost = j_post

            ltp_events = []
            while i < n_ev:
                (ev_t2, i_pre2, j_post2, dt2,
                 pr2_raw, po2_raw,
                 pr2_arr, po2_arr) = all_events[i]
                if dt2>0 and (ev_t2 == batch_time) and (j_post2 == batch_jpost):
                    # lumpsum
                    dW_2 = lambda_pmt * (w_0**(1 - mu)) * (w**mu) * np.exp(-dt2 / tau_plus_ms)
                    ltp_events.append((pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2))
                    lumpsum += dW_2
                    i += 1
                else:
                    break

            # Now we log partial lines => w_after=None
            for (pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2) in ltp_events:
                writer.writerow([
                    syn_id, event_idx,
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,   # same w_before for partial lines
                    dW_2,
                    None        # partial => no real update
                ])
                trajectory.append((
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,
                    dW_2,
                    None
                ))
                event_idx += 1

            # Single lumpsum line
            w_after_lumpsum = w_before + lumpsum
            lumpsum_dt = 0.0
            writer.writerow([
                syn_id, event_idx,
                pre_t_raw, None,
                pre_t_arr, None,
                lumpsum_dt,
                w_before,
                lumpsum,
                w_after_lumpsum
            ])
            trajectory.append((
                pre_t_raw, None,
                pre_t_arr, None,
                lumpsum_dt,
                w_before,
                lumpsum,
                w_after_lumpsum
            ))
            event_idx += 1

            w = w_after_lumpsum

        else:
            # dt_ms < 0 => LTD lumpsum => group all consecutive events with the same (pre_t_arr, i_pre)
            w_before = w
            lumpsum = 0.0
            batch_time = current_time
            batch_ipre = i_pre

            ltd_events = []
            while i < n_ev:
                (ev_t2, i_pre2, j_post2, dt2,
                 pr2_raw, po2_raw,
                 pr2_arr, po2_arr) = all_events[i]
                if dt2 < 0 and (ev_t2 == batch_time) and (i_pre2 == batch_ipre):
                    dW_2 = -lambda_pmt * alpha * w * np.exp(dt2 / tau_plus_ms)
                    ltd_events.append((pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2))
                    lumpsum += dW_2
                    i += 1
                else:
                    break

            # partial lines => w_after=None
            for (pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2) in ltd_events:
                writer.writerow([
                    syn_id, event_idx,
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,
                    dW_2,
                    None
                ])
                trajectory.append((
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,
                    dW_2,
                    None
                ))
                event_idx += 1

            # lumpsum line
            w_after_lumpsum = w_before + lumpsum
            lumpsum_dt = 0.0
            writer.writerow([
                syn_id, event_idx,
                pre_t_raw, None,
                pre_t_arr, None,
                lumpsum_dt,
                w_before,
                lumpsum,
                w_after_lumpsum
            ])
            trajectory.append((
                pre_t_raw, None,
                pre_t_arr, None,
                lumpsum_dt,
                w_before,
                lumpsum,
                w_after_lumpsum
            ))
            event_idx += 1

            w = w_after_lumpsum

    return w, trajectory


###############################################################################
# Plotting
###############################################################################

def get_synapse_color(syn_id):
    return f"C{syn_id - 1}"

def plot_synaptic_evolution(synapses_trajectories, time_min_ms, time_max_ms,
                            start_syn, end_syn, prediction_plot_save):
    """
    We'll define 'time_of_event' as post_arr if dt>0, or pre_arr if dt<0,
    lumpsum line => dt=0 => we might use pre_arr.
    Each event has (pre_s, post_s, pre_a, post_a, dt_ms, w_before, dW, w_after).
    If w_after=None => partial negative => no real update time.

    We'll plot only synapses in [start_syn..end_syn], skipping partial lines.
    """
    plt.figure()
    for syn_id, trajectory in synapses_trajectories.items():
        if syn_id < start_syn or syn_id > end_syn:
            continue  # skip

        if not trajectory:
            continue

        times_ms = []
        weights  = []

        for evt in trajectory:
            (pre_s, post_s, pre_a, post_a, dt_ms, wB, dW, wA) = evt
            if pre_s is None and post_s is None:
                # the "fake" row => place it at time_min
                times_ms.append(time_min_ms)
                weights.append(wA)
            else:
                # skip partial lines
                if wA is None:
                    continue
                if dt_ms > 0:
                    # lumpsum => use post_a
                    times_ms.append(post_a)
                else:
                    # lumpsum => dt<=0 => use pre_a
                    times_ms.append(pre_a if pre_a is not None else time_min_ms)
                weights.append(wA)

        color = get_synapse_color(syn_id)
        plt.plot(times_ms, weights, label=f"Syn {syn_id}", color=color, marker='o')

    plt.title(f"Synaptic Evolution, synapses {start_syn}..{end_syn}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Weight")
    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    # no plt.show() here, user does it in main script
    if prediction_plot_save:
        plt.savefig("predicted_synaptic_evolution.png")

def plot_pre_raster(pre_spikes_dict, N, time_min_ms, time_max_ms,
                    start_syn, end_syn, prediction_plot_save):
    """
    Raster for *pre-synaptic* neurons, only synapses in [start_syn..end_syn].
    We'll set integer y-ticks from start_syn..end_syn only.
    """
    plt.figure()
    plt.title(f"Raster Plot: Pre-Synaptic Neurons {start_syn}..{end_syn} (ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID (Pre)")

    for i in range(start_syn, end_syn+1):
        times_ms = pre_spikes_dict[i]
        color    = get_synapse_color(i)
        y_vals   = [i]*len(times_ms)
        plt.scatter(times_ms, y_vals, color=color, marker='.', label=f"Pre {i}")

    plt.xlim(time_min_ms, time_max_ms)
    plt.yticks(range(start_syn, end_syn+1))
    plt.legend()
    if prediction_plot_save:
        plt.savefig("predicted_presyn_rastegram.png")
    # no plt.show()

def plot_post_raster(post_spikes_dict, N, time_min_ms, time_max_ms,
                     start_syn, end_syn, prediction_plot_save):
    """
    Raster for *post-synaptic* neurons in [start_syn..end_syn].
    We'll set integer y-ticks from (start_syn+N)..(end_syn+N).
    """
    plt.figure()
    plt.title(f"Raster Plot: Post-Synaptic Neurons {start_syn}..{end_syn} (ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID (Post)")

    for i in range(start_syn, end_syn+1):
        times_ms = post_spikes_dict[i]
        color    = get_synapse_color(i)
        neuron_id= i + N
        y_vals   = [neuron_id]*len(times_ms)
        plt.scatter(times_ms, y_vals, color=color, marker='.', label=f"Post {neuron_id}")

    plt.xlim(time_min_ms, time_max_ms)
    plt.yticks(range(start_syn+N, end_syn+N+1))
    plt.legend()
    if prediction_plot_save:
        plt.savefig("predicted_postsyn_rastegram.png")
    # no plt.show()

###############################################################################
# Main
###############################################################################

def predict_stdp_alpha_forced_pl(config_check_stdp_filename):
    """
    1) Read config,
    2) Load spikes,
    3) For each synapse => lumpsum LTP & lumpsum LTD,
    4) Save CSV 'stdp_evolution_line_summary.csv',
    5) Print final summary for [start_syn..end_syn],
    6) Raster & evolution plot only for [start_syn..end_syn],
    7) Return minimal "analysis_summary" for the user => {syn_i: {"syn_ID", "start_syn_value", "final_syn_value"}} 
    No plt.show() calls here.
    """
    import yaml
    with open(config_check_stdp_filename, "r") as f:
        config = yaml.safe_load(f)

    prediction_plot_save         = config["prediction_plot_save"]

    N             = config["Total_number_described_synapses_for_sim"]
    csv_file_pre  = config["csv_file_pre"]
    csv_file_post = config["csv_file_post"]
    start_syn     = config.get("start_synapse", 1)
    end_syn       = config.get("end_synapse", N)
    verbose_pred  = config.get("verbose_prediction_summary", True)

    if not isinstance(start_syn, int) or not isinstance(end_syn, int):
        raise ValueError("start_synapse and end_synapse must be integers.")
    if start_syn < 1 or end_syn > N or start_syn > end_syn:
        raise ValueError(f"Invalid syn range: {start_syn}..{end_syn}, must be in [1..{N}]")

    stdp_params  = config["stdp_params"]
    tau_plus_ms  = stdp_params["tau_plus"]
    lambda_pmt   = stdp_params["lambda"]
    alpha        = stdp_params["alpha"]
    mu           = stdp_params["mu"]
    w_0          = config["w_0"]

    init_weights_list     = config.get("W_init", None)
    axonal_delays_list    = config.get("axonal_delay_ms", None)
    dendritic_delays_list = config.get("dendritic_delay_ms", None)
    if init_weights_list is None or len(init_weights_list) != N:
        raise ValueError("W_init must be list of length N.")

    axon_default  = 5.0
    dend_default  = 0.1

    pre_spikes_dict, pre_all_times, _ = load_spikes_pre(csv_file_pre, N)
    post_spikes_dict, post_all_times,_= load_spikes_post(csv_file_post, N)

    arrived_pre_dict  = {}
    arrived_post_dict = {}
    for i in range(1, N+1):
        axon_d  = float( safe_get_list_item(axonal_delays_list, i-1, axon_default) )
        dend_d  = float( safe_get_list_item(dendritic_delays_list, i-1, dend_default) )

        arrived_pre_dict[i]  = [t + axon_d  for t in pre_spikes_dict[i]]
        arrived_post_dict[i] = [t + dend_d  for t in post_spikes_dict[i]]

    all_arr_pre  = [t for i in range(1,N+1) for t in arrived_pre_dict[i]]
    all_arr_post = [t for i in range(1,N+1) for t in arrived_post_dict[i]]
    arrived_all  = all_arr_pre + all_arr_post
    if len(arrived_all)==0:
        global_min_time_ms = 0.0
        global_max_time_ms = 100.0
    else:
        global_min_time_ms = min(arrived_all) - 10
        global_max_time_ms = max(arrived_all) + 10

    final_weights         = [0.0]*N
    synapses_trajectories = {}

    summary_file = "stdp_evolution_line_summary.csv"
    with open(summary_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "synapse_id", "event_idx",
            "pre_spike_ms", "post_spike_ms",
            "pre_arrived_ms", "post_arrived_ms",
            "dt_ms",
            "w_before",
            "dW",
            "w_after"
        ])

        for syn_i in range(1, N+1):
            w_init = init_weights_list[syn_i-1]
            w_final, trajectory = stdp_pl_synapse_hom_causal(
                syn_i,
                pre_spikes_dict, post_spikes_dict,
                arrived_pre_dict, arrived_post_dict,
                tau_plus_ms, lambda_pmt, alpha, mu, w_0,
                w_init,
                writer,
                initial_time_for_plot_ms=global_min_time_ms
            )
            final_weights[syn_i-1] = w_final
            synapses_trajectories[syn_i] = trajectory

    print(f"Lumpsum STDP summary logged to {summary_file}")
    print("--------------------------------------------------")
    print(f"FINAL DIAGNOSTIC for synapses {start_syn}..{end_syn}")

    analysis_summary = {}
    for syn_i in range(1, N+1):
        if syn_i<start_syn or syn_i>end_syn: 
            continue

        w_init  = init_weights_list[syn_i-1]
        w_final = final_weights[syn_i-1]
        analysis_summary[syn_i] = {
            "syn_ID": syn_i,
            "start_syn_value": w_init,
            "final_syn_value": w_final
        }

        # optionally print
        if verbose_pred:
            pre_c = len(pre_spikes_dict[syn_i])
            post_c= len(post_spikes_dict[syn_i])
            changes= pre_c*post_c
            track  = synapses_trajectories[syn_i]
            if track:
                event_lines=[]
                for evt in track:
                    if evt[0] is None and evt[1] is None:
                        wA= evt[-1]
                        line_e= f"(NoSpikes, dt_ms=0.000, dW=0.000, W={wA:.3f})"
                    else:
                        (p_s, po_s, p_a, po_a, dt_m, wB, dW, wA)= evt
                        wA_str= f"{wA:.5f}" if wA is not None else "None"
                        line_e=(
                            f"(pre_spike_ms={p_s}, post_spike_ms={po_s}, pre_arr_ms={p_a}, post_arr_ms={po_a}, "
                            f"dt_ms={dt_m:.3f}, w_before={wB:.3f}, dW={dW:.3f}, w_after={wA_str})"
                        )
                    event_lines.append(line_e)
                all_event_str= " ".join(event_lines)
                print(f"Syn {syn_i}: #changes={changes}, init={w_init:.4f}, final={w_final:.4f} => {all_event_str}")
            else:
                print(f"Syn {syn_i}: #changes={changes}, init={w_init:.4f}, final={w_final:.4f} => No step-by-step data")

    print("--------------------------------------------------")

    # Plot
    plot_pre_raster(arrived_pre_dict,   N, global_min_time_ms, global_max_time_ms,
                    start_syn, end_syn, prediction_plot_save)
    plot_post_raster(arrived_post_dict, N, global_min_time_ms, global_max_time_ms,
                     start_syn, end_syn, prediction_plot_save)
    plot_synaptic_evolution(synapses_trajectories, global_min_time_ms, global_max_time_ms,
                            start_syn, end_syn, prediction_plot_save)

    return analysis_summary
