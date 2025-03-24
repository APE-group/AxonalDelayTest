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
    all_neuron_ids  = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neuron_id     = int(row["senders"])
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
      post_spikes_dict : { i: list of float spike_time_ms }   (i in 1..N => post neuron = i+N)
      all_spike_times_ms
      all_neuron_ids
    """
    post_spikes_dict = {i: [] for i in range(1, N+1)}
    all_spike_times_ms = []
    all_neuron_ids     = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neuron_id     = int(row["senders"])
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
# Causal STDP for pl_synapse_hom
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
    A "chronological" and "causal" version of the pl_synapse_hom rule.
    1) We gather *all pairs* (pre_spike, post_spike).
    2) For each pair, define:
         if dt>0 => event_time = post_arr,
                   dW = +...
         if dt<0 => event_time = pre_arr,
                   dW = -...
    3) We sort all events by event_time in ascending order.
    4) For each negative event that shares the same pre spike *and same event_time*,
       we sum up the negative dW in a lumpsum. We log each event individually,
       but do not update the weight in between. Then at the end of that lumpsum group,
       we apply the total negative change at once.
    5) For positive events, we update the weight immediately.

    We store every event as a 7-item tuple:
      (pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, dW, w_after).
    """

    pre_times_raw   = pre_spikes_dict[syn_id]
    post_times_raw  = post_spikes_dict[syn_id]
    pre_times_arr   = arrived_pre_dict[syn_id]
    post_times_arr  = arrived_post_dict[syn_id]

    w = w_init
    trajectory = []

    # Write an initial row => no real spikes
    writer.writerow([syn_id, 0, None, None, None, None, None, 0.0, w])
    trajectory.append((None, None, None, None, 0.0, 0.0, w))

    # Build a global list of all pairs
    # We'll store: (event_time, is_ltp, pre_index, post_index, dt_ms)
    #   is_ltp is True if dt>0, else False
    #   event_time = post_arr if dt>0, else pre_arr if dt<0
    #   We also store the raw times so we can compute the dW formula
    all_events = []

    for i_pre, pre_t_raw in enumerate(pre_times_raw):
        pre_t_arr = pre_times_arr[i_pre]

        for j_post, post_t_raw in enumerate(post_times_raw):
            post_t_arr = post_times_arr[j_post]
            dt_ms = post_t_arr - pre_t_arr

            if dt_ms > 0:
                # LTP => immediate => event_time = post_t_arr
                event_time = post_t_arr
            else:
                # LTD => lumps => event_time = pre_t_arr
                event_time = pre_t_arr

            all_events.append((event_time, i_pre, j_post, dt_ms,
                               pre_t_raw, post_t_raw,
                               pre_t_arr, post_t_arr))

    # Sort by event_time ascending
    all_events.sort(key=lambda x: x[0])

    event_idx = 1

    # We'll iterate in ascending order, applying lumps for negative events that share
    # the same syn_id, same pre spike, same event_time.
    idx = 0
    n_events = len(all_events)
    while idx < n_events:
        (current_time, i_pre, j_post, dt_ms,
         pre_t_raw, post_t_raw,
         pre_t_arr, post_t_arr) = all_events[idx]

        if dt_ms > 0:
            # LTP => immediate
            # formula from pl_synapse_hom for dt>0
            dW = (lambda_pmt *
                  (w_0**(1 - mu)) *
                  (w**mu) *
                  np.exp(-dt_ms / tau_plus_ms))
            w_new = w + dW

            # Log
            writer.writerow([
                syn_id, event_idx,
                pre_t_raw, post_t_raw,
                pre_t_arr, post_t_arr,
                dt_ms, dW, w_new
            ])
            trajectory.append((
                pre_t_raw, post_t_raw,
                pre_t_arr, post_t_arr,
                dt_ms, dW, w_new
            ))

            w = w_new
            event_idx += 1
            idx += 1

        else:
            # dt_ms <= 0 => negative => lumps
            # We'll gather all consecutive negative events for the same event_time
            # and the same i_pre (the same pre spike).
            lumpsum = 0.0
            negative_batch = []

            # We'll keep reading consecutive negative events with the same
            # pre arrival time = current_time, same i_pre, same syn_id
            this_pre_arr = pre_t_arr

            while idx < n_events:
                (etm, i_pre2, j_post2, dt2,
                 pr2_raw, po2_raw,
                 pr2_arr, po2_arr) = all_events[idx]

                if (etm == current_time and i_pre2 == i_pre and dt2 <= 0):
                    # part of the lumpsum
                    # formula for dt<0
                    dW_2 = -lambda_pmt * alpha * w * np.exp(dt2 / tau_plus_ms)
                    lumpsum += dW_2

                    # We'll log each negative event individually with partial w, but not apply it
                    w_temp = w + dW_2  # partial
                    writer.writerow([
                        syn_id, event_idx,
                        pr2_raw, po2_raw,
                        pr2_arr, po2_arr,
                        dt2, dW_2, w_temp
                    ])
                    trajectory.append((
                        pr2_raw, po2_raw,
                        pr2_arr, po2_arr,
                        dt2, dW_2, w_temp
                    ))

                    idx += 1
                    event_idx += 1
                else:
                    break

            # after we finish that batch => apply lumpsum
            w = w + lumpsum

    return w, trajectory


###############################################################################
# Plotting
###############################################################################

def get_synapse_color(syn_id):
    return f"C{syn_id - 1}"

def plot_synaptic_evolution(synapses_trajectories, time_min_ms, time_max_ms):
    """
    We'll define the 'time of event' as either post_arr if dt>0 or pre_arr if dt<0,
    but we do that again in post-processing from the 7-tuple.
    """
    plt.figure()
    for syn_id, trajectory in synapses_trajectories.items():
        if not trajectory:
            continue

        times_ms = []
        weights  = []

        for evt in trajectory:
            # 7 items => (pre_s, post_s, pre_a, post_a, dt_ms, dW, w_after)
            pre_s, post_s, pre_a, post_a, dt_ms, dW, w_after = evt

            if (pre_s is None) and (post_s is None):
                # The "fake" row
                times_ms.append(time_min_ms)
                weights.append(w_after)
            else:
                if dt_ms > 0:
                    # LTP => time = post_a
                    times_ms.append(post_a)
                else:
                    # LTD => time = pre_a
                    times_ms.append(pre_a)
                weights.append(w_after)

        color = get_synapse_color(syn_id)
        plt.plot(times_ms, weights, label=f"Syn {syn_id}", color=color, marker='o')

    plt.title("Synaptic Evolution (Causal pl_synapse_hom, Chronological)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Weight")
    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    plt.show()


def plot_pre_raster(pre_spikes_dict, N, time_min_ms, time_max_ms):
    """
    Raster for *pre-synaptic* neurons, presumably using arrival times.
    """
    plt.figure()
    plt.title("Raster Plot: Pre-Synaptic Neurons (ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID (Pre)")

    for i in range(1, N+1):
        times_ms = pre_spikes_dict[i]
        color    = get_synapse_color(i)
        y_vals   = [i]*len(times_ms)
        plt.scatter(times_ms, y_vals, color=color, marker='.', label=f"Pre {i}")

    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    plt.show()


def plot_post_raster(post_spikes_dict, N, time_min_ms, time_max_ms):
    """
    Raster for *post-synaptic* neurons (N+1..2N), presumably using arrival times.
    """
    plt.figure()
    plt.title("Raster Plot: Post-Synaptic Neurons (ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID (Post)")

    for i in range(1, N+1):
        times_ms = post_spikes_dict[i]
        color    = get_synapse_color(i)
        neuron_id= i + N
        y_vals   = [neuron_id]*len(times_ms)
        plt.scatter(times_ms, y_vals, color=color, marker='.', label=f"Post {neuron_id}")

    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    plt.show()


###############################################################################
# Main
###############################################################################

def test_stdp_main():
    """
    Main function:
      1) Reads config,
      2) Loads spikes,
      3) Applies single 'stdp_pl_synapse_hom_causal' rule in chronological order,
      4) Logs results to 'stdp_summary.csv',
      5) Prints final summary,
      6) Plots rasters & weight evolution.
    """
    with open("config_check_stdp.yaml", "r") as f:
        config = yaml.safe_load(f)

    N = config["N"]
    csv_file_pre  = config["csv_file_pre"]
    csv_file_post = config["csv_file_post"]

    # pl_synapse_hom STDP parameters
    tau_plus_ms  = config.get("tau_plus_ms", 20.0)
    lambda_pmt   = config.get("lambda_pmt", 0.9)
    alpha        = config.get("alpha", 0.11)
    mu           = config.get("mu", 0.4)
    w_0          = config.get("w_0", 1.0)

    # Delays & Weights
    init_weights_list     = config.get("initial_weights", None)
    axonal_delays_list    = config.get("axonal_delays_ms", None)
    dendritic_delays_list = config.get("dendritic_delays_ms", None)

    if init_weights_list is None or len(init_weights_list) != N:
        raise ValueError("initial_weights must be a list of length N.")

    # Defaults if missing
    axon_default  = 5.0
    dend_default  = 0.1

    # Load spikes
    pre_spikes_dict, pre_all_times, _   = load_spikes_pre(csv_file_pre, N)
    post_spikes_dict, post_all_times, _ = load_spikes_post(csv_file_post, N)

    # Build arrival-time dicts
    arrived_pre_dict  = {}
    arrived_post_dict = {}

    for i in range(1, N+1):
        axon_d   = safe_get_list_item(axonal_delays_list, i-1, axon_default)
        dend_d   = safe_get_list_item(dendritic_delays_list, i-1, dend_default)

        arrived_pre_dict[i]  = [t + axon_d for t in pre_spikes_dict[i]]
        arrived_post_dict[i] = [t + dend_d for t in post_spikes_dict[i]]

    # Global time range
    all_arr_pre  = [t for i in range(1, N+1) for t in arrived_pre_dict[i]]
    all_arr_post = [t for i in range(1, N+1) for t in arrived_post_dict[i]]
    arrived_all  = all_arr_pre + all_arr_post

    if len(arrived_all) == 0:
        global_min_time_ms = 0.0
        global_max_time_ms = 100.0
    else:
        global_min_time_ms = min(arrived_all) - 10
        global_max_time_ms = max(arrived_all) + 10

    # Run the "causal pl_synapse_hom"
    final_weights = [0.0]*N
    synapses_trajectories = {}

    summary_file = "stdp_summary.csv"
    with open(summary_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        # 9 columns: syn_id, event_idx,
        #   pre_spike_ms, post_spike_ms, pre_arrived_ms, post_arrived_ms, dt_ms, dW, w
        writer.writerow([
            "synapse_id", "event_idx",
            "pre_spike_ms", "post_spike_ms",
            "pre_arrived_ms", "post_arrived_ms",
            "dt_ms", "dW", "W"
        ])

        event_idx_counter = 1
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

    print(f"CAUSAL pl_synapse_hom summary logged to {summary_file}")

    # Final diagnostic summary
    print("--------------------------------------------------")
    print("FINAL DIAGNOSTIC SUMMARY (Causal pl_synapse_hom, Chronological)")

    for syn_i in range(1, N+1):
        w_init  = init_weights_list[syn_i-1]
        w_final = final_weights[syn_i-1]

        axon_d   = safe_get_list_item(axonal_delays_list,   syn_i-1, axon_default)
        dend_d   = safe_get_list_item(dendritic_delays_list, syn_i-1, dend_default)

        num_changes = (len(pre_spikes_dict[syn_i]) *
                       len(post_spikes_dict[syn_i]))

        trajectory = synapses_trajectories[syn_i]
        if trajectory:
            events_str = []
            for evt in trajectory:
                # 7 items => (pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, dW, w_after)
                if evt[0] is None and evt[1] is None:
                    # The "fake" row
                    wA = evt[-1]
                    evt_str = f"(NoSpikes, dt_ms=0.000, dW=0.000, W={wA:.3f})"
                else:
                    (p_s, po_s, p_a, po_a, dt_m, dW, wA) = evt
                    evt_str = (
                        f"(pre_spike_ms={p_s}, post_spike_ms={po_s}, "
                        f"pre_arr_ms={p_a}, post_arr_ms={po_a}, "
                        f"dt_ms={dt_m:.3f}, dW={dW:.3f}, W={wA:.3f})"
                    )
                events_str.append(evt_str)
            all_events_line = " ".join(events_str)
        else:
            all_events_line = "No step-by-step data"

        line = (f"Synapse {syn_i}: "
                f"axon_delay_ms={axon_d}, dend_delay_ms={dend_d}, "
                f"#changes={num_changes}, init={w_init:.3f}, final={w_final:.3f} => "
                f"{all_events_line}")
        print(line)

    print("--------------------------------------------------")

    # Rasters with arrival times
    plot_pre_raster(arrived_pre_dict,  N, global_min_time_ms, global_max_time_ms)
    plot_post_raster(arrived_post_dict, N, global_min_time_ms, global_max_time_ms)

    # Synaptic evolution
    plot_synaptic_evolution(synapses_trajectories, global_min_time_ms, global_max_time_ms)
