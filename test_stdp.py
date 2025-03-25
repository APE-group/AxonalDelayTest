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
      post_spikes_dict : { i: list of float spike_time_ms }
                         (i in 1..N => actual post neuron = i+N)
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

    1) Gather all pre–post pairs for this synapse => build a list of events.
       For dt>0 => event_time = post_arr; for dt<0 => event_time = pre_arr.
    2) Sort by event_time ascending.
    3) LTP (dt>0) => immediate update: log each event with w_before, w_after.
    4) LTD (dt<0) => lumpsum approach:
         - log each partial negative event with w_after=None (since no real update yet),
         - after finishing that lumpsum group, produce one lumpsum line with the real new weight.
    5) We store 8 items in each event:
         (pre_s, post_s, pre_a, post_a, dt_ms, w_before, dW, w_after)
    """

    pre_times_raw   = pre_spikes_dict[syn_id]
    post_times_raw  = post_spikes_dict[syn_id]
    pre_times_arr   = arrived_pre_dict[syn_id]
    post_times_arr  = arrived_post_dict[syn_id]

    w = w_init
    trajectory = []

    # Write an initial row => no real spikes
    writer.writerow([syn_id, 0, None, None, None, None, None, 0.0, w])
    # We store an 8-tuple => the partial final is None for this "fake" row
    trajectory.append((None, None, None, None, 0.0, 0.0, 0.0, w))

    # Build a global list of all pairs
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
                # dt<=0 => lumps => event_time = pre_t_arr
                event_time = pre_t_arr

            all_events.append((event_time,
                               i_pre, j_post,
                               dt_ms,
                               pre_t_raw, post_t_raw,
                               pre_t_arr, post_t_arr))

    # Sort by event_time ascending
    all_events.sort(key=lambda x: x[0])

    event_idx = 1
    i = 0
    n_ev = len(all_events)

    while i < n_ev:
        (ev_time, i_pre, j_post, dt_ms,
         pre_t_raw, post_t_raw,
         pre_t_arr, post_t_arr) = all_events[i]

        if dt_ms > 0:
            # LTP => immediate
            w_before = w
            dW = (lambda_pmt *
                  (w_0**(1 - mu)) *
                  (w**mu) *
                  np.exp(-dt_ms / tau_plus_ms))
            w_after = w_before + dW

            # CSV line => 10 columns:
            # syn_id, event_idx, pre_s, post_s, pre_arr, post_arr, dt_ms, w_before, dW, w_after
            writer.writerow([
                syn_id, event_idx,
                pre_t_raw, post_t_raw,
                pre_t_arr, post_t_arr,
                dt_ms,
                w_before,
                dW,
                w_after
            ])
            # Store 8 items => (pre_s, post_s, pre_a, post_a, dt_ms, w_before, dW, w_after)
            trajectory.append((
                pre_t_raw, post_t_raw,
                pre_t_arr, post_t_arr,
                dt_ms,
                w_before,
                dW,
                w_after
            ))

            w = w_after
            event_idx += 1
            i += 1

        else:
            # dt_ms <= 0 => lumpsum of negative changes
            lumpsum = 0.0
            w_before = w
            batch_time = ev_time
            batch_pre_idx = i_pre

            negative_events = []
            # gather consecutive negative events with the same event_time + same i_pre
            while i < n_ev:
                (ev_t2, i_pre2, j_post2, dt2,
                 pr2_raw, po2_raw,
                 pr2_arr, po2_arr) = all_events[i]
                if (ev_t2 == batch_time) and (i_pre2 == batch_pre_idx) and (dt2 <= 0):
                    dW_2 = -lambda_pmt * alpha * w * np.exp(dt2 / tau_plus_ms)
                    negative_events.append((pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2))
                    lumpsum += dW_2
                    i += 1
                else:
                    break

            # Log each negative event individually => w_after=None
            for (pr2_raw, po2_raw, pr2_arr, po2_arr, dt2, dW_2) in negative_events:
                # partial line => no real w update
                writer.writerow([
                    syn_id, event_idx,
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,  # same for each partial line
                    dW_2,
                    None        # we show "None" => no update
                ])
                trajectory.append((
                    pr2_raw, po2_raw,
                    pr2_arr, po2_arr,
                    dt2,
                    w_before,
                    dW_2,
                    None   # partial negative => no real final weight
                ))
                event_idx += 1

            # Now apply the lumpsum
            w_after_lumpsum = w_before + lumpsum
            # We'll do a lumpsum line => dt=0 or "LUMPSUM"
            lumpsum_dt = 0.0

            writer.writerow([
                syn_id, event_idx,
                pre_t_raw, None,  # we can omit post spike info here
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

def plot_synaptic_evolution(synapses_trajectories, time_min_ms, time_max_ms):
    """
    We'll define the 'time_of_event' as post_arr if dt>0, or pre_arr if dt<0.
    For lumpsum line => dt=0 => we can pick the pre_arr time or the lumpsum time.
    Each event in the trajectory is:
      (pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, w_before, dW, w_after)
    If w_after=None => partial negative => no real update time.
    If lumpsum line => dt=0 => we might use pre_arr as the update time.
    """
    plt.figure()
    for syn_id, trajectory in synapses_trajectories.items():
        if not trajectory:
            continue

        times_ms = []
        weights  = []

        for evt in trajectory:
            # 8 items
            (p_s, po_s, p_a, po_a, dt_ms, wB, dW, wA) = evt

            if p_s is None and po_s is None:
                # initial row => use time_min
                times_ms.append(time_min_ms)
                weights.append(wA)
            else:
                if wA is None:
                    # partial negative => we could skip it or put a partial time
                    # let's skip it from the curve
                    continue
                else:
                    # either LTP or lumpsum => dt>0 => time=po_a, dt<0 => time= p_a, dt=0 => lumpsum => p_a
                    if dt_ms > 0:
                        times_ms.append(po_a)
                    else:
                        # lumpsum => or dt<0 => use pre_a
                        times_ms.append(p_a if p_a is not None else time_min_ms)
                    weights.append(wA)

        color = get_synapse_color(syn_id)
        plt.plot(times_ms, weights, label=f"Syn {syn_id}", color=color, marker='o')

    plt.title("Synaptic Evolution (Causal lumpsum, partial negative lines omitted from curve)")
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
      4) Logs results to 'stdp_summary.csv' (with lumpsum line),
      5) Prints final summary,
      6) Plots rasters & weight evolution, skipping partial negative lines.
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
        axon_d_str   = safe_get_list_item(axonal_delays_list,   i-1, axon_default)
        dend_d_str   = safe_get_list_item(dendritic_delays_list, i-1, dend_default)
        # ensure float
        axon_d  = float(axon_d_str)
        dend_d  = float(dend_d_str)

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

    # Run the "causal pl_synapse_hom" rule
    final_weights = [0.0]*N
    synapses_trajectories = {}

    summary_file = "stdp_summary.csv"
    with open(summary_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        # 10 columns:
        # syn_id, event_idx,
        # pre_spike_ms, post_spike_ms,
        # pre_arrived_ms, post_arrived_ms,
        # dt_ms,
        # w_before,
        # dW,
        # w_after
        writer.writerow([
            "synapse_id", "event_idx",
            "pre_spike_ms", "post_spike_ms",
            "pre_arrived_ms", "post_arrived_ms",
            "dt_ms",
            "w_before",
            "dW",
            "w_after"
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

    # Final diagnostics
    print("--------------------------------------------------")
    print("FINAL DIAGNOSTIC SUMMARY (Causal pl_synapse_hom with lumpsum line, partial negative => w_after=None)")

    for syn_i in range(1, N+1):
        w_init  = init_weights_list[syn_i-1]
        w_final = final_weights[syn_i-1]

        # get delays
        axon_d_str   = safe_get_list_item(axonal_delays_list, syn_i-1, axon_default)
        dend_d_str   = safe_get_list_item(dendritic_delays_list,syn_i-1,dend_default)
        axon_d  = float(axon_d_str)
        dend_d  = float(dend_d_str)

        pre_count  = len(pre_spikes_dict[syn_i])
        post_count = len(post_spikes_dict[syn_i])
        num_changes = pre_count * post_count

        trajectory = synapses_trajectories[syn_i]
        if trajectory:
            events_str = []
            for evt in trajectory:
                # 8 items => (pre_s, post_s, pre_a, post_a, dt, w_before, dW, w_after)
                if evt[0] is None and evt[1] is None:
                    # The "fake" row
                    wA = evt[-1]
                    evt_str = f"(NoSpikes, dt_ms=0.000, dW=0.000, W={wA:.3f})"
                else:
                    (p_s, po_s, p_a, po_a, dt_m, wB, dW, wA) = evt
                    # if wA is None => partial negative
                    wA_str = f"{wA:.3f}" if wA is not None else "None"
                    evt_str = (
                        f"(pre_spike_ms={p_s}, post_spike_ms={po_s}, "
                        f"pre_arr_ms={p_a}, post_arr_ms={po_a}, "
                        f"dt_ms={dt_m:.3f}, w_before={wB:.3f}, dW={dW:.3f}, w_after={wA_str})"
                    )
                events_str.append(evt_str)
            all_events_line = " ".join(events_str)
        else:
            all_events_line = "No step-by-step data"

        line = (f"Synapse {syn_i}: axon_delay_ms={axon_d}, dend_delay_ms={dend_d}, "
                f"#changes={num_changes}, init={w_init:.3f}, final={w_final:.3f} => "
                f"{all_events_line}")
        print(line)
    print("--------------------------------------------------")

    # Rasters with arrival times
    plot_pre_raster(arrived_pre_dict,  N, global_min_time_ms, global_max_time_ms)
    plot_post_raster(arrived_post_dict, N, global_min_time_ms, global_max_time_ms)

    # Weight evolution
    plot_synaptic_evolution(synapses_trajectories, global_min_time_ms, global_max_time_ms)
