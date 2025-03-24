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

    # Sort
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
                         i in 1..N -> actual post neuron = i+N
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

    # Sort
    for i in range(1, N+1):
        post_spikes_dict[i].sort()

    return post_spikes_dict, all_spike_times_ms, all_neuron_ids


###############################################################################
# Causal STDP Rule (pl_synapse_hom)
###############################################################################

def stdp_pl_synapse_hom_causal(
    pre_spike_times_ms, post_spike_times_ms,
    pre_spike_arrivals_ms, post_spike_arrivals_ms,
    tau_plus_ms, lambda_pmt, alpha, mu, w_0,
    w_init,
    writer,
    synapse_id,
    initial_time_for_plot_ms
):
    """
    A "causal" version of the pl_synapse_hom rule.
    Key changes:
      1) LTP events (dt>0) are updated immediately at the post arrival time.
      2) LTD events (dt<0) per pre spike are summed and applied in one lump
         at the pre arrival time.

    We store each event as a 7-item tuple:
      ( pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, dW, w_after )
    so that the final summary can index them consistently.
    """

    w = w_init
    trajectory = []

    # Write an initial row => no real spikes, dt=0, dW=0, w_init
    # We'll pass None for the spike times, arrival times
    writer.writerow([synapse_id, 0, None, None, None, None, None, 0.0, w])
    trajectory.append((None, None, None, None, 0.0, 0.0, w))

    event_idx = 1
    num_pre_spikes = len(pre_spike_times_ms)

    for i_pre in range(num_pre_spikes):
        pre_t_orig = pre_spike_times_ms[i_pre]
        pre_t_arr  = pre_spike_arrivals_ms[i_pre]

        # We'll separate the post spikes into LTP (dt>0) and LTD (dt<0) for this pre spike.
        ltp_list = []
        ltd_list = []

        for j_post, post_t_orig in enumerate(post_spike_times_ms):
            post_t_arr = post_spike_arrivals_ms[j_post]
            dt_ms = post_t_arr - pre_t_arr

            if dt_ms > 0:
                # LTP
                # same formula from standard pl_synapse_hom for dt>0
                dW = (lambda_pmt *
                      (w_0**(1 - mu)) *
                      (w**mu) *
                      np.exp(-dt_ms / tau_plus_ms))
                ltp_list.append((post_t_orig, post_t_arr, dt_ms, dW))
            elif dt_ms < 0:
                # LTD
                dW = -lambda_pmt * alpha * w * np.exp(dt_ms / tau_plus_ms)
                ltd_list.append((post_t_orig, post_t_arr, dt_ms, dW))
            else:
                # dt=0 => skip or handle separately
                continue

        # 1) Handle LTP events immediately
        for (post_t_orig, post_t_arr, dt_ms, dW) in ltp_list:
            w_new = w + dW

            # Log to CSV
            # Note the post "arrived" time is effectively the event time,
            # but we store the original 7 items for consistency
            writer.writerow([
                synapse_id,
                event_idx,
                pre_t_orig,   # raw pre spike
                post_t_orig,  # raw post spike
                pre_t_arr,    # arrived pre
                post_t_arr,   # arrived post
                dt_ms,
                dW,
                w_new
            ])
            # Add to trajectory (7 items)
            trajectory.append((
                pre_t_orig,
                post_t_orig,
                pre_t_arr,
                post_t_arr,
                dt_ms,
                dW,
                w_new
            ))

            w = w_new
            event_idx += 1

        # 2) Accumulate negative changes for LTD, apply once
        if len(ltd_list) > 0:
            sum_neg = 0.0
            for (_, _, _, dW) in ltd_list:
                sum_neg += dW

            w_before = w
            w_after_ltd = w + sum_neg

            # We'll log each negative event individually, but do not update 'w' each time
            for (post_t_orig, post_t_arr, dt_ms, dW) in ltd_list:
                w_temp = w_before + dW  # just for logging partial effect

                writer.writerow([
                    synapse_id,
                    event_idx,
                    pre_t_orig,
                    post_t_orig,
                    pre_t_arr,
                    post_t_arr,
                    dt_ms,
                    dW,
                    w_temp
                ])
                trajectory.append((
                    pre_t_orig,
                    post_t_orig,
                    pre_t_arr,
                    post_t_arr,
                    dt_ms,
                    dW,
                    w_temp
                ))

                event_idx += 1

            # Now apply the total negative sum
            w = w_after_ltd

    return w, trajectory


###############################################################################
# Plotting
###############################################################################

def get_synapse_color(syn_id):
    return f"C{syn_id - 1}"

def plot_synaptic_evolution(synapses_trajectories, time_min_ms, time_max_ms):
    """
    We plot the 'time_of_update' vs the weight. 
    But for simplicity, we use either post_t_arr or pre_t_arr or a midpoint 
    – whichever you prefer. 
    In this minimal code, let's just do a midpoint for LTP, pre_t_arr for LTD, etc. 
    But we've stored only the 7 items. 
    We'll define logic to pick time_of_update based on dt_ms > 0 or dt_ms < 0.
    """
    plt.figure()
    for syn_id, trajectory in synapses_trajectories.items():
        if not trajectory or len(trajectory) == 0:
            continue

        times_ms = []
        weights  = []
        for idx, evt in enumerate(trajectory):
            # 7 items => (pre_s, post_s, pre_arr, post_arr, dt_ms, dW, w_after)
            pre_s, post_s, pre_a, post_a, dt_ms, dW, w_after = evt

            if (pre_s is None) and (post_s is None):
                # The initial row => place it at time_min
                times_ms.append(time_min_ms)
                weights.append(w_after)
            else:
                # LTP => dt_ms > 0 => time_of_update = post_a
                # LTD => dt_ms < 0 => time_of_update = pre_a
                # if dt_ms=0 => skip or define logic
                if dt_ms > 0:
                    time_of_update = post_a
                else:
                    time_of_update = pre_a

                times_ms.append(time_of_update)
                weights.append(w_after)

        color = get_synapse_color(syn_id)
        plt.plot(times_ms, weights, label=f"Syn {syn_id}", color=color, marker='o')

    plt.title("Synaptic Evolution (Weight vs Time in ms, causal approach)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Weight")
    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    plt.show()


def plot_pre_raster(pre_spikes_dict, N, time_min_ms, time_max_ms):
    """
    Raster for *pre-synaptic* neurons, presumably using arrival times if you pass that dict.
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
      3) Applies a single "stdp_pl_synapse_hom_causal" rule, 
      4) Logs results, 
      5) Plots rasters and weight evolution, 
      6) Prints final summary.
    """
    with open("config_check_stdp.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Basic parameters
    N = config["N"]
    csv_file_pre  = config["csv_file_pre"]
    csv_file_post = config["csv_file_post"]

    # Extract STDP parameters
    tau_plus_ms  = config.get("tau_plus_ms", 20.0)
    lambda_pmt   = config.get("lambda_pmt", 0.9)
    alpha        = config.get("alpha", 0.11)
    mu           = config.get("mu", 0.4)
    w_0          = config.get("w_0", 1.0)

    # Default delay values
    axon_default  = 5.0
    dend_default  = 0.1

    # We read them as lists of length N
    init_weights_list     = config.get("initial_weights", None)
    axonal_delays_list    = config.get("axonal_delays_ms", None)
    dendritic_delays_list = config.get("dendritic_delays_ms", None)

    if init_weights_list is None or len(init_weights_list) != N:
        raise ValueError("initial_weights must be a list of length N.")

    # Load spike data
    pre_spikes_dict, pre_all_times, _   = load_spikes_pre(csv_file_pre, N)
    post_spikes_dict, post_all_times, _ = load_spikes_post(csv_file_post, N)

    # Apply axonal / dendritic delays
    arrived_pre_dict  = {}
    arrived_post_dict = {}

    for i in range(1, N+1):
        axon_delay_ms  = safe_get_list_item(axonal_delays_list, i-1, axon_default)
        dend_delay_ms  = safe_get_list_item(dendritic_delays_list, i-1, dend_default)

        arrived_pre_dict[i]  = [t + axon_delay_ms  for t in pre_spikes_dict[i]]
        arrived_post_dict[i] = [t + dend_delay_ms  for t in post_spikes_dict[i]]

    # Determine global min/max time for plotting
    all_arr_pre  = [t for i in range(1, N+1) for t in arrived_pre_dict[i]]
    all_arr_post = [t for i in range(1, N+1) for t in arrived_post_dict[i]]
    arrived_all  = all_arr_pre + all_arr_post

    if len(arrived_all) == 0:
        global_min_time_ms = 0.0
        global_max_time_ms = 100.0
    else:
        global_min_time_ms = min(arrived_all) - 10
        global_max_time_ms = max(arrived_all) + 10

    # 5) Run the CAUSAL pl_synapse_hom rule
    final_weights = [0.0]*N
    synapses_trajectories = {}

    summary_file = "stdp_summary.csv"
    with open(summary_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "synapse_id", "event_idx",
            "pre_spike_ms", "post_spike_ms",
            "pre_arrived_ms", "post_arrived_ms",
            "dt_ms", "dW", "W"
        ])

        # For each synapse
        event_idx_counter = 1  # if you want a global event index across all synapses
        for syn_i in range(1, N+1):
            w_init = init_weights_list[syn_i-1]
            w_final, trajectory = stdp_pl_synapse_hom_causal(
                pre_spikes_dict[syn_i],
                post_spikes_dict[syn_i],
                arrived_pre_dict[syn_i],
                arrived_post_dict[syn_i],
                tau_plus_ms, lambda_pmt, alpha, mu, w_0,
                w_init,
                writer,
                synapse_id=syn_i,
                initial_time_for_plot_ms=global_min_time_ms
            )
            final_weights[syn_i-1] = w_final
            synapses_trajectories[syn_i] = trajectory

    print(f"CAUSAL pl_synapse_hom summary logged to {summary_file}")

    # 6) Print final diagnostics
    print("--------------------------------------------------")
    print("FINAL DIAGNOSTIC SUMMARY (Causal pl_synapse_hom)")
    for syn_i in range(1, N+1):
        w_init  = init_weights_list[syn_i-1]
        w_final = final_weights[syn_i-1]
        axon_delay_ms  = safe_get_list_item(axonal_delays_list,   syn_i-1, axon_default)
        dend_delay_ms  = safe_get_list_item(dendritic_delays_list,syn_i-1, dend_default)

        pre_count  = len(pre_spikes_dict[syn_i])
        post_count = len(post_spikes_dict[syn_i])
        num_changes = pre_count * post_count

        trajectory = synapses_trajectories[syn_i]
        if trajectory:
            events_str = []
            for evt in trajectory:
                # 7 items => ( pre_s, post_s, pre_a, post_a, dt, dW, wA )
                if evt[0] is None and evt[1] is None:
                    # The initial "fake" row
                    wA = evt[-1]
                    evt_str = f"(NoSpikes, dt_ms=0.000, dW=0.000, W={wA:.3f})"
                else:
                    (p_s, po_s, p_a, po_a, dt_ms, dW, wA) = evt
                    evt_str = (
                        f"(pre_spike_ms={p_s}, post_spike_ms={po_s}, "
                        f"pre_arr_ms={p_a}, post_arr_ms={po_a}, "
                        f"dt_ms={dt_ms:.3f}, dW={dW:.3f}, W={wA:.3f})"
                    )
                events_str.append(evt_str)
            all_events_line = " ".join(events_str)
        else:
            all_events_line = "No step-by-step data"

        line = (f"Synapse {syn_i}: axon_delay_ms={axon_delay_ms}, dend_delay_ms={dend_delay_ms}, "
                f"#changes={num_changes}, init={w_init:.3f}, final={w_final:.3f} => "
                f"{all_events_line}")
        print(line)
    print("--------------------------------------------------")

    # 7) Raster plots with arrival times
    plot_pre_raster(arrived_pre_dict,  N, global_min_time_ms, global_max_time_ms)
    plot_post_raster(arrived_post_dict, N, global_min_time_ms, global_max_time_ms)

    # 8) Plot synaptic evolution
    plot_synaptic_evolution(synapses_trajectories,
                            global_min_time_ms, global_max_time_ms)
