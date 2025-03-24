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
# STDP Functions
###############################################################################

def std_stdp(
    pre_spike_times_ms, post_spike_times_ms,
    pre_spike_arrivals_ms, post_spike_arrivals_ms,
    A_plus, A_minus, tau_plus_ms, tau_minus_ms,
    w_init,
    writer,
    synapse_id,
    initial_time_for_plot_ms
):
    """
    Standard pair-based STDP.
    We store a trajectory of (pre_spike_ms, post_spike_ms,
                             pre_arr_ms,  post_arr_ms,
                             dt_ms, dW, w_after).

    - pre_spike_times_ms, post_spike_times_ms : the original times of the spikes.
    - pre_spike_arrivals_ms, post_spike_arrivals_ms : the arrival times (with delay).
    We do a double loop over them in parallel.

    The index in 'pre_spike_times_ms' corresponds to the same index in 'pre_spike_arrivals_ms'.
    i.e. pre_spike_times_ms[k] + axonal_delay = pre_spike_arrivals_ms[k].
    The same for post.

    We define dt_ms = post_spike_arrivals_ms[j] - pre_spike_arrivals_ms[i].
    """
    w = w_init
    trajectory = []

    # Write an initial row => no real spikes, dt=0, dW=0, W_init
    writer.writerow([synapse_id, 0, None, None, None, None, None, 0.0, w])
    # We'll store a "fake" event with no real times
    trajectory.append((None, None, None, None, 0.0, 0.0, w))

    event_idx = 1

    # Double loop over all presyn indexes, postsyn indexes
    for i_pre, pre_t_orig in enumerate(pre_spike_times_ms):
        pre_t_arr = pre_spike_arrivals_ms[i_pre]

        for j_post, post_t_orig in enumerate(post_spike_times_ms):
            post_t_arr = post_spike_arrivals_ms[j_post]

            dt_ms = post_t_arr - pre_t_arr
            if dt_ms > 0:
                dW = A_plus * np.exp(-dt_ms / tau_plus_ms)
            else:
                dW = -A_minus * np.exp(dt_ms / tau_minus_ms)

            w_new = w + dW
            time_event_ms = 0.5*(pre_t_arr + post_t_arr)  # midpoint in arrival time

            # Write to CSV: we log both original spike times and arrival times
            writer.writerow([
                synapse_id,
                event_idx,
                pre_t_orig,
                post_t_orig,
                pre_t_arr,
                post_t_arr,
                dt_ms,
                dW,
                w_new
            ])

            trajectory.append((
                pre_t_orig, post_t_orig,
                pre_t_arr, post_t_arr,
                dt_ms, dW, w_new
            ))
            w = w_new
            event_idx += 1

    return w, trajectory


def spec_stdp(
    pre_spike_times_ms, post_spike_times_ms,
    pre_spike_arrivals_ms, post_spike_arrivals_ms,
    A_plus, A_minus, tau_plus_ms, tau_minus_ms,
    w_init,
    initial_time_for_plot_ms
):
    """
    Specialized STDP (placeholder).
    We'll store a single 'fake' event with no real updates:
    (None, None, None, None, dt_ms=0, dW=0, w_init).
    """
    traj = [(None, None, None, None, 0.0, 0.0, w_init)]
    return w_init, traj


###############################################################################
# Plotting
###############################################################################

def get_synapse_color(syn_id):
    return f"C{syn_id - 1}"

def plot_synaptic_evolution(synapses_trajectories, time_min_ms, time_max_ms):
    """
    We plot the arrival-time midpoint vs the weight.
    For each event: time_event_ms = 0.5*(pre_arr_ms + post_arr_ms)
    """
    plt.figure()
    for syn_id, trajectory in synapses_trajectories.items():
        if not trajectory or len(trajectory) == 0:
            continue

        # The first row is a "fake" row => might have None times
        # We'll build arrays skipping that if needed
        times_ms = []
        weights  = []
        for idx, event in enumerate(trajectory):
            (pre_t_orig, post_t_orig,
             pre_arr, post_arr,
             dt_ms, dW, w_after) = event
            if pre_arr is not None and post_arr is not None:
                # midpoint
                time_event_ms = 0.5*(pre_arr + post_arr)
                times_ms.append(time_event_ms)
                weights.append(w_after)
            else:
                # This is the initial row => we can place it at time_min
                times_ms.append(time_min_ms)
                weights.append(w_after)

        color = get_synapse_color(syn_id)
        plt.plot(times_ms, weights, label=f"Syn {syn_id}", color=color, marker='o')

    plt.title("Synaptic Evolution (Weight vs Time in ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Weight")
    plt.xlim(time_min_ms, time_max_ms)
    plt.legend()
    plt.show()


def plot_pre_raster(pre_spikes_dict, N, time_min_ms, time_max_ms):
    """
    We plot the *raw* pre spike times (no delay) or arrival times?
    The user didn't clarify. 
    Typically, you'd either show the raw spikes or the arrival times.
    Let's show the *arrival times* if you prefer the actual timing at the synapse.

    But the user specifically said we want the same times used for the final STDP. 
    So let's confirm we want to see arrival times or raw? 
    Let's assume we want to see the "raw" times in the pre_raster and post_raster 
    so the user can see the difference.

    Actually, they've asked to unify everything to the same time axis. 
    The question is ambiguous. 
    We'll keep it consistent with our prior approach: 
    we show *arrival times* in the raster. 
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
    Raster for *post-synaptic* neurons (N+1..2N), showing arrival times if we want consistency.
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
    # 1) Load config
    with open("config_check_stdp.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2) Basic parameters
    N             = config["N"]
    stdp_rule     = config["stdp_rule"]
    csv_file_pre  = config["csv_file_pre"]
    csv_file_post = config["csv_file_post"]

    A_plus       = config.get("A_plus", 0.01)
    A_minus      = config.get("A_minus", 0.012)
    tau_plus_ms  = config.get("tau_plus_ms", 20.0)
    tau_minus_ms = config.get("tau_minus_ms", 20.0)

    # Default delay values
    axon_default  = 5.0
    dend_default  = 0.1

    # We now read them as lists of length N
    init_weights_list        = config.get("initial_weights", None)
    axonal_delays_list       = config.get("axonal_delays_ms", None)
    dendritic_delays_list    = config.get("dendritic_delays_ms", None)

    if init_weights_list is None or len(init_weights_list) != N:
        raise ValueError("initial_weights must be a list of length N.")

    # 3) Load the raw spike data
    pre_spikes_dict, pre_all_times, _   = load_spikes_pre(csv_file_pre, N)
    post_spikes_dict, post_all_times, _ = load_spikes_post(csv_file_post, N)

    # 4) Construct the arrival-time dicts based on per-synapse axonal/dend delays
    arrived_pre_dict  = {}
    arrived_post_dict = {}

    for i in range(1, N+1):
        # i-th synapse => pre neuron i, post neuron i+N
        # Get the i-th delays from the config lists
        axon_delay_ms  = safe_get_list_item(axonal_delays_list, i-1, axon_default)
        dend_delay_ms  = safe_get_list_item(dendritic_delays_list, i-1, dend_default)

        # Shift each pre spike by axon_delay, each post by dend_delay
        arrived_pre_dict[i] = [t + axon_delay_ms  for t in pre_spikes_dict[i]]
        arrived_post_dict[i]= [t + dend_delay_ms  for t in post_spikes_dict[i]]

    # Combine arrival times for min/max range
    all_arr_pre = [t for i in range(1, N+1) for t in arrived_pre_dict[i]]
    all_arr_post= [t for i in range(1, N+1) for t in arrived_post_dict[i]]
    arrived_all = all_arr_pre + all_arr_post

    if len(arrived_all) == 0:
        global_min_time_ms = 0.0
        global_max_time_ms = 100.0
    else:
        global_min_time_ms = min(arrived_all) - 10
        global_max_time_ms = max(arrived_all) + 10

    # 5) Run STDP
    final_weights = [0.0]*N
    synapses_trajectories = {}

    if stdp_rule == "std_stdp":
        summary_file = "stdp_summary.csv"
        with open(summary_file, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                "synapse_id", "event_idx",
                "pre_spike_ms", "post_spike_ms",
                "pre_arrived_ms", "post_arrived_ms",
                "dt_ms", "dW", "W"
            ])

            for syn_i in range(1, N+1):
                w_init = init_weights_list[syn_i-1]

                w_final, trajectory = std_stdp(
                    pre_spikes_dict[syn_i],   # raw pre times
                    post_spikes_dict[syn_i],  # raw post times
                    arrived_pre_dict[syn_i],  # arrived pre times
                    arrived_post_dict[syn_i], # arrived post times
                    A_plus, A_minus,
                    tau_plus_ms, tau_minus_ms,
                    w_init,
                    writer,
                    synapse_id=syn_i,
                    initial_time_for_plot_ms=global_min_time_ms
                )
                final_weights[syn_i-1] = w_final
                synapses_trajectories[syn_i] = trajectory
        print(f"STDP summary logged to {summary_file}")

    elif stdp_rule == "spec_stdp":
        for syn_i in range(1, N+1):
            w_init = init_weights_list[syn_i-1]
            w_final, trajectory = spec_stdp(
                pre_spikes_dict[syn_i],
                post_spikes_dict[syn_i],
                arrived_pre_dict[syn_i],
                arrived_post_dict[syn_i],
                A_plus, A_minus,
                tau_plus_ms, tau_minus_ms,
                w_init,
                initial_time_for_plot_ms=global_min_time_ms
            )
            final_weights[syn_i-1] = w_final
            synapses_trajectories[syn_i] = trajectory
    else:
        raise ValueError(f"Unknown STDP rule: {stdp_rule}")

    # 6) Final diagnostic print
    print("--------------------------------------------------")
    print("FINAL DIAGNOSTIC SUMMARY")
    print("For each synapse, we print: axon_delay_ms, dend_delay_ms, #changes, init, final => event list:")
    for syn_i in range(1, N+1):
        w_init  = init_weights_list[syn_i-1]
        w_final = final_weights[syn_i-1]
        axon_delay_ms  = safe_get_list_item(axonal_delays_list,   syn_i-1, 5.0)
        dend_delay_ms  = safe_get_list_item(dendritic_delays_list,syn_i-1, 0.1)

        # #changes = #pre spikes * #post spikes
        num_changes = (len(pre_spikes_dict[syn_i]) *
                       len(post_spikes_dict[syn_i]))

        # Retrieve the trajectory
        trajectory = synapses_trajectories[syn_i]
        if trajectory:
            # each item: (pre_spike_ms, post_spike_ms, pre_arr_ms, post_arr_ms, dt_ms, dW, w_after)
            events_str = []
            for evt in trajectory:
                (raw_pre, raw_post,
                 arr_pre, arr_post,
                 dt_ms, dW, w_after) = evt

                if raw_pre is None and raw_post is None:
                    # The initial "fake" row
                    evt_str = "(NoSpikes, dt_ms=0.000, dW=0.000, W={:.3f})".format(w_after)
                else:
                    evt_str = (
                        f"(pre_spike_ms={raw_pre:.3f}, post_spike_ms={raw_post:.3f}, "
                        f"pre_arr_ms={arr_pre:.3f}, post_arr_ms={arr_post:.3f}, "
                        f"dt_ms={dt_ms:.3f}, dW={dW:.3f}, W={w_after:.3f})"
                    )
                events_str.append(evt_str)
            all_events_line = " ".join(events_str)
        else:
            all_events_line = "No step-by-step data (spec_stdp)"

        line = (f"Synapse {syn_i}: "
                f"axon_delay_ms={axon_delay_ms}, dend_delay_ms={dend_delay_ms}, "
                f"#changes={num_changes}, init={w_init:.3f}, final={w_final:.3f} => "
                f"{all_events_line}")
        print(line)
    print("--------------------------------------------------")

    # 7) Raster: we want to show *arrival times* in the plots
    #    Let's rename arrived_pre_dict -> pre_raster_dict, etc.
    #    so we pass them directly to the plot. 
    pre_raster_dict  = arrived_pre_dict
    post_raster_dict = arrived_post_dict

    plot_pre_raster(pre_raster_dict, N, global_min_time_ms, global_max_time_ms)
    plot_post_raster(post_raster_dict, N, global_min_time_ms, global_max_time_ms)

    # 8) Synaptic evolution
    plot_synaptic_evolution(synapses_trajectories,
                            global_min_time_ms, global_max_time_ms)
    return






