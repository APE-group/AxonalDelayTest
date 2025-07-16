#  predict_stdp_alpha_forced_pl_lib_zero_based.py
#  Version: 2025‑07‑16  —  works with 0‑based spike files
#
#  Copyright © 2025   Pier Stanislao Paolucci   <pier.paolucci@roma1.infn.it>
#  Copyright © 2025   Elena Pastorelli          <elena.pastorelli@roma1.infn.it>
#  SPDX‑License‑Identifier: GPL‑3.0‑or‑later
# ----------------------------------------------------------------------

import csv
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Helper utilities
###############################################################################

def safe_get_list_item(lst, index, default_val):
    """Return lst[index] if it exists, otherwise default_val."""
    return default_val if lst is None or index >= len(lst) else lst[index]


# ------------------------------------------------------------------ #
# ---- 0‑based spike‑file loaders ---------------------------------- #
# ------------------------------------------------------------------ #
def load_spikes_pre(csv_file, N):
    """
    Expect CSV columns:  senders, times
    PRE neuron IDs: 0 … N‑1
    """
    pre = {i: [] for i in range(N)}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            nid  = int(row["senders"])
            time = float(row["times"])
            if not (0 <= nid < N):
                raise ValueError(f"PRE neuron_id {nid} not in 0..{N-1}")
            pre[nid].append(time)
    for lst in pre.values():
        lst.sort()
    return pre


def load_spikes_post(csv_file, N):
    """
    Expect CSV columns:  senders, times
    POST neuron IDs: N … 2 N‑1  (index   i = nid - N   runs 0 … N‑1)
    """
    post = {i: [] for i in range(N)}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            nid  = int(row["senders"])
            time = float(row["times"])
            if not (N <= nid < 2*N):
                raise ValueError(f"POST neuron_id {nid} not in {N}..{2*N-1}")
            post[nid - N].append(time)
    for lst in post.values():
        lst.sort()
    return post


###############################################################################
# STDP kernel  (unchanged except for 0‑based IDs)
###############################################################################

def stdp_pl_synapse_hom_causal(
        syn_id,
        pre_raw, post_raw,
        pre_arr, post_arr,
        tau_plus_ms, lam, alpha, mu, w0,
        w_init,
        writer):

    w = w_init
    traj = []

    # dummy first line
    writer.writerow([syn_id, 0, None, None, None, None, None, 0.0, w])
    traj.append((None, None, None, None, 0.0, 0.0, 0.0, w))

    # enumerate all pairs
    ev = []
    for i_pre, t_pre_raw in enumerate(pre_raw):
        t_pre_arr = pre_arr[i_pre]
        for j_post, t_post_raw in enumerate(post_raw):
            t_post_arr = post_arr[j_post]
            dt = t_post_arr - t_pre_arr
            ev_time = t_post_arr if dt > 0 else t_pre_arr
            ev.append((ev_time, i_pre, j_post,
                       dt, t_pre_raw, t_post_raw,
                       t_pre_arr, t_post_arr))
    ev.sort(key=lambda x: x[0])

    idx = 1
    i = 0
    while i < len(ev):
        cur_time, i_pre, j_post, dt, tpr, tpo, tpa, tpoa = ev[i]

        if dt == 0:        # ignore exact coincidences
            i += 1
            continue

        batch = []
        if dt > 0:         # LTP  – group by (post_arr, j_post)
            while i < len(ev) and ev[i][3] > 0 and ev[i][0] == cur_time and ev[i][2] == j_post:
                batch.append(ev[i]); i += 1
            dws = [lam * (w0**(1-mu)) * (w**mu) * np.exp(-b[3]/tau_plus_ms) for b in batch]
        else:               # LTD  – group by (pre_arr, i_pre)
            while i < len(ev) and ev[i][3] < 0 and ev[i][0] == cur_time and ev[i][1] == i_pre:
                batch.append(ev[i]); i += 1
            dws = [-lam * alpha * w * np.exp(b[3]/tau_plus_ms) for b in batch]

        w_before = w
        for ev_line, dW in zip(batch, dws):
            tpr, tpo, tpa, tpoa = ev_line[4:8]
            writer.writerow([syn_id, idx, tpr, tpo, tpa, tpoa,
                             ev_line[3], w_before, dW, None])
            traj.append((tpr, tpo, tpa, tpoa, ev_line[3], w_before, dW, None))
            idx += 1
        w += sum(dws)
        writer.writerow([syn_id, idx, tpr, None, tpa, None, 0.0,
                         w_before, sum(dws), w])
        traj.append((tpr, None, tpa, None, 0.0, w_before, sum(dws), w))
        idx += 1

    return w, traj


###############################################################################
# Plot helpers  (now 0‑based labels)
###############################################################################

def get_syn_color(k):      # cycle through C0…C9
    return f"C{k % 10}"


def plot_syn_evolution(trajs, tmin, tmax, start_syn, end_syn, save):
    plt.figure()
    for sid, tr in trajs.items():
        if not (start_syn <= sid <= end_syn):
            continue
        x, y = [], []
        for ev in tr:
            if ev[-1] is None:           # partial lines
                continue
            when = ev[3] if ev[4] > 0 else ev[2]  # post_arr vs pre_arr
            if when is None:
                when = tmin
            x.append(when); y.append(ev[-1])
        plt.plot(x, y, marker="o", color=get_syn_color(sid), label=f"Syn {sid}")
    plt.xlim(tmin, tmax)
    plt.xlabel("Time (ms)"); plt.ylabel("Weight")
    plt.title(f"PRED: Synaptic evolution (Syn {start_syn}…{end_syn})")
    plt.legend()
    if save:
        plt.savefig("predicted_synaptic_evolution.png")


def plot_raster(spike_dict, offset, tmin, tmax, start_syn, end_syn, label, fname):
    plt.figure()
    plt.title(label); plt.xlabel("Time (ms)"); plt.ylabel("Neuron ID")
    for i in range(start_syn, end_syn+1):
        times = spike_dict[i]
        plt.scatter(times, [i+offset]*len(times),
                    color=get_syn_color(i), marker='.')
    plt.xlim(tmin, tmax)
    plt.yticks(range(start_syn+offset, end_syn+offset+1))
    if fname:
        plt.savefig(fname)


###############################################################################
# Public entry point
###############################################################################

def predict_stdp_alpha_forced_pl(cfg):
    """
    Analytical predictor for the new 0‑based spike files.
    Returns {syn_id: {"syn_ID", "start_syn_value", "final_syn_value"}}
    """

    # --- config -------------------------------------------------------
    verbose      = cfg["verbose_pred"]
    tau_plus_ms  = cfg["stdp_params"]["tau_plus"]
    lam          = cfg["stdp_params"]["lambda"]
    alpha        = cfg["stdp_params"]["alpha"]
    mu           = cfg["stdp_params"]["mu"]
    w0           = cfg["w_0"]

    N            = cfg["N"]
    csv_pre      = cfg["csv_file_pre"]
    csv_post     = cfg["csv_file_post"]

    start_syn           = cfg["start_syn"]
    end_syn           = cfg["end_syn"]
    if not (0 <= start_syn <= end_syn < N):
        raise ValueError("start_syn / end_synapse outside 0‑based range")

    plot_save    = cfg["prediction_plot_save"]

    W_init       = cfg["W_init"]
    ax_delays    = cfg.get("axonal_delay_ms",    [])
    den_delays   = cfg.get("dendritic_delay_ms", [])

    # --- load spikes --------------------------------------------------
    pre_raw  = load_spikes_pre(csv_pre,  N)
    post_raw = load_spikes_post(csv_post, N)

    pre_arr  = {i: [t + safe_get_list_item(ax_delays,  i, 5.0)
                    for t in pre_raw[i]]              for i in range(N)}
    post_arr = {i: [t + safe_get_list_item(den_delays, i, 0.1)
                    for t in post_raw[i]]             for i in range(N)}

    all_arr  = [t for lst in pre_arr.values()  for t in lst] + \
               [t for lst in post_arr.values() for t in lst]
    tmin, tmax = (0.0, 100.0) if not all_arr else (min(all_arr)-10, max(all_arr)+10)

    # --- main loop ----------------------------------------------------
    trajs = {}
    w_final = []
    with open("stdp_evolution_line_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["synapse_id", "event_idx",
                         "pre_spike_ms", "post_spike_ms",
                         "pre_arrived_ms", "post_arrived_ms",
                         "dt_ms", "w_before", "dW", "w_after"])

        for i in range(N):
            wf, tr = stdp_pl_synapse_hom_causal(
                i, pre_raw[i], post_raw[i],
                pre_arr[i], post_arr[i],
                tau_plus_ms, lam, alpha, mu, w0,
                W_init[i], writer)
            w_final.append(wf); trajs[i] = tr

    # --- summary dict -------------------------------------------------
    summary = {i: {"syn_ID": i,
                   "start_syn_value": W_init[i],
                   "final_syn_value": w_final[i]}
               for i in range(start_syn, end_syn+1)}

    # --- optional verbose --------------------------------------------
    if verbose:
        for i in range(start_syn, end_syn+1):
            print(f"Syn {i}: init={W_init[i]:.4f}  →  final={w_final[i]:.4f}")

    # --- plots --------------------------------------------------------
    plot_raster(pre_arr, 0, tmin, tmax, start_syn, end_syn,
                "PRED: PRE raster", 
                "predicted_presyn_raster.png" if plot_save else None)
    plot_raster(post_arr, N, tmin, tmax, start_syn, end_syn,
                "PRED: POST raster", 
                "predicted_postsyn_raster.png" if plot_save else None)
    plot_syn_evolution(trajs, tmin, tmax, start_syn, end_syn, plot_save)

    return summary
