"""Corrected SEP analysis for OpenNeuro ds004388 (median nerve, left wrist).

Replaces the first-pass peak picking in real_data_pipeline.py, which took the
largest absolute value across all channels. Here each component is read from
one channel chosen in advance, with the sign fixed in advance:

  N9   Erb's point (EP)                 negative peak, 7-13 ms
  N13  cervical electrode SC6, ref AC   negative peak, 11-16 ms
  N20  right parietal CP4, ref Fz       negative peak, 17-24 ms

The stimulus artefact is interpolated out of the continuous data before any
filtering, so it cannot ring into the response window. A peak only counts if it
is 3 baseline SDs from zero and not on the edge of its search window.

Cardiac artefact is not removed. Stimuli are not locked to the heartbeat, so
averaging reduces it, but single-trial work would need proper removal.
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ds004388"
RESULTS = ROOT / "results"

BAND = (30.0, 400.0)
ARTEFACT_MS = (-2.0, 5.0)
EPOCH_MS = (-100.0, 80.0)
BASELINE_MS = (-100.0, -10.0)
MIN_Z = 3.0
EDGE_MS = 0.4

# component: (signal channel, reference channel or None, search window in ms)
COMPONENTS = {
    "N9": ("EP", None, (7.0, 13.0)),
    "N13": ("SC6", "AC", (11.0, 16.0)),
    "N20": ("CP4", "Fz", (17.0, 24.0)),
}


def interpolate_artefact(data, sfreq, stim_samples, window_ms=ARTEFACT_MS):
    """PCHIP across the stimulus artefact for each channel, in place."""
    pre = int(round(-window_ms[0] * sfreq / 1000))
    post = int(round(window_ms[1] * sfreq / 1000))
    support = max(4, int(round(0.002 * sfreq)))
    for s in stim_samples:
        a, b = s - pre, s + post
        if a - support < 0 or b + support >= data.shape[1]:
            continue
        x = np.r_[np.arange(a - support, a), np.arange(b, b + support)]
        for ch in range(data.shape[0]):
            data[ch, a:b] = PchipInterpolator(x, data[ch, x])(np.arange(a, b))
    return data


def stimulus_samples(raw):
    events, ids = mne.events_from_annotations(raw, verbose=False)
    codes = [v for k, v in ids.items() if "stim" in k.lower()] or list(ids.values())
    return events[np.isin(events[:, 2], codes), 0] - raw.first_samp


def average_response(raw, signal, reference, stim):
    """Mean response of signal minus reference, in microvolts."""
    sfreq = raw.info["sfreq"]
    picks = [signal] + ([reference] if reference else [])
    data = raw.copy().pick(picks).load_data().get_data()
    data = interpolate_artefact(data, sfreq, stim)
    x = data[0] - data[1] if reference else data[0]
    x = mne.filter.filter_data(x[np.newaxis], sfreq, *BAND, verbose=False)[0]

    pre = int(round(-EPOCH_MS[0] * sfreq / 1000))
    post = int(round(EPOCH_MS[1] * sfreq / 1000))
    keep = stim[(stim - pre >= 0) & (stim + post < x.size)]
    trials = np.stack([x[s - pre:s + post] for s in keep])
    times_ms = np.arange(-pre, post) / sfreq * 1000
    base = (times_ms >= BASELINE_MS[0]) & (times_ms <= BASELINE_MS[1])
    trials = trials - trials[:, base].mean(axis=1, keepdims=True)
    return times_ms, trials.mean(axis=0) * 1e6, len(keep)


def find_peak(times_ms, avg_uv, window):
    """Most negative point in the window, and whether it can be trusted."""
    idx = np.where((times_ms >= window[0]) & (times_ms <= window[1]))[0]
    i = idx[np.argmin(avg_uv[idx])]
    base = (times_ms >= BASELINE_MS[0]) & (times_ms <= BASELINE_MS[1])
    z = abs(avg_uv[i]) / avg_uv[base].std()
    lat = float(times_ms[i])
    edge = lat - window[0] < EDGE_MS or window[1] - lat < EDGE_MS
    is_local_min = avg_uv[i] <= avg_uv[i - 1] and avg_uv[i] <= avg_uv[i + 1]
    return {
        "latency_ms": round(lat, 2),
        "amplitude_uv": round(float(avg_uv[i]), 3),
        "z": round(float(z), 2),
        "valid": bool(z >= MIN_Z and is_local_min and not edge and avg_uv[i] < 0),
    }


def analyse_raw(raw):
    stim = stimulus_samples(raw)
    rows, traces = [], {}
    for comp, (signal, reference, window) in COMPONENTS.items():
        t, avg, n = average_response(raw, signal, reference, stim)
        traces[comp] = (t, avg)
        rows.append({"component": comp, "n_trials": n, **find_peak(t, avg, window)})
    return pd.DataFrame(rows), traces


def conduction_time(df):
    """N13 to N20 for each subject where both peaks are valid."""
    good = df[df.valid].pivot(index="subject", columns="component", values="latency_ms")
    if not {"N13", "N20"} <= set(good.columns):
        return pd.Series(dtype=float)
    return (good["N20"] - good["N13"]).dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True, help="e.g. sub-001 sub-002")
    args = parser.parse_args()

    tables, figure_traces = [], {}
    for sub in args.subjects:
        files = sorted((DATA / sub / "eeg").glob(f"{sub}_task-median_run-*_eeg.set"))
        if not files:
            print(f"{sub}: no median-nerve file, skipped")
            continue
        raw = mne.io.read_raw_eeglab(files[0], preload=False, verbose=False)
        table, traces = analyse_raw(raw)
        table.insert(0, "subject", sub)
        tables.append(table)
        figure_traces[sub] = traces
        print(table.to_string(index=False))

    df = pd.concat(tables, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sep_fixed_channels.csv", index=False)
    summary = df[df.valid].groupby("component").agg(
        n_valid=("latency_ms", "size"),
        latency_mean_ms=("latency_ms", "mean"),
        latency_sd_ms=("latency_ms", "std"),
        amplitude_mean_uv=("amplitude_uv", "mean"),
    ).round(2)
    summary.to_csv(RESULTS / "sep_fixed_channels_summary.csv")
    print(summary.to_string())
    cct = conduction_time(df)
    print(f"Subjects: {df.subject.nunique()} | N13 to N20: n={len(cct)}, "
          f"mean={cct.mean():.2f} ms, sd={cct.std():.2f} ms")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, comp in zip(axes, COMPONENTS):
        stack = []
        for traces in figure_traces.values():
            t, avg = traces[comp]
            ax.plot(t, avg, color="0.8", lw=0.8)
            stack.append(avg)
        ax.plot(t, np.mean(stack, axis=0), color="k", lw=1.8)
        lo, hi = COMPONENTS[comp][2]
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.1)
        ax.set(title=f"{comp}: {COMPONENTS[comp][0]}", xlabel="Time (ms)", xlim=(-10, 50))
    axes[0].set_ylabel("Amplitude (µV)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(exist_ok=True)
    fig.savefig(RESULTS / "figures" / "sep_fixed_channels.png", dpi=150)


if __name__ == "__main__":
    main()
