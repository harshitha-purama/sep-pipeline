"""
Cross-Modal SEP Comparison
Compares evoked responses across two sensor modalities.

Development data: MNE somato — MEG gradiometers (grad) vs magnetometers (mag)
Maps to ds004388:  EEG (scalp cortical) vs ESG (spinal surface)

Scientific question:
  Do different recording modalities capture the same SEP components?
  How does SNR, latency, and amplitude compare between modalities?
"""

import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mne.set_log_level('WARNING')

FIGURES_DIR = ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = {
    'Gradiometer (proxy: ESG)': 'grad',   # spatially specific — like spinal ESG
    'Magnetometer (proxy: EEG)': 'mag',   # more global — like scalp EEG
}

# Search windows in ms for each SEP component
SEARCH_WINDOWS = {
    'N20': (15, 30),
    'P25': (22, 40),
    'N35': (30, 55),
    'P45': (40, 70),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_scale(raw):
    ch_types = set(raw.get_channel_types())
    if 'eeg' in ch_types:
        return 1e6, 'µV'
    elif 'grad' in ch_types:
        return 1e13, 'fT/cm'
    else:
        return 1e12, 'pT'


def process_modality(raw_orig, ch_type):
    """Full pipeline for one modality. Returns evoked and SNR array."""
    raw = raw_orig.copy().pick([ch_type, 'stim'])

    h_freq = min(200.0, raw.info['sfreq'] / 2 - 10)
    raw.filter(1.0, h_freq, fir_window='hamming')
    raw.notch_filter(50.0)

    if ch_type == 'eeg':
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()

    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)

    reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 150e-6}.get(ch_type, None)
    reject_dict = {ch_type: reject} if reject else None

    epochs = mne.Epochs(
        raw, events,
        tmin=-0.05, tmax=0.30,
        baseline=(-0.05, 0),
        reject=reject_dict,
        preload=True,
        verbose=False
    )

    evoked = epochs.average()

    # SNR: signal window 20–150ms vs baseline noise
    times = evoked.times
    sig_mask   = (times >= 0.020) & (times <= 0.150)
    noise_mask = (times >= -0.050) & (times <= 0.0)
    sig_rms   = np.sqrt(np.mean(evoked.data[:, sig_mask]  ** 2, axis=1))
    noise_rms = np.sqrt(np.mean(evoked.data[:, noise_mask] ** 2, axis=1))
    snr_db = 20 * np.log10(sig_rms / (noise_rms + 1e-30))

    return evoked, snr_db, len(epochs)


def extract_components(evoked, scale):
    """Extract peak latency and amplitude for each search window."""
    data = evoked.data * scale
    times_ms = evoked.times * 1000
    results = {}
    for name, (tmin, tmax) in SEARCH_WINDOWS.items():
        mask = (times_ms >= tmin) & (times_ms <= tmax)
        win = data[:, mask]
        win_t = times_ms[mask]
        idx = np.argmax(np.abs(win))
        ci, ti = np.unravel_index(idx, win.shape)
        results[name] = {
            'latency_ms': float(win_t[ti]),
            'amplitude':  float(win[ci, ti]),
            'channel':    evoked.ch_names[ci],
        }
    return results


def bootstrap_snr_ci(evoked, n_boot=1000, alpha=0.05):
    """Bootstrap 95% CI for mean SNR."""
    times = evoked.times
    sig_mask   = (times >= 0.020) & (times <= 0.150)
    noise_mask = (times >= -0.050) & (times <= 0.0)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, evoked.data.shape[0], evoked.data.shape[0])
        d = evoked.data[idx]
        s = np.sqrt(np.mean(d[:, sig_mask] ** 2, axis=1))
        n = np.sqrt(np.mean(d[:, noise_mask] ** 2, axis=1))
        means.append(np.mean(20 * np.log10(s / (n + 1e-30))))
    means = np.array(means)
    return means.mean(), np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_comparison(results_by_modality):
    """
    results_by_modality: dict mapping modality label →
        {'evoked', 'snr_db', 'components', 'scale', 'unit', 'n_epochs', 'snr_ci'}
    """
    modality_labels = list(results_by_modality.keys())
    colors = ['steelblue', 'crimson']
    n_mod = len(modality_labels)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # ── Panel A: Evoked waveforms overlaid (mean across channels) ─────────────
    ax_wave = fig.add_subplot(gs[0, :])
    for (label, res), color in zip(results_by_modality.items(), colors):
        evoked = res['evoked']
        scale, unit = res['scale'], res['unit']
        times_ms = evoked.times * 1000
        mean_sig = evoked.data.mean(axis=0) * scale
        sem_sig  = evoked.data.std(axis=0) * scale / np.sqrt(evoked.data.shape[0])
        ax_wave.plot(times_ms, mean_sig, color=color, linewidth=1.8, label=f'{label} (n={res["n_epochs"]} epochs)')
        ax_wave.fill_between(times_ms, mean_sig - sem_sig, mean_sig + sem_sig, alpha=0.2, color=color)

    ax_wave.axvline(0, color='gray', linestyle='--', linewidth=0.9)
    ax_wave.axhline(0, color='gray', linewidth=0.4)
    ax_wave.set_xlabel('Time (ms)')
    ax_wave.set_ylabel('Amplitude (scaled units)')
    ax_wave.set_title('A — Mean Evoked SEP Response: Gradiometer vs Magnetometer')
    ax_wave.legend(fontsize=9)
    ax_wave.set_xlim(times_ms[0], times_ms[-1])

    # Mark component windows
    for name, (tmin, tmax) in SEARCH_WINDOWS.items():
        ax_wave.axvspan(tmin, tmax, alpha=0.07, color='gold')
        ax_wave.text((tmin + tmax) / 2, ax_wave.get_ylim()[1] * 0.92, name,
                     ha='center', fontsize=7, color='darkgoldenrod')

    # ── Panel B: SNR comparison bar chart ────────────────────────────────────
    ax_snr = fig.add_subplot(gs[1, 0])
    snr_means = [results_by_modality[l]['snr_ci'][0] for l in modality_labels]
    snr_lo    = [results_by_modality[l]['snr_ci'][0] - results_by_modality[l]['snr_ci'][1] for l in modality_labels]
    snr_hi    = [results_by_modality[l]['snr_ci'][2] - results_by_modality[l]['snr_ci'][0] for l in modality_labels]
    bars = ax_snr.bar(modality_labels, snr_means, color=colors, alpha=0.75, width=0.5)
    ax_snr.errorbar(modality_labels, snr_means, yerr=[snr_lo, snr_hi],
                    fmt='none', color='black', capsize=6, linewidth=1.5)
    ax_snr.set_ylabel('Mean SNR (dB)')
    ax_snr.set_title('B — SNR Comparison (95% CI bootstrapped)')
    ax_snr.set_xticklabels(modality_labels, fontsize=8, rotation=10)
    for bar, val in zip(bars, snr_means):
        ax_snr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f} dB', ha='center', fontsize=9, fontweight='bold')

    # ── Panel C: Component latency comparison ────────────────────────────────
    ax_lat = fig.add_subplot(gs[1, 1])
    comp_names = list(SEARCH_WINDOWS.keys())
    x = np.arange(len(comp_names))
    width = 0.35
    for i, (label, color) in enumerate(zip(modality_labels, colors)):
        latencies = [results_by_modality[label]['components'][c]['latency_ms'] for c in comp_names]
        ax_lat.bar(x + i * width, latencies, width, label=label, color=color, alpha=0.75)
    ax_lat.set_xticks(x + width / 2)
    ax_lat.set_xticklabels(comp_names)
    ax_lat.set_ylabel('Peak Latency (ms)')
    ax_lat.set_title('C — SEP Component Latencies by Modality')
    ax_lat.legend(fontsize=8)

    # ── Panel D: SNR distribution violin ─────────────────────────────────────
    ax_vio = fig.add_subplot(gs[2, :])
    snr_data  = [results_by_modality[l]['snr_db'] for l in modality_labels]
    parts = ax_vio.violinplot(snr_data, showmedians=True, showextrema=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax_vio.set_xticks([1, 2])
    ax_vio.set_xticklabels(modality_labels, fontsize=9)
    ax_vio.set_ylabel('SNR per channel (dB)')
    ax_vio.set_title('D — SNR Distribution Across All Channels per Modality')

    # Wilcoxon test between modalities
    stat, pval = stats.mannwhitneyu(snr_data[0], snr_data[1], alternative='two-sided')
    sig_label = f'Mann-Whitney U: p = {pval:.4f}' + (' *' if pval < 0.05 else ' (n.s.)')
    ax_vio.text(1.5, ax_vio.get_ylim()[1] * 0.95, sig_label,
                ha='center', fontsize=9, style='italic')

    out = FIGURES_DIR / 'cross_modal_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Loading raw data ===")
    data_path = mne.datasets.somato.data_path()
    raw_fname = data_path / 'sub-01' / 'meg' / 'sub-01_task-somato_meg.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    results = {}
    for label, ch_type in MODALITIES.items():
        print(f"\n=== Processing: {label} ===")
        evoked, snr_db, n_epochs = process_modality(raw, ch_type)

        # pick right scale for this modality
        tmp = raw.copy().pick([ch_type])
        scale, unit = get_scale(tmp)
        del tmp

        components = extract_components(evoked, scale)
        print(f"  Epochs: {n_epochs} | Mean SNR: {snr_db.mean():.2f} dB")
        for cname, cvals in components.items():
            print(f"    {cname}: {cvals['latency_ms']:.1f} ms, "
                  f"{cvals['amplitude']:.3f} {unit} [{cvals['channel']}]")

        print("  Computing bootstrap CI for SNR...")
        snr_mean, snr_lo, snr_hi = bootstrap_snr_ci(evoked)

        results[label] = {
            'evoked':     evoked,
            'snr_db':     snr_db,
            'components': components,
            'scale':      scale,
            'unit':       unit,
            'n_epochs':   n_epochs,
            'snr_ci':     (snr_mean, snr_lo, snr_hi),
        }

    print("\n=== Statistical summary ===")
    labels = list(results.keys())
    snr_a = results[labels[0]]['snr_db']
    snr_b = results[labels[1]]['snr_db']
    stat, pval = stats.mannwhitneyu(snr_a, snr_b, alternative='two-sided')
    print(f"  SNR comparison ({labels[0]} vs {labels[1]})")
    print(f"  Mean SNR: {snr_a.mean():.2f} dB vs {snr_b.mean():.2f} dB")
    print(f"  Mann-Whitney U = {stat:.1f}, p = {pval:.4f}")

    print("\n=== Plotting ===")
    plot_comparison(results)

    print("\n=== Done ===")
