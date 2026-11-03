"""
Real Data SEP Pipeline — OpenNeuro ds004388, sub-001, median nerve run
Three modalities from the same recording:
  ENG (Erb's point EP)  → N9  ~9ms   peripheral nerve arrival
  ESG (spinal surface)  → N13 ~13ms  spinal cord arrival
  EEG (scalp cortex)    → N20 ~20ms  primary somatosensory cortex

This demonstrates the full ascending somatosensory pathway.
"""

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mne.set_log_level('WARNING')

DATA_FILE   = ROOT / 'data/ds004388/sub-001/eeg/sub-001_task-median_run-03_eeg.set'
CHAN_FILE    = ROOT / 'data/ds004388/sub-001/eeg/sub-001_task-median_run-03_channels.tsv'
FIGURES_DIR = ROOT / 'results' / 'figures'
RESULTS_DIR = ROOT / 'results'

# SEP component search windows per modality (ms) — non-overlapping
COMPONENT_WINDOWS = {
    'ENG':  {'N9':  (6,  14)},
    'ESG':  {'N13': (10, 18)},
    'EEG':  {'N20': (16, 26), 'P25': (26, 38), 'N35': (38, 55)},
}

# Highpass cutoff per modality
L_FREQ = {'EEG': 30.0, 'ESG': 5.0, 'ENG': 5.0}

# ── Channel mapping ───────────────────────────────────────────────────────────

def get_channel_groups():
    ch = pd.read_csv(CHAN_FILE, sep='\t')
    eeg_ch  = ch[ch['type'] == 'EEG']['name'].tolist()
    misc_ch = ch[ch['type'] == 'MISC']['name'].tolist()

    # ESG: spinal surface electrodes (S3–S36, SC1, SC6, AC, AL, Iz, L1, L4)
    esg_ch = [c for c in misc_ch if c.startswith('S') or c in
              ('SC1', 'SC6', 'AC', 'AL', 'Iz', 'L1', 'L4')]
    # ENG: peripheral nerve recording at Erb's point
    eng_ch = ['EP']

    return {'EEG': eeg_ch, 'ESG': esg_ch, 'ENG': eng_ch}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_sep(raw, l_freq=5.0):
    """
    Per-modality filter settings:
      EEG (scalp):      l_freq=30Hz — removes slow drifts, keeps cortical SEPs
      ESG (spinal):     l_freq=5Hz  — spinal SEPs have lower frequency content
      ENG (peripheral): l_freq=5Hz  — peripheral nerve signals also lower frequency
    Upper cutoff 2000Hz across all (well below 5kHz Nyquist after resample).
    """
    raw.filter(l_freq, 2000.0, fir_window='hamming')
    raw.notch_filter(50.0)
    raw.resample(5000, npad='auto')
    return raw


# ── Epoch and average ─────────────────────────────────────────────────────────

def make_epochs(raw, ch_names, reject_uv=500):
    """Epoch around stimulus, return evoked for a specific channel group."""
    raw_sub = raw.copy().pick(ch_names + ['STI 014'] if 'STI 014' in raw.ch_names else ch_names)

    events, _ = mne.events_from_annotations(raw_sub, verbose=False)
    if len(events) == 0:
        return None, 0

    epochs = mne.Epochs(
        raw_sub, events,
        tmin=-0.020, tmax=0.080,    # -20ms to +80ms — tight window for SEPs
        baseline=(-0.020, 0),
        reject=None,                # skip auto-rejection, keep all 500 trials
        preload=True,
        verbose=False
    )
    evoked = epochs.average()
    return evoked, len(epochs)


# ── Component extraction ──────────────────────────────────────────────────────

def extract_components(evoked, windows):
    """Extract peak latency and amplitude for each search window."""
    results = {}
    times_ms = evoked.times * 1000
    data_uv  = evoked.data * 1e6   # convert to µV

    for comp, (tmin, tmax) in windows.items():
        mask = (times_ms >= tmin) & (times_ms <= tmax)
        win  = data_uv[:, mask]
        win_t = times_ms[mask]
        if win.size == 0:
            continue
        ci, ti = np.unravel_index(np.argmax(np.abs(win)), win.shape)
        results[comp] = {
            'latency_ms':   round(float(win_t[ti]), 2),
            'amplitude_uv': round(float(win[ci, ti]), 4),
            'channel':      evoked.ch_names[ci],
        }
    return results


def compute_snr(evoked, signal_tmin=0.005, signal_tmax=0.060):
    times = evoked.times
    sig   = (times >= signal_tmin) & (times <= signal_tmax)
    noise = (times >= -0.020)      & (times <= 0.0)
    srms  = np.sqrt(np.mean(evoked.data[:, sig]   ** 2, axis=1))
    nrms  = np.sqrt(np.mean(evoked.data[:, noise] ** 2, axis=1))
    return 20 * np.log10(srms / (nrms + 1e-30))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ascending_pathway(results_by_modality):
    """
    Main figure: evoked waveforms for ENG → ESG → EEG side by side,
    showing the signal propagating up the somatosensory pathway.
    """
    modality_order = ['ENG', 'ESG', 'EEG']
    colors = {'ENG': 'darkorange', 'ESG': 'steelblue', 'EEG': 'forestgreen'}
    comp_colors = {'N9': 'darkorange', 'N13': 'steelblue',
                   'N20': 'forestgreen', 'P25': 'crimson', 'N35': 'purple'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Somatosensory Evoked Potentials — Ascending Pathway\n'
                 'ds004388 sub-001, median nerve stimulation',
                 fontsize=12, fontweight='bold')

    for ax, mod in zip(axes, modality_order):
        if mod not in results_by_modality:
            ax.set_visible(False)
            continue
        res = results_by_modality[mod]
        evoked = res['evoked']
        times_ms = evoked.times * 1000
        mean_sig = evoked.data.mean(axis=0) * 1e6
        sem_sig  = evoked.data.std(axis=0)  * 1e6 / np.sqrt(evoked.data.shape[0])

        ax.plot(times_ms, mean_sig, color=colors[mod], lw=1.8,
                label=f'{mod} (n={res["n_epochs"]} epochs)')
        ax.fill_between(times_ms, mean_sig - sem_sig, mean_sig + sem_sig,
                        alpha=0.2, color=colors[mod])
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.axhline(0, color='gray', lw=0.4)

        # Mark components
        for comp, vals in res['components'].items():
            c = comp_colors.get(comp, 'gray')
            ax.axvline(vals['latency_ms'], color=c, ls=':', lw=1.5)
            ax.annotate(f"{comp}\n{vals['latency_ms']:.1f}ms\n{vals['amplitude_uv']:.2f}µV",
                        xy=(vals['latency_ms'], vals['amplitude_uv']),
                        xytext=(vals['latency_ms'] + 3, vals['amplitude_uv']),
                        fontsize=7.5, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.8))

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'{mod} — {res["location"]}')
        ax.set_xlim(times_ms[0], times_ms[-1])
        snr = res['snr_db']
        ax.legend(fontsize=8, title=f'SNR: {snr.mean():.1f} dB')

    plt.tight_layout()
    out = FIGURES_DIR / 'real_data_ascending_pathway.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_snr_comparison(results_by_modality):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cross-Modal SNR — Real ds004388 Data', fontsize=11, fontweight='bold')

    modalities = [m for m in ['ENG', 'ESG', 'EEG'] if m in results_by_modality]
    colors = {'ENG': 'darkorange', 'ESG': 'steelblue', 'EEG': 'forestgreen'}

    # Bar chart: mean SNR per modality
    ax = axes[0]
    means = [results_by_modality[m]['snr_db'].mean() for m in modalities]
    bars = ax.bar(modalities, means,
                  color=[colors[m] for m in modalities], alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f} dB', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean SNR (dB)')
    ax.set_title('Mean SNR per Modality')

    # Violin: SNR distribution
    ax = axes[1]
    snr_data = [results_by_modality[m]['snr_db'] for m in modalities]
    parts = ax.violinplot(snr_data, showmedians=True)
    for pc, mod in zip(parts['bodies'], modalities):
        pc.set_facecolor(colors[mod])
        pc.set_alpha(0.65)
    ax.set_xticks(range(1, len(modalities) + 1))
    ax.set_xticklabels(modalities)
    ax.set_ylabel('SNR per channel (dB)')
    ax.set_title('SNR Distribution Across Channels')

    if len(modalities) >= 2:
        stat, pval = stats.kruskal(*snr_data)
        ax.text(0.5, 0.97, f'Kruskal-Wallis p = {pval:.4f}',
                transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    out = FIGURES_DIR / 'real_data_snr_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

MODALITY_LOCATIONS = {
    'ENG': 'Erb\'s point (peripheral nerve)',
    'ESG': 'Spinal surface electrodes',
    'EEG': 'Scalp (somatosensory cortex)',
}

if __name__ == '__main__':
    print("=== Loading channel map ===")
    ch_groups = get_channel_groups()
    for mod, chs in ch_groups.items():
        print(f"  {mod}: {len(chs)} channels — {chs[:5]}{'...' if len(chs)>5 else ''}")

    # Peek at channel names without loading data
    raw_ref = mne.io.read_raw_eeglab(DATA_FILE, preload=False)
    all_ch = raw_ref.ch_names
    del raw_ref
    print("\n=== Processing each modality in isolation (memory-efficient) ===")

    results = {}
    for mod, ch_names in ch_groups.items():
        valid = [c for c in ch_names if c in all_ch]
        if not valid:
            print(f"\n  {mod}: no matching channels found, skipping")
            continue

        print(f"\n=== {mod} ({len(valid)} channels) ===")
        # Load ONLY this modality's channels into memory
        raw = mne.io.read_raw_eeglab(DATA_FILE, preload=False)
        raw.pick(valid)
        raw.load_data()
        raw = preprocess_for_sep(raw, l_freq=L_FREQ[mod])
        print(f"  Loaded and preprocessed @ {raw.info['sfreq']} Hz  (highpass {L_FREQ[mod]} Hz)")

        evoked, n_epochs = make_epochs(raw, valid)
        del raw   # free memory immediately after epoching
        if evoked is None:
            print(f"  No events found, skipping")
            continue

        components = extract_components(evoked, COMPONENT_WINDOWS[mod])
        snr_db = compute_snr(evoked)

        print(f"  Epochs: {n_epochs} | Mean SNR: {snr_db.mean():.2f} dB")
        for comp, vals in components.items():
            print(f"    {comp}: {vals['latency_ms']} ms, "
                  f"{vals['amplitude_uv']} µV [{vals['channel']}]")

        results[mod] = {
            'evoked':     evoked,
            'snr_db':     snr_db,
            'components': components,
            'n_epochs':   n_epochs,
            'location':   MODALITY_LOCATIONS[mod],
        }

    print("\n=== Generating figures ===")
    plot_ascending_pathway(results)
    plot_snr_comparison(results)

    print("\n=== Saving results table ===")
    rows = []
    for mod, res in results.items():
        for comp, vals in res['components'].items():
            rows.append({
                'Modality': mod, 'Location': res['location'],
                'Component': comp,
                'Latency (ms)': vals['latency_ms'],
                'Amplitude (µV)': vals['amplitude_uv'],
                'Channel': vals['channel'],
                'Mean SNR (dB)': round(float(res['snr_db'].mean()), 2),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / 'real_data_results.csv', index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'real_data_results.csv'}")
    print("\n=== Done ===")
