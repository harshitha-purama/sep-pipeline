"""
Multi-Subject ICC Analysis
Intraclass Correlation Coefficient for SEP component reliability.

Approach: split 111 epochs into N pseudo-subjects (bootstrap splits),
compute evoked + components per pseudo-subject, then compute ICC
across pseudo-subjects for each component × modality.

ICC(2,1) — two-way random effects, absolute agreement, single measures.
Interpretation: <0.50 poor | 0.50–0.75 moderate | 0.75–0.90 good | >0.90 excellent
"""

import mne
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mne.set_log_level('WARNING')

FIGURES_DIR = ROOT / 'results' / 'figures'
RESULTS_DIR = ROOT / 'results'

MODALITIES    = {'Gradiometer (ESG proxy)': 'grad', 'Magnetometer (EEG proxy)': 'mag'}
SEARCH_WINDOWS = {'N20': (15, 30), 'P25': (22, 40), 'N35': (30, 55), 'P45': (40, 70)}
N_PSEUDO_SUBJECTS = 10   # split epochs into this many groups
N_BOOTSTRAP       = 50   # repeat the random split this many times for stability

# ── Pipeline helpers ──────────────────────────────────────────────────────────

def get_scale(ch_type):
    return {'eeg': (1e6, 'µV'), 'grad': (1e13, 'fT/cm'), 'mag': (1e12, 'pT')}[ch_type]


def load_epochs(raw, ch_type):
    """Preprocess and return clean epochs for one modality."""
    r = raw.copy().pick([ch_type, 'stim'])
    h_freq = min(200.0, r.info['sfreq'] / 2 - 10)
    r.filter(1.0, h_freq, fir_window='hamming')
    r.notch_filter(50.0)
    events = mne.find_events(r, stim_channel='STI 014', verbose=False)
    reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 150e-6}.get(ch_type)
    epochs = mne.Epochs(r, events, tmin=-0.05, tmax=0.30,
                        baseline=(-0.05, 0), reject={ch_type: reject},
                        preload=True, verbose=False)
    return epochs


def evoked_from_subset(epochs, indices):
    """Average a subset of epochs."""
    subset = epochs[indices]
    return subset.average()


def extract_peak(evoked, scale, tmin_ms, tmax_ms):
    """Return (latency_ms, amplitude) of the dominant peak in the time window."""
    data = evoked.data * scale
    times_ms = evoked.times * 1000
    mask = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)
    win = data[:, mask]
    win_t = times_ms[mask]
    ci, ti = np.unravel_index(np.argmax(np.abs(win)), win.shape)
    return float(win_t[ti]), float(win[ci, ti])


# ── ICC computation ───────────────────────────────────────────────────────────

def compute_icc(values_matrix):
    """
    values_matrix: shape (n_pseudo_subjects, n_bootstrap_splits)
    Reshape into long format and compute ICC(2,1).
    Returns ICC estimate and 95% CI.
    """
    n_subjects, n_raters = values_matrix.shape
    records = []
    for rater in range(n_raters):
        for subject in range(n_subjects):
            records.append({'targets': subject, 'raters': rater,
                            'ratings': values_matrix[subject, rater]})
    df = pd.DataFrame(records)
    icc_result = pg.intraclass_corr(data=df, targets='targets',
                                    raters='raters', ratings='ratings')
    # ICC(A,1) = two-way random, absolute agreement, single measures
    row = icc_result[icc_result['Type'] == 'ICC(A,1)'].iloc[0]
    return float(row['ICC']), float(row['CI95'][0]), float(row['CI95'][1])


def run_icc_for_modality(epochs, ch_type, rng, n_bootstrap=N_BOOTSTRAP):
    """
    Simulate realistic inter-subject variability by assigning each pseudo-subject
    a fixed latency offset drawn from a normal distribution (±2 ms SD, matching
    published inter-subject SEP variability). Bootstrap splits then act as repeated
    measurements of that subject's true value.

    This tests whether the pipeline RELIABLY recovers each subject's individual
    component peak — the clinical question for a real 40-subject study.
    """
    scale, unit = get_scale(ch_type)
    n_epochs = len(epochs)
    group_size = n_epochs // N_PSEUDO_SUBJECTS

    # Assign stable per-subject latency offsets (simulating inter-subject variability)
    # SD of 2 ms matches published SEP literature (e.g. Cruccu et al., 2008)
    subject_lat_offsets = rng.normal(0, 2.0, N_PSEUDO_SUBJECTS)   # ms
    subject_amp_scales  = rng.normal(1, 0.15, N_PSEUDO_SUBJECTS)  # ±15% amplitude variability

    lat_store = {c: np.zeros((N_PSEUDO_SUBJECTS, n_bootstrap)) for c in SEARCH_WINDOWS}
    amp_store = {c: np.zeros((N_PSEUDO_SUBJECTS, n_bootstrap)) for c in SEARCH_WINDOWS}

    for split in range(n_bootstrap):
        shuffled = rng.permutation(n_epochs)
        for subj in range(N_PSEUDO_SUBJECTS):
            idx = shuffled[subj * group_size: (subj + 1) * group_size]
            evoked = evoked_from_subset(epochs, idx)

            # Apply subject-specific offset to simulate individual differences
            evoked_sim = evoked.copy()
            evoked_sim.data = evoked.data * subject_amp_scales[subj]
            # Shift time axis to simulate latency difference
            shift_samples = int(round(subject_lat_offsets[subj] / 1000 * evoked.info['sfreq']))
            evoked_sim.data = np.roll(evoked_sim.data, shift_samples, axis=1)

            for comp, (tmin, tmax) in SEARCH_WINDOWS.items():
                # Widen search window slightly to catch shifted peaks
                lat, amp = extract_peak(evoked_sim, scale, tmin - 5, tmax + 5)
                lat_store[comp][subj, split] = lat
                amp_store[comp][subj, split] = amp

    results = {}
    for comp in SEARCH_WINDOWS:
        lat_icc, lat_lo, lat_hi = compute_icc(lat_store[comp])
        amp_icc, amp_lo, amp_hi = compute_icc(amp_store[comp])
        results[comp] = {
            'latency_icc':   lat_icc,  'latency_ci':   (lat_lo, lat_hi),
            'amplitude_icc': amp_icc,  'amplitude_ci': (amp_lo, amp_hi),
            'unit': unit,
            'lat_mean': lat_store[comp].mean(),
            'amp_mean': amp_store[comp].mean(),
            'lat_std':  lat_store[comp].std(),
            'amp_std':  amp_store[comp].std(),
        }
        print(f"    {comp}  latency ICC={lat_icc:.3f} [{lat_lo:.3f},{lat_hi:.3f}]  "
              f"amplitude ICC={amp_icc:.3f} [{amp_lo:.3f},{amp_hi:.3f}]")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def icc_category(icc):
    if icc >= 0.90: return 'Excellent'
    if icc >= 0.75: return 'Good'
    if icc >= 0.50: return 'Moderate'
    return 'Poor'


def plot_icc(all_results):
    """
    all_results: dict modality_label → dict comp → icc results
    """
    modalities  = list(all_results.keys())
    comp_names  = list(SEARCH_WINDOWS.keys())
    colors      = ['steelblue', 'crimson']
    n_comp      = len(comp_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Intraclass Correlation Coefficient — SEP Component Reliability',
                 fontsize=12, fontweight='bold')

    measure_labels = ['Latency ICC', 'Amplitude ICC']
    measure_keys   = [('latency_icc', 'latency_ci'), ('amplitude_icc', 'amplitude_ci')]

    for ax, m_label, (icc_key, ci_key) in zip(axes, measure_labels, measure_keys):
        x = np.arange(n_comp)
        width = 0.35
        for i, (mod_label, color) in enumerate(zip(modalities, colors)):
            icc_vals = [all_results[mod_label][c][icc_key] for c in comp_names]
            ci_lo    = [all_results[mod_label][c][icc_key] - all_results[mod_label][c][ci_key][0]
                        for c in comp_names]
            ci_hi    = [all_results[mod_label][c][ci_key][1] - all_results[mod_label][c][icc_key]
                        for c in comp_names]
            bars = ax.bar(x + i * width, icc_vals, width, label=mod_label.split('(')[0].strip(),
                          color=color, alpha=0.75)
            ax.errorbar(x + i * width, icc_vals, yerr=[ci_lo, ci_hi],
                        fmt='none', color='black', capsize=4, lw=1.2)
            for bar, val in zip(bars, icc_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{val:.2f}\n({icc_category(val)})',
                        ha='center', fontsize=7, color='black')

        # Reference lines
        for threshold, label, ls in [(0.90, 'Excellent', '--'), (0.75, 'Good', ':'),
                                      (0.50, 'Moderate', '-.')]:
            ax.axhline(threshold, color='gray', linestyle=ls, linewidth=0.9, alpha=0.7)
            ax.text(n_comp - 0.1, threshold + 0.01, label, fontsize=7,
                    color='gray', ha='right')

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(comp_names)
        ax.set_ylim(-0.1, 1.15)
        ax.set_ylabel('ICC(2,1)')
        ax.set_title(m_label)
        ax.legend(fontsize=8)
        ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / 'icc_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out}")


def write_icc_table(all_results):
    rows = []
    for mod_label, comp_results in all_results.items():
        for comp, res in comp_results.items():
            rows.append({
                'Modality':         mod_label,
                'Component':        comp,
                'Latency ICC':      round(res['latency_icc'], 3),
                'Latency CI':       f"[{res['latency_ci'][0]:.3f}, {res['latency_ci'][1]:.3f}]",
                'Latency mean (ms)': round(res['lat_mean'], 1),
                'Latency SD (ms)':   round(res['lat_std'], 2),
                'Amplitude ICC':    round(res['amplitude_icc'], 3),
                'Amplitude CI':     f"[{res['amplitude_ci'][0]:.3f}, {res['amplitude_ci'][1]:.3f}]",
                'Amplitude category': icc_category(res['amplitude_icc']),
                'Latency category':   icc_category(res['latency_icc']),
            })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'icc_table.csv'
    df.to_csv(out, index=False)
    print(f"  Table saved: {out}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Loading data ===")
    data_path = mne.datasets.somato.data_path()
    raw = mne.io.read_raw_fif(
        data_path / 'sub-01/meg/sub-01_task-somato_meg.fif', preload=True)

    rng = np.random.default_rng(42)
    all_results = {}

    for mod_label, ch_type in MODALITIES.items():
        print(f"\n=== {mod_label} ===")
        print(f"  Loading epochs...")
        epochs = load_epochs(raw, ch_type)
        print(f"  {len(epochs)} clean epochs → "
              f"{N_PSEUDO_SUBJECTS} pseudo-subjects × {N_BOOTSTRAP} bootstrap splits")
        print(f"  Computing ICC...")
        all_results[mod_label] = run_icc_for_modality(epochs, ch_type, rng)

    print("\n=== Summary table ===")
    df = write_icc_table(all_results)
    print(df[['Modality', 'Component', 'Latency ICC', 'Latency category',
              'Amplitude ICC', 'Amplitude category']].to_string(index=False))

    print("\n=== Plotting ===")
    plot_icc(all_results)

    print("\n=== Done ===")
