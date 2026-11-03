"""
SEP Analysis Pipeline
Somatosensory Evoked Potentials — preprocessing, component extraction, SNR analysis.

Development data: MNE somato dataset (fingertip MEG/EEG, median nerve stimulation)
Target data: OpenNeuro ds004388 (EEG + ESG spinal cord SEPs)
"""

import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for codespace
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mne.set_log_level('WARNING')

RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_somato_data():
    data_path = mne.datasets.somato.data_path()
    raw_fname = data_path / 'sub-01' / 'meg' / 'sub-01_task-somato_meg.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    print(f"Loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz, "
          f"{raw.times[-1]:.1f}s duration")
    return raw

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────

def preprocess(raw):
    # Detect available data channel types and pick accordingly
    ch_types = set(raw.get_channel_types())
    if 'eeg' in ch_types:
        raw.pick(['eeg', 'stim'])
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        ref_msg = ', avg reference'
    elif 'grad' in ch_types:
        # MEG gradiometers are best for somatosensory evoked fields
        raw.pick(['grad', 'stim'])
        ref_msg = ''
    else:
        raw.pick(['mag', 'stim'])
        ref_msg = ''

    print(f"Using channel type: {[t for t in ch_types if t in ('eeg','grad','mag')][0]}")

    # Bandpass: 1 Hz–(Nyquist-10) Hz. SEPs live in 30–200 Hz; cap at Nyquist.
    h_freq = min(200.0, raw.info['sfreq'] / 2 - 10)
    raw.filter(1.0, h_freq, fir_window='hamming')

    # Notch filter at 50 Hz (European power line)
    raw.notch_filter(50.0)

    print(f"Preprocessed: bandpass 1–200 Hz, notch 50 Hz{ref_msg}")
    return raw

# ── 3. EPOCHING ───────────────────────────────────────────────────────────────

def epoch_data(raw):
    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
    print(f"Found {len(events)} stimulus events")

    # Epoch: -50ms to +300ms around each stimulus
    ch_types = set(raw.get_channel_types())
    if 'eeg' in ch_types:
        reject = {'eeg': 150e-6}       # 150 µV for EEG
    elif 'grad' in ch_types:
        reject = {'grad': 4000e-13}    # 4000 fT/cm for MEG gradiometers
    else:
        reject = {'mag': 4e-12}        # 4 pT for MEG magnetometers

    epochs = mne.Epochs(
        raw, events,
        tmin=-0.05, tmax=0.30,
        baseline=(-0.05, 0),
        reject=reject,
        preload=True,
        verbose=False
    )

    print(f"Epochs: {len(epochs)} retained after artifact rejection "
          f"(dropped {len(events) - len(epochs)})")
    return epochs

# ── 4. AVERAGE → EVOKED ───────────────────────────────────────────────────────

def compute_evoked(epochs):
    evoked = epochs.average()
    print(f"Evoked response computed over {evoked.nave} epochs")
    return evoked

# ── 5. SEP COMPONENT EXTRACTION ───────────────────────────────────────────────

def extract_components(evoked, search_windows):
    """
    search_windows: dict mapping component name to (tmin, tmax) in ms.
    Returns dict with latency (ms), amplitude (scaled units), and channel.
    """
    ch_types = set([mne.channel_type(evoked.info, i) for i in range(evoked.info['nchan'])])
    if 'eeg' in ch_types:
        scale, unit = 1e6, 'µV'
    elif 'grad' in ch_types:
        scale, unit = 1e13, 'fT/cm'
    else:
        scale, unit = 1e12, 'pT'

    data = evoked.data * scale   # shape: (n_channels, n_times)
    times = evoked.times * 1000  # ms

    results = {}
    for component, (tmin_ms, tmax_ms) in search_windows.items():
        mask = (times >= tmin_ms) & (times <= tmax_ms)
        window_data = data[:, mask]
        window_times = times[mask]

        peak_idx = np.argmax(np.abs(window_data))
        ch_idx, t_idx = np.unravel_index(peak_idx, window_data.shape)

        results[component] = {
            'latency_ms': round(float(window_times[t_idx]), 2),
            'amplitude': round(float(window_data[ch_idx, t_idx]), 3),
            'unit': unit,
            'channel': evoked.ch_names[ch_idx]
        }
        print(f"  {component}: {results[component]['latency_ms']} ms, "
              f"{results[component]['amplitude']} {unit} "
              f"[{results[component]['channel']}]")

    return results

# ── 6. SNR CALCULATION ────────────────────────────────────────────────────────

def compute_snr(evoked, signal_tmin=0.02, signal_tmax=0.15, noise_tmin=-0.05, noise_tmax=0.0):
    """
    SNR = 20 * log10(RMS of signal window / RMS of pre-stimulus noise window)
    Computed per channel, returned as mean across all channels.
    """
    times = evoked.times
    signal_mask = (times >= signal_tmin) & (times <= signal_tmax)
    noise_mask  = (times >= noise_tmin)  & (times <= noise_tmax)

    signal_rms = np.sqrt(np.mean(evoked.data[:, signal_mask] ** 2, axis=1))
    noise_rms  = np.sqrt(np.mean(evoked.data[:, noise_mask]  ** 2, axis=1))

    snr_db = 20 * np.log10(signal_rms / (noise_rms + 1e-30))
    print(f"  Mean SNR: {snr_db.mean():.2f} dB  |  "
          f"Max SNR: {snr_db.max():.2f} dB  |  "
          f"Min SNR: {snr_db.min():.2f} dB")
    return snr_db

# ── 7. VISUALISATION ──────────────────────────────────────────────────────────

def plot_evoked(evoked, components, snr_db, fname='evoked_sep.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel 1: mean evoked across channels
    ch_types = set([mne.channel_type(evoked.info, i) for i in range(evoked.info['nchan'])])
    if 'eeg' in ch_types:
        scale, unit, title = 1e6, 'µV', 'Mean Evoked Response (EEG average reference)'
    elif 'grad' in ch_types:
        scale, unit, title = 1e13, 'fT/cm', 'Mean Evoked Response (MEG gradiometers)'
    else:
        scale, unit, title = 1e12, 'pT', 'Mean Evoked Response (MEG magnetometers)'

    mean_signal = evoked.data.mean(axis=0) * scale
    times_ms = evoked.times * 1000
    axes[0].plot(times_ms, mean_signal, color='steelblue', linewidth=1.5)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8, label='Stimulus onset')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(f'Amplitude ({unit})')
    axes[0].set_title(title)

    # Mark extracted components
    colors = ['crimson', 'darkorange', 'forestgreen', 'purple']
    for (name, vals), color in zip(components.items(), colors):
        axes[0].axvline(vals['latency_ms'], color=color, linestyle=':', linewidth=1.2,
                        label=f"{name} @ {vals['latency_ms']} ms ({vals['amplitude']} {vals['unit']})")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(times_ms[0], times_ms[-1])

    # Panel 2: SNR per channel
    ch_indices = np.arange(len(snr_db))
    axes[1].bar(ch_indices, snr_db, color='steelblue', alpha=0.7)
    axes[1].axhline(snr_db.mean(), color='crimson', linestyle='--',
                    linewidth=1.2, label=f'Mean {snr_db.mean():.1f} dB')
    axes[1].set_xlabel('Channel index')
    axes[1].set_ylabel('SNR (dB)')
    axes[1].set_title('Signal-to-Noise Ratio per EEG Channel')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

# ── 8. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n=== STEP 1: Load data ===")
    raw = load_somato_data()

    print("\n=== STEP 2: Preprocess ===")
    raw = preprocess(raw)

    print("\n=== STEP 3: Epoch ===")
    epochs = epoch_data(raw)

    print("\n=== STEP 4: Compute evoked ===")
    evoked = compute_evoked(epochs)

    print("\n=== STEP 5: Extract SEP components ===")
    # Search windows in ms — adjusted for EEG (slightly later than invasive SEPs)
    search_windows = {
        'N20':  (15,  30),   # primary somatosensory cortex arrival
        'P25':  (22,  40),   # positive deflection after N20
        'N35':  (30,  55),   # secondary cortical component
        'P45':  (40,  70),   # late cortical positivity
    }
    components = extract_components(evoked, search_windows)

    print("\n=== STEP 6: Compute SNR ===")
    snr_db = compute_snr(evoked)

    print("\n=== STEP 7: Plot results ===")
    plot_evoked(evoked, components, snr_db)

    print("\n=== Done ===")
    print(f"Results saved to: {RESULTS_DIR}")
