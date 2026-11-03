import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sep_fixed_channels as sep  # noqa: E402

SF = 5000.0
TRUE_MS = {"N9": 10.0, "N13": 13.5, "N20": 20.0}


def simulated_raw(seed=0, n_stim=400, flip=False, amp_uv=2.0):
    rng = np.random.default_rng(seed)
    names = ["EP", "SC6", "AC", "CP4", "Fz"]
    n = int((n_stim + 2) * 0.25 * SF)
    data = rng.normal(0, 5e-6, (5, n))
    stim = (np.arange(1, n_stim + 1) * 0.25 * SF).astype(int)
    t = np.arange(-int(0.03 * SF), int(0.03 * SF)) / SF
    bump = lambda ms: np.exp(-((t * 1000) ** 2) / (2 * 1.2 ** 2)) * amp_uv * 1e-6
    sign = 1 if flip else -1
    for s in stim:
        data[:, s - 2:s + 3] += 150e-6 * rng.choice([-1, 1])  # stimulus artefact
        for comp, ch in [("N9", 0), ("N13", 1), ("N20", 3)]:
            c = s + int(TRUE_MS[comp] / 1000 * SF)
            data[ch, c + int(t[0] * SF): c + int(t[0] * SF) + t.size] += sign * bump(0)
    raw = mne.io.RawArray(data, mne.create_info(names, SF, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(stim / SF, 0, "stimulus"))
    return raw


def test_recovers_known_latencies_through_artefact():
    df, _ = sep.analyse_raw(simulated_raw())
    for _, row in df.iterrows():
        assert row.latency_ms == pytest.approx(TRUE_MS[row.component], abs=1.0)
        assert row.valid


def test_wrong_polarity_is_not_reported_as_valid():
    df, _ = sep.analyse_raw(simulated_raw(flip=True))
    assert not df.valid.any()


def test_no_response_is_not_reported_as_valid():
    df, _ = sep.analyse_raw(simulated_raw(amp_uv=0.0))
    assert df.valid.sum() == 0


def test_conduction_time_uses_only_valid_peaks():
    df = pd.DataFrame({"subject": ["a", "a", "b", "b"], "component": ["N13", "N20"] * 2,
                       "latency_ms": [13.0, 20.0, 13.0, 21.0], "valid": [True, True, False, True]})
    assert sep.conduction_time(df).tolist() == [7.0]
