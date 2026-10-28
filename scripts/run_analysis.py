#!/usr/bin/env python3
"""Generate signals, extract diagnostic features and create figures."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from vibration_analysis import analyse_signal, classify_condition, generate_signal


def main() -> None:
    fs = 12_000
    conditions = ["normal", "imbalance", "bearing_fault"]
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    analyses = {}
    rows = []
    raw = {}
    for index, condition in enumerate(conditions):
        t, signal = generate_signal(condition, fs=fs, seed=42 + index)
        features = analyse_signal(signal, fs)
        prediction = classify_condition(features)
        analyses[condition] = features
        raw[condition] = (t, signal)
        rows.append({
            "Condition": condition,
            "Predicted condition": prediction,
            "RMS (g)": features["rms"],
            "Peak (g)": features["peak"],
            "Crest factor": features["crest_factor"],
            "Kurtosis": features["kurtosis"],
            "Dominant frequency (Hz)": features["dominant_frequency_hz"],
            "Envelope peak (Hz)": features["envelope_peak_hz"],
            "Resonance energy": features["resonance_energy"],
        })
        pd.DataFrame({"Time (s)": t, "Acceleration (g)": signal}).to_csv(
            output / f"{condition}_signal.csv", index=False, float_format="%.6f"
        )
    features_table = pd.DataFrame(rows).set_index("Condition")
    features_table.to_csv(output / "diagnostic_features.csv", float_format="%.4f")

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    display_names = {"normal": "Normal", "imbalance": "Rotor imbalance",
                     "bearing_fault": "Bearing-impact fault"}
    for row, condition in enumerate(conditions):
        t, signal = raw[condition]
        f = analyses[condition]
        axes[row, 0].plot(t[:int(0.15 * fs)], signal[:int(0.15 * fs)], color="#2563eb", lw=0.8)
        axes[row, 1].plot(f["frequency"], f["amplitude"], color="#dc2626", lw=0.9)
        axes[row, 1].set_xlim(0, 4000)
        axes[row, 2].plot(f["envelope_frequency"], f["envelope_amplitude"], color="#059669", lw=0.9)
        axes[row, 2].set_xlim(0, 300)
        axes[row, 0].set_ylabel(f"{display_names[condition]}\nAcceleration (g)")
        for ax in axes[row]:
            ax.grid(alpha=0.22)
    axes[0, 0].set_title("Time signal", weight="bold")
    axes[0, 1].set_title("FFT spectrum", weight="bold")
    axes[0, 2].set_title("Envelope spectrum", weight="bold")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[-1, 2].set_xlabel("Frequency (Hz)")
    fig.suptitle("Simulated machinery condition monitoring", weight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "fault_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    accuracy = (features_table["Predicted condition"] == features_table.index).mean()
    summary = [
        "# Fault-detection results", "",
        "The transparent demonstration classifier correctly identified "
        f"**{accuracy:.0%} of the three simulated conditions**.", "",
        "```text", features_table.round(3).to_string(), "```", "",
        "The imbalance case is dominated by the 30 Hz shaft frequency. The bearing-fault case",
        "contains repeated impacts near 120 Hz that excite a 2.5 kHz resonance and become visible",
        "through envelope analysis. Thresholds were designed for these simulations and must not",
        "be treated as validated industrial alarm limits.",
    ]
    (output / "analysis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(features_table.round(3).to_string())
    print(f"\nResults saved in {output}")


if __name__ == "__main__":
    main()

