"""
run_validation_report.py
---------------------------
Runs every literature dataset in services/literature_data.py through the
solver for every EOS model and writes a dataset x EOS AAD% table to
validation_report.md at the repo root.

Usage:
    uv run python scripts/run_validation_report.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from hydrate_project.services.literature_data import PRESET_DATA
from hydrate_project.services.solver_runner import run_model
from hydrate_project.services.metrics import calculate_aad

EOS_NAMES = ["Peng-Robinson", "Soave-Redlich-Kwong", "Patel-Teja"]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "validation_report.md"


def main():
    logging.basicConfig(level=logging.WARNING)

    rows = []
    for name, preset in PRESET_DATA.items():
        T_range = np.array(sorted(set(preset["T (K)"])))
        row = {"dataset": name}
        for eos in EOS_NAMES:
            results, err = run_model(
                preset["gas_comp"], preset["liq_comp"], T_range, [eos], lambda m: None
            )
            if err is not None:
                row[eos] = f"ERROR: {err}"
                continue
            aad = calculate_aad(results[eos], preset)
            row[eos] = f"{aad:.2f}%" if not np.isnan(aad) else "NaN"
        rows.append(row)

    lines = [
        "# Hydrate Solver Accuracy Validation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "AAD% = average absolute deviation between computed and literature",
        "equilibrium pressure. Lower is better. This is a diagnostic snapshot,",
        "not a pass/fail gate — use it to see where the model currently",
        "diverges from published data.",
        "",
        "| Dataset | " + " | ".join(EOS_NAMES) + " |",
        "|---" + "|---" * len(EOS_NAMES) + "|",
    ]
    for row in rows:
        cells = [row.get(eos, "-") for eos in EOS_NAMES]
        lines.append(f"| {row['dataset']} | " + " | ".join(cells) + " |")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
