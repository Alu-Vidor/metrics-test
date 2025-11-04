"""Reporting utilities for experiment outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import pandas as pd


def tables_to_csv(results: Mapping[str, Dict[str, float]], output_path: Path) -> None:
    """Persist scenario tables to CSV for later analysis."""

    frame = pd.DataFrame.from_dict(results, orient="index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)


def summary_html(report_items: Mapping[str, pd.DataFrame], output_path: Path) -> None:
    """Render a minimalist HTML report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        for title, frame in report_items.items():
            f.write(f"<h2>{title}</h2>\n")
            f.write(frame.to_html(float_format="{:.3f}".format))
        f.write("</body></html>\n")
