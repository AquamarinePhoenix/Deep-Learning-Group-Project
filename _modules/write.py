from __future__ import annotations

import os

from _modules.config import *


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    if elapsed_seconds >= 3600:
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"

    if elapsed_seconds >= 60:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        return f"{minutes}m {seconds:.2f}s"

    return f"{elapsed_seconds:.2f}s"

def write_row(ss: str, elapsed_seconds: float | None = None, source: str = "main") -> str:
    ss = f"[{source}] {ss}"
    if elapsed_seconds is not None:
        ss = f"{ss} ({_format_elapsed_seconds(elapsed_seconds)})"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(ss + "\n")
    print(ss + "\n")
    
def clear_file() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")
