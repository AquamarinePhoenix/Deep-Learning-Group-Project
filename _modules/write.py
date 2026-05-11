from __future__ import annotations

from _modules.config import *

def write_row(ss: str, elapsed_seconds: float | None = None, source: str = "main") -> str:
    ss = f"[{source}] {ss}"
    if elapsed_seconds is not None:
        ss = f"{ss} ({elapsed_seconds:.2f}s)"
    with open(RESULTS_DIR + RESULTS_FILENAME, "a", encoding="utf-8") as f:
        f.write(ss + "\n")
    print(ss + "\n")
    
def clear_file() -> None:
    with open(RESULTS_DIR + RESULTS_FILENAME, "w", encoding="utf-8") as f:
        f.write("")