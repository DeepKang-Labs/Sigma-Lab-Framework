# engine/sigma_lab_v421.py
# Adapter module for Sigma-Lab v4.2 core to be imported by v5.x tools/GUI
# NON-DISRUPTIVE: it only re-exports from sigma_lab_v4_2.py (root).
# License: CC-BY 4.0 — DeepKang Labs (Yuri & AI Kang)

__version__ = "4.2.1-adapter"

# We import the public API from the existing core file at repo root:
#   sigma_lab_v4_2.py
# If your class/function names differ, add more re-exports below.

try:
    from sigma_lab_v4_2 import (
        SigmaLab,
        SigmaConfig,
        OptionContext,
        Stakeholder,
        # If your core exposes a demo context helper, re-export it too:
        demo_context,
    )
except ImportError as e:
    # Friendly error to help newcomers if the file gets moved/renamed
    raise ImportError(
        "engine/sigma_lab_v421.py could not import from sigma_lab_v4_2.py.\n"
        "Make sure sigma_lab_v4_2.py exists at the repo root and exposes the expected symbols "
        "(SigmaLab, SigmaConfig, OptionContext, Stakeholder, demo_context)."
    ) from e

# Optional: explicit export list for clarity
__all__ = [
    "SigmaLab",
    "SigmaConfig",
    "OptionContext",
    "Stakeholder",
    "demo_context",
    "__version__",
]
