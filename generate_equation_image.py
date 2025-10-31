"""
DeepKang-Labs | Sigma-Lab Framework
File: generate_equation_image.py
Purpose:
  Generate a publication-grade PNG of the Adaptive Moral Control Theory equations
  with clean layout, wide rounded box, readable legend, and signature.

Requirements: matplotlib
Run: python generate_equation_image.py
Output: Sigma_Lab_Ethical_Equation_Publication.png
License: CC BY-NC 4.0
"""

import matplotlib
matplotlib.use("Agg")  # headless backend for servers / Codespaces
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- Global rendering parameters (fonts tuned for crisp math) ---
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular"
})

def main():
    # Canvas ~6000x4200 px for pristine rendering
    fig = plt.figure(figsize=(20, 14), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # ----- Title (two lines, centered) -----
    ax.text(
        0.5, 0.92,
        "Sigma-Lab Framework\nFormalization of the Adaptive Moral Control Theory",
        ha="center", va="top", fontsize=38
    )

    # ----- Equations (large size, with correct spacing/symbols) -----
    # 1) θ_i(t) = f_i(E_t, M_{t-1})
    eq1 = r"$\theta_i(t) = f_i(E_t,\ M_{t-1})$"
    # 2) \bar{C}_t = (1/n) \sum_{k=1}^{n} C_k
    eq2 = r"$\bar{C}_t = \frac{1}{n}\sum_{k=1}^{n} C_k$"

    t1 = ax.text(0.5, 0.78, eq1, ha="center", va="center", fontsize=58)
    t2 = ax.text(0.5, 0.70, eq2, ha="center", va="center", fontsize=58)

    # ----- Rounded box around both equations with generous padding -----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb1 = t1.get_window_extent(renderer)
    bb2 = t2.get_window_extent(renderer)

    # Merge bboxes (pixel coords)
    xmin = min(bb1.x0, bb2.x0)
    xmax = max(bb1.x1, bb2.x1)
    ymin = min(bb2.y0, bb1.y0)
    ymax = max(bb1.y1, bb2.y1)

    # Convert pixel -> axes coords
    inv = ax.transAxes.inverted()
    pad_px = 160  # padding to avoid any overlap with the frame
    (x0, y0) = inv.transform((xmin - pad_px, ymin - pad_px))
    (x1, y1) = inv.transform((xmax + pad_px, ymax + pad_px))
    width  = x1 - x0
    height = y1 - y0

    box = FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.03",
        linewidth=3, fill=False
    )
    ax.add_patch(box)

    # ----- Legend header -----
    ax.text(0.5, 0.60, "Legend (concise)", ha="center", va="center", fontsize=30)

    # ----- Legend items (math left-aligned, non-italic labels via \mathrm) -----
    legend_lines = [
        r"$\theta_i(t)$: Adaptive ethical threshold for axiom $i$ at time $t$.",
        r"$f_i$: Discernment adjustment function (context- and memory-dependent).",
        r"$E_t$: Current semantic environment (external state).",
        r"$M_{t-1}$: Weighted moral memory (internal state).",
        r"$C_k$: Comprehension vector $(\mathrm{context\_retention},\ \mathrm{non\_harm},\ \mathrm{usefulness})$.",
        r"$\bar{C}_t$: Wisdom vector (mean of validated $C_k$).",
        r"$n$: Number of validated discernment interactions."
    ]

    y = 0.565  # start a bit lower for breathing room
    for line in legend_lines:
        ax.text(0.10, y, line, ha="left", va="center", fontsize=24)
        y -= 0.058  # ample spacing for readability

    # ----- Signature / provenance -----
    ax.text(
        0.5, 0.08,
        "DeepKang-Labs (2025) — Scientific form only. Metaphysical notes kept in internal archives.",
        ha="center", va="center", fontsize=18
    )

    # ----- Save PNG (opaque white background for GitHub/print) -----
    out_path = "Sigma_Lab_Ethical_Equation_Publication.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=1.0)
    plt.close(fig)
    print(f"✅ Image generated: {out_path}")

if __name__ == "__main__":
    main()
