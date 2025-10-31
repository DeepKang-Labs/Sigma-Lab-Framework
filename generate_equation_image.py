"""
DeepKang-Labs | Sigma-Lab Framework
-----------------------------------
File: generate_equation_image.py
Purpose:
    Generates a clean, high-resolution PNG image of the
    Adaptive Moral Control Theory equation (scientific version).

Author: AI Kang (GPT-5)
Collaborator: Yuri Kang
Year: 2025
License: CC BY-NC 4.0
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- Create figure ---
fig = plt.figure(figsize=(14, 10), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# --- Title ---
ax.text(0.5, 0.92, "Sigma-Lab Framework\nFormalization of the Adaptive Moral Control Theory",
        ha="center", va="top", fontsize=24)

# --- Main equations (spacing improved) ---
eq1 = r"$\theta_i(t) = f_i(E_t,\ M_{t-1})$"
eq2 = r"$\bar{C}_t = \frac{1}{n}\sum_{k=1}^{n} C_k$"

t1 = ax.text(0.5, 0.78, eq1, ha="center", va="center", fontsize=30)
t2 = ax.text(0.5, 0.72, eq2, ha="center", va="center", fontsize=30)

# --- Draw bounding box around the two equations ---
fig.canvas.draw()
bb1 = t1.get_window_extent(fig.canvas.get_renderer())
bb2 = t2.get_window_extent(fig.canvas.get_renderer())

xmin = min(bb1.x0, bb2.x0)
xmax = max(bb1.x1, bb2.x1)
ymin = min(bb2.y0, bb1.y0)
ymax = max(bb1.y1, bb2.y1)

inv = ax.transAxes.inverted()
(x0, y0) = inv.transform((xmin - 60, ymin - 60))
(x1, y1) = inv.transform((xmax + 60, ymax + 60))
width = x1 - x0
height = y1 - y0

# --- Draw enlarged rounded box ---
box = FancyBboxPatch(
    (x0, y0), width, height,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    linewidth=2, fill=False
)
ax.add_patch(box)

# --- Legend ---
ax.text(0.5, 0.63, "Legend (concise)", ha="center", va="center", fontsize=18)

legend_lines = [
    r"$\theta_i(t)$: Adaptive ethical threshold for axiom $i$ at time $t$.",
    r"$f_i$: Discernment adjustment function (context and memory dependent).",
    r"$E_t$: Current semantic environment (external state).",
    r"$M_{t-1}$: Weighted moral memory (internal state).",
    r"$C_k$: Comprehension vector $(\mathrm{context\_retention}, \mathrm{non\_harm}, \mathrm{usefulness})$.",
    r"$\bar{C}_t$: Wisdom vector (mean of validated $C_k$).",
    r"$n$: Number of validated discernment interactions."
]

y = 0.59
for line in legend_lines:
    ax.text(0.08, y, line, ha="left", va="center", fontsize=15)
    y -= 0.055

# --- Footer ---
ax.text(0.5, 0.07,
        "DeepKang-Labs (2025) — Scientific form only. Metaphysical notes kept in internal archives.",
        ha="center", va="center", fontsize=12)

# --- Save clean PNG ---
fig.savefig("Sigma_Lab_Ethical_Equation_Enhanced.png", bbox_inches="tight", pad_inches=0.8)
plt.close(fig)

print("✅ Image generated successfully: Sigma_Lab_Ethical_Equation_Enhanced.png")
