"""MTLB, the processor programme, as one animated figure (ADR-0001).

Top: one 256-bit word, and the instruction decides what it means.
Bottom: the ladder, each rung labelled with what it is:
    U-CPU        MEASURED    the emulator and cost model on real workloads
                             (numbers from README / docs/kernel-cost.md)
    U-TPU        PLANNED     a systolic matrix unit (illustration only)
    U-QPU        PLANNED     a simulated quantum coprocessor, ~30 qubits on 16 GB
    QGPU, QRAM   HYPOTHETICAL  a bucket-brigade QRAM drawn as an illustration; no hardware
                               is claimed

Writes docs/img/mtlb_programme.gif. The image never names the company.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, FG, DIM = "#0a0b10", "#e8e6e1", "#6b6a64"
VIEWS = [
    ("4 × i64 scalars", 4, "#7F77DD"),
    ("32 × u8 packed bytes", 32, "#1D9E75"),
    ("8 × f32 lanes", 8, "#D85A30"),
    ("2 × complex128", 2, "#D4537E"),
    ("4 ring coefficients (post-quantum lattice)", 4, "#EF9F27"),
]
KERNELS = [("llm_matvec", 13.60, 85), ("ising_energy", 10.48, 66), ("mps_chain", 69.57, 82)]


def rung(ax, x, w, title, status, colour, dashed=False, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, 0.05), w, 0.86, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="none", ec=colour, lw=1.6, ls="--" if dashed else "-", alpha=alpha))
    ax.text(x + w / 2, 0.83, title, color=FG, ha="center", fontsize=12, alpha=alpha)
    ax.text(x + w / 2, 0.73, status, color=colour, ha="center", fontsize=9,
            family="monospace", alpha=alpha)


def frame(t: float) -> Image.Image:
    fig = plt.figure(figsize=(14, 6.4), dpi=78)
    fig.patch.set_facecolor(BG)
    top = fig.add_axes([0.04, 0.66, 0.92, 0.24])
    top.set_facecolor(BG)
    top.axis("off")
    v = int(t * len(VIEWS)) % len(VIEWS)
    label, lanes, colour = VIEWS[v]
    for i in range(lanes):
        shade = 0.55 + 0.45 * ((i % 2) == 0)
        top.add_patch(Rectangle((i / lanes, 0.25), 1 / lanes * 0.96, 0.5, fc=colour, alpha=shade * 0.85))
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.text(0.5, 0.88, "one 256-bit register · the instruction decides what it means",
             color=FG, ha="center", fontsize=13)
    top.text(0.5, 0.05, label, color=colour, ha="center", fontsize=12, family="monospace")

    ax = fig.add_axes([0.02, 0.06, 0.96, 0.54])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-0.012, 1.012)  # room for the rounded box edges
    ax.set_ylim(0, 1)
    # U-CPU, measured: flops per instruction of three real kernels
    rung(ax, 0.005, 0.285, "U-CPU", "MEASURED", "#1D9E75")
    for k, (name, fpi, pct) in enumerate(KERNELS):
        y = 0.52 - k * 0.16
        ax.add_patch(Rectangle((0.02, y), 0.26 * fpi / 70.0, 0.09, fc="#1D9E75", alpha=0.85))
        ax.text(0.02, y + 0.105, f"{name}: {fpi:.2f} flops/instr · {pct} % of ceiling",
                color="#b9e8d6", fontsize=8.5)
    # U-TPU, planned: a systolic array with a moving wavefront (illustration)
    rung(ax, 0.315, 0.215, "U-TPU", "PLANNED", "#7F77DD")
    n = 6
    for i in range(n):
        for j in range(n):
            phase = (i + j) / (2 * n) - t * 3
            glow = 0.25 + 0.75 * max(0.0, np.cos(2 * np.pi * phase)) ** 6
            ax.add_patch(Rectangle((0.348 + j * 0.026, 0.14 + i * 0.075), 0.02, 0.055,
                                   fc="#7F77DD", alpha=glow))
    # U-QPU, planned: a state vector's amplitudes turning (illustration)
    rung(ax, 0.555, 0.215, "U-QPU", "PLANNED · ~30 qubits simulated", "#D4537E")
    q = np.arange(16)
    amp = 0.5 + 0.5 * np.cos(2 * np.pi * (q / 16 + t * 2))
    ax.bar(0.583 + q * 0.0105, amp * 0.5, width=0.008, bottom=0.12, color="#D4537E", alpha=0.85)
    # QGPU / QRAM, hypothetical
    # QGPU / QRAM, hypothetical: a bucket-brigade QRAM (illustration). The address is in
    # superposition, |010> + |101>, so two root-to-leaf paths route at once.
    rung(ax, 0.795, 0.200, "QGPU · QRAM", "HYPOTHETICAL", "#9a988f", dashed=True)
    qc, amber = 0.895, "#EF9F27"
    levels = [[qc], [qc - 0.045, qc + 0.045], [qc - 0.0675 + 0.045 * k for k in range(4)]]
    ys = [0.58, 0.46, 0.34]
    leaves = (2, 5)  # |010> and |101>
    lit = {(lv, leaf >> (3 - lv)) for leaf in leaves for lv in range(3)}
    pulse = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(2 * np.pi * t * 4))
    for lv in range(2):
        for i, x in enumerate(levels[lv]):
            for c in (2 * i, 2 * i + 1):
                on = (lv, i) in lit and (lv + 1, c) in lit
                ax.plot([x, levels[lv + 1][c]], [ys[lv], ys[lv + 1]], color=amber if on else DIM,
                        lw=1.8 if on else 0.8, alpha=pulse if on else 0.6)
    leaf_x = [qc - 0.0735 + 0.021 * c for c in range(8)]
    for i, x in enumerate(levels[2]):
        for c in (2 * i, 2 * i + 1):
            on = c in leaves
            ax.plot([x, leaf_x[c]], [ys[2], 0.22], color=amber if on else DIM,
                    lw=1.8 if on else 0.8, alpha=pulse if on else 0.6)
    for lv, xs in enumerate(levels):
        for i, x in enumerate(xs):
            ax.scatter(x, ys[lv], s=46, color=amber if (lv, i) in lit else DIM, zorder=3)
    for c in range(8):
        on = c in leaves
        ax.add_patch(Rectangle((leaf_x[c] - 0.0075, 0.14), 0.015, 0.08,
                               fc=amber if on else "#2a2a28", alpha=pulse if on else 1.0))
    ax.text(qc, 0.075, "|010> + |101>  ·  bucket-brigade", color="#9a988f", ha="center",
            fontsize=8, family="monospace")
    fig.text(0.5, 0.955, "MTLB · metal + lab · one processor programme: CPU → TPU → QPU",
             color=FG, ha="center", fontsize=15)
    fig.text(0.5, 0.012, "U-CPU numbers measured on the emulator (Xeon host), no throughput claim · "
             "U-TPU / U-QPU planned, QGPU / QRAM hypothetical: drawn as illustrations",
             color=DIM, ha="center", fontsize=8.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return Image.open(buf).convert("RGB").quantize(colors=96, method=Image.MEDIANCUT)


def main() -> None:
    n_frames = 100
    frames = [frame(f / n_frames) for f in range(n_frames)]
    out = ROOT / "docs" / "img" / "mtlb_programme.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
