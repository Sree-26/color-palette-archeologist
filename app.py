"""
╔══════════════════════════════════════════════════════════════╗
║         THE COLOR PALETTE ARCHEOLOGIST  v3                   ║
║         Fix: hue-bucketing guarantees all hue families       ║
║         appear — reds are never buried by green backgrounds  ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import gradio as gr
import colorsys
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# SECTION 1 ▸ COLOR MATH UTILITIES
# ─────────────────────────────────────────────────────────────

def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


def complementary_color(r, g, b):
    r0, g0, b0 = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(r0, g0, b0), min(r0, g0, b0)
    return (
        int(np.clip(((cmax + cmin) - r0) * 255, 0, 255)),
        int(np.clip(((cmax + cmin) - g0) * 255, 0, 255)),
        int(np.clip(((cmax + cmin) - b0) * 255, 0, 255)),
    )


def analogous_colors(r, g, b, angle_deg=30):
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    results = []
    for delta in [-angle_deg / 360.0, angle_deg / 360.0]:
        new_h = (h + delta) % 1.0
        nr, ng, nb = colorsys.hls_to_rgb(new_h, l, s)
        results.append((
            int(np.clip(nr * 255, 0, 255)),
            int(np.clip(ng * 255, 0, 255)),
            int(np.clip(nb * 255, 0, 255)),
        ))
    return results


def relative_luminance(r, g, b):
    vals = []
    for c in [r, g, b]:
        c = c / 255.0
        vals.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def label_color(r, g, b):
    return "#000000" if relative_luminance(r, g, b) > 0.179 else "#FFFFFF"


# ─────────────────────────────────────────────────────────────
# SECTION 2 ▸ IMAGE → DOMINANT COLORS  (NumPy + Pandas)
# ─────────────────────────────────────────────────────────────

def extract_dominant_colors(image_array, n_colors=6, sensitivity=15):
    """
    Hue-bucketing extraction:
    1. Divide the 360° colour wheel into equal sectors
    2. Score each pixel by saturation × brightness
    3. Pick the highest-scoring colour from each sector
    4. Sort sectors by total score → n_colors palette

    This guarantees that ALL hue families present in the image
    (reds, greens, blues…) get a representative swatch —
    a large green background can no longer bury the reds.
    """
    if image_array is None:
        raise ValueError("No image provided.")

    # Ensure RGB
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # ── 2a. Quantise pixels (NumPy) ──────────────────────────
    step = max(1, int(sensitivity))
    quantised = (image_array // step * step).astype(np.uint8)
    pixels = quantised.reshape(-1, 3)

    # ── 2b. Compute hue, saturation, value per pixel ─────────
    r_n = pixels[:, 0] / 255.0
    g_n = pixels[:, 1] / 255.0
    b_n = pixels[:, 2] / 255.0

    cmax = np.maximum(np.maximum(r_n, g_n), b_n)
    cmin = np.minimum(np.minimum(r_n, g_n), b_n)
    delta = cmax - cmin

    # HSV saturation & value
    sat = np.where(cmax > 0, delta / cmax, 0.0)
    val = cmax

    # Hue in degrees [0, 360)
    hue = np.zeros(len(pixels), dtype=np.float32)
    mask = delta > 1e-6
    mr = mask & (cmax == r_n)
    mg = mask & (cmax == g_n)
    mb = mask & (cmax == b_n)
    hue[mr] = (60.0 * ((g_n[mr] - b_n[mr]) / delta[mr])) % 360.0
    hue[mg] = (60.0 * ((b_n[mg] - r_n[mg]) / delta[mg]) + 120.0) % 360.0
    hue[mb] = (60.0 * ((r_n[mb] - g_n[mb]) / delta[mb]) + 240.0) % 360.0

    # ── 2c. Build Pandas DataFrame ────────────────────────────
    df = pd.DataFrame(pixels, columns=["R", "G", "B"])
    df["hue"]    = hue
    df["sat"]    = sat
    df["val"]    = val
    # Weight: sat × val — rewards vivid, bright pixels
    # Grey / near-black pixels get near-zero weight
    df["weight"] = np.where(sat > 0.15, sat * val, 0.005)

    # Bucket hue into 12 sectors of 30° each
    # (grey/unsaturated pixels go to bucket 12 — handled separately)
    df["hue_bucket"] = np.where(sat > 0.15,
                                (hue // 30).astype(int),
                                12)

    # ── 2d. One best colour per hue bucket ───────────────────
    grouped = (
        df.groupby(["hue_bucket", "R", "G", "B"])["weight"]
        .sum()
        .reset_index()
    )

    # Best representative per bucket (highest weight)
    best_per_bucket = (
        grouped
        .sort_values("weight", ascending=False)
        .groupby("hue_bucket", as_index=False)
        .first()
        .sort_values("weight", ascending=False)
        .head(n_colors)
        .reset_index(drop=True)
    )

    # ── 2e. Compute display percentages ──────────────────────
    total = best_per_bucket["weight"].sum()
    best_per_bucket["Dominance_%"] = (
        best_per_bucket["weight"] / total * 100
    ).round(2)

    best_per_bucket["Hex"] = [
        rgb_to_hex(int(row.R), int(row.G), int(row.B))
        for _, row in best_per_bucket.iterrows()
    ]

    return best_per_bucket[["Hex", "R", "G", "B", "weight", "Dominance_%"]].rename(
        columns={"weight": "Count"}
    )


# ─────────────────────────────────────────────────────────────
# SECTION 3 ▸ BUILD THE MATPLOTLIB FIGURE
# ─────────────────────────────────────────────────────────────

def build_figure(df_colors):
    n = len(df_colors)
    fig = plt.figure(figsize=(14, 9), facecolor="#0F0F0F")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04)
    gs = GridSpec(4, 1, figure=fig,
                  height_ratios=[1.4, 1.6, 1.6, 1.6], hspace=0.6)

    fig.text(0.5, 0.95, "COLOR PALETTE ARCHEOLOGIST",
             ha="center", va="top", color="#F5F0E8", fontsize=15,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.905, "Extracted · Analyzed · Theorized",
             ha="center", va="top", color="#888880", fontsize=8,
             fontfamily="monospace")

    def clean_ax(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Panel 0 – Proportion Bar
    ax0 = fig.add_subplot(gs[0])
    clean_ax(ax0)
    ax0.text(-0.01, 1.35, "01 / PROPORTION", transform=ax0.transAxes,
             color="#888880", fontsize=7.5,
             fontfamily="monospace", fontweight="bold")
    x_cursor = 0.0
    for _, row in df_colors.iterrows():
        frac = row["Dominance_%"] / 100.0
        rect = mpatches.FancyBboxPatch(
            (x_cursor, 0.14), frac, 0.72,
            boxstyle="square,pad=0",
            facecolor=row["Hex"], edgecolor="none"
        )
        ax0.add_patch(rect)
        if frac > 0.055:
            lc = label_color(int(row["R"]), int(row["G"]), int(row["B"]))
            ax0.text(x_cursor + frac / 2, 0.50,
                     f"{row['Dominance_%']:.1f}%",
                     ha="center", va="center", color=lc,
                     fontsize=7.5, fontfamily="monospace", fontweight="bold")
        x_cursor += frac

    # Panel 1 – Dominant Swatches
    ax1 = fig.add_subplot(gs[1])
    clean_ax(ax1)
    ax1.text(-0.01, 1.18, "02 / DOMINANT PALETTE",
             transform=ax1.transAxes, color="#888880",
             fontsize=7.5, fontfamily="monospace", fontweight="bold")
    _draw_swatch_row(ax1, df_colors)

    # Panel 2 – Complementary
    ax2 = fig.add_subplot(gs[2])
    clean_ax(ax2)
    ax2.text(-0.01, 1.18, "03 / COMPLEMENTARY THEORY",
             transform=ax2.transAxes, color="#888880",
             fontsize=7.5, fontfamily="monospace", fontweight="bold")
    comp_rows = []
    for _, row in df_colors.iterrows():
        cr, cg, cb = complementary_color(int(row["R"]), int(row["G"]), int(row["B"]))
        comp_rows.append({
            "Hex": rgb_to_hex(cr, cg, cb),
            "R": cr, "G": cg, "B": cb,
            "Dominance_%": row["Dominance_%"]
        })
    _draw_swatch_row(ax2, pd.DataFrame(comp_rows))

    # Panel 3 – Analogous
    ax3 = fig.add_subplot(gs[3])
    clean_ax(ax3)
    ax3.text(-0.01, 1.18, "04 / ANALOGOUS THEORY",
             transform=ax3.transAxes, color="#888880",
             fontsize=7.5, fontfamily="monospace", fontweight="bold")
    analog_rows = []
    for _, row in df_colors.iterrows():
        for ar, ag, ab in analogous_colors(int(row["R"]), int(row["G"]), int(row["B"])):
            analog_rows.append({
                "Hex": rgb_to_hex(ar, ag, ab),
                "R": ar, "G": ag, "B": ab,
                "Dominance_%": row["Dominance_%"] / 2
            })
    _draw_swatch_row(ax3, pd.DataFrame(analog_rows).head(n))

    return fig


def _draw_swatch_row(ax, df):
    n = len(df)
    if n == 0:
        return
    swatch_w = 0.96 / n * 0.88
    gap      = 0.96 / n * 0.12
    for i, (_, row) in enumerate(df.iterrows()):
        x = i * (swatch_w + gap) + gap / 2
        rect = mpatches.FancyBboxPatch(
            (x, 0.22), swatch_w, 0.62,
            boxstyle="round,pad=0.005",
            facecolor=row["Hex"],
            edgecolor="#2A2A2A", linewidth=0.6
        )
        ax.add_patch(rect)
        ax.text(x + swatch_w / 2, 0.10, row["Hex"],
                ha="center", va="top", color="#AAAAAA",
                fontsize=6.2, fontfamily="monospace")


# ─────────────────────────────────────────────────────────────
# SECTION 4 ▸ GRADIO HANDLER
# ─────────────────────────────────────────────────────────────

def analyze_image(image, n_colors, sensitivity):
    if image is None:
        raise gr.Error("Please upload an image first.")

    df_colors = extract_dominant_colors(
        image_array=image,
        n_colors=int(n_colors),
        sensitivity=int(sensitivity)
    )

    fig = build_figure(df_colors)

    df_display = df_colors[["Hex", "R", "G", "B", "Dominance_%"]].copy()
    df_display.columns = ["Hex Code", "R", "G", "B", "Dominance %"]

    return fig, df_display


# ─────────────────────────────────────────────────────────────
# SECTION 5 ▸ GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Color Palette Archeologist") as demo:

    gr.Markdown("""
# ◈ Color Palette Archeologist
*Upload any image — unearth its hidden color DNA.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="Upload Image",
                type="numpy",
                image_mode="RGB"
            )
            n_colors_slider = gr.Slider(
                minimum=3, maximum=12, value=6, step=1,
                label="Number of Colors to Extract",
                info="How many dominant hue families to surface"
            )
            sensitivity_slider = gr.Slider(
                minimum=5, maximum=50, value=15, step=5,
                label="Color Sensitivity (Quantization Step)",
                info="Lower = more precise colour buckets"
            )
            analyze_btn = gr.Button("ANALYZE IMAGE", variant="primary")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Color Analysis")
            df_output = gr.Dataframe(
                label="Dominant Colors",
                headers=["Hex Code", "R", "G", "B", "Dominance %"],
                interactive=False
            )

    analyze_btn.click(
        fn=analyze_image,
        inputs=[img_input, n_colors_slider, sensitivity_slider],
        outputs=[plot_output, df_output]
    )

    img_input.change(
        fn=analyze_image,
        inputs=[img_input, n_colors_slider, sensitivity_slider],
        outputs=[plot_output, df_output]
    )

    gr.Markdown("""
---
**Panels:** `01 PROPORTION` · `02 DOMINANT` · `03 COMPLEMENTARY` · `04 ANALOGOUS`
    """)


# ─────────────────────────────────────────────────────────────
# SECTION 6 ▸ ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )
