
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# SETTINGS
# ============================================================
SOURCE = DATA_ROOT / 'output/native_bg_rf_replication/figure2_arcgis_2017_2024'

OUT = SOURCE / "publication_maps"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 400

METROS = [
    ("Philadelphia", 32129, 4598),
    ("DMV", 26985, 3968),
]

# ============================================================
# COLORBLIND-FRIENDLY COLORS
# ============================================================

# Ordered rent-change categories:
# blue -> cyan -> pale yellow -> orange -> dark red
RENT_COLORS = [
    "#2166AC",
    "#67A9CF",
    "#F7F7BF",
    "#FDAE61",
    "#B2182B",
]

NO_RENT_DATA = "#B8B8B8"

# Okabe-Ito colorblind-friendly palette

# Station-count change categories
ACCESS_COLORS = {
    0: "#E5E5E5",  # No stations
    1: "#0072B2",  # Gained access
    2: "#009E73",  # Increased count
    3: "#CC79A7",  # Unchanged count
    4: "#E69F00",  # Decreased count
    5: "#7F3C8D",  # Lost access
}

ACCESS_LABELS = {
    0: "No nearby stations (both years)",
    1: "New charging access in 2024",
    2: "Increased station density",
    3: "Unchanged station density",
    4: "Decreased station density",
    5: "Lost charging access",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# LOAD DATA
# ============================================================
data = {}

for metro, epsg, expected_n in METROS:
    path = SOURCE / f"Figure2_{metro}_StatePlane.gpkg"

    gdf = gpd.read_file(path, layer="block_groups")

    if gdf.crs is None:
        raise ValueError(f"{metro}: missing coordinate system")

    gdf = gdf.to_crs(epsg=epsg)

    if len(gdf) != expected_n:
        raise ValueError(
            f"{metro}: expected {expected_n} BGs, "
            f"found {len(gdf)}"
        )

    for col in [
        "rent_diff", "rent_class",
        "public17", "public24", "public_tr"
    ]:
        gdf[col] = pd.to_numeric(
            gdf[col], errors="coerce"
        )

    if gdf[["public17", "public24", "public_tr"]].isna().any().any():
        raise ValueError(f"{metro}: missing charging data")

    a17 = gdf["public17"].gt(0)
    a24 = gdf["public24"].gt(0)

    expected = np.select(
        [
            ~a17 & ~a24,
            ~a17 & a24,
            a17 & a24,
            a17 & ~a24,
        ],
        [0, 1, 2, 3],
        default=-1
    )

    if not np.array_equal(
        gdf["public_tr"].to_numpy(), expected
    ):
        raise ValueError(
            f"{metro}: charging transition mismatch"
        )

    data[metro] = gdf
    print(
        metro,
        "BGs:", len(gdf),
        "CRS:", gdf.crs.to_epsg()
    )

    
for metro, gdf in data.items():

    n17 = gdf["public17"]
    n24 = gdf["public24"]

    conditions = [
        (n17 == 0) & (n24 == 0),
        (n17 == 0) & (n24 > 0),
        (n17 > 0) & (n24 > n17),
        (n17 > 0) & (n24 == n17),
        (n17 > n24) & (n24 > 0),
        (n17 > 0) & (n24 == 0),
    ]

    gdf["station_change_class"] = np.select(
        conditions,
        [0, 1, 2, 3, 4, 5],
        default=-1
    )

    if (gdf["station_change_class"] == -1).any():
        raise ValueError(
            f"{metro}: Some station counts could not be classified."
        )

    print(f"\n{metro}")
    print(
        gdf["station_change_class"]
        .value_counts()
        .sort_index()
        .rename(index=ACCESS_LABELS)
    )

# ============================================================
# RENT LEGEND — COMMON POOLED CLASSIFICATIONS
# ============================================================
rent_values = pd.concat(
    [
        gdf[["rent_diff", "rent_class"]]
        for gdf in data.values()
    ],
    ignore_index=True
).dropna()

ranges = (
    rent_values
    .groupby("rent_class")["rent_diff"]
    .agg(["min", "max"])
    .reindex([1, 2, 3, 4, 5])
)


# ============================================================
# CORRECT RENT LEGEND LABELS
# ============================================================

def fmt_money(value):
    value = round(float(value))
    if value < 0:
        return f"−\\${abs(value):,}"
    return f"\\${value:,}"

RENT_LABELS = [
    f"{fmt_money(row['min'])} to {fmt_money(row['max'])}"
    for _, row in ranges.iterrows()
]


# ============================================================
# DRAW MAPS
# ============================================================
def plot_map(ax, gdf, kind, letter):

    if kind == "rent":
        categories = gdf["rent_class"]
        colors = np.full(
            len(gdf), NO_RENT_DATA, dtype=object
        )

        for cls, color in enumerate(RENT_COLORS, start=1):
            colors[
                categories.eq(cls).fillna(False).to_numpy()
            ] = color

    else:
        categories = gdf["station_change_class"]

        colors = np.full(
            len(gdf), ACCESS_COLORS[0], dtype=object
        )

        for cls, color in ACCESS_COLORS.items():
            colors[
                categories.eq(cls).to_numpy()
            ] = color

    # Plot BGs
    gdf.plot(
        ax=ax,
        color=colors.tolist(),
        linewidth=0,
        edgecolor="none",
        zorder=1
    )

    # Metro boundary
    boundary = gpd.GeoSeries(
        [gdf.geometry.unary_union],
        crs=gdf.crs
    ).boundary

    boundary.plot(
        ax=ax,
        color="#333333",
        linewidth=0.65,
        zorder=3
    )

    # Tight geographic extent
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = xmax - xmin
    height = ymax - ymin

    ax.set_xlim(
        xmin - width * 0.012,
        xmax + width * 0.012
    )
    ax.set_ylim(
        ymin - height * 0.012,
        ymax + height * 0.012
    )

    # Preserve actual geographic shapes
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Panel letter only
    ax.text(
        0.01, 0.98,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111111"
    )

# ============================================================
# LEGEND PATCHES
# ============================================================

rent_handles = [
    Patch(
        facecolor=color,
        edgecolor="#555555",
        linewidth=0.4,
        label=label
    )
    for color, label in zip(RENT_COLORS, RENT_LABELS)
]

rent_handles.append(
    Patch(
        facecolor=NO_RENT_DATA,
        edgecolor="#555555",
        linewidth=0.4,
        label="No rent data"
    )
)


access_handles = [
    Patch(
        facecolor=ACCESS_COLORS[k],
        edgecolor="#555555",
        linewidth=0.4,
        label=ACCESS_LABELS[k]
    )
    for k in range(6)
]


from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

def add_scale_bar(ax, bounds, miles, fontsize=11):
    """
    Add a scale bar inside the bottom-left corner of a map.
    Assumes projected coordinates are in meters.
    """
    xmin, ymin, xmax, ymax = bounds
    w, h = xmax - xmin, ymax - ymin

    distance = miles * 1609.344

    # Position relative to map bounds
    x0 = xmin - 0.055 * w
    x1 = x0 + distance
    y = ymin + 0.015 * h

    # White background behind scale bar
    pad_x = 0.018 * w
    pad_y = 0.025 * h

    background = Rectangle(
        (x0 - pad_x, y - pad_y),
        distance + 2 * pad_x,
        0.095 * h,
        facecolor="white",
        edgecolor="none",
        alpha=0.90,
        zorder=9
    )
    ax.add_patch(background)

    # Scale bar
    ax.plot(
        [x0, x1],
        [y, y],
        color="black",
        linewidth=2,
        zorder=10
    )

    tick_height = 0.012 * h

    for x in [x0, x1]:
        ax.plot(
            [x, x],
            [y - tick_height, y + tick_height],
            color="black",
            linewidth=1.5,
            zorder=10
        )

    # Distance label
    ax.text(
        (x0 + x1) / 2,
        y + 0.020 * h,
        f"{miles} miles",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="medium",
        zorder=11
    )



def add_north_arrow(ax):
    """Add a north arrow in axes-relative coordinates."""
    ax.annotate(
        "N",
        xy=(0.93, 0.91),
        xytext=(0.93, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            linewidth=1.7,
            mutation_scale=18
        ),
        zorder=12
    )


# ============================================================
# COMPACT FOUR-PANEL LAYOUT
# ============================================================

# Nearly square layout to reduce horizontal whitespace.
fig = plt.figure(figsize=(10.2, 10.3))

# Main maps use almost all available figure height.
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1, 1, 0.27],
    left=0.015,
    right=0.985,
    top=0.92,
    bottom=0.025,
    hspace=0.14,
    wspace=0.3
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

plot_map(
    ax_a, data["Philadelphia"],
    "rent", "a"
)
plot_map(
    ax_b, data["Philadelphia"],
    "charging", "b"
)
plot_map(
    ax_c, data["DMV"],
    "rent", "c"
)
plot_map(
    ax_d, data["DMV"],
    "charging", "d"
)


# ============================================================
# REMOVE HORIZONTAL WHITE SPACE BETWEEN MAPS
# ============================================================

# Push left-column maps toward the center
ax_a.set_anchor("E")
ax_c.set_anchor("E")

# Push right-column maps toward the center
ax_b.set_anchor("W")
ax_d.set_anchor("W")


# ============================================================
# CARTOGRAPHIC ELEMENTS
# ============================================================

# Philadelphia — apply to panel (b) only
add_scale_bar(
    ax_b,
    data["Philadelphia"].total_bounds,
    miles=10
)
add_north_arrow(ax_b)

# DMV — apply to panel (d) only
add_scale_bar(
    ax_d,
    data["DMV"].total_bounds,
    miles=25
)
add_north_arrow(ax_d)



# ============================================================
# METROPOLITAN AREA NAMES ABOVE EACH ROW
# ============================================================

# Calculate positions after plotting
fig.canvas.draw()

# Actual subplot positions in figure coordinates
pos_a = ax_a.get_position()
pos_b = ax_b.get_position()
pos_c = ax_c.get_position()
pos_d = ax_d.get_position()

# Center across both columns
row_center = (
    min(pos_a.x0, pos_c.x0) +
    max(pos_b.x1, pos_d.x1)
) / 2

# Position titles above the corresponding row
fig.text(
    row_center,
    max(pos_a.y1, pos_b.y1) + 0.012,
    "Philadelphia–Camden–Wilmington Metropolitan Area",
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold"
)

fig.text(
    row_center,
    max(pos_c.y1, pos_d.y1) + 0.012,
    "Washington–Arlington–Alexandria Metropolitan Area",
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold"
)


# ============================================================
# LARGE LEGENDS
# ============================================================
ax_rent_leg = fig.add_subplot(gs[2, 0])
ax_access_leg = fig.add_subplot(gs[2, 1])

ax_rent_leg.axis("off")
ax_access_leg.axis("off")

ax_rent_leg.legend(
    handles=rent_handles,
    title="Median gross rent change, 2017–2024\nUSD/month",
    loc="center",
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    ncol=2,
    labelspacing=0.75,
    columnspacing=1.0,
    handlelength=1.5,
    handleheight=1.0,
    borderaxespad=0
)

ax_access_leg.legend(
    handles=access_handles,
    title="Change in nearby public charging stations, 2017–2024",
    loc="center",
    frameon=False,
    fontsize=12,
    title_fontsize=12,
    ncol=2,
    labelspacing=0.8,
    columnspacing=1.0,
    handlelength=1.5,
    handleheight=1.0,
    borderaxespad=0
)

# ============================================================
# SAVE
# ============================================================
# Overwrite the original Figure 2 output files
png = OUT / "Figure2_Final_2017_2024.png"
pdf = OUT / "Figure2_Final_2017_2024.pdf"

fig.savefig(
    png,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white"
)

fig.savefig(
    pdf,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white"
)

plt.show()

print("Updated PNG:", png)
print("Updated PDF:", pdf)