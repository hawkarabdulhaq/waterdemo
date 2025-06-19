"""
2-D groundwater surface (thin-plate-spline RBF) – Streamlit

Data pulled live from GitHub:
  • Monthly_Sea_Level_Data.csv  (Date, W1…Wn)
  • wells.csv                   (well IDs + lat/lon)

The map is drawn on a fixed bounding box:
  lat  35.80 – 36.40
  lon  43.60 – 44.30
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf

# ---------------------------------------------------------------------------
# Fixed map bounds
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30

# ---------------------------------------------------------------------------
# Raw URLs
# ---------------------------------------------------------------------------
LEVELS_URL = (
    "https://raw.githubusercontent.com/"
    "hawkarabdulhaq/waterdemo/main/Monthly_Sea_Level_Data.csv"
)
COORDS_URL = (
    "https://raw.githubusercontent.com/"
    "hawkarabdulhaq/waterdemo/main/wells.csv"
)

# ---------------------------------------------------------------------------


@st.cache_data
def load_levels() -> pd.DataFrame:
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])


@st.cache_data
def load_coords() -> pd.DataFrame:
    """Load wells.csv and normalise column names, keeping only the first ID/lat/lon columns found."""
    df = pd.read_csv(COORDS_URL)

    id_syn  = {"well", "no", "id", "well_name"}
    lat_syn = {"lat", "latitude", "y", "northing"}
    lon_syn = {"lon", "lng", "longitude", "x", "easting"}

    rename = {}
    id_done = lat_done = lon_done = False
    for col in df.columns:
        c = col.lower()
        if c in id_syn and not id_done:
            rename[col] = "well"
            id_done = True
        elif c in lat_syn and not lat_done:
            rename[col] = "lat"
            lat_done = True
        elif c in lon_syn and not lon_done:
            rename[col] = "lon"
            lon_done = True

    df = df.rename(columns=rename).loc[:, ~df.columns.duplicated(keep="first")]

    missing = {"well", "lat", "lon"} - set(df.columns)
    if missing:
        st.error(f"`wells.csv` is missing column(s): {', '.join(sorted(missing))}")
        st.stop()

    return df[["well", "lat", "lon"]]


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Return meshgrid (lon_g, lat_g, z_g) using thin-plate RBF over fixed bounds."""
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Map (RBF)")
    st.caption(
        f"Interpolates wells **W1–W20** for any month on a fixed map "
        f"({LAT_MIN}–{LAT_MAX} lat, {LON_MIN}–{LON_MAX} lon)."
    )

    # ---- Load data --------------------------------------------------------
    levels = load_levels()
    coords = load_coords()
    well_cols = [c for c in levels.columns if c.upper().startswith("W")]

    if not well_cols:
        st.error("No W1…Wn columns found in the levels CSV.")
        st.stop()

    # ---- Sidebar ----------------------------------------------------------
    st.sidebar.header("Controls")
    date_opts = levels["Date"].dt.strftime("%Y-m-d")
    date_sel = st.sidebar.selectbox("Month", date_opts, index=len(date_opts) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)

    # ---- Merge chosen month with coords -----------------------------------
    row = levels.loc[date_opts == date_sel, well_cols].iloc[0]
    month_df = (
        row.rename_axis("well")
        .reset_index(name="level")
        .merge(coords, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if month_df.empty:
        st.warning("No matching wells between the two CSV files.")
        st.stop()

    # ---- Interpolate surface ---------------------------------------------
    lon = month_df["lon"].to_numpy(float)
    lat = month_df["lat"].to_numpy(float)
    z   = month_df["level"].to_numpy(float)

    lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)

    # ---- Plot -------------------------------------------------------------
    fig, ax = plt.subplots()
    cf = ax.contourf(
        lon_g,
        lat_g,
        z_g,
        levels=n_levels,
        cmap="viridis",
        alpha=0.75,
    )
    ax.scatter(
        lon,
        lat,
        c=z,
        edgecolors="black",
        s=80,
        label="Wells",
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {date_sel}")
    fig.colorbar(cf, ax=ax, label="Water level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # ---- Data table -------------------------------------------------------
    with st.expander("Raw data for this month"):
        st.dataframe(
            month_df[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
