"""
2-D Water-Table Map (thin-plate-spline RBF) – Streamlit

Data sources (live from GitHub):
  • Water levels : Monthly_Sea_Level_Data.csv
  • Well coords  : wells.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf

# ---------------------------------------------------------------------------
# RAW GitHub URLs
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
# FIXED MAP EXTENT  (degrees)
# ---------------------------------------------------------------------------

LON_MIN, LON_MAX = 43.4, 44.4
LAT_MIN, LAT_MAX = 35.55, 36.40

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------


@st.cache_data
def load_levels() -> pd.DataFrame:
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])


@st.cache_data
def load_coords() -> pd.DataFrame:
    """
    Load wells.csv and standardise column names while avoiding duplicates.

    ID  : well | no | id | well_name  
    Lat : lat  | latitude | y | northing  
    Lon : lon  | lng | longitude | x | easting
    """
    df = pd.read_csv(COORDS_URL)

    id_syn   = {"well", "no", "id", "well_name"}
    lat_syn  = {"lat", "latitude", "y", "northing"}
    lon_syn  = {"lon", "lng", "longitude", "x", "easting"}

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

    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]  # drop dupes

    missing = {"well", "lat", "lon"} - set(df.columns)
    if missing:
        st.error("`wells.csv` is missing: " + ", ".join(sorted(missing)))
        st.stop()

    return df[["well", "lat", "lon"]]


# ---------------------------------------------------------------------------
# INTERPOLATION
# ---------------------------------------------------------------------------


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Return meshgrid confined to the fixed map box."""
    rbf = Rbf(lon, lat, z, function="thin_plate")

    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("2-D Water-Table Map (RBF)")
    st.caption(
        f"Interpolated surface for wells **W1 … W20** "
        f"within {LAT_MIN}–{LAT_MAX} °N, {LON_MIN}–{LON_MAX} °E."
    )

    # ---- Load data --------------------------------------------------------
    levels = load_levels()
    coords = load_coords()
    well_cols = [c for c in levels.columns if c.upper().startswith("W")]
    if not well_cols:
        st.error("No well columns (W1 … Wn) found in Monthly_Sea_Level_Data.csv.")
        st.stop()

    # ---- Sidebar ----------------------------------------------------------
    st.sidebar.header("Controls")
    dates = levels["Date"].dt.strftime("%Y-m-d")
    date_sel = st.sidebar.selectbox("Month", dates, index=len(dates) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 600, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)

    # ---- Merge chosen month with coords -----------------------------------
    levels_row = levels.loc[dates == date_sel, well_cols].iloc[0]
    df_month = (
        levels_row.rename_axis("well")
        .reset_index(name="level")
        .merge(coords, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )
    if df_month.empty:
        st.warning("No matching wells between level and coordinate files.")
        st.stop()

    # ---- Interpolate surface ---------------------------------------------
    lon = df_month["lon"].to_numpy(float)
    lat = df_month["lat"].to_numpy(float)
    z   = df_month["level"].to_numpy(float)

    lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)

    # ---- Plot -------------------------------------------------------------
    fig, ax = plt.subplots()
    cf = ax.contourf(
        lon_g, lat_g, z_g,
        levels=n_levels,
        cmap="viridis",
        alpha=0.8,
        antialiased=True,
    )
    ax.scatter(
        lon, lat, c=z,
        edgecolors="black", s=90, zorder=3, label="Wells"
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

    # ---- Raw table --------------------------------------------------------
    with st.expander("Raw data for this month"):
        st.dataframe(
            df_month[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
