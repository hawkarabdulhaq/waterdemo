"""
2-D RBF interpolation of monthly groundwater levels (Streamlit).

Data sources (GitHub raw):
  • Water levels  : https://raw.githubusercontent.com/hawkarabdulhaq/waterdemo/main/Monthly_Sea_Level_Data.csv
  • Well coords   : https://raw.githubusercontent.com/hawkarabdulhaq/waterdemo/main/wells.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf

# ---------------------------------------------------------------------------
# URLs – change here if the repo path ever moves
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
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data
def load_levels() -> pd.DataFrame:
    """Download water-level CSV and parse dates."""
    df = pd.read_csv(LEVELS_URL, parse_dates=["Date"])
    return df


@st.cache_data
def load_coords() -> pd.DataFrame:
    """Download well-coordinate CSV and normalise column names."""
    df = pd.read_csv(COORDS_URL)

    # --- Map possible column names to standard ones ------------------------
    rename = {}
    for col in df.columns:
        c = col.lower()
        if c in {"well", "name", "id"}:
            rename[col] = "well"
        elif c in {"lat", "latitude", "y", "northing"}:
            rename[col] = "lat"
        elif c in {"lon", "lng", "longitude", "x", "easting"}:
            rename[col] = "lon"
    df = df.rename(columns=rename)

    # --- Basic sanity check -------------------------------------------------
    for must in ("well", "lat", "lon"):
        if must not in df.columns:
            st.error(f"`wells.csv` needs a “{must}” column (case-insensitive).")
            st.stop()

    return df[["well", "lat", "lon"]]


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Return meshgrid (lon_g, lat_g, z_g) using thin-plate-spline RBF."""
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(lon.min(), lon.max(), res),
        np.linspace(lat.min(), lat.max(), res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Map (RBF)")
    st.caption(
        "Interactively maps groundwater levels for wells **W1 … W20** "
        "using a thin-plate-spline Radial Basis Function surface."
    )

    # ---- Load data --------------------------------------------------------
    levels = load_levels()
    coords = load_coords()
    well_cols = [c for c in levels.columns if c.upper().startswith("W")]

    if not well_cols:
        st.error("No well columns (W1 … Wn) found in the levels CSV.")
        st.stop()

    # ---- Sidebar controls -------------------------------------------------
    st.sidebar.header("Controls")
    date_options = levels["Date"].dt.strftime("%Y-m-d")
    date_sel = st.sidebar.selectbox("Month", date_options, index=len(date_options) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 250, 50)
    n_levels = st.sidebar.slider("Number of contour levels", 5, 30, 15, 1)

    # ---- Extract chosen month & merge with coords -------------------------
    row = levels.loc[date_options == date_sel, well_cols].iloc[0]
    df_month = (
        row.rename_axis("well")
        .reset_index(name="level")
        .merge(coords, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if df_month.empty:
        st.warning("No matching well names between the two CSV files.")
        st.stop()

    # ---- Interpolate surface ---------------------------------------------
    lon_arr = df_month["lon"].to_numpy(float)
    lat_arr = df_month["lat"].to_numpy(float)
    z_arr = df_month["level"].to_numpy(float)

    lon_g, lat_g, z_g = rbf_surface(lon_arr, lat_arr, z_arr, grid_res)

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
        lon_arr,
        lat_arr,
        c=z_arr,
        edgecolors="black",
        s=80,
        label="Wells",
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {date_sel}")
    fig.colorbar(cf, ax=ax, label="Water level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # ---- Data table -------------------------------------------------------
    with st.expander("Show raw data for this month"):
        st.dataframe(
            df_month[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
