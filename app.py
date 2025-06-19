"""
Streamlit 2-D RBF interpolation of monthly water-level data.

Data:
- Water levels:   Monthly_Sea_Level_Data.csv  (columns: Date, W1…W20)
- Well coords:    wells.csv  (columns: well, lat, lon)

Author: <your-name>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf

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
    df = pd.read_csv(LEVELS_URL, parse_dates=["Date"])
    return df


@st.cache_data
def load_coords() -> pd.DataFrame:
    df = pd.read_csv(COORDS_URL)

    # --- Rename to canonical columns ---------------------------------------
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c in {"well", "name", "id"}:
            rename_map[col] = "well"
        elif c in {"lat", "latitude", "y"}:
            rename_map[col] = "lat"
        elif c in {"lon", "lng", "longitude", "x"}:
            rename_map[col] = "lon"
    df = df.rename(columns=rename_map)

    needed = {"well", "lat", "lon"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        st.error(f"`wells.csv` is missing column(s): {missing}")
        st.stop()

    return df[["well", "lat", "lon"]]


def rbf_interpolate(lon, lat, z, grid_res=200):
    """Return (grid_lon, grid_lat, grid_z) on regular mesh via thin-plate RBF."""
    rbf = Rbf(lon, lat, z, function="thin_plate")

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, grid_res),
        np.linspace(lat_min, lat_max, grid_res),
    )
    grid_z = rbf(grid_lon, grid_lat)
    return grid_lon, grid_lat, grid_z


# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Interpolation (RBF)")

    st.caption(
        "Plots a thin-plate-spline surface from wells **W1–W20** for any month."
    )

    # ---- Load data ---------------------------------------------------------
    levels_df = load_levels()
    coords_df = load_coords()

    well_cols = [c for c in levels_df.columns if c.startswith("W")]
    if len(well_cols) == 0:
        st.error("No well columns (W1…Wn) found in levels CSV.")
        st.stop()

    # ---- Sidebar controls --------------------------------------------------
    st.sidebar.header("Controls")
    date_strings = levels_df["Date"].dt.strftime("%Y-m-d")
    date_choice = st.sidebar.selectbox("Month", date_strings, index=len(date_strings) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 400, 250, 50)
    contour_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)

    # ---- Extract selected month -------------------------------------------
    month_data = levels_df.loc[date_strings == date_choice, well_cols].iloc[0]
    month_long = (
        month_data.rename_axis("well")
        .reset_index(name="level")
        .merge(coords_df, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if month_long.empty:
        st.warning("No matching wells between level and coordinate files.")
        st.stop()

    # ---- RBF interpolation -------------------------------------------------
    lon = month_long["lon"].to_numpy(float)
    lat = month_long["lat"].to_numpy(float)
    z   = month_long["level"].to_numpy(float)

    grid_lon, grid_lat, grid_z = rbf_interpolate(lon, lat, z, grid_res)

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots()
    cf = ax.contourf(
        grid_lon,
        grid_lat,
        grid_z,
        levels=contour_levels,
        alpha=0.75,
    )
    scat = ax.scatter(
        lon,
        lat,
        c=z,
        edgecolors="black",
        s=80,
        label="Wells",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {date_choice}")
    cbar = fig.colorbar(cf, ax=ax, label="Water level")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    # ---- Raw data table ----------------------------------------------------
    with st.expander("Show raw data for this month"):
        st.dataframe(
            month_long[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
