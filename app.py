"""
2-D water-table map (thin-plate-spline RBF) – Streamlit

Data pulled live from your GitHub repo:
  • Monthly_Sea_Level_Data.csv  – water levels (Date, W1…W20)
  • wells.csv                   – well coordinates & metadata
"""

from __future__ import annotations

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
    """Read monthly water levels and parse the Date column."""
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])


@st.cache_data
def load_coords() -> pd.DataFrame:
    """
    Read well-coordinate file and normalise column names.

    Accepts any of these (case-insensitive):
      • ID  : well, well_name, no, id
      • Lat : lat, latitude, y, northing
      • Lon : lon, lng, longitude, x, easting
    """
    df = pd.read_csv(COORDS_URL)

    # --- Map possible names to canonical ones ------------------------------
    rename = {}
    for col in df.columns:
        c = col.lower()
        if c in {"well", "well_name", "no", "id"}:
            rename[col] = "well"
        elif c in {"lat", "latitude", "y", "northing"}:
            rename[col] = "lat"
        elif c in {"lon", "lng", "longitude", "x", "easting"}:
            rename[col] = "lon"
    df = df.rename(columns=rename)

    # --- Verify we now have the required columns ---------------------------
    missing = {"well", "lat", "lon"} - set(df.columns)
    if missing:
        st.error(
            "`wells.csv` must contain well IDs and coordinates. "
            f"Missing column(s): {', '.join(sorted(missing))}"
        )
        st.stop()

    return df[["well", "lat", "lon"]]


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Return (lon_grid, lat_grid, z_grid) via thin-plate-spline RBF."""
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(lon.min(), lon.max(), res),
        np.linspace(lat.min(), lat.max(), res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Surface (RBF)")
    st.caption(
        "Interpolates wells **W1–W20** for any month using a thin-plate-spline "
        "Radial Basis Function surface."
    )

    # ---- Load data --------------------------------------------------------
    levels_df = load_levels()
    coords_df = load_coords()
    well_cols = [c for c in levels_df.columns if c.upper().startswith("W")]

    if not well_cols:
        st.error("No well columns (W1 … Wn) found in the levels CSV.")
        st.stop()

    # ---- Sidebar ----------------------------------------------------------
    st.sidebar.header("Controls")
    date_strings = levels_df["Date"].dt.strftime("%Y-m-d")
    date_sel = st.sidebar.selectbox("Month", date_strings, index=len(date_strings) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 250, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)

    # ---- Select month & merge --------------------------------------------
    levels_row = levels_df.loc[date_strings == date_sel, well_cols].iloc[0]
    month_df = (
        levels_row.rename_axis("well")
        .reset_index(name="level")
        .merge(coords_df, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if month_df.empty:
        st.warning("No matching wells between level and coordinate files.")
        st.stop()

    # ---- Interpolate ------------------------------------------------------
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
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {date_sel}")
    fig.colorbar(cf, ax=ax, label="Water level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # ---- Table ------------------------------------------------------------
    with st.expander("Raw data for this month"):
        st.dataframe(
            month_df[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
