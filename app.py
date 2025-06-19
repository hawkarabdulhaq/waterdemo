"""
2-D water-table surface (RBF) – Streamlit

Data sources pulled live from GitHub:
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
# Loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_levels() -> pd.DataFrame:
    """Monthly levels (Date, W1…Wn)."""
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])


@st.cache_data
def load_coords() -> pd.DataFrame:
    """
    Well coordinates & IDs.  Handles many column spellings and avoids duplicates.

    ID synonyms  : well | no | id | well_name  
    Lat synonyms : lat  | latitude | y | northing  
    Lon synonyms : lon  | lng | longitude | x | easting
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
        # any further duplicates are ignored (left with original name)

    df = df.rename(columns=rename)

    # Drop any duplicate columns that still share a name
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Sanity check
    missing = {"well", "lat", "lon"} - set(df.columns)
    if missing:
        st.error(
            "The well-coordinate file must provide ID and coordinates. "
            f"Missing column(s): {', '.join(sorted(missing))}"
        )
        st.stop()

    return df[["well", "lat", "lon"]]


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Thin-plate-spline RBF -> regular meshgrid."""
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(lon.min(), lon.max(), res),
        np.linspace(lat.min(), lat.max(), res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Map (RBF)")
    st.caption(
        "Interpolates wells **W1–W20** for any month using a thin-plate-spline "
        "Radial Basis Function surface."
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
    date_opts = levels["Date"].dt.strftime("%Y-m-d")
    date_sel = st.sidebar.selectbox("Month", date_opts, index=len(date_opts) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 250, 50)
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
        st.warning("No matching wells between level and coordinate files.")
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
