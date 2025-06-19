
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Optional: install pykrige locally if available
try:
    from pykrige.ok import OrdinaryKriging
    HAVE_PYKRIGE = True
except ImportError:
    HAVE_PYKRIGE = False

RAW_URL = "https://raw.githubusercontent.com/hawkarabdulhaq/waterdemo/main/Monthly_Sea_Level_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(RAW_URL, parse_dates=['Date'])
    # Ensure numeric columns
    well_cols = [c for c in df.columns if c.startswith('W')]
    df[well_cols] = df[well_cols].apply(pd.to_numeric, errors='coerce')
    return df

def main():
    st.title("Water‑Level Interpolation (RBF / Kriging)")
    st.markdown(
        """This Streamlit app downloads the **Monthly_Sea_Level_Data.csv** from the GitHub repo  
        and lets you interactively interpolate water‑level measurements from wells **W1–W20**  
        for any given month using either Radial Basis Function (RBF) or Ordinary Kriging."""
    )

    df = load_data()
    well_cols = [c for c in df.columns if c.startswith('W')]

    # --- Sidebar controls ---------------------------------------------------
    st.sidebar.header("Controls")
    date_choice = st.sidebar.selectbox(
        "Select month",
        df["Date"].dt.strftime("%Y‑m‑d"),
        index=len(df) - 1  # default = most recent
    )
    method = st.sidebar.radio(
        "Interpolation method",
        ("RBF", "Ordinary Kriging" if HAVE_PYKRIGE else "Ordinary Kriging (pykrige not installed)"),
        index=0,
        help="Install `pykrige` if you’d like to use ordinary kriging."
    )
    resolution = st.sidebar.slider(
        "Interpolation resolution (grid points)",
        min_value=50,
        max_value=400,
        value=200,
        step=25
    )

    # --- Prepare data -------------------------------------------------------
    row = df[df["Date"].dt.strftime("%Y‑m‑d") == date_choice].iloc[0]
    x = np.arange(1, len(well_cols) + 1)               # well index as 1‑D coordinate
    y = row[well_cols].values.astype(float)

    grid_x = np.linspace(x.min(), x.max(), resolution)

    # --- Interpolation ------------------------------------------------------
    if method.startswith("RBF"):
        rbf = Rbf(x, y, function="thin_plate")
        z = rbf(grid_x)
    else:
        if not HAVE_PYKRIGE:
            st.error("pykrige not installed. Please `pip install pykrige` in your environment.")
            st.stop()
        # pykrige expects 2‑D coordinates; pass zeros for Y
        ok = OrdinaryKriging(
            x, np.zeros_like(x), y,
            variogram_model="linear",
            verbose=False, enable_plotting=False
        )
        z, ss = ok.execute("grid", grid_x, np.array([0.0]))
        z = z[:, 0]

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(x, y, "o", label="Observed wells")
    ax.plot(grid_x, z, "-", label=f"{method} interpolation")
    ax.set(
        xlabel="Well index",
        ylabel="Water level (units)",
        title=f"Interpolated profile – {date_choice}"
    )
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # --- Data table ---------------------------------------------------------
    with st.expander("Show raw data for this month"):
        st.dataframe(
            pd.DataFrame({"well": well_cols, "value": y}).set_index("well"),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
