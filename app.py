import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Probabilitas Hujan NTT",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — putih bersih, aksen biru-teal
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background putih ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f3460 0%, #16213e 100%) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] * { color: #e8f4f8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #a8d8ea !important; font-weight: 600; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%) !important;
    border: 1px solid #b8d4e8 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,100,160,0.08) !important;
}
[data-testid="metric-container"] label { color: #0f3460 !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0f3460 !important; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, #0f3460, #1a6fa8);
    color: white !important;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f3460 0%, #1a6fa8 50%, #16213e 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(15,52,96,0.18);
}
.hero-banner h1 { color: white !important; font-size: 2rem; margin: 0 0 6px 0; }
.hero-banner p  { color: #b8d4e8 !important; margin: 0; font-size: 1rem; }

/* ── Season badge ── */
.season-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
    margin: 2px;
}
.badge-JFM { background:#dbeafe; color:#1e40af; }
.badge-AMJ { background:#dcfce7; color:#15803d; }
.badge-JAS { background:#fef9c3; color:#92400e; }
.badge-OND { background:#fce7f3; color:#9d174d; }

/* ── Divider ── */
.custom-divider {
    height: 3px;
    background: linear-gradient(90deg, #0f3460, #1a6fa8, #64b5f6, transparent);
    border-radius: 2px;
    margin: 20px 0;
}

/* ── Info box ── */
.info-box {
    background: #f0f7ff;
    border-left: 4px solid #1a6fa8;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #1a1a2e;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #0f3460 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #1a6fa8 !important;
    color: #1a6fa8 !important;
}

/* ── Selectbox in sidebar ── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* ── Remove default streamlit padding in places ── */
.block-container { padding-top: 1rem !important; }

/* ══════════════════════════════════════════════
   MOBILE RESPONSIVE — max-width: 768px
   ══════════════════════════════════════════════ */
@media (max-width: 768px) {

    /* Block container padding */
    .block-container {
        padding: 0.5rem 0.75rem !important;
    }

    /* Hero banner */
    .hero-banner {
        padding: 16px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
    }
    .hero-banner h1 {
        font-size: 1.2rem !important;
    }
    .hero-banner p {
        font-size: 0.82rem !important;
    }

    /* Section header */
    .section-header {
        font-size: 0.88rem !important;
        padding: 8px 12px !important;
    }

    /* Metric cards — stack 2 per row */
    [data-testid="metric-container"] {
        padding: 10px 10px !important;
        border-radius: 10px !important;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }

    /* Columns — force single column stack on very small screens */
    [data-testid="column"] {
        min-width: 100% !important;
        width: 100% !important;
    }

    /* Tab labels smaller */
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.78rem !important;
        padding: 6px 8px !important;
    }

    /* Sidebar toggle easier tap */
    [data-testid="collapsedControl"] {
        top: 10px !important;
    }

    /* Sidebar width on mobile */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 80vw !important;
    }

    /* Info box font */
    .info-box {
        font-size: 0.8rem !important;
        padding: 10px 12px !important;
    }

    /* Divider margin */
    .custom-divider {
        margin: 12px 0 !important;
    }

    /* Plotly charts — ensure they don't overflow */
    .js-plotly-plot, .plotly, .plot-container {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Matplotlib figures */
    img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Dataframe / table overflow */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }

    /* Selectbox & multiselect full width */
    .stSelectbox, .stMultiSelect {
        width: 100% !important;
    }

    /* Footer text */
    footer, .footer-text {
        font-size: 0.7rem !important;
    }
}

/* ══════════════════════════════════════════════
   EXTRA SMALL — max-width: 480px (headphone/watch-size excluded,
   targets very narrow phone screens)
   ══════════════════════════════════════════════ */
@media (max-width: 480px) {

    .hero-banner h1 {
        font-size: 1rem !important;
    }
    .hero-banner p {
        font-size: 0.75rem !important;
    }

    .section-header {
        font-size: 0.8rem !important;
        padding: 7px 10px !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.72rem !important;
        padding: 5px 6px !important;
    }

    [data-testid="metric-container"] label {
        font-size: 0.7rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEASON CONFIG
# ─────────────────────────────────────────────
SEASON_ORDER = ['JFM', 'AMJ', 'JAS', 'OND']
SEASON_COLORS = {
    'JFM': '#3b82f6',
    'AMJ': '#22c55e',
    'JAS': '#eab308',
    'OND': '#ec4899',
}
SEASON_MONTHS = {
    'JFM': 'Jan–Mar',
    'AMJ': 'Apr–Jun',
    'JAS': 'Jul–Sep',
    'OND': 'Okt–Des',
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_spasial(path="spasial_data.csv"):
    candidates = [
        path,
        "spasial_data.csv",
        os.path.join(os.path.dirname(__file__), "spasial_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            # Normalise season column name
            for c in df.columns:
                if c.lower() in ('season', 'musim'):
                    df.rename(columns={c: 'Season'}, inplace=True)
            # Normalise rain column
            for c in df.columns:
                if c.lower() in ('hujan', 'rain', 'prob', 'probability'):
                    df.rename(columns={c: 'Hujan'}, inplace=True)
            df['Season'] = df['Season'].str.upper().str.strip()
            return df
    # ── synthetic fallback so app never crashes ──
    st.warning("⚠️  'spasial_data.csv' tidak ditemukan — menggunakan data simulasi.", icon="⚠️")
    rng = np.random.default_rng(42)
    lats = np.linspace(-12, -7, 25)
    lons = np.linspace(118, 126, 25)
    rows = []
    for y in range(1985, 2016):
        for s in SEASON_ORDER:
            for la in lats:
                for lo in lons:
                    base = 40 + 30 * np.sin(np.radians((lo - 118) * 20))
                    rows.append({'Year': y, 'Season': s,
                                 'latitude': la, 'longitude': lo,
                                 'Hujan': float(np.clip(base + rng.normal(0, 15), 0, 100))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_oni(path="ONI.csv"):
    candidates = [path, "ONI.csv", os.path.join(os.path.dirname(__file__), "ONI.csv")]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            df = df[(df['Year'] >= 1985) & (df['Year'] <= 2015)]
            cols_keep = [c for c in ['Year', 'JFM', 'AMJ', 'JAS', 'OND'] if c in df.columns]
            return df[cols_keep]
    # synthetic ONI
    rng = np.random.default_rng(7)
    years = np.arange(1985, 2016)
    return pd.DataFrame({
        'Year': years,
        'JFM': rng.normal(0, 0.7, len(years)),
        'AMJ': rng.normal(0, 0.7, len(years)),
        'JAS': rng.normal(0, 0.7, len(years)),
        'OND': rng.normal(0, 0.7, len(years)),
    })


@st.cache_resource(show_spinner=False)
def load_shapefile(path="indonesia_kab.shp"):
    candidates = [path, "indonesia_kab.shp",
                  os.path.join(os.path.dirname(__file__), "indonesia_kab.shp")]
    for p in candidates:
        if os.path.exists(p):
            try:
                return gpd.read_file(p)
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def season_badge(s):
    return f'<span class="season-badge badge-{s}">{s} ({SEASON_MONTHS.get(s,"")})</span>'


def build_rainfall_timeseries(spasial):
    """Mean probabilitas hujan per Year × Season."""
    ts = (
        spasial.groupby(['Year', 'Season'])['Hujan']
        .mean()
        .reset_index()
        .pivot(index='Year', columns='Season', values='Hujan')
        .reindex(columns=SEASON_ORDER)
        .reset_index()
    )
    return ts


def build_corr_df(spasial, oni):
    """Merge spatial mean with ONI for correlation.
    Renames columns BEFORE merge to avoid suffix ambiguity.
    Result columns: Year, CH_JFM, CH_AMJ, CH_JAS, CH_OND,
                              ONI_JFM, ONI_AMJ, ONI_JAS, ONI_OND
    """
    ts = build_rainfall_timeseries(spasial)

    # Rename CH columns first: JFM -> CH_JFM, etc.
    ch_rename = {s: f"CH_{s}" for s in SEASON_ORDER if s in ts.columns}
    ts_renamed = ts.rename(columns=ch_rename)

    # Rename ONI columns: JFM -> ONI_JFM, etc.
    oni_rename = {s: f"ONI_{s}" for s in SEASON_ORDER if s in oni.columns}
    oni_renamed = oni.rename(columns=oni_rename)

    # Merge on Year — no suffix conflicts now
    merged = ts_renamed.merge(oni_renamed, on='Year', how='inner')
    return merged


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 18px 0;'>
        <div style='font-size:2.4rem;'>🌧️</div>
        <div style='font-size:1.05rem; font-weight:700; color:#e8f4f8; letter-spacing:1px;'>
            DASHBOARD<br/>PROBABILITAS HUJAN
        </div>
        <div style='color:#64b5f6; font-size:0.78rem; margin-top:4px;'>
            NTT · 1985–2015
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.15); margin:0 0 16px 0;'/>
    """, unsafe_allow_html=True)

    spasial = load_spasial()
    oni     = load_oni()
    indo    = load_shapefile()

    all_years   = sorted(spasial['Year'].unique())
    all_seasons = [s for s in SEASON_ORDER if s in spasial['Season'].unique()]

    st.markdown("**🗓️ Filter Tahun**")
    sel_year = st.selectbox("Pilih Tahun", all_years, index=len(all_years)//2)

    st.markdown("**🌤️ Filter Musim**")
    sel_season = st.selectbox("Pilih Musim", all_seasons,
                              format_func=lambda s: f"{s}  ({SEASON_MONTHS[s]})")

    st.markdown("**📊 Perbandingan Musim**")
    multi_seasons = st.multiselect("Pilih Musim (Time Series)",
                                   all_seasons, default=all_seasons,
                                   format_func=lambda s: f"{s} ({SEASON_MONTHS[s]})")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#64b5f6; text-align:center; padding:8px 0;'>
        Data: ERA5 Reanalysis × ONI NOAA<br/>
        Wilayah: NTT (118°–126°E, 12°–7°S)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <h1>Dashboard Probabilitas Hujan di NTT</h1>
    <p>Analisis Spasial · Korelasi ONI · Distribusi Musiman &nbsp;|&nbsp;
       Tahun: <strong style='color:#64d8ff;'>{sel_year}</strong> &nbsp;|&nbsp;
       Musim: <strong style='color:#64d8ff;'>{sel_season}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
subset_map = spasial[(spasial['Year'] == sel_year) & (spasial['Season'] == sel_season)]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📍 Titik Data", f"{len(subset_map):,}", help="Jumlah grid points")
with col2:
    mean_val = subset_map['Hujan'].mean() if not subset_map.empty else 0
    st.metric("Rata-rata Probabilitas", f"{mean_val:.1f}%")
with col3:
    max_val = subset_map['Hujan'].max() if not subset_map.empty else 0
    st.metric("🔴 Prob. Tertinggi", f"{max_val:.1f}%")
with col4:
    min_val = subset_map['Hujan'].min() if not subset_map.empty else 0
    st.metric("🔵 Prob. Terendah", f"{min_val:.1f}%")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Peta Spasial",
    "Time Series",
    "Korelasi",
    "Distribusi Musiman",
])

# ══════════════════════════════════════════════
# TAB 1 — PETA SPASIAL
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🗺️ Peta Probabilitas Hujan Spasial</div>',
                unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 1])

    with left_col:
        if subset_map.empty:
            st.warning("Tidak ada data untuk filter ini.")
        else:
            try:
                pivot = (
                    subset_map
                    .pivot(index='latitude', columns='longitude', values='Hujan')
                    .sort_index()
                )
                lon_arr = pivot.columns.values
                lat_arr = pivot.index.values
                Lon, Lat = np.meshgrid(lon_arr, lat_arr)
                Z = pivot.values

                fig_map, ax_map = plt.subplots(figsize=(11, 7), facecolor='white')
                ax_map.set_facecolor('#e8f4f8')

                # contourf
                cmap_custom = plt.cm.RdYlBu_r
                im = ax_map.contourf(Lon, Lat, Z, levels=20,
                                     cmap=cmap_custom, extend='both',
                                     vmin=0, vmax=100, alpha=0.88)

                # contour lines
                cs = ax_map.contour(Lon, Lat, Z, levels=8,
                                    colors='white', linewidths=0.4, alpha=0.5)
                ax_map.clabel(cs, inline=True, fontsize=7, fmt='%.0f%%')

                # shapefile
                if indo is not None:
                    indo.plot(ax=ax_map, color='none', edgecolor='#2c3e50',
                              linewidth=0.7, zorder=5)

                # colorbar
                cbar = fig_map.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02)
                cbar.set_label('Probabilitas Hujan (%)', fontsize=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=9)

                # bounds NTT
                ax_map.set_xlim(118, 125.5)
                ax_map.set_ylim(-11.5, -8)
                ax_map.set_xlabel('Longitude (°E)', fontsize=9)
                ax_map.set_ylabel('Latitude (°S)', fontsize=9)
                ax_map.set_title(
                    f'Probabilitas Hujan — {sel_year}  |  {sel_season} ({SEASON_MONTHS[sel_season]})',
                    fontsize=13, fontweight='bold', pad=12, color='#0f3460'
                )
                ax_map.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
                ax_map.tick_params(labelsize=8)

                # north arrow
                ax_map.annotate('N', xy=(125.6, -7.4), fontsize=14, fontweight='bold',
                                 ha='center', color='#0f3460')
                ax_map.annotate('', xy=(125.6, -7.2), xytext=(125.6, -7.6),
                                arrowprops=dict(arrowstyle='->', color='#0f3460', lw=2))

                fig_map.tight_layout()
                st.pyplot(fig_map, use_container_width=True)
                plt.close(fig_map)

            except Exception as e:
                st.error(f"Gagal render peta: {e}")

    with right_col:
        st.markdown("**ℹ️ Info Musim**")
        for s in SEASON_ORDER:
            active = "→ " if s == sel_season else "   "
            color = SEASON_COLORS[s]
            st.markdown(
                f"<div style='padding:8px 12px; margin:4px 0; border-radius:8px;"
                f" border-left:4px solid {color}; background:#f8fafb; font-size:0.85rem;'>"
                f"<b>{active}{s}</b><br/><span style='color:#555;'>{SEASON_MONTHS[s]}</span></div>",
                unsafe_allow_html=True
            )
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown("**Batas Wilayah**")
        st.markdown(
            "<div class='info-box'>Lon: 118°–125.5°E<br/>Lat: -8°–11.5°S<br/>Wilayah: NTT</div>",
            unsafe_allow_html=True
        )

        # Stats box
        if not subset_map.empty:
            st.markdown("**Statistik**")
            stats = subset_map['Hujan'].describe()
            for k, v in [('Min', stats['min']), ('Mean', stats['mean']),
                          ('Median', stats['50%']), ('Max', stats['max']),
                          ('Std', stats['std'])]:
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:4px 0; border-bottom:1px solid #eee; font-size:0.85rem;'>"
                    f"<span style='color:#666;'>{k}</span>"
                    f"<b style='color:#0f3460;'>{v:.1f}%</b></div>",
                    unsafe_allow_html=True
                )

    # ── Small multiples: all seasons for selected year ──
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Perbandingan 4 Musim — Tahun Terpilih</div>',
                unsafe_allow_html=True)

    fig_multi, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='white')
    for idx, s in enumerate(SEASON_ORDER):
        ax = axes[idx]
        ax.set_facecolor('#e8f4f8')
        sub = spasial[(spasial['Year'] == sel_year) & (spasial['Season'] == s)]
        if not sub.empty:
            try:
                pv = sub.pivot(index='latitude', columns='longitude', values='Hujan').sort_index()
                Lo, La = np.meshgrid(pv.columns.values, pv.index.values)
                im2 = ax.contourf(Lo, La, pv.values, levels=15,
                                  cmap='RdYlBu_r', extend='both', vmin=0, vmax=100)
                if indo is not None:
                    indo.plot(ax=ax, color='none', edgecolor='#2c3e50', linewidth=0.6, zorder=5)
                ax.set_xlim(118, 125.5)
                ax.set_ylim(-11.5, -8)
                ax.set_title(f'{s} ({SEASON_MONTHS[s]})', fontsize=11,
                             fontweight='bold', color='#0f3460')
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle='--', alpha=0.3)
                plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
            except Exception:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    fig_multi.suptitle(f'Probabilitas Hujan per Musim — {sel_year}',
                       fontsize=14, fontweight='bold', y=1.01, color='#0f3460')
    fig_multi.tight_layout()
    st.pyplot(fig_multi, use_container_width=True)
    plt.close(fig_multi)


# ══════════════════════════════════════════════
# TAB 2 — TIME SERIES & ONI
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Time Series Probabilitas Hujan dan Indeks ONI</div>',
                unsafe_allow_html=True)

    ts_df = build_rainfall_timeseries(spasial)

    if not multi_seasons:
        st.info("Pilih minimal satu musim di sidebar.")
    else:
        # ── Plotly time series ──
        fig_ts = go.Figure()
        for s in multi_seasons:
            col_name = s
            if col_name in ts_df.columns:
                fig_ts.add_trace(go.Scatter(
                    x=ts_df['Year'],
                    y=ts_df[col_name],
                    name=f"CH {s} ({SEASON_MONTHS[s]})",
                    line=dict(color=SEASON_COLORS[s], width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                    hovertemplate=f'<b>{s}</b><br>Tahun: %{{x}}<br>Prob: %{{y:.1f}}%<extra></extra>'
                ))

        fig_ts.update_layout(
            title=dict(text='Probabilitas Hujan Rerata per Musim (1985–2015)',
                       font=dict(size=15, color='#0f3460'), x=0.01),
            xaxis=dict(title='Tahun', showgrid=True, gridcolor='#eee'),
            yaxis=dict(title='Probabilitas Hujan (%)', range=[0, 100],
                       showgrid=True, gridcolor='#eee'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=380,
            margin=dict(l=60, r=20, t=60, b=50),
            hovermode='x unified'
        )
        fig_ts.add_hrect(y0=60, y1=100, fillcolor='rgba(239,68,68,0.06)', line_width=0)
        fig_ts.add_hrect(y0=0, y1=30, fillcolor='rgba(59,130,246,0.06)', line_width=0)
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── ONI line chart ──
        st.markdown('<div class="section-header">Indeks ONI (Oceanic Niño Index)</div>',
                    unsafe_allow_html=True)

        fig_oni = go.Figure()
        for s in multi_seasons:
            if s in oni.columns:
                fig_oni.add_trace(go.Scatter(
                    x=oni['Year'], y=oni[s],
                    name=f"ONI {s}",
                    line=dict(color=SEASON_COLORS[s], width=2),
                    mode='lines+markers',
                    marker=dict(size=4),
                    hovertemplate=f'<b>ONI {s}</b><br>Tahun: %{{x}}<br>ONI: %{{y:.2f}}<extra></extra>'
                ))

        # El Niño / La Niña bands
        fig_oni.add_hrect(y0=0.5, y1=3.5, fillcolor='rgba(239,68,68,0.08)',
                          annotation_text="El Niño", annotation_position="top right",
                          annotation_font_color='#dc2626', line_width=0)
        fig_oni.add_hrect(y0=-3.5, y1=-0.5, fillcolor='rgba(59,130,246,0.08)',
                          annotation_text="La Niña", annotation_position="bottom right",
                          annotation_font_color='#2563eb', line_width=0)
        fig_oni.add_hline(y=0, line_dash='dash', line_color='#999', line_width=1)

        fig_oni.update_layout(
            title=dict(text='Indeks ONI per Musim (1985–2015)',
                       font=dict(size=15, color='#0f3460'), x=0.01),
            xaxis=dict(title='Tahun', showgrid=True, gridcolor='#eee'),
            yaxis=dict(title='ONI Index', showgrid=True, gridcolor='#eee'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=360,
            margin=dict(l=60, r=20, t=60, b=50),
            hovermode='x unified'
        )
        st.plotly_chart(fig_oni, use_container_width=True)

        # ── Dual-axis: CH vs ONI for selected season ──
        st.markdown(f'<div class="section-header">CH vs ONI — Musim {sel_season}</div>',
                    unsafe_allow_html=True)

        if sel_season in ts_df.columns and sel_season in oni.columns:
            merged_view = ts_df[['Year', sel_season]].merge(
                oni[['Year', sel_season]], on='Year', suffixes=('_CH', '_ONI')
            )

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Bar(x=merged_view['Year'],
                       y=merged_view[f'{sel_season}_CH'],
                       name='Prob. Hujan (%)',
                       marker_color='rgba(26,111,168,0.55)',
                       marker_line=dict(width=0),
                       hovertemplate='Prob: %{y:.1f}%<extra></extra>'),
                secondary_y=False
            )
            fig_dual.add_trace(
                go.Scatter(x=merged_view['Year'],
                           y=merged_view[f'{sel_season}_ONI'],
                           name='ONI Index',
                           line=dict(color='#ef4444', width=2.5),
                           mode='lines+markers',
                           marker=dict(size=5),
                           hovertemplate='ONI: %{y:.2f}<extra></extra>'),
                secondary_y=True
            )
            fig_dual.update_layout(
                title=dict(
                    text=f'Probabilitas Hujan vs ONI — {sel_season} ({SEASON_MONTHS[sel_season]})',
                    font=dict(size=14, color='#0f3460'), x=0.01),
                plot_bgcolor='white', paper_bgcolor='white',
                height=340,
                margin=dict(l=60, r=60, t=55, b=50),
                legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
                hovermode='x unified'
            )
            fig_dual.update_yaxes(title_text='Probabilitas Hujan (%)',
                                  secondary_y=False, gridcolor='#eee')
            fig_dual.update_yaxes(title_text='ONI Index', secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — KORELASI & HEATMAP
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Matriks Korelasi — Curah Hujan × Oceanic Niño Index</div>',
                unsafe_allow_html=True)

    corr_df = build_corr_df(spasial, oni)
    num_cols = [c for c in corr_df.columns if c != 'Year']

    if len(num_cols) < 2:
        st.warning("Kolom tidak cukup untuk membuat matriks korelasi.")
    else:
        corr_mat = corr_df[num_cols].corr()

        # ── Plotly heatmap — FULL ──
        fig_heat_full = px.imshow(
            corr_mat,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect='auto',
            title='Heatmap Korelasi Lengkap (CH × ONI)',
        )
        fig_heat_full.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#0f3460'),
            coloraxis_colorbar=dict(title='r', tickfont=dict(size=10)),
            height=500,
            margin=dict(l=20, r=20, t=55, b=20),
            xaxis=dict(tickangle=-35),
        )
        fig_heat_full.update_traces(textfont_size=11)
        st.plotly_chart(fig_heat_full, use_container_width=True)

        # ── Partial: CH vs ONI only ──
        st.markdown('<div class="section-header">Korelasi Parsial — CH vs ONI per Musim</div>',
                    unsafe_allow_html=True)

        ch_cols  = [c for c in num_cols if c.startswith('CH_')]
        oni_cols = [c for c in num_cols if c.startswith('ONI_')]

        if ch_cols and oni_cols:
            partial_corr = corr_df[ch_cols + oni_cols].corr().loc[ch_cols, oni_cols]

            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                fig_partial = px.imshow(
                    partial_corr,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    labels=dict(x='ONI Season', y='Curah Hujan Season', color='r'),
                    title='Korelasi CH vs ONI per Musim',
                )
                fig_partial.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    title_font=dict(size=13, color='#0f3460'),
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                fig_partial.update_traces(textfont_size=13)
                st.plotly_chart(fig_partial, use_container_width=True)

            with col_h2:
                st.markdown("**Nilai Korelasi**")
                styled = partial_corr.style.background_gradient(
                    cmap='RdBu_r', vmin=-1, vmax=1
                ).format('{:.3f}')
                st.dataframe(styled, use_container_width=True, height=340)

        # ── Scatter: CH vs ONI tiap musim ──
        st.markdown('<div class="section-header">Scatter Plot CH vs ONI — per Musim</div>',
                    unsafe_allow_html=True)

        valid_seasons = [s for s in SEASON_ORDER
                         if f'CH_{s}' in corr_df.columns and f'ONI_{s}' in corr_df.columns]
        n = len(valid_seasons)
        if n:
            fig_sc = make_subplots(rows=1, cols=n,
                                   subplot_titles=[f'{s} ({SEASON_MONTHS[s]})' for s in valid_seasons])
            for i, s in enumerate(valid_seasons, 1):
                x_data = corr_df[f'ONI_{s}']
                y_data = corr_df[f'CH_{s}']
                color  = SEASON_COLORS[s]

                # regression line
                mask = x_data.notna() & y_data.notna()
                if mask.sum() > 2:
                    coef = np.polyfit(x_data[mask], y_data[mask], 1)
                    x_line = np.linspace(x_data.min(), x_data.max(), 50)
                    y_line = np.polyval(coef, x_line)
                    r_val  = np.corrcoef(x_data[mask], y_data[mask])[0, 1]
                    fig_sc.add_trace(
                        go.Scatter(x=x_line, y=y_line, mode='lines',
                                   line=dict(color=color, width=2, dash='dash'),
                                   name=f'Trend {s}', showlegend=False),
                        row=1, col=i
                    )
                    # label r
                    fig_sc.add_annotation(
                        x=x_data.max(), y=y_data.max(),
                        text=f'r={r_val:.2f}',
                        showarrow=False, font=dict(size=11, color=color),
                        xref=f'x{i}', yref=f'y{i}'
                    )

                fig_sc.add_trace(
                    go.Scatter(
                        x=x_data, y=y_data,
                        mode='markers',
                        marker=dict(color=color, size=7, opacity=0.75,
                                    line=dict(color='white', width=1)),
                        text=corr_df['Year'],
                        hovertemplate='Tahun: %{text}<br>ONI: %{x:.2f}<br>CH: %{y:.1f}%<extra></extra>',
                        name=s, showlegend=False
                    ),
                    row=1, col=i
                )
                fig_sc.update_xaxes(title_text='ONI', row=1, col=i, gridcolor='#eee')
                fig_sc.update_yaxes(title_text='Prob. Hujan (%)' if i == 1 else '',
                                    row=1, col=i, gridcolor='#eee')

            fig_sc.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                height=360,
                margin=dict(l=55, r=20, t=50, b=50),
                title=dict(text='Scatter: Probabilitas Hujan vs ONI per Musim',
                           font=dict(size=14, color='#0f3460'))
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — DISTRIBUSI MUSIMAN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Distribusi dan Persentase per Musim</div>',
                unsafe_allow_html=True)

    ts_df2 = build_rainfall_timeseries(spasial)

    # ── Radar / Polar chart ──
    season_means = {s: ts_df2[s].mean() for s in SEASON_ORDER if s in ts_df2.columns}

    col_r, col_b = st.columns([1, 1])

    with col_r:
        categories = list(season_means.keys())
        values     = list(season_means.values())
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill='toself',
            fillcolor='rgba(26,111,168,0.18)',
            line=dict(color='#1a6fa8', width=2.5),
            marker=dict(size=9, color='#1a6fa8'),
            name='Rerata CH (%)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='#ddd', tickfont=dict(size=9)),
                angularaxis=dict(gridcolor='#ddd')
            ),
            showlegend=False,
            title=dict(text='Probabilitas Hujan Rerata per Musim (Radar)',
                       font=dict(size=13, color='#0f3460'), x=0.5, xanchor='center'),
            paper_bgcolor='white',
            height=360,
            margin=dict(l=40, r=40, t=55, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        # Box plot distributions
        fig_box = go.Figure()
        for s in SEASON_ORDER:
            if s in ts_df2.columns:
                fig_box.add_trace(go.Box(
                    y=ts_df2[s],
                    name=f"{s}<br>{SEASON_MONTHS[s]}",
                    marker_color=SEASON_COLORS[s],
                    boxmean='sd',
                    hovertemplate=f'<b>{s}</b><br>%{{y:.1f}}%<extra></extra>'
                ))
        fig_box.update_layout(
            title=dict(text='Distribusi Probabilitas Hujan per Musim (Box Plot)',
                       font=dict(size=13, color='#0f3460'), x=0.5, xanchor='center'),
            yaxis=dict(title='Probabilitas Hujan (%)', range=[0, 100], gridcolor='#eee'),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False,
            height=360,
            margin=dict(l=55, r=20, t=55, b=50)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Stacked bar: persentase per musim per tahun ──
    st.markdown('<div class="section-header">Persentase Hujan per Musim — Tiap Tahun</div>',
                unsafe_allow_html=True)

    avail_s = [s for s in SEASON_ORDER if s in ts_df2.columns]
    row_sums = ts_df2[avail_s].sum(axis=1)
    pct_df = ts_df2[['Year']].copy()
    for s in avail_s:
        pct_df[s] = ts_df2[s] / row_sums * 100

    fig_stack = go.Figure()
    for s in avail_s:
        fig_stack.add_trace(go.Bar(
            x=pct_df['Year'],
            y=pct_df[s],
            name=f'{s} ({SEASON_MONTHS[s]})',
            marker_color=SEASON_COLORS[s],
            hovertemplate=f'<b>{s}</b><br>Tahun: %{{x}}<br>Porsi: %{{y:.1f}}%<extra></extra>'
        ))

    fig_stack.update_layout(
        barmode='stack',
        title=dict(text='Persentase Kontribusi Hujan per Musim (1985–2015)',
                   font=dict(size=14, color='#0f3460'), x=0.01),
        xaxis=dict(title='Tahun', showgrid=False),
        yaxis=dict(title='Persentase (%)', range=[0, 100], gridcolor='#eee'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=380,
        margin=dict(l=60, r=20, t=60, b=50),
        hovermode='x unified'
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Violin plot ──
    st.markdown('<div class="section-header">Violin Plot — Distribusi Full</div>',
                unsafe_allow_html=True)

    fig_viol = go.Figure()
    for s in SEASON_ORDER:
        if s in spasial['Season'].unique():
            vals = spasial[spasial['Season'] == s]['Hujan'].dropna().values
            fig_viol.add_trace(go.Violin(
                y=vals,
                name=f'{s}<br>{SEASON_MONTHS[s]}',
                box_visible=True,
                meanline_visible=True,
                fillcolor=SEASON_COLORS[s],
                opacity=0.7,
                line_color=SEASON_COLORS[s],
                hovertemplate=f'<b>{s}</b><br>%{{y:.1f}}%<extra></extra>'
            ))
    fig_viol.update_layout(
        title=dict(text='Distribusi Probabilitas Hujan per Grid & Musim (1985–2015)',
                   font=dict(size=14, color='#0f3460'), x=0.01),
        yaxis=dict(title='Probabilitas Hujan (%)', gridcolor='#eee'),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        height=380,
        margin=dict(l=60, r=20, t=55, b=50)
    )
    st.plotly_chart(fig_viol, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:16px 0; color:#888; font-size:0.8rem;'>
    <b>Dashboard Probabilitas Hujan di NTT</b> &nbsp;·&nbsp;
    Data: ERA5 Reanalysis &amp; ONI NOAA &nbsp;·&nbsp;
    Analisis: 1985–2015 &nbsp;·&nbsp;
    Wilayah: Nusa Tenggara Timur (118°–126°E, 7°–12°S)
</div>
""", unsafe_allow_html=True)