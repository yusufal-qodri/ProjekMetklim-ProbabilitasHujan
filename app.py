import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
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
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%) !important;
    border: 1px solid #b8d4e8 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,100,160,0.08) !important;
}
[data-testid="metric-container"] label { color: #0f3460 !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0f3460 !important; }
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
.season-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.82rem; margin: 2px;
}
.badge-JFM { background:#dbeafe; color:#1e40af; }
.badge-AMJ { background:#dcfce7; color:#15803d; }
.badge-JAS { background:#fef9c3; color:#92400e; }
.badge-OND { background:#fce7f3; color:#9d174d; }
.custom-divider {
    height: 3px;
    background: linear-gradient(90deg, #0f3460, #1a6fa8, #64b5f6, transparent);
    border-radius: 2px; margin: 20px 0;
}
.info-box {
    background: #f0f7ff; border-left: 4px solid #1a6fa8;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    margin: 8px 0; font-size: 0.9rem; color: #1a1a2e;
}
[data-testid="stTabs"] [role="tab"] { font-weight: 600 !important; font-size: 0.95rem !important; color: #0f3460 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { border-bottom: 3px solid #1a6fa8 !important; color: #1a6fa8 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important; border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important; color: white !important;
}
.block-container { padding-top: 1rem !important; }
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.75rem !important; }
    .hero-banner { padding: 16px !important; border-radius: 12px !important; margin-bottom: 16px !important; }
    .hero-banner h1 { font-size: 1.2rem !important; }
    .hero-banner p { font-size: 0.82rem !important; }
    .section-header { font-size: 0.88rem !important; padding: 8px 12px !important; }
    [data-testid="metric-container"] { padding: 10px !important; border-radius: 10px !important; }
    [data-testid="metric-container"] label { font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="column"] { min-width: 100% !important; width: 100% !important; }
    [data-testid="stTabs"] [role="tab"] { font-size: 0.78rem !important; padding: 6px 8px !important; }
    [data-testid="stSidebar"] { min-width: 260px !important; max-width: 80vw !important; }
    .info-box { font-size: 0.8rem !important; padding: 10px 12px !important; }
    .custom-divider { margin: 12px 0 !important; }
    .js-plotly-plot, .plotly, .plot-container { width: 100% !important; max-width: 100% !important; overflow-x: auto !important; }
    img { max-width: 100% !important; height: auto !important; }
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
    .stSelectbox, .stMultiSelect { width: 100% !important; }
}
@media (max-width: 480px) {
    .hero-banner h1 { font-size: 1rem !important; }
    .hero-banner p { font-size: 0.75rem !important; }
    .section-header { font-size: 0.8rem !important; padding: 7px 10px !important; }
    [data-testid="stTabs"] [role="tab"] { font-size: 0.72rem !important; padding: 5px 6px !important; }
    [data-testid="metric-container"] label { font-size: 0.7rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1rem !important; }
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
# COLORMAP KUSTOM: Merah (rendah) → Biru (tinggi)
# Kontras tinggi, khusus untuk probabilitas hujan
# ─────────────────────────────────────────────
CMAP_RAIN = LinearSegmentedColormap.from_list(
    'rain_contrast',
    [
        '#8B0000',   # 0%   merah tua (probabilitas sangat rendah)
        '#DC143C',   # 15%  merah cerah
        '#FF4500',   # 28%  oranye-merah
        '#FF8C00',   # 40%  oranye
        '#FFD700',   # 50%  kuning emas (tengah)
        '#9ACD32',   # 62%  hijau kekuningan
        '#00BFFF',   # 73%  biru langit
        '#1E90FF',   # 84%  biru cerah
        '#0000CD',   # 92%  biru tua
        '#00008B',   # 100% biru gelap pekat (probabilitas sangat tinggi)
    ],
    N=256
)

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
            for c in df.columns:
                if c.lower() in ('season', 'musim'):
                    df.rename(columns={c: 'Season'}, inplace=True)
            for c in df.columns:
                if c.lower() in ('hujan', 'rain', 'prob', 'probability'):
                    df.rename(columns={c: 'Hujan'}, inplace=True)
            df['Season'] = df['Season'].str.upper().str.strip()
            return df
    # synthetic fallback
    st.warning("'spasial_data.csv' tidak ditemukan — menggunakan data simulasi.", icon="⚠️")
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
def build_rainfall_timeseries(spasial):
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
    ts = build_rainfall_timeseries(spasial)
    ch_rename  = {s: f"CH_{s}"  for s in SEASON_ORDER if s in ts.columns}
    oni_rename = {s: f"ONI_{s}" for s in SEASON_ORDER if s in oni.columns}
    ts_renamed  = ts.rename(columns=ch_rename)
    oni_renamed = oni.rename(columns=oni_rename)
    return ts_renamed.merge(oni_renamed, on='Year', how='inner')


def build_norm_df(spasial, oni):
    """
    Normalisasi Min-Max CH dan ONI per musim — identik dengan notebook analisis.
    Output: Year + CH_JFM..CH_OND + JFM..OND, semua skala 0-1.
    """
    ts = build_rainfall_timeseries(spasial)
    ch_rename = {s: f"CH_{s}" for s in SEASON_ORDER if s in ts.columns}
    ts_renamed = ts.rename(columns=ch_rename)
    gabung = ts_renamed.merge(oni, on='Year', how='inner')

    cols_norm = (
        [f'CH_{s}' for s in SEASON_ORDER if f'CH_{s}' in gabung.columns] +
        [s for s in SEASON_ORDER if s in gabung.columns]
    )
    gabung_norm = gabung.copy()
    if cols_norm:
        scaler = MinMaxScaler()
        gabung_norm[cols_norm] = scaler.fit_transform(gabung[cols_norm])
    return gabung_norm


def _build_land_clip_path(indo, bbox=(117.5, -12.5, 126.5, -6.5)):
    """
    Bangun matplotlib Path gabungan dari semua polygon daratan NTT dalam bbox.
    Digunakan untuk meng-clip contourf agar hanya terlihat di atas daratan.
    """
    import shapely.geometry as sgeom
    from matplotlib.path import Path
    from shapely.ops import unary_union

    try:
        ntt_box  = sgeom.box(*bbox)
        ntt_land = indo[indo.geometry.intersects(ntt_box)].copy()
        if ntt_land.empty:
            return None, None

        land_union = unary_union(ntt_land.geometry)

        # Kumpulkan semua polygon (termasuk MultiPolygon)
        polys = (list(land_union.geoms)
                 if land_union.geom_type == 'MultiPolygon'
                 else [land_union])

        vertices, codes = [], []
        for poly in polys:
            # Exterior ring
            ext_coords = np.array(poly.exterior.coords)
            vertices.append(ext_coords)
            c = [Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY]
            codes.extend(c)
            # Interior rings (holes)
            for interior in poly.interiors:
                int_coords = np.array(interior.coords)
                vertices.append(int_coords)
                c = [Path.MOVETO] + [Path.LINETO] * (len(int_coords) - 2) + [Path.CLOSEPOLY]
                codes.extend(c)

        clip_path = Path(np.vstack(vertices), np.array(codes, dtype=np.uint8))
        return clip_path, land_union
    except Exception:
        return None, None


def _rasterize_land_mask(land_union, lon_arr, lat_arr):
    """
    Buat boolean mask (True = daratan) menggunakan rasterio.features.rasterize
    jika tersedia; fallback ke shapely vectorized contains.
    """
    nrows, ncols = len(lat_arr), len(lon_arr)
    mask = np.zeros((nrows, ncols), dtype=bool)

    try:
        # ── Metode 1: rasterio.features.rasterize (presisi tinggi, cepat) ──
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        lon_min, lon_max = lon_arr[0], lon_arr[-1]
        lat_min, lat_max = lat_arr[0], lat_arr[-1]
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

        burned = rasterize(
            [(land_union, 1)],
            out_shape=(nrows, ncols),
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True,
        )
        # rasterize menggunakan row order (lat dari atas ke bawah), perlu flip jika lat ascending
        if lat_arr[0] < lat_arr[-1]:   # ascending → flip vertical
            burned = burned[::-1, :]
        mask = burned.astype(bool)

    except ImportError:
        # ── Metode 2: shapely STRtree vectorized (lebih cepat dari loop biasa) ──
        try:
            import shapely.vectorized as sv
            Lon2d, Lat2d = np.meshgrid(lon_arr, lat_arr)
            mask = sv.contains(land_union, Lon2d, Lat2d)
        except Exception:
            # ── Fallback terakhir: loop (lambat tapi aman) ──
            import shapely.geometry as sgeom
            for i, la in enumerate(lat_arr):
                for j, lo in enumerate(lon_arr):
                    if land_union.contains(sgeom.Point(lo, la)):
                        mask[i, j] = True

    return mask


def render_land_map(ax, subset_data, indo, is_small=False):
    """
    Render peta probabilitas hujan — HANYA daratan NTT.

    Strategi masking berlapis:
      1. Interpolasi penuh di grid reguler (contourf)
      2. Clip contourf dengan matplotlib Path dari polygon shapefile
         → semua warna di luar batas daratan otomatis tersembunyi
      3. Rasterize shapefile → boolean mask → NaN di luar daratan
         (untuk garis kontur & label agar tidak melintasi lautan)
      4. Overlay batas administrasi di atas segalanya

    Background: putih (lautan tidak berwarna sama sekali).
    """
    from matplotlib.patches import PathPatch
    import matplotlib.transforms as mtransforms

    # ── Setup axes ──
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

    xlim = (118, 125.5)
    ylim = (-11.5, -7.5)
    bbox_ntt = (117.5, -12.5, 126.5, -6.5)

    if subset_data.empty:
        ax.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#555')
        if indo is not None:
            indo.plot(ax=ax, color='#f0f0f0', edgecolor='#444', linewidth=0.5, zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return None

    # ── Pivot ke grid 2D ──
    try:
        pivot = (
            subset_data
            .pivot(index='latitude', columns='longitude', values='Hujan')
            .sort_index()
        )
    except Exception:
        ax.text(0.5, 0.5, 'Error pivot data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='red')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return None

    lon_arr = pivot.columns.values.astype(float)
    lat_arr = pivot.index.values.astype(float)
    Lon, Lat = np.meshgrid(lon_arr, lat_arr)
    Z = pivot.values.astype(float).copy()

    # ── Bangun clip path & land mask dari shapefile ──
    clip_path  = None
    land_union = None
    if indo is not None:
        clip_path, land_union = _build_land_clip_path(indo, bbox=bbox_ntt)

    # ── Rasterize mask untuk NaN (kontur lines) ──
    if land_union is not None:
        land_mask = _rasterize_land_mask(land_union, lon_arr, lat_arr)
        Z_masked  = Z.copy()
        Z_masked[~land_mask] = np.nan
    else:
        Z_masked = Z.copy()

    im = None
    if np.any(~np.isnan(Z_masked)):
        levels = np.linspace(0, 100, 22)

        # ── contourf pada data PENUH (clip path akan menyembunyikan lautan) ──
        im = ax.contourf(
            Lon, Lat, Z,
            levels=levels,
            cmap=CMAP_RAIN,
            extend='both',
            vmin=0, vmax=100,
            alpha=1.0,
            zorder=2,
        )

        # ── Terapkan clip path ke contourf (kompatibel semua versi matplotlib) ──
        if clip_path is not None:
            clip_patch = PathPatch(
                clip_path,
                transform=ax.transData,
                facecolor='none',
                edgecolor='none',
            )
            ax.add_patch(clip_patch)
            # matplotlib < 3.8  pakai .collections (list of PolyCollection)
            # matplotlib >= 3.8 .collections dihapus; iterasi ax.get_children()
            _clipped = False
            try:
                cols = im.collections          # versi lama
                for c in cols:
                    c.set_clip_path(clip_patch)
                _clipped = True
            except AttributeError:
                pass
            if not _clipped:
                # versi baru: set clip lewat semua children axes yang bukan patch/text
                import matplotlib.collections as mcoll
                for child in ax.get_children():
                    if isinstance(child, mcoll.Collection):
                        try:
                            child.set_clip_path(clip_patch)
                        except Exception:
                            pass

        # ── Garis kontur pada data yang sudah di-mask ──
        cs_levels = np.linspace(0, 100, 11)
        try:
            cs = ax.contour(
                Lon, Lat, Z_masked,
                levels=cs_levels,
                colors='white',
                linewidths=0.3,
                alpha=0.4,
                zorder=4,
            )
            if not is_small:
                try:
                    ax.clabel(cs, inline=True, fontsize=6, fmt='%.0f%%')
                except Exception:
                    pass
        except Exception:
            pass

    # ── Overlay shapefile: isi putih di luar bbox ──
    if indo is not None:
        lw = 0.7 if not is_small else 0.45
        # Fill daratan putih → area di luar shapefile tetap putih
        # (ini menimpa sisa artefak contourf yang lolos di tepi)
        import shapely.geometry as sgeom
        ntt_box  = sgeom.box(*bbox_ntt)
        ntt_land = indo[indo.geometry.intersects(ntt_box)].copy()
        if not ntt_land.empty:
            # Gambar "kebalikan" daratan: area bukan daratan diberi warna putih
            try:
                import shapely.ops as sops
                full_box   = sgeom.box(xlim[0] - 1, ylim[0] - 1, xlim[1] + 1, ylim[1] + 1)
                land_union2 = sops.unary_union(ntt_land.geometry)
                ocean_geom  = full_box.difference(land_union2)
                ocean_gdf   = gpd.GeoDataFrame(geometry=[ocean_geom], crs=indo.crs)
                ocean_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=5)
            except Exception:
                pass
        # Border administrasi paling atas
        indo.plot(ax=ax, color='none', edgecolor='#1a1a1a', linewidth=lw, zorder=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return im


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

    st.markdown("**Filter Tahun**")
    sel_year = st.selectbox("Pilih Tahun", all_years, index=len(all_years)//2)

    st.markdown("**Filter Musim**")
    sel_season = st.selectbox("Pilih Musim", all_seasons,
                              format_func=lambda s: f"{s}  ({SEASON_MONTHS[s]})")

    st.markdown("**Perbandingan Musim**")
    multi_seasons = st.multiselect("Pilih Musim (Time Series)",
                                   all_seasons, default=all_seasons,
                                   format_func=lambda s: f"{s} ({SEASON_MONTHS[s]})")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#64b5f6; text-align:center; padding:8px 0;'>
        Data: ERA5 Reanalysis × ONI NOAA<br/>
        Wilayah: NTT (118°–125.5°E, -7.5°–11.5°S)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <h1>Dashboard Probabilitas Hujan di Nusa Tenggara Timur</h1>
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
    st.metric("🔵 Prob. Tertinggi", f"{max_val:.1f}%")
with col4:
    min_val = subset_map['Hujan'].min() if not subset_map.empty else 0
    st.metric("🔴 Prob. Terendah", f"{min_val:.1f}%")

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
# TAB 1 — PETA SPASIAL (Daratan, warna kontras)
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🗺️ Peta Probabilitas Hujan Spasial</div>',
                unsafe_allow_html=True)

    # st.markdown("""
    # <div style='display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap;'>
    #     <span style='font-size:0.84rem; font-weight:600; color:#333;'>Legenda Warna:</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#8B0000; color:white; font-size:0.77rem;'>🔴 0% — Sangat Rendah</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#FF8C00; color:white; font-size:0.77rem;'>🟠 40% — Sedang</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#FFD700; color:#333; font-size:0.77rem;'>🟡 50% — Tengah</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#1E90FF; color:white; font-size:0.77rem;'>🔵 80% — Tinggi</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#00008B; color:white; font-size:0.77rem;'>🟣 100% — Sangat Tinggi</span>
    #     <span style='padding:3px 10px; border-radius:4px; background:#ffffff; color:#333; font-size:0.77rem; border:1px solid #ccc;'>⬜ Lautan / Non-Daratan</span>
    # </div>
    # """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 1])

    with left_col:
        try:
            fig_map, ax_map = plt.subplots(figsize=(11, 7), facecolor='white')
            ax_map.set_facecolor('white')

            im = render_land_map(ax_map, subset_map, indo, is_small=False)

            if im is not None:
                cbar = fig_map.colorbar(im, ax=ax_map, fraction=0.033, pad=0.02)
                cbar.set_label('Probabilitas Hujan (%)', fontsize=10, fontweight='bold')
                cbar.set_ticks([0, 20, 40, 60, 80, 100])
                cbar.ax.set_yticklabels(['0%\n(Rendah)', '20%', '40%', '60%', '80%', '100%\n(Tinggi)'])
                cbar.ax.tick_params(labelsize=8)

            ax_map.set_xlabel('Longitude (°E)', fontsize=9)
            ax_map.set_ylabel('Latitude (°S)', fontsize=9)
            ax_map.set_title(
                f'Probabilitas Hujan — {sel_year}  |  {sel_season} ({SEASON_MONTHS[sel_season]})',
                fontsize=13, fontweight='bold', pad=12, color='#0f3460'
            )
            ax_map.grid(True, linestyle='--', alpha=0.25, linewidth=0.5)
            ax_map.tick_params(labelsize=8)

            # Penanda arah utara
            ax_map.annotate('N', xy=(125.1, -7.65), fontsize=13, fontweight='bold',
                             ha='center', color='#0f3460')
            ax_map.annotate('', xy=(125.1, -7.55), xytext=(125.1, -7.85),
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
            color  = SEASON_COLORS[s]
            st.markdown(
                f"<div style='padding:8px 12px; margin:4px 0; border-radius:8px;"
                f" border-left:4px solid {color}; background:#f8fafb; font-size:0.85rem;'>"
                f"<b>{active}{s}</b><br/><span style='color:#555;'>{SEASON_MONTHS[s]}</span></div>",
                unsafe_allow_html=True
            )
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown("**Batas Wilayah**")
        st.markdown(
            "<div class='info-box'>Lon: 118°–125.5°E<br/>Lat: -7.5°–11.5°S<br/>Wilayah: NTT</div>",
            unsafe_allow_html=True
        )
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

    # ── Perbandingan 4 musim ──
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Perbandingan 4 Musim di Tahun Terpilih</div>',
                unsafe_allow_html=True)

    fig_multi, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white')
    for idx, s in enumerate(SEASON_ORDER):
        ax = axes[idx]
        ax.set_facecolor('white')
        sub = spasial[(spasial['Year'] == sel_year) & (spasial['Season'] == s)]
        im2 = render_land_map(ax, sub, indo, is_small=True)
        ax.set_title(f'{s}\n({SEASON_MONTHS[s]})', fontsize=10,
                     fontweight='bold', color='#0f3460')
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.4)
        ax.set_xlabel('Lon (°E)', fontsize=7)
        ax.set_ylabel('Lat (°S)', fontsize=7)
        if im2 is not None:
            cb = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6)
            cb.set_ticks([0, 50, 100])
            cb.ax.set_yticklabels(['0%', '50%', '100%'])

    fig_multi.suptitle(f'Probabilitas Hujan Daratan NTT per Musim — {sel_year}',
                       fontsize=13, fontweight='bold', y=1.01, color='#0f3460')
    fig_multi.tight_layout()
    st.pyplot(fig_multi, use_container_width=True)
    plt.close(fig_multi)


# ══════════════════════════════════════════════
# TAB 2 — TIME SERIES & ONI & NORMALISASI
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Time Series Probabilitas Hujan dan Indeks ONI</div>',
                unsafe_allow_html=True)

    ts_df = build_rainfall_timeseries(spasial)

    if not multi_seasons:
        st.info("Pilih minimal satu musim di sidebar.")
    else:
        # ── Time series CH ──
        fig_ts = go.Figure()
        for s in multi_seasons:
            if s in ts_df.columns:
                fig_ts.add_trace(go.Scatter(
                    x=ts_df['Year'],
                    y=ts_df[s],
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
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=380, margin=dict(l=60, r=20, t=60, b=50), hovermode='x unified'
        )
        fig_ts.add_hrect(y0=60, y1=100, fillcolor='rgba(239,68,68,0.06)', line_width=0)
        fig_ts.add_hrect(y0=0, y1=30, fillcolor='rgba(59,130,246,0.06)', line_width=0)
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── ONI time series ──
        st.markdown('<div class="section-header">Indeks ONI (Oceanic Niño Index)</div>',
                    unsafe_allow_html=True)

        fig_oni = go.Figure()
        for s in multi_seasons:
            if s in oni.columns:
                fig_oni.add_trace(go.Scatter(
                    x=oni['Year'], y=oni[s],
                    name=f"ONI {s}",
                    line=dict(color=SEASON_COLORS[s], width=2),
                    mode='lines+markers', marker=dict(size=4),
                    hovertemplate=f'<b>ONI {s}</b><br>Tahun: %{{x}}<br>ONI: %{{y:.2f}}<extra></extra>'
                ))
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
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=360, margin=dict(l=60, r=20, t=60, b=50), hovermode='x unified'
        )
        st.plotly_chart(fig_oni, use_container_width=True)

        # ── Dual-axis CH vs ONI musim terpilih ──
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
                           mode='lines+markers', marker=dict(size=5),
                           hovertemplate='ONI: %{y:.2f}<extra></extra>'),
                secondary_y=True
            )
            fig_dual.update_layout(
                title=dict(
                    text=f'Probabilitas Hujan vs ONI — {sel_season} ({SEASON_MONTHS[sel_season]})',
                    font=dict(size=14, color='#0f3460'), x=0.01),
                plot_bgcolor='white', paper_bgcolor='white',
                height=340, margin=dict(l=60, r=60, t=55, b=50),
                legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
                hovermode='x unified'
            )
            fig_dual.update_yaxes(title_text='Probabilitas Hujan (%)',
                                  secondary_y=False, gridcolor='#eee')
            fig_dual.update_yaxes(title_text='ONI Index', secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

        # ════════════════════════════════════════════
        # NORMALISASI MIN-MAX — dari notebook analisis
        # ════════════════════════════════════════════
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Perbandingan CH vs ONI — Normalisasi Min-Max per Musim</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class='info-box'>
        Probabilitas hujan dan indeks ONI dinormalisasi ke skala <b>0–1</b> menggunakan 
        <b>Min-Max Scaling</b> agar dapat dibandingkan dalam satu sumbu meski satuan aslinya berbeda 
        (CH: 0–100%, ONI: −3 s/d +3). Metode ini identik dengan yang digunakan pada notebook 
        <em>Analisis Probabilitas Hujan NTT</em>.
        </div>
        """, unsafe_allow_html=True)

        try:
            norm_df = build_norm_df(spasial, oni)
            seasons_to_show = [s for s in multi_seasons
                               if f'CH_{s}' in norm_df.columns and s in norm_df.columns]

            if seasons_to_show:
                for s in seasons_to_show:
                    color = SEASON_COLORS[s]
                    fig_norm = go.Figure()

                    fig_norm.add_trace(go.Scatter(
                        x=norm_df['Year'],
                        y=norm_df[f'CH_{s}'],
                        name=f'CH {s} (ternormalisasi)',
                        line=dict(color=color, width=2.5),
                        mode='lines+markers',
                        marker=dict(size=5),
                        hovertemplate=f'<b>CH {s}</b><br>Tahun: %{{x}}<br>Nilai norm: %{{y:.3f}}<extra></extra>'
                    ))
                    fig_norm.add_trace(go.Scatter(
                        x=norm_df['Year'],
                        y=norm_df[s],
                        name=f'ONI {s} (ternormalisasi)',
                        line=dict(color=color, width=1.8, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=4, symbol='diamond'),
                        opacity=0.72,
                        hovertemplate=f'<b>ONI {s}</b><br>Tahun: %{{x}}<br>Nilai norm: %{{y:.3f}}<extra></extra>'
                    ))
                    fig_norm.add_hline(
                        y=0.5, line_dash='dot', line_color='gray',
                        line_width=1, opacity=0.45,
                        annotation_text='Tengah (0.5)',
                        annotation_position='right',
                        annotation_font_color='gray'
                    )
                    fig_norm.update_layout(
                        title=dict(
                            text=f'Musim {s} ({SEASON_MONTHS[s]}) — CH vs ONI Ternormalisasi Min-Max',
                            font=dict(size=13, color='#0f3460'), x=0.01
                        ),
                        xaxis=dict(title='Tahun', showgrid=True, gridcolor='#eee', range=[1984, 2016]),
                        yaxis=dict(title='Nilai Ternormalisasi (0–1)', range=[-0.05, 1.1],
                                   showgrid=True, gridcolor='#eee'),
                        plot_bgcolor='white', paper_bgcolor='white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=320, margin=dict(l=65, r=30, t=55, b=50),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_norm, use_container_width=True)
            else:
                st.info("Tidak ada musim yang valid untuk ditampilkan.")

        except Exception as e:
            st.error(f"Gagal membangun grafik normalisasi: {e}")


# ══════════════════════════════════════════════
# TAB 3 — KORELASI & SCATTER PLOT
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Matriks Korelasi — Curah Hujan × Oceanic Niño Index</div>',
                unsafe_allow_html=True)

    corr_df  = build_corr_df(spasial, oni)
    num_cols = [c for c in corr_df.columns if c != 'Year']

    if len(num_cols) < 2:
        st.warning("Kolom tidak cukup untuk membuat matriks korelasi.")
    else:
        corr_mat = corr_df[num_cols].corr()

        # ── Heatmap full ──
        fig_heat_full = px.imshow(
            corr_mat, text_auto='.2f', color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, aspect='auto', title='Heatmap Korelasi Lengkap (CH × ONI)',
        )
        fig_heat_full.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=14, color='#0f3460'),
            coloraxis_colorbar=dict(title='r', tickfont=dict(size=10)),
            height=500, margin=dict(l=20, r=20, t=55, b=20),
            xaxis=dict(tickangle=-35),
        )
        fig_heat_full.update_traces(textfont_size=11)
        st.plotly_chart(fig_heat_full, use_container_width=True)

        # ── Heatmap parsial CH vs ONI ──
        st.markdown('<div class="section-header">Korelasi Parsial — CH vs ONI per Musim</div>',
                    unsafe_allow_html=True)

        ch_cols  = [c for c in num_cols if c.startswith('CH_')]
        oni_cols = [c for c in num_cols if c.startswith('ONI_')]

        if ch_cols and oni_cols:
            partial_corr = corr_df[ch_cols + oni_cols].corr().loc[ch_cols, oni_cols]
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                fig_partial = px.imshow(
                    partial_corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    labels=dict(x='ONI Season', y='Curah Hujan Season', color='r'),
                    title='Korelasi CH vs ONI per Musim',
                )
                fig_partial.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    title_font=dict(size=13, color='#0f3460'),
                    height=380, margin=dict(l=20, r=20, t=50, b=20),
                )
                fig_partial.update_traces(textfont_size=13)
                st.plotly_chart(fig_partial, use_container_width=True)
            with col_h2:
                st.markdown("**Nilai Korelasi**")
                styled = partial_corr.style.background_gradient(
                    cmap='RdBu_r', vmin=-1, vmax=1
                ).format('{:.3f}')
                st.dataframe(styled, use_container_width=True, height=340)

        # ── Scatter: setiap musim satu baris sendiri ──
        st.markdown('<div class="section-header">Scatter Plot CH vs ONI</div>',
                    unsafe_allow_html=True)

        valid_seasons = [s for s in SEASON_ORDER
                         if f'CH_{s}' in corr_df.columns and f'ONI_{s}' in corr_df.columns]

        for s in valid_seasons:
            x_data = corr_df[f'ONI_{s}'].values
            y_data = corr_df[f'CH_{s}'].values
            color  = SEASON_COLORS[s]

            mask       = ~np.isnan(x_data) & ~np.isnan(y_data)
            x_clean    = x_data[mask]
            y_clean    = y_data[mask]
            years_arr  = corr_df['Year'].values[mask]

            fig_sc = go.Figure()

            if len(x_clean) > 2:
                model  = LinearRegression().fit(x_clean.reshape(-1, 1), y_clean)
                y_pred = model.predict(x_clean.reshape(-1, 1))
                r2     = r2_score(y_clean, y_pred)
                slope  = model.coef_[0]
                inter  = model.intercept_

                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = model.predict(x_line.reshape(-1, 1))

                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color=color, width=2.5, dash='dash'),
                    name='Garis Regresi', showlegend=True
                ))

                sign     = '+' if inter >= 0 else '-'
                eq_label = f"y = {slope:.3f}x {sign} {abs(inter):.3f}"
                r2_label = f"R² = {r2:.3f}"

                fig_sc.add_annotation(
                    x=0.03, y=0.97,
                    xref='paper', yref='paper',
                    text=f"<b>{r2_label}</b><br>{eq_label}",
                    showarrow=False,
                    align='left',
                    bgcolor='rgba(255,255,255,0.88)',
                    bordercolor=color,
                    borderwidth=1.5,
                    borderpad=6,
                    font=dict(size=12, color=color)
                )

            fig_sc.add_trace(go.Scatter(
                x=x_clean, y=y_clean,
                mode='markers',
                marker=dict(color=color, size=9, opacity=0.85,
                            line=dict(color='white', width=1.2)),
                text=years_arr,
                name='Data',
                hovertemplate=(
                    f'<b>Musim {s}</b><br>'
                    'Tahun: %{text}<br>'
                    'ONI: %{x:.2f}<br>'
                    'CH: %{y:.1f}%<extra></extra>'
                )
            ))

            fig_sc.update_layout(
                title=dict(
                    text=f'Scatter CH vs ONI — Musim {s} ({SEASON_MONTHS[s]})',
                    font=dict(size=14, color='#0f3460'), x=0.01
                ),
                xaxis=dict(
                    title=f'ONI Index ({s})',
                    showgrid=True, gridcolor='#eee',
                    zeroline=True, zerolinecolor='#bbb', zerolinewidth=1
                ),
                yaxis=dict(
                    title='Probabilitas Hujan (%)',
                    showgrid=True, gridcolor='#eee'
                ),
                plot_bgcolor='white', paper_bgcolor='white',
                height=420, margin=dict(l=65, r=30, t=60, b=60),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Tabel ringkasan regresi ──
        st.markdown('<div class="section-header">Ringkasan Hasil Regresi Linear</div>',
                    unsafe_allow_html=True)

        reg_rows = []
        for s in valid_seasons:
            x_d  = corr_df[f'ONI_{s}'].values
            y_d  = corr_df[f'CH_{s}'].values
            mask = ~np.isnan(x_d) & ~np.isnan(y_d)
            if mask.sum() > 2:
                model  = LinearRegression().fit(x_d[mask].reshape(-1, 1), y_d[mask])
                y_pred = model.predict(x_d[mask].reshape(-1, 1))
                r2     = r2_score(y_d[mask], y_pred)
                r_val  = np.corrcoef(x_d[mask], y_d[mask])[0, 1]
                reg_rows.append({
                    'Musim': f'{s} ({SEASON_MONTHS[s]})',
                    'Slope (a)': round(model.coef_[0], 4),
                    'Intercept (b)': round(model.intercept_, 4),
                    'r (Pearson)': round(r_val, 4),
                    'R² (R-squared)': round(r2, 4),
                    'n': int(mask.sum()),
                })

        if reg_rows:
            df_reg = pd.DataFrame(reg_rows)
            st.dataframe(
                df_reg.style.background_gradient(
                    subset=['r (Pearson)', 'R² (R-squared)'],
                    cmap='RdBu_r', vmin=-1, vmax=1
                ).format({
                    'Slope (a)': '{:.4f}',
                    'Intercept (b)': '{:.4f}',
                    'r (Pearson)': '{:.4f}',
                    'R² (R-squared)': '{:.4f}',
                }),
                use_container_width=True
            )


# ══════════════════════════════════════════════
# TAB 4 — DISTRIBUSI MUSIMAN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Distribusi dan Persentase per Musim</div>',
                unsafe_allow_html=True)

    ts_df2 = build_rainfall_timeseries(spasial)
    season_means = {s: ts_df2[s].mean() for s in SEASON_ORDER if s in ts_df2.columns}

    col_r, col_b = st.columns([1, 1])
    with col_r:
        categories    = list(season_means.keys())
        values        = list(season_means.values())
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill='toself', fillcolor='rgba(26,111,168,0.18)',
            line=dict(color='#1a6fa8', width=2.5),
            marker=dict(size=9, color='#1a6fa8'), name='Rerata CH (%)'
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
            paper_bgcolor='white', height=360,
            margin=dict(l=40, r=40, t=55, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        fig_box = go.Figure()
        for s in SEASON_ORDER:
            if s in ts_df2.columns:
                fig_box.add_trace(go.Box(
                    y=ts_df2[s], name=f"{s}<br>{SEASON_MONTHS[s]}",
                    marker_color=SEASON_COLORS[s], boxmean='sd',
                    hovertemplate=f'<b>{s}</b><br>%{{y:.1f}}%<extra></extra>'
                ))
        fig_box.update_layout(
            title=dict(text='Distribusi Probabilitas Hujan per Musim (Box Plot)',
                       font=dict(size=13, color='#0f3460'), x=0.5, xanchor='center'),
            yaxis=dict(title='Probabilitas Hujan (%)', range=[0, 100], gridcolor='#eee'),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False, height=360, margin=dict(l=55, r=20, t=55, b=50)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Stacked bar ──
    st.markdown('<div class="section-header">Persentase Hujan per Musim — Tiap Tahun</div>',
                unsafe_allow_html=True)
    avail_s  = [s for s in SEASON_ORDER if s in ts_df2.columns]
    row_sums = ts_df2[avail_s].sum(axis=1)
    pct_df   = ts_df2[['Year']].copy()
    for s in avail_s:
        pct_df[s] = ts_df2[s] / row_sums * 100

    fig_stack = go.Figure()
    for s in avail_s:
        fig_stack.add_trace(go.Bar(
            x=pct_df['Year'], y=pct_df[s],
            name=f'{s} ({SEASON_MONTHS[s]})', marker_color=SEASON_COLORS[s],
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
        height=380, margin=dict(l=60, r=20, t=60, b=50), hovermode='x unified'
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Violin ──
    st.markdown('<div class="section-header">Violin Plot — Distribusi Full</div>',
                unsafe_allow_html=True)
    fig_viol = go.Figure()
    for s in SEASON_ORDER:
        if s in spasial['Season'].unique():
            vals = spasial[spasial['Season'] == s]['Hujan'].dropna().values
            fig_viol.add_trace(go.Violin(
                y=vals, name=f'{s}<br>{SEASON_MONTHS[s]}',
                box_visible=True, meanline_visible=True,
                fillcolor=SEASON_COLORS[s], opacity=0.7,
                line_color=SEASON_COLORS[s],
                hovertemplate=f'<b>{s}</b><br>%{{y:.1f}}%<extra></extra>'
            ))
    fig_viol.update_layout(
        title=dict(text='Distribusi Probabilitas Hujan per Grid & Musim (1985–2015)',
                   font=dict(size=14, color='#0f3460'), x=0.01),
        yaxis=dict(title='Probabilitas Hujan (%)', gridcolor='#eee'),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, height=380, margin=dict(l=60, r=20, t=55, b=50)
    )
    st.plotly_chart(fig_viol, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:16px 0; color:#888; font-size:0.8rem;'>
    <b>Dashboard Probabilitas Hujan di NTT</b> &nbsp;·&nbsp;
    Data: ERA5 Reanalysis dan ONI NOAA &nbsp;·&nbsp;
    Analisis: 1985–2015 &nbsp;·&nbsp;
    Wilayah: Nusa Tenggara Timur (118°–125.5°E, -7.5°–11.5°S)
</div>
""", unsafe_allow_html=True)