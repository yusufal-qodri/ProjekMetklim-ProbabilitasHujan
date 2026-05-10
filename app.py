import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Set Page Config
st.set_page_config(page_title="Dashboard Spasial NTT", layout="wide")

# =====================================
# 1. LOAD DATA & SHAPEFILE
# =====================================

@st.cache_data
def load_data():
    # Gantilah baris ini dengan loading data asli kamu:
    # return pd.read_csv('data_spasial_kamu.csv')
    
    # Ini data simulasi agar dashboard tidak kosong saat pertama run
    lats = np.linspace(-12, -7, 20)
    lons = np.linspace(118, 126, 20)
    data_list = []
    for y in [2021, 2022, 2023]:
        for s in ['DJF', 'MAM', 'JJA', 'SON']:
            for la in lats:
                for lo in lons:
                    data_list.append({
                        'Year': y, 'Season': s, 
                        'latitude': la, 'longitude': lo, 
                        'Hujan': np.random.uniform(0, 100)
                    })
    return pd.DataFrame(data_list)

@st.cache_resource
def load_shp():
    # --- PERBAIKAN DI SINI ---
    # Masukkan path ke file SHP Indonesia milikmu
    path_shp = "C:/Metklim pbl/indonesia_kab.shp" # <--- SESUAIKAN PATH INI
    
    try:
        indo_shp = gpd.read_file(path_shp)
        return indo_shp
    except:
        st.error(f"File SHP tidak ditemukan di {path_shp}. Pastikan path benar!")
        # Jika file belum ada, kita buat GeoDataFrame kosong supaya tidak crash
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')

# Jalankan loading
spasial = load_data()
indo = load_shp()

# =====================================
# 2. SIDEBAR FILTER
# =====================================
st.sidebar.header("Filter Visualisasi")

if not spasial.empty:
    list_tahun = sorted(spasial['Year'].unique())
    list_musim = spasial['Season'].unique().tolist()

    selected_tahun = st.sidebar.selectbox("Pilih Tahun", list_tahun)
    selected_musim = st.sidebar.selectbox("Pilih Musim", list_musim)

    # Filter data
    subset = spasial[
        (spasial['Year'] == selected_tahun) & 
        (spasial['Season'] == selected_musim)
    ]

    # =====================================
    # 3. PROSES & PLOTTING
    # =====================================
    st.title(f"📊 Dashboard Spasial: {selected_tahun} - {selected_musim}")

    if not subset.empty:
        # Pivot
        pivot = subset.pivot(index='latitude', columns='longitude', values='Hujan').sort_index()
        lon = pivot.columns.values
        lat = pivot.index.values
        Lon, Lat = np.meshgrid(lon, lat)
        Z = pivot.values

        fig, ax = plt.subplots(figsize=(10, 6))

        # Contourf
        im = ax.contourf(Lon, Lat, Z, levels=15, cmap='RdYlBu_r', extend='both')

        # Plot SHP Indonesia
        if not indo.empty:
            indo.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)

        # Batas NTT
        ax.set_xlim(118, 126)
        ax.set_ylim(-12, -7)
        ax.set_title(f'Probabilitas Hujan {selected_tahun} ({selected_musim})')
        plt.colorbar(im, ax=ax, label='Probabilitas (%)')

        st.pyplot(fig)
        
        # Tabel
        st.write("### Data Table")
        st.dataframe(subset.head(20))
    else:
        st.warning("Data tidak tersedia untuk filter ini.")