"""
build_subregion_masks.py
========================
Complete pipeline to:
  1. Download GADM sub-national boundary data (Level 1 = states/provinces)
  2. Convert it to a countrymasks-style NetCDF (one binary variable per sub-region)
  3. Plug that mask into the RIME-X regional averages pipeline

Requirements:
    pip install geopandas regionmask xarray netCDF4 numpy requests tqdm pyogrio

Usage:
    python build_subregion_masks.py

Output:
    subregion_masks.nc   -- drop-in replacement for countrymasks.nc
"""

import os
import sys
import zipfile
import requests
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
from tqdm import tqdm
import gc
import time


# =============================================================================
# CONFIGURATION — edit these to match your setup
# =============================================================================

# Output directory for the mask file
OUTPUT_DIR = "./masks"

# Output mask filename (drop-in replacement for countrymasks.nc)
MASK_OUTPUT_PATH = f"{OUTPUT_DIR}/subregion_masks.nc"

# GADM admin level to use:
#   1 = States / Provinces (recommended for RIME-X at 2.5deg resolution)
#   2 = Districts / Counties (finer, but many will be smaller than 1 grid cell)
GADM_LEVEL = 1

# Filter to specific countries (ISO 3166-1 alpha-3 codes), or set to None for ALL countries
# Example: ["PAK", "IND", "CHN", "USA", "DEU"]
# None = all countries in GADM (will produce a very large mask file)
COUNTRIES_FILTER = None  # e.g. ["PAK", "IND"] for just Pakistan and India

# Target grid — 2.5 degree global grid matching CMIP6-ng g025 standard
# (same grid the RIME-X pipeline expects)
LON_START  = -178.75
LON_INC    = 2.5
LON_SIZE   = 144
LAT_START  = -88.75
LAT_INC    = 2.5
LAT_SIZE   = 72

# =============================================================================
# STEP 1 — DOWNLOAD GADM DATA
# =============================================================================

def download_gadm(output_dir: str, level: int = 1) -> str:
    """
    Downloads the GADM GeoPackage for the specified admin level.
    Returns path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # GADM 4.1 global GeoPackage (all levels in one file)
    gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip"
    zip_path  = os.path.join(output_dir, "gadm_410-levels.zip")
    gpkg_path = os.path.join(output_dir, "gadm_410-levels.gpkg")

    if os.path.exists(gpkg_path):
        print(f"[INFO] GADM GeoPackage already exists at: {gpkg_path}")
        return gpkg_path

    print(f"[INFO] Downloading GADM 4.1 from:\n       {gadm_url}")
    print("[INFO] This is a ~900MB file — may take a few minutes...")

    response = requests.get(gadm_url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading GADM",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[INFO] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)
    print(f"[INFO] GADM GeoPackage ready at: {gpkg_path}")
    return gpkg_path


# =============================================================================
# STEP 2 — LOAD GADM AND FILTER
# =============================================================================

def load_gadm(gpkg_path: str, level: int, countries_filter: list) -> gpd.GeoDataFrame:
    """
    Loads the specified GADM admin level from the GeoPackage.
    Optionally filters to a list of country ISO codes.
    """
    layer_name = f"ADM_{level}"
    print(f"[INFO] Loading GADM layer: {layer_name} ...")

    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    print(f"[INFO] Total sub-regions loaded: {len(gdf)}")
    print(f"[INFO] Columns available: {list(gdf.columns)}")

    if countries_filter:
        # GADM uses GID_0 for ISO 3-letter country codes
        gdf = gdf[gdf["GID_0"].isin(countries_filter)].copy()
        print(f"[INFO] After country filter {countries_filter}: {len(gdf)} sub-regions")

    if len(gdf) == 0:
        raise ValueError(
            "No sub-regions found after filtering. "
            "Check your COUNTRIES_FILTER values — use ISO 3166-1 alpha-3 codes (e.g. 'PAK', 'IND')."
        )

    # Clean up names: remove special characters that cause issues in NetCDF variable names
    name_col = f"NAME_{level}"
    country_col = "GID_0"
    gdf["mask_name"] = (
        gdf[country_col].str.upper()
        + "_"
        + gdf[name_col]
        .str.replace(r"[^A-Za-z0-9]", "_", regex=True)
        .str.upper()
    )

    # Ensure uniqueness (some regions share names across countries)
    gdf["mask_name"] = gdf["mask_name"] + "_" + gdf.index.astype(str)

    return gdf


# =============================================================================
# STEP 3 — BUILD THE TARGET GRID
# =============================================================================

def build_target_grid(
    lon_start: float, lon_inc: float, lon_size: int,
    lat_start: float, lat_inc: float, lat_size: int,
):
    """
    Builds the 2.5-degree global grid matching CMIP6-ng g025 convention.
    Returns (lons, lats) as 1D numpy arrays.
    """
    lons = np.array([lon_start + i * lon_inc for i in range(lon_size)])
    lats = np.array([lat_start + i * lat_inc for i in range(lat_size)])
    print(f"[INFO] Target grid: {lon_size} lons x {lat_size} lats "
          f"({lons[0]:.2f} to {lons[-1]:.2f}, {lats[0]:.2f} to {lats[-1]:.2f})")
    return lons, lats


# =============================================================================
# STEP 4 — CREATE MASK NETCDF
# =============================================================================

def build_mask_netcdf(
    gdf: gpd.GeoDataFrame,
    lons: np.ndarray,
    lats: np.ndarray,
    output_path: str,
):
    """
    Creates a countrymasks-style NetCDF where each variable is a binary
    mask for one sub-region (1 = inside, 0 = outside).

    Variable naming: m_{COUNTRY}_{REGION} e.g. m_PAK_PUNJAB_12
    This matches the m_ prefix convention in the RIME-X pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Building mask for {len(gdf)} sub-regions ...")
    print("[INFO] This may take a few minutes for large datasets ...")

    # Build regionmask Regions object from the GeoDataFrame
    regions = regionmask.from_geopandas(
        gdf,
        names="mask_name",
        name="subregions",
    )

    # Create the 2D integer mask: each cell contains the region index (or NaN if none)
    mask_int = regions.mask(lons, lats)  # shape: (lat, lon), values = region index

    # Convert to Dataset: one binary variable per sub-region
    print("[INFO] Converting to binary variables per sub-region ...")
    ds = xr.Dataset(coords={"lat": lats, "lon": lons})

    for i, name in enumerate(tqdm(regions.names, desc="Building mask variables")):
        var_name = f"m_{name}"
        # Truncate to 64 chars max (NetCDF variable name limit)
        var_name = var_name[:64]
        binary = xr.where(mask_int == i, 1, 0).astype(np.int8)
        ds[var_name] = binary
        ds[var_name].attrs["long_name"] = f"Binary mask for sub-region: {name}"
        ds[var_name].attrs["units"] = "1"

    # Global attributes
    ds.attrs["title"] = "Sub-national region masks for RIME-X pipeline"
    ds.attrs["source"] = "GADM 4.1 (https://gadm.org)"
    ds.attrs["grid"] = "2.5 degree global (144x72), matching CMIP6-ng g025"
    ds.attrs["convention"] = "1 = inside region, 0 = outside"

    print(f"[INFO] Saving mask NetCDF to: {output_path}")
    ds.to_netcdf(output_path)
    print(f"[INFO] Done. Variables in mask: {len(list(ds.data_vars))}")
    print(f"[INFO] Example variable names: {list(ds.data_vars)[:5]}")

    return ds


# =============================================================================
# STEP 5 — RIME-X PIPELINE (adapted from create_regional_averages_cmip6)
# =============================================================================

def create_regional_averages_cmip6(
    simulation_directory: str,
    simulation_file: str,
    target_directory: str,
    masks: xr.Dataset,
    indicator: str,
    need_global_mean: bool = False,
):
    """
    Computes area-weighted regional averages from a CMIP6 NetCDF file
    using the sub-region masks built above.

    Identical logic to the original RIME-X script, works with any mask
    NetCDF — country-level or sub-national.
    """
    simulation_path = f"{simulation_directory}/{simulation_file}"

    MODEL     = simulation_file.split("_")[2]
    scenario  = simulation_file.split("_")[3]
    ensemble  = simulation_file.split("_")[4]
    SCENARIO  = f"{scenario}-{ensemble}"
    INDICATOR = simulation_file.split("_")[0]

    print(f"[INFO] Processing: {MODEL} | {SCENARIO} | {INDICATOR}")

    simulation = xr.open_dataset(simulation_path).load()

    # Fix lon convention: 0-360 → -180 to 180
    simulation["lon"] = xr.where(
        simulation["lon"] > 180,
        simulation["lon"] - 360,
        simulation["lon"],
    )
    simulation = simulation.sortby("lon")

    # Align mask to simulation grid
    masks_aligned = masks.broadcast_like(simulation.isel(time=0))
    masks_aligned = xr.where(masks_aligned == 0, np.nan, masks_aligned)

    # Area-weighted regional average: sum(mask * data) / sum(mask)
    # (masks already include cos(lat) weighting applied before this function)
    averages = masks_aligned * simulation[indicator]
    averages  = averages.sum(dim=["lat", "lon"]) / masks_aligned.sum(dim=["lat", "lon"])

    # Optionally write global mean CSV (tas only)
    if need_global_mean and "m_GLOBAL" in averages:
        df = pd.DataFrame({
            "time": averages["time"].values,
            "tas":  averages["m_GLOBAL"].values,
        })
        gm_dir = f"{target_directory}/cmip-6_global_mean/{indicator}"
        os.makedirs(gm_dir, exist_ok=True)
        df.to_csv(
            f"{gm_dir}/globalmean_{INDICATOR.lower()}_{MODEL.lower()}_{SCENARIO.lower()}.csv",
            index=False,
        )

    # Write one CSV per sub-region
    for region in list(averages.keys()):
        region_clean = region.replace("m_", "")
        df = pd.DataFrame({
            "time":        averages["time"].values,
            region_clean:  averages[region].values,
        })
        directory = f"{target_directory}/isimip_regional_data/{region_clean}/latWeight"
        filename  = f"{MODEL}_{SCENARIO}_{INDICATOR}_{region_clean}_latweight.csv".lower()

        os.makedirs(directory, exist_ok=True)
        df.to_csv(f"{directory}/{filename}", index=False)

    simulation.close()
    del simulation
    del averages
    del masks_aligned
    gc.collect()
    time.sleep(2)


# =============================================================================
# MAIN — ORCHESTRATES EVERYTHING
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------
    # PART A: Build the sub-region mask NetCDF
    # -------------------------------------------------------------------

    print("=" * 60)
    print("STEP 1: Download GADM")
    print("=" * 60)
    gpkg_path = download_gadm(OUTPUT_DIR, GADM_LEVEL)

    print("\n" + "=" * 60)
    print("STEP 2: Load and filter GADM boundaries")
    print("=" * 60)
    gdf = load_gadm(gpkg_path, GADM_LEVEL, COUNTRIES_FILTER)

    print("\n" + "=" * 60)
    print("STEP 3: Build target grid")
    print("=" * 60)
    lons, lats = build_target_grid(
        LON_START, LON_INC, LON_SIZE,
        LAT_START, LAT_INC, LAT_SIZE,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Build and save sub-region mask NetCDF")
    print("=" * 60)
    mask_ds = build_mask_netcdf(gdf, lons, lats, MASK_OUTPUT_PATH)

    print(f"\n[SUCCESS] Sub-region mask saved to: {MASK_OUTPUT_PATH}")
    print(f"          Total sub-regions in mask: {len(list(mask_ds.data_vars))}")

    # -------------------------------------------------------------------
    # PART B: Run RIME-X pipeline with the new sub-region mask
    # (only runs if you have CMIP6 NetCDF simulation files available)
    # -------------------------------------------------------------------

    # !! UPDATE THESE PATHS to wherever your CMIP6 files are !!
    SIMULATION_DIRECTORIES = {
        "tas":    "/path/to/your/cmip6/tas/mon/g025",
        "pr":     "/path/to/your/cmip6/pr/mon/g025",
        "tasmax": "/path/to/your/cmip6/tasmax/mon/g025",
        "tasmin": "/path/to/your/cmip6/tasmin/mon/g025",
        "rsds":   "/path/to/your/cmip6/rsds/mon/g025",
        "mrso":   "/path/to/your/cmip6/mrso/mon/g025",
        "hurs":   "/path/to/your/cmip6/hurs/mon/g025",
    }
    TARGET_DIRECTORY = "./output_regional_averages"

    SCENARIOS_I_WANT = [
        "historical", "ssp126", "ssp245",
        "ssp370", "ssp460", "ssp585",
    ]

    # Check if any simulation directories actually exist
    valid_indicators = [
        ind for ind, path in SIMULATION_DIRECTORIES.items()
        if os.path.isdir(path)
    ]

    if not valid_indicators:
        print("\n" + "=" * 60)
        print("PART B SKIPPED: No CMIP6 simulation directories found.")
        print("Update SIMULATION_DIRECTORIES paths at the bottom of this")
        print("script once you have your NetCDF files downloaded.")
        print("=" * 60)
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STEP 5: Run RIME-X regional averages pipeline")
    print("=" * 60)

    for indicator in valid_indicators:

        print(f"\n[INFO] Processing indicator: {indicator}")

        # Reload mask fresh each indicator (cos_lat weighting mutates it)
        masks = xr.open_dataset(MASK_OUTPUT_PATH)

        # Apply cos(latitude) area weighting to the masks
        lat_da  = xr.DataArray(masks.lat.values, dims="lat")
        cos_lat = np.cos(np.deg2rad(lat_da))
        cos_lat = xr.DataArray(
            cos_lat, coords=[lat_da], dims=["lat"]
        ).broadcast_like(masks)

        masks       = masks * cos_lat
        masks["m_GLOBAL"] = cos_lat  # add global mask

        simulation_directory = SIMULATION_DIRECTORIES[indicator]
        all_files = os.listdir(simulation_directory)

        # Filter to desired scenarios by position 3 in filename split by _
        all_files = [
            f for f in all_files
            if len(f.split("_")) > 3
            and f.split("_")[3] in SCENARIOS_I_WANT
            and f.endswith(".nc")
        ]

        if not all_files:
            print(f"[WARN] No matching files found in {simulation_directory}")
            continue

        print(f"[INFO] Found {len(all_files)} files to process")

        need_global_mean = (indicator == "tas")

        for file in tqdm(all_files, desc=f"Processing {indicator}"):
            try:
                create_regional_averages_cmip6(
                    simulation_directory,
                    file,
                    TARGET_DIRECTORY,
                    masks,
                    indicator,
                    need_global_mean,
                )
            except Exception as e:
                print(f"[ERROR] Failed on {file}: {e}")
                continue

    print("\n[DONE] All indicators processed.")
    print(f"       Output CSVs written to: {TARGET_DIRECTORY}")
