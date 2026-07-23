import xarray as xr
import numpy as np
import pandas as pd
import os

COUNTRYMASKS = "/media/wcs/Disk3/imran/Rime-X/mnt/PROVIDE/rime-paper-reproduction/data/preprocessing/countrymasks.nc"
REGION_VAR = "m_DEU"

masks = xr.open_dataset(COUNTRYMASKS)
mask = masks[REGION_VAR]

cos_lat = np.cos(np.deg2rad(mask.lat))
cos_lat_2d = cos_lat.broadcast_like(mask)

files = {
    "ipsl-cm6a-lr": {
        "path": "/media/wcs/Disk3/imran/GCM_models/IPSL-CM6A-LR/tas/historical/tas_mon_IPSL-CM6A-LR_historical.nc",
        "outfile": "/media/wcs/Disk3/imran/GCM_models/IPSL-CM6A-LR/regional_averages_output/isimip_regional_data/DEU/latWeight/ipsl-cm6a-lr_historical-r1i1p1f1_tas_deu_latweight.csv",
    },
    "ukesm1-0-ll": {
        "path": "/media/wcs/Disk3/imran/GCM_models/UKESM1-0-LL/tas/historical/tas_mon_UKESM1-0-LL_historical.nc",
        "outfile": "/media/wcs/Disk3/imran/GCM_models/UKESM1-0-LL/regional_averages_output/isimip_regional_data/DEU/latWeight/ukesm1-0-ll_historical-r8i1p1f2_tas_deu_latweight.csv",
    },
}

for model, info in files.items():
    print(f"--- {model} ---")
    sim = xr.open_dataset(info["path"], chunks={"time": 120})
    sim["lon"] = xr.where(sim["lon"] > 180, sim["lon"] - 360, sim["lon"])
    sim = sim.sortby("lon")

    mask_i = mask.interp(lat=sim.lat, lon=sim.lon, method="nearest")
    weights = (cos_lat_2d.interp(lat=sim.lat, lon=sim.lon, method="nearest") * (mask_i > 0))

    tas = sim["tas"]
    weighted = (tas * weights).sum(dim=["lat", "lon"]) / weights.sum(dim=["lat", "lon"])
    weighted = weighted.load()

    df = weighted.to_pandas()
    df.name = "DEU"
    df.index.name = "time"

    os.makedirs(os.path.dirname(info["outfile"]), exist_ok=True)
    df.to_csv(info["outfile"])
    print(f"Saved: {info['outfile']}")
    print(df.head())
    sim.close()
