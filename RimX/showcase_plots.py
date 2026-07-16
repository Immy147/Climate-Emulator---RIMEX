import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.io.shapereader as shpreader
from shapely.vectorized import contains
from rimeX.preproc.quantilemaps import make_quantilemap_prediction
from rimeX.emulator import recombine_gmt_ensemble, load_magicc_ensemble

import random

np.random.seed(42)   # for numpy
random.seed(42)      # for Python's built-in random





def load_CAT_current_policy_scenario(magicc_files):

    test_scenario = []

    for magicc_file in magicc_files:
        test_scenario.append(load_magicc_ensemble(file=magicc_file, projection_baseline=[1995, 2014],
                                                                projection_baseline_offset=0.85))
    #print(test_scenario)
    scenario = pd.concat(test_scenario, axis=1).reindex(np.arange(2015,2100+1,5))

    return scenario

def get_distribution(impact_data_path, indicator, country, scenario_file, years, region=None):
    if region is None:
        region = country

    with xa.open_dataset(impact_data_path) as ds:

        impact_data_quantile_map = ds[indicator].sel(region=region).load()

    scenario = load_CAT_current_policy_scenario(scenario_file)

    emulated_data_new_method = make_quantilemap_prediction(impact_data_quantile_map, scenario,
                                                           quantiles=[float(0 + i * 0.01) for i in range(101)],
                                                           samples=1000, mode="deterministic", clip=True).sel(
        year=years).to_dataframe(name='lfgag', dim_order=['year', 'quantile'])

    emulated_data_new_method = emulated_data_new_method.unstack(level='quantile')

    emulated_data_new_method.columns = emulated_data_new_method.columns.droplevel(0)

    ts0 = emulated_data_new_method

    return ts0

def plot_distribution_results_kde(INDICATOR_DATA_PATH, INDICATOR, COUNTRY, SCENARIO, SCENARIO_FILE, YEARS, UNIT, actual_names, REGION=None, cut=False):
    if REGION is None:
        REGION = COUNTRY

    df = get_distribution(INDICATOR_DATA_PATH, INDICATOR, COUNTRY, SCENARIO_FILE, YEARS, region=REGION)

    if INDICATOR == 'tas':
        df = df - 273.15  

    df.columns = df.columns.astype(float)
    df = df.sort_index(axis=1)

    years_to_plot = YEARS
    df.columns = [round(value,2) for value in df.columns.values]
    quantile_levels = df.columns.values

    # Filter valid years (monotonic values)
    valid_years = [y for y in years_to_plot if np.all(np.diff(df.loc[y].values) > 0)]

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=min(valid_years) - 0.5 * (max(valid_years) - min(valid_years)),
                         vmax=max(valid_years))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4), sharex=False, dpi = 400)

    for year in valid_years:
        quantiles = df.columns.values
        values = df.loc[year].values

        if cut:
            inv_cdf = PchipInterpolator(quantiles, values)
            u_samples = np.random.uniform(0.001, 0.999, 1000)
        else:
            inv_cdf = PchipInterpolator(quantiles, values, extrapolate=True)
            u_samples = np.random.uniform(0.0, 1.0, 2000)

        x_samples = inv_cdf(u_samples)
        kde = gaussian_kde(x_samples)
        x_grid = np.linspace(x_samples.min(), x_samples.max(), 300)
        pdf_vals = kde(x_grid)

        color = cmap(norm(year))
        axs[0].plot(x_grid, pdf_vals, color=color, label=f'{year}')
        axs[0].fill_between(x_grid, pdf_vals, alpha=0.2, color=color)

    #axs[0].set_title(f"Estimated PDFs of {INDICATOR}")
    axs[0].set_xlabel(f'{actual_names.get(INDICATOR,INDICATOR)} Value ({UNIT})')
    axs[0].set_ylabel('Density')
    axs[0].legend(title='Year', loc='upper right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = fig.colorbar(sm, ax=axs[0], label='Year')

    for q in quantile_levels[1:-1]:
        if q == quantile_levels[1]:
            axs[1].plot(df.index, df[q], color='gray', alpha=0.2, linewidth=1, label='Percentiles')
        else:
            axs[1].plot(df.index, df[q], color='gray', alpha=0.2, linewidth=1)

    # Use the last valid year’s color for emphasis
    if valid_years:
        final_year = max(valid_years)
        final_color = cmap(norm(final_year))
    else:
        final_color = 'blue'

    if 0.05 in quantile_levels and 0.95 in quantile_levels:
        axs[1].fill_between(df.index, df[0.05], df[0.95], color=final_color, alpha=0.1, label='5–95% interval')
        axs[1].plot(df.index, df[0.05], color=final_color, alpha=0.8, linestyle='--', label='5th percentile')
        axs[1].plot(df.index, df[0.95], color=final_color, alpha=0.8, linestyle='--', label='95th percentile')

    if 0.5 in quantile_levels:
        axs[1].plot(df.index, df[0.5], color=final_color, linewidth=2, label='Median')

    #axs[1].set_title("Quantile Time Series")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel(f'{actual_names.get(INDICATOR,INDICATOR)} Value ({UNIT})')
    axs[1].legend(loc='best')

    if REGION == COUNTRY:
        title = f"Distribution of {actual_names.get(INDICATOR,INDICATOR)} in {actual_names.get(REGION,REGION)} under {SCENARIO}"
    else:
        title = f"Distribution of {actual_names.get(INDICATOR,INDICATOR)} in {actual_names.get(REGION,REGION)} ({actual_names.get(COUNTRY,COUNTRY)}) under {SCENARIO}"
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def get_distribution_map(
    indicator_data_path,
    indicator,
    years,
    magicc_files
):

    with xa.open_dataset(
        indicator_data_path,
        chunks={
            "lat": 20,
            "lon": 20
        }
    ) as ds:

        impact_data_quantile_map = ds[indicator].isel(
        lat=slice(0, 20),
        lon=slice(0, 20)
)

        scenario = load_CAT_current_policy_scenario(
            magicc_files=magicc_files
        )

        emulated_data_new_method = make_quantilemap_prediction(
            impact_data_quantile_map,
            scenario,
            quantiles=[0.05, 0.5, 0.95],
            samples=1000,
            mode="deterministic",
            clip=True
        ).sel(year=years)

        ts0 = emulated_data_new_method.compute()

    return ts0

def plot_quantile_map(
    indicator_data_path,
    indicator,
    quantiles,
    quantile_labels,
    years,
    scenario_file
):

    da = get_distribution_map(
        indicator_data_path=indicator_data_path,
        indicator=indicator,
        years=years,
        magicc_files=scenario_file
    )

    # Downsample grid massively
    da = da.isel(
        lat=slice(None,None,4),
        lon=slice(None,None,4)
    )

    baseline_year = years[0]
    baseline_2020 = da.sel(year=baseline_year)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(quantiles),
        figsize=(10,4),
        dpi=100,
        subplot_kw={'projection': ccrs.Robinson()}
    )

    if len(quantiles) == 1:
        axes = [axes]

    for i, q in enumerate(quantiles):

        ax = axes[i]

        current = da.sel(quantile=q).mean(dim='year')
        baseline = baseline_2020.sel(quantile=q)

        rel_change = current - baseline

        im = ax.pcolormesh(
            da.lon,
            da.lat,
            rel_change,
            transform=ccrs.PlateCarree(),
            cmap='Reds',
            shading='auto'
        )

        ax.coastlines(resolution='110m')

        ax.set_title(
            f"{quantile_labels[i]}"
        )

        plt.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            pad=0.05
        )

    plt.tight_layout()

    plt.show()

    plt.close('all')

    import gc
    gc.collect()