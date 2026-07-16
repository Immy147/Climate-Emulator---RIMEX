from rimeX.preproc.generic import GenericIndicator
import os, sys
from scipy.stats import gaussian_kde
from tqdm import tqdm
import xarray as xr
import copy
from rimeX.preproc.quantilemaps import make_quantilemap_prediction
from rimeX.emulator import load_magicc_ensemble, recombine_gmt_vectorized, recombine_gmt_ensemble
from rimeX.config import CONFIG as CONFIG_RIMEX
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import numpy as np
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

def find_historical_simulation(simulation, simulations_list):
    for sim in simulations_list:
        if (sim["climate_scenario"] == "historical" and
                sim["climate_forcing"] == simulation["climate_forcing"] and
                sim["ensemble"] == simulation["ensemble"]):
            return sim
    return None


class Indicator(GenericIndicator):
    def get_path(self, region=None, regional=True, regional_weight="latWeight", **simulation_specifiers):
        climate_scenario = simulation_specifiers["climate_scenario"]
        climate_forcing = simulation_specifiers["climate_forcing"]
        ensemble = simulation_specifiers["ensemble"]
        if climate_scenario == 'historical':
            year_range = '1850_2014'
        else:
            year_range = '2015_2100'
        if regional:
            assert regional_weight is not None
            path = Path(f'data/validation/impact_data/raw_averages/{self.name}/{climate_scenario}/{climate_forcing}/'
                        f'{climate_forcing}_{ensemble}_{climate_scenario}_{self.name}_regional_latweight_monthly_{year_range}.csv')
        return path


def get_simulations(indicator_name, raw_averages_path):
    def _get_simulations(name, data_path):
        simulations = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.csv'):
                    parts = file.split('_')
                    model, ensemble, scenario, indicator, iso = parts[0], parts[1], parts[2], parts[3], parts[4]
                    experiment = {'climate_scenario': scenario, 'climate_forcing': model, 'ensemble': ensemble}
                    if indicator == name and experiment not in simulations:
                        simulations.append(experiment)
        return simulations

    simulations_initial = _get_simulations(indicator_name, raw_averages_path)
    simulations = []
    for simulation in simulations_initial:
        if simulation['climate_scenario'] != 'historical':
            if {'climate_scenario': 'historical', 'climate_forcing': simulation['climate_forcing'],
                'ensemble': simulation['ensemble']} in simulations_initial:
                simulations.append(simulation)
        else:
            simulations.append(simulation)
    return simulations


def process_simulation_data_qq(indicator, region, scenario):
    yearly_data = {}
    all_years = []
    for simulation in indicator.simulations:
        if simulation['climate_scenario'] != scenario:
            continue
        if indicator.name == 'tas':
            df_historical = pd.read_csv(
                indicator.get_path(**find_historical_simulation(simulation, indicator.simulations)))
            base_value = df_historical[(df_historical['time'] >= '1995-01-01') &
                                       (df_historical['time'] <= '2014-12-31')][region].mean() - 273.15
            df = pd.read_csv(indicator.get_path(**simulation))
            df['year'] = pd.to_datetime(df['time']).dt.year
            df[region] = df[region] - 273.15 - base_value
        elif indicator.name == 'pr':
            df_historical = pd.read_csv(
                indicator.get_path(**find_historical_simulation(simulation, indicator.simulations)))
            base_value = df_historical[(df_historical['time'] >= '1995-01-01') &
                                       (df_historical['time'] <= '2014-12-31')][region].mean()
            df = pd.read_csv(indicator.get_path(**simulation))
            df['year'] = pd.to_datetime(df['time']).dt.year
            df[region] = (df[region] / base_value - 1) * 100
        else:
            raise NotImplementedError(f'Not implemented for {indicator.name}')
        yearly_df = (
            df[df[region] >= -100]
                .groupby('year', as_index=False)[region]
                .mean()
                .assign(**{
                region: lambda x: x[region].rolling(21, center=True, min_periods=21).mean()})
                .dropna(subset=[region])
        )
        sim_key = (simulation['climate_scenario'], simulation['climate_forcing'], simulation['ensemble'])
        yearly_data[sim_key] = yearly_df
        all_years.append(yearly_df)
    combined_df = pd.concat(all_years, ignore_index=True)
    return yearly_data, combined_df.groupby('year')[region].quantile(
        [p / 100 for p in range(1, 100)]).unstack().reset_index()


def emulate_data_qq(indicator_data_path, gmt_file_path, indicator, region, scenario):
    gmt_file = f"{gmt_file_path}/{scenario}_MAGICC_artificial_cmip6.csv"
    test_scenario = load_magicc_ensemble(file=gmt_file).reindex(np.arange(2025, 2091))
    clean_scenario_information = f'_no{scenario}'
    indicator_file = f'{indicator_data_path}/{indicator}_annual_noadmin_latweight_noneq{clean_scenario_information}_baseline_only_first_wl.nc'
    with xr.open_dataset(indicator_file) as ds:
        impact_data_quantile_map = ds[indicator].sel(region=region)
    emulated_data_new_method = make_quantilemap_prediction(impact_data_quantile_map, test_scenario,
                                                           quantiles=[p / 100 for p in range(1, 100)],
                                                           samples=40000, mode="deterministic")
    return emulated_data_new_method

def emulate_data_qq_maps(MAGICC_file_path, indicator_data_path, scenario, indicator, region):
    magicc_file = f"{MAGICC_file_path}/{scenario}_MAGICC_artificial_cmip6.csv"
    test_scenario = load_magicc_ensemble(
        file=magicc_file#, projection_baseline=[2010, 2019], projection_baseline_offset=1.07
    ).reindex(np.arange(2025, 2090 + 1, 5))
    clean_scenario_information = f'_no{scenario}'
    impact_data_path = f'{indicator_data_path}/{indicator}_annual_noadmin_latweight_noneq{clean_scenario_information}_baseline_only_first_wl.nc'
    with xr.open_dataset(impact_data_path) as ds:
        impact_data_quantile_map = ds[indicator].sel(region=region)

    emulated_data_new_method = make_quantilemap_prediction(impact_data_quantile_map, test_scenario, quantiles=[p/100 for p in range(1,100)], samples=8000, mode="deterministic")

    return emulated_data_new_method


def plot_qq_ax(ax, percentiles_df: pd.DataFrame, data_array: xr.DataArray, years: list, unit: str, return_legend,
               median, lower_bound, upper_bound, i, j):
    # Validate years present in both datasets.
    valid_years = [year for year in years if
                   (year in percentiles_df['year'].values and year in data_array['year'].values)]
    valid_years = sorted(valid_years)
    if not valid_years:
        raise ValueError("No valid years found in both datasets.")
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=min(valid_years) - 30, vmax=max(valid_years))
    all_df_quantiles = []
    all_da_quantiles = []
    year_handles = {}
    fill_between_handles = []
    highlight_styles = {
        4: {"label": "5th percentile", "marker": "o", "edgecolor": "red"},
        49: {"label": "50th percentile", "marker": "s", "edgecolor": "green"},
        94: {"label": "95th percentile", "marker": "^", "edgecolor": "blue"}
    }
    for year in valid_years:
        df_row = percentiles_df[percentiles_df['year'] == year]
        quantile_cols = sorted([col for col in df_row.columns if col != 'year'], key=float)
        df_quantiles = df_row[quantile_cols].values.flatten()
        da_quantiles = data_array.sel(year=year).values
        if len(df_quantiles) != len(da_quantiles):
            print(f"Skipping {year} - quantile mismatch")
            continue
        handle = ax.scatter(df_quantiles, da_quantiles, color=cmap(norm(year)), alpha=0.6,
                            edgecolors='w', linewidths=0.5)
        year_handles[year] = handle

        # Input start
        filling_handle = ax.fill_between(median.sel(year=year).values, lower_bound.sel(year=year).values,
                                         upper_bound.sel(year=year).values,
                                         color=cmap(norm(2020)),
                                         alpha=0.9,  # adjust transparency as needed
                                         label="Minimum and Maximum Error ",
                                         zorder=0)
        # fill_between_handles.append(fill_between_handle)
        # Input end

        for idx, style in highlight_styles.items():
            if idx < len(df_quantiles):
                ax.scatter(df_quantiles[idx], da_quantiles[idx], s=70, marker=style['marker'],
                           facecolors='none', edgecolors=style['edgecolor'], linewidths=2.5)

        all_df_quantiles.extend(df_quantiles)
        all_da_quantiles.extend(da_quantiles)
    min_val = min(min(all_df_quantiles), min(all_da_quantiles))
    max_val = max(max(all_df_quantiles), max(all_da_quantiles))
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='#444444', alpha=0.8)
    if j == 1:
        ax.set_xlabel(f'Simulated Quantiles ({unit})', fontsize=16)
    if j == 0:
        ax.set_ylabel(f'Emulated Quantiles ({unit})', fontsize=16)
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    sorted_years = sorted(year_handles)
    year_handles_list = [year_handles[yr] for yr in sorted_years]
    year_labels = [str(yr) for yr in sorted_years]
    percentile_handles = []
    for idx in [4, 49, 94]:
        style = highlight_styles[idx]
        percentile_handles.append(Line2D([0], [0], marker=style['marker'], color='none',
                                         markerfacecolor='none', markeredgecolor=style['edgecolor'],
                                         markersize=8, markeredgewidth=2.5, label=style['label']))
    ref_handle = Line2D([0], [0], linestyle='--', color='#444444', label='1:1 Reference')
    all_handles = year_handles_list + percentile_handles + [ref_handle] + [filling_handle]
    all_labels = year_labels + [highlight_styles[4]['label'], highlight_styles[49]['label'],
                                highlight_styles[94]['label']] + ['1:1 Reference'] + ['Expected\ndeviation']
    if return_legend:
        return all_handles, all_labels


def generate_sample_gaussian_kernal(percentiles_df, years, number_samples, quantile):
    """
    percentiles_df: pd.DataFrame with years as index and all percentiles of simulation data in the years
    years: list with years we want to generate samples for
    number_samples: number of samples generated per sample set
    quantile: quantile of each sampleset

    output:
    """

    sampled_percentiles_data = []

    for year in years:
        # print(percentiles_df)
        # print(percentiles_df['year'])

        percentiles = copy.deepcopy(percentiles_df)

        percentile_values = percentiles.loc[percentiles.year == int(year), :].values.flatten()[1:]
        kde = gaussian_kde(percentile_values)
        # quantile of the sorted samples -> you get the
        min_max = np.quantile(np.array([sorted(kde.resample(number_samples).flatten().tolist()) for i in range(1000)]),
                              [quantile], axis=0)
        percentiles_samples = np.quantile(min_max, [0.01 + 0.01 * i for i in range(99)])
        sampled_percentiles_data.append(list(percentiles_samples))

    sampled_percentiles = {"coords": {
        "quantile": {"dims": "quantile", "data": [0.01 + 0.01 * i for i in range(99)]},
        "year": {"dims": "year", "data": years},
    },
        "dims": ["quantile", "year"],
        "data": np.array(sampled_percentiles_data).T
    }

    return xr.DataArray.from_dict(sampled_percentiles)


def create_qq_validation_plot(RAW_AVERAGES_PATH, GMT_FILE_PATH, INDICATOR_DATA_PATH, SCENARIO, IND_REGIONS_MAP):
    # Create a figure with 2 rows x 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=400)
    years_to_plot = [2035 + i * 5 for i in range(12)]

    # We'll capture legend handles/labels from the first subplot
    legend_handles = None
    legend_labels = None

    for i, indicator in enumerate(IND_REGIONS_MAP.keys()):
        for j, region in enumerate(IND_REGIONS_MAP[indicator]):
            print(f'Processing {indicator}, {region}')
            # Update CONFIG for this subplot
            if indicator == 'tas':
                unit = '°C'
            elif indicator == 'pr':
                unit = '%'

            # Create Indicator instance and process data
            ind_obj = Indicator(name=indicator, simulations=get_simulations(indicator, RAW_AVERAGES_PATH))
            print(f'Got {ind_obj.name}')
            yearly_data, percentiles_df = process_simulation_data_qq(ind_obj, region,SCENARIO)
            print(f'Processed sim data {ind_obj.name}')
            emulated_data = emulate_data_qq(INDICATOR_DATA_PATH, GMT_FILE_PATH, indicator, region, SCENARIO)
            print(f'Got emulations {ind_obj.name}')
            ax = axes[i, j]

            # years_to_plot = [2090]# + i*5 for i in range(0, 12)]

            gaussian_samples_95 = generate_sample_gaussian_kernal(percentiles_df, years_to_plot, 100, quantile=0.99)
            gaussian_samples_05 = generate_sample_gaussian_kernal(percentiles_df, years_to_plot, 100, quantile=0.01)
            gaussian_samples_median = generate_sample_gaussian_kernal(percentiles_df, years_to_plot, 100, quantile=0.5)
            # For the first subplot, get legend handles
            if i == 0 and j == 0:
                legend_handles, legend_labels = plot_qq_ax(ax, percentiles_df, emulated_data, years_to_plot, unit, True,
                                                           gaussian_samples_median, gaussian_samples_05,
                                                           gaussian_samples_95, i, j)
            else:
                plot_qq_ax(ax, percentiles_df, emulated_data, years_to_plot, unit, False, gaussian_samples_median,
                           gaussian_samples_05, gaussian_samples_95, i, j)
            # Label each subplot with letters (a) through (f)
            # subplot_label = chr(97 + i*3 + j)  # 97 is 'a'
            subplot_label = f"Annual mean {indicator} in {region}"
            ax.set_title(subplot_label, fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)

    # Add a single global legend on the right side
    fig.legend(handles=legend_handles, labels=legend_labels, loc='center right', fontsize=16)
    # fig.suptitle("Q-Q comparison of emulated vs. simulated percentiles for selected indicators and countries", fontsize=22)
    plt.tight_layout(rect=[0, 0, 0.85, 0.93])
    plt.show()


def calculate_normalized_mae(percentiles_df, simulated_da):
    """
    Calculates the mean absolute error (MAE) between emulated and simulated quantiles,
    normalized by the range (max-min) of simulated quantiles.

    Parameters:
      percentiles_df (pd.DataFrame): DataFrame containing year in the first column
                                     and quantiles as column names (as strings or floats).
      simulated_da (xr.DataArray): Xarray DataArray containing simulated quantiles
                                   with dimensions (quantile, year).

    Returns:
      dict: Dictionary mapping years to their normalized MAE values.
    """
    # Extract years that are common in both datasets
    common_years = sorted(set(percentiles_df["year"]).intersection(simulated_da.year.values))

    mae_values = {}

    for year in common_years:
        # Extract emulated values for the given year
        emulated_values = percentiles_df.loc[percentiles_df["year"] == year].drop(columns="year").values.flatten()

        # Extract corresponding simulated values
        simulated_values = simulated_da.sel(year=year).values

        # Ensure they are comparable
        if len(emulated_values) != len(simulated_values):
            raise ValueError(f"Mismatch in number of quantiles for year {year}: "
                             f"emulated={len(emulated_values)}, simulated={len(simulated_values)}")

        # Calculate MAE
        mae = np.mean(np.abs(emulated_values - simulated_values))

        # Normalize by the range of simulated values
        sim_range = np.max(simulated_values) - np.min(simulated_values)
        normalized_mae = mae / sim_range if sim_range != 0 else np.nan  # Avoid division by zero

        mae_values[year] = normalized_mae

    return mae_values


def calculate_maximum_normalized_mae(percentiles_df, simulated_da):
    """
    Calculates the mean absolute error between emulated and simulated quantiles,
    normalized by the range (max-min) of simulated quantiles.

    Parameters:
      percentiles_df (pd.DataFrame): DataFrame containing year in the first column
                                     and quantiles as column names (as strings or floats).
      simulated_da (xr.DataArray): Xarray DataArray containing simulated quantiles
                                   with dimensions (quantile, year).

    Returns:
      dict: Dictionary mapping years to their normalized MAE values.
    """
    # Extract years that are common in both datasets
    common_years = sorted(set(percentiles_df["year"]).intersection(simulated_da.year.values))

    mae_values = {}

    for year in common_years:
        # Extract emulated values for the given year
        emulated_values = percentiles_df.loc[percentiles_df["year"] == year].drop(columns="year").values.flatten()

        # Extract corresponding simulated values
        simulated_values = simulated_da.sel(year=year).values

        # Ensure they are comparable
        if len(emulated_values) != len(simulated_values):
            raise ValueError(f"Mismatch in number of quantiles for year {year}: "
                             f"emulated={len(emulated_values)}, simulated={len(simulated_values)}")

        # Calculate MAE
        mae = np.max(np.abs(emulated_values - simulated_values))

        # Normalize by the range of simulated values
        sim_range = np.max(simulated_values) - np.min(simulated_values)
        normalized_mae = mae / sim_range if sim_range != 0 else np.nan  # Avoid division by zero

        mae_values[year] = normalized_mae

    return mae_values


def average_normalized_mae(normalised_mae_per_year):
    full_sum = 0
    for year in normalised_mae_per_year.keys():
        full_sum += normalised_mae_per_year[year]

    if len(normalised_mae_per_year.keys()) != 0:

        return full_sum / len(normalised_mae_per_year.keys())

    else:
        return None


def maximum_normalized_mae(normalised_mae_per_year):
    full_sum = 0
    for year in normalised_mae_per_year.keys():
        full_sum += normalised_mae_per_year[year]

    if len(normalised_mae_per_year.keys()) != 0:

        return full_sum / len(normalised_mae_per_year.keys())

    else:
        return None


def save_dict_to_json(data, path):
    path = Path(path)  # Ensure path is a Path object
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _get_simulations(name, data_path=CONFIG_RIMEX["indicators.folder"]):
    simulations = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):

                parts = file.split('_')
                model, ensemble, scenario, indicator, iso = parts[0], parts[1], parts[2], parts[3], parts[4]
                experiment = {'climate_scenario': scenario, 'climate_forcing': model, 'ensemble': ensemble}
                if indicator == name and experiment not in simulations:
                    simulations.append(experiment)

    return simulations

import gc

def create_mae_data(out_path, impact_data_path, MAGICC_file_path):
    for scenario_name in ['ssp245']:
        for indicator_name in ['tas', 'pr']:

            IND_CONFIG = {
                "SCENARIO": scenario_name,
                "INDICATOR": indicator_name}

            indicator_name = IND_CONFIG['INDICATOR']

            simulations_initial = _get_simulations(indicator_name)

            simulations = []
            for simulation in simulations_initial:
                if simulation['climate_scenario'] != 'historical':
                    if {'climate_scenario': 'historical', 'climate_forcing': simulation['climate_forcing'],
                        'ensemble': simulation['ensemble']} in simulations_initial:
                        simulations.append(simulation)
                else:
                    simulations.append(simulation)

            indicator = Indicator(name=IND_CONFIG["INDICATOR"], simulations=simulations)

            clean_scenario_information = f'_no{IND_CONFIG["SCENARIO"]}'
            impact_data_set = f'{impact_data_path}/{IND_CONFIG["INDICATOR"]}_annual_noadmin_latweight_noneq{clean_scenario_information}_baseline_only_first_wl.nc'

            with xr.open_dataset(impact_data_set) as ds:
                regions = ds[IND_CONFIG['INDICATOR']].region.load().values

            mae_regions = {}
            maximum_mae_regions = {}

            for region in tqdm(regions[regions]):

                yearly_data, percentiles_df = process_simulation_data_qq(indicator, region, IND_CONFIG["SCENARIO"])
                emulations_df = emulate_data_qq_maps(MAGICC_file_path, impact_data_path, IND_CONFIG["SCENARIO"], IND_CONFIG["INDICATOR"], region)

                normalised_mae_per_year = calculate_normalized_mae(percentiles_df, emulations_df)

                mae_regions[region] = average_normalized_mae(normalised_mae_per_year)

                normalised_maximum_mae_per_year = calculate_maximum_normalized_mae(percentiles_df, emulations_df)

                maximum_mae_regions[region] = maximum_normalized_mae(normalised_maximum_mae_per_year)

                  # Explicitly delete large intermediates and force GC each iteration
                del percentiles_df, emulations_df
                del yearly_data
                del normalised_mae_per_year, normalised_maximum_mae_per_year
                gc.collect()
            
            path_average = f'{out_path}/amae_{indicator_name}_{scenario_name}.json'
            path_maximum = f'{out_path}/maximum_amae_{indicator_name}_{scenario_name}.json'

            save_dict_to_json(mae_regions, path_average)
            save_dict_to_json(maximum_mae_regions, path_maximum)

def open_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_country_values_ax(value_dict, ax, title, vmin=None, vmax=None):
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(url)
    if 'ISO3166-1-Alpha-3' in world.columns:
        world = world.rename(columns={'ISO3166-1-Alpha-3': 'iso_a3'})
    df_values = pd.DataFrame(list(value_dict.items()), columns=['iso_a3', 'value'])
    world = world.merge(df_values, on='iso_a3', how='left')
    world.plot(column='value', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8',
               legend=False, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

def create_amae_map_validation_plot(path_error_data, scenario, useOwnErrorData):

    if not useOwnErrorData:

        path_tas = f'{path_error_data}/amae_tas_{scenario}_precalculated.json'
        path_pr  = f'{path_error_data}/amae_pr_{scenario}_precalculated.json'

    else:

        path_tas = f'{path_error_data}/amae_tas_{scenario}.json'
        path_pr  = f'{path_error_data}/amae_pr_{scenario}.json'

        mae_regions_tas = open_json_file(path_tas)
        mae_regions_pr  = open_json_file(path_pr)

        # Safe handling of Sudan/South Sudan
        if 'SDN' in mae_regions_tas:
            mae_regions_tas['SSD'] = mae_regions_tas['SDN']
        else:
            print("Warning: SDN not found in TAS data")

        if 'SDN' in mae_regions_pr:
            mae_regions_pr['SSD'] = mae_regions_pr['SDN']
        else:
            print("Warning: SDN not found in PR data")

    # rest of your plotting code continues here


    # Create side-by-side maps
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=400)
    plot_country_values_ax(mae_regions_tas, axes[0], "(a)", vmin=common_vmin, vmax=common_vmax)
    plot_country_values_ax(mae_regions_pr, axes[1], "(b)", vmin=common_vmin, vmax=common_vmax)

    plt.subplots_adjust(right=0.85)
    norm = colors.Normalize(vmin=common_vmin, vmax=common_vmax)
    sm = ScalarMappable(norm=norm, cmap="Reds")
    sm.set_array([])
    # Thinner colorbar: reduce width from 0.02 to 0.015
    cbar_ax = fig.add_axes([0.88, 0.12, 0.015, 0.76])
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="ANMAE")

    # Overall title with reduced white space between title and maps
    fig.suptitle("ANMAE of emulated vs. simulated percentiles by country", fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 0.85, 0.94])
    plt.show()
