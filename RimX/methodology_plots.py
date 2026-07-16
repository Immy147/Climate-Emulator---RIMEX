import pandas as pd
import numpy as np
import math
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap



def sort_three_lists(list1, list2, list3):
    # Zip the three lists together, sort based on list1, then unzip
    sorted_triples = sorted(zip(list1, list2, list3))
    sorted_list1, sorted_list2, sorted_list3 = zip(*sorted_triples)
    return list(sorted_list1), list(sorted_list2), list(sorted_list3)

def load_impact_data(path, warming_level=1.5):
    data = pd.read_csv(path)
    data = data[data['season'] == 'annual']
    data = data[data['warming_level'] == warming_level]
    return data[["value", "warming_level"]]

def load_scenario_data(paths, year, num_values='all'):
    # If a single path is given, make it a list
    if isinstance(paths, str):
        paths = [paths]

    # Load and concatenate all CSVs
    data_list = [pd.read_csv(p) for p in paths]
    data = pd.concat(data_list, ignore_index=True)

    # Extract the column for the specified year
    values = data[year]

    # Return all values or a random sample
    if num_values == 'all':
        return values
    else:
        return values.sample(n=num_values, random_state=42).reset_index(drop=True)



def plot_data(probabilitys_bins, warming_levels):
    plt.figure(figsize=(16, 6))
    x_values = list(probabilitys_bins.keys())
    probabilities = list(probabilitys_bins.values())
    probabilities_abs = [np.sum(np.array(probabilities[:i])) for i in range(len(probabilities))]
    probabilitys_bins_full = {}
    #for i,key in enumerate(probabilitys_bins.keys()):
    #    probabilitys_bins[key] = probabilities_abs[i]
    plt.step(x_values, probabilitys_bins.values(), where='post', label='Probability Density', color='red', linewidth=2)
    #for i in range(len(x_values) - 1):
    #    plt.hlines(probabilities[i], x_values[i], x_values[i + 1], color='b', linewidth=2)
    # Add warming levels as scatter points on the x-axis
    plt.scatter(warming_levels, [0] * len(warming_levels), color='red', marker='o', label='Warming Levels', s=2)
    print(probabilitys_bins)
    # Draw vertical lines from each warming level point on the x-axis to the step function
    for level in warming_levels:
        # Find the nearest x-value bin to determine where the step function is
        nearest_bin = np.round(np.float64(math.floor(level*10)/10),1)
        y_value = probabilitys_bins.get(nearest_bin,0)

        # Draw a vertical line from the warming level on the x-axis to the probability function graph
        #plt.vlines(level, 0, y_value, color='green', linestyles='dashed',linewidth=1)

    # Adding labels and title
    plt.xlabel('X-Value')
    plt.ylabel('Probability Density')
    plt.title('Step Function of Probability Density with Warming Levels')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def get_probability_bins(possible_warming_levels,number_to_select = 600, bin_size=0.1):
    min_warming_level = math.floor(np.min(possible_warming_levels) * 10) / 10
    max_warming_level = math.floor(np.max(possible_warming_levels) * 10 + 10) / 10
    number_bins = int((max_warming_level - min_warming_level) / bin_size)
    bins = np.linspace(min_warming_level, max_warming_level, num=number_bins)
    binned_warming_levels = np.digitize(possible_warming_levels, bins)
    unique, counts = np.unique(binned_warming_levels, return_counts=True)
    item_number_bin = dict(zip(unique, counts))
    probabilitys_bins = {np.round(min_warming_level + bin * bin_size, 1): item_number_bin.get(bin, 0) / number_to_select
                         for bin in item_number_bin}
    return probabilitys_bins

def plot_distribution(probabilities, color,year, warming_levels=None):

    probabilities = {key: probabilities[key] for key in sorted(probabilities.keys())}
    x_values = list(probabilities.keys())
    probabilities = list(probabilities.values())
    probabilities_abs = [np.sum(np.array(probabilities[:i])) for i in range(len(probabilities))]
    probabilitys_bins_full = {}
    for i,key in enumerate(x_values):
        probabilitys_bins_full[key] = probabilities_abs[i]
    plt.step(x_values, probabilities_abs, where='post', color=color, linewidth=2, label=f"{year}")
    #for i in range(len(x_values) - 1):
    #   plt.hlines(probabilities[i], x_values[i], x_values[i + 1], color=color, linewidth=2)
    if warming_levels is not None:
        if year == '2100':
            plt.scatter(warming_levels, [-0.05] * len(warming_levels), color=color, marker='o', label=f'Sample', s=2)
        else:
            plt.scatter(warming_levels, [-0.05] * len(warming_levels), color=color, marker='o',  s=2)

        # Draw vertical lines from each warming level point on the x-axis to the step function
        for level in warming_levels:
            # Find the nearest x-value bin to determine where the step function is
            nearest_bin = np.round(np.float64(math.floor(level * 10) / 10), 1)
            y_value = probabilitys_bins_full.get(nearest_bin, 0)

            # Draw a vertical line from the warming level on the x-axis to the probability function graph
            #plt.vlines(level, 0, y_value, color='green', linestyles='dashed', linewidth=1)



def create_yearly_warming_level_bins(years, scenario_data_paths, number_of_SCM_ensembles=600):
    probabilities = {}
    bins = []
    #plt.figure(figsize=(16, 6))
    for year in years:
        possible_warming_levels = np.array(load_scenario_data(scenario_data_paths, year).values)
        bin_size = 0.1
        probabilitys_bins = get_probability_bins(possible_warming_levels, number_to_select = number_of_SCM_ensembles)
        probabilities[year] = probabilitys_bins
        bins.extend(probabilitys_bins.keys())

    bins = sorted(list(set(bins)))

    return probabilities, bins



def plot_probability_functions(scenario_data_paths,  is_main_figure = False, number_selected = 600):

    if is_main_figure:
        plt.figure(figsize=(8, 6), dpi = 400)
    else:
        plt.figure(figsize=(10, 5), facecolor='white', dpi = 400) 
    probabilities, bins = create_yearly_warming_level_bins([str(2015 + x) for x in range(0, 86, 5)], scenario_data_paths, number_of_SCM_ensembles=number_selected)
    norm = Normalize(vmin=0, vmax=16)
    colors = cm.Reds(norm(range(19)))
    for i,year in enumerate([str(2020+x) for x in range(0,86,10)]):
        for bin in bins:
            if bin in probabilities[year].keys():
                continue
            else:
                probabilities[year][bin] = 0
        if year == '2100' or year == '2020':
            warming_levels = np.array(load_scenario_data(scenario_data_paths, year, number_selected).values)
        else:
            warming_levels = None
        plot_distribution(probabilities[year],colors[10+i],year, warming_levels)

    #plt.title("Comulative Distribution Function (CDF) for the GMT reached in the given year under the Climate Action Tracker's Current Policies Scenario")
    if not is_main_figure:
        plt.title('Probability Function for GMT reached in the given Year \n under the CAT Current Policies Scenario', fontsize = 16)
        plt.xlabel('GMT [anomaly in °C above pre-industrial]', fontsize = 16)
        plt.ylabel('Probability', fontsize = 16)
        plt.grid(True)
        plt.legend(fontsize=14)

    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()

def plot_MAGICC_timeseries(
    scenario_data_paths,
    number_of_SCM_ensembles=None,
    is_main_figure=False
):

    probabilities = {}
    bins = []

    if is_main_figure:
        plt.figure(figsize=(8, 6), dpi=400)
    else:
        plt.figure(figsize=(10, 5), dpi=400)

    years = [str(2000 + x) for x in range(0, 101, 5)]

    # Automatically determine ensemble size if not provided
    if number_of_SCM_ensembles is None:

        sample_data = load_scenario_data(
            scenario_data_paths,
            years[0]
        )

        number_of_SCM_ensembles = len(sample_data)

    # Create storage array
    warming_timeseries = np.ones(
        (len(years), number_of_SCM_ensembles)
    )

    # Load warming data
    for i, year in enumerate(years):

        possible_warming_levels = np.array(
            load_scenario_data(
                scenario_data_paths,
                year
            ).values
        )

        warming_timeseries[i, :] = possible_warming_levels

        probabilitys_bins = get_probability_bins(
            possible_warming_levels
        )

        probabilities[year] = probabilitys_bins

        bins.extend(probabilitys_bins.keys())

    # Plot ensemble members
    for i in range(number_of_SCM_ensembles - 1):

        plt.plot(
            years,
            warming_timeseries[:, i],
            alpha=0.1,
            color="grey",
            linewidth=1,
            zorder=-1
        )

    plt.plot(
        years,
        warming_timeseries[:, number_of_SCM_ensembles - 1],
        color="grey",
        linewidth=1,
        zorder=-1,
        label="MAGICC Ensemble Member"
    )

    norm = Normalize(vmin=0, vmax=16)

    colors = cm.Reds(norm(range(19)))

    years_counted = 0

    if not is_main_figure:
        plt.grid(True, which='major', zorder=1)

    # Scatter points
    for i, year in enumerate(years):

        if year in [str(2020 + x) for x in range(0, 86, 10)]:

            if year == '2100':

                plt.scatter(
                    [year] * number_of_SCM_ensembles,
                    warming_timeseries[i, :],
                    color=colors[10 + years_counted],
                    zorder=10,
                    s=5,
                    label="GMT reached by Ensemble Member in Year"
                )

            else:

                plt.scatter(
                    [year] * number_of_SCM_ensembles,
                    warming_timeseries[i, :],
                    color=colors[10 + years_counted],
                    zorder=10,
                    s=5
                )

            years_counted += 1

    if not is_main_figure:

        plt.title(
            "GMT Projections from the MAGICC SCM for the CAT Current Policies Scenario",
            fontsize=16
        )

        plt.xlabel('Year', fontsize=16)

        plt.ylabel(
            'GMT [anomaly in °C above pre-industrial]',
            fontsize=13
        )

        plt.legend(fontsize=14)

    plt.xticks(years[::2])

    plt.tick_params(
        axis='both',
        which='major',
        labelsize=14
    )

    plt.show()

    plt.close('all')

def plot_regional_impact_distribution_from_warming_levels(impact_data_path, scenario_data_paths, is_main_figure = False):
    years = [str(2020 + x) for x in range(0, 86, 10)]
    probabilities, bins = create_yearly_warming_level_bins(years, scenario_data_paths)
    if is_main_figure:
        fig, ax = plt.subplots(1,1,figsize=(8, 6), dpi = 400)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi = 400) # -> For poster
    norm = Normalize(vmin=0, vmax=(10+len(bins)))
    colors = cm.Reds(norm(range(10+len(bins))))
    colors_grey = cm.Greys(norm(range((10 + len(bins)))))
    for i, warming_level in enumerate(bins):
        data = load_impact_data(
            impact_data_path,
            warming_level)

        ax.plot(sorted(data["value"]), [i / (len(data["value"]) - 1) for i in range(len(data["value"]))], color=colors[10 + i],
                 alpha=0.7)
        if warming_level == max(bins):
            ax.scatter(sorted(data["value"]), [i / (len(data["value"]) - 1) for i in range(len(data["value"]))],
                        color=colors[10 + i], label = f'Samples')
        else:
            ax.scatter(sorted(data["value"]), [i / (len(data["value"]) - 1) for i in range(len(data["value"]))],
                       color=colors[10 + i])


    start_color = colors[10]
    end_color = colors[9 + len(bins)]
    cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=len(bins))

    norm = plt.Normalize(min(bins), max(bins))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('GMT [anomaly in °C above pre-industrial]', fontsize = 16)

    #ax.set_title("Cumulative Distribution Function of Regional Temperature Increase at each GMT")
    if not is_main_figure:
        ax.set_title("CDFs of Temperature Increase in Germany at each GMT", fontsize = 16)
        ax.set_xlabel('Temperature Increase in Germany [°C above 2005 to 2015 baseline]', fontsize = 16)
        ax.set_ylabel('Probability', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize = 14)

    plt.show()

def plot_complete_CDF(years, scenario_data_paths, impact_data_path, number_of_SCM_ensembles, is_main_figure = False):

    probabilities, bins = create_yearly_warming_level_bins(years, scenario_data_paths, number_of_SCM_ensembles)

    if is_main_figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=400) # -> for Paper
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=400) # -> For paper

    values = {}
    probabilities_value = {}
    color = {}
    norm = Normalize(vmin=0, vmax=10 + len(bins))
    colors = cm.Reds(norm(range((10 + len(bins)))))
    for year in years:
        values[year] = []
        probabilities_value[year] = []
        color[year] = []
        for i, warming_level in enumerate(bins):
            if warming_level in probabilities[year].keys():
                data = load_impact_data(
                    impact_data_path,
                    warming_level)

                values[year].extend(data["value"])
                probabilities_value[year].extend(list(
                    np.array([1 / (len(data["value"]) - 1) for i in range(len(data["value"]))]) * probabilities[year][
                        warming_level]))
                color[year].extend([colors[10 + i] for j in range(len(data["value"]))])

        values[year], probabilities_value[year], color[year] = sort_three_lists(values[year], probabilities_value[year],
                                                                                color[year])

    norm = Normalize(vmin=0, vmax=10 + len(years))
    colors_grey = cm.Greys(norm(range((10 + len(years)))))
    for i, year in enumerate(years):
        alpha = 0.7
        ax.plot(values[year], np.cumsum(np.array(probabilities_value[year])), color=colors_grey[5 + i], alpha=alpha,
                label=f'Year: {year}')
        if year == '2100':
            ax.scatter(values[year], np.cumsum(np.array(probabilities_value[year])), color=color[year], s=20,
                       label='Samples')
        else:
            ax.scatter(values[year], np.cumsum(np.array(probabilities_value[year])), color=color[year], s=20)

    start_color = colors[10]
    end_color = colors[9 + len(bins)]

    # Create the color gradient
    norm = plt.Normalize(0, 1)
    gradient = np.linspace(0, 1, 100)

    cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=100)

    #ax.set_title(
    #    f"Cumulative Distribution Function of Regional Temperature Increase over Germany in the Years 2020 to 2100 under Current Policies")
    #ax.set_title("                                   (c)")
    if not  is_main_figure:
        ax.set_title("CDFs for Temperature Increase in Germany under the CAT Current Policies Scenario in the given Years", fontsize = 16)
        ax.set_xlabel(f'Temperature Increase in Germany [°C above 2005 to 2015 baseline]', fontsize = 16)
        ax.set_ylabel('Probability', fontsize= 16)
        norm = plt.Normalize(min(bins), max(bins))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('GMT [anomaly in °C above pre-industrial]', fontsize = 16)
        ax.legend(fontsize = 14)
        ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()



