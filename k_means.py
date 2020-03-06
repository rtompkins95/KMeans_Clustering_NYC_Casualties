"""
File: Tompkins_Ryan.py
Author: Ryan Tompkins
Class: Principles of Data Mining
Prof: Dr. Kinsman
"""
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.cluster import KMeans


def numpy_hist(data, title, xlabel, ylabel):
    plt.hist2d(data[0], data[1])  # arguments are passed to np.histogram
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def pandas_csv(filename):
    df = pd.read_csv(filename, parse_dates=True)
    return df


def lat_long(data):
    locs = data[['LATITUDE', 'LONGITUDE']].dropna()
    return np.array(locs)


def casualty_heat_map_by_location(data):
    locs = lat_long(data)
    numpy_hist(locs, 'Contour for Number of Accidents at Latitude/Longitude')


def pedestrian_casulaties_with_dates(data):
    """
        DATE
        NUMBER OF PEDESTRIANS INJURED,
        NUMBER OF PEDESTRIANS KILLED,
    """
    injury_data = data[['DATE', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED']].dropna()
    injuries = np.array(injury_data)
    return injuries


def total_casualties_with_dates(data):
    injury_data = data[['DATE', 'NUMBER OF PERSONS KILLED',
                        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].dropna()

    injuries = np.array(injury_data)

    return injuries


def total_casualties_with_locations(data):
    injury_data = data[['LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS KILLED',
                        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].dropna()

    return injury_data


def total_borough_data(data):
    injury_data = data[['DATE', 'BOROUGH', 'NUMBER OF PERSONS KILLED',
                        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].dropna()

    injuries = np.array(injury_data)

    return injuries


def detailed_injury_data(data):
    injury_data = data[['DATE', 'BOROUGH', 'NUMBER OF PERSONS KILLED',
                        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].dropna()

    boroughs = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'STATEN ISLAND', 'QUEENS']

    injuries = []
    for b in boroughs:
        curr = injury_data.loc[injury_data['BOROUGH'] == b]
        injuries.append(injury_data.loc[injury_data['BOROUGH'] == b])

    borough_map = {'BRONX': np.array(injuries[0]),
                   'BROOKLYN': np.array(injuries[1]),
                   'MANHATTAN': np.array(injuries[2]),
                   'STATEN ISLAND': np.array(injuries[3]),
                   'QUEENS': np.array(injuries[4])}

    return borough_map


def cyclist_casualties_with_dates(data):
    injury_data = data[['DATE', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', ]].dropna()
    injuries = np.array(injury_data)

    return injuries


def motorist_casualties_with_dates(data):
    injury_data = data[['DATE', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].dropna()
    injuries = np.array(injury_data)

    return injuries


def casualties_by_month_graph(injuries, partition_index):
    # print('num injuries: ')
    # print(len(injuries))

    months = {}

    for i in range(13):
        if i == 0:
            continue
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)
        months[i] = 0

    days = {}

    for i in range(32):
        if i == 0:
            continue
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)
        days[i] = 0

    dates_to_casualties = {}

    total_casualties = 0
    for i in injuries:
        # print(i)
        # print(np.sum(i[1:]))
        # print(i[0].split('/'))

        curr_injury_sum = np.sum(i[partition_index:])

        total_casualties += curr_injury_sum

        month = i[0].split('/')[0]
        months[month] += curr_injury_sum

        if i[0] not in dates_to_casualties.keys():
            dates_to_casualties[i[0]] = curr_injury_sum
        else:
            dates_to_casualties[i[0]] += curr_injury_sum

    print('months')
    print(months)
    print('total causualties:')
    print(total_casualties)
    # print('dates to casualties:')
    # print(dates_to_casualties)

    lists = sorted(months.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    # NUM_ENTRIES = str(len(injuries) + 1)
    # title = 'Number of Casualties by Month (first ' + NUM_ENTRIES + ' entries)'
    # plt.title(title)
    # plt.xlabel('Months (\'MM\' format)')
    # plt.ylabel('Number of casualties (injuries + deaths)')
    plt.plot(x, y)


def casualties_by_month_data(injuries, sum_index):
    months = {}
    for i in range(13):
        if i == 0:
            continue
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)
        months[i] = 0

    for i in injuries:
        curr_injury_sum = np.sum(i[sum_index:])

        month = i[0].split('/')[0]
        months[month] += curr_injury_sum

    lists = sorted(months.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    return (x, y)


def casualties_by_borough_graph(injuries, casualty_type):
    # print('num injuries: ')
    # print(len(injuries))

    boroughs = {'BRONX': 0, 'BROOKLYN': 0, 'MANHATTAN': 0, 'STATEN ISLAND': 0, 'QUEENS': 0}
    for i in injuries:
        curr_injury_sum = np.sum(i[2:])
        boroughs[i[1]] += curr_injury_sum

    lists = sorted(boroughs.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    NUM_ENTRIES = str(len(injuries) + 1)
    title = 'Number of ' + casualty_type + ' Casualties by Borough (first ' + NUM_ENTRIES + ' entries)'
    plt.title(title)
    plt.xlabel('Borough')
    plt.ylabel('Number of casualties (injuries + deaths)')
    plt.plot(x, y)
    plt.show()


def borough_casualties_by_month_graph(data):
    index_of_borough_col = 2
    boroughs = {'BRONX': 0, 'BROOKLYN': 0, 'MANHATTAN': 0, 'STATEN ISLAND': 0, 'QUEENS': 0}
    borough_names = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'STATEN ISLAND', 'QUEENS']
    NUM_ENTRIES = str(len(data) + 1)

    detailed_borough = detailed_injury_data(data)
    for b in detailed_borough.keys():
        borough_injuries = detailed_borough[b]
        boroughs[b] = casualties_by_month_data(borough_injuries, index_of_borough_col)

    title = 'Number of Casualties by Month (first ' + NUM_ENTRIES + ' entries)'
    plt.title(title)
    plt.xlabel('Months (\'MM\' format)')
    plt.ylabel('Number of casualties (injuries + deaths)')

    for b in boroughs.keys():
        x, y = boroughs[b]
        plt.plot(x, y)

    plt.legend(borough_names, loc='upper left')
    plt.show()


def casualty_by_month_analysis(data):
    total_injuries = total_casualties_with_dates(data)

    total_borough = total_borough_data(data)
    casualties_by_borough_graph(total_borough, 'Total')

    cyclist_injuries = cyclist_casualties_with_dates(data)
    motorist_injuries = motorist_casualties_with_dates(data)
    ped_injuries = pedestrian_casulaties_with_dates(data)

    NUM_ENTRIES = str(len(data) + 1)
    title = 'Number of Casualties by Month (first ' + NUM_ENTRIES + ' entries)'
    plt.title(title)
    plt.xlabel('Months (\'MM\' format)')
    plt.ylabel('Number of casualties (injuries + deaths)')

    casualties_by_month_graph(ped_injuries, partition_index=1)
    casualties_by_month_graph(cyclist_injuries, partition_index=1)
    casualties_by_month_graph(motorist_injuries, partition_index=1)
    casualties_by_month_graph(total_injuries, partition_index=1)

    plt.legend(['Pedestrian', 'Cyclist', 'Motorist', 'Total'], loc='upper left')
    plt.show()


def graph_cumulutive_explained_variance(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    plt.show()


def do_pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)
    print("original shape:   ", data.shape)
    print("transformed shape:", data_pca.shape)

    return data_pca


def impute_missing_values(data):
    imp = SimpleImputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_data = imp.fit_transform(data)

    return imputed_data


def one_hot_encoder(data):
    le = preprocessing.LabelEncoder()

    data_2 = data.apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    return enc.fit_transform(data_2)


def scikit_kmeans_graph(data, num_clusters):
    # data = np.swapaxes(data, 0, 1)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)

    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=200, cmap='viridis')

    centers = kmeans.cluster_centers_
    print('cluster_centers:')
    print(centers)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=500, alpha=0.5)

    plt.title('K-Means Clustering (K = ' + str(num_clusters) + ' )')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def clean_lat_longs(data):
    data = np.array(data)

    cleaned_data = []

    for d in data:
        if d[1] < -150:
            d[0] = 40.77
            d[1] = -73.95
        if -70 < d[1] < -15:
            continue
        if d[1] == 0.0 and d[2] == 0.0:
            continue

        cleaned_data.append([d[0], d[1]])

    cleaned_data = np.array(cleaned_data)

    return cleaned_data


def parse_args():
    """Get a dictionary of the command line arguments

        Returns:
            args: A dictionary of CLI name -> flag value (e.g. args['all'] -> True)

    """
    ap = argparse.ArgumentParser()

    # a string arg, '--' prefix is arg name, '-' prefix are aliases to same argument
    ap.add_argument("-f", "-file", "--file", default="small.csv",
                    required=False, help="Print all stops for the given route name")

    args = vars(ap.parse_args())

    return args


def main():
    args = parse_args()
    f = args["file"]

    data = pandas_csv(f)
    data = lat_long(data)

    cleaned_data = clean_lat_longs(data)

    scikit_kmeans_graph(cleaned_data, 5)


main()
