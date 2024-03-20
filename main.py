#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
import pandas as pd
import numpy as np
import json
import time
from os.path import join, isfile
import csv
from sklearn.cluster import MeanShift,DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
import contextily as cx

from utils import *

"""
PATH TO OpenSfM: ADD ABSOLUTE PATH TO THE INSTALLATION FOLDER OF OpenSfM LIBRARY.
"""
opensfm_path = opensfm_path = r'C:\Users\micha\Desktop\Gekon\znacky\OpenSfM'

"""
PATH TO THE DATASET
"""
dataset_path = r'C:\Users\micha\Desktop\Gekon\znacky\OpenSfM\data\test_pano3'

"""
PATH TO THE TRAJECTORY
"""
trajectory_path = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\original_panorama_Bechovice\Praha21Bechexp_panorama.csv'
trajectory_df = pd.read_csv(trajectory_path, sep=';')

"""
PATH TO THE ximilar folder
"""
ximilar_path = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\ximilar_detect'

"""
PATH TO THE panoramas
"""
panos_path = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\original_panorama_Bechovice'


def prepare_exif_file(trajectory, dataset_path):
    df = pd.read_csv(trajectory, delimiter=';')
    print(df)
    json_data = {}

    for index, row in df.iterrows():
        panorama_name = row['panorama_file_name'] + '.jpg'
        latitude = row['latitude[deg]']
        longitude = row['longitude[deg]']
        altitude = row['altitude_ellipsoidal[m]']
        
        json_data[panorama_name] = {
            'gps': {
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude
            }
        }

    # Save JSON to a file
    with open(os.path.join(dataset_path, 'exif_overrides.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def prepare_track_row(pixel_row):
    name = pixel_row['pano_ID'] + '.jpg'
    x = pixel_row['x']
    y = pixel_row['y']
    x_norm = (x + 0.5 - 8000. / 2.0) / 8000
    y_norm = (y + 0.5 - 4000. / 2.0) / 8000
    pt_2D = [name, 99999, -1, x_norm, y_norm, -1, 1, 1, 1, -1, -1]

    return pt_2D


def append_track_row(row, tracks):
    print(tracks.columns)
    a_series = pd.Series(row, index=tracks.columns)
    tracks.append(a_series, ignore_index=True)



def compute_coords():
    if os.path.exists(os.path.join(dataset_path, 'exif_overrides.json')) is False:
        prepare_exif_file(trajectory_path, dataset_path)

    # paths to bat files with opensfm steps
    bat_prepare = os.path.join(opensfm_path, 'bin\opensfm_prepare.bat')
    bat_reconstruct = os.path.join(opensfm_path, 'bin\opensfm_reconstruct.bat')
    
    # prepare list of images
    img_path = os.path.join(dataset_path, 'images')
    img_files = [f for f in os.listdir(img_path) if isfile(join(img_path, f))]

    # load image coords of traffic sign
    pixel_coords_path = os.path.join(dataset_path, 'test_pano.csv')
    pixel_coords = pd.read_csv(pixel_coords_path, delimiter=';')
    pixel_coords.apply(pd.to_numeric, errors='coerce').fillna(pixel_coords)
    
    # run preparation of tracks
    subprocess.run([bat_prepare, dataset_path])
    print('Tracks prepared')

    # add tracks of traffic sign into tracks.csv
    tracks_path = os.path.join(dataset_path, 'tracks.csv')
    with open(tracks_path, 'a', newline='') as csvfile:
        tracks = csv.writer(csvfile, delimiter='\t')
        for i in range(len(pixel_coords)):
            p_row = pixel_coords.iloc[i]
            track_row = prepare_track_row(p_row)
            tracks.writerow(track_row)

    # run reconstruction process
    subprocess.run([bat_reconstruct, dataset_path])

    # load reconstruction file with geo-coordinates
    reconstruction_path = os.path.join(dataset_path, 'reconstruction.geocoords.json')
    reconstruction = open(reconstruction_path)
    reconstruction_json = json.loads(reconstruction.read())

    # search for geo-coordinate of the traffic sign
    tsign_coords = reconstruction_json[0]['points']['99999']['coordinates']
    print(tsign_coords)

    reconstruction.close()


def ximilar_to_df():
    """
    Converting info from ximilar txts into df
    """
    # List to store extracted data
    data = []

    # Iterate over each file in the folder
    for filename in os.listdir(ximilar_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(ximilar_path, filename)
            codes = filename[:-4].split('_')
            with open(file_path, "r") as file:
                #s = file.read()
                #print(s)
                #s = s.replace("\'", "\"")
                #print(file)
                #print(file.readlines())
                try:
                    json_lines = [remove_comments(line) for line in file.readlines()]
                    #print(json_lines)
                    json_str = "\n".join(json_lines)
                    json_data = json.loads(json_str)
                    #print(json_data)

                    x, y = map_cube(json_data["points"][0]["point"][0], json_data["points"][0]["point"][1], codes[-2])
                    #print(json_data["traffic signs"])
                    #print([sign["traffic sign code"] for sign in json_data["traffic signs"]])
                    extracted_info = {
                        "filename": filename[:-4],
                        "panorama_file_name": codes[-3],
                        "x": x,  # Extract x-coordinate from 'pole bottom'
                        "y": y,  # Extract y-coordinate from 'pole bottom'
                        "traffic_sign": ', '.join([sign["traffic sign code"] for sign in json_data["traffic signs"]])  # Extract traffic sign codes
                    }
                    data.append(extracted_info)
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)
    
    # Edit the dataframe
    pano_files = os.listdir(panos_path)
    # print(pano_files)
    filename_dict = {file_name[-7:-4]: file_name[:-4] for file_name in pano_files}
    df['panorama_file_name'] = df['panorama_file_name'].map(filename_dict.get)
    #df.drop('pano_code', axis=1)

    #print(df)
    
    return df


def find_pois(df):
    # 
    merged_df = pd.merge(df, trajectory_df, on='panorama_file_name', how='inner')
    geometry = [Point(xy) for xy in zip(merged_df['longitude[deg]'], merged_df['latitude[deg]'])] 
    gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=4326)
    grouped_gdf = gdf.groupby('traffic_sign')
    for sign, group in grouped_gdf:
        print(sign)
        cluster_dbscan(group)
        get_intersection(group)
        plot_situation(sign, group)
        

def cluster_dbscan(group):
    coords = np.vstack(zip(group['latitude[deg]'], group['longitude[deg]']))
    # print(coords)
    clusterer = DBSCAN(eps=0.0005).fit(coords)
    group['labels'] = clusterer.labels_
    gdf_copy = group.copy()
    lon_2 = []
    lat_2 = []
    for i, row in gdf_copy.iterrows():
        #print(i)
        arr = rectification(row)
        lon_2.append(row['longitude[deg]'] + arr[0]/1000)
        lat_2.append(row['latitude[deg]'] + arr[1]/1000)
    group['lon_2'] = lon_2
    group['lat_2'] = lat_2


    #g = sns.scatterplot(x='longitude[deg]', y='latitude[deg]', hue='labels', data=group)
    #plt.show()


def get_intersection(group_gdf):
    #print(group_gdf)
    cluster_gdf = group_gdf.groupby('labels')
    cluster_list = []
    for sign, cluster in cluster_gdf:
        print(sign)        
        idxs = cluster.index.tolist()
        intersections = []
        for i in range(len(idxs) - 1):
            id1 = idxs[i]
            id2 = idxs[i+1]
            row1 = cluster.loc[id1]
            row2 = cluster.loc[id2]
            line1 = create_linestring(row1)
            line2 = create_linestring(row2)
            intersection = line1.intersects(line2)
            if row1['panorama_file_name'] == row2['panorama_file_name']:
                intersection = False
            intersections.append(intersection)

        inter_arr = np.array(intersections)
        print(inter_arr)
        cluster_list.extend(cluster_by_intersection(idxs, inter_arr))
    print(cluster_list)
    cluster_ids = get_cluster_ids(cluster_list)
    print(cluster_ids)
    print(group_gdf['labels'])
    group_gdf['labels'] = cluster_ids
    group_gdf.sort('index')
    print(group_gdf['labels'])
                               




def cluster_by_intersection(idxs, intersections):
    split_indices = np.where(~intersections)[0] +1
    return np.split(idxs, split_indices)
    

def get_cluster_ids(cluster_list):
    cluster_ids = []
    valid_clusters = 0
    for cluster in cluster_list:
        if len(cluster) > 1:
            cluster_ids.extend([valid_clusters] * len(cluster))
            valid_clusters += 1
        else:
            cluster_ids.append(-1)
    return np.array(cluster_ids)


def plot_situation(sign, group):
    ax = group.plot(column='labels', figsize=(10,10), colormap='viridis')
    for row in zip(group['longitude[deg]'], group['latitude[deg]'], group['lon_2'], group['lat_2']):
        # plt.axline(row[:2], slope=np.tan(row[4]), c='blue')
        plt.axline(row[:2], row[2:4], c='red')
    plt.axis('equal')
    cx.add_basemap(ax, crs=group.crs)
    plt.title('{}, {}'.format(sign, np.unique(group['labels'])))

def main():
    df = ximilar_to_df()
    find_pois(df)
    

if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)