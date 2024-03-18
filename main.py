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
import math
from sklearn.cluster import MeanShift,DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import geopandas as gpd
import contextily as cx
import math as m

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

def get_theta_phi( _x, _y, _z):
    dv = math.sqrt(_x*_x + _y*_y + _z*_z)
    x = _x/dv
    y = _y/dv
    z = _z/dv
    theta = math.atan2(y, x)
    phi = math.asin(z)
    return theta, phi


# x,y position in cubemap
# cw  cube width
# W,H size of equirectangular image
def map_cube(x, y, side):
    cw = 2033
    W = 8000
    H = 4000

    u = 2*(float(x)/cw - 0.5)
    v = 2*(float(y)/cw - 0.5)

    if side == "0":
        theta, phi = get_theta_phi( 1, u, v )
    elif side == "1":
        theta, phi = get_theta_phi( -u, 1, v )
    elif side == "3":
        theta, phi = get_theta_phi( u, -1, v )
    elif side == "2":
        theta, phi = get_theta_phi( -1, -u, v )
    elif side == "5":
        theta, phi = get_theta_phi( -v, u, 1 )
    elif side == "4":
        theta, phi = get_theta_phi( v, u, -1 )

    _u = 0.5+0.5*(theta/math.pi)
    _v = 0.5+(phi/math.pi)
    return int(_u*W),  int(_v*H)

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


def remove_comments(line):
    return line.split("//")[0].strip('\n')



def find_pois(df):
    # 
    merged_df = pd.merge(df, trajectory_df, on='panorama_file_name', how='inner')
    grouped_df = merged_df.groupby('traffic_sign')
    for sign, group in grouped_df:
        cluster_meanshift(sign, group)
        

def cluster_meanshift(sign, group):
    coords = np.vstack(zip(group['latitude[deg]'], group['longitude[deg]']))
    geometry = [Point(xy) for xy in zip(group['longitude[deg]'], group['latitude[deg]'])] 
    gdf = gpd.GeoDataFrame(group, geometry=geometry, crs=4326)
    # print(coords)
    clusterer = DBSCAN(eps=0.0005).fit(coords)
    gdf['labels'] = clusterer.labels_
    # calculate_angle(gdf)
    # (gdf['angle'])
    ax = gdf.plot(column='labels', figsize=(10,10), colormap='viridis')
    for row in zip(gdf['longitude[deg]'], gdf['latitude[deg]'], gdf['angle']):
        plt.axline(row[:2], slope=np.tan(row[2]))
    plt.axis('equal')
    cx.add_basemap(ax, crs=gdf.crs)
    plt.title('{}, {}'.format(sign, np.unique(gdf['labels'])))

    #g = sns.scatterplot(x='longitude[deg]', y='latitude[deg]', hue='labels', data=group)
    #plt.show()

def calculate_angle(gdf):
    car_o = np.deg2rad(gdf['heading[deg]'] + 180)
    pix_o = np.deg2rad(gdf['x'] / 8000 * 360)
    gdf['angle'] = (car_o + pix_o) % (2*m.pi)


def rotation_matrix(omega, phi, kappa):
    R_x = np.array([[1, 0, 0],
                    [0, m.cos(omega), m.sin(omega)],
                    [0, -m.sin(omega), m.cos(omega)]
                    ])

    R_y = np.array([[m.cos(phi), 0, -m.sin(phi)],
                    [0, 1, 0],
                    [m.sin(phi), 0, m.cos(phi)]
                    ])

    R_z = np.array([[m.cos(kappa), -m.sin(kappa), 0],
                    [m.sin(kappa), m.cos(kappa), 0],
                    [0, 0, 1]
                    ])

    R = R_z @ R_y @ R_x
    return R


def polar2world(lat, lon):
    x = -m.sin(lat) * m.cos(lon)
    y = -m.sin(lat) * m.sin(lon)
    z = m.cos(lat)
    return np.array([x, y, z])


def polar2world_(lat, lon):
    y = m.sin(lat) * m.cos(lon)
    x = m.sin(lat) * m.sin(lon)
    z = m.cos(lat)
    return np.array([x, y, z])


def world2polar(arr):
    lat = m.acos(arr[2])
    lon = m.atan2(arr[1], arr[0])
    #lat = pi/2 - lat
    return lat, lon


def rectification(row):

    lam = m.radians(row['x'] / 8000 * 360.0)
    phi = m.radians(row['y'] / 4000 * 180.0)

    # rotation angles of panoramic images
    roll = m.radians(row['roll[deg]'])
    pitch = m.radians(row['pitch[deg]'])
    yaw = m.radians(row['heading[deg]'])
    #roll = 0.0
    #pitch = 0.0

    arr = polar2world(phi, lam)

    rotation = rotation_matrix(roll, pitch, yaw)

    arr_rec = rotation @ np.transpose(arr)

    phi2, lam2 = world2polar(arr_rec)

    arr_final = polar2world_(phi2, lam2)

    return arr_final


def main():
    df = ximilar_to_df()
    find_pois(df)
    

if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)