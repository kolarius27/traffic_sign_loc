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

"""
PATH TO THE ximilar folder
"""
ximilar_path = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\ximilar_detect'


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
    return _u*W,  _v*H

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
                    
                    extracted_info = {
                        "filename": filename[:-4],
                        "pano_code": int(codes[-3]),
                        "cube": int(codes[-2]),
                        "pole": int(codes[-1]),
                        "x_c": json_data["points"][0]["point"][0],  # Extract x-coordinate from 'pole bottom'
                        "y_c": json_data["points"][0]["point"][1],  # Extract y-coordinate from 'pole bottom'
                        "traffic_sign": [sign["traffic sign code"] for sign in json_data["traffic signs"]]  # Extract traffic sign codes
                    }
                    data.append(extracted_info)
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)

    # Display the DataFrame
    return df


def remove_comments(line):
    return line.split("//")[0].strip('\n')

def add_panofile(df):
    pass

def main():
    df = ximilar_to_df()
    

if __name__ == '__main__':
    start = time.time()

    ximilar_to_df()

    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)