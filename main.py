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
PATH TO DATASET
"""
dataset_path = r'C:\Users\micha\Desktop\Gekon\znacky\OpenSfM\data\test_pano2'


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
def map_cube(x, y, side, cw, W, H):

    u = 2*(float(x)/cw - 0.5)
    v = 2*(float(y)/cw - 0.5)

    if side == "front":
        theta, phi = get_theta_phi( 1, u, v )
    elif side == "right":
        theta, phi = get_theta_phi( -u, 1, v )
    elif side == "left":
        theta, phi = get_theta_phi( u, -1, v )
    elif side == "back":
        theta, phi = get_theta_phi( -1, -u, v )
    elif side == "bottom":
        theta, phi = get_theta_phi( -v, u, 1 )
    elif side == "top":
        theta, phi = get_theta_phi( v, u, -1 )

    _u = 0.5+0.5*(theta/math.pi)
    _v = 0.5+(phi/math.pi)
    return _u*W,  _v*H


if __name__ == '__main__':
    start = time.time()

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

    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)