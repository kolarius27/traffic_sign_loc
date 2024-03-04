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


if __name__ == '__main__':
    start = time.time()

    bat_prepare = os.path.join(opensfm_path, 'bin\opensfm_prepare.bat')
    bat_reconstruct = os.path.join(opensfm_path, 'bin\opensfm_reconstruct.bat')

    img_path = os.path.join(dataset_path, 'images')
    img_files = [f for f in os.listdir(img_path) if isfile(join(img_path, f))]

    pixel_coords_path = os.path.join(dataset_path, 'test_pano.csv')
    pixel_coords = pd.read_csv(pixel_coords_path, delimiter=';')
    pixel_coords.apply(pd.to_numeric, errors='coerce').fillna(pixel_coords)
    print(dataset_path)

    subprocess.run([bat_prepare, dataset_path])
    print('Tracks prepared')

    tracks_path = os.path.join(dataset_path, 'tracks.csv')
    with open(tracks_path, 'a', newline='') as csvfile:
        tracks = csv.writer(csvfile, delimiter='\t')
        for i in range(len(pixel_coords)):
            p_row = pixel_coords.iloc[i]
            track_row = prepare_track_row(p_row)
            tracks.writerow(track_row)

    subprocess.run([bat_reconstruct, dataset_path])

    reconstruction_path = os.path.join(dataset_path, 'reconstruction.geocoords.json')
    reconstruction = open(reconstruction_path)
    reconstruction_json = json.loads(reconstruction.read())


    tsign_coords = reconstruction_json
    print(reconstruction_json[0]['points']['99999']['coordinates'])

    reconstruction.close()

    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)