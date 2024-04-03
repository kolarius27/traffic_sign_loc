#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import argparse
import json
import time
from os.path import join, isfile
import csv
import matplotlib.pyplot as plt
import contextily as cx

from utils import *



#PATH TO OpenSfM: ADD ABSOLUTE PATH TO THE INSTALLATION FOLDER OF OpenSfM LIBRARY.

opensfm_path = r'C:\Users\micha\Desktop\Gekon\znacky\OpenSfM'


#PATH TO THE DATASET

dataset_path = r'C:\Users\micha\Desktop\Gekon\znacky\OpenSfM\data\test_pano4'
clean_folder(dataset_path)


def prepare_exif_file(
        trajectory_df: pd.DataFrame, 
        dataset_path: str
        ) -> None:
    """Convert trajectory csv into json with EXIF info of images"""
    json_data = {}

    # iterate through rows of trajectory DataFrame and add it into json
    for _, row in trajectory_df.iterrows():
        panorama_name = row['panorama_file_name'] + '.jpg'
        latitude = row['latitude[deg]']
        longitude = row['longitude[deg]']
        altitude = row['altitude_ellipsoidal[m]']
        
        json_data[panorama_name] = {
            'gps': {
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude
            },
            'projection_type': 'equirectangular',
            'width': 8000,
            'height': 4000,
            'camera': 'v2 point grey research ladybug 8000 4000 equirectangular 0.0'
        }

    # Save JSON to a file
    with open(os.path.join(dataset_path, 'exif_overrides.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def add_images(
        pois: gpd.GeoDataFrame,
        panos_path: str,
        buffer: int
        ) -> None:
    """Find images and copy them into temp folder."""
    # Create images folder in the temp folder
    images_folder = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    # Array of panoramic images with observations
    images = np.array([os.path.join(panos_path, file + '.jpg') for file in pois['panorama_file_name']])

    # Array of all images from folder with panoramic images
    pano_files = np.array(list_files(panos_path, '.jpg'))

    # get indexes from array of all images of images with observations
    idx = np.where(np.isin(pano_files, images))[0]
    
    # get range of indexes with images for reconstruction
    id_range = get_image_range(idx, buffer, len(pano_files))
    print(id_range)

    # copy those images into temp folder
    for id in id_range:
        file_path = pano_files[id]
        shutil.copyfile(file_path, os.path.join(images_folder, os.path.split(file_path)[-1]))


def compute_coords(pois, trajectory_df):
    
    if os.path.exists(os.path.join(dataset_path, 'exif_overrides.json')) is False:
        prepare_exif_file(trajectory_df, dataset_path)

    # paths to bat files with opensfm steps
    bat_prepare = os.path.join(opensfm_path, 'bin\opensfm_prepare.bat')
    bat_reconstruct = os.path.join(opensfm_path, 'bin\opensfm_reconstruct.bat') 
    
    # run preparation of tracks
    subprocess.run([bat_prepare, dataset_path])
    print('Tracks prepared')

    # add tracks of traffic sign into tracks.csv
    tracks_path = os.path.join(dataset_path, 'tracks.csv')
    with open(tracks_path, 'a', newline='') as csvfile:
        tracks = csv.writer(csvfile, delimiter='\t')
        for _, row in pois.iterrows():
            track_row = prepare_track_row(row)
            tracks.writerow(track_row)

    # run reconstruction process
    subprocess.run([bat_reconstruct, dataset_path])

    # load reconstruction file with geo-coordinates
    reconstruction_path = os.path.join(dataset_path, 'reconstruction.geocoords.json')
    with open(reconstruction_path, 'r') as reconstruction:
        reconstruction_json = json.loads(reconstruction.read())

        # search for geo-coordinate of the traffic sign
        tsign_coords = reconstruction_json[0]['points']['99999']['coordinates']
        print(tsign_coords)
    clean_folder(dataset_path)
    return tsign_coords


def ximilar_to_df(
        ximilar_path: str, 
        panos_path: str
        ) -> pd.DataFrame:
    """
    Converting info from ximilar txts into df

    Args:
        ximilar_path: a path to a folder with ximilar observation txts,
        panos_path: a path to panoramic images

    Returns: 
        pd.Dataframe of observations linked to panoramic images
    """
    # List to store extracted data
    data = []

    # Iterate over each file in the folder
    for filename in os.listdir(ximilar_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(ximilar_path, filename)
            codes = filename[:-4].split('_')
            # convert a txt observation into a row in the Dataframe
            with open(file_path, "r") as file:
                try:
                    # load txt as a json
                    json_lines = [remove_comments(line) for line in file.readlines()]
                    json_str = "\n".join(json_lines)
                    json_data = json.loads(json_str)

                    # convert cube image coordinates into equirectangular image coordinates
                    x, y = map_cube(json_data["points"][0]["point"][0], json_data["points"][0]["point"][1], codes[-2])

                    # save all relevant info into a dict
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
    
    # Search for panoramic images
    pano_files = os.listdir(panos_path)

    # Switch names of split panorama to full panorama
    filename_dict = {file_name[-7:-4]: file_name[:-4] for file_name in pano_files}
    df['panorama_file_name'] = df['panorama_file_name'].map(filename_dict.get)

    return df


def find_pois(
        ximilar_df: pd.DataFrame,
        trajectory_df: pd.DataFrame,
        panos_path: str,
        plot_arg: bool
        ) -> pd.DataFrame:
    """
    Hello World

    Args:
        ximilar_df: Dataframe of ximilar observations,
        trajectory_df: Dataframe with trajectory,
        panos_path: A path to panoramic images,
        plot_arg: Boolean value, if true, plot situation

    Returns:
        pd.DataFrame with traffic signs, pd.DataFrame with failed geolocation
    """
    # initialization of dataframes
    tsign_df = pd.DataFrame(columns = ['kod', 'pois', 'geometry'])
    failed_df = pd.DataFrame()

    # merge ximilar dataframe and trajectory dataframe
    merged_df = pd.merge(ximilar_df, trajectory_df, on='panorama_file_name', how='inner')
    merged_df.drop(columns=['filename', 'gps_seconds[s]'], inplace=True)

    # convert Dataframe into GeoDataFrame
    geometry = [Point(xy) for xy in zip(merged_df['longitude[deg]'], merged_df['latitude[deg]'])] 
    gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=4326)

    # Pre-cluster GeoDataFrame by traffic sign type
    grouped_gdf = gdf.groupby('traffic_sign')
    for sign, group in grouped_gdf:
        print(group)
        # Pre-cluster via DBSCAN
        cluster_dbscan(group)
        # Add direction vectors for every observation
        add_directions(group)
        # Check if direction vectors intersect between each other and finish clustering
        failed_idx = get_intersection(group)
        failed_obs = group[group.index.isin(failed_idx)]
        print(failed_obs)
        failed_df = pd.concat([failed_df, failed_obs])
        # Create final clusters with observations of one traffic sign
        pois_gdf = group.groupby('labels')
        # iterate through clusters
        for id, pois in pois_gdf:
            if id != -1:
                print(sign, id)
                print(pois)
                # Plotting the situation
                if plot_arg is True:
                    plot_situation(sign, pois)
                # Drop columns
                pois.drop(columns=['latitude[deg]', 'longitude[deg]', 'lon_2', 'lat_2', 'altitude_ellipsoidal[m]', 'geometry'], inplace=True)
                # SFM framework
                sfm(sign, id,  pois, trajectory_df, panos_path, tsign_df, buffer=0)
                print(tsign_df)
    return tsign_df, failed_df


def sfm(
        sign: str,
        id: int,
        pois: gpd.GeoDataFrame,
        trajectory_df: pd.DataFrame,
        panos_path: str,
        tsign_df: pd.DataFrame,
        buffer: int
        ) -> None:
    """
    SFM framework using OpenSfM library. 
    """
    # Copy images and into temp file and create exif_override.json
    add_images(pois, panos_path, buffer)

    # Compute WGS-84 coordinates of the traffic sign
    tsign_coords = compute_coords(pois, trajectory_df)

    # Add new line into final GeoDataFrame
    tsign_df.loc[len(tsign_df.index)] = [sign, '{}_{}'.format(sign, id), Point(tsign_coords)]


def add_directions(
        group: gpd.GeoDataFrame
        ) -> None:
    """Add directions of observations into pre-clustered geodataframes"""
    gdf_copy = group.copy()
    lon_2 = []
    lat_2 = []
    for _, row in gdf_copy.iterrows():
        # rectify panoramic image and calculate second point of direction vector
        arr = rectification(row)
        lon_2.append(row['longitude[deg]'] + arr[0]/500)
        lat_2.append(row['latitude[deg]'] + arr[1]/500)
    group['lon_2'] = lon_2
    group['lat_2'] = lat_2

    # drop columns
    group.drop(columns=['roll[deg]', 'pitch[deg]', 'heading[deg]'], inplace=True)


def get_intersection(
        group: gpd.GeoDataFrame
        ) -> None:
    """Create LineStrings from direction vectors and check if the LineStrings intersect. Then, split the pre-clusters into final clusters."""
    # Load pre-clusters
    cluster_gdf = group.groupby('labels')
    cluster_list = []

    # iterate through pre-clusters
    for _, cluster in cluster_gdf:
        # get original indexes of clusters
        idxs = cluster.index.tolist()
        intersections = []
        # check intersection of subsequent observations, if observations intersect, return True 
        for i in range(len(idxs) - 1):

            id1 = idxs[i]
            id2 = idxs[i+1]
            row1 = cluster.loc[id1]
            row2 = cluster.loc[id2]
            line1 = create_linestring(row1)
            line2 = create_linestring(row2)
            intersection = line1.intersects(line2)

            # if observation is from the same panorama, return False
            if row1['panorama_file_name'] == row2['panorama_file_name']:
                intersection = False
            intersections.append(intersection)

        # Find subsequent intersectioning observations
        inter_arr = np.array(intersections)
        cluster_list.extend(cluster_by_intersection(idxs, inter_arr))

    # prepare cluster ids and save them to the 'labels' column
    cluster_ids, failed_ids = get_cluster_ids(cluster_list)
    print('cluster ids', cluster_ids)
    print('failed ids', failed_ids)
    group['labels'] = cluster_ids
    return failed_ids
                               

def cluster_by_intersection(
        idxs: np.array, 
        intersections: np.array
        ) -> np.array:
    """Find borders between clusters"""   
    try:
        split_indices = np.where(~intersections)[0] +1
        return np.split(idxs, split_indices)
    except TypeError:
        return idxs
    

def get_cluster_ids(
        cluster_list: list
        ) -> np.array:
    """Prepare ids of clusters"""
    print(cluster_list)
    cluster_ids = []
    failed_ids = []
    valid_clusters = 0
    for cluster in cluster_list:   
        if type(cluster) is int:
            cluster_ids.append(-1)
            failed_ids.append(cluster)
        elif len(cluster) > 1 :
            cluster_ids.extend([valid_clusters] * len(cluster))
            valid_clusters += 1
        else:
            cluster_ids.append(-1)
            failed_ids.append(cluster[0])
    return np.array(cluster_ids), np.array(failed_ids)


def plot_situation(
        sign: str,
        group: gpd.GeoDataFrame
        ) -> None:
    """Plotting the situation"""
    ax = group.plot(column='labels', figsize=(10,10), colormap='viridis')
    for row in zip(group['longitude[deg]'], group['latitude[deg]'], group['lon_2'], group['lat_2']):
        plt.axline(row[:2], row[2:4], c='red')
    plt.axis('equal')
    cx.add_basemap(ax, crs=group.crs)
    plt.title('{}, {}'.format(sign, np.unique(group['labels'])))
    plt.show()


def main():
    # Get arguments
    parser = argparse.ArgumentParser(prog='PanoLoc', description='Aplikace na výpočet geolokace z panoramatických snímků')
    parser.add_argument('ximilar', help='Cesta ke složce, kde jsou uložené txt soubory od Ximilaru')
    parser.add_argument('trajektorie', help='Cesta k csv souboru s trajektorií snímání')
    parser.add_argument('fotky', help='Cesta ke složce s panoramatickými snímky')
    parser.add_argument('-p', '--plot', help='Grafické znázornění situace', action='store_true')
    args = parser.parse_args()

    # Convert ximilar observations to pandas dataframe
    ximilar_df = ximilar_to_df(args.ximilar, args.fotky)
    # Read trajectory csv
    trajectory_df = pd.read_csv(args.trajektorie, delimiter=';')
    # Main pipeline
    tsign_df, failed_df = find_pois(ximilar_df, trajectory_df, args.fotky, args.plot)
    # Save DataFrame as a shapefile
    save_shp(tsign_df, 'test_full.shp')
    failed_df.to_csv('failed.csv')


    

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    runtime = end - start
    print('Runtime: ', runtime)