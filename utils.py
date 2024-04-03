from shapely.geometry import Point, LineString
import math as m
import numpy as np
import pandas as pd
import os
import shutil
import glob
from sklearn.cluster import DBSCAN
import geopandas as gpd
from typing import Tuple


def save_shp(
        df: pd.DataFrame,
        shp_path: str
        ) -> None:
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs = 4326)
    gdf.to_file(shp_path)


def list_files(
        path: str,
        extension: str
        ) -> list:
    return glob.glob(os.path.join(path, '*{}'.format(extension)))


def clean_folder(
        folder: str
        ) -> None:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def prepare_track_row(
        pixel_row: pd.Series
        ) -> list:
    name = pixel_row['panorama_file_name'] + '.jpg'
    x = pixel_row['x']
    y = pixel_row['y']
    x_norm = (x + 0.5 - 8000. / 2.0) / 8000
    y_norm = (y + 0.5 - 4000. / 2.0) / 8000
    pt_2D = [name, 99999, -1, x_norm, y_norm, -1, 1, 1, 1, -1, -1]

    return pt_2D


def remove_comments(
        line: str
        ) -> str:
    """remove comments from ximilar txt line"""
    return line.split("//")[0].strip('\n')


def create_linestring(
        row: pd.Series
        ) -> LineString:
    return LineString([row['geometry'], Point(row['lon_2'], row['lat_2'])])



def get_theta_phi(
         _x: float,
         _y: float,
         _z: float
         ) -> Tuple[float, float]:
    """Convert cube coordinates into equirectangular based on the side of the cube map"""
    dv = m.sqrt(_x*_x + _y*_y + _z*_z)
    x = _x/dv
    y = _y/dv
    z = _z/dv
    theta = m.atan2(y, x)
    phi = m.asin(z)
    return theta, phi


def map_cube(
        x: int,
        y: int,
        side: int
        ) -> Tuple[int, int]:
    """
    Project a point in cube image coordinates to equirectangular image coordinates
    
    Args:
        x: x-coordinate in the cube projection,
        y: y-coordinate in the cube pprojeciton,
        side: side of the cube map
    """
    # cw  cube width
    # W,H size of equirectangular image
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

    _u = 0.5+0.5*(theta/m.pi)
    _v = 0.5+(phi/m.pi)

    return int(_u*W),  int(_v*H)


def rotation_matrix(
        omega: float,
        phi: float,
        kappa: float
        ) -> np.ndarray:
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


def polar2world(
        lat: float,
        lon: float
        ) -> np.array:
    x = -m.sin(lat) * m.cos(lon)
    y = -m.sin(lat) * m.sin(lon)
    z = m.cos(lat)
    return np.array([x, y, z])


def polar2world_(
        lat: float,
        lon: float
        ) -> np.array:
    y = m.sin(lat) * m.cos(lon)
    x = m.sin(lat) * m.sin(lon)
    z = m.cos(lat)
    return np.array([x, y, z])


def world2polar(
        arr: np.array
        ) -> Tuple[float, float]:
    lat = m.acos(arr[2])
    lon = m.atan2(arr[1], arr[0])
    return lat, lon


def rectification(
        row: pd.Series
        ) -> np.array:
    """Rectify image coordinates of the traffic sign into correct direction."""

    # convert coordinates into radians
    lam = m.radians(row['x'] / 8000 * 360.0)
    phi = m.radians(row['y'] / 4000 * 180.0)

    # rotation angles of panoramic images
    roll = m.radians(row['roll[deg]'])
    pitch = m.radians(row['pitch[deg]'])
    yaw = m.radians(row['heading[deg]'])

    # convert polar coordinates to cartesian
    arr = polar2world(phi, lam)

    # rotate point
    rotation = rotation_matrix(roll, pitch, yaw)
    arr_rec = rotation @ np.transpose(arr)

    # fix orientation and convert back to cartesian coordinates
    phi2, lam2 = world2polar(arr_rec)
    arr_final = polar2world_(phi2, lam2)

    return arr_final


def cluster_dbscan(
        group: gpd.GeoDataFrame
        ) -> None:
    """Pre-cluster observations via spatial clustering algorithm DBSCAN"""
    coords = np.vstack(zip(group['latitude[deg]'], group['longitude[deg]']))
    clusterer = DBSCAN(eps=0.1/6371., min_samples = 2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    group['labels'] = clusterer.labels_


def get_image_range(
        idx: np.array,
        buffer: int,
        list_len: int
        ) -> range:
    min_img_code = min(idx)
    max_img_code = max(idx)
    if min_img_code >= buffer:
        if (list_len - max_img_code) >= buffer:
            return range(min_img_code - buffer, max_img_code + buffer + 1, 1)
        elif (list_len - max_img_code) < buffer:
            return range(min_img_code - buffer, max_img_code + 1, 1)
    elif min_img_code < buffer:
        if (list_len - max_img_code) >= buffer:
            return range(min_img_code, max_img_code + buffer + 1, 1)
        elif (list_len - max_img_code) < buffer:
            return range(min_img_code, max_img_code + 1, 1)