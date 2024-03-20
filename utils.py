from shapely.geometry import Point, LineString
import math as m
import numpy as np



def remove_comments(line):
    return line.split("//")[0].strip('\n')

def create_linestring(row):
    return LineString([row['geometry'], Point(row['lon_2'], row['lat_2'])])

def calculate_angle(gdf):
    car_o = np.deg2rad(gdf['heading[deg]'])
    pix_o = np.deg2rad(gdf['x'] / 8000 * 360)
    gdf['angle'] = (car_o + pix_o) % (2*m.pi)

def get_theta_phi( _x, _y, _z):
    dv = m.sqrt(_x*_x + _y*_y + _z*_z)
    x = _x/dv
    y = _y/dv
    z = _z/dv
    theta = m.atan2(y, x)
    phi = m.asin(z)
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

    _u = 0.5+0.5*(theta/m.pi)
    _v = 0.5+(phi/m.pi)
    return int(_u*W),  int(_v*H)

def calculate_angle(gdf):
    car_o = np.deg2rad(gdf['heading[deg]'])
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