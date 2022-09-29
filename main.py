# -*- coding: utf-8 -*-
"""
The script is used to generate simulated sensor scan with given trajectory, map and model of sensor.

@author: 123
"""
import open3d as o3d
import numpy as np
import math
import open3d as o3d
import numpy as np
import glob
import time
from bitarray import bitarray
from bitarray import util
from bitarray.util import ba2int, zeros
from octree_hero import *
import os
import icp

def get_direction(array_ray):  # get the unit vector of a ray in sensor system
    x = 1 * math.cos(array_ray[1] * math.pi / 180) * math.cos(array_ray[0] * math.pi / 180)
    y = 1 * math.cos(array_ray[1] * math.pi / 180) * math.sin(array_ray[0] * math.pi / 180)
    z = 1 * math.sin(array_ray[1] * math.pi / 180)
    unit_vector = np.asarray([x, y, z])
    return unit_vector  # Sensor system right direction of car is x, heading direction is y, z.


def transform_in_utm(sensor_pose, coordinate_in_sensor):  # sensor_pose=[East,North,Up,Roll,Pitch,Yall]
    rotation = np.asarray(sensor_pose[3:] * math.pi / 180)
    Rotationmatrix = o3d.geometry.get_rotation_matrix_from_yxz(rotation)
    world_coordinate = np.dot(Rotationmatrix, coordinate_in_sensor)
    return world_coordinate

def sensor(reference_trajectory,noise = False):
    sensor_orientation_array = reference_trajectory[:,5:]
    sensor_position_array = reference_trajectory[:,2:5]
    if noise == True:
        sensor_orientation_array = reference_trajectory[:,3:]
        sensor_position_array = reference_trajectory[:,:3]
    min_horizental = 0
    max_horizental = 360
    min_vertical = -15
    max_vertical = 15
    h_stepsize = 0.2
    layers = 16
    distance = 100
    horizental_array = np.arange(min_horizental, max_horizental, h_stepsize)
    vertical_array = np.linspace(min_vertical, max_vertical, layers)
    array_rays = np.array([[1, 2]])
    for i in np.nditer(horizental_array):
        i = np.float32(i)
        for j in np.nditer(vertical_array):
            j = np.float32(j)
            array_rays = np.append(array_rays, [[i, j]], axis=0)
    array_rays = np.delete(array_rays, 0, axis=0)

    x = 1 * np.cos(array_rays[:, 1] * math.pi / 180) * np.cos(array_rays[:, 0] * math.pi / 180)
    y = 1 * np.cos(array_rays[:, 1] * math.pi / 180) * np.sin(array_rays[:, 0] * math.pi / 180)
    z = 1 * np.sin(array_rays[:, 1] * math.pi / 180)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    z = z[:, np.newaxis]

    unit_vector = np.concatenate((x, y), 1)
    unit_vector = np.concatenate((unit_vector, z), 1)

    rotation = np.asarray(sensor_orientation_array * math.pi / 180)
    Rotationmatrix = [o3d.geometry.get_rotation_matrix_from_yxz(rot) for rot in rotation]
    Rotationmatrix = np.array(Rotationmatrix)
    world_coordinate = np.dot(Rotationmatrix, unit_vector.T)
    world_coordinate = world_coordinate.transpose((0, 2, 1))  #让向量坐标从列向量编程行向量

    return sensor_position_array, world_coordinate

def transform_in_vehicle(intersected_array,pose_without_noise):
    rotation = pose_without_noise[3:]
    translation = pose_without_noise[:3]
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_yxz(rotation)
    intersected_array_vehicle = np.transpose(np.dot(np.linalg.inv(rotation_matrix),(intersected_array - translation).T))
    return intersected_array_vehicle

def add_noise(noised_position,position,simulated_point):
    #noise np.ndarray((250,6),np.float)
    translation = noised_position[:3]
    rotation = noised_position[3:]
    simulated_point_vehicle = transform_in_vehicle(simulated_point,position[2:])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_yxz(rotation)
    simulated_point_noise_imu = np.dot(rotation_matrix,simulated_point_vehicle.T).T + translation
    return simulated_point_noise_imu



if __name__ == "__main__":
    map_="005"  #Merged_122831_122944_123114
    pcd=o3d.io.read_point_cloud('/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/'+map_+'.ply')
    reference_trajectory = np.loadtxt('/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_VelodyneOrigin_noinv.txt', delimiter=",")
    position_noise = np.random.uniform(-0.5,0.5,(250,3))
    orientation_noise_roll = np.random.uniform(0,0.25,(250,1))
    orientation_noise_yaw = np.random.uniform(0,0.1,(250,1))
    orientation_noise = np.concatenate((orientation_noise_roll,np.concatenate((np.zeros((250,1)),orientation_noise_yaw),axis=1)),axis = 1)
    noise = np.concatenate((position_noise,orientation_noise),axis=1)
    reference_trajectory_noised = reference_trajectory[:250,2:8] + noise
    np.savetxt('/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_noised.txt',reference_trajectory_noised,fmt='%.14f',delimiter=",")


    save_path = "/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/simulated_different_noise/"
    if os.path.isdir(save_path)==False:
       os.mkdir(save_path)
    start_tree=time.time()
    root=OctreeNode(0,0,pcd,13,True)
    set_octree(root)
    end_tree=time.time()
    sensor_position_array,world_coordinate = sensor(reference_trajectory,False)
    print("time of build a tree:", end_tree-start_tree,'s')
    start=0
    narray=np.asarray
    for position in range(0,len(sensor_position_array)):
        # sensor_start=time.time()
        sensor_position=sensor_position_array[position]
        unit_vector_world=world_coordinate[position]
        intersected_array=loop_ray(unit_vector_world,sensor_position,root)
        simulated_noised_point = add_noise(reference_trajectory_noised[position],reference_trajectory[position],intersected_array)
        np.savetxt(save_path+str(start).zfill(4)+"_simulation.txt",intersected_array,fmt='%.14f',delimiter=",")
        np.savetxt(save_path+str(start).zfill(4)+"_simulation_noised.txt",simulated_noised_point,fmt='%.14f',delimiter=",")
        start += 1
        print(start," ","success")