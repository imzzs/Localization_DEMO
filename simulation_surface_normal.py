import numpy as np

from icp import *

def show_in_o3d(array_target_normal):
    vector3d = o3d.utility.Vector3dVector(array_target_normal[:,:3])
    pcd = o3d.geometry.PointCloud(vector3d)
    vector3d_normal = o3d.utility.Vector3dVector(array_target_normal[:,3:])
    pcd.normals = vector3d_normal
    o3d.visualization.draw_geometries([pcd],point_show_normal = True)

trajectory = np.loadtxt(
    '/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_VelodyneOrigin_noinv.txt',
    delimiter=",")  # np.array.shape = (N,8)
trajectory_noised = np.loadtxt(
    '/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_noised.txt',
    delimiter=",")  # np.array.shape = (N,6)

positions = 5
iteration_times = 5
E_static = np.zeros((iteration_times, positions))
D_static = np.zeros((iteration_times,positions))
for i in range(positions):
    R_mean = np.eye(3, 3)
    T_mean = np.zeros((3,))
    target = np.loadtxt(
        "/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/simulated_different_noise/000" + str(
            i) + "_simulation.txt", delimiter=",")
    source = np.loadtxt(
        "/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/simulated_different_noise/000" + str(
            i) + "_simulation_noised.txt", delimiter=",")
    source = np.unique(source,axis = 0)
    target = np.unique(target,axis = 0)
    array_index_dist_k = data_association_k(target,target,11)   #np.array((N,k,2))
    array_target_normal = np.zeros((len(target),6))


    target_origin = trajectory[i,2:5]
    source_origin = trajectory_noised[i,:3]
    position_source_icp = source_origin
    for j in range(len(target)):
        k_neighbors = target[array_index_dist_k[j, :, 0].astype(np.int32)]
        k_neighbors_normal, _ = normalize(k_neighbors)
        production = k_neighbors_normal.T @ k_neighbors_normal
        u, sigma_matrix, v = svd(production)
        normal_vector = u[:, 2]
        array_target_normal[j, :3] = target[j]
        array_target_normal[j, 3:] = normal_vector
    for time in range(iteration_times):
        array_index_dist = data_association(source, target)
        array_source_index_dist = np.concatenate((source, array_index_dist), axis=1)  # np.array((N,5),  source,index,dist)
        sorted_array_index_dist = discard_large_dist(0, array_source_index_dist)   # np.array((N,5)， source,index,dist)
        sorted_array_index_dist = np.unique(sorted_array_index_dist,axis = 0)

        E = np.sum(np.abs(((sorted_array_index_dist[:,:3] - target[sorted_array_index_dist[:,3].astype(np.int32)]) @ array_target_normal[sorted_array_index_dist[:,3].astype(np.int32),3:].T)).diagonal())
        E_static[time,i] = E

        A = np.zeros((len(sorted_array_index_dist),6))
        b = np.zeros((len(sorted_array_index_dist)))
        for k in range(len(sorted_array_index_dist)):
            target_q = array_target_normal[int(sorted_array_index_dist[k,3]),:3]
            normal_q = array_target_normal[int(sorted_array_index_dist[k,3]),3:]
            source_p = sorted_array_index_dist[k,:3]
            a1 = normal_q[2] * source_p[1] - normal_q[1] * source_p[2]
            a2 = normal_q[0] * source_p[2] - normal_q[2] * source_p[0]
            a3 = normal_q[1] * source_p[0] - normal_q[0] * source_p[1]
            A[k,:] = np.array([[a1,a2,a3,normal_q[0],normal_q[1],normal_q[2]]])
            b[k] = normal_q[0] * target_q[0] + normal_q[1] * target_q[1] + normal_q[2] * target_q[2] -normal_q[0] * source_p[0] - normal_q[1] * source_p[1] - normal_q[2] * source_p[2]
        x_hat = np.linalg.inv(A.T @ A) @ A.T @ b
        R = np.array([[1,-x_hat[2],x_hat[1]],[x_hat[2],1,-x_hat[0]],[-x_hat[1],x_hat[0],1]])
        t = x_hat[3:]
        source = (R @ source.T + t[:,np.newaxis]).T

        T_mean = R @ T_mean + t
        R_mean = R @ R_mean
        position_source_icp = R @ position_source_icp + t
        D_static[time, i] = np.linalg.norm(position_source_icp - target_origin, axis=0)
        np.savetxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/demo"+str(time)+".txt",source,fmt="%.14f",delimiter=",")
    position_source = R_mean @ source_origin + T_mean.reshape((3,))

print(E_static)
print(D_static)
print(position_source)
print(position_source_icp)
