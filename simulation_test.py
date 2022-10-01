from icp import *
positions = 1
iteration_times = 5
target_static = np.zeros((positions,3))
source_static= np.zeros((positions,iteration_times,3))
E_static = np.zeros((iteration_times, positions))
D_static = np.zeros((iteration_times, positions))
for i in range(positions):
        target = np.loadtxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/simulated_different_noise/000"+str(i)+"_simulation.txt",delimiter=",")
        #source = np.loadtxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/demo.txt",delimiter=",")
        trajectory = np.loadtxt('/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_VelodyneOrigin_noinv.txt', delimiter=",") #np.array.shape = (N,8)
        trajectory_noised = np.loadtxt('/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/extractet_traj_points_noised.txt', delimiter=",")   #np.array.shape = (N,6)
        source = np.loadtxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/simulated_different_noise/0000_simulation_noised.txt",delimiter=",")
        R_noised = o3d.geometry.get_rotation_matrix_from_yxz(trajectory_noised[i,3:])
        T_noised = trajectory_noised[i,:3]
        source, mean_source = normalize(source)
        T_target = trajectory[i,2:5]
        T_target_mean = T_target - mean_source
        target, mean_target = normalize(target)
        T_noised_mean = T_noised - mean_target
        position_source_icp = T_noised_mean
        R_mean = np.eye(3,3)
        T_mean = np.zeros((3,1))
        np.savetxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/target+"+str(i)+".txt",target,fmt="%.14f",delimiter=",")
        target_static[i,:] = T_target_mean
        for j in range(iteration_times):
            array_index_dist = data_association(source,target)    #按照source顺序返回target index和距离 返回 np.array([[target_index,dist]...[]])
            array_source_index_dist = np.concatenate((source,array_index_dist),axis = 1)   #np.array((N,5),  source,index,dist)
            sorted_array_index_dist = discard_large_dist(0,array_source_index_dist)
            source_result = sorted_array_index_dist[:,:3]
            target_result = target[sorted_array_index_dist[:,3].astype(np.int32),:]
            E = np.sum(np.linalg.norm((source_result - target_result),axis = 1))
            E_static[j,i] = E
            normalized_source_result,_ = normalize(source_result)
            normalized_target_result,_ = normalize(target_result)

            production = np.dot(normalized_target_result.T,normalized_source_result)
            #u,s,vh = np.linalg.svd(production)
            u,s,vh = svd(production)
            production_ = np.dot(np.dot(u,s),vh)
            R = np.dot(u,vh)
            one_matrix = np.ones((target.shape[0],1))
            t = np.dot(((target / target.shape[0]).T - np.dot(R, (source/source.shape[0]).T)), one_matrix) # np.shape(3,1)
            source = (np.dot(R,source.T) + t).T
            np.savetxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/demo"+str(i)+str(j)+".txt",source,fmt="%.14f",delimiter=",")
            R_mean = R @ R_mean
            T_mean = R_mean @ t + T_mean
            position_source_icp = R @ position_source_icp + t.reshape((3,))
            source_static[i,j,:] = position_source_icp
            distance = np.linalg.norm(position_source_icp - T_target_mean)
            D_static[j,i] = distance

print(T_target_mean)
print(target_static)
print(source_static)
print(E_static)
print(D_static)


# y = E_static
# plt.plot(x,y)
# plt.show()