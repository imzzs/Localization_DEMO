from icp import *

target = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,-1,1]])
target_origin = np.array([[2,1,3]])

R = np.array([[1,0,0],[0,np.sqrt(2)/2,-np.sqrt(2)/2],[0,np.sqrt(2)/2,np.sqrt(2)/2]])
t = np.array([[0.5,0.5,0.5]])

R = np.dot(R,np.array([[np.sqrt(2)/2,-np.sqrt(2)/2,0],[np.sqrt(2)/2,np.sqrt(2)/2,0],[0,0,1]]))
source = (R @ target.T + t.T).T   #
source_origin = ((R @ target_origin.T) + t.T).T
source_origin_ = source_origin
iteration_times = 10

R_mean = np.eye(3, 3)
T_mean = np.zeros((3, 1))

d_static = []
e_static = []
for i in range(iteration_times):
    array_index_dist = data_association(source,target)
    array_source_index_dist = np.concatenate((source, array_index_dist), axis=1)  # np.array((N,5),  source,index,dist)
    sorted_array_index_dist = discard_large_dist(0, array_source_index_dist)
    source_result = sorted_array_index_dist[:, :3]
    target_result = target[sorted_array_index_dist[:, 3].astype(np.int32), :]

    E = np.sum(np.linalg.norm((source_result - target_result),axis = 1))
    normalized_source_result,_ = normalize(source)
    normalized_target_result,_ = normalize(target)
    production = np.dot(normalized_target_result.T, normalized_source_result)
    e_static.append(E)
    u, s, vh = svd(production)
    production_ = np.dot(np.dot(u, s), vh)
    R = np.dot(u, vh)
    one_matrix = np.ones((normalized_source_result.shape[0], 1))
    t = np.dot(((target / target.shape[0]).T - np.dot(R, (source/source.shape[0]).T)), one_matrix).T   # np.shape(1,3)
    source = (np.dot(R, source.T) + t.T).T
    T_mean = R @ T_mean + t.T
    R_mean = R @ R_mean
    distance = np.linalg.norm(source_origin - target_origin)
    source_origin = ((R @ source_origin.T) + t.T).T
    d_static.append(distance)
    np.savetxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/demo_" + str(i) + ".txt",
               source, fmt="%.14f", delimiter=",")
np.savetxt("/home/zhao/sda3/masterarbeit/preparetion/localization_demo/data/target.txt",
               target, fmt="%.14f", delimiter=",")
position_source_icp =(R_mean @ source_origin_.T + T_mean).T

print(d_static)
print(e_static)
print(source_origin)
print(position_source_icp)
print(target_origin)
