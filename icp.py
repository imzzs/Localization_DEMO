"""
This script registers 2 cluster point cloud using algorithm icp.
Given: 2 point cloud
return: rigid transformation parameter R, t.
"""
import numpy as np
import math
import copy
import open3d as o3d
import matplotlib.pyplot as plt
class Node:
    def __init__(self,value,axis,left,right,point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices
    def is_leaf(self):
        if self.value == None:
            return True
        else:
            return False

class DistIndex:
    def __init__(self,distance,index):
        self.distance = distance
        self.index = index     #初始化结果点集之中，下标用来做什么。


class KNNResultSet:
    def __init__(self,capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist,0))
        self.comparison_counter = 0
    def add_point(self,dist,index):
        self.comparison_counter += 1
        if dist > self.worst_dist:
            return
        if self.count < self.capacity:
            self.count += 1
        i = self.count - 1
        while(i > 0):
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                i -= 1
            else:
                break
        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = int(index)
        self.worst_dist = self.dist_index_list[self.capacity-1].distance


class RNNResultset:
    def __init__(self,radius):
        self.radius = radius
        self.dist_result_set = []
        self.count = 0
        self.comparison_counter = 0
    def add_point(self,dist,index):
        self.comparison_counter += 1
        if dist > self.radius:
            return
        self.dist_result_set.append(DistIndex(dist,index))

def sort_key_by_value(keys,values):
    assert keys.shape == values.shape
    assert len(keys.shape) == 1
    sorted_indices = np.argsort(values)
    sorted_keys = keys[sorted_indices]
    sorted_values = values[sorted_indices]
    return sorted_keys,sorted_values

def axis_round_robin(axis,dim):
    if axis == dim - 1 :
        axis = 0
    else:
        axis += 1
    return axis

def kdtree_recursive_build(root,db,point_indices,axis,leaf_size):
    if root is None:
        root = Node(None,axis,None,None,point_indices)
    if len(point_indices) > leaf_size:   #判断如果不是叶子节点
        sorted_keys,_ = sort_key_by_value(point_indices,db[point_indices,axis])
        middle_left_idx = math.ceil(sorted_keys.shape[0]/2) - 1
        middle_left_point_idx = sorted_keys[middle_left_idx]
        middle_left_value = db[middle_left_point_idx,axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = sorted_keys[middle_right_idx]
        middle_right_value = db[middle_right_point_idx,axis]

        root.value = (middle_left_value + middle_right_value)*0.5

        root.left = kdtree_recursive_build(root.left,db,sorted_keys[:middle_left_idx],axis_round_robin(axis,db.shape[1]),leaf_size)
        root.right = kdtree_recursive_build(root.right,db,sorted_keys[middle_right_idx:],axis_round_robin(axis,db.shape[1]),leaf_size)

    return root

def kdtree_knn_search(root:Node,db:np.ndarray,knnsearchset:KNNResultSet,query:np.ndarray):
    if root is None:
        return
    if root.is_leaf():
        leaf_points = db[root.point_indices,:]
        diff = np.linalg.norm(np.expand_dims(query,0)-leaf_points,axis = 1)
        for i in range(len(diff)):
            knnsearchset.add_point(diff[i],root.point_indices[i])
        return False
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left,db,knnsearchset,query)
        if math.fabs(query[root.axis] - root.value) < knnsearchset.worst_dist:
            kdtree_knn_search(root.right, db, knnsearchset, query)
    else:
        kdtree_knn_search(root.right,db,knnsearchset,query)
        if math.fabs(query[root.axis] - root.value) < knnsearchset.worst_dist:
            kdtree_knn_search(root.left, db, knnsearchset, query)
    return False

def kdtree_radius_search(root:Node,db:np.ndarray,rsearchset:RNNResultset,query:np.ndarray):
    if root is None:
        return
    if root.is_leaf():
        leaf_points = db[root.point_indices,:]
        diff = np.linalg.norm(np.expand_dims(query,0) - leaf_points,axis = 1)
        for i in range(len(diff)):
            rsearchset.add_point(diff[i],root.point_indices[i])
        return False
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left,db,rsearchset,query)
        if math.fabs(query[root.axis] - root.value) < rsearchset.radius:
            kdtree_radius_search(root.right, db, rsearchset, query)
    else:
        kdtree_radius_search(root.right,db,rsearchset,query)
        if math.fabs(query[root.axis] - root.value) < rsearchset.radius:
            kdtree_radius_search(root.left, db, rsearchset, query)
    return False



def kdtree_construction(db_np,leaf_size):
    N,dim = db_np.shape[0],db_np.shape[1]
    root = None
    root = kdtree_recursive_build(root,db_np,np.arange(N),0,leaf_size)
    return root

def kd_test():
    db_size = 64
    dim = 3
    leaf_size = 1

    db_np = np.random.rand(db_size,dim)

    root = kdtree_construction(db_np,leaf_size)

    Resultset_knn = KNNResultSet(2)
    Resultset_r = RNNResultset(12)

    kdtree_knn_search(root,db_np,Resultset_knn,np.array([0,0,3]))
    kdtree_radius_search(root,db_np,Resultset_r,np.array([0,0,10]))
    for i in range(len(Resultset_r.dist_index_list)):
        print(Resultset_knn.dist_index_list[i].distance, "-", Resultset_knn.dist_index_list[i].index)
    print("radius:")
    for i in range(len(Resultset_r.dist_result_set)):
        print(Resultset_r.dist_result_set[i].distance, "-", Resultset_r.dist_result_set[i].index)

def normalize(pointcloud):
# pointcloud: np.array((N,3),dtype=float))
# return normalized: np.array((N,3),dtype=float))  mean_repeat[0]  np.shape = (3,)
    mean_repeat = np.repeat(np.array([np.mean(pointcloud,axis = 0).T]),pointcloud.shape[0],axis = 0)
    normalized = pointcloud - mean_repeat
    return normalized, mean_repeat[0]

def discard_large_dist(percent,array_source_index_dist):
#discard percent of point from association with largest distance
    sorted_index_array = np.argsort(array_source_index_dist[:,4],axis = 0)   # 按距离排序的下标，原顺序是source         sorted np.array([[source,index,dist],..[]])
    discard_index = int(len(sorted_index_array) - np.ceil(percent*len(sorted_index_array)))   #所有数组剪掉距离远的
    sorted_array_index_dist = array_source_index_dist[sorted_index_array[:discard_index]]
    return sorted_array_index_dist    #np.array([[source,index,dist1]...[source,index,dist2]])   按照dist排序之后， index为target下标

def data_association(source,target):
#Data association
    # source_normalized = normalize(source)
    # target_normalized = normalize(target)
    source_normalized = source
    target_normalized = target
    root = kdtree_construction(target_normalized,1)
    array_index_dist = np.zeros((source.shape[0],2))
    count = 0
    for point in source_normalized:
        KNNresultSet = KNNResultSet(1)
        kdtree_knn_search(root,target_normalized,KNNresultSet,point)
        array_index_dist[count] = np.concatenate((np.array([KNNresultSet.dist_index_list[0].index],dtype=int),np.array([KNNresultSet.dist_index_list[0].distance])),axis = 0)
        count += 1
        print(count)
#Iteration

    return array_index_dist

def svd(A):
    m, n = A.shape
    if m > n:
        sigma, V = np.linalg.eig(A.T @ A)
        # 将sigma 和V 按照特征值从大到小排列
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        V = V[:, arg_sort]

        # 对sigma进行平方根处理
        sigma_matrix = np.diag(np.sqrt(sigma))

        sigma_inv = np.linalg.inv(sigma_matrix)

        U = A @ V.T @ sigma_inv
        U = np.pad(U, pad_width=((0, 0), (0, m - n)))
        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, m - n), (0, 0)))
        return (U, sigma_matrix, V)
    else:
        # 同m>n 只不过换成从U开始计算
        sigma, U = np.linalg.eig(A @ A.T)
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        U = U[:, arg_sort]

        sigma_matrix = np.diag(np.sqrt(sigma))
        sigma_inv = np.linalg.inv(sigma_matrix)
        V = sigma_inv @ U.T @ A
        V = np.pad(V, pad_width=((0, n - m), (0, 0)))

        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, 0), (0, n - m)))
        return (U, sigma_matrix, V)






