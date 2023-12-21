# https://github.com/davidcaron/pclpy
# create by --zyrant

import numpy as np
from pclpy import pcl
import time
import os
from multiprocessing import Pool
import multiprocessing

def process_file(file):
    save_path = os.path.join(save_folder, os.path.basename(file))

    if os.path.exists(save_path):
        print('File: ', save_path, ' already exists, skip processing.')
        return

    start_time = time.time()
    
    depth = np.fromfile(file, dtype=np.float32).reshape(-1,6)  # x, y, z, r, g, b
    xyz = depth[:, :3]
    rgb = depth[:,3:]
    pc = pcl.PointCloud.PointXYZRGBA(xyz, rgb)
    normals = pc.compute_normals(radius=0.25)

    '''
    Params are from :
    G. Gao, M. Lauri, J. Zhang and S. Frintrop, 
    "Saliency-guided adaptive seeding for supervoxel segmentation," 
    2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 
    Vancouver, BC, Canada, 2017, pp. 4938-4943, doi: 10.1109/IROS.2017.8206374.
    '''

    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(voxel_resolution=0.02,
                                                            seed_resolution=0.25)
    vox.setInputCloud(pc)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(0.15)
    vox.setNormalImportance(0.15)
    vox.setColorImportance(0.7)
    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    labeled_cloud = vox.getLabeledCloud()

    labels = np.zeros((labeled_cloud.size(), ), dtype=np.int64)
    for i in range(labeled_cloud.size()):
        labels[i] = labeled_cloud.at(i).label
    
    labels.tofile(save_path) 
    end_time = time.time()
    total_time = end_time - start_time
    print('Saved to: ', save_path, 'Current superpoint shape: ', len(list(output.items())), 'Current time cost: ', total_time)

data_path = os.listdir('/opt/data/private/all_data/sunrgbd/points/')
depth_files = [os.path.join('/opt/data/private/all_data/sunrgbd/points/', f) for f in data_path]
depth_files.sort()

save_folder = '/opt/data/private/all_data/sunrgbd/superpoints/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

pool = Pool(multiprocessing.cpu_count())
print('Current multiprocessing cout: ', multiprocessing.cpu_count())
# Use imap to apply the process_file function to each file, and the iterator will yield results as tasks are completed.
for _ in pool.imap(process_file, depth_files):
    pass

pool.close()
pool.join()
