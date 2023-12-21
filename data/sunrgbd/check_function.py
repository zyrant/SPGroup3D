import numpy as np
import torch 
import open3d as o3d
from torch_scatter import scatter_mean


# check superpoint
depth = np.fromfile('/opt/data/private/all_data/sunrgbd/points/006887.bin', dtype=np.float32).reshape(-1,6)  # x, y, z, r,g,b
print(depth.shape)
labels = np.fromfile('/opt/data/private/all_data/sunrgbd/superpoints/006887.bin', dtype=np.int64)
print(labels.shape, 'num_superpoints: ', labels.max()+1)

depth_tensor = torch.from_numpy(depth)
labels_tensor = torch.from_numpy(labels)
superpoints = scatter_mean(depth_tensor, labels_tensor, dim=0)
print(labels.shape, 'num_superpoints: ', labels.max()+1)

map_color = []
for i in range(labels.max()+1):
    r = torch.rand(1,1) + 0.1
    g = torch.rand(1,1) + 0.1
    b = torch.rand(1,1) + 0.1
    xx = torch.cat((r,g,b),dim=1)
    map_color.append(xx)
map_color = torch.cat(map_color, dim=0)
orgin_rgb = map_color[labels]
orgin_data = torch.from_numpy(depth)
orgin_data_rgb = torch.cat((orgin_data[:,:3], orgin_rgb),dim=1)

np.savetxt('./all_scene.txt', orgin_data_rgb)
# show
pcd =o3d.io.read_point_cloud('./all_scene.txt', format='xyzrgb') # x y z r g b
o3d.visualization.draw_geometries([pcd], width=1000, height=2000)
