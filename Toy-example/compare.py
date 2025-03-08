import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tda_utils import *
from tda import *
import glob
import open3d as o3d



path = '/mnt/g/Research/Projects/Embankment/simulated/'
save_path = "/".join(path.rstrip("/").split("/")[:-1]) + '/results/'

def compute(path, prefix, circle=True, single=True):
    save_path = "/".join(path.rstrip("/").split("/")[:-1]) + '/results/'
    pcfile_lst = sorted(glob.glob(f"{path}/{prefix}*.ply")) 
    dim = 1 if circle else 2
    pcd_lst = []
    print('Now processing: ', prefix, single)
    if circle:
        random2d = o3d.io.read_point_cloud(path + 'random2d.ply')
        pcd_lst.append(np.asarray(random2d.points)) 
    else:
        random3d = o3d.io.read_point_cloud(path + 'random3d.ply')
        pcd_lst.append(np.asarray(random3d.points)) 

    radius, num_obj = [0], [0]
    for ply in pcfile_lst:
        if single:
            radius.append(int(ply.split('/')[-1].split('_')[-1].split('.')[0]))
        else:
            num_obj.append(int(ply.split('/')[-1].split('_')[-1].split('.')[0]))
        pcd = o3d.io.read_point_cloud(ply)
        points = np.asarray(pcd.points)
        pcd_lst.append(points)
        
    if single:
        LifetimeOverRadius = pd.DataFrame(columns=['Radius', 'LifeTime'])
        liftimes = ransac_persistence_m(pcd_lst, dim=dim, k=10, m=1000)
        LifetimeOverRadius['Radius'] = radius
        LifetimeOverRadius['LifeTime'] = liftimes[:, -1]
        LifetimeOverRadius.to_csv(f"{save_path}{prefix}_LifetimeOverRadius.csv", index=False) 
    else:
        H1OverNumCircles = pd.DataFrame(columns=['NumCircles', 'H'+str(dim)])
        H1_arr = ransac_tda_m(pcd_lst, dim=dim, k=10, m=1000)
        H1OverNumCircles['NumCircles'] = num_obj 
        H1OverNumCircles['H'+str(dim)] = H1_arr[:, -1]
        H1OverNumCircles.to_csv(f"{save_path}{prefix}_H{dim}OverNumCircles.csv", index=False)




# 2D circles
# Single circle with variant sizes
prefix="circle_"
compute(path, prefix, circle=True, single=True)


# Multiple circles with same size
prefix="multicircles_"
compute(path, prefix, circle=True, single=False)



# Single sphere with variant sizes
prefix="sphere_"
compute(path, prefix, circle=False, single=True)

    
# Multiple spheres with same size
prefix="multisphere_"   
compute(path, prefix, circle=False, single=False)





