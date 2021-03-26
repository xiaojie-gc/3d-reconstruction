
import ujson as json
import time
from itertools import combinations
import numpy as np
    
def compute_cam_cpointM(file_path):
    with open(json_file_path) as data_file:
        data = json.load(data_file)
        num_views = len(data['extrinsics'])
        points3d = data['structure'] 
        num_pts = len(points3d)
        # Create matrix 
        m = np.zeros((num_views, num_pts))
        
        pt_cnt = 0
        for key in points3d:
            matches = [k['key'] for k in key['value']['observations']]
            for cam_id in matches:
                # put 1's where there is match
                m[cam_id][pt_cnt] = 1
                # print("cam id = ", cam_id, "\tpt_cnt = ", pt_cnt)
            pt_cnt += 1
        return m, num_views, num_pts


    
def my_getk_optimal_cams(k , json_file_path):
    points_ik, num_views, num_pts = compute_cam_cpointM(json_file_path)
    view_range = range(0,num_views)
    combos = list( combinations(view_range, k) )
    
    points_sat = {}
    # Loop through each combination and return one with most points satisfied

    for c in combos:
        #initialize cameras bools
        bools = np.zeros((num_views,1)) 
        for i in c:
            bools[i,0] = 1
        #multiply each col by camera bools
        p = points_ik * bools
        # sum total amount of cameras covering each point
        p = p.sum(axis=0)
        # return count of >= 2
        num_pts_covered = np.count_nonzero(p >= 2, 0)
        points_sat[num_pts_covered] = c
    best_key = max(points_sat.keys())
    best_k = points_sat[best_key]
    return best_k
        
    

if __name__ == "__main__":
    #json_file_path = "/Users/andrewhlton/3d_reconstruction/dance_dataset/data/running_openMVS/test_8cam_orig_out/export1.json"
    # json_file_path = "/Users/andrewhlton/3d_reconstruction/dance_dataset/data/diff_pipelines/fg1_SfM2_default/sfm/sfm_data_final.json"
    # json_file_path = r"C:\Users\Andre\Dropbox\quality_eval\export1.json"
    json_file_path = r"E:\camera_opt\export_example_10_intr.json"
    times = []
    for i in range(10):
        start = time.perf_counter()
        camera_bools = my_getk_optimal_cams(6 , json_file_path)
        end = time.perf_counter()
        times.append(end-start)
    print("Average Time: {}".format(sum(times) / len(times)))
    print("Best k cameras:", camera_bools)

