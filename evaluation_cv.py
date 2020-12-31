import json
import time
import numpy as np
import cv2

class Extrinsics:
    def __init__(self, rotation_matrix, translation_vector, r_t_matrix):
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
    def print_extrinsics(self):
        print("Rotation Matrix: ", self.rotation_matrix, "\nTranslation Vector: ", self.translation_vector)


class Intrinsics:
    def __init__(self, view_num, k_matrix, width, height, distortion):
        self.view_num = view_num
        self.k_matrix = k_matrix
        self.width = width
        self.height = height
        self.distortion = distortion

    def print_intrinsics(self):
        print("View Number = ", self.view_num, "\nWidth = ", self.width, "\nHeight = ", self.height, "\nDistortion = ", self.distortion)
        np.set_printoptions(suppress=True)
        print("Printing k matrix: \n", self.k_matrix)

class ProjectionParams:
    def __init__(self,rotation, translation, k_matrix, distortion):
        self.X = []
        self.rotation = rotation
        self.translation = translation
        self.k_matrix = k_matrix
        self.distortion = distortion
        self.x = []

    def get_params(self):
        return self.X, self.rotation, self.translation, self.k_matrix, self.distortion, self.x

    def print_param(self):
        print('-'*50, "\nX length: ", len(self.X), "\ngt_x length: ", len(self.x),  "\nRotation: ", self.rotation,
              "\nTranslation: ", self.translation, "\nK Matrix: ", self.k_matrix,
              "\nDistortion: ", self.distortion, )

# data is a json object
def get_camera_data(data):
    camera_intrinsics = {}
    camera_extrinsics = {}
    num_of_extrinsics = len(data['extrinsics'])

    # Getting camera extrinsics
    extr = data['extrinsics']
    for i in range(0, num_of_extrinsics):  # This will preserve order as the json file is not in order
        for key in extr:
            if key['key'] == i:
                current_view = key['key']
                rotation_matrix = np.array( key['value']['rotation'])
                translation_vector = np.array(key['value']['center'])
                translation_vector = np.vstack(translation_vector)
                translation_vector = -rotation_matrix @ translation_vector
                r_t_matrix = np.append(rotation_matrix, translation_vector  , axis=1)
                current_extrinsics = Extrinsics(rotation_matrix, translation_vector, r_t_matrix)
                camera_extrinsics[current_view] = current_extrinsics

    # Getting camera intrinsics
    for key in data['intrinsics']:
        view_num = key['key']
        img_width = key['value']['ptr_wrapper']['data']['width']
        img_height = key['value']['ptr_wrapper']['data']['height']
        focal_length = key['value']['ptr_wrapper']['data']['focal_length']
        p_x = key['value']['ptr_wrapper']['data']['principal_point'][0]
        p_y = key['value']['ptr_wrapper']['data']['principal_point'][1]
        dist_name = list( key['value']['ptr_wrapper']['data'])[4] # name can change depending on type of distortion
        distortion = np.array([key['value']['ptr_wrapper']['data'][dist_name]])
        k_matrix = np.array([focal_length, 0, p_x, 0, focal_length, p_y, 0, 0, 1]).reshape(3, 3)
        camera_intrinsics[view_num] = Intrinsics(view_num, k_matrix, img_width, img_height, distortion)

    return camera_extrinsics, camera_intrinsics

# Running code
def evaluation(json_file_path):
    start = time.perf_counter()
    
    with open(json_file_path) as data_file:
        data = json.load(data_file)
        cam_extr, cam_intr = get_camera_data(data)
        num_of_cams = len(cam_extr)
        views = {}
        #initialize each view with correct extrinsics and intrinsics
        for i in range(0,num_of_cams):
            if len(cam_intr) == 1:
                views[i] = ProjectionParams(cam_extr[i].rotation_matrix, cam_extr[i].translation_vector,
                                            cam_intr[0].k_matrix, cam_intr[0].distortion)
            else:
                views[i] = ProjectionParams(cam_extr[i].rotation_matrix, cam_extr[i].translation_vector,
                                            cam_intr[i].k_matrix, cam_intr[i].distortion)
    
        # add 3d points and ground truth 2d poinits for each view
        points3d = data['structure']
        print("Number of Feature points: ", len(points3d) )
        for key in points3d:
            X = key['value']['X']
            # loop through 2d points and perform the projection
            for view in key['value']['observations']:
                view_num = view['key']
                gt_x = view['value']['x']
                # append 3d point to corresponding view
                views[view_num].X.append(X)
                # append 2d gt point to corresponding view
                views[view_num].x.append(gt_x)
    
    
    mean_error = 0
    for view in views:
        X, rotation, translation, k_matrix, distortion, x = views[view].get_params()
        X = np.float32(X)
        rotVec,_ = cv2.Rodrigues(rotation)
        #fake_dist = np.append(distortion, np.array([0]))
        imgpoints2, _ = cv2.projectPoints(X, rotVec, translation, k_matrix, None )
        imgpoints2 = imgpoints2.reshape(-1,2)
        gt_x = np.array( x )
        mean_error += np.absolute( gt_x - imgpoints2).mean(axis=None)
        print("Average Error for View: ", np.absolute( gt_x - imgpoints2).mean(axis=None), '\n', '-'*50)
    
    print("Average error is:", mean_error / num_of_cams )
    print("Total time: ", time.perf_counter() - start)
    return mean_error / num_of_cams
        
# test

#json_file = r"C:\Users\Andre/export1.json"
json_file = r"C:\Users\Andre/export_example.json"
error = evaluation(json_file)