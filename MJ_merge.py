"""
Auther:  Mingjun Li
Date:    12/23/2020
Purpose: To merge two different .ply files which contain the info
         of 3D points and the corresopnding faces got from the 3D
         reconstruction process via using OpenMVG and OpenMVS.

The idea of function-4 (get_transform) comes from the ICP algorithm,
but in our case, we don't need to do the iterative process to get
the matching-point sets, because we got those from .json files.

Problem so far is the user has to input the paths by hand, hope
this can be solved tomorrow.
"""

import json
from plyfile import PlyData
import time
import numpy as np
from numpy import linalg as la

#######################################################################
# 0
# Load the .json files
def load_json(path):
    f = open(path)
    F = json.load(f)
    return F    


#######################################################################
# 1
# Function to get 3D positions, image id, and 2D positions from .json file
# input : a .json file
# output: 3 lists
def getPoints(f):
    structure = f['structure']
    l = len(structure)
    position_3D = []
    image_id = [[]for i in range(l)]
    position_2D = [[]for i in range(l)]
    for i in range(l):
        position_3D.append(structure[i]['value']['X'])
        ob = structure[i]['value']['observations']
        for j in range(len(ob)):
            image_id[i].append(ob[j]['key'])
            position_2D[i].append(ob[j]['value']['x'])

    return position_3D, image_id, position_2D


#######################################################################
# 2
# Get matching 3D points in both foreground & background
# input : 6 lists
# output: 2 lists
def get_matching_points(fg_image_id, fg_2d, fg_3d, bg_image_id, bg_2d, bg_3d):
    match_bg_3d = []
    match_fg_3d = []

    # go through all the 3D points in foreground
    for i in range(len(fg_3d)):
        
        # for each 3D point in the foreground
        # go through all the corresponding points in images
        num_fg_image_id = len(fg_image_id[i])
        for j in range(num_fg_image_id):

            # compare with all the 2D point in original images
            for m in range(len(bg_3d)):
                num_bg_image_id = len(bg_image_id[m])
                for n in range(num_bg_image_id):

                    # if the images are same
                    if fg_image_id[i][j] == bg_image_id[m][n]:
                        x_diff = fg_2d[i][j][0]-bg_2d[m][n][0]
                        y_diff = fg_2d[i][j][1]-bg_2d[m][n][1]
                        if abs(x_diff)<0.1 and abs(y_diff)<0.1:
                            match_fg_3d.append(fg_3d[i])
                            match_bg_3d.append(bg_3d[m])
    match_bg_3d_2 = []
    match_fg_3d_2 = []
    
    match_fg_3d_2.append(match_fg_3d[0])
    match_bg_3d_2.append(match_bg_3d[0])
    for i in range(1,len(match_fg_3d)):
        if not match_fg_3d[i] in match_fg_3d_2:
            match_fg_3d_2.append(match_fg_3d[i])
            match_bg_3d_2.append(match_bg_3d[i])
    #print('Matching 3D points:',len(match_fg_3d_2))
    return match_fg_3d_2, match_bg_3d_2


#######################################################################
# 3
# Adjust the size of 2 3D objects to be the same
# input : 2 lists
# output: 4 np.arrays
def rescale(match_fg_3d, match_bg_3d):
    fg = np.array(match_fg_3d)
    bg = np.array(match_bg_3d)

    center_fg = np.mean(fg, axis=0)
    center_bg = np.mean(bg, axis=0)

    x1 = np.sum(np.square(fg - center_fg), axis=1)
    x2 = np.sum(np.square(bg - center_bg), axis=1)
    dis_points2center_fg = np.sqrt(x1)
    dis_points2center_bg = np.sqrt(x2)

    scaler = []
    for i in range(len(dis_points2center_fg)):
        scaler.append(dis_points2center_bg[i]/dis_points2center_fg[i])

    scaler_mean = np.mean(scaler)

    fg2 = scaler_mean * fg
    center_fg2 = np.mean(fg2, axis=0)
    return fg2, center_fg2, bg, center_bg, scaler_mean
    

#######################################################################
# 4
# Get transformation info
# input : 2 N*3 np.array & 2 3D positions
# output: a 3*3 matrix & a vector
def get_transform(fg, bg, center_fg, center_bg):
    fg2 = fg - center_fg
    bg2 = bg - center_bg

    H = np.dot(bg2.T, fg2)
    U, s, VT = la.svd(H)

    # rotation matrix
    R = np.dot(U, VT)

    # special reflection case
    if la.det(R) < 0:
       VT[2,:] *= -1
       R = np.dot(U, VT)

    # translation
    t = center_bg.T - np.dot(R,center_fg.T)

    return R, t


#######################################################################
# 5
# Get number of points in .ply file
# input : str(.ply file)
def get_num_points(S):
    flag_index = S.find('vertex')
    num_index = flag_index + 7
    num_str = S[num_index]
    i = 1
    while S[num_index+i] != ' ' and S[num_index+i] != '\n':
        num_str = num_str + S[num_index+i]
        i = i+1
    num_int = int(num_str)
    return num_index, i, num_str, num_int


#######################################################################
# 6
# Get number of faces in .ply file
# input : str(.ply file)
def get_num_faces(S):
    flag_index = S.find('face')
    num_index = flag_index + 5
    num_str = S[num_index]
    i = 1
    while S[num_index+i] != ' ' and S[num_index+i] != '\n':
        num_str = num_str + S[num_index+i]
        i = i+1
    num_int = int(num_str)
    return num_index, i, num_str, num_int



#######################################################################
# 7 - final 
# Do the merge
def do_merge(path1, path2, path3, path4, path5):

    fg = load_json(path1)
    bg = load_json(path2)

    fg_3d, fg_image_id, fg_2d = getPoints(fg)
    bg_3d, bg_image_id, bg_2d = getPoints(bg)

    match_fg, match_bg = get_matching_points(fg_image_id, fg_2d, fg_3d,
                                             bg_image_id, bg_2d, bg_3d)

    fg2, cen_fg, bg2, cen_bg, scaler = rescale(match_fg, match_bg)

    R, t = get_transform(fg2, bg2, cen_fg, cen_bg)

    fg_texture = PlyData.read(path3)

    # Get the number of points
    number_points_fg = len(fg_texture.elements[0].data)
    number_elements_for_each_point = len(fg_texture.elements[0].data[0])

    print('Number of points:', number_elements_for_each_point, 'points.')

    # Get the number of faces
    number_faces_fg = len(fg_texture.elements[1].data)

    a = len(fg_texture.elements[1].data[0][0])
    b = len(fg_texture.elements[1].data[0][1])
    number_of_elements_for_each_face = a+b+2

    print('Each face has',number_of_elements_for_each_face,'elements.')

    B = PlyData.read(path4)
    BB = str(B)

    B_P_index, B_P_bits, B_P_num_str, B_P_num_int = get_num_points(BB)
    B_F_index, B_F_bits, B_F_num_str, B_F_num_int = get_num_faces(BB)
    aa = len(B.elements[1].data[0][0])
    bb = len(B.elements[1].data[0][1])
    B_pc = B['vertex'].data
    B_pc_array = np.array([[x,y,z] for x,y,z in B_pc])


    # Get the positions of all the points
    pc = fg_texture['vertex'].data
    pc_array = np.array([[x,y,z] for x,y,z in pc])
    #print(pc_array.shape)

    # Get the faces
    faces = fg_texture['face'].data

    # Do the transfomation
    changed_fg_texture = np.dot((scaler * pc_array), R.T) + t

    with open(path5,'w')as f:
        f.seek(0)
        f.write('ply\nformat ascii 1.0\ncomment VCGLIB generated\n')
        f.write('element vertex ')
        f.write(str(number_points_fg + B_P_num_int))
        f.write('\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('element face ')
        f.write(str(number_faces_fg + B_F_num_int))
        f.write('\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('property list uchar float texcoord\nend_header\n')

        # output the vertices (A goes first)
        for i in range(number_points_fg):
          for j in changed_fg_texture[i]:
            f.write(str(j))
            f.write(' ')
          f.write('\n')
        f.write('\n')

        # out put the vertices (B goes 2nd)
        for m in range(B_P_num_int):
            for n in B_pc_array[m]:
                f.write(str(n))
                f.write(' ')
            f.write('\n')

        # output the faces (A goes first)
        for i in range(number_faces_fg):
          f.write(str(a))
          f.write(' ')
          for j in range(a):
            f.write(str(fg_texture.elements[1].data[i][0][j]))
            f.write(' ')
          f.write(str(b))
          f.write(' ')
          for k in range(b):
            f.write(str(fg_texture.elements[1].data[i][1][k]))
            f.write(' ')
          f.write('\n')

        ''' output the faces (B goes 2nd) '''
        for m in range(B_F_num_int):
            f.write(str(aa))
            f.write(' ')
            for n in range(aa):
                f.write(str(B.elements[1].data[m][0][n] + number_points_fg))
                f.write(' ')
            f.write(str(bb))
            f.write(' ')
            for k in range(bb):
                f.write(str(B.elements[1].data[m][1][k]))
                f.write(' ')
            f.write('\n')
