import cv2
import sys
import os
import time
import background_sub as bsub
import MJ_merge as merge
from pathlib import Path
import argparse
import subprocess
import shutil
import json

parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='data/originals',
                    help="the directory which contains the pictures set.")
parser.add_argument('--fg_dir', type=str, default='data/fg',
                    help="the directory which contains the foreground pictures set.")
parser.add_argument('--bg_dir', type=str, default='data/bg',
                    help="the directory which contains the background pictures set.")
parser.add_argument('--output_dir', type=str, default='data/results',
                    help="the directory which contains the final results.")
parser.add_argument('--reconstructor', type=str, default='MvgMvsPipeline.py',
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--fg_adc', type=int, default=200, help="the advancement of foreground mask.")
parser.add_argument('--bg_adc', type=int, default=160, help="the advancement of background mask.")


args = parser.parse_args()

with open("time.json", "r") as jsonFile:
    time_file = json.load(jsonFile)

time_file["timeList"] = []

with open('time.json', 'w', encoding='utf-8') as f:
    json.dump(time_file, f, indent=4)

shutil.rmtree(args.fg_dir)
shutil.rmtree(args.bg_dir)
# shutil.rmtree(args.output_dir)

Path(args.fg_dir).mkdir(parents=True, exist_ok=True)
Path(args.bg_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# create cv2 background module for each camera
backSub = {}
for image_dir in os.listdir(args.data_dir):
    backSub[image_dir] = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=216, detectShadows=False)

fail = { "timestamp": [] }

for timestamp in range(0, 21):
    str_timestamp = str(timestamp).zfill(5)
    print("current timestamp: ", str_timestamp, '-' * 50)

    if timestamp > 0:
        fg_dir = os.path.join(args.fg_dir, str_timestamp)
        bg_dir = os.path.join(args.bg_dir, str_timestamp)

        Path(fg_dir).mkdir(parents=True, exist_ok=True)
        # os.chmod(fg_dir, 0o777)
        Path(bg_dir).mkdir(parents=True, exist_ok=True)
        # os.chmod(bg_dir, 0o777)

    # go through each camera
    for image_dir in os.listdir(args.data_dir):
        # image file format = 000004.png
        img_file_name = str_timestamp + ".png"
        print("\tload image {} from camera #{}".format(img_file_name, image_dir))

        image = cv2.imread(os.path.join(args.data_dir, image_dir, img_file_name))

        # print(os.path.join(args.data_dir, image_dir, img_file_name))

        if timestamp == 0:  # run background subtraction algorithm on first image
            backSub[image_dir].apply(image)
            continue

        extracted_binary_foreground = backSub[image_dir].apply(image)

        # create background and foreground images
        backsub_success, fg, bg_to_fg, box_areas = bsub.create_fg_mask(extracted_binary_foreground, image,
                                                                       fg_advancement=120, bg_advancement=90,
                                                                       color=True)
        cv2.imwrite(os.path.join(fg_dir, image_dir + "_" + str_timestamp + "_fg.png"), fg)  # save foreground image
        bg = bsub.create_background(bg_to_fg, image, color=True)  # create background image
        cv2.imwrite(os.path.join(bg_dir, image_dir + "_" + str_timestamp + "_bg.png"), bg)  # save background image

    if timestamp == 0:
        continue
    # start to run openMvg + openMvs for foreground and background

    fg_output_dir = os.path.join(args.fg_dir, str_timestamp + "_output")
    bg_output_dir = os.path.join(args.bg_dir, str_timestamp + "_output")

    try:
        # start to run openMvg + openMvs for foreground
        print("start to reconstruct {}/{}".format(fg_dir, str_timestamp))
        start = time.time()
        p = subprocess.Popen(["python3", args.reconstructor, fg_dir, fg_output_dir])
        p.wait()
        if p.returncode != 0:
            break
        print("foreground finished in {}".format(time.time() - start))

        # start to run openMvg + openMvs for background
        print("start to reconstruct {}/{}".format(bg_dir, str_timestamp))
        start = time.time()
        p = subprocess.Popen(["python3", args.reconstructor, bg_dir, bg_output_dir])
        p.wait()
        if p.returncode != 0:
            break
        print("foreground finished in {}".format(time.time() - start))
    except Exception as e:
        print(e)
        # sys.exit('\r\nProcess canceled by user, all files remains')
        continue

    # start to merge foreground and background
    if sys.platform.startswith('win'):
        cmd = "where"
    else:
        cmd = "which"

    try:
        ret = subprocess.run([cmd, "openMVG_main_SfMInit_ImageListing"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=True)
        OPENMVG_BIN = os.path.split(ret.stdout.decode())[0]

        pChange = subprocess.Popen(
            [os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i", fg_output_dir + "/sfm/sfm_data.bin",
             "-o", fg_output_dir + "/sfm/sfm_data.json"])
        pChange.wait()

        pChange = subprocess.Popen(
            [os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"), "-i", bg_output_dir + "/sfm/sfm_data.bin",
             "-o", bg_output_dir + "/sfm/sfm_data.json"])
        pChange.wait()

        # path 1 -> foreground.json
        path1 = fg_output_dir + "/sfm/sfm_data.json"

        # path 2 -> background.json
        path2 = bg_output_dir + "/sfm/sfm_data.json"

        # path 3 -> foreground final texture.ply
        path3 = fg_output_dir + "/mvs/scene_dense_mesh_refine_texture.ply"

        # path 4 -> background final texture.ply
        path4 = bg_output_dir + "/mvs/scene_dense_mesh_refine_texture.ply"

        # path 5 -> output.ply
        path5 = args.output_dir + '/result_' + str_timestamp + '.ply'

        merge.do_merge(path1, path2, path3, path4, path5)

    except:
        fail["timestamp"].append(str_timestamp)


with open('fail.json', 'w', encoding='utf-8') as f:
    json.dump(fail, f, indent=4)





