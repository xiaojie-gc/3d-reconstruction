import cv2
import sys
import os
import time
import background_sub as bsub
import MJ_merge as merge_process
from pathlib import Path
import argparse
import subprocess
import shutil
import json
import resolution


def execution_time_monitor(model, step, running_time, new=True):
    try:
        with open("time.json", "r") as jsonFile:
            time_file = json.load(jsonFile)

        if new is True:
            time_file["timeList"].append({"model": model, step: running_time})
        else:
            for item in time_file["timeList"]:
                if item["model"] == model:
                    item[step] = running_time

        with open('time.json', 'w', encoding='utf-8') as f:
            json.dump(time_file, f, indent=4)

    except Exception as e:
        print(e)


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
parser.add_argument('--fg_adc', type=int, default=120, help="the advancement of foreground mask.")
parser.add_argument('--bg_adc', type=int, default=90, help="the advancement of background mask.")
parser.add_argument('--resolution', type=str, default="golden", help="the advancement of background mask.")


args = parser.parse_args()

with open("time.json", "r") as jsonFile:
    time_file = json.load(jsonFile)

time_file["timeList"] = []

with open('time.json', 'w', encoding='utf-8') as f:
    json.dump(time_file, f, indent=4)

try:
    shutil.rmtree(args.fg_dir)
    shutil.rmtree(args.bg_dir)
except:
    print(".....")


Path(args.fg_dir).mkdir(parents=True, exist_ok=True)
Path(args.bg_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# create cv2 background module for each camera
backSub = {}
for image_dir in os.listdir(args.data_dir):
    backSub[image_dir] = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=216, detectShadows=False)

fail = {
    "timestamp": []
}

R = {
    "golden": (1920, 1080),
    "720p": (1280, 720),
    "480p": (720, 480)
}

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

    start = time.time()

    # go through each camera
    for image_dir in os.listdir(args.data_dir):
        # image file format = 000004.png
        img_file_name = str_timestamp + ".png"
        print("\tload image {} from camera #{}".format(img_file_name, image_dir))

        image = cv2.imread(os.path.join(args.data_dir, image_dir, img_file_name))

        if args.resolution != "golden":
            new_file_name = str_timestamp + "_" + args.resolution + ".png"
            image = cv2.resize(image, R[args.resolution])

        if timestamp == 0:  # run background subtraction algorithm on first image
            backSub[image_dir].apply(image)
            continue

        extracted_binary_foreground = backSub[image_dir].apply(image)
        # create background and foreground images
        success, img_mask, bg_to_fg, box_coords = bsub.create_fg_mask( extracted_binary_foreground, image, fg_advancement = args.fg_adc,
                                                                     bg_advancement = args.bg_adc, color = True)
        background_mask =  bsub.create_background(img_mask, image, color=True)
        # box coords 0 index is top left coordinate for box 1, 1 index is bottom right box 1, 2 index is top left for box 2 if neccassarry and so on
        # coordinates themselves list 1x2 numpy arrays containing x then y

        # # create background and foreground images
        cv2.imwrite(os.path.join(fg_dir, image_dir + "_" + str_timestamp + "_fg.png"), img_mask)
        cv2.imwrite(os.path.join(bg_dir, image_dir + "_" + str_timestamp + "_bg.png"), background_mask)

    if timestamp == 0:
        continue
    # start to run openMvg + openMvs for foreground and background

    fg_output_dir = os.path.join(args.fg_dir, str_timestamp + "_output")
    bg_output_dir = os.path.join(args.bg_dir, str_timestamp + "_output")

    execution_time_monitor(fg_dir, "background subtraction", time.time() - start, new=True)
    execution_time_monitor(bg_dir, "background subtraction", 0)

    try:

        # start to run openMvg + openMvs for foreground
        print("start to reconstruct {}/{}".format(fg_dir, str_timestamp))
        start = time.time()
        p = subprocess.Popen(["python3", args.reconstructor, fg_dir, fg_output_dir])
        p.wait()
        if p.returncode != 0:
            break
        execution_time_monitor(fg_dir, "foreground reconstruction", time.time() - start, new=False)

        # start to run openMvg + openMvs for background
        print("start to reconstruct {}/{}".format(bg_dir, str_timestamp))
        start = time.time()
        p = subprocess.Popen(["python3", args.reconstructor, bg_dir, bg_output_dir])
        p.wait()
        if p.returncode != 0:
            break
        execution_time_monitor(bg_dir, "background reconstruction", time.time() - start, new=False)

    except Exception as e:
        print(e)
        # sys.exit('\r\nProcess canceled by user, all files remains')
        continue

    # start to merge foreground and background
    try:
        start = time.time()

        if sys.platform.startswith('win'):
            cmd = "where"
        else:
            cmd = "which"

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
        # path 2 -> background.json
        # path 3 -> foreground final texture.ply
        # path 4 -> background final texture.ply
        # path 5 -> output.ply
        merge_process.do_merge(fg_output_dir + "/sfm/sfm_data.json", bg_output_dir + "/sfm/sfm_data.json",
                               fg_output_dir + "/mvs/scene_dense_mesh_refine_texture.ply",
                               bg_output_dir + "/mvs/scene_dense_mesh_refine_texture.ply",
                               args.output_dir + '/result_' + str_timestamp + '.ply')
        execution_time_monitor(fg_dir, "merge", time.time() - start, new=False)
    except Exception as e:
        print(e)
        fail["timestamp"].append(str_timestamp)


with open('fail.json', 'w', encoding='utf-8') as f:
    json.dump(fail, f, indent=4)
