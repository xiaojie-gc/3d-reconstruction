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


parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='data/originals',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/gold_results',
                    help="the directory which contains the final results.")
parser.add_argument('--reconstructor', type=str, default='MvgMvsPipeline.py',
                    help="the directory which contains the reconstructor python script.")

args = parser.parse_args()

try:
    shutil.rmtree(args.fg_dir)
    shutil.rmtree(args.bg_dir)
except:
    print(".....")
# shutil.rmtree(args.output_dir)

Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

for timestamp in range(0, 21):
    str_timestamp = str(timestamp).zfill(5)
    print("current timestamp: ", str_timestamp, '-' * 50)

    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)

    Path(collect_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    # go through each camera
    inx = 0
    for image_dir in os.listdir(args.data_dir):
        # image file format = 000004.png
        img_file_name = str_timestamp + ".png"
        print("\tload image {} from camera #{}".format(img_file_name, image_dir))
        shutil.copy2(os.path.join(args.data_dir, image_dir, img_file_name), os.path.join(collect_dir, str(inx) + ".png"))
        inx += 1

    with open("time.json", "r") as jsonFile:
        time_file = json.load(jsonFile)

    time_file["timeList"].append({"model": collect_dir})

    with open('time.json', 'w', encoding='utf-8') as f:
        json.dump(time_file, f, indent=4)

    # start to run openMvg + openMvs for foreground
    print("start to reconstruct {}".format(str_timestamp))
    start = time.time()
    p = subprocess.Popen(["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output")])
    p.wait()
    if p.returncode != 0:
        break

    with open("time.json", "r") as jsonFile:
        time_file = json.load(jsonFile)

    for item in time_file["timeList"]:
        if item["model"] == collect_dir:
            item["total"] = time.time() - start
            break

    with open('time.json', 'w', encoding='utf-8') as f:
        json.dump(time_file, f, indent=4)