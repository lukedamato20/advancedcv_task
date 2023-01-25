# import keras
import keras
import shutil
import os
import numpy as np
import time
import subprocess


path = os.getcwd()
 
model_path = os.path.join(path, "yolov7_custom.pt")
image_path = os.path.join(path, "/data/testing", "imgTEST.jpg")
image_output_path = os.path.join(path, "/runs/detect", "imgDETECT.jpg")

result_path = 'C:/Users/lldam/OneDrive/UOM/3rd Year/Advanced Computer Vision/advancedcv_task/Task2and3/yolov7/runs/detect/result'

if os.path.exists(result_path) and os.path.isdir(result_path):
    shutil.rmtree(result_path)
    print("results folder refreshed... \n")

os.chdir('C:/Users/lldam/OneDrive/UOM/3rd Year/Advanced Computer Vision/advancedcv_task/Task2and3/yolov7')

detect_script = "python detect.py --weights yolov7_custom.pt --conf 0.10 --img-size 640 --source data/testing/task3-6.jpg --name result --save-txt"
print("\nRunning detect.py script...\n")
os.system(detect_script)

def save_rows(filename):
    keys = ['batch', 'x', 'y', 'val']
    dicts = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(',')
            d = dict(zip(keys, line))
            dicts.append(d)
    return dicts

os.chdir('C:/Users/lldam/OneDrive/UOM/3rd Year/Advanced Computer Vision/advancedcv_task/Task2and3/yolov7')
# Converting arrays to dictionaries
arraydict = save_rows('C:/Users/lldam/OneDrive/UOM/3rd Year/Advanced Computer Vision/advancedcv_task/Task2and3/yolov7/runs/detect/result/labels/task3-6.txt')                
print("Results file found... \n")
print("Creating JSON... \n")
# Writing bounding boxes to json file
import json
import codecs

json_file = "result.json" 
json.dump(str(arraydict), codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)
print("JSON Created!")