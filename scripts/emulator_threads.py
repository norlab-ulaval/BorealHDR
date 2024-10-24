import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import yaml
from pathlib import Path
import threading
import argparse

import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from classes.class_image_emulator import Image_Emulator
from classes.class_image_display import Display

from classes.class_auto_exposure_methods import Metric

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def create_dataframe(path_imgs, bracket_values):
    bracket_lists = []
    for bracket_value in bracket_values:
        path_bracket = Path(path_imgs, str(bracket_value))
        bracket_list = []
        for img_filename in sorted(os.listdir(path_bracket)):
            bracket_list.append(img_filename)
        bracket_lists.append(bracket_list)
    df = pd.DataFrame(bracket_lists)
    df.index = bracket_values
    return df

def over_write_experiment_folder(path):
    if os.path.isdir(path):
        os.system(f"rm -rf {path}")
    os.makedirs(path)
    return

def emulate(metric_full):

    metric = metric_full.split("-")[0]
    brightness_percentage = int(metric_full.split("-")[-1])

    display_class = Display()
    emulator_left_class = Image_Emulator(PATH_BRACKETING_IMGS_LEFT, "radiance", "closer_least_sat", COLOR)
    emulator_right_class = Image_Emulator(PATH_BRACKETING_IMGS_RIGHT, "radiance", "closer_least_sat", COLOR)

    metric_class = Metric(metric, brightness_percentage)

    dataframe_left = create_dataframe(PATH_BRACKETING_IMGS_LEFT, BRACKETING_VALUES)
    dataframe_right = create_dataframe(PATH_BRACKETING_IMGS_RIGHT, BRACKETING_VALUES)
    
    if (metric != "classical"):
        over_write_experiment_folder(SAVE_PATH / f"ae-{metric}")
    else:
        over_write_experiment_folder(SAVE_PATH / f"ae-{metric}-{brightness_percentage}")

    # INPUT: 6 folders with the trajectory for each bracket time
    # 1) For a timestamp, emulate image with current exposure time
    # 2) Benchmark -> From the metric, select next exposure time (target)
    # 3) Back to step (1), but with next timestamp

    exposure_time_target = EXPOSURE_TIME_INIT
    for timestamp in tqdm(range(0, dataframe_left.shape[1]-1)):
        emulator_left_class.update_image_list(dataframe_left.loc[:][timestamp].to_list())
        emulator_right_class.update_image_list(dataframe_right.loc[:][timestamp].to_list())
        emulated_image_left = emulator_left_class.emulate_image(exposure_time_target)
        emulated_image_right = emulator_right_class.emulate_image(exposure_time_target)

        img_left = display_class.resulting_img(emulated_image_left, bit=SAVE_DEPTH, color=COLOR)
        img_right = display_class.resulting_img(emulated_image_right, bit=SAVE_DEPTH, color=COLOR)
        
        if ACTION == "show":
            display_class.show_imgs(img_left, img_right)
        elif ACTION == "save":
            if (metric != "classical"):
                display_class.save_imgs(emulated_image_left, img_left, emulated_image_right, img_right, path=SAVE_PATH / f"ae-{metric}", index=timestamp)
            else:
                display_class.save_imgs(emulated_image_left, img_left, emulated_image_right, img_right, path=SAVE_PATH / f"ae-{metric}-{brightness_percentage}", index=timestamp)
        exposure_time_target = metric_class.find_next_exposure_time(emulated_image_left["emulated_img"], exposure_time_target)
    return


#######################################################
# VARIABLES
PARENT_PATH = Path(__file__).resolve().parent

parameters_file = PARENT_PATH / "../parameters.yaml"
with open(parameters_file, 'r') as file:
    parameters = yaml.safe_load(file)
    
parser = argparse.ArgumentParser(description="Emulate HDR images using different automatic exposure techniques.")
parser.add_argument('--dataset_folder', type=str, default=parameters["EMULATION"]["dataset_folder"], help='Path to the dataset folder.')
parser.add_argument('--experiment', type=str, default=parameters["EMULATION"]["experiment"], help='Name of the experiment.')
parser.add_argument('--depth_emulated_imgs', type=int, default=parameters["EMULATION"]["depth_emulated_imgs"], help='Bit depth of the emulated images.')
parser.add_argument('--emulated_in_color', type=bool, default=parameters["EMULATION"]["emulated_in_color"], help='Whether to emulate images in color.')
parser.add_argument('--automatic_exposure_techniques', type=str, nargs='+', default=parameters["EMULATION"]["automatic_exposure_techniques"], help='List of automatic exposure techniques to use.')
parser.add_argument('--save_or_show_emulated_imgs', type=str, choices=['save', 'show'], default=parameters["EMULATION"]["save_or_show_emulated_imgs"], help='Whether to save or show the emulated images.')
parser.add_argument('--save_path', type=str, default=parameters["EMULATION"]["save_path"], help='Path to save the emulated images.')
parser.add_argument('--exposure_time_init', type=float, default=parameters["EMULATION"]["exposure_time_init"], help='Initial exposure time.')

args = parser.parse_args()
    
DATASET_FOLDER = PARENT_PATH / Path(args.dataset_folder)
EXPERIMENT = args.experiment
SAVE_DEPTH = args.depth_emulated_imgs
COLOR = args.emulated_in_color
AE_METRIC = args.automatic_exposure_techniques
ACTION = args.save_or_show_emulated_imgs
SAVE_PATH = PARENT_PATH / Path(args.save_path) / EXPERIMENT

PATH_BRACKETING_IMGS_LEFT = DATASET_FOLDER / EXPERIMENT / "camera_left"
PATH_BRACKETING_IMGS_RIGHT = DATASET_FOLDER / EXPERIMENT / "camera_right"

BRACKETING_VALUES = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
EXPOSURE_TIME_INIT = args.exposure_time_init
#########################################################

def main():

    # Run the experiments in parallel
    threads = []
    for metric in AE_METRIC:
        t = threading.Thread(target=emulate, args=(metric,))
        threads.append(t)
        t.start()

    print("All threads started")
    for index, thread in enumerate(threads):
        thread.join()
    
    if ACTION == "save":
        graphics = Display()
        graphics.plot_exposure(SAVE_PATH)


if __name__ == "__main__":
    main()
