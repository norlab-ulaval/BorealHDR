import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import argparse
from pathlib import Path
import threading

from classes.class_image_emulator import Image_Emulator
from classes.class_image_display import Image_Display

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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Emulate from a bracketing sequence")
    parser.add_argument("-x", "--experiment", help="Folder name of the sequence", required=False, default="backpack_2023-09-25-15-05-03")
    parser.add_argument("-s", "--save_path", help="Save path", required=False, default="/home/user/code/output/emulated_images/")
    args = parser.parse_args()

    #######################################################
    # VARIABLES
    BASE_PATH = Path(__file__).absolute().parents[1]

    EMULATION_METHOD = "radiance"
    SELECTION_METHOD = "closer_least_sat"
    ACTION = "save"
    SAVE_DEPTH = 8
    COLOR = True
    AE_METRIC = [
        # "classical-30",
        "classical-50",
        # "classical-70",
        # "manual-0",
        # "gradient-0",
        # "ewg-0",
        # "softperc-0"
    ]

    EXPERIMENT = args.experiment
    SAVE_PATH = Path(args.save_path)
    if "09-25" in EXPERIMENT:
        LOCATION_ACQUISITION = "ulaval_campus"
        DATASET_PATH = Path(f"../data_sample/{LOCATION_ACQUISITION}/")
    elif "09-27" in EXPERIMENT:
        LOCATION_ACQUISITION = "belair"
        DATASET_PATH = Path(f"../dataset_mount_point/{LOCATION_ACQUISITION}/")
    elif "04-20" in EXPERIMENT:
        LOCATION_ACQUISITION = "forest_20"
        DATASET_PATH = Path(f"../dataset_mount_point/{LOCATION_ACQUISITION}/")
    elif "04-21" in EXPERIMENT:
        LOCATION_ACQUISITION = "forest_21"
        DATASET_PATH = Path(f"../dataset_mount_point/{LOCATION_ACQUISITION}/")
    else:
        print("Wrong dataset name!")
    PATH_BRACKETING_IMGS_LEFT = DATASET_PATH / EXPERIMENT / "camera_l"
    PATH_BRACKETING_IMGS_RIGHT = DATASET_PATH / EXPERIMENT / "camera_r"

    BRACKETING_VALUES = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    EXPOSURE_TIME_INIT = 4.0
    #########################################################

def emulate(metric_full):

    metric = metric_full.split("-")[0]
    brightness_percentage = int(metric_full.split("-")[-1])

    display_left_class = Image_Display("left")
    display_right_class = Image_Display("right")
    emulator_left_class = Image_Emulator(PATH_BRACKETING_IMGS_LEFT, EMULATION_METHOD, SELECTION_METHOD, COLOR)
    emulator_right_class = Image_Emulator(PATH_BRACKETING_IMGS_RIGHT, EMULATION_METHOD, SELECTION_METHOD, COLOR)

    metric_class = Metric(metric, brightness_percentage)

    dataframe_left = create_dataframe(PATH_BRACKETING_IMGS_LEFT, BRACKETING_VALUES)
    dataframe_right = create_dataframe(PATH_BRACKETING_IMGS_RIGHT, BRACKETING_VALUES)

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

        if (metric != "classical"):
            display_left_class.resulting_img(emulated_image_left, action=ACTION, path=SAVE_PATH / f"ae-{metric}", index=timestamp, bit=SAVE_DEPTH, color=COLOR)
            display_right_class.resulting_img(emulated_image_right, action=ACTION, path=SAVE_PATH / f"ae-{metric}", index=timestamp, bit=SAVE_DEPTH, color=COLOR)
        else:
            display_left_class.resulting_img(emulated_image_left, action=ACTION, path=SAVE_PATH / f"ae-{metric}-{brightness_percentage}", index=timestamp, bit=SAVE_DEPTH, color=COLOR)
            display_right_class.resulting_img(emulated_image_right, action=ACTION, path=SAVE_PATH / f"ae-{metric}-{brightness_percentage}", index=timestamp, bit=SAVE_DEPTH, color=COLOR)
        exposure_time_target = metric_class.find_next_exposure_time(emulated_image_left["emulated_img"], exposure_time_target)

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


if __name__ == "__main__":
    main()
