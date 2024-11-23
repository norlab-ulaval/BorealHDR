import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import yaml

from classes.class_image_emulator import Image_Emulator
from classes.class_image_display import Display

# Constants
OUTPUT_PATH = Path("/home/user/data/results/emulated_images")
CAMERA_FOLDER = "camera_left"  # or "camera_right"
METHOD_EXPOSURE_TIMES_FILE = "exposure_times_left.csv"
BRACKETING_VALUES = np.array([2.0**i for i in range(6)])
METHOD_FOLDERS = [
    "classical-30",
    "classical-50",
    "classical-70",
    "drl_exposure_ctrl",
    "fixed",
    "kim",
    "shim",
    "zhang",
]
TARGET_EXPOSURE_TIMES = np.linspace(1, 50, 15)


def find_folder(start_path: str, search_folder: str) -> Path:
    """
    Recursively search for a folder starting from `start_path`.

    :param start_path: Root directory to start the search.
    :param search_folder: Folder name to search for.
    :return: Path to the found folder or None if not found.
    """
    for root, dirs, _ in os.walk(start_path):
        if search_folder in dirs:
            return Path(root) / search_folder
    return None


def find_closest_value(csv_file: Path, target_timestamp: int) -> float:
    """
    Find the value corresponding to the closest timestamp in a CSV file.

    :param csv_file: Path to the CSV file.
    :param target_timestamp: Target timestamp to search for.
    :return: Value corresponding to the closest timestamp.
    """
    df = pd.read_csv(csv_file, header=None, names=["timestamp", "value"])
    closest_idx = (df["timestamp"] - target_timestamp).abs().idxmin()
    return df.loc[closest_idx, "value"]


def create_dataframe(
    path_imgs: Path, timestamp: int, bracket_values: list
) -> pd.DataFrame:
    """
    Create a DataFrame mapping bracket values to the closest image filename.

    :param path_imgs: Path to the images folder.
    :param timestamp: Target timestamp to match.
    :param bracket_values: List of bracketing values.
    :return: Pandas DataFrame mapping bracket values to filenames.
    """
    bracket_lists = []
    for bracket_value in bracket_values:
        path_bracket = path_imgs / str(bracket_value)
        closest_img = None
        closest_diff = float("inf")

        for img_filename in sorted(os.listdir(path_bracket)):
            img_timestamp = int(img_filename.split(".")[0])
            diff = abs(img_timestamp - timestamp)
            if diff < closest_diff:
                closest_diff = diff
                closest_img = img_filename

        bracket_lists.append([closest_img] if closest_img else [])
    df = pd.DataFrame(bracket_lists, index=bracket_values)
    return df


def main(input_paths: list, results: str, timestamp: int, exact_timestamps: bool):
    """
    Main function to emulate images based on exposure times and save results.

    :param input_paths: List of input data paths.
    :param results: Path to the results folder.
    :param timestamp: Target timestamp for emulation.
    :param exact_timestamps: Whether to use exact timestamps or approximate.
    """
    display = Display()

    for path in input_paths:
        path = path[0]  # Extract string from list
        camera_path = find_folder(path, CAMERA_FOLDER)
        if camera_path is None:
            print(f"Could not find {CAMERA_FOLDER} folder in {path}")
            continue

        results_path = find_folder(results, camera_path.parts[-2])
        if results_path is None:
            print(f"Could not find results folder for {camera_path.parts[-2]}")
            continue

        emulator = Image_Emulator(camera_path, "radiance", "HIGHERNOSAT", True)

        method_exposure_dict = {}
        for method in METHOD_FOLDERS:
            method_time_file = results_path / method / METHOD_EXPOSURE_TIMES_FILE
            value = find_closest_value(method_time_file, timestamp)

            if exact_timestamps:
                method_exposure_dict[method] = float(value)
            else:
                closest_exposure = TARGET_EXPOSURE_TIMES[
                    np.abs(TARGET_EXPOSURE_TIMES - value).argmin()
                ]
                method_exposure_dict[method] = float(closest_exposure)

        output_path = OUTPUT_PATH / path.split("/")[-1]
        display.verify_if_folder_exist_or_create_it(output_path)

        print(f"Saving data to {output_path}")
        with open(output_path / "method_exposure_times.yaml", "w") as f:
            yaml.dump(method_exposure_dict, f)

        for target_exposure_time in TARGET_EXPOSURE_TIMES:
            emulated_image = emulator.emulate_image(target_exposure_time)
            image = display.resulting_img(emulated_image, bit=8, color=True)
            filepath_out = output_path / f"{target_exposure_time:.1f}.png"
            cv2.imwrite(str(filepath_out), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that takes a timestamp and an experiment \
        and emulates images, outputing the emulated images together \
        with a file with closest matching exposure times for each method."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="One or more input data paths.",
        type=str,
        action="append",
        nargs="+",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        required=True,
        type=int,
        help="Timestamp to emulate, in [ns].",
    )
    parser.add_argument(
        "--results", required=True, help="Path to the results folder.", type=str
    )
    parser.add_argument(
        "--exact_timestamps",
        help="Use exact timestamps for exposure emulation. Defaults to nearest neighbor.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    main(args.input, args.results, args.timestamp, args.exact_timestamps)
