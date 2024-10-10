import sys
import os
from pathlib import Path
sys.path.append(Path(__file__).parents[2])

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares


class Image_Emulator:

    def __init__(self, path_imgs_bracketing, emulation_method="radiance", selection_method="closest", color_bayer=False):

        self.base_path = Path(__file__).parents[2]
        self.path_bracketing = path_imgs_bracketing
        
        self.bracketing_values = np.array(sorted([float(folder) for folder in os.listdir(self.path_bracketing)]))
        self.emulation_method = emulation_method
        self.selection_method = selection_method
        self.color_bayer = color_bayer

        self.ICRF, self.CRF = self.get_CRF()
        self.update_image_list()
    
    def get_CRF(self):

        intensity_values = np.linspace(0,4095,256)
        values_inverse_CRF = np.loadtxt(self.base_path / "calibration_files" / "pcalib_inside1.txt")

        digital_number = intensity_values
        irradiance = values_inverse_CRF*(16.0)
        irradiance[0] = 0
        irradiance[-1] = 4095

        icrf = interp1d(digital_number, irradiance, kind='linear', fill_value=(0,4095))
        crf = interp1d(irradiance, digital_number, kind='linear', fill_value=(0,4095))
        return icrf, crf
    
    def create_exposure_time_logspace(self, min_exposure_time, max_exposure_time, number_values):
        min_log = np.log(min_exposure_time)
        max_log = np.log(max_exposure_time)
        values_linspace = np.linspace(min_log, max_log, number_values)
        values_logspace = np.exp(values_linspace)
        return values_logspace

    def update_image_list(self, img_list=None):

        self.bracket_images = []
        self.bracket_images_filenames = []
        if img_list is not None:    # Use provided filenames
            for filename, bracket in zip(img_list, self.bracketing_values):
                image = cv2.imread(str(self.path_bracketing / str(bracket) / filename), cv2.IMREAD_ANYDEPTH)
                if not self.color_bayer:
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
                if(image is None):
                    print(f"Image {filename} not found in {self.path_bracketing / str(bracket)}")
                self.bracket_images.append(image)
                self.bracket_images_filenames.append(filename)
        else:                       # Use first files in each folder
            for bracket in self.bracketing_values:
                filename = os.listdir(self.path_bracketing / str(bracket))[0]
                image = cv2.imread(str(self.path_bracketing / str(bracket) / filename), cv2.IMREAD_ANYDEPTH)
                if not self.color_bayer:
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
                if(image is None):
                    print(f"Image {filename} not found in {self.path_bracketing / str(bracket)}")
                self.bracket_images.append(image)
                self.bracket_images_filenames.append(filename)
        self.update_saturation_levels()
        

    def update_saturation_levels(self):

        self.over_sat_levels = []
        self.under_sat_levels = []
        for bracket_idx in range(len(self.bracketing_values)):
            image = self.get_bracket_image(bracket_idx)
            under_saturated_pixels = image[image == 0]
            over_saturated_pixels = image[image == 4094]
            self.under_sat_levels.append(under_saturated_pixels.size/image.size)
            self.over_sat_levels.append(over_saturated_pixels.size/image.size)
            self.saturation_levels = np.array(self.under_sat_levels) + np.array(self.over_sat_levels)


    def emulate_image(self, target_exp_time):

        bracket_idx = self.select_best_image(target_exp_time)
        if bracket_idx == -1 : bracket_idx = len(self.bracketing_values)-1

        if (self.emulation_method == "linear"):
            emulated_image, multiplication_factor = self.linear_emulation(target_exp_time, bracket_idx)
        elif (self.emulation_method == "radiance"):
            emulated_image, multiplication_factor = self.radiance_emulation(target_exp_time, bracket_idx)
        elif (self.emulation_method == "kaist"):
            emulated_image, multiplication_factor = self.kaist_emulation(target_exp_time, bracket_idx)
        elif (self.emulation_method == "no_factor"):
            emulated_image, multiplication_factor = self.get_bracket_image(bracket_idx), 1

        image_dic = {"path":self.path_bracketing,
                     "filename":self.bracket_images_filenames[bracket_idx],
                     "exposure_time":self.bracketing_values[bracket_idx],
                     "timestamp":str(self.bracket_images_filenames[bracket_idx]).split("-")[0],
                     "target_exposure_time":target_exp_time,
                     "emulation_factor":multiplication_factor,
                     "emulated_img":emulated_image,
                     "bracket_idx":bracket_idx}
        return image_dic
    

    def select_best_image(self, target_exp_time):

        SATURATION_THRESHOLD = 0.01

        if (self.selection_method == "closer_least_sat"):
            higher_values = np.where(self.bracketing_values >= float(target_exp_time))[0]
            bracket_idx = higher_values[0] if len(higher_values) > 0 else -1
            if bracket_idx != 0 and bracket_idx != -1 and self.saturation_levels[bracket_idx] > self.saturation_levels[bracket_idx-1]:
                bracket_idx -= 1
        else:
            raise Exception(f"Selection method '{self.selection_method}' not implemented")
        
        return bracket_idx
    

    def linear_emulation(self, target_exp_time, bracket_idx):

        bracket_image = self.get_bracket_image(bracket_idx)
        multiplication_factor = target_exp_time/float(self.bracketing_values[bracket_idx])
        emulated_image = (bracket_image*multiplication_factor).astype(np.uint16)
        emulated_image = np.clip(emulated_image, 0, 4095)
        return emulated_image, multiplication_factor
    

    def radiance_emulation(self, target_exp_time, bracket_idx):

        bracket_image = self.get_bracket_image(bracket_idx)
        multiplication_factor = target_exp_time/float(self.bracketing_values[bracket_idx])
        radiance_image = self.ICRF(bracket_image)
        emulated_radiance_image = radiance_image*multiplication_factor
        emulated_radiance_image = np.clip(emulated_radiance_image, 0, 4095)
        emulated_image = self.CRF(emulated_radiance_image.astype(np.float64)).astype(np.uint16)
        return emulated_image, multiplication_factor
    

    def get_bracket_image(self, bracket_idx):

        img = self.bracket_images[bracket_idx]
        return img