import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares


class Image_Emulator:

    def __init__(self, path_imgs_bracketing, emulation_method="radiance", selection_method="closest", color_bayer=False, device='cpu'):

        self.device = device
        self.base_path = Path(__file__).parents[2]
        self.path_bracketing = path_imgs_bracketing

        # Collect only folder names that can be converted to float
        bracketing_folders = []
        for folder in os.listdir(self.path_bracketing):
            folder_path = self.path_bracketing / folder
            if os.path.isdir(folder_path):
                try:
                    float(folder)
                    bracketing_folders.append(folder)
                except ValueError:
                    continue
        self.bracketing_values = torch.tensor(sorted([float(f) for f in bracketing_folders]), device=self.device)

        self.emulation_method = emulation_method
        self.selection_method = selection_method
        self.color_bayer = color_bayer

        self.ICRF_x, self.ICRF_y, self.CRF_x, self.CRF_y = self.get_CRF()
        self.update_image_list()
    
    def get_CRF(self):
        # Load calibration data and convert to PyTorch tensors
        intensity_values = torch.linspace(0, 4095, 256, device=self.device)
        values_inverse_CRF = torch.tensor(np.loadtxt(self.base_path / "calibration_files" / "pcalib_inside1.txt"), 
                                         device=self.device, dtype=torch.float32)

        digital_number = intensity_values
        irradiance = values_inverse_CRF * 16.0
        irradiance[0] = 0
        irradiance[-1] = 4095

        # Store interpolation points for differentiable interpolation
        return digital_number, irradiance, irradiance, digital_number
    
    def differentiable_interp1d(self, x_points, y_points, x_query):
        """Differentiable 1D interpolation using PyTorch"""
        # Ensure x_query is within bounds
        x_query = torch.clamp(x_query, x_points[0], x_points[-1])
        
        # Find the indices for interpolation
        indices = torch.searchsorted(x_points[1:], x_query, right=False)
        indices = torch.clamp(indices, 0, len(x_points) - 2)
        
        # Get the surrounding points
        x0 = x_points[indices]
        x1 = x_points[indices + 1]
        y0 = y_points[indices]
        y1 = y_points[indices + 1]
        
        # Linear interpolation
        t = (x_query - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    def create_exposure_time_logspace(self, min_exposure_time, max_exposure_time, number_values):
        min_log = torch.log(torch.tensor(min_exposure_time, device=self.device))
        max_log = torch.log(torch.tensor(max_exposure_time, device=self.device))
        values_linspace = torch.linspace(min_log, max_log, number_values, device=self.device)
        values_logspace = torch.exp(values_linspace)
        return values_logspace

    def load_image_as_tensor(self, image_path):
        """Load image and convert to PyTorch tensor while maintaining differentiability for future operations"""
        # Use OpenCV for loading but convert to PyTorch tensor immediately
        image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
        if not self.color_bayer:
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        
        if image is None:
            return None
            
        # Convert to PyTorch tensor
        image_tensor = torch.tensor(image, device=self.device, dtype=torch.float32)
        return image_tensor

    def update_image_list(self, img_list=None):

        self.bracket_images = []
        self.bracket_images_filenames = []
        if img_list is not None:    # Use provided filenames
            for filename, bracket in zip(img_list, self.bracketing_values):
                bracket_val = float(bracket.item()) if torch.is_tensor(bracket) else float(bracket)
                image_path = self.path_bracketing / str(bracket_val) / filename
                image_tensor = self.load_image_as_tensor(image_path)
                if image_tensor is None:
                    print(f"Image {filename} not found in {self.path_bracketing / str(bracket_val)}")
                self.bracket_images.append(image_tensor)
                self.bracket_images_filenames.append(filename)
        else:                       # Use first files in each folder
            for bracket in self.bracketing_values:
                bracket_val = float(bracket.item()) if torch.is_tensor(bracket) else float(bracket)
                filename = os.listdir(self.path_bracketing / str(bracket_val))[0]
                image_path = self.path_bracketing / str(bracket_val) / filename
                image_tensor = self.load_image_as_tensor(image_path)
                if image_tensor is None:
                    print(f"Image {filename} not found in {self.path_bracketing / str(bracket_val)}")
                self.bracket_images.append(image_tensor)
                self.bracket_images_filenames.append(filename)
        self.update_saturation_levels()
        

    def update_saturation_levels(self):

        self.over_sat_levels = []
        self.under_sat_levels = []
        for bracket_idx in range(len(self.bracketing_values)):
            image = self.get_bracket_image(bracket_idx)
            under_saturated_pixels = (image == 0).sum()
            over_saturated_pixels = (image == 4094).sum()
            total_pixels = image.numel()
            self.under_sat_levels.append(float(under_saturated_pixels) / total_pixels)
            self.over_sat_levels.append(float(over_saturated_pixels) / total_pixels)
        
        self.saturation_levels = torch.tensor(self.under_sat_levels, device=self.device) + \
                                torch.tensor(self.over_sat_levels, device=self.device)


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

        # Convert tensor values to appropriate types for the dictionary
        exposure_time = float(self.bracketing_values[bracket_idx])
        
        image_dic = {"path":self.path_bracketing,
                     "filename":self.bracket_images_filenames[bracket_idx],
                     "exposure_time":exposure_time,
                     "timestamp":str(self.bracket_images_filenames[bracket_idx]).split("-")[0],
                     "target_exposure_time":target_exp_time,
                     "emulation_factor":multiplication_factor,
                     "emulated_img":emulated_image,
                     "bracket_idx":bracket_idx}
        return image_dic
    

    def select_best_image(self, target_exp_time):

        SATURATION_THRESHOLD = 0.01
        # target_exp_tensor = torch.tensor(target_exp_time, device=self.device)

        if (self.selection_method == "closer_least_sat"):
            higher_values = torch.where(self.bracketing_values >= target_exp_time)[0]
            bracket_idx = int(higher_values[0]) if len(higher_values) > 0 else -1
            if bracket_idx != 0 and bracket_idx != -1 and self.saturation_levels[bracket_idx] > self.saturation_levels[bracket_idx-1]:
                bracket_idx -= 1
        else:
            raise Exception(f"Selection method '{self.selection_method}' not implemented")
        
        return bracket_idx
    

    def linear_emulation(self, target_exp_time, bracket_idx):

        bracket_image = self.get_bracket_image(bracket_idx)
        # target_exp_tensor = torch.tensor(target_exp_time, device=self.device)
        bracket_exp = self.bracketing_values[bracket_idx]
        multiplication_factor = target_exp_time / bracket_exp
        emulated_image = bracket_image * multiplication_factor
        emulated_image = torch.clamp(emulated_image, 0, 4095)
        return emulated_image, float(multiplication_factor)
    

    def radiance_emulation(self, target_exp_time, bracket_idx):

        bracket_image = self.get_bracket_image(bracket_idx)
        bracket_exp = self.bracketing_values[bracket_idx]
        multiplication_factor = target_exp_time / bracket_exp
        
        # Use differentiable interpolation for ICRF
        radiance_image = self.differentiable_interp1d(self.ICRF_x, self.ICRF_y, bracket_image)
        emulated_radiance_image = radiance_image * multiplication_factor
        emulated_radiance_image = torch.clamp(emulated_radiance_image, 0, 4095)
        
        # Use differentiable interpolation for CRF
        emulated_image = self.differentiable_interp1d(self.CRF_x, self.CRF_y, emulated_radiance_image)
        return emulated_image, float(multiplication_factor)
    

    def get_bracket_image(self, bracket_idx):

        img = self.bracket_images[bracket_idx]
        return img
    
    def kaist_emulation(self, target_exp_time, bracket_idx):
        """Placeholder for KAIST emulation method - not implemented yet"""
        raise NotImplementedError("KAIST emulation method not implemented")
    
    def to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array for compatibility"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def enable_gradient(self):
        """Enable gradient computation for all internal tensors"""
        for i, img in enumerate(self.bracket_images):
            if torch.is_tensor(img):
                self.bracket_images[i] = img.requires_grad_(True)
    
    def disable_gradient(self):
        """Disable gradient computation for all internal tensors"""
        for i, img in enumerate(self.bracket_images):
            if torch.is_tensor(img):
                self.bracket_images[i] = img.requires_grad_(False)