import cv2
import numpy as np
import os
from pathlib import Path

import matplotlib
import threading

# Simple solution: Always use Agg backend for thread safety
# matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

class Display():
    def __init__(self, use_matplotlib=True, display_mode="display_only"):
        """
        Initialize Display class
        
        Args:
            use_matplotlib: Whether to use matplotlib for rendering
            display_mode:
                - "display_only": Only try to display (may fail in threads)
        """
        self.use_matplotlib = use_matplotlib
        self.display_mode = display_mode
        self.fig = None
        self.ax = None
        self.im = None
        self.is_main_thread = threading.current_thread() is threading.main_thread()
        self.frame_counter = 0
    
        return

    def verify_if_folder_exist_or_create_it(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        return
    
    def resulting_img(self, img, bit=8, color=False):
        emulated_img = img["emulated_img"]
        # Check if emulated_img is a PyTorch tensor
        if hasattr(emulated_img, 'detach') and hasattr(emulated_img, 'cpu'):
            emulated_img = emulated_img.detach().cpu().numpy()
        if bit == 8:
            img_to_show = (emulated_img / 16.0).astype('uint8')
            if color:
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BAYER_RG2RGB)
        elif bit == 12:
            img_to_show = emulated_img
        else:
            print("Wrong bit format. Should be 8 or 12.")
            return
        
        return img_to_show
            
    def show_imgs(self, img_l, img_r):
        imgs_side_by_side = np.hstack((img_l, img_r))
        imgs_side_by_side = cv2.resize(imgs_side_by_side, (0, 0), fx=0.4, fy=0.4)
        
        # Convert BGR to RGB for matplotlib
        imgs_side_by_side_rgb = cv2.cvtColor(imgs_side_by_side, cv2.COLOR_BGR2RGB)

        if self.use_matplotlib:
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.im = self.ax.imshow(imgs_side_by_side_rgb)
            else:
                self.im.set_array(imgs_side_by_side_rgb)
            plt.pause(0.001)

        return
    
    
    def save_imgs(self, img_l, img_to_show_l, img_r, img_to_show_r, action="show", path="", index=0):
        self.verify_if_folder_exist_or_create_it(path)
        for camera, img, img_to_show in zip(["images_left", "images_right"], [img_l, img_r], [img_to_show_l, img_to_show_r]):

            complete_path = path / camera
            self.verify_if_folder_exist_or_create_it(complete_path)
            
            save_image_filename = f"{index:0>5d}"
            cv2.imwrite(str(complete_path / f"{save_image_filename}.png"), img_to_show)
            f = open(path / f"times_{camera}.txt","a")
            msg = save_image_filename+" "+str(img["timestamp"]).split(".")[0]+" "+str(img["target_exposure_time"].detach().cpu().numpy())+"\n"
            f.write(msg)
            f.close()
            
    def plot_exposure(self, path):
        for experiment in os.listdir(path):
            if not os.path.isdir(path / experiment):
                continue
            f = open(path / experiment / "times_images_left.txt","r")
            lines = f.readlines()
            f.close()
            
            timestamps = []
            exposure_time_values = []
            for line in lines:
                timestamps.append(int(line.split(" ")[1]))
                exposure_time_values.append(float(line.split(" ")[2]))
            duration = np.array([(t - timestamps[0])*1e-9 for t in timestamps])
            exposure_time_values = np.array(exposure_time_values)
            plt.plot(duration, exposure_time_values, label=experiment)
        plt.ylabel('Exposure time (ms)')
        plt.xlabel('Duration (s)')
        plt.title('Exposure time for each metric')
        plt.legend()
        
        plt.savefig(path / "exposure_times.png")
        return
