import cv2
import numpy as np
import os
from time import time

class Image_Display():
    def __init__(self, camera):
        if camera == "left":
            self.save_folder = "images_l"
        elif camera == "right":
            self.save_folder = "images_r"
        return
    
    def show_imgs_og_and_simulated(self, img_og, img_simulated):
        img_original = cv2.cvtColor(img_og, cv2.COLOR_BayerRG2RGB)
        img_simulated = cv2.cvtColor(img_simulated, cv2.COLOR_BayerRG2RGB)
        imgs_side_by_side = np.hstack(((img_original/16.0).astype('uint8'), (img_simulated/16.0).astype('uint8')))
        cv2.imshow("Left: Image Original / Right: Image Simulated", imgs_side_by_side)
        cv2.waitKey()
    
    def save_imgs_side_by_side(self, img_og, imgs_simulated):
        img_original = cv2.cvtColor(img_og, cv2.COLOR_BayerRG2RGB)
        if (len(imgs_simulated) == 2):
            img_simulated_small_bracket = cv2.cvtColor(imgs_simulated[0]["simulated_img"], cv2.COLOR_BayerRG2RGB)
            img_simulated_long_bracket = cv2.cvtColor(imgs_simulated[1]["simulated_img"], cv2.COLOR_BayerRG2RGB)
            imgs_side_by_side = np.hstack(((img_simulated_small_bracket/16.0).astype('uint8'), (img_original/16.0).astype('uint8'), (img_simulated_long_bracket/16.0).astype('uint8')))
            img_name = "target_"+str(imgs_simulated[0]["target_exposure_time"])+"ms-small-factor_"+str(imgs_simulated[0]["simulation_factor"])+"-long-factor_"+str(imgs_simulated[1]["simulation_factor"])
        else:
            img_bracketing = cv2.cvtColor(imgs_simulated[0]["simulated_img"], cv2.COLOR_BayerRG2RGB)
            imgs_side_by_side = np.hstack(((img_original/16.0).astype('uint8'), (img_bracketing/16.0).astype('uint8')))
            img_name = "bracket"+str(imgs_simulated[0]["exposure_time"])
        cv2.imwrite(f"../../illumination_calibration/simulated_images/{img_name}.png", imgs_side_by_side)

    def save_imgs_raw_og_and_simulated(self, img_og, imgs_simulated):
        for img in imgs_simulated:
            img_name = "target_"+str(img["target_exposure_time"])+"ms-factor_"+str(img["simulation_factor"])
            cv2.imwrite(f"../../illumination_calibration/simulated_images/{img_name}.tif", img["simulated_img"])
        img_name = "ground_truth_target_"+str(imgs_simulated[0]["target_exposure_time"])+"ms"
        cv2.imwrite(f"../../illumination_calibration/simulated_images/{img_name}.tif", img_og)

    def show_img(self, img, title):
        img_to_show = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        img_to_show = (img_to_show/16.0).astype('uint8')
        cv2.imshow(title, img_to_show)
        cv2.waitKey()

    def resulting_img(self, img, action="show", path="", index=0, bit=8, color=False):
        if bit == 8:
            img_to_show = (img["emulated_img"]/16.0).astype('uint8')
            if color:
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BAYER_RG2RGB)
        elif bit == 12:
            img_to_show = img["emulated_img"]
        else:
            print("Wrong bit format. Should be 8 or 12.")
            return

        if action == "show":
            cv2.imshow("Image", img_to_show)
            cv2.waitKey(33)
        elif action == "save":
            self.verify_if_folder_exist_or_create_it(path)
            complete_path = path / self.save_folder
            self.verify_if_folder_exist_or_create_it(complete_path)
            
            save_image_filename = f"{index:0>5d}"
            cv2.imwrite(str(complete_path / f"{save_image_filename}.png"), img_to_show)
            f = open(path / f"times_{self.save_folder}.txt","a")
            msg = save_image_filename+" "+str(img["timestamp"].split(".")[0])+" "+str(img["target_exposure_time"])+"\n"
            f.write(msg)
            f.close()

    def result_one_img(self, img, exposure_time, action="show", path="", index=0, color_bayer=False):
        if color_bayer:
            img["emulated_img"] = cv2.cvtColor(img["emulated_img"], cv2.COLOR_BAYER_RG2BGR)
        img_to_show = (img["emulated_img"]/16.0).astype('uint8')

        # text = f"Exp.Time: {exposure_time:.1f} ms"
        # origin = (50,50)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 2
        # color = (0,0,255)
        # thickness = 2
        # img_to_show = cv2.putText(img_to_show, text, origin, font, font_scale, color, thickness)

        if action == "show":
            cv2.imshow("Image", img_to_show)
            cv2.waitKey(33)
        elif action == "save":
            self.verify_if_folder_exist_or_create_it(path)
            complete_path = path / self.save_folder
            self.verify_if_folder_exist_or_create_it(complete_path)
            
            save_image_filename = f"img_{exposure_time:.2f}"
            cv2.imwrite(str(complete_path / f"{save_image_filename}.png"), img_to_show)

    def verify_if_folder_exist_or_create_it(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        return