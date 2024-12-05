import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.stats as sc_stats
from skimage.filters.rank import entropy
from skimage.morphology import disk
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from scipy import signal
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .drl_exposure_ctrl.env import ExposureEnv
from .drl_exposure_ctrl.agent import Actor
import torch
import pathlib

##################################################################################################################################################
class Metric():
    def __init__(self, metric_name, brightness_target=50):
        self.metric_name = metric_name
        self.brightness_target = brightness_target
        # print(f"Auto-Exposure Metric: {self.metric_name}")

        if self.metric_name == "shim":
            self.metric_class = Metric_Shim()
        elif self.metric_name == "classical":
            self.metric_class = Metric_Classical(self.brightness_target)
        elif self.metric_name == "fixed":
            self.metric_class = Metric_Fixed()
        elif self.metric_name == "kim":
            self.metric_class = Metric_Kim()
        elif self.metric_name == "zhang":
            self.metric_class = Metric_Zhang()
        elif self.metric_name == "drl_exposure_ctrl":
            self.metric_class = Metric_Drl_Exposure_Ctrl()
        elif self.metric_name == "wang":
            self.metric_class = Metric_Wang()
        else:
            raise Exception(f"Method {self.metric_name} not implemented!")
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        next_exposure_time = self.metric_class.find_next_exposure_time(img, exposure_time)
        return next_exposure_time


################################################################################################################################################
class Metric_Shim():
    """
    Shim: Auto-adjusting Camera Exposure for Outdoor Robotics using Gradient Information
    """

    def __init__(self):
        self.delta = 0.06
        self.lambda_var = 10**3
        self.N = np.log10(self.lambda_var*(1-self.delta)+1)
        self.kp = 0.2
        self.d = 0.75

        self.gamma_values = np.array([1/1.9, 1/1.5, 1/1.2, 1.0, 1.2, 1.5, 1.9])
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img_preprocess = self.img_preproccessing(img)
        img_normalize = (img_preprocess - np.min(img_preprocess))/(np.max(img_preprocess) - np.min(img_preprocess)) # Modified
        best_gamma = self.find_best_gamma_factor(img_normalize)
        next_exposure_time = self.update_fct_linear(best_gamma, exposure_time)
        return next_exposure_time
    
    def update_fct_not_linear(self, gamma, exposure_time):
        alpha = 1/2 if (gamma >= 1) else 1
        R = self.d*np.tan((2-gamma)*np.arctan(1/self.d)-np.arctan(1/self.d))+1
        next_exposure_time = (1 + alpha*self.kp*(R - 1))*exposure_time
        return next_exposure_time
    
    def update_fct_linear(self, gamma, exposure_time):
        alpha = 1/2 if (gamma >= 1) else 1
        next_exposure_time = (1 + alpha*self.kp*(1 - gamma))*exposure_time
        return next_exposure_time
    
    def find_best_gamma_factor(self, img_og):
        M = []
        for gamma in self.gamma_values:
            gamma_gray_img = self.apply_gamma(img_og, gamma)
            M.append(self.gradient_calculation_shim2018(gamma_gray_img))
        polynomial_fit = np.polyfit(self.gamma_values, M, deg=5) # f = a*x**5 + b*x**4 + c*x**3 + ... + f
        gamma_fit_values = np.linspace(np.min(self.gamma_values), np.max(self.gamma_values), 100)
        polynomial_fit_fct = np.poly1d(polynomial_fit)
        arg_max_value = np.argmax(polynomial_fit_fct(gamma_fit_values))
        best_gamma = gamma_fit_values[arg_max_value]
        return best_gamma
    
    def apply_gamma(self, img, gamma):
        resulting_img = img**(gamma)
        return resulting_img
    
    def gradient_calculation_shim2018(self, img):
        sobel_gradient_x = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=1, dy=0, ksize=3)
        sobel_gradient_y = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=0, dy=1, ksize=3)
        sobel_gradient_img = np.sqrt(sobel_gradient_x**2 + sobel_gradient_y**2)
        m_i = (sobel_gradient_img - np.min(sobel_gradient_img))/(np.max(sobel_gradient_img) - np.min(sobel_gradient_img)) # Modified

        m_i_mean = np.zeros_like(m_i)
        m_i_mean[m_i >= self.delta] = (1/self.N)*np.log10(self.lambda_var*(m_i[m_i >= self.delta] - self.delta)+1)
        M_total_img = np.sum(m_i_mean)
        return M_total_img
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        return img
    
################################################################################################################################################
class Metric_Classical():
    def __init__(self, brightness_target, proportional_factor=0.002, encoding=12):
        self.encoding = encoding
        self.proportional_factor = proportional_factor
        self.brightness_target = int((brightness_target/100)*(2**self.encoding))
        self.threshold = 15
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img_preprocess = self.img_preproccessing(img)
        mean_brightness_value = np.mean(img_preprocess)
        distance_from_target = mean_brightness_value - self.brightness_target

        if np.abs(distance_from_target) < self.threshold:
            return exposure_time
        else:
            if distance_from_target < 0:
                new_exposure = np.abs(exposure_time + (self.proportional_factor*np.abs(distance_from_target)))
                return new_exposure
            else:
                new_exposure = np.abs(exposure_time - (self.proportional_factor*np.abs(distance_from_target)))
                return new_exposure
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        return img

####################################################################################################################################

class Metric_Fixed():
    def __init__(self, number_frames_auto=3):
        self.number_frames_auto = number_frames_auto
        self.count_number_frames = 0

        self.brightness_target = 50
        self.classical_auto_exposure = Metric_Classical(self.brightness_target)
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        if (self.count_number_frames <= self.number_frames_auto):
            self.count_number_frames += 1
            next_exposure_time = self.classical_auto_exposure.find_next_exposure_time(img, exposure_time)
            return next_exposure_time
        else:
            return exposure_time


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

class Metric_Kim():
    """
    Kim: Exposure Control using Bayesian Optimization based on Entropy Weighted Image Gradient
    """

    def __init__(self):
        self.encoding = 8 # For now, makes entropy way quicker!!!
        self.sigma = 0
        self.alpha = 2**4
        self.tau = 2**2
        self.entropy_threshold = 0.6 # Find value

        self.gpr = GaussianProcessRegressor()
        self.training_exposure_time = []
        self.training_metric_values = []
        self.evaluate_exposure_time = np.linspace(0.5,35,50)
        self.previous_eta = 1
        self.size_sliding_window = 5
        self.gamma = np.log(2/0.25)
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img_preprocess = self.img_preproccessing(img)

        # Calculate Gradient and Entropy
        img_entropy = self.calculate_entropy(img_preprocess)
        img_gradient_x, img_gradient_y = self.calculate_gradient(img_preprocess)
        norm_square_img_gradient = img_gradient_x**2 + img_gradient_y**2

        weight = self.calculate_weigth(img_entropy)

        metric = self.calculate_metric(weight, norm_square_img_gradient, img_entropy)
        # print(f"Final metric value: {metric}")

        next_exposure_time = self.exposure_control_scheme(exposure_time, metric)
        # print(f"Next exposure time: {next_exposure_time} ms")
        return next_exposure_time
    
    def exposure_control_scheme(self, exposure_time, metric):
        self.training_exposure_time.insert(0,exposure_time)
        self.training_metric_values.insert(0,metric)

        mu, std_dev = self.gaussian_process()
        # plt.errorbar(self.evaluate_exposure_time, mu, yerr=std_dev, marker='o', markersize=2, alpha=0.5, label="GP")
        # plt.plot(self.training_exposure_time, self.training_metric_values, 'o', label="Actual image values")
        # plt.plot(exposure_time, metric, '*', label="Live exp time")
        # plt.legend()
        # plt.show()

        x_n = self.acquisition_function(mu, std_dev, "MAXMI")
        is_optimal = self.check_optimal(x_n, std_dev)
        if is_optimal[0]:
            if len(self.training_exposure_time) == self.size_sliding_window:
                self.manage_training_points()
            return is_optimal[1] # If time already optimal, dont change
        else:
            if len(self.training_exposure_time) == self.size_sliding_window:
                self.manage_training_points()
            return self.evaluate_exposure_time[x_n]
    
    def manage_training_points(self):
        self.training_exposure_time.pop()
        self.training_metric_values.pop()
        return
    
    def check_optimal(self, index_next, std):
        if (std[index_next] <= 5):
            optimal = True
            query_exposures = np.linspace(0.5,35,1000) 
            optimal_exposure_time = query_exposures[np.argmax(self.gpr.predict(query_exposures.reshape(-1,1)))]
            return optimal, optimal_exposure_time
        else: 
            optimal = False
            return optimal, None
    
    def gaussian_process(self):
        gp_kernel = 1.0 * RBF()
        alpha_array = np.full_like(self.training_exposure_time, 0.2)
        self.gpr = GaussianProcessRegressor(kernel=gp_kernel, alpha=alpha_array, n_restarts_optimizer=10).fit(np.array(self.training_exposure_time).reshape(-1,1), np.array(self.training_metric_values))
        gpr_prediction_mean, gpr_prediction_stdev = self.gpr.predict(np.array(self.evaluate_exposure_time).reshape(-1,1), return_std=True)
        return gpr_prediction_mean, gpr_prediction_stdev

    def acquisition_function(self, mean, std, type_acquisition):
        variance = std**2
        if type_acquisition == "MAXVAR":
            x_n = np.argmax(variance)
            # print(f"X_N: {x_n}")
        elif type_acquisition == "MAXMI":
            phi = np.sqrt(self.gamma)*(np.sqrt(variance + self.previous_eta) - np.sqrt(self.previous_eta))
            x_n = np.argmax(mean + phi)
            eta_t = self.previous_eta + variance[x_n]
            self.previous_eta = eta_t
            # print(f"Eta: {eta_t}")
        return x_n
    
    def calculate_metric(self, weight, norm_square_gradient, entropy):
        N = norm_square_gradient.shape[0]
        g = (weight*norm_square_gradient) + self.activation_function(entropy)*self.saturation_mask(entropy)*weight*(1/N)*np.sum(norm_square_gradient)
        g_sum = np.sum(g)
        return g_sum
    
    def calculate_gradient(self, img):
        sobel_gradient_x = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=1, dy=0, ksize=3)
        sobel_gradient_y = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=0, dy=1, ksize=3)
        return sobel_gradient_x.ravel(), sobel_gradient_y.ravel()
    
    def calculate_entropy(self, img):
        # img_entropy = entropy(img, disk(1))
        bins = int(2**self.encoding)
        histogram,_ = np.histogram(img.ravel(), bins=bins, range=(0,bins))
        probability = histogram / histogram.sum()
        img_entropy = -probability[img.ravel()]*np.log2(probability[img.ravel()])
        # img_entropy = entropy(probability, base=2)
        return img_entropy.ravel()
    
    def calculate_weigth(self, entropy):

        self.sigma = np.std(entropy)
        weight = (1/self.sigma)*np.exp(((entropy - np.mean(entropy))**2)/(2*(self.sigma**2)))
        weight_normalized = weight/np.sum(weight)
        return weight_normalized
    
    def activation_function(self, entropy):
        entropy = (entropy - np.min(entropy))/(np.max(entropy) - np.min(entropy))
        pi = (2/(1 + np.exp(-self.alpha*entropy + self.tau))) - 1
        return pi
    
    def saturation_mask(self, entropy):
        entropy = (entropy - np.min(entropy))/(np.max(entropy) - np.min(entropy))
        mask = np.zeros_like(entropy)
        mask[entropy < self.entropy_threshold] = 1
        return mask
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        img = (img/16.0).astype(np.uint8)
        return img


################################################################################################################################################
class Metric_Zhang():
    """
    Zhang: Active Exposure Control for Robust Visual Odometry in HDR Environments (SoftPerceptile)
    """

    def __init__(self):
        self.encoding = 8
        self.p = 0.7
        # self.p = 0.8
        self.k = 5
        # self.gamma = 1e-7
        self.gamma = 1e3
        # self.gamma = 1e4
        # self.weights = self.calculate_weight(1200*1920)
        self.weights = self.create_sine_weights(1200*1920, self.k, self.p)
        self.icrf_derivative = self.get_response_functions()

        self.brightness_target = 50
        self.classical_auto_exposure = Metric_Classical(self.brightness_target, self.encoding)
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        exposure_time = exposure_time*1000
        img_preprocess = self.img_preproccessing(img)
        mean_img = np.mean(img_preprocess)
        small_bounday = 70
        high_boundary = 190

        if ((mean_img <= small_bounday) or (mean_img >= high_boundary)):
            print("Classical Auto-Exposure")
            next_exposure_time = self.classical_auto_exposure.find_next_exposure_time(img, exposure_time)
        else:
            img_gradient = self.calculate_gradient(img_preprocess)

            # Exposure Control
            icrf_gradient = self.calculate_gradient(1.0/(self.icrf_derivative(img_preprocess)*exposure_time))

            gradient_derivative = 2*(icrf_gradient*img_gradient)
            
            arg_sorted = np.argsort(gradient_derivative)
            final_gradient = self.handle_over_under_exposed(gradient_derivative, img_preprocess.ravel())
            # final_gradient = gradient_derivative
            m_softperc_derivative = np.sum(self.weights * final_gradient[arg_sorted])

            next_exposure_time = exposure_time + self.gamma*m_softperc_derivative # Gamma 0.7

        # if next_exposure_time < 0.02: #lower than camera limit
        #     next_exposure_time = 0.02

        next_exposure_time = next_exposure_time/1000
        return next_exposure_time

    def handle_over_under_exposed(self, gradient_derivative, img):
        gradient_derivative[img == 0] = 2.0
        gradient_derivative[img == 255] = -2.0
        return gradient_derivative
    
    # def calculate_gradient(self, img):
    #     sobel_gradient_x = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=1, dy=0)
    #     sobel_gradient_y = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=0, dy=1)
    #     return sobel_gradient_x, sobel_gradient_y
    
    def calculate_gradient(self, img):
        
        gradient_x = cv2.Scharr(img.astype(np.float32), ddepth=cv2.CV_32FC1, dx=1, dy=0)
        gradient_x /= 32.0
        gradient_y = cv2.Scharr(img.astype(np.float32), ddepth=cv2.CV_32FC1, dx=0, dy=1)
        gradient_y /= 32.0
        gradient = gradient_x**2 + gradient_y**2                  
        return gradient.ravel()
    
    def calculate_weight(self, img_size):
        s = img_size
        threshold = math.floor(self.p * s)
        first_indexes = np.arange(0, threshold)

        first_indexes_second_eq = np.arange(0, s-threshold)
        last_indexes = np.arange(threshold, s)

        weight = np.zeros_like(np.arange(0,s, dtype=np.float32), dtype=np.float32)
        weight[first_indexes] = np.sin(((np.pi)/(2*threshold))*first_indexes)**self.k

        weight[last_indexes] = np.sin((np.pi/2) - (np.pi/(2*(s - threshold)))*first_indexes_second_eq)**self.k # Their equations don't work
        # weight = weight/np.linalg.norm(weight) # Sum
        weight = weight/np.sum(weight) # Sum

        return weight
    
    def create_sine_weights(self, num, order, percentile_ratio):
        """
        Create sine-based weights.
        
        Args:
            num (int): Total number of weights.
            order (float): Exponent applied to sine values.
            percentile_ratio (float): Ratio determining the split between the two parts of the sine function.
            
        Returns:
            list: Normalized sine-based weights.
        """
        if not (0.0 < percentile_ratio < 1.0):
            raise ValueError("percentile_ratio must be in the range (0, 1).")

        weights = np.zeros(num)
        
        # Determine split points and steps
        num_first = int(num * percentile_ratio)
        num_second = num - num_first

        # Generate first part: [0, pi/2]
        step_first = np.pi / 2 / (num_first - 1)
        for i in range(num_first):
            weights[i] = np.sin(i * step_first) ** order
        weights[num_first - 1] = 1.0  # Explicitly set the last value of the first part

        # Generate second part: (pi/2, pi]
        step_second = np.pi / 2 / num_second
        for i in range(num_first, num):
            weights[i] = np.sin(np.pi / 2 - (i - num_first + 1) * step_second) ** order

        # Normalize weights
        weights /= np.sum(weights)
    
        return weights.tolist()
    
    def get_response_functions(self):
        intensity_values = np.linspace(0,255,256)
        base_path = Path(__file__).parents[2]
        # values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_inside1.txt") #"pcalib_inside2.txt"
        values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_forest2024.txt") #"pcalib_inside2.txt"

        digital_number = intensity_values
        # irradiance = values_inverse_CRF*(16.0) # 12bits (2^4)
        # irradiance[0] = 0
        # irradiance[-1] = 4095

        # icrf = interp1d(digital_number, irradiance, kind='linear', fill_value=(0,4095))
        # crf = interp1d(irradiance, digital_number, kind='linear', fill_value=(0,4095))

        # icrf_derivate = np.diff(crf(np.linspace(0,4095,4097)))
        # icrf_derivate = np.diff(icrf(np.linspace(0,4095,4097))) # This is the correct one
        icrf = np.poly1d(np.polyfit(digital_number, np.log(values_inverse_CRF), 10))
        icrf_derivative = icrf.deriv()
        
        return icrf_derivative
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        img = (img/16.0).astype(np.uint8)
        return img


################################################################################################################################################
class Metric_Wang():
    """
    Wang
    """

    def __init__(self):
        self.encoding = 8
        self.lambda_factor = 5.0
        self.number_iterations = 3
        self.zeta = 0.001
        self.k = 0.5
        self.crf, self.icrf, self.icrf_derivative = self.get_response_functions()

        self.brightness_target = 50
        self.classical_auto_exposure = Metric_Classical(self.brightness_target, self.encoding)
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        exposure_time = exposure_time
        img_process = self.img_preproccessing(img)
    
        # print("-----------------------------------------------------------")
        exposure_time_opt, image_opt, gamma_search = self.heuristic_exposure_prediction(img_process, exposure_time)
        exposure_time_search = exposure_time * (1 + self.k*(gamma_search - 1))
        
        # print(f"Update Exposure Time: {(exposure_time_opt - exposure_time) * 0.2}")
        # exposure_time_search = exposure_time_opt
        
        weights = self.calculate_weights(img_process)
        gradient_img_x, gradient_img_y = self.calculate_gradient(img_process)
        gradient_opt_x, gradient_opt_y = self.calculate_gradient(1.0/(exposure_time_search * self.icrf_derivative(img_process)))
        gradient_factor = 2 * weights * (gradient_img_x * gradient_opt_x + gradient_img_y * gradient_opt_y)
        
        gradient_factor[image_opt < 30] = 2.0
        gradient_factor[image_opt > 225] = -2.0
        
        # print(f"Number pixel under 30: {np.count_nonzero(image_opt < 30)/len(image_opt.ravel())}")
        # print(f"Number pixel over 225: {np.count_nonzero(image_opt > 225)/len(image_opt.ravel())}")
        # print(f"Gradient from saturate pixels: {np.sum(gradient_factor[image_opt < 30]) + np.sum(gradient_factor[image_opt > 225])/np.sum(gradient_factor)}")
        # print(f"Total gradient: {np.sum(gradient_factor)}")
        
        # print(f"Min image optimal: {np.min(image_opt)}")
        # print(f"Max image optimal: {np.max(image_opt)}")
        # print(f"Min gradient factor: {np.min(gradient_factor)}")
        # print(f"Max gradient factor: {np.max(gradient_factor)}")
        
        gradient_factor = np.sum(gradient_factor)/np.sum(weights)
        
        # print(f"Gradient Factor: {gradient_factor}")

        # if 0.5 < np.abs(gradient_factor) < 2.1:
        #     exposure_time_refined = exposure_time_search * self.zeta * gradient_factor
            
        next_exposure_time = exposure_time_search + self.zeta * gradient_factor

        return next_exposure_time
    
    def heuristic_exposure_prediction(self, img, exposure_time, alpha=0.5, beta=2.0):
        
        # print(f"Current exposure time: {exposure_time}")
        t_opt = exposure_time
        I_opt = img
        G_opt = self.quality_metric(I_opt)
        gamma = 1
        for i in range(self.number_iterations):
            t_left = alpha * t_opt
            t_right = beta * t_opt
            I_emulated_left = self.emulate_image(img, exposure_time, t_left)
            I_emulated_right = self.emulate_image(img, exposure_time, t_right)
            G_left = self.quality_metric(I_emulated_left)
            G_right = self.quality_metric(I_emulated_right)
            
            # self.show_images(I_emulated_left, I_opt, I_emulated_right, t_left, t_opt, t_right, G_left, G_opt, G_right)
            
            if G_left >= G_opt and G_left >= G_right:
                # print(f"Left higher than optimal")
                t_opt = t_left
                I_opt = I_emulated_left
                G_opt = G_left
                gamma *= alpha 
                beta = 0.5 * (1 + beta)
            elif G_right >= G_opt:
                # print(f"Right higher than optimal")
                t_opt = t_right
                I_opt = I_emulated_right
                G_opt = G_right
                gamma *= beta
                alpha = 0.5 * (1 + alpha)
            else:
                # print(f"Optimal")
                alpha = 0.5 * (1 + alpha)
                beta = 0.5 * (1 + beta)
        
        # print(f"Optimal exposure time: {t_opt}")
        return t_opt, I_opt, gamma
    
    def show_images(self, img_left, img_opt, img_right, t_left, t_opt, t_right, G_left, G_opt, G_right):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axes[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Left (ET: {t_left})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img_opt, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Optimal (ET: {t_opt})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'Right (ET: {t_right})')
        axes[0, 2].axis('off')
        
        # Gradient images
        grad_left_x, grad_left_y = self.calculate_gradient(img_left)
        grad_left = np.sqrt(grad_left_x**2 + grad_left_y**2)
        axes[1, 0].imshow(grad_left, cmap='gray')
        axes[1, 0].set_title(f'Gradient Left ({G_left*1e-6})')
        axes[1, 0].axis('off')
        
        grad_opt_x, grad_opt_y = self.calculate_gradient(img_opt)
        grad_opt = np.sqrt(grad_opt_x**2 + grad_opt_y**2)
        axes[1, 1].imshow(grad_opt, cmap='gray')
        axes[1, 1].set_title(f'Gradient Optimal ({G_opt*1e-6})')
        axes[1, 1].axis('off')
        
        grad_right_x, grad_right_y = self.calculate_gradient(img_right)
        grad_right = np.sqrt(grad_right_x**2 + grad_right_y**2)
        axes[1, 2].imshow(grad_right, cmap='gray')
        axes[1, 2].set_title(f'Gradient Right ({G_right*1e-6})')
        axes[1, 2].axis('off')
        
        plt.show()
            
    def emulate_image(self, img_source, exp_source, exp_target):
        
        image_emulated = self.crf(exp_target/exp_source*self.icrf(img_source))
        image_emulated = np.clip(image_emulated, 0.0, 255.0).astype(np.uint8)
    
        return image_emulated
            
    def quality_metric(self, img):
        weights = self.calculate_weights(img)
        img_gradient_x, img_gradient_y = self.calculate_gradient(img)
        img_gradient = img_gradient_x**2 + img_gradient_y**2
        metric = np.sum(np.multiply(weights, img_gradient))
        
        return metric
    
    def calculate_gradient(self, img):
        gradient_x = cv2.Sobel(img.astype(np.float64), ddepth=cv2.CV_64FC1, dx=1, dy=0)
        gradient_y = cv2.Sobel(img.astype(np.float64), ddepth=cv2.CV_64FC1, dx=0, dy=1)              
        return gradient_x, gradient_y
    
    def calculate_weights(self, img):
        weight = np.zeros_like(img, dtype=np.float32)
        weight[img < 128] = 1/(1 + self.lambda_factor*np.exp((63.5 - img[img < 128])/5)) - 0.001
        weight[img >= 128] = 1/(1 + self.lambda_factor*np.exp((img[img >= 128] - 191.5)/5)) - 0.001
        
        return weight
    
    def get_response_functions(self):
        intensity_values = np.linspace(0,255,256)
        base_path = Path(__file__).parents[2]
        # values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_inside1.txt") #"pcalib_inside2.txt"
        values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_forest2024.txt") #"pcalib_inside2.txt"

        digital_number = intensity_values
        crf = np.poly1d(np.polyfit(values_inverse_CRF, digital_number, 5))
        icrf = np.poly1d(np.polyfit(digital_number, values_inverse_CRF, 5))
        icrf_derivative = np.poly1d(np.polyfit(digital_number, np.log(values_inverse_CRF), 5))
        
        # # Plot the ICRF and CRF side-by-side
        # fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # x = np.linspace(0, 255, 256)

        # Plot CRF
        # y_crf = crf(x)
        # axes[0].plot(x, y_crf, label='CRF')
        # axes[0].set_xlabel('Digital Number')
        # axes[0].set_ylabel('Irradiance')
        # axes[0].set_title('Camera Response Function (CRF)')
        # axes[0].legend()
        # axes[0].grid(True)

        # # Plot ICRF
        # y_icrf = icrf(x)
        # axes[1].plot(x, y_icrf, label='ICRF')
        # axes[1].set_xlabel('Digital Number')
        # axes[1].set_ylabel('Irradiance')
        # axes[1].set_title('Inverse Camera Response Function (ICRF)')
        # axes[1].legend()
        # axes[1].grid(True)

        # plt.tight_layout()
        # plt.show()
        
        return crf, icrf, icrf_derivative
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        img = (img/16.0).astype(np.uint8)
        return img


################################################################################################################################################   
class Metric_Drl_Exposure_Ctrl():
    def __init__(self, number_frames_auto=3):
        self.number_frames_auto = number_frames_auto
        self.count_number_frames = 0

        self.brightness_target = 50
        self.classical_auto_exposure = Metric_Classical(self.brightness_target)
        
        self.params = {'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
          'state_dim': (4, 84, 84),
          'action_dim': 1,
          'len_episode': 100000,

          'env_mode_test': True,
          'env_data_argumentation': True,
          'env_seq_filepath': "config/infer.yaml",
          'env_img_ori_h': 192,
          'env_img_ori_w': 256,
          'env_expo_lb': 50,
          'env_expo_ub': 2000000,
          'env_expo_init': 10000,

          #   'rwd_mode': "stat",
          'rwd_mode': "feat",
          'rwd_mean_target': 0.5,
          'rwd_w_flk': 0.2,
          'rwd_w_detect': 0.005,
          'rwd_w_match': 0.005,

          'sac_hidden_dim': 512
          }
        base_path = Path(__file__).parents[0]
        self.agent = Actor(self.params['state_dim'], self.params['action_dim'], self.params['sac_hidden_dim'])
        self.agent.load_state_dict(torch.load(f'{base_path}/drl_exposure_ctrl/model/actor_drl_feat_10000.pth'))
        self.agent.eval()
        
        # setup env
        self.env = ExposureEnv(None, self.params, self.params['len_episode'])
        self.s_network, _ = self.env.reset(frame_id=0)
        self.s_network = np.array([])
        self.s_reward = np.array([])
        
        # set first exposure time DRL
        self.first_frame_DRL = True
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img_network_resized = cv2.resize(img, (self.params["state_dim"][1], self.params["state_dim"][2]))
        img_reward_resized = cv2.resize(img, (self.params["env_img_ori_w"], self.params["env_img_ori_h"]))
        img_normalized = img_network_resized / 2**12
        if len(self.s_network) == 0:
            self.s_network = np.expand_dims(img_normalized, axis=0)
            self.s_reward = np.expand_dims(img_reward_resized, axis=0)
        else:
            self.s_network = np.concatenate((self.s_network, np.expand_dims(img_normalized, axis=0)), axis=0)
            self.s_reward = np.concatenate((self.s_reward, np.expand_dims(img_reward_resized, axis=0)), axis=0)
            if self.s_network.shape[0] == 5:
                self.s_network = self.s_network[1:, :]
                self.s_reward = self.s_reward[1:, :]
            
        if (self.count_number_frames <= self.number_frames_auto):
            self.count_number_frames += 1
            next_exposure_time = self.classical_auto_exposure.find_next_exposure_time(img, exposure_time)
            return next_exposure_time
        else:
            if self.first_frame_DRL:
                self.env.expo = exposure_time * 1000
                self.first_frame_DRL = False
            s_in = torch.unsqueeze(torch.tensor(self.s_network, dtype=torch.float), 0)
            a, _ = self.agent(s_in, True, False)
            a = a.data.numpy().flatten()[0]
            s_, r, done, next_exposure_time = self.env.step(a, self.s_reward)
            self.s = s_
            return next_exposure_time/1000