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

##################################################################################################################################################
class Metric():
    def __init__(self, metric_name, brightness_target=50):
        self.metric_name = metric_name
        self.brightness_target = brightness_target
        # print(f"Auto-Exposure Metric: {self.metric_name}")

        if self.metric_name == "gradient":
            self.metric_class = Metric_Gradient()
        elif self.metric_name == "classical":
            self.metric_class = Metric_Classical(self.brightness_target)
        elif self.metric_name == "manual":
            self.metric_class = Metric_Manual()
        elif self.metric_name == "ewg":
            self.metric_class = Metric_Entropy_Weighted_Gradient()
        elif self.metric_name == "softperc":
            self.metric_class = Metric_Active_Exposure_HDR_Softperc()
        # elif self.metric_name == "perc":
        #     self.metric_class = Metric_Active_Exposure_HDR_Perc()
        elif self.metric_name == "noise":
            self.metric_class = Metric_Noise_Aware()
        else:
            raise Exception(f"Method {self.metric_name} not implemented!")
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        next_exposure_time = self.metric_class.find_next_exposure_time(img, exposure_time)
        return next_exposure_time


################################################################################################################################################
class Metric_Gradient():
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
    def __init__(self, brightness_target, proportional_factor=0.002):
        self.encoding = 12
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

class Metric_Manual():
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

class Metric_Entropy_Weighted_Gradient():
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
class Metric_Active_Exposure_HDR_Softperc():
    """
    Zhang: Active Exposure Control for Robust Visual Odometry in HDR Environments (SoftPerceptile)
    """

    def __init__(self):
        self.encoding = 12
        self.p = 0.8
        self.k = 5
        self.gamma = 1e-6
        self.weights = self.calculate_weight(1200*1920)
        self.icrf, self.crf, self.icrf_derivative = self.get_response_functions()

        self.brightness_target = 50
        self.classical_auto_exposure = Metric_Classical(self.brightness_target)
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img_preprocess = self.img_preproccessing(img)
        mean_img = np.mean(img_preprocess)
        small_bounday = (70/256)*2**self.encoding
        high_boundary = (190/256)*2**self.encoding

        if ((mean_img <= small_bounday) or (mean_img >= high_boundary)):
            # print(f"Classical: {mean_img}")
            next_exposure_time = self.classical_auto_exposure.find_next_exposure_time(img, exposure_time)
            # print(f"Next exposure time: {next_exposure_time:.2f} ms")
        else:
            # print(f"Softperc: {mean_img}")
            img_derivative_x, img_derivative_y = self.custom_gradient_float(img_preprocess)

            # Exposure Control
            icrf_derivative_x, icrf_derivative_y = self.custom_gradient_float(1.0/(self.icrf_derivative(img_preprocess)*exposure_time))

            gradient_derivative = 2*((img_derivative_x * icrf_derivative_x) + (img_derivative_y * icrf_derivative_y))
            arg_sorted = np.argsort(gradient_derivative.ravel())
            m_softperc_derivative = np.sum(self.weights * gradient_derivative.ravel()[arg_sorted])

            # plt.hist(gradient_derivative.ravel()[arg_sorted], bins=1000)
            # plt.show()

            # print(f"Factor: {self.gamma*m_softperc_derivative}")
            next_exposure_time = exposure_time + self.gamma*m_softperc_derivative
            # print(f"Next exposure time: {next_exposure_time:.2f} ms")
        if next_exposure_time < 0.02:
            next_exposure_time = 0.02
            # print(f"Next exposure time: {next_exposure_time:.2f} ms")
        return next_exposure_time
    
    def calculate_gradient(self, img):
        sobel_gradient_x = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=1, dy=0)
        sobel_gradient_y = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=0, dy=1)
        return sobel_gradient_x, sobel_gradient_y
    
    def custom_gradient_float(self, img):
        scharr_kernel_x = np.array([[3, 0, -3],
                                    [10, 0, -10],
                                    [3, 0, -3]], dtype=np.float32)
        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0, 0, 0],
                                    [-3, -10, -3]], dtype=np.float32)

        sobel_kernel_x = np.array([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=np.float32)
        sobel_kernel_y = np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=np.float32)

        simple_kernel_x = np.array([1, 0, -1], dtype=np.float32).reshape((1,3))
        simple_kernel_y = np.array([[1],
                                    [0],
                                    [-1]], dtype=np.float32)
        
        gradient_x = signal.convolve2d(img, scharr_kernel_x, boundary='symm', mode='same')
        gradient_y = signal.convolve2d(img, scharr_kernel_y, boundary='symm', mode='same')
        return np.float128(gradient_x), np.float128(gradient_y)
    
    def calculate_weight(self, img_size):
        s = img_size
        threshold = math.floor(self.p * s)
        first_indexes = np.arange(0, threshold)

        first_indexes_second_eq = np.arange(0, s-threshold)
        last_indexes = np.arange(threshold, s)

        weight = np.zeros_like(np.arange(0,s, dtype=np.float32), dtype=np.float32)
        weight[first_indexes] = np.sin(((np.pi)/(2*threshold))*first_indexes)**self.k

        weight[last_indexes] = np.sin((np.pi/2) - (np.pi/(2*(s - threshold)))*first_indexes_second_eq)**self.k # Their equations are not true
        weight = weight/np.linalg.norm(weight)

        # plt.plot(weight)
        # plt.xlabel("Pixel index")
        # plt.ylabel("Weight")
        # plt.show()
        return weight
    
    def get_response_functions(self):
        intensity_values = np.linspace(0,4095,256)
        base_path = Path(__file__).parents[2]
        values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_inside1.txt") #"pcalib_inside2.txt"

        digital_number = intensity_values
        irradiance = values_inverse_CRF*(16.0)
        irradiance[0] = 0
        irradiance[-1] = 4095

        icrf = interp1d(digital_number, irradiance, kind='linear', fill_value=(0,4095))
        crf = interp1d(irradiance, digital_number, kind='linear', fill_value=(0,4095))

        icrf_derivate = np.diff(icrf(np.linspace(0,4095,4097)))
        icrf_derivative = UnivariateSpline(np.linspace(0,4095,4096), icrf_derivate)#, fill_value=(0,4095))
        return icrf, crf, icrf_derivative
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        return img


################################################################################################################################################
class Metric_Noise_Aware():
    """
    Shin: Camera Exposure Control for Robust Robot Vision with Noise-Aware Image Quality Assessment
    """
    def __init__(self):
        self.gamma = 0.06 #from article
        self.lambda_var = 10**3 #from article
        self.N = np.log10(self.lambda_var*(1-self.delta)+1)

        self.N_c = 100 #from article
        self.K_g = 2 #from article
        self.encoding = 12
        self.delta_gradient = 1
        self.tau_l = 15*16 #from article
        self.tau_h = 235*16 #from article
        self.alpha = 0.4 #from article
        self.beta = 0.4 #from article
        self.epsilon = 1.7 #from article
        return
    
    def find_next_exposure_time(self, img, exposure_time):
        img = self.img_preproccessing(img)
        img_normalize = (img - np.min(img))/(np.max(img) - np.min(img))
        L_gradient, image_gradient = self.gradient_mapping(img_normalize)
        L_entropy = self.entropy_metric(img_normalize)
        sigma_noise = self.noise_metric(img_normalize, image_gradient)

        metric_f = self.alpha*L_gradient + (1-self.alpha)*L_entropy - self.beta*sigma_noise
        # print(f"Final metric: {metric_f}")

        initial_simplex = self.evaluate_initial_simplex(img, exposure_time)
        next_exposure_time = self.nelder_mead(initial_simplex)

        raise ValueError("Stop. Not finish implementation")
        return next_exposure_time
    
    def evaluate_initial_simplex(self, img, et):
        mean_image = np.mean(img)
        if ((0 <= mean_image) and (mean_image <= ((2**self.encoding)/2))):
            h = self.epsilon*(1 - mean_image/(2**self.encoding - 1))
        else:
            h = -1*(1/self.epsilon)*(mean_image/(2**self.encoding - 1))

        x = et*(1 + h) #Pas de e_i a cause pas de gain (1D)
        return x
    
    def noise_metric(self, img, gradient):
        homogeneous_mask = np.zeros_like(img)
        unsaturated_mask = np.zeros_like(img)
        kernel_M = np.array([[1, -2, 1],
                             [-2, 4, -2],
                             [1, -2, 1]])
        image_noise = np.abs(signal.convolve2d(img, kernel_M, boundary='symm', mode='same'))
        homogeneous_mask[gradient <= self.delta_gradient] = 1
        unsaturated_mask[self.tau_l <= img] = 1
        unsaturated_mask[img <= self.tau_h] = 1
        image_valid = homogeneous_mask * unsaturated_mask * image_noise
        N_s = np.count_nonzero(image_valid)
        metric_noise = np.sqrt(np.pi/2)*(1/N_s)*np.sum(image_valid)
        return metric_noise
    
    def entropy_metric(self, img):
        bins = int(2**self.encoding)
        histogram,_ = np.histogram(img.ravel(), bins=bins, range=(0,bins))
        probability = histogram / histogram.sum()
        entropy_value = sc_stats.entropy(probability, base=2)
        return entropy_value
    
    def gradient_mapping(self, img):
        sobel_gradient_x = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=1, dy=0, ksize=3)
        sobel_gradient_y = cv2.Sobel(img, ddepth=cv2.CV_16UC1, dx=0, dy=1, ksize=3)
        sobel_gradient_img = np.sqrt(sobel_gradient_x**2 + sobel_gradient_y**2)
        g_i = (sobel_gradient_img - np.min(sobel_gradient_img))/(np.max(sobel_gradient_img) - np.min(sobel_gradient_img))

        img_height = g_i.shape[0]
        img_width = g_i.shape[1]
        grid_height = int(img_height/self.N_c)
        grid_width = int(img_width/self.N_c)
        tiles = []
        for i in range(0,self.N_c):
            for j in range(0,self.N_c):
                tiles.append(g_i[j*grid_height:(j+1)*grid_height, i*grid_width:(i+1)*grid_width])

        G_j = []
        for tile in tiles:
            g_i_mean = np.zeros_like(tile)
            g_i_mean[tile >= self.gamma] = (1/self.N)*np.log10(self.lambda_var*(tile[tile >= self.gamma] - self.gamma)+1)
            G_j.append(np.sum(g_i_mean))

        mean_G_j = np.mean(G_j)
        std_G_j = np.std(G_j)
        return (self.K_g * mean_G_j/std_G_j), g_i
    
    def img_preproccessing(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
        return img



################################################################################################################################################
# class Metric_Active_Exposure_HDR_Perc():
#     """
#     Zhang: Active Exposure Control for Robust Visual Odometry in HDR Environments (Perceptile)
#     """

#     def __init__(self):
#         self.encoding = 12
#         self.p = 0.9
#         self.gamma = 5e-6
#         self.icrf, self.crf, self.icrf_derivative = self.get_response_functions()
#         return
    
#     def find_next_exposure_time(self, img, exposure_time):
#         img = self.img_preproccessing(img)

#         # Exposure Control
#         img_derivative_x, img_derivative_y = self.custom_gradient_float(img)
#         icrf_derivative_x, icrf_derivative_y = self.custom_gradient_float(1.0/(self.icrf_derivative(img)*exposure_time))

#         gradient_derivative = 2*((img_derivative_x * icrf_derivative_x) + (img_derivative_y * icrf_derivative_y))
#         m_perc_derivative = np.percentile(gradient_derivative.ravel(), self.p)

#         # print(f"M_perc_derivative: {m_perc_derivative:.2f}")
#         next_exposure_time = exposure_time + self.gamma*m_perc_derivative
#         # print(f"Next_et: {next_exposure_time:.2f} ms")
#         return next_exposure_time
    
#     def custom_gradient_float(self, img):
#         scharr_kernel_x = np.array([[3, 0, -3],
#                                     [10, 0, -10],
#                                     [3, 0, -3]], dtype=np.float32)
#         scharr_kernel_y = np.array([[3, 10, 3],
#                                     [0, 0, 0],
#                                     [-3, -10, -3]], dtype=np.float32)

#         sobel_kernel_x = np.array([[1, 0, -1],
#                                    [2, 0, -2],
#                                    [1, 0, -1]], dtype=np.float32)
#         sobel_kernel_y = np.array([[1, 2, 1],
#                                    [0, 0, 0],
#                                    [-1, -2, -1]], dtype=np.float32)

#         simple_kernel_x = np.array([1, 0, -1], dtype=np.float32).reshape((1,3))
#         simple_kernel_y = np.array([[1],
#                                     [0],
#                                     [-1]], dtype=np.float32)
        
#         gradient_x = signal.convolve2d(img, scharr_kernel_x, boundary='symm', mode='same')
#         gradient_y = signal.convolve2d(img, scharr_kernel_y, boundary='symm', mode='same')
#         return np.float128(gradient_x), np.float128(gradient_y)
    
#     def get_response_functions(self):
#         intensity_values = np.linspace(0,4095,256)
#         base_path = Path(__file__).parents[2]
#         values_inverse_CRF = np.loadtxt(base_path / "calibration_files" / "pcalib_outside.txt")

#         digital_number = intensity_values
#         irradiance = values_inverse_CRF*(16.0)
#         irradiance[0] = 0
#         irradiance[-1] = 4095

#         icrf = interp1d(digital_number, irradiance, kind='linear', fill_value=(0,4095))
#         crf = interp1d(irradiance, digital_number, kind='linear', fill_value=(0,4095))

#         icrf_derivate = np.diff(icrf(np.linspace(0,4095,4097)))
#         icrf_derivative = UnivariateSpline(np.linspace(0,4095,4096), icrf_derivate)#, fill_value=(0,4095))
#         return icrf, crf, icrf_derivative
    
#     def img_preproccessing(self, image):
#         img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
#         return img




# class Metric_Entropy():
#     def __init__(self):
#         self.encoding = 12
#         self.last_entropy = 0
#         self.delta_entropy_threshold = 0.1
#         self.exposure_time_stepsize = 1
#         return
    
#     def find_next_exposure_time(self, img, exposure_time):
#         print("################")
#         print(f"Actual exposure time: {exposure_time} ms")
#         img = self.img_preproccessing(img)
#         entropy_value = self.calculate_entropy(img)
#         change_entropy = entropy_value - self.last_entropy
#         print(f"Image entropy: {entropy_value}\nChange of entropy: {change_entropy}")
#         if np.abs(change_entropy) < self.delta_entropy_threshold:
#             return exposure_time
#         else:
#             if change_entropy > 0:
#                 self.last_entropy = entropy_value
#                 print(f"Increase to: {exposure_time + self.exposure_time_stepsize} ms")
#                 return exposure_time + self.exposure_time_stepsize
#             else:
#                 self.last_entropy = entropy_value
#                 print(f"Decrease to: {exposure_time - self.exposure_time_stepsize} ms")
#                 return exposure_time - self.exposure_time_stepsize
    
#     def img_preproccessing(self, image):
#         # img = cv2.imread(image, cv2.IMREAD_ANYDEPTH)
#         img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2GRAY)
#         return img

#     def calculate_entropy(self, img):
#         bins = int(2**self.encoding)
#         histogram,_ = np.histogram(img.ravel(), bins=bins, range=(0,bins))
#         probability = histogram / histogram.sum()
#         img_entropy = entropy(probability, base=2)
#         return img_entropy