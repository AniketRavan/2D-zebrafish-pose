import numpy as np
from construct_model import f_x_to_model_evaluation

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def evaluate_images(im_real, pose_prediction):
    buffer_x = 101 - im_real.shape[1]
    offset_x = int(buffer_x / 2)
    buffer_y = 101 - im_real.shape[0]
    offset_y = int(buffer_y / 2)
    pt = pose_prediction
    pt[0,0:12] = pt[0,0:12] - offset_x 
    pt[1,0:12] = pt[1,0:12] - offset_y 
    vec = np.diff(pt, n = 1, axis = 1)
    theta_0 = np.arctan2(vec[1, 0], vec[0, 0])
    theta_i = np.diff(np.arctan2(vec[1, 0:9], vec[0,0:9]))
    t = normalize_data([0, 0.8, 0.8 + np.cumsum(np.ones((1, 9), dtype = np.float64))])
    pt_for_interpolation = np.array([np.mean(pt[:, 10 : 12], axis = 1), pt[:, 0 : 10]])
    pt = interparc(t, pt_for_interpolation(0, :), pt_for_interpolation(1, :)).T

    pt = pt[:, 1:11]
    x[0] = pt[0, 0]
    x[1] = pt[1, 0]
    x[2] = theta_0
    x[3:11] = theta_i
    seglen = np.mean(np.sqrt(vec[0, 0 : 9] ** 2 + vec[1, 0 : 9) ** 2))
    im_gray, pt = f_x_to_model_evaluation(x, seglen, 0, im_real.shape[1], im_real.shape[0])
    im_masked = mask_real_image(im_real, im_gray)
    corr_coeff = np.corrcoef(im_masked[im_masked > 0], im_gray[im_masked > 0])
    return corr_coeff

