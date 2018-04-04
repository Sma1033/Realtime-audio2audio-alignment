"""
Created on Thu Jul 13 00:23:37 2017 @ author: Sma1033
use this function to find bast match local dtw path
"""
import numpy as np
import librosa


def align_2_target(X, m, new_length):
    X_adaptive = np.zeros([m, np.int(new_length)])  # create a matrix samm size as target
    X_target_ratio = np.float(np.int(new_length)) / X.shape[1]  
    for i in range(0, X_adaptive.shape[1]):
        X_adaptive[:, i] = X[:, np.int(i/X_target_ratio)]
    return X_adaptive


def quicksort(arr):
    """ Quicksort a list
    :type arr: list
    :param arr: List to sort
    :returns: list -- Sorted list
    """
    if not arr:
        return []

    pivots = [x for x in arr if x == arr[0]]
    lesser = quicksort([x for x in arr if x < arr[0]])
    greater = quicksort([x for x in arr if x > arr[0]])

    return lesser + pivots + greater


def alignment_dtw(gt_cens_all,       \
                  gt_tempo,          \
                  gt_start_frame,    \
                  gt_end_frame,      \
                  input_clip_cens,   \
                  input_clip_tempo,  \
                  tempo_adj_man,     \
                  clip_length,       \
                  cens_fps,          \
                  tempo_change_ratio_limit_dtw,  \
                  tempo_change_ratio_limit_clip, \
                  tempo_max_song,    \
                  tempo_min_song,    \
                  ):

    # find estemated ground truth frame from all song frame
    est_gt_cens = gt_cens_all[:, gt_start_frame:gt_end_frame]

    # calculate scaled input
    gt_audio_length = est_gt_cens.shape[1]
    input_audio_row_num = input_clip_cens.shape[0]
    scaled_input = align_2_target(input_clip_cens, input_audio_row_num, gt_audio_length)

    # run DTW(scaled input, estimated GT audio) here
    cost_matrix, wp = librosa.dtw(scaled_input, est_gt_cens,  \
                        global_constraints=True,    \
                        band_rad=tempo_change_ratio_limit_dtw,  \
                        subseq=True)


    pre_reg_x = wp[:, 1]
    pre_reg_x = pre_reg_x[::-1]
    pre_reg_x_with_coef = np.vstack([pre_reg_x, np.ones(len(pre_reg_x))]).T
    pre_reg_y = wp[:, 0]
    pre_reg_y = pre_reg_y[::-1]

    reg_slope, reg_coef = np.linalg.lstsq(pre_reg_x_with_coef, pre_reg_y)[0]
    reg_residuals = np.linalg.lstsq(pre_reg_x_with_coef, pre_reg_y)[1]

    start_chp = int(0.03 * len(pre_reg_x))
    end_chp = int(0.97 * len(pre_reg_x))
    x_change_point = []  # find all x change point
    for i in range(start_chp, end_chp) :
        if pre_reg_x[i+1] > pre_reg_x[i]:
            x_change_point.append(i+1)

    slope_x_length = np.int(len(x_change_point)*0.032)
    #slope_x_length = 20
    slope_list = []
    for j in range(0, len(x_change_point)-slope_x_length-1):
        delta_x = np.float(pre_reg_x[x_change_point[j+slope_x_length]] - pre_reg_x[x_change_point[j]])
        delta_y = np.float(pre_reg_y[x_change_point[j+slope_x_length]] - pre_reg_y[x_change_point[j]])
        if (delta_x > 0) and (delta_y > 0) :
            slop_n = delta_y / delta_x
            slope_list.append(slop_n)


    sorted_slope_list = quicksort(slope_list)

    slope_list_len = len(sorted_slope_list)
    start_list = int(slope_list_len * 0.25)
    end_list = int(slope_list_len * 0.75)
    final_slope = np.mean(sorted_slope_list[start_list:end_list])


    middle_x = np.int((pre_reg_x[len(pre_reg_x)-1]+pre_reg_x[0])/2)

    middle_x_index = 0
    for k in range(0, len(pre_reg_x)) :
        if (pre_reg_x[k] == middle_x):
            middle_x_index = k

    old_line_y = reg_slope*pre_reg_x[middle_x_index] + reg_coef
    new_line_y = final_slope*pre_reg_x[middle_x_index] + reg_coef
    line_dy_center = old_line_y - new_line_y

    zzz_dtw_input_is_faster = 0; zzz_dtw_input_is_slower = 0;
    
    if (final_slope*est_gt_cens.shape[1] + reg_coef + line_dy_center > est_gt_cens.shape[1]):
        zzz_dtw_input_is_slower = 1
    else:
        zzz_dtw_input_is_faster = 1

    zzz_dtw_cal_tempo_ratio = est_gt_cens.shape[1] / np.float(final_slope*est_gt_cens.shape[1] + reg_coef + line_dy_center) 


    # set change ratio limit
    zzz_dtw_cal_tempo_ratio = min((1.0+tempo_change_ratio_limit_clip), zzz_dtw_cal_tempo_ratio)
    zzz_dtw_cal_tempo_ratio = max(1.0/(1+tempo_change_ratio_limit_clip), zzz_dtw_cal_tempo_ratio)        
    #print (zzz_dtw_cal_tempo_ratio)
    
    # manually overwrite tempo
    zzz_dtw_cal_tempo_ratio = zzz_dtw_cal_tempo_ratio * (1.0 + float(tempo_adj_man)/100.0)

    zzz_dtw_cal_input_tempo = np.float(input_clip_tempo) * zzz_dtw_cal_tempo_ratio

    # force output tempo in a range
    zzz_dtw_cal_input_tempo = max(tempo_min_song , zzz_dtw_cal_input_tempo)
    zzz_dtw_cal_input_tempo = min(tempo_max_song, zzz_dtw_cal_input_tempo)

    zzz_est_gt_endframe = gt_start_frame + np.int(est_gt_cens.shape[1]*zzz_dtw_cal_tempo_ratio)

    zzz_pre_reg_x = pre_reg_x
    zzz_pre_reg_y = pre_reg_y
    
    zzz_final_reg_slope = final_slope
    
    zzz_line_offset = reg_coef + line_dy_center

    zzz_reg_residuals = reg_residuals
    
    # calculate total cost value
    wp_length = wp.shape[0]
    X_start = wp[wp_length-1, 0]
    X_end = wp[0,0]    
    Y_start = wp[wp_length-1, 1]
    Y_end = wp[0,1]    
    total_best_path_cost = abs(cost_matrix[X_end, Y_end] - cost_matrix[X_start, Y_start])
    
    zzz_dtw_cost = total_best_path_cost
    
    zzz_gt_length = est_gt_cens.shape[1]

    
    return  zzz_pre_reg_x,            \
            zzz_pre_reg_y,            \
            zzz_final_reg_slope,      \
            zzz_line_offset,          \
            zzz_reg_residuals,        \
            zzz_gt_length,            \
            zzz_dtw_cost,             \
            zzz_dtw_cal_input_tempo,  \
            zzz_dtw_cal_tempo_ratio,  \
            zzz_est_gt_endframe,      \
            zzz_dtw_input_is_slower,  \
            zzz_dtw_input_is_faster           


