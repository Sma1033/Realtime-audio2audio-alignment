"""
Created on Thu Jul 13 00:23:37 2017 @ author: Sma1033
"""
import time
import multiprocessing
from multiprocessing import Queue
import numpy as np
import soundfile
import librosa
import alignment_dtw as adtw
import dill
import os
import pandas as pd


# define DTW cauculations parameters here
class dtw_constant(object):
    def __init__(self):
        self.dtw_length = 6.0
        self.inc_step_length = 0.15
        self.dtw_workers = 8
        
        self.filter_length = 2
        self.tempo_change_ratio_limit_dtw = 0.30
        self.tempo_change_ratio_limit_clip = 0.30
        self.tempo_max_song = 120.0
        self.tempo_min_song = 30.0
        self.gt_tempo = 60.0


# define chroma transfer parameters here
class chroma_constant(object):
    def __init__(self):
        self.frame_hop_length = 256
        self.lowest_start_freq = 64
        self.n_chroma_bin_in_octave = 24
        self.n_octave_overtone_analysis = 2
        self.n_octave_stack = 6
        self.time_filter_n = 4
        self.pitch_filter_n = 3
        self.chroma_mul_const = 4.0
        self.re_run_transfer = 0


# define multi-octave chroma function
def chroma_n_octave_func(audio_data,                    \
                         samp_rate,                     \
                         frame_hop_length,              \
                         lowest_start_freq,             \
                         n_chroma_bin_in_octave,        \
                         n_octave_overtone_analysis,    \
                         n_octave_stack,                \
                         ):

    chroma_single_octave = range(0, n_octave_stack)

    # 2^0.5 = 1.41421 , 2^0.33333 = 1.25992
    octave_overlap_parameter = 1.5
    oo_ratio = 2**(1.0/octave_overlap_parameter)
    for x in range(n_octave_stack) :
        chroma_single_octave[x] = librosa.feature.chroma_cens(y=audio_data,                           \
                                                              sr=samp_rate,                           \
                                                              hop_length=frame_hop_length,            \
                                                              n_chroma=n_chroma_bin_in_octave,        \
                                                              n_octaves=n_octave_overtone_analysis,   \
                                                              fmin=lowest_start_freq * (oo_ratio**x), \
                                                              win_len_smooth=1,                       \
                                                              )
        if (x == 0) :
            chroma_n_octave = chroma_single_octave[x]
        else:
            if chroma_n_octave.shape[1] > 0 and chroma_single_octave[x].shape[1] > 0 :
                chroma_n_octave = np.vstack([chroma_n_octave, chroma_single_octave[x]])
            else :
                print ("Error !! array shapes are un-equal !!")
                print (chroma_n_octave.shape[1])
                print (chroma_single_octave[x].shape[1])

    return chroma_n_octave


###   define single DTW worker here   ###
def dtw_single_worker(saved_pickle_file_path,                            \
                      gt_tempo,                                          \
                      dtw_single_worker_gt_in_start_frame,               \
                      dtw_single_worker_gt_in_end_frame,                 \
                      input_clip_array_q,                                \
                      samp_rate_i,                                       \
                      dtw_single_worker_input_clip_tempo,                \
                      dtw_single_worker_tempo_adj_man,                   \
                      clip_length_5s,                                    \
                      gt_cens_fps,                                       \
                      dtw_single_worker_est_gt_endframe,                 \
                      cal_input_tempo,                                   \
                      dtw_single_worker_calculation_time,                \
                      dtw_single_worker_alignment_cost,                  \
                      dtw_single_worker_job_5s_obj_done_flag,            \
                      dtw_single_worker_kill_sig,                        \
                      dtw_single_worker_run_flag,                        \
                      dtw_single_worker_job_corresponding_time_index,    \
                      dtw_single_worker_is_alive,                        \
                      dtw_single_worker_formal_run,                      \
                      ) :
    
    # process is running, set running flag = 1
    dtw_single_worker_is_alive.value = 1
    #dtw_single_worker_job_5s_obj_done_flag.value = 1
    while(dtw_single_worker_kill_sig.value == 0) :
        if (dtw_single_worker_run_flag.value == 1) :  # run init. by worker head.
            
            ### run dtw jobs here ###
            dtw_segment_start_time = time.time()
            # load all ref audio data from pickle file
            with open(saved_pickle_file_path, 'rb') as saved_pickle_file:
                ref_audio_pkg = dill.load(saved_pickle_file)
        
            gt_cens_all = ref_audio_pkg.audio_data_chroma_cens
        
            # input_clip_data = np.array(mp_q_input_clip_data[:])   
            if (input_clip_array_q.qsize() != 0) :
                input_clip_data = input_clip_array_q.get()
            else :
                input_clip_data = np.zeros((12,500))
            
            c_constant = chroma_constant()
            # time axis smoothing
            time_filter_n = c_constant.time_filter_n
            # pitch axis smoothing
            pitch_filter_n = c_constant.pitch_filter_n

            # converter chroma to pandas dataframe for time and pitch filter
            input_clip_data_temp1 = pd.DataFrame(input_clip_data)
            input_clip_data_temp2 = input_clip_data_temp1.rolling(window=time_filter_n, 
                                                                  axis=1,center=True, 
                                                                  min_periods=1).mean()

            input_clip_cens = np.array(input_clip_data_temp2.rolling(window=pitch_filter_n,
                                                                     axis=0,center=True, 
                                                                     min_periods=1).mean())
            # increase dtw compare distance for high energy note
            input_clip_cens = input_clip_cens * c_constant.chroma_mul_const

            gt_start_frame = int(dtw_single_worker_gt_in_start_frame.value)
            gt_end_frame = int(dtw_single_worker_gt_in_end_frame.value)
            input_clip_tempo = dtw_single_worker_input_clip_tempo.value
            tempo_adj_man = dtw_single_worker_tempo_adj_man.value
            
            # import DTW parameters
            d_contant = dtw_constant()
            tempo_change_ratio_limit_dtw = d_contant.tempo_change_ratio_limit_dtw
            tempo_change_ratio_limit_clip = d_contant.tempo_change_ratio_limit_clip
            tempo_max_song = d_contant.tempo_max_song
            tempo_min_song = d_contant.tempo_min_song
            
            # cauculate DTW result
            dtw_pre_reg_x,        \
            dtw_pre_reg_y,        \
            dtw_final_reg_slope,  \
            dtw_line_offset,      \
            dtw_reg_residuals,    \
            dtw_gt_length,        \
            dtw_dtw_cost,         \
            dtw_cal_input_tempo,  \
            dtw_cal_tempo_ratio,  \
            dtw_est_gt_endframe,  \
            dtw_input_is_slower,  \
            dtw_input_is_faster = adtw.alignment_dtw(gt_cens_all,      \
                                                     gt_tempo,         \
                                                     gt_start_frame,   \
                                                     gt_end_frame,     \
                                                     input_clip_cens,  \
                                                     input_clip_tempo, \
                                                     tempo_adj_man,    \
                                                     clip_length_5s,   \
                                                     gt_cens_fps,      \
                                                     tempo_change_ratio_limit_dtw,  \
                                                     tempo_change_ratio_limit_clip, \
                                                     tempo_max_song,    \
                                                     tempo_min_song)
                                                     
            
            dtw_single_worker_est_gt_endframe.value = dtw_est_gt_endframe
            cal_input_tempo.value = dtw_cal_input_tempo
            dtw_single_worker_alignment_cost.value = dtw_dtw_cost

            dtw_single_worker_calculation_time.value = time.time() - dtw_segment_start_time
            #let host know there's a workd done in this worker   
            if (dtw_single_worker_formal_run.value == 1) :
                dtw_single_worker_job_5s_obj_done_flag.value = 1
            dtw_single_worker_formal_run.value = 0
            # let host know this worker is not running
            dtw_single_worker_run_flag.value = 0
            
        else :   ## worker in idle state, process stays here ##           
            time.sleep(0.01)
            
    # process is end, set alive flag = 0
    dtw_single_worker_is_alive.value = 0

###   define single DTW worker block ends.   ###



#def cut_5s_data(sound_data_i, current_time, samp_rate_i, clip_length_5s) :
#    data_in_end = np.int( (current_time*samp_rate_i)-1 )
#    data_in_start = max(np.int( data_in_end - clip_length_5s*samp_rate_i ), 0)
#    latest_input_5s = sound_data_i[data_in_start:data_in_end,]
#    return latest_input_5s


def time_2_5s_dtw_index(ref_time, job_time_list, time_step_5s) :
    job_list_index = 0
    for x in range(len(job_time_list)) :
        if np.abs(job_time_list[x] - ref_time) < 0.25 * time_step_5s:
            job_list_index = x
    return job_list_index


class ref_audio_data:
    def __init__(self, file_path):
        self.audio_data, self.audio_samp_rate = soundfile.read(file_path)

        audio_data = self.audio_data
        samp_rate = self.audio_samp_rate

        # import chroma calculation constant
        c_constant = chroma_constant()
        
        frame_hop_length = c_constant.frame_hop_length
        lowest_start_freq = c_constant.frame_hop_length 
        n_chroma_bin_in_octave = c_constant.n_chroma_bin_in_octave
        n_octave_overtone_analysis = c_constant.n_octave_overtone_analysis
        n_octave_stack = c_constant.n_octave_stack

        ref_audio_data_temp0 = chroma_n_octave_func(audio_data,     \
                                samp_rate,                          \
                                frame_hop_length,                   \
                                lowest_start_freq,                  \
                                n_chroma_bin_in_octave,             \
                                n_octave_overtone_analysis,         \
                                n_octave_stack,                     \
                                )

        c_constant = chroma_constant()

        # time axis smoothing
        time_filter_n = c_constant.time_filter_n
        # pitch axis smoothing
        pitch_filter_n = c_constant.pitch_filter_n

        # converter chroma to pandas dataframe
        ref_audio_data_temp1 = pd.DataFrame(ref_audio_data_temp0)

        ref_audio_data_temp2 = ref_audio_data_temp1.rolling(window=time_filter_n, 
                                                                  axis=1,center=True, 
                                                                  min_periods=1).mean()

        self.audio_data_chroma_cens = np.array(ref_audio_data_temp2.rolling(window=pitch_filter_n,
                                                                     axis=0,center=True, 
                                                                     min_periods=1).mean())
        
        # increase dtw compare distance for high energy note
        self.audio_data_chroma_cens = self.audio_data_chroma_cens * c_constant.chroma_mul_const
        
        self.audio_data_cens_fps = self.audio_data_chroma_cens.shape[1]/(np.float(len(self.audio_data))/self.audio_samp_rate)
        
##### ref audio transer function ends here #####


def dtw_mission_control(mp_program_run,                        \
                        mp_dtw_control_ready,                  \
                        mp_input_array,                        \
                        mp_sync_saved_sampling_frame,          \
                        mp_dtw_segment_calculation_time,       \
                        mp_tempo_report,                       \
                        mp_position_current,                   \
                        mp_position_next_5s,                   \
                        mp_total_done_5s_dtw_job,              \
                        mp_metronome_tempo,                    \
                        mp_tempo_adj_man,                      \
                        mp_sampling_segment_len,               \
                        mp_dtw_job_started,                    \
                        mp_est_gt_time,                        \
                        mp_est_gt_endframe,                    \
                        mp_gt_cens_fps,                        \
                        mp_dtw_ref_song_end,                   \
                        mp_dtw_control_exit,                   \
                        ):    
    
    print "DTW Mission Control Start..."
    
    # import all chroma constant
    c_constant = chroma_constant()


    # set ref-audio file paths here
    ref_audio_file_path = os.getcwd() + "\\" + "audio\\original.wav"
    saved_pickle_file_path = os.getcwd() + "\\" + "saved_chroma_data.pickle"


    # remove file if re-transfer is needed
    if c_constant.re_run_transfer == 1 :
        with open(saved_pickle_file_path, 'wb') as saved_pickle_file:
            dill.dump("Empty", saved_pickle_file, protocol=2)
        os.remove(saved_pickle_file_path)
        print ("Old file is deleted !")

    # load saved pickle data
    try:
        with open(saved_pickle_file_path, 'rb') as saved_pickle_file:
            # reload audio data from pickle #
            ref_audio_pkg = dill.load(saved_pickle_file)
            print "[info] : Pickle file load successfully !!"
    except IOError:
        print('[Error] : Can\'t open picke file, re-run transfer now')
        ref_audio_pkg = ref_audio_data(ref_audio_file_path)
        # save audio data to pickle #
        with open(saved_pickle_file_path, 'wb') as saved_pickle_file:                 
            dill.dump(ref_audio_pkg, saved_pickle_file, protocol=2)


    gt_cens_fps = ref_audio_pkg.audio_data_cens_fps
    mp_gt_cens_fps.value = ref_audio_pkg.audio_data_cens_fps
    samp_rate_i = ref_audio_pkg.audio_samp_rate

    d_contant = dtw_constant()
    gt_tempo = d_contant.gt_tempo   
    clip_length_5s = d_contant.dtw_length
    time_step_5s = d_contant.inc_step_length

    ### init. single dtw workers ###
    total_num_workers = d_contant.dtw_workers
    dtw_single_worker_job_corresponding_time_index = range(total_num_workers)
    dtw_single_worker_gt_in_start_frame = range(total_num_workers)
    dtw_single_worker_gt_in_end_frame = range(total_num_workers)
    dtw_single_worker_input_clip_array_q = range(total_num_workers)
    dtw_single_worker_input_clip_tempo = range(total_num_workers)
    dtw_single_worker_tempo_adj_man = range(total_num_workers)
    dtw_single_worker_est_gt_endframe = range(total_num_workers)
    dtw_single_worker_cal_input_tempo = range(total_num_workers)
    dtw_single_worker_calculation_time = range(total_num_workers)
    dtw_single_worker_alignment_cost = range(total_num_workers)
    dtw_single_worker_job_5s_obj_done_flag = range(total_num_workers)
    dtw_single_worker_kill_sig = range(total_num_workers)
    dtw_single_worker_run_flag = range(total_num_workers)
    dtw_single_worker_is_alive = range(total_num_workers)
    dtw_single_worker_formal_run = range(total_num_workers)
    dtw_worker_obj = range(total_num_workers)
    
    for worker_idx in range(total_num_workers):
        dtw_single_worker_gt_in_start_frame[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_gt_in_end_frame[worker_idx] = multiprocessing.Value('i', 1000)
        dtw_single_worker_input_clip_array_q[worker_idx] = Queue()
        dtw_single_worker_input_clip_tempo[worker_idx] = multiprocessing.Value('d', 0.0)
        dtw_single_worker_tempo_adj_man[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_est_gt_endframe[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_cal_input_tempo[worker_idx] = multiprocessing.Value('d', 0.0)
        dtw_single_worker_calculation_time[worker_idx] = multiprocessing.Value('d', 0.0)
        dtw_single_worker_alignment_cost[worker_idx] = multiprocessing.Value('d', 0.0)
        dtw_single_worker_job_5s_obj_done_flag[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_kill_sig[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_run_flag[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_job_corresponding_time_index[worker_idx] = multiprocessing.Value('i', 0)        
        dtw_single_worker_is_alive[worker_idx] = multiprocessing.Value('i', 0)
        dtw_single_worker_formal_run[worker_idx] = multiprocessing.Value('i', 0)
        dtw_worker_obj[worker_idx] = multiprocessing.Process(target = dtw_single_worker,                                  \
                                                    args = (saved_pickle_file_path,                                       \
                                                            gt_tempo,                                                     \
                                                            dtw_single_worker_gt_in_start_frame[worker_idx],              \
                                                            dtw_single_worker_gt_in_end_frame[worker_idx],                \
                                                            dtw_single_worker_input_clip_array_q[worker_idx],             \
                                                            samp_rate_i,                                                  \
                                                            dtw_single_worker_input_clip_tempo[worker_idx],               \
                                                            dtw_single_worker_tempo_adj_man[worker_idx],                  \
                                                            clip_length_5s,                                               \
                                                            gt_cens_fps,                                                  \
                                                            dtw_single_worker_est_gt_endframe[worker_idx],                \
                                                            dtw_single_worker_cal_input_tempo[worker_idx],                \
                                                            dtw_single_worker_calculation_time[worker_idx],               \
                                                            dtw_single_worker_alignment_cost[worker_idx],                 \
                                                            dtw_single_worker_job_5s_obj_done_flag[worker_idx],           \
                                                            dtw_single_worker_kill_sig[worker_idx],                       \
                                                            dtw_single_worker_run_flag[worker_idx],                       \
                                                            dtw_single_worker_job_corresponding_time_index[worker_idx],   \
                                                            dtw_single_worker_is_alive[worker_idx],                       \
                                                            dtw_single_worker_formal_run[worker_idx],                     \
                                                            )                                                             \
                                                    )
        dtw_worker_obj[worker_idx].daemon = True
        dtw_worker_obj[worker_idx].start()
        time.sleep(0.4)
    run_job_index = 0
    ### single dtw workers init done. ###


    # wait for all workers be ready
    show_refersh = 1.0
    while_loop_runed = 0
    total_alive_worker = 0

    while(total_alive_worker != total_num_workers) :
        # check alive worker result
        total_alive_worker = 0
        for x in range(total_num_workers) :
            if (dtw_single_worker_is_alive[x].value == 1) :
                total_alive_worker += 1

        # show lateset worker info
        mod_time = (time.time() % show_refersh)
        if (while_loop_runed == 0) :
            if mod_time < (show_refersh * 0.5) :
                #print ("[info] wait for workers ready : {}/{}".format(total_alive_worker, total_num_workers))
                while_loop_runed = 1
        else : # loop already runned , reset loop
            if mod_time > (show_refersh * 0.5) :
                while_loop_runed = 0
            time.sleep(0.05)
    ################################################

    # tell main monitor DTW process is ready
    mp_dtw_control_ready.value = 1
    print ("[info] all workers are ready now")


    # avoid empty chroma transfer
    while(mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value < clip_length_5s * 0.5):
        time.sleep(0.1)
    print ("Sound card input buffer is not empty,  Start fist input chroma conversion...")

    # start the first chroma transfer    
    converted_data_n = np.int(mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value * samp_rate_i)
    c_data_n = np.array(mp_input_array[0 : converted_data_n])
    c_data_unit = mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value

    frame_hop_length = c_constant.frame_hop_length
    lowest_start_freq = c_constant.frame_hop_length #32.7
    n_chroma_bin_in_octave = c_constant.n_chroma_bin_in_octave
    n_octave_overtone_analysis = c_constant.n_octave_overtone_analysis
    n_octave_stack = c_constant.n_octave_stack
    
    input_audio_data_temp0 = chroma_n_octave_func(c_data_n,      \
                             samp_rate_i,                        \
                             frame_hop_length,                   \
                             lowest_start_freq,                  \
                             n_chroma_bin_in_octave,             \
                             n_octave_overtone_analysis,         \
                             n_octave_stack,                     \
                             )

    chroma_cens_array = input_audio_data_temp0



    # run a dtw worker every 0.2 sec.
    refersh_period = 0.20
    refresh_count = 0
    while_loop_runed = 0

    # wait for main monitor give command to start
    while ( (mp_program_run.value == 0) and (mp_dtw_control_exit.value == 0) ) :
        # run non-formal job beforhand to avoid slow start jam
        mod_time = (time.time() % refersh_period)
        if (while_loop_runed == 0) :
            if mod_time < (refersh_period * 0.5) :
                input_extra_end_position = np.int(mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value * samp_rate_i)
                input_extra_start_position = np.int(c_data_unit * samp_rate_i)                
                latest_extra_data = np.array(mp_input_array[input_extra_start_position : input_extra_end_position])
                c_data_unit = mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value

                # convert input audio to chroma
                input_audio_data_temp0 = chroma_n_octave_func(latest_extra_data,      \
                                         samp_rate_i,                        \
                                         frame_hop_length,                   \
                                         lowest_start_freq,                  \
                                         n_chroma_bin_in_octave,             \
                                         n_octave_overtone_analysis,         \
                                         n_octave_stack,                     \
                                         )

                # prepare data for workers
                chroma_cens_array = np.hstack([chroma_cens_array, input_audio_data_temp0])
                end = chroma_cens_array.shape[1]
                start = max(0, np.int(end - gt_cens_fps * clip_length_5s))
                latest_input_chroma = chroma_cens_array[:, start:end]

                while(True) :
                    if (dtw_single_worker_is_alive[run_job_index].value == 1) and  \
                       (dtw_single_worker_run_flag[run_job_index].value == 0) and  \
                       (dtw_single_worker_kill_sig[run_job_index].value == 0) and  \
                       (dtw_single_worker_job_5s_obj_done_flag[run_job_index].value == 0):
                       # start sending data to worker from here
                            dtw_single_worker_gt_in_start_frame[run_job_index].value = 0
                            dtw_single_worker_gt_in_end_frame[run_job_index].value = np.int(d_contant.dtw_length * gt_cens_fps)
                            dtw_single_worker_input_clip_array_q[run_job_index].put(latest_input_chroma)
                            dtw_single_worker_input_clip_tempo[run_job_index].value = gt_tempo
                            dtw_single_worker_tempo_adj_man[run_job_index].value = 0
                            dtw_single_worker_formal_run[run_job_index].value = 0
                            dtw_single_worker_job_corresponding_time_index[run_job_index].value = 0
                                
                            # data sync is finished, start run jobs   
                            dtw_single_worker_run_flag[run_job_index].value = 1                        
                                
                            if (run_job_index == total_num_workers - 1):
                                run_job_index = 0
                            else :
                                run_job_index += 1
                            break
                            
                    else : # switch to next worker
                        if (run_job_index == total_num_workers - 1):
                            run_job_index = 0
                        else :
                            run_job_index += 1                
                while_loop_runed = 1
                
                # show waiting message for every 5 * refersh_period Sec.
                refresh_count += 1
                if ((refresh_count+1) % 3) == 0 :
                    if ((refresh_count/3)-1 > 0) :
                        print ("[info.] {} Initialization is done, Press \"p\" key on GUI to start alignment.".format((refresh_count/3)-1))
                
        else : # loop already runned , reset loop
            if mod_time > (refersh_period * 0.5) :
                while_loop_runed = 0
            time.sleep(0.05)


    #prelude_total_beat = 6.0
    # use mp_metronome_tempo info to calculate prelude length
    #beat_length = (60.0 / mp_metronome_tempo.value) # how long(Sec.) is the period for 1 beat
    #prelude_length = np.int(beat_length * prelude_total_beat * 10.0) / 10.0  # 6-beat time length (around 6.5 Sec.)
    
    prelude_length = 6.0

    # save start time stamp
    start_time = time.time() - prelude_length - 0.10

    elapse_time = time.time() - start_time

    # start DTW job parameters Init.
    song_length = 600          # use this number to creat list
    dtw_5s_start = prelude_length + 0.05   # define the first dtw job


    #### do 5s job initialization here ####
    # set job run time list for 5s job
    job_list_5s = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()
    # job_5s_obj = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()
    # use this flag to check if this job is done
    job_5s_obj_done_flag = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()    
    # use list to store series result    
    est_gt_endframe = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()
    # sendout data for monitor
    cal_input_tempo = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()    
    gt_in_start_frame = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()    
    gt_in_end_frame = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()    
    input_clip_tempo = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist() 
    dtw_segment_calculation_time = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()
    dtw_alignment_cost = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()

    for x in range(0, len(job_list_5s)) :  # set this done flag to 1 if it's done (default : 0)
        job_5s_obj_done_flag[x] = 0     
        est_gt_endframe[x] = 0   # estimated corsponding GT end frame
        cal_input_tempo[x] = 0.0   # estimate tempo
        dtw_segment_calculation_time[x] = 0.0   # estimate tempo
        dtw_alignment_cost[x] = 0.0

        
    job_list_5s_not_finished = 1
    job_list_5s_marker = 0  
    do_5s_job = 0

    default_input_tempo = mp_metronome_tempo.value
    latest_5s_input_tempo_avg = default_input_tempo 

    # create list to store dtw result filter input value
    filter_len = d_contant.filter_length
    latest_5s_input_tempo = range(filter_len)
    latest_done_5s_job_flag = range(filter_len)
    input_clip_tempo_temp = range(filter_len)
    gt_in_end_frame_temp = range(filter_len)
    gt_in_start_frame_temp = range(filter_len)
    dtw_alignment_cost_temp = range(filter_len)
    for x in range(0, filter_len) : 
        latest_5s_input_tempo[x] = default_input_tempo
        latest_done_5s_job_flag[x] = -1
        input_clip_tempo_temp[x] = 0
        gt_in_end_frame_temp[x] = 0
        gt_in_start_frame_temp[x] = 0
        dtw_alignment_cost_temp[x] = 0
                        
    total_done_5s_dtw_job_old = 0
    
    # "show lateset info" part parameters
    #show_refersh_sec = time_step_5s    
    while_loop_runed = 0
    
    print ("start formal job !")

    # main loop run repeatedly here
    while(mp_dtw_control_exit.value == 0) :
        ############################################
        ## lateset information are collected here ##
        ############################################        
        elapse_time = time.time() - start_time

        total_alive_worker = 0
        total_busy_worker = 0
        total_idle_worker = 0
        for x in range(total_num_workers) :
            if (dtw_single_worker_is_alive[x].value == 1) :
                total_alive_worker += 1
                if (dtw_single_worker_run_flag[x].value == 1):
                    total_busy_worker += 1
                else :
                    total_idle_worker += 1

        # collect calculation result
        for x in range(total_num_workers) :
            if (dtw_single_worker_job_5s_obj_done_flag[x].value == 1) :
                #print x
                corresponding_time_index = dtw_single_worker_job_corresponding_time_index[x].value
                #print corresponding_time_index
                est_gt_endframe[corresponding_time_index] = dtw_single_worker_est_gt_endframe[x].value
                #dtw_single_worker_est_gt_endframe[x].value = 0
                cal_input_tempo[corresponding_time_index] = dtw_single_worker_cal_input_tempo[x].value
                #dtw_single_worker_cal_input_tempo[x].value = 0
                dtw_segment_calculation_time[corresponding_time_index] = dtw_single_worker_calculation_time[x].value
                #dtw_single_worker_calculation_time[x].value = 0
                dtw_alignment_cost[corresponding_time_index] = dtw_single_worker_alignment_cost[x].value
                job_5s_obj_done_flag[corresponding_time_index] = 1
                # clear worker done flag
                dtw_single_worker_job_5s_obj_done_flag[x].value = 0

                print "    [info] Collect finished job index : {}".format(corresponding_time_index)
                #print "    [info] Calculated tempo : {}".format(cal_input_tempo[corresponding_time_index])
                #print "    [info] est_gt_endframe : {}".format(est_gt_endframe[corresponding_time_index])
                print "    [info] Cost Func. Value : {}".format(int(dtw_alignment_cost[corresponding_time_index]))


        # find the latest done 5s dtw job       
        total_done_5s_dtw_job = 0
        total_done_5s_dtw_job_list = []
        for x in range(len(job_list_5s)-1, -1, -1) :      
            if (job_5s_obj_done_flag[x] == 1):
                total_done_5s_dtw_job_list.append(x)
                total_done_5s_dtw_job += 1


        # if there's new job done , collect information
        if (total_done_5s_dtw_job_old != total_done_5s_dtw_job) :     
            filter_len_adp = min(total_done_5s_dtw_job, filter_len)
            if (total_done_5s_dtw_job > 0) :
                latest_5s_input_tempo_avg = 0
                for x in range(0, filter_len_adp) :
                    latest_done_5s_job_flag[x] = total_done_5s_dtw_job_list[x]
                    latest_5s_input_tempo[x] = cal_input_tempo[latest_done_5s_job_flag[x]]
                for x in range(0, filter_len_adp) :
                    latest_5s_input_tempo_avg += latest_5s_input_tempo[x]
                latest_5s_input_tempo_avg = latest_5s_input_tempo_avg / np.float(filter_len_adp)
                                        
            # update info to main monitor            
            mp_tempo_report.value = latest_5s_input_tempo_avg
            mp_total_done_5s_dtw_job.value = total_done_5s_dtw_job
            mp_dtw_segment_calculation_time.value = dtw_segment_calculation_time[latest_done_5s_job_flag[0]]
            # enter this "if" section only when there's new job done
            total_done_5s_dtw_job_old = total_done_5s_dtw_job
        ######### if section finished here #########


        # show lateset info every 'show_refersh_sec' Sec.                                 
#        mod_time = (elapse_time % show_refersh_sec)
#        if (while_loop_runed == 0) :
#            if mod_time < (show_refersh_sec * 0.5) :
#                print 'Elapse time : {:1.2f} s'.format(elapse_time)
#                print '    5s dtw job done : {}  ,  Est. input tempo : {:1.3f}'.format(total_done_5s_dtw_job, latest_5s_input_tempo_avg)
#                print '    job calculation time : {:1.2f} Sec.'.format(dtw_segment_calculation_time[latest_done_5s_job_flag[0]])
#                print '    alive worker : {}'.format(total_alive_worker)
#                print '    busy worker : {}'.format(total_busy_worker)
#                print '    idle worker : {}'.format(total_idle_worker)
#                
#                while_loop_runed = 1
#        else : # loop already runned , reset loop
#            if mod_time > (show_refersh_sec * 0.5) :
#                while_loop_runed = 0
        ############################################                                  


                               
        # check if all jobs are done first
        if (job_list_5s_not_finished==1) : #or    \
#           (job_list_10s_not_finished==1) or    \
#           (job_list_15s_not_finished==1) :
               
            # check if it's time to do job
            if job_list_5s_not_finished and (elapse_time > job_list_5s[job_list_5s_marker]) :
                do_5s_job = 1 # set run job flag in this loop  
                #print job_list_5s_marker
                #print job_list_5s[job_list_5s_marker]
                #print elapse_time
                #print "set do_5s_job = 1"
                if (job_list_5s_marker < len(job_list_5s)-1) : # there are 5s jobs left, move to next target
                    job_list_5s_marker += 1
                else : # all 5s job done
                    job_list_5s_not_finished = 0
                

            # check if there's any job to do in this loop
            if (do_5s_job==1) : # or (do_5s_job==1) or (do_5s_job==1):
                #print "time is {0:1.1f} Sec. now".format(elapse_time)
                #current_time = int(elapse_time) # calculate current time
                elapse_time = time.time() - start_time
                current_time = elapse_time # calculate current time
                                  
                if (do_5s_job==1): # do 5s job in this loop   
                    # calculate current job list index from current_time
                    current_5s_index = time_2_5s_dtw_index(current_time,    \
                                                           job_list_5s,     \
                                                           time_step_5s)
                    
                    #update corsponding groundtrouth frame
                    if (total_done_5s_dtw_job <= 1 ) : # no any job has been done
                        input_clip_tempo[current_5s_index] = default_input_tempo
                        in_gt_tempo_ratio = np.float(input_clip_tempo[current_5s_index]) / gt_tempo
                        gt_in_end_frame[current_5s_index] = np.int(current_time * in_gt_tempo_ratio * gt_cens_fps)
                        gt_in_start_frame[current_5s_index] = max( 0, gt_in_end_frame[current_5s_index] - np.int(clip_length_5s * in_gt_tempo_ratio * gt_cens_fps) )                        
                    if (total_done_5s_dtw_job > 1) : # use the latest calculate result
                        #filter_len_adp = min(total_done_5s_dtw_job, filter_len)
                        for x in range(0, filter_len_adp):
                            input_clip_tempo_temp[x] = cal_input_tempo[latest_done_5s_job_flag[x]]
                            in_gt_tempo_ratio = np.float(input_clip_tempo_temp[x]) / gt_tempo
                            gt_in_end_frame_temp[x] = np.int(est_gt_endframe[latest_done_5s_job_flag[x]] + (current_time - job_list_5s[latest_done_5s_job_flag[x]]) * in_gt_tempo_ratio * gt_cens_fps)
                            gt_in_end_frame_temp[x] = min(ref_audio_pkg.audio_data_chroma_cens.shape[1], gt_in_end_frame_temp[x])
                            gt_in_start_frame_temp[x] = max( 0, gt_in_end_frame_temp[x] - np.int(clip_length_5s * in_gt_tempo_ratio * gt_cens_fps) )
                            dtw_alignment_cost_temp[x] = dtw_alignment_cost[latest_done_5s_job_flag[x]]
                        # calculate average value here(tempo / start_frame / end_frame)

                        # test different type filter                        
#                        input_clip_tempo_temp_sorted = np.sort(input_clip_tempo_temp)
#                        gt_in_end_frame_temp_sorted = np.sort(gt_in_end_frame_temp)
#                        gt_in_start_frame_temp_sorted = np.sort(gt_in_start_frame_temp)
#                        
#                        filter_start = int(round(filter_len_adp*0.2))
#                        filter_end = int(filter_len_adp-round(filter_len_adp*0.2))

                        
#                        low_cost_func_list = list(np.argsort(gt_in_end_frame_temp[0:filter_len_adp]))
#                        print dtw_alignment_cost_temp
#                        print low_cost_func_list
#                        ratio = 0.0
#                        if filter_len_adp <= 8:
#                            start_choose_element = 0
#                            latest_choose_element = filter_len_adp
#                        else:
#                            start_choose_element = int(filter_len_adp * ratio)
#                            latest_choose_element = filter_len_adp - int(filter_len_adp * ratio)
#                            
#                        #print (latest_choose_element)
#                        low_cost_func_list = low_cost_func_list[start_choose_element:latest_choose_element]
#                        print low_cost_func_list


                        input_clip_tempo_sum = 0.0
                        gt_in_end_frame_sum = 0.0
                        gt_in_start_frame_sum = 0.0
                        for x in range(0,  filter_len_adp):
                        #for x in low_cost_func_list :
                            input_clip_tempo_sum += input_clip_tempo_temp[x]
                            gt_in_end_frame_sum += gt_in_end_frame_temp[x]
                            gt_in_start_frame_sum += gt_in_start_frame_temp[x]
                        input_clip_tempo[current_5s_index] = int(input_clip_tempo_sum / (filter_len_adp))
                        gt_in_end_frame[current_5s_index] = int(gt_in_end_frame_sum / (filter_len_adp))
                        gt_in_start_frame[current_5s_index] = int(gt_in_start_frame_sum / (filter_len_adp))                              

                    
                    # send estimated time and frame to main monitor
                    mp_est_gt_time[mp_dtw_job_started.value] = current_time
                    mp_est_gt_endframe[mp_dtw_job_started.value] = gt_in_end_frame[current_5s_index]
                    mp_dtw_job_started.value += 1                                  
                    #if (gt_in_end_frame[current_5s_index] > ref_audio_pkg.audio_data_chroma_cens.shape[1] - ref_audio_pkg.audio_data_cens_fps * 4.0) :
                    #    mp_dtw_ref_song_end.value = 1
                    #    for x in range(total_num_workers) :
                    #        dtw_single_worker_kill_sig[x].value = 1
                        
                        
                    # cut latest input from audio data
                    #latest_input_5s = cut_5s_data(sound_data_i, current_time, samp_rate_i, clip_length_5s)
                    
                    #latest_input_5s = mp_input_array[input_start_position : input_end_position]                    
                    #input_clip_array[current_5s_index] = multiprocessing.Array('d', latest_input_5s)

                    # calculate latest corsponding input data
#                    input_end_position = np.int(mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value * samp_rate_i)
#                    input_start_position = max( 0, input_end_position - np.int(clip_length_5s * samp_rate_i) )                    
#                    latest_input_5s = np.array(mp_input_array[input_start_position : input_end_position])


                    # convert extra chroma from extra input
                    input_extra_end_position = np.int(mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value * samp_rate_i)
                    input_extra_start_position = np.int(c_data_unit * samp_rate_i)                
                    latest_extra_data = np.array(mp_input_array[input_extra_start_position : input_extra_end_position])
                    c_data_unit = mp_sync_saved_sampling_frame.value * mp_sampling_segment_len.value
    
                    input_audio_data_temp0 = chroma_n_octave_func(latest_extra_data,      \
                                             samp_rate_i,                        \
                                             frame_hop_length,                   \
                                             lowest_start_freq,                  \
                                             n_chroma_bin_in_octave,             \
                                             n_octave_overtone_analysis,         \
                                             n_octave_stack,                     \
                                             )
    
                    
                    chroma_cens_array = np.hstack([chroma_cens_array, input_audio_data_temp0])
                    start = max(0, np.int(end - gt_cens_fps * clip_length_5s))
                    end = chroma_cens_array.shape[1]
                    latest_input_chroma = chroma_cens_array[:, start:end]


                    if ((dtw_single_worker_kill_sig[run_job_index].value == 0) and (mp_dtw_control_exit.value == 0)) :
                        # worker distributor arrange workers here                    
                        while(True) :
                            if (dtw_single_worker_is_alive[run_job_index].value == 1) and  \
                               (dtw_single_worker_run_flag[run_job_index].value == 0) and  \
                               (dtw_single_worker_kill_sig[run_job_index].value == 0) and  \
                               (dtw_single_worker_job_5s_obj_done_flag[run_job_index].value == 0):
                                # send data to worker here
                                dtw_single_worker_gt_in_start_frame[run_job_index].value = gt_in_start_frame[current_5s_index]
                                dtw_single_worker_gt_in_end_frame[run_job_index].value = gt_in_end_frame[current_5s_index]
                                #dtw_single_worker_input_clip_array[run_job_index] = multiprocessing.Array('d', mp_input_array[input_start_position : input_end_position])
                                #dtw_single_worker_input_clip_array_q[run_job_index].put(latest_input_5s)
                                dtw_single_worker_input_clip_array_q[run_job_index].put(latest_input_chroma)
                                dtw_single_worker_input_clip_tempo[run_job_index].value = input_clip_tempo[current_5s_index]
                                dtw_single_worker_tempo_adj_man[run_job_index].value = int(mp_tempo_adj_man.value)
                                dtw_single_worker_formal_run[run_job_index].value = 1
                                dtw_single_worker_job_corresponding_time_index[run_job_index].value = current_5s_index
                                
                                # data sync is finished, start run jobs   
                                dtw_single_worker_run_flag[run_job_index].value = 1                        
                                print '    start job, worker ID : {}'.format(run_job_index)
                                
                                if (run_job_index == total_num_workers - 1):
                                    run_job_index = 0
                                else :
                                    run_job_index += 1
                                    #print "this worker start working, move to next worker for next time"
                                break
                            
                            else : # switch to next worker
                                if (run_job_index == total_num_workers - 1):
                                    run_job_index = 0
                                else :
                                    run_job_index += 1
                                    print "wait for worker ready"
                    
                    
                    print '    start job, time index : {}'.format(current_5s_index)
                    
                    if (job_list_5s_not_finished==0) :
                        print "no 5s jobs left"                 


                        
#                if (do_10s_job==1): # check and do 5s job in this loop
#                    print "doing 10s-{} job".format(current_time)
#
#                    # share memory for passing arguments                    
#                    result_10s[current_time] = multiprocessing.Value('i', 0)
#                    
#                    # set this done flag to 1 if it's done
#                    #job_10s_obj_done_flag[current_time] = multiprocessing.Value('i', 0) 
#                    
#                    job_10s_obj[current_time] = multiprocessing.Process(target=dtw_worker_10s,                \
#                                                                        args = (current_time,                 \
#                                                                                result_10s[current_time],     \
#                                                                                job_10s_obj_done_flag[current_time]))
#                    job_10s_obj[current_time].start()
#                    
#                    if (job_list_10s_not_finished==0) :
#                        print "no 10s jobs left"           
                        
#                if (do_15s_job==1): # check and do 5s job in this loop
#                    print "doing 15s-{} job".format(current_time)
#
#                    # share memory for passing arguments                    
#                    result_15s[current_time] = multiprocessing.Value('i', 0)
#                    
#                    # set this done flag to 1 if it's done
#                    #job_15s_obj_done_flag[current_time] = multiprocessing.Value('i', 0) 
#                    
#                    job_15s_obj[current_time] = multiprocessing.Process(target=dtw_worker_15s,                \
#                                                                        args = (current_time,                 \
#                                                                                result_15s[current_time],     \
#                                                                                job_15s_obj_done_flag[current_time]))
#                    job_15s_obj[current_time].start()
#                   
#                    if (job_list_15s_not_finished==0) :
#                        print "no 15s jobs left"                    



                # clear run job flag in this loop
                do_5s_job = 0                
                #do_5s_job, do_10s_job, do_15s_job = (0,0,0)


        else :  # all jobs are done, quit while loop
            print "###################################"
            print "###\tall jobs are done\t###"
            print "###################################"
            break

        
    #kill all sub-workers
    for x in range(total_num_workers) :
        dtw_single_worker_kill_sig[x].value = 1

    ### programs ends here ###


if __name__== "__main__" :
    
    mp_program_run = multiprocessing.Value('i', 0)   
    mp_dtw_control_ready = multiprocessing.Value('i', 0)
    mp_input_array = multiprocessing.Array('d', 22050 * 1200)
    mp_sync_saved_sampling_frame = multiprocessing.Value('i', 100)
    mp_dtw_segment_calculation_time = multiprocessing.Value('d', 0.0)
    mp_tempo_report = multiprocessing.Value('d', 60.0)
    mp_position_current = multiprocessing.Value('d', 0.0)
    mp_position_next_5s = multiprocessing.Value('d', 0.0)
    mp_total_done_5s_dtw_job = multiprocessing.Value('i', 0)
    mp_metronome_tempo = multiprocessing.Value('d', 60.0)
    mp_sampling_segment_len = multiprocessing.Value('d', 0.2)

    prelude_length = 6.6
    dtw_5s_start = prelude_length + 0.2
    song_length = 1200          
    time_step_5s = 0.5
    job_list_5s = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()
    
    mp_dtw_job_started = multiprocessing.Value('i', 0)
    mp_est_gt_time = multiprocessing.Array('d', len(job_list_5s))
    mp_est_gt_endframe = multiprocessing.Array('i', len(job_list_5s))
    mp_gt_cens_fps = multiprocessing.Value('d', 172.267475)
    
    mp_dtw_ref_song_end = multiprocessing.Value('i', 0) 
    mp_dtw_control_exit = multiprocessing.Value('i', 0)        
    
                
    mp_program_run.value = 1
    
    dtw_mission_control(mp_program_run,                        \
                        mp_dtw_control_ready,                  \
                        mp_input_array,                        \
                        mp_sync_saved_sampling_frame,          \
                        mp_dtw_segment_calculation_time,       \
                        mp_tempo_report,                       \
                        mp_position_current,                   \
                        mp_position_next_5s,                   \
                        mp_total_done_5s_dtw_job,              \
                        mp_metronome_tempo,                    \
                        mp_sampling_segment_len,               \
                        mp_dtw_job_started,                    \
                        mp_est_gt_time,                        \
                        mp_est_gt_endframe,                    \
                        mp_gt_cens_fps,                        \
                        mp_dtw_ref_song_end,                   \
                        mp_dtw_control_exit,                   \
                        ) 

    
    



