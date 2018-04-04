
"""
Created on Wed Jun 21 10:33:58 2017

@author: iis519
"""

#import Tkinter as tk
#from Tkinter import *
#from PIL import ImageTk
#from scipy import signal
#import report_time

import Tkinter
import numpy as np
#import psutil
import os, pyaudio, wave, time
import dill
from datetime import datetime
#import datetime
import librosa
import multiprocessing
import threading
import dtw_job_control as djc
import ast


from scipy.signal import butter, lfilter
#from scipy.signal import freqz

#fs = 22050.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def volume_process(value):
    temp_value = value
    temp_value = (np.log10(temp_value + 0.0001) + 1.7)  * 33
    temp_value = min(100.0, temp_value)
    temp_value = max(0.0, temp_value)
    if temp_value > 0.0 and temp_value <= 20 :
        level = "1"
    elif temp_value > 20.0 and temp_value <= 40  : 
        level = "2"
    elif temp_value > 40.0 and temp_value <= 60 : 
        level = "3"
    elif temp_value > 60.0 and temp_value <= 80 : 
        level = "4"
    elif temp_value > 80.0 and temp_value <= 100 : 
        level = "5"
    else :
        level = "1"
        
    return level


class playback_service():
    def __init__(self):
        self.playback_file_path = "audio\\stretched.wav"
        self.play_back_wave_file = wave.open(self.playback_file_path, 'rb')
        self.song_data ,self.sr = librosa.load(self.playback_file_path, sr=22050)
        self.song_length = len(self.song_data)/float(self.sr)
        self.pyaud = pyaudio.PyAudio()
        self.define_play_stream()

        print "playing service is initialized."

    def playback_callback(self, in_data, frame_count, time_info, status):
        playback_data = self.play_back_wave_file.readframes(frame_count)
        return (playback_data, pyaudio.paContinue)

    def define_play_stream(self):
        self.play_stream = self.pyaud.open(format=self.pyaud.get_format_from_width(self.play_back_wave_file.getsampwidth()),
                                           channels=self.play_back_wave_file.getnchannels(),
                                           rate=self.play_back_wave_file.getframerate(),
                                           output=True,
                                           stream_callback=self.playback_callback)
        self.play_stream.stop_stream()
        
    def start_play_stream(self):
        self.play_stream.start_stream()
        print "start playing file..."

    def close_play_stream(self): 
        self.play_stream.stop_stream()
        self.play_stream.close()
        self.play_back_wave_file.close()
        self.pyaud.terminate()
        
# global variable for data transfer
saved_sampling_data = []
saved_sampling_data_str = []
saved_sampling_frame = 0

class record_service():
    def __init__(self):
        global saved_sampling_data
        global saved_sampling_data_str
        global saved_sampling_frame        
        saved_sampling_data = []
        saved_sampling_data_str = []
        saved_sampling_frame = 0
        #global sampling_is_running        
        self.WIDTH = 2
        self.SAMPLING_WINDOW = 0.05
        self.SAMPLING_CHANNELS = 1
        self.SAMPLING_RATE = 22050
        self.MONITORING_SAMPLING_INPUT = False #False
        self.NUM_SAMPLES = np.int(self.SAMPLING_RATE * np.float(self.SAMPLING_WINDOW))
        self.pyaud = pyaudio.PyAudio()
        self.sampling_is_running = False
        
        print "recording service is initialized."


    def save_wave_file(self, filename, save_data): 
        # no need to save file
#        self.wf = wave.open(filename, 'wb') 
#        self.wf.setnchannels(self.SAMPLING_CHANNELS) 
#        self.wf.setsampwidth(2) 
#        self.wf.setframerate(self.SAMPLING_RATE) 
#        self.wf.writeframes("".join(save_data)) 
#        self.wf.close()
        print ("no file:", filename, "is saved.")

    def callback(self, in_data, frame_count, time_info, status):
        global saved_sampling_frame
        saved_sampling_data.append(in_data)
        
        saved_sampling_data_str.append( np.fromstring(in_data, dtype=np.int16) )
        saved_sampling_frame += 1
    
        return (in_data, pyaudio.paContinue)

    def run_sampling_stream(self):
        #global sampling_is_running
        if (self.sampling_is_running == False):
            self.audio_stream = self.pyaud.open(format=pyaudio.paInt16,
                                                channels=self.SAMPLING_CHANNELS,
                                                rate=self.SAMPLING_RATE,
                                                input=True,
                                                output=self.MONITORING_SAMPLING_INPUT,
                                                frames_per_buffer=self.NUM_SAMPLES,
                                                stream_callback=self.callback)
            self.audio_stream.start_stream()
            self.sampling_is_running = True
            print "Start recording."
        else:    
            print "Recording is already running"

        #self.recording_service_update()
        #print "sampling is Started."

    def close_sampling_stream(self): 
        #global sampling_is_running
        if (self.sampling_is_running == True):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.pyaud.terminate()
            self.sampling_is_running = False
            print "saved audio frame is {0}".format(saved_sampling_frame)
            print "Stop recording."
#        else:
#            print "Recording is already stopped"

    def save_data_2_file(self):
        if len(saved_sampling_data) > 0:
            filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".wav" 
            self.save_wave_file(filename, saved_sampling_data)




class demo_window(Tkinter.Tk):
    def __init__(self):
        self.root = Tkinter.Tk.__init__(self)
        
        self.my_frame = Tkinter.Frame(self)
        # set key function
        #self.my_frame.bind("<space>", self.callback_key_SPACE)
        #self.my_frame.bind("<b>", self.callback_key_B) 
        self.my_frame.bind("<p>", self.callback_key_P) 
        self.my_frame.bind("<Escape>", self.callback_key_ESCAPE)

        # manually change tempo
        self.tempo_adj_man = 0
        self.my_frame.bind("<Left>", self.callback_key_LEFT) # manually slow down
        self.my_frame.bind("<Right>", self.callback_key_RIGHT) # manually speed up
        self.my_frame.bind("<Down>", self.callback_key_DOWN) # reset tempo adj
        
        self.my_frame.pack()
        self.my_frame.focus_set()

        self.title("     Realtime Alignment System      ")
        self.resizable(0,0)   # prevent user from resize window shape              
        
        self.win_width = 800
        self.win_height = 620                               
        # creat new canvas here           
        self.canvas = Tkinter.Canvas(self.my_frame,
                                width=self.win_width,
                                height=self.win_height,
                                background='black')        
                                #background='light cyan')          
        self.canvas.pack()

        self.out_cmd_path = "cur_event_cmd.txt"
        self.out_alignment_path = "alignment_result.txt"

        # paint upper fram here
        self.upFrame = Tkinter.Frame(self.my_frame, width=792, height=49, bg="DarkSlateGray3")
        self.upFrame.pack() ; 
        self.upFrame.place(x=5, y=5)
        
        # split up/down frame
        line_thickness = 3
        self.split_line1 = self.canvas.create_rectangle(0, 54, self.win_width, 54+line_thickness+2, fill="white")
        self.split_line2 = self.canvas.create_rectangle(0, 1, self.win_width, 2+line_thickness, fill="black")
        self.split_line3 = self.canvas.create_rectangle(0, 0, 1+line_thickness, self.win_height, fill="black")
        self.split_line4 = self.canvas.create_rectangle(0, self.win_height-3, self.win_width, self.win_height, fill="black")
        self.split_line5 = self.canvas.create_rectangle(self.win_width-3, 0, self.win_width, self.win_height, fill="black")

        # create volume bar
        self.volume_bar_update_time = 0.05   # update every 0.1 Sec.
        self.volume_bar_value = 0
        self.volume_bar_height = 538
        self.update_vbar_loop_runed = 0
        self.volume_bar = self.canvas.create_rectangle(10, 70, 66, 70+self.volume_bar_height, fill="alice blue", outline="white")
        self.volume_bar_power = self.canvas.create_rectangle(15, 580, 61, 65+self.volume_bar_height, fill="green", outline="black")
        self.volume_bar_bar = self.canvas.create_rectangle(15, 580, 61, 580, fill="yellow", outline="black")


        # create Exit button
        self.exit_button = Tkinter.Button(self.my_frame, 
                                     text="Exit", 
                                     command=self.quit_prog, 
                                     bg='light yellow2', borderwidth=4)
        self.exit_button.pack()          
        self.exit_button.place(x=(self.win_width-52), y=15)


        # show info on upper frame
        self.input_audio_time = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="Input Time  : 0.0 Sec.")
        self.input_audio_time.config(font=("Courier", 11))
        self.input_audio_time.pack()
        self.input_audio_time.place(x=270, y=8)

        self.input_audio_tempo = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="Input Tempo : 0.0 BPM.")
        self.input_audio_tempo.config(font=("Courier", 11))
        self.input_audio_tempo.pack()
        self.input_audio_tempo.place(x=270, y=29)

        self.ref_audio_time = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="Ref. Time  : 0.0 Sec.")
        self.ref_audio_time.config(font=("Courier", 11))
        self.ref_audio_time.pack()
        self.ref_audio_time.place(x=525, y=8)

        self.ref_audio_tempo = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="Ref. Tempo : 60.0 BPM.")
        self.ref_audio_tempo.config(font=("Courier", 11))
        self.ref_audio_tempo.pack()
        self.ref_audio_tempo.place(x=525, y=29)

        self.tempo_adj_man_label = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="Tempo Adj : 0 %")
        self.tempo_adj_man_label.config(font=("Courier", 11))
        self.tempo_adj_man_label.pack()
        self.tempo_adj_man_label.place(x=30, y=8)

        self.dtw_done_n_label = Tkinter.Label(self.my_frame, bg="DarkSlateGray3", text="DTW Task Done : 0")
        self.dtw_done_n_label.config(font=("Courier", 11))
        self.dtw_done_n_label.pack()
        self.dtw_done_n_label.place(x=30, y=29)

        # set time tempo HUD parameter
        self.time_tempo_update_time = 0.01        # update every 0.01 Sec.
        self.update_time_tempo_loop_runed = 0

        # set spectrogram path
        self.file_path_srt_01 = os.getcwd() + "\\spectrogram\\" + "000_015.gif"
        self.file_path_srt_02 = os.getcwd() + "\\spectrogram\\" + "015_030.gif"
        self.file_path_srt_03 = os.getcwd() + "\\spectrogram\\" + "030_045.gif"
        self.file_path_srt_04 = os.getcwd() + "\\spectrogram\\" + "045_060.gif"
        self.file_path_srt_05 = os.getcwd() + "\\spectrogram\\" + "060_075.gif"
        self.file_path_srt_06 = os.getcwd() + "\\spectrogram\\" + "075_090.gif"
        self.file_path_srt_07 = os.getcwd() + "\\spectrogram\\" + "090_105.gif"
        self.file_path_srt_08 = os.getcwd() + "\\spectrogram\\" + "105_120.gif"
        self.file_path_srt_09 = os.getcwd() + "\\spectrogram\\" + "120_135.gif"
        self.file_path_srt_10 = os.getcwd() + "\\spectrogram\\" + "135_150.gif"
        self.file_path_srt_11 = os.getcwd() + "\\spectrogram\\" + "150_165.gif"
        self.file_path_srt_12 = os.getcwd() + "\\spectrogram\\" + "165_180.gif"
        self.file_path_srt_13 = os.getcwd() + "\\spectrogram\\" + "180_195.gif"
        self.file_path_srt_14 = os.getcwd() + "\\spectrogram\\" + "195_210.gif"
        self.file_path_srt_15 = os.getcwd() + "\\spectrogram\\" + "210_225.gif"
        self.file_path_srt_16 = os.getcwd() + "\\spectrogram\\" + "225_240.gif"
        self.file_path_srt_17 = os.getcwd() + "\\spectrogram\\" + "240_255.gif"
        self.file_path_srt_18 = os.getcwd() + "\\spectrogram\\" + "255_270.gif"
        self.file_path_srt_19 = os.getcwd() + "\\spectrogram\\" + "270_285.gif"
        self.file_path_srt_20 = os.getcwd() + "\\spectrogram\\" + "285_300.gif"
        self.file_path_srt_21 = os.getcwd() + "\\spectrogram\\" + "300_315.gif"



        # draw first spectrogram as 1st page
        self.bground_file_01 = Tkinter.PhotoImage(file=self.file_path_srt_01)
        self.bground_file_02 = Tkinter.PhotoImage(file=self.file_path_srt_02)
        self.bground_file_03 = Tkinter.PhotoImage(file=self.file_path_srt_03)
        self.bground_file_04 = Tkinter.PhotoImage(file=self.file_path_srt_04)
        self.bground_file_05 = Tkinter.PhotoImage(file=self.file_path_srt_05)
        self.bground_file_06 = Tkinter.PhotoImage(file=self.file_path_srt_06)
        self.bground_file_07 = Tkinter.PhotoImage(file=self.file_path_srt_07)
        self.bground_file_08 = Tkinter.PhotoImage(file=self.file_path_srt_08)
        self.bground_file_09 = Tkinter.PhotoImage(file=self.file_path_srt_09)
        self.bground_file_10 = Tkinter.PhotoImage(file=self.file_path_srt_10)
        self.bground_file_11 = Tkinter.PhotoImage(file=self.file_path_srt_11)
        self.bground_file_12 = Tkinter.PhotoImage(file=self.file_path_srt_12)
        self.bground_file_13 = Tkinter.PhotoImage(file=self.file_path_srt_13)
        self.bground_file_14 = Tkinter.PhotoImage(file=self.file_path_srt_14)
        self.bground_file_15 = Tkinter.PhotoImage(file=self.file_path_srt_15)
        self.bground_file_16 = Tkinter.PhotoImage(file=self.file_path_srt_16)
        self.bground_file_17 = Tkinter.PhotoImage(file=self.file_path_srt_17)
        self.bground_file_18 = Tkinter.PhotoImage(file=self.file_path_srt_18)
        self.bground_file_19 = Tkinter.PhotoImage(file=self.file_path_srt_19)
        self.bground_file_20 = Tkinter.PhotoImage(file=self.file_path_srt_20)
        self.bground_file_21 = Tkinter.PhotoImage(file=self.file_path_srt_21)


        # set background to the 1st page
        self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_01 )

        # set spectrogram data
        self.default_ref_tempo = 60.0
        self.est_input_tempo = self.default_ref_tempo
        self.total_spectrogram_pages = 21
        self.spectrogram_width = 645
        self.period_per_page = 15.0

        # draw running timebar
        self.xcor_init = 77
        self.time_bar = self.canvas.create_line(self.xcor_init, 70,
                                                self.xcor_init, 610,
                                                fill="yellow2", width=3)
        
        self.refresh_period_ms = 10    # (update GUI frame for every 0.015 Sec.)
        self.running_flag = False

        self.reset_spectrogram_view()
        
        self.set_next_spectrogram_target(0, 0)        
        
        #self.last_runed_time = 0.0

        # initialize tempo calculater
        self.space_key_time = list(range(2400))
        self.space_key_count = 0

        

        # initialize audio record
        self.my_sd = record_service()
        #self.my_sd.SAMPLING_WINDOW
        
        self.my_playback = playback_service()
        
        # main loop update period
        self.refersh_sec = 1.0   # show message every 0.5 Sec.
        self.show_loop_runed = 0
        
        self.init_dtw_process()
        
        self.ref_time = 0.0

        # init. inst. volume
        self.v_volume = 0.0
        self.vv_volume = 0.0
        self.a_volume = 0.0
        self.c_volume = 0.0

        # start soundcard capture here
        try :
            self.my_sd.run_sampling_stream()
        except IOError:
            print "Microphone is not ready"
            self.my_sd.__init__()
            self.program_reset()

        # load saved pickle data
        self.saved_midi_time_pickle_file_path = os.getcwd() + "\\mozart_midi_time_array.pickle"
        with open(self.saved_midi_time_pickle_file_path, 'rb') as saved_midi_time_pickle_file:
            self.saved_midi_time = dill.load(saved_midi_time_pickle_file)
            
        self.saved_performance_time_pickle_file_path = os.getcwd() + "\\mozart_performance_time_array.pickle"        
        with open(self.saved_performance_time_pickle_file_path, 'rb') as saved_performance_time_pickle_file:
            self.saved_performance_time = dill.load(saved_performance_time_pickle_file)
        
        self.time_conv_index = 0
        self.midi_time = 0.0


        with open('RefTimeList_Mozart.txt') as self.RefTimeList_Mozart:
            self.RefTimeList_tmp = list(self.RefTimeList_Mozart.read().splitlines())
    
        self.RefTimeList = []    
        for x in range(0, len(self.RefTimeList_tmp)):
            self.RefTimeList.append(ast.literal_eval(self.RefTimeList_tmp[x]))
            
        with open('RefEventList_Mozart.txt') as self.RefEventList_Mozart:
            self.RefEventList = list(self.RefEventList_Mozart.read().splitlines())

        self.ref_list_index = 0
        # create an empty file "cur_event_cmd.txt"
        #open(self.out_cmd_path, 'w').close()
        
        open(self.out_alignment_path, 'w').close()

        self.start_time = time.time()
        
        # time to start automatically
        self.scheduled_start_time = 6.0
        self.playback_is_running = 0
        
        # call system loop
        self.system_run()
        




    def init_dtw_process(self):
        self.sync_saved_sampling_frame = 0
        self.mp_input_array_len_in_sec = 1200   # 1200 Sec. = 20 mins
        self.mp_input_array_size = self.my_sd.SAMPLING_RATE * self.mp_input_array_len_in_sec
        self.mp_elapse_time = multiprocessing.Value('d', 0.0)        
        self.mp_input_array = multiprocessing.Array('d', self.mp_input_array_size)
        self.mp_sync_saved_sampling_frame = multiprocessing.Value('i', 0)
        self.mp_program_run = multiprocessing.Value('i', 0)
        self.mp_dtw_control_ready = multiprocessing.Value('i', 0)
        self.mp_dtw_segment_calculation_time = multiprocessing.Value('d', 0.0)
        self.mp_tempo_report = multiprocessing.Value('d', 0.0)
        self.mp_position_current = multiprocessing.Value('d', 0.0)
        self.mp_position_next_5s = multiprocessing.Value('d', 0.0)
        self.mp_total_done_5s_dtw_job = multiprocessing.Value('i', 0)
        self.mp_dtw_control_exit = multiprocessing.Value('i', 0)
        self.mp_total_done_5s_dtw_job = multiprocessing.Value('i', 0)
        self.mp_metronome_tempo = multiprocessing.Value('d', self.default_ref_tempo)
        self.mp_tempo_adj_man = multiprocessing.Value('i', self.tempo_adj_man)
        self.mp_sampling_segment_len = multiprocessing.Value('d', self.my_sd.SAMPLING_WINDOW)

        prelude_length = 6.0
        dtw_5s_start = prelude_length + 0.1
        song_length = 1200
        time_step_5s = 0.1
        job_list_5s = np.arange(dtw_5s_start, (song_length+time_step_5s), time_step_5s).tolist()    
        self.mp_dtw_job_started = multiprocessing.Value('i', 0)
        self.mp_est_gt_time = multiprocessing.Array('d', len(job_list_5s))
        self.mp_est_gt_endframe = multiprocessing.Array('i', len(job_list_5s))
        self.mp_dtw_ref_song_end = multiprocessing.Value('i', 0)
        self.mp_gt_cens_fps = multiprocessing.Value('d', 172)
        
        self.dtw_job_control = multiprocessing.Process(target = djc.dtw_mission_control,               \
                                                       args = (self.mp_program_run,                    \
                                                               self.mp_dtw_control_ready,              \
                                                               self.mp_input_array,                    \
                                                               self.mp_sync_saved_sampling_frame,      \
                                                               self.mp_dtw_segment_calculation_time,   \
                                                               self.mp_tempo_report,                   \
                                                               self.mp_position_current,               \
                                                               self.mp_position_next_5s,               \
                                                               self.mp_total_done_5s_dtw_job,          \
                                                               self.mp_metronome_tempo,                \
                                                               self.mp_tempo_adj_man,                  \
                                                               self.mp_sampling_segment_len,           \
                                                               self.mp_dtw_job_started,                \
                                                               self.mp_est_gt_time,                    \
                                                               self.mp_est_gt_endframe,                \
                                                               self.mp_gt_cens_fps,                    \
                                                               self.mp_dtw_ref_song_end,               \
                                                               self.mp_dtw_control_exit,               \
                                                               )                                       \
                                                       )
        self.dtw_job_control.start()

        while (self.mp_dtw_control_ready.value == 0):
            print "(Waiting for DTW process initialization...)"
            time.sleep(1)

        time.sleep(0.3)
        print "===  DTW Initialization Done !!  ==="
        

    def update_volume_bar(self):
        global saved_sampling_frame
        # show lateset info every 'refersh_sec' Sec.                                 
        mod_time = (time.time() % self.volume_bar_update_time)
        if (self.update_vbar_loop_runed == 0) :
            if mod_time < (self.volume_bar_update_time * 0.4) :
                #print 'Elapse time : {:1.2f} s'.format(time.time())
                if (saved_sampling_frame < 2) :
                    tot_value = 0
                    start_frame = 0
                    end_frame = 0
                    self.vbar_value_avg = 0
                else :
                    frame_unit = int(self.my_sd.SAMPLING_RATE * self.my_sd.SAMPLING_WINDOW) # 22050 * 0.1 Sec. = 2205
                    end_frame = min(len(self.input_data_array)-1, saved_sampling_frame * frame_unit)
                    start_frame = max(0, end_frame - (frame_unit * 1))
                    tot_value = np.mean(np.abs(self.input_data_array[start_frame : end_frame])) * 100
                    
                    tot_value = (np.log10(tot_value + 0.0001) + 1.7)  * 33
                    tot_value = min(100.0, tot_value)
                    tot_value = max(0.0, tot_value)
#                    if (saved_sampling_frame <= 2) :
#                        self.vbar_value_avg = 0
#                    self.vbar_value_avg = (self.vbar_value_avg + value ) / 2.0
                    #self.vbar_value_old = value

                    # calculate band passed sig of violin1
                    v1_lowcut, v1_highcut = 400, 4000
                    v1_band_pass = butter_bandpass_filter(self.input_data_array[start_frame : end_frame], 
                                                          v1_lowcut, 
                                                          v1_highcut, 
                                                          self.my_sd.SAMPLING_RATE, 
                                                          order=2)
                    v1_band_pass_value = np.mean(np.abs(v1_band_pass)) * 150

                    # calculate band passed sig of violin2
                    v2_lowcut, v2_highcut = 200, 1000
                    v2_band_pass = butter_bandpass_filter(self.input_data_array[start_frame : end_frame], 
                                                          v2_lowcut, 
                                                          v2_highcut, 
                                                          self.my_sd.SAMPLING_RATE, 
                                                          order=2)
                    v2_band_pass_value = np.mean(np.abs(v2_band_pass)) * 150
                                                
                    # calculate band passed sig of viola
                    va_lowcut, va_highcut = 80, 500
                    va_band_pass = butter_bandpass_filter(self.input_data_array[start_frame : end_frame], 
                                                          va_lowcut, 
                                                          va_highcut, 
                                                          self.my_sd.SAMPLING_RATE, 
                                                          order=2)
                    va_band_pass_value = np.mean(np.abs(va_band_pass)) * 150
                                                
                    # calculate band passed sig of cello
                    co_lowcut, co_highcut = 10, 120
                    co_band_pass = butter_bandpass_filter(self.input_data_array[start_frame : end_frame], 
                                                          co_lowcut, 
                                                          co_highcut, 
                                                          self.my_sd.SAMPLING_RATE, 
                                                          order=2)
                    co_band_pass_value = np.mean(np.abs(co_band_pass)) * 150

                    self.v_volume = v1_band_pass_value
                    self.vv_volume = v2_band_pass_value
                    self.a_volume = va_band_pass_value
                    self.c_volume = co_band_pass_value
                                                
                    #value = v1_band_pass_value
                    #value = v2_band_pass_value
                    #value = va_band_pass_value
                    #value = co_band_pass_value
                    #v1_band_pass_value += 1

                    #tot_value = (np.log10(tot_value + 0.0001) + 1.7)  * 33
                    #tot_value = min(100.0, tot_value)
                    #tot_value = max(0.0, tot_value)


                    
                self.canvas.delete(self.volume_bar_bar) 
                self.volume_bar_bar = self.canvas.create_rectangle(15, 
                                                                   580-(5*tot_value), 
                                                                   61, 
                                                                   580, 
                                                                   fill="yellow", 
                                                                   outline="black")
                #print value

                
                self.update_vbar_loop_runed = 1
        else : # loop already runned , reset loop
            if mod_time > (self.volume_bar_update_time * 0.4) :
                self.update_vbar_loop_runed = 0
        ############################################     

    # calculate corrsponding midi time from performance time
    def calc_midi_time(self):
        while (self.saved_performance_time[min(len(self.saved_performance_time)-1, self.time_conv_index)] < self.ref_time) :
            self.time_conv_index += 1
            if self.time_conv_index > len(self.saved_performance_time) :
                break
        self.midi_time = self.saved_midi_time[min(len(self.saved_midi_time)-1, self.time_conv_index)]
        #print self.midi_time

    def write_output(self):
        while (self.RefTimeList[min(len(self.RefTimeList)-1, self.ref_list_index)] < self.midi_time) : 
            if self.ref_list_index == 0 :
                with open(self.out_cmd_path, 'a+') as self.cur_event_cmd:
                    self.cur_event_cmd.write('start' + '\n')
                    
            if self.ref_list_index > len(self.RefTimeList)-1 :
                break
            
            with open(self.out_cmd_path, 'a+') as self.cur_event_cmd:
                if "Pitch1" in self.RefEventList[self.ref_list_index] :
                    if self.ref_list_index < 238 :   # first pitch1 case
                        self.cur_event_cmd.write("Pitch1" + "\t" +\
                                                 str(int(float(self.RefTimeList[self.ref_list_index]) * 314.6)) + \
                                                "\t" + "\n")
                        #pass
                    else :    # second pitch1 case
                        self.cur_event_cmd.write("Pitch1" + "\t" + \
                                                 str(int((float(self.RefTimeList[self.ref_list_index])-93) * 314.6)) + \
                                                "\t" + "\n")                    
                        #pass
                        
                elif "Pitch2" in self.RefEventList[self.ref_list_index] :
                    self.cur_event_cmd.write("Pitch1" + "\t" + \
                                             str(int((float(self.RefTimeList[self.ref_list_index])-207) * 314.6)) + \
                                             "\t" + "\n")
                        
                    
                # write out volume information
                elif "Loud" in self.RefEventList[self.ref_list_index] :
                    self.cur_event_cmd.write(self.RefEventList[self.ref_list_index] +           \
                                             "v"  + volume_process(self.v_volume) + "\t" +                                      \
                                             "vv" + volume_process(self.vv_volume) + "\t" +                                      \
                                             "a"  + volume_process(self.a_volume) + "\t" +                                      \
                                             "c"  + volume_process(self.c_volume) + "\t" +                                      \
                                             "\n")
                
                
                else : 
                    if self.RefTimeList[min(len(self.RefTimeList)-1, self.ref_list_index)] > 299.0 :
                        self.cur_event_cmd.write("end" + "\n")
                    else :
                        self.cur_event_cmd.write(self.RefEventList[self.ref_list_index] + "\n")
            #print ("Write Line : {}".format(self.ref_list_index))
            self.ref_list_index += 1            
            #if self.ref_list_index > len(self.RefTimeList)-1 :
            #    break
        #print self.ref_list_index
        #print (self.RefEventList[self.ref_list_index])
        


    def write_alignment_path(self):
        #cpu_usage = psutil.cpu_percent()
        with open(self.out_alignment_path, 'a+') as self.alignment_path :
            self.alignment_path.write(str("{:.3f}".format(self.elapse_time-0.2)) + '    ') # input time
            self.alignment_path.write(str("{:.3f}".format(self.ref_time)) + '    ') # record reference time
            self.alignment_path.write(str("{:.3f}".format(self.elapse_time-0.2 - self.ref_time)) + '    ')  # record input time - reference time
            #self.alignment_path.write(str("{:.3f}".format(self.mp_dtw_segment_calculation_time.value)) + '    ') # record calculation delay
            #self.alignment_path.write(datetime.now().time().strftime("%H:%M:%S")) # record calculation delay
            self.alignment_path.write('\n')

    def update_time_tempo_lable(self, input_time, input_tempo, ref_time, ref_tempo, tempo_adj, dtw_done):
        if (self.running_flag is True) :
            x_time = input_time
            x_tempo = input_tempo
        else :
            x_time = 0.0
            x_tempo = 0.0
            
        mod_time = (time.time() % self.time_tempo_update_time)
        if (self.update_time_tempo_loop_runed == 0) :
            if mod_time < (self.time_tempo_update_time * 0.4) :
                self.input_audio_time.config(text="Input Time  : {:1.1f} Sec.".format(x_time))
                self.input_audio_tempo.config(text="Input Tempo : {:1.1f} BPM.".format(x_tempo))
                self.ref_audio_time.config(text="Ref. Time  : {:1.1f} Sec.".format(ref_time))
                self.ref_audio_tempo.config(text="Ref. Tempo : {:1.1f} BPM.".format(ref_tempo))
                self.tempo_adj_man_label.config(text="Tempo Adj : {} %".format(tempo_adj))
                self.dtw_done_n_label.config(text="DTW Task Done : {}".format(dtw_done))

                self.update_time_tempo_loop_runed = 1
        else : # loop already runned , reset loop
            if mod_time > (self.time_tempo_update_time * 0.4) :
                self.update_time_tempo_loop_runed = 0
        ############################################     
        


    def reset_spectrogram_view(self):
        self.canvas.delete(self.time_bar)   # delete current time bar
        self.time_bar = self.canvas.create_line(self.xcor_init, 70,
                                                self.xcor_init, 610,
                                                fill="yellow2", width=3)
        self.time_bar_pos = self.canvas.coords(self.time_bar)         
        #self.timer.config(text="Elapse Time : 0.0 Sec.")    # reset time lable
        self.spectrogram_marker = 0.0        # marker position for whole song
        self.spectrogram_page_marker = 0.0   # marker position for single page
        self.current_spectrogram_page = 1        
        
        self.elapse_time = 0
        
        

    def set_next_spectrogram_target(self, target, need_time):
        self.spectrogram_next_target = np.float(target)         # run the marker to spectrogram's 60 Sec. position
        self.spectrogram_target_reach_period = np.float(need_time)   # take how much time to achieve target
        # estimated time to reach target
        self.spectrogram_target_reach_countdown = time.time() + self.spectrogram_target_reach_period


    def change_spectrogram(self, next_page):    
        print ("change to page %d" %(next_page))    
    
        #reset time bar position  
        self.current_spectrogram_page = next_page
        
        self.canvas.delete(self.bg_spec)
        if (next_page == 1) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_01 )
        elif (next_page == 2) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_02 )
        elif (next_page == 3) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_03 )
        elif (next_page == 4) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_04 )
        elif (next_page == 5) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_05 )
        elif (next_page == 6) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_06 )
        elif (next_page == 7) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_07 )
        elif (next_page == 8) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_08 )
        elif (next_page == 9) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_09 )
        elif (next_page == 10) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_10 )
        elif (next_page == 11) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_11 )
        elif (next_page == 12) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_12 )
        elif (next_page == 13) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_13 )
        elif (next_page == 14) :            
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_14 )
        elif (next_page == 15) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_15 )
        elif (next_page == 16) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_16 )
        elif (next_page == 17) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_17 )
        elif (next_page == 18) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_18 )
        elif (next_page == 19) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_19 )
        elif (next_page == 20) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_20 )
        elif (next_page == 21) :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_21 )
        else :
            self.bg_spec = self.canvas.create_image(400, 340, image=self.bground_file_01 )

        self.canvas.delete(self.time_bar)
        self.time_bar = self.canvas.create_line(self.xcor_init, 70,
                                                self.xcor_init, 610,
                                                fill="yellow2", width=3)   
        self.time_bar_pos = self.canvas.coords(self.time_bar)    


    def update_spectrogram_marker(self):        # Cal. correct spectrogram and update
        correct_page = min(int(self.spectrogram_marker / self.period_per_page) \
                            + 1, self.total_spectrogram_pages)
        self.spectrogram_page_marker = self.spectrogram_marker -   \
            (correct_page - 1.0) * self.period_per_page

        if (self.current_spectrogram_page != correct_page):
            self.change_spectrogram(correct_page)


    def update_dtw_data(self):
        global saved_sampling_frame
        self.mp_elapse_time.value = self.elapse_time
        if (self.mp_dtw_ref_song_end.value == 1) :
            self.mp_dtw_ref_song_end.value = 0
            self.program_reset()
        
        # when get new frame, save data into "self.input_data_array"
        if (saved_sampling_frame > self.sync_saved_sampling_frame) :
            frame_unit = int(self.my_sd.SAMPLING_RATE * self.my_sd.SAMPLING_WINDOW)   # 2205 here
            if (saved_sampling_frame == 1) : 
                self.input_data_array = np.asarray(saved_sampling_data_str[0]) / 32768.0
                self.mp_input_array[0 : frame_unit] = self.input_data_array[0 : frame_unit]
                self.sync_saved_sampling_frame = 1
                self.mp_sync_saved_sampling_frame.value = self.sync_saved_sampling_frame
            elif (saved_sampling_frame > 1) : 
                frame_difference = saved_sampling_frame - self.sync_saved_sampling_frame
                for index in range(frame_difference):
                    self.input_data_array = np.hstack((self.input_data_array, np.asarray(saved_sampling_data_str[min(saved_sampling_frame-1, self.sync_saved_sampling_frame+index)]) / 32768.0))
                    start_frame = int(frame_unit * self.sync_saved_sampling_frame + index)
                    end_frame = start_frame + frame_unit
                    if len(self.mp_input_array[start_frame:end_frame]) == len(self.input_data_array[start_frame:end_frame]) :
                        self.mp_input_array[start_frame:end_frame] = self.input_data_array[start_frame:end_frame]
                    self.sync_saved_sampling_frame += 1
                    self.mp_sync_saved_sampling_frame.value = self.sync_saved_sampling_frame



    def count_tempo(self):
        #print time.time()    
        self.space_key_time[self.space_key_count] = time.time()
        #self.keypress_ref_audio_time[self.space_key_count] = self.ref_time
        self.space_key_count += 1
        print "\"SPACE\" Pressed : {}".format(self.space_key_count)
        
        
        
        if (self.space_key_count >= 2) :
            key_stroke_period = range(self.space_key_count - 1)
            key_stroke_period_avg = 0
            i = 0
            filter_num = 3
            for x in range(max(0, self.space_key_count-1-filter_num), self.space_key_count-1) :
                key_stroke_period[x] = self.space_key_time[x+1] - self.space_key_time[x]
                key_stroke_period_avg = key_stroke_period_avg + key_stroke_period[x]
                i += 1
                if (x == self.space_key_count-2) :
                    key_stroke_period_avg = key_stroke_period_avg / np.float(i)

            self.mp_metronome_tempo.value = (self.default_ref_tempo / key_stroke_period_avg)
            print '    => Tempo : {:1.2f}'.format(self.mp_metronome_tempo.value)

    def callback_key_SPACE(self, event):
        self.count_tempo()
                

    def callback_key_B(self, event): # start run DTW here
        print time.time() 
        print "\"B\" Key pressed ! , DTW Process running"
        self.count_tempo()
        self.program_start()
           
        #self.mp_program_run.value = 1

    def callback_key_P(self, event): # start run DTW here
        self.playback_is_running = 1
        self.playback_start_time = time.time()
        print "\"P\" Key pressed ! , Audio Playback start !"
        self.my_playback.start_play_stream()




    def callback_key_ESCAPE(self, event):
        #print time.time() 
        print "\"Escape\" Key pressed !"
        self.quit_prog()

    def callback_key_LEFT(self, event):
        print "[info] Slow Down !"
        if (self.tempo_adj_man <= 5) and (self.tempo_adj_man >= -5):
            self.tempo_adj_man -= 1
        else :
            self.tempo_adj_man -= 2
            
        self.mp_tempo_adj_man.value = int(self.tempo_adj_man)
        self.update_time_tempo_lable(self.elapse_time,        \
                                     self.est_input_tempo,        \
                                     self.ref_time,           \
                                     self.default_ref_tempo,  \
                                     self.tempo_adj_man,      \
                                     self.mp_total_done_5s_dtw_job.value)
        
        
    def callback_key_RIGHT(self, event):
        print "[info] Speed Up !"
        if (self.tempo_adj_man <= 5) and (self.tempo_adj_man >= -5):
            self.tempo_adj_man += 1
        else :
            self.tempo_adj_man += 2

        self.mp_tempo_adj_man.value = int(self.tempo_adj_man)
        self.update_time_tempo_lable(self.elapse_time,        \
                                     self.est_input_tempo,        \
                                     self.ref_time,           \
                                     self.default_ref_tempo,  \
                                     self.tempo_adj_man,      \
                                     self.mp_total_done_5s_dtw_job.value)
        
    def callback_key_DOWN(self, event):
        print "[info] Reset original tempo !"
        self.tempo_adj_man = 0
        self.mp_tempo_adj_man.value = int(self.tempo_adj_man)
        self.update_time_tempo_lable(self.elapse_time,        \
                                     self.est_input_tempo,        \
                                     self.ref_time,           \
                                     self.default_ref_tempo,  \
                                     self.tempo_adj_man,      \
                                     self.mp_total_done_5s_dtw_job.value)

    def program_start(self):
        print ("program start")        
        self.running_flag = True
        self.mp_program_run.value = 1 if (self.running_flag == True) else 0        
        self.reset_spectrogram_view() 

        #prelude_total_beat = 6.0
        #beat_length = (self.default_ref_tempo / self.mp_metronome_tempo.value)
        #prelude_length = np.int(beat_length * prelude_total_beat * 10.0) / 10.0  # 6-beat time length (around 6.7 Sec.)
        prelude_length = 6.0
        
        # save start time stamp
        self.start_time = time.time() - prelude_length
#        open('saved_tempo_info', 'w').close()
#        self.file_note_info = open('saved_tempo_info', 'a+')
#        self.file_note_info.write("start\n")  # write start here
#        self.file_note_info.close() 

                                   
        #self.start_time = time.time()        
       

        # Set next target and estimated time to reach target
        #self.set_next_spectrogram_target(10, 10)    #  (target, need_time)
        
    
        try :
            self.my_sd.run_sampling_stream()
        except IOError:
            print "Microphone is not ready"
            self.my_sd.__init__()
            self.program_reset()
        

        
        #self.system_run()  # actual GUI program runs here



    def program_reset(self):    
        print ("program reset")        
        self.running_flag = False
        self.mp_program_run = 1 if (self.running_flag == True) else 0   
                                   
        # kill dtw sub process
        self.mp_dtw_control_exit.value = 1
        
        self.reset_spectrogram_view()
        self.change_spectrogram(1)
                
        # Re-Set next target and estimated time to reach target
        self.set_next_spectrogram_target(0, 0)  
        

        # reset recording service
        self.my_sd.close_sampling_stream()
        self.my_sd.save_data_2_file()
        self.my_sd.__init__()

        
    def quit_prog(self):
        # reset recording service
        self.my_sd.close_sampling_stream()
        #self.my_sd.save_data_2_file()        
        self.my_sd.__init__()
        self.my_playback.close_play_stream()
        
        self.mp_dtw_control_exit.value = 1
        #time.sleep(1)
        self.destroy()
        print "GUI window is closed."


    def system_run(self):
        
        if (self.running_flag is False) and (self.playback_is_running == 1):
            self.playback_time = time.time() - self.playback_start_time
        
            if (self.playback_time > self.scheduled_start_time) and (self.playback_time < (self.scheduled_start_time + 2.0)):
                self.program_start()
        
        if (self.running_flag is True) :
            if ((time.time() - self.playback_start_time) > self.my_playback.song_length) :
                self.running_flag = False
                self.my_sd.close_sampling_stream()
                self.my_sd.__init__()
                self.my_playback.close_play_stream()
                self.mp_dtw_control_exit.value = 1
                

        self.elapse_time = time.time() - self.start_time
                                    
        self.update_dtw_data()
        
        self.update_volume_bar()
        
        self.update_time_tempo_lable(self.elapse_time,        \
                                     self.est_input_tempo,        \
                                     self.ref_time,           \
                                     self.default_ref_tempo,  \
                                     self.tempo_adj_man,      \
                                     self.mp_total_done_5s_dtw_job.value)

                      
        if (self.running_flag is True):
       
            #self.elapse_time = time.time() - self.start_time
            
            # update show tempo  
            if (self.mp_total_done_5s_dtw_job.value > 0) :
                self.est_input_tempo = self.mp_tempo_report.value
            elif (self.space_key_count > 2) :
                self.est_input_tempo = self.mp_metronome_tempo.value
            else :
                self.est_input_tempo = self.default_ref_tempo
            
            time_offset = 0.25
            # update referenece time here
            if (self.mp_total_done_5s_dtw_job.value <= 2) :
                temp_value = self.elapse_time * (self.est_input_tempo / self.default_ref_tempo)
                temp_value += time_offset
                if (temp_value > self.ref_time) :
                    self.ref_time = temp_value
            else :
                temp_value = (self.mp_est_gt_endframe[self.mp_dtw_job_started.value-1] / self.mp_gt_cens_fps.value) + (self.mp_tempo_report.value / self.default_ref_tempo) * (self.elapse_time - self.mp_est_gt_time[self.mp_dtw_job_started.value-1])
                temp_value += time_offset
                if temp_value > self.ref_time :
                    self.ref_time = temp_value

            #write out alignment path after enough dtw result collected
            if (self.mp_total_done_5s_dtw_job.value >= 3) and \
                (time.time() - self.playback_start_time) > (self.scheduled_start_time + 4.0):
                self.write_alignment_path()
                                         
            # show lateset info every 'refersh_sec' Sec.                                               
            self.mod_time = (self.elapse_time % self.refersh_sec)
            if (self.show_loop_runed == 0) :
                if self.mod_time < (self.refersh_sec * 0.3) :
                    self.show_loop_runed = 1
                    print 'Elapse time : {:1.2f} s'.format(self.elapse_time)
                    print '    total DTW segment calculation done : {}'.format(self.mp_total_done_5s_dtw_job.value)
                    if (self.mp_total_done_5s_dtw_job.value > 0) :
                        print '    DTW calculation time : {:1.2f}'.format(self.mp_dtw_segment_calculation_time.value)
                        print '    Input tempo : {:1.2f}'.format(self.mp_tempo_report.value)

                    self.ref_time_5s_later = self.ref_time + (self.mp_tempo_report.value / 60.0) * 5.0
                    self.set_next_spectrogram_target(self.ref_time_5s_later, 5.0)    #  (target, need_time)
    
                    print (self.ref_time - self.elapse_time)
            else : # show loop already runned , reset loop
                if self.mod_time > (self.refersh_sec * 0.3) :
                    self.show_loop_runed = 0
         
                
            if (self.running_flag==True) :
                self.elapse_time = time.time() - self.start_time # calculate elapse time since start
                self.time_bar_pos = self.canvas.coords(self.time_bar) # get time_bar current position
                
                if (False):
                    self.running_flag = True

                else :   # when still try to reach target

                    self.spectrogram_marker = self.ref_time + 0.25
                    
                    # Syncronize spectrogram_marker and spectrogram_page_marker
                    self.update_spectrogram_marker()

                        
                    # move time bar to catch spectrogram_page_marker
                    while ( self.time_bar_pos[2] < (self.xcor_init + self.spectrogram_width*(self.spectrogram_page_marker/15.0)) ) :
                        self.canvas.move(self.time_bar, 1, 0)                 
                        self.time_bar_pos = self.canvas.coords(self.time_bar)                        
            
                    
        self.after(self.refresh_period_ms, self.system_run)  # re-run loop every refresh_period ms
                

        
if __name__ == '__main__':       
    # window thread is running here (work fine code)
    print "starting GUI window..."
    running_window = demo_window()    
    window_thread = threading.Thread(target=running_window.mainloop())
    window_thread.start()




    







    
    