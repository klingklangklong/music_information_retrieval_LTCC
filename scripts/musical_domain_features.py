import music21
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as s

from utils import pitches_to_classes, key_number_to_key_name, get_frequency_registers, save_txt_array, get_2Dpitch_curves, midi_to_list, get_start_end_frame, compute_local_average, save_txt_list
from plot_functions import plot_pitch_histogram

import yaml
from yaml.loader import SafeLoader
import sys
sys.path.append('../')

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)
Fs_msp = global_var["Fs_msp"]
n_pitches = global_var["n_pitches"]
lowest_pitch = global_var["lowest_pitch"]
keys = global_var["keys"]


def get_info_from_midi_list(midi_list, show_histograms=False):
    """
    Given a midi list of events, this function returns some musical information extracted from this list.

    Args:
        midi_list (list): input list of midi files
        show_histograms (bool, optional): if true, the histograms are plotted and showed. Defaults to False.
    """
    dur_midi = max(list(map(lambda x:x[1], midi_list)))
    pitches = sorted(list(map(lambda x:x[2], midi_list)))
    unique_pitches = list(set(pitches))

    pitch_classes_density, pitch_classes_found = pitches_to_classes(pitches)
    num_pitch_classes = len(pitch_classes_found)
    
    num_notes = len(pitches)
    num_unique_pitches = len(unique_pitches)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    most_frequent_pitch = int(s.mode(pitches)[0])
    range_pitches = max_pitch - min_pitch
    avg_pitch = int(np.rint(np.average(pitches)))
    std_pitch = np.std(pitches)
    instruments = sorted(list(set(map(lambda x:x[4], midi_list)))) 

    print("N° notes: ", num_notes)  
    print("N° unique pitches", num_unique_pitches) 
    print("N° pitch classes: ", num_pitch_classes)  
    print("Duration: ", dur_midi, " seconds")
    print("Min pitch: ", min_pitch, '->', key_number_to_key_name(min_pitch))
    print("Max pitch: ", max_pitch, '->', key_number_to_key_name(max_pitch))
    print("Range pitches: ", range_pitches)
    print("Average pitch: ", avg_pitch, '->', key_number_to_key_name(int(avg_pitch)))
    print("Standard deviation pitches: ", std_pitch)
    print("Most frequent pitch: ", most_frequent_pitch, '->', key_number_to_key_name(most_frequent_pitch))
    
    get_frequency_registers(pitches)

    print("Instruments: ", instruments) 

    if (show_histograms):

        #hist of all pitches from 0-128
        plot_pitch_histogram(pitches, 
                            num_bin = 128,
                            binrange = 128,
                            xlabel = "Pitch [MIDI]")
        

        #hist of pitches in the found range
        fig = plt.figure(figsize = (20,6))
        sns.histplot(data = pitches,
                     binrange = (min_pitch, max_pitch),
                     discrete=True,
                      kde=True
                        )
        plt.xlabel("Pitch [MIDI]")
        xticks = list(range(min_pitch, max_pitch+1))
        plt.xticks(xticks)
        plt.show()
        
        #hist of the chroma classes
        fig = plt.figure(figsize=(20,6))
        plt.bar(x = keys,
                height = pitch_classes_density
                )
        plt.ylabel("Count")
        plt.xlabel("Chroma")
        plt.show()


def get_harmony_info(filename):
    """
    Given the filename path of a midi file, the function returns some harmonic and structure information from this song.
    The function is able to detect the tempo, the time signature, and the key of the input midi file. 

    Args:
        filename (str): filename of the input midi file.
    """

    score_data = music21.converter.parse(filename)

    #first instrument track
    part = score_data.parts[0]

    tempo_list = []
    bpm_list = []
    keys = []
    
    #Tempo and time signature detection
    for p in part:
        for n in p:
            if type(n) == music21.meter.TimeSignature: 
                numerator = n.numerator
                denominator = n.denominator
                ratioString = n.ratioString
                tempo_list.append(ratioString)
                
            if type(n) == music21.tempo.MetronomeMark:          #modify global tempo
                bpm = n.number
                bpm_list.append(bpm)

    print("\nTempo list: ", *tempo_list)
    print("BPM list: ", *bpm_list)

    #Key detection
    try:
        key = score_data.analyze('key')
        print("Key: ", key, " | correlation coefficient: ", key.correlationCoefficient*100, "%")
        keys.append(key)

    except:    
        for p in part:
            for n in p:
                if type(n) == music21.key.Key:
                    print("Key: ", n)
                    keys.append(n)

    if not keys:
        print("No key found")



def retrieve_tracks_pitch_curves(midi_list, get_info_flag=True, plot_pitches=True, save_flag=False):
    """
    Given a list of midi notes, this function extracts and plot one curve of pitches for each track.

    Args:
        midi_list (list): list of midi notes
        get_info_flag (bool, optional): if True, it show information for each track. Defaults to True.
        plot_pitches (bool, optional): if True, it plots and show the curves. Defaults to True.
        save_flag (bool, optional): if True, a .txt file with the retrieved information is saved. Defaults to False.
    """
    #it gets a midi_list and returns 1 pitch curve for each instrumental track
    
    df = pd.DataFrame()
    single_2D_curve = []
    list_df = []        #list of dataframes
    i=0
    off = 5
    figure = plt.figure(figsize=(20, 8))

    instruments = sorted(list(set(map(lambda x:x[4], midi_list))))
    multitrack_midi_list = [[y for y in midi_list if y[4]==x] for x in instruments]     #each element is a midi list of a single track     

    for track_midi_list in multitrack_midi_list:
        if(get_info_flag):
            print("\nInstrument track: ", instruments[i])
            get_info_from_midi_list(track_midi_list)


        pitch_curve2D = get_2Dpitch_curves(track_midi_list)
        
        pitch_curve2D_array = np.array(pitch_curve2D).T

        plt.plot(pitch_curve2D_array[0], pitch_curve2D_array[1], marker = 'o', linestyle = 'dashed')
        
        plt.ylim([lowest_pitch,lowest_pitch+n_pitches])

        if(save_flag):
            save_txt_array(times = pitch_curve2D_array[0],
                           values = pitch_curve2D_array[1],
                           filename = instruments[i] + ".txt")
        
        i+=1

    plt.legend(instruments)   
    plt.xlabel("Time [s]")
    plt.ylabel("MIDI Pitch")
    plt.grid()
    plt.show()   





def get_statistics_from_mono_midi(filename, 
                                  average_flag = False,
                                  average_param = 7,
                                  save_flag=False,
                                  plot_flag=False,
                                  save_path= ""):

    """
    This function retrieves information from a mono-track midi file.
    A statistic regarding the local maxima and minima of the musical pitches is extracted.

    Args:
        filename (str): path of the midi file
        average_flag (bool): if True, the average of the extracted curves of values is computed
        average_param (int): window of samples for the average calculation-
        save_flag (boolean): if True, the .txt files with the curve values are saved
        plot_flag (boolean): if True, the plots are showed
        save_path (str): path where we can save 
    Returns:
        max_array (np.ndarray): curve representing the local maxima of the musical pitches of the input song
        min_array: (np.ndarray): curve representing the local minima of the musical pitches of the input song
    """

    midi_data = pretty_midi.PrettyMIDI(save_path + "/" + filename)
    midi_list = midi_to_list(save_path + "/" + filename)

    max_list = []
    min_list = []
    avg_list = []
    min_list = []

    dur = midi_list[-1][1]
    time_frames = (np.arange(0, dur+1, 1/Fs_msp, dtype=float))

    max_array = np.zeros(shape=len(time_frames), dtype=int)
    min_array = np.zeros(shape=len(time_frames), dtype=int)
    avg_array = np.zeros(shape=len(time_frames), dtype=int)
    range_array = np.zeros(shape=len(time_frames), dtype=int)

    print(type(min_array))
    # sort midi list by start time, and by pitch
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2])) 

    # group by start time
    values = sorted(set(map(lambda x:x[0], midi_list)))

    #each element of the list is a group of note with the same starting point
    group_by_start_time = [[y for y in midi_list if y[0]==x] for x in values]       

    selected_notes = [] 
    selected_harmony_groups = []

    for group in group_by_start_time:
        if len(group) > 1:
            
            midi_min = group[0]
            midi_max = group[-1]
            range = midi_max[2] - midi_min[2]
            avg_value = range / 2
            
            #time -> frame
            start_min, end_min = get_start_end_frame(midi_min, Fs=Fs_msp)
            start_max, end_max = get_start_end_frame(midi_max, Fs=Fs_msp)

            #function that writes the pitch in the list for max
            min_array[start_min:end_min] = midi_min[2]
            max_array[start_max:end_max] = midi_max[2]

        else:
            note = group[0]

    
    #to find the local maxima
    print(len(min_array))
    
    avg_array[:] = (max_array[:] + min_array[:]) / 2

    range_array[:] = max_array[:] - min_array[:]

    if(average_flag):
        #avg_array = compute_local_average(avg_array, M=average_param)
        range_array = compute_local_average(range_array, M=average_param)
        min_array = compute_local_average(min_array, M=average_param)
        max_array = compute_local_average(max_array, M=average_param)


    if(save_flag):
        save_txt_list(list(min_array), "min.txt")
        save_txt_list(list(max_array), "max.txt")
        save_txt_list(list(avg_array), "avg.txt")

    if(plot_flag):
        fig = plt.figure(figsize=(20,8))
        plt.plot(min_array, color="green")
        plt.plot(max_array, color="red")

        x_ticks = np.arange(0, len(time_frames))
        x_labels = time_frames / Fs_msp

        plt.xticks(ticks = x_ticks[::300],
                labels = time_frames[::300])
        
        plt.ylabel("MIDI Pitch")
        plt.xlabel("Time [s]")
        plt.legend(["Min curve", "Max curve"], fontsize=15)

        plt.show()

    return max_array, min_array






