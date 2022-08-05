import music21
import pretty_midi
import matplotlib.pyplot as plt
import os
import pypianoroll
import numpy as np
from libfmp import *
import libfmp.c6
import pandas as pd
import music21 as m21
import random
from scipy import signal
import seaborn as sns
import statistics
from scipy import stats as s
from scipy.interpolate import interp1d
from scipy import signal
from scipy.ndimage import filters
import IPython.display as ipd
import yaml
from yaml.loader import SafeLoader


###############################################
#Load variables from .yml file

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)

Fs = Fs_midi = Fs_symb = global_var["Fs"]
Fs_msp = global_var["Fs_msp"]
n_pitches = global_var["n_pitches"]
lowest_pitch = global_var["lowest_pitch"]
num_midi_notes = global_var["num_midi_notes"]
num_steps = global_var["num_steps"]
keys = global_var["keys"]
keys_ordered = global_var["keys_ordered"]

###############################################




def group_notes_in_time_frame(midi_list,
                              num_steps=11):

    """
    Function that divides the whole piece in equally spaced frames, and group
    in each of them the notes that start in that time period
    """

    dur = midi_list[-1][1]

    step_time_duration = dur/num_steps
    print(dur)

    time_frames = list(np.arange(0, dur+1, step_time_duration, dtype=float))
    #time_frames = list(np.arange(0, dur+1, 1/Fs_msp, dtype=float))
    print("Time frames: ", time_frames)

    time_frames_groups = [[y for y in midi_list if (y[0]>=time_frames[i] and y[0]<time_frames[i+1])] for i in range(len(time_frames)-1)]  

    return time_frames_groups


def filter_chord_repetition(detected_chords):

    """
        Function that checks if in different time frames there are the same detected notes
    """


    #order lists of notes for each time frame
    ordered_chords =  detected_chords.copy()
    for i in range(len(ordered_chords)):
        ordered_chords[i] = sorted(ordered_chords[i])

    #count list of notes in a time frame with length >0 before filtering
    len_before_filtering=0
    for k in range(len(ordered_chords)):
        if(len(ordered_chords[k])>0):   
            len_before_filtering+=1

    for i in range(num_steps):
        indices_same_value = [i]
        curr_pitches = ordered_chords[i]
        
        #skip if no detected notes here
        if(len(curr_pitches)==0):
            continue
        #find same values
        j=i+1
        while(j<num_steps):
            if (curr_pitches == ordered_chords[j]):
                indices_same_value.append(j)
            j+=1

        #choose random index between the time frames with the same notes
        chosen_index = random.choice(indices_same_value)
        #print("\nIteration", i, " | values: ", ordered_chords[i] , "indices with same value",   indices_same_value   ," | chosen value: ", chosen_index)

        #print("Indices with same value: ", indices_same_value)

        indices_same_value.remove(chosen_index)

        for index in indices_same_value:
            ordered_chords[index] = []

    #count values >0 after filtering
    len_after_filtering=0
    for k in range(len(ordered_chords)):
        if(len(ordered_chords[k])>0):
            len_after_filtering+=1


    print(len_before_filtering, len_after_filtering)

    return ordered_chords



def find_most_played_note_in_song_portion(detected_harmony,
                                          portion_size = 10):

    """
        Function that finds the most played notes in a certain portion of the input song.
        It takes an input the detected harmony for each time frame. 
        The number of elements belonging to a portion are counted only if the num of notes in the 
        detected harmony are more than 0.
        The function returns an array with the same size of the detected harmony,
        with the indication of the main note for each time frame.
    """


    count =0
    portion_harmony = []
    frame_pitches = []    #format: [limit frame, most played pitch until that frame]
    frame_limits = [0]
    portion_size = 10

    for i, curr_harm in enumerate(detected_harmony):
        len_harm = len(curr_harm)

        if(len_harm>0):

            #print("Frame n°", i, " | N° notes: ", len_harm, " | MPN: ", curr_harm[0], " | Count: ", count)
            portion_harmony.append(curr_harm)
            count += 1

            #retrieve most played pitch in the current portion
            if((count == portion_size) or (i == num_steps-1)):

                concat_harmony = [x for xs in portion_harmony for x in xs]   #flat list 
                most_played_pitches = get_sorted_list_most_played_pitches(concat_harmony)
                most_played_pitch = most_played_pitches[0]

                frame_pitches.append(most_played_pitch)
                frame_limits.append(i)
                #print("Frame limit: ", i , " | Most played note: ", most_played_pitch)
                #print("__________________\n")
                count = 0

                portion_harmony = []

    
    #assign most played note to each portion
    expanded_main_notes_list = np.zeros(num_steps, dtype = int)
    
    #frame_pitch_limit_list
    for i in range(len(frame_limits)-1):

        pitch = frame_pitches[i]
        low_limit = frame_limits[i]
        up_limit = frame_limits[i+1]
        #print("limits: ", low_limit, up_limit, " | pitch: ", pitch)
        expanded_main_notes_list[low_limit:up_limit] = pitch

    expanded_main_notes_list[up_limit:] = pitch    #to append the last note


    return expanded_main_notes_list




def get_num_pitches_per_time_frame(input_midi_list,
                                   num_steps = 101,
                                   plot_flag=False):

    """
    1) We divide the input song into equally and temporaly spaced time frames. (tot num of frames = num_steps)
    2) In each frame we group the notes that start in that time frame
    3) We calculate the highest and lowest pitch of that time frame, and we put these values into 2 different arrays
    4) We calculate the indication of a harmony range, which is the difference between the highest and lowest values
    5) We find the most relevant events harmony range. We consider those events the maximum and minimum relative peaks.
    6) Map these values in the range [0,10] to have for each step the number of pitches to play
    Note: the max num of notes to play in a frame is a parameter that has to be decided
    a-priori in order to let it work on the Max patch
    """


    time_frames_groups = group_notes_in_time_frame(input_midi_list,
                                                   num_steps=num_steps)

    max_arr = np.zeros(num_steps)
    min_arr = np.zeros(num_steps)
    range_arr = np.zeros(num_steps)

    for i, group in enumerate(time_frames_groups):
        
        if(len(group)>0):
            pitches = []
            for note in group:
                pitches.append(note[2])

            max_arr[i] = max(pitches)
            min_arr[i] = min(pitches)


    range_arr = max_arr - min_arr


    #find it with peaks

    #settings peaks
    
    #peaks, properties = signal.find_peaks(spectral_flux, 
    #                                      prominence=0.1,
    #                                      height = height,
    #                                      distance = 15)




    height = filters.median_filter(range_arr, 
                                   size=filter_size) + offset
                                   
    local_maxima, _ = signal.find_peaks(range_arr, 
                                        height=height,
                                        prominence=prominence,
                                        distance=distance)


    inverted_range_arr = (range_arr*-1) + max(range_arr)
    height = filters.median_filter(inverted_range_arr, 
                                   size=filter_size) + offset

    local_minima, _ = signal.find_peaks(inverted_range_arr,
                                        height=height,
                                        prominence=prominence,
                                        distance=distance)



    #map between 0 and 10 (num of notes of MAX)
    mapped_range_arr = np.array(map_values(range_arr,
                                           prev_min = min(range_arr),
                                           prev_max = max(range_arr),
                                           new_min = 0,
                                           new_max= 12), 
                                dtype=int)
    
    #array of detected events
    mapped_range_array_w_zeros = np.zeros(num_steps, dtype=int)

    mapped_range_array_w_zeros[local_maxima] = mapped_range_arr[local_maxima]
    mapped_range_array_w_zeros[local_minima] = mapped_range_arr[local_minima]


    #ToDo: filter values which are 0: in theory there is no problem because they are followed by other zero values




    if(plot_flag):

        fig = plt.figure(figsize=(20,8))
        plt.title("Range curve with detected events")
        plt.xlabel("Time frame")
        plt.ylabel("Range values")
        plt.plot(range_arr, color="green")
        plt.plot(local_maxima, range_arr[local_maxima], 'bo', label="Local Max")
        plt.plot(local_minima, range_arr[local_minima], 'ro', label="Local Min")
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()



        fig = plt.figure(figsize=(20,8))
        plt.title("Range curve with detected events (mapped in the number of notes to play)")
        plt.xlabel("Time frame")
        plt.ylabel("Num of note to play")
        plt.plot(mapped_range_arr, color="green")
        plt.plot(local_maxima, mapped_range_arr[local_maxima], 'bo', label="Local Max")
        plt.plot(local_minima, mapped_range_arr[local_minima], 'ro', label="Local Min")
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()


        fig = plt.figure(figsize=(20,8))
        plt.title("Detected events before filtering")
        plt.xlabel("Time frame")
        plt.ylabel("Num of note to play")
        plt.stem(mapped_range_array_w_zeros)

        plt.legend(loc="upper left")
        plt.grid()
        plt.show()


        fig = plt.figure(figsize=(20,8))


        #plt.plot(mapped_range_array_w_zeros, 'bo')
        plt.show()

    return mapped_range_array_w_zeros



def get_sorted_list_most_played_pitches(input_pitches):

    """
    Given an input list of pitches, it returns the list of the most played pitches,
    sorted from the most played to the less played
    """

    #save the number of time that a certain pitch has been detected (format = note: repetition)
    pitches_occurencies = {i:input_pitches.count(i) for i in input_pitches}

    #order the couple (note: repetition) from the less to the most repeated
    sorted_list_most_played_pitches = {k: v for k, v in sorted(pitches_occurencies.items(), key=lambda item: item[1])}

    #order list of pitches from the most repeated to the least repeated
    sorted_list_most_played_pitches = list(sorted_list_most_played_pitches)[::-1]

    return sorted_list_most_played_pitches



def retrieve_most_important_notes(midi_list, 
                                  num_steps=num_steps,
                                  chroma_flag=True,
                                  save_flag=False,
                                  plot_flag=True):

    """
        Importance = most played notes in the piano score in a certain time frame
    """


    time_frames_groups = group_notes_in_time_frame(midi_list,
                                                    num_steps=num_steps)

    #array which contains for each time frame the number of notes to send
    num_pitches_per_frame = get_num_pitches_per_time_frame(input_midi_list = midi_list,
                                                           num_steps = num_steps,
                                                           plot_flag = plot_flag)
    
    most_important_pitches = []
    detected_harmony_text = []
    detected_harmony = []
    list_chroma_letters = []


    for count, frame_group in enumerate(time_frames_groups):
        
        #retrieve all the pitches from a time group and sort them
        if(chroma_flag):
            detected_pitches = sorted(map(lambda x:x[2]%12 + 60, frame_group)) #offset + 60
        else:
            detected_pitches = sorted(map(lambda x:x[2], frame_group))


        pitches_occurencies_sorted_by_value = get_sorted_list_most_played_pitches(detected_pitches)

        try:
            most_important_pitch = pitches_occurencies_sorted_by_value[0]
            most_important_pitches.append(most_important_pitch)
        except:
            most_important_pitches.append(0)

        
        detected_pitches = pitches_occurencies_sorted_by_value[0:num_pitches_per_frame[count]]
        #print(detected_pitches)

        detected_harmony.append(detected_pitches)


    
    detected_filtered_harmony = filter_chord_repetition(detected_harmony)
    list_chroma_letters = midi_to_chroma_letter(detected_filtered_harmony)
    detected_filtered_harmony_text = list_to_text(detected_filtered_harmony)

    detected_harmony_chroma_letters = list_to_text(list_chroma_letters)

    if(save_flag):
        save_txt_list(values = detected_filtered_harmony_text,
                    filename= "top_played_notes.txt")
        
        save_txt_list(values = detected_harmony_chroma_letters,
            filename= "top_played_notes(chroma_letters).txt")

        ##save the most played note for each frame
        #save_txt_list(values = most_important_pitches,
        #             filename= "most_important_pitches.txt")


    return detected_harmony