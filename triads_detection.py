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



def harmony_to_filtered_triads(detected_harmony):
    """
        Given a harmony (list of most played notes), the function finds the triads for each time frame following these rules:
        1) For each time frame, we get the major and minor triads for each of the detected notes.
        2) We take only the triads where all the notes are contained in the current detected harmony
    """

    file = open("demo.txt", 'w')

    found_triads = []

    for i, harm in enumerate(detected_harmony):
        separator_txt = ("\n\n-----------------------------------------------------------------\n")
        file.write(separator_txt)

        frame_txt = "Frame n° " + str(i).zfill(2) + "| Most played note: " + keys[expanded_main_notes_list[i]%12] + " | Detected harmony: " + str(midi_to_chroma_list(harm))
        #print(frame_txt)
        file.write(frame_txt)

        #print("Frame n°", str(i).zfill(2), "| Most played note: ", keys[expanded_main_note_list[i]%12], " | Detected harmony: ", midi_to_chroma_list(harm))
        if(len(harm)>0):
            file.write("\n\nPossible triads:")

            found_triads_in_frame = []


            for note in harm:

                #for each note of the detected harmony, we find its triads
                curr_note_found_triads = get_triad_from_notes(note, 
                                                              harm,
                                                              file)

                #append to all the triads detected in a frame
                found_triads_in_frame.append(curr_note_found_triads)


            #flat list of triads detected in a certain time frame
            found_triads_in_frame = [x for xs in found_triads_in_frame for x in xs]

            found_triads.append(found_triads_in_frame)

        else:
            found_triads.append([])

    file.close()

    return found_triads



def get_one_triad_per_frame(found_triads):
    """
    Function that select one triad per time frame following these rules:
    Given the filtered triads we analyze each time frame
    If in a frame there is more than one triad, we select the most similar one to the triad found in the previous step.
    The first triad is chosen randomly from the triads found in a certain frame.
    #Note: most similarity is the sum of the differences between each sorted chroma value of the 2 triads. 
    #lowest = most similar
    """
    file = open("found triads.txt", 'w')

    i = 0
    repetitions_counter = 0
    previous_triad = []
    output_list_triads = []
    output_midi_triads = []

    while i<num_steps:
        
        #triads belonging to the current time frame
        curr_frame_triads = found_triads[i]

        #go to next step if there are not triads in the current time frame
        if(len(curr_frame_triads)==0):
            output_list_triads.append([])
            output_midi_triads.append([])
            file.write("\nFrame " + str(i).zfill(2) + " | Main note: " + keys[expanded_main_notes_list[i]%12])

            i = i+1
            continue

        else:

            #case: we haven't found a triad yet. choose randomly one of the available triads.
            #NOTE: now it takes the first triad.
            if(len(previous_triad)==0):
                #previous_triad = random.choice(curr_frame_triads)
                previous_triad = curr_frame_triads[2]
                output_list_triads.append(midi_to_chroma_list(previous_triad))
                output_midi_triads.append(previous_triad)
                file.write("\nFrame " + str(i).zfill(2) + " | Main note: " +  keys[expanded_main_notes_list[i]%12] + " | Triad : " + str(midi_to_chroma_list(previous_triad)))

            else:

                differences = []

                #calculate similarity with the previously found triad and the ones found in this time frame
                for triad in curr_frame_triads:

                    sorted_previous_triad = sorted(previous_triad)
                    sorted_current_triad = sorted(triad)

                    #sum of the differences of sorted values between sorted triads
                    diff_triads = sum([abs(sorted_previous_triad[i] - sorted_current_triad[i]) for i in range(3) ])
                    differences.append(diff_triads)
                
                #find the index of the triad in the current frame which is the most similar to the triad found in the previous step
                min_difference = min(differences)
                index_min = differences.index(min_difference)
                current_triad = curr_frame_triads[index_min]

                output_list_triads.append(midi_to_chroma_list(current_triad))
                output_midi_triads.append(current_triad)

                previous_triad = current_triad

                
                file.write("\nFrame " + str(i).zfill(2) + " | Main note: " +  keys[expanded_main_notes_list[i]%12] + " | Triad : " + str(midi_to_chroma_list(curr_frame_triads[index_min])))


        i = i+1

    file.close()

    return output_midi_triads


def order_triads(input_list_triads,
                 detected_harmony):
    
    """
    Function that orders the notes of the triad according to their position in the most played notes list. (detected harmony)
    For each note of the triad, we find the index of this note into the detected harmony in that time frame.
    Then, we create a list for each triad where each cell is [index in the harmony, note].
    We sort this list according to the index values, 
    and we get the notes of the triad ordered.

    """

    file = open("ordered triads.txt", 'w')

    ordered_list_triads = []

    for i, triad in enumerate(input_list_triads):

        input_midi_harmony = [x%12 for x in detected_harmony[i]]

        current_indexed_triad = []

        if(len(input_midi_harmony)>0):

            for note in triad:
                index = input_midi_harmony.index(note)
                current_indexed_triad.append([index, note])
            
            #sort notes by the indeces of the detected harmony
            current_indexed_triad = sorted(current_indexed_triad, key=lambda x: (x[0]))  

            #retrieve the sorted chroma values
            current_triad = [x[1] for x in current_indexed_triad]
 
            ordered_list_triads.append(current_triad)


            if(len(current_triad)>0):

                file.write("\nFrame " + str(i).zfill(2) +
                           " | Main note: " +  keys[expanded_main_notes_list[i]%12] +
                           " | Ordered triad : " + str(midi_to_chroma_list(current_triad)) +
                           "  ( Non-ordered triad : " + str(midi_to_chroma_list(triad))+
                           " | Detected Harmony: " + str(midi_to_chroma_list(input_midi_harmony)) + " )"
                           )
                
            else:
                file.write("\nFrame " + str(i).zfill(2) +
                           " | Main note: " +  keys[expanded_main_notes_list[i]%12] 
                           )




        else:
            file.write("\nFrame " + str(i).zfill(2) + " | Main note: " + keys[expanded_main_notes_list[i]%12])   
            ordered_list_triads.append([])  

    file.close()
    
    return ordered_list_triads


def invert_and_filter_triads(input_list_triads,
                            detected_harmony):
    
    """

    Input: we have maximum one triad per time frame.
    
    1) Inversion: we move from triad to triad with the smoothest possible line in the melody voice (first note of the triad).
        Starting from the second triad, we order it putting on top the most similar note to the first one of the previous triad
    2) Filtering: a chord can be repeated only once in two consecutive steps. At the third time, we do an inversion.

    """

    file = open("inverted filtered triads.txt", 'w')

    ordered_list_triads = []
    previous_triad = []
    current_triad = []
    repetition_counter = 0

    for i, triad in enumerate(input_list_triads):

        input_midi_harmony = [x%12 for x in detected_harmony[i]]

        current_indexed_triad = []

        if(len(triad)>0):
            
            if(len(previous_triad)==0):
                previous_triad = triad
                #print(previous_triad)

                
                ordered_list_triads.append(previous_triad)
                continue

            else:

                current_triad = triad

                first_note_previous_triad = previous_triad[0]

                differences_with_prev_main_note = []

                #calculate the relative difference between the first pitch of the
                #previous triad and all the pitches of the current triad
                for note_index, note in enumerate(current_triad):
                    #print(keys[first_note_previous_triad], keys[note])
                    difference = min(np.abs(first_note_previous_triad - note), 
                                     np.abs((first_note_previous_triad+12) - note),
                                     np.abs((first_note_previous_triad-12) - note))
                    differences_with_prev_main_note.append([note_index, difference])

                #sort list from lowest to highest difference
                differences_with_prev_main_note = sorted(differences_with_prev_main_note, key=lambda x: (x[1]))  

                #select the shift value which is the index position of the most similar note in the current triad
                shift_value = differences_with_prev_main_note[0][0]

                unrolled_triad = current_triad

                #roll the triad putting on top the most similar note
                current_triad = list(np.roll(current_triad, 
                                             -shift_value))

                ################################################################

                #Check repetitions: If we have more than 2 consecutive triads, 
                #we roll it obtaining a new inversion
                sum = 0
                for j in range(len(current_triad)):
                    sum += np.abs(current_triad[j] - previous_triad[j])
                #print("sum of differences: ", sum)

                #if we have no repetition in 2 consecutive frames, the counter is set to 0
                if(sum==0):
                    repetition_counter+=1
                else:
                    repetition_counter = max(0, repetition_counter-1)
                

                if(repetition_counter>1):
                    #print("Triad rolled!")
                    current_triad = list(np.roll(current_triad, 
                                                 1))       
                    repetition_counter=0            

                ################################################################

                previous_triad = current_triad

                ordered_list_triads.append(current_triad)



        else:
                
                ordered_list_triads.append([])

    #write output file
    write_harmony_file(input_triads = ordered_list_triads,
                       expanded_main_notes_list = expanded_main_notes_list,
                       input_harmony = detected_harmony,
                       name_file = song_name +".txt")
    
    return ordered_list_triads


def shift_first_triad(input_list,
                       shift=0):
    """
    Given a list of triads, it shifts only the first triad by a number of position 
    given by the parameter "shift".
    """ 

    for i in range(len(input_list)):
        if (len(input_list[i])>0):
            input_list[i] = list(np.roll(input_list[i], shift))
            print(input_list[i])
            return input_list

    return input_list


def shift_list_notes(input_list,
                     num_shift=3):

    """
    Function used to create 3 different versions of a list of triads, 
    by shifting the order of the notes of each triad.

    """

    #for each shift
    for i in range(num_shift):
        shifted_list = []
        for j, elem in enumerate(input_list):

            shifted_list.append(list(np.roll(input_list[j], i)))


        write_harmony_file(input_triads = shifted_list,
                           expanded_main_notes_list = expanded_main_notes_list,
                           input_harmony = detected_harmony,
                           name_file = song_name + "_shift_" + str(i) +".txt")
        print(shifted_list)


#shift_list_notes(inv_filt_triads_list,num_shift=3)


