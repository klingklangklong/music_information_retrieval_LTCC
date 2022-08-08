import numpy as np
import os
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from scipy.ndimage import filters
from scipy import signal
import matplotlib.pyplot as plt
from utils import *
import random


###############################################
#Load variables from .yml file

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)

#Fs = Fs_midi = Fs_symb = global_var["Fs"]
#Fs_msp = global_var["Fs_msp"]
#n_pitches = global_var["n_pitches"]
#lowest_pitch = global_var["lowest_pitch"]
#num_midi_notes = global_var["num_midi_notes"]
#num_steps = global_var["num_steps"]
#keys = global_var["keys"]
#keys_ordered = global_var["keys_ordered"]

###############################################


def get_triad_from_notes(midi_note, 
                         input_midi_harmony,
                         file_txt):

    """
    Function that given a midi note, it returns the major and minor triads 
    where the considered note is the tonic.
    The triads are accepted only if all the notes are contained into the input midi harmony
    Note: the triad values are in the chroma range (0-12)
    """

    major_triad_offset = [0, 4, 7]
    minor_triad_offset = [0, 3, 7]

    major_triad = []
    minor_triad = []

    major_triad_keys = []
    minor_triad_keys = []

    input_midi_harmony = [x%12 for x in input_midi_harmony]

    out_triads = []


    for offset in major_triad_offset:
        curr_note = ( (midi_note%12) + offset ) % 12
        major_triad.append(curr_note)
        major_triad_keys.append(key_number_to_key_name(curr_note, show_octave=False))

    for offset in minor_triad_offset:
        curr_note = ( (midi_note%12) + offset ) % 12
        minor_triad.append(curr_note)
        minor_triad_keys.append(key_number_to_key_name(curr_note, show_octave=False))


    #print("Note: ", key_number_to_key_name(midi_note, show_octave=False), "| Major triad :" , major_triad_keys, " | Minor Triad: ", minor_triad_keys)


    if (set(major_triad) <= set(input_midi_harmony)):
        #print("Tonic note -> ", key_number_to_key_name(midi_note, show_octave=False), "| Major triad :" , major_triad_keys)
        found_chord = "\nTonic note -> " + str(key_number_to_key_name(midi_note, show_octave=False)) + "| Major triad :" + str(major_triad_keys)
        file_txt.write(found_chord)

        out_triads.append(major_triad)

    if (set(minor_triad) <= set(input_midi_harmony)):
        #print("Tonic note -> ", key_number_to_key_name(midi_note, show_octave=False), " | Minor Triad: ", minor_triad_keys)
        found_chord = "\nTonic note -> " + str(key_number_to_key_name(midi_note, show_octave=False)) + " | Minor Triad: " + str(minor_triad_keys)
        file_txt.write(found_chord)

        out_triads.append(minor_triad)

    out_triads = [x for x in out_triads]
    return out_triads




def harmony_to_filtered_triads(detected_harmony, expanded_main_notes_list, output_folder=''):
    """
        Given a harmony (list of most played notes), the function finds the triads for each time frame following these rules:
        1) For each time frame, we get the major and minor triads for each of the detected notes.
        2) We take only the triads where all the notes are contained in the current detected harmony
    """

    out_filename = "all_found_triads.txt"
    out_file_path = os.path.join(output_folder, out_filename)

    file = open(out_file_path, 'w')

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
    print("Written output file: ", out_filename)

    return found_triads



def get_one_triad_per_frame(found_triads, expanded_main_notes_list, num_steps, output_folder):
    """
    Function that select one triad per time frame following these rules:
    Given the filtered triads we analyze each time frame
    If in a frame there is more than one triad, we select the most similar one to the triad found in the previous step.
    The first triad is chosen randomly from the triads found in a certain frame.
    #Note: most similarity is the sum of the differences between each sorted chroma value of the 2 triads. 
    #lowest = most similar
    """

    out_filename = "one_triad_per_frame.txt"
    out_file_path = os.path.join(output_folder, out_filename)

    file = open(out_file_path, 'w')

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
    print("Written output file: ", out_filename)


    return output_midi_triads


def order_triads(input_list_triads,
                 detected_harmony,
                 expanded_main_notes_list,
                 output_folder = "",
                 save_flag=True):
    
    """
    Function that orders the notes of the triad according to their position in the most played notes list. (detected harmony)
    For each note of the triad, we find the index of this note into the detected harmony in that time frame.
    Then, we create a list for each triad where each cell is [index in the harmony, note].
    We sort this list according to the index values, 
    and we get the notes of the triad ordered.

    """
    
  
    #out_filename = "ordered_triads.txt"
    #out_file_path = os.path.join(output_folder, out_filename)
    #file = open(out_file_path, 'w')

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


            #if(len(current_triad)>0):

            #    file.write("\nFrame " + str(i).zfill(2) +
            #               " | Main note: " +  keys[expanded_main_notes_list[i]%12] +
            #               " | Ordered triad : " + str(midi_to_chroma_list(current_triad)) +
            #               "  ( Non-ordered triad : " + str(midi_to_chroma_list(triad))+
            #               " | Detected Harmony: " + str(midi_to_chroma_list(input_midi_harmony)) + " )"
            #               )

            #else:
            #    file.write("\nFrame " + str(i).zfill(2) +
            #               " | Main note: " +  keys[expanded_main_notes_list[i]%12] 
            #               )


        #else:
        #    
        #    file.write("\nFrame " + str(i).zfill(2) + " | Main note: " + keys[expanded_main_notes_list[i]%12])   
        #    ordered_list_triads.append([])  

    #file.close()
    #print("Written output file: ", out_file_path)
    
    return ordered_list_triads



def invert_and_filter_triads(input_list_triads,
                            detected_harmony):
    
    """

    Input: we have maximum one triad per time frame.
    
    1) Inversion: we move from triad to triad with the smoothest possible line in the melody voice (first note of the triad).
        Starting from the second triad, we order it putting on top the most similar note to the first one of the previous triad
    2) Filtering: a chord can be repeated only once in two consecutive steps. At the third time, we do an inversion.

    """

    #file = open("inverted filtered triads.txt", 'w')

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
    #write_harmony_file(input_triads = ordered_list_triads,
    #                   expanded_main_notes_list = expanded_main_notes_list,
    #                   input_harmony = detected_harmony,
    #                   name_file = song_name +".txt")
    
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
            return input_list

    return input_list


@DeprecationWarning
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



def get_three_shifted_triad_list(triads_list, detected_harmony, expanded_main_notes_list, song_name="", output_folder=""):

    """
    Given a list of triads, it returns the three different version considering the three different shift.
    ToDo: explain it the steps
    """


    for shift in range(3):
        ordered_triads_list = order_triads(triads_list,
                                           detected_harmony,
                                           expanded_main_notes_list,
                                           output_folder,
                                           save_flag=False)

        ordered_triads_list = shift_first_triad(ordered_triads_list,
                                                shift=shift)

        inv_filt_triads_list = invert_and_filter_triads(ordered_triads_list,
                                                    detected_harmony)

        filename_output = song_name + "_shift_" + str(shift) +".txt"
        write_harmony_file(input_triads = inv_filt_triads_list,
                           expanded_main_notes_list = expanded_main_notes_list,
                           input_harmony = detected_harmony,
                           name_file = filename_output,
                           output_folder = output_folder)

        print("Written: ", filename_output)




def filter_chord_repetition(detected_chords):

    """
        Function that checks if in different time frames there are the same detected notes
    """


    #order lists of notes for each time frame
    ordered_chords =  detected_chords.copy()

    num_steps = len(ordered_chords)

    for i in range(num_steps):
        ordered_chords[i] = sorted(ordered_chords[i])

    

    #count list of notes in a time frame with length >0 before filtering
    len_before_filtering=0
    for k in range(num_steps):
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


    #print(len_before_filtering, len_after_filtering)

    return ordered_chords





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






def find_most_played_note_in_song_portion(detected_harmony,
                                          portion_size = 10,
                                          num_steps = 100):

    """
        Function that finds the most played notes in a certain portion of the input song.
        It takes an input the detected harmony for each time frame. 
        The number of elements belonging to a portion are counted only if the num of notes in the 
        detected harmony are more than 0.
        The function returns an array with the same size of the detected harmony,
        with the indication of the main note for each time frame.


        portion_size: num of steps in which finding the pad note. To do, parametrize it.
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




def retrieve_most_important_notes(midi_list, 
                                  num_steps=100,
                                  output_folder = "",
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
                        filename= "top_played_notes.txt",
                        save_path=output_folder)
        
        save_txt_list(values = detected_harmony_chroma_letters,
                      filename= "top_played_notes(chroma_letters).txt",
                      save_path=output_folder)

        ##save the most played note for each frame
        #save_txt_list(values = most_important_pitches,
        #             filename= "most_important_pitches.txt")


    return detected_harmony


def group_notes_in_time_frame(midi_list,
                              num_steps=11):

    """
    Function that divides the whole piece in equally spaced frames, and group
    in each of them the notes that start in that time period
    """

    dur = midi_list[-1][1]

    step_time_duration = dur/num_steps
    #print("Song duration: ", dur)

    time_frames = list(np.arange(0, dur+1, step_time_duration, dtype=float))
    #time_frames = list(np.arange(0, dur+1, 1/Fs_msp, dtype=float))
    #print("Time frames: ", time_frames)

    time_frames_groups = [[y for y in midi_list if (y[0]>=time_frames[i] and y[0]<time_frames[i+1])] for i in range(len(time_frames)-1)]  

    return time_frames_groups



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


    prominence=7
    distance=2
    offset=2
    filter_size=25



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


