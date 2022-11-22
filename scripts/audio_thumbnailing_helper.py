import pretty_midi
import matplotlib.pyplot as plt
import os
import numpy as np
from libfmp import *
from matplotlib.colors import ListedColormap
from utils import find_index_in_array


def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb from libfmp library.

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S


def get_thumbnailing_segment(SP_all):
    """This function returns the indices of the feature vector containing the segmen with maximum fitness

    Args:
        SP_all (np.ndarray): array of fitness matrices

    Returns:
        seg (list): list of frames delimiting the retrieved segment of audio frames corresponding to the motiv
    """
    SP = SP_all[0]
    N = SP.shape[0]
    print(N)
    value_max = np.max(SP)
    arg_max = np.argmax(SP)
    ind_max = np.unravel_index(arg_max, [N, N])

    seg = [ind_max[1], ind_max[1]+ind_max[0]]

    return seg


def retrieve_midi_thumbnail(segment, original_midi_list, Fs_frame, output_folder="", save_flag=False, out_name = ""):

    """
    This function creates and save the retrieved MIDI motiv


    Args:
        segment (list): list of frames delimiting the retrieved segment of audio frames corresponding to the motiv
        original_midi_list (list): list of midi notes
        Fs_frame (int): sample rate of the feature vector
        output_folder (str): path where the retrieved midi motiv is saved
        save_flag (boolean): if true, it saves the motiv
        out_name(str): name of the output file.
    """

    out_file = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=program)


    time_start = segment[0] / Fs_frame
    time_end = segment[1] / Fs_frame

    print("Segment [frame]", segment[0], segment[1])
    print("Segment [seconds]:" ,time_start, time_end)

    for note_event in original_midi_list:

        start = note_event[0]
        end = note_event[1]
        
        #note = pretty_midi.Note(   velocity=100, pitch=note_number, start=0, end=.5)
    
        if((end>time_start) and (start<time_end)):
            note = pretty_midi.Note(velocity = note_event[3],
                                    pitch = note_event[2],
                                    start = start - time_start,
                                    end = end - time_start)
            cello.notes.append(note)


    if(save_flag):

        out_file.instruments.append(cello)
        out_motiv_path = os.path.join(output_folder, 'retrieved' + out_name + '.mid')
        out_file.write(out_motiv_path)
        print("Retrieved MIDI thumbnail saved: ", out_motiv_path)


def retrieve_motivs_from_SP(midi_list, num_motivs, Fs_frame, SP_path, output_folder):

    """
    This function retrieve a certain number of motivs from a melody. 
    The motivs are retrieved using a pre-computed fitness of similarty matrix on a feature vector, 
    following this audio thumbnailing algorithm:
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_AudioThumbnailing.html


    Args:
        midi_list (list): list of notes 
        num_motivs (int): num of motivs we want to retrieve
        Fs_frame (int): sample rate of the feature vector
        SP_path (str): path of the pre-computed SP (fitness of the similarity matrix)
        output_folder (str): path of the saved midi file corresponding to the motivs
    """

    SP = np.load(SP_path)         #load pre-computed SP (fitness of similarity matrix)
    N = SP.shape[0]

    SP_flat = SP.flatten();                                         #flat SP into 1d array
    SP_flat_sorted = sorted(SP_flat, reverse=True);                 #sort values from higher to lower
    list_max_values = SP_flat_sorted[0:num_motivs]                  #take only the first N5 values


    #for each value retrieved, find the segment of the melody that contains the motiv
    #Notes from experiments: it finds always the same area more or less, that means that there are more path in the same zone 
    for i, max_value in enumerate(list_max_values):

        curr_value = max_value
        curr_idx_max = find_index_in_array(curr_value, SP_flat)
        curr_indices = np.unravel_index(curr_idx_max, [N,N])

        seg_low_bound = seg_high_bound = 0
        seg_low_bound = curr_indices[1]
        seg_high_bound = curr_indices[1] + curr_indices[0]

        time_low_bound = seg_low_bound / Fs_frame
        time_high_bound = seg_high_bound / Fs_frame

        curr_frame_segment = [seg_low_bound, seg_high_bound]
        curr_time_segment = [time_low_bound, time_high_bound]

        print("\n" + str(i) + ")")

        retrieve_midi_thumbnail(segment = curr_frame_segment, 
                        original_midi_list = midi_list,
                        Fs_frame = Fs_frame,
                        output_folder = output_folder,
                        save_flag=True,
                        out_name = str(i))