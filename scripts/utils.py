import pretty_midi
import os
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import pypianoroll

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)
keys = global_var["keys"]



def midi_to_list(filename):
    """
    Function that takes a the filename of a midi file in input and returns a midi list of event extracted from it
    """

    #load file
    midi_data = pretty_midi.PrettyMIDI(filename)
    midi_list = []   

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity

            midi_list.append([start, end, pitch, velocity, instrument.name])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))   

    return midi_list


def key_number_to_key_name(key_number, show_octave=True):
    """
    #Modified from Pretty Midi
    Convert a key number to a key string.
    Parameters
    ----------
    key_number : int
        Uses pitch classes to represent major and minor keys.
    Returns
    -------
    key_name : str
        Format: note name + num octave

    """

    if not isinstance(key_number, int):
        raise ValueError('`key_number` is not int!')
    if not ((key_number >= 0) and (key_number <= 128)):
        raise ValueError('`key_number` is lower than 0 or larger than 128')

    #chroma index
    key_idx = key_number % 12

    
    if(show_octave):
        octave = key_number // 12
        output = keys[key_idx] + str(octave)
    
    else:
        output = keys[key_idx]


    return output



def pitches_to_classes(pitches):
    """
    Given a list of pitches, it returns the list of pitch classes found
    """

    pitch_classes_density = np.zeros(12, dtype=int)
    #pitch_classes = []

    pitch_classes_found = []


    for pitch in pitches:
        pitch_class = key_number_to_key_name(pitch, show_octave=False)
        pitch_classes_density[pitch%12] +=1
        pitch_classes_found.append(pitch_class)

    pitch_classes_found = list(set(pitch_classes_found))


    return pitch_classes_density, pitch_classes_found



def get_frequency_registers(pitches):

    l_reg = []
    m_reg = []
    h_reg = []

    first_limit = 54
    second_limit = 72

    num_pitches = len(pitches)

    for p in pitches:
        if((p>=0) and (p<=first_limit)):
            l_reg.append(p)
        elif((p>first_limit) and (p<=second_limit)):
            m_reg.append(p)
        elif((p>second_limit) and (p<128)):
            h_reg.append(p)

    l_reg_perc = np.round((len(l_reg) / num_pitches)*100,2)
    m_reg_perc = np.round((len(m_reg) / num_pitches)*100,2)
    h_reg_perc = np.round((len(h_reg) / num_pitches)*100,2)

    print("Fraction of notes between MIDI pitches 0 and 54: ", l_reg_perc, " %")
    print("Fraction of notes between MIDI pitches 55 and 72: ", m_reg_perc, " %")
    print("Fraction of notes between MIDI pitches 73 and 127: ", h_reg_perc, " %")



def convert_feature_to_time_domain(input_list, Fs_frame):
    output_list = [x / Fs_frame for x in input_list]
    return output_list


def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average


def compute_relative_tempo_difference(times):

    relative_times = np.diff(times, )
    
    return relative_times


def map_values(input_values, 
               prev_min=0,
               prev_max=1,
               new_min=0,
               new_max=128):



    if((prev_min==0) and (prev_max==0)):
        prev_min = min(input_values)
        prev_max = max(input_values)

    left_span = prev_max - prev_min
    right_span = new_max - new_min

    mapped_values = []

    for v in input_values:
        mapped_v = int(np.round(new_min + (float(v - prev_min) / float(left_span)) * right_span))     #round is to get the int part
        
        mapped_values.append(mapped_v)

    return mapped_values


def list_to_text(input_list):

    """
        Utility function used to make a list in a format readable by a Max collection
    """

    output_text = []

    for row in input_list:
        row_txt = str(row)
        row_txt = row_txt.replace(",", " ")
        row_txt = row_txt.replace("[", "")
        row_txt = row_txt.replace("]", "")

        output_text.append(row_txt)

    return output_text


def midi_to_chroma_list(input_midi_list):

    output_key_list = []

    for elem in input_midi_list:
        key = key_number_to_key_name(int(elem), show_octave=False)
        output_key_list.append(key)

    return output_key_list


def midi_to_chroma_letter(list_midi_values):


    list_chroma_letters = []

    if(len(list_midi_values) > 0):
        for frame_notes in list_midi_values:
            curr_chroma_letters = []

            for midi_note in frame_notes:
                curr_chroma_letter = keys[midi_note%12]
                curr_chroma_letters.append(curr_chroma_letter)

            list_chroma_letters.append(curr_chroma_letters)
            #curr_chroma_letter = keys[val%12]
            #list_chroma_letters.append(curr_chroma_letter)

    return list_chroma_letters


def add_pitch_offset_to_midi_list(input_list,
                                  offset_pitch = 0):

    """
    Function that adds an offset to each midi pitch of a list
    """

    output_list = []
    for i in range(len(input_list)):
        if(len(input_list[i])>0):
            for j in range(len(input_list[i])):
                input_list[i][j] += offset_pitch


    return input_list



def delete_zero_values(input_list):
    
    """
    Function used to substitue zero values with the first non-zero value 
    that comes in chronological order
    """

    index_list = np.arange(len(input_list))

    df = pd.DataFrame(data=input_list, index=index_list, columns = ['A'])

    output_list = df['A'].replace(to_replace=0, method='ffill').values
    
    print(output_list)

    return output_list


def get_start_end_frame(midi_note, Fs):
    """
    Given a midi note, it returns the start and the end frame, according to a certain Fs
    """

    start_time = midi_note[0]
    dur_time = midi_note[1]

    start_frame = int(start_time*Fs)
    end_frame = int(dur_time*Fs)

    return start_frame, end_frame


def get_2Dpitch_curves(track_midi_list):
    
    #given a midi list of one instrument, it returns a 2D list of the x and y axis of the pitch curve

    x_axis = [] #start time positions
    y_axis = [] #pitches

    
    for note_event in track_midi_list:
        x_axis.append(note_event[0])
        y_axis.append(note_event[2])

    pitch_curve2D = list(zip(x_axis, y_axis))

    return pitch_curve2D



def write_harmony_file(input_triads,
                       expanded_main_notes_list,
                       input_harmony,
                       name_file = "",
                       output_folder = ""):
    
    """
        Given a list of triads, it writes the triads and the harmony detected
        only in the time frame where we have a detected triad
    """
    filename = os.path.join(output_folder, name_file)
    file = open(filename, 'w')


    for i in range(len(input_triads)):
        
        if(len(input_triads[i])== 0 ):      #and (len(input_harmony[i])==0)
            file.write("\nFrame " + str(i).zfill(2) +
                " )  Pad note: " +  str(keys[expanded_main_notes_list[i]%12]) )            
            
        #elif(len(input_triads[i])== 0 and (len(input_harmony)==0)):
        #    file.write("\nFrame " + str(i).zfill(2) +
        #        " )  Pad note: " +  str(keys[expanded_main_notes_list[i]%12]) )    

        else:
            file.write("\nFrame " + str(i).zfill(2) +
                " )  Pad note: " +  str(keys[expanded_main_notes_list[i]%12]) +
                " || Chosen triad : " + str(midi_to_chroma_list(input_triads[i])) +
                " || Most played notes: " + str(midi_to_chroma_list(input_harmony[i])) )

    file.close()

    return



def save_txt_list(values, filename="out.txt",  save_path=""):
    
    filename = os.path.join(save_path, filename)

    file = open(filename, 'w')


    for i, value in enumerate(values):
        row_txt = str(i) + ", " + str(value) + ";\n"
        file.write(row_txt)


    file.close()
    return



def save_txt_array(times, values, filename="out.txt", save_path=""):

    filename = os.path.join(save_path, filename) 
    
    file = open(filename, 'w')

    for i, (time,value) in enumerate(zip(times,values)):
        #row_txt = str(i) + ", " + str(time) + " " + str(value) + ";\n"
        row_txt = str(i) + ", " + str(value) + ";\n"
        file.write(row_txt)

    file.close()
    return




def save_txt_2Darray(values, filename="out.txt", save_path=""):

    filename = os.path.join(save_path, filename)
    file = open(filename, 'w')

    for i, value in enumerate(values):

        row_txt = str(value[0]) + ", " + str(value[1]) + ";\n"
        file.write(row_txt)
    

    file.close()
    return


def list_to_pitch_activations(note_list, num_frames, frame_rate):
    
    'The parameter peakheight_scalefactor was taken from the function "midird4_to_pitchOnsetPeaks" '
    
    offset = 1
    P = np.zeros((128, num_frames))

    for l in note_list:
        
        start_frame = max(0, int(l[0] * frame_rate))  
        end_frame = min(num_frames, int(l[1] * frame_rate) ) + offset  ##Original +1. Try +2 to align to the maltab one
        P[int(l[2]), start_frame:end_frame] = l[3]         
        
    return P


def multitrack_to_score(filename, Fs, Fs_frame, n_pitches, lowest_pitch, H):
    """
    Given a filename of a midi file, it returns a matrix version of the musical score
    """

    pypianoroll_algorithm_flag = False


    if (pypianoroll_algorithm_flag==True):
        beat_resolution = 4  #n° of time steps in a quarter note (16th-note allowed)
        #beat_resolution = 24  #n° of time steps in a quarter note (16th-note allowed)


        multitrack  = pypianoroll.read(filename, resolution=beat_resolution)   

        n_tracks = len(multitrack.tracks)
        n_timesteps = multitrack[0].pianoroll.shape[0]
        piano_roll_shape = (n_timesteps, n_pitches)

        output_score = np.zeros(shape = piano_roll_shape)

        i=0
        while(i<n_tracks):
            print(multitrack.tracks[i].pianoroll.shape)
            output_score[:,:] += multitrack.tracks[i].pianoroll[:,lowest_pitch:lowest_pitch + n_pitches]
            i+=1

        output_score = output_score.T

    else:
        
        midi_list = midi_to_list(filename)
        dur_symb = midi_list[-1][1]    ##release of the last note = dur midi file
        print("time duration: ", dur_symb)
        print("Fs: ", Fs)
        len_symb = int(np.ceil(dur_symb * Fs))    #n samples with that Fs
        print("len symb: ", len_symb)
        num_features = int(len_symb / H)   #n° features
        print("n features: ", num_features)

        output_score_128_pitches = list_to_pitch_activations(midi_list, num_features, Fs_frame)

        output_score = np.zeros((n_pitches, num_features))
        output_score[:,:] = output_score_128_pitches[lowest_pitch:lowest_pitch + n_pitches, :]
        
    print(output_score.shape)
    
    return output_score


def find_index_in_array(value, input_array):

    for i, elem in enumerate(input_array):
        if(elem == value):
            #print("index: ", i)
            found_idx = i

    return found_idx