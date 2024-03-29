
import matplotlib.pyplot as plt
import pypianoroll
import numpy as np
import libfmp.c6
import random
from scipy import signal
from scipy.ndimage import filters

from plot_functions import plot_2Darray, plot_1Darray
from utils import midi_to_list, convert_feature_to_time_domain, map_values, save_txt_list

import yaml
from yaml.loader import SafeLoader
import sys
sys.path.append('../')

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



def multitrack_to_score(filename, Fs_frame, pypianoroll_flag = False):
    """
    Given a filename of a midi file, this function returns the corresponding feature in the frequency domain of the musical score

    Args:
        filename (str): path of the midi file
        Fs_frame (int): sample rate of the feature vector
        pypianoroll_flag (bool, optional): if true, it uses the pypianoroll library to create the feature. 
                                           Otherwise it uses the pitch activation from libfmp library. Defaults to False.

    Returns:
        output_score (np.ndarray): output feature vector
    """


    H = Fs/Fs_frame
    if(pypianoroll_flag==True):
        beat_resolution = 4  #n° of time steps in a quarter note (16th-note allowed)
        
        multitrack  = pypianoroll.read(filename, resolution=beat_resolution)   

        n_tracks = len(multitrack.tracks)
        n_timesteps = multitrack[0].pianoroll.shape[0]
        piano_roll_shape = (n_timesteps, n_pitches)

        output_score = np.zeros(shape = piano_roll_shape)

        i=0
        while(i<n_tracks):
            output_score[:,:] += multitrack.tracks[i].pianoroll[:,lowest_pitch:lowest_pitch + n_pitches]
            i+=1

        output_score = output_score.T
    
    else:
        midi_list = midi_to_list(filename)
        dur_symb = midi_list[-1][1]    ##release of the last note = dur midi file
        len_symb = int(np.ceil(dur_symb * Fs))    #n samples with that Fs
        print("len symb: ", len_symb)
        num_features = int(len_symb / H)   #n° features
        print("n features: ", num_features)


        output_score_128_pitches = list_to_pitch_activations(midi_list, num_features, Fs_frame)

        output_score = np.zeros((n_pitches, num_features))
        output_score[:,:] = output_score_128_pitches[lowest_pitch:lowest_pitch + n_pitches, :]


    plot_2Darray(output_score,
                 Fs_frame,
                 xlabel="Time[s]",
                 ylabel="MIDI Pitch",
                 title="Frequency score")

    return output_score


def list_to_pitch_activations(note_list, num_frames, frame_rate):
    """
    This function is used to create a pitch activation frequency feature starting from a list of notes

    Args:
        note_list (list): input list of midi notes
        num_frames (int): num frames of the feature
        frame_rate (int): sample rate of the feature

    Returns:
        P (np.ndarray): pitch activation feature vector
    """
    
    offset = 1
    P = np.zeros((128, num_frames))

    for l in note_list:
        
        start_frame = max(0, int(l[0] * frame_rate))  
        end_frame = min(num_frames, int(l[1] * frame_rate) ) + offset  ##Original +1. Try +2 to align to the maltab one
        P[int(l[2]), start_frame:end_frame] = l[3]         
        
    return P


def normalize_feature_sequence(X, norm=2, threshold=0.0001, v=None):
    """    
    This function normalized the input feature sequence
    The norm value in input is used as the order of normalization of the vector of ones. 
    The resulting normalization is a kind of l^p norm

    Args:
        X (np.ndarray): input feature
        norm (int, optional): normalization order. Defaults to 2.
        threshold (float, optional): _description_. Defaults to 0.0001.
        v (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    K, N = X.shape
    X_norm = np.zeros((K, N))
    

    if v is None:
        v = np.ones(12, dtype=np.float64)
        v = v / np.linalg.norm(v, norm);
        
    for n in range(N):
        s = np.linalg.norm(X[:,n], norm)
        
        if s > threshold:
            X_norm[:, n] = X[:, n] / s
            
        else:
            X_norm[:, n] = v


    return X_norm


def get_midi_chroma(score, Fs_frame):
    """
    Given an input feature in the frequency domain, this function calculates the corresponding normalized chroma feature

    Args:
        score (np.ndarray): input feature in the frequency domain
        Fs_frame (int): sample rate of the feature vectors

    Returns:
        chroma_midi_norm (np.ndarray): normalized chroma feature
    """

    chroma_midi = np.zeros((12, score.shape[1]))

    for i in range(score.shape[0]):
        indChroma = np.mod(i+24,12)
        chroma_midi[indChroma, :] += score[i, :]
        
    threshold=0.001
    chroma_midi_norm = normalize_feature_sequence(chroma_midi, norm=2, threshold=threshold)         #normalization

    plot_2Darray(chroma_midi_norm,
                 Fs_frame,
                 xlabel = "Time [s]",
                 ylabel = "Pitch",
                 yticks = keys,
                 title="Chromagram")

    

    return chroma_midi_norm



def compute_decaying(input_feature, decaying_steps=10):    
    """
    This function computes a decay to the input feature

    Args:
        input_feature (np.ndarray): input feature
        decaying_steps (int, optional): number of decaying steps. Defaults to 10.

    Returns:
        feature_decay (np.ndarray): feature with decaying
    """


    len_novelty_spectrum = input_feature.shape[0]       
    novelty_spectrum_decay = np.zeros(len_novelty_spectrum)       
    
    filtercoef = np.sqrt(1./np.arange(1,decaying_steps+1))
    num_coef = len(filtercoef)    
        

    v_shift = input_feature
    v_help = np.zeros((num_coef, len_novelty_spectrum))

    
    for n in range(num_coef):
        v_help[n, :] = filtercoef[n] * v_shift
        v_shift = np.roll(v_shift, 1)
        v_shift[0] = 0  
        
    feature_decay = np.max(v_help, axis=0)

    return feature_decay




def compute_spectral_flux(chroma, Fs_frame, decaying_flag=False, decaying_steps=10, show_peaks=False, title=""):
    """
    This function calculates the spectral flux feature.

    Args:
        chroma (np.ndarray): chroma feature
        Fs_frame (_type_): sample rate of the feature vectors
        decaying_flag (bool, optional): if true, it calculates a decaying feature vector. Defaults to False.
        decaying_steps (int, optional): num of decaying steps. Defaults to 10.
        show_peaks (bool, optional): if true, the feature plot is showd. Defaults to False.
        title (str, optional): title of the plot. Defaults to "".

    Returns:
        spectral flux (np.ndarray): spectral flux feature
    """
    #chroma -> spectral flux

    Y = chroma
    Y_diff = np.diff(Y)

    symb_flag = True
    if(symb_flag==True):   ##first sample problem in symbolic file 
        for i in range(Y.shape[0]):
            Y_diff[i,0] = Y[i,0] - Y[i,1]

    Y_diff[Y_diff < 0] = 0; Y_diff.shape
    spectral_flux = np.sum(Y_diff, axis=0); #sum over the 12-pitches axes

    #parameters
    norm = 1
    M = 10 

    #add 0 at the beginning (?)
    spectral_flux = np.concatenate((spectral_flux, np.array([0.0])))

    if M > 0:
        local_average = libfmp.c6.compute_local_average(spectral_flux, M)    #calculate the local average in each point
        spectral_flux = spectral_flux - local_average                     #curve - average    
        spectral_flux[spectral_flux < 0] = 0.0                            #rectification
    
    if norm == 1:
        max_value = max(spectral_flux)
        if max_value > 0:
            spectral_flux = spectral_flux / max_value                     #normalization

    if(decaying_flag):
        spectral_flux = compute_decaying(spectral_flux, decaying_steps)

    #conversion features -> time (deprecated)
    x_axis_features = list(range(0,len(spectral_flux)))
    x_axis_time = convert_feature_to_time_domain(x_axis_features, Fs_frame)


    #peaks calculation
    height = filters.median_filter(spectral_flux, size=8) + 0.1
    peaks, properties = signal.find_peaks(spectral_flux, 
                                          prominence=0.1,
                                          height = height,
                                          distance = 15)

    #time coefficients
    T_coef = np.arange(spectral_flux.shape[0]) / Fs_frame

    
    plot_1Darray(T_coef, spectral_flux, xlabel="Time[s]", ylabel="Amplitude")

    if(show_peaks):
        peaks_sec = T_coef[peaks]
        plt.plot(peaks_sec, spectral_flux[peaks], 'ro')
    
    plt.title(title)


    plt.show()


    #define a new array with values only where the peaks are
    peaks_array = np.zeros(Y.shape[1])
    peaks_array[peaks] = spectral_flux[peaks]

    #map float values in the new range 0-128
    peaks_array = map_values(peaks_array,
                             prev_min = min(peaks_array),
                             prev_max = max(peaks_array),
                             new_min = 0,
                             new_max= 128)

    #              filename="spectral_peaks.txt")
    
    

    return spectral_flux