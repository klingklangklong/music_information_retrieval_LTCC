import matplotlib.pyplot as plt
import pretty_midi
import numpy as np
import seaborn as sns
import yaml
from yaml.loader import SafeLoader
import pandas as pd
from utils import get_2Dpitch_curves, convert_feature_to_time_domain, midi_to_list
import sys
sys.path.append('../')

###############################################
#Load variables from .yml file

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)

Fs_msp = global_var["Fs_msp"]
n_pitches = global_var["n_pitches"]
lowest_pitch = global_var["lowest_pitch"]
###############################################


def plot_1Darray(x_axis, y_axis, figsize = (16,8), xlabel="", ylabel=""):
    """
    This function creates a plot given a x-axis and a y-axis array. 

    Args:
        x_axis (np.ndarray): x-axis array
        y_axis (np.ndarray): y-axis array
        figsize (tuple, optional): size of the figure. Defaults to (16,8).
        xlabel (str, optional): label of the x-axis. Defaults to "".
        ylabel (str, optional): label of the y-axis. Defaults to "".
    """
    fig = plt.figure(figsize = figsize)
    plt.plot(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_2Darray(array, Fs_frame, figsize=(16,8), cmap='gray_r', xlabel="", ylabel="", yticks ="", title="",):

    """
    This function creates a plot given a n input 2D array

    Args:
        array (np.ndarray): input 2D array
        Fs_frame (int): sample rate
        figsize (tuple, optional): size of the figure. Defaults to (16,8).
        cmap (str): color map of the plot
        xlabel (str, optional): label of the x-axis. Defaults to "".
        ylabel (str, optional): label of the y-axis. Defaults to "".
        yticks (str, optional): labels of the value on the y-axis.
        title (str, optional): title of the plot
    """

    fig,ax = plt.subplots(1,1,figsize = figsize, dpi=72)

    im = ax.imshow(array,
        cmap=cmap,
        aspect='auto',
        origin = 'lower')
    
    x_axis_features = list(range(0,array.shape[1]))
    x_axis_time = convert_feature_to_time_domain(x_axis_features, Fs_frame)

    #print tick every 10 seconds
    time_step = int(10*Fs_msp)
    plt.xticks(x_axis_features[0::time_step], x_axis_time[0::time_step])

    if yticks:
        plt.yticks(ticks = list(range(0, array.shape[0])),
                   labels = yticks)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im)
    plt.title(title)
    plt.show()


def plot_curve_pitches(filename):
    """
    This function plots one pitch curve for each instrumental track of the input midi file

    Args:
        filename (str): path of the input midi file
    """
    df = pd.DataFrame()
    single_2D_curve = []
    list_df = []        #list of dataframes
    i=0
    off = 5
    figure = plt.figure(figsize=(20, 8))

    midi_data = pretty_midi.PrettyMIDI(filename)
    midi_list = midi_to_list(filename)
    instruments = list(set(map(lambda x:x[4], midi_list)))
    multitrack_midi_list = [[y for y in midi_list if y[4]==x] for x in instruments]     #each element is a midi list a single track     

    for track_midi_list in multitrack_midi_list:
        pitch_curve2D = get_2Dpitch_curves(track_midi_list)
        pitch_curve2D_array = np.array(pitch_curve2D).T
        plt.plot(pitch_curve2D_array[0], pitch_curve2D_array[1])   #,  linestyle = 'dashed'
        plt.ylim([lowest_pitch,lowest_pitch+n_pitches])
        i+=1

    plt.legend(instruments)   
    plt.xlabel("Time [s]")
    plt.ylabel("MIDI Pitch")
    plt.grid()
    plt.show()   


def plot_pitch_histogram(data, num_bin, binrange, xlabel):
    """
    This function plots a histogram of the input data

    Args:
        data (np.ndarray): input data
        num_bin (int): number of bins of the histogram
        binrange (int): highest value for bin edges
        xlabel (str): label of the x-axis
    """
    
    fig = plt.figure(figsize = (20,6))

    x = sns.histplot(data = data,
                     binrange = (0,binrange),
                     bins = num_bin,
                     kde=True,
                    )
    
    plt.xlabel(xlabel)
    plt.show()
