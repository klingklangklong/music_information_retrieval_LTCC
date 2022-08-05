import matplotlib.pyplot as plt
from utils import convert_feature_to_time_domain, midi_to_list
import pretty_midi
import numpy as np
import seaborn as sns
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


def plot_1Darray(x_axis, y_axis, figsize = (16,8), xlabel="", ylabel=""):

    fig = plt.figure(figsize = figsize)
    plt.plot(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #ax.plot(x_axis, y_axis)


def plot_2Darray(array, figsize=(16,8), cmap='gray_r', xlabel="", ylabel="", yticks =""):
    fig,ax = plt.subplots(1,1,figsize = figsize, dpi=72)

    im = ax.imshow(array,
        cmap=cmap,
        aspect='auto',
        origin = 'lower')
    

    x_axis_features = list(range(0,array.shape[1]))
    x_axis_time = convert_feature_to_time_domain(x_axis_features)

    #print tick every 10 seconds
    time_step = int(10*Fs_frame)
    plt.xticks(x_axis_features[0::time_step], x_axis_time[0::time_step])

    if yticks:
        plt.yticks(ticks = list(range(0, array.shape[0])),
                   labels = yticks)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im)
    plt.show()






def plot_curve_pitches(filename):
    
    #it gets a midi_list and returns 1 pitch curve for each instrumental track

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
    
    fig = plt.figure(figsize = (20,6))

    x = sns.histplot(data = data,
                     binrange = (0,binrange),
                     bins = num_bin,
                     kde=True,
                    )
    
    plt.xlabel(xlabel)
    plt.show()
