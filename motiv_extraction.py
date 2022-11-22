import os
import numpy as np
import libfmp.c4
import libfmp.b
import sys
sys.path.append('scripts')

from utils import midi_to_list, multitrack_to_score
from frequency_domain_features import get_midi_chroma
from plot_functions import plot_2Darray
from audio_thumbnailing_helper import *
from motiv_harmony_separation import *

#Load variables from .yml file
import yaml
from yaml.loader import SafeLoader
import argparse


#argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_filename", default="Rebel-Le Cahos.mid", help="filename of the midi file we want to parse")
ap.add_argument("-f", "--input_folder", default="input_data", help="folder which contains the midi files")
ap.add_argument("-o", "--output_folder", default="output_data", help="folder which contains the output data")

args = vars(ap.parse_args())


input_filename = args["input_filename"]
input_folder = args["input_folder"]
output_folder = args["output_folder"]
input_name = input_filename[:-4]
print(input_name)

output_folder = os.path.join(output_folder,input_name)



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


Fs_feature_motiv_extraction = global_var["Fs_feature_motiv_extraction"]
temporal_resolution = 1/Fs_feature_motiv_extraction
H = int(temporal_resolution * Fs)
W = int(H*2)
Fs_frame=Fs/H





def motiv_extraction(file_path):
    """
    This functions compute a motiv extraction from a midi file

    Args:
        file_path (str): path of the input MIDI file
    """
    split_motiv_harmony(file_path, output_folder)



    #filename_motiv = mir_path + name_song + "/" + "motiv.mid"
    filename_motiv = os.path.join(output_folder, "motiv.mid")

    #filename_mono = mir_path + name_song + "/" + "mono.mid"
    filename_mono = os.path.join(output_folder, "mono.mid")


    midi_list = midi_to_list(filename_mono)

    ##Case when Sp already exists, we don't need to re-compute it
    if(os.path.exists("SP.npy")):
        retrieve_motivs_from_SP(midi_list,
                                num_motivs=5,
                                Fs_frame = Fs_frame,
                                SP_path = Sp_path,
                                output_folder=output_folder)
        return
    

    score = multitrack_to_score(filename_mono, Fs, Fs_frame, n_pitches, lowest_pitch, H)

    ##get chroma
    chroma = get_midi_chroma(score, Fs_frame)

    ##compute similarity matrix
    S = compute_sm_dot(chroma,chroma)

    ##plot similarity matrix
    plot_2Darray(S, Fs_frame, figsize=(30,30), title="")


    print("calculating fitness...")
    #compute the fitness of all the segments
    SP_all = libfmp.c4.compute_fitness_scape_plot(S)        #SP_all = fitness, score, normalized score, coverage, normlized coverage. SP is the fitness
    SP=SP_all[0]
    np.save(os.path.join(output_folder, "SP"), SP)
    print("Fitness calculated!")

    seg = get_thumbnailing_segment(SP_all)

    #retrieve the main retrieved motiv
    retrieve_midi_thumbnail(seg, 
                            midi_list,
                            Fs_frame, 
                            output_folder = output_folder,
                            original_midi_list = midi_list,
                            save_flag=True)



    ###Retrieve from SP.npy file, more then one motiv (potentially, because they could be the same motiv if you see the segment corresponding)

    #Sp_path = os.path.join(output_folder, "SP.npy")
    #retrieve_motivs_from_SP(midi_list,
    #                        num_motivs=5,
    #                        Fs_frame = Fs_frame,
    #                        SP_path = Sp_path,
    #                        output_folder=output_folder)






if __name__ == "__main__":



    file_path = os.path.join(input_folder, input_filename)
    
    print("Input file: ", input_filename)
    print("Input folder: ", input_folder)
    print("Output folder: ", output_folder)
    os.makedirs(output_folder, exist_ok=True) 


    motiv_extraction(file_path)



    