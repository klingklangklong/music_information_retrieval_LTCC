import os
import sys
sys.path.append('scripts')
from utils import midi_to_list
from frequency_domain_features import multitrack_to_score, get_midi_chroma, compute_spectral_flux
from motiv_harmony_separation import split_motiv_harmony
from musical_domain_features import get_harmony_info, get_info_from_midi_list, retrieve_tracks_pitch_curves
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


#Load variables from .yml file
import yaml
from yaml.loader import SafeLoader

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

#for max msp
temporal_resolution = 1/Fs_msp
H = int(temporal_resolution * Fs)
W = int(H*2)
Fs_frame=Fs/H




def music_information_retrieval_MIDI(file_path):

    """
    Given a filename of a midi file, it estracts some musical features from it.
    """


    midi_list = midi_to_list(file_path)    

    #get some harmonic and structure information from the whole piece
    get_harmony_info(file_path)

    #get some musical info from the whole piece
    get_info_from_midi_list(midi_list, show_histograms=True)

    #get some musical info track by track
    retrieve_tracks_pitch_curves(midi_list)

    #split the song into a 2-track midi file: harmony and motiv.
    split_motiv_harmony(file_path, output_folder)


    #compute some frequency features
    score = multitrack_to_score(file_path, Fs_frame = Fs_frame, pypianoroll_flag=False)

    chroma = get_midi_chroma(score, Fs_frame)

    spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, show_peaks=False, title="Spectral Flux")

    spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, show_peaks=True, title="Spectral Flux with Onset Peaks")

    decaying_spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, decaying_flag=True, decaying_steps=40, title="Decaying Spectral Flux")


if __name__ == "__main__":

    file_path = os.path.join(input_folder, input_filename)

    print("Input file: ", input_filename)
    print("Input folder: ", input_folder)
    print("Output folder: ", output_folder)
    os.makedirs(output_folder, exist_ok=True) 

    input_folder = "input_data"
    filename = "Rebel-Le Cahos.mid"
    file_path = os.path.join(input_folder, filename)

    music_information_retrieval_MIDI(file_path)

    


    