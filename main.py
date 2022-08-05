##################################################################################################
import os
from utils import midi_to_list
from frequency_domain_features import multitrack_to_score, get_midi_chroma, compute_spectral_flux
from motiv_harmony_separation import split_motiv_harmony
#from harmony_generation import *
from musical_domain_features import get_harmony_info, get_info_from_midi_list, retrieve_tracks_info
#from triads_detection import *
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

    get_harmony_info(file_path)

    get_info_from_midi_list(midi_list, show_histograms=True)

    retrieve_tracks_info(midi_list)

    split_motiv_harmony(file_path)

    score = multitrack_to_score(file_path, Fs_frame = Fs_frame, pypianoroll_flag=False)

    chroma = get_midi_chroma(score)

    spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, show_peaks=False)

    spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, show_peaks=True)

    decaying_spectral_flux = compute_spectral_flux(chroma, Fs_frame=Fs_frame, decaying_flag=True, decaying_steps=40)


if __name__ == "__main__":
    input_folder = "input_data"
    filename = "Rebel-Le Cahos.mid"
    file_path = os.path.join(input_folder, filename)

    print(file_path)
    music_information_retrieval_MIDI(file_path)


    