import os
import pretty_midi
from utils import midi_to_list
import argparse
from harmony_extraction_helper import *


#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--num_steps", default=100, type=int, help="num of steps in which we divide the input song")
ap.add_argument("-i", "--input_filename", default="Rebel-Le Cahos.mid", help="filename of the midi file we want to parse")
ap.add_argument("-f", "--input_folder", default="input_data", help="folder which contains the midi files")
ap.add_argument("-o", "--output_folder", default="output_data", help="folder which contains the output data")

args = vars(ap.parse_args())
num_steps = args["num_steps"]
input_filename = args["input_filename"]
input_folder = args["input_folder"]
output_folder = args["output_folder"]







def triads_list_extractions(file_path):

    midi_data = pretty_midi.PrettyMIDI(file_path)
    midi_list = midi_to_list(file_path)


    detected_harmony = retrieve_most_important_notes(midi_list,
                                                    num_steps=num_steps,
                                                    output_folder=output_folder,
                                                    save_flag=False,
                                                    plot_flag=False)

    #get pad notes
    expanded_main_notes_list = find_most_played_note_in_song_portion(detected_harmony,
                                                                    portion_size = 10,
                                                                    num_steps=num_steps)

    
    #get the triads
    found_triads = harmony_to_filtered_triads(detected_harmony, 
                                              expanded_main_notes_list,
                                              output_folder)     


    #get one triad per frame
    triads_list = get_one_triad_per_frame(found_triads, 
                                          expanded_main_notes_list,
                                          num_steps=num_steps,
                                          output_folder = output_folder); 


    ##order notes in the triad
    #ordered_triads_list = order_triads(triads_list,
    #                               detected_harmony)
#
    ##order notes in the triad
    #inv_filt_triads_list = invert_and_filter_triads(ordered_triads_list,
    #                                                detected_harmony)

    #shift the triad list
    get_three_shifted_triad_list(triads_list, 
                                 detected_harmony, 
                                 expanded_main_notes_list, 
                                 song_name=input_filename[:-4], 
                                 output_folder = output_folder)                                             


if __name__ == "__main__":

    
    file_path = os.path.join(input_folder, input_filename)

    print("Input file: ", input_filename)
    print("Input folder: ", input_folder)
    print("Output folder: ", output_folder)

    os.makedirs(output_folder, exist_ok=True) 

    triads_list_extractions(file_path)