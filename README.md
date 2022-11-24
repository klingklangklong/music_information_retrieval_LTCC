# Music Information Retrieval Tools 
This tool has been used to extract meaningul information from existing musical pieces.
This information has been used to compose new songs for the Listening to Climate Change concert produced by kling klang klong.


## 1. Setup


1- Clone the repository

```
git clone https://klingklangklong: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@github.com/klingklangklong/Listening_to_Climate_Change.git
```

2- Install requirements:

```
pip install libfmp pretty_midi pypianoroll 

pip install music21==7.3.1 
```


## 2. Execution

### I) Music information Retrieval algorithm

Extract MIR information from an input MIDI file.

```
python 'music_information_retrieval.py' --input_filename 'Rebel-Le Cahos.mid' --input_folder 'input_data' --output_folder 'output_data'
```

Specify the ```input_filename``` of your input MIDI file, located in the folder ```input_folder```. 

In the ```output_folder```, a new folder with the input song name will be created (in this case Rebel-Le Cahos). It will contain the generated output files
 

- ```mono.mid``` : input MIDI file compressed in one single track
- ```motiv_harmony_split.mid```: input file separated in two different tracks, the harmony and motiv.
- ```motiv.mid```: the motiv track extracted from the input file
- ```harmony.mid```: harmony track extracted from the input file: the harmony track extracted from the input track



### II) Harmony generation algorithm

Generate a harmony structure taking inspiration from an input MIDI file with a MIR-based algorithm.

The algorithm is explained step-by-step in ``documentation/Harmony Generation Algorithm.txt``

```
python 'harmony_generation.py' --input_filename 'Rebel-Le Cahos.mid' --input_folder 'input_data' --output_folder 'output_data'
```

Output files generated:

- ```all_found_triads.txt``` : file which indicates for each frames the resulting detected harmony (most played notes), the most played in the region and all the possible minor and major triads that it's possible to generate.

- ```one_triad_per_frame.txt```: file with only one selected triad per frame.

- ```.._shift_0.txt / .._shift_1.txt / .._shift_2.txt```: final output list using the three different shift for the first triad.



### III) Motiv extraction algorithm

The algoritm is based on https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_AudioThumbnailing.html

```
python 'motiv_extraction.py' --input_filename 'Rebel-Le Cahos.mid' --input_folder 'input_data' --output_folder 'output_data'
```

Output file generated:

- ```retrieved.mid```: MIDI file of the retrieved motiv.


## 3. Global parameters
Global parameters are exposed in the ```global.yaml``` file, and can be changed if necessary.


## 4. Colab notebook

Path: ```notebooks/LTCC_Colab.ipynb```

## 5. Contact
For any question, or problem contact Carmelo: carmelo@klingklangklong.com
