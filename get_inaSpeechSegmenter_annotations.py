"""
get_inaSpeechSegmenter_annotations.py

This script:
* Truncates each podcast audio to 30 seconds. 
* Runs inaSpeechSegmenter on that 30 seconds of audio, to obtain labeled segments: male, female, music, noEnergy, noise.
* Totals the number of seconds for each category, and writes that total into the output df. 
* Writes the resulting df out to file. 

inaSpeechSegmenter: https://github.com/ina-foss/inaSpeechSegmenter

env = inaSpeechSegementer8, which was created by:
conda create -n inaSpeechSegmenter8 python=3.10 pip ipykernel
conda activate inaSpeechSegmenter8
pip install tensorflow[and-cuda]
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cd inaSpeechSegmenter/
pip install -e .
cd ..
export LD_LIBRARY_PATH=/home/grads/m/mariateleki/anaconda3/envs/inaSpeechSegementer8/lib/

conda activate inaSpeechSegmenter8; CUDA_VISIBLE_DEVICES=0 python get_inaSpeechSegmenter_annotations.py -p 0
conda activate inaSpeechSegmenter8; CUDA_VISIBLE_DEVICES=1 python get_inaSpeechSegmenter_annotations.py -p 1
conda activate inaSpeechSegmenter8; CUDA_VISIBLE_DEVICES=2 python get_inaSpeechSegmenter_annotations.py -p 2
conda activate inaSpeechSegmenter8; CUDA_VISIBLE_DEVICES=3 python get_inaSpeechSegmenter_annotations.py -p 3
"""

import os
import json
import pathlib
import subprocess
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import pandas as pd
import traceback

# allows the import of utils files from the upper directory
import sys
sys.path.append("..")
import utils_general
import utils_podcasts

import time
import math

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid

# save environment information for each run of this file
result = subprocess.run("conda list", shell=True, capture_output=True, text=True)
with open(f"./env/{os.path.basename(os.path.abspath(__file__))}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as file:
    file.write(os.environ['CONDA_DEFAULT_ENV'] + "\n")
    file.write(result.stdout)

# start the timer
start_time = time.time()

# set var
module_name = "inaSpeechSegmenter"

# set up logging
utils_general.just_create_this_dir("./logs")
logging.basicConfig(filename=f"./logs/{module_name}-{datetime.now().isoformat(timespec='seconds')}.log", level=logging.DEBUG)

# set up argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--part", type=int, choices=[0,1,2,3], required=False, help="Select the split of the data to run.")
args = parser.parse_args()

# get ogg audio file
def get_ogg_filepath(show_filename_prefix, episode_filename_prefix):
    
    dir1 = show_filename_prefix.split("_")[1][0].upper()
    dir2 = show_filename_prefix.split("_")[1][1].upper()
    
    # correct based on bug with dir names on Spotify's end
    if (dir1 == "7") and (dir2 == "Q"):
        dir2 = "Q (aasishp@spotify.com)"
    
    # correct based on bug with dir names on Spotify's end
    if show_filename_prefix in ["show_2E5eZu8zXmIAOpyd7dRJG1","show_46S1p4KzB0aeEZdYCs2mHb","show_4HrZvmf6lHa8Nm50sKTu8E"]:
        show_filename_prefix = show_filename_prefix+" (aasishp@spotify.com 2)"
    
    filepath = os.path.join("/data2/maria/Spotify-Podcasts/podcasts-audio-only-2TB/podcasts-audio", dir1, dir2, show_filename_prefix, episode_filename_prefix+".ogg")
    
    return filepath

# function for truncating audio to 30 seconds
def get_subprocess_cmd(input_path, output_path, time_to_truncate_to_in_seconds):
    cmd = f"""ffmpeg -hide_banner -loglevel error -ss 0 -t {time_to_truncate_to_in_seconds} -i"""
    cmd = cmd.split()
    
    # the paths may have spaces in them from the Spotify dataset, so their paths get appended next
    cmd.append(f"{input_path}")
    cmd.append(f"{output_path}")
    return cmd

df = pd.read_csv(f"./csv/df-4-{args.part}.csv", index_col=0)

pbar = tqdm(total=len(df))

# initialize df output for the new cols
df["segmentation"] = None
for attr in ["female", "male", "music", "noEnergy", "noise"]:
    df[attr] = None

# initialize segmenter
seg = Segmenter()

# set up temp dir
utils_general.just_create_this_dir("./temp")

# iterate through df and run the parser on each file/transcript
for index, row in df.iterrows():
    try: 

        # get the input audio ogg file
        input_filepath = get_ogg_filepath(row["show_filename_prefix"], row["episode_filename_prefix"])

        # set up temp result filepath for ffmpeg
        temp_result_filepath = f"./temp-files/temp-result-{module_name}-4-{str(args.part)}.ogg"
        utils_general.delete_file_if_already_exists(temp_result_filepath)

        # trim and convert the file
        result = subprocess.run(get_subprocess_cmd(input_path=input_filepath, 
                                          output_path=temp_result_filepath, 
                                          time_to_truncate_to_in_seconds=30))

        # run the module
        segmentation = seg(temp_result_filepath)
        df.loc[index, "segmentation"] = str(segmentation)

        # write out the file as a csv
        output_filepath = f"/home/grads/m/mariateleki/analysis-spotify/temp/temp-result-df-{args.part}.txt"
        seg2csv(segmentation, output_filepath)

        # read the results in a table
        episode_id = df["episode_filename_prefix"]
        ina_df = pd.read_table(output_filepath)

        # compute the length of each sequence
        ina_df["length"] = ina_df['stop'] - ina_df['start']

        # store the aggregated data in a new data frame
        ina_df_aggregated = ina_df[['labels', 'length']].groupby("labels").sum()

        for attribute in ["female", "male", "music", "noEnergy", "noise"]:
            if attribute in ina_df_aggregated.index:
                df.loc[index, attribute] = ina_df_aggregated["length"][attribute]
            else:
                # for example: there were no male voice segments present in the 30 second audio clip
                df.loc[index, attribute] = 0.0

    except Exception as e:

        for attribute in ["female", "male", "music", "noEnergy", "noise", "segmentation"]:
            df.loc[index, attribute] = module_name

        logging.debug(input_filepath, ":", e)
        traceback.print_exc()
            
    # update the progress bar 
    pbar.update(1)
    
    # check if it's time to exit
    elapsed_time = time.time() - start_time

    # write out results and reset the clock if 10 minutes have passed, or it's the last df run
    if elapsed_time >= 600:
        
        # write out results
        print("Writing out results so far to df.")
        csv_path = f"./csv/df-{module_name}-4-{args.part}.csv"
        df.to_csv(csv_path, header=True)
                                                  
        # reset the clock
        start_time = time.time()
        
# write out results
print("Writing out final results.")
csv_path = f"./csv/df-{module_name}-4-{args.part}.csv"
df.to_csv(csv_path, header=True)

# close the progress bar
pbar.close()

