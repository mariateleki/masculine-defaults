"""
get_english-fisher-annoatations.py

This script:
* Divides the podcast transcripts into 2 parts, and runs 2 model instances in parallel across 2 NVIDIA TITAN Xp to annotate the transcripts.
* Iterates through the podcast transcripts within each part to obtain the sentence-level parse tree for each transcript, and writes the parse tree (along with the dys, and the orig_dys outputs) into the df. 
* Saves the df at the end of annotating. 

Warning: This script takes a long time (days) to run.

conda activate english-fisher-annotations; CUDA_VISIBLE_DEVICES=0 python get_english-fisher-annotations.py -p 0
conda activate english-fisher-annotations; CUDA_VISIBLE_DEVICES=1 python get_english-fisher-annotations.py -p 1
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

# start the timer
start_time = time.time()

# set var
module_name = "english-fisher-annotations"

# set up logging
utils_general.just_create_this_dir("./logs")
logging.basicConfig(filename=f"./logs/{module_name}-{datetime.now().isoformat(timespec='seconds')}.log", level=logging.DEBUG)

# set up argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--part", type=int, choices=[0,1], required=False, help="Select the split of the data to run.")
args = parser.parse_args()

# function for running the annotator
def get_subprocess_cmd(input_path, output_path):
    cmd = f"""python main.py 
              --input-path {input_path}
              --output-path {output_path}
              --model ./model/swbd_fisher_bert_Edev.0.9078.pt"""
    cmd = cmd.split()
    return cmd


df = pd.read_csv(f"../csv/df-{args.part}.csv", index_col=0)

pbar = tqdm(total=len(df))

# initialize df output
df["parse"] = ""
df["orig_dys"] = ""
df["dys"] = ""

# set up temp dir
utils_general.just_create_this_dir("./temp")

# iterate through df and run the parser on each file/transcript
for index, row in df.iterrows():
    
    # some podcasts have no transcript
    if row["transcript"] != "":
        
        input_filepath = f"/home/grads/m/mariateleki/analysis-spotify/english-fisher-annotations/temp/temp-result-df-{args.part}.txt"
        utils_general.write_file(new_filename=os.path.basename(input_filepath), 
                                 directory=os.path.dirname(input_filepath), 
                                 text=str(row["transcript"]))
        
        output_filepath = f"/home/grads/m/mariateleki/analysis-spotify/english-fisher-annotations/temp/temp-result-df-{args.part}.txt"
        
        try: 
            # run the parser
            result = subprocess.call(get_subprocess_cmd(input_path=input_filepath, output_path=output_filepath))
            
            # then write the results into the df
            df.loc[index, "parse"] = utils_general.read_file(output_filepath.replace(".txt", "_parse.txt"))
            df.loc[index, "orig_dys"] = utils_general.read_file(output_filepath.replace(".txt", "_orig_dys.txt"))
            df.loc[index, "dys"] = utils_general.read_file(output_filepath.replace(".txt", "_dys.txt"))
            
        except Exception as e:
            
            df.loc[index, "parse"] = module_name
            df.loc[index, "orig_dys"] = module_name
            df.loc[index, "dys"] = module_name
            
            logging.debug(input_filepath, ":", e)
            traceback.print_exc()
            
            
    # update the progress bar 
    pbar.update(1)
    
    # check if it's time to exit
    elapsed_time = time.time() - start_time

    # Exit the script if 10 minutes have passed
    if elapsed_time >= 600:  # 10 minutes = 600 seconds
        
        # write out the results
        print("Writing out results so far to df.")
        csv_path = f"../csv/df-{module_name}-{args.part}.csv"
        df.to_csv(csv_path, header=True)
        
        # reset the clock
        start_time = time.time()
        
# write out results
print("Writing out final results.")
csv_path = f"../csv/df-{module_name}-{args.part}.csv"
df.to_csv(csv_path, header=True)

# close the progress bar
pbar.close()
