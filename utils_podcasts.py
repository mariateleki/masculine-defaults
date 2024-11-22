import json
import os
import pandas as pd
import numpy as np
import string

from urlextract import URLExtract

import utils_general

def get_file(show_filename_prefix, episode_filename_prefix):
    
    dir1 = show_filename_prefix.split("_")[1][0].upper()
    dir2 = show_filename_prefix.split("_")[1][1].upper()
    
    filepath = os.path.join(utils_general.PATH_TO_TEXT_DIR, dir1, dir2, show_filename_prefix, episode_filename_prefix+".txt")
    
    text = ""
    text = utils_general.read_file(filepath)
    
    return text, filepath

def clean_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, "")
    return text

def get_description_for_episode_id(episode_id, testset_or_trainset):
    # get the description from the appropriate dataframe
    if testset_or_trainset == "testset":
        description_series = TESTSET_METADATA_DF.loc[TESTSET_METADATA_DF["episode_filename_prefix"] == episode_id, "episode_description"]
        description = str(description_series.values[0])
    else:  # testset_or_trainset == "trainset"
        description_series = TRAINSET_METADATA_DF.loc[TRAINSET_METADATA_DF["episode_filename_prefix"] == episode_id, "episode_description"]
        description = str(description_series.values[0])
        
    # clean the descriptions the same way as the transcripts
    description = clean_urls(description)
    description = description.encode("ascii", "ignore").decode()
    
    return description

# modified from https://github.com/potsawee/podcast_trec2020/blob/main/data/processor.py
# also performs some basic cleaning
def get_transcript_text_from_json_asr_file(json_asr_file):
    transcript_list = []
    with open(json_asr_file) as f:
        transcript_dict = json.loads(f.read())
        
        results_list = [r for r in transcript_dict["results"]]
        last_result = results_list[-1]
        
        for word_dict in last_result["alternatives"][0]["words"]:
            transcript_list.append(word_dict["word"])
        
        transcript_string = " ".join(transcript_list)
        
        # clean the transcripts the same way as the descriptions
        transcript_string = clean_urls(transcript_string)
        transcript_string = transcript_string.encode("ascii", "ignore").decode()
        
        if transcript_string[-1] not in string.punctuation:
            transcript_string += "."
        
        return transcript_string
