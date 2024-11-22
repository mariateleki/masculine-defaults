# imports and setup
import os
import re
import string
import sys
import subprocess
import datetime
import argparse

import numpy as np

import utils_general
from tqdm import tqdm

import torch
from torch.nn.functional import cosine_similarity

import pandas as pd
pd.set_option('display.max_columns', None)

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

import nltk
nltk.download('punkt')

import utils_embeddings

def get_topic_list(file):
    with open(file, "r") as file:
        file_contents = file.read()
        words = file_contents.split()
        return list(words)

def find_common_words_with_positions(list1, list2):
    common_words = []
    for word in list1:
        if word in list2:
            positions = (list1.index(word), list2.index(word))
            common_words.append((word, positions))
    return common_words

def get_embeddings_for_string(string, model):
    if model == "chatgpt":
        return torch.tensor(client.embeddings.create(input = [string], model="text-embedding-3-large").data[0].embedding)
    else:  # "llama"
        ret = utils_embeddings.get_llama_embeddings([string])[0][0]
        print(ret)
        return ret

def get_m_w_cosine_total(input_string, model):

    input_string_embedding = get_embeddings_for_string(input_string, model)
    m_cosine_total = 0.0
    w_cosine_total = 0.0

    # https://github.com/pliang279/sent_debias/blob/master/debias-BERT/gender_tests/weat6b.jsonl
    for mw_tuple in [("male","female"),("man","woman"),("boy","girl"),("brother","sister"),("he","she"),("him","her"),("his","hers"),("son","daughter")]:
        m = mw_tuple[0]
        w = mw_tuple[1]

        m_embedding = get_embeddings_for_string(m, model)
        w_embedding = get_embeddings_for_string(w, model)

        m_cosine_sim = cosine_similarity(input_string_embedding.unsqueeze(0), m_embedding.unsqueeze(0))
        m_cosine_total += m_cosine_sim
        # print("cosine_sim between '{}' and '{}': {:.4f}".format(word, m, cosine_sim.item()))

        w_cosine_sim = cosine_similarity(input_string_embedding.unsqueeze(0), w_embedding.unsqueeze(0))
        w_cosine_total += w_cosine_sim
        # print("cosine_sim between '{}' and '{}': {:.4f}".format(word, w, cosine_sim.item()))

    return (m_cosine_total.item(), w_cosine_total.item())

def count_words_from_list_in_string(word_list, input_string):
    input_string = re.sub(r'[^\w\s]', '', input_string).lower()

    word_occurrences = {}
    for word in word_list:
        pattern = r'\b' + re.escape(word) + r'\b'
        count = len(re.findall(pattern, input_string, re.IGNORECASE))
        word_occurrences[word] = count

    return [(word, count) for word, count in word_occurrences.items() if count > 0]

def get_random_sentences(text, num_samples=5):
    sentences = nltk.sent_tokenize(text)

    # Generate random starting indices for selecting 3 consecutive sentences
    random_indices = np.random.randint(0, len(sentences) - 2, num_samples)

    # Extract 3 consecutive sentences starting from each random index
    random_sets = []
    for idx in random_indices:
        three_sentences = sentences[idx:idx+3]
        random_sets.append(" ".join(three_sentences))

    return random_sets

def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, "")
    return text

def word_matches(word, word_list):
    cleaned_word = remove_punctuation(word).lower()
    return any(cleaned_word == w for w in word_list)

def is_first_char_capitalized(s):
    if len(s) > 0:
        return s[0].isupper()
    else:
        return False

# save environment information for each run
result = subprocess.run("conda list", shell=True, capture_output=True, text=True)
with open(f"./env/{os.path.basename(os.path.abspath(__file__))}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as file:
    file.write(os.environ['CONDA_DEFAULT_ENV'] + "\n")
    file.write(result.stdout)

# Now redirect all file output to stdout
output_file = f"./experiment_runs/{os.path.basename(os.path.abspath(__file__))}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
file_handle = open(output_file, "w")
sys.stdout = file_handle

# set up parameters
parser = argparse.ArgumentParser()
parser.add_argument("--N_SAMPLES", type=int, default=100)
parser.add_argument("--TOP_N", type=int, default=5)
parser.add_argument("--GENDER", type=str, default="male")
parser.add_argument("--NUM_SAMPLES", type=int, default=3)
parser.add_argument("--MIN_SWAPS", type=int, default=3)
parser.add_argument("--NUM_SECONDS", type=float, default=20.0)
parser.add_argument("--SEED", type=int, default=42)
parser.add_argument("--NUM_API", type=int, default=5)
parser.add_argument("--MODEL", type=str)
parser.add_argument("--BERTOPICS", type=int, default=0)  # 1 is True, 0 is False
args = parser.parse_args()

N_SAMPLES = args.N_SAMPLES
TOP_N = args.TOP_N
GENDER = args.GENDER
NUM_SAMPLES = args.NUM_SAMPLES
MIN_SWAPS = args.MIN_SWAPS
NUM_SECONDS = args.NUM_SECONDS
SEED = args.SEED
NUM_API = args.NUM_API
MODEL = args.MODEL
BERTOPICS = args.BERTOPICS

np.random.seed(SEED)

print("N_SAMPLES:", N_SAMPLES)
print("TOP_N:", TOP_N)
print("GENDER:", GENDER)
print("NUM_SAMPLES:", NUM_SAMPLES)
print("MIN_SWAPS:", MIN_SWAPS)
print("NUM_SECONDS:", NUM_SECONDS)
print("SEED:", SEED)
print("MODEL:", MODEL)
print("BERTOPICS:", BERTOPICS)

df = pd.read_csv("./csv/df-all-features.csv")
df_female = df[df["female"]>=NUM_SECONDS][0:N_SAMPLES].reset_index(drop=True)
df_male = df[df["male"]>=NUM_SECONDS][0:N_SAMPLES].reset_index(drop=True)

if BERTOPICS == 0:
    original_male_word_list = get_topic_list("topic60.txt")[0:15]
    original_female_word_list = get_topic_list("topic62.txt")[0:15]

    print(len(original_male_word_list), ":", original_male_word_list)
    print(len(original_female_word_list), ":", original_female_word_list, "\n")

    list1 = original_male_word_list
    list2 = original_female_word_list
else:
    list1 = "like, yeah, know, oh, right, podcast, got, going, think, really, okay, people, fucking, na, want, good, shit, gon, gon na, mean, time, let, say, kind, make".split(", ")
    list2 = "life, know, things, really, people, feel, like, want, love, going, way, think, person, time, right, need, say, lot, feeling, let, kind, make, thing, world, energy".split(", ")

common_words_with_positions = find_common_words_with_positions(list1, list2)
print(common_words_with_positions)

for (word,(pos1,pos2)) in common_words_with_positions:
    if pos1 == pos2:
        list1.remove(word)
        list2.remove(word)
    elif pos1 < pos2:
        list2.remove(word)
    else:  # pos2 > pos1
        list1.remove(word)

male_word_list = list1[0:TOP_N]
female_word_list = list2[0:TOP_N]

print(len(male_word_list), ":", male_word_list)
print(len(female_word_list), ":", female_word_list)

if GENDER == "male":
    current_list = female_word_list
    other_list = male_word_list
    the_df = df_male
else:  # female
    current_list = male_word_list
    other_list = female_word_list
    the_df = df_female

random_substrings = []
for index, row in the_df.iterrows():
    random_substrings.extend(get_random_sentences(the_df.loc[index,"transcript"], num_samples=NUM_SAMPLES))

M_wins = 0
W_wins = 0
for random_substring in random_substrings:

    swaps = 0

    new_random_substring_list = []
    for word in random_substring.split(" "):
        if word_matches(word, other_list):
            swaps += 1

            random_word = np.random.choice(current_list)

            # check if word is capitalized
            if is_first_char_capitalized(word):
                random_word = random_word.capitalize()

            # check if punctuation before word
            if word[0] in string.punctuation:
                random_word = word[0] + random_word

            # check if punctuation after word
            if word[-1] in string.punctuation:
                random_word = random_word + word[-1]

            new_random_substring_list.append(random_word)
        else:
            new_random_substring_list.append(word)

    new_random_substring = " ".join(new_random_substring_list)

    inc_before_list = count_words_from_list_in_string(current_list, random_substring)
    inc_after_list = count_words_from_list_in_string(current_list, new_random_substring)
    dec_before_list = count_words_from_list_in_string(other_list, random_substring)
    dec_after_list = count_words_from_list_in_string(other_list, new_random_substring)

    if (inc_before_list != inc_after_list) and (dec_before_list != dec_after_list) and (swaps >= MIN_SWAPS):

        print("\nCURRENT LIST / INSERT =", current_list, "\n")
        print("OTHER LIST / REMOVE =", other_list, "\n")
        print("INC. BEFORE - Total count of words from list in string:", inc_before_list, "\n")
        print("INC. AFTER - Total count of words from list in string:", inc_after_list, "\n")
        print("DEC. BEFORE - Total count of words from list in string:", dec_before_list, "\n")
        print("DEC. AFTER - Total count of words from list in string:", dec_after_list, "\n")
        print("BEFORE - string:", random_substring, "\n")
        print("AFTER - string:", new_random_substring, "\n")

        old_m_list = []
        old_w_list = []
        new_m_list = []
        new_w_list = []
        for i in range(NUM_API):  # 5

            old_m, old_w = get_m_w_cosine_total(random_substring, MODEL)
            new_m, new_w = get_m_w_cosine_total(new_random_substring, MODEL)

            old_m_list.append(old_m)
            old_w_list.append(old_w)
            new_m_list.append(new_m)
            new_w_list.append(new_w)

        def get_avg(numbers):
            s = 0.0
            for num in numbers:
                s += num
            return s/float(len(numbers))

        old_m = get_avg(old_m_list)
        old_w = get_avg(old_w_list)
        new_m = get_avg(new_m_list)
        new_w = get_avg(new_w_list) 

        m_diff = new_m - old_m
        w_diff = new_w - old_w

        print("                M:               W:")
        print("old:", old_m, old_w)
        print("new:", new_m, new_w)

        print("m_diff:", m_diff)
        print("w_diff:", w_diff)

        if m_diff > 0 and w_diff > 0:
            if m_diff > w_diff:
                print("M wins") # got more M-ly
                M_wins += 1
            elif w_diff > m_diff: 
                print("W wins") # got more W-ly
                W_wins += 1
        elif m_diff < 0 and w_diff < 0: 
            if np.abs(m_diff) > np.abs(w_diff):
                print("W wins")
                W_wins += 1
            elif np.abs(w_diff) > np.abs(m_diff):
                print("M wins")
                M_wins += 1
        elif m_diff < w_diff: 
            print("W wins")
            W_wins += 1
        else:
            print("M wins")
            M_wins += 1

        print("Total M_wins=", M_wins)
        print("Total W_wins=", W_wins)
        print("=============================================================================")


# stdout back to stdout
sys.stdout = sys.__stdout__
file_handle.close()

print("Total M_wins=", M_wins)
print("Total W_wins=", W_wins)
