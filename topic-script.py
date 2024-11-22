# used BERTopic - Best Practices notebook: https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing#scrollTo=xyIFi06Vzeg3

# imports and setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import re
import string
import sys
import subprocess
import datetime
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import cosine_similarity

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

import spacy
spacy_model = spacy.load('en_core_web_sm')


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from hdbscan import HDBSCAN
from umap import UMAP

from transformers.pipelines import pipeline

import openai

# custom imports
import utils_general
import utils_embeddings

def save_top_words(model, feature_names, n_top_words, filename):
    with open(filename, "w") as f:
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]  # Get indices of top n words
            top_words = [feature_names[i] for i in top_words_idx]
            f.write(f"{topic_idx+1},")
            for word in top_words:
                f.write(word+",")
            f.write("\n")

# set up parameters
parser = argparse.ArgumentParser()
parser.add_argument("--DATETIME_STR", type=str)
parser.add_argument("--EMBEDDING_MODEL", type=str, choices=["Count", "Bert", "ChatGPT", "Llama"])
parser.add_argument("--TOPIC_MODEL", type=str, choices=["LDA", "BERTopic"])
parser.add_argument("--LEMMATIZE", type=str, default="False")
parser.add_argument("--SEED", type=int, default=42)

args = parser.parse_args()

DATETIME_STR = args.DATETIME_STR
EMBEDDING_MODEL = args.EMBEDDING_MODEL
TOPIC_MODEL = args.TOPIC_MODEL
if args.LEMMATIZE == "True":
    LEMMATIZE = True
else:
    LEMMATIZE = False
SEED = args.SEED

# set up filenames
# datetimestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
datetimestamp = DATETIME_STR
output_filename = os.path.join(".","csv",f"{EMBEDDING_MODEL}-{TOPIC_MODEL}-{LEMMATIZE}-{datetimestamp}.csv")
lda_filename = os.path.join(".", "csv", f"{EMBEDDING_MODEL}-{TOPIC_MODEL}-{LEMMATIZE}-{datetimestamp}-LDA_TOP_WORDS.csv")
bertopic_filename = os.path.join(".","csv",f"{EMBEDDING_MODEL}-{TOPIC_MODEL}-{LEMMATIZE}-{datetimestamp}-BERTOPIC_TOP_WORDS.csv")

# set the np seed
np.random.seed(SEED)

# set up docs list
pd.set_option('display.max_columns', None)
df = pd.read_csv("./csv/df-all-features.csv")
df = df.drop_duplicates(subset="show_uri", keep="first")
df = df.drop(columns=df.columns[df.columns.str.contains("Topic")])  # drop old topic distributions, redoing in this file
df = df.sample(n=10000, random_state=SEED)
df["transcript"] = df["transcript"].fillna("")
docs = list(df["transcript"])
print(docs[0])

# remove nltk stopwords from docs
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenized_documents = [word_tokenize(doc.lower()) for doc in docs]
docs = [
    [word for word in doc if word.isalnum() and word not in stop_words]
    for doc in tokenized_documents
]
docs = [" ".join(doc) for doc in docs]

if LEMMATIZE:
    def lemmatize_text(text):
        doc = spacy_model(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct])

    docs = [lemmatize_text(doc) for doc in docs]
    print("LEMMATIZED DOC:", docs[0])
    
# get embeddings for docs
embeddings = None
embedding_model = None
vectorizer = None
if EMBEDDING_MODEL == "Count":
    vectorizer = CountVectorizer()
    embeddings = vectorizer.fit_transform(docs)
elif EMBEDDING_MODEL == "Bert":
    pass  # uses SBERT model built into BERTopic
elif EMBEDDING_MODEL == "ChatGPT":
    embeddings = utils_embeddings.get_chatgpt_embeddings(docs)
elif EMBEDDING_MODEL == "Llama":
    embeddings = utils_embeddings.get_llama_embeddings(docs)
    
    
print(embeddings)

if TOPIC_MODEL == "LDA":
    
    # train LDA model
    n_topics = 100
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        random_state=0, 
        max_iter=5,
        evaluate_every=1,
        verbose=1
        # we use the default value for perplexity tolerance
    )
    lda.fit(embeddings)
    
    # save the topic words
    count_feature_names = vectorizer.get_feature_names_out()
    save_top_words(lda, count_feature_names, 25, lda_filename)

    # Transform the fitted LDA model to get document-topic distribution
    document_topic_probs = lda.transform(embeddings)
    print(document_topic_probs[0])

    # Create a new DataFrame with the original data and add the topic probabilities
    prob_df = pd.DataFrame(document_topic_probs, columns=[f'Topic_{i+1}' for i in range(n_topics)])
    df = pd.concat([df.reset_index(), prob_df], axis=1)
    columns_to_drop = [col for col in df.columns if "Unnamed" in col]  # drop columns containing 'Unnamed' in their name because these are old indexes
    df = df.drop(columns=columns_to_drop)

    # save to file
    df.to_csv(output_filename, header=True)
    
else:  # BERTopic
    # From BERTopic Best Practices: https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing#scrollTo=CmfKtFIcvkrx
    # set up BERTopic based on the EMBEDDING_MODEL: https://colab.research.google.com/drive/18arPPe50szvcCp_Y6xS56H2tY0m-RLqv?usp=sharing
    umap_params = {
        "n_neighbors": 15,  # 15
        "n_components": 5, # 5
        "min_dist": 0.0, 
        "metric": 'cosine',
        "random_state": 42}
    print("umap_params:", umap_params)
    umap_model = UMAP(n_neighbors=umap_params["n_neighbors"],
                      n_components=umap_params["n_components"], 
                      min_dist=umap_params["min_dist"], 
                      metric=umap_params["metric"], 
                      random_state=umap_params["random_state"])
    
    hdbscan_params = {
        "min_cluster_size": 2, # 15 # previously 2 for the 2024-08-31_08-34-07 runs
        "metric": 'euclidean',
        "cluster_selection_method": 'eom',
        "prediction_data": True}
    print("hdbscan_params:", hdbscan_params)
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_params["min_cluster_size"], 
                            metric=hdbscan_params["metric"], 
                            cluster_selection_method=hdbscan_params["cluster_selection_method"], 
                            prediction_data=hdbscan_params["prediction_data"])
    
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    # Representation models
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech("en_core_web_sm")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    prompt = """
    I have a topic that is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. In the case of no topic, note the style. For example: conversational, formal, scripted, etc. Make sure it is in the following format:
    topic: <topic label>
    """  # I modified this prompt
    client = openai.OpenAI(api_key="")
    openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)
    representation_model = {
        "KeyBERT": keybert_model,
        # "OpenAI": openai_model,
        "MMR": mmr_model,
        "POS": pos_model
    }

    if TOPIC_MODEL != "Bert":
        topic_model = BERTopic(  # pipeline models and hyperparameters
          umap_model=umap_model,
          hdbscan_model=hdbscan_model,
          vectorizer_model=vectorizer_model,
          representation_model=representation_model,
          top_n_words=25,
          verbose=True,
          calculate_probabilities=True)
    else:
        topic_model = BERTopic(  # pipeline models and hyperparameters
          embedding_model=embedding_model,
          umap_model=umap_model,
          hdbscan_model=hdbscan_model,
          vectorizer_model=vectorizer_model,
          representation_model=representation_model,
          top_n_words=25,
          verbose=True,
          calculate_probabilities=True)

    # train model
    topics, probs = topic_model.fit_transform(docs)
    
    print("Saving umap embeds...")
    umap_embeds = pd.DataFrame(umap_model.embedding_)
    print(umap_embeds)
    umap_df = pd.DataFrame(umap_embeds)
    print(umap_df.head())
    umap_df.to_csv(f"./csv/umap_embeddings_{EMBEDDING_MODEL}.csv", index=False)

    # show topics
    bertopic_df = topic_model.get_topic_info()  # TODO: Might need to save this too.
    bertopic_df.to_csv(bertopic_filename, header=True)

    # save probs into pd df
    prob_df = pd.DataFrame(np.array(probs), columns=[f"Topic_{i}" for i in range(0, np.array(probs).shape[1])])  # this line will fail for small sample sizes (e.g. 100ish podcasts) because they all get put into the same topic
    prob_df.insert(0, "Document_Num", range(len(df)))
    
    # Create a new DataFrame with the original data and add the topic probabilities
    df = pd.concat([df.reset_index(), prob_df], axis=1)
    columns_to_drop = [col for col in df.columns if "Unnamed" in col]  # drop columns containing 'Unnamed' in their name because these are old indexes
    df = df.drop(columns=columns_to_drop)
    
    # save to file
    df.to_csv(output_filename, header=True)