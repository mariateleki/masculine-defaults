# utils_embeddings.py
# some from https://github.com/ZBWpro/PretCoTandKE/blob/main/evaluation.py
import sys
import time
import fcntl

import torch
from torch.utils.data import DataLoader

from openai import OpenAI

from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import tiktoken

from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer

def init_count_embeddings(docs):
    # must pass the entire corpus
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    print(X.shape)
    return vectorizer, X

def get_count_embeddings(doc, vectorizer, X):
    doc_embedding = torch.from_numpy(vectorizer.transform([doc]).toarray()).float()
    print(doc_embedding.shape)
    return doc_embedding
    
def get_chatgpt_embeddings(docs):
    # set model and client up
    model = "text-embedding-3-large"
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    
    # get embeddings for docs
    embeddings = []
     # print("Getting ChatGPT embeddings...")
    for doc in tqdm(docs):
        # truncate the input to max number of tokens
        enc = tiktoken.encoding_for_model(model)
        doc = enc.decode(enc.encode(doc)[:8191])
        
        # get embedding from API
        embeddings.append(torch.tensor(client.embeddings.create(input=[doc], model=model).data[0].embedding))
    
    return embeddings

def get_llama_embeddings(docs):

    # from https://github.com/ZBWpro/PretCoTandKE/blob/main/evaluation.py
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # How to use llama 3.1: https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/modelUpgradeExample.py
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token="YOUR_TOKEN_HERE")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token="YOUR_TOKEN_HERE",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    all_messages = []
    for d in docs:
        all_messages.append([{"role": "user", "content": f"This sentence: \" {d} \" means in one word: \""}])

    embeddings = []
    # print("Getting Llama embeddings...")
    for message in tqdm(all_messages):
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # print("\nINPUT:")
        decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=False) for seq in input_ids]
        # for seq in decoded_sequences:
        #     print(seq)

        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            attention_mask=attention_mask,
            num_return_sequences=1, #
            return_dict_in_generate=True, # 
            output_hidden_states=True, # 
        )

        # why hidden_states[0]: https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput
        # then [-1][:, -1, :] is from: https://github.com/ZBWpro/PretCoTandKE/blob/main/evaluation.py
        embeddings.append(outputs["hidden_states"][0][-1][:, -1, :].cpu().float())
       
    return embeddings

