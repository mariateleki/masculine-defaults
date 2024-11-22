import os
import subprocess
import datetime
import sys

CURRENT_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

output_file = f"./experiment_runs/{os.path.basename(os.path.abspath(__file__))}-{CURRENT_DATETIME}.txt"
file_handle = open(output_file, "w+")
sys.stdout = file_handle

def run_script(embedding_model, topic_model, lemmatize):
    l = ["python", "topic-script.py", 
         "--DATETIME_STR", str(CURRENT_DATETIME),
         "--EMBEDDING_MODEL", str(embedding_model), 
         "--TOPIC_MODEL", str(topic_model), 
         "--LEMMATIZE", str(lemmatize)
        ]
    print(" ".join(l[2:]))

    result = subprocess.run(l, capture_output=True, text=True)

    # Check if the subprocess ran successfully
    if result.returncode == 0:
        pass
    else:
        print("Script execution failed.")

    # Print the output and error messages (if any)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
                               
for topic_model in ["LDA"]:
    for embedding_model in ["Count"]:
        for lemmatize in [False, True]:
            run_script(embedding_model, topic_model, lemmatize)
    
lemmatize = False
for topic_model in ["BERTopic"]:
    for embedding_model in ["Bert","ChatGPT","Llama"]: 
        run_script(embedding_model, topic_model, lemmatize)
                            
# stdout back to stdout
sys.stdout = sys.__stdout__
file_handle.close()
