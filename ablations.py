import os
import subprocess
import datetime
import sys

output_file = f"./experiment_runs/{os.path.basename(os.path.abspath(__file__))}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
file_handle = open(output_file, "w+")
sys.stdout = file_handle

for n_samples in [100]:
    for top_n in [10]:
        for gender in ["male", "female"]:
            for num_samples in [3]:
                for min_swaps in [1,2,3,4,5,6,7,8,9,10]:
                    for num_seconds in [20.0, 25.0, 30.0]: 
                        for seed in range(0,2):
                            for num_api in [1]:
                                
                                model = "llama"
                                BERTOPICS = 0
                                
                                l = ["python", "run_experiment.py", 
                                                         "--N_SAMPLES", str(n_samples), 
                                                         "--TOP_N", str(top_n), 
                                                         "--GENDER", str(gender), 
                                                         "--NUM_SAMPLES", str(num_samples), 
                                                         "--MIN_SWAPS", str(min_swaps), 
                                                         "--NUM_SECONDS", str(num_seconds), 
                                                         "--SEED", str(seed),
                                                         "--NUM_API", str(num_api),
                                                         "--MODEL", str(model),
                                                         "--BERTOPICS", str(BERTOPICS)
                                                        ]

                                print(" ".join(l[2:]))

                                result = subprocess.run(l, capture_output=True, text=True)

                                # Check if the subprocess ran successfully
                                if result.returncode == 0:
                                    pass
                                else:
                                    print("Script execution failed.")

                                # Print the output and error messages
                                print(result.stdout)

                                print("Error:")
                                print(result.stderr)
                            
# stdout back to stdout
sys.stdout = sys.__stdout__
file_handle.close()
