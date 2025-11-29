import torch
import numpy as np
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import time
import warnings
import pandas as pd
import polars as pl
from vllm import LLM, SamplingParams
import sys
import json
import argparse
import logging
from textwrap import indent

from prompts import (
    step1_prompt,
    self_improvement_prompt,
    correction_prompt,
    verification_system_prompt,
    verification_remider
)
from logs_functions import (
    log_print, 
    set_log_file, 
    close_log_file, 
    read_file_content,
    extract_detailed_solution,
    verify_solution,
    check_if_solution_claimed_complete,
    init_explorations,
    agent
)

pd.set_option('display.max_colwidth', None)
warnings.simplefilter('ignore')

start_time = time.time()
cutoff_time = start_time + (4 * 60 + 53) * 60

llm_model_pth = '/mnt/d/Qwen3-1.7B'

MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 4096

llm = LLM(
    llm_model_pth,
    dtype="bfloat16",                # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=False,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=1,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.9, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=391,
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(
            temperature=0.2,              # randomness of the sampling
            min_p=0.05,
            top_p=0.9,
            skip_special_tokens=True,     # Whether to skip special tokens in the output
            max_tokens=32768 ,
            # stop=["</think>"]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent (SDK Version)')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt', 
                       help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Comma-separated other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    
    args = parser.parse_args()

    max_runs = args.max_runs
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)

    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    
    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i+1} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts)
            if sol is not None:
                print(f">>>>>>> Found a correct solution in run {i+1}.")
                # 最终解决方案会以清晰的格式打印在 agent 函数的末尾
                # print(sol) # 可以取消注释以再次打印
                break
        except Exception as e:
            print(f">>>>>>> Error in run {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    close_log_file()