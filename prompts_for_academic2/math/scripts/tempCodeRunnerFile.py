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

from prompts import (
    step1_prompt,
    self_improvement_prompt,
    correction_prompt,
    verification_system_prompt,
    verification_remider
)
print(step1_prompt[:200]) 