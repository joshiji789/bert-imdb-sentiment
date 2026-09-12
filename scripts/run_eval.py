# Author: Anubhav Joshi
"""
Evaluate the model on the untouched IMDB test set - either the raw pretrained model (zero-shot baseline)
or a saved checkpoint from run_train.py
"""

import argparse
import json 
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hugging class that can load a BERT classification model, either from the HUB
# or from the local folder.
from transformers import AutoModelForSequenceClassification
from src.data import build_eval_dataloader, load_tokenizer_imdb
from src.evaluate import evaluate_model
from src.utils import NUM_LABELS, get_device

def parse_args():
    """
    Reads the command-line flags the user typed and turns them into the Python object.
    """
    parser = argparse.ArgumentParser()

    # --pretrained and --checkpoint are mutually exclusive (never both, never neither)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrianed", help = "HF model name to evalute zero-shot, bert-base-uncased")
    group.add_argument("--checkpoint", help = "Path to checkpoint directory saved by run_train.py")


    # name usef for the output .json filename
    parser.add_argument("--run_name", required=True)

    # Must match what the model was trained/tokenized with
    parser.add_argument("--max_seq_length", type = int, default= 256)

    parser.add_argument("--eval_batch_size", type = int, default = 32)
    return parser.parse_args()


def _load_model_for_eval(checkpoint_dir):
    """
    LoRA/QLoRA checkpoints only save adapter weights (not the full base model),
    so they need to be reloaded differently from a full-weight checkpoint
    (head-only/frozen-transformer/full-finetune, which save every weight).
    """
    