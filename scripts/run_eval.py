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

    # peft's save_pretrained() writes this exact filename ONLY for LoRA/QLoRA
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")

    # this branch only run for LoRA/QLoRA checkpoints
    if os.path.exists(adapter_config_path):
        from peft import PeftConfig, PeftModel

        # adapter_config.json records which base model this adpater was trained on top
        # of so we know what to reload before attaching the adapter.
        peft_config = PeftConfig.from_pretrained(checkpoint_dir)

        # rebuilds a FRESH bert-base-uncased classifier - at this point
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path, num_labels = NUM_LABELS
        )
        # this is the step that actually applies the trained LoRA adapter, 
        # weights and the trained classifier head on top of that fresh model.
        return PeftModel.from_pretrained(base_model, checkpoint_dir)

    return AutoModelForSequenceClassification.from_pretrained(base_model, checkpoint_dir)

def main():
    # parse the command-line flags first
    args = parse_args()

    # If args.checkpoint is non-empty string use it; otherwise use args.pretrained
    tokenizer_soruce = args.checkpoint or args.pretrained

    _, test_dataset, _ = load_tokenizer_imdb(
        tokenizer_soruce, 
        max_seq_length = args.max_seq_length,
        load_train = False
    )

    #
    eval_loader = build_eval_dataloader(test_dataset, batch_size=args.eval_batch_size)

    if args.checkpoint:
        model = _load_model_for_eval(args.checkpoint) # reload a saved fine-tuned model

    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, num_labels = NUM_LABELS)

    device = get_device()
    metrics = evaluate_model(model, eval_loader, device)

    result = {
        "run_name": args.run_name,
        "model_source": args.checkpoint or args.pretrained, # records what was actually evaluated
        **metrics
    }

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents = True, exist_ok = True)

    with open(metrics_dir/f"{args.run_name}.json", "w") as f:
        json.dump(result, f, indent = 2)

    print(json.dump(result, indent = 2))


if __name__ == "__main__":
    main()