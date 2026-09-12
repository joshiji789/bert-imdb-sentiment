# Author: Anubhav Joshi

"""
main script to run the fine tune models.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level = logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from src.data import build_eval_dataloader, build_train_dataloader, load_tokenized_imdb
from src.evaluate import evaluate_model
from src.model import build_model
from src.train import save_checkpoint, train_model
from src.utils import get_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True, help = "Path to the YAML config")
    parser.add_argument("--learning_rate", type = float, default=None, help = "Override the config's learning_rate")
    parser.add_argument("--run_name", type = str, default = None, help = "Overrides the config's run_name")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["learning_rate"] = float(config["learning_rate"])

    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    if args.run_name is not None:
        config["run_name"] = args.run_name

    run_name = config.get("run_name") or f"{config['strategy']}-lr{config['learning_rate']}"

    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Run '{run_name}' ({config['strategy']}) on device: {device}")

    train_dataset, test_dataset, tokenizer = load_tokenized_imdb(
        config["model_name"],
        max_seq_length = config.get("max_seq_length", 256),
        train_subset_size = config.get("train_subset_size"),
        seed = config.get("seed", 42)
    )

    train_loader = build_train_dataloader(train_dataset, batch_size = config.get("batch_size", 16))
    eval_loader = build_eval_dataloader(test_dataset, batch_size = config.get("eval_batch_size", 32))

    """
    lora_r/lora_alpha/lora_droupout/lora_target_modules only matter for strategy = 'lora/qlora',
    build_model() simple never looks at them for any other strategy, so it's safe to always pass
    config.get(ket, <same default build_model alrady uses>) regardless of which strategy this config is for.
    """

    model = build_model(
        strategy = config["strategy"],
        model_name = config["model_name"],
        lora_r = config.get("lora_r", 8),
        lora_alpha = config.get("lora_alpha", 16),
        lora_dropout = config.get("lora_dropout", 0.1),
        lora_target_modules = config.get("lora_target_modules", ["query", "values"])
    )

    history = train_model(
        model, 
        train_loader, 
        device,
        learning_rate = config["learning_rate"],
        num_epochs = config.get("num_epochs", 10),
        weight_decay = config.get("weight_decay", 0.01),
        warmup_ratio = config.get("warmup_ratio", 0.1)
    )

    metrics = evaluate_model(model, eval_loader, device)

    checkpoint_dir = save_checkpoint(
        model, tokenizer, Path(config.get("output_dir", "result/checkpoints"))/run_name
    )

    results = {
        "run_name": run_name,
        "strategy": config["strategy"],
        "model_name": config["model_name"],
        "learning_rate": config["learning_rate"],
        "num_epochs": config["num_epochs", 10],
        "checkpoint_dir": str(checkpoint_dir),
        "train_history": history,
        **metrics
    }

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok = True)
    with open(metrics_dir / f"{run_name}.json", "w") as f:
        json.dump(results, f, indent = 2)
    
    print(json.dumps({k: v for k, v in results.items() if k!="train_history"}, indent = 2))

if __name__ == "__main__":
    main()