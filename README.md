# bert-imdb-sentiment
## GOAL
Compare 6 things on the exact same untouched IMDB test set (accuracy, precision, recall, F1, confusion matrix)
1. **Pretrained BERT** — zero-shot, no training at all. Raw `bert-base-uncased` with a randomly initialized classifier head; expected to land close to chance (~50%), since the head has never learned anything.
2. **Head-only fine-tune** — BERT itself (embeddings + all 12 transformer layers + pooler) fully frozen; only the final linear classifier layer is trained. The purest "linear probe on frozen BERT features" baseline.
3. **Frozen** — same as head-only, but the pooler layer (the small dense+tanh layer that turns BERT's `[CLS]` token into a fixed-size vector) is also unfrozen and trained alongside the classifier.
4. **Full fine-tuning** — every parameter in the model is trainable (embeddings, all 12 transformer layers, pooler, classifier).
5. **LoRA fine-tuning** — BERT's original weights stay frozen; small low-rank adapter matrices are injected into the attention Query/Value projections and trained, along with the classifier head.
6. **QLoRA fine-tuning** — same as LoRA, but the frozen base weights are loaded in 4-bit precision. Requires a CUDA GPU (bitsandbytes); falls back to standard LoRA on this machine (Apple Silicon / MPS, no CUDA).

Plus we'll sweep a couple of different learning rates per method (methods 2–6 — there's nothing to tune for the pretrained baseline).

The Results below shows the after fine-tuning: \
![Training Loss vs. Epoch](results\figures\loss_vs_epoch.png)

Here's the proposed project structure:

```
 bert-imdb-sentiment/
 ├── README.md
 ├── requirements.txt
 ├── configs/
 │ ├── frozen.yaml
 │ ├── full_finetune.yaml
 │ ├── lora.yaml
 │ └── qlora.yaml
 ├── src/
 │ ├── data.py          # load IMDB, tokenize, build DataLoaders (shared, untouched test set)
 │ ├── models.py        # build model per strategy: frozen head / full FT / LoRA / QLoRA
 │ ├── train.py         # training loop, takes a config, saves checkpoint + metrics
 │ ├── evaluate.py      # runs a saved model (or the raw pretrained model) on the test set ->
 accuracy, F1, confusion matrix
 │ └── utils.py         # seeding, device selection (mps/cuda/cpu), logging helpers
 ├── scripts/
 │ ├── run_train.py         # CLI: python scripts/run_train.py --config configs/lora.yaml
 │ └── run_eval.py          # CLI: python scripts/run_eval.py --checkpoint
 ├── notebooks/
 │ ├── 01_download_and_explore.ipynb    #
 │ └── 02_compare_results.ipynb          # loads all saved metrics, builds final comparison
 table/plots
 ├── results/
 │ ├── metrics/                     # one JSON per run: {strategy, lr, accuracy, f1, confusion_matrix,
 ...}
 │ └── checkpoints/                 # saved fine-tuned weights (gitignored — large files)
 └── tests/
 └── test_data.py                   # sanity checks on tokenization/splits
 ```

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126    
```

### Commands to run the code
- Command to run the head only fine-tuning with learning rate 10^(-3) and 10^(-4)
python scripts/run_train.py --config configs/head-only.yaml --learning_rate 1e-3 --run_name head-only_lr1e-3
python scripts/run_train.py --config configs/head-only.yaml --learning_rate 1e-4 --run_name head-only_lr1e-4


- Command to run the frozen-transformer fine-tuning with learning rate 10^(-3) and 10^(-4)
python scripts/run_train.py --config configs/frozen-transformer.yaml --learning_rate 1e-3 --run_name frozen-transformer_lr1e-3
python scripts/run_train.py --config configs/frozen-transformer.yaml --learning_rate 1e-4 --run_name frozen-transformer_lr1e-4

- Commands to run the full-finetune where learning rate is 10^(-3) and 10^(-4) (But these values are two high for the full-fine tuning), so we have also used the \
values of O(10^(-5)).
python scripts/run_train.py --config configs/full-finetune.yaml --learning_rate 1e-3 --run_name full-finetune_lr1e-3
python scripts/run_train.py --config configs/full-finetune.yaml --learning_rate 1e-4 --run_name full-finetune_lr1e-4
python scripts/run_train.py --config configs/full-finetune.yaml --learning_rate 2e-5 --run_name full-finetune_lr2e-5
python scripts/run_train.py --config configs/full-finetune.yaml --learning_rate 5e-5 --run_name full-finetune_lr5e-5

- Commands to run the LoRA and QLoRA with the learning rate of 2e-4 and 5e-5.
python scripts/run_train.py --config configs/lora.yaml --learning_rate 2e-4 --run_name lora_lr2e-4
python scripts/run_train.py --config configs/lora.yaml --learning_rate 5e-5 --run_name lora_lr5e-5