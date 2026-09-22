# Author: Anubhav Joshi

"""
Evaluates the model:
- accuracy: fraction of predictions that exactly matches the true label.
- precision: of everything which we have predicted "positive" (label 1), what fraction of it is actually positive. (TP/(TP+FP))
- recall: of eveything that actuall is "predicted", what fraction did we correctly catch. (TP/(TP+FN))
- f1: harmonic mean of the precision and recall - one number balancing both
- confusion matrix: full breakdown of correct/incorrect predictions per class
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.utils import is_quantized

# disables gradient tracking - on backward pass happends during evaluation
@torch.no_grad()
def evaluate_model(model, eval_loader, device):
    if not is_quantized(model):
        model.to(device)
    model.eval()    # turn off dropout, so predictions are deterministic

    all_preds = []
    all_labels = []

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids = input_ids, attention_mask = attention_mask)

        # argmax: the predicted class (0 or 1)
        preds = outputs.logits.argmax(dim=-1).cpu() # sklearn can't read CUDA/MPS tensors

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # return the matrics
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
    }