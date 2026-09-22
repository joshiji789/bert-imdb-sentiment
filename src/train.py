# Author: Anubhav Joshi
"""
Trains a model for a fixed number of epochs, and returns a per-epoch loss history.
Also provides save_checkpoint(), used to persist a trained model + its tokenizer to disk.
"""

import logging
import time
from pathlib import Path

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.utils import is_quantized

logger = logging.getLogger(__name__)

def train_model(model, train_loader, device, learning_rate, num_epochs,
                weight_decay = 0.01, warmup_ratio = 0.1):
    """
    Why AdamW: different weights often need different learning rate.
    - Take largest step when gradients are small
    - Take smaller steps when gradients are noisy

    model:           Name of model to use
    train_loader:   load the dataset for training
    device:         CUDA, CPU, or MAC
    learning_rate:  how big a step during each update
    wewight_decay:  prevent weights from becoming excessing large (reduces overfitting, keep parameters small)
    warmup ratio:   gradually increase the learning rate at the start of training to stabilize optimzation and
                    preserve pretrained knowledge.
    """
    if not is_quantized(model):
        model.to(device)
    model.train()

    # Only pass parameters that actually required gradients to the optimizer.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader)*num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps     = int(total_steps*warmup_ratio),
                                                num_training_steps  = total_steps)

    history = []
    log_every = 1 # Constant for printing log for every 1 iterations
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0 # accumulates over the WHOLE epoch, for the end of epoch summary


        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Passing "labels" makes the model compute and return the loss itself
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += outputs.loss.item()

        avg_loss = running_loss / len(train_loader)

        elapsed = time.time() - start
        if epoch%log_every == 0:
            logger.info("epoch %d/%d finished - avg loss %.5f-%.1fs", 
                        epoch+1, num_epochs, avg_loss, elapsed)

        history.append({"epoch": epoch+1,
                        "loss": avg_loss,
                        "seconds": elapsed})

    return history


def save_checkpoint(model, tokenizer, output_dir):
    """
    Saves the model's weights + config and the tokenizer's files to utput_dir,
    so the exact same model can be reloaded later for evaluation without retraining.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir