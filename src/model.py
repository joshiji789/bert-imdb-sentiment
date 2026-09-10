# Author: Anubhav Joshi
"""
Building a BERT model configured for one of serveral fine-tuning methods.
Every strategy starts from the SAME pretrained Checkpoint - Only which parameters
are allowed to change during traing differs.

Strategy    |   Frozen   |  Trainable
----------------------------------------

pretrained  |  Everything (frozed) | nothing- zero-shot baseline

zero-shot baseline head only | embeddings + all 12 trainsformer layers (frozen) + pooler | only the final classification head 

frozen transformer | embeddings + all 12 transformer layer (frozen) | pooler layer + classification layer

full-fine-tuning | nothing | every parameter in the model

LoRA | the original BERT weights | small LoRA adapter matrics + classification 

QLoRA | same as LoRA, base weights loaded in 4 bit (CUDA only)
"""
import logging
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification
from src.utils import NUM_LABELS
logger = logging.getLogger(__name__)

STRATEGIES = ("pretrained", "head-only", "frozen-transformer", "full-finetune", "lora", "qlora")

# freeze all except function
def _freeze_all_except(model, trainable_prefixes):
    """
    Set requires_grad = False on every parameter, except ones whose name starts with one of the
    given prefixes (e.g. 'classifier', 'bert.pooler)
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)
    return model

# Build LoRA model
def _build_lora_model(strategy, model_name, lora_r, lora_alpha, lora_dropout, lora_target_modules):
    """
    Build a LoRA model based on the given strategy and parameters
    """
    # Using Prameter-Efficient Fine-Tuning (peft) HuggingFace library to build LoRA model
    from peft import LoraConfig, TaskType, get_peft_model
    quantization_kwarg = {}

    if strategy == "qlora":
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization_kwarg["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit = True, 
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.bfloat16
            )
        else:
            # bitsandbytes 4 bit kernels and CUDA-only.
            # On CPU, or MAC we can't do 4bit quantization, so we run just the standard LoRA model instead.
            logger.warning("QLoRA is only supported on CUDA. QLoRA needs bitsandbytes 4bit kernels.")

    # Build the base model (pretrained BERT) with or without quantization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels = NUM_LABELS, attn_implementation = "eager", **quantization_kwarg
        )

    lora_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        r = lora_r,                         # rank of the LoRA adapter matrics - higher = more capacity/params
        lora_alpha = lora_alpha,            # scaling factor applied to the LoRA update
        lora_droupout = lora_dropout,       # dropout inside the adapter, for regularization
        target_modules = lora_target_modules, # Which layer get adapter (attention query/value projections)
        modules_to_save = ["classifier"]      # which modules to save when saving the model
    )

    return get_peft_model(base_model, lora_config)

def build_model():
    return None