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

