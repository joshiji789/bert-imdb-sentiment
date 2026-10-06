# Author: Anubhav Joshi
"""
Building a BERT sequence-classification model with fine tuning strategies
for the "prompt family" of methods. 

Strategy        |      Where the soft prompt lives     |    Reparameterized (update raw embedding or MLP weights)
-------------------------------------------------------------------------------------------------------------
prompt-tuning   | prepended to the input embeddings only (layer 0) | no - raw embeddings trained directly
prefix-tuning   | prepended to the K/V of EVERY transformer layer  | yes - 2 layer MLP
p-tuning        | prepended to the input embeddings only (layer 0) | yes - LSTM/MLP "prompt encoder"
p-tuning-v2     | prepended to K/V of EVERY transformer layer      | no - per layer prefix trained directly

prompt-tuning (Lester et al. 2021) and p-tuning (Liu et al 2021) are both shallow: The soft (added) tokens
only touch the first embedding layer, so every later layer sees them just like normal input embeddings.
p-tuning twwist over prompt-tuning is that the osft embeddings are produced by a small MLP (prompt encoder)
instead of trained directly.

prefix tuning (Li and Liang 2021) and p-tuning-v2 (Liu et al. 2022) are both deep: The soft token is injected
into the attention K/V of every transfomer layer. prefix tuning reparameterized the prefix through an MLP,
while p-tuning-v2 removes that MLP and optimizes the per-layer prefix vectors directly.
"""

import logging
from transformers import AutoModelForSequenceClassification
from src.utils import NUM_LABELS

logger = logging.getLogger(__name__)

PROMPT_FAMILY_STRATEGIES = ["prompt-tuning", "prefix-tuning", 'p-tuning', "p-tuning-v2"]

def _build_prompt_tuning_config(model_name, nnum_virtual_tokens, prompt_tuning_init, 
                                prompt_tuning_init_text):
    """
    prompt-tuning: num_virtual_tokens trainable "soft prompt" embeddings
    are prepended to the input embeddings.
    """