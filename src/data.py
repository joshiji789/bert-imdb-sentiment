# Author: Anubhav Joshi
import re

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.utils import IMDB_DATASET_NAME

TENSOR_COUMNS = ["input_ids", "attention_mask", "label"]

def _clean_text(text):
    """
    Function to clean the text by removing unwanted characters and formatting.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip() # Remove leading and trailing spaces

def _drop_duplicates_texts(dataset):
    """
    Function to drop duplicate text from the dataset
    """
    seen = set()
    keep_indicies = []
    for i, text in enumerate(dataset["text"]):
        if text not in seen:
            seen.add(text)
            keep_indicies.append(i)
    return dataset.select(keep_indicies)

## Load Tokenized IMDB Dataset
def load_tokenized_imdb(model_name, max_seq_length = 256, 
                        train_subset_size = None, seed = 42, load_train = True):
    """
    Load the IMDB dataset, and tokenize it using the specified model's tokenizer.
    The function also allows for optional subsetting of the training data and setting
    a random seed for reproducibility.
    """

    dataset = load_dataset(IMDB_DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        cleaned = [_clean_text(text) for text in batch["text"]]
        return tokenizer(cleaned, truncation = True, 
                         padding = "max_length", max_length = max_seq_length)

    # Test Split: no dedup, no shuffle, no subset
    # Same order in every single run so every method is compared on the same ground.
    test_dataset = dataset["test"].map(preprocess, batched = True)
    test_dataset.set_format(type = "torch", columns=TENSOR_COUMNS)

    # train split: dedup, shuffle, subsetting
    train_dataset = None
    if load_train:
        train_split = _drop_duplicates_texts(dataset["train"])
        train_split = train_split.shuffle(seed = seed)
        if train_subset_size is not None:
            train_split = train_split.select(range(train_subset_size))
        train_dataset = train_split.map(preprocess, batched=True)
        train_dataset.set_format(type="torch", columns=TENSOR_COUMNS)

    return train_dataset, test_dataset, tokenizer


## Build the train dataset and test dataset loaders
def build_train_dataloader(train_dataset, batch_size =16):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def build_eval_dataloader(test_dataset, batch_size = 16):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)