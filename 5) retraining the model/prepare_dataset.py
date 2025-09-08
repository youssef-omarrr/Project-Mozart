import re
from datasets import load_dataset
from transformers import GPT2Tokenizer
import random

TRAIN_FILE = "../dataset/train_file.txt"
TEST_FILE = "../dataset/test_file.txt"

MUSIC_NOTES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b",
    "r",  # rest
    "<MASK>",
    "<|startofpiece|>", "<NAME=", "><BPM=", "><DURATION_BEATS=", "><DURATION_MINUTES=",
    "<TRACKS>", "Piano:", "Guitar:", "<TRACKSEP>"
]

def pre_tokenize_musical_text(text):
    """Pre-tokenize musical text to preserve musical tokens"""
    # Protect musical tokens by adding spaces around them
    # Musical note pattern: NoteOctave_Duration (G2_0.75, C4_q, Rest_s)
    text = re.sub(r'([A-G][#b]?[0-9]+_[a-z0-9/\.]+)', r' \1 ', text)
    # Rest pattern
    text = re.sub(r'(Rest_[a-z0-9/\.]+)', r' \1 ', text)
    # Instrument names with colons
    text = re.sub(r'([A-Za-z][A-Za-z0-9_#\- ]{0,40}?):', r' \1: ', text)
    return text

def enhance_training_data(dataset):
    """Pre-tokenize the dataset before GPT-2 tokenization"""
    def pre_tokenize_examples(examples):
        pre_tokenized_texts = []
        for text in examples["text"]:
            pre_tokenized_texts.append(pre_tokenize_musical_text(text))
        return {"text": pre_tokenized_texts}
    
    return dataset.map(
        pre_tokenize_examples,
        batched=True,
        desc="Pre-tokenizing musical data"
    )
    
def prepare_dataset(tokenizer, mask_probability=0.15):
    """Enhanced dataset preparation with pre-tokenization"""
    
    dataset = load_dataset("text", data_files={
        "train": TRAIN_FILE,
        "validation": TEST_FILE
    })

    # Pre-tokenize to preserve musical tokens
    dataset = enhance_training_data(dataset)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing pre-tokenized dataset"
    )

    return tokenized

def mask_music_tokens(text, mask_probability=0.15):
    parts = text.split("<MASK>")
    if len(parts) < 2:
        return text
        
    # Only mask the music sections after <MASK> tokens
    for i in range(1, len(parts)):
        tokens = parts[i].split()
        masked_tokens = []
        for token in tokens:
            if token in MUSIC_NOTES and random.random() < mask_probability:
                masked_tokens.append("<MASK>")
            else:
                masked_tokens.append(token)
        parts[i] = " ".join(masked_tokens)
        
    return "<MASK>".join(parts)