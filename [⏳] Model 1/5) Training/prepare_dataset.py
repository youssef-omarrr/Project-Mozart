from datasets import load_dataset
from filter import is_music_token 

# Paths to your processed train/test files
TRAIN_INPUTS = "D:/Codess & Projects/Project Mozart/dataset/train_inputs.txt"
TRAIN_TARGETS = "D:/Codess & Projects/Project Mozart/dataset/train_targets.txt"
TEST_INPUTS = "D:/Codess & Projects/Project Mozart/dataset/test_inputs.txt"
TEST_TARGETS = "D:/Codess & Projects/Project Mozart/dataset/test_targets.txt"


def pre_tokenize_musical_text(text: str) -> str:
    """
    Pre-tokenize musical text using is_music_token() and preserve special tokens.
    Only keeps valid musical notes or special tokens (like <TRACKS>, <MASK>, etc.).
    """
    tokens = text.split()
    filtered = [tok for tok in tokens if is_music_token(tok) or tok.startswith("<")]
    return " ".join(filtered)


def enhance_dataset(dataset):
    """
    Apply pre-tokenization to the entire dataset.
    """
    def pre_tokenize_examples(examples):
        return {"text": [pre_tokenize_musical_text(t) for t in examples["text"]]}
    
    return dataset.map(
        pre_tokenize_examples,
        batched=True,
        desc="Pre-tokenizing musical data"
    )


def prepare_dataset(tokenizer):
    """
    Prepare tokenized dataset ready for training or evaluation.
    Uses train_inputs/train_targets and test_inputs/test_targets.
    Returns a dictionary with train and test tokenized datasets.
    """
    
    data_files = {
        "train_input": TRAIN_INPUTS,
        "train_target": TRAIN_TARGETS,  # These should be PARALLEL files
        "test_input": TEST_INPUTS,
        "test_target": TEST_TARGETS     # line 1 of targets should correspond to line 1 of inputs
    }

    # Load raw text datasets
    dataset = load_dataset("text", data_files=data_files)
    
    # CRITICAL: Verify input-target alignment
    print(f"Train inputs: {len(dataset['train_input'])}")
    print(f"Train targets: {len(dataset['train_target'])}")
    print(f"Test inputs: {len(dataset['test_input'])}")
    print(f"Test targets: {len(dataset['test_target'])}")
    
    # Sample check - first few lines should correspond
    print("\nSample input:", dataset['train_input'][0]['text'][:100])
    print("Sample target:", dataset['train_target'][0]['text'][:100])
    
    # Tokenize using tokenizer
    def tokenize_input_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )
    
    def tokenize_target_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True,
            max_length=1024,
            padding=False
        )
    
    tokenized = {}
    tokenized["train_input"] = dataset["train_input"].map(
        tokenize_input_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train inputs"
    )
    
    tokenized["train_target"] = dataset["train_target"].map(
        tokenize_target_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train targets"
    )
    
    # Repeat for test
    tokenized["test_input"] = dataset["test_input"].map(
        tokenize_input_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing test inputs"
    )
    
    tokenized["test_target"] = dataset["test_target"].map(
        tokenize_target_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing test targets"
    )
    
    return tokenized
