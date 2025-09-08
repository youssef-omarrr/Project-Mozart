from datasets import load_dataset
from filter import is_music_token 

# Paths to your processed train/test files
TRAIN_INPUTS = "../../dataset/train_inputs.txt"
TRAIN_TARGETS = "../../dataset/train_targets.txt"
TEST_INPUTS = "../../dataset/test_inputs.txt"
TEST_TARGETS = "../../dataset/test_targets.txt"


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
        "train_target": TRAIN_TARGETS,
        "test_input": TEST_INPUTS,
        "test_target": TEST_TARGETS
    }

    # Load raw text datasets
    dataset = load_dataset("text", data_files=data_files)
    
    # Only pre-tokenize if needed - your files might already be properly formatted
    # Comment out this section if your input/target files are already clean
    """
    # Pre-tokenize musical tokens (optional - only if files contain non-music tokens)
    for split in dataset.keys():
        dataset[split] = enhance_dataset(dataset[split])
    """
    
    # Tokenize using tokenizer
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )
    
    tokenized = {}
    for split in dataset.keys():
        tokenized[split] = dataset[split].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split} data"
        )
    
    return tokenized
