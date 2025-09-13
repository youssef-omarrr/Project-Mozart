from datasets import load_dataset
from filter import is_music_token 
import os

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


def build_features(examples, tokenizer, mask_token_id, max_length=1024):
    """
    Build tokenized features from parallel input/target texts.
    - input contains <MASK>
    - target contains full notes
    - labels are -100 except where <MASK> occurs
    """
    inputs = [pre_tokenize_musical_text(t) for t in examples["input"]]
    targets = [pre_tokenize_musical_text(t) for t in examples["target"]]

    enc_input = tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    enc_target = tokenizer(
        targets,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    input_ids = enc_input["input_ids"]
    attention_mask = enc_input["attention_mask"]
    target_ids = enc_target["input_ids"]

    labels = []
    mask_positions = []

    for inp, tgt in zip(input_ids, target_ids):
        lbl = [-100] * len(inp)
        mask_pos = [0] * len(inp)

        for i, tok in enumerate(inp):
            if tok == mask_token_id:
                lbl[i] = tgt[i]
                mask_pos[i] = 1

        labels.append(lbl)
        mask_positions.append(mask_pos)
        
        
        
    # Debug: Check mask positions
    total_masks = sum(sum(mp) for mp in mask_positions)
    print(f"🔍 Total mask positions created: {total_masks}")

    if total_masks == 0:
        print("❌ WARNING: No mask positions found! Check your <MASK> tokens.")
        # Show a sample
        print("Sample input:", inputs[0][:200])
        print("Sample target:", targets[0][:200])
        
        

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mask_positions": mask_positions,
    }

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

    # Load raw parallel datasets
    raw_inputs = load_dataset("text", data_files={"train": data_files["train_input"], "test": data_files["test_input"]})
    raw_targets = load_dataset("text", data_files={"train": data_files["train_target"], "test": data_files["test_target"]})

    # Align inputs + targets into one dataset
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = raw_inputs[split].add_column("target", raw_targets[split]["text"])
        dataset[split] = dataset[split].rename_column("text", "input")

    mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")

    # Apply feature building
    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(
            lambda ex: build_features(ex, tokenizer, mask_token_id),
            batched=True,
            remove_columns=["input", "target"],
            desc=f"Building {split} features",
            num_proc=os.cpu_count()
        )

    return dataset
