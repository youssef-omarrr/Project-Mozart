from datasets import load_dataset
from filter import is_music_token 

# Paths to your processed train/test files
TRAIN_INPUTS = "D:/Codess & Projects/Project Mozart/dataset/train_inputs.txt"
TRAIN_TARGETS = "D:/Codess & Projects/Project Mozart/dataset/train_targets.txt"
TEST_INPUTS = "D:/Codess & Projects/Project Mozart/dataset/test_inputs.txt"
TEST_TARGETS = "D:/Codess & Projects/Project Mozart/dataset/test_targets.txt"


def pre_tokenize_musical_text(text: str) -> str:
    """
    FIXED: More careful pre-tokenization that preserves ALL special tokens and music tokens.
    The issue was that only tokens starting with '<' were being preserved, but many
    metadata tokens (like numbers, punctuation) were being filtered out.
    """
    tokens = text.split()
    filtered = []
    
    for tok in tokens:
        # Keep music tokens
        if is_music_token(tok):
            filtered.append(tok)
        # Keep ALL special tokens (not just those starting with '<')
        elif tok.startswith("<") and tok.endswith(">"):
            filtered.append(tok)
        # CRITICAL FIX: Keep metadata tokens that are part of the structure
        elif any(meta in tok for meta in ["<NAME=", "<BPM=", "<DURATION_", "StringInstrument_", "_", ":", "="]):
            filtered.append(tok)
        # Keep individual characters/numbers that are part of metadata
        elif tok in "0123456789.,-_: =":
            filtered.append(tok)
        # Keep alphabetic parts of metadata
        elif len(tok) <= 10 and (tok.isalnum() or any(c in tok for c in ".,#-_")):
            filtered.append(tok)
    
    result = " ".join(filtered)
    
    # Debug: Show what was kept vs removed
    original_tokens = text.split()
    filtered_tokens = result.split()
    
    if len(original_tokens) != len(filtered_tokens):
        print(f"WARNING: Token count changed from {len(original_tokens)} to {len(filtered_tokens)}")
        removed = set(original_tokens) - set(filtered_tokens)
        if removed:
            print(f"  Removed tokens (first 10): {list(removed)[:10]}")
    
    return result


def build_features(examples, tokenizer, mask_token_id, max_length=1024):
    """
    FIXED: Better feature building with more robust alignment checking
    """
    # Use the FIXED preprocessing
    inputs = [pre_tokenize_musical_text(t) for t in examples["input"]]
    targets = [pre_tokenize_musical_text(t) for t in examples["target"]]
    
    # Debug first few examples
    print(f"\nDEBUG: First input after preprocessing: {inputs[0][:200]}...")
    print(f"DEBUG: First target after preprocessing: {targets[0][:200]}...")

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
    
    # Statistics
    total_mask_tokens_found = 0
    total_sequences = len(input_ids)

    for inp, tgt in zip(input_ids, target_ids):
        lbl = [-100] * len(inp)
        mask_pos = [0] * len(inp)
        
        sequence_mask_count = 0

        for i, tok in enumerate(inp):
            if tok == mask_token_id:
                # CRITICAL: Ensure we don't go out of bounds
                if i < len(tgt):
                    lbl[i] = tgt[i]
                    mask_pos[i] = 1
                    sequence_mask_count += 1
                    total_mask_tokens_found += 1
                else:
                    print(f"WARNING: Mask position {i} out of bounds for target sequence length {len(tgt)}")

        labels.append(lbl)
        mask_positions.append(mask_pos)
        
        # Debug first sequence
        if len(labels) == 1:
            print(f"DEBUG: First sequence mask positions: {sum(mask_pos)}")
            mask_indices = [i for i, m in enumerate(mask_pos) if m == 1]
            print(f"DEBUG: Mask indices: {mask_indices}")
            for idx in mask_indices[:5]:  # Show first 5
                if idx < len(tgt):
                    expected_token = tokenizer.convert_ids_to_tokens([tgt[idx]])[0]
                    print(f"  Position {idx}: Should predict '{expected_token}'")
        
    print(f"PREPROCESSING SUMMARY:")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Total mask tokens found: {total_mask_tokens_found}")
    print(f"  Average masks per sequence: {total_mask_tokens_found / total_sequences:.2f}")

    if total_mask_tokens_found == 0:
        print("❌ CRITICAL ERROR: No mask positions found!")
        print("❌ This means your preprocessing is removing <MASK> tokens!")
        # Show a problematic example
        print(f"❌ Sample original input: {examples['input'][0][:200]}")
        print(f"❌ Sample processed input: {inputs[0][:200]}")
        return None
    
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
    print(f"Using mask token '<MASK>' with ID: {mask_token_id}")

    # Apply feature building
    for split in ["train", "test"]:
        print(f"\nProcessing {split} split...")
        dataset[split] = dataset[split].map(
            lambda ex: build_features(ex, tokenizer, mask_token_id),
            batched=True,
            remove_columns=["input", "target"],
            desc=f"Building {split} features",
            num_proc=1  # Use single process for better debugging
        )
        
        # Verify the split was processed correctly
        if dataset[split] is None:
            raise ValueError(f"Failed to process {split} dataset - no mask positions found!")

    return dataset