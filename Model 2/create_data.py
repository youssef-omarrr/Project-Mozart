from miditok import REMI
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


# ----------------------------------------------------
# MIDI Tokenization
# ----------------------------------------------------
def tokenize_MIDI():
    """
    Tokenize all MIDI files in the given directory using the REMI tokenizer.
    Returns a list of token sequences.
    """
    tokenizer = REMI()
    MIDI_DIR = Path("../data/audio")

    tokenized = []
    for midi_path in tqdm(list(MIDI_DIR.glob("*.mid")), desc="Tokenizing MIDI files"):
        tokens = tokenizer.encode(midi_path)  # pass path directly
        tokenized.append(tokens)

    return tokenized


# ----------------------------------------------------
# Utility: Convert token sequences to int lists
# ----------------------------------------------------
def to_int_list(seq):
    """
    Convert any seq (TokSequence, list of TokSequence, or list of ints) 
    into a flat list of ints.
    """
    if hasattr(seq, "ids"):  # direct TokSequence
        return list(seq.ids)

    elif isinstance(seq, (list, tuple)):
        out = []
        for x in seq:
            if hasattr(x, "ids"):      # nested TokSequence
                out.extend(list(x.ids))
            elif isinstance(x, int):
                out.append(x)
            else:
                raise TypeError(f"Unexpected element type: {type(x)}")
        return out

    elif isinstance(seq, int):
        return [seq]

    else:
        raise TypeError(f"Unexpected type at top level: {type(seq)}")


# ----------------------------------------------------
# Vocab Size Extraction
# ----------------------------------------------------
def get_vocab_size(test_print=False):
    """
    Tokenizes the MIDI dataset and computes vocabulary size.
    Returns (vocab_size, int_seqs).
    """
    tokenized = tokenize_MIDI()
    int_seqs = [to_int_list(seq) for seq in tokenized]

    vocab_size = max(max(seq) for seq in int_seqs) + 1

    if test_print:
        print(f"Vocab size: {vocab_size}")
        print("First sequence:", int_seqs[0][:20])

    return vocab_size, int_seqs


# ----------------------------------------------------
# Dataset Definition
# ----------------------------------------------------
class MIDIDataset(Dataset):
    """
    Custom Dataset for training a Transformer on MIDI token sequences.
    Produces (x, y) pairs with optional stride-based sliding windows.
    """

    def __init__(self, sequences, seq_len=32, pad_id=0, stride=1):
        self.data = []
        self.seq_len = seq_len
        self.pad_id = pad_id

        for seq in tqdm(sequences, desc="Creating dataset"):
            ids = list(seq)

            if len(ids) < 2:  # skip too-short sequences
                continue

            # Sliding windows with configurable stride
            for i in range(0, len(ids) - 1, stride):
                x = ids[i:i+seq_len]
                y = ids[i+1:i+seq_len+1]

                # Pad if shorter than seq_len
                if len(x) < seq_len:
                    x = pad(torch.tensor(x, dtype=torch.long),
                            (0, seq_len - len(x)), value=pad_id)
                    y = pad(torch.tensor(y, dtype=torch.long),
                            (0, seq_len - len(y)), value=pad_id)
                else:
                    x = torch.tensor(x, dtype=torch.long)
                    y = torch.tensor(y, dtype=torch.long)

                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------------------------------
# Collate Function
# ----------------------------------------------------
def collate_fn(batch, pad_id=0):
    """
    Pads a batch of variable-length sequences to the max length in the batch.
    """
    xs, ys = zip(*batch)  # unzip
    xs = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    ys = pad_sequence(ys, batch_first=True, padding_value=pad_id)
    return xs, ys


# ----------------------------------------------------
# Dataset + DataLoader Creation
# ----------------------------------------------------
def create_dataset_and_dataloader(seq_len=32,
                                pad_id=0,
                                stride=8,
                                batch_size=32):
    """
    Creates a dataset and dataloader for training.
    """
    _, int_seqs = get_vocab_size(test_print=False)

    dataset = MIDIDataset(int_seqs,
                        seq_len=seq_len,
                        pad_id=pad_id,
                        stride=stride)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_id=pad_id))

    return dataset, loader
