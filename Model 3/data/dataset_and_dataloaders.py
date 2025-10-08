import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.nn.functional import pad

# Dataset definition
# -------------------
class MidiDataset(Dataset):
    """Dataset that converts token-id sequences into sliding-window (input, target) tensor pairs.

    Each sample is a pair (x, y) where x is a tensor of token ids of length `seq_len`
    and y is the same sequence shifted by one position (next-token targets). Short
    windows at the end are padded with `pad_id`.
    """

    def __init__(
            self,
            sequences: list = None,
            seq_len: int = 512,
            pad_id: int = 0,
            stride: int = 1):
        """Build the dataset by sliding a window over each token sequence.

        Args:
                sequences (list): List of token-id sequences (iterables of ints).
                seq_len (int): Length of each input window (default 512).
                pad_id (int): Token id used for padding short windows (default 0).
                stride (int): Sliding window stride (default 1).
                prebuilt_dataset (list): Pre-built list of (x, y) pairs to load directly.
        """
        
        # 0. Init dataset list
        # -----------------------
        self.dataset = []

        # 1. Go throgh each sequence of ids
        # ----------------------------------
        for seq in tqdm(sequences, desc="Creating dataset..."):

            # 2. Create the sliding window and make them into tensors
            # --------------------------------------------------------
            for i in range(0, len(seq)-1, stride):
                x = torch.tensor(seq[i:i+seq_len], dtype=torch.long)
                y = torch.tensor(seq[i+1:i+seq_len+1],  dtype=torch.long)

                # 3. Pad last set of data if it is less than seq_len
                # and also quit the loop to avoid entries with many pad tokens
                # But before if the seq is less than half the seq_len ignore it
                # --------------------------------------------------------------
                if len(x) < seq_len//2 :
                    break # to avoid noisy sequences that are mostly pad tokens
                
                elif len(x) < seq_len:
                    x = pad(input=x,
                            # pad = (pad_left, pad_right)
                            pad=(0, seq_len-len(x)),
                            value=pad_id)
                    y = pad(input=y,
                            # pad = (pad_left, pad_right)
                            pad=(0, seq_len-len(y)),
                            value=pad_id)
                    break

                # 4. Append pair to the dataset list
                # -----------------------------------
                self.dataset.append((x, y))

    # get length
    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.dataset)

    # get item
    def __getitem__(self, idx):
        """Return the (x, y) sample at the given index."""
        return self.dataset[idx]


# Dataset and Dataloader function
# --------------------------------
def create_dataloaders(
				tokenized_data: list,
				seq_len: int = 512,
				pad_id: int = 0,
				stride: int = 8,
				batch_size: int = 32,
				val_split: float = 0.1,
				random_seed: int = 42):
    """Create training and validation DataLoaders from tokenized MIDI data.

    Args:
        tokenized_data (list): List of token-id sequences.
        load_dataset (str): Path to loads presaved dataset.
        seq_len (int): Sequence length for sliding windows.
        pad_id (int): Padding token id.
        stride (int): Sliding window stride.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data to reserve for validation (0.0-1.0).
        random_seed (int): Seed for deterministic train/val split.

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """

    # 1. Create dataset
    # ------------------
    dataset = MidiDataset(
                tokenized_data,
                seq_len=seq_len,
                pad_id=pad_id,
                stride=stride)

    # 2. Split dataset into training and validation
    # ----------------------------------------------
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed))

    # 3. Create dataloaders
    # ----------------------
    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset,
								batch_size=batch_size,
								shuffle=True,
								pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True)

    # For now i will only return the dataloaders (if needed i may return the dataset too)
    return train_dataloader, val_dataloader
