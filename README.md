# Project Mozart 🎶

![alt text](imgs/mozart.jpg)

A `PyTorch` implementation of a Transformer-based music generation pipeline using `REMI` tokenization. This repo documents the full experimentation path, successful Model 2 plus the earlier failed-but-informative attempts, and highlights the technical decisions and lessons learned.

---

## Quick summary

- **Project Mozart** processes `MIDI` files, tokenizes them with `REMI`, trains custom `Transformer` models (several experiments), and generates MIDI output which can be converted to `WAV` using FluidSynth + custom SoundFonts. 
- **Model 2** is the working implementation that produces coherent piano music after training, although the output is still not the best (needs more training).
-  Models 0 and 1 did not produce usable final output but were vital experiments that demonstrate advanced engineering and debugging skills.

---

## Highlights / Why this is interesting

- **Engineering-first workflow:** multiple architecture experiments, tokenizer engineering, and loss-function design.
- **Practical music-token engineering** with REMI and structural constraints to improve validity of generated sequences.
- **Custom Transformer architecture** + training tricks (sinusoidal positional encodings, causal masking, label smoothing, gradient clipping).
- **Experimentation with parameter-efficient finetuning (LoRA)**, seq2seq models, and custom regularizers, shown as deliberate, reproducible experiments even when they failed.
- **End-to-end pipeline:** MIDI → tokenization → model → generate MIDI → synthesize WAV (FluidSynth + SoundFont).

---

## Project structure (short)

```
├── Model 0/              # failed experiment: GPT-2 medium + LoRA (tokenizer issues)
├── Model 1/              # failed experiment: facebook-BART seq2seq + LoRA (masking/loss issues)
├── Model 2/              # working model — current best
│   ├── create_data.py       # MIDI tokenization & dataset preparation (REMI)
│   ├── create_model.py      # MusicTransformer2 architecture
│   ├── train_model.py       # training loop, validation, checkpointing
│   ├── generate.py          # sampling & generation utilities
│   ├── notes_utils.py       # conversion, MIDI/WAV helpers, music-theory helpers
│   └── checkpoints/         # saved model checkpoints
└── Encoder and Decoder/  # original encoder/decoder components (used during earlier experiments)
```

---
## Model comparison
The main idea was to make a chatbot where I can tell it the name of the symphony and how long I want it to be and optionally the bpm, and then the model would generate all of this.

My train of thought was like this:
	1. Create an encoder that takes these MIDI files and turs them to text and extract all the data needed (e.g. Name, bpm, instruments)
	2. Use a pretrained model to be able to fill these data and finetune to learn and be able to generate music sequences like the one that the encoder created.
	3. Once the model is done writing text like the ones the encoder created, this text out would then go to the decoder to convert the text back to audio

> The encoder and decoder where build from scratch.
> I quickly realized that the model being a chatbot is way too advanced so I just extracted the prompt manually.
### **Model 0**

**Idea:**

* Fine-tune GPT-2 (causal LM) using LoRA for efficient adaptation.
* Represent MIDI data as flattened JSON-like sequences with BPM, instruments, and notes.
* Add penalties in the loss function for repeated `REST` tokens.

**Output:**

* Generated sequences mostly filled with `REST` tokens and repetitive patterns.
* If the input had multiple instruments, it would only generate a sequence in the first instrument and ignore the rest.

**Observation:**

- The used pretrained tokenizer only had normal vocabulary token not music tokens, that's why it only generated `REST` tokens, as it is a normal English word.
* Tokenizer split music tokens incorrectly (e.g., `C4_q` → `C`, `4`, `q`).
* Inputs became extremely long (>100k chars), making training unstable.
* Approach unsuitable for structured symbolic data.

---

### **Model 1**

**Idea:**

* Switch to a seq2seq model (facebook-BART) for more controlled generation.
* Build dataset with masked inputs and unmasked targets.
* Add all musical notes to the tokenizer to preserve full tokens.
* Apply custom loss to penalize non-music outputs.

**Improvement over Model 0:**

* Used a model better suited for structured sequence generation (seq2seq).
* Fixed token fragmentation by expanding tokenizer vocabulary.
* Split the dataset so that instead of really long lines, containing all the symphony in one line, to have a max of 20-30 tokens per line.
* Used Masking to show the model where it needs to generate the music sequence (to fix the model only generating one instrument)

**Output:**

* Generated mostly non-music tokens and failed to maintain musical structure.

**Observation:**

* Masking strategy was too fragile and the model always found ways to overcome the loss penalty in the wrong ways.
* Loss-based enforcement wasn’t enough to make the model follow structure.
* Improved tokenizer helped preserve note tokens but didn’t solve sequence coherence.

---

### **Model 2**

**Idea:**

* Build a custom Transformer from scratch designed specifically for symbolic music (to avoid pre-learned non-music tokens).
* Use **REMI tokenization** (Bar → Position → Pitch → Velocity → Duration) (avoids tokenizers with non-music tokens).
* Add structural constraints in token order (because the REMI tokenizer splits the music note into multiple tokens for a smaller vocab size ) and a composite loss (cross-entropy + structural loss).
* Apply training improvements: label smoothing, gradient clipping, LayerNorm before projection.

**Improvement over Model 1:**

* Designed architecture and tokenization tailored to the task.
* Added explicit structure and stability mechanisms during training.

**Output:**

* Generated coherent piano pieces with consistent rhythm and melody.
* The output only uses the piano (not other instruments), and is random.

**Observation:**

* Model successfully learned musical structure and produced normal music sequences.
* Still limited in expressiveness and variation, but first fully working system.

---

## Checkpoint / resume tip (common bug fix)

If you load checkpoints and want to print `val_loss` in an f-string, use single quotes inside the f-string to avoid syntax errors:

```python
if resume_training:
    ckpt_path = "checkpoints/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    print(f"Loaded best model (val_loss={checkpoint['val_loss']:.4f})")
```

---

## Technical achievements

- Built a complete **tokenization** pipeline for MIDI data (REMI), including vocabulary analysis and data sharding for training.
- Designed and implemented **structural token constraints** to enforce valid musical transitions (Bar → Position → Pitch → Velocity → Duration).
- Implemented a custom **Transformer-based model** (sinusoidal position encodings, causal masking, layernorm placement, embedding scaling).
- Engineered a **custom combined loss** (cross-entropy + structural loss) and diagnostic tooling to quantify non-musical tokens.
- Applied **training regularization**: label smoothing, gradient clipping, validation loop with checkpointing and resume support.
- Practical systems work: MIDI→WAV conversion with FluidSynth, SoundFont support, and sample generation scripts.
- Experimented with `LoRA` finetuning and `seq2seq` baselines; identified failure modes and iterated with principled fixes.
- Learned the practical difference between **causal LLMs** (predict next token given history) and **seq2seq models** (encode input sequence then decode output sequence).
- Built a functional `encoder` (MIDI → JSON-like text) and `decoder` (text → WAV) that, while not used in the final pipeline, *remain fully usable for other applications*.

---

## What worked (Model 2)

- **Tokenization** with REMI produced compact, music-aware vocabulary.
- **Causal attention** and **triangular masks** prevented token leakage.
- **Structural constraints** reduced syntactic invalid outputs and improved musical validity.
- **Training stability** improved with LayerNorm-before-projection, embedding scaling, and gradient clipping.
- End-to-end generation pipeline (tokenize → train → sample → synthesize) is reproducible.

---

## What failed (and why that’s valuable)

### 1. Data & preprocessing

- Flattening entire pieces into text lines >100k chars: learned to respect model input windows and the need to shard/segment sequences.
- Overly large datasets stressed memory and training time, highlighting the importance of data preprocessing, batching strategies, and even curriculum learning.

### 2. Tokenization

- Pretrained tokenizers (GPT-2) split musical tokens incorrectly (e.g., "C4_q" → "C", "4", "_", "q"): showed why domain-specific tokenization is essential.
- Experiments confirmed that language-based tokenizers assume natural text distributions, which don’t map well to symbolic music — leading to the design of a custom REMI tokenizer.
- Even when music notes are added to the pretrained tokenizer, they represent a smaller portion of the total number of tokens leading to ignoring them in most cases.

### 3. Modeling & architecture

- Loss-based enforcement with complex penalties encouraged adversarial behavior: models found shortcuts instead of following rules — a lesson in constrained optimization and unintended equilibria.
- Masking strategies in seq2seq (BART) still produced non-music tokens despite heavy penalties, teaching robustness testing, dataset balancing, and loss debugging.
- Learned the difference between causal LLMs (autoregressive, natural fit for sequence continuation) vs seq2seq models (encode input, then decode output) — and why causal LMs were better for open-ended music generation.

### 4. Training techniques

- LoRA finetuning reduced training cost but exposed the limits of parameter-efficient methods when pretrained weights are misaligned with the domain (language vs symbolic music).
- Exploding gradients and unstable training in earlier attempts underscored the need for gradient clipping, label smoothing, and better initialization.

### 5. Engineering lessons

- Encoder/decoder components (MIDI → text, text → WAV) didn’t integrate into the final model, but they remain functional standalone tools that can be reused in other projects.
- The iterative failures made it clear that building reusable modules (data pipeline, tokenizer, encoder/decoder, generation scripts) is just as valuable as the final model.

---
