# Project Mozart 🎶

![alt text](imgs/mozart.jpg)
> The project is almost done, but it will require alot of training and the results are not good right now.\

---

A small research project for *symbolic music processing and generation*. The repository is organized into two main parts: an **encoder/decoder** pipeline that converts between MIDI/audio and a text JSON representation, and an **LLM-based** music generation pipeline.

### Summary
- Purpose: 
   1. turn MIDI into a tokenized text format for model training
   2. Generate new token sequences with an LLM
   3. Decode them back to audio using a soundfont (for better quality).
- Status: end-to-end pipeline and many helper notebooks/scripts are included. Results are preliminary and training requires significant compute.

### Project layout (high level)
- **1) Audio part/:** encoder and decoder utilities, examples and test audio outputs.
- **2) Dataset creation/:** scripts and notebooks used to create tokenized datasets from MIDI files.
- **dataset/:** raw MIDI collection and generated text datasets (train/test splits and unique token lists).
- **[⏳] Model 1/:** data creation, model/tokenizer creation, training and evaluation utilities for Model 1 experiments.


### **1) Encoder & Decoder**
#### **Encoder:** 
- converts MIDI files into a compact JSON/text representation of musical events (notes, durations, velocities, time shifts, program changes, etc.). 
- Token files are stored at dataset/text/ .
- Use the encoder scripts/notebooks in the "1) Audio part/" and "2) Dataset creation/" folders to produce the tokenized text. Notebooks demonstrate example usage.

#### **Decoder:** 
- converts the generated token text back into a MIDI or directly to WAV using a soundfont (e.g., FluidSynth). This improves quality compared to raw waveforms.
- Audio generation helpers are in 1) Audio part/.
- Typical workflow: tokens/text -> decoder -> temporary MIDI -> fluidsynth/timidity -> WAV.

### **1) LLM for Music Generation**
- [❌] Model 0 (GPT-2 / gpt2-medium): an initial attempt using GPT-2 medium. Training this model required more GPU memory and compute than available, so it did not produce usable results on limited hardware.
- Model 1 (BART-small): a lighter encoder-decoder setup (BART-small) with the training pipeline and tokenizer prepared. This model is expected to be easier to train given resource constraints. The pipeline (data creation, tokenization, training, evaluation) is implemented under "[⏳] Model 1".
  - Note: With sufficient epochs and compute, Model 1 should improve, but **results may still be constrained by available resources**.

### Finetuning & technical highlights
- **Fine-tuning approach:** used LoRA (Low-Rank Adapters) to finetune LLMs efficiently on musical token data. This reduces GPU memory and speeds up experiments while keeping base model weights intact.
- **Tokenization:** music is represented as tokenized musical events (notes, durations, velocities, time-shifts, program changes). This event-based tokenization and JSON/text format enables sequence modeling with standard LLMs.
- **Practical skills demonstrated:**
  - MIDI/audio processing and synthesis (encoder/decoder workflows, FluidSynth/soundfont usage).
  - Tokenizer creation and data pipeline for symbolic music.
  - Model finetuning with LoRA and training/evaluation of Transformer-based models (PyTorch / Hugging Face Transformers).
  - Experiment design for resource-constrained environments (model selection, batch sizing, gradient accumulation).
  - Reproducible notebook-driven workflows, scripting, and dataset preparation.
  - Prompting, generation utilities, and post-processing to convert tokens back to audio.


### Notes & suggestions
- ".wav" files have not been pushed due to large size.
- Use a modern soundfont (SF2) and FluidSynth for best playback quality for more details check "1) Audio part/".
- If you have limited GPU RAM, prefer smaller models (BART-small or distilled variants) and smaller batch sizes.
