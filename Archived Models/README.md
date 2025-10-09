## Model comparison

The old original idea was to make a chatbot where I can tell it the name of the symphony and how long I want it to be and optionally the bpm, and then the model would generate all of this.

My train of thought was like this:
	1. Create an encoder that takes these MIDI files and turs them to text and extract all the data needed (e.g. Name, bpm, instruments)
	2. Use a pretrained model to be able to fill these data and finetune to learn and be able to generate music sequences like the one that the encoder created.
	3. Once the model is done writing text like the ones the encoder created, this text out would then go to the decoder to convert the text back to audio

> * The encoder and decoder where build from scratch.
> * I quickly realized that the model being a chatbot is way too advanced so I just extracted the prompt manually.
> * For a more detailed models comparison, view the *Archived Models* folder.
### **Model 0**

**Idea:**

- Fine-tune GPT-2 (causal LM) using LoRA for efficient adaptation.
- Represent MIDI data as flattened JSON-like sequences with BPM, instruments, and notes.
- Add penalties in the loss function for repeated `REST` tokens.

**Output:**

- Generated sequences mostly filled with `REST` tokens and repetitive patterns.
- If the input had multiple instruments, it would only generate a sequence in the first instrument and ignore the rest.

**Observation:**

- The used pretrained tokenizer only had normal vocabulary token not music tokens, that's why it only generated `REST` tokens, as it is a normal English word.
- Tokenizer split music tokens incorrectly (e.g., `C4_q` → `C`, `4`, `q`).
- Inputs became extremely long (>100k chars), making training unstable.
- Approach unsuitable for structured symbolic data.

---

### **Model 1**

**Idea:**

- Switch to a seq2seq model (facebook-BART) for more controlled generation.
- Build dataset with masked inputs and unmasked targets.
- Add all musical notes to the tokenizer to preserve full tokens.
- Apply custom loss to penalize non-music outputs.

**Improvement over Model 0:**

- Used a model better suited for structured sequence generation (seq2seq).
- Fixed token fragmentation by expanding tokenizer vocabulary.
- Split the dataset so that instead of really long lines, containing all the symphony in one line, to have a max of 20-30 tokens per line.
- Used Masking to show the model where it needs to generate the music sequence (to fix the model only generating one instrument)

**Output:**

- Generated mostly non-music tokens and failed to maintain musical structure.

**Observation:**

- Masking strategy was too fragile and the model always found ways to overcome the loss penalty in the wrong ways.
- Loss-based enforcement wasn’t enough to make the model follow structure.
- Improved tokenizer helped preserve note tokens but didn’t solve sequence coherence.

---

### **Model 2**

**Idea:**

- Build a custom Transformer from scratch designed specifically for symbolic music (to avoid pre-learned non-music tokens).
- Use **REMI tokenization** (Bar → Position → Pitch → Velocity → Duration) (avoids tokenizers with non-music tokens).
- Add structural constraints in token order (because the REMI tokenizer splits the music note into multiple tokens for a smaller vocab size ) and a composite loss (cross-entropy + structural loss).
- Apply training improvements: label smoothing, gradient clipping, LayerNorm before projection.

**Improvement over Model 1:**

- Designed architecture and tokenization tailored to the task.
- Added explicit structure and stability mechanisms during training.

**Output:**

- Generated coherent piano pieces with consistent rhythm and melody.
- The output only uses the piano (not other instruments), and is random.

**Observation:**

- Model successfully learned musical structure and produced normal music sequences.
- Still limited in expressiveness and variation, but first fully working system.

---

| Model       | Type / approach                                | Key techniques attempted                                                                                                                                                                                                                                                                                                                                                                           | Strengths / what was learned                                                                                                                                                                                                                                                        | Outcome                                                                                                                                      |
| ----------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model 0** | GPT-2 Medium (causal LM) with LoRA fine-tuning | - Integrated **LoRA** for parameter-efficient adaptation<br>- Custom loss with penalties for repeated `REST` tokens<br>- Flattened JSON-like MIDI representation with BPM, instruments, notes                                                                                                                                                                                                      | - Hands-on experience extending GPT-2 vocab with domain-specific tokens<br>- Learned tokenizer limitations for symbolic data (splitting tokens like `C4_q` to C, 4, q)<br>- Practiced scaling LoRA to large pretrained models                                                       | ❌ Generation collapsed to mostly `REST`; tokenizer fragmentation and extremely long input lines (>100k chars) made this approach impractical |
| **Model 1** | facebook-BART (seq2seq) with LoRA fine-tuning  | - Built dataset with masked inputs and unmasked targets (up to 50 tokens per line)<br>- Added all musical notes into tokenizer<br>- Custom loss penalizing non-music tokens<br>- Attempted rule-based enforcement via loss                                                                                                                                                                         | - Demonstrated ability to adapt `seq2seq` architectures to symbolic sequence generation<br>- Successfully re-engineered the tokenizer to preserve music tokens<br>- Learned about model “adversarial” behavior against loss rules (ignoring penalties)                              | ❌ Generated mostly non-music tokens; masking strategy too brittle; loss not sufficient to enforce structure                                  |
| **Model 2** | Custom Transformer (from scratch)              | - REMI tokenization (compact, music-aware)<br>- Sinusoidal positional encoding<br>- Causal triangular attention mask<br>- GPT-style embedding scaling<br>- LayerNorm before projection<br>- **Structural constraints** on token order (Bar → Position → Pitch → Velocity → Duration)<br>- Composite loss: cross-entropy + structural loss<br>- Training tricks: label smoothing, gradient clipping | - Designed and implemented full Transformer architecture with PyTorch<br>- Integrated domain knowledge into sequence rules<br>- Stabilized training with architectural/training refinements<br>- Built end-to-end pipeline: tokenize → train → validate → generate → synthesize WAV | ✅ First working system: produces coherent piano pieces; still limited quality but proves pipeline and model design                           |


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
