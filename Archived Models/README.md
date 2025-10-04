## Model comparison

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
