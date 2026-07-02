# TinyStories GPT — Training a Small Story-Generation Model

Fine-tuning GPT-2 (124M) on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset to build a small language model that writes simple children's stories. Built as a hands-on exercise to understand the full LLM training loop — data streaming, tokenization, causal-LM loss, checkpointing, and sampling — rather than treating models as black boxes.

> Started from a course exercise on custom text-generation engines, then reimplemented and debugged it end-to-end (see "What I fixed" below).

## Why TinyStories?

TinyStories (Eldan & Li, 2023) contains ~2.1M short stories written with a deliberately small vocabulary. That makes it possible for a small model, trained on a single free-tier GPU, to produce genuinely coherent text — which is normally impossible at this scale.

## Project structure

```
tinystories-gpt2/
├── train.py          # streaming dataset + training loop + checkpointing
├── generate.py       # sample stories from any checkpoint
├── requirements.txt
└── checkpoints/      # created during training (one folder per epoch)
```

## How it works

1. **Streaming data** — the dataset is too large for Colab RAM, so `TinyStoriesStream` (an `IterableDataset`) tokenizes stories on the fly.
2. **Tokenization** — GPT-2's BPE tokenizer; EOS reused as the pad token since GPT-2 has none. Sequences padded/truncated to 128 tokens.
3. **Loss masking** — labels at padding positions are set to `-100` so the cross-entropy loss ignores them.
4. **Training** — AdamW (lr 5e-5), gradient clipping at norm 1.0, average loss logged per epoch.
5. **Checkpointing** — model + tokenizer saved to disk every epoch, so a Colab disconnect never loses more than one epoch.
6. **Qualitative checks** — a sample story is generated after every epoch to catch problems that the loss number alone won't show.

## What I fixed along the way

My first version produced garbled output like:

```
Once upon a time there a girl Lily She to her's. was years and to. day she to wi...
```

even though the loss was steadily decreasing (1.47 → 1.12). Lessons learned:

- **Don't "clean" natural text.** Aggressive preprocessing (dropping short/common tokens) destroys grammar. The model can only learn what it sees.
- **Loss going down ≠ model is good.** The model was minimizing loss on broken data. Sampling every epoch caught what the metric hid.
- **Set the attention mask and pad token explicitly** — otherwise HF warns and generation quality suffers.
- **Tokenizer/model must match.** The reference `TinyStories-3M` model uses the GPT-Neo tokenizer, not GPT-2's.

## Usage

```bash
pip install -r requirements.txt

# Train (defaults: 3 epochs, batch 16, seq len 128)
python train.py --epochs 3 --batch_size 16

# Generate from your checkpoint
python generate.py --model checkpoints/epoch_3 --prompt "Once upon a time"

# Baseline comparison with the official pretrained model
python generate.py --model roneneldan/TinyStories-3M
```

On a Colab T4, one epoch over ~100k stories takes roughly 1–1.5 hours.

## Results

| Epoch | Avg loss |
|-------|----------|
| 1     | (fill in from your run) |
| 2     | ... |
| 3     | ... |

_(Add your own loss curve plot from `checkpoints/loss_history.json` and 2–3 sample stories here after training.)_

## References

- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023)
- Hugging Face `transformers` and `datasets` documentation
- Course exercise on building a custom text-generation engine (starting point for this project)
