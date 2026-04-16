# NeuroMera - Emotion Classification

A BERT-based text emotion classifier that detects 6 emotions: **anger**, **fear**, **joy**, **love**, **sadness**, and **surprise**.

## Quick Start

1. Open `NeuroMera_Fixed.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime > Change runtime type > T4 GPU`
3. Update `CSV_PATH` in Step 1 to point to your dataset on Google Drive
4. `Runtime > Run all`
5. Model saves to Google Drive automatically after training

## Dataset Format

Your CSV needs exactly two columns:

```
text,label
"i feel so happy today",joy
"this makes me angry",anger
"i am really scared",fear
```

**Supported labels:** `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`

## Using a Trained Model

To predict without retraining, skip to **Step 7** in the notebook and run from there. It loads the saved model from Google Drive and gives you an interactive prompt.

```python
predict_emotion("I love you so much")       # -> love
predict_emotion("I am not happy at all")     # -> sadness
predict_emotion("My heart is singing")       # -> joy
```

## Architecture

- **Model:** `bert-base-uncased` fine-tuned for sequence classification
- **Tokenizer:** BERT WordPiece (max_length=128)
- **Training:** 3 epochs, lr=2e-5, batch_size=16, fp16
- **No manual text preprocessing** — raw text goes directly to BERT's tokenizer, which preserves sentence structure and negation

## Project Structure

```
NeuroMera/
  NeuroMera_Fixed.ipynb   # Training + evaluation + prediction notebook
  READ_THIS_FIRST.txt     # Setup guide for contributors
  README.md
```
