# NeuroMera - Multimodal Emotion Classification

Detects emotions from **text**, **audio**, and **video** using fine-tuned transformer models.

## Models

| Modality | Model | Notebook | Emotions |
|----------|-------|----------|----------|
| Text | BERT (`bert-base-uncased`) | `NeuroMera_Fixed.ipynb` | anger, fear, joy, love, sadness, surprise |
| Audio | Wav2Vec2 (`facebook/wav2vec2-base`) | `audio.ipynb` | anger, fear, joy, neutral, sadness |
| Video | ViT (`google/vit-base-patch16-224`) | `video.ipynb` | auto-detected from folder names |

## Quick Start (all three notebooks)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime > Change runtime type > T4 GPU`
3. Update the dataset path in Step 1/2
4. `Runtime > Run all`
5. Model saves to Google Drive automatically

To predict without retraining, skip to the last step in any notebook and run from there.

---

## Text — `NeuroMera_Fixed.ipynb`

**Dataset format:** CSV with two columns

```
text,label
"i feel so happy today",joy
"this makes me angry",anger
```

**Labels:** `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`

```python
predict_emotion("I love you so much")       # -> love
predict_emotion("I am not happy at all")     # -> sadness
```

---

## Audio — `audio.ipynb`

**Dataset format:** `.wav` files in a single folder. Emotion code is extracted from position 3 of the filename (underscore-separated):

```
AudioWAV/
  01_01_ANG_01.wav      # ANG = anger
  01_01_HAP_02.wav      # HAP = joy
  01_01_SAD_01.wav      # SAD = sadness
  ...
```

**Emotion codes:** `ANG` (anger), `SAD` (sadness), `HAP` (joy), `FEA` (fear), `NEU` (neutral), `DIS` (anger)

```python
predict_emotion("/path/to/clip.wav")        # -> "joy"
```

---

## Video — `video.ipynb`

**Dataset format:** Video files organized in labeled folders:

```
VideoData/
  anger/
    clip1.mp4
    clip2.mp4
  joy/
    clip1.mp4
  sadness/
    ...
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

The notebook automatically extracts frames, detects faces (OpenCV Haar cascade), and trains a ViT classifier on the face crops. Prediction uses majority vote across frames.

```python
predict_video_emotion("/path/to/video.mp4")  # -> "anger"
```

---

## Project Structure

```
NeuroMera/
  NeuroMera_Fixed.ipynb   # Text emotion classification (BERT)
  audio.ipynb             # Audio emotion classification (Wav2Vec2)
  video.ipynb             # Video emotion classification (ViT)
  READ_THIS_FIRST.txt     # Explains what was fixed and why
  README.md
```
