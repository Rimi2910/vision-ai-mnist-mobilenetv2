# Vision MNIST — MobileNetV2 (Transfer Learning)

**One-line:** Trained MobileNetV2 on MNIST (grayscale → RGB) using a memory-safe `tf.data` pipeline with transfer learning and fine-tuning.

---

## Results

* **Final test accuracy:** **97.20%**
* Short summary: Converted MNIST 28×28 grayscale digits to RGB and resized to 96×96, used MobileNetV2 (ImageNet weights) with a small classification head, trained with on-the-fly preprocessing to avoid RAM crashes, then fine-tuned the top layers.

---

## Files in this repo

* `vision_ai_.ipynb` — runnable Colab notebook (recommended way to reproduce).
* `mnist_mobilenetv2_tl.keras` — final saved model **(if included; otherwise available via Drive link below)**.
* `training_curves.png` — accuracy & loss plots.
* `confusion_matrix.png` — confusion matrix heatmap.
* `sample_preds/` — example images with predicted vs true labels.
* `presentation.pdf` — 5-slide summary.
* `demo_video.mp4` — 30-second demo (if included; otherwise Drive link).
* `predict.py` — small CLI script to run a prediction on a single image.
* `requirements.txt` — Python dependencies.

---

## How to run (Colab — recommended)

1. Open `vision_ai_.ipynb` in Google Colab.
2. `Runtime → Change runtime type` → set **Hardware accelerator: GPU** (optional but faster).
3. Run all cells from top to bottom. The notebook mounts your Google Drive and saves artifacts to `/content/drive/MyDrive/vision_mnist_project` by default.

**Quick cell to load saved model in Colab:**

```python
from tensorflow import keras
model = keras.models.load_model('/content/drive/MyDrive/vision_mnist_project/mnist_mobilenetv2_tl.keras')
```

---

## How to run locally (quick)

1. Clone the repo.
2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the prediction script:

```bash
python predict.py --model /path/to/mnist_mobilenetv2_tl.keras --image sample_preds/sample_0.png
```

---

## Model file (if not uploaded to GitHub)

If the model file is too large to include in the repo, download it from Google Drive:
**Model (Drive):** `https://drive.google.com/your-model-link-here`
(To load in Colab after downloading to `/content/drive/MyDrive/...`, use the code snippet above.)

---

## Notes & design choices

* Used **on-the-fly preprocessing** (`tf.data`) to convert grayscale → RGB and resize per batch to avoid memory issues.
* Two-stage training: (1) freeze MobileNetV2 base and train classification head; (2) unfreeze top layers and fine-tune with a lower learning rate.
* Chose 96×96 input size to balance resolution vs memory/speed.

---

## Contact

If you have questions or want improvements (e.g., Grad-CAM visualizations, better augmentation, or a Flask demo), open an issue or contact me at `<cybersuman@yahoo.com>`.

---

*Prepared with ❤️ — replace placeholders (Drive links / contact) before final submission.*
