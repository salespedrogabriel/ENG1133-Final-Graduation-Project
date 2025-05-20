# One-Shot Learning with Omniglot using a Siamese Network

This project implements an image classifier based on One-Shot Learning using Siamese Neural Networks and the [Omniglot dataset](https://github.com/brendenlake/omniglot).

## 📁 Project Structure

```
my_project/
│
├── data/
│   ├── images_background_restructured/
│   └── images_evaluation_restructured/
│
├── datasets/
│   ├── __init__.py
│   ├── omniglot_train.py
│   └── omniglot_test.py
│
├── models/
│   ├── __init__.py
│   └── siamese.py
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py
│   └── logging_utils.py
│
├── train.py
├── training_results.csv
├── epoch_history.csv
└── README.md
```

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- Pillow

Install with:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn Pillow
```

## 🚀 Running Training

Make sure the `images_background_restructured` and `images_evaluation_restructured` folders are correctly organized with one subfolder per class.

Run training with:

```bash
python train.py
```

During training:
- Models are saved in the `model/` folder every 100 iterations.
- One-Shot evaluations are performed periodically on both train and test sets.
- Results are saved to:
  - `epoch_history.csv`: metrics per batch.
  - `training_results.csv`: final metrics and training parameters.

## 📊 Results

At the end of training, the script prints key metrics:
- Accuracy
- Precision
- Recall
- F1 Score

These metrics help assess model performance comprehensively.

## 📝 Notes

- The model architecture is based on the one proposed in *Koch et al., 2015*, with some adaptations.
- Images are converted to grayscale and transformed using `RandomAffine`.

## 📚 References

- [Siamese Neural Networks for One-shot Image Recognition – Koch et al. (2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Omniglot Dataset](https://github.com/brendenlake/omniglot)

---

Developed with ❤️ for educational and experimental purposes.
