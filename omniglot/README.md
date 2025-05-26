# One-Shot Learning with Omniglot using a Siamese Network

This project implements an image classifier based on One-Shot Learning using Siamese Neural Networks and the [Omniglot dataset](https://github.com/brendenlake/omniglot). Make sure to unzip [data.rar](https://github.com/salespedrogabriel/One-Shot-Inspecoes-Submarinas/blob/main/omniglot/data.rar) folder

## Project Structure

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

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- torchvision 0.16.0
- numpy 1.24.3
- matplotlib
- seaborn
- scikit-learn 1.3.0
- Pillow 9.5.0

Install with:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn Pillow
```

## Running Training

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

## Results

At the end of training, the script prints key metrics:
- Accuracy
- Precision
- Recall
- F1 Score

These metrics help assess model performance comprehensively.

## Notes

- The model architecture is based on the one proposed in *Koch et al., 2015*, with some adaptations.
- Images are converted to grayscale and transformed using `RandomAffine`.

## References

- [Siamese Neural Networks for One-shot Image Recognition – Koch et al. (2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Omniglot Dataset](https://github.com/brendenlake/omniglot)

---
