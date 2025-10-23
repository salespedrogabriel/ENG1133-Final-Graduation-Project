# One-Shot Learning Approach for Underwater Inspection Image Classification

## About the Project
This project was developed as part of the **Undergraduate Thesis** in Computer Engineering.
The goal is to **classify underwater** images obtained from inspections performed by **ROVs (Remotely Operated Vehicles)** using a **One-Shot Learning** approach based on **Siamese Networks**.  

The proposal aims to reduce manual analysis time of inspection images — a process that can take up to **40 hours per specialist** — by automating part of the identification of underwater structures and objects.

## Objectives

- Implement a Siamese Neural Network (SNN) architecture to learn image similarity.  
- Adapt the classical Omniglot dataset approach to a new set of real underwater inspection images.  
- Evaluate model performance on N-way One-Shot tasks.  
- Generate complete performance metrics (accuracy, precision, recall, F1-score, and confusion matrix).  

## Technical Approach

The model was developed using **PyTorch**, with the following configuration:

- **Architecture:** Siamese CNN with 3 convolutional blocks  
  and two fully connected layers  
- **Loss function:** `BCEWithLogitsLoss`  
- **Optimizer:** `Adam (lr=0.0001)`  
- **Evaluation:** *N-way* tasks (e.g., 3-way, 20-way)  
- **Custom dataset:** real underwater inspection images divided into the following classes:  
  - `Pipeline`  
  - `Equipment`  
  - `Flange`  
  - `ROV`  
  - `Manipulator`  
  - `Object`

## How to Run

### Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib

Install dependencies with:  

```
pip install -r requirements.txt
```

### Model Training

Run the main training script:  

```
python train.py
```

During training, metrics are saved in .csv files, and model weights (.pth) are stored in the model folder.

### Results

Accuracy (Train): 96.5%  
Accuracy (Test): 76.5%  

F1-Score (Train): 0.963  
F1-Score (Test): 0.764  

### Future Work

* Test less complex architectures  
* Explore other approaches such as Few-Shot Learning  
* Increase intra-class variation  
* Expand the number of samples while maintaining dataset balance  
