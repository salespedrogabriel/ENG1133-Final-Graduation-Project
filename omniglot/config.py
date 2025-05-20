import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "images_background_reestruturado")
TEST_DIR = os.path.join(DATA_DIR, "images_evaluation_reestruturado")

BATCH_SIZE = 128
MAX_ITER = 15000
WAY = 20
TIMES = 400
LEARNING_RATE = 0.0006
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = os.path.join(BASE_DIR, "model")
HISTORY_CSV = os.path.join(BASE_DIR, "historico_epocas.csv")
RESULTS_CSV = os.path.join(BASE_DIR, "resultados_treino.csv")
