import os
import torch

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(BASE_DIR, "data_split_1000")
    TRAIN_DIR = os.path.join(DATA_DIR, "images_background_reestructured")
    TEST_DIR = os.path.join(DATA_DIR, "images_evaluation_reestructured")

    BATCH_SIZE = 128
    MAX_ITER = 40_000
    WAY = 3
    TIMES = 200
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_DIR = os.path.join(BASE_DIR, "model")
    HISTORY_CSV = os.path.join(BASE_DIR, "epoch_history.csv")
    RESULTS_CSV = os.path.join(BASE_DIR, "training_results.csv")

    log_csv = HISTORY_CSV
    resultados_csv = RESULTS_CSV
    loss_pickle = os.path.join(BASE_DIR, "loss_history.pkl")
    train_dataset = TRAIN_DIR
    test_dataset = TEST_DIR
    weight_decay = WEIGHT_DECAY
    batch_size = BATCH_SIZE
    max_iter = MAX_ITER
    way = WAY
    times = TIMES
    learning_rate = LEARNING_RATE
    model_dir = MODEL_DIR
    show_every = 100
    save_every = 500
    test_every = 1000
    num_workers = 2
