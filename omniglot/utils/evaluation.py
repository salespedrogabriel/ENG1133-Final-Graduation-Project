import torch
import random
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from config import Config

def evaluate_oneshot(net, dataset, way=20, times=400, device="cpu"):
    net.eval()
    cfg = Config()
    correct = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for _ in range(times):
            classes = list(set([img[1] for img in dataset.dataset.imgs]))
            selected_classes = random.sample(classes, way)
            correct_class_idx = random.randint(0, way - 1)
            correct_class = selected_classes[correct_class_idx]

            # Imagem de suporte (positiva)
            support_imgs = [img for img in dataset.dataset.imgs if img[1] == correct_class]
            support_img = Image.open(random.choice(support_imgs)[0]).convert("L")
            support_img = dataset.transform(support_img).unsqueeze(0).to(cfg.DEVICE)

            # Imagens candidatas
            candidates = []
            for cls in selected_classes:
                candidate_imgs = [img for img in dataset.dataset.imgs if img[1] == cls]
                candidate_img = Image.open(random.choice(candidate_imgs)[0]).convert("L")
                candidate_img = dataset.transform(candidate_img).unsqueeze(0).to(cfg.DEVICE)
                candidates.append(candidate_img)

            emb_support = net.forward_one(support_img)
            distances = [F.pairwise_distance(emb_support, net.forward_one(c)).item() for c in candidates]
            pred = np.argmin(distances)

            # Avaliação binária (acertou ou não)
            y_true.append(correct_class_idx)  # índice da classe correta
            y_pred.append(pred)                # índice da classe prevista
            correct += (pred == correct_class_idx)

    # Verificando se o y_true está fixo
    #print("y_true:", y_true[:10])
    #print("y_pred:", y_pred[:10])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\nOne-Shot Evaluation on {way}-way {times}-trial")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    return acc, prec, rec, f1
