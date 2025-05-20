import torch
import random
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

def evaluate_oneshot(net, dataset, way=20, times=400, cuda=True):
    net.eval()
    correct = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for _ in range(times):
            classes = list(set([img[1] for img in dataset.dataset.imgs]))
            selected_classes = random.sample(classes, way)
            correct_class = selected_classes[0]

            support_imgs = [img for img in dataset.dataset.imgs if img[1] == correct_class]
            support_img = Image.open(random.choice(support_imgs)[0]).convert("L")
            support_img = dataset.transform(support_img).unsqueeze(0)
            if cuda:
                support_img = support_img.cuda()

            candidates = []
            for cls in selected_classes:
                candidate_img = Image.open(random.choice([img for img in dataset.dataset.imgs if img[1] == cls])[0]).convert("L")
                candidate_img = dataset.transform(candidate_img).unsqueeze(0)
                if cuda:
                    candidate_img = candidate_img.cuda()
                candidates.append(candidate_img)

            emb_support = net.forward_one(support_img)
            distances = [F.pairwise_distance(emb_support, net.forward_one(c)).item() for c in candidates]
            pred = np.argmin(distances)
            y_true.append(1)
            y_pred.append(1 if pred == 0 else 0)
            correct += (pred == 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nOne-Shot Evaluation on {way}-way {times}-trial")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return acc, prec, rec, f1
