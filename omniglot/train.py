import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import time
import os
import pickle
from config import Config
from datasets.omniglot_dataset import OmniglotTrain, OmniglotTest
from models.siamese import Siamese
from utils.evaluation import evaluate_oneshot
from utils.logger import salvar_log_epoca_csv, salvar_resultados_csv
from torchvision import transforms

def main():
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"Nº de GPUs disponíveis: {torch.cuda.device_count()}")
        print(f"Dispositivo atual: {torch.cuda.current_device()}")
    else:
        print("Usando CPU")

    # Config
    cfg = Config()
    cuda = torch.cuda.is_available()

    # Transforms
    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    # Datasets e Dataloaders
    raw_train_dataset = ImageFolder(cfg.train_dataset)
    raw_test_dataset = ImageFolder(cfg.test_dataset)

    train_dataset = OmniglotTrain(raw_train_dataset, transform=data_transforms)
    test_dataset = OmniglotTest(raw_test_dataset, transform=transforms.ToTensor(), times=cfg.times, way=cfg.way)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


    # Modelo
    model = Siamese()
    if cuda:
        model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Treinamento
    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)

    train_loss = []
    loss_val = 0
    batch_ids = []
    losses = []

    model.train()
    train_iter = iter(train_loader)
    for batch_id in range(1, cfg.max_iter + 1):
        try:
            img1, img2, label = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            img1, img2, label = next(train_iter)

        start = time.time()

        
        img1, img2, label = img1.to(cfg.DEVICE), img2.to(cfg.DEVICE), label.to(cfg.DEVICE)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()

        if batch_id % cfg.show_every == 0:
            avg_loss = loss_val / cfg.show_every
            print(f"[{batch_id}]\tloss: {loss_val / cfg.show_every:.5f}\tTime: {(time.time() - start) * cfg.show_every:.2f}s")
            batch_ids.append(batch_id)
            losses.append(loss_val / cfg.show_every)
            loss_val = 0

        if batch_id % cfg.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.model_dir, f"model-batch-{batch_id}.pth"))

        if batch_id % cfg.test_every == 0:
            print("\nAvaliação One-Shot - Treino:")
            acc_train, prec_train, rec_train, f1_train = evaluate_oneshot(model, train_dataset, cfg.way, cfg.times, cfg.DEVICE)


            print("Avaliação One-Shot - Teste:")
            acc_test, prec_test, rec_test, f1_test = evaluate_oneshot(model, test_dataset, cfg.way, cfg.times, cfg.DEVICE)

            dados_epoca = {
                "iteracao": batch_id,
                "loss": round(avg_loss, 5),
                "train_accuracy": round(acc_train * 100, 2),
                "train_precision": round(prec_train, 4),
                "train_recall": round(rec_train, 4),
                "train_f1": round(f1_train, 4),
                "test_accuracy": round(acc_test * 100, 2),
                "test_precision": round(prec_test, 4),
                "test_recall": round(rec_test, 4),
                "test_f1": round(f1_test, 4)
            }
            salvar_log_epoca_csv(cfg.log_csv, dados_epoca)
            train_loss.append(loss_val)

    # Salvar curva de perda
    with open(cfg.loss_pickle, 'wb') as f:
        pickle.dump(train_loss, f)

    # Avaliação final
    eval_acc, eval_prec, eval_rec, eval_f1 = evaluate_oneshot(model, test_dataset, cfg.way, cfg.times, cfg.DEVICE)

    # Salvar resultados finais
    resultados_finais = {
        "learning_rate": cfg.learning_rate,
        "optimizer": "Adam",
        "batch_size": cfg.batch_size,
        "epochs": cfg.max_iter * cfg.batch_size // len(raw_train_dataset.imgs),
        "n_way": cfg.way,
        "n_trials": cfg.times,
        "loss_function": "BCEWithLogitsLoss",
        "transforms": "RandomAffine(15), ToTensor",
        "accuracy": round(eval_acc * 100, 2),
        "precision": round(eval_prec, 4),
        "recall": round(eval_rec, 4),
        "f1_score": round(eval_f1, 4)
    }
    salvar_resultados_csv(cfg.resultados_csv, resultados_finais)
    print("\nTreinamento finalizado.")

if __name__ == "__main__":
    main()
