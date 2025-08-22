import csv
import os

def salvar_log_epoca_csv(nome_arquivo, dados_epoca):
    file_exists = os.path.isfile(nome_arquivo)
    with open(nome_arquivo, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dados_epoca.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(dados_epoca)

def salvar_resultados_csv(nome_arquivo, dados):
    file_exists = os.path.isfile(nome_arquivo)
    with open(nome_arquivo, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dados.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(dados)
