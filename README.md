# Abordagem One-Shot Learning para Classificação de Imagens de Inspeções Submarinas

### Sobre o Projeto
Este projeto foi desenvolvido como parte do **Trabalho de Conclusão de Curso** em Engenharia de Computação.  
O objetivo é **classificar imagens submarinas** provenientes de inspeções realizadas por **ROVs (Remotely Operated Vehicles)** utilizando uma abordagem de **One-Shot Learning** com **redes siamesas**.

A proposta busca **reduzir o tempo de análise manual** das imagens de inspeções, um processo que pode levar até **40 horas por especialista**, automatizando parte da identificação de estruturas e objetos submarinos.

## Objetivos

- Implementar uma **arquitetura Siamese Neural Network (SNN)** para aprender similaridade entre imagens.  
- Adaptar a abordagem clássica do **dataset Omniglot** para um novo conjunto de dados de **inspeções submarinas reais**.  
- Avaliar o desempenho do modelo em tarefas **N-way One-Shot**.  
- Gerar métricas completas (acurácia, precisão, recall, F1-score e matriz de confusão).

## Abordagem Técnica

O modelo foi desenvolvido em **PyTorch**, utilizando as seguintes configurações:

- **Arquitetura:** Siamese CNN com 3 blocos convolucionais  
  e duas camadas totalmente conectada 
- **Função de perda:** `BCEWithLogitsLoss`  
- **Otimização:** `Adam (lr=0.0001)`  
- **Avaliação:** tarefas *N-way* (ex: 3-way, 20-way)  
- **Dataset customizado:** imagens reais de inspeções submarinas, divididas em classes:  
  - `Duto`   
  - `Equipamento`  
  - `Flange`  
  - `ROV`
  - `Manipulador`
  - `Objeto`

## Como Executar

### Pré-requisitos

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib

Instale as dependências com:

```
pip install -r requirements.txt
```

### Treinamento do Modelo

Execute o script principal

```
python train.py
```

Durante o treinamento, as métricas são salvas em arquivos .csv e os pesos .pth serão salvos na pasta model.

### Resultados Obtidos

Acurácia (Treino): 96.5%  
F1-Score (Treino): 0.963  
Acurácia (Teste): 76.5%  
F1-Score (Teste): 0.764  

### Trabalhos Futuros

* Testar arquiteturas menos complexas
* Explorar outros tipos de abordagens como a Few-Shot Learning
* Buscar uma maior variação intra-classe
* Aumentar o número de amostras porém sem comprometer no balanceamento do dataset
