# Imagem base com CUDA, CUDNN, e Python 3.9 já incluído
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Instala apenas os pacotes necessários para seu código
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas os arquivos necessários
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional) Crie diretório para o código e copie o código para ele
# COPY . /app
# WORKDIR /app

CMD ["bash"]
