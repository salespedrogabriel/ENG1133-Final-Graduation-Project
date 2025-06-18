import pandas as pd
import matplotlib.pyplot as plt

# Caminhos de entrada e saída
csv_path = r"C:\Users\pedrosales\Desktop\saida_adicional.csv"
output_graph_path = r"C:\Users\pedrosales\Desktop\atributos_unicos.png"

# Atributos binários do CSV
atributos = ['filename', 'width', 'height',
              'MarcacaoCircular', 'Corda', 'ROV', 
              'Anodo', 'ResiduoDeAnodo', 'CrescimentoVidaMarinha', 
              'Duto', 'Cruzamento', 'EndFitting',
              'Sucata_bb', 'Flange', 'Flutuador', 
              'Equipamento', 'Mangueira' ]

# Lê o CSV
df = pd.read_csv(csv_path)

# Soma quantos atributos estão ativos por linha
df['atributos_ativos'] = df[atributos].sum(axis=1)

# Inicializa o dicionário de contagem
contagem = {atributo: 0 for atributo in atributos}

# Loop por linha para aplicar regras personalizadas
for _, row in df.iterrows():
    ativos = [attr for attr in atributos if row[attr] == 1]

    if len(ativos) == 1:
        # Caso geral: apenas 1 atributo ativo
        contagem[ativos[0]] += 1

    elif sorted(ativos) == ['Cruzamento', 'Duto']:
        # Caso especial: somente 'Cruzamento' e 'Duto'
        contagem['Cruzamento'] += 1

# Ordena as contagens do maior para o menor
contagem = dict(sorted(contagem.items(), key=lambda x: x[1], reverse=True))

# Cria o gráfico
plt.figure(figsize=(10, 6))
bars = plt.bar(contagem.keys(), contagem.values(), color='teal')

# Adiciona os valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.2, str(int(height)), ha='center', va='bottom')

# Configurações do gráfico
plt.title('Distribuição de exemplos com atributos únicos (regra especial para Cruzamento)')
plt.xlabel('Atributo')
plt.ylabel('Quantidade de exemplos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 8000)  

# Ajuste de layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15)

# Salva o gráfico
plt.savefig(output_graph_path)
print(f"Gráfico salvo com sucesso em: {output_graph_path}")