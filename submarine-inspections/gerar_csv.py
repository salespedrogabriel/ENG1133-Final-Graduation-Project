import json
import csv
import os

# Caminho onde estão os seus arquivos JSON
json_dir = r"D:\PUC-Rio\2025.1\TCC\mantis\april-2024-dutos"

# Caminho onde você quer salvar o CSV
output_csv_path = r"D:\PUC-Rio\2025.1\TCC\mantis\saida.csv"

# Campos do CSV
csv_fields = ['filename', 'width', 'height', 'ROV', 'Duto', 'Cruzamento', 'Sucata_bb', 'Flange', 'Flutuador', 'Equipamento']

# Lista de linhas do CSV
csv_rows = []

# Processar cada arquivo JSON
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        slot = data['item']['slots'][0]
        annotations = data.get('annotations', [])

        # Inicializa a linha com valores padrões
        row = {
            'filename': data['item']['name'],
            'width': slot['width'],
            'height': slot['height'],
            'ROV': 0,
            'Duto': 0,
            'Cruzamento': 0,
            'Sucata_bb': 0,
            'Flange': 0,
            'Flutuador': 0,
            'Equipamento': 0
        }

        # Verificar presença de cada categoria
        for ann in annotations:
            name = ann.get('name', '')
            attrs = ann.get('attributes', [])
            
            if name in row:
                row[name] = 1
            for attr in attrs:
                if attr in row:
                    row[attr] = 1

        csv_rows.append(row)

# Exportar para CSV
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"CSV gerado com sucesso em: {output_csv_path}")