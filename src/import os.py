import os
import shutil

# Caminho da pasta raiz
root_dir = r"C:\caminho\da\pasta"

# Lista para armazenar os arquivos PDF
pdf_files = []

# Percorrer todas as pastas e subpastas
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(".pdf"):
            old_path = os.path.join(dirpath, filename)  # Caminho atual do arquivo
            
            # Nome da pasta onde o PDF estava
            folder_name = os.path.basename(dirpath)
            
            # Novo nome: pasta_nome_arquivo.pdf
            new_filename = f"{folder_name}_{filename}"
            new_path = os.path.join(dirpath, new_filename)
            
            # Se já existir, evitar sobrescrever
            counter = 1
            while os.path.exists(new_path):
                name, ext = os.path.splitext(new_filename)
                new_filename = f"{name}_{counter}{ext}"
                new_path = os.path.join(dirpath, new_filename)
                counter += 1
            
            # Renomear o arquivo
            os.rename(old_path, new_path)
            
            # Adicionar à lista
            pdf_files.append(new_path)

# Exibir todos os arquivos encontrados e renomeados
print("Arquivos PDF encontrados e renomeados:")
for pdf in pdf_files:
    print(pdf)
