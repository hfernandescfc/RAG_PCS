import os
import json
from collections import Counter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔍 DIAGNÓSTICO DO BANCO VETORIAL\n")
print("="*70)

db_path = "./chroma_db_multiplos"

if not os.path.exists(db_path):
    print("❌ Banco não encontrado!")
    exit()

# Carregar embeddings
print("Carregando banco...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db = Chroma(persist_directory=db_path, embedding_function=embeddings)
print("✅ Banco carregado\n")

# Obter todos os dados
all_data = db.get()

total_chunks = len(all_data['ids'])
print(f"📦 Total de chunks no banco: {total_chunks}\n")

# ============================================================================
# ANÁLISE 1: Fontes (sources)
# ============================================================================
print("="*70)
print("📄 ANÁLISE DE FONTES (campo 'source')")
print("="*70)

sources = [meta.get('source', 'SEM_SOURCE') for meta in all_data['metadatas']]
source_counts = Counter(sources)

print(f"\nTotal de fontes únicas: {len(source_counts)}")
print(f"\nTop 20 fontes por quantidade de chunks:")
print("-"*70)

for i, (source, count) in enumerate(source_counts.most_common(20), 1):
    print(f"{i:2d}. {source:50s} → {count:4d} chunks")

# Verificar se há problema
if len(source_counts) == 1:
    print("\n⚠️  PROBLEMA DETECTADO:")
    print("   Apenas 1 fonte encontrada! Todos os chunks têm o mesmo 'source'.")
    print("   Possíveis causas:")
    print("   1. Metadados foram sobrescritos durante processamento")
    print("   2. Bug no script de processamento")

# ============================================================================
# ANÁLISE 2: Campos de metadados disponíveis
# ============================================================================
print("\n" + "="*70)
print("🔑 CAMPOS DE METADADOS DISPONÍVEIS")
print("="*70)

# Pegar sample de metadados
sample_metadata = all_data['metadatas'][:5]

print("\nExemplos de metadados (primeiros 5 chunks):")
print("-"*70)

for i, meta in enumerate(sample_metadata, 1):
    print(f"\nChunk {i}:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

# Todos os campos únicos
all_keys = set()
for meta in all_data['metadatas']:
    all_keys.update(meta.keys())

print(f"\n\nCampos únicos encontrados: {sorted(all_keys)}")

# ============================================================================
# ANÁLISE 3: Verificar file_path (se existir)
# ============================================================================
print("\n" + "="*70)
print("📁 ANÁLISE DE CAMINHOS DE ARQUIVO")
print("="*70)

file_paths = [meta.get('file_path', None) for meta in all_data['metadatas']]
file_paths_unique = set(fp for fp in file_paths if fp)

if file_paths_unique:
    print(f"\nTotal de arquivos únicos (file_path): {len(file_paths_unique)}")
    print("\nArquivos processados:")
    for i, fp in enumerate(sorted(file_paths_unique), 1):
        basename = os.path.basename(fp) if fp else 'N/A'
        chunks_count = file_paths.count(fp)
        print(f"{i:2d}. {basename:50s} → {chunks_count:4d} chunks")
else:
    print("\n⚠️  Campo 'file_path' não encontrado nos metadados")

# ============================================================================
# ANÁLISE 4: Comparar com arquivos_processados.json
# ============================================================================
print("\n" + "="*70)
print("📋 COMPARAÇÃO COM REGISTRO DE PROCESSAMENTO")
print("="*70)

registro_path = "./arquivos_processados.json"

if os.path.exists(registro_path):
    with open(registro_path, 'r', encoding='utf-8') as f:
        registro = json.load(f)
    
    print(f"\nArquivos no registro: {len(registro)}")
    print(f"Fontes únicas no banco: {len(source_counts)}")
    
    if len(registro) != len(source_counts):
        print("\n⚠️  DISCREPÂNCIA DETECTADA!")
        print(f"   Registro diz: {len(registro)} arquivos")
        print(f"   Banco tem: {len(source_counts)} fontes únicas")
        print("\n   Isso indica que os metadados não foram salvos corretamente.")
    
    print("\n\nArquivos no registro:")
    for i, (hash_key, info) in enumerate(registro.items(), 1):
        nome = info.get('nome', 'N/A')
        paginas = info.get('paginas', '?')
        metodo = info.get('metodo', '?')
        print(f"{i:2d}. {nome:50s} ({paginas} pág, {metodo})")
    
    # Verificar quais estão faltando
    nomes_registro = {info.get('nome') for info in registro.values()}
    nomes_banco = set(source_counts.keys())
    
    faltando_no_banco = nomes_registro - nomes_banco
    extras_no_banco = nomes_banco - nomes_registro
    
    if faltando_no_banco:
        print(f"\n❌ Arquivos no registro mas NÃO no banco: {len(faltando_no_banco)}")
        for nome in sorted(faltando_no_banco)[:10]:
            print(f"   • {nome}")
    
    if extras_no_banco:
        print(f"\n⚠️  Fontes no banco mas NÃO no registro: {len(extras_no_banco)}")
        for nome in sorted(extras_no_banco)[:10]:
            print(f"   • {nome}")
else:
    print("\n⚠️  Arquivo arquivos_processados.json não encontrado")

# ============================================================================
# RECOMENDAÇÕES
# ============================================================================
print("\n" + "="*70)
print("💡 DIAGNÓSTICO E RECOMENDAÇÕES")
print("="*70)

if len(source_counts) == 1 and total_chunks > 100:
    print("\n🔴 PROBLEMA CRÍTICO CONFIRMADO:")
    print("   Todos os chunks têm a mesma fonte!")
    print("\n✅ SOLUÇÃO:")
    print("   1. Reprocessar documentos com script corrigido")
    print("   2. Deletar banco atual: rm -rf chroma_db_multiplos")
    print("   3. Executar script de processamento atualizado")
    
elif len(source_counts) < len(registro) * 0.5:
    print("\n🟡 PROBLEMA PARCIAL:")
    print("   Muitos documentos estão com metadados incorretos")
    print("\n✅ SOLUÇÃO:")
    print("   Reprocessar incrementalmente os arquivos problemáticos")

else:
    print("\n🟢 BANCO PARECE OK!")
    print(f"   {len(source_counts)} fontes únicas encontradas")
    print(f"   {total_chunks} chunks totais")

print("\n" + "="*70)