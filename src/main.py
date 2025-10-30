import os
import sys
import glob
import hashlib
import json
import gc

# AVISO: Este script 'main.py' está obsoleto.
# Use os entrypoints dedicados:
# - Ingestão: python src/rag_processar_corrigido.py
# - CLI:      python src/query_cli.py
# - Web App:  streamlit run src/rag_pipeline/app.py

# Verificar e configurar Tesseract
def verificar_tesseract():
    caminhos_possiveis = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    for caminho in caminhos_possiveis:
        if os.path.exists(caminho):
            return caminho
    return None

def calcular_hash_arquivo(filepath):
    """Calcula hash MD5 de um arquivo"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def carregar_registro_processados(registro_path="./arquivos_processados.json"):
    """Carrega registro de arquivos já processados"""
    if os.path.exists(registro_path):
        with open(registro_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def salvar_registro_processados(registro, registro_path="./arquivos_processados.json"):
    """Salva registro de arquivos processados"""
    with open(registro_path, 'w', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

def verificar_pdf_tem_texto(pdf_path):
    """Verifica se PDF tem texto extraível"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        
        for i, page in enumerate(reader.pages[:3]):
            text = page.extract_text().strip()
            if len(text) > 100:
                return True
        return False
    except:
        return False

def processar_pdf_com_texto(pdf_path, nome_arquivo):
    """Extrai texto direto do PDF"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.docstore.document import Document
    
    print("   📝 Extração direta...")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        if documents:
            for i, doc in enumerate(documents):
                doc.metadata['source'] = nome_arquivo
                doc.metadata['page'] = i + 1
                doc.metadata['file_path'] = pdf_path
                doc.metadata['metodo'] = 'extração_direta'
            
            return documents, 'extração_direta'
    except Exception as e:
        print(f"   ⚠️  Erro: {e}")
    
    return None, None

def processar_pdf_com_ocr(pdf_path, nome_arquivo, pytesseract, convert_from_path):
    """Extrai texto com OCR - OTIMIZADO PARA MEMÓRIA"""
    from langchain.docstore.document import Document
    
    print("   🖼️  OCR (processando página por página)...")
    
    try:
        # Descobrir número de páginas SEM carregar tudo
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        total_paginas = len(reader.pages)
        print(f"   Total: {total_paginas} páginas")
        
        documents = []
        
        # PROCESSAR PÁGINA POR PÁGINA (economia de memória)
        for page_num in range(1, total_paginas + 1):
            try:
                percent = (page_num / total_paginas) * 100
                print(f"      [{page_num}/{total_paginas}] {percent:.0f}%", end="\r")
                
                # Converter apenas UMA página por vez
                images = convert_from_path(
                    pdf_path,
                    dpi=200,  # Reduzido de 300 para economizar memória
                    first_page=page_num,
                    last_page=page_num,
                    fmt='jpeg'
                )
                
                if images:
                    image = images[0]
                    
                    # OCR
                    text = pytesseract.image_to_string(
                        image,
                        lang='por',
                        config='--psm 3 --oem 1'
                    )
                    
                    if text.strip():
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                "source": nome_arquivo,
                                "page": page_num,
                                "file_path": pdf_path,
                                "metodo": "ocr"
                            }
                        )
                        documents.append(doc)
                    
                    # LIBERAR MEMÓRIA
                    del images
                    del image
                    gc.collect()
                    
            except Exception as e:
                print(f"\n      ⚠️  Erro página {page_num}: {e}")
        
        print(f"\n   ✅ {len(documents)} páginas extraídas")
        return documents, 'ocr'
        
    except Exception as e:
        print(f"   ❌ Erro no OCR: {e}")
        return None, None

def processar_e_salvar_arquivo(arq_info, registro, embeddings, db_path, ocr_disponivel, 
                               pytesseract=None, convert_from_path=None):
    """Processa UM arquivo e salva IMEDIATAMENTE no banco"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from datetime import datetime
    
    pdf_path = arq_info['path']
    nome_arquivo = arq_info['nome']
    hash_arquivo = arq_info['hash']
    
    print(f"\n{'='*70}")
    print(f"📄 {nome_arquivo}")
    print('='*70)
    
    # Tentar extração direta primeiro
    print("1️⃣  Verificando se tem texto extraível...")
    tem_texto = verificar_pdf_tem_texto(pdf_path)
    
    documents = None
    metodo = None
    
    if tem_texto:
        print("   ✅ Tem texto!")
        documents, metodo = processar_pdf_com_texto(pdf_path, nome_arquivo)
    
    # Se falhou ou não tem texto, tentar OCR
    if not documents and ocr_disponivel:
        print("2️⃣  Usando OCR...")
        documents, metodo = processar_pdf_com_ocr(pdf_path, nome_arquivo, 
                                                  pytesseract, convert_from_path)
    
    if not documents:
        print("   ❌ Falhou ao processar")
        return False
    
    # Dividir em chunks
    print("   ✂️  Criando chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    docs = [doc for doc in docs if doc.page_content.strip()]
    print(f"   ✅ {len(docs)} chunks")
    
    # SALVAR NO BANCO IMEDIATAMENTE
    print("   💾 Salvando no banco...")
    
    try:
        if os.path.exists(db_path):
            # Adicionar ao banco existente
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            db.add_documents(docs)
        else:
            # Criar novo banco
            db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
        
        print("   ✅ Salvo!")
        
        # REGISTRAR SUCESSO
        registro[hash_arquivo] = {
            'nome': nome_arquivo,
            'caminho': pdf_path,
            'metodo': metodo,
            'paginas': len(documents),
            'chunks': len(docs),
            'data_processo': datetime.now().isoformat()
        }
        salvar_registro_processados(registro)
        
        # LIBERAR MEMÓRIA
        del documents
        del docs
        del db
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro ao salvar: {e}")
        return False

print("🔧 Sistema RAG Otimizado para Baixa Memória (8GB)\n")

# Verificar Tesseract
tesseract_path = verificar_tesseract()

if not tesseract_path:
    print("⚠️  Tesseract não encontrado - OCR desabilitado")
    ocr_disponivel = False
else:
    print(f"✅ Tesseract: {tesseract_path}")
    ocr_disponivel = True

# Importar bibliotecas
try:
    from langchain_community.document_loaders import PyPDFLoader
    from PyPDF2 import PdfReader
    import requests
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
except ImportError as e:
    print(f"❌ Erro: {e}")
    print("Instale: pip install langchain-community PyPDF2 requests")
    sys.exit(1)

if ocr_disponivel:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except ImportError:
        print("⚠️  pdf2image/pytesseract não disponíveis")
        ocr_disponivel = False
        pytesseract = None
        convert_from_path = None

print()

# ============================================================================
# CONFIGURAÇÃO: Adicione seus PDFs aqui
# ============================================================================

pdf_folder = r"C:\Users\compesa\Desktop\ppp_RAG"

# A busca agora é recursiva para encontrar arquivos .pdf e .PDF em todas as subpastas
# O '**' indica que a busca deve incluir todos os subdiretórios
pdf_files = glob.glob(os.path.join(pdf_folder, "**", "*.pdf"), recursive=True)
pdf_files += glob.glob(os.path.join(pdf_folder, "**", "*.PDF"), recursive=True)
# ============================================================================

pdf_files = [f for f in pdf_files if os.path.exists(f)]

if not pdf_files:
    print("❌ Nenhum PDF encontrado!")
    sys.exit(1)

# Carregar registro
registro = carregar_registro_processados()
print(f"📋 Arquivos já processados: {len(registro)}\n")

# Verificar o que processar
arquivos_para_processar = []
hashes_vistos = set()

print("🔍 Verificando arquivos...\n")

for pdf_path in pdf_files:
    nome_arquivo = os.path.basename(pdf_path)
    hash_arquivo = calcular_hash_arquivo(pdf_path)
    
    if hash_arquivo in registro:
        info = registro[hash_arquivo]
        print(f"⏭️  {nome_arquivo} (já processado em {info.get('metodo', '?')})")
        continue
    
    if hash_arquivo in hashes_vistos:
        print(f"⚠️  {nome_arquivo} (duplicado)")
        continue
    
    hashes_vistos.add(hash_arquivo)
    size_mb = os.path.getsize(pdf_path) / (1024*1024)
    print(f"✅ {nome_arquivo} ({size_mb:.2f} MB)")
    
    arquivos_para_processar.append({
        'path': pdf_path,
        'nome': nome_arquivo,
        'hash': hash_arquivo
    })

print(f"\n{'='*70}")
print(f"📊 Para processar: {len(arquivos_para_processar)} arquivos")
print('='*70)

if not arquivos_para_processar:
    print("\n✅ Todos já processados! Carregando banco...\n")
else:
    # Criar embeddings UMA VEZ
    print("\n🧠 Inicializando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}  # Batch menor
    )
    print("✅ Embeddings prontos")
    
    db_path = "./chroma_db_multiplos"
    
    # PROCESSAR ARQUIVO POR ARQUIVO
    sucesso = 0
    falhas = 0
    
    for i, arq_info in enumerate(arquivos_para_processar, 1):
        print(f"\n[{i}/{len(arquivos_para_processar)}]")
        
        try:
            if processar_e_salvar_arquivo(
                arq_info, registro, embeddings, db_path, 
                ocr_disponivel, pytesseract, convert_from_path
            ):
                sucesso += 1
            else:
                falhas += 1
        except Exception as e:
            print(f"   ❌ ERRO CRÍTICO: {e}")
            falhas += 1
        
        # Forçar limpeza de memória
        gc.collect()
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESSAMENTO FINALIZADO")
    print('='*70)
    print(f"✅ Sucesso: {sucesso} arquivos")
    print(f"❌ Falhas: {falhas} arquivos")

# Carregar banco para consultas
print("\n💾 Carregando banco vetorial...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

db = Chroma(persist_directory="./chroma_db_multiplos", embedding_function=embeddings)
print("✅ Banco carregado")

# Verificar Ollama
print("\n🤖 Verificando Ollama...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code != 200:
        raise Exception()
    models = [m['name'] for m in response.json().get('models', [])]
    print(f"✅ Ollama OK! Modelos: {models}")
except:
    print("❌ Ollama não está rodando!")
    print("Execute: ollama serve")
    sys.exit(1)

llm = Ollama(model="llama3.2:3b", temperature=0, num_ctx=2048)

retriever = db.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\n" + "="*70)
print("✅ SISTEMA PRONTO!")
print("="*70)
print(f"\n📚 Base: {len(registro)} arquivos processados")

# Interface
while True:
    print("\n❓ Pergunta (ou 'sair'):")
    query = input(">>> ").strip()
    
    if query.lower() in ['sair', 'exit', 'quit', '']:
        break
    
    print("\n🔍 Buscando...\n")
    
    try:
        result = qa.invoke({"query": query})
        
        print("📝 Resposta:")
        print("-" * 70)
        print(result['result'])
        print("-" * 70)
        
        print("\n📚 Fontes:")
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', '?')
            page = doc.metadata.get('page', '?')
            metodo = doc.metadata.get('metodo', '?')
            print(f"   [{i}] {source} (pág. {page}) [{metodo}]")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

print("\n✅ Encerrado!")
