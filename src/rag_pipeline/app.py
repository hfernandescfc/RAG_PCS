# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import requests
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuração da página
st.set_page_config(
    page_title="Sistema RAG - Consulta de Documentos",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .fonte-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<h1 class="main-header">📚 Sistema de Consulta de Documentos - Programa Cidade Saneada</h1>', unsafe_allow_html=True)

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Caminho do banco
    db_path = st.text_input(
        "Caminho do Banco Vetorial",
        value="./chroma_db_multiplos",
        help="Pasta onde está o banco vetorial"
    )
    
    # Modelo de embeddings
    embedding_model = st.selectbox(
        "Modelo de Embeddings",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "neuralmind/bert-base-portuguese-cased",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ],
        help="Use o mesmo modelo usado no processamento"
    )
    
    # Configurações de busca
    st.subheader("🔍 Parâmetros de Busca")
    
    num_docs = st.slider(
        "Número de documentos",
        min_value=3,
        max_value=10,
        value=6,
        help="Quantos trechos buscar nos documentos"
    )
    
    search_type = st.radio(
        "Tipo de busca",
        ["MMR (Diversidade)", "Similaridade"],
        help="MMR evita resultados repetitivos"
    )
    
    # Configurações do LLM
    st.subheader("🤖 Modelo de IA")
    
    ollama_model = st.selectbox(
        "Modelo Ollama",
        ["llama3.2:3b", "llama3.1:8b", "mistral", "phi3:mini"],
        help="Modelo para gerar respostas"
    )
    
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = mais preciso, 1 = mais criativo"
    )
    
    # Informações do sistema
    st.divider()
    st.subheader("ℹ️ Informações")
    
    if os.path.exists("./arquivos_processados.json"):
        try:
            with open("./arquivos_processados.json", 'r', encoding='utf-8') as f:
                registro = json.load(f)
            
            st.metric("Documentos processados", len(registro))
            
            with st.expander("Ver documentos"):
                for hash_key, info in registro.items():
                    st.text(f"📄 {info.get('nome', 'N/A')}")
                    st.caption(f"   Páginas: {info.get('paginas', '?')} | Método: {info.get('metodo', '?')}")
        except UnicodeDecodeError:
            st.warning("⚠️ Erro ao ler arquivos_processados.json. Verifique a codificação do arquivo.")

# Função de inicialização (cache para não recarregar sempre)
@st.cache_resource
def inicializar_sistema(db_path, embedding_model, ollama_model, temperature):
    """Inicializa o sistema RAG"""
    
    # Verificar se banco existe
    if not os.path.exists(db_path):
        st.error(f"❌ Banco vetorial não encontrado em: {db_path}")
        st.stop()
    
    # Verificar Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            raise Exception()
    except:
        st.error("❌ Ollama não está rodando! Execute: `ollama serve`")
        st.stop()
    
    # Carregar embeddings
    with st.spinner("🧠 Carregando modelo de embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    # Carregar banco
    with st.spinner("💾 Carregando banco vetorial..."):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Configurar LLM
    llm = Ollama(
        model=ollama_model,
        temperature=temperature,
        num_ctx=4096,
        num_predict=1024
    )
    
    # Prompt customizado
    template_prompt = """Você é um assistente especializado em análise de documentos contratuais e técnicos.

Use APENAS as informações dos documentos fornecidos abaixo para responder à pergunta.

REGRAS IMPORTANTES:
1. Se a informação NÃO estiver nos documentos, diga claramente: "Não encontrei essa informação nos documentos fornecidos"
2. Cite especificamente de qual documento e página você tirou a informação
3. Se houver informações conflitantes, mencione ambas e suas fontes
4. Seja preciso e objetivo
5. Use linguagem clara e profissional

DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA DETALHADA:"""

    PROMPT = PromptTemplate(
        template=template_prompt,
        input_variables=["context", "question"]
    )
    
    return db, llm, PROMPT, embeddings

# Inicializar sistema
try:
    db, llm, PROMPT, embeddings = inicializar_sistema(
        db_path, 
        embedding_model, 
        ollama_model, 
        temperature
    )
    
    st.success("✅ Sistema carregado com sucesso!")
    
except Exception as e:
    st.error(f"❌ Erro ao inicializar: {e}")
    st.stop()

# Tabs principais
tab1, tab2, tab3 = st.tabs(["💬 Consulta", "🔍 Busca Literal", "📊 Estatísticas"])

# ============================================================================
# TAB 1: CONSULTA PRINCIPAL
# ============================================================================
with tab1:
    st.header("Faça sua pergunta sobre os documentos")
    
    # Sugestões de perguntas
    with st.expander("💡 Exemplos de perguntas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Perguntas diretas:**
            - Qual o prazo do contrato?
            - Quem são as partes contratantes?
            - Qual o valor total do contrato?
            - Quais são as obrigações da concessionária?
            """)
        
        with col2:
            st.markdown("""
            **Perguntas específicas:**
            - Há previsão de multas?
            - Qual o prazo para universalização?
            - A quem cabe a elaboração dos projetos?
            - Quais são as garantias contratuais?
            """)
    
    # Campo de pergunta
    query = st.text_area(
        "Sua pergunta:",
        height=100,
        placeholder="Digite aqui sua pergunta sobre os documentos..."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        buscar_btn = st.button("🔍 Buscar Resposta", type="primary", use_container_width=True)
    
    with col2:
        mostrar_preview = st.checkbox("Mostrar preview", value=False)
    
    with col3:
        mostrar_fontes = st.checkbox("Mostrar fontes", value=True)
    
    if buscar_btn and query:
        # Configurar retriever
        search_type_str = "mmr" if "MMR" in search_type else "similarity"
        
        if search_type_str == "mmr":
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": num_docs,
                    "fetch_k": num_docs * 3,
                    "lambda_mult": 0.5
                }
            )
        else:
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": num_docs}
            )
        
        # Preview dos documentos encontrados
        if mostrar_preview:
            with st.expander("📄 Preview dos documentos encontrados"):
                docs_preview = retriever.invoke(query)
                
                for i, doc in enumerate(docs_preview, 1):
                    st.markdown(f"**[{i}] {doc.metadata.get('source', '?')} (pág. {doc.metadata.get('page', '?')})**")
                    st.caption(doc.page_content[:200] + "...")
                    st.divider()
        
        # Criar chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Buscar resposta
        with st.spinner("🤖 Gerando resposta... (isso pode levar 10-30 segundos)"):
            try:
                result = qa_chain.invoke({"query": query})
                
                # Mostrar resposta
                st.markdown("### 📝 Resposta:")
                st.info(result['result'])
                
                # Mostrar fontes
                if mostrar_fontes and result.get('source_documents'):
                    st.markdown("### 📚 Fontes utilizadas:")
                    
                    for i, doc in enumerate(result['source_documents'], 1):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**[{i}] {doc.metadata.get('source', 'Desconhecido')}**")
                            
                            with col2:
                                st.caption(f"Página {doc.metadata.get('page', '?')}")
                            
                            with col3:
                                metodo = doc.metadata.get('metodo', '?')
                                badge_color = "🟢" if metodo == "extração_direta" else "🔵"
                                st.caption(f"{badge_color} {metodo}")
                            
                            with st.expander("Ver trecho"):
                                st.text(doc.page_content[:500] + "...")
                
                # Salvar no histórico
                if 'historico' not in st.session_state:
                    st.session_state.historico = []
                
                st.session_state.historico.append({
                    'timestamp': datetime.now().isoformat(),
                    'pergunta': query,
                    'resposta': result['result'],
                    'num_fontes': len(result.get('source_documents', []))
                })
                
            except Exception as e:
                st.error(f"❌ Erro ao gerar resposta: {e}")

# ============================================================================
# TAB 2: BUSCA LITERAL
# ============================================================================
with tab2:
    st.header("🔎 Busca literal nos documentos")
    st.caption("Busca por termos exatos no texto dos documentos")
    
    termo_busca = st.text_input("Digite o termo para buscar:", placeholder="Ex: universalização")
    
    if st.button("🔍 Buscar Termo", use_container_width=True):
        if termo_busca:
            with st.spinner(f"Buscando '{termo_busca}'..."):
                all_docs = db.get()
                encontrados = []
                
                for doc_id, text, metadata in zip(
                    all_docs['ids'], 
                    all_docs['documents'], 
                    all_docs['metadatas']
                ):
                    if termo_busca.lower() in text.lower():
                        encontrados.append((text, metadata))
                
                if encontrados:
                    st.success(f"✅ Encontrado em {len(encontrados)} trechos")
                    
                    for i, (text, meta) in enumerate(encontrados[:20], 1):  # Mostrar até 20
                        source = meta.get('source', '?')
                        page = meta.get('page', '?')
                        
                        with st.expander(f"[{i}] {source} - Página {page}"):
                            # Destacar termo
                            idx = text.lower().find(termo_busca.lower())
                            preview_start = max(0, idx - 100)
                            preview_end = min(len(text), idx + 200)
                            preview = text[preview_start:preview_end]
                            
                            st.text(f"...{preview}...")
                    
                    if len(encontrados) > 20:
                        st.info(f"Mostrando 20 de {len(encontrados)} resultados")
                else:
                    st.warning(f"❌ Termo '{termo_busca}' não encontrado")

# ============================================================================
# TAB 3: ESTATÍSTICAS
# ============================================================================
with tab3:
    st.header("📊 Estatísticas do Sistema")
    
    # Informações do banco
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists("./arquivos_processados.json"):
            try:
                with open("./arquivos_processados.json", 'r', encoding='utf-8') as f:
                    registro = json.load(f)
                st.metric("📄 Documentos", len(registro))
            except:
                st.metric("📄 Documentos", "Erro")
    
    with col2:
        all_docs = db.get()
        st.metric("📦 Chunks", len(all_docs['ids']))
    
    with col3:
        if 'historico' in st.session_state:
            st.metric("💬 Consultas", len(st.session_state.historico))
    
    # Histórico de consultas
    if 'historico' in st.session_state and st.session_state.historico:
        st.subheader("📝 Histórico de Consultas")
        
        for i, item in enumerate(reversed(st.session_state.historico[-10:]), 1):
            with st.expander(f"{i}. {item['pergunta'][:80]}..."):
                st.caption(f"🕐 {item['timestamp']}")
                st.markdown(f"**Pergunta:** {item['pergunta']}")
                st.markdown(f"**Resposta:** {item['resposta']}")
                st.caption(f"📚 Fontes consultadas: {item['num_fontes']}")
        
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.historico = []
            st.rerun()
    else:
        st.info("Nenhuma consulta realizada ainda")
    
    # Informações técnicas
    with st.expander("🔧 Informações Técnicas"):
        st.json({
            "Banco Vetorial": db_path,
            "Modelo Embeddings": embedding_model,
            "Modelo LLM": ollama_model,
            "Temperatura": temperature,
            "Tipo de Busca": search_type,
            "Documentos por Busca": num_docs
        })