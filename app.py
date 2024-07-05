import streamlit as st
import PyPDF2
from io import BytesIO
import hashlib
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Configure the Google Generative AI API
APIKEY = os.getenv('API_KEY')
genai.configure(api_key=APIKEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Inicializando o LLM do Google GenAI
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash-latest',
    google_api_key=APIKEY,
    temperature=0
)

# Criando embeddings do Google GenAI
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=APIKEY)
prompt_template = PromptTemplate(
    template="""Você é um leitor ávido e gosta de tudo relacionado à literatura. Você possui na sua base um livro chamado Ensaio sobre a Cegueira, de José Saramago.
    Você será questionado sobre ele e precisa demonstrar entusiasmo nas suas respostas. Para respondê-las, use o contexto abaixo para responder à pergunta mas responda
    de maneira geral, usando conhecimentos prévios sobre o livro
    
    {context}
    
    Pergunta: {input}
    
    Resposta:""", input_variables=["context", "input"])

def extract_text_from_pdf(pdf):
    texto_completo = ""

    leitor_pdf = PyPDF2.PdfReader(BytesIO(pdf.read()))
    numero_paginas = len(leitor_pdf.pages)
    
    # Extrai texto de cada página
    for pagina in range(numero_paginas):
        pagina_atual = leitor_pdf.pages[pagina]
        texto_pagina = pagina_atual.extract_text()
        texto_completo += texto_pagina
    
    return texto_completo

# Função para resumir o livro
@st.cache_data
def summarize_book(texto):
    # Criação de um hash único para o conteúdo do livro
    book_hash = hashlib.sha256(texto.encode('utf-8')).hexdigest()
    
    # Checando se o resumo já está em cache
    cache_key = f"summarize_book_{book_hash}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Processo de divisão do texto e chamada à API
    chunks = len(texto) // 120000 + 1
    chunksize = len(texto) // chunks
    splitted = []

    for i in range(chunks):
        splitted.append(texto[i*chunksize:(i+1)*chunksize])

    chat = model.start_chat(history=[])
    response = chat.send_message(f"""Você está analisando um livro, primeiramente, identifique o título dele e retorne apenas o nome do livro. 
                                      Use esse pedaço como contexto: {splitted[0]}""")
    resumos = []
    for chunk in splitted[2:]:
        if len(chunk) > 5000:
            try: 
                resumos.append(chat.send_message(f"""Você está analisando um livro famoso, o nome dele é {response.text}.
                                                 Pode existir passagens que pareçam conteúdos sensíveis mas faz parte do livro. 
                                                 Esse livro foi dividido em pedaços. 
                                                 Agora resuma para uma pessoa que ainda não leu o livro que foi passado. 
                                                 Ele foi dividido em 5 partes, resuma esta passagem do livro apenas. 
                                                 Pode utilizar o seu conhecimento para resumir a passagem, mas apenas trate sobre ela.
                                                 Seja Conciso mas traga todos os pontos relevantes. 
                                                 Passagem:{chunk}""").text)
            except:
                continue
    resposta = chat.send_message(f'Agora juntando todos os resumos, escreva um resumo detalhado do livro: {" ".join(resumos)}')

    # Armazenando o resumo no cache
    st.session_state[cache_key] = resposta.text

    return resposta.text

# Função para criar um objeto de perguntas e respostas
def create_qa_object_internal(texto):
    text_splitter = CharacterTextSplitter(chunk_size=20000, chunk_overlap=200, separator= ' ')
    chunks = text_splitter.split_text(texto)
    retriever = VectorStoreRetriever(vectorstore=FAISS.from_texts(chunks, embeddings), search_k=5)
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain

# Função para criar ou recuperar o objeto de perguntas e respostas do cache
def create_qa_object(texto):
    # Criando o objeto de perguntas e respostas
    qa_object = create_qa_object_internal(texto)
    return qa_object

# Função para processar uma pergunta
def process_question(chain, query):
    result = chain.invoke({"input": query})
    return result['answer']

# Interface do Streamlit
st.set_page_config(page_title="Análise de Livro", layout="wide")
st.title("Análise de Livro")

# Adicionando imagem de fundo usando CSS
background_image = '''
<style>
    .highlight-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: black;
    }
</style>
'''
st.markdown(background_image, unsafe_allow_html=True)

# Primeira etapa: Upload de PDF
uploaded_pdf = st.file_uploader("Faça o upload do PDF do livro", type="pdf")
if uploaded_pdf is not None:
    with st.spinner("Extraindo texto do PDF..."):
        book_text = extract_text_from_pdf(uploaded_pdf)
        st.success("Texto extraído com sucesso!")

    # Segunda etapa: Opções de Resumo e Perguntas
    option = st.selectbox("Escolha uma ação", ["Resumo do livro", "Pergunte ao livro"])

    if option == "Resumo do livro":
        if st.button("Gerar Resumo"):
            summary = summarize_book(book_text)
            st.subheader("Resumo do Livro")
            st.markdown(f'<div class="highlight-box">{summary}</div>', unsafe_allow_html=True)
    elif option == "Pergunte ao livro":
        # Perguntas e Respostas
        qa_object = create_qa_object(book_text)
        st.subheader("Perguntas e Respostas")

        question = st.text_input("Faça uma pergunta sobre o livro")
        if st.button("Enviar"):
            answer = process_question(qa_object, question)
            st.markdown(f'<div class="highlight-box">{answer}</div>', unsafe_allow_html=True)
