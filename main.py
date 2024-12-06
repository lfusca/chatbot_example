import os
import json
import requests
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Carregar variáveis do arquivo .env
load_dotenv()

# Configuração da OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configurações da Evolution API
EVOLUTION_API_URL = 'sua_url'  # URL base
EVOLUTION_API_TOKEN = os.getenv('EVOLUTION_API_TOKEN')  # Seu token de autenticação

# Verificação das variáveis de ambiente
print(f"EVOLUTION_API_URL: {EVOLUTION_API_URL}")
print(f"EVOLUTION_API_TOKEN: {EVOLUTION_API_TOKEN}")

# Inicializar o aplicativo Flask
app = Flask(__name__)

INDEX_METADATA_FILE = 'index_metadata.json'

def obter_metadados_arquivos():
    metadados = {}
    if not os.path.exists('docs'):
        os.makedirs('docs')
    for filename in sorted(os.listdir('docs')):
        filepath = os.path.join('docs', filename)
        if os.path.isfile(filepath):
            metadados[filename] = int(os.path.getmtime(filepath))
    return metadados

def carregar_metadados_indexados():
    if os.path.exists(INDEX_METADATA_FILE):
        with open(INDEX_METADATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

def salvar_metadados_indexados(metadados):
    with open(INDEX_METADATA_FILE, 'w') as f:
        json.dump(metadados, f)

def verificar_se_arquivos_alteraram():
    metadados_atuais = obter_metadados_arquivos()
    metadados_indexados = carregar_metadados_indexados()
    arquivos_alterados = metadados_atuais != metadados_indexados
    print(f"Metadados atuais: {metadados_atuais}")
    print(f"Metadados indexados: {metadados_indexados}")
    print(f"Arquivos alterados? {arquivos_alterados}")
    return arquivos_alterados

def carregar_e_indexar_documentos():
    print("Carregando e indexando documentos...")
    docs = []
    for filename in os.listdir('docs'):
        filepath = os.path.join('docs', filename)
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath, encoding='utf-8')
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")

    metadados_atuais = obter_metadados_arquivos()
    salvar_metadados_indexados(metadados_atuais)

    return vectorstore

if os.path.exists("faiss_index") and not verificar_se_arquivos_alteraram():
    print("Carregando índice existente...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings)
else:
    vectorstore = carregar_e_indexar_documentos()

def recuperar_documentos(consulta, k=4):
    docs_relevantes = vectorstore.similarity_search(consulta, k=k)
    return docs_relevantes

def gerar_resposta(mensagem: str) -> str:
    try:
        docs_relevantes = recuperar_documentos(mensagem)
        contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])

        # Construir o prompt com o contexto
        prompt = f"""
        Você é um assistente que utiliza o contexto fornecido para responder de forma precisa.

        Contexto:
        {contexto}

        Pergunta:
        {mensagem}

        Resposta:
        """

        resposta = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        resposta_texto = resposta['choices'][0]['message']['content'].strip()
        print(f"Mensagem gerada pela IA: {resposta_texto}")
        return resposta_texto
    except Exception as e:
        print(f"Erro ao gerar resposta com OpenAI: {e}")
        return "Desculpe, não consegui processar sua mensagem agora."

def enviar_mensagem(numero: str, mensagem: str):
    numero_limpo = numero.strip()
    mensagem_limpa = mensagem.strip()

    if not numero_limpo or not mensagem_limpa:
        print("Erro: Número de telefone ou mensagem está vazio.")
        return None

    payload = {
        "number": numero_limpo,
        "text": mensagem_limpa
    }

    print(f"Payload enviado: {json.dumps(payload, ensure_ascii=False, indent=4)}")

    url = f"{EVOLUTION_API_URL}/message/sendText/FIAPTest" #mudar para seu endpoint

    headers = {
        "apikey": EVOLUTION_API_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Resposta da API: {response.status_code}, {response.text}")
        if response.status_code in [200, 201]:
            print(f"Status: Mensagem enviada para {numero_limpo}")
            return response.json()
        else:
            print(f"Status: Erro ao enviar mensagem para {numero_limpo}: {response.text}")
            return None
    except Exception as e:
        print(f"Erro na requisição: {e}")
        return None

@app.route('/webhook/whatsapp', methods=['POST'])
def webhook():
    try:
        data = request.json
        if not data:
            print("Nenhum dado recebido.")
            return jsonify({"error": "No data received"}), 400

        print(f"Dados brutos recebidos: {json.dumps(data, ensure_ascii=False, indent=4)}")

        if data.get('event') == 'messages.upsert':
            message_data = data.get('data', {})
            key = message_data.get('key', {})
            message_content = message_data.get('message', {})
            from_me = key.get('fromMe', False)

            if not from_me:
                remote_jid = key.get('remoteJid', '')
                sender = remote_jid.split('@')[0] if remote_jid else None

                message = message_content.get('conversation', '')

                if not sender or not message:
                    print("Erro: Estrutura de dados inválida.")
                    return jsonify({"error": "Invalid data structure"}), 400

                print(f"Mensagem recebida de {sender}: {message}")

                resposta = gerar_resposta(message)

                print(f"Mensagem a ser enviada: {resposta}")

                envio = enviar_mensagem(sender, resposta)
                if envio:
                    print(f"Status: Resposta enviada para {sender}")
                else:
                    print(f"Status: Erro ao enviar a resposta para {sender}")

                return jsonify({"status": "success"}), 200
            else:
                print("Mensagem enviada pelo bot ignorada.")
                return jsonify({"status": "ignored"}), 200
        else:
            print("Tipo de evento não suportado.")
            return jsonify({"status": "ignored"}), 200

    except Exception as e:
        print(f"Erro no processamento do webhook: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Servidor Flask iniciado. Aguardando mensagens da Evolution API...")
    app.run(host="0.0.0.0", port=5000)