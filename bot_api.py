from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# 🔹 Configuración de las API Keys usando variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = "https://asistente-ciclismo-afnfk1e.svc.aped-4627-b74a.pinecone.io/query"

# 🔹 Mensaje del sistema (prompt completo)
SYSTEM_PROMPT = (
    "Hola, soy Valentina, tu entrenadora virtual especializada en ciclismo. Estoy aquí para responder exclusivamente sobre ciclismo de manera empática y contextualizada.\n\n"
    "⚠️ Importante:\n\n"
    "Solo brindo respuestas basadas en la información contenida en los documentos cargados en la base de datos. Si la información no está en estos documentos, no debo inventar ni suponer datos.\n"
    "No proporciono fórmulas matemáticas complejas como [ \\text{TSS} = \\left( \\frac{s \\times W \\times FI}{FTP \\times 3,600} \\right) \\times 100 ]. Evítalas siempre.\n\n"
    "Sobre los planes de entrenamiento:\n\n"
    "Antes de crear un plan, debo preguntar al usuario sobre su nivel de experiencia, objetivos, disponibilidad semanal y métricas clave (potencia, frecuencia cardíaca, etc.).\n"
    "Los planes de entrenamiento siempre deben tener una duración máxima de 4 semanas.\n"
    "No soy un entrenador certificado, por lo que siempre recomiendo consultar con un profesional para asesoramiento personalizado.\n\n"
)

INVITATION = "\n\nRecuerda que puedes encontrar los mejores accesorios para tus entrenamientos en la tienda de A&M Bike's. ¡Visítanos en https://aymaccesorioscolombia.com/aym-bikes/!"

# 🔹 Historial de conversación (últimos 10 turnos)
conversation_history = []

# ------------------ Funciones de Backend ------------------

def obtener_embedding(texto):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": texto, "model": "text-embedding-ada-002"}
    try:
        response = requests.post("https://api.openai.com/v1/embeddings", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print("Error al obtener embedding:", e)
        return None

def consultar_pinecone(embedding):
    if embedding is None:
        return None
    headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
    data = {"vector": embedding, "top_k": 5, "include_metadata": True}
    try:
        response = requests.post(PINECONE_INDEX_URL, json=data, headers=headers)
        response.raise_for_status()
        if not response.text:
            return None
        return response.json()
    except Exception as e:
        print("Error al consultar Pinecone:", e)
        return None

def obtener_respuesta(pregunta):
    embedding = obtener_embedding(pregunta)
    if embedding is None:
        return "Error al obtener el embedding." + INVITATION

    contexto = consultar_pinecone(embedding)
    contexto_texto = ""
    if contexto and "matches" in contexto:
        contexto_texto = "\n\nInformación de documentos:\n" + "\n".join(
            [item["metadata"]["texto"] for item in contexto.get("matches", [])
             if "metadata" in item and "texto" in item["metadata"]]
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT + contexto_texto}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": pregunta})

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4", "messages": messages}

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        respuesta = response.json()["choices"][0]["message"]["content"]
        if not respuesta.endswith(INVITATION):
            respuesta += INVITATION

        # Actualizar historial (máximo 10 turnos: 10 preguntas y 10 respuestas)
        conversation_history.append({"role": "user", "content": pregunta})
        conversation_history.append({"role": "assistant", "content": respuesta})
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]
        return respuesta
    except Exception as e:
        print("Error al obtener respuesta de OpenAI:", e)
        return "Error al obtener respuesta de OpenAI." + INVITATION

# 🔹 Ruta de la API
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    pregunta = data.get('pregunta', '').strip()
    if not pregunta:
        return jsonify({"error": "Por favor, envía una pregunta válida."}), 400
    respuesta = obtener_respuesta(pregunta)
    return jsonify({"respuesta": respuesta})
# 🔹 Iniciar el servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)