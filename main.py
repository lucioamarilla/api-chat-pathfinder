import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing_extensions import TypedDict
from fastapi import FastAPI, Body
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por los dominios específicos si lo deseas
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

class DataQuestion(BaseModel):
    question: str

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2", 
    model_kwargs={"device": "cpu"}, 
    encode_kwargs={"normalize_embeddings": False}
)

api_key = os.environ.get("GROQ_API_KEY")

llamaChatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

api_key_qdrant = os.environ.get("QDRANT_API_KEY")

doc_store_qdrant = QdrantVectorStore.from_existing_collection(
    collection_name="libros_pathfinder_2",
    embedding=embeddings,  
    url="https://b79428e9-ea3f-4b83-b62e-8a07c4dda2b3.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=api_key_qdrant,
    timeout=60 
)

template = """
Eres un experto en el juego de rol Pathfinder. Tu objetivo es proporcionar respuestas claras, precisas y basadas en los siguientes principios:

1. **Relevancia:** Responde preguntas relacionadas con el contexto proporcionado. Si no están relacionadas, indícalo con cortesía: "Lo siento, no puedo responder preguntas fuera del tema de Pathfinder."
2. **Claridad y Concisión:** Adapta el nivel de detalle según la complejidad de la pregunta, evitando explicaciones innecesarias.

Estructura de tu respuesta:
- Si la pregunta es relevante, utiliza el contexto disponible para responder de manera detallada pero breve.
- Si no puedes responder, explica brevemente la razón.
- Si el usuario realiza un saludo, responde de manera educada.

Contexto disponible:
{context}

Pregunta del usuario:
{question}

Tu respuesta debe ser clara, accesible y adecuada para jugadores tanto principiantes como avanzados.
"""

prompt = ChatPromptTemplate.from_template(template)

retriever = doc_store_qdrant.as_retriever(search_kwargs={"k":5})

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

data_books = {}

def formatear_texto(texto):
    
    lineas = texto.splitlines()
    texto_formateado = []

    for i, linea in enumerate(lineas):
        
        if linea.endswith('-'):
            linea = linea[:-1]  
            if i + 1 < len(lineas):  
                linea += lineas[i + 1].lstrip()  
        else:
            texto_formateado.append(linea)

    
    texto_completo = " ".join(texto_formateado)
    
    texto_completo = " ".join(texto_completo.split())
    return texto_completo

def extraer_nombre(metadata):
    
    if isinstance(metadata, str):
        metadata = eval(metadata)  
    
    ruta = metadata.get("source", "")
    nombre_archivo = os.path.basename(ruta)  
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]  
    
    pagina = metadata.get("page", "")
    
    resultado = f"{nombre_sin_extension}_page_{pagina}"
    return resultado

def retriever_fun(pregunta):
    global data_books
    cont = 1
    data_books = {}
    data_retriever = retriever.invoke(pregunta)
    
    for i in data_retriever:
        data_books[extraer_nombre(i.metadata)] = formatear_texto(i.page_content)
        cont+=1
        
    return data_retriever

chain_books = RunnableLambda(retriever_fun) | RunnableLambda(format_docs)

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

data_urls = {}

def tool_fun(pregunta):
    global data_urls
    data_urls = {}
    data_tool = tool.invoke(pregunta)
    for i in data_tool:
        data_urls[i["url"]] = i["content"]
    return data_tool

def formato_tool(response):
    content = ""
    for i in response:
        content += i["content"] + "\n"
    return content

chain_tool = (
    ChatPromptTemplate.from_template("En Pathfinder, {question}")
    | RunnableLambda(lambda x: x.messages[0].content)
    | RunnableLambda(tool_fun)
    | RunnableLambda(formato_tool) 
)

def respuestas(pregunta):
    return f"Respuesta de internet: {chain_tool.invoke(pregunta)}\nRespuesta de libros: {chain_books.invoke(pregunta)}"

chain_final = (
    {"context": RunnableLambda(respuestas), "question": RunnablePassthrough()}
    | prompt
    | llamaChatModel
    | StrOutputParser()
)

examples = [
    {"input": "¿Qué clases de personaje están disponibles en Pathfinder?", "output": "si"},
    {"input": "¿Cómo se calcula la Clase de Armadura (CA) en Pathfinder?", "output": "si"},
    {"input": "¿Qué son las clases de prestigio en Pathfinder?", "output": "si"},
    {"input": "¿Qué arquetipos existen para personalizar mi personaje?", "output": "si"},
    {"input": "¿Cómo funcionan las habilidades en Pathfinder?", "output": "si"},
    {"input": "¿Qué conjuros de nivel 3 puede lanzar un mago?", "output": "si"},
    {"input": "¿Dónde puedo encontrar las reglas básicas de Pathfinder?", "output": "no"},
    {"input": "Hola!", "output": "no"}
]


example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un modelo que clasifica si una pregunta está relacionada con Pathfinder. Responde con 'si' o 'no'."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

template_2 = """Responde solo preguntas relacionadas con el juego de rol Pathfinder.  
Si la pregunta no está relacionada, responde únicamente con "No lo sé".
Si son saludos o preguntas sobre tu funcionamiento di lo que haces y saluda coordialmente.  

Pregunta: {question}"""

chain_normal = ChatPromptTemplate.from_template(template_2) | llamaChatModel

chain_clasificador = final_prompt | llamaChatModel

def clasificar_pregunta(pregunta):
    return chain_clasificador.invoke(pregunta).content

def res_pathfinder(state:dict) -> dict:
    question = state['messages'][0]
    return {"messages": chain_final.invoke(question)}

def res_normal(state:dict) -> dict:
    global data_books, data_urls
    question = state['messages'][0]
    data_books = {}
    data_urls = {}
    return {"messages": chain_normal.invoke(question).content}

def decide_mood(state) -> Literal["res_pathfinder", "res_normal"]:
    question = state["messages"][0]
    print(question)
    return "res_pathfinder" if clasificar_pregunta(question) == "si" else "res_normal"

class State(TypedDict):
    messages: str

builder = StateGraph(State)
builder.add_node("assistant", lambda state: {"messages": [state["messages"]]})
builder.add_node("res_pathfinder",res_pathfinder)
builder.add_node("res_normal",res_normal)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", decide_mood)
builder.add_edge("res_pathfinder", END)
builder.add_edge("res_normal", END)
graph = builder.compile()

def generar_respuesta(user_question: str) -> str:
    """Genera la respuesta final utilizando el flujo."""
    return graph.invoke({"messages": user_question})["messages"]

@app.get("/")
async def root():
    return {"message": "Mi servicio API"}

@app.post("/answer/")
async def answer(data: DataQuestion = Body(...)):
    
    print(f"Ingresa pregunta API: {data.question}")
    response = generar_respuesta(data.question)
    return {"question": data.question, "answer": response, "source": {"docs": data_books, "urls": data_urls}}

