import tkinter as tk
from tkinter import simpledialog, scrolledtext
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import ollama

# -----------------------------
# CONEXIÓN A BASE DE DATOS
# -----------------------------
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot_db"
        )
    except mysql.connector.Error as e:
        print(f"Error de base de datos: {e}")
        return None

def get_all_data():
    db = connect_db()
    if db is None:
        return []
    
    try:
        cursor = db.cursor()
        cursor.execute("SELECT question, answer FROM knowledge")
        data = cursor.fetchall()
        return data
    except mysql.connector.Error as e:
        print(f"Error al obtener datos: {e}")
        return []
    finally:
        if db and db.is_connected():
            db.close()

def insert_new_qa(question, answer):
    db = connect_db()
    if db is None:
        return False
    
    try:
        cursor = db.cursor()
        cursor.execute("INSERT INTO knowledge (question, answer) VALUES (%s, %s)", (question, answer))
        db.commit()
        return True
    except mysql.connector.Error as e:
        print(f"Error al insertar datos: {e}")
        return False
    finally:
        if db.is_connected():
            db.close()

# -----------------------------
# CHATBOT LÓGICO (BASE DE DATOS - CORREGIDO)
# -----------------------------
def get_db_response(user_input):
    data = get_all_data()
    if not data:
        return None, 0.0

    questions = [row[0] for row in data]
    answers = [row[1] for row in data]

    try:
        # CORRECCIÓN: Eliminado el parámetro stop_words no válido
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(questions + [user_input])
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        index = similarity.argmax()

        if similarity[0][index] > 0.45:
            return answers[index], similarity[0][index]
        else:
            return None, similarity[0][index]
    except Exception as e:
        print(f"Error en procesamiento de texto: {e}")
        return None, 0.0

# -----------------------------
# RESPUESTA CON OLLAMA (LOCAL - SIN INTERNET)
# -----------------------------
def get_ollama_response(prompt):
    """
    Usa Ollama localmente para generar respuestas
    """
    low = prompt.lower()

    # Respuestas locales rápidas
    if "hora" in low:
        return f"🕐 La hora actual es {datetime.datetime.now().strftime('%H:%M:%S')}."
    if "fecha" in low:
        return f"📅 Hoy es {datetime.datetime.now().strftime('%d/%m/%Y')}."
    if "hola" in low or "hello" in low:
        return "¡Hola! 😊 Soy tu asistente con IA local. ¿En qué puedo ayudarte?"
    if "adiós" in low or "chao" in low or "bye" in low:
        return "¡Hasta luego! 👋 Fue un gusto ayudarte."

    try:
        # Usar Ollama para generar respuesta [citation:1][citation:3]
        response = ollama.chat(
            model='llama3.2:1b',  # Usa el modelo que instalaste
            messages=[
                {
                    'role': 'system', 
                    'content': 'Eres un asistente útil y amigable. Responde de forma concisa y clara en español.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"🤖 Respuesta local: He procesado tu pregunta sobre '{prompt}'. Actualmente estoy aprendiendo de tus preguntas para mejorar mis respuestas."

# -----------------------------
# RESPUESTA EN STREAMING CON OLLAMA (OPCIONAL)
# -----------------------------
def get_ollama_streaming_response(prompt, chat_window):
    """
    Versión con streaming para ver la respuesta en tiempo real
    """
    try:
        full_response = ""
        chat_window.insert(tk.END, "🤖 Bot: ", "bot")
        
        stream = ollama.chat(
            model='llama3.2:1b',
            messages=[
                {'role': 'system', 'content': 'Eres un asistente útil en español.'},
                {'role': 'user', 'content': prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                chat_window.insert(tk.END, content, "bot")
                chat_window.see(tk.END)
                full_response += content
                
        chat_window.insert(tk.END, "\n\n", "bot")
        return full_response
    except Exception as e:
        error_msg = f"🤖 Respuesta local: Interesante pregunta sobre '{prompt}'. Voy a aprender más sobre este tema.\n\n"
        chat_window.insert(tk.END, error_msg, "bot")
        return error_msg

# -----------------------------
# INTEGRACIÓN GENERAL MEJORADA
# -----------------------------
def get_response(user_input, chat_window=None):
    # Primero buscar en base de datos
    response, score = get_db_response(user_input)
    if response:
        return f"💡 {response}"

    # Usar Ollama localmente
    if chat_window:
        return get_ollama_streaming_response(user_input, chat_window)
    else:
        return get_ollama_response(user_input)

def learn_new_answer(user_input, user_answer):
    if user_answer and user_answer.strip():
        success = insert_new_qa(user_input, user_answer.strip())
        return success
    return False

# -----------------------------
# INTERFAZ GRÁFICA (MODIFICADA)
# -----------------------------
def send_message():
    user_input = entry.get().strip()
    if not user_input:
        return

    # Mostrar mensaje del usuario
    chat_window.insert(tk.END, f"👤 Tú: {user_input}\n", "user")
    chat_window.see(tk.END)
    entry.delete(0, tk.END)

    # Deshabilitar entrada mientras procesa
    entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)
    root.config(cursor="watch")
    
    # Procesar en segundo plano
    root.after(100, process_message, user_input)

def process_message(user_input):
    try:
        bot_response = get_response(user_input, chat_window)
        
        # Si no usamos streaming, mostrar la respuesta completa
        if "💡" in bot_response or "🤖 Respuesta local" in bot_response:
            chat_window.insert(tk.END, f"🤖 Bot: {bot_response}\n\n", "bot")
            chat_window.see(tk.END)

        # Verificar si es un caso para aprendizaje
        if not any(tag in bot_response for tag in ["💡", "🕐", "📅"]):
            root.after(1000, ask_for_learning, user_input)
        else:
            enable_input()
            
    except Exception as e:
        chat_window.insert(tk.END, f"🤖 Bot: ❌ Error: {str(e)}\n\n", "bot")
        enable_input()

def enable_input():
    entry.config(state=tk.NORMAL)
    send_button.config(state=tk.NORMAL)
    root.config(cursor="")
    entry.focus()

def ask_for_learning(user_input):
    user_answer = simpledialog.askstring(
        "🎓 Aprendizaje",
        "¿Quieres guardar esta pregunta y respuesta en la base local?\n\nPregunta: " + user_input + "\n\nEscribe la respuesta ideal (o Cancelar para omitir):",
        parent=root
    )
    
    if user_answer and user_answer.strip():
        success = learn_new_answer(user_input, user_answer.strip())
        if success:
            chat_window.insert(tk.END, "🤖 Bot: ¡✅ Gracias! He aprendido algo nuevo. 😊\n\n", "bot")
        else:
            chat_window.insert(tk.END, "🤖 Bot: ❌ Hubo un error al guardar. Intenta más tarde.\n\n", "bot")
        chat_window.see(tk.END)
    
    enable_input()

# -----------------------------
# UI Tkinter
# -----------------------------
root = tk.Tk()
root.title("🤖 Chatbot Inteligente - Ollama Local")
root.geometry("750x600")

# Configurar estilo
root.configure(bg='#f0f0f0')

# Frame principal
main_frame = tk.Frame(root, bg='#f0f0f0')
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

# Título
title_label = tk.Label(main_frame, text="💬 Chatbot con Ollama Local", 
                      font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#333')
title_label.pack(pady=(0, 10))

# Área de chat
chat_frame = tk.Frame(main_frame, relief=tk.GROOVE, bd=2)
chat_frame.pack(fill=tk.BOTH, expand=True)

chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                      font=("Arial", 11),
                                      bg='#fafafa', 
                                      padx=10, pady=10)
chat_window.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

# Configurar tags
chat_window.tag_config("user", foreground="#1e40af", font=("Arial", 11, "bold"))
chat_window.tag_config("bot", foreground="#059669", font=("Arial", 11))

# Frame de entrada
input_frame = tk.Frame(main_frame, bg='#f0f0f0')
input_frame.pack(fill=tk.X, pady=(15, 0))

entry = tk.Entry(input_frame, font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.bind("<Return>", lambda event: send_message())
entry.focus()

send_button = tk.Button(input_frame, text="🚀 Enviar", 
                       command=send_message, 
                       bg="#3b82f6", fg="white", 
                       font=("Arial", 11, "bold"),
                       padx=20)
send_button.pack(side=tk.RIGHT)

# Mensaje de bienvenida
welcome_message = """🤖 Bot: ¡Hola! Soy tu nuevo asistente con Ollama Local 🎉

Ahora funciono completamente offline con:
• 🚀 IA local (sin dependencias de internet)
• 💾 Base de datos propia
• 🎓 Aprendizaje continuo

¡Pregúntame lo que quieras!

Ejemplos:
• "¿Qué es la inteligencia artificial?"
• "¿Qué hora es?"
• "Explícame cómo funciona Python"
• "Háblame sobre machine learning"

"""

chat_window.insert(tk.END, welcome_message, "bot")
chat_window.see(tk.END)

root.mainloop()