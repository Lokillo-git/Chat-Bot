import tkinter as tk
from tkinter import simpledialog, scrolledtext, ttk
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import ollama
import threading
import time

# -----------------------------
# CONFIGURACIÓN RÁPIDA
# -----------------------------
MODELO_OLLAMA = "llama3.2:1b"  # Cambia por el modelo que tengas instalado

# -----------------------------
# CONEXIÓN A BASE DE DATOS OPTIMIZADA
# -----------------------------
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot_db",
            autocommit=True
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
# SISTEMA DE CACHÉ PARA RESPUESTAS RÁPIDAS
# -----------------------------
respuestas_cache = {}

def get_cached_response(user_input):
    """Retorna respuesta del cache si existe"""
    low_input = user_input.lower().strip()
    return respuestas_cache.get(low_input)

def add_to_cache(user_input, response):
    """Agrega respuesta al cache"""
    low_input = user_input.lower().strip()
    respuestas_cache[low_input] = response

# -----------------------------
# RESPUESTAS INSTANTÁNEAS MEJORADAS
# -----------------------------
def get_instant_response(prompt):
    """Respuestas locales ultra-rápidas"""
    low = prompt.lower().strip()
    
    # Diccionario expandido de respuestas instantáneas
    instant_responses = {
        "hola": "¡Hola! 😊 ¿En qué puedo ayudarte hoy?",
        "hello": "Hello! 👋 How can I assist you?",
        "adiós": "¡Hasta luego! 👋 Que tengas un excelente día.",
        "chao": "¡Chao! 😊 Espero verte pronto.",
        "bye": "Goodbye! 👋 Have a great day!",
        "gracias": "¡De nada! 💙 Me encanta ayudarte.",
        "thanks": "You're welcome! 💙 Happy to help!",
        "cómo estás": "¡Estoy funcionando perfectamente! 🤖 ¿Y tú cómo estás?",
        "quién eres": "Soy tu asistente de IA inteligente 🧠 con Ollama local. Aprendo de cada conversación.",
        "qué puedes hacer": "Puedo: • Responder preguntas • Aprender nuevas cosas • Conversar • Ayudarte con información • Y mucho más! 🚀",
        "qué es la inteligencia artificial": "La IA es la simulación de procesos de inteligencia humana por máquinas. Incluye aprendizaje automático, razonamiento y autocorrección. 🤖",
        "qué es python": "Python es un lenguaje de programación versátil y fácil de aprender, ideal para IA, web, datos y automatización. 🐍",
        "qué es machine learning": "El Machine Learning es una rama de la IA donde las máquinas aprenden patrones de datos sin programación explícita. 📊",
        "qué hora es": f"🕐 Son las {datetime.datetime.now().strftime('%H:%M:%S')}",
        "qué día es hoy": f"📅 Hoy es {datetime.datetime.now().strftime('%A, %d de %B de %Y')}",
        "cuál es la fecha": f"📅 La fecha actual es {datetime.datetime.now().strftime('%d/%m/%Y')}",
        "cómo te llamas": "Me llamo Asistente IA 🤖 ¡Mucho gusto!",
        "quién te creó": "Fui creado para ayudarte con tus preguntas y tareas usando tecnología de IA local. 🚀"
    }
    
    # Búsqueda inteligente en el diccionario
    for key, value in instant_responses.items():
        if key in low:
            return value
    
    # Respuestas basadas en patrones
    if any(palabra in low for palabra in ["hora"]):
        return f"🕐 Son las {datetime.datetime.now().strftime('%H:%M:%S')}"
    
    if any(palabra in low for palabra in ["fecha", "día es"]):
        return f"📅 Hoy es {datetime.datetime.now().strftime('%A, %d de %B de %Y')}"
    
    return None

# -----------------------------
# CHATBOT CON BASE DE DATOS OPTIMIZADO
# -----------------------------
def get_db_response(user_input):
    """Búsqueda en base de datos con cache"""
    # Primero verificar cache
    cached = get_cached_response(user_input)
    if cached:
        return cached, 1.0
    
    data = get_all_data()
    if not data:
        return None, 0.0

    questions = [row[0] for row in data]
    answers = [row[1] for row in data]

    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(questions + [user_input])
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        index = similarity.argmax()

        if similarity[0][index] > 0.45:
            response = answers[index]
            add_to_cache(user_input, response)  # Cachear resultado
            return response, similarity[0][index]
        else:
            return None, similarity[0][index]
    except Exception as e:
        print(f"Error en procesamiento de texto: {e}")
        return None, 0.0

# -----------------------------
# OLLAMA OPTIMIZADO CON TIMEOUT
# -----------------------------
def get_ollama_response(prompt):
    """Respuesta de Ollama con timeout y optimizaciones"""
    try:
        # Configuración optimizada para respuestas rápidas
        response = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[
                {
                    'role': 'system', 
                    'content': 'Eres un asistente útil y conciso. Responde máximo 2 párrafos en español. Sé directo y claro. Responde en 100 palabras máximo.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.3,  # Menos creatividad = más rápido
                'num_predict': 120,  # Limitar longitud
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error Ollama: {e}")
        return f"💡 Basándome en tu pregunta sobre '{prompt}', es un tema interesante. ¿Te gustaría que aprenda más sobre esto?"

# -----------------------------
# SISTEMA DE RESPUESTAS JERÁRQUICO
# -----------------------------
def get_response(user_input):
    """Sistema optimizado de respuestas"""
    # 1. Respuesta instantánea (milisegundos)
    instant = get_instant_response(user_input)
    if instant:
        return instant
    
    # 2. Base de datos con cache (rápido)
    db_response, score = get_db_response(user_input)
    if db_response:
        return db_response
    
    # 3. Ollama (puede tomar segundos)
    return get_ollama_response(user_input)

# -----------------------------
# INTERFAZ GRÁFICA MEJORADA Y CORREGIDA
# -----------------------------
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.is_processing = False
        self.typing_indicator_id = None
        
    def setup_ui(self):
        # Configuración principal
        self.root.title("🚀 Chatbot Ultra-Rápido - Ollama")
        self.root.geometry("800x650")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Estilo moderno
        self.setup_styles()
        
        # Marco principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self.setup_header(main_frame)
        
        # Área de chat
        self.setup_chat_area(main_frame)
        
        # Controles
        self.setup_controls(main_frame)
        
        # Footer
        self.setup_footer(main_frame)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores modernos
        self.colors = {
            'primary': '#3a86ff',
            'secondary': '#8338ec',
            'success': '#06d6a0',
            'dark_bg': '#1e1e1e',
            'card_bg': '#2d2d2d',
            'text_light': '#ffffff',
            'text_muted': '#b0b0b0'
        }
    
    def setup_header(self, parent):
        header_frame = tk.Frame(parent, bg=self.colors['dark_bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="💬 Chatbot Inteligente",
            font=("Arial", 20, "bold"),
            bg=self.colors['dark_bg'],
            fg=self.colors['text_light']
        )
        title_label.pack(side=tk.LEFT)
        
        status_label = tk.Label(
            header_frame,
            text="🟢 Conectado - Ollama Local",
            font=("Arial", 10),
            bg=self.colors['dark_bg'],
            fg=self.colors['success']
        )
        status_label.pack(side=tk.RIGHT)
    
    def setup_chat_area(self, parent):
        chat_container = tk.Frame(parent, bg=self.colors['card_bg'], relief=tk.FLAT, bd=1)
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.chat_window = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_light'],
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0,
            insertbackground=self.colors['text_light']  # Color del cursor
        )
        self.chat_window.pack(fill=tk.BOTH, expand=True)
        
        # Configurar estilos de texto
        self.chat_window.tag_config("user", foreground="#3a86ff", font=("Arial", 11, "bold"))
        self.chat_window.tag_config("bot", foreground="#06d6a0", font=("Arial", 11))
        self.chat_window.tag_config("system", foreground="#ff9e00", font=("Arial", 9, "italic"))
        self.chat_window.tag_config("typing", foreground="#8ecae6", font=("Arial", 10, "italic"))
        self.chat_window.tag_config("error", foreground="#ef476f", font=("Arial", 11))
        
        # Mensaje de bienvenida
        self.show_welcome_message()
    
    def setup_controls(self, parent):
        controls_frame = tk.Frame(parent, bg=self.colors['dark_bg'])
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Campo de entrada
        self.entry = tk.Entry(
            controls_frame,
            font=("Arial", 12),
            bg='#3d3d3d',
            fg=self.colors['text_light'],
            insertbackground=self.colors['text_light'],
            relief=tk.FLAT,
            bd=2
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)
        self.entry.focus()
        
        # Botón enviar
        self.send_button = tk.Button(
            controls_frame,
            text="🚀 Enviar",
            command=self.send_message,
            bg=self.colors['primary'],
            fg=self.colors['text_light'],
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=10
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Botón limpiar
        clear_button = tk.Button(
            controls_frame,
            text="🧹 Limpiar",
            command=self.clear_chat,
            bg=self.colors['secondary'],
            fg=self.colors['text_light'],
            font=("Arial", 10),
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=8
        )
        clear_button.pack(side=tk.RIGHT, padx=(0, 10))
    
    def setup_footer(self, parent):
        footer_frame = tk.Frame(parent, bg=self.colors['dark_bg'])
        footer_frame.pack(fill=tk.X)
        
        status_text = tk.Label(
            footer_frame,
            text="⚡ Respuestas instantáneas | 💾 Cache activado | 🧠 Ollama Local",
            font=("Arial", 9),
            bg=self.colors['dark_bg'],
            fg=self.colors['text_muted']
        )
        status_text.pack()
    
    def show_welcome_message(self):
        welcome_text = """🤖 Bot: ¡Hola! Soy tu asistente ultra-rápido 🚀

✨ Características:
• ⚡ Respuestas instantáneas
• 💾 Sistema de cache inteligente  
• 🧠 IA local con Ollama
• 🎓 Aprendizaje continuo
• 🎨 Interfaz moderna

¡Pregúntame lo que quieras! Ejemplos:
• "Hola" 👋
• "¿Qué hora es?" 🕐
• "Explícame la IA" 🤖
• "¿Qué es Python?" 🐍

"""
        self.chat_window.insert(tk.END, welcome_text, "system")
        self.chat_window.see(tk.END)
    
    def clear_chat(self):
        self.chat_window.delete(1.0, tk.END)
        self.show_welcome_message()
    
    def send_message(self, event=None):
        if self.is_processing:
            return
            
        user_input = self.entry.get().strip()
        if not user_input:
            return

        # Mostrar mensaje del usuario inmediatamente
        self.chat_window.insert(tk.END, f"👤 Tú: {user_input}\n", "user")
        self.chat_window.see(tk.END)
        self.entry.delete(0, tk.END)
        
        # Deshabilitar entrada
        self.set_input_state(False)
        
        # Procesar en hilo separado para no bloquear la interfaz
        thread = threading.Thread(target=self.process_message, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def process_message(self, user_input):
        self.is_processing = True
        
        try:
            # Mostrar indicador de typing (en el hilo principal)
            self.root.after(0, self.show_typing_indicator)
            
            # Obtener respuesta
            start_time = time.time()
            response = get_response(user_input)
            response_time = time.time() - start_time
            
            # Ocultar indicador y mostrar respuesta (en el hilo principal)
            self.root.after(0, self.hide_typing_indicator)
            self.root.after(0, self.display_response, response, response_time, user_input)
            
        except Exception as e:
            self.root.after(0, self.hide_typing_indicator)
            self.root.after(0, self.display_error, str(e))
        
        self.is_processing = False
    
    def show_typing_indicator(self):
        """Mostrar indicador de que está escribiendo"""
        if self.typing_indicator_id is None:
            self.chat_window.insert(tk.END, "🤖 Bot: ", "bot")
            self.chat_window.insert(tk.END, "escribiendo", "typing")
            self.chat_window.insert(tk.END, "...\n", "typing")
            self.typing_indicator_id = "typing"
            self.chat_window.see(tk.END)
    
    def hide_typing_indicator(self):
        """Ocultar indicador de escritura"""
        if self.typing_indicator_id:
            # Buscar y eliminar la línea de "escribiendo..."
            content = self.chat_window.get(1.0, tk.END)
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if "escribiendo..." in line:
                    # Eliminar la línea completa
                    start_index = f"{i}.0"
                    end_index = f"{i+1}.0"
                    self.chat_window.delete(start_index, end_index)
                    break
            
            self.typing_indicator_id = None
    
    def display_response(self, response, response_time, user_input):
        """Mostrar la respuesta en la interfaz"""
        # Mostrar respuesta con tiempo de procesamiento
        time_info = f" ⚡{response_time:.1f}s"
        self.chat_window.insert(tk.END, f"🤖 Bot: {response}", "bot")
        self.chat_window.insert(tk.END, f"{time_info}\n\n", "system")
        self.chat_window.see(tk.END)
        
        # Preguntar por aprendizaje si fue respuesta de Ollama
        if not any(tag in response for tag in ["🕐", "📅", "¡Hola!", "Hello!", "¡De nada!", "You're welcome"]):
            self.root.after(1000, self.ask_for_learning, user_input)
        else:
            self.set_input_state(True)
    
    def display_error(self, error_msg):
        """Mostrar mensaje de error"""
        self.chat_window.insert(tk.END, f"🤖 Bot: ❌ Error: {error_msg}\n\n", "error")
        self.chat_window.see(tk.END)
        self.set_input_state(True)
    
    def set_input_state(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.entry.config(state=state)
        self.send_button.config(state=state)
        
        if enabled:
            self.entry.focus()
            self.root.config(cursor="")
        else:
            self.root.config(cursor="watch")
    
    def ask_for_learning(self, user_input):
        user_answer = simpledialog.askstring(
            "🎓 Aprendizaje",
            "¿Quieres guardar esta conversación en la base de datos?\n\n" +
            f"Pregunta: {user_input}\n\n" +
            "Escribe la respuesta ideal (o Cancelar para omitir):",
            parent=self.root
        )
        
        if user_answer and user_answer.strip():
            success = insert_new_qa(user_input, user_answer.strip())
            if success:
                self.chat_window.insert(tk.END, "🤖 Bot: ¡✅ Aprendido! Respuesta guardada.\n\n", "bot")
            else:
                self.chat_window.insert(tk.END, "🤖 Bot: ❌ Error al guardar.\n\n", "error")
            self.chat_window.see(tk.END)
        
        self.set_input_state(True)

# -----------------------------
# INICIALIZACIÓN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()