import streamlit as st
import cv2
import os
import numpy as np
import warnings
from PIL import Image
import pytesseract
import PyPDF2
from fpdf import FPDF
from langdetect import detect, DetectorFactory
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import queue
from gtts import gTTS
import io
from groq import Groq
import base64

# --- 1. CONFIGURATION SYSTÈME ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

# --- 2. CONFIGURATION DES SECRETS (CLÉ GROQ) ---
api_key_vault = st.secrets.get("GROQ_API_KEY", "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")
client = Groq(api_key=api_key_vault)

# --- 3. CONFIGURATION TESSERACT (AUTO-DÉTECTION) ---
if os.path.exists("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
else:
    # Chemin local Windows
    tesseract_exe = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_exe):
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe

# --- 4. CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_all_models():
    import tensorflow as tf
    # Note: On n'importe pas NLLB ici car on utilise Groq pour la traduction (plus rapide et léger)
    try:
        asl = tf.keras.models.load_model("asl_model.h5", compile=False)
        return asl
    except:
        return None

asl_model = load_all_models()

# --- 5. STYLE & DESIGN (TOUR EIFFEL) ---
def get_base64(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except: return None
    return None

img_b64 = get_base64("eiffel.jpg")

st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpg;base64,{img_b64 if img_b64 else ''}");
        background-size: cover;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(40, 40, 40, 0.7) !important; 
    }}
    .stApp {{ background-color: #000000; }}
    div.stButton > button {{
        background-color: #C11F1F !important;
        color: white !important;
        border-radius: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 6. FONCTIONS UTILITAIRES ---
def translate_with_groq(text, target_lang):
    if not text: return ""
    try:
        prompt = f"Traduire le texte suivant en {target_lang}. Ne donne que la traduction : {text}"
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur de traduction : {e}"

def text_to_speech(text, lang_name):
    lang_map = {"Français": "fr", "Anglais": "en", "Arabe": "ar", "Espagnol": "es", "Allemand": "de"}
    clean_lang = lang_map.get(lang_name, "fr")
    tts = gTTS(text=text, lang=clean_lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0) # IMPORTANT pour que st.audio fonctionne
    return fp

def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return pdf.output(dest='S').encode('latin-1')

# --- 7. INTERFACE ---
st.title("👂💬 Traducteur Intelligent Multi-Modal")

if "history" not in st.session_state: st.session_state.history = []
if "messages" not in st.session_state: st.session_state.messages = []
if "text_from_signs" not in st.session_state: st.session_state.text_from_signs = ""

with st.sidebar:
    st.header("⚙️ Configuration")
    languages = ["Français", "Anglais", "Arabe", "Espagnol", "Allemand"]
    src_mode = st.selectbox("Langue source :", ["Auto"] + languages)
    target_lang = st.selectbox("Traduire vers :", languages)
    
    st.divider()
    if st.button("Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
    for entry in st.session_state.history[:5]:
        with st.expander(f"📜 {entry['source'][:20]}..."):
            st.write(entry['result'])

tabs = st.tabs(["📝 Texte & PDF", "📸 Image (OCR)", "🎤 Vocal", "🤟 Signes", "💬 Chatbot"])
source_text = ""

# --- ONGLET 1: TEXTE ---
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        txt_in = st.text_area("Saisissez votre texte :", height=150)
        if txt_in: source_text = txt_in
    with col2:
        pdf_file = st.file_uploader("Ou importez un PDF", type="pdf")
        if pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            source_text = " ".join([p.extract_text() for p in reader.pages])

# --- ONGLET 2: IMAGE ---
with tabs[1]:
    img_file = st.file_uploader("Charger une image", type=["jpg", "png", "jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=300)
        if st.button("🔍 Extraire le texte"):
            source_text = pytesseract.image_to_string(img)
            st.info("Texte extrait !")

# --- ONGLET 3: VOCAL ---
with tabs[2]:
    st.warning("La reconnaissance vocale via micro direct ne fonctionne que sur PC local. Sur le Web, utilisez les autres onglets.")
    import speech_recognition as sr
    if st.button("🎤 Écouter (Local uniquement)"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            source_text = r.recognize_google(audio, language="fr-FR")
            st.success(f"Reconnu : {source_text}")

# --- ONGLET 4: SIGNES ---
with tabs[3]:
    res_q = queue.Queue()
    class SignProc(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (96, 96)) / 255.0
            if asl_model:
                pred = asl_model.predict(np.expand_dims(resized, axis=0), verbose=0)
                letter = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[np.argmax(pred)]
                res_q.put(letter)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="asl", video_processor_factory=SignProc)
    
    col_c1, col_c2 = st.columns(2)
    current_letter = res_q.get() if not res_q.empty() else "-"
    col_c1.metric("Lettre", current_letter)
    
    if col_c2.button("Ajouter lettre"):
        if current_letter != "-": st.session_state.text_from_signs += current_letter
    
    source_text = st.text_area("Texte accumulé :", value=st.session_state.text_from_signs)

# --- ONGLET 5: CHATBOT ---
with tabs[4]:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Une question ?"):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("assistant"):
            r = client.chat.completions.create(model="llama-3.1-8b-instant", messages=st.session_state.messages)
            resp = r.choices[0].message.content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})

# --- MOTEUR DE TRADUCTION FINAL ---
if source_text:
    st.divider()
    with st.spinner("Traduction..."):
        result = translate_with_groq(source_text, target_lang)
        st.subheader("Résultat :")
        st.success(result)
        
        # Audio & PDF
        c1, c2 = st.columns(2)
        with c1:
            st.audio(text_to_speech(result, target_lang), format="audio/mp3")
        with c2:
            st.download_button("📥 PDF", data=generate_pdf(result), file_name="traduction.pdf")
        
        # Historique
        if not st.session_state.history or st.session_state.history[0]["source"] != source_text[:20]:
            st.session_state.history.insert(0, {"source": source_text, "result": result})
