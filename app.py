import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
import warnings
from PIL import Image
import pytesseract
import PyPDF2
import speech_recognition as sr
from fpdf import FPDF
from langdetect import detect, DetectorFactory
from nllb_engine import TranslatorNLLB
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import queue
from gtts import gTTS
import io
from groq import Groq
import base64


# --- CONFIGURATION DU CLIENT GROQ ---
# Placez votre clé API ici

# Chercher la clé dans le coffre-fort caché de Streamlit
api_key_vault = st.secrets.get("GROQ_API_KEY", "METS_TA_CLE_ICI_POUR_TES_TESTS_LOCAUX")

client = Groq(api_key=api_key_vault)
# --- CONFIGURATION SYSTÈME EXISTANTE ---
warnings.filterwarnings("ignore")
# Configuration pour la stabilité de la détection de langue
DetectorFactory.seed = 0

# --- CONFIGURATION SYSTÈME ---
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- CONFIGURATION TESSERACT (CRUCIAL) ---
# On définit le chemin standard. Vérifiez qu'il existe sur votre disque C.
tesseract_exe = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

def add_letter(letter):
    st.session_state.detected_text += letter

# Initialisation du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialisation de l'historique dans la session
if "history" not in st.session_state:
    st.session_state.history = []

if os.path.exists(tesseract_exe):
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
else:
    st.sidebar.error("⚠️ Tesseract.exe introuvable dans Program Files. L'onglet Photo ne fonctionnera pas.")

st.set_page_config(page_title="IA Traducteur Universel", layout="wide")


# --- 2. FONCTION POUR L'IMAGE ---
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img_b64 = get_base64("eiffel.jpg")

# --- 3. STYLE CSS AVEC RECTANGLE TRANSPARENT ---
st.markdown(f"""
    <style>
    /* 1. On place la photo de la Tour Eiffel en fond de la sidebar */
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpg;base64,{img_b64}");
        background-size: cover;
        background-position: center;
    }}

    /* 2. On crée le rectangle gris transparent par-dessus l'image */
    /* La valeur 0.6 correspond à l'opacité (0 = invisible, 1 = gris total) */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(40, 40, 40, 0.6) !important; 
        padding-top: 2rem;
    }}

    /* 3. On s'assure que le texte dans la sidebar est bien blanc et lisible */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
        font-weight: 500;
    }}

    /* 4. Fond de l'application en gris foncé */
    .stApp {{
        background-color: #000000;
    }}

    /* 5. Boutons Rouge Londres */
    div.stButton > button {{
        background-color: #C11F1F !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)
# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_all_models():
    try:
        asl = tf.keras.models.load_model("asl_model.h5")
        nllb = TranslatorNLLB()
        return asl, nllb
    except:
        # Si le modèle ASL manque, on charge au moins le traducteur
        nllb = TranslatorNLLB()
        return None, nllb

asl_model, translator = load_all_models()

# --- FONCTION EXPORT PDF ---
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Remplacement des caractères non-compatibles avec fpdf standard
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return pdf.output(dest='S').encode('latin-1')
def text_to_speech(text, lang_code):
    short_lang = lang_code.split('_')[0] 
    # AJOUT de "deu": "de" pour l'Allemand
    lang_map = {
        "fra": "fr", 
        "eng": "en", 
        "arb": "ar", 
        "spa": "es", 
        "deu": "de"  # <--- NE PAS OUBLIER CELUI-CI
    }
    clean_lang = lang_map.get(short_lang, "en")
    
    tts = gTTS(text=text, lang=clean_lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp

# --- INTERFACE ---
st.title("👂💬 Traducteur Intelligent Multi-Modal")

# Sidebar
st.sidebar.header("⚙️ Configuration")

languages = {
    "Français": "fra_Latn",
    "Anglais": "eng_Latn",
    "Arabe": "arb_Arab",
    "Espagnol": "spa_Latn",
    "Allemand": "deu_Latn"
}

# Mode de détection
src_mode = st.sidebar.selectbox("Langue source :", 
    ["Détection Automatique", "Français", "Anglais", "Arabe", "Espagnol", "Allemand"])
target_lang = st.sidebar.selectbox("Traduire vers :", list(languages.keys()))
target_code = languages[target_lang]

tabs = st.tabs(["📝 Texte & PDF", "📸 Image (OCR)", "🎤 Vocal", "🤟 Signes", "💬 Chatbot"])
source_text = ""

# --- SECTION HISTORIQUE DANS LA SIDEBAR ---
st.sidebar.divider()
st.sidebar.subheader("history_edu Historique")

if st.sidebar.button("Effacer l'historique"):
    st.session_state.history = []
    st.rerun()

for i, entry in enumerate(st.session_state.history[:10]): # Affiche les 10 derniers
    with st.sidebar.expander(f"🔍 {entry['source']}"):
        st.write(f"**Vers :** {entry['target_lang']}")
        st.write(entry['result'])
        if st.button("Récupérer", key=f"hist_{i}"):
            # Optionnel : permet de ré-afficher le texte dans l'interface
            st.info("Texte copié dans l'historique (Visualisation uniquement)")

# 1. Onglet Texte & PDF
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        txt_in = st.text_area("Saisissez votre texte :", height=200)
        if txt_in: source_text = txt_in
    with col2:
        pdf_file = st.file_uploader("Ou importez un PDF", type="pdf")
        if pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            source_text = " ".join([page.extract_text() for page in reader.pages])
            st.success("Texte du PDF extrait !")

# 2. Onglet Image (OCR)
with tabs[1]:
    img_file = st.file_uploader("Charger une image contenant du texte", type=["jpg", "png", "jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=400)
        if st.button("🔍 Extraire le texte"):
            try:
                source_text = pytesseract.image_to_string(img)
                st.info("Texte extrait via OCR.")
            except Exception:
                st.error("Erreur : Tesseract n'est pas configuré correctement sur votre système.")

# 3. Onglet Vocal (PyAudio est prêt !)
with tabs[2]:
    st.subheader("🎤 Traduction Vocale")
    
    # Dictionnaire étendu pour la reconnaissance vocale
    speech_langs = {
        "Français": "fr-FR",
        "Anglais": "en-US",
        "Arabe": "ar-SA",
        "Espagnol": "es-ES",
        "Allemand": "de-DE",
        "Détection Automatique": "fr-FR" # On met une langue par défaut pour l'écoute
    }

    if st.button("🎤 Commencer l'écoute"):
        r = sr.Recognizer()
        # On récupère le code correspondant au choix de la barre latérale
        current_lang_code = speech_langs.get(src_mode, "fr-FR")
        
        try:
            with sr.Microphone() as source:
                st.write(f"Configuration du micro pour : **{src_mode}**...")
                r.adjust_for_ambient_noise(source, duration=1)
                st.info("Je vous écoute... Parlez maintenant.")
                
                audio = r.listen(source, timeout=7, phrase_time_limit=10)
                
                # Reconnaissance avec la langue sélectionnée
                source_text = r.recognize_google(audio, language=current_lang_code)
                
                st.success(f"Texte reconnu : {source_text}")
                # Le moteur de traduction global prendra le relais automatiquement 
                # car source_text n'est plus vide.
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas compris l'audio. Assurez-vous de parler la langue sélectionnée.")
        except sr.RequestError:
            st.error("Erreur de connexion avec le service de reconnaissance vocale.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# 4. Onglet Signes
# --- Initialisation de la file d'attente ---
# On utilise st.cache_resource pour que la queue survive au rechargement de la page
@st.cache_resource
def get_queue():
    return queue.Queue()

result_queue = get_queue()

with tabs[3]:
    st.subheader("🤟 Saisie par Langue des Signes")
    
    col_cam, col_res = st.columns([2, 1])
    
    with col_cam:
        class SignLanguageProcessor:
            def recv(self, frame):
                # 1. Récupération de l'image brute (format BGR standard OpenCV)
                img = frame.to_ndarray(format="bgr24")
                
                # 2. Inversion BGR -> RGB (Étape manquante n°1)
                # MobileNet a été entraîné sur du RGB, sans ça il ne reconnaît rien
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 3. Redimensionnement précis
                img_resized = cv2.resize(img_rgb, (96, 96))
                
                img_array = np.expand_dims(img_resized, axis=0)
                
                
                
                # 6. Prédiction
                prediction = asl_model.predict(img_array, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # On ne valide la lettre que si le modèle est sûr à plus de 70%
                if confidence > 0.7:
                    letter = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[class_idx]
                    result_queue.put(letter)
                else:
                    letter = "Analyse..."

                # Affichage du résultat sur la vidéo
                cv2.putText(img, f"Prediction: {letter} ({int(confidence*100)}%)", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        webrtc_streamer(
            key="sign-input",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

    with col_res:
        # RÉCUPÉRATION DE LA LETTRE DEPUIS LA QUEUE
        latest_letter = "-"
        try:
            # On récupère la dernière lettre envoyée par la caméra
            while not result_queue.empty():
                latest_letter = result_queue.get_nowait()
        except queue.Empty:
            latest_letter = "-"

        st.write("### Contrôle")
        st.metric("Lettre actuelle", latest_letter)
        
        # Initialiser le texte si vide
        if "text_from_signs" not in st.session_state:
            st.session_state.text_from_signs = ""

        if st.button("📥 Valider cette lettre"):
            if latest_letter != "-":
                st.session_state.text_from_signs += latest_letter
                st.rerun() # Force la mise à jour de l'affichage
            
        if st.button("Space ␣"):
            st.session_state.text_from_signs += " "
            st.rerun()
            
        if st.button("Effacer ❌"):
            st.session_state.text_from_signs = ""
            st.rerun()
            
        # Zone de texte finale
        final_text = st.text_area("Texte construit :", value=st.session_state.text_from_signs)
        
        if st.button("Traduire ce texte"):
            # On injecte le texte construit dans la variable globale de traduction
            source_text = st.session_state.text_from_signs
# --- 5. Onglet Chatbot ---
with tabs[4]:
    st.subheader("💬 Assistant IA Ultra-Rapide (Groq)")

    # 1. Configuration du comportement
    system_prompt = (
        "Tu es un assistant linguistique expert intégré dans une application de traduction. "
        "Tu maîtrises parfaitement le Français, l'Anglais, l'Arabe, l'Espagnol et l'Allemand. "
        "Réponds de manière concise et toujours dans la langue utilisée par l'utilisateur."
    )

    # 2. Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Zone de saisie utilisateur
    if prompt := st.chat_input("Posez une question à l'IA..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Appel à l'API Groq avec Streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                completion = client.chat.completions.create(
                    # On remplace 'llama3-8b-8192' par une version plus récente
                    model="llama-3.1-8b-instant", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    stream=True,
                )

                # Boucle de lecture du stream
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # Mise à jour de l'affichage en temps réel
                        response_placeholder.markdown(full_response + "▌")
                
                # Affichage final sans le curseur
                response_placeholder.markdown(full_response)
                
                # Sauvegarde de la réponse de l'assistant
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Erreur de connexion à Groq : {e}")

# --- MOTEUR DE TRADUCTION FINAL ---
if source_text:
    st.divider()
    
    # Gestion Auto-Détection
    if src_mode == "Détection Automatique":
        try:
            det = detect(source_text)
            # Mapping complet pour NLLB
            mapping_auto = {
                'fr': 'fra_Latn',
                'en': 'eng_Latn',
                'ar': 'arb_Arab',
                'es': 'spa_Latn',
                'de': 'deu_Latn'
            }
            src_code = mapping_auto.get(det, "fra_Latn")
            st.caption(f"✨ Langue détectée : {det.upper()}")
        except:
            src_code = "fra_Latn"
    else:
        src_code = languages[src_mode]

    with st.spinner("Traduction en cours..."):
        result = translator.translate(source_text, src_code, target_code)
        
        # --- ENREGISTREMENT DANS L'HISTORIQUE ---
        new_entry = {
            "source": source_text[:50] + "..." if len(source_text) > 50 else source_text,
            "result": result,
            "target_lang": target_lang
        }
        # On vérifie si c'est déjà le dernier élément pour éviter les doublons au rerun
        if not st.session_state.history or st.session_state.history[0]["source"] != new_entry["source"]:
            st.session_state.history.insert(0, new_entry)

        # --- AFFICHAGE DES RÉSULTATS ---
        st.subheader("Résultat de la traduction :")
        st.success(result)
        
        col_audio, col_pdf = st.columns(2)
        
        with col_audio:
            st.write("🎵 Écouter :")
            try:
                audio_fp = text_to_speech(result, target_code)
                st.audio(audio_fp, format="audio/mp3")
            except Exception as e:
                st.error("Impossible de générer l'audio.")

        with col_pdf:
            st.write("📄 Document :")
            pdf_bytes = generate_pdf(result)
            st.download_button("📥 Télécharger en PDF", data=pdf_bytes, file_name="ma_traduction.pdf")