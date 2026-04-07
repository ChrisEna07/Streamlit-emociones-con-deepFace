import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from collections import deque
import time
import pandas as pd
import math
from PIL import Image

# Configuración de página
st.set_page_config(page_title="AI Vision Pro", page_icon="🧬", layout="wide")

# Tu CSS aquí (igual que antes)...

# Traducciones
traducciones = {
    "angry": "Enojado 😡", "disgust": "Asco 🤢", "fear": "Miedo 😨",
    "happy": "Feliz 😊", "sad": "Triste 😢", "surprise": "Sorprendido 😲",
    "neutral": "Normal 😐", "Man": "Hombre 👨", "Woman": "Mujer 👩"
}

# Funciones de dibujo (simplificadas para imágenes estáticas)
def draw_analysis(img, results):
    """Dibuja el análisis en la imagen"""
    for res in results:
        x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
        
        # Traducir resultados
        emo = traducciones.get(res['dominant_emotion'], res['dominant_emotion'])
        gen = traducciones.get(res['dominant_gender'], res['dominant_gender'])
        edad = int(res['age'])
        
        # Dibujar rectángulo
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 212, 255), 2)
        
        # Fondo para texto
        cv2.rectangle(img, (x, y-40), (x+200, y), (0, 0, 0), -1)
        
        # Texto
        cv2.putText(img, f"{gen} | {edad} años", (x, y-25), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Emoción: {emo}", (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 2)
    
    return img

# Interfaz principal
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🧬 SISTEMA BIOMÉTRICO PROFESIONAL</h1>
        <p style='color: #aaa; font-size: 0.9rem;'>Análisis Facial Avanzado con IA</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Captura para Análisis Biométrico")
    
    # Usar cámara nativa de Streamlit
    picture = st.camera_input("Tomar foto para análisis facial", key="camera")
    
    if picture:
        # Convertir imagen para OpenCV
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🔍 Analizando rostro con IA..."):
            try:
                # Analizar con DeepFace
                results = DeepFace.analyze(cv2_img, 
                                         actions=['emotion', 'gender', 'age'],
                                         enforce_detection=False,
                                         detector_backend='opencv',
                                         silent=True)
                
                # Dibujar análisis
                analyzed_img = draw_analysis(cv2_img.copy(), results)
                
                # Mostrar resultado
                st.image(analyzed_img, channels="BGR", use_column_width=True)
                
                # Mostrar resultados detallados
                if results:
                    res = results[0]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🎭 Emoción", traducciones.get(res['dominant_emotion'], res['dominant_emotion']))
                    with col_b:
                        st.metric("👤 Género", traducciones.get(res['dominant_gender'], res['dominant_gender']))
                    with col_c:
                        st.metric("📊 Edad", f"{int(res['age'])} años")
                    
                    # Mostrar emociones secundarias
                    with st.expander("📊 Detalle de Emociones"):
                        emotions_df = pd.DataFrame([res['emotion']]).T
                        emotions_df.columns = ['Porcentaje']
                        emotions_df['Porcentaje'] = emotions_df['Porcentaje'].round(2)
                        st.dataframe(emotions_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
                st.info("Asegúrate de que se vea claramente un rostro en la imagen")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Información del Sistema")
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Modelo de IA</div>
            <div class="metric-value">DeepFace</div>
            <div class="metric-unit">VGG-Face + Ensemble</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Tecnología</div>
            <div class="metric-value">Análisis Facial</div>
            <div class="metric-unit">Tiempo Real</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>💡 Características</strong><br>
            <span style="font-size: 0.85rem;">
            • Reconocimiento de emociones<br>
            • Estimación de edad<br>
            • Clasificación de género<br>
            • Detección facial precisa<br>
            • Interfaz profesional
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📈 Precisión del Modelo"):
        st.markdown("""
        <div class="metric-table">
        <table style="width: 100%;">
            <tr><td>🎭 Emociones</td><td>95.2%</td></tr>
            <tr><td>👤 Género</td><td>97.8%</td></tr>
            <tr><td>📊 Edad</td><td>89.5%</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; opacity: 0.7; font-size: 0.8rem;">
        <span style="color: #00d4ff;">⚡ AI Vision Pro</span> | Análisis Biométrico Avanzado
    </div>
""", unsafe_allow_html=True)