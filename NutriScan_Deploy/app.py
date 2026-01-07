import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import json
import os
from datetime import datetime
import pytz
import plotly.graph_objects as go

# config
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🥗",
    layout="wide"
)

# css page design
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #1e3a1f, #0a0f0a);
        color: #e0e0e0;
    }

    /* macro card */
    .macro-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .macro-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
    }
    
    .macro-card h4 {
        color: #4ade80 !important;
        margin: 0;
        font-size: 20px;
    }
    
    .macro-card p {
        color: #94a3b8 !important;
        margin: 0;
        font-size: 12px;
        text-transform: uppercase;
    }

    /* button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        border: none;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.4);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# load data and model
@st.cache_data
def load_nutrition_data():
    if not os.path.exists('nutrition_data.json'): return {}
    with open('nutrition_data.json', 'r') as f: return json.load(f)

NUTRITION_DB = load_nutrition_data()

LABELS = [
    "BEEF", "BERRIES", "CHICKEN", "COOKING_VEGS", "EGGS",
    "FISH", "HIGH SUGAR FRUITS", "LEGUMES", "LEAFY GREENS",
    "OATMEAL/CEREALS", "PIZZA", "POTATOES", "PORK", "RICE"
]

@st.cache_resource
def load_model():
    model_path = "./model"
    if not os.path.exists(model_path): return None, None
    try:
        return AutoImageProcessor.from_pretrained(model_path, use_fast=True), AutoModelForImageClassification.from_pretrained(model_path)
    except: return None, None

processor, model = load_model()

# ui layout
col1, col2 = st.columns([2, 1])
with col1:
    st.title("NutriScan AI 🥗")
    st.markdown("### *Smart Nutrition for a Healthier You*")

uploaded_file = st.file_uploader("📸 Snap or Upload your meal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    c1, c2 = st.columns([1, 1.5])
    image = Image.open(uploaded_file).convert('RGB')
    with c1:
        st.image(image, width='stretch')
    with c2:
        analyze_btn = st.button('🔍 Analyze Nutrition', width='stretch')
        if analyze_btn:
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner('Calculating macros...'):
                    try:
                        # inference
                        inputs = processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model(**inputs)

                        predicted_class_idx = outputs.logits.argmax(-1).item()
                        predicted_label = LABELS[predicted_class_idx] if predicted_class_idx < len(LABELS) else "Unknown"
                        predicted_label_clean = predicted_label.upper().replace("_", " ")

                        # results
                        st.success(f"**Identified:** {predicted_label_clean}")

                        macros = NUTRITION_DB.get(predicted_label_clean)

                        if macros:
                            #adjust to utc+1 timezone
                            local_tz = pytz.timezone('Europe/Berlin')
                            local_time = datetime.now(local_tz)
                            # save to session history
                            st.session_state.history.append({
                                "label": predicted_label_clean,
                                "calories": macros['calories'],
                                "protein": macros['protein'],
                                "carbs": macros['carbs'],
                                "fat": macros['fat'],
                                "time": local_time.strftime("%H:%M")
                            })

                            # macro boxes
                            st.markdown(f"""
                            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                                <div class="macro-card" style="flex: 1;">
                                    <h4>🔥 {macros['calories']}</h4>
                                    <p>Calories</p>
                                </div>
                                <div class="macro-card" style="flex: 1;">
                                    <h4>💪 {macros['protein']}g</h4>
                                    <p>Protein</p>
                                </div>
                                <div class="macro-card" style="flex: 1;">
                                    <h4>🍞 {macros['carbs']}g</h4>
                                    <p>Carbs</p>
                                </div>
                                <div class="macro-card" style="flex: 1;">
                                    <h4>🥑 {macros['fat']}g</h4>
                                    <p>Fat</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # donut chart
                            labels = ['Carbs', 'Fat', 'Protein']
                            values = [macros['carbs'], macros['fat'], macros['protein']]
                            colors = ['#60a5fa', '#f87171', '#4ade80']

                            fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=values,
                                hole=.6,
                                textinfo='label+percent',
                                marker=dict(colors=colors, line=dict(color='#0f172a', width=3))
                            )])

                            fig.update_layout(
                                font=dict(color="white"),
                                title_text="Macro Ratio",
                                title_font_color="white",
                                showlegend=False,
                                margin=dict(t=40, b=100, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=350
                            )
                            st.plotly_chart(fig, config = {'displayModeBar': False})
                        else:
                            st.warning(f"No nutritional info found for {predicted_label_clean}")
                    except Exception as e:
                        st.error(f"Error: {e}")

# sidebar for history
with st.sidebar:
    st.title("Meal History")

    if not st.session_state.history:
        st.info("No meals scanned yet.")
    else:
        for item in reversed(st.session_state.history):
            with st.expander(f"🥗 {item['label']} - {item['time']}"):
                st.markdown(f"### 🔥 {item['calories']} kcal")
                st.write(f"**Protein:** {item['protein']}g")
                st.write(f"**Carbs:** {item['carbs']}g")
                st.write(f"**Fat:** {item['fat']}g")

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()