"""
Streamlit frontend for Neural Machine Translation
Interactive web interface for translation and visualization
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Neural Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .baseline-badge {
        background-color: #3b82f6;
        color: white;
    }
    .attention-badge {
        background-color: #8b5cf6;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API Configuration
# ============================================================================

API_URL = config.API_ENDPOINT

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def call_translate_api(text, model_type):
    """Call translation API"""
    try:
        response = requests.post(
            f"{API_URL}/translate",
            json={"text": text, "model_type": model_type},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def call_compare_api(text):
    """Call comparison API"""
    try:
        response = requests.post(
            f"{API_URL}/compare",
            params={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None


def plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens):
    """Create interactive attention heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=src_tokens,
        y=tgt_tokens,
        colorscale='YlOrRd',
        text=np.array(attention_weights).round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Weight"),
        hoverongaps=False,
        hovertemplate='Source: %{x}<br>Target: %{y}<br>Weight: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Attention Weights Heatmap",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#667eea'}
        },
        xaxis_title="Source Tokens (English)",
        yaxis_title="Target Tokens (French)",
        height=max(400, len(tgt_tokens) * 50),
        width=max(600, len(src_tokens) * 60),
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=-45)  # FIXED: Set tickangle in layout instead
    )
    
    return fig


# ============================================================================
# Session State Initialization
# ============================================================================

if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

# ============================================================================
# Main App
# ============================================================================

# Header
st.markdown('<div class="main-header">🌐 Neural Machine Translation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">English → French Translation with LSTM & Attention</div>', unsafe_allow_html=True)

# Check API status
api_healthy = check_api_health()

if not api_healthy:
    st.error("⚠️ Cannot connect to backend API. Please ensure the FastAPI server is running.")
    st.info(f"Start the server with: `cd backend && uvicorn main:app --reload --port {config.API_PORT}`")
    st.stop()

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_type = st.radio(
        "Select Model",
        ["attention", "baseline"],
        format_func=lambda x: "🟣 LSTM with Attention" if x == "attention" else "🔵 Baseline LSTM",
        help="Choose between baseline LSTM and attention-based model"
    )
    
    st.markdown("---")
    
    # Show model info
    st.subheader("📊 Model Information")
    model_info = get_model_info()
    
    if model_info and model_type in model_info:
        info = model_info[model_type]
        st.metric("Parameters", f"{info['total_parameters']:,}")
        st.metric("Vocab Size (EN)", f"{info['vocab_size_en']:,}")
        st.metric("Vocab Size (FR)", f"{info['vocab_size_fr']:,}")
    
    st.markdown("---")
    
    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    This app demonstrates:
    
    **Baseline LSTM**
    - Fixed context vector
    - Limited on long sentences
    - Bottleneck problem
    
    **LSTM with Attention**
    - Dynamic context vectors
    - Better on long sentences
    - Visualizable attention
    """)
    
    st.markdown("---")
    
    # Comparison mode
    comparison_mode = st.checkbox("🔀 Comparison Mode", 
                                  help="Translate with both models side-by-side")

# ============================================================================
# Main Content
# ============================================================================

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔤 Translation", "📊 Comparison", "📈 Analytics"])

# ============================================================================
# TAB 1: Translation
# ============================================================================

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input (English)")
        input_text = st.text_area(
            "Enter English text to translate:",
            height=200,
            placeholder="Type something in English...",
            help="Enter an English sentence to translate to French",
            key="input_text_tab1"
        )
        
        translate_button = st.button(
            "🔄 Translate",
            type="primary",
            use_container_width=True,
            key="translate_button_tab1"
        )
    
    with col2:
        st.subheader("🇫🇷 Output (French)")
        output_placeholder = st.empty()
    
    # Handle translation
    if translate_button and input_text.strip():
        with st.spinner(f"Translating with {model_type} model..."):
            result = call_translate_api(input_text.strip(), model_type)
            
            if result:
                # Display translation
                with output_placeholder:
                    st.success(result['translation'])
                
                # Add to history
                st.session_state.translation_history.insert(0, {
                    'source': input_text.strip(),
                    'translation': result['translation'],
                    'model': model_type
                })
                
                # Keep only last 10
                st.session_state.translation_history = st.session_state.translation_history[:10]
                
                # Show attention visualization
                if model_type == "attention" and result.get('attention_weights'):
                    st.markdown("---")
                    st.subheader("🎯 Attention Visualization")
                    
                    attention = np.array(result['attention_weights'])
                    src_tokens = result['source_tokens']
                    tgt_tokens = result['target_tokens']
                    
                    # Create heatmap
                    fig = plot_attention_heatmap(attention, src_tokens, tgt_tokens)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.info("""
                    💡 **How to read this heatmap:**
                    - Each row shows which source words the model focused on when generating that target word
                    - Brighter colors (red/orange) indicate higher attention weights
                    - You should see a roughly diagonal pattern for good word alignments
                    - The model dynamically adjusts its attention for each output word
                    """)
    
    elif translate_button:
        st.warning("⚠️ Please enter some text to translate!")
    
    # Example sentences
    st.markdown("---")
    st.subheader("📋 Example Sentences")
    
    examples = [
        "Hello, how are you?",
        "I love learning about artificial intelligence.",
        "The weather is beautiful today.",
        "What is your name?",
        "Machine translation is fascinating."
    ]
    
    cols = st.columns(len(examples))
    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"Try #{i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_selected = example
                st.rerun()
    
    # Handle example selection
    if hasattr(st.session_state, 'example_selected'):
        st.info(f"💡 Selected: {st.session_state.example_selected}")
        del st.session_state.example_selected

# ============================================================================
# TAB 2: Comparison
# ============================================================================

with tab2:
    st.subheader("🔀 Compare Both Models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        compare_text = st.text_area(
            "Enter text to compare:",
            height=150,
            placeholder="Type an English sentence...",
            key="compare_text"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        compare_button = st.button(
            "⚖️ Compare Models",
            type="primary",
            use_container_width=True,
            key="compare_button"
        )
    
    if compare_button and compare_text.strip():
        with st.spinner("Comparing models..."):
            results = call_compare_api(compare_text.strip())
            
            if results:
                st.markdown("---")
                
                # Show source
                st.markdown(f"**📝 Source:** {results['source']}")
                st.markdown("")
                
                # Create comparison columns
                col_base, col_attn = st.columns(2)
                
                # Baseline result
                with col_base:
                    st.markdown('<div class="model-badge baseline-badge">🔵 Baseline LSTM</div>', 
                              unsafe_allow_html=True)
                    if 'baseline' in results and 'translation' in results['baseline']:
                        st.info(results['baseline']['translation'])
                    else:
                        st.error("Baseline model not available")
                
                # Attention result
                with col_attn:
                    st.markdown('<div class="model-badge attention-badge">🟣 LSTM + Attention</div>',
                              unsafe_allow_html=True)
                    if 'attention' in results and 'translation' in results['attention']:
                        st.success(results['attention']['translation'])
                    else:
                        st.error("Attention model not available")
                
                # Show attention heatmap
                if 'attention' in results and results['attention'].get('attention_weights'):
                    st.markdown("---")
                    st.subheader("🎯 Attention Weights")
                    
                    attention = np.array(results['attention']['attention_weights'])
                    src_tokens = results['attention']['source_tokens']
                    tgt_tokens = results['attention']['target_tokens']
                    
                    fig = plot_attention_heatmap(attention, src_tokens, tgt_tokens)
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: Analytics
# ============================================================================

with tab3:
    st.subheader("📈 Translation History & Analytics")
    
    if st.session_state.translation_history:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.translation_history)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Translations", len(df))
        
        with col2:
            baseline_count = len(df[df['model'] == 'baseline'])
            st.metric("Baseline Used", baseline_count)
        
        with col3:
            attention_count = len(df[df['model'] == 'attention'])
            st.metric("Attention Used", attention_count)
        
        st.markdown("---")
        
        # Show history table
        st.subheader("Recent Translations")
        
        # Format table
        display_df = df.copy()
        display_df['model'] = display_df['model'].apply(
            lambda x: '🔵 Baseline' if x == 'baseline' else '🟣 Attention'
        )
        display_df.columns = ['Source (EN)', 'Translation (FR)', 'Model']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.translation_history = []
            st.rerun()
    
    else:
        st.info("📭 No translations yet. Start translating to see analytics!")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Neural Machine Translation Project</strong></p>
    <p>Built with PyTorch • FastAPI • Streamlit</p>
    <p>Demonstrating LSTM and Attention mechanisms for sequence-to-sequence translation</p>
</div>
""", unsafe_allow_html=True)