import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Sampah AI",
    page_icon="♻️",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #52b788 100%);
        border-radius: 16px; padding: 2.5rem 2rem 2rem; color: white; margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem; }
    .hero p  { font-size: 1rem; opacity: 0.85; margin: 0; }
    .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 6px rgba(0,0,0,0.07); margin-bottom: 1rem; }
    .metric-box { background: #f0faf4; border-left: 4px solid #2d6a4f; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
    .metric-label { font-size: 0.78rem; color: #555; margin-bottom: 0.15rem; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #1b4332; }
    .result-highlight { background: linear-gradient(135deg, #d8f3dc, #b7e4c7); border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; margin: 1rem 0; }
    .result-highlight h2 { color: #1b4332; margin: 0; font-size: 1.8rem; }
    .result-highlight p  { color: #2d6a4f; margin: 0.3rem 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Download model ───────────────────────────────────────────
MODEL_URLS = {
    'cnn_model.tflite': 'https://github.com/asshyariintanpermataningrum/UAS-AI-EDU_klasifikasi-sampah/releases/download/v1.0/cnn_model.tflite',
    'vgg_model.tflite': 'https://github.com/asshyariintanpermataningrum/UAS-AI-EDU_klasifikasi-sampah/releases/download/v1.0/vgg_model.tflite',
}

def download_model(filename, url):
    if not os.path.exists(filename):
        with st.spinner(f"⏳ Mengunduh {filename}..."):
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            progress = st.progress(0)
            downloaded = 0
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(downloaded / total, 1.0))
            progress.empty()

# ── Load TFLite model ────────────────────────────────────────
@st.cache_resource
def load_models():
    for fname, url in MODEL_URLS.items():
        download_model(fname, url)
    try:
        import tflite_runtime.interpreter as tflite
        cnn = tflite.Interpreter(model_path='cnn_model.tflite')
        vgg = tflite.Interpreter(model_path='vgg_model.tflite')
    except ImportError:
        import tensorflow as tf
        cnn = tf.lite.Interpreter(model_path='cnn_model.tflite')
        vgg = tf.lite.Interpreter(model_path='vgg_model.tflite')
    cnn.allocate_tensors()
    vgg.allocate_tensors()
    return cnn, vgg

def predict_tflite(interpreter, input_data):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ── Helpers ──────────────────────────────────────────────────
# PENTING: urutan kelas harus sama persis dengan class_indices dari training di Colab
# Hasil training: {'anorganik': 0, 'b3': 1, 'organik': 2}
CLASS_NAMES = ['anorganik', 'b3', 'organik']
CLASS_INFO  = {
    'organik':   {'icon':'🍃','desc':'Sampah yang dapat terurai secara alami.','contoh':'Sisa makanan, daun kering, kulit buah','badge':'badge-organik'},
    'anorganik': {'icon':'🧴','desc':'Sampah yang tidak mudah terurai secara alami.','contoh':'Plastik, kaleng, kaca, kertas','badge':'badge-anorganik'},
    'b3':        {'icon':'⚠️','desc':'Bahan Berbahaya dan Beracun — butuh penanganan khusus.','contoh':'Baterai, lampu neon, obat kadaluarsa, cat','badge':'badge-b3'},
}

# PERBAIKAN PENTING: kedua model (CNN dan VGG16) dilatih dengan input 224x224
# dan normalisasi sederhana (rescale 1./255), BUKAN preprocessing standar VGG16
# (mean subtraction ImageNet). Preprocessing di app HARUS sama persis dengan
# preprocessing saat training di Colab, supaya hasil prediksi konsisten/akurat.

def preprocess_cnn(img):
    img = img.resize((224, 224)).convert('RGB')   # diperbaiki dari 150x150 -> 224x224
    arr = np.array(img, dtype=np.float32) / 255.0  # normalisasi sama seperti training
    return np.expand_dims(arr, 0)

def preprocess_vgg(img):
    img = img.resize((224, 224)).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0  # diperbaiki: rescale 1/255, BUKAN mean subtraction
    return np.expand_dims(arr, 0)

def confidence_bar(probs, title):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = ['#4361ee','#f77f00','#2d6a4f']
    bars = ax.barh(CLASS_NAMES, probs*100, color=colors, height=0.5)
    ax.set_xlim(0, 110); ax.set_xlabel('Confidence (%)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f'{p*100:.1f}%', va='center', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig

# ── Load ─────────────────────────────────────────────────────
try:
    cnn_model, vgg_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Hero ─────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>♻️ Klasifikasi Sampah AI</h1><p>Unggah foto sampah — CNN & VGG16 akan mengklasifikasikannya secara otomatis</p></div>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"❌ Gagal memuat model: {load_error}")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Gambar", "📊 Perbandingan Model", "ℹ️ Info Dataset"])

# ════════════════════════════════════════════════════════════
with tab1:
    col_up, col_res = st.columns([1, 1.6], gap="large")
    with col_up:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Gambar")
        uploaded = st.file_uploader("Pilih gambar sampah", type=['jpg','jpeg','png'], label_visibility="collapsed")
        model_choice = st.radio("Gunakan model:", ["VGG16 (Lebih Akurat)", "CNN from Scratch", "Keduanya"], index=0)
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded:
            st.image(Image.open(uploaded), caption="Gambar yang diupload", use_container_width=True)

    with col_res:
        if uploaded:
            img = Image.open(uploaded)
            label_vgg = label_cnn = None

            if model_choice in ["VGG16 (Lebih Akurat)", "Keduanya"]:
                arr_vgg  = preprocess_vgg(img)
                probs_vgg = predict_tflite(vgg_model, arr_vgg)
                label_vgg = CLASS_NAMES[np.argmax(probs_vgg)]
                info = CLASS_INFO[label_vgg]
                st.markdown(f'<div class="result-highlight"><h2>{info["icon"]} {label_vgg.upper()}</h2><p>VGG16 · Confidence: {max(probs_vgg)*100:.1f}%</p></div>', unsafe_allow_html=True)
                st.pyplot(confidence_bar(probs_vgg, "Confidence Score — VGG16"))

            if model_choice in ["CNN from Scratch", "Keduanya"]:
                arr_cnn  = preprocess_cnn(img)
                probs_cnn = predict_tflite(cnn_model, arr_cnn)
                label_cnn = CLASS_NAMES[np.argmax(probs_cnn)]
                info_cnn = CLASS_INFO[label_cnn]
                st.markdown(f'<div class="result-highlight"><h2>{info_cnn["icon"]} {label_cnn.upper()}</h2><p>CNN from Scratch · Confidence: {max(probs_cnn)*100:.1f}%</p></div>', unsafe_allow_html=True)
                st.pyplot(confidence_bar(probs_cnn, "Confidence Score — CNN from Scratch"))

            if model_choice == "Keduanya" and label_cnn and label_vgg:
                if label_cnn == label_vgg:
                    st.success("✅ Kedua model sepakat pada hasil prediksi yang sama!")
                else:
                    st.warning("⚠️ Kedua model memberikan prediksi berbeda.")

            display_label = label_vgg if "VGG16" in model_choice else label_cnn
            if display_label:
                info = CLASS_INFO[display_label]
                st.markdown(f'<div class="card"><b>{info["icon"]} Tentang sampah {display_label}</b><br><i>{info["desc"]}</i><br><b>Contoh:</b> {info["contoh"]}</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Upload gambar sampah untuk mulai prediksi.")

# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Perbandingan Performa CNN vs VGG16")
    # Angka di bawah ini diambil dari hasil evaluasi nyata pada data TEST
    # (lihat notebook Colab, bagian Evaluasi Model)
    metrics = {'Accuracy':(73.91,91.30),'Precision':(80.56,93.33),'Recall':(75.00,90.48),'F1-Score':(74.01,90.74)}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>CNN from Scratch</h4>', unsafe_allow_html=True)
        for m,(v,_) in metrics.items():
            st.markdown(f'<div class="metric-box"><div class="metric-label">{m}</div><div class="metric-value">{v:.2f}%</div></div>', unsafe_allow_html=True)
        st.markdown("**Waktu training:** ± 3.5 menit (43 epoch)</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>VGG16 Transfer Learning</h4>', unsafe_allow_html=True)
        for m,(_,v) in metrics.items():
            st.markdown(f'<div class="metric-box"><div class="metric-label">{m}</div><div class="metric-value">{v:.2f}%</div></div>', unsafe_allow_html=True)
        st.markdown("**Waktu training:** ± 2 menit (30 epoch)</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9,4))
    x = np.arange(len(metrics)); w = 0.35
    b1 = ax.bar(x-w/2,[v[0] for v in metrics.values()],w,label='CNN from Scratch',color='#4361ee')
    b2 = ax.bar(x+w/2,[v[1] for v in metrics.values()],w,label='VGG16 Transfer Learning',color='#2d6a4f')
    ax.set_ylim(0,105); ax.set_xticks(x); ax.set_xticklabels(metrics.keys())
    ax.set_ylabel('Score (%)'); ax.set_title('Perbandingan Metrik Evaluasi',fontweight='bold')
    ax.legend(); ax.grid(axis='y',alpha=0.3); ax.spines[['top','right']].set_visible(False)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{bar.get_height():.2f}%',ha='center',va='bottom',fontsize=8)
    plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="card">💡 <b>Analisis:</b> VGG16 mengungguli CNN di semua metrik (Accuracy 91.30% vs 73.91%) berkat fitur visual dari ImageNet yang sudah teroptimasi, sekaligus waktu training lebih singkat. CNN from Scratch tetap solid (~74-80%) meski dilatih dari nol dengan dataset kecil (150 foto). Untuk kasus dataset terbatas seperti ini, Transfer Learning sangat direkomendasikan.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ Informasi Dataset")
    col_a, col_b, col_c = st.columns(3)
    for col, (kelas, info) in zip([col_a,col_b,col_c], CLASS_INFO.items()):
        with col:
            st.markdown(f'<div class="card"><h2 style="margin:0">{info["icon"]}</h2><b>{kelas.upper()}</b><br><br><i>{info["desc"]}</i><br><b>Contoh:</b> {info["contoh"]}</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(7,3))
    bars = ax2.bar(['Train','Validasi','Test'],[105,22,23],color=['#2d6a4f','#52b788','#b7e4c7'],width=0.4)
    ax2.set_ylabel('Jumlah Gambar'); ax2.set_title('Distribusi Dataset (Total: 150 gambar)',fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)
    for bar in bars:
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,str(int(bar.get_height())),ha='center',fontweight='bold')
    plt.tight_layout(); st.pyplot(fig2)

    st.markdown('<div class="card">🏫 <b>Konteks Pendidikan:</b> Dataset dibuat sendiri dengan memfoto langsung sampah di lingkungan sekolah/kampus. Klasifikasi sampah menjadi <b>organik</b>, <b>anorganik</b>, dan <b>B3</b> bertujuan membantu warga sekolah memilah sampah dengan benar.</div>', unsafe_allow_html=True)
