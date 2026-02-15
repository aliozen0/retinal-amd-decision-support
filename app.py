"""
Retinal AMD Klinik Karar Destek Paneli
=======================================
EfficientNet-B4 modeli ile retinal OCT görüntülerinden patoloji tespiti,
Grad-CAM görselleştirmesi ve interaktif olasılık grafikleri sunan,
otomatik klinik rapor üreten Streamlit tabanlı web arayüzü.

Kullanım:
    streamlit run app.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone, timedelta

# Turkiye saat dilimi (GMT+3)
TZ_TR = timezone(timedelta(hours=3))

# ============================================================================
# Proje modülleri
# ============================================================================
from models import (
    MODEL_OPTIONS,
    DISABLED_MODELS,
    load_model,
    get_target_layer,
    get_classes,
)
from utils.preprocessing import preprocess_image, prepare_display_image
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.reporting import generate_clinical_report
from utils.pdf_export import generate_pdf_report

# ============================================================================
# Sayfa konfigürasyonu
# ============================================================================
st.set_page_config(
    page_title="Retinal AMD | Klinik Karar Destek Paneli",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# Premium CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* ── Hero ── */
    .hero-banner {
        background: linear-gradient(135deg, #020617 0%, #0f172a 30%, #1e1b4b 60%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 80px rgba(99, 102, 241, 0.08), 0 20px 60px rgba(0,0,0,0.3);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff, #a5b4fc, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; position: relative; z-index: 1;
    }
    .hero-subtitle {
        color: #94a3b8; font-size: 1rem; margin: 0.4rem 0 0 0;
        font-weight: 400; position: relative; z-index: 1;
    }
    .hero-badges {
        margin-top: 0.8rem; position: relative; z-index: 1;
        display: flex; gap: 0.5rem; flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(14,165,233,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 100px; padding: 0.25rem 0.8rem;
        font-size: 0.72rem; color: #a5b4fc; font-weight: 500; letter-spacing: 0.5px;
    }

    /* ── Stat Grid ── */
    .stat-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 1rem; margin-bottom: 1.5rem;
    }
    .stat-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,27,75,0.6));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px; padding: 1.1rem 1.3rem;
        backdrop-filter: blur(20px); transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.1);
    }
    .stat-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
    .stat-value { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin: 0; }
    .stat-label {
        font-size: 0.72rem; color: #64748b; margin: 0.15rem 0 0 0;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px;
    }

    /* ── Upload Zone ── */
    .upload-zone {
        background: linear-gradient(145deg, rgba(15,23,42,0.6), rgba(30,27,75,0.3));
        border: 2px dashed rgba(99,102,241,0.25);
        border-radius: 20px; padding: 3rem 2rem; text-align: center;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(99,102,241,0.5);
        background: linear-gradient(145deg, rgba(15,23,42,0.8), rgba(30,27,75,0.4));
    }
    .upload-zone .icon { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
    .upload-zone h3 { color: #e2e8f0; font-size: 1.2rem; margin: 0 0 0.4rem 0; }
    .upload-zone p { color: #64748b; font-size: 0.9rem; margin: 0; }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.7rem;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(99,102,241,0.12);
    }
    .section-header h2 { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 0; }

    /* ── Image Cards ── */
    .image-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px; padding: 1rem; backdrop-filter: blur(20px);
    }
    .image-card-title {
        font-size: 0.8rem; font-weight: 600; color: #a5b4fc;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;
    }

    /* ── Result Chip ── */
    .result-chip {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.7rem 1.3rem; border-radius: 100px;
        font-weight: 700; font-size: 1.05rem; margin: 0.5rem 0;
    }
    .result-chip.pathology {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
        border: 1px solid rgba(239,68,68,0.3); color: #fca5a5;
    }
    .result-chip.normal {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
        border: 1px solid rgba(34,197,94,0.3); color: #86efac;
    }

    /* ── Report Card ── */
    .report-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,27,75,0.4));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px; padding: 1.5rem; margin-top: 1rem;
    }
    .report-card h3 { color: #a5b4fc; font-size: 1rem; margin: 0 0 0.8rem 0; }

    /* ── Steps ── */
    .steps-container {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 1rem; margin-top: 1.5rem;
    }
    .step-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.8), rgba(30,27,75,0.4));
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 14px; padding: 1.2rem; text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: rgba(99,102,241,0.3);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.08);
    }
    .step-number {
        width: 34px; height: 34px; border-radius: 50%;
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white; font-weight: 700; font-size: 0.9rem;
        display: inline-flex; align-items: center; justify-content: center;
        margin-bottom: 0.7rem;
    }
    .step-card h4 { color: #e2e8f0; font-size: 0.9rem; margin: 0 0 0.3rem 0; }
    .step-card p { color: #64748b; font-size: 0.78rem; margin: 0; line-height: 1.4; }

    /* ── Footer ── */
    .footer-bar {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.08);
        border-radius: 12px; padding: 0.8rem 1.2rem; margin-top: 2rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .footer-bar p { color: #475569; font-size: 0.72rem; margin: 0; }

    /* ── Coming Soon Tag ── */
    .coming-soon-tag {
        display: inline-block;
        background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.08));
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 8px; padding: 0.2rem 0.6rem;
        font-size: 0.65rem; color: #fbbf24; font-weight: 600;
    }

    /* ── Sidebar Info ── */
    .sidebar-info {
        background: linear-gradient(145deg, rgba(99,102,241,0.08), rgba(14,165,233,0.05));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px; padding: 0.9rem; margin: 0.6rem 0;
    }
    .sidebar-info p { font-size: 0.78rem !important; line-height: 1.5 !important; margin: 0 !important; }
    .sidebar-info strong { color: #a5b4fc !important; }
    .sidebar-logo { text-align: center; padding: 1.2rem 0; border-bottom: 1px solid rgba(99,102,241,0.1); margin-bottom: 1.2rem; }
    .sidebar-logo h2 { background: linear-gradient(135deg, #e0e7ff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.3rem; font-weight: 800; margin: 0; }
    .sidebar-logo p { color: #64748b; font-size: 0.7rem; margin: 0.2rem 0 0 0; text-transform: uppercase; letter-spacing: 2px; font-weight: 500; }

    /* ── Pulse Button ── */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
        70% { box-shadow: 0 0 0 12px rgba(99,102,241,0); }
        100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #6366f1, #818cf8) !important;
        border: none !important; border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important; font-weight: 700 !important;
        font-size: 0.95rem !important; animation: pulse-glow 2s infinite;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.3) !important;
    }

    /* ── Override ── */
    .stDivider { border-color: rgba(99,102,241,0.08) !important; }
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, rgba(15,23,42,0.6), rgba(30,27,75,0.3));
        border: 1px solid rgba(99,102,241,0.1) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Cihaz tespiti
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = str(DEVICE)


def create_confidence_chart(
    class_names: list,
    probabilities: np.ndarray,
    predicted_class: str,
) -> go.Figure:
    """Premium Plotly güven grafiği."""
    colors = []
    for name in class_names:
        if name == predicted_class:
            colors.append("rgba(99, 102, 241, 0.9)")
        else:
            colors.append("rgba(51, 65, 85, 0.6)")

    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities * 100,
            marker=dict(
                color=colors,
                line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
            ),
            text=[f"%{p*100:.1f}" for p in probabilities],
            textposition="outside",
            textfont=dict(color="#a5b4fc", size=14, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Güven: <b>%{y:.1f}%</b><extra></extra>",
        )
    ])

    fig.update_layout(
        title=dict(text="Sınıf Güven Dağılımı", font=dict(color="#94a3b8", size=14), x=0.0),
        xaxis=dict(color="#64748b", gridcolor="rgba(99,102,241,0.05)",
                   tickfont=dict(size=13, color="#94a3b8")),
        yaxis=dict(color="#64748b", gridcolor="rgba(99,102,241,0.05)", range=[0, 110],
                   tickfont=dict(size=11, color="#475569"),
                   title=dict(text="Güven (%)", font=dict(size=11, color="#475569"))),
        plot_bgcolor="rgba(2,6,23,0.5)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=350,
        margin=dict(l=50, r=20, t=50, b=50), bargap=0.35,
    )
    return fig


def run_inference(model, input_tensor, model_type):
    """Model üzerinde çıkarım yapar."""
    classes = get_classes(model_type)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = classes[predicted_idx]
    return predicted_class, predicted_idx, probabilities


# ============================================================================
# SIDEBAR — Sadece ayarlar (katlanabilir)
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🧬 Retinal AMD</h2>
        <p>Ayarlar</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧠 Model Seçimi")
    selected_model_label = st.selectbox(
        "Model", options=list(MODEL_OPTIONS.keys()), index=0, label_visibility="collapsed",
    )
    model_type = MODEL_OPTIONS[selected_model_label]

    if model_type in DISABLED_MODELS:
        st.markdown("""
        <div class="sidebar-info">
            <p>🔒 <strong>Bu model yakında aktifleştirilecektir.</strong><br>
            Şu an EfficientNet-B4 kullanılmaktadır.</p>
        </div>
        """, unsafe_allow_html=True)
        model_type = "efficientnet_b4"

    st.markdown("---")
    with st.expander("📋 Model Detayları"):
        st.markdown("""
        <div class="sidebar-info">
            <p><strong>EfficientNet-B4</strong><br>
            • 4 Sınıf: CNV, DME, DRUSEN, NORMAL<br>
            • Compound Scaling · %99+ doğruluk</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-info">
            <p><strong>Swin-V2 + SupCon</strong> <span class="coming-soon-tag">YAKINDA</span><br>
            • 3 Sınıf: AMD, DME, NORMAL<br>
            • Supervised Contrastive Learning</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    device_icon = "🟢 GPU" if DEVICE.type == "cuda" else "🔵 CPU"
    st.caption(f"Cihaz: **{device_icon}** · v1.0.0")

# ============================================================================
# ANA PANEL
# ============================================================================

# Hero
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🧬 Retinal AMD</h1>
    <p class="hero-subtitle">Retinal OCT Görüntülerinden Yapay Zekâ Destekli Patoloji Tespiti</p>
    <div class="hero-badges">
        <span class="hero-badge">🔬 EfficientNet-B4</span>
        <span class="hero-badge">🔥 Grad-CAM</span>
        <span class="hero-badge">📋 Klinik Rapor</span>
        <span class="hero-badge">📄 PDF Dışa Aktarma</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model yükleme
with st.spinner("⏳ Model yükleniyor..."):
    model, is_demo_mode = load_model(model_type, DEVICE_STR)

# Stat kartları
classes = get_classes(model_type)
model_display_name = "EfficientNet-B4" if model_type == "efficientnet_b4" else "Swin-V2"
now = datetime.now(TZ_TR).strftime("%d.%m.%Y")

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <p class="stat-value">{model_display_name}</p>
        <p class="stat-label">Aktif Model</p>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🏷️</div>
        <p class="stat-value">{len(classes)} Sınıf</p>
        <p class="stat-label">Tanı Kapasitesi</p>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📅</div>
        <p class="stat-value">{now}</p>
        <p class="stat-label">Analiz Tarihi</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# GÖRÜNTÜ YÜKLEME — Ana panelde
# ============================================================================
st.markdown("""
<div class="section-header">
    <span class="icon">📤</span>
    <h2>Görüntü Yükle & Analiz Et</h2>
</div>
""", unsafe_allow_html=True)

col_upload, col_button = st.columns([3, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Retinal OCT görüntüsü seçin (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

with col_button:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button(
        "🔬  Analizi Başlat",
        use_container_width=True,
        type="primary",
        disabled=uploaded_file is None,
    )

# ============================================================================
# ANALİZ AKIŞI
# ============================================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Görüntü yüklenirken hata oluştu: {e}")
        st.stop()

    # Ön izleme
    if not analyze_button:
        col_preview, col_info = st.columns([1, 1], gap="large")
        with col_preview:
            st.markdown('<div class="image-card"><div class="image-card-title">🖼️ Ön İzleme</div></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div class="image-card" style="padding:1.5rem;">
                <div class="image-card-title">📊 Görüntü Bilgileri</div>
                <table style="width:100%; color:#cbd5e1; font-size:0.88rem;">
                    <tr><td style="color:#64748b; padding:0.5rem 0;">Dosya</td>
                        <td style="text-align:right; color:#a5b4fc; font-weight:500;">{uploaded_file.name}</td></tr>
                    <tr><td style="color:#64748b; padding:0.5rem 0;">Çözünürlük</td>
                        <td style="text-align:right; color:#a5b4fc; font-weight:500;">{image.size[0]}×{image.size[1]} px</td></tr>
                    <tr><td style="color:#64748b; padding:0.5rem 0;">Format</td>
                        <td style="text-align:right; color:#a5b4fc; font-weight:500;">{image.mode}</td></tr>
                    <tr><td style="color:#64748b; padding:0.5rem 0;">Model</td>
                        <td style="text-align:right; color:#a5b4fc; font-weight:500;">{model_display_name}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ── ANALİZ ──
    if analyze_button:
        with st.spinner("🔬 Analiz yapılıyor..."):
            input_tensor = preprocess_image(image, DEVICE)
            display_image = prepare_display_image(image)
            predicted_class, predicted_idx, probabilities = run_inference(
                model, input_tensor, model_type
            )

            try:
                target_layer = get_target_layer(model, model_type)
                with torch.enable_grad():
                    heatmap = generate_gradcam(model, input_tensor, predicted_idx, target_layer)
                overlaid_image = overlay_gradcam(display_image, heatmap, alpha=0.5)
                gradcam_success = True
            except Exception as e:
                st.warning(f"⚠️ Grad-CAM hatası: {e}")
                overlaid_image = display_image
                heatmap = np.zeros((224, 224), dtype=np.float32)
                gradcam_success = False

            is_swin = model_type == "swin_v2"
            report = generate_clinical_report(
                model_name=model_display_name,
                predicted_class=predicted_class,
                confidence=float(probabilities[predicted_idx]),
                is_swin_v2=is_swin,
            )

        # ════════════════════════════════════════════════
        # SONUÇLAR
        # ════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">
            <span class="icon">🔬</span>
            <h2>Analiz Sonuçları</h2>
        </div>
        """, unsafe_allow_html=True)

        # Sonuç chip
        is_normal = predicted_class == "NORMAL"
        chip_class = "normal" if is_normal else "pathology"
        chip_icon = "✅" if is_normal else "🔴"
        conf_pct = float(probabilities[predicted_idx]) * 100

        st.markdown(f"""
        <div class="result-chip {chip_class}">
            {chip_icon} {predicted_class} — %{conf_pct:.1f} Güven
        </div>
        """, unsafe_allow_html=True)

        # Görüntüler
        col_orig, col_gradcam = st.columns([1, 1], gap="large")
        with col_orig:
            st.markdown('<div class="image-card"><div class="image-card-title">🖼️ Orijinal Görüntü</div></div>', unsafe_allow_html=True)
            st.image(display_image, use_container_width=True)
        with col_gradcam:
            st.markdown(f'<div class="image-card"><div class="image-card-title">🔥 Grad-CAM · {predicted_class}</div></div>', unsafe_allow_html=True)
            if gradcam_success:
                st.image(overlaid_image, use_container_width=True)
            else:
                st.image(display_image, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Olasılık Grafiği
        fig = create_confidence_chart(classes, probabilities, predicted_class)
        st.plotly_chart(fig, use_container_width=True)

        # Klinik Rapor
        st.markdown('<div class="report-card"><h3>📋 Klinik Analiz Raporu</h3></div>', unsafe_allow_html=True)
        if is_normal:
            st.success(report)
        elif is_swin and predicted_class == "AMD":
            st.warning(report)
        else:
            st.info(report)

        # ════════════════════════════════════════════════
        # PDF İNDİRME
        # ════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">
            <span class="icon">📄</span>
            <h2>Raporu İndir</h2>
        </div>
        """, unsafe_allow_html=True)

        try:
            pdf_bytes = generate_pdf_report(
                original_image=display_image,
                gradcam_image=overlaid_image,
                predicted_class=predicted_class,
                confidence=float(probabilities[predicted_idx]),
                class_names=classes,
                probabilities=probabilities,
                model_name=model_display_name,
                report_text=report,
            )

            timestamp = datetime.now(TZ_TR).strftime("%Y%m%d_%H%M%S")
            filename = f"RetinalAMD_Rapor_{predicted_class}_{timestamp}.pdf"

            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
            with col_dl2:
                st.download_button(
                    label="📄  PDF Raporu İndir",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
        except Exception as e:
            st.error(f"PDF oluşturulurken hata: {e}")

        # Footer
        st.markdown(f"""
        <div class="footer-bar">
            <p>🧬 Retinal AMD v1.0.0 · {model_display_name}</p>
            <p>Bu sonuçlar yapay zekâ desteğidir, kesin tanı niteliği taşımaz.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Hoş geldiniz
    st.markdown("""
    <div class="upload-zone">
        <span class="icon">🧬</span>
        <h3>Retinal OCT Görüntüsü Yükleyin</h3>
        <p>Yukarıdaki alandan JPG/PNG formatında görüntü seçerek analizi başlatın</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-number">1</div>
            <h4>Görüntü Seçin</h4>
            <p>JPG veya PNG formatında retinal OCT görüntüsü yükleyin</p>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <h4>Analiz Edin</h4>
            <p>"Analizi Başlat" butonuyla yapay zekâ analizini tetikleyin</p>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <h4>Raporu İndirin</h4>
            <p>Grad-CAM, olasılık grafikleri ve klinik raporu PDF olarak indirin</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-bar">
        <p>🧬 Retinal AMD v1.0.0 · Retinal OCT Analizi</p>
        <p>EfficientNet-B4 · Grad-CAM · PDF Raporlama</p>
    </div>
    """, unsafe_allow_html=True)
