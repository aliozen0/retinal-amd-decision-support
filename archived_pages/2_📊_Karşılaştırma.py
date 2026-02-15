"""
Retinal AMD — Karşılaştırma Paneli
=====================================
Aynı hastanın farklı tarihlerdeki OCT analizlerini
karşılaştırarak ilerleme/gerileme trendini gösterir.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import numpy as np

TZ_TR = timezone(timedelta(hours=3))

st.set_page_config(
    page_title="Karşılaştırma | Retinal AMD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .page-header {
        background: linear-gradient(135deg, #020617 0%, #0f172a 30%, #1e1b4b 60%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
        box-shadow: 0 0 80px rgba(99, 102, 241, 0.08), 0 20px 60px rgba(0,0,0,0.3);
    }
    .page-header h1 {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff, #a5b4fc, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .page-header p { color: #94a3b8; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    .section-header {
        display: flex; align-items: center; gap: 0.7rem;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(99,102,241,0.12);
    }
    .section-header h2 { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 0; }

    .trend-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px; padding: 1.2rem; text-align: center;
    }
    .trend-card .emoji { font-size: 2rem; margin-bottom: 0.3rem; }
    .trend-card .label { font-size: 0.78rem; color: #94a3b8; }
    .trend-card .value { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }

    .image-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px; padding: 1rem;
    }
    .image-card-title {
        font-size: 0.8rem; font-weight: 600; color: #a5b4fc;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    .footer-bar {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.08);
        border-radius: 12px; padding: 0.8rem 1.2rem; margin-top: 2rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .footer-bar p { color: #475569; font-size: 0.72rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

from utils.database import (
    is_db_available, search_patients,
    get_patient_analyses, base64_to_image,
)

# ── Sayfa Başlığı ──
st.markdown("""
<div class="page-header">
    <h1>📊 Karşılaştırma Paneli</h1>
    <p>Hastanın geçmiş OCT analizlerini karşılaştırarak ilerleme/gerileme trendini izleyin</p>
</div>
""", unsafe_allow_html=True)

if not is_db_available():
    st.error("❌ Veritabanı bağlantısı kurulamadı.")
    st.stop()

# ── Hasta Seçimi ──
all_patients = search_patients("")
if not all_patients:
    st.info("📋 Henüz kayıtlı hasta bulunmamaktadır.")
    st.stop()

patient_options = {
    f"{p['ad']} {p['soyad']} — {p['dosya_no']}": p["id"]
    for p in all_patients
}

selected_label = st.selectbox("Hasta Seçin", options=list(patient_options.keys()))
selected_id = patient_options[selected_label]

analyses = get_patient_analyses(selected_id)

if not analyses:
    st.info("🔬 Bu hasta için henüz analiz kaydı bulunmamaktadır.")
    st.stop()

if len(analyses) < 2:
    st.warning("⚠️ Karşılaştırma yapabilmek için en az 2 analiz kaydı gereklidir. Şu an **1** analiz mevcut.")

# ── Sınıf renk haritası ──
CLASS_COLORS = {
    "CNV": "#ef4444",
    "DME": "#f59e0b",
    "DRUSEN": "#8b5cf6",
    "AMD": "#ec4899",
    "NORMAL": "#22c55e",
}

# ================================================================
# 1. TREND GRAFİĞİ — Güven Skoru Zaman Serisi
# ================================================================
st.markdown("""
<div class="section-header">
    <span>📈</span>
    <h2>Güven Skoru Trendi</h2>
</div>
""", unsafe_allow_html=True)

# Kronolojik sıra (eskiden yeniye)
analyses_chrono = sorted(analyses, key=lambda a: a["analysis_date"])

dates = []
confidences = []
classes = []
colors = []

for a in analyses_chrono:
    try:
        dt = datetime.fromisoformat(a["analysis_date"].replace("Z", "+00:00"))
        dates.append(dt)
    except Exception:
        dates.append(a["analysis_date"])
    confidences.append(a["confidence"] * 100)
    classes.append(a["predicted_class"])
    colors.append(CLASS_COLORS.get(a["predicted_class"], "#6366f1"))

fig = go.Figure()

# Ana çizgi
fig.add_trace(go.Scatter(
    x=dates, y=confidences,
    mode="lines+markers",
    line=dict(color="rgba(99, 102, 241, 0.7)", width=3),
    marker=dict(
        size=14, color=colors,
        line=dict(color="white", width=2),
        symbol="circle",
    ),
    text=[f"{cls} — %{conf:.1f}" for cls, conf in zip(classes, confidences)],
    hovertemplate="<b>%{text}</b><br>Tarih: %{x|%d.%m.%Y %H:%M}<extra></extra>",
    name="Güven Skoru",
))

# Sınıf değişimlerini vurgula
for i in range(1, len(classes)):
    if classes[i] != classes[i - 1]:
        fig.add_vline(
            x=dates[i], line_dash="dot",
            line_color="rgba(245, 158, 11, 0.5)",
        )
        fig.add_annotation(
            x=dates[i], y=105,
            text=f"{classes[i-1]} → {classes[i]}",
            showarrow=False,
            font=dict(size=10, color="#fbbf24"),
            bgcolor="rgba(2,6,23,0.8)",
        )

fig.update_layout(
    yaxis=dict(
        title="Güven (%)", range=[0, 110],
        gridcolor="rgba(99,102,241,0.05)", color="#64748b",
    ),
    xaxis=dict(gridcolor="rgba(99,102,241,0.05)", color="#64748b"),
    plot_bgcolor="rgba(2,6,23,0.5)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    height=400, margin=dict(l=50, r=20, t=30, b=50),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# ── Trend Özet Kartları ──
if len(analyses_chrono) >= 2:
    first_a = analyses_chrono[0]
    last_a = analyses_chrono[-1]
    conf_change = (last_a["confidence"] - first_a["confidence"]) * 100
    class_changed = first_a["predicted_class"] != last_a["predicted_class"]

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(f"""
        <div class="trend-card">
            <div class="emoji">📊</div>
            <div class="value">{len(analyses_chrono)}</div>
            <div class="label">Toplam Analiz</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        trend_icon = "📈" if conf_change > 0 else ("📉" if conf_change < 0 else "➡️")
        st.markdown(f"""
        <div class="trend-card">
            <div class="emoji">{trend_icon}</div>
            <div class="value">{conf_change:+.1f}%</div>
            <div class="label">Güven Değişimi</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown(f"""
        <div class="trend-card">
            <div class="emoji">{"🔄" if class_changed else "✅"}</div>
            <div class="value">{"Değişti" if class_changed else "Sabit"}</div>
            <div class="label">Sınıf Durumu</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t4:
        last_cls = last_a["predicted_class"]
        cls_icon = "✅" if last_cls == "NORMAL" else "🔴"
        st.markdown(f"""
        <div class="trend-card">
            <div class="emoji">{cls_icon}</div>
            <div class="value">{last_cls}</div>
            <div class="label">Son Tanı</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Otomatik Değerlendirme ──
    st.markdown("<br>", unsafe_allow_html=True)

    if last_a["predicted_class"] == "NORMAL" and first_a["predicted_class"] != "NORMAL":
        st.success("🎉 **İyileşme Trendi:** Hasta patolojik bulgulardan normal duruma geçiş göstermiştir.")
    elif last_a["predicted_class"] != "NORMAL" and first_a["predicted_class"] == "NORMAL":
        st.error("⚠️ **Kötüleşme Trendi:** Hasta normal durumdan patolojik bulgu tespit edilen duruma geçmiştir.")
    elif last_a["predicted_class"] != "NORMAL" and class_changed:
        st.warning(f"🔄 **Sınıf Değişimi:** {first_a['predicted_class']} → {last_a['predicted_class']}. Klinik değerlendirme önerilir.")
    elif last_a["predicted_class"] == "NORMAL":
        st.success("✅ **Stabil:** Hasta tüm kontrollerde normal bulgu göstermiştir.")
    else:
        if conf_change > 5:
            st.warning("📈 Güven skoru artış eğiliminde — patolojik bulgu güçleniyor olabilir.")
        elif conf_change < -5:
            st.info("📉 Güven skoru azalış eğiliminde — tedaviye yanıt olabilir.")
        else:
            st.info("➡️ Güven skoru stabil seyretmektedir.")

# ================================================================
# 2. YAN YANA KARŞILAŞTIRMA
# ================================================================
if len(analyses_chrono) >= 2:
    st.markdown("""
    <div class="section-header">
        <span>🔍</span>
        <h2>Yan Yana Karşılaştırma</h2>
    </div>
    """, unsafe_allow_html=True)

    # Analiz seçimleri
    analysis_labels = []
    for a in analyses_chrono:
        try:
            dt = datetime.fromisoformat(a["analysis_date"].replace("Z", "+00:00"))
            label = f"{a['predicted_class']} — %{a['confidence']*100:.1f} — {dt.strftime('%d.%m.%Y %H:%M')}"
        except Exception:
            label = f"{a['predicted_class']} — %{a['confidence']*100:.1f}"
        analysis_labels.append(label)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        idx1 = st.selectbox("1. Analiz (Eski)", range(len(analysis_labels)),
                            format_func=lambda x: analysis_labels[x], index=0)
    with col_sel2:
        default_idx2 = min(len(analysis_labels) - 1, idx1 + 1) if idx1 < len(analysis_labels) - 1 else len(analysis_labels) - 1
        idx2 = st.selectbox("2. Analiz (Yeni)", range(len(analysis_labels)),
                            format_func=lambda x: analysis_labels[x], index=default_idx2)

    a1 = analyses_chrono[idx1]
    a2 = analyses_chrono[idx2]

    # Orijinal görüntüler
    st.markdown("#### 🖼️ Orijinal Görüntüler")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        try:
            dt1 = datetime.fromisoformat(a1["analysis_date"].replace("Z", "+00:00")).strftime("%d.%m.%Y")
        except Exception:
            dt1 = "?"
        st.markdown(f'<div class="image-card"><div class="image-card-title">📅 {dt1} · {a1["predicted_class"]}</div></div>', unsafe_allow_html=True)
        if a1.get("original_image_b64"):
            st.image(base64_to_image(a1["original_image_b64"]), use_container_width=True)
        else:
            st.info("Görüntü mevcut değil")
    with col_img2:
        try:
            dt2 = datetime.fromisoformat(a2["analysis_date"].replace("Z", "+00:00")).strftime("%d.%m.%Y")
        except Exception:
            dt2 = "?"
        st.markdown(f'<div class="image-card"><div class="image-card-title">📅 {dt2} · {a2["predicted_class"]}</div></div>', unsafe_allow_html=True)
        if a2.get("original_image_b64"):
            st.image(base64_to_image(a2["original_image_b64"]), use_container_width=True)
        else:
            st.info("Görüntü mevcut değil")

    # Grad-CAM görüntüleri
    st.markdown("#### 🔥 Grad-CAM Karşılaştırması")
    col_gc1, col_gc2 = st.columns(2)
    with col_gc1:
        st.markdown(f'<div class="image-card"><div class="image-card-title">🔥 Grad-CAM · {dt1}</div></div>', unsafe_allow_html=True)
        if a1.get("gradcam_image_b64"):
            st.image(base64_to_image(a1["gradcam_image_b64"]), use_container_width=True)
        else:
            st.info("Grad-CAM mevcut değil")
    with col_gc2:
        st.markdown(f'<div class="image-card"><div class="image-card-title">🔥 Grad-CAM · {dt2}</div></div>', unsafe_allow_html=True)
        if a2.get("gradcam_image_b64"):
            st.image(base64_to_image(a2["gradcam_image_b64"]), use_container_width=True)
        else:
            st.info("Grad-CAM mevcut değil")

    # Olasılık dağılımı karşılaştırması
    st.markdown("#### 📊 Olasılık Dağılımı Karşılaştırması")

    probs1 = a1.get("probabilities", [])
    probs2 = a2.get("probabilities", [])

    if probs1 and probs2 and len(probs1) == len(probs2):
        # Sınıf isimleri — EfficientNet-B4 varsayılan
        class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
        if len(probs1) == 3:
            class_names = ["AMD", "DME", "NORMAL"]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name=f"📅 {dt1}", x=class_names,
            y=[p * 100 for p in probs1],
            marker_color="rgba(99, 102, 241, 0.6)",
            text=[f"%{p*100:.1f}" for p in probs1],
            textposition="outside",
        ))
        fig_comp.add_trace(go.Bar(
            name=f"📅 {dt2}", x=class_names,
            y=[p * 100 for p in probs2],
            marker_color="rgba(14, 165, 233, 0.6)",
            text=[f"%{p*100:.1f}" for p in probs2],
            textposition="outside",
        ))

        fig_comp.update_layout(
            barmode="group",
            yaxis=dict(title="Olasılık (%)", range=[0, 115],
                       gridcolor="rgba(99,102,241,0.05)", color="#64748b"),
            xaxis=dict(color="#64748b"),
            plot_bgcolor="rgba(2,6,23,0.5)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            height=380, margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ── Footer ──
st.markdown("""
<div class="footer-bar">
    <p>📊 Karşılaştırma Paneli · Retinal AMD v1.0.0</p>
    <p>Bu sonuçlar yapay zekâ desteğidir, kesin tanı niteliği taşımaz.</p>
</div>
""", unsafe_allow_html=True)
