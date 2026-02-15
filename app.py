"""
Retinal AMD — Klinik Karar Destek Paneli v3.1
==============================================
İki sekmeli arayüz:
  Tab 1 — 🔬 Analiz Paneli (görüntü yükleme, sonuç, karşılaştırma)
  Tab 2 — 🏥 Hasta Yönetimi (hasta listesi, arama, ekleme, profil)
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image as PILImage
from datetime import datetime, timezone, timedelta

# ── Proje Modülleri ──
import utils.preprocessing as preprocessing
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.reporting import generate_clinical_report
from utils.pdf_export import generate_pdf_report, generate_comparative_pdf
from utils.llm_reporting import is_llm_available, generate_llm_report, generate_llm_comparative_report
from models import load_model, get_classes, get_target_layer
from utils.database import (
    is_db_available, save_analysis, get_patient_analyses,
    image_to_base64, base64_to_image,
    search_patients, add_patient, get_all_patients,
)

TZ_TR = timezone(timedelta(hours=3))
MODEL_KEY = "efficientnet_b4"
MODEL_DISPLAY = "EfficientNet-B4"

# ── Sayfa ──
st.set_page_config(
    page_title="Retinal AMD · Klinik Panel",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{
  --bg:#f8fafc;--card:#ffffff;--card-h:#f1f5f9;
  --bdr:rgba(100,116,139,.2);--bdr-a:rgba(79,70,229,.3);
  --t1:#0f172a;--t2:#1e293b;--tm:#334155;
  --acc:#4f46e5;--acc-l:#4338ca;
  --ok:#15803d;--ng:#b91c1c;--warn:#b45309;
  --r:16px;
}
.stApp{font-family:'Inter',sans-serif;background:var(--bg)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#ffffff,#f1f5f9);border-right:1px solid var(--bdr)}
header[data-testid="stHeader"]{background:rgba(248,250,252,.9);backdrop-filter:blur(8px)}
h1,h2,h3,h4{color:var(--t1)!important;font-weight:700!important}
p,span,label,li{color:var(--t2)!important}
.stMarkdown p{color:var(--t2)!important}
.stCaption p, [data-testid="stCaptionContainer"] p{color:var(--tm)!important}
[data-testid="stTab"] button p{color:var(--t1)!important;font-weight:600!important}
.stButton button{border-radius:10px;font-weight:600;transition:.2s}
.stButton button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(79,70,229,.15)}
.stTextInput label, .stSelectbox label, .stFileUploader label{color:var(--t1)!important;font-weight:500!important}
.stExpander summary span{color:var(--t1)!important;font-weight:600!important}

.hero{
  background:linear-gradient(135deg,rgba(79,70,229,.08),rgba(139,92,246,.04));
  border:1px solid rgba(79,70,229,.2);border-radius:var(--r);padding:1rem 1.5rem;
  margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;
}
.hero .name{font-size:1.25rem;font-weight:700;color:var(--t1)!important;margin:0}
.hero .meta{color:var(--tm)!important;font-size:.82rem;margin:.2rem 0 0}
.hero .badge{background:rgba(21,128,61,.1);color:#15803d;padding:.3rem .8rem;
  border-radius:100px;font-size:.72rem;font-weight:700;border:1px solid rgba(21,128,61,.2)}

.dx-card{border-radius:var(--r);padding:1.2rem;margin:.8rem 0;text-align:center}
.dx-card.ok{background:rgba(21,128,61,.06);border:1px solid rgba(21,128,61,.2)}
.dx-card.ng{background:rgba(185,28,28,.06);border:1px solid rgba(185,28,28,.2)}
.dx-card .dx{font-size:1.6rem;font-weight:800;margin:0;letter-spacing:-.02em;color:var(--t1)!important}
.dx-card .sub{color:var(--tm)!important;font-size:.85rem;margin:.2rem 0 0}

.cmp-col{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
  padding:.8rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.cmp-col .tag{display:inline-block;background:rgba(79,70,229,.1);color:var(--acc)!important;
  padding:.15rem .6rem;border-radius:100px;font-size:.68rem;font-weight:700;margin-bottom:.4rem}
.cmp-col h4{margin:.2rem 0;color:var(--t1)!important}
.cmp-col .conf{color:var(--tm)!important;font-size:.82rem;margin:0}

.sec-title{color:var(--t1)!important;font-size:1.05rem;font-weight:700;margin-bottom:.6rem;
  padding-left:.7rem;border-left:3px solid var(--acc)}

.hist-row{display:flex;justify-content:space-between;align-items:center;
  background:var(--card);border:1px solid var(--bdr);border-radius:10px;
  padding:.6rem .8rem;margin-bottom:.35rem;transition:.2s;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.hist-row:hover{border-color:var(--bdr-a);background:var(--card-h)}
.hist-row .dt{color:var(--tm)!important;font-size:.72rem}
.hist-row .cls{color:var(--t1)!important;font-weight:700;font-size:.85rem}
.hist-row .sc{color:var(--acc)!important;font-size:.82rem;font-weight:600}

.pt-card{background:var(--card);border:1px solid var(--bdr);
  border-radius:12px;padding:.8rem 1rem;margin-bottom:.4rem;
  display:flex;justify-content:space-between;align-items:center;transition:.2s;
  box-shadow:0 1px 2px rgba(0,0,0,.04)}
.pt-card:hover{border-color:var(--bdr-a);background:var(--card-h)}
.pt-card .nm{color:var(--t1)!important;font-weight:700;font-size:.9rem}
.pt-card .no{color:var(--tm)!important;font-size:.78rem}

.empty-box{text-align:center;padding:2.5rem 1rem;color:var(--tm)!important}
.empty-box .ic{font-size:2.5rem;margin-bottom:.3rem}
</style>
""", unsafe_allow_html=True)

# ── Session ──
for k, v in {"selected_patient": None, "current_result": None, "compare_selections": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

db_ok = is_db_available()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Aktif Hasta & Hızlı Seçim
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.8rem 0 .3rem">
        <span style="font-size:2.2rem">👁️</span>
        <h3 style="margin:.2rem 0 0;font-weight:800;letter-spacing:-.03em">Retinal AMD</h3>
        <p style="color:#475569;font-size:.72rem;margin:0">Klinik Karar Destek Paneli</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    sp = st.session_state["selected_patient"]
    if sp:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.15);
                    border-radius:12px;padding:.7rem .8rem;text-align:center">
            <p style="color:#86efac;font-weight:700;font-size:.9rem;margin:0">
                ✅ {sp['ad']} {sp['soyad']}
            </p>
            <p style="color:#475569;font-size:.72rem;margin:.15rem 0 0">📁 {sp['dosya_no']}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Hasta Değiştir", use_container_width=True):
            st.session_state["selected_patient"] = None
            st.session_state["current_result"] = None
            st.session_state["compare_selections"] = []
            st.rerun()
    else:
        st.caption("Hasta seçilmedi. '🏥 Hasta Yönetimi' sekmesinden seçin.")

    st.markdown("---")

    # Hızlı arama
    if db_ok:
        st.markdown("**🔍 Hızlı Hasta Ara**")
        qq = st.text_input("Ad, soyad veya dosya no", placeholder="Örn: Mehmet veya 12345",
                           label_visibility="collapsed", key="sidebar_q")
        if qq:
            found = search_patients(qq)
            for p in found[:5]:
                if st.button(f"👤 {p['ad']} {p['soyad']} · {p['dosya_no']}", key=f"sq_{p['id']}",
                             use_container_width=True):
                    st.session_state["selected_patient"] = p
                    st.session_state["current_result"] = None
                    st.session_state["compare_selections"] = []
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ANA SEKMELER
# ══════════════════════════════════════════════════════════════════════════════

# Hasta seçim ekranına yönlendirme (JS ile tab tıklama)
tab_analysis, tab_patients = st.tabs(["🔬 Analiz Paneli", "🏥 Hasta Yönetimi"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALİZ PANELİ
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    patient = st.session_state["selected_patient"]

    # ── Hasta Banner ──
    if patient:
        dogum = patient.get("dogum_tarihi") or "—"
        st.markdown(f"""
        <div class="hero">
            <div>
                <p class="name">{patient['ad']} {patient['soyad']}</p>
                <p class="meta">📁 {patient['dosya_no']}  ·  🎂 {dogum}</p>
            </div>
            <span class="badge">● Aktif Hasta</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Hasta seçmeden de tek seferlik analiz yapabilirsiniz.")
        # Hasta hızlı seçici — doğrudan analiz sekmesinde
        if db_ok:
            if st.button("🏥 Hasta Seç", use_container_width=True, type="primary"):
                st.session_state["show_patient_picker"] = True

            if st.session_state.get("show_patient_picker"):
                all_p = get_all_patients()
                if all_p:
                    opts = {f"{p['ad']} {p['soyad']} · {p['dosya_no']}": p for p in all_p}
                    chosen = st.selectbox("Hasta seçin:", [""] + list(opts.keys()), key="inline_pick")
                    if chosen and chosen in opts:
                        st.session_state["selected_patient"] = opts[chosen]
                        st.session_state["current_result"] = None
                        st.session_state["compare_selections"] = []
                        st.session_state["show_patient_picker"] = False
                        st.rerun()
                else:
                    st.caption("Henüz kayıtlı hasta yok. '🏥 Hasta Yönetimi' sekmesinden ekleyin.")

    # ── 2 Kolon ──
    col_left, col_right = st.columns([3, 2], gap="large")

    # ────────── SOL: ANALİZ ──────────
    with col_left:
        st.markdown('<p class="sec-title">📤 OCT Analizi</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Görüntü", type=["jpg", "jpeg", "png", "bmp", "tiff"],
                                    label_visibility="collapsed")

        if uploaded:
            pil_image = PILImage.open(uploaded).convert("RGB")
            display_image = preprocessing.prepare_display_image(pil_image)

            # Küçük önizleme + buton
            c_img, c_btn = st.columns([2, 1])
            with c_img:
                st.image(display_image, width=180, caption="Yüklenen OCT")
            with c_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_btn = st.button("🚀 Analiz Et", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Analiz ediliyor…"):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, is_demo = load_model(MODEL_KEY, str(device))
                    target_layer = get_target_layer(model, MODEL_KEY)

                    if is_demo:
                        st.warning("⚠️ Demo modu — ağırlıklar yüklenmedi.")

                    input_tensor = preprocessing.preprocess_image(pil_image, device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

                    idx = int(np.argmax(probs))
                    class_names = get_classes(MODEL_KEY)
                    predicted_class = class_names[idx]
                    confidence = float(probs[idx])

                    cam = generate_gradcam(model, input_tensor, None, target_layer)
                    overlaid = overlay_gradcam(display_image, cam)

                    report_text = generate_clinical_report(
                        model_name=MODEL_DISPLAY,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        is_swin_v2=False,
                    )

                    saved_id = None
                    if patient and db_ok:
                        saved = save_analysis(
                            patient_id=patient["id"],
                            predicted_class=predicted_class,
                            confidence=confidence,
                            probabilities=probs.tolist(),
                            model_name=MODEL_DISPLAY,
                            original_image=display_image,
                            gradcam_image=overlaid,
                            report_text=report_text,
                        )
                        if saved:
                            saved_id = saved.get("id")

                    now_str = datetime.now(TZ_TR).isoformat()
                    st.session_state["current_result"] = {
                        "id": saved_id,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probs.tolist(),
                        "model_name": MODEL_DISPLAY,
                        "report_text": report_text,
                        "display_image": display_image,
                        "overlaid_image": overlaid,
                        "class_names": class_names,
                        "analysis_date": now_str,
                    }

                    if patient and db_ok:
                        past = get_patient_analyses(patient["id"])
                        if past and len(past) >= 2:
                            st.session_state["compare_selections"] = [past[1]["id"]]
                        else:
                            st.session_state["compare_selections"] = []

        # ── SONUÇ ──
        result = st.session_state.get("current_result")
        if result:
            is_ok = result["predicted_class"] == "NORMAL"
            css = "ok" if is_ok else "ng"
            icon = "✅" if is_ok else "🔴"
            col = "#4ade80" if is_ok else "#f87171"

            st.markdown(f"""
            <div class="dx-card {css}">
                <p class="dx" style="color:{col}">{icon} {result['predicted_class']}</p>
                <p class="sub">%{result['confidence']*100:.1f} güven  ·  {result['model_name']}</p>
            </div>
            """, unsafe_allow_html=True)

            if result.get("id"):
                st.success("💾 Analiz kaydedildi.")

            # Grad-CAM — küçük göster, tıkla büyüt
            with st.expander("🔥 Grad-CAM Dikkat Haritası (büyütmek için tıklayın)", expanded=True):
                st.image(result["overlaid_image"], width=280, caption="Grad-CAM Aktivasyonu")

            # Olasılık çubukları
            import plotly.graph_objects as go
            cls_c = {"CNV": "#ef4444", "DME": "#f59e0b", "DRUSEN": "#8b5cf6", "NORMAL": "#22c55e", "AMD": "#ec4899"}
            fig = go.Figure(go.Bar(
                x=result["class_names"],
                y=[p * 100 for p in result["probabilities"]],
                marker_color=[cls_c.get(c, "#6366f1") for c in result["class_names"]],
                text=[f"%{p*100:.1f}" for p in result["probabilities"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=170, margin=dict(l=10, r=10, t=5, b=25),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.03)", range=[0, 115]),
                xaxis=dict(showgrid=False),
                font=dict(color="#94a3b8", size=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Rapor & PDF & AI
            with st.expander("📋 Klinik Rapor (Kural Tabanlı)", expanded=False):
                st.markdown(result["report_text"])

            # 🤖 LLM Rapor
            if is_llm_available():
                if st.button("🤖 Yapay Zekâ ile Detaylı Rapor Üret", key="llm_single", use_container_width=True):
                    import time
                    progress_bar = st.progress(0, text="🧠 Yapay zekâ modeli hazırlanıyor…")
                    for i in range(30):
                        time.sleep(0.04)
                        progress_bar.progress(i + 1, text="🧠 Yapay zekâ modeli hazırlanıyor…")
                    progress_bar.progress(35, text="📡 io.net sunucusuna bağlanılıyor…")
                    for i in range(35, 55):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1, text="📡 io.net sunucusuna bağlanılıyor…")
                    progress_bar.progress(60, text="✍️ DeepSeek-V3.2 rapor yazıyor…")

                    llm_text = generate_llm_report(
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"],
                        probabilities=result["probabilities"],
                        class_names=result["class_names"],
                        model_name=result["model_name"],
                        patient_info=patient,
                    )

                    for i in range(60, 95):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1, text="✍️ DeepSeek-V3.2 rapor yazıyor…")
                    progress_bar.progress(100, text="✅ Rapor hazır!")
                    time.sleep(0.5)
                    progress_bar.empty()

                    if llm_text:
                        st.session_state["llm_single_report"] = llm_text
                        st.rerun()

                if st.session_state.get("llm_single_report"):
                    st.markdown("---")
                    st.markdown("✨ **🤖 DeepSeek-V3.2 Klinik Rapor**")
                    st.markdown(st.session_state["llm_single_report"])
                    st.markdown("---")

            # PDF İndir
            try:
                hist = get_patient_analyses(patient["id"]) if patient and db_ok else None
                report_for_pdf = st.session_state.get("llm_single_report") or result["report_text"]
                pdf_bytes = generate_pdf_report(
                    original_image=result["display_image"],
                    gradcam_image=result["overlaid_image"],
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    class_names=result["class_names"],
                    probabilities=np.array(result["probabilities"]),
                    model_name=result["model_name"],
                    report_text=report_for_pdf,
                    patient_info=patient,
                    analysis_history=hist[:5] if hist else None,
                )
                st.download_button(
                    "📄 PDF İndir", data=pdf_bytes,
                    file_name=f"Rapor_{result['predicted_class']}_{datetime.now(TZ_TR).strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", type="primary", use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF hatası: {e}")

    # ────────── SAĞ: GEÇMİŞ ──────────
    with col_right:
        if patient and db_ok:
            st.markdown('<p class="sec-title">📋 Geçmiş Analizler</p>', unsafe_allow_html=True)
            analyses = get_patient_analyses(patient["id"])

            if analyses:
                st.caption(f"{len(analyses)} kayıt · Karşılaştırmak için ☑️ seçin (max 3)")

                current_sels = list(st.session_state.get("compare_selections", []))
                EMO = {"CNV": "🔴", "DME": "🟡", "DRUSEN": "🟣", "NORMAL": "🟢", "AMD": "🩷"}

                for a in analyses:
                    a_id = a["id"]
                    raw_d = a.get("analysis_date", "")
                    d = raw_d[:10] if len(raw_d) >= 10 else "?"
                    t = raw_d[11:16] if len(raw_d) >= 16 else ""
                    cls = a.get("predicted_class", "?")
                    conf = a.get("confidence", 0) * 100
                    emo = EMO.get(cls, "⚪")

                    c1, c2 = st.columns([1, 9])
                    with c1:
                        chk = st.checkbox("x", value=a_id in current_sels, key=f"s_{a_id}",
                                          label_visibility="collapsed")
                        if chk and a_id not in current_sels and len(current_sels) < 3:
                            current_sels.append(a_id)
                        elif not chk and a_id in current_sels:
                            current_sels.remove(a_id)
                    with c2:
                        with st.expander(f"{emo} **{cls}** — %{conf:.0f}  ·  {d} {t}", expanded=False):
                            st.caption(f"**Tanı:** {cls}  ·  **Güven:** %{conf:.1f}")
                            st.caption(f"**Model:** {a.get('model_name', '—')}  ·  **Tarih:** {d} {t}")
                            if a.get("gradcam_image_b64"):
                                st.image(base64_to_image(a["gradcam_image_b64"]),
                                         width=180, caption="Grad-CAM")
                            elif a.get("original_image_b64"):
                                st.image(base64_to_image(a["original_image_b64"]),
                                         width=180, caption="Orijinal")
                            if a.get("report_text"):
                                txt = a["report_text"]
                                st.markdown(f"<p style='font-size:.78rem;color:#94a3b8;margin-top:.3rem'>"
                                            f"{txt[:300]}{'…' if len(txt) > 300 else ''}</p>",
                                            unsafe_allow_html=True)

                st.session_state["compare_selections"] = current_sels[:3]

                # Trend
                if len(analyses) >= 2:
                    import plotly.graph_objects as go
                    sa = sorted(analyses, key=lambda x: x["analysis_date"])
                    CC = {"CNV": "#ef4444", "DME": "#f59e0b", "DRUSEN": "#8b5cf6",
                          "AMD": "#ec4899", "NORMAL": "#22c55e"}
                    fig = go.Figure(go.Scatter(
                        x=[x["analysis_date"] for x in sa],
                        y=[x["confidence"] * 100 for x in sa],
                        mode="lines+markers",
                        line=dict(color="rgba(99,102,241,.4)", width=2),
                        marker=dict(color=[CC.get(x["predicted_class"], "#6366f1") for x in sa],
                                    size=8, line=dict(width=1.5, color="white")),
                        hovertemplate="%{y:.0f}%<extra></extra>",
                    ))
                    fig.update_layout(
                        height=160, margin=dict(l=15, r=15, t=5, b=15),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)", range=[0, 110],
                                   tickfont=dict(color="#475569", size=9)),
                        showlegend=False,
                    )
                    st.markdown("#### 📈 Trend")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<div class="empty-box"><div class="ic">📭</div><p>Henüz analiz yok.</p></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-box"><div class="ic">📋</div><p>Hasta seçin.</p></div>',
                        unsafe_allow_html=True)

    # ────────── KARŞILAŞTIRMA ──────────
    current = st.session_state.get("current_result")
    compare_ids = st.session_state.get("compare_selections", [])

    items = []
    if current:
        items.append({
            "label": "🆕 Yeni Analiz", "predicted_class": current["predicted_class"],
            "confidence": current["confidence"], "analysis_date": current["analysis_date"],
            "model_name": current["model_name"], "gradcam_image": current["overlaid_image"],
            "report_text": current.get("report_text", ""),
        })

    if compare_ids and patient and db_ok:
        all_h = get_patient_analyses(patient["id"])
        hm = {a["id"]: a for a in all_h} if all_h else {}
        for cid in compare_ids:
            a = hm.get(cid)
            if a:
                gimg = base64_to_image(a["gradcam_image_b64"]) if a.get("gradcam_image_b64") else None
                items.append({
                    "label": f"📅 {a.get('analysis_date', '?')[:10]}",
                    "predicted_class": a.get("predicted_class", "?"),
                    "confidence": a.get("confidence", 0),
                    "analysis_date": a.get("analysis_date", ""),
                    "model_name": a.get("model_name", "?"),
                    "gradcam_image": gimg,
                    "report_text": a.get("report_text", ""),
                })

    if len(items) >= 2:
        st.markdown("---")
        st.markdown('<p class="sec-title">🔍 Karşılaştırma</p>', unsafe_allow_html=True)

        cols = st.columns(len(items))
        for col, item in zip(cols, items):
            with col:
                ok = item["predicted_class"] == "NORMAL"
                c = "#4ade80" if ok else "#f87171"
                st.markdown(f"""
                <div class="cmp-col">
                    <span class="tag">{item['label']}</span>
                    <h4 style="color:{c};font-size:1.2rem">{item['predicted_class']}</h4>
                    <p class="conf">%{item['confidence']*100:.1f}</p>
                </div>
                """, unsafe_allow_html=True)

                if item.get("gradcam_image") is not None:
                    with st.expander("Grad-CAM", expanded=True):
                        st.image(item["gradcam_image"], width=200)

        # Degisim
        n, o = items[0], items[-1]
        if n["predicted_class"] != o["predicted_class"]:
            st.warning(f"⚠️ Sınıf değişimi: {o['predicted_class']} → {n['predicted_class']}")
        else:
            diff = (n["confidence"] - o["confidence"]) * 100
            if abs(diff) > 5:
                st.info(f"{'📈' if diff > 0 else '📉'} Güven farkı: {diff:+.1f}%")
            else:
                st.success(f"✅ Stabil ({n['predicted_class']}, fark: {diff:+.1f}%)")

        # 🤖 LLM Karşılaştırma Raporu
        if is_llm_available():
            if st.button("🤖 AI ile Karşılaştırma Raporu Yaz", key="llm_cmp", use_container_width=True):
                import time
                pb = st.progress(0, text="🧠 Yapay zekâ hazırlanıyor…")
                for i in range(25):
                    time.sleep(0.04)
                    pb.progress(i + 1, text="🧠 Yapay zekâ hazırlanıyor…")
                pb.progress(30, text="📡 io.net bağlantısı kuruluyor…")
                for i in range(30, 50):
                    time.sleep(0.03)
                    pb.progress(i + 1, text="📡 io.net bağlantısı kuruluyor…")
                pb.progress(55, text="🔬 DeepSeek karşılaştırma analizi yapıyor…")

                llm_cmp = generate_llm_comparative_report(
                    analyses=items,
                    patient_info=patient,
                )

                for i in range(55, 95):
                    time.sleep(0.02)
                    pb.progress(i + 1, text="🔬 DeepSeek karşılaştırma analizi yapıyor…")
                pb.progress(100, text="✅ Karşılaştırma raporu hazır!")
                time.sleep(0.5)
                pb.empty()

                if llm_cmp:
                    st.session_state["llm_cmp_report"] = llm_cmp
                    st.rerun()

            if st.session_state.get("llm_cmp_report"):
                st.markdown("---")
                st.markdown("✨ **🤖 DeepSeek-V3.2 Karşılaştırma Raporu**")
                st.markdown(st.session_state["llm_cmp_report"])
                st.markdown("---")

        # PDF
        try:
            llm_cmp_text = st.session_state.get("llm_cmp_report")
            cpdf = generate_comparative_pdf(analyses=items, patient_info=patient, llm_report=llm_cmp_text)
            st.download_button("📊 Karşılaştırma PDF", data=cpdf,
                               file_name=f"Karsilastirma_{datetime.now(TZ_TR).strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF hatası: {e}")

    elif current and not compare_ids:
        st.markdown("---")
        st.caption("ℹ️ Karşılaştırma için sağ panelden geçmiş analiz seçin.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HASTA YÖNETİMİ
# ══════════════════════════════════════════════════════════════════════════════
with tab_patients:
    if not db_ok:
        st.warning("⚠️ Veritabanı bağlantısı yok.")
    else:
        pt_sub1, pt_sub2, pt_sub3 = st.tabs(["📋 Hasta Listesi", "🔍 Hasta Ara", "➕ Yeni Hasta"])

        # ── Liste ──
        with pt_sub1:
            st.markdown('<p class="sec-title">📋 Kayıtlı Hastalar</p>', unsafe_allow_html=True)
            all_patients = get_all_patients()
            if all_patients:
                st.caption(f"Toplam {len(all_patients)} hasta")
                for p in all_patients:
                    c_info, c_act = st.columns([5, 1])
                    with c_info:
                        ad_soyad = f"{p['ad']} {p['soyad']}"
                        dogum = p.get("dogum_tarihi") or ""
                        st.markdown(f"""
                        <div class="pt-card">
                            <div>
                                <span class="nm">👤 {ad_soyad}</span><br>
                                <span class="no">📁 {p['dosya_no']}  {('· 🎂 ' + dogum) if dogum else ''}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_act:
                        if st.button("Seç", key=f"pl_{p['id']}", use_container_width=True):
                            st.session_state["selected_patient"] = p
                            st.session_state["current_result"] = None
                            st.session_state["compare_selections"] = []
                            st.rerun()
            else:
                st.markdown('<div class="empty-box"><div class="ic">📭</div><p>Henüz hasta kaydı yok.</p></div>',
                            unsafe_allow_html=True)

        # ── Arama ──
        with pt_sub2:
            st.markdown('<p class="sec-title">🔍 Detaylı Hasta Arama</p>', unsafe_allow_html=True)

            c_s1, c_s2 = st.columns(2)
            with c_s1:
                s_name = st.text_input("Ad / Soyad", placeholder="Örn: Mehmet Yılmaz", key="pt_s_name")
            with c_s2:
                s_dosya = st.text_input("Dosya No", placeholder="Örn: 12345", key="pt_s_dosya")

            sq = s_name or s_dosya or ""
            if sq:
                results = search_patients(sq)
                if results:
                    st.success(f"{len(results)} sonuç bulundu")
                    for p in results:
                        c_i, c_a = st.columns([5, 1])
                        with c_i:
                            an_count = len(get_patient_analyses(p["id"])) if db_ok else 0
                            st.markdown(f"""
                            <div class="pt-card">
                                <div>
                                    <span class="nm">👤 {p['ad']} {p['soyad']}</span><br>
                                    <span class="no">📁 {p['dosya_no']} · 📊 {an_count} analiz</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c_a:
                            if st.button("Seç", key=f"ps_{p['id']}", use_container_width=True):
                                st.session_state["selected_patient"] = p
                                st.session_state["current_result"] = None
                                st.session_state["compare_selections"] = []
                                st.rerun()
                else:
                    st.info("Sonuç bulunamadı.")

        # ── Yeni Hasta ──
        with pt_sub3:
            st.markdown('<p class="sec-title">➕ Yeni Hasta Kaydı</p>', unsafe_allow_html=True)

            with st.form("full_add_patient", clear_on_submit=True):
                st.markdown("**Zorunlu Bilgiler**")
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    f_ad = st.text_input("Ad *")
                with r1c2:
                    f_soyad = st.text_input("Soyad *")
                f_dosya = st.text_input("Dosya No *")

                st.markdown("**Ek Bilgiler** *(isteğe bağlı)*")
                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    f_dogum = st.date_input("Doğum Tarihi", value=None)
                with r2c2:
                    f_tel = st.text_input("Telefon")
                f_email = st.text_input("E-posta")
                f_not = st.text_area("Notlar", height=80)

                submit = st.form_submit_button("💾 Hastayı Kaydet", type="primary", use_container_width=True)
                if submit:
                    if f_ad and f_soyad and f_dosya:
                        new_p = add_patient(
                            dosya_no=f_dosya, ad=f_ad, soyad=f_soyad,
                            dogum_tarihi=str(f_dogum) if f_dogum else None,
                            telefon=f_tel if f_tel else None,
                            email=f_email if f_email else None,
                            notlar=f_not if f_not else None,
                        )
                        if new_p:
                            st.session_state["selected_patient"] = new_p
                            st.success(f"✅ {f_ad} {f_soyad} kaydedildi ve seçildi!")
                            st.balloons()
                        else:
                            st.error("Kayıt başarısız. Dosya no benzersiz mi kontrol edin.")
                    else:
                        st.warning("Ad, Soyad ve Dosya No alanları zorunludur.")


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:.4rem">
    <p style="color:#1e293b;font-size:.7rem;margin:0">
        👁️ Retinal AMD v3.1 · Yapay zekâ desteği — kesin tanı niteliği taşımaz.
    </p>
</div>
""", unsafe_allow_html=True)
