"""
Retinal AMD — UI Bileşenleri Modülü
===================================
Tekrar kullanılabilir Streamlit arayüz bileşenleri:
- Hasta seçici / oluşturucu (Sidebar)
- Hasta özet kartı
- Analiz geçmişi listesi/tablosu
- Trend grafiği
- Karşılaştırma görünümü
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import base64
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

from utils.database import (
    search_patients, add_patient,
    get_patient_analyses, base64_to_image,
)

TZ_TR = timezone(timedelta(hours=3))

# Sınıf renkleri
CLASS_COLORS = {
    "CNV": "#ef4444",
    "DME": "#f59e0b",
    "DRUSEN": "#8b5cf6",
    "AMD": "#ec4899",
    "NORMAL": "#22c55e",
}

def render_sidebar_patient_selector() -> Optional[Dict]:
    """
    Sidebar'da hasta arama ve ekleme işlemlerini yönetir.
    Seçili hastayı session_state'e kaydeder ve döndürür.
    """
    st.sidebar.markdown("### 👤 Hasta İşlemleri")
    
    tab_search, tab_add = st.sidebar.tabs(["🔍 Ara", "➕ Ekle"])
    
    # ── TAB 1: ARA ──
    with tab_search:
        search_query = st.text_input("Hasta Ara", placeholder="Ad, Soyad, Dosya No...", label_visibility="collapsed")
        patients = search_patients(search_query)
        
        if not patients:
            st.info("Kayıt bulunamadı.")
            return None
            
        # Selectbox için seçenekler
        options = {f"{p['ad']} {p['soyad']} ({p['dosya_no']})": p for p in patients}
        selected_label = st.selectbox("Sonuçlar", options=list(options.keys()), label_visibility="collapsed")
        
        if selected_label:
            selected_patient = options[selected_label]
            if st.button("✅ Hastayı Seç", use_container_width=True, type="primary"):
                st.session_state["selected_patient"] = selected_patient
                st.rerun()

    # ── TAB 2: EKLE ──
    with tab_add:
        with st.form("quick_add_patient"):
            ad = st.text_input("Ad")
            soyad = st.text_input("Soyad")
            dosya_no = st.text_input("Dosya No")
            submitted = st.form_submit_button("💾 Kaydet & Seç")
            
            if submitted:
                if ad and soyad and dosya_no:
                    new_patient = add_patient(dosya_no, ad, soyad)
                    if new_patient:
                        st.success("Kaydedildi!")
                        st.session_state["selected_patient"] = new_patient
                        st.rerun()
                    else:
                        st.error("Hata oluştu.")
                else:
                    st.warning("Eksik bilgi.")

    return st.session_state.get("selected_patient")


def render_patient_summary(patient: Dict):
    """Sayfanın üst kısmında hasta özet bilgilerini gösterir."""
    if not patient:
        return

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,27,75,0.6));
        border: 1px solid rgba(99,102,241,0.2); border-radius: 16px; padding: 1.2rem;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;
    ">
        <div>
            <h2 style="margin:0; font-size:1.4rem; color:#e2e8f0;">
                {patient['ad']} {patient['soyad']}
            </h2>
            <p style="margin:0.2rem 0 0 0; color:#94a3b8; font-size:0.9rem;">
                📁 {patient['dosya_no']} &nbsp;·&nbsp; 
                🎂 {patient.get('dogum_tarihi') or '-'}
            </p>
        </div>
        <div style="text-align:right;">
             <span style="background:rgba(99,102,241,0.15); color:#a5b4fc; padding:0.3rem 0.8rem; border-radius:8px; font-size:0.8rem; border:1px solid rgba(99,102,241,0.3);">
                Aktif Hasta
             </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis_history_list(patient_id: str):
    """Hastanın geçmiş analizlerini listeleyen bir bileşen."""
    analyses = get_patient_analyses(patient_id)
    
    if not analyses:
        st.info("Henüz analiz kaydı yok.")
        return

    st.markdown("### 📋 Geçmiş Analizler")
    
    # Scrollable container workaround
    st.markdown("""
    <style>
    .history-container {
        max-height: 400px; overflow-y: auto; padding-right: 5px;
    }
    .history-item {
        background: rgba(30,41,59,0.5); border: 1px solid rgba(148,163,184,0.1);
        border-radius: 8px; padding: 10px; margin-bottom: 8px; cursor: pointer;
        transition: all 0.2s;
    }
    .history-item:hover {
        background: rgba(30,41,59,0.8); border-color: rgba(99,102,241,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Her analiz için buton (seçim yapmak amacıyla)
    for a in analyses:
        d = a.get("analysis_date", "").replace("T", " ")[:16]
        c_score = a.get("confidence", 0) * 100
        p_cls = a.get("predicted_class", "?")
        color = CLASS_COLORS.get(p_cls, "#cbd5e1")
        
        # Buton etiketi
        label = f"{d} | {p_cls} (%{c_score:.1f})"
        
        if st.button(label, key=f"hist_btn_{a['id']}", use_container_width=True):
            st.session_state["selected_history_analysis"] = a
            
    return analyses


def render_trend_chart(analyses: List[Dict]):
    """Güven skoru trend grafiğini çizer."""
    if not analyses or len(analyses) < 2:
        return

    # Kronolojik sıralama
    analyses_sorted = sorted(analyses, key=lambda x: x["analysis_date"])
    
    dates = []
    scores = []
    colors = []
    
    for a in analyses_sorted:
        dates.append(a["analysis_date"])
        scores.append(a["confidence"] * 100)
        colors.append(CLASS_COLORS.get(a["predicted_class"], "#6366f1"))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=scores, mode='lines+markers',
        line=dict(color='rgba(99, 102, 241, 0.5)', width=2),
        marker=dict(color=colors, size=8, line=dict(width=1, color='white'))
    ))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 110]),
    )
    
    st.markdown("#### 📈 Gelişim Trendi")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_comparison_view(current_analysis: Dict, past_analysis: Dict):
    """
    İki analizi karşılaştırmalı olarak gösterir.
    current_analysis: Yeni yapılan veya seçili olan ana analiz
    past_analysis: Geçmişten seçilen referans analiz
    """
    st.markdown("---")
    st.markdown("### 🔍 Karşılaştırmalı İnceleme")
    
    col1, col2 = st.columns(2)
    
    # Sol: Mevcut / Yeni
    with col1:
        st.caption(f"YENİ: {current_analysis.get('analysis_date', 'Şimdi')[:16]}")
        if current_analysis.get("gradcam_image_b64"):
            img = base64_to_image(current_analysis["gradcam_image_b64"])
            st.image(img, use_container_width=True, caption=f"{current_analysis['predicted_class']}")
            
    # Sağ: Referans / Eski
    with col2:
        st.caption(f"ESKİ: {past_analysis.get('analysis_date', '?')[:16]}")
        if past_analysis.get("gradcam_image_b64"):
            img = base64_to_image(past_analysis["gradcam_image_b64"])
            st.image(img, use_container_width=True, caption=f"{past_analysis['predicted_class']}")
            
    # Değişim Yorumu
    c_curr = current_analysis.get("confidence", 0)
    c_past = past_analysis.get("confidence", 0)
    diff = (c_curr - c_past) * 100
    
    if current_analysis['predicted_class'] != past_analysis['predicted_class']:
        st.warning(f"⚠️ Sınıf Değişimi: {past_analysis['predicted_class']} ➔ {current_analysis['predicted_class']}")
    else:
        if diff > 5:
            st.info(f"Yapay zeka güveninde artış: +%{diff:.1f}")
        elif diff < -5:
            st.info(f"Yapay zeka güveninde düşüş: %{diff:.1f}")
