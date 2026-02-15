"""
Retinal AMD — Hasta Yönetimi Sayfası
========================================
Hasta ekleme, arama, listeleme ve profil görüntüleme.
"""

import streamlit as st
from datetime import datetime, timezone, timedelta

TZ_TR = timezone(timedelta(hours=3))

st.set_page_config(
    page_title="Hasta Yönetimi | Retinal AMD",
    page_icon="🏥",
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
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .page-header p { color: #94a3b8; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    .patient-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,27,75,0.5));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px; padding: 1.2rem; margin-bottom: 0.8rem;
        transition: all 0.3s ease; cursor: pointer;
    }
    .patient-card:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.1);
    }
    .patient-name { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin: 0; }
    .patient-meta { font-size: 0.78rem; color: #64748b; margin: 0.2rem 0 0 0; }

    .section-header {
        display: flex; align-items: center; gap: 0.7rem;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(99,102,241,0.12);
    }
    .section-header h2 { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 0; }

    .info-box {
        background: linear-gradient(145deg, rgba(99,102,241,0.08), rgba(14,165,233,0.05));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
    }
    .info-box p { font-size: 0.85rem; line-height: 1.5; margin: 0; color: #cbd5e1; }
    .info-box strong { color: #a5b4fc; }

    .stat-mini {
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,27,75,0.6));
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 12px; padding: 0.8rem 1rem; text-align: center;
    }
    .stat-mini .val { font-size: 1.4rem; font-weight: 800; color: #a5b4fc; }
    .stat-mini .lbl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }

    .analysis-row {
        background: linear-gradient(145deg, rgba(15,23,42,0.7), rgba(30,27,75,0.3));
        border: 1px solid rgba(99,102,241,0.08);
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .analysis-row .cls { font-weight: 700; color: #e2e8f0; }
    .analysis-row .dt { font-size: 0.78rem; color: #64748b; }
    .analysis-row .conf { font-weight: 600; color: #a5b4fc; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #6366f1, #818cf8) !important;
        border: none !important; border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important; font-weight: 700 !important;
        transition: all 0.3s ease !important;
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

# ── Import veritabanı modülü ──
from utils.database import (
    is_db_available, add_patient, search_patients,
    get_patient, update_patient, delete_patient,
    get_patient_analyses, get_patient_analysis_count,
    base64_to_image,
)

# ── Sayfa Başlığı ──
st.markdown("""
<div class="page-header">
    <h1>🏥 Hasta Yönetimi</h1>
    <p>Hasta kaydı oluşturma, arama ve profil yönetimi</p>
</div>
""", unsafe_allow_html=True)

# ── Veritabanı Bağlantı Kontrolü ──
if not is_db_available():
    st.error("❌ Veritabanı bağlantısı kurulamadı. Lütfen `.streamlit/secrets.toml` dosyasını kontrol edin.")
    st.stop()

# ── Sekmeler ──
tab_add, tab_search, tab_profile = st.tabs(["➕ Hasta Ekle", "🔍 Hasta Ara", "👤 Hasta Profili"])

# ================================================================
# TAB 1: HASTA EKLE
# ================================================================
with tab_add:
    st.markdown("""
    <div class="section-header">
        <span>➕</span>
        <h2>Yeni Hasta Kaydı</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.form("add_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            ad = st.text_input("👤 Ad *", placeholder="Örn: Mehmet")
            soyad = st.text_input("👤 Soyad *", placeholder="Örn: Yılmaz")
            dosya_no = st.text_input("📁 Dosya No *", placeholder="Örn: HAS-2024-001")
        with col2:
            dogum_tarihi = st.date_input("📅 Doğum Tarihi", value=None)
            telefon = st.text_input("📱 Telefon", placeholder="Örn: 0532 XXX XX XX")
            email = st.text_input("📧 E-posta", placeholder="Örn: hasta@mail.com")

        notlar = st.text_area("📝 Notlar", placeholder="Hasta ile ilgili ek bilgiler...", height=80)

        submitted = st.form_submit_button("💾 Hastayı Kaydet", type="primary", use_container_width=True)

        if submitted:
            if not ad or not soyad or not dosya_no:
                st.error("❌ Ad, Soyad ve Dosya No alanları zorunludur!")
            else:
                dt_str = dogum_tarihi.isoformat() if dogum_tarihi else None
                result = add_patient(
                    dosya_no=dosya_no, ad=ad, soyad=soyad,
                    dogum_tarihi=dt_str, telefon=telefon,
                    email=email, notlar=notlar,
                )
                if result:
                    st.success(f"✅ Hasta başarıyla kaydedildi: **{ad} {soyad}** (Dosya No: {dosya_no})")
                    st.balloons()

# ================================================================
# TAB 2: HASTA ARA
# ================================================================
with tab_search:
    st.markdown("""
    <div class="section-header">
        <span>🔍</span>
        <h2>Hasta Arama</h2>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input(
        "Arama", placeholder="Ad, soyad veya dosya no ile arayın...",
        label_visibility="collapsed",
    )

    patients = search_patients(search_query)

    if patients:
        st.markdown(f"**{len(patients)}** hasta bulundu")
        for p in patients:
            analysis_count = get_patient_analysis_count(p["id"])
            col_info, col_action = st.columns([4, 1])
            with col_info:
                created = ""
                if p.get("created_at"):
                    try:
                        dt = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
                        created = dt.strftime("%d.%m.%Y")
                    except Exception:
                        created = ""

                st.markdown(f"""
                <div class="patient-card">
                    <p class="patient-name">{p['ad']} {p['soyad']}</p>
                    <p class="patient-meta">
                        📁 {p['dosya_no']}
                        {'· 📱 ' + p['telefon'] if p.get('telefon') else ''}
                        · 🔬 {analysis_count} analiz
                        · 📅 {created}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_action:
                if st.button("👁️ Profil", key=f"view_{p['id']}", use_container_width=True):
                    st.session_state["selected_patient_id"] = p["id"]
                    st.session_state["active_tab"] = "profile"
                    st.rerun()
    elif search_query:
        st.info("🔍 Aramanızla eşleşen hasta bulunamadı.")
    else:
        st.info("📋 Henüz kayıtlı hasta bulunmamaktadır. 'Hasta Ekle' sekmesinden yeni hasta ekleyebilirsiniz.")

# ================================================================
# TAB 3: HASTA PROFİLİ
# ================================================================
with tab_profile:
    st.markdown("""
    <div class="section-header">
        <span>👤</span>
        <h2>Hasta Profili</h2>
    </div>
    """, unsafe_allow_html=True)

    # Hasta seçimi
    all_patients = search_patients("")
    if not all_patients:
        st.info("📋 Henüz kayıtlı hasta bulunmamaktadır.")
    else:
        patient_options = {
            f"{p['ad']} {p['soyad']} — {p['dosya_no']}": p["id"]
            for p in all_patients
        }

        # Önceden seçilmiş hasta varsa indexle
        default_idx = 0
        if "selected_patient_id" in st.session_state:
            for i, (label, pid) in enumerate(patient_options.items()):
                if pid == st.session_state["selected_patient_id"]:
                    default_idx = i
                    break

        selected_label = st.selectbox(
            "Hasta Seçin", options=list(patient_options.keys()),
            index=default_idx, label_visibility="visible",
        )
        selected_id = patient_options[selected_label]
        patient = get_patient(selected_id)

        if patient:
            # ── Profil Bilgileri ──
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(f"""
                <div class="stat-mini">
                    <div class="val">📁</div>
                    <div class="lbl">{patient['dosya_no']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_p2:
                analysis_count = get_patient_analysis_count(patient["id"])
                st.markdown(f"""
                <div class="stat-mini">
                    <div class="val">{analysis_count}</div>
                    <div class="lbl">Toplam Analiz</div>
                </div>
                """, unsafe_allow_html=True)
            with col_p3:
                dogum = patient.get("dogum_tarihi", "-") or "-"
                st.markdown(f"""
                <div class="stat-mini">
                    <div class="val">📅</div>
                    <div class="lbl">{dogum}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Detay bilgileri ──
            with st.expander("📋 Hasta Detayları", expanded=True):
                det_c1, det_c2 = st.columns(2)
                with det_c1:
                    st.markdown(f"""
                    <div class="info-box">
                        <p><strong>Ad Soyad:</strong> {patient['ad']} {patient['soyad']}</p>
                        <p><strong>Dosya No:</strong> {patient['dosya_no']}</p>
                        <p><strong>Doğum Tarihi:</strong> {patient.get('dogum_tarihi') or '-'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with det_c2:
                    st.markdown(f"""
                    <div class="info-box">
                        <p><strong>Telefon:</strong> {patient.get('telefon') or '-'}</p>
                        <p><strong>E-posta:</strong> {patient.get('email') or '-'}</p>
                        <p><strong>Notlar:</strong> {patient.get('notlar') or '-'}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Analiz Geçmişi ──
            st.markdown("""
            <div class="section-header">
                <span>🔬</span>
                <h2>Analiz Geçmişi</h2>
            </div>
            """, unsafe_allow_html=True)

            analyses = get_patient_analyses(patient["id"])
            if analyses:
                for a in analyses:
                    try:
                        dt = datetime.fromisoformat(a["analysis_date"].replace("Z", "+00:00"))
                        date_str = dt.strftime("%d.%m.%Y %H:%M")
                    except Exception:
                        date_str = a.get("analysis_date", "-")

                    conf_pct = a["confidence"] * 100
                    is_normal = a["predicted_class"] == "NORMAL"
                    cls_icon = "✅" if is_normal else "🔴"

                    st.markdown(f"""
                    <div class="analysis-row">
                        <span class="cls">{cls_icon} {a['predicted_class']}</span>
                        <span class="conf">%{conf_pct:.1f}</span>
                        <span class="dt">🧠 {a.get('model_name', '-')} · {date_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Görüntüleri göster
                    with st.expander(f"📸 Görüntüler — {date_str}", expanded=False):
                        img_c1, img_c2 = st.columns(2)
                        with img_c1:
                            if a.get("original_image_b64"):
                                st.image(
                                    base64_to_image(a["original_image_b64"]),
                                    caption="Orijinal", use_container_width=True,
                                )
                            else:
                                st.info("Görüntü kaydedilmemiş")
                        with img_c2:
                            if a.get("gradcam_image_b64"):
                                st.image(
                                    base64_to_image(a["gradcam_image_b64"]),
                                    caption="Grad-CAM", use_container_width=True,
                                )
                            else:
                                st.info("Grad-CAM kaydedilmemiş")
            else:
                st.info("🔬 Bu hasta için henüz analiz kaydı bulunmamaktadır. Ana sayfadan analiz yaparak kayıt oluşturabilirsiniz.")

            # ── Hasta Silme ──
            st.markdown("---")
            with st.expander("⚠️ Tehlikeli İşlemler"):
                st.warning("Bu işlem hastayı ve tüm analiz geçmişini kalıcı olarak siler!")
                if st.button("🗑️ Hastayı Sil", type="primary"):
                    if delete_patient(patient["id"]):
                        st.success("Hasta silindi.")
                        if "selected_patient_id" in st.session_state:
                            del st.session_state["selected_patient_id"]
                        st.rerun()

# ── Footer ──
st.markdown("""
<div class="footer-bar">
    <p>🏥 Hasta Yönetimi · Retinal AMD v1.0.0</p>
    <p>Hasta bilgileri KVKK kapsamında korunmaktadır.</p>
</div>
""", unsafe_allow_html=True)
