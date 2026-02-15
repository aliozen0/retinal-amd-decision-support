"""
Retinal AMD — Veritabanı Modülü (Supabase)
=============================================
Supabase PostgreSQL üzerinde hasta ve analiz CRUD operasyonları.
Streamlit Secrets ile bağlantı yönetimi.
"""

import base64
import io
import numpy as np
from PIL import Image as PILImage
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any

import streamlit as st

# Türkiye saat dilimi (GMT+3)
TZ_TR = timezone(timedelta(hours=3))


# ============================================================================
# Supabase Bağlantısı
# ============================================================================
@st.cache_resource
def init_supabase():
    """
    Streamlit Secrets'dan Supabase bağlantı bilgilerini alır ve client döndürür.
    st.cache_resource ile sarmalanarak tekrar tekrar bağlantı kurulması engellenir.

    Returns:
        Supabase client nesnesi veya None (yapılandırma eksikse)
    """
    try:
        from supabase import create_client, Client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        client: Client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"⚠️ Supabase bağlantısı kurulamadı: {e}")
        return None


def is_db_available() -> bool:
    """Veritabanı bağlantısının mevcut olup olmadığını kontrol eder."""
    client = init_supabase()
    return client is not None


# ============================================================================
# Görüntü Dönüşüm Yardımcıları
# ============================================================================
def image_to_base64(np_image: np.ndarray, max_size: int = 224) -> str:
    """
    Numpy dizisindeki görüntüyü base64 string'e dönüştürür.
    Veritabanında saklamak için küçültülmüş boyutta encode eder.

    Args:
        np_image: [H, W, 3] boyutunda uint8 numpy görüntü
        max_size: Maksimum kenar uzunluğu (piksel)

    Returns:
        Base64 encode edilmiş PNG string
    """
    img = PILImage.fromarray(np_image)
    # Boyut küçültme
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str) -> np.ndarray:
    """
    Base64 string'i numpy dizisine dönüştürür.

    Args:
        b64_str: Base64 encode edilmiş PNG string

    Returns:
        [H, W, 3] boyutunda uint8 numpy dizisi
    """
    img_bytes = base64.b64decode(b64_str)
    img = PILImage.open(io.BytesIO(img_bytes))
    return np.array(img.convert("RGB"))


# ============================================================================
# Hasta CRUD Operasyonları
# ============================================================================
def add_patient(
    dosya_no: str,
    ad: str,
    soyad: str,
    dogum_tarihi: Optional[str] = None,
    telefon: Optional[str] = None,
    email: Optional[str] = None,
    notlar: Optional[str] = None,
) -> Optional[Dict]:
    """
    Yeni hasta kaydı ekler.

    Args:
        dosya_no: Hasta dosya numarası (benzersiz)
        ad: Hasta adı
        soyad: Hasta soyadı
        dogum_tarihi: Doğum tarihi (YYYY-MM-DD)
        telefon: Telefon numarası
        email: E-posta adresi
        notlar: Ek notlar

    Returns:
        Eklenen hasta verisi veya None (hata durumunda)
    """
    client = init_supabase()
    if not client:
        return None

    data = {
        "dosya_no": dosya_no.strip(),
        "ad": ad.strip(),
        "soyad": soyad.strip(),
    }
    if dogum_tarihi:
        data["dogum_tarihi"] = dogum_tarihi
    if telefon:
        data["telefon"] = telefon.strip()
    if email:
        data["email"] = email.strip()
    if notlar:
        data["notlar"] = notlar.strip()

    try:
        result = client.table("patients").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Hasta eklenirken hata: {e}")
        return None


def search_patients(query: str = "") -> List[Dict]:
    """
    Hasta arama — ad, soyad veya dosya no ile filtreleme.

    Args:
        query: Arama metni (boş ise tüm hastalar döner)

    Returns:
        Eşleşen hasta listesi
    """
    client = init_supabase()
    if not client:
        return []

    try:
        if query.strip():
            q = query.strip()
            result = (
                client.table("patients")
                .select("*")
                .or_(
                    f"ad.ilike.%{q}%,"
                    f"soyad.ilike.%{q}%,"
                    f"dosya_no.ilike.%{q}%"
                )
                .order("created_at", desc=True)
                .execute()
            )
        else:
            result = (
                client.table("patients")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
        return result.data or []
    except Exception as e:
        st.error(f"Hasta aranırken hata: {e}")
        return []


def get_patient(patient_id: str) -> Optional[Dict]:
    """Belirtilen ID'ye sahip hastayı getirir."""
    client = init_supabase()
    if not client:
        return None

    try:
        result = (
            client.table("patients")
            .select("*")
            .eq("id", patient_id)
            .single()
            .execute()
        )
        return result.data
    except Exception as e:
        st.error(f"Hasta bilgisi alınırken hata: {e}")
        return None


def update_patient(patient_id: str, **kwargs) -> Optional[Dict]:
    """
    Hasta bilgilerini günceller.

    Args:
        patient_id: Hasta UUID
        **kwargs: Güncellenecek alanlar
    """
    client = init_supabase()
    if not client:
        return None

    try:
        kwargs["updated_at"] = datetime.now(TZ_TR).isoformat()
        result = (
            client.table("patients")
            .update(kwargs)
            .eq("id", patient_id)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Hasta güncellenirken hata: {e}")
        return None


def delete_patient(patient_id: str) -> bool:
    """Hastayı ve ilişkili analizlerini siler."""
    client = init_supabase()
    if not client:
        return False

    try:
        client.table("patients").delete().eq("id", patient_id).execute()
        return True
    except Exception as e:
        st.error(f"Hasta silinirken hata: {e}")
        return False


def get_all_patients() -> List[Dict]:
    """Tüm hastaları listeler (selectbox için)."""
    return search_patients("")


# ============================================================================
# Analiz CRUD Operasyonları
# ============================================================================
def save_analysis(
    patient_id: str,
    predicted_class: str,
    confidence: float,
    probabilities: list,
    model_name: str,
    original_image: Optional[np.ndarray] = None,
    gradcam_image: Optional[np.ndarray] = None,
    report_text: Optional[str] = None,
) -> Optional[Dict]:
    """
    Analiz sonucunu veritabanına kaydeder.

    Args:
        patient_id: Hasta UUID
        predicted_class: Tahmin edilen sınıf
        confidence: Güven skoru (0-1)
        probabilities: Sınıf olasılıkları listesi
        model_name: Kullanılan model adı
        original_image: Orijinal görüntü (numpy)
        gradcam_image: Grad-CAM görüntüsü (numpy)
        report_text: Klinik rapor metni

    Returns:
        Kaydedilen analiz verisi veya None
    """
    client = init_supabase()
    if not client:
        return None

    data = {
        "patient_id": patient_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_name": model_name,
        "analysis_date": datetime.now(TZ_TR).isoformat(),
    }

    if original_image is not None:
        data["original_image_b64"] = image_to_base64(original_image)
    if gradcam_image is not None:
        data["gradcam_image_b64"] = image_to_base64(gradcam_image)
    if report_text:
        data["report_text"] = report_text

    try:
        result = client.table("analyses").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Analiz kaydedilirken hata: {e}")
        return None


def get_patient_analyses(patient_id: str) -> List[Dict]:
    """
    Hastanın tüm analizlerini kronolojik sırayla getirir.

    Args:
        patient_id: Hasta UUID

    Returns:
        Analiz listesi (yeniden eskiye sıralı)
    """
    client = init_supabase()
    if not client:
        return []

    try:
        result = (
            client.table("analyses")
            .select("*")
            .eq("patient_id", patient_id)
            .order("analysis_date", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        st.error(f"Analiz geçmişi alınırken hata: {e}")
        return []


def get_analysis(analysis_id: str) -> Optional[Dict]:
    """Belirtilen ID'ye sahip analizi getirir."""
    client = init_supabase()
    if not client:
        return None

    try:
        result = (
            client.table("analyses")
            .select("*")
            .eq("id", analysis_id)
            .single()
            .execute()
        )
        return result.data
    except Exception as e:
        st.error(f"Analiz bilgisi alınırken hata: {e}")
        return None


def get_patient_analysis_count(patient_id: str) -> int:
    """Hastanın toplam analiz sayısını döndürür."""
    client = init_supabase()
    if not client:
        return 0

    try:
        result = (
            client.table("analyses")
            .select("id", count="exact")
            .eq("patient_id", patient_id)
            .execute()
        )
        return result.count or 0
    except Exception:
        return 0
