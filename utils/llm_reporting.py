"""
Retinal AMD — LLM Raporlama Modülü
====================================
io.net API üzerinden DeepSeek-V3.2 ile klinik rapor üretimi.
OpenAI uyumlu endpoint kullanır.
"""

import streamlit as st
from openai import OpenAI
from typing import Optional


def _get_client() -> Optional[OpenAI]:
    """
    Streamlit Secrets'dan io.net API bilgilerini alarak OpenAI client döndürür.
    """
    try:
        api_key = st.secrets["io_net"]["api_key"]
        base_url = st.secrets["io_net"].get("base_url", "https://api.intelligence.io.solutions/api/v1/")
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


def _get_model() -> str:
    """Kullanılacak model adını döndürür."""
    try:
        return st.secrets["io_net"].get("model", "deepseek-ai/DeepSeek-V3.2")
    except Exception:
        return "deepseek-ai/DeepSeek-V3.2"


def is_llm_available() -> bool:
    """LLM API bağlantısının mevcut olup olmadığını kontrol eder."""
    return _get_client() is not None


def generate_llm_report(
    predicted_class: str,
    confidence: float,
    probabilities: list,
    class_names: list,
    model_name: str,
    patient_info: Optional[dict] = None,
) -> Optional[str]:
    """
    Tek analiz sonucu için LLM destekli klinik rapor üretir.

    Args:
        predicted_class: Tahmin edilen sınıf
        confidence: Güven skoru (0-1)
        probabilities: Tüm sınıf olasılıkları
        class_names: Sınıf isimleri
        model_name: Kullanılan model adı
        patient_info: Hasta bilgileri (opsiyonel)

    Returns:
        LLM tarafından üretilen rapor metni veya None
    """
    client = _get_client()
    if not client:
        return None

    # Olasılık dağılımı metni
    prob_text = "\n".join(
        f"  - {name}: %{prob*100:.1f}" for name, prob in zip(class_names, probabilities)
    )

    # Hasta bilgisi metni
    patient_text = ""
    if patient_info:
        ad = patient_info.get("ad", "")
        soyad = patient_info.get("soyad", "")
        dosya = patient_info.get("dosya_no", "")
        dogum = patient_info.get("dogum_tarihi", "")
        patient_text = f"""
Hasta Bilgileri:
  - Ad Soyad: {ad} {soyad}
  - Dosya No: {dosya}
  - Doğum Tarihi: {dogum}
"""

    system_prompt = """Sen retinal OCT (Optik Koherens Tomografi) görüntülerini analiz eden bir yapay zekâ klinik karar destek sisteminin rapor yazarısın. Türkçe ve profesyonel tıbbi dilde yaz.

Kurallar:
- Kısa ve öz yaz, gereksiz detaylardan kaçın.
- Bulgular, değerlendirme ve öneriler bölümlerini kullan.
- Bu bir KARAR DESTEK sistemidir, kesin tanı koymaz. Bunu her zaman belirt.
- Güven skoru %80 altındaysa bunu özellikle vurgula.
- Sonuçları klinik bağlamda açıkla.
- Markdown formatı kullan (başlıklar, kalın, listeler)."""

    user_prompt = f"""{patient_text}
Analiz Sonuçları:
  - Yapay Zekâ Modeli: {model_name}
  - Tahmin Edilen Sınıf: {predicted_class}
  - Güven Skoru: %{confidence*100:.1f}

Olasılık Dağılımı:
{prob_text}

Bu analiz sonuçlarına dayalı kısa bir klinik rapor yaz."""

    try:
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM rapor üretimi sırasında hata oluştu: {str(e)}"


def generate_llm_comparative_report(
    analyses: list,
    patient_info: Optional[dict] = None,
) -> Optional[str]:
    """
    Birden fazla analizi karşılaştıran LLM destekli detaylı rapor üretir.

    Args:
        analyses: Karşılaştırılacak analiz listesi.
                  Her biri dict: predicted_class, confidence, analysis_date,
                  model_name, report_text
        patient_info: Hasta bilgileri (opsiyonel)

    Returns:
        LLM tarafından üretilen karşılaştırma raporu veya None
    """
    client = _get_client()
    if not client:
        return None

    # Analiz özetleri
    analysis_texts = []
    for i, a in enumerate(analyses):
        date_str = a.get("analysis_date", "?")[:16].replace("T", " ")
        analysis_texts.append(
            f"Analiz #{i+1} ({date_str}):\n"
            f"  - Tanı: {a.get('predicted_class', '?')}\n"
            f"  - Güven: %{a.get('confidence', 0)*100:.1f}\n"
            f"  - Model: {a.get('model_name', '?')}"
        )
    analyses_block = "\n\n".join(analysis_texts)

    # Hasta bilgisi
    patient_text = ""
    if patient_info:
        ad = patient_info.get("ad", "")
        soyad = patient_info.get("soyad", "")
        dosya = patient_info.get("dosya_no", "")
        patient_text = f"Hasta: {ad} {soyad} (Dosya No: {dosya})\n"

    system_prompt = """Sen retinal OCT görüntülerini analiz eden bir yapay zekâ klinik karar destek sisteminin karşılaştırma raporu yazarısın. Türkçe ve profesyonel tıbbi dilde yaz.

Kurallar:
- Bu bir KARŞILAŞTIRMALI rapordur, analizler arasındaki değişimleri analiz et.
- Hastalık ilerlemesi veya iyileşme durumunu değerlendir.
- Güven skoru değişimlerini yorumla.
- Sınıf değişikliği varsa bu kritik bir bulgudur, özellikle vurgula.
- Tedavi sürecine ilişkin genel bir değerlendirme yap.
- Takip önerilerinde bulun.
- Markdown formatı kullan.
- Bu bir karar destek sistemidir, kesin tanı koymaz."""

    user_prompt = f"""{patient_text}
{analyses_block}

Bu {len(analyses)} analizi karşılaştırarak bir klinik değişim raporu yaz. Hastalık seyri, güven değişimleri ve takip önerileri hakkında detaylı değerlendirme yap."""

    try:
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM karşılaştırma raporu üretimi sırasında hata: {str(e)}"
