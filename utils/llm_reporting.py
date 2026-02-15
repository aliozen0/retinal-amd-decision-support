"""
Retinal AMD — LLM Raporlama Modülü
====================================
io.net API üzerinden çoklu LLM model desteği ile klinik rapor üretimi.
OpenAI uyumlu endpoint kullanır.
Loglama ve timeout desteği içerir.
"""

import streamlit as st
import logging
from openai import OpenAI
from typing import Optional

# ── Loglama ──
logger = logging.getLogger("llm_reporting")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)


# ── Kullanılabilir Modeller ──
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-R1-0528",
    "moonshotai/Kimi-K2-Thinking",
    "moonshotai/Kimi-K2-Instruct-0905",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-ar",
    "zai-org/GLM-4.7",
    "zai-org/GLM-4.6",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Devstral-Small-2505",
]

# Kısa görünen isimler (UI için)
MODEL_DISPLAY_NAMES = {m: m.split("/")[-1] for m in AVAILABLE_MODELS}

# Varsayılan model
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"

# API timeout (saniye)
API_TIMEOUT = 120


def _get_client() -> Optional[OpenAI]:
    """
    Streamlit Secrets'dan io.net API bilgilerini alarak OpenAI client döndürür.
    """
    try:
        api_key = st.secrets["io_net"]["api_key"]
        base_url = st.secrets["io_net"].get("base_url", "https://api.intelligence.io.solutions/api/v1/")
        logger.info("OpenAI client oluşturuluyor → base_url=%s", base_url)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=API_TIMEOUT)
    except KeyError as e:
        logger.error("Secrets eksik: %s — io_net bölümünü kontrol edin.", e)
        return None
    except Exception as e:
        logger.error("Client oluşturma hatası: %s", e)
        return None


def get_available_models() -> list[str]:
    """Kullanılabilir model listesini döndürür."""
    return AVAILABLE_MODELS.copy()


def get_model_display_name(model_id: str) -> str:
    """Model ID'sinden kısa görünen ismini döndürür."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id.split("/")[-1])


def is_llm_available() -> bool:
    """LLM API bağlantısının mevcut olup olmadığını kontrol eder."""
    available = _get_client() is not None
    logger.info("LLM kullanılabilirlik: %s", available)
    return available


def generate_llm_report(
    predicted_class: str,
    confidence: float,
    probabilities: list,
    class_names: list,
    model_name: str,
    patient_info: Optional[dict] = None,
    llm_model: Optional[str] = None,
) -> Optional[str]:
    """
    Tek analiz sonucu için LLM destekli klinik rapor üretir.

    Args:
        predicted_class: Tahmin edilen sınıf
        confidence: Güven skoru (0-1)
        probabilities: Tüm sınıf olasılıkları
        class_names: Sınıf isimleri
        model_name: Kullanılan analiz model adı
        patient_info: Hasta bilgileri (opsiyonel)
        llm_model: Kullanılacak LLM modeli (None ise varsayılan)

    Returns:
        LLM tarafından üretilen rapor metni veya None
    """
    client = _get_client()
    if not client:
        logger.warning("Client oluşturulamadı, rapor üretilemiyor.")
        return None

    selected_model = llm_model or DEFAULT_MODEL
    logger.info("Tekli rapor üretimi başlıyor → model=%s, sınıf=%s, güven=%.2f",
                selected_model, predicted_class, confidence)

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
        logger.info("API isteği gönderiliyor → %s (timeout=%ds)", selected_model, API_TIMEOUT)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        result = response.choices[0].message.content
        logger.info("Tekli rapor başarıyla üretildi (%d karakter)", len(result) if result else 0)
        return result
    except Exception as e:
        logger.error("LLM API hatası: %s", e, exc_info=True)
        return f"⚠️ LLM rapor üretimi sırasında hata oluştu: {str(e)}"


def generate_llm_comparative_report(
    analyses: list,
    patient_info: Optional[dict] = None,
    llm_model: Optional[str] = None,
) -> Optional[str]:
    """
    Birden fazla analizi karşılaştıran LLM destekli detaylı rapor üretir.

    Args:
        analyses: Karşılaştırılacak analiz listesi.
        patient_info: Hasta bilgileri (opsiyonel)
        llm_model: Kullanılacak LLM modeli (None ise varsayılan)

    Returns:
        LLM tarafından üretilen karşılaştırma raporu veya None
    """
    client = _get_client()
    if not client:
        logger.warning("Client oluşturulamadı, karşılaştırma raporu üretilemiyor.")
        return None

    selected_model = llm_model or DEFAULT_MODEL
    logger.info("Karşılaştırma raporu başlıyor → model=%s, analiz_sayısı=%d",
                selected_model, len(analyses))

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
        logger.info("Karşılaştırma API isteği → %s (timeout=%ds)", selected_model, API_TIMEOUT)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        result = response.choices[0].message.content
        logger.info("Karşılaştırma raporu başarıyla üretildi (%d karakter)", len(result) if result else 0)
        return result
    except Exception as e:
        logger.error("LLM karşılaştırma API hatası: %s", e, exc_info=True)
        return f"⚠️ LLM karşılaştırma raporu üretimi sırasında hata: {str(e)}"
