"""
HÜMA-MED Klinik Raporlama Modülü
==================================
Tahmin sonuçlarına göre otomatik Türkçe klinik rapor metni üreten modül.

Yazar: HÜMA-MED Ekibi
"""

from typing import Dict


# ============================================================================
# Patoloji açıklamaları — klinik rapor metinlerinde kullanılır
# ============================================================================
PATHOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "CNV": (
        "Koroidal Neovaskülarizasyon (CNV) bulgusu saptanmıştır. "
        "Retina altında anormal damar oluşumu gözlemlenmektedir. "
        "Anti-VEGF tedavi uygunluğunun değerlendirilmesi önerilir."
    ),
    "DME": (
        "Diyabetik Makula Ödemi (DME) bulgusu saptanmıştır. "
        "Makula bölgesinde sıvı birikimi gözlemlenmektedir. "
        "Diyabet yönetiminin gözden geçirilmesi ve göz içi tedavi seçeneklerinin "
        "değerlendirilmesi önerilir."
    ),
    "DRUSEN": (
        "Drusen birikimi saptanmıştır. "
        "Retina pigment epiteli altında sarımsı birikintiler gözlemlenmektedir. "
        "Yaşa bağlı makula dejenerasyonu (AMD) riski açısından düzenli takip önerilir."
    ),
    "AMD": (
        "Yaşa Bağlı Makula Dejenerasyonu (AMD) bulgusu saptanmıştır. "
        "Makula bölgesinde dejeneratif değişiklikler gözlemlenmektedir. "
        "İleri tetkik ve tedavi planlaması önerilir."
    ),
    "NORMAL": (
        "Görüntüde patolojik bulgu saptanmamıştır. "
        "Retinal yapılar normal sınırlarda gözlemlenmektedir. "
        "Rutin kontrol takvimine devam edilmesi önerilir."
    ),
}


def generate_clinical_report(
    model_name: str,
    predicted_class: str,
    confidence: float,
    is_swin_v2: bool = False,
) -> str:
    """
    Tahmin sonucuna göre otomatik Türkçe klinik rapor metni üretir.

    Args:
        model_name: Kullanılan model adı (arayüzde gösterim için)
        predicted_class: Tahmin edilen sınıf adı (CNV, DME, DRUSEN, AMD, NORMAL)
        confidence: Güven skoru (0-1 arası)
        is_swin_v2: Swin-V2 modeli kullanılıp kullanılmadığı

    Returns:
        Formatlanmış klinik rapor metni
    """
    confidence_pct = confidence * 100

    # Rapor başlığı
    report_lines = [
        f"📋 **KLİNİK ANALİZ RAPORU**",
        f"",
        f"**Kullanılan Model:** {model_name}",
        f"**Tahmin:** {predicted_class}",
        f"**Güven Oranı:** %{confidence_pct:.1f}",
        f"",
        "---",
        f"",
    ]

    if predicted_class == "NORMAL":
        # Normal bulgu
        report_lines.append(
            f"✅ {PATHOLOGY_DESCRIPTIONS['NORMAL']}"
        )
    else:
        # Patolojik bulgu
        description = PATHOLOGY_DESCRIPTIONS.get(
            predicted_class,
            f"{predicted_class} bulgusu saptanmıştır."
        )
        report_lines.append(
            f"🔴 **{model_name}** analizi sonucunda, görüntüde "
            f"**%{confidence_pct:.1f}** güven oranıyla **{predicted_class}** "
            f"bulgusu saptanmıştır."
        )
        report_lines.append(f"")
        report_lines.append(description)
        report_lines.append(f"")
        report_lines.append(
            "🔍 Grad-CAM ısı haritasında işaretlenen bölgelere "
            "odaklanılması önerilir."
        )

    # Swin-V2'de AMD tanısı geldiğinde ek bilgilendirme notu
    if is_swin_v2 and predicted_class == "AMD":
        report_lines.append(f"")
        report_lines.append(
            "⚠️ **Önemli Not:** Swin-V2 modelinde AMD tanısı, "
            "**CNV (Koroidal Neovaskülarizasyon)** veya **DRUSEN** "
            "kaynaklı olabilir. Bu iki alt tip, model eğitiminde AMD "
            "başlığı altında birleştirilmiştir. Kesin ayırıcı tanı için "
            "detaylı klinik inceleme önerilir."
        )

    # Güven skoru düşükse uyarı
    if confidence_pct < 70.0:
        report_lines.append(f"")
        report_lines.append(
            f"⚡ **Düşük Güven Uyarısı:** Güven oranı %{confidence_pct:.1f} "
            f"olup, bu sonucun dikkatli yorumlanması ve klinik korelasyon "
            f"ile doğrulanması önerilir."
        )

    # Yasal uyarı
    report_lines.append(f"")
    report_lines.append("---")
    report_lines.append(
        "*Bu rapor yapay zekâ destekli bir analiz sonucudur ve kesin tanı "
        "niteliği taşımaz. Klinik karar verme sürecinde uzman hekim "
        "değerlendirmesi esastır.*"
    )

    return "\n".join(report_lines)
