"""
Retinal AMD — Model Katmanı
============================
EfficientNet-B4 model mimarisinin tanımlanması,
oluşturulması ve ağırlık dosyalarından yüklenmesi.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from typing import Tuple, List, Optional

# ============================================================================
# Model dosya yolları — eğitim tamamlandığında buradan güncelleyebilirsiniz
# ============================================================================
MODEL_V1_PATH = os.path.join(os.path.dirname(__file__), "sota_99acc.pth")
MODEL_V2_PATH = os.path.join(os.path.dirname(__file__), "supcon_swin_v2_best_sota.pth")

# ============================================================================
# Sınıf eşlemeleri
# ============================================================================
CLASSES_V1: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASSES_V2: List[str] = ["AMD", "DME", "NORMAL"]

# Model seçenekleri (sidebar için)
MODEL_OPTIONS = {
    "EfficientNet-B4 (Yüksek Hız/Kararlılık)": "efficientnet_b4",
    "🔒 Swin-V2 + SupCon (Yakında)": "swin_v2",
}

# Pasif (henüz ağırlığı olmayan) modeller
DISABLED_MODELS = {"swin_v2"}


def create_efficientnet_b4(num_classes: int = 4) -> nn.Module:
    """
    EfficientNet-B4 modelini oluşturur ve son sınıflandırıcı katmanını
    belirtilen sınıf sayısına göre konfigüre eder.

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 4 — CNV, DME, DRUSEN, NORMAL)

    Returns:
        Konfigüre edilmiş EfficientNet-B4 modeli
    """
    # Önceden eğitilmiş ağırlıklar olmadan model oluştur
    model = models.efficientnet_b4(weights=None)

    # Son sınıflandırıcı katmanını hedef sınıf sayısına göre değiştir
    # Kaydedilen .pth dosyasındaki yapı: classifier.1.1 (iç içe Sequential)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def create_swin_v2(num_classes: int = 3) -> nn.Module:
    """
    Swin-V2-B modelini oluşturur. Omurgayı (backbone) dondurur ve
    son katmanı belirtilen sınıf sayısına göre konfigüre eder.

    Not: Bu modelde CNV ve DRUSEN, "AMD" başlığı altında birleştirilmiştir.

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 3 — AMD, DME, NORMAL)

    Returns:
        Konfigüre edilmiş ve omurgası dondurulmuş Swin-V2-B modeli
    """
    # Önceden eğitilmiş ağırlıklar olmadan model oluştur
    model = models.swin_v2_b(weights=None)

    # Omurgayı dondur — sadece head eğitilebilir
    for param in model.parameters():
        param.requires_grad = False

    # Son sınıflandırıcı katmanını değiştir
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    # Head katmanını eğitilebilir yap
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def get_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """
    Grad-CAM için hedef katmanı döndürür.

    Args:
        model: PyTorch modeli
        model_type: Model tipi ("efficientnet_b4" veya "swin_v2")

    Returns:
        Grad-CAM için hedef katman modülü
    """
    if model_type == "efficientnet_b4":
        # EfficientNet'in son özellik çıkarma bloğu
        return model.features[-1]
    else:
        # Swin-V2'nin normalizasyon katmanı
        return model.norm


def get_classes(model_type: str) -> List[str]:
    """
    Model tipine göre sınıf isimlerini döndürür.

    Args:
        model_type: Model tipi ("efficientnet_b4" veya "swin_v2")

    Returns:
        Sınıf isimlerinin listesi
    """
    if model_type == "efficientnet_b4":
        return CLASSES_V1
    return CLASSES_V2


@st.cache_resource
def load_model(model_type: str, device_str: str) -> Tuple[nn.Module, bool]:
    """
    Belirtilen model tipini yükler. Ağırlık dosyası mevcutsa diskten yükler,
    yoksa demo modunda (rastgele ağırlıklarla) çalışır.

    st.cache_resource ile sarmalanarak tekrar tekrar yükleme engellenir.

    Args:
        model_type: Model tipi ("efficientnet_b4" veya "swin_v2")
        device_str: Hedef cihaz string'i ("cuda" veya "cpu")

    Returns:
        (model, is_demo_mode) tuple'ı
    """
    device = torch.device(device_str)
    is_demo_mode = False

    # Model mimarisini oluştur
    if model_type == "efficientnet_b4":
        model = create_efficientnet_b4(num_classes=4)
        weight_path = MODEL_V1_PATH
    else:
        model = create_swin_v2(num_classes=3)
        weight_path = MODEL_V2_PATH

    # Ağırlık dosyasını yüklemeye çalış
    if os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)

            # Eğer state_dict bir dict içinde sarmalanmışsa çöz
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict, strict=False)
            st.success(f"✅ Model ağırlıkları başarıyla yüklendi: `{weight_path}`")
        except Exception as e:
            st.warning(
                f"⚠️ Model ağırlıkları yüklenirken hata oluştu: {e}\n"
                f"Demo modunda devam ediliyor."
            )
            is_demo_mode = True
    else:
        st.warning(
            f"⚠️ Model dosyası bulunamadı: `{weight_path}`\n"
            f"Demo modunda (rastgele ağırlıklarla) devam ediliyor."
        )
        is_demo_mode = True

    # Modeli değerlendirme moduna al ve cihaza taşı
    model = model.to(device)
    model.eval()

    return model, is_demo_mode
