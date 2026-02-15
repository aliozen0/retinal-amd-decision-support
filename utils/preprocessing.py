"""
Retinal AMD — Ön İşleme Modülü
================================
Retinal OCT görüntülerini model girişine uygun tensörlere dönüştüren
ön işleme pipeline'ı.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple

# ============================================================================
# ImageNet normalizasyon değerleri (her iki model için ortak)
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_transforms() -> transforms.Compose:
    """
    Model girişi için standart dönüşüm pipeline'ını döndürür.
    Resize → CenterCrop → ToTensor → Normalize (ImageNet)

    Returns:
        torchvision.transforms.Compose nesnesi
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    PIL görüntüsünü model girişine uygun tensöre dönüştürür.

    Args:
        image: PIL formatında giriş görüntüsü
        device: Hedef hesaplama cihazı (CPU/CUDA)

    Returns:
        [1, 3, 224, 224] boyutunda normalize edilmiş tensör
    """
    # Görüntüyü RGB'ye dönüştür (gri tonlamalı veya RGBA olabilir)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Dönüşüm pipeline'ını uygula
    transform = get_transforms()
    tensor = transform(image)

    # Batch boyutu ekle ve cihaza taşı
    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def prepare_display_image(image: Image.Image) -> np.ndarray:
    """
    PIL görüntüsünü görselleştirme için numpy dizisine dönüştürür.
    224x224 boyutuna yeniden boyutlandırır.

    Args:
        image: PIL formatında giriş görüntüsü

    Returns:
        [224, 224, 3] boyutunda uint8 numpy dizisi
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Görüntüyü model giriş boyutuna yeniden boyutlandır
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

    return np.array(image)
