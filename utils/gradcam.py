"""
Retinal AMD — Grad-CAM Modülü
===============================
Hook tabanlı saf PyTorch Grad-CAM implementasyonu.
Harici kütüphane bağımlılığı yoktur (pytorch-grad-cam vb. kullanılmaz).
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """
    Hook tabanlı Gradient-weighted Class Activation Mapping (Grad-CAM).

    register_forward_hook ve register_full_backward_hook kullanarak
    hedef katmandan aktivasyon ve gradyan bilgisi yakalar, ardından
    sınıfa özgü ısı haritası üretir.

    Attributes:
        model: PyTorch modeli
        target_layer: Grad-CAM için hedef katman
        activations: Forward pass'te yakalanan aktivasyonlar
        gradients: Backward pass'te yakalanan gradyanlar
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        GradCAM nesnesini oluşturur ve hook'ları kaydeder.

        Args:
            model: Değerlendirme modundaki PyTorch modeli
            target_layer: Aktivasyon ve gradyan yakalanacak katman
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hook'ları kaydet
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """Forward hook: Aktivasyonları yakala ve kaydet."""
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        """Backward hook: Gradyanları yakala ve kaydet."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Verilen giriş tensörü için Grad-CAM ısı haritası üretir.

        Args:
            input_tensor: [1, 3, H, W] boyutunda giriş tensörü
            target_class: Hedef sınıf indeksi. None ise en yüksek
                          olasılıklı sınıf kullanılır.

        Returns:
            [H, W] boyutunda 0-1 arasında normalize edilmiş ısı haritası (numpy)
        """
        # Gradyan hesaplaması için requires_grad aktifleştir
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Hedef sınıfı belirle
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass — hedef sınıfın skoruna göre
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Grad-CAM hesapla
        if self.gradients is None or self.activations is None:
            # Hook'lar çalışmadıysa boş harita döndür
            return np.zeros((224, 224), dtype=np.float32)

        # Global Average Pooling — gradyanlar üzerinde
        weights = self.gradients.mean(dim=[2, 3] if self.gradients.dim() == 4 else [-1], keepdim=True)

        # Ağırlıklı toplam — aktivasyonlar × ağırlıklar
        if self.activations.dim() == 4:
            # CNN tarzı çıktı: [B, C, H, W]
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
        elif self.activations.dim() == 3:
            # Transformer tarzı çıktı: [B, N, C]
            cam = (weights * self.activations).sum(dim=-1, keepdim=True)
            # Token dizisini 2D haritaya dönüştür
            num_tokens = cam.shape[1]
            h = w = int(num_tokens ** 0.5)
            if h * w != num_tokens:
                # CLS token varsa veya boyut uyuşmuyorsa en yakın kareyi al
                h = w = int(np.ceil(num_tokens ** 0.5))
                cam = cam[:, :h * w, :]
            cam = cam.reshape(1, 1, h, w)

        # ReLU — sadece pozitif katkılar
        cam = F.relu(cam)

        # Giriş boyutuna yeniden boyutlandır
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # 0-1 arasında normalize et
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def remove_hooks(self) -> None:
        """Kayıtlı hook'ları temizle (bellek sızıntısını önlemek için)."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int],
    target_layer: nn.Module,
) -> np.ndarray:
    """
    Model ve giriş tensörü için Grad-CAM ısı haritası üretir.

    Args:
        model: PyTorch modeli (eval modunda)
        input_tensor: [1, 3, 224, 224] boyutunda giriş tensörü
        target_class: Hedef sınıf indeksi (None ise argmax)
        target_layer: Grad-CAM hedef katmanı

    Returns:
        [224, 224] boyutunda 0-1 normalize ısı haritası
    """
    grad_cam = GradCAM(model, target_layer)
    try:
        heatmap = grad_cam.generate(input_tensor, target_class)
    finally:
        # Hook'ları temizle
        grad_cam.remove_hooks()
    return heatmap


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Grad-CAM ısı haritasını orijinal görüntü üzerine bindirir.

    Args:
        original_image: [H, W, 3] boyutunda uint8 RGB görüntü
        heatmap: [H, W] boyutunda 0-1 arasında normalize ısı haritası
        alpha: Bindirme oranı (0=sadece görüntü, 1=sadece ısı haritası)

    Returns:
        [H, W, 3] boyutunda bindirilmiş uint8 RGB görüntü
    """
    # Isı haritasını 0-255 aralığına dönüştür
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Isı haritasını görüntü boyutuna yeniden boyutlandır
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    # JET colormap uygula (mavi→yeşil→kırmızı)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # BGR → RGB dönüşümü (OpenCV BGR formatında çalışır)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Orijinal görüntü ile ısı haritasını birleştir
    overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlaid
