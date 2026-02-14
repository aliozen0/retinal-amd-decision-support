"""
HÜMA-MED PDF Raporlama Modülü
==============================
Analiz sonuçlarını (orijinal görüntü, Grad-CAM ısı haritası,
olasılık değerleri ve klinik rapor) profesyonel PDF formatında dışa aktarır.

Yazar: HÜMA-MED Ekibi
"""

import os
import tempfile
import numpy as np
from PIL import Image as PILImage
from fpdf import FPDF
from datetime import datetime
from typing import List


# ============================================================================
# Windows sistem fontu — Türkçe karakter desteği için
# ============================================================================
ARIAL_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
ARIAL_BOLD_PATH = r"C:\Windows\Fonts\arialbd.ttf"
ARIAL_ITALIC_PATH = r"C:\Windows\Fonts\ariali.ttf"


class HumaMedPDF(FPDF):
    """HÜMA-MED için profesyonel PDF rapor sınıfı. Unicode destekli."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

        # Unicode destekli Arial fontunu kaydet
        if os.path.exists(ARIAL_FONT_PATH):
            self.add_font("Arial-TR", "", ARIAL_FONT_PATH, uni=True)
        if os.path.exists(ARIAL_BOLD_PATH):
            self.add_font("Arial-TR", "B", ARIAL_BOLD_PATH, uni=True)
        if os.path.exists(ARIAL_ITALIC_PATH):
            self.add_font("Arial-TR", "I", ARIAL_ITALIC_PATH, uni=True)

        self._font_name = "Arial-TR" if os.path.exists(ARIAL_FONT_PATH) else "Helvetica"

    def _set(self, style: str = "", size: int = 10) -> None:
        """Kısa font ayar yardımcısı."""
        self.set_font(self._font_name, style, size)

    def header(self) -> None:
        """Sayfa üst bilgisi — profesyonel beyaz tasarım."""
        # Üst çizgi — indigo accent
        self.set_fill_color(79, 70, 229)
        self.rect(0, 0, 210, 3, "F")

        # Logo & başlık
        self.set_y(8)
        self._set("B", 16)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, "HÜMA-MED", ln=False)

        # Sağ üst — tarih
        self._set("", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, datetime.now().strftime("%d.%m.%Y — %H:%M"), align="R")
        self.ln(6)

        # Alt başlık
        self._set("", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Klinik Karar Destek Raporu", ln=True)

        # Ayırıcı çizgi
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 3, 200, self.get_y() + 3)
        self.ln(8)

    def footer(self) -> None:
        """Sayfa alt bilgisi."""
        self.set_y(-18)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self._set("I", 7)
        self.set_text_color(148, 163, 184)
        self.cell(
            0, 5,
            "Bu rapor yapay zekâ destekli bir analiz sonucudur ve "
            "kesin tanı niteliği taşımaz. Klinik karar sürecinde uzman hekim "
            "değerlendirmesi esastır.",
            align="C",
        )
        self.ln(3)
        self._set("", 7)
        self.cell(0, 5, f"HÜMA-MED v1.0  |  Sayfa {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str) -> None:
        """Bölüm başlığı — sol kenar indigo çizgili."""
        self.ln(4)
        # Sol accent bar
        y = self.get_y()
        self.set_fill_color(79, 70, 229)
        self.rect(10, y, 2.5, 7, "F")

        self._set("B", 12)
        self.set_text_color(30, 41, 59)
        self.set_x(16)
        self.cell(0, 7, title, ln=True)
        self.ln(2)


def generate_pdf_report(
    original_image: np.ndarray,
    gradcam_image: np.ndarray,
    predicted_class: str,
    confidence: float,
    class_names: List[str],
    probabilities: np.ndarray,
    model_name: str,
    report_text: str,
) -> bytes:
    """
    Analiz sonuçlarını profesyonel PDF formatında üretir.

    Returns:
        PDF dosyasının bytes içeriği
    """
    pdf = HumaMedPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ══════════════════════════════════════════════════
    # 1. ANALİZ ÖZETİ
    # ══════════════════════════════════════════════════
    pdf.section_title("Analiz Özeti")

    # Bilgi tablosu — 2 sütunlu düzen
    info_data = [
        ("Kullanılan Model", model_name),
        ("Tahmin Edilen Tanı", predicted_class),
        ("Güven Oranı", f"%{confidence * 100:.1f}"),
        ("Sınıf Sayısı", str(len(class_names))),
        ("Analiz Tarihi", datetime.now().strftime("%d.%m.%Y — %H:%M:%S")),
    ]

    for label, value in info_data:
        pdf._set("", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(50, 6, label, ln=False)

        pdf._set("B", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, value, ln=True)

    pdf.ln(4)

    # ══════════════════════════════════════════════════
    # 2. GÖRÜNTÜ ANALİZİ
    # ══════════════════════════════════════════════════
    pdf.section_title("Görüntü Analizi")

    tmp_files = []
    try:
        # Orijinal
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            PILImage.fromarray(original_image).save(f, format="PNG")
            tmp_orig = f.name
            tmp_files.append(tmp_orig)

        # Grad-CAM
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            PILImage.fromarray(gradcam_image).save(f, format="PNG")
            tmp_gradcam = f.name
            tmp_files.append(tmp_gradcam)

        img_w = 85  # mm
        y_label = pdf.get_y()

        # Etiketler
        pdf._set("B", 8)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(img_w + 5, 5, "Orijinal Görüntü", align="C", ln=False)
        pdf.cell(img_w + 5, 5, "Grad-CAM Isı Haritası", align="C", ln=True)

        # Görüntüler
        img_y = pdf.get_y() + 1
        pdf.image(tmp_orig, x=12, y=img_y, w=img_w)
        pdf.image(tmp_gradcam, x=12 + img_w + 6, y=img_y, w=img_w)

        # Görüntülerin altına geçiş
        pdf.set_y(img_y + img_w + 4)

    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    pdf.ln(2)

    # ══════════════════════════════════════════════════
    # 3. OLASILIK DAĞILIMI
    # ══════════════════════════════════════════════════
    pdf.section_title("Sınıf Olasılık Dağılımı")

    # Tablo başlığı
    col_widths = [40, 30, 30, 50]  # toplam = 150mm (sayfa içinde kalır)
    headers = ["Sınıf", "Olasılık", "Durum", "Görsel"]

    pdf._set("B", 9)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_text_color(51, 65, 85)
    pdf.set_draw_color(226, 232, 240)

    x_start = (210 - sum(col_widths)) / 2  # Tabloyu ortala
    pdf.set_x(x_start)
    for header, w in zip(headers, col_widths):
        pdf.cell(w, 7, header, border=1, fill=True, align="C")
    pdf.ln()

    # Tablo satırları
    for name, prob in zip(class_names, probabilities):
        is_predicted = name == predicted_class
        prob_pct = prob * 100

        if is_predicted:
            pdf.set_fill_color(238, 242, 255)
            pdf._set("B", 9)
            pdf.set_text_color(67, 56, 202)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf._set("", 9)
            pdf.set_text_color(51, 65, 85)

        pdf.set_x(x_start)

        # Sınıf adı
        pdf.cell(col_widths[0], 7, name, border=1, fill=True, align="C")

        # Olasılık
        pdf.cell(col_widths[1], 7, f"%{prob_pct:.2f}", border=1, fill=True, align="C")

        # Durum
        status = "● Tahmin" if is_predicted else ""
        pdf.cell(col_widths[2], 7, status, border=1, fill=True, align="C")

        # Görsel bar — max 20 karakter (sütuna sığacak şekilde)
        bar_count = max(1, int(prob * 20))
        bar_text = "█" * bar_count
        pdf.cell(col_widths[3], 7, bar_text, border=1, fill=True, align="L")
        pdf.ln()

    pdf.ln(4)

    # ══════════════════════════════════════════════════
    # 4. KLİNİK RAPOR
    # ══════════════════════════════════════════════════
    pdf.section_title("Klinik Değerlendirme Raporu")

    # Markdown sembollerini temizle
    clean_report = report_text
    for marker in ["**", "📋", "🔴", "✅", "🔍", "⚡", "⚠️", "---"]:
        clean_report = clean_report.replace(marker, "")

    # Italic yıldız temizliği
    while clean_report.startswith("*"):
        clean_report = clean_report[1:]
    while clean_report.endswith("*"):
        clean_report = clean_report[:-1]
    clean_report = clean_report.replace("*", "")

    # Rapor kutucuğu — hafif gri arka plan
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    box_y = pdf.get_y()

    # Rapor metnini yaz
    pdf._set("", 9)
    pdf.set_text_color(51, 65, 85)

    lines = [line.strip() for line in clean_report.strip().split("\n") if line.strip()]
    for line in lines:
        pdf.multi_cell(180, 5, line, align="L")
        pdf.ln(1)

    pdf.ln(4)

    # ══════════════════════════════════════════════════
    # 5. SORUMLULUK REDDİ
    # ══════════════════════════════════════════════════
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_fill_color(255, 251, 235)  # Sarımsı arka plan
    pdf.set_draw_color(253, 224, 71)

    pdf._set("B", 8)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 5, "Yasal Uyarı", ln=True)

    pdf._set("I", 7)
    pdf.set_text_color(146, 64, 14)
    pdf.multi_cell(
        0, 4,
        "Bu rapor HÜMA-MED Klinik Karar Destek Sistemi tarafından otomatik "
        "olarak üretilmiştir. Yapay zekâ destekli analiz sonuçları kesin tanı "
        "niteliği taşımamaktadır. Tüm bulgular uzman hekim tarafından klinik "
        "korelasyon ile değerlendirilmelidir.",
        align="L",
    )

    return bytes(pdf.output())
