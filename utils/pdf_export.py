"""
Retinal AMD — PDF Raporlama Modülü
====================================
Analiz sonuçlarını (orijinal görüntü, Grad-CAM ısı haritası,
olasılık değerleri ve klinik rapor) profesyonel PDF formatında dışa aktarır.

Cross-platform Unicode desteği: Windows (Arial), Linux (DejaVu Sans).
"""

import os
import tempfile
import numpy as np
from PIL import Image as PILImage
from fpdf import FPDF
from datetime import datetime, timezone, timedelta

# Turkiye saat dilimi (GMT+3)
TZ_TR = timezone(timedelta(hours=3))
from typing import List, Optional


# ============================================================================
# Cross-platform font arama — hem Windows hem Linux (Streamlit Cloud) desteği
# ============================================================================
def _find_font(candidates: list) -> Optional[str]:
    """Verilen aday yollarından ilk mevcut olanı döndürür."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# Regular font adayları
_REGULAR_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",                              # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Debian/Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                      # Arch
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",        # Fedora
]

_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
]

_ITALIC_CANDIDATES = [
    r"C:\Windows\Fonts\ariali.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Oblique.ttf",
]

FONT_REGULAR = _find_font(_REGULAR_CANDIDATES)
FONT_BOLD = _find_font(_BOLD_CANDIDATES)
FONT_ITALIC = _find_font(_ITALIC_CANDIDATES)


class RetinalPDF(FPDF):
    """Profesyonel PDF rapor sinifi. Cross-platform Unicode destekli."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

        # Unicode destekli fontu kaydet (platform-agnostic)
        self._font_name = "Helvetica"  # fallback
        self._available_styles = {""}  # her zaman regular mevcut

        if FONT_REGULAR:
            self.add_font("UniFont", "", FONT_REGULAR, uni=True)
            self._font_name = "UniFont"
        if FONT_BOLD:
            self.add_font("UniFont", "B", FONT_BOLD, uni=True)
            self._available_styles.add("B")
        if FONT_ITALIC:
            self.add_font("UniFont", "I", FONT_ITALIC, uni=True)
            self._available_styles.add("I")

    def _set(self, style: str = "", size: int = 10) -> None:
        """Kisa font ayar yardimcisi. Mevcut olmayan stiller icin fallback."""
        safe_style = style if style in self._available_styles else ""
        self.set_font(self._font_name, safe_style, size)

    def header(self) -> None:
        """Sayfa ust bilgisi — profesyonel beyaz tasarim."""
        # Ust cizgi — indigo accent
        self.set_fill_color(79, 70, 229)
        self.rect(0, 0, 210, 3, "F")

        # Logo & baslik
        self.set_y(8)
        self._set("B", 16)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, "Retinal AMD Klinik Karar Destek", ln=False)

        # Sag ust — tarih
        self._set("", 8)
        self.set_text_color(100, 116, 139)
        date_str = datetime.now(TZ_TR).strftime("%d.%m.%Y - %H:%M")
        self.cell(0, 8, date_str, align="R")
        self.ln(6)

        # Alt baslik
        self._set("", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Klinik Karar Destek Raporu", ln=True)

        # Ayirici cizgi
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
            "Bu rapor yapay zeka destekli bir analiz sonucudur ve "
            "kesin tani niteligi tasimaz. Klinik karar surecinde uzman hekim "
            "degerlendirmesi esastir.",
            align="C",
        )
        self.ln(3)
        self._set("", 7)
        self.cell(0, 5, f"Retinal AMD v1.0  |  Sayfa {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str) -> None:
        """Bolum basligi — sol kenar indigo cizgili."""
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
    Analiz sonuclarini profesyonel PDF formatinda uretir.

    Returns:
        PDF dosyasinin bytes icerigi
    """
    pdf = RetinalPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ====================================================
    # 1. ANALIZ OZETI
    # ====================================================
    pdf.section_title("Analiz Ozeti")

    # Bilgi tablosu — 2 sutunlu duzen
    info_data = [
        ("Kullanilan Model", model_name),
        ("Tahmin Edilen Tani", predicted_class),
        ("Guven Orani", f"%{confidence * 100:.1f}"),
        ("Sinif Sayisi", str(len(class_names))),
        ("Analiz Tarihi", datetime.now(TZ_TR).strftime("%d.%m.%Y - %H:%M:%S")),
    ]

    for label, value in info_data:
        pdf._set("", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(50, 6, label, ln=False)

        pdf._set("B", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, value, ln=True)

    pdf.ln(4)

    # ====================================================
    # 2. GORUNTU ANALIZI
    # ====================================================
    pdf.section_title("Goruntu Analizi")

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
        pdf.cell(img_w + 5, 5, "Orijinal Goruntu", align="C", ln=False)
        pdf.cell(img_w + 5, 5, "Grad-CAM Isi Haritasi", align="C", ln=True)

        # Goruntuler
        img_y = pdf.get_y() + 1
        pdf.image(tmp_orig, x=12, y=img_y, w=img_w)
        pdf.image(tmp_gradcam, x=12 + img_w + 6, y=img_y, w=img_w)

        # Goruntulerin altina gecis
        pdf.set_y(img_y + img_w + 4)

    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    pdf.ln(2)

    # ====================================================
    # 3. OLASILIK DAGILIMI
    # ====================================================
    pdf.section_title("Sinif Olasilik Dagilimi")

    # Tablo basligi
    col_widths = [40, 30, 30, 50]  # toplam = 150mm
    headers = ["Sinif", "Olasilik", "Durum", "Gorsel"]

    pdf._set("B", 9)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_text_color(51, 65, 85)
    pdf.set_draw_color(226, 232, 240)

    x_start = (210 - sum(col_widths)) / 2
    pdf.set_x(x_start)
    for header, w in zip(headers, col_widths):
        pdf.cell(w, 7, header, border=1, fill=True, align="C")
    pdf.ln()

    # Tablo satirlari
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

        # Sinif adi
        pdf.cell(col_widths[0], 7, name, border=1, fill=True, align="C")

        # Olasilik
        pdf.cell(col_widths[1], 7, f"%{prob_pct:.2f}", border=1, fill=True, align="C")

        # Durum
        status = "[Tahmin]" if is_predicted else ""
        pdf.cell(col_widths[2], 7, status, border=1, fill=True, align="C")

        # Gorsel bar — ASCII-safe karakter
        bar_count = max(1, int(prob * 20))
        bar_text = "|" * bar_count
        pdf.cell(col_widths[3], 7, bar_text, border=1, fill=True, align="L")
        pdf.ln()

    pdf.ln(4)

    # ====================================================
    # 4. KLINIK RAPOR
    # ====================================================
    pdf.section_title("Klinik Degerlendirme Raporu")

    # Markdown sembollerini temizle
    clean_report = report_text
    for marker in ["**", "---", "*"]:
        clean_report = clean_report.replace(marker, "")

    # Emoji temizligi
    import re
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF"
        "\u23E9-\u23F3\u23F8-\u23FA\u200d\uFE0F\u20E3\u2640\u2642\u2695"
        "\u26A1\u2B50\u2B55\u2934\u2935\u25AA\u25AB\u25FB-\u25FE]+",
        flags=re.UNICODE,
    )
    clean_report = emoji_pattern.sub("", clean_report)

    # Em dash -> tire
    clean_report = clean_report.replace("\u2014", "-").replace("\u2013", "-")

    # Rapor metnini yaz
    pdf._set("", 9)
    pdf.set_text_color(51, 65, 85)

    lines = [line.strip() for line in clean_report.strip().split("\n") if line.strip()]
    for line in lines:
        pdf.multi_cell(180, 5, line, align="L")
        pdf.ln(1)

    pdf.ln(4)

    # ====================================================
    # 5. SORUMLULUK REDDI
    # ====================================================
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(253, 224, 71)

    pdf._set("B", 8)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 5, "Yasal Uyari", ln=True)

    pdf._set("I", 7)
    pdf.set_text_color(146, 64, 14)
    pdf.multi_cell(
        0, 4,
        "Bu rapor Retinal AMD Klinik Karar Destek Sistemi tarafindan otomatik "
        "olarak uretilmistir. Yapay zeka destekli analiz sonuclari kesin tani "
        "niteligi tasimamaktadir. Tum bulgular uzman hekim tarafindan klinik "
        "korelasyon ile degerlendirilmelidir.",
        align="L",
    )

    return bytes(pdf.output())
