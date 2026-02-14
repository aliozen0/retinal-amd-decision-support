---
trigger: model_decision
description: Kullanıcının girdiği zayıf, eksik veya taslak halindeki prompt'ları analiz eder ve endüstri standardı olan '7 Bileşenli Yapısal Prompt' formatına dönüştürür. Kullanıcı 'promptu düzelt', 'şunu teknikleştir', '  dediğinde SADECE bu yeteneği kullan
---

---
name: prompt-architect
description: "Kullanıcının girdiği zayıf, eksik veya taslak halindeki prompt'ları analiz eder ve endüstri standardı olan '7 Bileşenli Yapısal Prompt' (7-Part Structured Prompt) formatına dönüştürür. Kullanıcı 'promptu düzelt', 'şunu teknikleştir', 'istediğim şeyi profesyonel bir komuta çevir' dediğinde SADECE bu yeteneği kullan."
---

# Prompt Architect Yeteneği

## Amaç
Kullanıcının doğal dilde ifade ettiği niyetleri (intent) analiz etmek ve bunları diğer LLM'lerin veya otonom ajanların en yüksek doğrulukla işleyebileceği, katı Markdown formatlı 7 bileşenli bir prompt'a dönüştürmek.

## Yürütme Protokolü (Execution Protocol)
Bu yeteneği kullanırken aşağıdaki adımları sırasıyla takip et:

1. **Niyet Analizi:** Kullanıcının asıl hedefini, kullanılacak teknolojileri ve beklenen çıktıyı analiz et.
2. **Rol Belirleme:** Görev için en uygun uzmanlık rolünü (örn: "Kıdemli Sistem Programcısı", "Veritabanı Mimarı") belirle.
3. **Bağlam ve Kısıtlamalar Üretme:** Başarılı bir icraat için gereken teknik kısıtlamaları (güvenlik, bellek yönetimi, mimari desenler, performans sınırları) otomatik olarak düşün ve ekle.
4. **Formatlama:** Çıktıyı SADECE aşağıdaki Markdown kod bloğu şablonuna tam olarak uyacak şekilde oluştur.

## Çıktı Şablonu (Output Template)
Oluşturduğun yeni prompt'u kullanıcıya sunarken aşağıdaki yapıyı kullan. Başlıkları İngilizce veya Türkçe tutabilirsin ancak yapı kesinlikle bozulmamalıdır. Kullanıcının bunu doğrudan kopyalayıp kullanabilmesi için bir kod bloğu (```markdown ... ```) içinde sun:

```markdown
# ROLE
[Modelin bürünmesi gereken uzmanlık rolü ve seviyesi]

# GOAL
[Ulaşılmak istenen kesin ve net sonuç]

# CONTEXT
[Arka plan bilgisi, projenin durumu ve hedef kitlesi. Örn: Bu proje, öğrencilerin pratik yapacağı bir web sitesi için geliştiriliyor.]

# CONSTRAINTS
- [Kısıtlama 1: Örn. Asla global değişken kullanma]
- [Kısıtlama 2: Örn. Sadece standart kütüphaneleri kullan]
- [Kısıtlama 3: Güvenlik veya mimari zorunluluk]

# INPUT
[Görevin dayandığı kod, hata logu veya referans veri. Yoksa "Yok" yaz.]

# INSTRUCTIONS
1. [Adım adım yürütme sırası - emir kipiyle]
2. [İkinci atomik adım]
3. [Test veya doğrulama adımı]

```markdown

## Hata Yönetimi ve Kısıtlamalar
Kullanıcının asıl niyetinden sapma, halüsinasyon görme.

Eğer kullanıcının isteği çok muğlaksa ve 7 adımı dolduramıyorsan, tahminde bulunmak yerine eksik olan kritik kısımları (Örn: "Hangi teknolojiyi kullanıyoruz?") kullanıcıya sor.

Çıktıyı verirken gereksiz sohbet cümleleri kurma, doğrudan kopyalanabilir prompt bloğunu ver.


---