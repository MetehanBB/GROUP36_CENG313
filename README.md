# CENG313 - Group 36 Project

## Project Overview
This repository contains the coursework and machine learning pipeline for the **Deep Past Challenge - Translate Akkadian to English** competition as part of the CENG313 course.

## Team Members
* **Metehan Bölükbaşı** - Student ID: `23118080031` - GitHub: `@MetehanBB`
* **Mustafa Karadeniz** - Student ID: `23118080030` - GitHub: `@Mustafa-Karadeniz`

## Chosen Competition
* **Name:** Deep Past Challenge - Translate Akkadian to English
* **Task:** Machine Translation (Old Assyrian / Transliterated Akkadian to English)
* **Evaluation Metric:** Geometric Mean of BLEU and chrF++

## Repository Structure
```text
GROUP36_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
## Dataset

Yarışma verisi `deep-past-initiative-machine-translation/` klasöründedir.

- `train.csv`: 1561 belge, sütunlar `oare_id`, `transliteration`, `translation` (eksik değer yok)
- `test.csv`: cümle düzeyinde, sütunlar `id`, `text_id`, `line_start`, `line_end`, `transliteration`
- `sample_submission.csv`: örnek gönderim formatı
- Ek yardımcı dosyalar (ör. `published_texts.csv`, `OA_Lexicon_eBL.csv`, `publications.csv`)

Train belge düzeyinde, test ise cümle düzeyindedir.

## Approach

**Assignment 1 (EDA ve baseline):**
- EDA: veri boyutları, eksik değerler, karakter uzunluğu dağılımları, en sık tokenlar, tekrar eden satırlar
- Üç basit baseline (`src/models.py`): Copy-input, sabit cümle, TF-IDF en yakın komşu
- Aynı train/validation bölmesi (%80 / %20, `random_state=42`)

**Assignment 2:** Önerilen çözüm, Aralık'ta eklenecek.

## Results (Assignment 1, validation seti)

| Baseline | BLEU | chrF |
|---|---|---|
| Copy-input | 1.29 | 10.55 |
| Sabit cümle | 2.52 | 22.82 |
| TF-IDF en yakın komşu | 15.49 | 35.51 |

Skorlar `sacrebleu` ile ham BLEU ve chrF olarak hesaplanmıştır. Yarışmanın resmi metriği chrF++ ve BLEU geometrik ortalamasıdır.

## How to Run

1. `assignment1_eda_baseline.ipynb` dosyasını Google Colab'de aç (GitHub'dan: **Open in Colab**).
2. **Çalışma zamanı → Tümünü çalıştır** de.
3. İlk hücre repoyu otomatik klonlar ve `sacrebleu` kurar. Veri repoda hazırdır, ayrıca indirmek gerekmez.

GPU Assignment 1 için gerekli değildir.

## Progress Log

- 2026-11-08: Mustafa veri dosyalarını repoya ekledi (`data files`, `Big Dataset` commit'leri).
- 2026-11-09: Metehan Colab-GitHub bağlantısını kurdu, ilk EDA (veri boyutları, eksik değerler, uzunluk histogramları) ve ilk baseline denemelerini yaptı.
- 2026-11-10: TF-IDF en yakın komşu baseline'ı eklendi; Copy-input, sabit cümle ve TF-IDF aynı train/validation bölmesiyle karşılaştırıldı.
- 2026-11-11: Notebook temizlendi, baseline kodları `src/models.py` içine taşındı, EDA yorumları ve baseline tablosu eklendi.
