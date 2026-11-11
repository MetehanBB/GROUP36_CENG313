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
```

## Dataset

Yarışma verisi `deep-past-initiative-machine-translation/` klasöründedir.

- `train.csv`: 1561 belge, sütunlar `oare_id`, `transliteration`, `translation` (eksik değer yok)
- `test.csv`: yalnızca 4 örneklik bir örnek dosyadır (gerçek test seti gizlidir); cümle düzeyindedir, sütunlar `id`, `text_id`, `line_start`, `line_end`, `transliteration`
- `sample_submission.csv`: örnek gönderim formatı
- Ek yardımcı dosyalar (ör. `published_texts.csv`, `OA_Lexicon_eBL.csv`, `eBL_Dictionary.csv`, `publications.csv`)

Train belge düzeyinde, test ise cümle düzeyindedir.

## Approach

**Assignment 1 (EDA ve baseline):**
- EDA: veri boyutları, eksik değerler, karakter uzunluğu dağılımları, en sık tokenlar, tekrar eden satırlar, uç değerler (en kısa/en uzun
