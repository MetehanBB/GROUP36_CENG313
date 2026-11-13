# CENG313 - Group 36 Project

## Project Overview
This repository contains the coursework and machine learning pipeline for the **Deep Past Challenge - Translate Akkadian to English** competition as part of the CENG313 course.

> **Kaggle skoru (Late Submission, 13.11.2026): Private 8.1842 · Public 8.5489**
> Yarışma 23 Mart 2026'da kapandı; gönderim *Late Submission* ile puanlandı (sıralamaya girmez). Ayar seçimi ve baseline karşılaştırması ayrıca `train.csv` verisinin `train_test_split(test_size=0.2, random_state=42)` ile ayrılan %20'lik held-out kısmında yapıldı.

## Team Members
* **Metehan Bölükbaşı** - Student ID: `23118080031` - GitHub: `@MetehanBB`
* **Mustafa Karadeniz** - Student ID: `23118080030` - GitHub: `@Mustafa-Karadeniz`

## Chosen Competition
* **Name:** Deep Past Challenge - Translate Akkadian to English
* **Link:** https://www.kaggle.com/c/deep-past-initiative-machine-translation
* **Task:** Machine Translation (Old Assyrian / Transliterated Akkadian to English)
* **Evaluation Metric:** Geometric Mean of BLEU and chrF++, `Score = sqrt(BLEU × chrF++)` (micro-averaged, `sacrebleu`)

## Repository Structure
```text
GROUP36_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                       # data/ ve __pycache__/ hariç tutulur
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA & Baseline
├── assignment2_solution.ipynb       # Assignment 2: Improved Solution
├── src/
│   ├── __init__.py
│   ├── data.py                      # load_data, clean_akkadian_text, preprocess, split_data
│   ├── models.py                    # baseline modelleri: train_baseline, predict, run_baselines
│   └── utils.py                     # compute_metrics, EDA grafikleri, save_submission
├── data/                            # Git'e eklenmez
└── submissions/
    ├── kaggle_submission.ipynb      # Kaggle'a gönderilen, otomatik üretilen notebook
    └── late_submission_score.png    # Late Submission skor ekran görüntüsü
```

## Dataset

Veri Git'e eklenmez (Kaggle kuralları ve dosya boyutu). İndirmek için önce yarışma sayfasında kuralları kabul edin, sonra iki yoldan birini kullanın:

**1) Kaggle API**

```bash
pip install kaggle
```

```bash
kaggle competitions download -c deep-past-initiative-machine-translation -p data
```

İnen zip dosyasını `data/` klasörüne açın. Kaggle API anahtarı için: Kaggle → Settings → API → *Create New Token*.

**2) Elle**

Yarışmanın *Data* sekmesinden dosyaları indirip `data/` klasörüne koyun.

Assignment 1 için gereken dosyalar: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.

- `train.csv`: 1561 belge; sütunlar `oare_id`, `transliteration`, `translation` (eksik değer yok)
- `test.csv`: yalnızca 4 satırlık bir örnek (gerçek test seti gizli); cümle düzeyinde; sütunlar `id`, `text_id`, `line_start`, `line_end`, `transliteration`
- `sample_submission.csv`: örnek gönderim formatı
- Ek kaynaklar (Assignment 2 için): `published_texts.csv`, `Sentences_Oare_FirstWord_LinNum.csv`, `OA_Lexicon_eBL.csv`, `eBL_Dictionary.csv`, `publications.csv`

Train belge düzeyinde, test cümle düzeyindedir.

## How to Run

**Google Colab**

1. `assignment1_eda_baseline.ipynb` dosyasını Colab'de açın: [Open in Colab](https://colab.research.google.com/github/MetehanBB/GROUP36_CENG313/blob/main/assignment1_eda_baseline.ipynb)
2. Veri için şunlardan birini yapın:
   - Colab'de sol paneldeki **Secrets** (anahtar simgesi) bölümüne `KAGGLE_USERNAME` ve `KAGGLE_KEY` ekleyip notebook erişimini açın; notebook veriyi kendisi indirir, veya
   - İlk hücre çalıştıktan sonra üç CSV dosyasını `GROUP36_CENG313/data/` klasörüne yükleyin.
3. **Çalışma zamanı → Yeniden başlat ve tümünü çalıştır** deyin.

İlk hücre depoyu klonlar ve `requirements.txt` içindeki paketleri kurar. Klonlanan dal `main`'dir: değişiklikler henüz `main`'e birleştirilmediyse ya önce birleştirin ya da ilk hücredeki `BRANCH` değerini çalıştığınız dalın adı yapın. Aksi hâlde eski `src/` gelir ve import hatası alınır. Assignment 1 için GPU gerekmez; notebook'un tamamı CPU'da bir dakikadan kısa sürer.

**Yerel**

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook assignment1_eda_baseline.ipynb
```

GitHub notebook'u görüntüleyemezse: [nbviewer](https://nbviewer.org/github/MetehanBB/GROUP36_CENG313/blob/main/assignment1_eda_baseline.ipynb)

## Approach

**Assignment 1 (EDA ve baseline)** — notebook 7 bölümden oluşur:

1. **Problem definition:** görev, hedef değişken, metrik.
2. **Data overview:** boyut, veri tipleri, `describe()`, eksik değerler, tekrarlar; train (belge) ve test (cümle) biçim farkı.
3. **EDA:** karakter ve kelime uzunluğu dağılımları, aykırı değerler, kaynak-hedef uzunluk ilişkisi, en sık tokenlar, Sümerogramlar ve determinatifler (`{d}`, `{ki}`), `<gap>` analizi, train-test yazım farkları, çelişkili tekrarlar.
4. **Preprocessing:** `Ḫ/ḫ → H/h`, alt simge rakamlar, bütün kırık biçimlerinin (`[x]`, `…`, `<big_gap>`) train'deki gibi `<gap>` olarak standartlaştırılması, determinatif yazımı, akademik işaretlerin silinmesi; her adımın nedeni.
5. **Baseline model:** train / validation / held-out bölmesi (%70 / %10 / %20, `random_state=42`) ve beş baseline.
6. **Kaggle submission:** Late Submission skoru, held-out skoru, leaderboard karşılaştırması, `submission.csv` ve Kaggle gönderim notebook'u.
7. **Conclusion:** bulgular ve Assignment 2 planı.

Öne çıkan bulgular:

- Train'deki uzun tabletlerin transliterasyonu yaklaşık 900 karakterde kesilmiş görünüyor, çevirileri ise tam (850+ karakterli belgelerde çeviri / kaynak oranı belirgin biçimde yükseliyor).
- Public `test.csv` içindeki 4 örnek cümle train'deki bir belgenin parçası; bu yüzden örnek test üzerinde skor raporlanmıyor.
- Train'de `Ḫ/ḫ` var, `H/h` yok; testte tersi. Ön işleme bu farkı kapatıyor.

**Assignment 2:** Önerilen çözüm (ByT5 ince ayarı, cümle düzeyi hizalama, ek veri) Aralık'ta eklenecek.

## Results (Assignment 1)

Resmi metrik: `Score = sqrt(BLEU × chrF++)`. Ayarlar validation üzerinde seçildi, held-out %20'de bir kez ölçüldü, sonra seçilen model tüm train ile kurulup Kaggle'a gönderildi.

**Kaggle gizli test (Late Submission)** — seçilen baseline, tüm train ile kuruldu:

| Baseline | Public | Private |
|---|---|---|
| TF-IDF karakter + uzunluk sınırı | 8.5489 | 8.1842 |

**Held-out test (%20, 313 belge)** — model train + validation (%80) ile kuruldu:

| Baseline | BLEU | chrF++ | Score |
|---|---|---|---|
| Copy-input | 1.29 | 8.81 | 3.37 |
| Sabit cümle | 2.19 | 19.14 | 6.47 |
| TF-IDF kelime en yakın komşu (ham metin) | 15.49 | 34.15 | 23.00 |
| TF-IDF karakter en yakın komşu (temiz metin) | 15.81 | 37.03 | 24.19 |
| **TF-IDF karakter + uzunluk sınırı (seçilen)** | **17.17** | **37.49** | **25.37** |

**Validation (%10, 156 belge)** — model train (%70) ile kuruldu:

| Baseline | BLEU | chrF++ | Score |
|---|---|---|---|
| Copy-input | 0.87 | 8.20 | 2.67 |
| Sabit cümle | 2.34 | 18.90 | 6.66 |
| TF-IDF kelime en yakın komşu (ham metin) | 13.93 | 35.39 | 22.20 |
| TF-IDF karakter en yakın komşu (temiz metin) | 15.42 | 38.65 | 24.41 |
| TF-IDF karakter + uzunluk sınırı (seçilen) | 15.97 | 38.40 | 24.76 |

**Leaderboard ile karşılaştırma:** Yarışmanın birincisi public'te 41.67, private'ta 42.90 aldı; en çok oy alan public notebook 35.9, popüler başlangıç notebook'u (DPC Starter Infer) 26.8 (public). Baseline'ımız birincinin private'ta 34.71, public'te 33.12 puan gerisinde.

**Held-out ile Kaggle arasındaki fark:** Aynı model held-out'ta 25.37, gizli testte 8.18 aldı. Belge düzeyindeki held-out, yarışmanın cümle düzeyindeki ve büyük olasılıkla train'de olmayan tabletlerden oluşan testini temsil etmiyor. Assignment 2'de modeller cümle düzeyinde ve tablet bazında ayrık bir doğrulama kümesinde karşılaştırılacak (ayrıntılar notebook'un 6.1 ve 7. bölümlerinde).

## Kaggle Submission History

Yarışma bir *Code Competition*: `submission.csv` elle yüklenmez; Kaggle üzerinde interneti kapalı çalışan bir notebook gönderilir. Gönderilecek notebook `submissions/kaggle_submission.ipynb` dosyasıdır (`src/utils.py` → `build_kaggle_notebook` tarafından üretilir, `src/` kodunu içinde taşır).

Late Submission adımları:

1. Yarışma sayfasında **Late Submission**'a tıklayın, kuralları kabul edin.
2. **Code → New Notebook → File → Import Notebook** ile `submissions/kaggle_submission.ipynb` dosyasını yükleyin.
3. Sağ panelde yarışma verisinin ekli ve **Internet: Off** olduğunu kontrol edin.
4. **Save Version → Save & Run All (Commit)**; çalışma bitince **Submit**.
5. Skoru aşağıdaki tabloya yazın, ekran görüntüsünü `submissions/` klasörüne ekleyin.

| Tarih | Yöntem | Public score | Private score | Ekran görüntüsü |
|---|---|---|---|---|
| 13.11.2026 | TF-IDF karakter + uzunluk sınırı (Late Submission) | 8.5489 | 8.1842 | [late_submission_score.png](submissions/late_submission_score.png) |

