"""Yardımcı araçlar: yarışma metriği, EDA tabloları ve grafikleri, submission dosyası.

Notebook'lar bu modülü şöyle kullanır:

    from src.utils import compute_metrics, plot_distributions, save_submission
"""
import json
import math
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Grafik stili (tek yerde tanımlı; tüm grafikler aynı görünür)
# --------------------------------------------------------------------------
COLOR_SOURCE = "#2a78d6"   # Akadca (kaynak) serisi
COLOR_TARGET = "#eb6834"   # İngilizce (hedef) serisi
COLOR_NEUTRAL = "#c3c2b7"  # vurgulanmayan çubuklar
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def set_plot_style():
    """Tüm grafikler için ortak, sade matplotlib stilini kurar."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # š, ṣ, ṭ gibi harfleri içerir
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",  # DejaVu Sans'ta "semibold" yok; uyarı verir
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
    })


def _fmt(value):
    """Grafik etiketleri için tam sayı biçimi (ondalık nokta ile karışmasın diye ayraçsız)."""
    return str(int(round(value)))


# --------------------------------------------------------------------------
# Yarışma metriği
# --------------------------------------------------------------------------
def compute_metrics(predictions, references):
    """Yarışmanın resmi metriğini hesaplar: Score = sqrt(BLEU * chrF++).

    BLEU ve chrF++ tüm korpus üzerinden (mikro ortalama) sacrebleu ile hesaplanır.
    """
    import sacrebleu  # yalnızca değerlendirmede gerekir; Kaggle'da tahmin için gerekmez

    preds = [str(p) for p in predictions]
    refs = [str(r) for r in references]
    if len(preds) != len(refs):
        raise ValueError(f"Tahmin ({len(preds)}) ve referans ({len(refs)}) sayısı farklı.")

    # sacrebleu referansları "akış listesi" olarak bekler: tek referans akışı -> [refs]
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    score = math.sqrt(max(0.0, bleu) * max(0.0, chrf))
    return {"BLEU": bleu, "chrF++": chrf, "Score": score}


# Eski ad; geriye dönük uyumluluk için.
compute_competition_metrics = compute_metrics


# --------------------------------------------------------------------------
# EDA tabloları
# --------------------------------------------------------------------------
def data_overview(df):
    """Sütun başına veri tipi, dolu/eksik değer ve benzersiz değer sayısı."""
    return pd.DataFrame({
        "veri tipi": df.dtypes.astype(str),
        "dolu": df.notna().sum(),
        "eksik": df.isna().sum(),
        "eksik %": (df.isna().mean() * 100).round(2),
        "benzersiz": df.nunique(),
    })


def duplicate_report(df, src_col="transliteration", tgt_col="translation"):
    """Tekrar eden satır, kaynak ve hedef sayıları ile çelişkili kaynak sayısı."""
    targets_per_source = df.groupby(src_col)[tgt_col].nunique()
    return pd.Series({
        "Tam satır tekrarı": int(df.duplicated().sum()),
        "Tekrar eden transliterasyon": int(df[src_col].duplicated().sum()),
        "Tekrar eden çeviri": int(df[tgt_col].duplicated().sum()),
        "Çelişkili kaynak (aynı transliterasyon, farklı çeviri)": int((targets_per_source > 1).sum()),
    }, name="adet").to_frame()


def add_length_features(df, src_col="transliteration", tgt_col="translation"):
    """Karakter ve kelime uzunluğu sütunlarını ekler."""
    out = df.copy()
    out["src_chars"] = out[src_col].astype(str).str.len()
    out["src_words"] = out[src_col].astype(str).str.split().str.len()
    if tgt_col in out.columns:
        out["tgt_chars"] = out[tgt_col].astype(str).str.len()
        out["tgt_words"] = out[tgt_col].astype(str).str.split().str.len()
        out["char_ratio"] = out["tgt_chars"] / out["src_chars"]
        out["word_ratio"] = out["tgt_words"] / out["src_words"]
    return out


def outlier_summary(df, columns):
    """IQR kuralına göre (Q3 + 1.5 * IQR üstü) aykırı değer özeti."""
    rows = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        n_out = int((df[col] > upper).sum())
        rows[col] = {
            "Q1": q1, "medyan": df[col].median(), "Q3": q3, "üst sınır": upper,
            "aykırı adet": n_out, "aykırı %": round(100 * n_out / len(df), 2),
            "maksimum": int(df[col].max()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def length_ratio_summary(df, edges=(850, 900)):
    """Kaynak uzunluk grubuna göre belge sayısı ve çeviri / kaynak karakter oranı özeti."""
    low, high = edges
    groups = {
        f"< {low}": df["src_chars"] < low,
        f"{low}-{high - 1}": (df["src_chars"] >= low) & (df["src_chars"] < high),
        f"{high}+": df["src_chars"] >= high,
        "tümü": df["src_chars"].notna(),
    }
    rows = {}
    for name, mask in groups.items():
        ratio = df.loc[mask, "char_ratio"]
        rows[name] = {
            "belge sayısı": int(mask.sum()),
            "belge %": round(100 * mask.mean(), 1),
            "oran medyanı": round(ratio.median(), 2),
            "oran < 0.5 (belge)": int((ratio < 0.5).sum()),
            "oran > 2 (belge)": int((ratio > 2).sum()),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "kaynak uzunluğu (karakter)"
    return table


def extreme_examples(df, by, n=3, largest=True, width=100,
                     src_col="transliteration", tgt_col="translation"):
    """Bir uzunluk sütununa göre en kısa / en uzun n örneği kısaltılmış metinle gösterir."""
    picked = df.nlargest(n, by) if largest else df.nsmallest(n, by)
    out = picked[["src_chars", "tgt_chars"]].copy()
    out[src_col] = picked[src_col].str[:width]
    out[tgt_col] = picked[tgt_col].str[:width]
    return out


def length_correlations(df, x, y):
    """İki uzunluk sütunu arasındaki Pearson ve Spearman korelasyonu."""
    return pd.Series({
        "Pearson": df[x].corr(df[y], method="pearson"),
        "Spearman": df[x].corr(df[y], method="spearman"),
    }, name=f"{x} ~ {y}").round(3).to_frame()


def token_counts(texts):
    """Boşlukla ayrılmış tokenların sıklığı (Counter)."""
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    return counter


def count_determinatives(texts):
    """Süslü parantez içindeki determinatiflerin sıklığı, örn. {d}, {ki}."""
    counter = Counter()
    for text in texts:
        counter.update(re.findall(r"\{[^{}]*\}", str(text)))
    return counter


def determinative_table(texts):
    """Her determinatifin toplam kullanımı ve geçtiği belge sayısı / yüzdesi."""
    texts = [str(text) for text in texts]
    rows = []
    for det, count in count_determinatives(texts).most_common():
        docs = sum(det in text for text in texts)
        rows.append({"determinatif": det, "adet": count, "geçtiği belge": docs,
                     "belge %": round(100 * docs / len(texts), 1)})
    return pd.DataFrame(rows)


def _token_type(token):
    if token in ("<gap>", "<big_gap>"):
        return "Kırık işareti (<gap>)"
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return "Sayı"
    letters = re.sub(r"\{[^{}]*\}", "", token)
    if letters.isupper():
        return "Sümerogram (tamamı BÜYÜK harf)"
    if letters.islower():
        return "Hece yazımı (küçük harf)"
    if any(ch.isupper() for ch in letters):
        return "Karışık (logogram + hece)"
    return "Diğer"


def token_type_table(texts):
    """Tokenları türüne göre sayar: hece yazımı, Sümerogram, sayı, kırık işareti."""
    counter = Counter()
    for text in texts:
        counter.update(_token_type(tok) for tok in str(text).split())
    table = pd.DataFrame(counter.most_common(), columns=["token türü", "adet"])
    table["pay %"] = (100 * table["adet"] / table["adet"].sum()).round(1)
    return table


NOTATIONS = {
    "Ḫ / ḫ harfi": r"[Ḫḫ]",
    "H / h harfi": r"[Hh]",
    "Alt simge rakam (₀-₉)": r"[₀-₉]",
    "Determinatif { }": r"\{[^{}]*\}",
    "Normal parantez ( )": r"\([^()]*\)",
    "Köşeli parantez [ ]": r"[\[\]]",
    "<gap>": r"<gap>",
    "<big_gap>": r"<big_gap>",
    "Üç nokta (… veya ...)": r"…|\.\.\.",
    "„ işareti": r"„",
    "! ? / : işaretleri": r"[!?/:]",
}


def notation_table(named_texts):
    """Özel işaretlerin her veri kümesinde kaç kez ve satırların yüzde kaçında geçtiği.

    named_texts: {"train": seri, "test": seri} gibi bir sözlük.
    """
    columns = {}
    for name, texts in named_texts.items():
        series = pd.Series(list(texts), dtype="object").astype(str)
        columns[f"{name}: adet"] = {
            label: int(series.str.count(pattern).sum()) for label, pattern in NOTATIONS.items()
        }
        columns[f"{name}: satır %"] = {
            label: round(100 * series.str.contains(pattern).mean(), 1)
            for label, pattern in NOTATIONS.items()
        }
    return pd.DataFrame(columns)


def char_inventory_diff(texts_a, texts_b):
    """İki metin kümesinin karakter envanteri farkı: (yalnız a'da, yalnız b'de)."""
    chars_a = set("".join(str(t) for t in texts_a))
    chars_b = set("".join(str(t) for t in texts_b))
    return sorted(chars_a - chars_b), sorted(chars_b - chars_a)


def before_after(texts, clean_fn):
    """Temizleme fonksiyonunun etkisini önce / sonra tablosu olarak gösterir."""
    return pd.DataFrame({"Önce": list(texts), "Sonra": [clean_fn(t) for t in texts]})


def vocab_stats(named_texts):
    """Her metin kümesi için toplam token, farklı token ve bir kez geçen token sayısı."""
    rows = {}
    for name, texts in named_texts.items():
        counter = token_counts(texts)
        total, unique = sum(counter.values()), len(counter)
        hapax = sum(1 for count in counter.values() if count == 1)
        rows[name] = {
            "toplam token": total, "farklı token": unique,
            "bir kez geçen token": hapax,
            "bir kez geçenlerin payı %": round(100 * hapax / unique, 1),
            "token / farklı token": round(total / unique, 1),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def determinative_words(texts, determinative):
    """Belirli bir determinatifi ({d}, {ki} ...) içeren kelimelerin sıklığı."""
    counter = Counter()
    for text in texts:
        counter.update(tok for tok in str(text).split() if determinative in tok)
    return counter


def gap_summary(df, src_col="transliteration", tgt_col="translation", token="<gap>"):
    """Kırık işaretinin (<gap>) kaynak ve hedefte ne sıklıkta geçtiğini özetler."""
    src_count = df[src_col].astype(str).str.count(token)
    tgt_count = df[tgt_col].astype(str).str.count(token)

    def describe(counts):
        return {
            "<gap> içeren belge %": round(100 * (counts > 0).mean(), 1),
            "toplam <gap>": int(counts.sum()),
            "belge başına ortalama": round(counts.mean(), 2),
            "tek belgede en çok": int(counts.max()),
        }

    summary = pd.DataFrame.from_dict(
        {"transliterasyon": describe(src_count), "çeviri": describe(tgt_count)}, orient="index")
    agreement = pd.crosstab(
        (src_count > 0).map({True: "kaynakta var", False: "kaynakta yok"}),
        (tgt_count > 0).map({True: "çeviride var", False: "çeviride yok"}),
    )
    agreement.index.name, agreement.columns.name = None, None
    return summary, agreement, round(src_count.corr(tgt_count), 3)


def conflicting_sources(df, src_col="transliteration", tgt_col="translation", width=110):
    """Aynı transliterasyona farklı çeviri verilmiş satırları kısaltılmış metinle listeler."""
    repeated = df[df[src_col].duplicated(keep=False)].sort_values(src_col)
    out = pd.DataFrame({
        "transliterasyon": repeated[src_col].str[:60] + "…",
        "çeviri (başı)": repeated[tgt_col].str[:width] + "…",
        "çeviri uzunluğu": repeated[tgt_col].str.len(),
    })
    return out


def sample_overlap(train_df, sample_sub, tgt_col="translation"):
    """Örnek test çevirilerinin train çevirileri içinde geçip geçmediğini kontrol eder.

    Boşluk farkları yok sayılır. Geçiyorsa örnek test, train'deki bir belgenin parçasıdır
    (sızıntı) ve üzerinde skor hesaplamak yanıltıcı olur.
    """
    def squeeze(text):
        return re.sub(r"\s+", "", str(text))

    train_texts = [squeeze(text) for text in train_df[tgt_col]]
    rows = []
    for row_id, text in zip(sample_sub["id"], sample_sub[tgt_col]):
        needle = squeeze(text).strip('"')
        hits = [index for index, hay in enumerate(train_texts) if needle and needle in hay]
        rows.append({"id": row_id, "örnek çeviri (başı)": str(text)[:60] + "…",
                     "train'de geçtiği belge sayısı": len(hits),
                     "train satır indeksi": hits[0] if hits else None})
    return pd.DataFrame(rows)


def preprocessing_effect(texts, clean_fn):
    """Temizlemenin kaynak metin üzerindeki toplu etkisini özetler (önce / sonra)."""
    texts = [str(text) for text in texts]
    cleaned = [clean_fn(text) for text in texts]
    changed = sum(a != b for a, b in zip(texts, cleaned))

    def summarize(items):
        return {
            "farklı token": len(token_counts(items)),
            "farklı karakter": len(set("".join(items))),
            "ortalama uzunluk (karakter)": round(sum(map(len, items)) / len(items), 1),
            "Ḫ/ḫ içeren belge %": round(100 * sum(bool(re.search("[Ḫḫ]", t)) for t in items) / len(items), 1),
            "alt simge içeren belge %": round(100 * sum(bool(re.search("[₀-₉]", t)) for t in items) / len(items), 1),
        }

    table = pd.DataFrame.from_dict({"önce": summarize(texts), "sonra": summarize(cleaned)},
                                   orient="index")
    return table, round(100 * changed / len(texts), 1)


def show_predictions(eval_df, predictions, rows=(0, 1, 2), width=220,
                     src_col="transliteration", tgt_col="translation"):
    """Seçilen satırlar için kaynak, referans ve tahmini kısaltarak alt alta gösterir."""
    table = []
    for position in rows:
        table.append({
            "satır": position,
            "kaynak": str(eval_df[src_col].iloc[position])[:width],
            "referans": str(eval_df[tgt_col].iloc[position])[:width],
            "tahmin": str(predictions[position])[:width],
        })
    return pd.DataFrame(table).set_index("satır")


# --------------------------------------------------------------------------
# EDA grafikleri
# --------------------------------------------------------------------------
def plot_distributions(df, columns, titles, xlabel, colors=None, bins=30):
    """Seçilen sütunların histogramlarını yan yana çizer; medyanı işaretler."""
    colors = colors or [COLOR_SOURCE, COLOR_TARGET]
    fig, axes = plt.subplots(1, len(columns), figsize=(5.6 * len(columns), 4.2))
    axes = np.atleast_1d(axes)
    for ax, col, title, color in zip(axes, columns, titles, colors):
        ax.hist(df[col], bins=bins, color=color, edgecolor=SURFACE, linewidth=1.2)
        ax.grid(axis="x", visible=False)
        median = df[col].median()
        ax.axvline(median, color=INK, linewidth=1.2)
        ax.text(median, ax.get_ylim()[1] * 0.97, f"  medyan {_fmt(median)}",
                ha="left", va="top", fontsize=9, color=INK_SECONDARY)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Belge sayısı")
    fig.tight_layout()
    plt.show()


def plot_boxplots(df, columns, labels, title, xlabel, colors=None):
    """Aynı birimdeki sütunları tek eksende yatay kutu grafiği olarak karşılaştırır."""
    colors = colors or [COLOR_SOURCE, COLOR_TARGET]
    fig, ax = plt.subplots(figsize=(9, 1.3 + 0.9 * len(columns)))
    boxes = ax.boxplot(
        [df[col] for col in columns], vert=False, patch_artist=True, widths=0.5,
        medianprops={"color": INK, "linewidth": 1.4},
        whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": INK_MUTED,
                    "markeredgecolor": SURFACE, "alpha": 0.6},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(columns) + 1))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    plt.show()


def plot_length_relationship(df, x, y, xlabel, ylabel, title):
    """Kaynak ve hedef uzunluğu arasındaki ilişkiyi saçılım grafiğiyle gösterir."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(df[x], df[y], s=22, color=COLOR_SOURCE, alpha=0.45,
               edgecolor=SURFACE, linewidth=0.5)
    slope, intercept = np.polyfit(df[x], df[y], 1)
    xs = np.array([df[x].min(), df[x].max()])
    ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.6, label="doğrusal eğilim")
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()


def plot_counter(counter, n=15, title="", xlabel="Adet", color=COLOR_SOURCE, ax=None):
    """Bir Counter'daki en sık n öğeyi sıralı yatay çubuklarla çizer."""
    items = counter.most_common(n)
    labels = [str(key) for key, _ in items]
    values = [value for _, value in items]
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(7, 0.6 + 0.32 * len(items)))
    ax.barh(labels, values, height=0.62, color=color)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for label, value in zip(labels, values):
        ax.text(value, label, f" {_fmt(value)}", va="center", ha="left",
                fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if own_figure:
        plt.tight_layout()
        plt.show()


def plot_top_tokens(source_texts, target_texts, n=15):
    """Kaynak ve hedef metinlerdeki en sık n tokenı yan yana çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 + 0.32 * n))
    plot_counter(token_counts(source_texts), n, "Transliterasyon: en sık tokenlar",
                 color=COLOR_SOURCE, ax=axes[0])
    plot_counter(token_counts(target_texts), n, "İngilizce çeviri: en sık tokenlar",
                 color=COLOR_TARGET, ax=axes[1])
    fig.tight_layout()
    plt.show()


def plot_determinative_words(texts, determinatives=("{d}", "{ki}"),
                             titles=("{d}: tanrı adı içeren kelimeler", "{ki}: yer adı içeren kelimeler"),
                             n=10):
    """Her determinatif için onu en sık taşıyan n kelimeyi yan yana çizer."""
    fig, axes = plt.subplots(1, len(determinatives), figsize=(6 * len(determinatives), 0.6 + 0.32 * n))
    for ax, det, title in zip(np.atleast_1d(axes), determinatives, titles):
        plot_counter(determinative_words(texts, det), n, title, color=COLOR_SOURCE, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_baseline_scores(results, label_col="Baseline", metrics=("BLEU", "chrF++", "Score")):
    """Baseline'ları her metrik için ayrı panelde karşılaştırır; en iyisi vurgulanır."""
    ordered = results.sort_values(metrics[-1], ascending=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 0.9 + 0.45 * len(ordered)),
                             sharey=True)
    # Üç metrik de 0-100 ölçeğinde; paneller karşılaştırılabilsin diye ortak eksen sınırı.
    limit = max(float(ordered[list(metrics)].to_numpy().max()) * 1.22, 1.0)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        values = ordered[metric].to_numpy()
        colors = [COLOR_SOURCE if v == values.max() else COLOR_NEUTRAL for v in values]
        ax.barh(ordered[label_col], values, height=0.6, color=colors)
        ax.grid(axis="y", visible=False)
        for label, value in zip(ordered[label_col], values):
            ax.text(value, label, f" {value:.2f}", va="center", ha="left",
                    fontsize=9, color=INK_SECONDARY)
        ax.set_xlim(0, limit)
        title = "Score = √(BLEU × chrF++)" if metric == "Score" else metric
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Submission dosyası
# --------------------------------------------------------------------------
EMPTY_PREDICTION = "<gap>"  # boş tahmin Kaggle'da hata verir; yerine konur


def save_submission(ids, predictions, path="submission.csv"):
    """id + translation sütunlu submission dosyasını yazar ve tabloyu döndürür."""
    cleaned = []
    for pred in predictions:
        pred = re.sub(r"\s+", " ", str(pred)).strip() if isinstance(pred, str) else ""
        cleaned.append(pred or EMPTY_PREDICTION)
    submission = pd.DataFrame({"id": list(ids), "translation": cleaned})
    submission.to_csv(path, index=False)
    return submission


def validate_submission(submission, expected_ids):
    """Submission biçimini doğrular; sorun varsa ValueError verir."""
    problems = []
    expected = list(expected_ids)
    if list(submission.columns) != ["id", "translation"]:
        problems.append(f"sütunlar ['id', 'translation'] olmalı, bulunan: {list(submission.columns)}")
    else:
        if len(submission) != len(expected):
            problems.append(f"satır sayısı {len(expected)} olmalı, bulunan: {len(submission)}")
        if submission["id"].duplicated().any():
            problems.append("tekrar eden id var")
        if set(submission["id"]) != set(expected):
            problems.append("id'ler test kümesiyle eşleşmiyor")
        empty = submission["translation"].isna() | (submission["translation"].astype(str).str.strip() == "")
        if empty.any():
            problems.append(f"{int(empty.sum())} boş tahmin var")
    if problems:
        raise ValueError("Geçersiz submission: " + "; ".join(problems))
    return True


# --------------------------------------------------------------------------
# Kaggle gönderim notebook'u (Code Competition: internet kapalı çalışır)
# --------------------------------------------------------------------------
_KAGGLE_INTRO = """# Deep Past Challenge — Baseline gönderimi (GROUP36 / CENG313)

Bu notebook `src/utils.py` içindeki `build_kaggle_notebook()` ile **otomatik üretilir**; elle düzenlenmez.

Yarışma bir *Code Competition* olduğu için gönderim, Kaggle üzerinde **interneti kapalı** çalışan bir notebook ile yapılır.
İnternet olmadığından GitHub deposu klonlanamaz; bu yüzden aşağıdaki hücreler `src/` modüllerini dosya olarak yeniden yazar,
ardından modeli tüm `train.csv` ile kurup gizli test kümesi için `submission.csv` üretir.

**Kullanım:** Yarışma sayfası → *Late Submission* → bu notebook'u içe aktar → sağ panelde *Internet: Off* → *Save Version (Save & Run All)* → *Submit*.
"""

_KAGGLE_SETUP = """import os

os.makedirs("src", exist_ok=True)
open("src/__init__.py", "w").close()
"""

_KAGGLE_INFERENCE = """import sys

sys.path.insert(0, os.getcwd())

from src.data import load_data, preprocess
from src.models import predict, train_baseline
from src.utils import save_submission, validate_submission

train_df, test_df, _ = load_data(download=False)
model = train_baseline(preprocess(train_df))  # gönderimde tüm train verisi kullanılır
predictions = predict(model, test_df)

submission = save_submission(test_df["id"], predictions, "submission.csv")
validate_submission(submission, test_df["id"])
print(f"Train: {len(train_df)} belge | Test: {len(test_df)} satır | submission.csv yazıldı")
submission.head()
"""


def build_kaggle_notebook(out_path="submissions/kaggle_submission.ipynb", src_dir="src"):
    """src/ modüllerini içine gömen, tek başına çalışan Kaggle notebook'unu üretir."""
    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": text.splitlines(keepends=True)}

    cells = [markdown(_KAGGLE_INTRO), code(_KAGGLE_SETUP)]
    for name in ("data.py", "models.py", "utils.py"):
        with open(os.path.join(src_dir, name), encoding="utf-8") as handle:
            cells.append(code(f"%%writefile src/{name}\n{handle.read()}"))
    cells.append(code(_KAGGLE_INFERENCE))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
    return out_path
