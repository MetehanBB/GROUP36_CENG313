"""Assignment 1 baseline modelleri.

Her baseline (train_df, val_df) alir ve val_df icin tahmin listesi dondurur.
Skorlar evaluate() ile (BLEU, chrF) olarak hesaplanir.
"""
import pandas as pd
import sacrebleu
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SRC_COL = "transliteration"
TGT_COL = "translation"


def split_data(df, test_size=0.2, seed=42):
    """Eksik satirlari atip train/validation olarak boler."""
    df = df.dropna(subset=[SRC_COL, TGT_COL])
    return train_test_split(df, test_size=test_size, random_state=seed)


def copy_input_baseline(train_df, val_df):
    """Baseline 1: Akadca girdiyi aynen 'ceviri' olarak ver."""
    return val_df[SRC_COL].astype(str).tolist()


def constant_baseline(train_df, val_df):
    """Baseline 2: Train'deki en sik ceviriyi her cumle icin ver."""
    most_common = train_df[TGT_COL].astype(str).value_counts().index[0]
    return [most_common] * len(val_df)


def nearest_neighbor_baseline(train_df, val_df, max_features=50000):
    """Baseline 3: TF-IDF ile train'deki en benzer girdinin cevirisini ver."""
    vec = TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+",
        ngram_range=(1, 2), max_features=max_features,
    )
    x_train = vec.fit_transform(train_df[SRC_COL].astype(str))
    x_val = vec.transform(val_df[SRC_COL].astype(str))
    best = linear_kernel(x_val, x_train).argmax(axis=1)
    return train_df[TGT_COL].astype(str).iloc[best].tolist()


def evaluate(preds, val_df):
    """(BLEU, chrF) dondurur. Referanslar val_df'in ceviri sutunudur."""
    refs = [val_df[TGT_COL].astype(str).tolist()]
    bleu = sacrebleu.corpus_bleu(preds, refs).score
    chrf = sacrebleu.corpus_chrf(preds, refs).score
    return bleu, chrf


def run_baselines(train_df, val_df):
    """Uc baseline'i calistirip sonuc tablosunu (DataFrame) dondurur."""
    methods = {
        "Copy-input": copy_input_baseline,
        "Sabit cümle": constant_baseline,
        "TF-IDF en yakın komşu": nearest_neighbor_baseline,
    }
    rows = []
    for name, fn in methods.items():
        bleu, chrf = evaluate(fn(train_df, val_df), val_df)
        rows.append({"Baseline": name, "BLEU": round(bleu, 2), "chrF": round(chrf, 2)})
    return pd.DataFrame(rows)
