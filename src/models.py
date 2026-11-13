"""Assignment 1 baseline modelleri.

Hepsi eğitim gerektirmeyen ya da çok hafif yöntemlerdir. Amaç, Assignment 2'de
geliştirilecek modellerin geçmesi gereken alt sınırı belirlemektir.

Notebook'lar bu modülü şöyle kullanır:

    from src.models import train_baseline, predict, run_baselines
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import SRC_COL, TGT_COL, clean_akkadian_text, clean_translation

# Validation skoruna göre seçilen ayarlar (bkz. notebook, "Baseline model" bölümü).
BEST_PARAMS = {"analyzer": "char_wb", "ngram_range": (2, 5), "clean": True, "max_len_ratio": 3.0}


class TfidfRetrievalBaseline:
    """Çeviri belleği: en benzer eğitim transliterasyonunun çevirisini döndürür.

    analyzer="word"    -> boşlukla ayrılmış kelimeler (1-2 gram)
    analyzer="char_wb" -> kelime içi karakter n-gramları; yazım farklarına daha dayanıklı
    max_len_ratio      -> verilirse, getirilen çeviri girdi uzunluğunun bu katını
                          aşmayacak şekilde cümle sınırından kesilir
    """

    def __init__(self, analyzer="char_wb", ngram_range=None, max_features=None,
                 clean=True, max_len_ratio=None, lowercase=True):
        self.analyzer = analyzer
        self.ngram_range = ngram_range or ((1, 2) if analyzer == "word" else (2, 5))
        self.max_features = max_features
        self.clean = clean
        self.max_len_ratio = max_len_ratio
        self.lowercase = lowercase

    def _prepare(self, texts):
        texts = [str(text) for text in texts]
        return [clean_akkadian_text(text) for text in texts] if self.clean else texts

    def fit(self, source_texts, target_texts):
        options = {"analyzer": self.analyzer, "ngram_range": tuple(self.ngram_range),
                   "max_features": self.max_features, "lowercase": self.lowercase}
        if self.analyzer == "word":
            options["token_pattern"] = r"\S+"  # tireli heceleri tek kelime olarak tut
        else:
            options["sublinear_tf"] = True
        self.vectorizer_ = TfidfVectorizer(**options)
        self.train_matrix_ = self.vectorizer_.fit_transform(self._prepare(source_texts))
        self.targets_ = [clean_translation(text) for text in target_texts]
        return self

    def _fit_length(self, target, query_len):
        if not self.max_len_ratio:
            return target
        limit = self.max_len_ratio * max(query_len, 1)
        if len(target) <= limit:
            return target
        kept, total = [], 0
        for sentence in re.split(r"(?<=[.!?;:])\s+", target):
            if kept and total + len(sentence) > limit:
                break
            kept.append(sentence)
            total += len(sentence) + 1
        return " ".join(kept)

    def predict(self, source_texts, batch_size=512):
        queries = self._prepare(source_texts)
        predictions = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            similarity = linear_kernel(self.vectorizer_.transform(batch), self.train_matrix_)
            for query, best in zip(batch, similarity.argmax(axis=1)):
                predictions.append(self._fit_length(self.targets_[best], len(query)))
        return predictions


def tokenization_demo(text):
    """Bir metnin kelime ve karakter n-gram tokenlarına nasıl ayrıldığını gösterir."""
    cleaned = clean_akkadian_text(text)
    word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2)).build_analyzer()
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).build_analyzer()
    word_tokens, char_tokens = word(cleaned), char(cleaned)
    return pd.DataFrame({
        "token sayısı": [len(word_tokens), len(char_tokens)],
        "ilk tokenlar": [" | ".join(word_tokens[:8]), " | ".join(repr(t) for t in char_tokens[:14])],
    }, index=["kelime (1-2 gram)", "karakter (2-5 gram)"])


def train_baseline(train_df, **params):
    """Seçilen baseline'ı (TF-IDF çeviri belleği) kurar ve döndürür."""
    config = {**BEST_PARAMS, **params}
    model = TfidfRetrievalBaseline(**config)
    return model.fit(train_df[SRC_COL], train_df[TGT_COL])


def predict(model, data):
    """Model ile çeviri üretir; data bir DataFrame ya da metin listesi olabilir."""
    texts = data[SRC_COL] if isinstance(data, pd.DataFrame) else data
    return model.predict(list(texts))


# --------------------------------------------------------------------------
# Karşılaştırma baseline'ları: hepsi (train_df, eval_df) alır, tahmin listesi döndürür
# --------------------------------------------------------------------------
def copy_input_baseline(train_df, eval_df):
    """Baseline 1: Akadca girdiyi aynen 'çeviri' olarak ver."""
    return eval_df[SRC_COL].astype(str).tolist()


def constant_baseline(train_df, eval_df):
    """Baseline 2: Train'deki en sık çeviriyi her örnek için ver."""
    most_common = train_df[TGT_COL].astype(str).value_counts().index[0]
    return [most_common] * len(eval_df)


def nearest_neighbor_baseline(train_df, eval_df, max_features=50000):
    """Baseline 3: Ham metinde kelime TF-IDF ile en benzer girdinin çevirisini ver."""
    model = TfidfRetrievalBaseline(analyzer="word", max_features=max_features, clean=False)
    model.fit(train_df[SRC_COL], train_df[TGT_COL])
    return model.predict(eval_df[SRC_COL].tolist())


def char_nearest_neighbor_baseline(train_df, eval_df):
    """Baseline 4: Temizlenmiş metinde karakter n-gram TF-IDF ile en yakın komşu."""
    return predict(train_baseline(train_df, max_len_ratio=None), eval_df)


def selected_baseline(train_df, eval_df):
    """Baseline 5 (seçilen): Baseline 4 + çıktı uzunluğu sınırı (BEST_PARAMS)."""
    return predict(train_baseline(train_df), eval_df)


BASELINES = {
    "Copy-input": copy_input_baseline,
    "Sabit cümle": constant_baseline,
    "TF-IDF kelime (ham metin)": nearest_neighbor_baseline,
    "TF-IDF karakter (temiz metin)": char_nearest_neighbor_baseline,
    "TF-IDF karakter + uzunluk sınırı": selected_baseline,
}


def evaluate(predictions, eval_df):
    """Tahminleri eval_df'in çeviri sütununa göre yarışma metriğiyle değerlendirir."""
    from .utils import compute_metrics

    return compute_metrics(predictions, eval_df[TGT_COL].astype(str).tolist())


def run_baselines(train_df, eval_df, baselines=None):
    """Baseline'ları çalıştırıp BLEU, chrF++ ve Score tablosunu döndürür."""
    rows = []
    for name, baseline in (baselines or BASELINES).items():
        scores = evaluate(baseline(train_df, eval_df), eval_df)
        rows.append({"Baseline": name, **{key: round(value, 2) for key, value in scores.items()}})
    return pd.DataFrame(rows)


def compare_settings(train_df, eval_df, settings, label="Ayar"):
    """Seçilen baseline'ın farklı ayarlarını karşılaştırır.

    settings: {"ayar adı": {train_baseline parametreleri}} sözlüğü.
    """
    rows = []
    for name, params in settings.items():
        predictions = predict(train_baseline(train_df, **params), eval_df)
        scores = evaluate(predictions, eval_df)
        rows.append({
            label: name,
            **{key: round(value, 2) for key, value in scores.items()},
            "ortalama çıktı (karakter)": round(sum(map(len, predictions)) / len(predictions)),
        })
    return pd.DataFrame(rows)


def compare_length_caps(train_df, eval_df, ratios=(None, 4.0, 3.0, 2.0, 1.5)):
    """Çıktı uzunluğu sınırının (max_len_ratio) farklı değerlerini karşılaştırır."""
    settings = {("sınır yok" if ratio is None else ratio): {"max_len_ratio": ratio} for ratio in ratios}
    return compare_settings(train_df, eval_df, settings, label="max_len_ratio")


def robustness_to_h_spelling(train_df, eval_df):
    """Test yazımını (Ḫ/ḫ yerine H/h) taklit edip temizlemenin etkisini ölçer.

    Değerlendirme kaynaklarında Ḫ -> H, ḫ -> h yapılır; her yöntem için skorun
    orijinal yazıma göre ne kadar değiştiği raporlanır.
    """
    test_style = eval_df.copy()
    test_style[SRC_COL] = test_style[SRC_COL].str.replace("ḫ", "h").str.replace("Ḫ", "H")
    rows = []
    for analyzer, label in (("word", "TF-IDF kelime"), ("char_wb", "TF-IDF karakter")):
        for clean in (False, True):
            model = TfidfRetrievalBaseline(analyzer=analyzer, clean=clean)
            model.fit(train_df[SRC_COL], train_df[TGT_COL])
            original = evaluate(model.predict(eval_df[SRC_COL].tolist()), eval_df)["Score"]
            changed = evaluate(model.predict(test_style[SRC_COL].tolist()), test_style)["Score"]
            rows.append({
                "Yöntem": label, "Ön işleme": "var" if clean else "yok",
                "Score (train yazımı)": round(original, 2),
                "Score (test yazımı: H/h)": round(changed, 2),
                "Fark": round(changed - original, 2),
            })
    return pd.DataFrame(rows)
