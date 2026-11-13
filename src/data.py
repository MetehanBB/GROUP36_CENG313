"""Veri yükleme, ön işleme ve bölme fonksiyonları (Deep Past Challenge).

Notebook'lar bu modülü şöyle kullanır:

    from src.data import load_data, preprocess, split_data
"""
import glob
import os
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

COMPETITION = "deep-past-initiative-machine-translation"
SRC_COL = "transliteration"
TGT_COL = "translation"
SRC_CLEAN = "src_clean"
TGT_CLEAN = "tgt_clean"
REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")

# Verinin aranacağı klasörler (sırayla): yerel/Colab, Kaggle, eski klasör adı.
DATA_DIR_CANDIDATES = ("data", f"/kaggle/input/{COMPETITION}", COMPETITION)

DATA_HELP = (
    "Yarışma verisi bulunamadı. Şunlardan birini yap:\n"
    "  1) train.csv, test.csv ve sample_submission.csv dosyalarını data/ klasörüne koy, veya\n"
    "  2) Kaggle API anahtarını tanımla (KAGGLE_USERNAME + KAGGLE_KEY; Colab'de Secrets "
    "bölümüne ekle) ve yarışma kurallarını Kaggle sitesinde kabul et.\n"
    f"Yarışma: https://www.kaggle.com/c/{COMPETITION}"
)


# --------------------------------------------------------------------------
# Veri yükleme
# --------------------------------------------------------------------------
def _has_required_files(folder):
    return all(os.path.exists(os.path.join(folder, name)) for name in REQUIRED_FILES)


def find_data_dir():
    """Gerekli 3 dosyayı içeren ilk klasörü döndürür; yoksa None."""
    for folder in DATA_DIR_CANDIDATES:
        if _has_required_files(folder):
            return folder
    # Kaggle veri klasörünü farklı bir alt klasöre bağlarsa onu da bul.
    for path in sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True)):
        folder = os.path.dirname(path)
        if _has_required_files(folder):
            return folder
    return None


def _load_colab_secrets():
    """Colab Secrets içindeki Kaggle anahtarlarını ortam değişkenine aktarır."""
    try:
        from google.colab import userdata
    except ImportError:
        return
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        if os.environ.get(key):
            continue
        try:
            value = userdata.get(key)
        except Exception:  # anahtar yok ya da erişim izni verilmedi
            continue
        if value:
            os.environ[key] = value


def download_data(data_dir="data"):
    """Gerekli 3 dosyayı Kaggle API ile indirir (API anahtarı gerekir)."""
    import subprocess
    import zipfile

    _load_colab_secrets()
    os.makedirs(data_dir, exist_ok=True)
    for name in REQUIRED_FILES:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", data_dir],
            check=True,
        )
        zip_path = os.path.join(data_dir, name + ".zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(data_dir)
            os.remove(zip_path)
    return data_dir


def load_data(data_dir=None, download=True):
    """(train_df, test_df, sample_submission) döndürür.

    data_dir verilmezse data/ ve Kaggle klasörleri aranır; hiçbiri yoksa
    Kaggle API ile indirme denenir.
    """
    folder = data_dir or find_data_dir()
    if folder is None and download:
        try:
            folder = download_data("data")
        except Exception as exc:
            raise FileNotFoundError(DATA_HELP) from exc
    if folder is None or not _has_required_files(folder):
        raise FileNotFoundError(DATA_HELP)

    train_df = pd.read_csv(os.path.join(folder, "train.csv"))
    test_df = pd.read_csv(os.path.join(folder, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(folder, "sample_submission.csv"))
    return train_df, test_df, sample_sub


# --------------------------------------------------------------------------
# Ön işleme
# --------------------------------------------------------------------------
# Alt simge rakamlar normal rakama çevrilir (il₅ -> il5).
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

# Kırık işareti temizlik sırasında korunsun diye geçici yer tutucu.
_GAP = ""

# Yarışma sayfasındaki determinatif listesi; test örneğinde (ki) biçiminde geçebiliyor.
_DETERMINATIVES = (
    "d|ki|m|mi|f|lu2|e2|uru|kur|mul|geš|ĝeš|giš|tug2|dub|id2|mušen|na4|kuš|u2"
)
_ELLIPSIS = r"(?:…|\.{3,})"


def clean_akkadian_text(text):
    """Transliterasyondaki akademik notasyonu sadeleştirir, kırıkları standartlaştırır.

    Adımların gerekçeleri notebook'un "Preprocessing" bölümünde anlatılır.
    """
    if not isinstance(text, str):
        return ""

    # 1) Aynı harfin farklı Unicode yazımlarını tek biçime indir.
    text = unicodedata.normalize("NFC", text)

    # 2) Train'de Ḫ/ḫ, testte yalnızca H/h var.
    text = text.replace("ḫ", "h").replace("Ḫ", "H")

    # 3) Alt simge rakamlar -> normal rakam. Test örneğinde ₄ yerine „ geçiyor.
    text = text.translate(SUBSCRIPT_MAP).replace("„", "4")

    # 4) Kırıkların hepsi <gap>. Yarışma uzun kırıklar için <big_gap> öneriyor, ama
    #    train'de (kaynak ve çeviri) yalnızca <gap> var; test, train ile aynı biçime iner.
    text = re.sub(r"<\s*(?:big[_ ])?gap\s*>", _GAP, text, flags=re.IGNORECASE)
    text = re.sub(rf"\[\s*{_ELLIPSIS}(?:\s*{_ELLIPSIS})*\s*\]", _GAP, text)
    text = re.sub(r"\[\s*x(?:\s+x)*\s*\]", _GAP, text)
    text = re.sub(_ELLIPSIS, _GAP, text)

    # 5) Determinatifler: (ki) -> {ki}; süslü parantez içi küçük harf.
    text = re.sub(rf"\(({_DETERMINATIVES})\)", r"{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\s*([^{}]*?)\s*\}", lambda m: "{" + m.group(1).lower() + "}", text)

    # 6) Modern akademik işaretler: hatalı işaretler (<< >>) içeriğiyle silinir,
    #    diğer parantezlerin yalnızca kendisi silinir (içerik kalır).
    text = re.sub(r"<<[^<>]*>>", " ", text)
    text = re.sub(r"[!?/:]", "", text)
    for char in "˹˺⌈⌉⌊⌋[]<>`´’":
        text = text.replace(char, "")

    # 7) Kırık etiketlerini geri koy; kelimeye tire ile bağlı değilse boşlukla ayır.
    #    Ardışık <gap>'ler birleştirilmez, çünkü train'de de ardışık geçiyorlar.
    text = re.sub(rf"(?<=[^\s-])(?={_GAP})", " ", text)
    text = re.sub(rf"(?<={_GAP})(?=[^\s-])", " ", text)
    text = text.replace(_GAP, "<gap>")

    # 8) Fazla boşlukları tek boşluğa indir.
    return re.sub(r"\s+", " ", text).strip()


def clean_translation(text):
    """İngilizce çeviride yalnızca Unicode ve boşluk normalizasyonu yapar.

    Çeviri metriğin ölçtüğü hedef olduğu için içeriğine dokunulmaz.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df):
    """Temizlenmiş sütunları ekler; eğitim verisinde boş ve tekrar eden çiftleri atar.

    Çeviri sütunu olmayan tabloda (test) satır atılmaz, çünkü her id için tahmin gerekir.
    """
    out = df.copy()
    out[SRC_CLEAN] = out[SRC_COL].apply(clean_akkadian_text)
    if TGT_COL in out.columns:
        out[TGT_CLEAN] = out[TGT_COL].apply(clean_translation)
        out = out[(out[SRC_CLEAN] != "") & (out[TGT_CLEAN] != "")]
        out = out.drop_duplicates(subset=[SRC_CLEAN, TGT_CLEAN])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Train / validation / held-out test bölmesi
# --------------------------------------------------------------------------
def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """(train, validation, held-out test) döndürür.

    Önce held-out test ayrılır (train_test_split(test_size=0.2, random_state=42)),
    sonra kalan kısımdan validation ayrılır. Oranlar tüm veriye göredir.
    """
    df = df.dropna(subset=[SRC_COL, TGT_COL])
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=random_state)
    return train, val, test
