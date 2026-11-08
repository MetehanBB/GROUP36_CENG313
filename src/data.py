import re
import pandas as pd

# Subscript rakamları standart rakamlara dönüştürme tablosu
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def clean_akkadian_text(text: str) -> str:
    """
    Akadca çivi yazısı transliterasyonundaki modern akademik 
    notasyonları ve gereksiz karakterleri temizler.
    """
    if not isinstance(text, str):
        return ""

    # Subscript (alt indis) rakamları normale çevir (örn: il₅ -> il5)
    text = text.translate(SUBSCRIPT_MAP)

    # Köşeli parantezleri ve içindekileri veya sadece parantez işaretlerini temizle
    text = re.sub(r'\[x\]', '<gap>', text)
    text = re.sub(r'(\.\.\.|\[\s*\.\.\.\s*\])', '<big_gap>', text)
    
    # Soru işareti, ünlem ve slash işaretlerini kaldır
    text = re.sub(r'[!?/]', '', text)

    # Kırık işaret parantezlerini sil (içerideki harf kalır)
    # Yarım parantezler (˹ ˺), tavan parantezler (⌈ ⌉) ve ters tırnaklar
    for char in ['˹', '˺', '⌈', '⌉', '`', '´', '’']:
        text = text.replace(char, '')
    
    # H ve h normalizasyonu (Test verisinde sadece standart H ve h var)
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')

    # Köşeli ve sivri parantez işaretlerini kaldır (içerideki kelime kalsın)
    # [KÙ.BABBAR] -> KÙ.BABBAR (gap etiketlerindeki <> işaretlerine dokunmadan)
    text = text.replace('[', '').replace(']', '')
    text = re.sub(r'(?<!<)>(?!>)|(?<!<)<(?!gap|big_gap)', '', text)

    # Fazla boşlukları tek boşluğa indir ve kenarlardaki boşlukları kes
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data(train_path: str, test_path: str):
    """
    Train ve test verilerini yükleyip temel temizliği uygular.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['clean_transliteration'] = train_df['transliteration'].apply(clean_akkadian_text)
    test_df['clean_transliteration'] = test_df['transliteration'].apply(clean_akkadian_text)
    
    return train_df, test_df
