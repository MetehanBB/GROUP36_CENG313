import re

def clean_akkadian_text(text):
    """
    Akadca çivi yazısı transliterasyonundaki modern akademik 
    notasyonları ve gereksiz karakterleri temizler.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Köşeli parantezleri ve içindekileri veya sadece parantez işaretlerini temizle
    text = re.sub(r'[\[\]\(\)\<\>]', '', text)
    
    # 2. Soru işareti, ünlem ve slash işaretlerini kaldır
    text = re.sub(r'[!?/]', '', text)
    
    # 3. Özel akademik işaretleri (örn: ˹ ˺) temizle
    text = re.sub(r'[˹˺]', '', text)
    
    # 4. Fazla boşlukları tek boşluğa indir ve kenarlardaki boşlukları kes
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Örnek kullanım testi:
if __name__ == "__main__":
    ornek_metin = "a-na [a-ti] ! wa-ar-ḫum / 1"
    temiz_metin = clean_akkadian_text(ornek_metin)
    print("Ham Metin: ", ornek_metin)
    print("Temiz Metin:", temiz_metin)
