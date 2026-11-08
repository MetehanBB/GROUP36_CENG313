import  math
import  sacrebleu

def compute_competition_metrics(predictions, references):
    """
    Yarışmanın resmi metriğini hesaplar:
    Score = sqrt(BLEU * chrF++)
    """
    # sacrebleu referans listesi list of lists bekler
    refs = [[ref] for ref in references]


    bleu = sacrebleu.corpus_bleu(predictions, refs).score
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2).score

    # Sıfır puan durumunda hata almamak için max(0.0, score) kullanıyoruz
    geom_mean = math.sqrt(max(0.0, bleu) * max(0.0, chrf))

    return {
        "BLEU": bleu,
        "chrF++": chrf,
        "Geometric_Mean": geom_mean
    }