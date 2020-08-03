from nltk.tokenize import sent_tokenize, word_tokenize
import os, re, math
from nltk import ngrams, FreqDist
from collections import Counter
from nltk import jaccard_distance

from nltk.corpus import stopwords

stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'revealed',
                   'use', 'follow', 'seen', 'changes', 'underwent', 'visit', 'done', 'admitted', 'reviewed', 'change',
                   'showed', 'denies', 'check', 'elevated', 'treated', 'found', 'taking', 'felt', 'shows', 'states',
                   'noted', 'reports', 'would', 'can', "given", 'plan', 'discussed', 'examined', 'evaluation',
                   'dictated', 'appearing', 'appears', 'controlled',
                   'feeling', 'feels', 'walking', 'pending', 'notable', 'impression', 'scheduled', 'work', 'study',
                   'started']
special_stopwords = ['date', 'record', 'patient', "history", "patient", "medications", "continue", "time", "although",
                     "significant", "exam", "prior", "home", "stable", "however", "last", "disease", "mild", "weeks",
                     'tid', 'two', 'asa', "negative", "positive", "recent", "symptoms", "daily", "since", "increased",
                     "days", "july", 'wbc', 'related', 'left', 'right', 'bilateral', 'problems', 'tablet', 'clinic',
                     'care', 'woman', 'man', 'slightly', 'increasing', 'female', 'wife', 'type', 'lives', 'good',
                     'current', 'report', 'setting', 'multiple', 'received', 'labs', 'hospitalization', 'episodes',
                     'old', 'presents', 'hpi', 'illness', 'severe', "week", 'mg_tablet', 'weight', 'discharge',
                     'including', 'decreased', 'improved', 'small', 'treatment', 'issues', 'minutes', 'wound',
                     'prn', 'month', 'none', 'ago', 'qhs', 'htn', 'mg', 'lower', 'regular', 'family', 'recently',
                     'general', 'high', 'control', 'age', 'resident', 'physical', 'rate', 'episode', 'morning',
                     'present', 'review', 'hold', 'due', 'strength', 'weakness', 'extremity', 'point', 'increase',
                     'reason', 'findings', 'bilaterally', 'area', 'regimen', 'primary', 'return', 'center'
                                                                                                  'note', 'units',
                     'social', "june", "march", "years", "card", "year", "admission", 'one', 'per',
                     'bid', 'copd', 'assessment', 'status', 'post', 'daughter', 'months', 'secondary', 'include',
                     'dose', 'times', 'results', 'developed', 'continues', 'difficulty', 'sleep', 'tab', 'failure',
                     'sob', 'neg', 'cxr', 'cmm_absolute', 'inr', 'vitd', 'low', 'baseline', 'loss', 'evidence', 'mid',
                     "back", "document", "hospital", "november", "september", "january", "street", "august", "april",
                     'day', 'medical', 'severe', 'clear', 'plasma', 'extremities', 'exercise', 'moderate', 'full',
                     "december", "start", "past", "without", "examination", "today", "also", "normal", "well",
                     'internal', 'notes', 'escription', 'etoh', 'approximately',
                     'likely', '|endoftext|', 'systems', 'male', 'management', 'consistent', 'consult', 'meds',
                     'medicine',
                     'night']

my_stopwords = stopwords.words('english') + stopwords_verbs + special_stopwords


def remove_stopwords(texts, stop_words):
    words = []
    for text in texts:
        if text not in stop_words:
            words.append(text)
    return words


def refine_text(text):
    text = re.sub(r'(\[\*{2}[0-9\-\sa-zA-Z\(\)\/]+\*{2}\])', '', text)
    # text = re.sub(r'[^\\n\.][A-Z\sa-z]+\:', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r"\b[a-zA-Z0-9]\b", "", text)

    text = re.sub(r'\s\d\.\s', ' ', text)
    text = re.sub(r'[\\.]\s\#[\\.]', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[\\.][\s\W][\\.]', ' ', text)
    text = re.sub(r'[\\.]\s\d[\\.]', ' ', text)
    text = re.sub(r'_{2,}', '', text)
    text = re.sub(r'\-{1,}', " ", text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text


nonPunct = re.compile('.*[A-Za-z0-9]{3,}.*')
if __name__ == "__main__":
    original_text = []

    original_word_count = 0
    wc = 0
    for file in os.listdir('i2b2_deid_data'):
        lines = open(os.path.join('i2b2_deid_data', file)).read()
        lines = refine_text(lines).lower()
        # original_sentences += sent_tokenize(lines)
        words = word_tokenize(lines)

        original_text += [w for w in words if (nonPunct.match(w) and w not in my_stopwords)]

        # text += lines + "\n"

    original_counts = Counter(original_text)

    # for size in 1, 2:
    #     all_counts[size] = FreqDist(ngrams(text, size))

    # corpus = open('/dspSharedData2/MominFiles/AdversarialClassifier/data/i2b2_seqgan_deid/merged_output.txt').read()
    corpus = open('/dspSharedData2/MominFiles/AdversarialClassifier/i2b2_original_DP.txt').read()
    # generated_sentences = []
    generated_text = []
    documents = corpus.split("\n" + ('=' * 20) + "\n")
    # documents = corpus.split("\n")
    for document in documents:
        lines = refine_text(document.strip()).lower()
        # generated_sentences += sent_tokenize(lines)

        # original_sentences += sent_tokenize(lines)
        words = word_tokenize(lines)
        generated_text += [w for w in words if (nonPunct.match(w) and w not in my_stopwords)]
        # text += lines + "\n"

    # generated_most_common += original_counts.most_common(500)
    c = len(original_text)
    d = len(generated_text)
    generated_counts = Counter(generated_text)
    original_most_common = original_counts.most_common(d)
    G2 = 0
    total = 0

    G2vals = []
    for w, a in original_most_common:
        b = generated_counts[w]
        E1 = c * (a + b) / (c + d)
        E2 = d * (a + b) / (c + d)

        G2val = 2 * (((b * math.log(b / E2)) if b != 0 else 0) + (a * math.log(a / E1)))
        ELL = G2val / (c + d) * math.log(min(E1, E2))
        Bayes = G2 - 1 * math.log(c + d)
        G2vals.append([w, a, b, G2val, ELL, Bayes])

        G2 += G2val
        total += 1

    for i in range(10):
        print(G2vals[i])
    print('G2 Test: {}, {}, {}'.format(G2 / total,c,d))

    print('Jaccard Distance:{}'.format(jaccard_distance(set(generated_text), set(original_text))))
