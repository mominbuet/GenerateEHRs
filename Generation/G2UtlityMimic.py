from docutils.nodes import docinfo
from nltk.tokenize import sent_tokenize, word_tokenize
import os, re, math
from nltk import ngrams, FreqDist
from collections import Counter
from nltk import jaccard_distance

from nltk.corpus import stopwords

from G2Utility import stopwords_verbs, special_stopwords

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
    text = re.sub(r'\s{2,}', ' ', text)

    return text


# MAX_DOC=1000
nonPunct = re.compile('.*[A-Za-z0-9]{3,}.*')
if __name__ == "__main__":
    original_text = []

    original_word_count = 0
    wc = 0
    mimic_dir = '/dspSharedData2/MominFiles/AdversarialClassifier/data/MIMICIII_original_sep'
    for folder in os.listdir(mimic_dir):
        # if 'hypertension' in folder:
            for file in os.listdir(os.path.join(mimic_dir, folder)):
                lines = open(os.path.join(mimic_dir,folder, file)).read()
                lines = refine_text(lines).lower()
                # original_sentences += sent_tokenize(lines)
                words = word_tokenize(lines)
                if len(words) > 512: continue
                original_text += [w for w in words if (nonPunct.match(w) and w not in my_stopwords)]
                # if len(original_text) > 5000: break
                # text += lines + "\n"

    original_counts = Counter(original_text)

    # for size in 1, 2:
    #     all_counts[size] = FreqDist(ngrams(text, size))

    # generated_sentences = []
    generated_text = []
    corpus_dir = '/dspSharedData2/MominFiles/AdversarialClassifier/data/mimic_dp'
    files = os.listdir(corpus_dir)
    for file in files:
        # if 'hypertension' in file:
            with open(os.path.join(corpus_dir, file)) as rfile:
                rfile.readline()  # skip 1st line
                corpus = rfile.read()
                documents = corpus.split("\n" + ('=' * 20) + "\n")
                # documents = corpus.split("\n" )
                for document in documents:
                    document = refine_text(document.strip()).lower()
                    # lines = document.split("\n")[1:]

                    # generated_sentences += sent_tokenize(lines)

                    # original_sentences += sent_tokenize(lines)
                    words = word_tokenize(document)

                    generated_text += [w for w in words if (nonPunct.match(w) and w not in my_stopwords)]
                    # if len(generated_text) > len(original_text): break
                    # text += lines + "\n"

    # generated_most_common += original_counts.most_common(500)
    c = len(original_text)
    d = len(generated_text)
    generated_counts = Counter(generated_text)
    original_most_common = original_counts.most_common(c)
    G2 = 0
    total = 0
    G2vals = []
    for w, a in original_most_common:
        b = generated_counts[w]
        if b > 0:
            E1 = c * (a + b) / (c + d)
            E2 = d * (a + b) / (c + d)

            G2val = 2 * (((b * math.log(b / E2)) if b != 0 else 0) + (a * math.log(a / E1)))

            ELL = 100 * G2val / ((c + d) * math.log(min(E1, E2)))
            Bayes = G2 - 1 * math.log(c + d)
            G2vals.append([w, a, b, G2val, ELL,Bayes])

            G2 += G2val  # / ((c + d) * math.log(min(a, b)))
            total += 1

    for i in range(100):
        print(G2vals[i])

    print('G2 Test: {},{},{}'.format(G2 / total, c, d))

    print('Jaccard Distance:{}'.format(jaccard_distance(set(generated_text), set(original_text))))