from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
import os, re

from G2Utility import nonPunct, my_stopwords, refine_text

original_sentences = []
for file in os.listdir('i2b2_data'):
    open(os.path.join('i2b2_data', file)).readline()
    lines = open(os.path.join('i2b2_data', file)).read()
    lines = refine_text(lines).lower()
    sentences = sent_tokenize(lines)
    # if len(tmp) > 5:
    document_sent = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        if len(tokens) > 5:
            tmp = [w for w in tokens if (nonPunct.match(w) and w not in my_stopwords)]
            if len(tmp) > 5:
                document_sent += tmp
    if len(document_sent) > 2:
        original_sentences.append(document_sent)

corpus = open('/dspSharedData2/MominFiles/AdversarialClassifier/i2b2_original_DP.txt').read()
# corpus = open('/dspSharedData2/MominFiles/AdversarialClassifier/data/i2b2_seqgan_deid/merged_output.txt').read()#i2b2
##mimic III start
# corpus = ''
# corpus_dir = '/dspSharedData2/MominFiles/AdversarialClassifier/data/generated_774'
# files = os.listdir(corpus_dir)
# for file in files:
#     with open(os.path.join(corpus_dir, file)) as read_file:
#         read_file.readline()
#         corpus += read_file.read()
##mimic III end

generated_sentences = []
# documents = corpus.split("\n" + ('=' * 20) + "\n")
documents = corpus.split("\n" )
for document in documents:
    lines = refine_text(document.strip()).lower()
    sentences = sent_tokenize(lines)
    document_sent = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tmp = [w for w in tokens if (nonPunct.match(w) and w not in my_stopwords)]
        if len(tmp) > 5:
            document_sent.append(tmp)
    generated_sentences.append(document_sent)
    # corpus_bleu(original_sentences, document_sent)

score = [0, 0, 0, 0]
total = 0
for docs in generated_sentences:
    for sentences in docs:
        score[0] += sentence_bleu(original_sentences, sentences, weights=(1, 0, 0, 0),smoothing_function=SmoothingFunction().method1)
        score[1] += sentence_bleu(original_sentences, sentences, weights=(0, 1, 0, 0),smoothing_function=SmoothingFunction().method1)
        score[2] += sentence_bleu(original_sentences, sentences, weights=(0, 0, 1, 0),smoothing_function=SmoothingFunction().method1)
        total += 1
        # if total > 3000:
        #     break
print('BLEU1,2,3 scores" {},{},{}'.format(score[0] / total, score[1] / total, score[2] / total))
