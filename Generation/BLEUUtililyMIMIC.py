from docutils.nodes import document
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
import os, re
from joblib import Parallel, delayed
import multiprocessing

from G2Utility import nonPunct, my_stopwords, refine_text


def getSentenceFromDocument(sentence):
    sentence = refine_text(sentence).lower()
    tokens = word_tokenize(sentence)
    result = []
    if 6 < len(tokens) < 20:
        tmp = [w for w in tokens if (nonPunct.match(w) and w not in my_stopwords)]
        if len(tmp) > 5:
            result += tmp
            # print(tmp)

    return result


num_cores = multiprocessing.cpu_count()


def getBLEU(original_sentences, generated_sentence, weights):
    B1 = sentence_bleu(original_sentences, generated_sentence, weights=weights,
                       smoothing_function=SmoothingFunction().method1)
    # print(B1)
    #  B2 = sentence_bleu(original_sentences, generated_sentence, weights=(0, 1, 0, 0))
    # # B3 = sentence_bleu(original_sentences, generated_sentence, weights=(0, 0, 1, 0))
    return B1

corpus = ''
corpus_dir = '/dspSharedData2/MominFiles/AdversarialClassifier/data/mimic_dp'
files = os.listdir(corpus_dir)
for file in files:
    # if 'hypertension' in file:

    with open(os.path.join(corpus_dir, file)) as read_file:
        read_file.readline()
        corpus += read_file.read()
##mimic III end

generated_sentences = []
# documents = corpus.split("\n" + ('=' * 20) + "\n")
documents = corpus.split("\n")
for document in documents:

    # document = ' '.join(l for l in document.split("\n"))  # skip 3 lines
    # lines = refine_text(document.strip()).lower()
    sentences = sent_tokenize(document)
    # print(sentences)
    document_sent = []
    tmp = Parallel(n_jobs=num_cores)(delayed(getSentenceFromDocument)(sent) for sent in sentences)
    tmp_refined = []
    for t in tmp:
        if len(t) >2:
            tmp_refined += (t)
            # print(t)
    if len(tmp_refined) > 5:
        generated_sentences.append([tmp_refined])

print("got all generated "+str(len(generated_sentences)))

original_sentences = []
mimic_dir = '/dspSharedData2/MominFiles/AdversarialClassifier/data/MIMICIII_original_sep'
total_sentences = 0
for folder in os.listdir(mimic_dir):
    # if 'hypertension' in folder:
    for file in os.listdir(os.path.join(mimic_dir, folder)):
        with open(os.path.join(mimic_dir, folder, file)) as rfile:
            # rfile.readline()
            lines = rfile.read()
            # lines = refine_text(lines).lower()
            sentences = sent_tokenize(lines)
            # if len(tmp) > 5:
            document_sent = []
            tmp = Parallel(n_jobs=num_cores)(delayed(getSentenceFromDocument)(sent) for sent in sentences)
            # print(tmp)
            for t in tmp:
                if len(t) > 5:
                    document_sent += t
                    # total_sentences += 1
            # if len(tmp) > 5:
            #     document_sent += tmp
            # print(document_sent)
            if len(document_sent) > 2:
                original_sentences.append(document_sent)
                # print(document_sent)
            # if len(original_sentences) > 10:
            #     break
print("got all originals")
print(len(original_sentences))
# corpus = open('i2b2_generated_deid2.txt').read()#i2b2
##mimic III start


    # if len(generated_sentences) > 10: break
    # corpus_bleu(original_sentences, document_sent)

# for sentences in docs:
#     sentence_bleu(original_sentences, sentences, weights=(1, 0, 0, 0))
# Parallel(n_jobs=num_cores)(
#     delayed(getBLEU)(tuple( i for i in sentences), (1, 0, 0, 0)) for sentences in docs)
# B1 = Parallel(n_jobs=num_cores)(
#     delayed(getBLEU)(original_sentences, sentences, (1, 0, 0, 0)) for sentences in generated_sentences)
# B2 = Parallel(n_jobs=num_cores)(
#     delayed(getBLEU)(original_sentences, sentences, (0, 1, 0, 0)) for sentences in generated_sentences)
# B3 = Parallel(n_jobs=num_cores)(
#     delayed(getBLEU)(original_sentences, sentences, (0, 0, 1, 0)) for sentences in generated_sentences)
# # print(sum(B2))
# # score += [sum(B1), sum(B2), sum(B3)]
#
# print('BLEU 1,2,3 scores" {},{},{}'.format(sum(B1) / len(B1), sum(B2) / len(B2), sum(B3) / len(B3)))


score = [0, 0, 0, 0]
total = 0
for docs in generated_sentences:
    for sentences in docs:
        score[0] += sentence_bleu(original_sentences, sentences, weights=(1, 0, 0, 0))
        score[1] += sentence_bleu(original_sentences, sentences, weights=(0, 1, 0, 0))
        score[2] += sentence_bleu(original_sentences, sentences, weights=(0, 0, 1, 0))
        total += 1
        if total > 3000:
            break
print('BLEU1,2,3 scores" {},{},{}'.format(score[0] / total, score[1] / total, score[2] / total))
