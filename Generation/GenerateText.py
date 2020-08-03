import os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gpt_2_simple as gpt2


def write_output(output_file, res):
    # output_file = os.path.join('generated_original_data', icd9.replace(' ', '') + '.txt')
    with open(output_file, 'a') as wf:
        for text in res:
            if text != '':
                try:
                    wf.write(text + '\n')
                    wf.write('=' * 20 + '\n')
                except:
                    print('Could not write ' + text)


model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print("Downloading {} model...".format(model_name))
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

file_name = "MIMICIII_original"
# if not os.path.isfile(file_name):
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     data = requests.get(url)
#
#     with open(file_name, 'w') as f:
#         f.write(data.text)

sess = gpt2.start_tf_sess(threads=6)

gpt2.finetune(sess, file_name, batch_size=1, multi_gpu=True, sample_every=500, sample_length=1024, combine=1024 * 5,
              model_name=model_name,noise_multiplier=5.0, optimizer='dpadam', NB_TRAIN=9817,#len(os.listdir(file_name)),
              run_name='MIMIC_original_124M_DP', steps=30000)  # steps is max number of training steps
#
ICD9s = ['hypertension', 'congestive heart failure', 'atrial fibrillation', 'coronary atherosclerosis',
         'aqcute kidney failure', 'diabetes mellitus type 2', 'acute respiratry failure',
         'urinary tract infection']
# ICD9s = ['']
for icd9 in ICD9s:

    res = []

    while len(res) < 1000:
        try:
            tmp = gpt2.generate(sess, run_name='MIMIC_original_124M_output', nsamples=10,
                                prefix='ICD9 ' + icd9 + '\nAdmission Date:  [**',
                                temperature=random.randrange(70, 75, 1) / 100, top_k=40,
                                return_as_list=True)
            res += tmp
            write_output(os.path.join('generated_dp_mimic', icd9.replace(' ', '') + '.txt'), tmp)
        except:
            continue

    print(icd9)

    # for icd9_2 in ICD9s:
    #     if icd9_2 != icd9:
    #         res = []
    #         while len(res) < 200:
    #             try:
    #                 tmp = gpt2.generate(sess, run_name='MIMIC_original_124M_output', nsamples=5,
    #                                      prefix='ICD9 ' + icd9 + '& ICD9 ' + icd9_2 + '\nHISTORY OF PRESENT ILLNESS:',
    #                                      temperature=random.randrange(70, 75, 1) / 100, top_k=40,
    #                                      return_as_list=True)
    #
    #                 of = os.path.join('generated_original_data5',
    #                                   icd9.replace(' ', '') + '_' + icd9_2.replace(' ', '') + '.txt')
    #                 write_output(of, tmp)
    #                 res += tmp
    #             except:
    #                 continue
