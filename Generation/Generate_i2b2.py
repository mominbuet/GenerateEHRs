import os,random
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

file_name = "i2b2_deid_data"
# if not os.path.isfile(file_name):
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     data = requests.get(url)
#
#     with open(file_name, 'w') as f:
#         f.write(data.text)

sess = gpt2.start_tf_sess(threads=4)
gpt2.finetune(sess,file_name, batch_size=1,multi_gpu=True,run_name='i2b2_deid_124M',
              sample_every=1000,sample_length=512,model_name=model_name, optimizer = 'dpadam',
              noise_multiplier=1, l2_norm_clip=1.5,NB_TRAIN=514,
              steps=60000)
res = []
while len(res) < 300:
    try:
        tmp = gpt2.generate(sess, run_name='i2b2_deid_124M', nsamples=5,
                                     prefix='Record date:',
                                     temperature=random.randrange(70, 75, 1) / 100, top_k=40,
                                     return_as_list=True)
        res += tmp
        write_output(os.path.join('i2b2_deid_dp_gendata124.txt'), tmp)
    except:
        continue
#gpt2.generate(sess,prefix='Record date: ',nsamples=200,run_name='i2b2_deid_774M',destination_path='i2b2_generated_deid.txt')
#gpt2.generate(sess,temperature=0.8,top_p=.8,prefix='Record date: ',nsamples=200,run_name='i2b2_deid_774M',destination_path='i2b2_generated_deid2.txt')
