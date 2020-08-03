import os
import numpy as np

paths = ['generated_original_data5','generated_original_data6']
ICD9s = ['hypertension', 'congestive heart failure', 'atrial fibrillation', 'coronary atherosclerosis',
         'aqcute kidney failure', 'diabetes mellitus type 2', 'acute respiratry failure', 'urinary tract infection']
data = []
index = 1
output_file = 'mimic3_original_generated2.npy'
for path in paths:
    files = os.listdir(path)



    # with open(output_file, 'w') as w_f:
    for file in files:
        if file.endswith('.txt'):
            text = open(os.path.join(path, file)).read()
            documents = text.split('\n' + '=' * 20 + '\n')
            for document in documents:
                document = document.strip()
                if document != '':
                    lines = document.split("\n")

                    label_text = lines[0].strip().split("&")

                    # text = ' '.join(lines[1:])
                    text = lines[1:]
                    labels = [0 for _ in range(len(ICD9s))]
                    if 'ICD9' in lines[0]:
                        for label in label_text:
                            if label != '':
                                labels[ICD9s.index(label.replace("ICD9", '').strip())] = 1
                    data.append([index, '\n'.join(text), labels])
                    # w_text = str(index) + ',"' + '\n'.join(text) + '"'
                    # for l in labels:
                    #     w_text += "," + str(l)
                    # w_f.write(w_text)
                    index += 1
print(len(data))
np.save(output_file, data)
