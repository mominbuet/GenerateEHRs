import os
import numpy as np
from pygments.lexer import include

if __name__ == "__main__":
    main_path = os.path.join('data', 'MIMICIII_original')
    files = os.listdir(main_path)
    for file in files:
        with open(os.path.join(main_path, file), 'r') as file:
            doc = file.read()
            lines = doc.split("\n")

            if 'Admission Date' in lines[1]:
                print(file.name)
    file = '/dspSharedData/TanbirFiles/ICD9-Classifier/mimic_data/adm_notes.npy'
    cui_list = ['4019', '4280', '42731', '41401', '5849', '25000', '51881', '5990']
    LABEL_COLUMNS = ['hypertension', 'congestive heart failure', 'atrial fibrillation', 'coronary atherosclerosis',
                     'aqcute kidney failure', 'diabetes mellitus type 2', 'acute respiratry failure',
                     'urinary tract infection']
    all_admin_notes = np.load(file, allow_pickle=True)
    for data in all_admin_notes:
        doc = data[5]
        if doc is not None:
            lines = doc.split("\n")
            # print(lines[0])
            inclusion = []
            for index, cui in enumerate(cui_list):
                if cui in data[len(data) - 2]:
                    inclusion.append(LABEL_COLUMNS[index])
            if len(inclusion) == 1:
                print(data[0], inclusion, lines[0])
                path_dir = os.path.join('data', 'MIMICIII_original_sep', inclusion[0])
                if not os.path.exists(path_dir):
                    os.mkdir(path_dir)
                with open(os.path.join(path_dir, str(data[0]) + '.txt'), 'w') as wfile:
                    wfile.write('ICD9 ' + inclusion[0] + '\n' + doc)
