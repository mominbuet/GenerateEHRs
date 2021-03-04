# Generation 
The generation folder contains the GPT-2 generation technique for i2b2 and MIMIC-III dataset. It also incorporates the differential privacy technique from [Tensorflow Privacy](https://github.com/tensorflow/privacy)
GenerateText.py and GenerateText_i2b2.py runs the GPT generators. The 'gpt_2_simple' folder contains the DP codes included to GPT-2

[Tensorflow Privacy](https://github.com/tensorflow/privacy)  needs to be installed prior to running the generation. The parameters for running GPT is included in `GenerateText.py` (or i2b2), change the input params in line 27. For example, optimizer is currently set as 'dpadam' which is the differentially private adam with noise multiplier 5.0. The epsilon can be calculated using [Tensorflow Privacy package](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/analysis)

The input and output data both have 'ICD9 Code' at the beginning of each document. This is used in the classifiers as well. 
# Utility
Utility folder contains the code that tests the three utility metrics used in the paper. Adding the run parameters or settings follow [BERT] (https://github.com/google-research/bert) classifiers. To change the input check this [function](https://github.com/mominbuet/GenerateEHRs/blob/fcdc74c5f66d9fb2de2ebbb0b57c1ff0e3be4b03/Utility/ICD9Classifier.py#L254)

Contact azizmma@cs.umanitoba.ca for details
