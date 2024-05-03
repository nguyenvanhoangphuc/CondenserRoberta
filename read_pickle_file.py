# read pickle file from path

import pickle


# file_path = r"pair_data_ja\bm_25_pairs_top20"
# file_path = r"saved_model_ja/documents_manual"
# file_path = r"saved_model/documents_manual"
# file_path = r"saved_model_ja/doc_refers_saved"
# file_path = r"saved_model/doc_refers_saved"
# file_path = r"pair_data_ja/bm_25_pairs_top20"
file_path = r"pair_data/bm_25_pairs_top20"
with open(file_path, "rb") as bm_file:
    bm25 = pickle.load(bm_file)

print(bm25[:20])
print(len(bm25))

# for item in bm25:
#     if item['relevant']==1:
#         print("có relevant 1")
# import os
# import json
# import argparse
# import numpy as np
# import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", default="zac2021-ltr-data-ja", type=str, help="path to input data")

# args = parser.parse_args()

# train_path = os.path.join(args.data_path, "train_question.json")
# training_ques = json.load(open(train_path, encoding='utf-8'))

# for idx, item in enumerate(training_ques):
#     print(idx, item["question"])
#     if idx == 2:
#         break