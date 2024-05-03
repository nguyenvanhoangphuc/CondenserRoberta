import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, load_json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=20, type=int)
    parser.add_argument("--model_path", default="saved_model_ja/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--data_path", default="zac2021-ltr-data-ja", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="pair_data_ja/", type=str, help="path to save pair sentence directory")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "train_question.json")
    training_ques = json.load(open(train_path, encoding='utf-8'))
    # training_ques = load_json(train_path)

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("saved_model_ja/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(open(os.path.join("generated_data_ja", "legal_dict.json")))

    save_pairs = []
    top_n = args.top_pair
    end = 0
    for idx, item in tqdm(enumerate(training_ques)):
        question_id = item["question_id"]
        question = item["question"]
        print("=======================================================================")
        print("question\n", question)
        # relevant_articles = item["relevant_articles"]
        # actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]
        # show list of top_n relevant articles
        print("Top n relevant articles")
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            concat_id = pred[0] + "_" + pred[1]
            print(idx, idx_pred, doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"])

        decision = input("your answer? (y/n/e): ")
        if decision == "n":
            # Save negative pairs (all top_n pairs)
            for idx, idx_pred in enumerate(predictions):
                pred = doc_refers[idx_pred]
                concat_id = pred[0] + "_" + pred[1]
                save_dict = {}
                save_dict["question"] = question
                save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
            continue
        elif decision == "e":
            break
        else: 
            posi_idx = list(map(int, decision.strip().split(" ")))
            # Save positive and negative pairs
            for idx, idx_pred in enumerate(predictions):
                pred = doc_refers[idx_pred]
                concat_id = pred[0] + "_" + pred[1]
                if idx in posi_idx:
                    save_dict = {}
                    save_dict["question"] = question
                    save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                    save_dict["relevant"] = 1
                    save_pairs.append(save_dict)
                else:
                    save_dict = {}
                    save_dict["question"] = question
                    save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)
        print(len(save_pairs))
        # # Save negative pairs
        # for idx, idx_pred in enumerate(predictions):
        #     pred = doc_refers[idx_pred]

        #     check = 0
        #     concat_id = pred[0] + "_" + pred[1]
        #     print(question)
        #     print(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"])
        #     # quyết định
        #     decision = input("Is this relevant? (y/n/a/e): ")
        #     if decision == "y":
        #         check += 1
        #     elif decision == "a":
        #         break
        #     elif decision == "e":
        #         end = 1
        #         break

        #     if check == 0:
        #         save_dict = {}
        #         save_dict["question"] = question
        #         save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
        #         save_dict["relevant"] = 0
        #         save_pairs.append(save_dict)
        #     else:
        #         save_dict = {}
        #         save_dict["question"] = question
        #         save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
        #         save_dict["relevant"] = 1
        #         save_pairs.append(save_dict)
        # if end == 1:
        #     break
                    
    save_path = args.save_pair_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"bm_25_pairs_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
    print(len(save_pairs))