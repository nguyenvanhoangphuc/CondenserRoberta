import json
import os
import re
from tqdm import tqdm
import argparse

def load_json(corpus_path):
    data = json.load(open(corpus_path, encoding="utf-8"))
    return data["items"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # --data_dir zac2021-ltr-data 
    parser.add_argument("--data_dir", default="./data", type=str, help="path to training data")  
    # --save_dir generated_data
    parser.add_argument("--save_dir", default="./generated_data_ja", type=str, help="path to training data")
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    # tạo file corpus.txt chứa nội dung của các laws
    cp = open(os.path.join(args.save_dir, "corpus.txt"), "w", encoding="utf-8")
    # mở file legal_corpus.json chứa nội dung của các question and laws tương ứng
    corpus_path = os.path.join(args.data_dir, "legal_corpus.json")
    # data chứa dữ liệu file legal_corpus.json
    data = json.load(open(corpus_path, encoding="utf-8"))

    save_dict = {}
    co_f = open(os.path.join(args.save_dir, "cocondenser_data.json"), "w", encoding="utf-8")
    count = 0
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["article_id"]
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            article_full = article_title + ". " + article_text
            article_full = article_full.replace("\n", " ")
            # viết full article vào file corpus.txt
            cp.write(article_full + "\n")
            
            # Save data for cocondenser 
            spans = [article_title]
            passages = re.split(r"\n[0-9]+\. |1\. ", article_text)
            for idx, p in enumerate(passages):
                if p != "":
                    article_full = article_title + ". " + p
                    article_full = article_full.replace("\n", " ")
                    spans.append(p)
            # viết data vào file cocondenser_data.json sau khi thực hiện chuẩn hóa dữ liệu (xuống dòng, đánh số thứ tự)
            co_f.write("#".join(spans) + "\n")
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {"title": article_title, "text": article_text}
    
    co_f.close()
    print(count)
    # exit()
    print("Create legal dict from raw data")
    # lưu dữ liệu vào file legal_dict.json
    with open(os.path.join(args.save_dir, "legal_dict.json"), "w", encoding="utf-8") as outfile:
        json.dump(save_dict, outfile)
    print("Finish")

    # đọc file train_question.json để lấy nội dung của các câu hỏi
    # sau đó viết nội dung của các câu hỏi vào file corpus.txt
    question_path = os.path.join(args.data_dir, "train_question.json")
    training_ques = json.load(open(question_path, encoding='utf-8'))
    for item in tqdm(training_ques):
        question = item["question"]
        cp.write(question + "\n")

    # ** đây là phần ẩn đi có thể mở lại **
    # corpus_path_train = os.path.join(args.data_dir, "train_question_answer.json")
    # items = load_json(corpus_path_train)
    # # ghi nội dung của các câu hỏi vào file corpus.txt
    # for item in tqdm(items):
    #     question = item["question"]
    #     cp.write(question + "\n")

    # corpus_path_test = os.path.join(args.data_dir, "public_test_question.json")
    # items = load_json(corpus_path_test)

    # for item in tqdm(items):
    #     question = item["question"]
    #     cp.write(question + "\n")
    # ****

    cp.close()

## conclusion:
# sử dụng file legal_corpus.json chứa dữ liệu của các laws và articles tương ứng
# sử dụng file train_question_answer.json chứa dữ liệu của các câu hỏi và laws tương ứng
# sử dụng file public_test_question.json chứa dữ liệu của các câu hỏi để test

# create file corpus.txt chứa nội dung của các laws và các câu hỏi
# create file legal_dict.json chứa nội dung của các laws theo mẫu law_id_article_id: {"title": article_title, "text": article_text}
# create file cocondenser_data.json chứa nội dung của các laws và các câu hỏi sau khi thực hiện chuẩn hóa dữ liệu (xuống dòng, đánh số thứ tự)

# end
# corpus.txt (for finetune language model)
# cocondenser_data.json (for finetune CoCondenser model)