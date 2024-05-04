## Lưu ý: 
- Nếu gặp lỗi sau:
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 142: character maps to <undefined>
Thì bổ sung ', encoding="utf-8"' trong câu lệnh open file
- Cài đặt MeCab xử lý tokenizer cho tiếng Nhật: 
pip install mecab-python3
pip install unidic-lite

## Các bước thực hiện: 

#### 1. Chạy lệnh tạo corpus: 

python create_corpus.py --data_dir zac2021-ltr-data

python create_corpus_ja.py --data_dir zac2021-ltr-data-ja

conclusion:
- sử dụng file legal_corpus.json chứa dữ liệu của các laws và articles tương ứng
- sử dụng file train_question_answer.json chứa dữ liệu của các câu hỏi và laws tương ứng
- sử dụng file public_test_question.json chứa dữ liệu của các câu hỏi để test

- create file corpus.txt chứa nội dung của các laws và các câu hỏi
- create file legal_dict.json chứa nội dung của các laws theo mẫu law_id_article_id: {"title": article_title, "text": article_text}
- create file cocondenser_data.json chứa nội dung của các laws và các câu hỏi sau khi thực hiện chuẩn hóa dữ liệu (xuống dòng, đánh số thứ tự)

target:
- corpus.txt (for finetune language model)
- cocondenser_data.json (for finetune CoCondenser model)
- legal_dict.json 

#### 2. bm25_train.py

python bm25_train.py --data_path zac2021-ltr-data
python bm25_train_ja.py --data_path zac2021-ltr-data-ja

conclusion: 
- sử dụng file legal_corpus.json để tạo ra 2 file mới sau khi tiền xử lý là documents_manual và doc_refers_saved
- huấn luyện BM25_Plus trên dữ liệu file documents_manual đã tạo ở trên => bm25_Plus_04_06_model_full_manual_stopword
- tính thêm một số kết quả F1, recall, precision của dữ liệu sau khi huấn luyện

#### 3. bm25_create_pairs.py

python bm25_create_pairs.py --model_path saved_model/bm25_Plus_04_06_model_full_manual_stopword --data_path zac2021-ltr-data --save_pair_path pair_data

python bm25_create_pairs_ja.py --model_path saved_model_ja/bm25_Plus_04_06_model_full_manual_stopword --data_path zac2021-ltr-data-ja --save_pair_path pair_data_ja

conclusion: 
- Tạo ra các cặp câu question và document tương ứng, trong đó document là 1 laws trong top 20 laws liên quan nhất question (dùng bm25 đã huấn luyện trước đó để đánh giá)
- Tương ứng với mỗi cặp câu question và document đó thì mình sẽ có một nhãn là relevant có 2 giá trị là 0 với 1. Bằng 0 khi question và document không có trong file train_question_answer.json và bằng 1 là có trong file đó.

#### 4. Finetune language model using Huggingface

export MODEL_NAME=vinai/phobert-base; DATA_FILE=/kaggle/working/generated_data/corpus.txt; SAVE_DIR=/kaggle/working/saved_model/phobert-base \
    python run_mlm.py \
        --model_name_or_path "$MODEL_NAME" \
        --train_file "$DATA_FILE" \
        --do_train \
        --do_eval \
        --output_dir "$SAVE_DIR" \
        --line_by_line \
        --overwrite_output_dir \
        --save_steps 2000 \
        --num_train_epochs 20 \
        --per_device_eval_batch_size 32 \
        --per_device_train_batch_size 32

Hiện tại mình tập trung vào tạo tập dữ liệu theo hướng negative (như ở bước 3) nên bước này chưa cần tập trung.
Như bên Hoàng và Quyền thì mình có model này sẵn (vd BERTJapanese) và coi như nó đã huấn luyện trên dữ liệu luật.

python run_mlm.py 
    --model_name_or_path vinai/phobert-base 
    --train_file generated_data/corpus.txt 
    --do_train --do_eval 
    --output_dir saved_model/phobert-base 
    --line_by_line --overwrite_output_dir 
    --save_steps 2000 --num_train_epochs 20 
    --max_seq_length 1024
    --per_device_eval_batch_size 32 
    --per_device_train_batch_size 32

python run_mlm.py arguments.json
python3 -c "import torch; print(torch.cuda.is_available())"
CUDA_VISIBLE_DEVICES=0 python run_mlm.py arguments_ja.json

#### 5. Train condenser and cocondenser from language model checkpoint

CUDA_VISIBLE_DEVICES=0
python train_sentence_bert_ja.py --pretrained_model /kaggle/working/phobert-base --max_seq_length 256 --pair_data_path /kaggle/working/pair_data/bm_25_pairs_top20 --round 1 --num_val 1000 --epochs 10 --saved_model /kaggle/working/saved_model --batch_size 32

tiếng việt
python train_sentence_bert.py --pretrained_model saved_model/phobert-base --max_seq_length 256 --pair_data_path pair_data/bm_25_pairs_top20 --round 1 --num_val 1000 --epochs 5 --saved_model saved_model_round1 --batch_size 32

tiếng nhật
python train_sentence_bert_ja.py --pretrained_model saved_model_ja/japanese-roberta-base --max_seq_length 256 --pair_data_path pair_data_ja/bm_25_pairs_top20 --round 1 --num_val 5 --epochs 10 --saved_model saved_model_round1_ja --batch_size 32

Train model BERT, code trên là thực hiện huấn luyện trên model phobert-base, thay vì phobert-base thì mình có thể có code để huấn luyện cho BERT-japanese với ContrastiveLoss tương tự như họ trong file train_sentence_bert.py. 

Hoàng với Quyền k biết hiện tại có làm theo hướng này không?

output: model BERT sau khi huấn luyện ContrastiveLoss đã học được cách phân biệt câu law liên quan và câu law không liên quan đến question đầu vào.

#### 6. using hard negative pairs create from BERT model

python hard_negative_mining.py \
    --model_path /path/to/your/sentence/bert/model\
    --data_path /path/to/the/lagal/corpus/json\
    --save_path /path/to/directory/to/save/neg/pairs\
    --top_k top_k_negative_pair

tiếng việt
python hard_negative_mining.py --model_path saved_model/bm25_Plus_04_06_model_full_manual_stopword --sentence_bert_path saved_model_round1 --data_path zac2021-ltr-data --save_path pair_data --top_k 20

tiếng nhật
python hard_negative_mining_ja.py --model_path saved_model_ja/bm25_Plus_04_06_model_full_manual_stopword --sentence_bert_path saved_model_round1_ja --data_path zac2021-ltr-data-ja --save_path pair_data_ja --top_k 20

conclusion: 
- Tạo negative mức 2 bằng cách lấy top 20 câu mà model 1 dự đoán sai để thực hiện đưa vào huấn luyện ContrastiveLoss tiếp.

output: tạo được danh sách các cặp question và law mới là negative cho model huấn luyện

#### 7. Training again BERT model

python train_sentence_bert.py 
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 2 \
    --num_val $NUM_VAL\
    --epochs 5\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\

Tiếng việt: 
python train_sentence_bert.py --pretrained_model saved_model/japanese-roberta-base --max_seq_length 256 --pair_data_path pair_data/save_pairs_vibert_top20.pkl --round 2 --num_val 5 --epochs 5 --saved_model saved_model_round2 --batch_size 32

Tiếng nhật: 
python train_sentence_bert_ja.py --pretrained_model saved_model_ja/japanese-roberta-base --max_seq_length 256 --pair_data_path pair_data_ja/save_pairs_vibert_top20.pkl --round 2 --num_val 5 --epochs 5 --saved_model saved_model_round2_ja --batch_size 32

conclusion: 
- tương tự như lần trước huấn luyện ở bước 5 nhưng lần này data mạnh hơn sử dụng data đã tạo ở bước 6

output: model sau khi huấn luyện sẽ khắc phục được các nhược điểm của model trước.

Cần đánh giá model sau mỗi bước để có cái nhìn khách quan nhất.