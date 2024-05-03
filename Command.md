python create_corpus.py --data_dir zac2021-ltr-data-ja

- create file corpus.txt chứa nội dung của các laws và các câu hỏi
- create file legal_dict.json chứa nội dung của các laws theo mẫu law_id_article_id: {"title": article_title, "text": article_text}
- create file cocondenser_data.json chứa nội dung của các laws và các câu hỏi sau khi thực hiện chuẩn hóa dữ liệu (xuống dòng, đánh số thứ tự)

python bm25_train.py --data_path zac2021-ltr-data-ja

- sử dụng file legal_corpus.json để tạo ra 2 file mới sau khi tiền xử lý là documents_manual và doc_refers_saved
- huấn luyện BM25_Plus trên dữ liệu file documents_manual đã tạo ở trên => bm25_Plus_04_06_model_full_manual_stopword

python find_law_bm25.py --model_path saved_model_ja/bm25_Plus_04_06_model_full_manual_stopword --data_path zac2021-ltr-data-ja --save_pair_path pair_data_ja

- tìm ra top 20 câu liên quan nhất đến từng question trong hợp đồng.
- hiển thị ra cho người dùng chọn danh sách câu liên quan đến question.
- tạo positive và negative data for training.