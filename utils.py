import string
from underthesea import word_tokenize
import os
import json
import MeCab  # or KyTea for Japanese tokenization

stop_word_japanese = ["です", "これ", "それ"]  # Add more stopwords as needed
number_japanese = ["１", "２", "３", "４", "５", "６", "７", "８", "９", "１０"]  # Japanese numbers
chars_japanese = ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ"]  # Hiragana characters

def remove_stopword_japanese(w):
    return w not in stop_word_japanese

def remove_punctuation_japanese(w):
    return w not in string.punctuation and w != "、" and w != "。"  # Japanese punctuation

def lower_case_japanese(w):
    return w.lower()

def japanese_tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")  # Initialize MeCab with desired options for Japanese tokenization
    tokens = tagger.parse(text).split()  # Tokenize Japanese text
    tokens = list(map(lower_case_japanese, tokens))
    tokens = list(filter(remove_punctuation_japanese, tokens))
    tokens = list(filter(remove_stopword_japanese, tokens))
    return tokens

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def load_json(path):
    return json.load(open(path))
