import os
import time
# from ..config import PADDING_LENGTH, stop_words, chars
from config import PADDING_LENGTH, stop_words, chars
stime = time.time()


def read_txt(path):
    # returns list of sentences separated by \n
    f = open(path, "r")
    data = f.readlines()
    for i in range(0, len(data)):
        data[i] = data[i].replace("\n", "")    
    f.close()

    return data


def _remove_chars(text):
    # sentence wise
    for i, t in enumerate(text):
        if t in chars:
            text = text.replace(t, "")
    # print(f"here {t}, {text}")
    return text.strip()


def _tokenize(text):
    # sentence wise
    
    words = text.split(" ")
    new_words = []
    for w, word in enumerate(words):
        word = _remove_chars(word)
        if word not in stop_words:
            new_words.append(word)

    return new_words


def tokenize(data):
    # list of sentence wise
    for i in range(len(data)):
        data[i] = _tokenize(data[i])    
    return data


def remove_chars(data):
    for i in range(len(data)):
        data[i] = _remove_chars(data[i])
    return data


def preprocess_raw_text(text):
    text = _remove_chars(text)
    text = _tokenize(text)
    return text


def preprocess_label(label_):
    for i in range(len(label_)):
        label_[i] = preprocess_raw_text(label_[i])
    return label_


def make_vocab(labels, max_limit = 1000):
    vocab = {}
    idx = 1 # starts from 1 because 0 will be used for padding
    for indi_label in labels:
        for sentence in indi_label:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
                    if idx == max_limit:
                        vocab["unk"] = idx
                        return vocab

    vocab["unk"] = idx

    return vocab


def _inverse_vocab(vocab):
    return {v: k for k, v in vocab.items()}


def pad(text, padding_length = PADDING_LENGTH):
    l = len(text)
    for i in range(l, padding_length):
        text.append(0)

    # print(len(text), text)    
    return text

def _numericalize(tokenized_text):
    # expects a tokenized list of a sentence -> ["quick", "brown", "fox"]
    transformed_text = []
    for word in tokenized_text:
        if word not in vocab:
            word = "unk"
        transformed_text.append(vocab[word])
    
    if len(transformed_text) < PADDING_LENGTH:
        transformed_text = pad(transformed_text)
    return transformed_text


def text2int(raw_text):
    raw_text = preprocess_raw_text(raw_text)
    # print(raw_text)
    # print(vocab)
    return pad(_numericalize(raw_text))

def int2text(ids):
    op_string = ""
    for i in range(len(ids)):
        if ids[i] != 0:
            # ids[i] = inverse_vocab[ids[i]]
            # print(inverse_vocab[ids[i]])
            op_string += inverse_vocab[ids[i]] + " "
    return op_string


def create_vocab_inv_labels(labels_path):
    # labels_path = "/mnt/d/work/datasets/img_text/labels/"
    labels = os.listdir(labels_path)
    for l in range(len(labels)):
        labels[l] = read_txt(os.path.join(labels_path, labels[l]))
        labels[l] = preprocess_label(labels[l])
        
    global vocab
    global inverse_vocab

    vocab = make_vocab(labels)
    for l in range(len(labels)):
        for t in range(len(labels[l])):
            # print(tok_text)
            labels[l][t] = _numericalize(labels[l][t])

    inverse_vocab = _inverse_vocab(vocab)

    return labels, vocab, inverse_vocab
