import os


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
    chars = "!@#$%^&*(),.;:'[]{}<?>"
    for i, t in enumerate(text):
        if t in chars:
            text = text.replace(t, "")
    # print(f"here {t}, {text}")
    return text.strip()


def _tokenize(text):
    # sentence wise
    stop_words = ["a", "is", "the", "etc", "its", "be", "to", "in", "of", "i", "with", "as", "but", "or", "an", "my", "so"]
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