import spacy
import textacy

nlp = spacy.load('en')
import re
import json
from pprint import pprint


def match_id_pattern(text):
    pattern = 'id: GO:[0-9]*$'
    m = re.search(pattern, text)
    if m is not None:
        return True
    else:
        return False


def match_def_pattern(text):
    pattern = '^def:*'
    m = re.search(pattern, text)
    if m is not None:
        return True
    else:
        return False


res_dict = {}
key = None
val = None
with open('go.obo', 'r') as f:
    l = f.readline()
    while l:
        l = l.strip('\n')
        txt = l.strip()
        if match_id_pattern(txt):
            key = txt
        if match_def_pattern(txt):
            val = txt
            res_dict[key] = val
        l = f.readline()

temp_json = 'temp_data.json'
with open(temp_json, 'w') as file:
    file.write(json.dumps(res_dict))
with open(temp_json) as tmp_file:
    data_dict = json.loads(tmp_file.read())

stop_words = ['OBSOLETE', 'def', '"']


def process_key(key_txt):
    res = key_txt.split(':')[-1]
    return res


def process_val(val_txt):
    global stop_words
    for s in stop_words:
        try:
            val_txt = val_txt.replace(s, '')
        except:
            pass
    return val_txt


def main_1():
    data_dict_1 = {}
    for k, v in data_dict.items():
        k = process_key(k)
        v = process_val(v)
        data_dict_1[k] = v

    temp_json_1 = 'temp_data_1.json'
    with open(temp_json_1, 'w') as file:
        file.write(json.dumps(data_dict_1))


def tokenize_text(txt):

    txt = textacy.preprocess.remove_punct(txt, marks=';,:[]()-+.=<>')
    txt = textacy.preprocess.replace_urls(txt, replace_with=' ')
    txt = textacy.preprocess.replace_numbers(txt, replace_with = ' ')
    txt = textacy.preprocess.replace_currency_symbols(txt, replace_with=None)

    # res = []
    # doc = nlp(txt)
    # tokens = [t for t in doc]
    # for i in range(len(tokens)):
    #     # Lemmatization
    #     t = str(tokens[i].lemma_)
    #     res.append(t)
    # txt = ' '.join(res)

    res = []
    txt = textacy.preprocess.normalize_whitespace(txt)
    # print (txt)
    doc = textacy.Doc(txt, lang='en')
    for s in textacy.extract.words(doc, exclude_pos=None, min_freq=1):
        if len(str(s)) > 2 :
            res.append(str(s).lower())
    return res


def main_2():
    temp_json_1 = 'temp_data_1.json'
    with open(temp_json_1) as tmp_file:
        data_dict_2 = json.loads(tmp_file.read())
    for k, v in data_dict_2.items():
        v = tokenize_text(v)
        data_dict_2[k] = v
    temp_json_2 = 'temp_data_2.json'
    with open(temp_json_2, 'w') as file:
        file.write(json.dumps(data_dict_2))

# -----------#

main_1()
main_2()