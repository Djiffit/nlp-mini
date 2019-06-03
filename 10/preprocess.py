import urllib.request as request
import argparse
import spacy
import ujson as json
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
from zipfile import ZipFile
from spacy.lang.en import English

def get_args():
    parser = argparse.ArgumentParser('SQuAD dataset pre-process pipeline')
    parser.add_argument('--glove_url', type=str, default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--train_path', type=str, default='./data/train-v2.0.json')
    parser.add_argument('--test_path', type=str, default='./data/test-v2.0.json')
    parser.add_argument('--dev_path', type=str, default='./data/dev-v2.0.json')
    return parser.parse_args()

def create_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])

class DownloadBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_data_from_url(url, out):
    with DownloadBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as bar:
        request.urlretrieve(url, filename=out, reporthook=bar.update_to)

def download(args):
    download_list = [
        args.glove_url,
    ]

    for url in download_list:
        out = create_data_path(url)

        if not os.path.isfile(out):
            print(f'Download {url}...')
            download_data_from_url(url, out)

        if os.path.exists(out) and out.split('.')[-1] == 'zip':
            extract = out.replace('.zip', '')
            if not os.path.exists(extract):
                print(f'Extracting the zipfile {url} ...')
                with ZipFile(out, 'r') as myzip:
                    myzip.extractall(extract)

def tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

def process_text(text):
    return text.replace("''", '" ').replace("``", '" ')

def read_file(filename, file_type, w_count, c_count):
    print(f'Process {file_type} entries')

    with open(filename, 'r') as file:
        src = json.load(file)
        total = 0
        data = []
        data_eval = {}
        for text in tqdm(src['data']):
            for p in text['paragraphs']:
                ctx = process_text(p['context'].replace("''", '" '))
                ctx_tkns = tokenizer(ctx)
                ctx_chars = [list(token) for token in ctx_tkns]

                pos, word_spans = 0, []

                for token in ctx_tkns:
                    # Can't naively expect position
                    pos = ctx.find(token, pos)
                    word_spans.append((pos, pos + len(token)))
                    pos += len(token)

                    w_count[token] += len(p['qas'])

                    for c in token:
                        c_count[c] += len(p['qas'])

                    for qa in p['qas']:
                        question = process_text(qa['question'])
                        q_tokens = tokenizer(question)
                        q_chars = [list(token) for token in q_tokens]
                        for token in q_tokens:
                            w_count[token] += 1

                            for c in token:
                                c_count[c] += 1

                        ans_starts, ans_ends, ans = [], [], []

                        for answer in qa['answers']:
                            ans_text = answer['text']
                            ans_start = answer['answer_start']
                            ans_end = ans_start + len(ans_text)
                            ans.append(ans_text)
                            ans_span = []
                            for i, span in enumerate(word_spans):
                                if not (ans_end <= span[0]) or (ans_start >= span[1]):
                                    ans_span.append(i)

                            ans_starts += [ans_span[0]]
                            ans_ends += [ans_span[-1]]

                        data += [{
                            'ctx_tokens': ctx_tkns,
                            'ctx_chars': ctx_chars,
                            'q_tokens': q_tokens,
                            'q_chars': q_chars,
                            'ans_starts': ans_starts,
                            'ans_ends': ans_ends,
                            'id': total
                        }]

                        data_eval[str(total)] = {
                            'ctx': ctx,
                            'question': question,
                            'spans': word_spans,
                            'answers': ans,
                            'squad-id': qa['id']
                        }

                        total += 1
        print(f'Found {len(data)} questions!')
    return data, data_eval

def get_embedding(counter, vec_size, embed_file=None):
    print('Processing word vectors')
    embeddings = {}

    if embed_file:
        with open(embed_file, 'r', encoding='utf-8') as embed:
            for row in tqdm(embed, total=2196017):
                split = row.split()
                word = ''.join(split[0:-vec_size])
                word_vec = [float(dim) for dim in split[-vec_size:]]

                # Filter words > count ?
                if word in counter:
                    embeddings[word] = word_vec
    else:
        # Initialize character vectors randomly
        for char in counter:
            embeddings[char] = [np.random.normal(scale=0.1) for i in range(vec_size)]

    null = '~~NULL~~'
    oov = '~~OOV~~'
    word2index = {token: i for i, token in enumerate(embeddings.keys(), 2)}
    word2index[null] = 1
    word2index[oov] = 0
    embeddings[null] = [0.] * vec_size
    embeddings[oov] = [0.] * vec_size
    index2embedding = {index: embeddings[token] for token, index in word2index.items()}
    embedding_matrix = [index2embedding[i] for i in range(len(index2embedding))]

    return embedding_matrix, word2index



def save(filename, obj):
    with open(filename, "w") as fh:
        json.dump(obj, fh)

def process(args):
    w_count, c_count = Counter(), Counter()
    dev_examples, dev_eval = read_file(args.dev_path, 'dev', w_count, c_count)
    word_embed_matrix, word2idx = get_embedding(w_count, 300, './data/glove.840B.300d/glove.840B.300d.txt')
    char_embed_matrix, char2idx = get_embedding(c_count, 100)

    save('./data/word_embed_mat.json', word_embed_matrix)
    save('./data/char_embed_mat.json', char_embed_matrix)
    save('./data/word2idx.json', word2idx)
    save('./data/char2idx.json', char2idx)
    save('./data/dev_data.json', dev_examples)
    save('./data/dev_evaluate.json', dev_eval)

arguments = get_args()
download(arguments)
nlp = English()
process(arguments)
