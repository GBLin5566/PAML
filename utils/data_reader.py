import os
import pickle
import random
from random import randint
import torch.utils.data as data
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from utils import config
import pprint
pp = pprint.PrettyPrinter(indent=1)


def DistJaccard(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return float(len(str1 & str2)) / len(str1 | str2)


def dist_matrix(array_str):
    matrix = []
    for s_r in array_str:
        row = []
        for s_c in array_str:
            row.append(DistJaccard(s_r, s_c))
        matrix.append(row)
    mat_ = np.array(matrix)
    print("Mean", np.mean(mat_))
    print("Var", np.var(mat_))
    return matrix


def plot_mat(mat):
    ax = sns.heatmap(mat, cmap="YlGnBu")
    # g = sns.clustermap(mat,cmap="YlGnBu", figsize=(8,8))
    plt.show()


def create_str_array(data):
    arr = []
    for _, v in data.items():
        arr.append(" ".join(v[0][0]))
    return arr


def show_example(mat_jac, arr, a, b):
    print("Example with {}<= VAL < {}".format(a, b))
    for i in range(len(mat_jac)):
        for j in range(len(mat_jac)):
            if(i > j and float(mat_jac[i][j]) >= a and float(mat_jac[i][j]) < b):
                print("Dial 1\n", arr[i])
                print("Dial 2\n", arr[j])
                return


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            config.UNK_idx: "UNK",
            config.PAD_idx: "PAD",
            config.EOS_idx: "EOS",
            config.SOS_idx: "SOS"}
        self.n_words = len(self.index2word)

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def transform(self, inputs, is_list_of_str):
        if is_list_of_str:
            sequence = [
                self.word2index[word]
                if word in self.word2index else config.UNK_idx
                for word in inputs.split(' ')] + [config.EOS_idx]
        else:
            sequence = [
                self.word2index[word]
                if word in self.word2index else config.UNK_idx
                for word in ' '.join(inputs).split(' ')]
        return torch.LongTensor(sequence)


class BertLang:
    def __init__(self):
        self.tokenizer = config._tokenizer

    def transform(self, inputs, is_list_of_str):
        if not is_list_of_str:
            inputs = ' '.join(inputs)
        return torch.LongTensor(self.tokenizer.encode(inputs, max_length=500))


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.src = []
        self.cands = []
        self.trg = []
        self.persona = []
        self.max_len_sent = 0
        self.max_len_words = 0
        self.max_len_answer = 0
        for d in data:
            if(len(d[0]) > self.max_len_sent):
                self.max_len_sent = len(d[0])
            for e in d[0]:
                if(len(e.split(' ')) > self.max_len_words):
                    self.max_len_words = len(e.split(' '))
            for e in d[1]:
                if(len(e.split(' ')) > self.max_len_words):
                    self.max_len_words = len(e.split(' '))
            if(len(d[1][d[2]].split(' ')) > self.max_len_answer):
                self.max_len_answer = len(d[1][d[2]].split(' '))

            self.src.append(d[0])
            self.cands.append(d[1])
            self.trg.append(d[1][d[2]])
            self.persona.append(d[3])
        self.vocab = vocab
        self.num_total_seqs = len(data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["input_txt"] = self.src[index]
        item["target_txt"] = self.trg[index]
        item["cand_txt"] = self.cands[index]
        item["cand_index"] = []
        for c in self.cands[index]:
            item["cand_index"].append(self.preprocess(c, is_list_of_str=True))
        item["persona_txt"] = self.persona[index]

        item["input_batch"] = self.preprocess(self.src[index])
        item["target_batch"] = self.preprocess(
            self.trg[index], is_list_of_str=True)
        if config.pointer_gen:
            item["input_ext_vocab_batch"], item["article_oovs"] = \
                self.process_input(item["input_txt"])
            item["target_ext_vocab_batch"] = self.process_target(
                item["target_txt"], item["article_oovs"])
        return item

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, inputs, is_list_of_str=False):
        """Converts words to ids."""
        return self.vocab.transform(inputs, is_list_of_str)

    def process_input(self, input_txt):
        seq = []
        oovs = []
        seq = torch.LongTensor(self.vocab.transform(input_txt, is_list_of_str=True))
        return seq, oovs

    def process_target(self, target_txt, oovs):
        seq = []
        for word in target_txt.strip().split():
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            elif word in oovs:
                seq.append(self.vocab.n_words + oovs.index(word))
            else:
                seq.append(config.UNK_idx)
        seq.append(config.EOS_idx)
        seq = torch.LongTensor(seq)
        return seq


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long() + \
            config.PAD_idx
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(
        key=lambda x: len(
            x["input_batch"]),
        reverse=True)  # sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    input_batch, input_lengths = merge(item_info['input_batch'])
    target_batch, target_lengths = merge(item_info['target_batch'])

    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    d = {}
    d["input_batch"] = input_batch
    d["target_batch"] = target_batch
    d["input_lengths"] = input_lengths
    d["target_lengths"] = target_lengths
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]
    d["cand_txt"] = item_info["cand_txt"]
    d["cand_index"] = item_info["cand_index"]
    d["persona_txt"] = item_info["persona_txt"]

    if 'input_ext_vocab_batch' in item_info:
        input_ext_vocab_batch, _ = merge(item_info['input_ext_vocab_batch'])
        target_ext_vocab_batch, _ = merge(item_info['target_ext_vocab_batch'])
        input_ext_vocab_batch = input_ext_vocab_batch.transpose(0, 1)
        target_ext_vocab_batch = target_ext_vocab_batch.transpose(0, 1)
        if config.USE_CUDA:
            input_ext_vocab_batch = input_ext_vocab_batch.cuda()
            target_ext_vocab_batch = target_ext_vocab_batch.cuda()
        d["input_ext_vocab_batch"] = input_ext_vocab_batch
        d["target_ext_vocab_batch"] = target_ext_vocab_batch
        if "article_oovs" in item_info:
            d["article_oovs"] = item_info["article_oovs"]
            d["max_art_oovs"] = max(len(art_oovs)
                                    for art_oovs in item_info["article_oovs"])
    return d


def read_langs(file_name, cand_list, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    # Read the file and split into lines
    persona = []
    dial = []
    lock = 0
    index_dial = 0
    data = {}
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if(int(nid) == 1 and lock == 1):
                if(str(sorted(persona)) in data):
                    data[str(sorted(persona))].append(dial)
                else:
                    data[str(sorted(persona))] = [dial]
                persona = []
                dial = []
                lock = 0
                index_dial = 0
            lock = 1
            if '\t' in line:
                # utterance line
                u, r, _, cand = line.split('\t')
                cand = cand.split('|')
                for c in cand:
                    if c in cand_list:
                        pass
                    else:
                        cand_list[c] = 1
                dial.append({"nid": index_dial, "u": u, "r": r, 'cand': cand})
                index_dial += 1
            else:
                # persona line
                r = line.split(":")[1][1:-1]
                persona.append(str(r))
    return data


def filter_data(data, cut):
    print("Full data:", len(data))
    newdata = {}
    cnt = 0
    for k, v in data.items():
        # print("PERSONA",k)
        # print(pp.pprint(v))
        if(len(v) > cut):
            cnt += 1
            newdata[k] = v
        # break
    print("Min {} dialog:".format(cut), cnt)
    return newdata


def cluster_persona(data, split):
    if split not in ['train', 'valid', 'test']:
        raise ValueError(
            "Invalid split, please choose one from train, valid, test")
    filename = 'data/ConvAI2/' + split + '_persona_map'
    with open(filename, 'rb') as f:
        persona_map = pickle.load(f)
    # persona_map = {persona_index:[similar personas list], }
    newdata = {}
    for k, v in data.items():
        p = eval(k)
        persona_index = 0
        for p_index, p_set in persona_map.items():
            if p in p_set:
                persona_index = p_index
        if persona_index in newdata:
            for dial in v.values():
                newdata[persona_index][len(newdata[persona_index])] = dial

        else:
            newdata[persona_index] = v
    return newdata


def preprocess(data):
    newdata = {}
    cnt_ptr = 0
    cnt_voc = 0
    for k, v in data.items():
        # string of list of string -> list of string
        p = eval(k)
        new_v = {i: [] for i in range(len(v))}
        for d_index, dial in enumerate(v):
            if(config.persona):
                context = list(p)
            else:
                context = []
            for turn in dial:
                context.append(turn["u"])
                for i, c in enumerate(turn['cand']):
                    if(turn["r"] == c):
                        answer = i

                new_v[d_index].append(
                    [list(context), turn['cand'], answer, eval(k)])

                # print(sum(context,[]).split(" "))
                # compute stats
                for key in turn["r"].split(" "):
                    index = [
                        loc for loc, val in enumerate(
                            " ".join(context).split(" ")) if (
                            val == key)]
                    if (index):
                        cnt_ptr += 1
                    else:
                        cnt_voc += 1
                context.append(turn["r"])
        newdata[k] = new_v
    print("Pointer percentace= {} ".format(cnt_ptr / (cnt_ptr + cnt_voc)))
    return newdata


def prepare_data_seq():
    file_paths = {
        'train': 'data/ConvAI2/train_self_original.txt',
        'valid': 'data/ConvAI2/valid_self_original.txt',
        'test': 'data/ConvAI2/test_self_original.txt',
    }
    cand = {}
    vocab = BertLang()
    data = [
        filter_data(
            cluster_persona(
                preprocess(
                    read_langs(path, cand_list=cand, max_line=None),
                    ),
                desp),
            cut=1)
        for desp, path in file_paths.items()
    ]
    data += [vocab]

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(config.save_path + 'dataset.p', "wb") as f:
        pickle.dump(data, f)
        print("Saved PICKLE")

    return data


def get_persona(data):
    per = []
    for k, _ in data.items():
        per.append(k)
    return per


class Personas:
    def __init__(self):
        random.seed(999)
        if(os.path.exists(config.save_path_dataset + 'dataset.p')):
            with open(config.save_path_dataset + 'dataset.p', "rb") as f:
                [self.meta_train, self.meta_valid,
                    self.meta_test, self.vocab] = pickle.load(f)
            self.type = {'train': self.meta_train,
                         'valid': self.meta_valid,
                         'test': self.meta_test}
            print("DATASET LOADED FROM PICKLE")
        else:
            self.meta_train, self.meta_valid, self.meta_test, self.vocab = \
                prepare_data_seq()
            self.type = {'train': self.meta_train,
                         'valid': self.meta_valid,
                         'test': self.meta_test}

    def get_len_dataset(self, split):
        return len(self.type[split])

    def get_personas(self, split="test"):
        persona = get_persona(self.type[split])  # array with personas
        if(split == "test" or split == "valid"):
            persona = [p for p in persona if len(self.type[split][p]) > 1]
        return persona

    def get_task(self, split):
        '''
        Return a random persona from a give set (split in [train,valid,test])
        '''
        persona = get_persona(self.type[split])  # array with personas
        t = randint(0, len(persona) - 1)
        while(len(self.type[split][persona[t]]) < 1):
            t = randint(0, len(persona) - 1)
        return persona[t]

    def get_num_of_dialog(
        self, persona, split): return len(
        self.type[split][persona])

    def get_balanced_loader(
            self,
            persona,
            batch_size,
            split,
            fold=-1,
            dial_num=1):
        dial_persona = self.type[split][persona]
        if len(dial_persona) == 1:
            raise ValueError("persona have less than two dialogs")
        tr = []
        val = []
        if (split == "train" or split == "valid"):
            val_dial = 0
            tr_dial = 0
            while val_dial == tr_dial:
                val_dial = randint(0, len(dial_persona) - 1)
                tr_dial = randint(0, len(dial_persona) - 1)
            for p in dial_persona[val_dial]:
                val.append(p)
            for p in dial_persona[tr_dial]:
                tr.append(p)
        elif(fold != -1 and (split == "test")):
            val_dial = fold
        else:
            val_dial = len(dial_persona) - 1
        if (split == "test"):
            for i in dial_persona:
                if(i == val_dial):
                    for p in dial_persona[i]:
                        val.append(p)
                else:
                    if dial_num == 0:
                        continue
                    for p in dial_persona[i]:
                        tr.append(p)
                    dial_num -= 1

        dataset_train = Dataset(tr, self.vocab)
        data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     collate_fn=collate_fn)

        dataset_valid = Dataset(val, self.vocab)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      collate_fn=collate_fn)

        return data_loader_tr, data_loader_val

    def get_data_loader(self, persona, batch_size, split, fold=-1):
        dial_persona = self.type[split][persona]
        if(len(dial_persona) == 1 and split == "train"):
            tr = []
            val = []
            for i in dial_persona:
                for p in dial_persona[i]:
                    val.append(p)
                    tr.append(p)
        else:
            tr = []
            val = []
            if (split == "train"):
                val_dial = randint(0, len(dial_persona) - 1)
            elif(fold != -1 and (split == "test" or split == "valid")):
                val_dial = fold
            else:
                val_dial = len(dial_persona) - 1
            for i in dial_persona:
                if(i == val_dial):
                    for p in dial_persona[i]:
                        val.append(p)
                else:
                    for p in dial_persona[i]:
                        tr.append(p)

        dataset_train = Dataset(tr, self.vocab)
        data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     collate_fn=collate_fn)

        dataset_valid = Dataset(val, self.vocab)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      collate_fn=collate_fn)

        return data_loader_tr, data_loader_val

    def get_all_data(self, batch_size):
        tr = []
        val = []
        test = []
        for persona in self.meta_train:
            for i in range(len(self.meta_train[persona])):
                for p in self.meta_train[persona][i]:
                    tr.append(p)

        for persona in self.meta_valid:
            for i in range(len(self.meta_valid[persona])):
                for p in self.meta_valid[persona][i]:
                    val.append(p)

        for persona in self.meta_test:
            for i in range(len(self.meta_test[persona])):
                for p in self.meta_test[persona][i]:
                    test.append(p)

        dataset_train = Dataset(tr, self.vocab)
        data_loader_tr = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn)

        dataset_val = Dataset(val, self.vocab)
        data_loader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn)

        dataset_test = Dataset(test, self.vocab)
        data_loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn)
        return data_loader_tr, data_loader_val, data_loader_test


if __name__ == '__main__':
    train, valid, test, vocab = prepare_data_seq()
    dial_num = []
    for k, v in train.items():
        dial_num.append(len(v))
    print(sum(dial_num) / len(dial_num))
    plt.hist(dial_num, color='red', bins=25, edgecolor='black', alpha=0.7)
    plt.title('Histogram of number of dialog for each persona')
    plt.xlabel('Number of dialogue')
    plt.ylabel('Number of persona')
    # plt.show()
    plt.savefig("hist.pdf")
