import sys, pickle, os, random
import numpy as np
import re

## tags, BIO
tag2label = {"O": 0,
             "Disease": 1, "Reason": 2,
             "Symptom": 3, "Test": 4,
             "Test_Value": 5, "I-ORG": 6,
             "Drug": 7, "Frequency": 8,
             "Amount": 9, "Method": 10,
             "Treatment": 11, "Operation": 12,
             "SideEff": 13, "Anatomy": 14,
             "Level": 15, "Duration": 16,
             }


def file_name(file_dir):
    #获取文件夹下所有文件名并拼接为文件路径返回
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def read_corpus(txt_path):
    #读取文件内容及标签
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    numbers = ['1','2','3','4','5','6','7','8','9','0']
    #获取文件目录下所有文件并读取为sentence和tags
    #标签数据处理有点问题
    #未处理重叠实体的问题解决
    f_names = file_name(txt_path)
    nums = 0
    tag_ = []
    sent_ = []
    for name in f_names:
        if nums % 2 == 0:
            tag_ = []
            sent_ = []
        if name[-3:] == 'pkl':
            break
        if name[-3:] == 'txt':
            path = name
            with open(path, encoding='utf-8') as tr:
                sent = tr.read()
                sent_ = sent
        else:
            path = name
            with open(path, encoding='utf-8') as lr:
                tag = []
                tags = lr.read()
                tags = tags.split('\n')
                for tag in tags:
                    tag = re.split(' |\t', tag)
                    if len(tag) >= 5:
                        tag_.append(tag)


    #for line in lines:
    #    if line != '\n':
    #        [char, label] = line.strip().split()
    #        sent_.append(char)
    #        tag_.append(label)
    #    else:
    #        data.append((sent_, tag_))
    #        sent_, tag_ = [], []
        if nums % 2 != 0:
            label_ = []
            dictions = {}
            for one_of_tag in tag_:
                x = 3
                for x in range(3, len(one_of_tag)):
                    if ';' in one_of_tag[x]:
                        continue
                    else:
                        break
                for i in range(int(one_of_tag[2]), int(one_of_tag[x])):
                    dictions[i] = one_of_tag[1]

            for i in range(len(sent_)):
                if i in dictions.keys():
                    label_.append(dictions[i])
                else:
                    label_.append('O')

            #此处之前搞错了，将tag加入了进去
            for i in range(int(len(sent_)/300)):
                if (i + 1) * 300 < len(sent_):
                    data.append((sent_[i*300:(i+1)*300], label_[i*300:(i+1)*300]))
                else:
                    data.append((sent_[i * 300:], label_[i * 300:]))
        nums += 1
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    #建立字典
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sents, tag_ in data:
        for sent_ in sents:
            for word in sent_:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
    low_freq_words = []
    #删除低频率的词，加入到低频率词典中
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]
    #对词典重新进行编号
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    #输出字典长度，并写入到字典路径下
    print(len(word2id))
    with open(vocab_path + '\\word2id.pkl', 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    #句子变为id
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    #读取字典
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    #随机编码字向量
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    #将序列进行补齐，补齐为长度一致的序列,pad_mark原来是0
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    #max_len = max(map(lambda x : len(x), sequences))
    max_len = 300
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
        #seq_len_list.append(len(seq))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    #返回句子与标签
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    #建立句子和对应的label列表
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)

        for i in range(len(tag_)):
            #如果不是数字，则用tag2label转化
            if not isinstance(tag_[i], int) and tag_[i] in tag2label.keys():
                tag_[i] = tag2label[tag_[i]]

        label_ = tag_

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

