import json
import itertools
import re
import copy
import math
from collections import OrderedDict
import numpy as np
from numpy import mean
import bisect
# from prepro.utils import _get_word_ngrams
import os
import random
from multiprocess import Pool
import argparse


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set



def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def rouge_sentence(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # abstract = abstract_sent_list
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    
    
    sent_len = len(doc_sent_list)
    #calculate rouge score
    c = list(range(len(doc_sent_list)))
    candidates_1 = [evaluated_1grams[idx] for idx in c]
    candidates_1 = set.union(*map(set, candidates_1))
    candidates_2 = [evaluated_2grams[idx] for idx in c]
    candidates_2 = set.union(*map(set, candidates_2))
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
    rouge_score = rouge_1 + rouge_2
    return rouge_score

def shapley_value(unique_keys, all_coalitions, source_data): 
    s_shapley_value = []
    n = len(unique_keys)
    for s_id in unique_keys:
        s_coalition = []
        no_s_coalition = []
        

        for c in copy.deepcopy(all_coalitions):
            if s_id in c:
                s_coalition.append((copy.deepcopy(c),source_data['#'.join([str(i) for i in c])]))
                c.remove(s_id)
                if len(c)==0:
                    no_s_coalition.append((c,0))
                else:
                    no_s_coalition.append((c,source_data['#'.join([str(i) for i in c])]))

        # print('s的联盟:', s_coalition)
        # print('剔除s后的联盟', no_s_coalition)


        shapley_value=0
        for i in range(len(s_coalition)):
            s = len(s_coalition[i][0])

            s_payoff = s_coalition[i][1] - no_s_coalition[i][1]

            s_weight = math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
            shapley_value += s_payoff * s_weight

        s_shapley_value.append((s_id, shapley_value))
    return s_shapley_value

def sample_discrete(vec):
    u = np.random.rand()
    start = 0
    for i, num in enumerate(vec):      
        if u > start:
            start += num
        else:
            return i-1
    return i


def monte_carlo_sampling(seq, n, p = None, T=1000000):
    """
    在离散序列上进行蒙特卡洛采样
    
    Args:
        S (list): 离散序列
        p (function): 离散序列的概率分布函数
        n (int): 采样数量
        N (int, optional): 采样次数，默认值为 1000000
        
    Returns:
        list: 采样结果的列表
    """
    if len(seq) < n:
        raise ValueError("总样本数不足")
    
    if p is None:
        p = lambda x: 1/len(seq)

    # F = [sum(p(S[:j+1])) for j in range(len(S))]
    F = [sum(p(i) for i in seq[:j+1]) for j in range(len(seq))]
    samples = []
    count = 0
    while count < T and len(samples) < n:
        u = random.random()
        for j in range(len(seq)):
            if F[j] >= u:
                samples.append(seq[j])
                break
        count += 1
    
    if len(samples) < n:
        raise ValueError("无法采样到足够的样本")

    return samples

def calulate_shapley_with_approximate_bakup(datas, save_path, times = 10, summary_sent_count = 9, in_max_len = False):
    # sample_times = 100
    all_data = []
    for index, data in enumerate(datas):
        clss = data['clss']
        max_sent_ids = bisect.bisect_left(clss, 512)
        labels = data['src_sent_labels']
        if in_max_len:
            src = data['src_txt'][:max_sent_ids]
        else:
            src = data['src_txt']
        tgt_txt = data['tgt_txt']
        tgt_sent = tgt_txt.split('<q>')

        abs_sent_list = [sent.split() for sent in tgt_sent]
        doc_sent_list = [sent.split() for sent in src]

        labels_id = [id for id,value in enumerate(labels) if value == 1]
        unique_keys = list(range(len(src)))
        n = len(unique_keys)
        p = [1/n] * n
        if n <= 10: #句子长度小于10，全采样
            sample_times = 1
            sample_len = n
        else: #否则设置采样的数量和采样次数
            sample_times = times
            # sample_len = 20
            sample_len = 10
        # print(sample_times)
        shapley_value_sampel_times = OrderedDict() #句子shaoley value的dict的初始化
        for sent_id in range(n):
            # shapley_value_sampel_times[sent_id] = [0.0] #防止出现nan的情况
            shapley_value_sampel_times[sent_id] = [] #防止出现nan的情况
            # shapley_value_sampel_times[sent_id] = [] #防止出现nan的情况
        # print(shapley_value_sampel_times)
        for _ in range(sample_times):
            # unique_keys  = np.sort(random.sample(unique_keys, sample_len)).tolist()
            # unique_keys = np.sort(np.random.choice(unique_keys, sample_len, replace=False))
            
            unique_keys = [] #蒙特卡洛采样,不会出现采样不到的现象
            if sample_times == 1:
                unique_keys = list(range(len(src)))
            else: 
                for _ in range(sample_len):#从总的句子数量均匀采样出sample_len个句子，然后采样samtimes 次数
                    unique_keys.append(sample_discrete(p)) 

            # all_coalitions=[list(j) for i in range(n) for j in itertools.combinations(unique_keys, r=i+1)]
            all_coalitions=[list(j) for i in range(len(labels_id)) for j in itertools.combinations(unique_keys, r=i+1)] ##times20_sentcount_maxlen

            source_data = {} 
            for unique_key in all_coalitions:
                sent_key = [doc_sent_list[id] for id in unique_key]
                rouge_score = rouge_sentence(sent_key, abs_sent_list)
                key = '#'.join([str(id) for id in unique_key])
                source_data[key] = rouge_score
        
            s_shapley_value = shapley_value(unique_keys, all_coalitions, source_data)#[sid, shapley_value]
            
            for sent_id, shapley_score  in s_shapley_value:
                shapley_value_sampel_times[sent_id].append(shapley_score)
            # print(shapley_value_sampel_times)
        # print('******************')
        # print(shapley_value_sampel_times)
        avg_s_shapley_value = [(sent_id, mean(score)) if len(score) > 0 else (sent_id, 0.0) for sent_id, score in shapley_value_sampel_times.items()]
        sent_shapley_value = [score for _, score in avg_s_shapley_value]
        # print('**************************')
        # print(avg_s_shapley_value)
        # print(sent_shapley_value)
        avg_s_shapley_value.sort(key = lambda x: x[1], reverse= True)

        shapley_label_id = [sent_id for sent_id, value in avg_s_shapley_value[:len(labels_id)] if value > 0.0] #shapley
        shapley_label_id_1 = [sent_id for sent_id, value in avg_s_shapley_value[: summary_sent_count] if value > 0.0] #shapley
        shapley_label = [0] * len(data['src_txt'])
        shapley_label_1 = [0]* len(data['src_txt'])
        for l in shapley_label_id:
            shapley_label[l] = 1

        for l in shapley_label_id_1:
            shapley_label_1[l] = 1

        
        if in_max_len:
            total_sentence_shapley_value = [0.0] * len(data['src_txt'])
            total_sentence_shapley_value[: max_sent_ids] = sent_shapley_value
            data['sent_shapley_value'] = total_sentence_shapley_value
        else:
            data['sent_shapley_value'] = sent_shapley_value

        data['shapley_label'] = shapley_label
        data['shapley_label_1'] = shapley_label_1
        
        all_data.append(data)
        print(index)

    
    with open(save_path,'w') as save:
        json.dump(all_data, save) 
        save.close() 

    datas_shapley = json.load(open(save_path))

    print('orgingal datas length:' + str(len(datas)))
    print('shapley datas length:' + str(len(datas_shapley)))
    

def integrate(base_path, count, save_path):
    total_datas = []
    for i in range(count):
        datas = json.load(open(os.path.join(base_path, 'train_'+ str(i+1) + '.json' )))
        total_datas.extend(datas)
    print('total datas'+ str(len(total_datas)))

    with open(save_path, 'w') as save:
        json.dump(total_datas, save)
    save.close()




def calulate_shapley_with_approximate(datas, save_path, max_sent_count = 10, times = 10, summary_sent_count = 2, in_max_len = False):
    all_data = []
    for index, data in enumerate(datas):
        clss = data['clss']
        max_sent_ids = bisect.bisect_left(clss, 512)
        sent_rouge_scores = data['sent_rouge_score']
        
        if in_max_len:
            src = data['src_txt'][:max_sent_ids]
            labels = data['src_sent_labels'][:max_sent_ids]
        else:
            src = data['src_txt']
            labels = data['src_sent_labels']
        tgt_txt = data['tgt_txt']
        tgt_sent = tgt_txt.split('<q>')

        print('************')
        print(len(src))

        abs_sent_list = [sent.split() for sent in tgt_sent]
        doc_sent_list = [sent.split() for sent in src]

        labels_id = [id for id, value in enumerate(labels) if value == 1]
        src_sent_ids = [id for id in range(len(src)) if sent_rouge_scores[id] >0] #筛出sent_rouge = 0的句子
        n = len(src_sent_ids)
        if n <= max_sent_count: #句子数量小于10，全采样
            sample_times = 1
            sample_len = n
        else: #否则设置采样的数量和采样次数
            sample_times = times
            sample_len = max_sent_count
        shapley_value_sampel_times = OrderedDict() #句子shapley value的dict的初始化

        # for sent_id in src_sent_ids: # intialize the dict
        #     shapley_value_sampel_times[sent_id] = [] #防止出现nan的情况

        for sent_id in range(len(data['src_txt'])): # intialize the dict,覆盖所有的源文档句子
            shapley_value_sampel_times[sent_id] = []


        for _ in range(sample_times):    
            unique_keys = [] #蒙特卡洛采样,不会出现采样不到的现象

            if sample_times == 1:
                unique_keys = src_sent_ids
            else: 
                sample_unique_keys = monte_carlo_sampling(src_sent_ids, sample_len)
                unique_keys = list(set(sample_unique_keys)) # remove the duplicate

            all_coalitions=[list(j) for i in range(len(unique_keys)) for j in itertools.combinations(unique_keys, r=i+1)]
            # all_coalitions=[list(j) for i in range(len(labels_id)) for j in itertools.combinations(unique_keys, r=i+1)]

            source_data = {} 
            for unique_key in all_coalitions:
                sent_key = [doc_sent_list[id] for id in unique_key]
                rouge_score = rouge_sentence(sent_key, abs_sent_list)
                key = '#'.join([str(id) for id in unique_key])
                source_data[key] = rouge_score
        
            s_shapley_value = shapley_value(unique_keys, all_coalitions, source_data)#[sid, shapley_value]
            
            for sent_id, shapley_score  in s_shapley_value:
                shapley_value_sampel_times[sent_id].append(shapley_score)
            
        avg_s_shapley_value = [(sent_id, mean(score)) if len(score) > 0 else (sent_id, 0.0) for sent_id, score in shapley_value_sampel_times.items()]
        sent_shapley_value = [score for _, score in avg_s_shapley_value]

        print(len(avg_s_shapley_value))
        print(sent_shapley_value)

        # avg_s_shapley_value = [(sent_id, mean(shapley_value_sampel_times[str(sent_id)])) if str(sent_id) in shapley_value_sampel_times.keys() else (sent_id, 0.0) for sent_id in range(len(data['src_txt']))]
        # sent_shapley_value = [avg_s_shapley_value[str(sent_id)] if str(sent_id) in avg_s_shapley_value.keys() else 0.0 for sent_id in range(len(data['src_txt']))]
       
        avg_s_shapley_value.sort(key = lambda x: x[1], reverse= True)

        shapley_label_id = [sent_id for sent_id, value in avg_s_shapley_value[:len(labels_id)] if value > 0.0] #shapley
        shapley_label_id_1 = [sent_id for sent_id, value in avg_s_shapley_value[: summary_sent_count] if value > 0.0] #shapley
        shapley_label = [0] * len(data['src_txt'])
        shapley_label_1 = [0]* len(data['src_txt'])
        for l in shapley_label_id:
            shapley_label[l] = 1

        for l in shapley_label_id_1:
            shapley_label_1[l] = 1

        data['sent_shapley_value'] = sent_shapley_value

        data['shapley_label'] = shapley_label
        data['shapley_label_1'] = shapley_label_1
        
        all_data.append(data)
        print(index)

    
    with open(save_path,'w') as save:
        json.dump(all_data, save) 
        save.close() 

    datas_shapley = json.load(open(save_path))

    print('orgingal datas length:' + str(len(datas)))
    print('shapley datas length:' + str(len(datas_shapley)))



def _shapley_algorithm(params):
    data, max_sent_count, times, summary_sent_count, in_max_len = params

    clss = data['clss']
    max_sent_ids = bisect.bisect_left(clss, 512)
    sent_rouge_scores = data['sent_rouge_score']
        
    if in_max_len:
        src = data['src_txt'][:max_sent_ids]
        labels = data['src_sent_labels'][:max_sent_ids]
    else:
        src = data['src_txt']
        labels = data['src_sent_labels']
    tgt_txt = data['tgt_txt']
    tgt_sent = tgt_txt.split('<q>')


    abs_sent_list = [sent.split() for sent in tgt_sent]
    doc_sent_list = [sent.split() for sent in src]

    labels_id = [id for id, value in enumerate(labels) if value == 1]
    src_sent_ids = [id for id in range(len(src)) if sent_rouge_scores[id] >0] #筛出sent_rouge = 0的句子
    n = len(src_sent_ids)
    if n <= max_sent_count: #句子数量小于10，全采样
        sample_times = 1
        sample_len = n
    else: #否则设置采样的数量和采样次数
        sample_times = times
        sample_len = max_sent_count
    shapley_value_sampel_times = OrderedDict() #句子shapley value的dict的初始化

    for sent_id in range(len(data['src_txt'])): # intialize the dict,覆盖所有的源文档句子
        shapley_value_sampel_times[sent_id] = []


    for _ in range(sample_times):    
        unique_keys = [] #蒙特卡洛采样,不会出现采样不到的现象

        if sample_times == 1:
            unique_keys = src_sent_ids
        else: 
            sample_unique_keys = monte_carlo_sampling(src_sent_ids, sample_len)
            unique_keys = list(set(sample_unique_keys)) # remove the duplicate

        all_coalitions=[list(j) for i in range(len(unique_keys)) for j in itertools.combinations(unique_keys, r=i+1)]
        # all_coalitions=[list(j) for i in range(len(labels_id)) for j in itertools.combinations(unique_keys, r=i+1)]

        source_data = {} 
        for unique_key in all_coalitions:
            sent_key = [doc_sent_list[id] for id in unique_key]
            rouge_score = rouge_sentence(sent_key, abs_sent_list)
            key = '#'.join([str(id) for id in unique_key])
            source_data[key] = rouge_score
    
        s_shapley_value = shapley_value(unique_keys, all_coalitions, source_data)#[sid, shapley_value]
        
        for sent_id, shapley_score  in s_shapley_value:
            shapley_value_sampel_times[sent_id].append(shapley_score)

    avg_s_shapley_value = [(sent_id, mean(score)) if len(score) > 0 else (sent_id, 0.0) for sent_id, score in shapley_value_sampel_times.items()]
    sent_shapley_value = [score for _, score in avg_s_shapley_value]

    # print(len(avg_s_shapley_value))
    # print(sent_shapley_value)
    
    avg_s_shapley_value.sort(key = lambda x: x[1], reverse= True)

    shapley_label_id = [sent_id for sent_id, value in avg_s_shapley_value[:len(labels_id)] if value > 0.0] #shapley
    shapley_label_id_1 = [sent_id for sent_id, value in avg_s_shapley_value[: summary_sent_count] if value > 0.0] #shapley
    shapley_label = [0] * len(data['src_txt'])
    shapley_label_1 = [0]* len(data['src_txt'])
    for l in shapley_label_id:
        shapley_label[l] = 1

    for l in shapley_label_id_1:
        shapley_label_1[l] = 1

    data['sent_shapley_value'] = sent_shapley_value

    data['shapley_label'] = shapley_label
    data['shapley_label_1'] = shapley_label_1

    return data

def calculate_shapley_with_approximate_pool(datas, save_path, max_sent_count = 10, times = 10, summary_sent_count = 2, in_max_len = False, n_cpus = 100):
    pool = Pool(n_cpus)

    data_list = []
    
    for data in datas:
        data_list.append((data, max_sent_count, times, summary_sent_count, in_max_len))

    # for d in pool.imap(_shapley_algorithm, data_list):
    #         pass

    result_list = pool.imap(_shapley_algorithm, data_list)

    result_list = list(result_list)
    
    pool.close()
    pool.join()


    with open(save_path,'w') as save:
        json.dump(result_list, save) 
        save.close() 

    datas_shapley = json.load(open(save_path))

    print('orgingal datas length:' + str(len(datas)))
    print('shapley datas length:' + str(len(datas_shapley)))


def save_dict(_log_path, dic, name):
    path = os.path.join(_log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f, indent = 4)
    f.close()

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, default='./source_path',help="Path to directory where data  are stored")
    # arg_parser.add_argument('--test_path', type=str, default='./test_path',help="Path to directory where data  are stored")
    # arg_parser.add_argument('--valid_path', type=str, default='./valid_path',help="Path to directory where data  are stored")
    # arg_parser.add_argument('--train_path', type=str, default='./train_path',help="Path to directory where data  are stored")
    arg_parser.add_argument('--save_path', type=str, default='./save_path',help="Path to directory where data  are stored")

    arg_parser.add_argument('--max_sent_count', type=int, default=10,help="max_sent_count") 
    arg_parser.add_argument('--times', type=int, default=10,help="sample times")
    arg_parser.add_argument('--summary_sent_count', type=int, default=2,help="sentence count in summary")
    arg_parser.add_argument('--in_max_len', action='store_true', default=False, help="get label in 512 tokens")
    arg_parser.add_argument('--within_summary_count', action='store_true', default=False, help="combination contains max sentence count same as summary count or not")
    arg_parser.add_argument('--normal', action='store_true', default=False, help="set shapley increment with zero or not")
    arg_parser.add_argument('--seed', type = int, default=114514, help="seed")


    args, _ = arg_parser.parse_known_args()
    print(args)


    for corpus_type in ['test', 'valid', 'train']:
    # for corpus_type in ['test', 'valid']:
    # for corpus_type in ['train']:
    # for corpus_type in ['test']:
        source_path = os.path.join(args.source_path, corpus_type + '.json')
        datas = json.load(open(source_path))
        save_path = os.path.join(args.save_path, corpus_type + '.json')

        ###if save path not exist, create save_path
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        ##save args to json
        save_dict(args.save_path, args, 'args')

        calculate_shapley_with_approximate_pool(datas, save_path, 
                                                max_sent_count = args.max_sent_count, 
                                                times = args.times, 
                                                summary_sent_count = args.summary_sent_count, 
                                                in_max_len = args.in_max_len)

   

    
    
