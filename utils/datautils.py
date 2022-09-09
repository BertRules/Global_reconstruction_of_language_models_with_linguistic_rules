import json
import numpy as np
from tqdm import tqdm
from rule import ruleutils,rulemine
from utils import utils, datautils

def get_indexed_word(w, dec=True):
    p = w.rfind('-')
    s = w[:p]
    idx = int(w[p + 1:])
    if dec:
        idx -= 1
    return s, idx


def read_sent_dep_tups_rbsep(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        rel, gov, dep = line.split(' ')
        w_gov, idx_gov = get_indexed_word(gov, False)
        w_dep, idx_dep = get_indexed_word(dep, False)
        tups.append((rel, (idx_gov, w_gov), (idx_dep, w_dep)))
        # tups.append(line.split(' '))
    return tups

def next_sent_pos(fin):
    pos_tags = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        pos_tags.append(line)
    return pos_tags


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        # tempfix one error in dep_relation_file
        if len(dep_tup)==4:
            dep_tup[1]= dep_tup[1]+dep_tup[2]
            dep_tup[2] = dep_tup[3]
            dep_tup.pop(3)
        wgov, idx_gov = get_indexed_word(dep_tup[0])
        wdep, idx_dep = get_indexed_word(dep_tup[1])
        dep_tup = (dep_tup[2], (idx_gov, wgov), (idx_dep, wdep))
        dep_list.append(dep_tup)
    return dep_list


def load_dep_tags_list(filename, space_sep=True):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    while True:
        if space_sep:
            dep_tags = next_sent_dependency(f)
        else:
            dep_tags = read_sent_dep_tups_rbsep(f)
        if not dep_tags:
            break
        sent_dep_tags_list.append(dep_tags)
    f.close()
    return sent_dep_tags_list


def load_pos_tags(filename):
    f = open(filename, encoding='utf-8')
    sent_pos_tags_list = list()
    while True:
        sent_pos_tags = next_sent_pos(f)
        if not sent_pos_tags:
            break
        sent_pos_tags_list.append(sent_pos_tags)
    f.close()
    return sent_pos_tags_list

def load_dep_tags_list_bst(filename):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    sentence=[]
    sent_dep_tags = []
    for line in f:
        if not line.strip():
            sent_dep_tags = next_sent_dependency(sentence)
            sentence = []
            sent_dep_tags_list.append(sent_dep_tags)
            sent_dep_tags = []
        else:
            sentence.append(line)
    f.close()
    return sent_dep_tags_list


def load_pos_tags_bst(filename):
    f = open(filename, encoding='utf-8')
    sent_pos_tags_list = list()
    sentence=[]
    sent_pos_tags = []
    for line in f:
        if not line.strip():
            sent_pos_tags = next_sent_pos(sentence)
            sentence = []
            sent_pos_tags_list.append(sent_pos_tags)
            sent_pos_tags = []
        else:
            sentence.append(line)
    f.close()
    return sent_pos_tags_list


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def load_json_objs(filename):
    f = open(filename, encoding='utf-8')
    objs = list()
    for line in f:
        objs.append(json.loads(line))
    f.close()
    return objs

def load_json_objs_own_data(filename):
    f = open(filename, encoding='utf-8')
    objs = json.load(f)
    f.close()
    return objs


def write_terms_list(terms_list, dst_file):
    fout = open(dst_file, 'w', encoding='utf-8')
    for terms in terms_list:
        fout.write('{}\n'.format(json.dumps(terms, ensure_ascii=False)))
    fout.close()


def __add_unk_word(word_vecs_matrix):
    n_words = word_vecs_matrix.shape[0]
    dim = word_vecs_matrix.shape[1]
    word_vecs = np.zeros((n_words + 1, dim), np.float32)
    for i in range(n_words):
        word_vecs[i] = word_vecs_matrix[i]
    word_vecs[n_words] = np.random.normal(0, 0.1, dim)
    # word_vecs[n_words] = np.random.uniform(-0.1, 0.1, dim)
    return word_vecs


def load_word_vecs(word_vecs_file, add_unk=True):
    import pickle

    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    if add_unk and (not vocab[-1] == '<UNK>'):
        print('add <UNK>')
        word_vecs_matrix = __add_unk_word(word_vecs_matrix)
        vocab.append('<UNK>')

    assert vocab[-1] == '<UNK>'
    return vocab, word_vecs_matrix


def load_json_own_labels(filename):
    data = json.load(open(filename, 'r'))

    true_op = []
    sentences2 = []
    sentencesx = []

    for i in data:
        sentencesx = (data[i]['sentence'])
        sentences2.append(' '.join(sentencesx))
        
    for i,j in enumerate(sentences2):
        sentences2[i] = j.lower()
        
    for key,value in data.items():
        temp = []
        for j,word in enumerate(value['sentence']):
            if value['label'][j] != 'O':
                temp.append(word.lower())
        true_op.append(temp)
        
    sents = []
    for i,sentence in enumerate(true_op[0:100]):
        tempdict = dict()
        tempdict['text']=sentences2[i]
        tempdict['opinions']=true_op[i]
        sents.append(tempdict)
    
    return sents



def __evaluate(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list, sent_texts):
    correct_sent_idxs = list()
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, (terms_sys, terms_true, dep_tags, pos_tags) in enumerate(
            zip(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list)):
        true_cnt += len(terms_true)
        sys_cnt += len(terms_sys)
        # new_hit_cnt = __count_hit(terms_true, aspect_terms)
        new_hit_cnt = utils.count_hit(terms_true, terms_sys)
        if new_hit_cnt == len(terms_true) and new_hit_cnt == len(terms_sys):
            correct_sent_idxs.append(sent_idx)
        hit_cnt += new_hit_cnt
        # if len(terms_true) and new_hit_cnt < len(terms_true):
        #     print(terms_true)
        #     print(terms_sys)
        #     print(sent_texts[sent_idx])
        #     print(pos_tags)
        #     print(dep_tags)
        #     print()

    # __save_never_hit_terms(sents, terms_sys_list, 'd:/data/aspect/semeval14/tmp.txt')

    print('hit={}, true={}, sys={}'.format(hit_cnt, true_cnt, sys_cnt))
    p = hit_cnt / (sys_cnt + 1e-8)
    r = hit_cnt / (true_cnt + 1e-8)
    print(p, r, 2 * p * r / (p + r + 1e-8))
#    hit_rates.append([hit_cnt,true_cnt,sys_cnt])
    return correct_sent_idxs

def __evaluate_combination_MH(terms_sys_list, terms_true_list, sent_texts):
    correct_sent_idxs = list()
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, (terms_sys, terms_true) in enumerate(
            zip(terms_sys_list, terms_true_list)):
        true_cnt += len(terms_true)
        sys_cnt += len(terms_sys)
        # new_hit_cnt = __count_hit(terms_true, aspect_terms)
        new_hit_cnt = utils.count_hit(terms_true, terms_sys)
        if new_hit_cnt == len(terms_true) and new_hit_cnt == len(terms_sys):
            correct_sent_idxs.append(sent_idx) #komplett richtige Sätze
        hit_cnt += new_hit_cnt
        # if len(terms_true) and new_hit_cnt < len(terms_true):
        #     print(terms_true)
        #     print(terms_sys)
        #     print(sent_texts[sent_idx])
        #     print(pos_tags)
        #     print(dep_tags)
        #     print()
    # __save_never_hit_terms(sents, terms_sys_list, 'd:/data/aspect/semeval14/tmp.txt')
#    print('hit={}, true={}, sys={}'.format(hit_cnt, true_cnt, sys_cnt))
#    p = hit_cnt / (sys_cnt + 1e-8)
#    r = hit_cnt / (true_cnt + 1e-8)
#    f1 = 2 * p * r / (p + r + 1e-8)
#    print('precision = {},\nrecall = {},\nf1-score = {}\n'.format(round(p,4),round(r,4), round(f1,4)))
    hit_rate={'hit_cnt':hit_cnt,'true_cnt':true_cnt,'sys_cnt':sys_cnt}
    return correct_sent_idxs,hit_rate

def __write_rule_results(terms_list, sent_texts, output_file):
    if output_file is not None:
        fout = open(output_file, 'w', encoding='utf-8', newline='\n')
        outlist = []
        for terms_sys, sent_text in zip(terms_list, sent_texts):
            # sent_obj = {'text': sent_text}
            # if terms_sys:
            #     sent_obj['terms'] = terms_sys
            fout.write('{}\n'.format(json.dumps(list(terms_sys), ensure_ascii=False)))
            outlist.append(list(terms_sys))
        fout.close()
    # import pickle
    # filename_pred_dai = 'pred_dai.txt'
    # with open(filename_pred_dai, 'wb') as outfile:
    #     pickle.dump(outlist,outfile)


def __run_with_mined_rules(mine_tool, rule_patterns_file, term_hit_rate_file, dep_tags_file, pos_tags_file,
                           sent_texts_file, filter_terms_vocab_file, mode, term_hit_rate_thres=0.6,
                           output_result_file=None, sents_file=None):
    l1_rules, l2_rules = ruleutils.load_rule_patterns_file(rule_patterns_file)
    term_vocab = ruleutils.get_term_vocab(term_hit_rate_file, term_hit_rate_thres)

    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sent_texts = datautils.read_lines(sent_texts_file)
    filter_terms_vocab = set(datautils.read_lines(filter_terms_vocab_file))
    # opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))


    terms_sys_list = list()
    for sent_idx, (dep_tag_seq, pos_tag_seq, sent_text) in enumerate(zip(dep_tags_list, pos_tags_list, sent_texts)):
        terms = set()
        l1_terms_new = set()
        for p in l1_rules:
            terms_new = ruleutils.find_terms_by_l1_pattern_applyrules(
                p, dep_tag_seq, pos_tag_seq, mine_tool, filter_terms_vocab,mode)
            terms.update(terms_new)
            l1_terms_new.update(terms_new)
        for p in l2_rules:
            terms_new = ruleutils.find_terms_by_l2_pattern(
                p, dep_tag_seq, pos_tag_seq, mine_tool, filter_terms_vocab, l1_terms_new,mode)
            terms.update(terms_new)

        terms_new = mine_tool.get_terms_by_matching(dep_tag_seq, pos_tag_seq, sent_text, term_vocab)
        terms.update(terms_new)

        terms_sys_list.append(list(terms))

    if output_result_file is not None:
        __write_rule_results(terms_sys_list, sent_texts, output_result_file)

    if sents_file is not None:
        sents = datautils.load_json_objs(sents_file)
        # comments for loading data in other format (maybe needed for aspect rule mining later)
        #sents = datautils.load_json_own_labels(sents_file)
        # aspect_terms_true = utils.aspect_terms_list_from_sents(sents)
        terms_list_true = mine_tool.terms_list_from_sents(sents)
        sent_texts = [sent['text'] for sent in sents]
        
        #hier könnten noch manuell erzeugte Regeln bzw. deren Outputs hinzugefügt/evauliert werden, aktuell aber nur list(set()) der predicted Opinion-Terms nach den Rule-Mining Regeln für Evaluation
        pred_comb = []
        for k in range(len(terms_sys_list)):
            #pred_comb.append(list(set(predictions_flat[k]+terms_sys_list[k])))
            pred_comb.append(list(set(terms_sys_list[k])))
  
        #correct_sent_idxs = __evaluate(terms_sys_list, terms_list_true, dep_tags_list, pos_tags_list, sent_texts)
        correct_sent_idxs = __evaluate(pred_comb, terms_list_true, dep_tags_list, pos_tags_list, sent_texts)
#        correct_sent_idxs = __evaluate(predictions_flat, terms_list_true, dep_tags_list, pos_tags_list, sent_texts)


def __run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids,):
    l0_rules,l1_rules, l2_rules = ruleutils.load_rule_patterns_file(rule_patterns_file)
    data_check = rulemine.get_data_by_ids_as_named_tuple(data,ids)
    pattern_with_sentidx = dict()
    for pattern in l0_rules:
        pattern_with_sentidx[str(pattern)] = dict()
    for pattern in l1_rules:
        pattern_with_sentidx[str(pattern)] = dict()
    for pattern in l2_rules:
        pattern_with_sentidx[str(pattern)] = dict()
    terms_sys_list = list()
    terms_sys_dict = dict()
    terms_with_sentidx = dict()

    for i, row in tqdm(list(enumerate(data_check.sents))):
        sent_idx = row['id']
        terms = set()
        terms_idx_per_sent = set()
        for pattern in l0_rules:
            terms_new, terms_idx = ruleutils.find_terms_by_l0_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, list())
                pattern_with_sentidx[str(pattern)][sent_idx].extend(terms_idx)
            terms.update(terms_new)
            terms_idx_per_sent.update(terms_idx)
            curr_terms_sys_list = terms_sys_dict.setdefault(str(pattern),list())
            curr_terms_sys_list.append(terms_new)
        for pattern in l1_rules:
            terms_new, terms_idx = ruleutils.find_terms_by_l1_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, list())
                pattern_with_sentidx[str(pattern)][sent_idx].extend(terms_idx)
            terms.update(terms_new)
            terms_idx_per_sent.update(terms_idx)
            curr_terms_sys_list = terms_sys_dict.setdefault(str(pattern),list())
            curr_terms_sys_list.append(terms_new)
        for pattern in l2_rules:
            terms_new, terms_idx = ruleutils.find_terms_by_l2_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, list())
                pattern_with_sentidx[str(pattern)][sent_idx].extend(terms_idx)
            terms.update(terms_new)
            terms_idx_per_sent.update(terms_idx)
            curr_terms_sys_list = terms_sys_dict.setdefault(str(pattern),list())
            curr_terms_sys_list.append(terms_new)
        terms_sys_list.append(list(terms))
        terms_with_sentidx[str(sent_idx)]=list(terms_idx_per_sent)
    sents = data_check.sents
    # comments for loading data in other format (maybe needed for aspect rule mining later)
    #sents = datautils.load_json_own_labels(sents_file)
    #aspect_terms_true = utils.aspect_terms_list_from_sents(sents)
    terms_list_true = mine_tool.terms_list_from_sents(sents)
    if 'CoLA' in rule_patterns_file:
        terms_list_true_new = [list(set(l)) for l in terms_list_true]
        terms_list_true = terms_list_true_new
    sent_texts = [sent['text'] for sent in sents]
    correct_sent_idxs,hit_rate = __evaluate_combination_MH(
            terms_sys_list, terms_list_true, sent_texts)
    corr_sent_hit_rates= {'correct_sent_idxs':correct_sent_idxs,'hit_rate':hit_rate}
    pattern_to_corr_sent_hit_rates = dict()
    for pattern,terms_sys_list in terms_sys_dict.items():
        if 'CoLA' in rule_patterns_file:
            terms_sys_list_new = [list(set(l)) for l in terms_sys_list]
            terms_sys_list = terms_sys_list_new
        correct_sent_idxs,hit_rate = __evaluate_combination_MH(
            terms_sys_list, terms_list_true, sent_texts)
        pattern_to_corr_sent_hit_rates[pattern]={'correct_sent_idxs':correct_sent_idxs,
                                      'hit_rate':hit_rate}
    return corr_sent_hit_rates, pattern_to_corr_sent_hit_rates, pattern_with_sentidx, terms_with_sentidx


def __get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te):
    true_words_with_sentidx = dict()
    ids = IDs_tr_dev_te['train']
    data_train = dict()
    for i in ids:
        data_train[i] = data[i].copy()
    for i,row in data_train.items():
        curr_terms = row['sents'][term_type+'s']
        curr_terms = [w.lower() for w in curr_terms]
        true_words_with_sentidx[i]=curr_terms        
    return true_words_with_sentidx

def __get_true_words_with_sentidx_dev(term_type, data,IDs_tr_dev_te):
    true_words_with_sentidx = dict()
    ids = IDs_tr_dev_te['dev']
    data_train = dict()
    for i in ids:
        data_train[i] = data[i].copy()
    for i,row in data_train.items():
        curr_terms = row['sents'][term_type+'s']
        curr_terms = [w.lower() for w in curr_terms]
        true_words_with_sentidx[i]=curr_terms        
    return true_words_with_sentidx

def load_data_complete_json(data_filename):
    data_temp = json.load(open(data_filename,'r'))
    data = dict()
    for k,v in data_temp.items():
        v_new = data.setdefault(k,dict())
        for k1,v1 in v.items():
            if 'rel_seqs' in k1:
                v2 = list()
                for l in v1:
                    v2.append((l[0], (l[1][0], l[1][1]), (l[2][0], l[2][1])))
            else:
                v2 = v1
            v_new[k1] = v2
    return data

def generate_random_IDs_tr_dev_te(keys):
    # generate IDs_tr_dev_te
    import random
    random.seed(1234)
    random.shuffle(keys)
    n = len(keys)
    tr,dev,te = 0.65,0.15,0.2
    assert tr+ dev + te == 1
    IDs_tr_dev_te = {'train':keys[0:round(n*tr)],
                     'dev':keys[round(n*tr):round(n*(tr+dev))],
                     'test':keys[round(n*(tr+dev)):]}
    assert n == sum(len(v) for v in IDs_tr_dev_te.values())
    assert n == len(set.union(set(IDs_tr_dev_te['train']),
                              set(IDs_tr_dev_te['dev']),
                              set(IDs_tr_dev_te['test'])))
    return IDs_tr_dev_te

def remove_bb_from_data(data,excluded_BB):
    for v in data.values():
        for k in v.keys():
            if excluded_BB+'_' in k:
                del v[k]
                break
    return data
