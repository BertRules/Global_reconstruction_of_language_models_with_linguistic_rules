import os
import json
from collections import namedtuple
import pandas as pd
from rule import ruleutils
from utils import datautils
from tqdm import tqdm
from time import time
import numpy as np
RuleMineData = namedtuple('RuleMineData', ['dep_tag_seqs', 'pos_tag_seqs', 'sents'])


def __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file):
    tvs_line = datautils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(sents_file)

    assert len(tvs_arr) == len(dep_tags_list)

    dep_tags_list_train, dep_tags_list_valid = list(), list()
    pos_tags_list_train, pos_tags_list_valid = list(), list()
    sents_train, sents_valid = list(), list()
    for tvs_label, dep_tags, pos_tags, sent in zip(tvs_arr, dep_tags_list, pos_tags_list, sents):
        if tvs_label == 0:
            dep_tags_list_train.append(dep_tags)
            pos_tags_list_train.append(pos_tags)
            sents_train.append(sent)
        else:
            dep_tags_list_valid.append(dep_tags)
            pos_tags_list_valid.append(pos_tags)
            sents_valid.append(sent)

    data_train = RuleMineData(dep_tags_list_train, pos_tags_list_train, sents_train)
    data_valid = RuleMineData(dep_tags_list_valid, pos_tags_list_valid, sents_valid)
    return data_train, data_valid

def __load_data_combination_MH(relation_files, tags_files, sents_file, tvs_arr):

    rel_lists = dict()
    tags_lists = dict()

    for k,v in relation_files.items():
        rel_lists[k] = datautils.load_dep_tags_list(v)
    for k,v in tags_files.items():
        tags_lists[k] = datautils.load_pos_tags(v)
    sents = datautils.load_json_objs(sents_file)
    
    # # temp, for fast debug
    # restricted_len = 100
    # for k in list(rel_lists.keys()):
    #     rel_lists[k] = rel_lists[k][:restricted_len]
    # for k in list(tags_lists.keys()):
    #     tags_lists[k] = tags_lists[k][:restricted_len]
    # sents = sents[:restricted_len]
    # tvs_arr = tvs_arr[:restricted_len]
    # # temp end
    
    for k,v in rel_lists.items():
        assert len(tvs_arr) == len(v)
    for k,v in tags_lists.items():
        try:
            assert len(tvs_arr) == len(v)
        except AssertionError:
            print(len(tvs_arr), len(v), k, v)
            raise NotImplementedError
    rel_lists_train = dict()
    for k in rel_lists.keys():
        rel_lists_train[k]=list()
    rel_lists_valid= dict()
    for k in rel_lists.keys():
        rel_lists_valid[k]=list()
    tags_lists_train = dict()
    for k in tags_lists.keys():
        tags_lists_train[k]=list()
    tags_lists_valid= dict()
    for k in tags_lists.keys():
        tags_lists_valid[k]=list()
    sents_train, sents_valid = list(), list()
    for i,tvs_label in enumerate(tvs_arr):
        for k,v in rel_lists.items():
            if tvs_label == 0: rel_lists_train[k].append(v[i])
            else: rel_lists_valid[k].append(v[i])
        for k,v in tags_lists.items():
            if tvs_label == 0: tags_lists_train[k].append(v[i])
            else: tags_lists_valid[k].append(v[i])
        if tvs_label == 0: sents_train.append(sents[i])
        else: sents_valid.append(sents[i])

    fields = []
    fields.extend([i+'_rel_seqs' for i in rel_lists.keys()])
    fields.extend([i+'_tag_seqs' for i in tags_lists.keys()])
    fields.extend(['sents'])
    RuleMineData_combination_MH = namedtuple('RuleMineData_comb_MH', fields)
    data_train = RuleMineData_combination_MH(*rel_lists_train.values(),*tags_lists_train.values(),sents_train)
    data_valid = RuleMineData_combination_MH(*rel_lists_valid.values(),*tags_lists_valid.values(),sents_valid)
    return data_train, data_valid

def get_tp_fp(true_words_with_sentidx,pred_words_with_sentidx,data):
    tp,fp = 0,0
    for sent_idx,token_idxs in pred_words_with_sentidx.items():
        all_words = data[sent_idx]['sents']['text']
        pred_word_ids = set(token_idxs)
        true_words = true_words_with_sentidx[sent_idx]
        true_words = list(set(true_words))
        true_words = [w.lower() for w in true_words]
        true_word_ids = list()
        for i,tok in enumerate(all_words):
            if tok.lower() in true_words:
                true_word_ids.append(i)
        for tok_id,word in enumerate(all_words):
            if tok_id in true_word_ids and tok_id in pred_word_ids:
                tp += 1
            if tok_id not in true_word_ids and tok_id in pred_word_ids:
                fp += 1
    return tp,fp 

def get_tp_fp_CoLA(true_words_with_sentidx,pred_words_with_sentidx,data):
    tp,fp = 0,0
    for sent_idx,token_idxs in pred_words_with_sentidx.items():
        pred_word_ids = set(token_idxs)
        true_words = true_words_with_sentidx[sent_idx]
        true_words = list(set(true_words))
        true_words = [w.lower() for w in true_words]
        if bool(true_words) and bool(pred_word_ids):
            tp += 1
        if not bool(true_words) and bool(pred_word_ids):
            fp += 1
    return tp,fp 

def __get_term_filter_dict(dep_tag_seqs, pos_tag_seqs, terms_list_train, filter_rate, tool,mode):
    term_cnts_dict = dict()
    for dep_tag_seq, pos_tag_seq, terms in zip(dep_tag_seqs, pos_tag_seqs, terms_list_train):
        term_cands = getattr(tool,'get_candidate_terms_'+str(mode))(dep_tag_seq, pos_tag_seq)
        for t in term_cands:
            cnts = term_cnts_dict.get(t, (0, 0))
            hit_cnt = cnts[0] + 1 if t in terms else cnts[0]
            term_cnts_dict[t] = (hit_cnt, cnts[1] + 1)

    filter_terms = set()
    for t, (hit_cnt, cnt) in term_cnts_dict.items():
        if hit_cnt / cnt < filter_rate:
            filter_terms.add(t)
    print(len(filter_terms))
    return filter_terms

def __get_term_filter_dict_combination_MH(data_train, terms_list_train, filter_rate, tool):
    # words, which are allocated to opinion terms only view times, get filtered out
    dep_tag_seqs = data_train.dep_rel_seqs
    term_cnts_dict = dict()
    fields_with_tag = [i for i in data_train._fields if 'tag' in i]
    for i,(dep_tag_seq, terms) in enumerate(zip(dep_tag_seqs, terms_list_train)):
        for field in fields_with_tag:
            curr_tag_seq = getattr(data_train,field)[i]
            mode = field.replace('_tag_seqs','')
            suffix='_'+str(mode)
            term_cands = getattr(tool,'get_candidate_terms'+suffix)(dep_tag_seq, curr_tag_seq)
            for t in term_cands:
                cnts = term_cnts_dict.get(t, (0, 0))
                hit_cnt = cnts[0] + 1 if t in terms else cnts[0]
                term_cnts_dict[t] = (hit_cnt, cnts[1] + 1)

    filter_terms = set()
    for t, (hit_cnt, cnt) in term_cnts_dict.items():
        if hit_cnt / cnt < filter_rate:
            filter_terms.add(t)
    print('number of terms after "__get_term_filter_dict_combination_MH": '+str(len(filter_terms)))
    return filter_terms

def __get_word_cnts_dict(word_cnts_file):
    with open(word_cnts_file, encoding='utf-8') as f:
        df = pd.read_csv(f)
    word_cnt_dict = dict()
    for w, cnt, _ in df.itertuples(False, None):
        word_cnt_dict[w] = cnt
    return word_cnt_dict


def __find_phrase_word_idx_span(phrase, sent_words):
    phrase_words = phrase.split()
    pleft = 0
    while pleft + len(phrase_words) <= len(sent_words):
        p = pleft
        while p - pleft < len(phrase_words) and sent_words[p] == phrase_words[p - pleft]:
            p += 1
        if p - pleft == len(phrase_words):
            return pleft, p
        pleft += 1
    return None

def __find_phrase_word_id_idx_span(term_id, sent_words):
    pleft, p = term_id,term_id+1
    return pleft, p

def __find_related_dep_patterns(dep_tags, pos_tags, term_word_idx_span, mine_tool,sent_idx, mode):
    widx_beg, widx_end = term_word_idx_span
    # print(' '.join(sent_words[widx_beg: widx_end]))
    # for widx in range(widx_beg, widx_end):
    related_dep_tag_idxs = set()
    for i, dep_tag in enumerate(dep_tags):
        rel, gov, dep = dep_tag
        igov, wgov = gov
        idep, wdep = dep
        if not widx_beg <= igov < widx_end and not widx_beg <= idep < widx_end:
            # continue if neither igov nor idep are contained in the term_word_idx_span
            continue
        if widx_beg <= igov < widx_end and widx_beg <= idep < widx_end:
            # continue if igov and idep are contained simultaneously in the term_word_idx_span
            continue
        # print(dep_tag)
        related_dep_tag_idxs.add(i)
        
    # if sent_idx==0:
    #     print(related_dep_tag_idxs)

    # print(related_dep_tag_idxs)
    # patterns_l1 = __patterns_from_l1_dep_tags(aspect_word_wc,
    #     [dep_tags[idx] for idx in related_dep_tag_idxs], pos_tags, term_word_idx_span, opinion_terms)
    # print(patterns_new)
    # related_l2 = __find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags)
    # patterns_l2 = __patterns_from_l2_dep_tags(aspect_word_wc, related_l2, pos_tags, term_word_idx_span, opinion_terms)
    # return patterns_l1, patterns_l2
    return_of_method = mine_tool.get_patterns_from_term(term_word_idx_span, related_dep_tag_idxs, dep_tags, pos_tags, mode)
    return return_of_method

def __find_related_dep_patterns_combination_MH(
                curr_rels, curr_tags, idx_span, mine_tool,sent_idx,
                ruletype):
    widx_beg, widx_end = idx_span
    # print(' '.join(sent_words[widx_beg: widx_end]))
    # for widx in range(widx_beg, widx_end):
    related_rels_idxs=dict()
    for k,v in curr_rels.items():
        curr_related_rels_idxs = set()
        for i, curr_rel in enumerate(v):
            rel, gov, dep = curr_rel
            igov, wgov = gov
            idep, wdep = dep
            if not widx_beg <= igov < widx_end and not widx_beg <= idep < widx_end:
                # continue if neither igov nor idep are contained in the idx_span
                continue
            if widx_beg <= igov < widx_end and widx_beg <= idep < widx_end:
                # continue if igov and idep are contained simultaneously in the idx_span
                continue
            # print(curr_rel)
            curr_related_rels_idxs.add(i)
        related_rels_idxs[k]=curr_related_rels_idxs
    
    # patterns_l0 adaption: adapt get_patterns_from_term_combination_MH method of
    # opinionminetool and aspectminetool
    return_of_method = mine_tool.get_patterns_from_term_combination_MH(idx_span,
                                        related_rels_idxs, curr_rels, curr_tags,
                                        ruletype)
    return return_of_method

def __l1_pattern_legal(pattern, word_freq_dict):
    rel, gov, dep = pattern
    if not gov.startswith('_A') and gov != '_OP' and not gov.isupper() and word_freq_dict.get(gov, 0) < 10:
        return False
    if not dep.startswith('_A') and dep != '_OP' and not dep.isupper() and word_freq_dict.get(dep, 0) < 10:
        return False
    return True


def __filter_l1_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        if __l1_pattern_legal(p, word_freq_dict):
            patterns_new.append(p)
    return patterns_new


def __filter_l2_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        pl, pr = p
        if not __l1_pattern_legal(pl[0], word_freq_dict) or not __l1_pattern_legal(pr[0], word_freq_dict):
            # print(p)
            continue
        patterns_new.append(p)
    return patterns_new


def __find_rule_candidates(dep_tags_list, pos_tags_list, mine_tool, aspect_terms_list, word_cnts_file,
                           freq_thres,mode):
    word_freq_dict = __get_word_cnts_dict(word_cnts_file) # hier wird bspw. unterschieden
    # zwischen "good" und "good."

    # sents = utils.load_json_objs(sents_file)
    cnt_miss = 0
    patterns_l1_cnts, patterns_l2_cnts = dict(), dict()
    for sent_idx, (dep_tags, pos_tags) in enumerate(zip(dep_tags_list, pos_tags_list)):
        assert len(dep_tags) == len(pos_tags)
        sent_words = [dep_tup[2][1] for dep_tup in dep_tags] # basiert aktuell darauf, dass in
        # dep_tags alle wörter des satzes enthalten sind.
        aspect_terms = aspect_terms_list[sent_idx]
        for term in aspect_terms:
            idx_span = __find_phrase_word_idx_span(term, sent_words)

            if idx_span is None:
                cnt_miss += 1
                continue
            # if sent_idx==0:
            #     print(dep_tags,pos_tags,idx_span)
            patterns_l1_new, patterns_l2_new = __find_related_dep_patterns(
                dep_tags, pos_tags, idx_span, mine_tool,sent_idx,mode) # hier werde nicht nur
            #  dependency rules erstellt, sondern auch schon mit POS tags verknüpft

            patterns_l1_new = __filter_l1_patterns(patterns_l1_new, word_freq_dict)
            patterns_l2_new = __filter_l2_patterns(patterns_l2_new, word_freq_dict)

            for p in patterns_l1_new:
                cnt = patterns_l1_cnts.get(p, 0)
                patterns_l1_cnts[p] = cnt + 1
            for p in patterns_l2_new:
                cnt = patterns_l2_cnts.get(p, 0)
                patterns_l2_cnts[p] = cnt + 1

            # patterns_l1.update(patterns_l1_new)
            # patterns_l2.update(patterns_l2_new)
        # if sent_idx >= 100:
        #     break
            # if sent_idx == 0:
            #     print(patterns_l1_new)

    # patterns_l1, patterns_l2 = set(), set()
    patterns_l1 = {p for p, cnt in patterns_l1_cnts.items() if cnt > freq_thres}
    patterns_l2 = {p for p, cnt in patterns_l2_cnts.items() if cnt > freq_thres}

    print(cnt_miss, 'terms missed')
    return patterns_l1, patterns_l2

def __find_rule_candidates_combination_MH_per_batch(batch_extended):
    batch,CoLA_flag,fields_with_rel,fields_with_tag, \
        data, IDs_tr_dev_te,mine_tool,term_ids_dict,ruletype = batch_extended
    ids = IDs_tr_dev_te['train']
    data_train = get_data_by_ids_as_named_tuple(data,ids)
    patterns_l0_cnts,patterns_l1_cnts, patterns_l2_cnts = dict(),dict(),dict()
    for sent_idx,sentence in batch:
        sent_words = sentence['text']
        id_sent = sentence['id']
        opinion_term_ids = term_ids_dict[id_sent]
                
        curr_rels = dict()
        curr_tags = dict()
        
        for field_rel in fields_with_rel:
            curr_rels_list = getattr(data_train,field_rel)
            curr_rels[field_rel]=curr_rels_list[sent_idx]
        for field_tag in fields_with_tag:
            curr_tags_list = getattr(data_train,field_tag)
            curr_tags[field_tag]=curr_tags_list[sent_idx]
        
        if CoLA_flag:
            if opinion_term_ids:
                idx_spans = [(i,i+1) for i in range(len(sent_words))]
                patterns_l0_new_list, patterns_l1_new_list, patterns_l2_new_list = list(),list(),list()
                for idx_span in idx_spans:
                    patterns_l0_new, patterns_l1_new, patterns_l2_new = \
                        __find_related_dep_patterns_combination_MH(
                        curr_rels, curr_tags, idx_span, mine_tool,sent_idx,
                        ruletype) # hier werden
                    #  nicht nur rules mit rels erstellt, sondern auch schon mit tags verknüpft
                    patterns_l0_new_list.extend(patterns_l0_new)
                    patterns_l1_new_list.extend(patterns_l1_new)
                    patterns_l2_new_list.extend(patterns_l2_new)
                patterns_l0_new = set(patterns_l0_new_list)
                patterns_l1_new = set(patterns_l1_new_list)
                patterns_l2_new = set(patterns_l2_new_list)
                for p in patterns_l0_new:
                    cnt = patterns_l0_cnts.get(p, 0)
                    patterns_l0_cnts[p] = cnt + 1
                for p in patterns_l1_new:
                    cnt = patterns_l1_cnts.get(p, 0)
                    patterns_l1_cnts[p] = cnt + 1
                for p in patterns_l2_new:
                    cnt = patterns_l2_cnts.get(p, 0)
                    patterns_l2_cnts[p] = cnt + 1
        else:
            for term_id in opinion_term_ids:
                idx_span = __find_phrase_word_id_idx_span(term_id, sent_words)
    
                if idx_span is None:
                    # cnt_miss += 1
                    continue
                patterns_l0_new, patterns_l1_new, patterns_l2_new = \
                    __find_related_dep_patterns_combination_MH(
                    curr_rels, curr_tags, idx_span, mine_tool,sent_idx,
                    ruletype) # hier werden
                #  nicht nur rules mit rels erstellt, sondern auch schon mit tags verknüpft
                
                # die nächsten beiden zeilen sind eig. überflüssig, da aktuell keine
                # direkten wörter aus den sätzen in den Regeln auftauchen
                # patterns_l1_new = __filter_l1_patterns(patterns_l1_new, word_freq_dict)
                # patterns_l2_new = __filter_l2_patterns(patterns_l2_new, word_freq_dict)

                for p in patterns_l0_new:
                    cnt = patterns_l0_cnts.get(p, 0)
                    patterns_l0_cnts[p] = cnt + 1
                for p in patterns_l1_new:
                    cnt = patterns_l1_cnts.get(p, 0)
                    patterns_l1_cnts[p] = cnt + 1
                for p in patterns_l2_new:
                    cnt = patterns_l2_cnts.get(p, 0)
                    patterns_l2_cnts[p] = cnt + 1
    return patterns_l0_cnts,patterns_l1_cnts, patterns_l2_cnts

def __find_rule_candidates_combination_MH(
            data, IDs_tr_dev_te, mine_tool, term_ids_dict_train,
            freq_thres,output_file,num_cores,
            use_topk_patterns_runrulemine_prec_filtering,k_topkprec,
            use_topk_patterns_runrulemine_absfreq_filtering,
            k_topkabsfreq,
            ruletype):
    term_ids_dict = term_ids_dict_train
    CoLA_flag = bool('_CoLA_' in output_file)
    
    if use_topk_patterns_runrulemine_prec_filtering and '2nd_it_' not in output_file:
        if '_topkprec_'+str(k_topkprec)+'.txt' not in output_file:
            raise NotImplementedError
        output_file = output_file.replace('_topkprec_'+str(k_topkprec)+'.txt','.txt')
    rulegeneration_file = output_file.replace('.txt','_rulegen.json')
    # >> or '1st_it_patterns_runrulemine' not in rulegeneration_file: <<
    # is put into a comment, s.t. rulegen file is also loaded for 2nd iteration,
    # otherwise, rulegen would be done always for 2nd iteration
    if not os.path.isfile(rulegeneration_file): #\
        # or '1st_it_patterns_runrulemine' not in rulegeneration_file:
        # sents = utils.load_json_objs(sents_file)
        # cnt_miss = 0
        patterns_l0_cnts,patterns_l1_cnts, patterns_l2_cnts =dict(), dict(), dict()
        ids = IDs_tr_dev_te['train']
        data_train = get_data_by_ids_as_named_tuple(data,ids)
        fields_with_rel = [i for i in data_train._fields if 'rel' in i]
        fields_with_tag = [i for i in data_train._fields if 'tag' in i]
        sentences = data_train.sents
        print("processing each sentence in __find_rule_candidates_combination_MH:")
        
        # t0 = time()
        batch_size = len(sentences)/num_cores
        batch_size = round(batch_size)+1 if round(batch_size)<batch_size else round(batch_size)
        batches = group_in_batches([(sent_idx,sentence) for sent_idx,sentence in \
                                    enumerate(sentences)],batch_size)
        batches_extended = [(batch,CoLA_flag,fields_with_rel,fields_with_tag,
                             data,IDs_tr_dev_te,mine_tool,
                             term_ids_dict,ruletype) for batch in batches]
        if num_cores == 1:
            batch_extended = batches_extended[0]
            result = [__find_rule_candidates_combination_MH_per_batch(batch_extended)]
        else:
            import multiprocessing as mp
            pool = mp.Pool(num_cores)
            result = pool.map(__find_rule_candidates_combination_MH_per_batch,
                              batches_extended)
        for patterns_l0_cnts_pb,patterns_l1_cnts_pb, patterns_l2_cnts_pb in result:
            for p,cnt in patterns_l0_cnts_pb.items():
                cnt_old = patterns_l0_cnts.get(p, 0)
                patterns_l0_cnts[p] = cnt+cnt_old
            for p,cnt in patterns_l1_cnts_pb.items():
                cnt_old = patterns_l1_cnts.get(p, 0)
                patterns_l1_cnts[p] = cnt+cnt_old
            for p,cnt in patterns_l2_cnts_pb.items():
                cnt_old = patterns_l2_cnts.get(p, 0)
                patterns_l2_cnts[p] = cnt+cnt_old


        # dismiss O "outside" labels of tags
        patterns_l0_cnts_new, patterns_l1_cnts_new, patterns_l2_cnts_new = list(),list(), list()    
        drop = ['ASP_O','NER_O','OP_O', 'NO_SYN', 'SENT_O', 'NO_TOP_RST', 'NO_LEAF_RST', 'NO_ANT', 'NO_RHYP']
        for pattern,cnts in patterns_l0_cnts.items():
            if not any(dropword in pattern for dropword in drop):
                   patterns_l0_cnts_new.append((pattern,cnts))
        for pattern,cnts in patterns_l1_cnts.items():
            if not any(dropword in str(pattern) for dropword in drop):
                    patterns_l1_cnts_new.append((pattern,cnts))
        for pattern,cnts in patterns_l2_cnts.items():
            if not any(dropword in str(pattern) for dropword in drop):
                    patterns_l2_cnts_new.append((pattern,cnts))
    
        patterns_l0_cnts = dict(patterns_l0_cnts_new)
        patterns_l1_cnts = dict(patterns_l1_cnts_new)
        patterns_l2_cnts = dict(patterns_l2_cnts_new)

        # print(cnt_miss, 'terms missed in "__find_rule_candidates_combination_MH"')
        patterns_l0_cnts = dict((p,cnt) for p, cnt in patterns_l0_cnts.items() if cnt >= 5)
        patterns_l1_cnts = dict((p,cnt) for p, cnt in patterns_l1_cnts.items() if cnt >= 5)
        patterns_l2_cnts = dict((p,cnt) for p, cnt in patterns_l2_cnts.items() if cnt >= 5)
        
        patterns_l0_cnts_str=dict([(k,v) for k,v in patterns_l0_cnts.items()])
        patterns_l1_cnts_str=dict([(' '.join(k),v) for k,v in patterns_l1_cnts.items()])
        patterns_l2_cnts_str=dict([(' '.join(k[0][0])+' '+str(k[0][1])+' '+' '.join(k[1][0])+' '+str(k[1][1]),v) \
                                   for k,v in patterns_l2_cnts.items()])
        rulegeneration_dict = {'patterns_l0_cnts_str':patterns_l0_cnts_str,
                               'patterns_l1_cnts_str':patterns_l1_cnts_str,
                               'patterns_l2_cnts_str':patterns_l2_cnts_str,
                               }
        with open(rulegeneration_file,'w') as rg_f:
            json.dump(rulegeneration_dict,rg_f)
        with open(rulegeneration_file,'r') as rg_f:
            rulegeneration_dict_ceck = json.load(rg_f)
        patterns_l0_cnts1 = rulegeneration_dict_ceck['patterns_l0_cnts_str']
        assert patterns_l0_cnts1 == patterns_l0_cnts
        patterns_l1_cnts1 = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict_ceck['patterns_l1_cnts_str'].items()])
        assert patterns_l1_cnts1 == patterns_l1_cnts                        
        patterns_l2_cnts1 = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict_ceck['patterns_l2_cnts_str'].items()])
        assert patterns_l2_cnts1 == patterns_l2_cnts

    else:
        with open(rulegeneration_file,'r') as rg_f:
            rulegeneration_dict = json.load(rg_f)
        patterns_l0_cnts = rulegeneration_dict['patterns_l0_cnts_str']
        patterns_l1_cnts = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict['patterns_l1_cnts_str'].items()])
        patterns_l2_cnts = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict['patterns_l2_cnts_str'].items()])
    number_of_patterns_rulegen_json = len(patterns_l0_cnts.keys()) + len(patterns_l1_cnts.keys()) + len(patterns_l2_cnts.keys())
    print(f"number_of_patterns_rulegen_json = {str(number_of_patterns_rulegen_json)}")

    if use_topk_patterns_runrulemine_absfreq_filtering:
        freq_to_count = dict()
        count_total = 0
        for lX_key,patterns_dict in rulegeneration_dict.items():
            for p,freq in patterns_dict.items():
                curr_count = freq_to_count.setdefault(freq,0)
                curr_count += 1
                freq_to_count[freq] = curr_count
                count_total+=1
        df = pd.DataFrame.from_dict(freq_to_count,orient='index')
        df.sort_index(inplace=True,ascending=False)
        cumulative = np.cumsum(df[0].values)
        frequencies = [i for i in df.index.values]
        for curr_cum_number_patterns,curr_freq in zip(cumulative,frequencies):
            if curr_cum_number_patterns > k_topkabsfreq:
                break
            else:
                cum_number_patterns = curr_cum_number_patterns 
                freq = curr_freq
        freq_thres = freq
        print(f"k_topkabsfreq = {str(k_topkabsfreq)}")
    print(f"use_topk_patterns_runrulemine_absfreq_filtering == {str(use_topk_patterns_runrulemine_absfreq_filtering)}")
    print(f"freq_thres = {str(freq_thres)}")
    patterns_l0 = {p for p, cnt in patterns_l0_cnts.items() if cnt >= freq_thres}
    patterns_l1 = {p for p, cnt in patterns_l1_cnts.items() if cnt >= freq_thres}
    patterns_l2 = {p for p, cnt in patterns_l2_cnts.items() if cnt >= freq_thres}
    number_of_patterns_absfreq_filtering = len(patterns_l0) + len(patterns_l1) + len(patterns_l2)
    print(f"number_of_patterns_absfreq_filtering = {str(number_of_patterns_absfreq_filtering)}")
    if use_topk_patterns_runrulemine_absfreq_filtering:
        assert number_of_patterns_absfreq_filtering == cum_number_patterns

    return patterns_l0,patterns_l1, patterns_l2

def __filter_l1_patterns_through_matching(patterns, dep_tags_list, pos_tags_list, terms_list, mine_tool,
                                          filter_terms_vocab, pattern_filter_rate,mode):
    patterns_keep = list()
    for i,p in enumerate(patterns):
        hit_cnt, cnt = 0, 0
        
        for dep_tags, pos_tags, terms_true in zip(dep_tags_list, pos_tags_list, terms_list):
            terms_true = set(terms_true)
            terms = ruleutils.find_terms_by_l1_pattern(
                p, dep_tags, pos_tags, mine_tool, filter_terms_vocab,mode)

            for term in terms:
                if term in filter_terms_vocab:
                    continue
                cnt += 1
                if term in terms_true:
                    hit_cnt += 1

        if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
        print(i)
    return patterns_keep


def __filter_l0_patterns_through_matching_combination_MH(
        patterns_l0,  data, IDs_tr_dev_te, terms_list_valid, true_words_with_sentidx,
        mine_tool, pattern_filter_rate,output_file,use_topk_patterns,k_topkprec,
        use_only_patterns_in_rulegen_lXwithprec_json):
    CoLA_flag = bool('_CoLA_' in output_file)
    if use_topk_patterns and '2nd_it_' not in output_file:
        if '_topkprec_'+str(k_topkprec)+'.txt' not in output_file:
            raise NotImplementedError
        output_file = output_file.replace('_topkprec_'+str(k_topkprec)+'.txt','.txt')
    rulegeneration_withprec_file = output_file.replace('.txt','_rulegen_l0withprec.json')
    
    if not os.path.isfile(rulegeneration_withprec_file): #\
        patterns_to_prec_dict = dict()
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict,rg_f)
    else:
        with open(rulegeneration_withprec_file,'r') as rg_f:
            patterns_to_prec_dict = json.load(rg_f)

    patterns_l0_todo = patterns_l0 - set(patterns_to_prec_dict.keys())

    if patterns_l0_todo and not use_only_patterns_in_rulegen_lXwithprec_json:
        print('filter l0 patterns:')
        
        for pattern in tqdm(patterns_l0_todo):
            ids = IDs_tr_dev_te['dev']
            data_valid = get_data_by_ids_as_named_tuple(data,ids)
            pattern_with_sentidx = ruleutils.get_pattern_with_sentidx(pattern,data_valid,mine_tool)
            pred_words_with_sentidx = pattern_with_sentidx[str(pattern)]
            if CoLA_flag:
                tp,fp  = get_tp_fp_CoLA(true_words_with_sentidx,pred_words_with_sentidx,data)
            else:
                tp,fp  = get_tp_fp(true_words_with_sentidx,pred_words_with_sentidx,data)
            prec = (tp+ 1e-8) / (tp+fp+ 1e-8)
            patterns_to_prec_dict[pattern] = prec
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict,rg_f)
        with open(rulegeneration_withprec_file,'r') as rg_f:
            pattern_to_prec1 = json.load(rg_f)
        assert pattern_to_prec1 == patterns_to_prec_dict

    patterns_keep = list()
    for pattern,prec in patterns_to_prec_dict.items():
        try: prec > pattern_filter_rate
        except: print(type(prec),type(pattern_filter_rate))
            
        if prec > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(pattern)
    if not use_topk_patterns:            
        print(f"patterns_keep: l0 patterns with prec > {str(pattern_filter_rate)}: {str(len(patterns_keep))}")
    return patterns_keep,patterns_to_prec_dict

def __filter_l1_pattern_through_matching_combination_MH(args):
    (pattern, data, IDs_tr_dev_te, terms_list_valid, mine_tool,pattern_filter_rate)=args
    ids = IDs_tr_dev_te['dev']
    data_valid = get_data_by_ids_as_named_tuple(data,ids)
    hit_cnt, cnt = 0, 0
    for sent_idx,terms_true in enumerate(terms_list_valid):
        terms_true = set(terms_true)
        terms = ruleutils.find_terms_by_l1_pattern_combination_MH_runrulemine(
            pattern, sent_idx, data_valid, mine_tool, [])
        if 'cls' in terms_true:
            terms = set(terms)
        for term in terms:
            cnt += 1
            if term in terms_true:
                hit_cnt += 1

    if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
        # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
        return pattern
        # print(pattern)
    else: return ()

def __filter_l1_pattern_batches_through_matching_combination_MH(batch_extended):
    output_batch=list()
    batch,data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx,\
        mine_tool,pattern_filter_rate,CoLA_flag = batch_extended
    for pattern in batch:
        ids = IDs_tr_dev_te['dev']
        data_valid = get_data_by_ids_as_named_tuple(data,ids)
        pattern_with_sentidx = ruleutils.get_pattern_with_sentidx(pattern,data_valid,mine_tool)
        pred_words_with_sentidx = pattern_with_sentidx[str(pattern)]
        if CoLA_flag:
            tp,fp  = get_tp_fp_CoLA(true_words_with_sentidx,pred_words_with_sentidx,data)
        else:
            tp,fp  = get_tp_fp(true_words_with_sentidx,pred_words_with_sentidx,data)
        prec = (tp+ 1e-8) / (tp+fp+ 1e-8)
        if  prec > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            output_batch.append((pattern,prec))
            # print(pattern)
    return output_batch

def group_in_batches(iterable,batch_size):
    iterable_batches = list()
    for i,row in enumerate(iterable):
        if i%batch_size == 0:
            curr_batch=[]
        curr_batch.append(row)
        if (i+1)%batch_size == 0 and i<len(iterable)-1:
            iterable_batches.append(curr_batch)
    iterable_batches.append(curr_batch)
    return iterable_batches

def __filter_l1_patterns_through_matching_combination_MH(
        patterns_l1, data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,
        pattern_filter_rate,num_cores,output_file,use_topk_patterns,
        k_topkprec,
        use_only_patterns_in_rulegen_lXwithprec_json):
    CoLA_flag = bool('_CoLA_' in output_file)
    if use_topk_patterns and '2nd_it_' not in output_file:
        if '_topkprec_'+str(k_topkprec)+'.txt' not in output_file:
            raise NotImplementedError
        output_file = output_file.replace('_topkprec_'+str(k_topkprec)+'.txt','.txt')
    rulegeneration_withprec_file = output_file.replace('.txt','_rulegen_l1withprec.json')
    
    if not os.path.isfile(rulegeneration_withprec_file): #\
        patterns_to_prec_dict = dict()
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict,rg_f)
    else:
        with open(rulegeneration_withprec_file,'r') as rg_f:
            patterns_to_prec_dict = json.load(rg_f)
        patterns_to_prec_dict = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in patterns_to_prec_dict.items()])
    
    patterns_l1_todo = patterns_l1 - set(patterns_to_prec_dict.keys())

    if patterns_l1_todo and not use_only_patterns_in_rulegen_lXwithprec_json:
        pattern_filter_rate_min = -1 # so this is actually not needed anymore
        print('filter l1 patterns:')
        # iterable = [(pattern,data, IDs_tr_dev_te, terms_list_valid, mine_tool,pattern_filter_rate_min)for pattern in patterns_l1]
        if num_cores == 1:
            batch_size = len(patterns_l1_todo)/num_cores
            batch_size = round(batch_size)+1 if round(batch_size)<batch_size else round(batch_size)
            batches = group_in_batches(patterns_l1_todo,batch_size)
            batches_extended = [(batch,data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,pattern_filter_rate_min,CoLA_flag) for batch in batches]
            batch_extended = batches_extended[0] 
            result = [__filter_l1_pattern_batches_through_matching_combination_MH(batch_extended)]
            result_new = list()
            for output_batch in result:
                result_new.extend(output_batch)
            patterns_to_prec = result_new
            patterns_to_prec = [p for p in patterns_to_prec if p[0]]
            # print('\ntime with mp: '+str(time()-t0)+'\n')
            for pattern,prec in patterns_to_prec:
                patterns_to_prec_dict[pattern]=prec
        else:
            import multiprocessing as mp
            # t0 = time()
            batch_size = len(patterns_l1_todo)/num_cores
            batch_size = round(batch_size)+1 if round(batch_size)<batch_size else round(batch_size)
            batches = group_in_batches(patterns_l1_todo,batch_size)
            batches_extended = [(batch,data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,pattern_filter_rate_min,CoLA_flag) for batch in batches]
            pool = mp.Pool(num_cores)
            result = pool.map(__filter_l1_pattern_batches_through_matching_combination_MH,
                              batches_extended)
            result_new = list()
            for output_batch in result:
                result_new.extend(output_batch)
            patterns_to_prec = result_new
            patterns_to_prec = [p for p in patterns_to_prec if p[0]]
            # print('\ntime with mp: '+str(time()-t0)+'\n')
            for pattern,prec in patterns_to_prec:
                patterns_to_prec_dict[pattern]=prec
        patterns_to_prec_dict_temp = dict([(' '.join(k),v) \
                                 for k,v in patterns_to_prec_dict.items()])
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict_temp,rg_f)
        with open(rulegeneration_withprec_file,'r') as rg_f:
            patterns_to_prec_dict1 = json.load(rg_f)
        patterns_to_prec_dict1 = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in patterns_to_prec_dict1.items()])
        assert patterns_to_prec_dict1 == patterns_to_prec_dict


    patterns_keep = list()
    
    for pattern,prec in patterns_to_prec_dict.items():
        if prec > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(pattern)
    if not use_topk_patterns:            
        print(f"patterns_keep: l1 patterns with prec > {str(pattern_filter_rate)}: {str(len(patterns_keep))}")
    return patterns_keep,patterns_to_prec_dict

def __filter_l2_patterns_through_matching(patterns, dep_tags_list, pos_tags_list, terms_list, mine_tool,
                                          filter_terms_vocab, pattern_filter_rate,mode):
    patterns_keep = list()
    print('filter l2 patterns:')
    for p in tqdm(patterns):
        hit_cnt, cnt = 0, 0
        for dep_tags, pos_tags, terms_true in zip(dep_tags_list, pos_tags_list, terms_list):
            terms_true = set(terms_true)
            terms = ruleutils.find_terms_by_l2_pattern(
                p, dep_tags, pos_tags, mine_tool, filter_terms_vocab,mode)
            for term in terms:
                cnt += 1
                if term in terms_true:
                    hit_cnt += 1
                    # print(p)
                    # print(dep_tag_l, dep_tag_r)
                    # print(term)         
        if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
    return patterns_keep

def __filter_l2_pattern_through_matching_combination_MH(args):
    (pattern, data, IDs_tr_dev_te, terms_list_valid, mine_tool,pattern_filter_rate)=args
    ids = IDs_tr_dev_te['dev']
    data_valid = get_data_by_ids_as_named_tuple(data,ids)
    hit_cnt, cnt = 0, 0
    for sent_idx,terms_true in enumerate(terms_list_valid):
        terms_true = set(terms_true)
        terms = ruleutils.find_terms_by_l2_pattern_combination_MH_runrulemine(
            pattern, sent_idx, data_valid, mine_tool, [])
        if 'cls' in terms_true:
            terms = set(terms)
        for term in terms:
            cnt += 1
            if term in terms_true:
                hit_cnt += 1

    if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
        # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
        return pattern
        # print(pattern)
    else: return ()

def __filter_l2_pattern_batches_through_matching_combination_MH(batch_extended):
    output_batch=list()
    batch, data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,pattern_filter_rate,CoLA_flag=batch_extended
    for pattern in batch:
        ids = IDs_tr_dev_te['dev']
        data_valid = get_data_by_ids_as_named_tuple(data,ids)
        pattern_with_sentidx = ruleutils.get_pattern_with_sentidx(pattern,data_valid,mine_tool)
        pred_words_with_sentidx = pattern_with_sentidx[str(pattern)]
        if CoLA_flag:
            tp,fp  = get_tp_fp_CoLA(true_words_with_sentidx,pred_words_with_sentidx,data)
        else:
            tp,fp  = get_tp_fp(true_words_with_sentidx,pred_words_with_sentidx,data)
        prec = (tp+ 1e-8) / (tp+fp+ 1e-8)

        if prec > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            output_batch.append((pattern,prec))
            # print(pattern)
        
    return output_batch

def __filter_l2_patterns_through_matching_combination_MH(
        patterns_l2, data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,
                                          pattern_filter_rate,num_cores,output_file,use_topk_patterns,
                                          k_topkprec,
                                          use_only_patterns_in_rulegen_lXwithprec_json):
    CoLA_flag = bool('_CoLA_' in output_file)
    if use_topk_patterns and '2nd_it_' not in output_file:
        if '_topkprec_'+str(k_topkprec)+'.txt' not in output_file:
            raise NotImplementedError
        output_file = output_file.replace('_topkprec_'+str(k_topkprec)+'.txt','.txt')
    rulegeneration_withprec_file = output_file.replace('.txt','_rulegen_l2withprec.json')
    
    if not os.path.isfile(rulegeneration_withprec_file): #\
        patterns_to_prec_dict = dict()
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict,rg_f)
    else:
        with open(rulegeneration_withprec_file,'r') as rg_f:
            patterns_to_prec_dict = json.load(rg_f)
        patterns_to_prec_dict = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in patterns_to_prec_dict.items()])
    patterns_l2_todo = patterns_l2 - set(patterns_to_prec_dict.keys())
        
    if patterns_l2_todo and not use_only_patterns_in_rulegen_lXwithprec_json:
        pattern_filter_rate_min = -1
        print('filter l2 patterns:')
        if num_cores == 1:
            batch_size = len(patterns_l2_todo)/num_cores
            batch_size = round(batch_size)+1 if round(batch_size)<batch_size else round(batch_size)
            batches = group_in_batches(patterns_l2_todo,batch_size)
            batches_extended = [(batch,data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,pattern_filter_rate_min,CoLA_flag)for batch in batches]
            batch_extended = batches_extended[0] 
            result = [__filter_l2_pattern_batches_through_matching_combination_MH(batch_extended)]
            result_new = list()
            for output_batch in result:
                result_new.extend(output_batch)
            patterns_to_prec = result_new
        else:
            import multiprocessing as mp
            # t0 = time()
            batch_size = len(patterns_l2_todo)/num_cores
            batch_size = round(batch_size)+1 if round(batch_size)<batch_size else round(batch_size)
            batches = group_in_batches(patterns_l2_todo,batch_size)
            batches_extended = [(batch,data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx, mine_tool,pattern_filter_rate_min,CoLA_flag)for batch in batches]
            pool = mp.Pool(num_cores)
            result = pool.map(__filter_l2_pattern_batches_through_matching_combination_MH,
                              batches_extended)
            result_new = list()
            for output_batch in result:
                result_new.extend(output_batch)
            patterns_to_prec = result_new
            # print('\ntime with mp: '+str(time()-t0)+'\n')
        patterns_to_prec = [p for p in patterns_to_prec if p[0]]
    
        for pattern,prec in patterns_to_prec:
            patterns_to_prec_dict[pattern]=prec
        patterns_to_prec_dict_temp = dict([(' '.join(k[0][0])+' '+str(k[0][1])+' '+' '.join(k[1][0])+' '+str(k[1][1]),v) \
                                 for k,v in patterns_to_prec_dict.items()])
    
        with open(rulegeneration_withprec_file,'w') as rg_f:
            json.dump(patterns_to_prec_dict_temp,rg_f)
        with open(rulegeneration_withprec_file,'r') as rg_f:
            patterns_to_prec_dict1 = json.load(rg_f)
        patterns_to_prec_dict1 = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in patterns_to_prec_dict1.items()])
        assert patterns_to_prec_dict1 == patterns_to_prec_dict

    patterns_keep = list()
    for pattern,prec in patterns_to_prec_dict.items():
        if prec > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(pattern)
    if not use_topk_patterns:            
        print(f"patterns_keep: l2 patterns with prec > {str(pattern_filter_rate)}: {str(len(patterns_keep))}")
            
    return patterns_keep,patterns_to_prec_dict

def __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool):
    prel, pgov, pdep = pattern
    rel, (igov, wgov), (idep, wdep) = dep_tag
    if rel != prel:
        return False
    return mine_tool.match_pattern_word(pgov, wgov, pos_tags[igov]) and mine_tool.match_pattern_word(
        pdep, wdep, pos_tags[idep])


def __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool):
    matched_idxs = list()
    for i, dep_tag in enumerate(dep_tags):
        if __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool):
            matched_idxs.append(i)
    return matched_idxs


def find_terms_by_l1_pattern(pattern, dep_tags, pos_tags, mine_tool, filter_terms_vocab):
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool)
    for idx in matched_dep_tag_idxs:
        term = mine_tool.get_term_from_matched_pattern(pattern, dep_tags, pos_tags, idx)

        if term in filter_terms_vocab:
            continue
        terms.append(term)
    return terms


def gen_rule_patterns(mine_tool, dep_tags_file, pos_tags_file, sents_file, train_valid_split_file,
                      word_cnts_file, freq_thres, term_filter_rate, pattern_filter_rate, output_file, mode):
    # opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))
    data_train, data_valid = __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file)

    # aspect_terms_list_train = utils.aspect_terms_list_from_sents(data_train.sents)
    terms_list_train = mine_tool.terms_list_from_sents(data_train.sents)
    filter_terms_vocab = __get_term_filter_dict(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, terms_list_train, term_filter_rate, mine_tool,mode)

    patterns_l1, patterns_l2 = __find_rule_candidates(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, mine_tool, terms_list_train,
        word_cnts_file, freq_thres,mode)
    print(len(patterns_l1), 'l1 patterns', len(patterns_l2), 'l2 patterns')

    terms_list_valid = mine_tool.terms_list_from_sents(data_valid.sents)

    patterns_l1 = __filter_l1_patterns_through_matching(
        patterns_l1, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, terms_list_valid,
        mine_tool, filter_terms_vocab, pattern_filter_rate,mode)

    patterns_l2 = __filter_l2_patterns_through_matching(
        patterns_l2, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, terms_list_valid,
        mine_tool, filter_terms_vocab, pattern_filter_rate,mode)

    patterns_l1.sort()
    patterns_l2.sort()
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for p in patterns_l1:
        fout.write('{}\n'.format(' '.join(p)))
    for p in patterns_l2:
        (pl, ipl), (pr, ipr) = p
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()

def get_data_by_ids_as_named_tuple(data,ids):
    data_with_ids = dict()
    for k in ids:
        data_with_ids[k]=data[k].copy()
    fields = list(list(data.values())[0])
    RuleMineData_combination_MH = namedtuple('RuleMineData_comb_MH', fields)
    field_to_columns_train = dict(zip(fields,[list() for f in fields]))
    for k,row in data_with_ids.items():
        for f,v in field_to_columns_train.items():
            v.append(row[f])
    return RuleMineData_combination_MH(*field_to_columns_train.values())

def gen_rule_patterns_combination_MH(mine_tool,term_type, data, IDs_tr_dev_te,
                                     freq_thres, 
                                     pattern_filter_rate, output_file,num_cores,
                                     use_topk_patterns_runrulemine_prec_filtering,k_topkprec,
                                     use_topk_patterns_runrulemine_absfreq_filtering,
                                     k_topkabsfreq,
                                     quit_before_runrulmine_final_output,
                                     use_only_patterns_in_rulegen_lXwithprec_json,
                                     ruletype):
    ids = IDs_tr_dev_te['train']
    data_train = get_data_by_ids_as_named_tuple(data,ids)
    ids = IDs_tr_dev_te['dev']
    data_valid = get_data_by_ids_as_named_tuple(data,ids)


    # aspect_terms_list_train = utils.aspect_terms_list_from_sents(data_train.sents)
    # terms_list_train = mine_tool.terms_list_from_sents(data_train.sents)
    term_ids_dict_train = mine_tool.term_ids_dict_from_sents(data_train.sents)

    from time import time
    t0 = time()
    patterns_l0,patterns_l1, patterns_l2 = __find_rule_candidates_combination_MH(
                data, IDs_tr_dev_te, mine_tool, term_ids_dict_train, freq_thres,output_file,num_cores,
                use_topk_patterns_runrulemine_prec_filtering,k_topkprec,
                use_topk_patterns_runrulemine_absfreq_filtering,
                k_topkabsfreq,ruletype)
    print('output of __find_rule_candidates (so after checking absolute frequency with freq_thres):')
    print('l0 patterns: ',len(patterns_l0), '\n',
      'l1 patterns: ',len(patterns_l1), '\n',
      'l2 patterns: ',len(patterns_l2), '\n')
    print(f'time for __find_rule_candidates_combination_MH and k_topkabsfreq = {str(k_topkabsfreq)}: {round(time()-t0)} s')

    # raise NotImplementedError
    

    use_topk_patterns = use_topk_patterns_runrulemine_prec_filtering
    t0 = time()

    terms_list_valid = mine_tool.terms_list_from_sents(data_valid.sents)
    true_words_with_sentidx = datautils.__get_true_words_with_sentidx_dev(term_type, data,IDs_tr_dev_te)
    patterns_l0,patterns_to_prec_dict_l0 = __filter_l0_patterns_through_matching_combination_MH(
        patterns_l0,  data, IDs_tr_dev_te, terms_list_valid, true_words_with_sentidx,
        mine_tool, pattern_filter_rate,output_file,use_topk_patterns,k_topkprec,
        use_only_patterns_in_rulegen_lXwithprec_json)
    patterns_l1,patterns_to_prec_dict_l1 = __filter_l1_patterns_through_matching_combination_MH(
        patterns_l1, data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx,
        mine_tool, pattern_filter_rate,num_cores,output_file,use_topk_patterns,k_topkprec,
        use_only_patterns_in_rulegen_lXwithprec_json)
    patterns_l2,patterns_to_prec_dict_l2 = __filter_l2_patterns_through_matching_combination_MH(
        patterns_l2, data, IDs_tr_dev_te, terms_list_valid,true_words_with_sentidx,
        mine_tool, pattern_filter_rate,num_cores,output_file,use_topk_patterns,k_topkprec,
        use_only_patterns_in_rulegen_lXwithprec_json)
    print(f'time for __filter_lX_patterns_through_matching_combination_MH and k_topkabsfreq = {str(k_topkabsfreq)}: {round(time()-t0)} s')

    if quit_before_runrulmine_final_output:
        raise NotImplementedError
    
    if use_topk_patterns:
        # do not use filtered patterns patterns_lX ("patterns_keep")
        # instead: use top K of patterns based on precision in patterns_to_prec_dict
        if True:
            print(f"use top k={str(k_topkprec)} for l0, l1 and l2 patterns each")
            patterns_to_prec_dict_l0_sorted = sorted(patterns_to_prec_dict_l0.items(),
                                      key = lambda item: (item[1]),
                                      reverse=True)

            patterns_to_prec_dict_l1_sorted = sorted(patterns_to_prec_dict_l1.items(),
                                      key = lambda item: (item[1]),
                                      reverse=True)
            patterns_to_prec_dict_l2_sorted = sorted(patterns_to_prec_dict_l2.items(),
                                      key = lambda item: (item[1]),
                                      reverse=True)
            
            # temp lower amount of l1 pattern (due to low precision after first few)
            # patterns_l0 = list(item[0] for item in patterns_to_prec_dict_l0_sorted)[:round(k_topkprec/5)]
            # patterns_l1 = list(item[0] for item in patterns_to_prec_dict_l1_sorted)[:round(k_topkprec/5)]
            patterns_l0 = list(item[0] for item in patterns_to_prec_dict_l0_sorted)[:k_topkprec]
            patterns_l1 = list(item[0] for item in patterns_to_prec_dict_l1_sorted)[:k_topkprec]
            patterns_l2 = list(item[0] for item in patterns_to_prec_dict_l2_sorted)[:k_topkprec]

        else:
            print(f"use patterns with topkprecision k={str(k_topkprec)}")
            patterns_to_prec_dict_l0_l1_l2 = dict()
            for d in [patterns_to_prec_dict_l0,
                      patterns_to_prec_dict_l1,
                      patterns_to_prec_dict_l2]:
                for k,v in d.items():
                    patterns_to_prec_dict_l0_l1_l2[k] = v
            patterns_to_prec_dict_l0_l1_l2_sorted = sorted(patterns_to_prec_dict_l0_l1_l2.items(),
                                          key = lambda item: (item[1]),
                                          reverse=True)
    
            patterns_l0_l1_l2 = list(item[0] for item in patterns_to_prec_dict_l0_l1_l2_sorted)[:k_topkprec]
    
            patterns_l0 = list(p for p in patterns_l0_l1_l2 if p in patterns_to_prec_dict_l0.keys())
            patterns_l1 = list(p for p in patterns_l0_l1_l2 if p in patterns_to_prec_dict_l1.keys())
            patterns_l2 = list(p for p in patterns_l0_l1_l2 if p in patterns_to_prec_dict_l2.keys())
    
    
    print('overview output runrulemine:')
    print('l0 patterns: ',len(patterns_l0), '\n',
      'l1 patterns: ',len(patterns_l1), '\n',
      'l2 patterns: ',len(patterns_l2), '\n')
    patterns_l0 = list(patterns_l0)
    patterns_l1 = list(patterns_l1)
    patterns_l2 = list(patterns_l2)
    
    # do not sort again to keep order by precision in rule pattern file
    #patterns_l0.sort()
    #patterns_l1.sort()
    #patterns_l2.sort()
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for p in patterns_l0:
        fout.write('{}\n'.format(p))
    for p in patterns_l1:
        fout.write('{}\n'.format(' '.join(p)))
    for p in patterns_l2:
        (pl, ipl), (pr, ipr) = p
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()



def gen_filter_terms_vocab_file(mine_tool, dep_tags_file, pos_tags_file, sents_file, term_filter_rate, output_file, mode):
    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(sents_file)
    # aspect_terms_list = datautils.aspect_terms_list_from_sents(sents)
    terms_list = mine_tool.terms_list_from_sents(sents)
    filter_terms_vocab = __get_term_filter_dict(
        dep_tags_list, pos_tags_list, terms_list, term_filter_rate, mine_tool,mode)
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        for t in filter_terms_vocab:
            fout.write('{}\n'.format(t))



def gen_filter_terms_vocab_file_combination_MH(mine_tool, relation_files,
                        tags_files, sents_file, term_filter_rate, term_filter_file):
    sents = datautils.load_json_objs(sents_file)
    tvs_arr = [0 for i in range(len(sents))]
    data,_ = __load_data_combination_MH(relation_files, tags_files, sents_file,tvs_arr)

    terms_list = mine_tool.terms_list_from_sents(sents)
    filter_terms_vocab = __get_term_filter_dict_combination_MH(
        data, terms_list, term_filter_rate, mine_tool)
    with open(term_filter_file, 'w', encoding='utf-8', newline='\n') as fout:
        for t in filter_terms_vocab:
            fout.write('{}\n'.format(t))


def gen_term_hit_rate_file(mine_tool, train_sents_file, dep_tags_file, pos_tags_file, dst_file):
    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(train_sents_file)
    terms_list = mine_tool.terms_list_from_sents(sents)
    term_hit_cnts = dict()
    for terms in terms_list:
        for t in terms:
            cnt = term_hit_cnts.get(t, 0)
            term_hit_cnts[t] = cnt + 1

    all_terms = set(term_hit_cnts.keys())
    print(len(all_terms), 'terms')
    term_cnts = {t: 0 for t in all_terms}
    # for t in term_hit_cnts.keys():
    for dep_tags, pos_tags, sent in zip(dep_tags_list, pos_tags_list, sents):
        sent_text = sent['text'].lower()
        terms = mine_tool.get_terms_by_matching(dep_tags, pos_tags, sent_text, all_terms)
        for t in terms:
            cnt = term_cnts.get(t, 0)
            term_cnts[t] = cnt + 1

    term_hit_rate_tups = list()
    for t, hit_cnt in term_hit_cnts.items():
        total_cnt = term_cnts.get(t, 0)
        if total_cnt > 0:
            term_hit_rate_tups.append((t, hit_cnt / (total_cnt + 1e-5)))

    term_hit_rate_tups.sort(key=lambda x: -x[1])

    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(term_hit_rate_tups, columns=['term', 'rate']).to_csv(
            fout, float_format='%.4f', index=False)

def gen_term_hit_rate_file_combination_MH(mine_tool, sents_file, relation_files,
                                          tags_files, term_hit_rate_file):
    sents = datautils.load_json_objs(sents_file)
    tvs_arr = [0 for i in range(len(sents))]
    data,_ = __load_data_combination_MH(relation_files, tags_files, sents_file,tvs_arr)
    # # temp, for fast debug
    # restricted_len = 100
    # sents = sents[:restricted_len]
    # # temp ende
    terms_list = mine_tool.terms_list_from_sents(sents)
    term_hit_cnts = dict()
    for terms in terms_list:
        for t in terms:
            cnt = term_hit_cnts.get(t, 0)
            term_hit_cnts[t] = cnt + 1

    all_terms = set(term_hit_cnts.keys())
    print(len(all_terms), 'terms')
    term_cnts = {t: 0 for t in all_terms}
    # for t in term_hit_cnts.keys():
    for sent_idx,sent in enumerate(sents):
        sent_text = sent['text'].lower()
        terms = mine_tool.get_terms_by_matching(data.dep_rel_seqs[sent_idx],
                                                data.pos_tag_seqs[sent_idx],
                                                sent_text, all_terms)
        for t in terms:
            cnt = term_cnts.get(t, 0)
            term_cnts[t] = cnt + 1

    term_hit_rate_tups = list()
    for t, hit_cnt in term_hit_cnts.items():
        total_cnt = term_cnts.get(t, 0)
        if total_cnt > 0:
            term_hit_rate_tups.append((t, hit_cnt / (total_cnt + 1e-5)))

    term_hit_rate_tups.sort(key=lambda x: -x[1])

    with open(term_hit_rate_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(term_hit_rate_tups, columns=['term', 'rate']).to_csv(
            fout, float_format='%.4f', index=False)