from rule.aspectrulemine import AspectMineTool
from rule.opinionrulemine import OpinionMineTool
from rule import rulemine, ruleutils
from rule.ruleutils import get_rule_pattern_from_string_v2
from utils import datautils
import pandas as pd
import json
from significance.utils import eval_sig
from tqdm import tqdm
import os

def runrulemine(run_folder,term_type,data,
                IDs_tr_dev_te,output_file,num_cores,use_topk_patterns_runrulemine_prec_filtering,
                k_topkprec,
                use_topk_patterns_runrulemine_absfreq_filtering,
                k_topkabsfreq, quit_before_runrulmine_final_output,
                use_only_patterns_in_rulegen_lXwithprec_json,
                ruletype,freq_thres = 5):

    ## rule mining parameters
    n_train = len(IDs_tr_dev_te['train'])
    if term_type=='aspect':
        pattern_filter_rate = 0.55
        # freq_thres = 5 # min(15,max(round(n_train/1000),6) )
    else:
        pattern_filter_rate = 0.55
        # freq_thres = 5 # min(15,max(round(n_train/1000),6) )
    # freq threshold ~0.1% of dataset size

    if term_type == 'aspect': mine_tool = AspectMineTool()
    elif term_type == 'opinion': mine_tool = OpinionMineTool()
    else: raise NotImplementedError
       
    # print('Output Rule Patterns File: '+str(output_file))
    rulemine.gen_rule_patterns_combination_MH(mine_tool,term_type, data, IDs_tr_dev_te,
                                              freq_thres, 
                                              pattern_filter_rate, output_file,num_cores,
                                              use_topk_patterns_runrulemine_prec_filtering,
                                              k_topkprec,
                                              use_topk_patterns_runrulemine_absfreq_filtering,
                                              k_topkabsfreq,
                                              quit_before_runrulmine_final_output,
                                              use_only_patterns_in_rulegen_lXwithprec_json,
                                              ruletype)

def intersect_tokens(T_1,T_2):
    T_intersect = dict()
    for k,v_1 in T_1.items():
        if k in T_2.keys():
            v_2 = T_2[k]
            v_intersection = list(set(v_1).intersection(v_2))
            T_intersect[k]=v_intersection
    return T_intersect

def difference_tokens(T_1,T_2):
    # substract tokens T_2 from T_1
    T_difference = dict()
    for k,v_1 in T_1.items():
        if k in T_2.keys():
            v_2 = T_2[k]
            v_difference = list(set(v_1).difference(v_2))
            T_difference[k]=v_difference
        else:
            T_difference[k]=v_1
    return T_difference

def count_tokens(T):
    n = 0
    for v in T.values():
        n+=len(v)
    return n

def ranking(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te):
    ids = IDs_tr_dev_te['train']
    # ids.extend(IDs_tr_dev_te['dev'])
    (corr_sent_hit_rates,pattern_to_corr_sent_hit_rates, patterns_with_sentidx,
     term_idxs) = datautils.__run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids)
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()

    # l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
    # lines=l0_rules+l1_rules+l2_rules
        
    rules_res = list()
    for pattern_idx, line in enumerate(lines):
            pattern = get_rule_pattern_from_string_v2(line)
            hit_rate = pattern_to_corr_sent_hit_rates[str(pattern)]['hit_rate']
            # hier weitermachen: hitrates pro pattern in lines abspeichern
            row ={'pattern_idx':pattern_idx,'pattern':pattern,'rule':line,**hit_rate}
            rules_res.append(row)

    patterns_with_p_r = dict()
    for pattern in rules_res:
        prec = pattern['hit_cnt']/(pattern['sys_cnt']+ 1e-8)
        rec = pattern['hit_cnt']/(pattern['true_cnt']+ 1e-8)
        pat = pattern['pattern']
        patterns_with_p_r[str(pat)] = {'prec':prec,'rec':rec,'pattern_idx':pattern['pattern_idx']}
        
    patterns_with_p_r_sorted = sorted(patterns_with_p_r.items(),
                                      key = lambda item: (item[1]['prec'],item[1]['rec']),
                                      reverse=True)
   
    return patterns_with_p_r_sorted, patterns_with_sentidx

def ranking_on_dev(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te):

    # load rule_patterns_file
    with open(rule_patterns_file, "r",encoding='utf8') as infile:
        lines = infile.readlines()
    # load precision jsons
    prec_jsons_files = [rule_patterns_file.split('runrulemine')[0]+\
                        'runrulemine_rulegen_l'+str(i)+'withprec.json' for i in range(3)]

    with open(prec_jsons_files[0],'r') as rg_f:l0_pattern_to_prec = json.load(rg_f)
    with open(prec_jsons_files[1],'r') as rg_f:l1_pattern_to_prec_str = json.load(rg_f)
    with open(prec_jsons_files[2],'r') as rg_f:l2_pattern_to_prec_str = json.load(rg_f)

    l1_pattern_to_prec = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                             for k,v in l1_pattern_to_prec_str.items()])

    l2_pattern_to_prec = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                             for k,v in l2_pattern_to_prec_str.items()])
    
    # load abs_freq json
    rulegeneration_file = rule_patterns_file.split('runrulemine')[0]+'runrulemine_rulegen.json'

    with open(rulegeneration_file,'r') as rg_f: rulegeneration_dict = json.load(rg_f)
    patterns_l0_cnts = rulegeneration_dict['patterns_l0_cnts_str']
    patterns_l1_cnts = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                             for k,v in rulegeneration_dict['patterns_l1_cnts_str'].items()])
    patterns_l2_cnts = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                             for k,v in rulegeneration_dict['patterns_l2_cnts_str'].items()])

    patterns_with_p_r = dict()
    for line in lines:
        pattern = get_rule_pattern_from_string_v2(line)
        if isinstance(pattern,str)==1:
            assert pattern in l0_pattern_to_prec.keys()
            assert pattern in patterns_l0_cnts.keys()
            prec_on_dev = l0_pattern_to_prec[pattern]
            abs_freq_on_train = patterns_l0_cnts[pattern]
        elif len(pattern)==3:
            pattern = tuple(pattern)
            assert pattern in l1_pattern_to_prec.keys()
            assert pattern in patterns_l1_cnts.keys()
            prec_on_dev = l1_pattern_to_prec[pattern]
            abs_freq_on_train = patterns_l1_cnts[pattern]
        elif len(pattern)==2:
            pattern = tuple(pattern)
            assert pattern in l2_pattern_to_prec.keys()
            assert pattern in patterns_l2_cnts.keys()
            prec_on_dev = l2_pattern_to_prec[pattern]
            abs_freq_on_train = patterns_l2_cnts[pattern]
        else:
            raise NotImplementedError
        
        patterns_with_p_r[pattern] = {'prec':prec_on_dev,'rec':abs_freq_on_train,'line_str':line}
    
    # round prec for more stable sorting
    patterns_with_p_r_sorted = sorted(patterns_with_p_r.items(),
                                      key = lambda item: (round(item[1]['prec'],2),item[1]['rec']),
                                      reverse=True)
    
    # reminder: validation of sorting since rules in log_output seem not to be completly sorted
    # doing sorting in two steps did not change output
    
    return patterns_with_p_r_sorted

def ranking_bal_acc(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te,term_type):
    ids = IDs_tr_dev_te['train']
    (_,_, patterns_with_sentidx,_) = datautils.__run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids)
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
#    
    sentidx_to_extr_tokenidx_BERT = dict()
    for k,v in data.items():
        if k in ids:
            row = v['sents']
            extr_words = row[term_type+'s']
            extr_words = [w.lower() for w in extr_words]
            toks = row['text']
            tokenidx = list()
            for i,tok in enumerate(toks):
                if tok.lower() in extr_words:
                    tokenidx.append(i)
            sentidx_to_extr_tokenidx_BERT[k]=tokenidx

    # adapt for CoLA
        if 'CoLA' in rule_patterns_file:
            for d in [sentidx_to_extr_tokenidx_BERT]:
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            for p,d in patterns_with_sentidx.items():
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            
    # l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
    # lines=l0_rules+l1_rules+l2_rules
    true_positives_negatives_per_sent = dict()
    for k,v in sentidx_to_extr_tokenidx_BERT.items():
        length = len(data[k]['sents']['text'])
        if 'CoLA' in rule_patterns_file:
            length = 1
            if len(v) >1 : raise NotImplementedError
        n = length - len(v)
        true_positives_negatives_per_sent[k] = {'positives': v,'len':length, 'negatives':n}
           
    patterns_with_tp_fp_tn_fn = dict()
    for k,v in patterns_with_sentidx.items():
        tp_pattern, fp_pattern, fn_pattern, tn_pattern = 0,0,0,0
        sent_trues = list()
        for sent_id, pnl in true_positives_negatives_per_sent.items():
            tp_sent,fp_sent,fn_sent,tn_sent = 0,0,0,0
            if sent_id in v.keys():
                extr_toks = set(v[sent_id])
                true_toks = pnl['positives']
                sent_trues.append({sent_id:list(true_toks)})
                for tok in extr_toks:
                    if tok in true_toks:
                        tp_sent += 1
                    else:
                        fp_sent +=1
                for tok in true_toks:
                    if tok not in extr_toks:
                        fn_sent += 1
                tn_sent = pnl['len'] - (tp_sent + fp_sent + fn_sent)
            else:
                fn_sent += len(pnl['positives'])
                tn_sent += pnl['negatives']
            tp_pattern += tp_sent
            fp_pattern += fp_sent
            fn_pattern += fn_sent
            tn_pattern += tn_sent
        patterns_with_tp_fp_tn_fn[k] = list()
        patterns_with_tp_fp_tn_fn[k].append({'tp':tp_pattern, 'fp':fp_pattern, 'fn':fn_pattern,'tn':tn_pattern})
        patterns_with_tp_fp_tn_fn[k].append(v)
        patterns_with_tp_fp_tn_fn[k].append(sent_trues)
    
    rules_res = list()
    for pattern_idx, line in enumerate(lines):
            pattern = get_rule_pattern_from_string_v2(line)
            tp_tn_fp_fn = patterns_with_tp_fp_tn_fn[str(pattern)][0]
            # hier weitermachen: hitrates pro pattern in lines abspeichern
            row ={'pattern_idx':pattern_idx,'pattern':pattern,'rule':line, **tp_tn_fp_fn}
            rules_res.append(row)
    
    patterns_with_bal_acc = dict()
    for pattern in rules_res:
        tp,fp,fn,tn = pattern['tp'],pattern['fp'],pattern['fn'],pattern['tn']
        tnr = (tn + 1e-8)/(fp+tn+1e-8)
        tpr = (tp + 1e-8)/(tp+fn+1e-8)
        bal_acc = (tpr+tnr/2)
        pat = pattern['pattern']
        patterns_with_bal_acc[str(pat)] = {'tnr':tnr,'tpr':tpr,'bal_acc':bal_acc, 'pattern_idx':pattern['pattern_idx']}

    patterns_with_bal_acc_sorted = sorted(patterns_with_bal_acc.items(),
                                      key = lambda item: (item[1]['bal_acc']),
                                      reverse=True)
   
    return patterns_with_bal_acc_sorted, patterns_with_sentidx


def ranking_prec_rec_with_bal_acc(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te,term_type):
    ids = IDs_tr_dev_te['train']
    (_,_, patterns_with_sentidx,_) = datautils.__run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids)
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
#    
    sentidx_to_extr_tokenidx_BERT = dict()
    for k,v in data.items():
        if k in ids:
            row = v['sents']
            extr_words = row[term_type+'s']
            extr_words = [w.lower() for w in extr_words]
            toks = row['text']
            tokenidx = list()
            for i,tok in enumerate(toks):
                if tok.lower() in extr_words:
                    tokenidx.append(i)
            sentidx_to_extr_tokenidx_BERT[k]=tokenidx

    # adapt for CoLA
        if 'CoLA' in rule_patterns_file:
            for d in [sentidx_to_extr_tokenidx_BERT]:
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            for p,d in patterns_with_sentidx.items():
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            
    # l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
    # lines=l0_rules+l1_rules+l2_rules
    true_positives_negatives_per_sent = dict()
    for k,v in sentidx_to_extr_tokenidx_BERT.items():
        length = len(data[k]['sents']['text'])
        if 'CoLA' in rule_patterns_file:
            length = 1
            if len(v) >1 : raise NotImplementedError
        n = length - len(v)
        true_positives_negatives_per_sent[k] = {'positives': v,'len':length, 'negatives':n}
           
    patterns_with_tp_fp_tn_fn = dict()
    for k,v in patterns_with_sentidx.items():
        tp_pattern, fp_pattern, fn_pattern, tn_pattern = 0,0,0,0
        sent_trues = list()
        for sent_id, pnl in true_positives_negatives_per_sent.items():
            tp_sent,fp_sent,fn_sent,tn_sent = 0,0,0,0
            if sent_id in v.keys():
                extr_toks = set(v[sent_id])
                true_toks = pnl['positives']
                sent_trues.append({sent_id:list(true_toks)})
                for tok in extr_toks:
                    if tok in true_toks:
                        tp_sent += 1
                    else:
                        fp_sent +=1
                for tok in true_toks:
                    if tok not in extr_toks:
                        fn_sent += 1
                tn_sent = pnl['len'] - (tp_sent + fp_sent + fn_sent)
            else:
                fn_sent += len(pnl['positives'])
                tn_sent += pnl['negatives']
            tp_pattern += tp_sent
            fp_pattern += fp_sent
            fn_pattern += fn_sent
            tn_pattern += tn_sent
        patterns_with_tp_fp_tn_fn[k] = list()
        patterns_with_tp_fp_tn_fn[k].append({'tp':tp_pattern, 'fp':fp_pattern, 'fn':fn_pattern,'tn':tn_pattern})
        patterns_with_tp_fp_tn_fn[k].append(v)
        patterns_with_tp_fp_tn_fn[k].append(sent_trues)
    
    rules_res = list()
    for pattern_idx, line in enumerate(lines):
            pattern = get_rule_pattern_from_string_v2(line)
            tp_tn_fp_fn = patterns_with_tp_fp_tn_fn[str(pattern)][0]
            # hier weitermachen: hitrates pro pattern in lines abspeichern
            row ={'pattern_idx':pattern_idx,'pattern':pattern,'rule':line, **tp_tn_fp_fn}
            rules_res.append(row)
    
    patterns_with_bal_acc = dict()
    for pattern in rules_res:
        tp,fp,fn,tn = pattern['tp'],pattern['fp'],pattern['fn'],pattern['tn']
        tnr = (tn + 1e-8)/(fp+tn+1e-8)
        tpr = (tp + 1e-8)/(tp+fn+1e-8)
        rec = (tp + 1e-8)/(tp+fn+1e-8)
        prec = (tp + 1e-8)/(tp+fp+1e-8)
        bal_acc = (tpr+tnr/2)
        pat = pattern['pattern']
        patterns_with_bal_acc[str(pat)] = {'prec':prec,'rec':rec,'tnr':tnr,'tpr':tpr,'bal_acc':bal_acc, 'pattern_idx':pattern['pattern_idx']}

    patterns_with_prec_rec_sorted_with_bal_acc = sorted(patterns_with_bal_acc.items(),
                                      key = lambda item: (item[1]['prec'],item[1]['rec']),
                                      reverse=True)
   
    return patterns_with_prec_rec_sorted_with_bal_acc, patterns_with_sentidx


def get_current_BB(pattern_str,term_class):
    current_BB = set()
    if '"' in pattern_str and 'SYNSET' in pattern_str:
        pattern_str=pattern_str.replace('"',"'")
        pattern_str=pattern_str.replace("SYNSET('","SYNSET_('")
    elif 'SYNSET' in pattern_str:
        pattern_str=pattern_str.replace("SYNSET","SYNSET_")
    if '"' in pattern_str and "POS_''" in pattern_str:
        pattern_str=pattern_str.replace('"',"'")    
    if '"' in pattern_str:
        raise NotImplementedError
    if 'WORD' in pattern_str:
        pattern_str=pattern_str.replace('WORD',"WORD_XX")
    rule_len = len(pattern_str.split("', '"))
    if rule_len == 1:
        req = pattern_str
        if req[:2] == term_class:
            req=req[2:]
        temp = req.split('_')
        if len(temp) == 2 or req.startswith("SYNSET_"):
            current_BB.add(temp[0])
        else:
            print(pattern_str)
            raise NotImplementedError
    elif rule_len == 3:
        pattern = pattern_str[2:-2].split("', '")
        for req in pattern:
            if req[:2] == term_class:
                req=req[2:]
            temp = req.split('_')
            if len(temp) == 2 or req.startswith("SYNSET_"):
                current_BB.add(temp[0])
            else:
                print(pattern_str)
                raise NotImplementedError
    else:
        pattern = []
        for temp in pattern_str[2:-2].split('), ('):
            pattern.extend(temp[2:-5].split("', '"))
        for req in pattern:
            if req[:2] == term_class:
                req=req[2:]
            temp = req.split('_')
            if len(temp) == 2 or req.startswith("SYNSET_"):
                current_BB.add(temp[0])
            else:
                print(pattern_str)
                raise NotImplementedError
    return current_BB


def get_all_BB(patterns_to_sentidx_to_extr_tokenidx,term_class):
    all_BB = set()
    for pattern_str,sentidx_to_extr_tokenidx in patterns_to_sentidx_to_extr_tokenidx.items():
        current_BB = get_current_BB(pattern_str,term_class)
        all_BB.update(current_BB)
    return all_BB


def restrict_rule_patterns_file(mine_tool,rule_patterns_file,
                                data,IDs_tr_dev_te,term_type):
    print('start rule patterns file restriction')
    th_limit =  70000 # size of sample from train data for fast evaluation and ranking
    th_patterns = 10000 # cut off from ranked pattern files
    th_all_cnt = 350 # if number of pattern for one specific BB in restricted patterns is less than th_all_cnt then add all patterns of that building block
    
    rule_patterns_file_restricted = \
        rule_patterns_file.replace('.txt','_restricted_v2_'+\
                                   str(th_limit)+'_'+str(th_patterns)+'.txt')
    import os
    if not os.path.isfile(rule_patterns_file_restricted):
        if isinstance(mine_tool,OpinionMineTool): term_prefix = '_O'
        else: term_prefix = '_A'
        import random
        random.seed(123)
        train_ids = IDs_tr_dev_te['train']
        min_val = min(len(train_ids),th_limit)
        temp = random.sample(range(len(train_ids)), min_val)
        train_ids_restr = [train_ids[i] for i in temp]
        ids = train_ids_restr
        (_,pattern_to_corr_sent_hit_rates, patterns_with_sentidx,_) = \
            datautils.__run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids)
        if 'CoLA' in rule_patterns_file:
            # pattern_to_corr_sent_hit_rates counts tokens per sentences, hence
            # this variable has to be rebuild for 'CoLA' based on patterns with_sentidx
            # CoLA is sentence classification -> so binary classification of sentences
            # -> classes are  tokenidx=[0] and tokenidx=[]
            
            sentidx_to_extr_tokenidx_BERT = dict()
            for k,v in data.items():
                if k in ids:
                    row = v['sents']
                    extr_words = row[term_type+'s']
                    extr_words = [w.lower() for w in extr_words]
                    toks = row['text']
                    tokenidx = list()
                    for i,tok in enumerate(toks):
                        if tok.lower() in extr_words:
                            tokenidx.append(i)
                    sentidx_to_extr_tokenidx_BERT[k]=tokenidx
            
            assert pattern_to_corr_sent_hit_rates.keys() == patterns_with_sentidx.keys()
            for d in [sentidx_to_extr_tokenidx_BERT]:
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            for p,d in patterns_with_sentidx.items():
                for k,tokenidx in list(d.items()):
                    if tokenidx:
                        tokenidx = [0]
                    assert tokenidx in [[],[0]]
                    d[k]=tokenidx
            # adapt pattern_to_corr_sent_hit_rates
            # pattern_to_corr_sent_hit_rates has two keys, and only 'hit_rate' is used here
            T_lm = sentidx_to_extr_tokenidx_BERT.copy()
            for k,v in pattern_to_corr_sent_hit_rates.items():
                old_hit_rate = v['hit_rate']
                # old_hit_rate has keys: dict_keys(['hit_cnt', 'true_cnt', 'sys_cnt'])
                # values of these keys are int numbers
                # hit_cnt = tp
                # true_cnt = tp+fn
                # sys_cnt = tp+fp
                curr_sentidx = patterns_with_sentidx[k]
                T_sigma = curr_sentidx.copy()
                   
                ## calculate tp, fp, fn
                tp = count_tokens(intersect_tokens(T_sigma,T_lm))
                fp = count_tokens(difference_tokens(T_sigma,T_lm))
                fn = count_tokens(difference_tokens(T_lm,T_sigma))
                tn = len(ids)-tp-fp-fn
                old_hit_rate['hit_cnt'] = tp
                old_hit_rate['true_cnt'] = tp+fn
                old_hit_rate['sys_cnt'] = tp+fp
            
        all_BB = list(get_all_BB(patterns_with_sentidx,term_prefix))
        if 'WORD' in all_BB:
            all_BB.remove('WORD')
        with open(rule_patterns_file, "r") as infile:
            lines = infile.readlines()
        # l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
        # lines=l0_rules+l1_rules+l2_rules
        lines_format = [str(get_rule_pattern_from_string_v2(line.strip())) for line in list(lines)]
        # split rule patterns such depending on BB
        bb_to_pattern_to_corr_sent_hit_rates_with_bb = dict()
        for bb in all_BB:
            curr_pattern_to_corr_sent_hit_rates_with_bb=dict()
            for k,v in pattern_to_corr_sent_hit_rates.items():
                if bb in get_current_BB(k,term_prefix):
                    curr_pattern_to_corr_sent_hit_rates_with_bb[k]=v
            bb_to_pattern_to_corr_sent_hit_rates_with_bb[bb]=curr_pattern_to_corr_sent_hit_rates_with_bb
       
        from tqdm import tqdm
        pattern_and_f1={'pattern':list(),'f1':list()}
        for p,d in tqdm(pattern_to_corr_sent_hit_rates.items()):
            assert (p in lines_format) #or str([p]) in lines_format)
            line_id = lines_format.index(p) #if p in lines_format else lines_format.index(str([p]))
            p_str = str(lines[line_id])
            tp = d['hit_rate']['hit_cnt']
            pred_pos = d['hit_rate']['sys_cnt']
            true_pos = d['hit_rate']['true_cnt']
            pr_temp = tp/ pred_pos if pred_pos >0 else 0
            rec_temp = tp/ true_pos if true_pos >0 else 0
            f1_temp = 2*pr_temp*rec_temp/(pr_temp+rec_temp) if (pr_temp+rec_temp) > 0 else 0
            pattern_and_f1['pattern'].append(p_str)
            pattern_and_f1['f1'].append(f1_temp)
        pattern_and_f1_df = pd.DataFrame.from_dict(pattern_and_f1)
        pattern_and_f1_df.sort_values(['f1'],ascending=[0],inplace=True)
        lines_new = set()
        for i,pattern in enumerate(list(pattern_and_f1_df.pattern.values)):
            if i >= th_patterns:
                break
            lines_new.add(pattern)
        lines_new = list(lines_new)
        for bb,curr_pattern_to_corr_sent_hit_rates_with_bb in bb_to_pattern_to_corr_sent_hit_rates_with_bb.items():
            temp_list = []
            for line in lines_new:
                if bb in get_current_BB(str(get_rule_pattern_from_string_v2(line.strip())),term_prefix):
                    temp_list.append(line)
            count = len(temp_list)
            if count < th_all_cnt:
                print(bb+' has only this amount of rules in the current rules file:')
                print(len(list(curr_pattern_to_corr_sent_hit_rates_with_bb.keys())))
                print('less rules than threshold, so all rules are added to restricted')
                for p,d in curr_pattern_to_corr_sent_hit_rates_with_bb.items():
                    assert (p in lines_format)# or str([p]) in lines_format)
                    line_id = lines_format.index(p) #if p in lines_format else lines_format.index(str([p]))
                    p_str = str(lines[line_id])
                    lines_new.append(p_str)
                    
        lines_new = list(set(lines_new))
        with open(rule_patterns_file_restricted, "w") as outfile:
            outfile.writelines(lines_new)
    print('end rule patterns file restriction')
    return rule_patterns_file_restricted

def ruleselection(eval_config,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file):
    term_type = eval_config[0] # eg. 'aspect'
    excluded_BB = eval_config[1] # eg. 'POS' # dep,srl,coref,prox,ner,pos,word,wordnet,ownasps,ownops
    excluded_BB = excluded_BB.upper()
    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    # restrict rule set considered for selection for faster runtime
    if isinstance(mine_tool,OpinionMineTool):
        rule_patterns_file = \
            restrict_rule_patterns_file(mine_tool,rule_patterns_file,
                                        data,IDs_tr_dev_te,term_type)
    
    # sort patterns as preparation for selection heuristic
    patterns_with_p_r_sorted, patterns_with_sentidx = ranking(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te)
    
    # patterns_with_p_r_sorted is a dictionary.
    # the keys are the patterns, the values are dicts like
    # {'prec':precision,'rec':recall,'pattern_idx':pattern_number}
    # patterns_with_sentidx is a dictionary
    # the keys are the patterns,
    #    the values are dicts again
    #    the keys of the dicts are sentence ids, and the values are ids of
    #    tokens, which are truly annotated by 'term_type' (i.e., aspects or opinions)
    
    # sort out all rules containing excluded_BB
    patterns_with_p_r_sorted_new = []
    if excluded_BB:
        for sp in patterns_with_p_r_sorted:
            if excluded_BB+'_' not in str(sp[0]) and excluded_BB+'(' not in str(sp[0]):
                patterns_with_p_r_sorted_new.append(sp)
        assert all ((not excluded_BB+'_' in sp[0] and excluded_BB+'(' not in str(sp[0]))  for sp in patterns_with_p_r_sorted_new)
        for sp in patterns_with_p_r_sorted:
            if sp not in patterns_with_p_r_sorted_new:
                patterns_with_sentidx.pop(sp[0])
        patterns_with_p_r_sorted = patterns_with_p_r_sorted_new

   
    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)

    selected_patterns = list()
    compare = dict()
    for i, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        compare[i] = {'true_words':true_words, 'curr_predictions': set()}
    final = compare #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 größer wird dann wird es auch in final geladen, 
    #final ist dann im näcshten schritt das neue compare
    
    true_words_count = 0 #true_cnt
    for sentence in compare.values():
        true_words_count += len(sentence['true_words'])
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
    # lines contains same patterns as keys of patterns_with_sentidx, but
    # in a slightly different format

    f1_max = 0
    for pattern in patterns_with_p_r_sorted:
        if compare != final:
            raise NotImplementedError
        compare = final
        hit = 0
        sys = 0
        for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
            curr_tokens = data[sent_idx]['sents']['text']
            curr_tokens = set([word for i,word in enumerate(curr_tokens) \
                               if i in token_idxs])
            curr_tokens = set([word.lower() for word in curr_tokens])
            compare[sent_idx]['curr_predictions'].update(curr_tokens)
        for sent in compare.values():
            for word in sent['curr_predictions']:
                sys += 1
                if word in sent['true_words']:
                    hit += 1
        p = hit / (sys + 1e-8)
        r = hit / (true_words_count + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)  
        if f1 > f1_max:
            selected_patterns.append(get_rule_pattern_from_string_v2(lines[pattern[1]['pattern_idx']]))
            for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
                curr_tokens = data[sent_idx]['sents']['text']
                curr_tokens = set([word for i,word in enumerate(curr_tokens) \
                                   if i in token_idxs])
                curr_tokens = set([word.lower() for word in curr_tokens])
                final[sent_idx]['curr_predictions'].update(curr_tokens)
            f1_max = f1
            p_max = p
            r_max = r
   
    # extraction of patterns from pattern string 
    patterns_l0 = []
    patterns_l1 = []
    patterns_l2 = []
    
    for pattern in selected_patterns:
        if isinstance(pattern,str)==1:
            patterns_l0.append(pattern)
        elif len(pattern)==3:
            patterns_l1.append(pattern)
        elif len(pattern)==2:
            patterns_l2.append(pattern)
        else:
            print(pattern)
            raise NotImplementedError
    
    if excluded_BB:
        sum_scores_rules_sorted_out  = [0,0]
        for i,(pattern,score) in enumerate(patterns_with_p_r_sorted):
            if excluded_BB+'_' in pattern or excluded_BB+'(' in pattern:
                sum_scores_rules_sorted_out[0] += score[0]
                sum_scores_rules_sorted_out[1] += score[1]
                print(i)
        assert(sum_scores_rules_sorted_out[0]==0 and sum_scores_rules_sorted_out[1]==0)
    
    fout = open(output_rule_selection_file, 'w', encoding='utf-8', newline='\n')
    for pat in patterns_l0:
        fout.write('{}\n'.format(pat))
    for pat in patterns_l1:
        fout.write('{}\n'.format(' '.join(pat)))
    for pat in patterns_l2:
        (pl, ipl), (pr, ipr) = pat
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()
    
    log_file_name = '/'.join(output_rule_selection_file.split('/')[:-1])
    log_file_name += '/log.txt'
    log_string = ('\n###'+output_rule_selection_file.split('/')[-1] +
        '\n[train-data] precision: '+str(round(p_max,4))+
        '\n[train-data] recall: '+str(round(r_max,4))+
        '\n[train-data] f1: '+str(round(f1_max,4))+'\n')
    # next lines are in comment, since f1_max,p_max,r_max are not correct currently,
    # since compare = final (without ".copy()")
    # with open(log_file_name,'a') as logfile:
    #     logfile.write(log_string)


def ruleselection_new(term_type,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file,
                                            restrict_in_ruleselection):
    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    if restrict_in_ruleselection:    
        # restrict rule set considered for selection for faster runtime
        rule_patterns_file = \
            restrict_rule_patterns_file(mine_tool,rule_patterns_file,
                                        data,IDs_tr_dev_te,term_type)
    
    # sort patterns as preparation for selection heuristic
    patterns_with_p_r_sorted, patterns_with_sentidx = ranking(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te)
    # patterns_with_p_r_sorted is a dictionary.
    # the keys are the patterns, the values are dicts like
    # {'prec':precision,'rec':recall,'pattern_idx':pattern_number}
    # patterns_with_sentidx is a dictionary
    # the keys are the patterns,
    #    the values are dicts again
    #    the keys of the dicts are sentence ids, and the values are ids of
    #    tokens, which are truly annotated by 'term_type' (i.e., aspects or opinions)
    
  
    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)

    selected_patterns = list()
    compare = dict()
    for i, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        compare[i] = {'true_words':true_words, 'curr_predictions': set()}
    final = compare.copy() #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 größer wird dann wird es auch in final geladen, 
    #final ist dann im näcshten schritt das neue compare
    
    true_words_count = 0 #true_cnt
    for sentence in compare.values():
        true_words_count += len(sentence['true_words'])
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
    # lines contains same patterns as keys of patterns_with_sentidx, but
    # in a slightly different format

    f1_max = 0
    for pattern in patterns_with_p_r_sorted:
        compare = final.copy()
        hit = 0
        sys = 0
        for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
            curr_tokens = data[sent_idx]['sents']['text']
            curr_tokens = set([word for i,word in enumerate(curr_tokens) \
                               if i in token_idxs])
            curr_tokens = set([word.lower() for word in curr_tokens])
            compare[sent_idx]['curr_predictions'].update(curr_tokens)
        for sent in compare.values():
            for word in sent['curr_predictions']:
                sys += 1
                if word in sent['true_words']:
                    hit += 1
        p = hit / (sys + 1e-8)
        r = hit / (true_words_count + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)  
        if f1 > f1_max:
            selected_patterns.append(get_rule_pattern_from_string_v2(lines[pattern[1]['pattern_idx']]))
            for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
                curr_tokens = data[sent_idx]['sents']['text']
                curr_tokens = set([word for i,word in enumerate(curr_tokens) \
                                   if i in token_idxs])
                curr_tokens = set([word.lower() for word in curr_tokens])
                final[sent_idx]['curr_predictions'].update(curr_tokens)
            f1_max = f1
            p_max = p
            r_max = r
            # write_to_log()
            log_file_name = output_rule_selection_file.replace('.txt','_log.txt')
            pattern_type = ''
            if isinstance(selected_patterns[-1],str)==1:
                pattern_type = 'l0'
            elif len(selected_patterns[-1])==3:
                pattern_type = 'l1'
            elif len(selected_patterns[-1])==2:
                pattern_type = 'l2'
            log_string = (
                'len(selected_patterns):'+str(len(selected_patterns)) +
                '; curr precision: '+str(round(p_max,4))+
                '; curr recall: '+str(round(r_max,4))+
                '; curr f1: '+str(round(f1_max,4))+
                '; precision of pattern: '+str(round(pattern[1]['prec'],4))+
                '; recall of pattern: '+str(round(pattern[1]['rec'],4))+
                '; pattern: '+str(selected_patterns[-1])+
                '; pattern type: '+str(pattern_type)+
                '\n')
            with open(log_file_name,'a') as logfile:
                logfile.write(log_string)
   
    # extraction of patterns from pattern string 
    patterns_l0 = []
    patterns_l1 = []
    patterns_l2 = []
    
    for pattern in selected_patterns:
        if isinstance(pattern,str)==1:
            patterns_l0.append(pattern)
        elif len(pattern)==3:
            patterns_l1.append(pattern)
        elif len(pattern)==2:
            patterns_l2.append(pattern)
        else:
            print(pattern)
            raise NotImplementedError
    
    fout = open(output_rule_selection_file, 'w', encoding='utf-8', newline='\n')
    for pat in patterns_l0:
        fout.write('{}\n'.format(pat))
    for pat in patterns_l1:
        fout.write('{}\n'.format(' '.join(pat)))
    for pat in patterns_l2:
        (pl, ipl), (pr, ipr) = pat
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()
    
    log_file_name = '/'.join(output_rule_selection_file.split('/')[:-1])
    log_file_name += '/log.txt'
    log_string = ('\n###'+output_rule_selection_file.split('/')[-1] +
        '\n[train-data] precision: '+str(round(p_max,4))+
        '\n[train-data] recall: '+str(round(r_max,4))+
        '\n[train-data] f1: '+str(round(f1_max,4))+'\n')
    with open(log_file_name,'a') as logfile:
        logfile.write(log_string)

def ruleselection_new_bal_acc(term_type,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file,
                                            restrict_in_ruleselection):

    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    if restrict_in_ruleselection:    
        # restrict rule set considered for selection for faster runtime
        rule_patterns_file = \
            restrict_rule_patterns_file(mine_tool,rule_patterns_file,
                                        data,IDs_tr_dev_te,term_type)
        print("WARNING: Rule patterns are restricted as for ruleselection regarding f1 score!!")
    
    # sort patterns as preparation for selection heuristic
    patterns_with_bal_acc_sorted, patterns_with_sentidx = ranking_bal_acc(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te, term_type)
    # patterns_with_bal_acc_sorted is a dictionary.
    # the keys are the patterns, the values are dicts like
    # {'tnr':tnr,'tpr':tpr,'bal_acc':bal_acc, 'pattern_idx':pattern_number}
    # patterns_with_sentidx is a dictionary
    # the keys are the patterns,
    #    the values are dicts again
    #    the keys of the dicts are sentence ids, and the values are ids of
    #    tokens, which are truly annotated by 'term_type' (i.e., aspects or opinions)

    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)

    selected_patterns = list()
    compare = dict()
    for sent_idx, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        compare[sent_idx] = {'true_words':true_words, 'curr_predictions': set(),
                      'all_words':set()}
        curr_tokens = data[sent_idx]['sents']['text']
        all_words = set([w.lower() for w in curr_tokens])
        # add all words of the current sentence to the currenct dict "compare"
        compare[sent_idx]['all_words']=all_words

    final = compare.copy() #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 größer wird dann wird es auch in final geladen, 
    #final ist dann im näcshten schritt das neue compare
    
    true_words_count = 0 #true_cnt
    for sentence in compare.values():
        true_words_count += len(sentence['true_words'])
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
    # lines contains same patterns as keys of patterns_with_sentidx, but
    # in a slightly different format

    bal_acc_max = 0
    for pattern in patterns_with_bal_acc_sorted:
        compare = final.copy()
        for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
            curr_tokens = data[sent_idx]['sents']['text']
            pred_words = set([word for i,word in enumerate(curr_tokens) \
                               if i in token_idxs])
            pred_words = set([word.lower() for word in pred_words])
            # add predictions of pattern to the currenct dict "compare"
            compare[sent_idx]['curr_predictions'].update(pred_words)
        tp,fp,tn,fn = 0,0,0,0
        for sent in compare.values():
            all_words = sent['all_words']
            pred_words = sent['curr_predictions']
            true_words = sent['true_words']
            for word in all_words:
                if word in true_words and word in pred_words:
                    tp += 1
                if word in true_words and word not in pred_words:
                    fn += 1
                if word not in true_words and word in pred_words:
                    fp += 1
                if word not in true_words and word not in pred_words:
                    tn += 1
        tpr = (tp+ 1e-8) / (tp+fn+ 1e-8)
        tnr = (tn+ 1e-8) / (tn+fp+ 1e-8)
        bal_acc = (tpr+tnr)/2
        if bal_acc > bal_acc_max:
            selected_patterns.append(get_rule_pattern_from_string_v2(lines[pattern[1]['pattern_idx']]))
            final = compare.copy()
            bal_acc_max = bal_acc
            tpr_max = tpr
            tnr_max = tnr
   
    # extraction of patterns from pattern string 
    patterns_l0 = []
    patterns_l1 = []
    patterns_l2 = []
    
    for pattern in selected_patterns:
        if isinstance(pattern,str)==1:
            patterns_l0.append(pattern)
        elif len(pattern)==3:
            patterns_l1.append(pattern)
        elif len(pattern)==2:
            patterns_l2.append(pattern)
        else:
            print(pattern)
            raise NotImplementedError
    
    fout = open(output_rule_selection_file, 'w', encoding='utf-8', newline='\n')
    for pat in patterns_l0:
        fout.write('{}\n'.format(pat))
    for pat in patterns_l1:
        fout.write('{}\n'.format(' '.join(pat)))
    for pat in patterns_l2:
        (pl, ipl), (pr, ipr) = pat
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()
    
    log_file_name = '/'.join(output_rule_selection_file.split('/')[:-1])
    log_file_name += '/log.txt'
    log_string = ('\n###'+output_rule_selection_file.split('/')[-1] +
        '\n[train-data] true positive rate: '+str(round(tpr_max,4))+
        '\n[train-data] true negative rate: '+str(round(tnr_max,4))+
        '\n[train-data] balanced accuracy: '+str(round(bal_acc_max,4))+'\n')
    with open(log_file_name,'a') as logfile:
        logfile.write(log_string)


def ruleselection_new_bal_acc_v2(term_type,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file,
                                            restrict_in_ruleselection):

    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    if restrict_in_ruleselection:    
        # restrict rule set considered for selection for faster runtime
        rule_patterns_file = \
            restrict_rule_patterns_file(mine_tool,rule_patterns_file,
                                        data,IDs_tr_dev_te,term_type)
        print("WARNING: Rule patterns are restricted as for ruleselection regarding f1 score!!")
    
    # sort patterns as preparation for selection heuristic
    patterns_with_prec_rec_sorted_with_bal_acc, patterns_with_sentidx = ranking_prec_rec_with_bal_acc(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te, term_type)
    # patterns_with_prec_rec_sorted_with_bal_acc is a dictionary.
    # the keys are the patterns, the values are dicts like
    # {'prec':prec,'rec':rec,tnr':tnr,'tpr':tpr,'bal_acc':bal_acc, 'pattern_idx':pattern_number}
    # patterns_with_sentidx is a dictionary
    # the keys are the patterns,
    #    the values are dicts again
    #    the keys of the dicts are sentence ids, and the values are ids of
    #    tokens, which are truly annotated by 'term_type' (i.e., aspects or opinions)

    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)

    selected_patterns = list()
    compare = dict()
    for sent_idx, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        compare[sent_idx] = {'true_words':true_words, 'curr_predictions': set(),
                      'all_words':set()}
        curr_tokens = data[sent_idx]['sents']['text']
        all_words = set([w.lower() for w in curr_tokens])
        # add all words of the current sentence to the currenct dict "compare"
        compare[sent_idx]['all_words']=all_words

    final = compare.copy() #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 größer wird dann wird es auch in final geladen, 
    #final ist dann im näcshten schritt das neue compare
    
    true_words_count = 0 #true_cnt
    for sentence in compare.values():
        true_words_count += len(sentence['true_words'])
    
    with open(rule_patterns_file, "r") as infile:
        lines = infile.readlines()
    # lines contains same patterns as keys of patterns_with_sentidx, but
    # in a slightly different format

    bal_acc_max = 0
    for pattern in patterns_with_prec_rec_sorted_with_bal_acc:
        compare = final.copy()
        for sent_idx,token_idxs in patterns_with_sentidx[pattern[0]].items():
            curr_tokens = data[sent_idx]['sents']['text']
            pred_words = set([word for i,word in enumerate(curr_tokens) \
                               if i in token_idxs])
            pred_words = set([word.lower() for word in pred_words])
            # add predictions of pattern to the currenct dict "compare"
            compare[sent_idx]['curr_predictions'].update(pred_words)
        tp,fp,tn,fn = 0,0,0,0
        for sent in compare.values():
            all_words = sent['all_words']
            pred_words = sent['curr_predictions']
            true_words = sent['true_words']
            for word in all_words:
                if word in true_words and word in pred_words:
                    tp += 1
                if word in true_words and word not in pred_words:
                    fn += 1
                if word not in true_words and word in pred_words:
                    fp += 1
                if word not in true_words and word not in pred_words:
                    tn += 1
        tpr = (tp+ 1e-8) / (tp+fn+ 1e-8)
        tnr = (tn+ 1e-8) / (tn+fp+ 1e-8)
        bal_acc = (tpr+tnr)/2
        if bal_acc > bal_acc_max:
            selected_patterns.append(get_rule_pattern_from_string_v2(lines[pattern[1]['pattern_idx']]))
            final = compare.copy()
            bal_acc_max = bal_acc
            tpr_max = tpr
            tnr_max = tnr
   
    # extraction of patterns from pattern string 
    patterns_l0 = []
    patterns_l1 = []
    patterns_l2 = []
    
    for pattern in selected_patterns:
        if isinstance(pattern,str)==1:
            patterns_l0.append(pattern)
        elif len(pattern)==3:
            patterns_l1.append(pattern)
        elif len(pattern)==2:
            patterns_l2.append(pattern)
        else:
            print(pattern)
            raise NotImplementedError
    
    fout = open(output_rule_selection_file, 'w', encoding='utf-8', newline='\n')
    for pat in patterns_l0:
        fout.write('{}\n'.format(pat))
    for pat in patterns_l1:
        fout.write('{}\n'.format(' '.join(pat)))
    for pat in patterns_l2:
        (pl, ipl), (pr, ipr) = pat
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()
    
    log_file_name = '/'.join(output_rule_selection_file.split('/')[:-1])
    log_file_name += '/log.txt'
    log_string = ('\n###'+output_rule_selection_file.split('/')[-1] +
        '\n[train-data] true positive rate: '+str(round(tpr_max,4))+
        '\n[train-data] true negative rate: '+str(round(tnr_max,4))+
        '\n[train-data] balanced accuracy: '+str(round(bal_acc_max,4))+'\n')
    with open(log_file_name,'a') as logfile:
        logfile.write(log_string)


def get_tp_fp_tn_fn(compare,curr_pred_type):
    tp,fp,tn,fn = 0,0,0,0
    for k,sent in compare.items():
        all_words = sent['all_words']
        pred_word_ids = sent[curr_pred_type]
        true_words = sent['true_words']
        true_words = [w.lower() for w in true_words]
        true_word_ids = list()
        for i,tok in enumerate(all_words):
            if tok.lower() in true_words:
                true_word_ids.append(i)
        for tok_id,word in enumerate(all_words):
            if tok_id in true_word_ids and tok_id in pred_word_ids:
                tp += 1
            if tok_id in true_word_ids and tok_id not in pred_word_ids:
                fn += 1
            if tok_id not in true_word_ids and tok_id in pred_word_ids:
                fp += 1
            if tok_id not in true_word_ids and tok_id not in pred_word_ids:
                tn += 1
    return tp,fp,tn,fn 

def get_tp_fp_tn_fn_CoLA(compare,curr_pred_type):
    tp,fp,tn,fn = 0,0,0,0
    for k,sent in compare.items():
        pred_word_ids = sent[curr_pred_type]
        true_words = sent['true_words']
        true_words = [w.lower() for w in true_words]
        if bool(true_words) and bool(pred_word_ids):
            tp += 1
        if bool(true_words) and not bool(pred_word_ids):
            fn += 1
        if not bool(true_words) and bool(pred_word_ids):
            fp += 1
        if not bool(true_words) and not bool(pred_word_ids):
            tn += 1
    return tp,fp,tn,fn 

def ruleselection_new_on_dev(term_type,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file):
    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    # sort patterns as preparation for selection heuristic
    #  load prec and freq jsons and then rank based on that
    patterns_with_p_r_sorted = ranking_on_dev(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te)
    # patterns_with_p_r_sorted is a sorted list of tuples
    # first entry is a pattern, the second entroy is a dict like
    # {'prec':prec_on_dev,'rec':abs_freq_on_train,'line_str':line}
  
    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)
    compare = dict()
    final = dict()
    for sent_idx, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        # add all words of the current sentence to the currenct dict "compare"
        curr_tokens = data[sent_idx]['sents']['text']
        compare[sent_idx] = {'true_words':true_words,
                             'curr_pred_f1': set(),
                             'curr_pred_bal_acc': set(),
                             'all_words':curr_tokens}
        final[sent_idx] = {'true_words':true_words,
                             'curr_pred_f1': set(),
                             'curr_pred_bal_acc': set(),
                             'all_words':curr_tokens}
    #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 bzw. bal_acc größer wird dann wird es auch in final übertragen, 
    #final ist dann im näcshten schritt das neue compare
    
    selected_patterns_f1 = list()
    #selected_patterns_bal_acc = list()
    f1_max = 0
    #bal_acc_max = 0
    data_check = rulemine.get_data_by_ids_as_named_tuple(data,IDs_tr_dev_te['train'])
    # from time import time
    for pattern,p_r in tqdm(list(patterns_with_p_r_sorted)):
        # t0 = time()
        # get pattern_with_sentidx for current pattern
        pattern_with_sentidx = ruleutils.get_pattern_with_sentidx(pattern,data_check,mine_tool)
        # print(f'time for get_pattern_with_sentidx: {round(time()-t0)} s')
        # t0 = time()

        for sent_idx,token_idxs in pattern_with_sentidx[str(pattern)].items():
            curr_tokens = data[sent_idx]['sents']['text']
            pred_word_ids = set(token_idxs)
            # add predictions of pattern to the currenct dict "compare"
            compare[sent_idx]['curr_pred_f1'].update(set(pred_word_ids))
            #compare[sent_idx]['curr_pred_bal_acc'].update(set(pred_word_ids))
        # print(f'time for add pred in compare dict: {round(time()-t0)} s')
        # t0 = time()
        #if 'CoLA' in rule_patterns_file:
        #    tp,fp,tn,fn  = get_tp_fp_tn_fn_CoLA(compare,curr_pred_type='curr_pred_f1')
        #else:
        tp,fp,tn,fn  = get_tp_fp_tn_fn(compare,curr_pred_type='curr_pred_f1')
        prec = (tp+ 1e-8) / (tp+fp+ 1e-8)
        rec = (tp+ 1e-8) / (tp+fn+ 1e-8)
        f1 = 2*prec*rec/(prec+rec+1e-8)
        # maybe restriction to clearer improvements in f1 [did not improve results on first try though]
        # bad prec of first rules for CoLA, maybe restrict based on prec, maybe validation of code/ term comparison for CoLA or prec-ranking on train again
        if f1 > f1_max:
            selected_patterns_f1.append(pattern)
            # save predictions in curr_pred_f1
            for sent_idx,t in compare.items():
                final[sent_idx]['curr_pred_f1'].update(set(t['curr_pred_f1']))
            f1_max = f1
            p_max = prec
            r_max =   rec
            # write_to_log()
            log_file_name = output_rule_selection_file.replace('.txt','_log_f1.txt')
            pattern_type = ''
            if isinstance(pattern,str):
                pattern_type = 'l0'
            elif len(pattern)==3:
                pattern_type = 'l1'
            elif len(pattern)==2:
                pattern_type = 'l2'
            log_string = (
                'len(selected_patterns):'+str(len(selected_patterns_f1)) +
                '; curr precision: '+str(round(p_max,4))+
                '; curr recall: '+str(round(r_max,4))+
                '; curr f1: '+str(round(f1_max,4))+
                '; curr tp: '+str(round(tp,4))+
                '; curr fp: '+str(round(fp,4))+
                '; curr fn: '+str(round(fn,4))+
                '; curr tn: '+str(round(tn,4))+
                '; precision of pattern: '+str(round(p_r['prec'],4))+
                '; tp, abs. recall of pattern: '+str(round(p_r['rec'],4))+
                '; pattern: '+str(pattern)+
                '; pattern type: '+str(pattern_type)+
                '\n')
            with open(log_file_name,'a') as logfile:
                logfile.write(log_string)
        # print(f'time for computation block f1: {round(time()-t0)} s')
        # t0 = time()

        #if 'CoLA' in rule_patterns_file:
        #    tp,fp,tn,fn  = get_tp_fp_tn_fn_CoLA(compare,curr_pred_type='curr_pred_bal_acc')
        #else:
        #    tp,fp,tn,fn  = get_tp_fp_tn_fn(compare,curr_pred_type='curr_pred_bal_acc')
        #tpr = (tp+ 1e-8) / (tp+fn+ 1e-8)
        #tnr = (tn+ 1e-8) / (tn+fp+ 1e-8)
        #bal_acc = (tpr+tnr)/2
        if False:
        #if bal_acc > bal_acc_max:
            selected_patterns_bal_acc.append(pattern)
            # save predictions in curr_pred_bal_acc
            for sent_idx,t in compare.items():
                final[sent_idx]['curr_pred_bal_acc'].update(set(t['curr_pred_bal_acc']))
            bal_acc_max = bal_acc
            tpr_max = tpr
            tnr_max = tnr    
            log_file_name = output_rule_selection_file.replace('.txt','_log_bal_acc_v2.txt')
            pattern_type = ''
            if isinstance(pattern,str):
                pattern_type = 'l0'
            elif len(pattern)==3:
                pattern_type = 'l1'
            elif len(pattern)==2:
                pattern_type = 'l2'
            log_string = (
                'len(selected_patterns):'+str(len(selected_patterns_bal_acc)) +
                '; curr tpr: '+str(round(tpr_max,4))+
                '; curr tnr: '+str(round(tnr_max,4))+
                '; curr bal_acc: '+str(round(bal_acc_max,4))+
                '; curr tp: '+str(round(tp,4))+
                '; curr fp: '+str(round(fp,4))+
                '; curr fn: '+str(round(fn,4))+
                '; curr tn: '+str(round(tn,4))+
                '; precision of pattern: '+str(round(p_r['prec'],4))+
                '; tp, abs. recall of pattern: '+str(round(p_r['rec'],4))+
                '; pattern: '+str(pattern)+
                '; pattern type: '+str(pattern_type)+
                '\n')
            with open(log_file_name,'a') as logfile:
                logfile.write(log_string)
        # print(f'time for computation block bal_acc: {round(time()-t0)} s')
        # t0 = time()
        for sent_idx,t in final.items():
            compare[sent_idx]['curr_pred_f1']=set(t['curr_pred_f1'])
            #compare[sent_idx]['curr_pred_bal_acc']=set(t['curr_pred_bal_acc'])
        # print(f'time for copy of final dict: {round(time()-t0)} s')
        # t0 = time()
        # assert None not in list(final.values())[0]['curr_pred_f1']
   
    for selected_patterns,file_suffix in [(selected_patterns_f1,'f1'),
                                          #(selected_patterns_bal_acc,'bal_acc_v2')
                                          ]:
        patterns_l0 = []
        patterns_l1 = []
        patterns_l2 = []
        for pattern in selected_patterns:
            if isinstance(pattern,str)==1:
                patterns_l0.append(pattern)
            elif len(pattern)==3:
                patterns_l1.append(pattern)
            elif len(pattern)==2:
                patterns_l2.append(pattern)
            else:
                print(pattern)
                raise NotImplementedError
        curr_outfile = output_rule_selection_file.replace('.txt','_'+file_suffix+'.txt')
        fout = open(curr_outfile, 'w', encoding='utf-8', newline='\n')
        for pat in patterns_l0:
            fout.write('{}\n'.format(pat))
        for pat in patterns_l1:
            fout.write('{}\n'.format(' '.join(pat)))
        for pat in patterns_l2:
            (pl, ipl), (pr, ipr) = pat
            fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
        fout.close()

def ruleselection_new_on_dev_f_beta(term_type,rule_patterns_file,data,IDs_tr_dev_te,output_rule_selection_file):
    if term_type == 'opinion': mine_tool = OpinionMineTool()
    elif term_type == 'aspect': mine_tool = AspectMineTool()
    else: raise NotImplementedError

    # sort patterns as preparation for selection heuristic
    #  load prec and freq jsons and then rank based on that
    patterns_with_p_r_sorted = ranking_on_dev(rule_patterns_file,mine_tool,
                    data,IDs_tr_dev_te)
    # patterns_with_p_r_sorted is a sorted list of tuples
    # first entry is a pattern, the second entroy is a dict like
    # {'prec':prec_on_dev,'rec':abs_freq_on_train,'line_str':line}
  
    true_words_with_sentidx = datautils.__get_true_words_with_sentidx(term_type, data,IDs_tr_dev_te)
    compare = dict()
    final = dict()
    for sent_idx, true_words in true_words_with_sentidx.items():
        true_words = list(set(true_words))
        # add all words of the current sentence to the currenct dict "compare"
        curr_tokens = data[sent_idx]['sents']['text']
        compare[sent_idx] = {'true_words':true_words,
                             'curr_pred_f1': set(),
                             'curr_pred_bal_acc': set(),
                             'all_words':curr_tokens}
        final[sent_idx] = {'true_words':true_words,
                             'curr_pred_f1': set(),
                             'curr_pred_bal_acc': set(),
                             'all_words':curr_tokens}
    #zwei anfangs gleiche dictionaries, 
    #in compare wird immer das neue pattern geladen, 
    #wenn da dann der f1 bzw. bal_acc größer wird dann wird es auch in final übertragen, 
    #final ist dann im näcshten schritt das neue compare
    
    selected_patterns_f1 = list()
    selected_patterns_bal_acc = list()
    f_beta_max = 0
    bal_acc_max = 0
    data_check = rulemine.get_data_by_ids_as_named_tuple(data,IDs_tr_dev_te['train'])
    # from time import time
    for pattern,p_r in tqdm(list(patterns_with_p_r_sorted)):
        # t0 = time()
        # get pattern_with_sentidx for current pattern
        pattern_with_sentidx = ruleutils.get_pattern_with_sentidx(pattern,data_check,mine_tool)
        # print(f'time for get_pattern_with_sentidx: {round(time()-t0)} s')
        # t0 = time()

        for sent_idx,token_idxs in pattern_with_sentidx[str(pattern)].items():
            curr_tokens = data[sent_idx]['sents']['text']
            pred_word_ids = set(token_idxs)
            # add predictions of pattern to the currenct dict "compare"
            compare[sent_idx]['curr_pred_f1'].update(set(pred_word_ids))
            compare[sent_idx]['curr_pred_bal_acc'].update(set(pred_word_ids))
        # print(f'time for add pred in compare dict: {round(time()-t0)} s')
        # t0 = time()
        if 'CoLA' in rule_patterns_file:
            tp,fp,tn,fn  = get_tp_fp_tn_fn_CoLA(compare,curr_pred_type='curr_pred_f1')
        else:
            tp,fp,tn,fn  = get_tp_fp_tn_fn(compare,curr_pred_type='curr_pred_f1')
        prec = (tp+ 1e-8) / (tp+fp+ 1e-8)
        rec = (tp+ 1e-8) / (tp+fn+ 1e-8)
        
        #hardcode fixed beta at the moment (common values 0.5 and 2)
        beta = 0.5
        
        f_beta = (1+beta*beta)*prec*rec/(beta*beta*prec+rec+1e-8)
        # maybe restriction to clearer improvements in f1 [did not improve results on first try though]
        # bad prec of first rules for CoLA, maybe restrict based on prec, maybe validation of code/ term comparison for CoLA or prec-ranking on train again
        if f_beta > f_beta_max:
            selected_patterns_f1.append(pattern)
            # save predictions in curr_pred_f1
            for sent_idx,t in compare.items():
                final[sent_idx]['curr_pred_f1'].update(set(t['curr_pred_f1']))
            f_beta_max = f_beta
            p_max = prec
            r_max =   rec
            # write_to_log()
            log_file_name = output_rule_selection_file.replace('.txt','_log_f1.txt')
            pattern_type = ''
            if isinstance(pattern,str):
                pattern_type = 'l0'
            elif len(pattern)==3:
                pattern_type = 'l1'
            elif len(pattern)==2:
                pattern_type = 'l2'
            log_string = (
                'len(selected_patterns):'+str(len(selected_patterns_f1)) +
                '; curr precision: '+str(round(p_max,4))+
                '; curr recall: '+str(round(r_max,4))+
                '; curr f1: '+str(round(f_beta_max,4))+
                '; curr tp: '+str(round(tp,4))+
                '; curr fp: '+str(round(fp,4))+
                '; curr fn: '+str(round(fn,4))+
                '; curr tn: '+str(round(tn,4))+
                '; precision of pattern: '+str(round(p_r['prec'],4))+
                '; tp, abs. recall of pattern: '+str(round(p_r['rec'],4))+
                '; pattern: '+str(pattern)+
                '; pattern type: '+str(pattern_type)+
                '\n')
            with open(log_file_name,'a') as logfile:
                logfile.write(log_string)
        # print(f'time for computation block f1: {round(time()-t0)} s')
        # t0 = time()

        if 'CoLA' in rule_patterns_file:
            tp,fp,tn,fn  = get_tp_fp_tn_fn_CoLA(compare,curr_pred_type='curr_pred_bal_acc')
        else:
            tp,fp,tn,fn  = get_tp_fp_tn_fn(compare,curr_pred_type='curr_pred_bal_acc')
        tpr = (tp+ 1e-8) / (tp+fn+ 1e-8)
        tnr = (tn+ 1e-8) / (tn+fp+ 1e-8)
        bal_acc = (tpr+tnr)/2
        if bal_acc > bal_acc_max:
            selected_patterns_bal_acc.append(pattern)
            # save predictions in curr_pred_bal_acc
            for sent_idx,t in compare.items():
                final[sent_idx]['curr_pred_bal_acc'].update(set(t['curr_pred_bal_acc']))
            bal_acc_max = bal_acc
            tpr_max = tpr
            tnr_max = tnr    
            log_file_name = output_rule_selection_file.replace('.txt','_log_bal_acc_v2.txt')
            pattern_type = ''
            if isinstance(pattern,str):
                pattern_type = 'l0'
            elif len(pattern)==3:
                pattern_type = 'l1'
            elif len(pattern)==2:
                pattern_type = 'l2'
            log_string = (
                'len(selected_patterns):'+str(len(selected_patterns_bal_acc)) +
                '; curr tpr: '+str(round(tpr_max,4))+
                '; curr tnr: '+str(round(tnr_max,4))+
                '; curr bal_acc: '+str(round(bal_acc_max,4))+
                '; curr tp: '+str(round(tp,4))+
                '; curr fp: '+str(round(fp,4))+
                '; curr fn: '+str(round(fn,4))+
                '; curr tn: '+str(round(tn,4))+
                '; precision of pattern: '+str(round(p_r['prec'],4))+
                '; tp, abs. recall of pattern: '+str(round(p_r['rec'],4))+
                '; pattern: '+str(pattern)+
                '; pattern type: '+str(pattern_type)+
                '\n')
            with open(log_file_name,'a') as logfile:
                logfile.write(log_string)
        # print(f'time for computation block bal_acc: {round(time()-t0)} s')
        # t0 = time()
        for sent_idx,t in final.items():
            compare[sent_idx]['curr_pred_f1']=set(t['curr_pred_f1'])
            compare[sent_idx]['curr_pred_bal_acc']=set(t['curr_pred_bal_acc'])
        # print(f'time for copy of final dict: {round(time()-t0)} s')
        # t0 = time()
        # assert None not in list(final.values())[0]['curr_pred_f1']
   
    for selected_patterns,file_suffix in [(selected_patterns_f1,'f1'),
                                          (selected_patterns_bal_acc,'bal_acc_v2')]:
        patterns_l0 = []
        patterns_l1 = []
        patterns_l2 = []
        for pattern in selected_patterns:
            if isinstance(pattern,str)==1:
                patterns_l0.append(pattern)
            elif len(pattern)==3:
                patterns_l1.append(pattern)
            elif len(pattern)==2:
                patterns_l2.append(pattern)
            else:
                print(pattern)
                raise NotImplementedError
        curr_outfile = output_rule_selection_file.replace('.txt','_'+file_suffix+'.txt')
        fout = open(curr_outfile, 'w', encoding='utf-8', newline='\n')
        for pat in patterns_l0:
            fout.write('{}\n'.format(pat))
        for pat in patterns_l1:
            fout.write('{}\n'.format(' '.join(pat)))
        for pat in patterns_l2:
            (pl, ipl), (pr, ipr) = pat
            fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
        fout.close()
    
    # log_file_name = '/'.join(output_rule_selection_file.split('/')[:-1])
    # log_file_name += '/log.txt'
    # log_string = ('\n###'+output_rule_selection_file.split('/')[-1] +
    #     '\n[train-data] precision: '+str(round(p_max,4))+
    #     '\n[train-data] recall: '+str(round(r_max,4))+
    #     '\n[train-data] f1: '+str(round(f_beta_max,4))+'\n')
    # with open(log_file_name,'a') as logfile:
    #     logfile.write(log_string)

def annotate_own_term_type(rule_patterns_file,data,term_type,output_data_own_term_type_file):
    data_own_term_type_filename = output_data_own_term_type_file
    if term_type == 'aspect':
        mine_tool = AspectMineTool()
        curr_own_term_type_tag = 'ASP'
    elif term_type == 'opinion':
        mine_tool = OpinionMineTool()
        curr_own_term_type_tag = 'OP'
    
    ids = (data.keys())
    corr_sent_hit_rates, pattern_to_corr_sent_hit_rates, pattern_with_sentidx, \
        terms_with_sentidx = datautils.__run_with_mined_rules_combination_MH(mine_tool,
                                rule_patterns_file,data,ids)
    
    data_own_term_type = dict()
    for sentidx,row in data.items():
        ids_extr_tokens = terms_with_sentidx[sentidx]
        tokens = row['sents']['text']
        # assert len(tokens) == len(row['pos_tag_seqs'])
        curr_term_type_tag_seq = [curr_own_term_type_tag+'_O' for i in range(len(tokens))]
        for i in range(len(tokens)):
            if i in ids_extr_tokens:
                curr_term_type_tag_seq[i] = curr_own_term_type_tag+'_A'
        data_own_term_type[sentidx] ={
            curr_own_term_type_tag.lower()+'_tag_seqs': curr_term_type_tag_seq}
    with open(data_own_term_type_filename,'w') as outfile:
        json.dump(data_own_term_type,outfile)
    
    data_comp = datautils.load_data_complete_json(data_own_term_type_filename)
    
    assert data_comp == data_own_term_type

def generate_rulemine_output_excluded_BB(output_file,output_file_excluded_BB,excluded_BB,
                                         term_type):
    excluded_BB_to_tag = {'pos':'POS', 
                          'dep':'DEP', 
                          'prox':'PROX', 
                          'srl':'SRL', 
                          'coref':'COREF', 
                          'wordnet':'SYNSET'}
    excluded_BB_tag = excluded_BB_to_tag[excluded_BB]
    prefix = '_O' if term_type == 'opinion' else '_A'
    if not os.path.isfile(output_file_excluded_BB):
        # load rule_patterns_file
        with open(output_file, "r") as file:
            lines = file.readlines()
        # filter rule_patterns_file
        lines_filtered = list()
        for line in lines:
            pattern = get_rule_pattern_from_string_v2(line)
            if isinstance(pattern,str)==1:
                if not (prefix+excluded_BB_tag in pattern):
                    lines_filtered.append(line)
            elif len(pattern)==3:
                pattern = tuple(pattern)
                if all(not (t.startswith(prefix+excluded_BB_tag)
                            or t.startswith(excluded_BB_tag)) for t in pattern):
                    lines_filtered.append(line)
            elif len(pattern)==2:
                pattern = tuple(pattern)
                tags_pattern = [t for i in pattern for t in i[0]]
                if all(not (t.startswith(prefix+excluded_BB_tag)
                            or t.startswith(excluded_BB_tag)) for t in tags_pattern):
                    lines_filtered.append(line)
            else:
                raise NotImplementedError
        for line in lines_filtered:
            if excluded_BB_tag in line:
                print(excluded_BB_tag)
                raise NotImplementedError
        with open(output_file_excluded_BB, "w") as outfile:
            outfile.writelines(lines_filtered)

    prec_jsons_files_filtered = [output_file_excluded_BB.split('runrulemine')[0]+\
                        'runrulemine_rulegen_l'+str(i)+'withprec.json' for i in range(3)]
    if any(not os.path.isfile(f) for f in prec_jsons_files_filtered):

        # load precision jsons
        prec_jsons_files = [output_file.split('runrulemine')[0]+\
                            'runrulemine_rulegen_l'+str(i)+'withprec.json' for i in range(3)]
    
        with open(prec_jsons_files[0],'r') as rg_f:l0_pattern_to_prec = json.load(rg_f)
        with open(prec_jsons_files[1],'r') as rg_f:l1_pattern_to_prec_str = json.load(rg_f)
        with open(prec_jsons_files[2],'r') as rg_f:l2_pattern_to_prec_str = json.load(rg_f)
    
        l1_pattern_to_prec = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in l1_pattern_to_prec_str.items()])
    
        l2_pattern_to_prec = dict([(tuple(get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in l2_pattern_to_prec_str.items()])
        # filter precision jsons
        l0_pattern_to_prec_filtered = dict()
        l1_pattern_to_prec_filtered = dict()
        l2_pattern_to_prec_filtered = dict()
        for pattern,prec in l0_pattern_to_prec.items():
            if not (prefix+excluded_BB_tag in pattern):
                l0_pattern_to_prec_filtered[pattern]=prec
        for pattern,prec in l1_pattern_to_prec.items():
            if all(not (t.startswith(prefix+excluded_BB_tag)
                            or t.startswith(excluded_BB_tag)) for t in pattern):
                l1_pattern_to_prec_filtered[pattern]=prec
        for pattern,prec in l2_pattern_to_prec.items():
            tags_pattern = [t for i in pattern for t in i[0]]
            if all(not (t.startswith(prefix+excluded_BB_tag)
                        or t.startswith(excluded_BB_tag)) for t in tags_pattern):
                l2_pattern_to_prec_filtered[pattern]=prec        
        with open(prec_jsons_files_filtered[0],'w') as rg_f:
            json.dump(l0_pattern_to_prec_filtered,rg_f)
        
        patterns_to_prec_dict_l1_temp = dict([(' '.join(k),v) \
                                 for k,v in l1_pattern_to_prec_filtered.items()])
        with open(prec_jsons_files_filtered[1],'w') as rg_f:
            json.dump(patterns_to_prec_dict_l1_temp,rg_f)
        
        patterns_to_prec_dict_temp = dict([(' '.join(k[0][0])+' '+str(k[0][1])+' '+' '.join(k[1][0])+' '+str(k[1][1]),v) \
                                 for k,v in l2_pattern_to_prec_filtered.items()])
        with open(prec_jsons_files_filtered[2],'w') as rg_f:
            json.dump(patterns_to_prec_dict_temp,rg_f)

            
    rulegeneration_file = output_file.split('runrulemine')[0]+\
                        'runrulemine_rulegen.json'
    rulegeneration_file_filtered = output_file_excluded_BB.split('runrulemine')[0]+\
                        'runrulemine_rulegen.json'
    if not os.path.isfile(rulegeneration_file_filtered):
        with open(rulegeneration_file,'r') as rg_f:
            rulegeneration_dict = json.load(rg_f)
        patterns_l0_cnts = rulegeneration_dict['patterns_l0_cnts_str']
        patterns_l1_cnts = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict['patterns_l1_cnts_str'].items()])
        patterns_l2_cnts = dict([(tuple(ruleutils.get_rule_pattern_from_string_v2(k)),v) \
                                 for k,v in rulegeneration_dict['patterns_l2_cnts_str'].items()])
        
           
        patterns_l0_cnts_new, patterns_l1_cnts_new, patterns_l2_cnts_new = list(),list(), list()    
        for pattern,cnts in patterns_l0_cnts.items():
            if not (prefix+excluded_BB_tag in pattern):
                patterns_l0_cnts_new.append((pattern,cnts))
        for pattern,cnts in patterns_l1_cnts.items():
            if all(not (t.startswith(prefix+excluded_BB_tag)
                            or t.startswith(excluded_BB_tag)) for t in pattern):
                patterns_l1_cnts_new.append((pattern,cnts))
        for pattern,cnts in patterns_l2_cnts.items():
            tags_pattern = [t for i in pattern for t in i[0]]
            if all(not (t.startswith(prefix+excluded_BB_tag)
                        or t.startswith(excluded_BB_tag)) for t in tags_pattern):
                patterns_l2_cnts_new.append((pattern,cnts))
                
        patterns_l0_cnts_new = dict(patterns_l0_cnts_new)
        patterns_l1_cnts_new = dict(patterns_l1_cnts_new)
        patterns_l2_cnts_new = dict(patterns_l2_cnts_new)
        
        patterns_l0_cnts = patterns_l0_cnts_new
        patterns_l1_cnts = patterns_l1_cnts_new
        patterns_l2_cnts = patterns_l2_cnts_new
        
            
        patterns_l0_cnts_str=dict([(k,v) for k,v in patterns_l0_cnts.items()])
        patterns_l1_cnts_str=dict([(' '.join(k),v) for k,v in patterns_l1_cnts.items()])
        patterns_l2_cnts_str=dict([(' '.join(k[0][0])+' '+str(k[0][1])+' '+' '.join(k[1][0])+' '+str(k[1][1]),v) \
                                   for k,v in patterns_l2_cnts.items()])
        rulegeneration_dict = {'patterns_l0_cnts_str':patterns_l0_cnts_str,
                               'patterns_l1_cnts_str':patterns_l1_cnts_str,
                               'patterns_l2_cnts_str':patterns_l2_cnts_str,
                               }
        with open(rulegeneration_file_filtered,'w') as rg_f:
            json.dump(rulegeneration_dict,rg_f)