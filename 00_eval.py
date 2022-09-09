# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from utils import datautils
from rule.ruleutils import load_rule_patterns_file
from rule.aspectrulemine import AspectMineTool
from rule.opinionrulemine import OpinionMineTool
import time
# # declaration of methods

def count_tokens(T):
    n = 0
    for v in T.values():
        n+=len(v)
    return n

def add_to_T_setBB(T_setBB,sentidx_to_extr_tokenidx):
    for sentidx,add_extr_tokenidx in sentidx_to_extr_tokenidx.items():
        curr_extr_tokenidx = set(T_setBB.setdefault(sentidx,list()))
        curr_extr_tokenidx.update(set(add_extr_tokenidx))
        T_setBB[sentidx] = list(curr_extr_tokenidx)
    return T_setBB

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

def aggregate_to_single_T_ruleset(patterns_to_sentidx_to_extr_tokenidx):
    T_ruleset = dict()
    for pattern_str,sentidx_to_extr_tokenidx in patterns_to_sentidx_to_extr_tokenidx.items():
        T_ruleset = add_to_T_setBB(T_ruleset,sentidx_to_extr_tokenidx)
    return T_ruleset

def get_patterns_with_sentidx(data,ids,rule_patterns_file,term_type):

    if term_type == 'aspect': mine_tool = AspectMineTool()
    elif term_type == 'opinion': mine_tool = OpinionMineTool()
    _, _, patterns_with_sentidx, _ = datautils.__run_with_mined_rules_combination_MH(
                mine_tool,rule_patterns_file,data,ids,)
    return patterns_with_sentidx

def get_pr_and_rec(data,ids,rule_patterns_file,term_type):
    # load data
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

    patterns_to_sentidx_to_extr_tokenidx = get_patterns_with_sentidx(data,ids,rule_patterns_file,
                                                                      term_type)
    sentidx_to_extr_tokenidx_rules = aggregate_to_single_T_ruleset(patterns_to_sentidx_to_extr_tokenidx)
    
    if 'CoLA' in rule_patterns_file:
        # CoLA is sentence classification -> so binary classification of sentences
        # -> classes are  tokenidx=[0] and tokenidx=[]
        for d in [sentidx_to_extr_tokenidx_BERT,
                  sentidx_to_extr_tokenidx_rules]:
            for k,tokenidx in list(d.items()):
                if tokenidx:
                    tokenidx = [0]
                assert tokenidx in [[],[0]]
                d[k]=tokenidx
            

    
    ## EVALUATION
    # "disentangle positive classifications of BERT":
    #    calculate precision and recall with "true"=BERT labels and prediction=rule labels
    # "disentangle false negative classifications of BERT":
    #    calculate precision and recall with
    #   "true"=true labels intersection BERT not labels
    #    and prediction=rule labels
    
    T_lm = sentidx_to_extr_tokenidx_BERT.copy()
    T_sigma = sentidx_to_extr_tokenidx_rules.copy()
    count_patterns = len(patterns_to_sentidx_to_extr_tokenidx.keys())
    
    ## calculate tp, fp, fn
    tp = count_tokens(intersect_tokens(T_sigma,T_lm))
    fp = count_tokens(difference_tokens(T_sigma,T_lm))
    fn = count_tokens(difference_tokens(T_lm,T_sigma))
    if 'CoLA' in rule_patterns_file:
        tn = len(ids)-tp-fp-fn
    else:
        n_all_instances = 0
        for i in ids:
            n_all_instances += len(data[i]['sents']['text'])
        tn = n_all_instances-tp-fp-fn

    
    ## calculate precision and recall
    pr = tp/(tp+fp) if tp>0 else 0
    rec = tp/(tp+fn) if tp>0 else 0
    # print("precision= "+str(round(pr*10000)/100)+"%")
    # print("recall= "+str(round(rec*10000)/100)+"%")
    return pr,rec,tp,fp,fn,tn,count_patterns,sentidx_to_extr_tokenidx_BERT,sentidx_to_extr_tokenidx_rules

def compute_results_with_rule_selection_files(data,ids,rule_patterns_file,
                                              term_type):
    # compute precision and recall for the complete pattern set of rule selection
    pr,rec,tp,fp,fn,tn,count_patterns,sentidx_to_extr_tokenidx_BERT,\
        sentidx_to_extr_tokenidx_rules = get_pr_and_rec(data,ids,rule_patterns_file,
                                                    term_type)
    f1=2*(pr+ 1e-8)*(rec+ 1e-8)/(pr+rec+ 2e-8)
    tpr = (tp + 1e-8)/(tp + fn + 1e-8)
    tnr = (tn + 1e-8)/(tn + fp + 1e-8)
    bal_acc = (tpr+tnr) / 2
    l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
    count_l0,count_l1,count_l2 = len(l0_rules),len(l1_rules),len(l2_rules)
    results = {'precision':pr,'recall':rec,
                         'f1':f1,
                         'tpr':tpr,
                         'tnr':tnr,
                         'bal_acc':bal_acc,
                         'tp':tp,
                         'fp':fp,
                         'fn':fn,
                         'tn':tn,
                         'count_patterns':count_patterns,
                         'count_l0':count_l0,
                         'count_l1':count_l1,
                         'count_l2':count_l2
                         }
    return results,sentidx_to_extr_tokenidx_BERT,sentidx_to_extr_tokenidx_rules

def compute_results_balanced_accuracy_with_rule_selection_files(data,ids,rule_patterns_file,
                                              term_type):
    # compute precision and recall for the complete pattern set of rule selection
    _,_,tp,fp,fn,tn,count_patterns,sentidx_to_extr_tokenidx_BERT,\
        sentidx_to_extr_tokenidx_rules = get_pr_and_rec(data,ids,rule_patterns_file,
                                                    term_type)
    tpr = (tp + 1e-8)/(tp + fn + 1e-8)
    tnr = (tn + 1e-8)/(tn + fp + 1e-8)
    bal_acc = (tpr+tnr) / 2
    results = {'tpr':tpr,'tnr':tnr,
                         'bal_acc':bal_acc,
                         'tp':tp,
                         'fp':fp,
                         'fn':fn,
                         'tn':tn,
                         'count_patterns':count_patterns,
                         }
    return results,sentidx_to_extr_tokenidx_BERT,sentidx_to_extr_tokenidx_rules


##### START OF MAIN-SCRIPT
## EVAL CONFIG
start_time = time.time()


## configuration 

# dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k'
dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k_only_tok_allrel_BB'
# dataset = 'laptops_amazon_ae_oe_150k'
# dataset = 'laptops_amazon_ae_oe_150k_only_tok_allrel_BB'


name_run = 'v3'
iteration = '1st_it'
# term_type = 'aspect'
term_type = 'opinion'
rs_new_on_dev = True # if True: ruleselection is conducted on dev data with f1 and bal_acc_v2 at once
eval_measure = '' # if rs_new_on_dev == True: this variable is not used
# eval_measure = 'bal_acc_v2'# 'bal_acc' # if eval_measure != 'bal_acc', f1 score is used for eval
use_topk_patterns_runrulemine_prec_filtering = True
k_topkprec = 30000
## end of configuration



data_filename = './data/input/'+dataset+'.json'


## load data
data = datautils.load_data_complete_json(data_filename)
run_folder = './data/run/'+dataset+'_'+term_type+'_'+name_run+'/'
IDs_tr_dev_te = json.load(open(run_folder+'IDs_tr_dev_te.json','r'))

run_folder_backup = run_folder
files_in_run_folder = os.listdir(run_folder)
excluded_BBs = [f.split('excluded_')[-1] \
                for f in files_in_run_folder if f.startswith("excluded_")]
# excluded_BBs = []#['pos','dep','wordnet']
excluded_BBs.append("")
excluded_BBs.sort()
    
for partitions in [['test']]:#,['train']]:
    outfilename = run_folder_backup+iteration+'_results_'+'_'.join(partitions)+'.csv'
    if use_topk_patterns_runrulemine_prec_filtering:
        outfilename = outfilename.replace('.csv','_topkprec_'+str(k_topkprec)+'.csv')
    if eval_measure == 'bal_acc':
        outfilename = outfilename.replace('.csv','_bal_acc.csv')
    elif eval_measure == 'bal_acc_v2':
        outfilename = outfilename.replace('.csv','_bal_acc_v2.csv')
    if os.path.isfile(outfilename):
        df_already_calculated = pd.read_csv(outfilename,sep=';')
        names_rows = list(df_already_calculated['Unnamed: 0'].values)
        excluded_BBs_already_calculated = list(set([s.split('_')[-1] for s in names_rows]))
        excluded_BBs = list(set(excluded_BBs).difference(excluded_BBs_already_calculated))

    # do evaluation for all_BB ('') and every BB excluded ('pos','ner',...)
    # only for excluded_BBs which results are not yet calculated for
    # excluded_BBs.append('')
    excluded_BBs = list(set(excluded_BBs))
    # raise NotImplementedError
    for excluded_BB in excluded_BBs:#['','pos','ner','wordnet','dep','srl','coref','prox']:
        data = datautils.load_data_complete_json(data_filename)
        if excluded_BB:
            run_folder = run_folder_backup+'excluded_'+excluded_BB+'/'
            if not os.path.isdir(run_folder): raise NotImplementedError
            data = datautils.remove_bb_from_data(data,excluded_BB)
        else:
            run_folder = run_folder_backup

        try:iteration
        except:iteration = '2nd_it'
        if 'CoLA' in data_filename or iteration == '1st_it':
            iteration = '1st_it'
        else:
            # add output_data_own_term_type_file to data
            output_data_own_term_type_file = run_folder + '1st_it_own_'+term_type+'.json'
            data_own_term_type = datautils.load_data_complete_json(output_data_own_term_type_file)
            for k,v in data.items():
                curr_own_term_type = data_own_term_type[k]
                for k1,v1 in curr_own_term_type.items():
                    v[k1] = v1
            # specify filename of rule patterns file that shall be evaluated
        rule_patterns_file = run_folder+iteration+'_patterns_ruleselection.txt'
        rule_patterns_files=list()
        if rs_new_on_dev:
            rule_patterns_files.append(rule_patterns_file.replace('.txt','_topkprec_'+str(k_topkprec)+
                                                                  '_f1.txt'))
            rule_patterns_files.append(rule_patterns_file.replace('.txt','_topkprec_'+str(k_topkprec)+
                                                                  '_bal_acc_v2.txt'))
        else:
            if use_topk_patterns_runrulemine_prec_filtering:
                rule_patterns_file = rule_patterns_file.replace('.txt','_topkprec_'+str(k_topkprec)+'.txt')
            if eval_measure == 'bal_acc':
                rule_patterns_file = rule_patterns_file.replace('.txt','_bal_acc.txt')
            elif eval_measure == 'bal_acc_v2':
                rule_patterns_file = rule_patterns_file.replace('.txt','_bal_acc_v2.txt')
            rule_patterns_files.append(rule_patterns_file)
        # eval with patterns from rule selection (per BB set)
        ids = []
        print(f'eval for rulepatternsfile: {rule_patterns_file}')
        for partition in partitions:
            ids.extend(IDs_tr_dev_te[partition])
        
        results = dict()
        if os.path.isfile(outfilename):
            df_already_calculated = pd.read_csv(outfilename,sep=';')
            for i in df_already_calculated.iterrows():
                results[i[1][0]] = i[1][1:].to_dict()


        for rule_patterns_file in rule_patterns_files:
            if 'bal_acc_v2.txt' in rule_patterns_file:
                continue
            res,sentidx_to_extr_tokenidx_BERT,sentidx_to_extr_tokenidx_rules \
                = compute_results_with_rule_selection_files(data,ids,rule_patterns_file, term_type)
            rs_type_str = rule_patterns_file.replace('.txt','').split('_topkprec_'+str(k_topkprec)+'_')[-1]
            results[rs_type_str+'_'+'_'.join(partitions)+'excluded_'+excluded_BB] = res
            # res,sentidx_to_extr_tokenidx_BERT,sentidx_to_extr_tokenidx_rules \
            #     = compute_results_balanced_accuracy_with_rule_selection_files(data,ids,rule_patterns_file, term_type)
            # results['_'.join(partitions)+'_bal_acc_'+'excluded_'+excluded_BB] = res
            res_detail = dict()
            res_per_sentence = dict()
            for k,v in sentidx_to_extr_tokenidx_BERT.items():
                if k in sentidx_to_extr_tokenidx_rules.keys():
                    v2 = sentidx_to_extr_tokenidx_rules[k]
                else:
                    v2 = []
                res_detail[k] = {'BERT':str(v),'rules':str(v2)}
                curr_tokens = data[k]['sents']['text']
                tp,fp,fn,tn = 0,0,0,0
                for i in range(len(curr_tokens)):
                    if i in v and i in v2: tp += 1
                    elif i in v and i not in v2: fn += 1
                    elif i not in v and i in v2: fp += 1
                    else: tn += 1
                res_per_sentence[k]={'tokens':curr_tokens,
                                     'BERT':v,'rules':v2,
                                     'tp':tp,
                                     'fp':fp,
                                     'fn':fn,
                                     'tn':tn
                    }
            # outfilename_res_per_sentence = outfilename.replace('.csv','_res_per_sentence.json')
            # with open (outfilename_res_per_sentence,'w') as of:
            #     json.dump(res_per_sentence,of)
            # # # df_res_d = pd.DataFrame.from_dict(res_detail,orient="index")
            # # # outfilename = run_folder+'results_detail_'+'_'.join(partitions)+excluded_BB+'.csv'
            # # # df_res_d.to_csv(outfilename,sep=';')
            # raise NotImplementedError
        df = pd.DataFrame.from_dict(results,orient="index")
        df.to_csv(outfilename,sep=';')
end_time = time.time()
print('time: '+str(end_time-start_time))