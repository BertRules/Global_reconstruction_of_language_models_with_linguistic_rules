import json
import os
from utils import datautils
import rule_extraction_utils
import time

# this script is the controller of the rule extraction process
# this script specifies ids for train/valid/test from _complete.json
# all output (intermediate results, rule patterns) of one run is saved in "./run/"

def main():
    start_time = time.time()
    ## start configuration 
    # dataset = 'laptops_amazon_ae_oe_150k'
    # dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k'
    # dataset = 'laptops_amazon_ae_oe_150k_only_tok_allrel_BB'
    dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k_only_tok_allrel_BB'

    name_run = 'v3'
    ruletype = 'l0l1'
    if not ruletype in ['l0','l1','l2','l0l1','l0l2','l1l2']: ruletype = 'l0l1l2'
    name_run += '_'+ruletype
    num_sentences_tr_dev_te = 150000 # currently 70.000 is maximum number of sentences in the data
    # term_type = 'aspect'
    term_type = 'opinion'
    freq_thres = 10
    bb_config = ['']#,'pos','ner','wordnet','dep','srl','coref','prox']#['','pos','ner','wordnet','dep','srl','coref','prox']
    num_cores = 1 # if ==1, then multiprocessing is not used
    rs_new_on_dev = True # if True: ruleselection is conducted on dev data with f1 and bal_acc_v2 at once
    eval_measure = '' # if rs_new_on_dev == True: this variable is not used
    # eval_measure = 'bal_acc_v2' #'bal_acc' # if eval_measure != 'bal_acc' or eval_measure != 'bal_acc_v2', f1 score is used for eval
    # bal_acc_v2 uses precision and recall for ranking in ruleselection_new_bal_acc_v2
    restrict_in_ruleselection = False
    use_topk_patterns_runrulemine_absfreq_filtering = False
    k_topkabsfreq = 10000000
    
    use_only_patterns_in_rulegen_lXwithprec_json = False
    
    use_topk_patterns_runrulemine_prec_filtering = True
    k_topkprec = 30000
    quit_before_runrulmine_final_output = False
    ## end configuration
    
    if num_sentences_tr_dev_te == 70000: freq_thres = 5
    data_filename = './data/input/'+dataset+'.json'
    data = datautils.load_data_complete_json(data_filename)
    if 'CoLA' in data_filename:
        term_type = 'aspect'
    # if run does not exist, create run folder, otherwise load run folder
    run_folder = './data/run/'+dataset+'_'+term_type+'_'+name_run+'/'
    # run_folder = './data/run/'+dataset+'_'+name_run+'/'
    
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
        # raise NotImplementedError
        keys = list(data.keys())
        keys = keys[:num_sentences_tr_dev_te]
        IDs_tr_dev_te = datautils.generate_random_IDs_tr_dev_te(keys)
        json.dump(IDs_tr_dev_te,open(run_folder+'IDs_tr_dev_te.json','w'))
    else:
        IDs_tr_dev_te = json.load(open(run_folder+'IDs_tr_dev_te.json','r'))
    run_folder_backup = run_folder
    # do 1)-5) for all_BB ('') and every BB excluded ('pos','ner',...)
    for excluded_BB in bb_config:
        
        data = datautils.load_data_complete_json(data_filename)
        if excluded_BB:
            run_folder = run_folder_backup+'excluded_'+excluded_BB+'/'
            if not os.path.isdir(run_folder): os.mkdir(run_folder)
            data = datautils.remove_bb_from_data(data,excluded_BB)
        else:
            run_folder = run_folder_backup
            
        # Vorgehen: 
        # 1) 1st iteration Regelmenge mit runrulemine
        iteration = '1st_it'
        output_file = run_folder+iteration+'_patterns_runrulemine.txt'
        if use_topk_patterns_runrulemine_prec_filtering:
            output_file = output_file.replace('.txt','_topkprec_'+str(k_topkprec)+'.txt')
        if not os.path.isfile(output_file):
            print('generate '+output_file)
            time_start_runrulemine = time.time()
            rule_extraction_utils.runrulemine(run_folder,term_type,
                                          data,IDs_tr_dev_te,output_file,num_cores,use_topk_patterns_runrulemine_prec_filtering,
                                          k_topkprec,
                                          use_topk_patterns_runrulemine_absfreq_filtering,
                                          k_topkabsfreq,
                                          quit_before_runrulmine_final_output,
                                          use_only_patterns_in_rulegen_lXwithprec_json,
                                          ruletype,freq_thres)
            time_end_runrulemine = time.time()
            print('time for runrulemine: '+str(time_end_runrulemine -time_start_runrulemine))

        # 2) 1st iteration Ruleselection
        iteration = '1st_it'
        rule_patterns_file = run_folder+iteration+'_patterns_runrulemine.txt'
        output_rule_selection_file = run_folder+iteration+'_patterns_ruleselection.txt'
        if use_topk_patterns_runrulemine_prec_filtering:
            rule_patterns_file = rule_patterns_file.replace('.txt','_topkprec_'+str(k_topkprec)+'.txt')
            output_rule_selection_file = output_rule_selection_file.replace('.txt','_topkprec_'+str(k_topkprec)+'.txt')
        if eval_measure == 'bal_acc':
            output_rule_selection_file = output_rule_selection_file.replace('.txt','_bal_acc.txt')
        elif eval_measure == 'bal_acc_v2':
            output_rule_selection_file = output_rule_selection_file.replace('.txt','_bal_acc_v2.txt')
        if not os.path.isfile(output_rule_selection_file):
            print('generate '+output_rule_selection_file)
            time_start_ruleselection = time.time()
            if rs_new_on_dev:
                rule_extraction_utils.ruleselection_new_on_dev(term_type,rule_patterns_file,data,
                                            IDs_tr_dev_te,output_rule_selection_file)
            else:
                if eval_measure == 'bal_acc':
                    rule_extraction_utils.ruleselection_new_bal_acc(term_type,rule_patterns_file,data,
                                                IDs_tr_dev_te,output_rule_selection_file,
                                                restrict_in_ruleselection)
                elif eval_measure == 'bal_acc_v2':
                    rule_extraction_utils.ruleselection_new_bal_acc_v2(term_type,rule_patterns_file,data,
                                                IDs_tr_dev_te,output_rule_selection_file,
                                                restrict_in_ruleselection)
                else:
                    rule_extraction_utils.ruleselection_new(term_type,rule_patterns_file,data,
                                                IDs_tr_dev_te,output_rule_selection_file,
                                                restrict_in_ruleselection)
            time_end_ruleselection = time.time()
            print('time for ruleselection: '+str(time_end_ruleselection -time_start_ruleselection))
    
    end_time = time.time()
    print('time: '+str(end_time-start_time))

if __name__ == "__main__":
  main()