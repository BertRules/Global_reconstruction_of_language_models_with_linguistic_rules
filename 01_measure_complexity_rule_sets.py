from rule.ruleutils import load_rule_patterns_file
import numpy as np

# dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k'
dataset = 'laptops_amazon_ae_oe_150k'
# dataset = 'restaurants_yelp_dsc_cl_ae_oe_150k_only_tok_allrel_BB'
# dataset = 'laptops_amazon_ae_oe_150k_only_tok_allrel_BB'

name_run = 'v3'
iteration = '1st_it'
# term_type = 'aspect'
term_type = 'opinion'
k_topkprec = 30000


run_folder = './data/run/'+dataset+'_'+term_type+'_'+name_run+'/'

iteration == '1st_it'
            # specify filename of rule patterns file that shall be evaluated
rule_patterns_file = run_folder+iteration+'_patterns_ruleselection_topkprec_'+\
    str(k_topkprec)+'_f1.txt'

l0_rules,l1_rules, l2_rules = load_rule_patterns_file(rule_patterns_file)
count_l0,count_l1,count_l2 = len(l0_rules),len(l1_rules),len(l2_rules)

measures = dict()

# if "l2" in name_run: raise NotImplementedError
rules = []
for r in l0_rules:
    rules.append([r])
rules.extend(l1_rules)
for r in l2_rules:
    line = []
    for p in r:
        for a in p[0]:
            line.append(a)
    rules.append(line)
'''
- number of rules
- average rule length
    - # antecedents
    (- # of relations)
- number of unique BB values
'''
# number of rules
measures['number of rules']=len(rules)
measures['number of l0 rules']=len(l0_rules)
measures['number of l1 rules']=len(l1_rules)
measures['number of l2 rules']=len(l2_rules)

# average rule length
## avg number of antecedents
number_of_antecedents_per_rule = []
num_ant_per_count = {1:0,2:0,3:0}
for r in rules:
    num_ant = 0
    for a in r:
        if a != "WORD":
            num_ant += 1
    number_of_antecedents_per_rule.append(num_ant)
    curr_count = num_ant_per_count.setdefault(num_ant,0)
    num_ant_per_count[num_ant] = curr_count+1

measures['avg number of antecedents']=np.mean(number_of_antecedents_per_rule)
for i in range(6):
    measures['rules with '+str(i+1)+' antec']=0
for k,v in num_ant_per_count.items():
    measures['rules with '+str(k)+' antec']=v

# number of unique BB values
bb_unique = set()
for r in rules:
    num_ant = 0
    for a in r:
        if a != "WORD":
            bb_unique.add(a)
measures['number of unique BB values']=len(bb_unique)

print("\t".join([k for k in measures.keys()]))
print("\t".join([str(v) for v in measures.values()]))


