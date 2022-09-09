from rule import ruleutils


class OpinionMineTool:
    def __init__(self):
        pass
    
    @staticmethod
    def __get_term_pos_type_pos(term_pos_tags):
        for t in term_pos_tags:
            if t in ruleutils.JJ_POS_TAGS:
                return 'POS_J'
        for t in term_pos_tags:
            if t in ruleutils.RB_POS_TAGS:
                return 'POS_R'
        for t in term_pos_tags:
            if t in ruleutils.VB_POS_TAGS:
                return 'POS_V'
        return None
    
  #  @staticmethod
#     def __get_term_pos_type_ner(term_pos_tags):
#         for t in term_pos_tags:
#             if t in ruleutils.NEB_TAGS_OP:
#                 return 'NER_B'
# #        for t in term_pos_tags:
# #            if t in ruleutils.NEO_TAGS_OP:
# #                return 'NER_O'               
#         return None
    
    @staticmethod    
    def __get_term_pos_type_ner(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NER_O':
                return 'NER_O'
        for t in term_pos_tags:
            if t != 'NER_O':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_asp(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'ASP_A':
                return 'ASP_A'
        for t in term_pos_tags:
            if t == 'ASP_O':
                return 'ASP_O'
        return None
    
    @staticmethod
    def __get_term_pos_type_word(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'WORD':
                return 'WORD'

        return None
    
    @staticmethod    
    def __get_term_pos_type_token(term_pos_tags):
        for t in term_pos_tags:
            return t
        return None

    @staticmethod    
    def __get_term_pos_type_wordnet(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_SYN':
                return 'NO_SYN'
        for t in term_pos_tags:
            if t != 'NO_SYN':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_rhyp(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_RHYP':
                return 'NO_RHYP'
        for t in term_pos_tags:
            if t != 'NO_RHYP':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_ant(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_ANT':
                return 'NO_ANT'
        for t in term_pos_tags:
            if t != 'NO_ANT':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_rstLEAF(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_LEAF_RST':
                return 'NO_LEAF_RST'
        for t in term_pos_tags:
            if t != 'NO_LEAF_RST':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_rstTOP(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_TOP_RST':
                return 'NO_TOP_RST'
        for t in term_pos_tags:
            if t != 'NO_TOP_RST':
                return t
        return None
    
    @staticmethod    
    def __get_term_pos_type_op(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'OP_A':
                return 'OP_A'
        for t in term_pos_tags:
            if t == 'OP_O':
                return 'OP_O'
        return None
  

    def get_patterns_from_term_combination_MH(self, idx_span, related_rels_idxs, curr_rels, curr_tags,
                                              ruletype):
        widx_beg, widx_end = idx_span
        opinion_word_wc_dict = dict()
        for k,v in curr_tags.items():
            term_tags = set([v[i] for i in range(widx_beg, widx_end)])# get al pos tags of term_word
            # die folgenden __get_term_pos_type methoden aggregieren gewisse Tags
            mode = k.replace('_tag_seqs','')

            if mode=='ner':
                term_tag_type = OpinionMineTool.__get_term_pos_type_ner(term_tags) #OpinionMineTool.
            elif mode=='pos':
                term_tag_type = OpinionMineTool.__get_term_pos_type_pos(term_tags)
            elif mode == 'asp':
                term_tag_type = OpinionMineTool.__get_term_pos_type_asp(term_tags)                
            elif mode=='cons':
                term_tag_type = OpinionMineTool.__get_term_pos_type_cons(term_tags)
            elif mode=='coref':
                term_tag_type = OpinionMineTool.__get_term_pos_type_coref(term_tags)
            elif mode=='op':
                term_tag_type = OpinionMineTool.__get_term_pos_type_op(term_tags)
            elif mode=='word':
                term_tag_type = OpinionMineTool.__get_term_pos_type_word(term_tags)                
            elif mode=='wordnet':
                term_tag_type = OpinionMineTool.__get_term_pos_type_wordnet(term_tags)
            elif mode == 'ant':
                term_tag_type = OpinionMineTool.__get_term_pos_type_ant(term_tags)
            elif mode == 'rhyp':
                term_tag_type = OpinionMineTool.__get_term_pos_type_rhyp(term_tags)
            elif mode == 'rstTOP':
                term_tag_type = OpinionMineTool.__get_term_pos_type_rstTOP(term_tags)
            elif mode == 'rstLEAF':
                term_tag_type = OpinionMineTool.__get_term_pos_type_rstLEAF(term_tags)
            elif mode=='token':
                term_tag_type = OpinionMineTool.__get_term_pos_type_token(term_tags)

            else:
                raise NotImplementedError()
            if term_tag_type is not None:
                opinion_word_wc_dict[k] = '_O{}'.format(term_tag_type)

        if opinion_word_wc_dict.keys():
            related_rels_dict=dict()
            for k,v in related_rels_idxs.items():
                related_rels_dict[k] = [curr_rels[k][idx] for idx in v]
            if 'l0' in ruletype:
                patterns_l0 = set(v for v in opinion_word_wc_dict.values())
            else:
                patterns_l0 = {}
            if 'l1' in ruletype:
                patterns_l1 = self.__patterns_from_l1_dep_tags_combination_MH(opinion_word_wc_dict, related_rels_dict,
                                                           curr_tags, idx_span) #self.
            else: patterns_l1 = {}
            patterns_new = set()
            for p in patterns_l1:
                flag__O = False
                for l in [1,2]:
                    if p[l].startswith("_O"):
                        flag__O = True
                if flag__O:
                    patterns_new.add(p)
            patterns_l1 = patterns_new
            if 'l2' in ruletype:
                related_l2 = ruleutils.find_related_l2_rels_combination_MH(related_rels_idxs, curr_rels)
                patterns_l2 = self.__patterns_from_l2_dep_tags_combination_MH(opinion_word_wc_dict, related_l2, curr_tags, idx_span) #self.
            else:
                patterns_l2 = {}
            return patterns_l0,patterns_l1, patterns_l2
        else:
            return set(),set(), set()
    
    @staticmethod
    def get_candidate_terms_pos(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.JJ_POS_TAGS, ruleutils.RB_POS_TAGS, ruleutils.VB_POS_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
                    terms.append(w)
        return terms

    @staticmethod
    def get_candidate_terms_ner(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        #pos_tag_sets = [ruleutils.NEB_TAGS_AS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            #for pos_tag_set in pos_tag_sets:
                if pos_tag != 'NER_O':
                    terms.append(w)
        return terms

    @staticmethod
    def get_candidate_terms_wordnet(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        #pos_tag_sets = [ruleutils.NEB_TAGS_OP]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            #for pos_tag_set in pos_tag_sets:
                if pos_tag != 'NO_SYN':
                    terms.append(w)
        return terms
    
    @staticmethod    
    def get_candidate_terms_asp(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.ASPB_TAGS, ruleutils.ASPO_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
                    terms.append(w)
        return terms
    
    @staticmethod    
    def get_candidate_terms_word(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.WORD_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
                    terms.append(w)
        return terms
    
    @staticmethod    
    def get_candidate_terms_op(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.OPO_TAGS, ruleutils.OPA_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
                    terms.append(w)
        return terms

    @staticmethod
    def get_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_O'):
            aspect_position = 1
        elif pattern[2].startswith('_O'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return w

    @staticmethod
    def get_term_from_matched_pattern_combination_MH(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_O'):
            aspect_position = 1
        elif pattern[2].startswith('_O'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return w
      

    
    @staticmethod
    def get_termidx_from_matched_l0pattern_combination_LK(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern.startswith('_O'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return widx


    @staticmethod
    def get_termidx_from_matched_pattern_combination_LK(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_O'):
            aspect_position = 1
        elif pattern[2].startswith('_O'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return widx

    @staticmethod
    def match_pattern_word_pos(pw, w, pos_tag):
        if pw == '_OPOS_J' and pos_tag in ruleutils.JJ_POS_TAGS:
            return True
        if pw == '_OPOS_R' and pos_tag in ruleutils.RB_POS_TAGS:
            return True
        if pw == '_OPOS_V' and pos_tag in ruleutils.VB_POS_TAGS:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    @staticmethod
    def match_pattern_word_ner(pw, w, pos_tag):
        if pw != 'NER_O' and pos_tag != 'NER_O':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_asp(pw, w, pos_tag):
        if pw == '_OASP_O' and pos_tag in ruleutils.ASPO_TAGS:
            return True
        if pw == '_OASP_A' and pos_tag in ruleutils.ASPB_TAGS:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w

    @staticmethod
    def match_pattern_word_token(pw, w, pos_tag):
        if pw.startswith('_O'):
            return pw[2:] == pos_tag
        else:
            return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_wordnet(pw, w, pos_tag):
        if pw != 'NO_SYN' and pos_tag != 'NO_SYN':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
        
    @staticmethod
    def match_pattern_word_ant(pw, w, pos_tag):
        if pw != 'NO_ANT' and pos_tag != 'NO_ANT':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rhyp(pw, w, pos_tag):
        if pw != 'NO_RHYP' and pos_tag != 'NO_RHYP':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rstLEAF(pw, w, pos_tag):
        if pw != 'NO_LEAF_RST' and pos_tag != 'NO_LEAF_RST':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rstTOP(pw, w, pos_tag):
        if pw != 'NO_TOP_RST' and pos_tag != 'NO_TOP_RST':
            if pw.startswith('_O'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_op(pw, w, pos_tag):
        if pw == '_OOP_O' and pos_tag == 'OP_O':
            return True
        if pw == '_OOP_A' and pos_tag == 'OP_A':
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    @staticmethod
    def match_pattern_word_word(pw, w, pos_tag):
        if pw == '_OWORD' and pos_tag == 'WORD':
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
#    @staticmethod
#    def match_pattern_word_cons_v1(pw, w, pos_tag):
#        if pw == '_OCONS_S' and pos_tag in ['S0']:
#            return True
#        if pw == '_OCONS_N' and pos_tag in ['NP0']:
#            return True
#        if pw == '_OCONS_V' and pos_tag in ['VP0']:
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w
#    
#    @staticmethod
#    def match_pattern_word_cons(pw, w, pos_tag):
#        if pw == '_OCONS_0' and pos_tag.endswith('0'):
#            return True
#        if pw == '_OCONS_1' and pos_tag.endswith('1'):
#            return True
#        if pw == '_OCONS_2' and pos_tag.endswith('2'):
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w

#    @staticmethod
#    def match_pattern_word_coref(pw, w, pos_tag):
#        if pw == '_OCOREF_O' and pos_tag in ['O']:
#            return True
#        if pw == '_OCOREF_1' and pos_tag in ['1']:
#            return True
#        if pw == '_OCOREF_2' and pos_tag in ['2']:
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w
    
    # @staticmethod
    # def match_pattern_word_srl(pw, w, pos_tag):
    #     if pw == '_OSRL_O' and pos_tag in ['O']:
    #         return True
    #     if pw == '_OSRL_SR' and pos_tag not in ['O']:
    #         return True
    #     return pw == w

    @staticmethod
    def get_terms_by_matching(dep_tags, pos_tags, sent_text, terms_vocab):
        terms = list()
        for t in terms_vocab:
            pbeg = sent_text.find(t)
            if pbeg < 0:
                continue
            pend = pbeg + len(t)
            if pbeg > 0 and sent_text[pbeg - 1].isalpha():
                continue
            if pend < len(sent_text) and sent_text[pend].isalpha():
                continue
            terms.append(t)
        return terms

    @staticmethod
    def terms_list_from_sents(sents):
        opinion_terms_list = list()
        for sent in sents:
            opinion_terms_list.append([t.lower() for t in sent.get('opinions', list())])
        return opinion_terms_list

    @staticmethod
    def term_ids_dict_from_sents(sents):
        opinion_term_ids_dict = dict()
        for sent in sents:
            opinion_terms = sent.get('opinions', list())
            opinion_terms = [w.lower() for w in opinion_terms]
            toks = sent['text']
            tokenidx = list()
            for i,tok in enumerate(toks):
                if tok.lower() in opinion_terms:
                    tokenidx.append(i)
                
            opinion_term_ids_dict[sent['id']]=tokenidx
        return opinion_term_ids_dict

    @staticmethod
    def __patterns_from_l1_dep_tags(opinion_word_wc, related_dep_tags, pos_tags, term_word_idx_span):
        widx_beg, widx_end = term_word_idx_span
        # print(related_dep_tags)
        patterns = set()
        for dep_tag in related_dep_tags:
            rel, (igov, wgov), (idep, wdep) = dep_tag
            if widx_beg <= igov < widx_end:
                # if govenor in opinion term
                patterns.add((rel, opinion_word_wc, wdep)) #baustein "exaktes wort"
                patterns.add((rel, opinion_word_wc, pos_tags[idep])) #baustein "pos"_tag
            elif widx_beg <= idep < widx_end:
                # if dependent in opinion term
                patterns.add((rel, wgov, opinion_word_wc)) #baustein "exaktes wort"
                patterns.add((rel, pos_tags[igov], opinion_word_wc)) #baustein "pos"_tag
            else:
                # if neither governor nor dependent is in opinion term
                patterns.add((rel, wgov, wdep)) #baustein "exaktes wort"
                patterns.add((rel, pos_tags[igov], wdep)) #baustein "pos"_tag
                patterns.add((rel, wgov, pos_tags[idep])) #baustein "pos"_tag
        return patterns


    @staticmethod
    def __patterns_from_l0_dep_tags_combination_MH(opinion_word_wc_dict,
                                                       curr_tags, idx_span):
        widx_beg, widx_end = idx_span
        # print(related_dep_tags)
        patterns = set()
        raise NotImplementedError
        for tag_idx, (tag_type,tags_v) in enumerate(curr_tags.items()):
            if widx_beg <= tag_idx < widx_end:
                # if word is opinion term
                if tag_type in opinion_word_wc_dict.keys():
                    patterns.add((opinion_word_wc_dict[tag_type]))
                    # combination
        return patterns

    @staticmethod
    def __patterns_from_l1_dep_tags_combination_MH(opinion_word_wc_dict, related_rels_dict,
                                                       curr_tags, idx_span):
        widx_beg, widx_end = idx_span
        # print(related_dep_tags)
        patterns = set()
        for _,rels_v in related_rels_dict.items():
            for rel_v in rels_v:
                rel, (igov, wgov), (idep, wdep) = rel_v
                for tag_type_gov,tags_v_gov in curr_tags.items():
                    for tag_type_dep,tags_v_dep in curr_tags.items():
                        if widx_beg <= igov < widx_end:
                            # if govenor in opinion term
                            if tag_type_gov in opinion_word_wc_dict.keys():
                                patterns.add((rel, opinion_word_wc_dict[tag_type_gov], tags_v_dep[idep]))
                                # combination
                        elif widx_beg <= idep < widx_end:
                            # if dependent in opinion term
                            if tag_type_dep in opinion_word_wc_dict.keys():
                                patterns.add((rel, tags_v_gov[igov], opinion_word_wc_dict[tag_type_dep])) # combination
                        patterns.add((rel, tags_v_gov[igov], tags_v_dep[idep]))
        return patterns

    def __patterns_from_l2_dep_tags(self, opinion_word_wc, related_dep_tag_tups, pos_tags, term_word_idx_span):
        # widx_beg, widx_end = term_word_idx_span
        patterns = set()
        for dep_tag_i, dep_tag_j in related_dep_tag_tups:
            patterns_i = self.__patterns_from_l1_dep_tags(
                opinion_word_wc, [dep_tag_i], pos_tags, term_word_idx_span)
            patterns_j = self.__patterns_from_l1_dep_tags(
                opinion_word_wc, [dep_tag_j], pos_tags, term_word_idx_span)
            # print(dep_tag_i, dep_tag_j)
            # print(patterns_i, patterns_j)

            if dep_tag_i[1][0] == dep_tag_j[1][0] or dep_tag_i[1][0] == dep_tag_j[2][0]:
                patterns_i = {(tup, 1) for tup in patterns_i}
            else:
                patterns_i = {(tup, 2) for tup in patterns_i}

            if dep_tag_j[1][0] == dep_tag_i[1][0] or dep_tag_j[1][0] == dep_tag_i[2][0]:
                patterns_j = {(tup, 1) for tup in patterns_j}
            else:
                patterns_j = {(tup, 2) for tup in patterns_j}
            # print(patterns_i, patterns_j)

            for pi in patterns_i:
                for pj in patterns_j:
                    if pi[0][pi[1]] != pj[0][pj[1]]:
                        # print(pi, pj)
                        continue
                    if pi < pj:
                        patterns.add((pi, pj))
                    else:
                        patterns.add((pj, pi))
        return patterns

    def __patterns_from_l2_dep_tags_combination_MH(self, opinion_word_wc_dict, related_l2, curr_tags, idx_span): #addself
        # widx_beg, widx_end = term_word_idx_span
        patterns = set()
        for dep_tag_i, dep_tag_j in related_l2:
            patterns_i_wo_index = self.__patterns_from_l1_dep_tags_combination_MH(
                opinion_word_wc_dict, {'dummy':[dep_tag_i]}, curr_tags, idx_span)
            patterns_j_wo_index = self.__patterns_from_l1_dep_tags_combination_MH(
                opinion_word_wc_dict, {'dummy':[dep_tag_j]}, curr_tags, idx_span)
            for k,l in [(1,1),(1,2),(2,1),(2,2)]:
                if dep_tag_i[k][0] == dep_tag_j[l][0]:
                    patterns_i = {(tup, k) for tup in patterns_i_wo_index}
                    patterns_j = {(tup, l) for tup in patterns_j_wo_index}
                    for pi in patterns_i:
                        for pj in patterns_j:
                            if pi[0][pi[1]] != pj[0][pj[1]]:
                                # print(pi, pj)
                                continue
                            patterns.add((pi, pj))
        # check that pattern has at least one tag with "_O" at the beginning
        patterns_new = set()
        for p in patterns:
            flag__O = False
            for p_l1 in p:
                for l in [1,2]:
                    if p_l1[0][l].startswith("_O"):
                        flag__O = True
            if flag__O:
                patterns_new.add(p)
        return patterns_new