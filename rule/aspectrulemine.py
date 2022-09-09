from rule import ruleutils
from utils import datautils


class AspectMineTool:
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
        for t in term_pos_tags:
            if t in ruleutils.NOUN_POS_TAGS:
                return 'POS_N'
        return None
    
#     @staticmethod
#     def __get_term_pos_type_ner(term_pos_tags):
#         for t in term_pos_tags:
#             if t in ruleutils.NEB_TAGS_AS:
#                 return 'NER_B' #nur die kommen als Aspects infrage
# #        for t in term_pos_tags:
# #            if t not in ['NER_O']:
# #                return 'NER_B'
#         return None
    
    @staticmethod    
    def __get_term_pos_type_token(term_pos_tags):
        for t in term_pos_tags:
            return t
        return None


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
    def __get_term_pos_type_ops(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'OP_A':
                return 'OP_A'
        for t in term_pos_tags:
            if t == 'OP_O':
                return 'OP_O'
        return None
    
    @staticmethod
    def __get_term_pos_type_word(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'WORD':
                return 'WORD'

        return None
   
    @staticmethod    
    def __get_term_pos_type_sent(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'SENT_S':
                return 'SENT_S'
        for t in term_pos_tags:
            if t == 'SENT_O':
                return 'SENT_O'
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
    def __get_term_pos_type_wordnet(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_SYN':
                return 'NO_SYN'
        for t in term_pos_tags:
            if t != 'NO_SYN':
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
    def __get_term_pos_type_rhyp(term_pos_tags):
        for t in term_pos_tags:
            if t  == 'NO_RHYP':
                return 'NO_RHYP'
        for t in term_pos_tags:
            if t != 'NO_RHYP':
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


    def get_patterns_from_term(self, term_word_idx_span, related_dep_tag_idxs, dep_tags, pos_tags):
        widx_beg, widx_end = term_word_idx_span
        term_pos_tags = set([pos_tags[i] for i in range(widx_beg, widx_end)])
        term_pos_type = AspectMineTool.__get_term_pos_type(term_pos_tags)
        if term_pos_type is None:
            # print(term)
            return set(), set()

        aspect_word_wc = '_A{}'.format(term_pos_type)

        related_dep_tags = [dep_tags[idx] for idx in related_dep_tag_idxs]
        patterns_l1 = self.__patterns_from_l1_dep_tags(aspect_word_wc, related_dep_tags, pos_tags, term_word_idx_span)
        related_l2 = ruleutils.find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags)
        patterns_l2 = self.__patterns_from_l2_dep_tags(aspect_word_wc, related_l2, pos_tags, term_word_idx_span)
        return patterns_l1, patterns_l2

    def get_patterns_from_term_combination_MH(self, idx_span, related_rels_idxs, curr_rels, curr_tags,
                                              ruletype):
        widx_beg, widx_end = idx_span
        aspect_word_wc_dict = dict()
        for k,v in curr_tags.items():
            term_tags = set([v[i] for i in range(widx_beg, widx_end)])# get al pos tags of term_word
            # die folgenden __get_term_pos_type methoden aggregieren gewisse Tags
            mode = k.replace('_tag_seqs','')
            if mode=='ner':
                term_tag_type = AspectMineTool.__get_term_pos_type_ner(term_tags) # add AspectMineTool.
            elif mode=='pos':
                term_tag_type = AspectMineTool.__get_term_pos_type_pos(term_tags)
            elif mode=='cons':
                term_tag_type = AspectMineTool.__get_term_pos_type_cons(term_tags)
            elif mode=='coref':
                term_tag_type = AspectMineTool.__get_term_pos_type_coref(term_tags)
            elif mode=='word':
                term_tag_type = AspectMineTool.__get_term_pos_type_word(term_tags)                
            elif mode=='wordnet':
                term_tag_type = AspectMineTool.__get_term_pos_type_wordnet(term_tags)
            elif mode == 'sent':
                term_tag_type = AspectMineTool.__get_term_pos_type_sent(term_tags)
            elif mode == 'ops':
                term_tag_type = AspectMineTool.__get_term_pos_type_ops(term_tags) 
            elif mode=='asp':
                term_tag_type = AspectMineTool.__get_term_pos_type_asp(term_tags)                
            elif mode == 'ant':
                term_tag_type = AspectMineTool.__get_term_pos_type_ant(term_tags)
            elif mode == 'rhyp':
                term_tag_type = AspectMineTool.__get_term_pos_type_rhyp(term_tags)
            elif mode == 'rstTOP':
                term_tag_type = AspectMineTool.__get_term_pos_type_rstTOP(term_tags)
            elif mode == 'rstLEAF':
                term_tag_type = AspectMineTool.__get_term_pos_type_rstLEAF(term_tags)
            elif mode=='token':
                term_tag_type = AspectMineTool.__get_term_pos_type_token(term_tags)
            else:
                raise NotImplementedError()   
            
            if term_tag_type is not None:
                aspect_word_wc_dict[k] = '_A{}'.format(term_tag_type)

        if aspect_word_wc_dict.keys():
            related_rels_dict=dict()
            for k,v in related_rels_idxs.items():
                related_rels_dict[k] = [curr_rels[k][idx] for idx in v]
            if 'l0' in ruletype:
                patterns_l0 = set(v for v in aspect_word_wc_dict.values())
            else:
                patterns_l0 = set()
            if 'l1' in ruletype:
                patterns_l1 = self.__patterns_from_l1_dep_tags_combination_MH(aspect_word_wc_dict, related_rels_dict,
                                                           curr_tags, idx_span)
            else:
                patterns_l1 = set()
            patterns_new = set()
            for p in patterns_l1:
                flag__A = False
                for l in [1,2]:
                    if p[l].startswith("_A"):
                        flag__A = True
                if flag__A:
                    patterns_new.add(p)
            patterns_l1 = patterns_new
            if 'l2' in ruletype:
                related_l2 = ruleutils.find_related_l2_rels_combination_MH(related_rels_idxs, curr_rels)
                patterns_l2 = self.__patterns_from_l2_dep_tags_combination_MH(aspect_word_wc_dict, related_l2, curr_tags, idx_span)
            else:
                patterns_l2 = set()
            return patterns_l0,patterns_l1, patterns_l2
        else:
            return set(), set(), set()

    @staticmethod
    def get_candidate_terms(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        noun_phrases = ruleutils.get_noun_phrases(words, pos_tag_seq, None)

        verbs = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            if pos_tag in ruleutils.VB_POS_TAGS:
                verbs.append(w)

        return noun_phrases + verbs

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
    def get_candidate_terms_sent(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.SENTS_TAGS, ruleutils.SENTO_TAGS]
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
    def get_candidate_terms_ops(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.OPO_TAGS, ruleutils.OPA_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
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
    def get_candidate_terms_ops(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.OPO_TAGS, ruleutils.OPA_TAGS]
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
      
    
#    @staticmethod
#    def get_candidate_terms_cons_v1(dep_tag_seq, pos_tag_seq):
#        words = [tup[2][1] for tup in dep_tag_seq]
#        pos_tag_sets = ['S0','NP0','VP0']
#        terms = list()
#        for w, pos_tag in zip(words, pos_tag_seq):
#            for pos_tag_set in pos_tag_sets:
#                if pos_tag in pos_tag_set:
#                    terms.append(w)
#        return terms

#    @staticmethod
#    def get_candidate_terms_cons(dep_tag_seq, pos_tag_seq):
#        words = [tup[2][1] for tup in dep_tag_seq]
#        pos_tag_sets = ['PP2', 'VP14', ',11', 'VB10', 'S1', 'CC16', 'VBZ1', '``5', 'UH4', 'SBAR3', '-LRB-4', '-RRB-3', 'SQ2', ':1', 'NNS0', 'DT1', 'VP13', 'VBP0', 'CC9', 'PRT3', '-LRB-1', 'SINV2', 'DT2', 'VP15', 'WHADVP4', '.2', 'SQ1', 'ADVP5', 'HYPH9', 'VP5', 'WHNP4', 'NN4', 'VBN1', '.4', 'VBP3', ':5', '-LRB-2', 'JJ2', 'VBZ4', 'NP16', 'PP7', 'WP0', ',7', 'ADJP6', ':3', 'ADJP9', 'PRN6', 'VBP1', 'VB4', 'RB7', 'VB5', 'UCP2', 'PRP$11', 'JJ4', 'NFP6', 'CC10', ':4', 'NNS2', 'LS0', 'ADJP2', 'VP0', 'S7', 'MD6', 'S9', 'ADVP6', '``1', 'LABEL0', 'FRAG2', 'NP0', 'CD3', 'JJS1', 'NNS4', 'NN12', 'S6', 'CC13', 'RB0', 'HYPH4', 'VP3', '-LRB-6', 'SYM2', 'INTJ1', 'SBAR8', 'SINV3', 'VP11', 'JJ0', 'CC2', 'PP4', 'ADJP5', 'NNP5', 'S3', '.18', 'INTJ2', 'VBD2', 'PP14', 'VP8', 'NP2', 'PRP$2', '.15', 'VBG8', 'CD2', 'IN6', 'NN3', 'PP5', 'PP6', 'IN3', 'CC3', 'VBN6', "''6", ',6', 'MD0', 'WHNP0', 'S8', 'IN4', 'HYPH2', '-RRB-5', '.10', 'VBD4', "''2", ',5', 'VBG1', "''3", ',3', 'VP4', 'VBG0', 'SBAR6', 'WDT0', ',9', 'NP14', 'HYPH5', 'NNP2', 'ADJP1', 'S12', 'X8', 'WHADVP0', 'VBD0', 'ADJP0', 'RB2', 'NFP2', '-RRB-9', 'TO0', '-RRB-11', 'RB5', 'PP1', 'NP3', 'VB0', 'VBN0', 'SQ4', 'PRP$1', ',10', 'CC1', 'VBZ0', 'PRT2', 'NNS3', 'WHNP1', 'VBD3', 'PRP1', 'TO1', 'NN2', '-RRB-6', 'NFP3', 'JJS0', 'HYPH6', 'SINV1', 'JJ5', 'INTJ0', ':0', 'NP13', 'CC7', '.1', '``2', '.5', 'NN5', 'HYPH3', 'SBAR4', ',1', 'NN0', 'VBP5', 'WHPP0', '-LRB-3', 'WRB0', 'RB3', 'S10', 'CC8', 'UH0', 'S17', ',4', '.9', 'ADVP3', 'PRT1', 'NNS6', '.11', 'LST0', 'VBZ3', 'CD1', 'VB1', '.0', 'INTJ6', ':2', 'WHNP3', 'NP9', '-RRB-1', '.7', 'ADVP1', 'PRN15', 'VBN2', 'PP0', '-RRB-7', 'VBD1', ',2', 'PRP$0', 'JJ6', 'DT9', '``0', "''9", '$1', 'NNP3', 'NP4', 'CONJP0', 'NN6', 'VP7', 'ADVP0', '$0', 'NP8', 'HYPH1', 'TO9', ',8', 'WHADVP1', 'IN2', 'VB2', 'PRT5', 'NN8', 'NP7', 'IN0', 'PRT7', 'VP9', "''4", '.16', 'NP5', 'S5', '-RRB-4', 'CD0', '.3', '.8', 'WHNP2', 'SBAR1', ',0', 'CC4', 'CC0', 'NP1', ',14', 'CC6', 'S4', 'RB1', 'TO4', 'NNP4', 'SYM4', 'SBAR5', 'ADVP4', 'ADJP4', 'IN1', 'TO2', 'VBP2', 'NN1', 'VP2', 'WHADVP6', 'ADJP3', 'RB6', '-LRB-8', 'VBP4', 'JJR0', 'RB4', '-LRB-0', 'NP10', 'MD1', 'S0', 'VP1', 'NFP1', 'NFP0', '-RRB-10', '.6', 'NP6', 'SQ3', 'SBAR7', 'VBZ2', 'X0', 'NNPS1', 'S15', 'FRAG0', 'POS3', 'NP12', 'DT5', 'CC5', 'MD3', '-RRB-2', 'S11', 'INTJ4', 'UH2', 'JJ8', 'JJ3', 'VBD5', '-LRB-5', 'ADVP2', 'CD6', 'NFP4', '-RRB-8', '-LRB-9', 'INTJ3', "''5", 'JJS2', 'NNP1', 'DT3', 'UCP4', 'VP17', '-RRB-0', 'VP10', 'VP6', 'DT0', '.12', 'SBAR2', 'JJ1', 'PP3', ',13', 'SBAR0', 'NNS1', 'S2', '-RRB-12', "''7", 'RBR3', 'NNP0']
#        terms = list()
#        for w, pos_tag in zip(words, pos_tag_seq):
#            for pos_tag_set in pos_tag_sets:
#                if pos_tag in pos_tag_set:
#                    terms.append(w)
#        return terms
    
#    @staticmethod
#    def get_candidate_terms_coref(dep_tag_seq, pos_tag_seq):
#        words = [tup[2][1] for tup in dep_tag_seq]
#        pos_tag_sets = ['O','1','2']
#        terms = list()
#        for w, pos_tag in zip(words, pos_tag_seq):
#            for pos_tag_set in pos_tag_sets:
#                if pos_tag in pos_tag_set:
#                    terms.append(w)
#        return terms
#    
#    @staticmethod
#    def get_candidate_terms_srl(dep_tag_seq, pos_tag_seq):
#        words = [tup[2][1] for tup in dep_tag_seq]
#        pos_tag_sets = ['O','SR']
#        terms = list()
#        for w, pos_tag in zip(words, pos_tag_seq):
#            for pos_tag_set in pos_tag_sets:
#                if pos_tag in pos_tag_set:
#                    terms.append(w)
#        return terms

    @staticmethod
    def get_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_A'):
            aspect_position = 1
        elif pattern[2].startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        if pattern[aspect_position] == '_AV':
            return w
        else:
            # hier gabs nen fehler
            return ruleutils.get_noun_phrase_from_seed(dep_tags, pos_tags, [widx])

    @staticmethod
    def get_term_from_matched_pattern_combination_MH(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_A'):
            aspect_position = 1
        elif pattern[2].startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return w

    @staticmethod
    def get_term_from_matched_l0pattern_combination_MH(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern.startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return w

    @staticmethod
    def get_termidx_from_matched_l0pattern_combination_LK(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern.startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return widx


    @staticmethod
    def get_termidx_from_matched_pattern_combination_LK(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_A'):
            aspect_position = 1
        elif pattern[2].startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return widx

    # def match_pattern_word(self, pw, w, pos_tag):
    #     if pw == '_AV' and pos_tag in ruleutils.VB_POS_TAGS:
    #         return True
    #     if pw == '_AN' and pos_tag in ruleutils.NOUN_POS_TAGS:
    #         return True
    #     if pw == '_OP' and w in self.opinion_terms_vocab:
    #         return True
    #     if pw.isupper() and pos_tag == pw:
    #         return True
    #     return pw == w

    @staticmethod
    def match_pattern_word_pos(pw, w, pos_tag):
        if pw == '_APOS_J' and pos_tag in ruleutils.JJ_POS_TAGS:
            return True
        if pw == '_APOS_R' and pos_tag in ruleutils.RB_POS_TAGS:
            return True
        if pw == '_APOS_V' and pos_tag in ruleutils.VB_POS_TAGS:
            return True
        if pw == '_APOS_N' and pos_tag in ruleutils.NOUN_POS_TAGS:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    # @staticmethod
    # def match_pattern_word_ner(pw, w, pos_tag):
    #     if pw == '_ANER_O' and pos_tag in ruleutils.NEO_TAGS_AS:
    #         return True
    #     if pw == '_ANER_B' and pos_tag in ruleutils.NEB_TAGS_AS:
    #         return True
    #     if pw.isupper() and pos_tag == pw:
    #         return True
    #     return pw == w
    
    @staticmethod
    def match_pattern_word_ner(pw, w, pos_tag):
        if pw != 'NER_O' and pos_tag != 'NER_O':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_wordnet(pw, w, pos_tag):
        if pw != 'NO_SYN' and pos_tag != 'NO_SYN':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w

    @staticmethod
    def match_pattern_word_token(pw, w, pos_tag):
        if pw.startswith('_A'):
            return pw[2:] == pos_tag
        else:
            return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_ant(pw, w, pos_tag):
        if pw != 'NO_ANT' and pos_tag != 'NO_ANT':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rhyp(pw, w, pos_tag):
        if pw != 'NO_RHYP' and pos_tag != 'NO_RHYP':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rstLEAF(pw, w, pos_tag):
        if pw != 'NO_LEAF_RST' and pos_tag != 'NO_LEAF_RST':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    
    @staticmethod
    def match_pattern_word_rstTOP(pw, w, pos_tag):
        if pw != 'NO_TOP_RST' and pos_tag != 'NO_TOP_RST':
            if pw.startswith('_A'):
                return pw[2:] == pos_tag
            else:
                return pw == pos_tag
        return pw == w
    

    @staticmethod
    def match_pattern_word_sent(pw, w, pos_tag):
        if pw == '_ASENT_O' and pos_tag in ruleutils.SENTO_TAGS:
            return True
        if pw == '_ASENT_S' and pos_tag in ruleutils.SENTS_TAGS:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    
    @staticmethod
    def match_pattern_word_ops(pw, w, pos_tag):
        if pw == '_AOP_O' and pos_tag  == 'OP_O':
            return True
        if pw == '_AOP_A' and pos_tag  == 'OP_A':
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    @staticmethod
    def match_pattern_word_word(pw, w, pos_tag):
        if pw == '_AWORD' and pos_tag == 'WORD':
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
    
    @staticmethod
    def match_pattern_word_asp(pw, w, pos_tag):
        if pw == '_AASP_O' and pos_tag == 'ASP_O':
            return True
        if pw == '_AASP_A' and pos_tag == 'ASP_A':
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w
#    @staticmethod
#    def match_pattern_word_cons_v1(pw, w, pos_tag):
#        if pw == '_ACONS_S' and pos_tag in ['S0']:
#            return True
#        if pw == '_ACONS_N' and pos_tag in ['NP0']:
#            return True
#        if pw == '_ACONS_V' and pos_tag in ['VP0']:
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w
#    
#    @staticmethod
#    def match_pattern_word_cons(pw, w, pos_tag):
#        if pw == '_ACONS_0' and pos_tag.endswith('0'):
#            return True
#        if pw == '_ACONS_1' and pos_tag.endswith('1'):
#            return True
#        if pw == '_ACONS_2' and pos_tag.endswith('2'):
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w
#
#    @staticmethod
#    def match_pattern_word_coref(pw, w, pos_tag):
#        if pw == '_ACOREF_O' and pos_tag in ['O']:
#            return True
#        if pw == '_ACOREF_1' and pos_tag in ['1']:
#            return True
#        if pw == '_ACOREF_2' and pos_tag in ['2']:
#            return True
#        if pw.isupper() and pos_tag == pw:
#            return True
#        return pw == w

    @staticmethod
    def get_terms_by_matching(dep_tags, pos_tags, sent_text, terms_vocab):
        sent_text_lower = sent_text.lower()
        matched_tups = list()
        for t in terms_vocab:
            pbeg = sent_text_lower.find(t)
            if pbeg < 0:
                continue
            if pbeg != 0 and sent_text_lower[pbeg - 1].isalpha():
                continue
            pend = pbeg + len(t)
            if pend != len(sent_text_lower) and sent_text_lower[pend].isalpha():
                continue
            matched_tups.append((pbeg, pend))
            # break

        matched_tups = AspectMineTool.__remove_embeded(matched_tups)
        sent_words = [tup[2][1] for tup in dep_tags]
        aspect_terms = set()
        for matched_span in matched_tups:
            phrase = ruleutils.pharse_for_span(matched_span, sent_text_lower, sent_words, pos_tags, dep_tags)
            if phrase is not None:
                aspect_terms.add(phrase)

        return aspect_terms

    @staticmethod
    def terms_list_from_sents(sents):
        aspect_terms_list = list()
        for sent in sents:
            aspect_terms_list.append([t.lower() for t in sent.get('aspects', list())])
        return aspect_terms_list

    @staticmethod
    def term_ids_dict_from_sents(sents):
        aspect_term_ids_dict = dict()
        for sent in sents:
            aspect_terms = sent.get('aspects', list())
            aspect_terms = [w.lower() for w in aspect_terms]
            toks = sent['text']
            tokenidx = list()
            for i,tok in enumerate(toks):
                if tok.lower() in aspect_terms:
                    tokenidx.append(i)
                
            aspect_term_ids_dict[sent['id']]=tokenidx
        return aspect_term_ids_dict

    @staticmethod
    def __remove_embeded(matched_tups):
        matched_tups_new = list()
        for i, t0 in enumerate(matched_tups):
            exist = False
            for j, t1 in enumerate(matched_tups):
                if i != j and t1[0] <= t0[0] and t1[1] >= t0[1]:
                    exist = True
                    break
            if not exist:
                matched_tups_new.append(t0)
        return matched_tups_new

    # def __patterns_from_l1_dep_tags(self, aspect_word_wc, related_dep_tags, pos_tags, term_word_idx_span):
    #     widx_beg, widx_end = term_word_idx_span
    #     # print(related_dep_tags)
    #     patterns = set()
    #     for dep_tag in related_dep_tags:
    #         rel, gov, dep = dep_tag
    #         igov, wgov = gov
    #         idep, wdep = dep
    #         if widx_beg <= igov < widx_end:
    #             patterns.add((rel, aspect_word_wc, wdep))
    #             patterns.add((rel, aspect_word_wc, pos_tags[idep]))
    #             if wdep in self.opinion_terms_vocab:
    #                 patterns.add((rel, aspect_word_wc, '_OP'))
    #         elif widx_beg <= idep < widx_end:
    #             patterns.add((rel, wgov, aspect_word_wc))
    #             patterns.add((rel, pos_tags[igov], aspect_word_wc))
    #             if wgov in self.opinion_terms_vocab:
    #                 patterns.add((rel, '_OP', aspect_word_wc))
    #         else:
    #             patterns.add((rel, wgov, wdep))
    #             patterns.add((rel, pos_tags[igov], wdep))
    #             patterns.add((rel, wgov, pos_tags[idep]))
    #             if wgov in self.opinion_terms_vocab:
    #                 patterns.add((rel, '_OP', wdep))
    #                 patterns.add((rel, '_OP', pos_tags[idep]))
    #             if wdep in self.opinion_terms_vocab:
    #                 patterns.add((rel, wgov, '_OP'))
    #                 patterns.add((rel, pos_tags[igov], '_OP'))
    #     return patterns

    @staticmethod
    def __patterns_from_l0_dep_tags_combination_MH(aspect_word_wc_dict,
                                                       curr_tags, idx_span):
        widx_beg, widx_end = idx_span
        # print(related_dep_tags)
        patterns = set()
        raise NotImplementedError
        for tag_idx, (tag_type,tags_v) in enumerate(curr_tags.items()):
            if widx_beg <= tag_idx < widx_end:
                # if word is opinion term
                if tag_type in aspect_word_wc_dict.keys():
                    patterns.add((aspect_word_wc_dict[tag_type]))
                    # combination
        return patterns

    @staticmethod
    def __patterns_from_l1_dep_tags_combination_MH(aspect_word_wc_dict, related_rels_dict,
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
                            if tag_type_gov in aspect_word_wc_dict.keys():
                                patterns.add((rel, aspect_word_wc_dict[tag_type_gov], tags_v_dep[idep])) # combination
                        elif widx_beg <= idep < widx_end:
                            # if dependent in opinion term
                            if tag_type_dep in aspect_word_wc_dict.keys():
                                patterns.add((rel, tags_v_gov[igov], aspect_word_wc_dict[tag_type_dep])) # combination
                        patterns.add((rel, tags_v_gov[igov], tags_v_dep[idep])) # combination
        return patterns

    def __patterns_from_l2_dep_tags(self, aspect_word_wc, related_dep_tag_tups, pos_tags, term_word_idx_span):
        # widx_beg, widx_end = term_word_idx_span
        patterns = set()
        for dep_tag_i, dep_tag_j in related_dep_tag_tups:
            patterns_i = self.__patterns_from_l1_dep_tags(
                aspect_word_wc, [dep_tag_i], pos_tags, term_word_idx_span)
            patterns_j = self.__patterns_from_l1_dep_tags(
                aspect_word_wc, [dep_tag_j], pos_tags, term_word_idx_span)
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

    @staticmethod
    def __get_term_pos_type(term_pos_tags):
        for t in term_pos_tags:
            if t in ruleutils.NOUN_POS_TAGS:
                return 'N'
        for t in term_pos_tags:
            if t in ruleutils.VB_POS_TAGS:
                return 'V'
        return None

    def __patterns_from_l2_dep_tags_combination_MH(self, aspect_word_wc_dict, related_l2, curr_tags, idx_span):
        patterns = set()
        for dep_tag_i, dep_tag_j in related_l2:
            patterns_i_wo_index = self.__patterns_from_l1_dep_tags_combination_MH(
                aspect_word_wc_dict, {'dummy':[dep_tag_i]}, curr_tags, idx_span)
            patterns_j_wo_index = self.__patterns_from_l1_dep_tags_combination_MH(
                aspect_word_wc_dict, {'dummy':[dep_tag_j]}, curr_tags, idx_span)
            for k,l in [(1,1),(1,2),(2,1),(2,2)]:
                if dep_tag_i[k][0] == dep_tag_j[l][0]:
                    patterns_i = {(tup, k) for tup in patterns_i_wo_index}
                    patterns_j = {(tup, l) for tup in patterns_j_wo_index}
                    for pi in patterns_i:
                        for pj in patterns_j:
                            if pi[0][pi[1]] != pj[0][pj[1]]:
                                continue
                            patterns.add((pi, pj))
        # check that pattern has at least one tag with "_A" at the beginning
        patterns_new = set()
        for p in patterns:
            flag__A = False
            for p_l1 in p:
                for l in [1,2]:
                    if p_l1[0][l].startswith("_A"):
                        flag__A = True
            if flag__A:
                patterns_new.add(p)
        return patterns_new