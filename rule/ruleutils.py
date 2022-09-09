NOUN_POS_TAGS = {'POS_NN', 'POS_NNP', 'POS_NNS', 'POS_NNPS','POS_NOUN'}
VB_POS_TAGS = {'POS_VB', 'POS_VBN', 'POS_VBP', 'POS_VBZ', 'POS_VBG', 'POS_VBD','POS_VERB'}
JJ_POS_TAGS = {'POS_JJ', 'POS_JJR', 'POS_JJS','POS_ADJ'}
RB_POS_TAGS = {'POS_RB', 'POS_RBR'}
NEB_TAGS_OP = {'NER_CARDINAL', 'NER_MONEY', 'NER_PERCENT', 'NER_PERSON', 'NER_QUANTITY', 'NER_TIME', 'NER_ORDINAL' }
NEO_TAGS_OP = {'NER_O', 'NER_DATE', 'NER_EVENT', 'NER_FAC', 'NER_GPE','NER_LOC', 'NER_NORP', 'NER_ORG', 'NER_PRODUCT', 'NER_WORK_OF_ART', 'NER_LANGUAGE'}
NEB_TAGS_AS = {'NER_PERSON','NER_EVENT', 'NER_FAC','NER_GPE','NER_LOC', 'NER_ORG','NER_PRODUCT', 'NER_WORK_OF_ART'}
NEO_TAGS_AS = {}
ASPB_TAGS = {'ASP_A'}
ASPO_TAGS = {'ASP_O'}
OPO_TAGS = {'OP_O'}
OPA_TAGS = {'OP_A'}
WORD_TAGS = {'WORD'}
SENTS_TAGS = {'SENT_S'}
SENTO_TAGS = {'SENT_O'}

def load_rule_patterns_file(filename):
    l0_rules, l1_rules, l2_rules = list(), list(), list()
    f = open(filename, encoding='utf-8')
    for line in f:
        vals = line.strip().split()
        assert len(vals) == 1 or len(vals) == 3 or len(vals) == 8
        if len(vals) == 1:
            if line.strip()[0]=='[' and line.strip()[-1]==']':
                l0_rules.append(line.strip()[2:-2])
            else:
                l0_rules.append(vals[0])
        elif len(vals) == 3:
            l1_rules.append(vals)
        else:
            l2_rules.append(
                (((vals[0], vals[1], vals[2]), int(vals[3])), ((vals[4], vals[5], vals[6]), int(vals[7]))))
    f.close()
    return l0_rules,l1_rules, l2_rules

def get_rule_pattern_from_string_v2(line):
    vals = line.strip().split()
    assert len(vals) == 1 or len(vals) == 3 or len(vals) == 8
    if len(vals) == 1:
        if line.strip()[0]=='[' and line.strip()[-1]==']':
            vals=line.strip()[2:-2]
        else:
            vals=vals[0]
    elif len(vals) == 8:
        vals = (
            (((vals[0], vals[1], vals[2]), int(vals[3])), ((vals[4], vals[5], vals[6]), int(vals[7]))))
    return vals

def get_rule_pattern_from_string(line):
    vals = line.strip().split()
    assert len(vals) == 1 or len(vals) == 3 or len(vals) == 8
    if len(vals) == 8:
        vals = (
            (((vals[0], vals[1], vals[2]), int(vals[3])), ((vals[4], vals[5], vals[6]), int(vals[7]))))
    return vals

def get_term_vocab(aspect_term_hit_rate_file, rate_thres):
    import pandas as pd

    df = pd.read_csv(aspect_term_hit_rate_file)
    df = df[df['rate'] > rate_thres]
    return set(df['term'])


def find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags):
    related = list()
    for i in related_dep_tag_idxs:
        rel, gov, dep = dep_tags[i]
        igov, wgov = gov
        idep, wdep = dep
        for j, dep_tag_j in enumerate(dep_tags):
            if i == j:
                continue
            rel_j, gov_j, dep_j = dep_tag_j
            igov_j, wgov_j = gov_j
            idep_j, wdep_j = dep_j
            if igov == igov_j or igov == idep_j or idep == igov_j or idep == idep_j:
                # print(dep_tags[i], dep_tag_j)
                related.append((dep_tags[i], dep_tag_j))
    return related

def find_related_l2_rels_combination_MH(related_rels_idxs, curr_rels):
    related=list()
    for k_i,v_i in curr_rels.items():
        related_rel_idxs = related_rels_idxs[k_i]
        for i in related_rel_idxs:
#            raise NotImplementedError
            dep_tag_i = v_i[i]
            _, gov_i, dep_i = dep_tag_i
            igov_i, _ = gov_i
            idep_i, _ = dep_i
            for k_j,v_j in curr_rels.items():
                for j, dep_tag_j in enumerate(v_j):
                    if k_i == k_j and i == j:
                        continue
                    _, gov_j, dep_j = dep_tag_j
                    igov_j, _ = gov_j
                    idep_j, _ = dep_j
                    if igov_i == igov_j or igov_i == idep_j or idep_i == igov_j or idep_i == idep_j:
                        related.append((dep_tag_i, dep_tag_j))
    return related

def get_noun_phrases(words, pos_tags, nouns_filter):
    assert len(words) == len(pos_tags)

    noun_phrases = list()
    pleft = 0
    while pleft < len(words):
        if pos_tags[pleft] not in NOUN_POS_TAGS:
            pleft += 1
            continue
        pright = pleft + 1
        while pright < len(words) and pos_tags[pright] in {'NN', 'NNS', 'NNP', 'CD', 'NOUN'}:
            pright += 1

        # if pleft > 0 and pos_tags[pleft - 1] == 'JJ' and words[pleft - 1] not in opinion_terms:
        #     pleft -= 1

        phrase = ' '.join(words[pleft: pright])
        if nouns_filter is None or phrase not in nouns_filter:
            noun_phrases.append(phrase)
        pleft = pright
    # print(' '.join(words))
    # print(noun_phrases)
    return noun_phrases


def __find_word_spans(text_lower, words):
    p = 0
    word_spans = list()
    for w in words:
        wp = text_lower[p:].find(w)
        if wp < 0:
            word_spans.append((-1, -1))
            continue
        word_spans.append((p + wp, p + wp + len(w)))
        p += wp + len(w)
    return word_spans


def get_noun_phrase_from_seed(dep_tags, pos_tags, base_word_idxs):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)

    ileft = min(phrase_word_idxs)
    iright = max(phrase_word_idxs)
    ileft_new, iright_new = ileft, iright
    while ileft_new > 0:
        # hier gabs nen fehler
        if pos_tags[ileft_new - 1] in NOUN_POS_TAGS:
            ileft_new -= 1
        else:
            break
    while iright_new < len(pos_tags) - 1:
        if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD','NOUN'}:
            iright_new += 1
        else:
            break

    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


def pharse_for_span(span, sent_text_lower, words, pos_tags, dep_tags):
    word_spans = __find_word_spans(sent_text_lower, words)
    widxs = list()
    for i, wspan in enumerate(word_spans):
        if (wspan[0] <= span[0] < wspan[1]) or (wspan[0] < span[1] <= wspan[1]):
            widxs.append(i)

    if not widxs:
        # print(span)
        # print(sent_text_lower[span[0]: span[1]])
        # print(sent_text_lower)
        # print(words)
        # print(word_spans)
        # exit()
        return None

    phrase = get_noun_phrase_from_seed(dep_tags, pos_tags, widxs)
    return phrase


def __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool,mode):
    prel, pgov, pdep = pattern
    rel, (igov, wgov), (idep, wdep) = dep_tag
    if rel != prel:
        return False
    if isinstance(mode,str): #tempfix
        return getattr(mine_tool,'match_pattern_word_'+mode)(pgov, wgov, pos_tags[igov]) \
            and getattr(mine_tool,'match_pattern_word_'+mode)(pdep, wdep, pos_tags[idep])
    else:
        return False

def __match_l1_pattern_combination_MH(pattern, curr_rel, sent_idx, data_valid, mine_tool):
    prel, pgov, pdep = pattern
    rel, (igov, wgov), (idep, wdep) = curr_rel
    if rel != prel:
        return False
    fields_with_tag = [i for i in data_valid._fields if 'tag' in i]
    matching_gov = False
    for curr_field_tag in fields_with_tag:
        mode = curr_field_tag.replace('_tag_seqs','')
        all_tags_of_field = getattr(data_valid,curr_field_tag)
        curr_tags_of_field = all_tags_of_field[sent_idx]
        matching_gov = bool(matching_gov or getattr(mine_tool,'match_pattern_word_'+mode)(pgov, wgov, curr_tags_of_field[igov]))
    matching_dep = False
    for curr_field_tag in fields_with_tag:
        mode = curr_field_tag.replace('_tag_seqs','')
        all_tags_of_field = getattr(data_valid,curr_field_tag)
        curr_tags_of_field = all_tags_of_field[sent_idx]
        matching_dep = bool(matching_dep or getattr(mine_tool,'match_pattern_word_'+mode)(pdep, wdep, curr_tags_of_field[idep]))
    return matching_gov and matching_dep


def __match_l0_pattern_combination_MH(pattern, sent_idx, id_tok, data_valid, mine_tool):
    pdep = pattern
    fields_with_tag = [i for i in data_valid._fields if 'tag' in i]
    matching_dep = False
    for curr_field_tag in fields_with_tag:
        mode = curr_field_tag.replace('_tag_seqs','')
        all_tags_of_field = getattr(data_valid,curr_field_tag)
        curr_tags_of_field = all_tags_of_field[sent_idx]
        matching_dep = bool(matching_dep or getattr(mine_tool,'match_pattern_word_'+mode)(pdep, '', curr_tags_of_field[id_tok]))
    return matching_dep


def __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool,mode):
    matched_idxs = list()
    for i, dep_tag in enumerate(dep_tags):
        if __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool,mode):
            matched_idxs.append(i)
    return matched_idxs


def __get_l1_pattern_matched_dep_tags_combination_MH(pattern, sent_idx, data_valid, mine_tool):
    matched_idxs_dict = dict()
    fields_with_rel = [i for i in data_valid._fields if 'rel' in i]
    for curr_field in fields_with_rel:
        matched_idxs = list()
        all_rels_of_field = getattr(data_valid,curr_field)
        curr_rels_of_field = all_rels_of_field[sent_idx]
        for i, curr_rel in enumerate(curr_rels_of_field):
            if __match_l1_pattern_combination_MH(pattern, curr_rel, sent_idx, data_valid, mine_tool):
                matched_idxs.append(i)
        matched_idxs_dict[curr_field] = matched_idxs
    return matched_idxs_dict


def __get_l0_pattern_matched_dep_tags_combination_MH(pattern, sent_idx, data_valid, mine_tool):
    matched_idxs = list()
    sentences = data_valid.sents
    curr_sentence = sentences[sent_idx]
    curr_tokens = curr_sentence['text']
    for id_tok, curr_token in enumerate(curr_tokens):
        if __match_l0_pattern_combination_MH(pattern, sent_idx, id_tok,
                                             data_valid, mine_tool):
            matched_idxs.append(id_tok)
    return matched_idxs


def find_terms_by_l1_pattern(pattern, dep_tags, pos_tags, mine_tool, filter_terms_vocab,mode):
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool,mode)
    for idx in matched_dep_tag_idxs:
        term = mine_tool.get_term_from_matched_pattern(pattern, dep_tags, pos_tags, idx)

        if term in filter_terms_vocab:
            continue
        terms.append(term)
        # print(pattern)
        # print(term)
        # print(dep_tags)
        # print([t[2][1] for t in dep_tags])
        # print()
    return terms

def find_terms_by_l0_pattern_combination_MH_applyrules(pattern, sent_idx, data_valid, 
                                            mine_tool):
    terms = list()
    terms_idx = list()
    matched_toks_dixs = __get_l0_pattern_matched_dep_tags_combination_MH(
            pattern, sent_idx, data_valid, mine_tool)
    sentences = data_valid.sents
    sentence = sentences[sent_idx]
    tokens = sentence['text']
    for idx in matched_toks_dixs:
        terms.append(tokens[idx])
        terms_idx.append(idx)
    return terms, terms_idx


def find_terms_by_l0_pattern_combination_MH_runrulemine(pattern, sent_idx, data_valid, 
                                            mine_tool):
    terms = list()
    #terms_idx = list()
    matched_toks_dixs = __get_l0_pattern_matched_dep_tags_combination_MH(
            pattern, sent_idx, data_valid, mine_tool)
    sentences = data_valid.sents
    sentence = sentences[sent_idx]
    tokens = sentence['text']
    for idx in matched_toks_dixs:
        terms.append(tokens[idx])
    return terms#, terms_idx

def find_terms_by_l1_pattern_combination_MH_applyrules(pattern, sent_idx, data_valid, 
                                            mine_tool):
    terms = list()
    terms_idx = list()
    matched_rel_idxs_dict = __get_l1_pattern_matched_dep_tags_combination_MH(
            pattern, sent_idx, data_valid, mine_tool)
    for rel_name,matched_rel_dixs in matched_rel_idxs_dict.items():
        all_rels_of_field = getattr(data_valid,rel_name)
        curr_rels = all_rels_of_field[sent_idx]
        for idx in matched_rel_dixs:
            term = mine_tool.get_term_from_matched_pattern_combination_MH(pattern, curr_rels, [], idx)
            term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(pattern, curr_rels, [], idx)
            terms.append(term)
            terms_idx.append(term_idx)
    return terms, terms_idx

def find_terms_by_l1_pattern_combination_MH_runrulemine(pattern, sent_idx, data_valid, 
                                            mine_tool, filter_terms_vocab):
    terms = list()
    #terms_idx = list()
    matched_rel_idxs_dict = __get_l1_pattern_matched_dep_tags_combination_MH(
            pattern, sent_idx, data_valid, mine_tool)
    for rel_name,matched_rel_dixs in matched_rel_idxs_dict.items():
        all_rels_of_field = getattr(data_valid,rel_name)
        curr_rels = all_rels_of_field[sent_idx]
        for idx in matched_rel_dixs:
            term = mine_tool.get_term_from_matched_pattern_combination_MH(pattern, curr_rels, [], idx)
            #term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(pattern, curr_rels, [], idx)
            terms.append(term)
            #terms_idx.append(term_idx)
    return terms#, terms_idx


def find_terms_by_l2_pattern(pattern, dep_tags, pos_tags, mine_tool, filter_terms_vocab, mode,existing_terms=None):
    (pl, ipl), (pr, ipr) = pattern
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pl, dep_tags, pos_tags, mine_tool,mode)
    for idx in matched_dep_tag_idxs:
        dep_tag_l = dep_tags[idx]
        sw_idx = dep_tag_l[ipl][0]  # index of the shared word

        for j, dep_tag_r in enumerate(dep_tags):
            if dep_tag_r[ipr][0] != sw_idx:
                continue
            if not __match_l1_pattern(pr, dep_tag_r, pos_tags, mine_tool,mode):
                continue

            term = mine_tool.get_term_from_matched_pattern(pl, dep_tags, pos_tags, idx)
            if term is None:
                term = mine_tool.get_term_from_matched_pattern(pr, dep_tags, pos_tags, j)
            if term is None or term in filter_terms_vocab:
                # print(p, 'term not found')
                continue

            # if term not in existing_terms:
            #     print(pattern)
            #     print(term)
            #     print([t[2][1] for t in dep_tags])
            #     print(dep_tags)
            #     print()

            terms.append(term)
    return terms



def find_terms_by_l2_pattern_combination_MH_applyrules(
                pattern, sent_idx, data_valid, mine_tool):
    (p1, ip1), (p2, ip2) = pattern
    terms = list()
    terms_idx = list()
    matched_rel_idxs_dict = __get_l1_pattern_matched_dep_tags_combination_MH(
            p1, sent_idx, data_valid, mine_tool)

    fields_with_rel = [i for i in data_valid._fields if 'rel' in i]
    for rel_name,matched_rel_idxs in matched_rel_idxs_dict.items():
        all_rels1 = getattr(data_valid,rel_name)
        curr_rels1 = all_rels1[sent_idx]
        for idx in matched_rel_idxs:
            rel1 = curr_rels1[idx]
            sw_idx = rel1[ip1][0]  # index of the shared word
            
            for curr_field in fields_with_rel:
                all_rels2 = getattr(data_valid,curr_field)
                curr_rels2 = all_rels2[sent_idx]
                for j, rel2 in enumerate(curr_rels2):
                    if rel2[ip2][0] != sw_idx: # check whether shared word is also in rel2
                        continue
                    if not __match_l1_pattern_combination_MH(
                            p2, rel2, sent_idx, data_valid, mine_tool):
                        # check whether p2 matches rel2
                        continue
                    term = mine_tool.get_term_from_matched_pattern_combination_MH(p1, curr_rels1, [], idx)
                    term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(p1, curr_rels1, [], idx)
                    if term is None:
                        term = mine_tool.get_term_from_matched_pattern_combination_MH(p2, curr_rels2, [], j)
                        term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(p2, curr_rels2, [], j)
                    terms.append(term)
                    terms_idx.append(term_idx)
    return terms, terms_idx

def find_terms_by_l2_pattern_combination_MH_runrulemine(
                pattern, sent_idx, data_valid, mine_tool, filter_terms_vocab):
    (p1, ip1), (p2, ip2) = pattern
    terms = list()
    #terms_idx = list()
    matched_rel_idxs_dict = __get_l1_pattern_matched_dep_tags_combination_MH(
            p1, sent_idx, data_valid, mine_tool)

    fields_with_rel = [i for i in data_valid._fields if 'rel' in i]
    for rel_name,matched_rel_idxs in matched_rel_idxs_dict.items():
        all_rels1 = getattr(data_valid,rel_name)
        curr_rels1 = all_rels1[sent_idx]
        for idx in matched_rel_idxs:
            rel1 = curr_rels1[idx]
            sw_idx = rel1[ip1][0]  # index of the shared word
            
            for curr_field in fields_with_rel:
                all_rels2 = getattr(data_valid,curr_field)
                curr_rels2 = all_rels2[sent_idx]
                for j, rel2 in enumerate(curr_rels2):
                    if rel2[ip2][0] != sw_idx: # check whether shared word is also in rel2
                        continue
                    if not __match_l1_pattern_combination_MH(
                            p2, rel2, sent_idx, data_valid, mine_tool):
                        # check whether p2 matches rel2
                        continue
                    term = mine_tool.get_term_from_matched_pattern_combination_MH(p1, curr_rels1, [], idx)
                    #term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(p1, curr_rels1, [], idx)
                    if term is None:
                        term = mine_tool.get_term_from_matched_pattern_combination_MH(p2, curr_rels2, [], j)
                        #term_idx = mine_tool.get_termidx_from_matched_pattern_combination_LK(p2, curr_rels2, [], j)
                    if term is None or term in filter_terms_vocab:
                        # print(p, 'term not found')
                        continue
                    terms.append(term)
                    #terms_idx.append(term_idx)
    return terms#, terms_idx

def get_pattern_with_sentidx(pattern,data_check,mine_tool):
    pattern_with_sentidx = {str(pattern):dict()}
    if isinstance(pattern,str):
        for i, row in list(enumerate(data_check.sents)):
            sent_idx = row['id']
            terms_new, terms_idx = find_terms_by_l0_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx and set(terms_idx) != {None}:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, set())
                pattern_with_sentidx[str(pattern)][sent_idx].update(terms_idx)
    elif len(pattern)==3:
        for i, row in list(enumerate(data_check.sents)):
            sent_idx = row['id']
            terms_new, terms_idx = find_terms_by_l1_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx and set(terms_idx) != {None}:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, set())
                pattern_with_sentidx[str(pattern)][sent_idx].update(terms_idx)
    elif len(pattern)==2:
        for i, row in list(enumerate(data_check.sents)):
            sent_idx = row['id']
            # if sent_idx == 'A2GT6G0AKVUQ01||##||B00006LK6Q||##||10':
            #     break
            terms_new, terms_idx = find_terms_by_l2_pattern_combination_MH_applyrules(
                pattern, i, data_check, mine_tool)
            if terms_idx and set(terms_idx) != {None}:
                pattern_with_sentidx[str(pattern)].setdefault(sent_idx, set())
                pattern_with_sentidx[str(pattern)][sent_idx].update(terms_idx)
    return pattern_with_sentidx