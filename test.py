from MissSpell import SymSpell
import gc
import string
import re
import pandas as pd
import numpy as np 


def find_in_list_ss1(row):
    for doc_word in row:
        suggestion = ss1.best_word(doc_word, silent = True)
        if suggestion is not None:
            return doc_word
    return ''

def find_in_list_ss2(row):
    for doc_word in row:
        suggestion = ss2.best_word(doc_word, silent = True)
        if suggestion is not None:
            return doc_word
    return ''

train = pd.read_table('train.tsv',
                      engine='c',
                      dtype={'item_condition_id': 'category',
                             'shipping': 'category'}
                      )
test = pd.read_table('test.tsv',
                     engine='c',
                     dtype={'item_condition_id': 'category',
                            'shipping': 'category'}
                     )

submission = test[['test_id']]

t1 = pd.concat([train[['brand_name', 'name', 'item_description']], test[['brand_name', 'name', 'item_description']]], axis = 0)

t1['name'] = t1['name'] \
    .fillna('') \
    .str.lower() \
    .astype(str)
t1['brand_name'] = t1['brand_name'] \
    .fillna('') \
    .str.lower() \
    .astype(str)
t1['item_description'] = t1['item_description'] \
    .fillna('') \
    .str.lower() \
    .replace(to_replace='No description yet', value='')
#    print(f'[{time() - start_time}] Missing filled.')

source_reg = r'[' + string.punctuation + '0-9]'
temp_source = t1['brand_name'].apply(lambda x: re.sub(source_reg, '', x))

vc = temp_source.value_counts()
source_cat = vc[vc > 0].index
reg = r'[a-z0-9]+'

one_word = source_cat[~source_cat.str.contains(' ')]       
many_words = source_cat[source_cat.str.contains(' ')]       

ss1 = SymSpell(max_edit_distance = 0, min_word_len=3)
ss1.create_dictionary_from_arr(one_word, token_pattern = r'.+')
ss1_dict = ss1.dictionary

ss2 = SymSpell(max_edit_distance = 0, min_word_len=3)
ss2.create_dictionary_from_arr(many_words, token_pattern = r'.+')
ss2_dict = ss2.dictionary    

reg = r'[a-z0-9]+'
temp = t1.loc[t1['brand_name'] == '']['name'].str.findall(pat = reg)
temp = [[(row1[i] + ' ' + row1[i + 1]).strip() if i < len(row1) - 1 else row1[i] for i in range(len(row1))]
        if len(row1) > 1 else row1 for row1 in temp]

final = [find_in_list_ss2(row) if len(row) > 1 else '' for row in temp]
final = pd.DataFrame(data = {'final': final})
comp = t1.loc[t1['brand_name'] == '']['brand_name']
comp = pd.DataFrame(data = {'comp': comp})

ind = comp.loc[comp.comp == ''].index
est = final.loc[final.index.isin(ind)]
temp2 = pd.DataFrame(data = {'temp': temp})

              
string = 'kate spade'

suggest_dict = {}
min_suggest_len = float('inf')

queue = [string]
q_dictionary = {}  # items other than string that we've checked
dictionary = ss2_dict.copy()

while len(queue) > 0:
    q_item = queue[0]  # pop
    queue = queue[1:]

    # early exit
    # process queue item
    if (q_item in dictionary) and (q_item not in suggest_dict):
        if dictionary[q_item][1] > 0:
            # word is in dictionary, and is a word from the corpus, and
            # not already in suggestion list so add to suggestion
            # dictionary, indexed by the word with value (frequency in
            # corpus, edit distance)
            # note q_items that are not the input string are shorter
            # than input string since only deletes are added (unless
            # manual dictionary corrections are added)
            suggest_dict[q_item] = (dictionary[q_item][1],
                                    len(string) - len(q_item))
            # early exit
            if (len(string) == len(q_item)):
                break
            elif (len(string) - len(q_item)) < min_suggest_len:
                min_suggest_len = len(string) - len(q_item)

        # the suggested corrections for q_item as stored in
        # dictionary (whether or not q_item itself is a valid word
        # or merely a delete) can be valid corrections
        for sc_item in dictionary[q_item][0]:
            if sc_item not in suggest_dict:

                # compute edit distance
                # suggested items should always be longer
                # (unless manual corrections are added)
                assert len(sc_item) > len(q_item)

                # q_items that are not input should be shorter
                # than original string
                # (unless manual corrections added)
                assert len(q_item) <= len(string)

                if len(q_item) == len(string):
                    assert q_item == string
                    item_dist = len(sc_item) - len(q_item)

                # item in suggestions list should not be the same as
                # the string itself
                assert sc_item != string

                # calculate edit distance using, for example,
                # Damerau-Levenshtein distance
                item_dist = dameraulevenshtein(sc_item, string)

                # do not add words with greater edit distance if
                # verbose setting not on
                if (verbose < 2) and (item_dist > min_suggest_len):
                    pass
                elif item_dist <= max_edit_distance:
                    assert sc_item in dictionary  # should already be in dictionary if in suggestion list
                    suggest_dict[sc_item] = (dictionary[sc_item][1], item_dist)
                    if item_dist < min_suggest_len:
                        min_suggest_len = item_dist

                # depending on order words are processed, some words
                # with different edit distances may be entered into
                # suggestions; trim suggestion dictionary if verbose
                # setting not on
                if verbose < 2:
                    suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

    # now generate deletes (e.g. a substring of string or of a delete)
    # from the queue item
    # as additional items to check -- add to end of queue
    assert len(string) >= len(q_item)

    # do not add words with greater edit distance if verbose setting
    # is not on
    if (verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
        pass
    elif (len(string) - len(q_item)) < max_edit_distance and len(q_item) > 1:
        for c in range(len(q_item)):  # character index
            word_minus_c = q_item[:c] + q_item[c + 1:]
            if word_minus_c not in q_dictionary:
                queue.append(word_minus_c)
                q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

as_list = suggest_dict.items()
# outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))
print outlist[0]
