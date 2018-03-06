from MissSpell import SymSpell
import gc
import string
import re

def miss_word_fill(data, source_name, target_list, max_edit_distance, verbose, min_word_len):
    source_reg = r'[' + string.punctuation + '0-9]'
    temp_source = data[source_name].apply(lambda x: re.sub(source_reg, '', x))
    vc = temp_source.value_counts()
    source_cat = vc[vc > 0].index
    reg = r'[a-z0-9]+'

    one_word = source_cat[~source_cat.str.contains(' ')]       
    many_words = source_cat[source_cat.str.contains(' ')]       
    
    ss1 = SymSpell(max_edit_distance = max_edit_distance, min_word_len=min_word_len)
    ss1.create_dictionary_from_arr(one_word, token_pattern = r'.+')
    
    ss2 = SymSpell(max_edit_distance = max_edit_distance, min_word_len=min_word_len)
    ss2.create_dictionary_from_arr(many_words, token_pattern = r'.+')
    
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
    
    for tar in target_list:
        temp = data.loc[data[source_name] == ''][tar].str.findall(pat = reg)
        temp = [[(row1[i] + ' ' + row1[i + 1]).strip() if i < len(row1) - 1 else row1[i] for i in range(len(row1))]
                if len(row1) > 1 else row1 for row1 in temp]
        data.loc[data[source_name] == '', source_name] = [find_in_list_ss2(row) if len(row) > 1 else '' for row in temp]
        print ('two words finish')
        temp = data.loc[data[source_name] == ''][tar].str.findall(pat = reg)
        data.loc[data[source_name] == '', source_name] = [find_in_list_ss1(row) for row in temp]
    
    gc.collect()
        
