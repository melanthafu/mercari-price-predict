import pandas as pd
import numpy as np
import gc
from wordfill import miss_word_fill
import re

def split_cat(text):
    try:
        cats = text.split('/')
        return cats[0], cats[1], cats[2]
    
    except:
        return 'other', 'other', 'other'

def preprocess_pandas(train, test, source_name, target_list, max_edit_distance, verbose, min_word_len):
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    merge = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
#    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
#    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
#    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.') 

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
#    print(f'[{time() - start_time}] Missing filled.')
    
#    merge['name'] = merge.name.apply(lambda x: re.sub(r'[^a-z0-9]', '', x))
#    merge['brand_name'] = merge.name.apply(lambda x: re.sub(r'[^a-z0-9]', '', x))
#    merge['item_description'] = merge.name.apply(lambda x: re.sub(r'[^a-z0-9]', '', x))


    miss_word_fill(merge, source_name=source_name, target_list=target_list,max_edit_distance=max_edit_distance, verbose = verbose, min_word_len = min_word_len) 
#    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
#    print(f'[{time() - start_time}] Name concancenated.')

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']
#    print(f'[{time() - start_time}] Item description concatenated.')

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, nrow_train

    
