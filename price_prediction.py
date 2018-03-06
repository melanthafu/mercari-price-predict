import pandas as pd
from time import time
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from preprocess_pandas import preprocess_pandas, split_cat
from vectorizer_process import ItemSelector, DropColumnsByDf
from intersect_drop_columns import intersect_drop_columns
import keras as ks
from keras.models import Model, load_model
import keras.backend.tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def timer(name):
    t0 = time.time()
    yield
    print('[%s] done in %s s' % (name, time.time() - t0))

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
merge, y_train, nrow_train = preprocess_pandas(train, test, source_name='brand_name', target_list=['name', 'item_description'], max_edit_distance=0, verbose = 0, min_word_len = 3)

########
tt1 = t1['brand_name']
tt2 = merge['brand_name']

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', 'its', 'are', 'am'])

vectorizer = FeatureUnion([
    ('name', Pipeline([
        ('select', ItemSelector('name')),
        ('transform', HashingVectorizer(
            ngram_range=(1, 2),
            n_features=2 ** 27,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('category_name', Pipeline([
        ('select', ItemSelector('category_name')),
        ('transform', HashingVectorizer(
            ngram_range=(1, 1),
            token_pattern='.+',
            tokenizer=split_cat,
            n_features=2 ** 27,
            norm='l2',
            lowercase=False
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('brand_name', Pipeline([
        ('select', ItemSelector('brand_name')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('gencat_cond', Pipeline([
        ('select', ItemSelector('gencat_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_1_cond', Pipeline([
        ('select', ItemSelector('subcat_1_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_2_cond', Pipeline([
        ('select', ItemSelector('subcat_2_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('has_brand', Pipeline([
        ('select', ItemSelector('has_brand')),
        ('ohe', OneHotEncoder())
    ])),
    ('shipping', Pipeline([
        ('select', ItemSelector('shipping')),
        ('ohe', OneHotEncoder())
    ])),
    ('item_condition_id', Pipeline([
        ('select', ItemSelector('item_condition_id')),
        ('ohe', OneHotEncoder())
    ])),
    ('item_description', Pipeline([
        ('select', ItemSelector('item_description')),
        ('hash', HashingVectorizer(
            ngram_range=(1, 3),
            n_features=2 ** 27,
            dtype=np.float32,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2)),
    ]))
], n_jobs=1)
            
sparse_merge = vectorizer.fit_transform(merge)

print(sparse_merge.shape)

tfdif = TfidfTransformer()
X = tfdif.fit_transform(sparse_merge)

X_train = X[:nrow_train]
print(X_train.shape)
X_test = X[nrow_train:]
print(X_test.shape)

X_train, X_test = intersect_drop_columns(X_train, X_test, min_df = 1)
y_scaler = StandardScaler()
X_save, y_save = X_train.copy(), y_train.copy()
y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 98431)


gpu_options=K.tf.GPUOptions(per_process_gpu_memory_fraction=0.85) # gpu setting

def predict_fit(X_train, y_train, sparse = True, train_round = 3, epoch = 20):
    with K.tf.device('/gpu:0'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,gpu_options=gpu_options)))
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=sparse)
        h1 = ks.layers.Dense(192, activation='relu')(model_in)
        h2 = ks.layers.Dense(64, activation='relu')(h1)
        h3 = ks.layers.Dense(64, activation='relu')(h2)
        out = ks.layers.Dense(1)(h3)
        model = Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    
    filepath=str('gap_shift_1.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    
    for i in range(train_round):
        if i == 0:
            model.fit(x=X_train, y=y_train, validation_data = (X_val, y_val), batch_size=2**(11 + i), epochs=1, verbose=2, \
                      callbacks = callbacks_list, nb_epoch = epoch)
            model.sample_weights('model_weights_%s' % i)
        else:
            model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
            model.load_weights('model_weights_%s' % (i - 1))
            model.fit(x=X_train, y=y_train, validation_data = (X_val, y_val), batch_size=2**(11 + i), epochs=1, verbose=2, \
                      callbacks = callbacks_list, nb_epoch = epoch)
            model.sample_weights('model_weights_%s' % i)

    return 

train_round = 3
predict_fit(X_train, y_train, train_round = train_round, epoch = 20)
model = load_model('gap_shift_1.hdf5')
model = model.load_weights('model_weights_%s' % train_round)   

pred_set = X_save.copy()         
y_pred = model.predict(pred_set)[:, 0]
y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])    
print('RMSLE: ', np.sqrt(mean_squared_log_error(y_save, y_pred)))