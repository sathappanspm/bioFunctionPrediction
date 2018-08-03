import glob
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json
import logging

log = logging.getLogger('root.DataLoader')

'''
We ignore 5 amino acids. Rest 20(k) form our vocab size.
These are delegated to the id  k+1
0 is used for padding.
'''

# --------------------------------- #
# Set up the data
# Done only once
# After the files are created (locally), this will be avoided
# Checks put in place
# --------------------------------- #

IGNORE_AA = ('B', 'O', 'J', 'X', 'U','Z')
CAT_GO = ('BP', 'MF')
ORIG_SETS = ('test', 'train')
# ---- #
# Effective  Test size : 0.15
# Effective validation size : 0.05 (0.25 * 0.20)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

file_name_dict_y = {
    'BP': 'bp.pkl',
    'MF': 'mf.pkl'
}

file_name_dict_x = {
    'BP': {
        'train': 'train-bp.pkl',
        'test': 'test-bp.pkl'
    },
    'MF': {
        'train': 'train-mf.pkl',
        'test': 'test-mf.pkl'
    }
}

orig_file_dir = os.path.join(os.path.dirname(__file__), './../../resources/data/data_orig_paper')
data_dir = os.path.join(os.path.dirname(__file__), './../../resources/data/data_paper')


def download_data():
    global orig_file_dir
    if not os.path.isdir(orig_file_dir):
        os.mkdir(orig_file_dir)


def get_file(cat_go, _set='train'):
    global orig_file_dir
    global orig_file_name_dict_x

    f_name = file_name_dict_x[cat_go][_set]
    file_path = orig_file_dir + '/' + f_name
    data1 = pd.read_pickle(file_path)
    return data1


'''
The original paper uses train and test sets.
We combine these and set up train , test, validations sets
'''


def combine_data():
    global CAT_GO
    global ORIG_SETS
    global data_dir

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    for _go in CAT_GO:
        # create a new dataframe
        df = pd.DataFrame(
            columns=['sequences', 'ngrams' ,'labels']
        )
        for _set in ORIG_SETS:
            data = get_file(_go, _set)
            data = data[['sequences', 'ngrams' ,'labels' ]]
            df = df.append(data, ignore_index=True, sort=True)

        train, test = train_test_split(df, test_size=0.2)
        test, val = train_test_split(test, test_size=0.25)
        _ref = {}
        _ref['train'] = train
        _ref['test'] = test
        _ref['val'] = val

        for k, _df in _ref.items():
            _df = _df.reset_index()
            file_name = ''.join([_go, '_', k, '.pkl'])
            res_path = os.path.join(data_dir, file_name)
            _df.to_pickle(res_path)


def initialize():
    global orig_file_dir
    global data_dir

    if not os.path.isdir(orig_file_dir):
        download_data()
    if not os.path.isdir(data_dir):
        combine_data()
    return


initialize()


# ---------------------------------------- #

class Amino_Acid_Map:

    def __init__(self):
        self.aminoacid_map = {}
        self.load()
        return

    def load(self):
        global IGNORE_AA
        data_dir =  os.path.join(os.path.dirname(__file__), './../../resources')

        with open(os.path.join(data_dir, 'aminoacids.txt'), 'r') as inpf:
            _id = 1
            aa_list = list(set(json.load(inpf)))

            for aa in aa_list:
                if aa not in IGNORE_AA:
                    self.aminoacid_map[aa] = _id
                    _id += 1

            aa_size = len(self.aminoacid_map)
            self.aminoacid_map['<unk>'] = aa_size + 1

            log.info(
                'loaded amino acid map of size-{}'.format(aa_size)
            )

    def amino_acid_to_id(self, amino_acid):
        if amino_acid not in self.aminoacid_map:
            log.info('unable to find {} in known aminoacids'.format(amino_acid))
            amino_acid = '<unk>'
        return self.aminoacid_map[amino_acid]

    def to_onehot(self, seq):
        res = [
            self.amino_acid_to_id(amino_acid) for amino_acid in seq
        ]
        return  res

# ------------------------- #
class BaseDataIterator:
    global data_dir
    data_loc = os.path.join(os.path.dirname(__file__), data_dir)

    def __init__(
            self,
            functype,
            batch_size,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        self.featuretype = featuretype
        self.functype = functype
        self.max_seq_len = seqlen
        self.seq_col_name = 'sequences'
        self.batch_size = batch_size
        self.aa_map_obj = Amino_Acid_Map()
        self.y_column = 'labels'
        self.reset()
        return

    def reset(self):
        # start of data file
        self.cur_idx = 0

    def filter_by_seq_len(self,df):
        df = df[df[self.seq_col_name].str.len() <= self.max_seq_len]
        return df


    def convert_seq_to_id(self):
        def pad_seq(res):
            pad = [0] * (self.max_seq_len - len(res))
            res.extend(pad)
            return res

        def aux(row):
            seq = row[self.x_column]
            res = self.aa_map_obj.to_onehot(seq)
            return pad_seq(res)

        self.df[self.x_column] = self.df.apply(aux,axis=1)
        return

    def format_x(self):
        if self.featuretype == 'onehot':
            self.x_column = 'sequences'
            self.convert_seq_to_id()
        elif self.featuretype == 'ngrams':
            self.x_column = 'ngrams'
        return

    def format_batch_data(self, _df):

        print('format_batch_data >> batch length', len(_df))
        y = list(_df[self.y_column])
        y = np.asarray(y)

        x_data = _df[self.x_column].values
        x = np.hstack([np.array(i) for i in x_data])
        x = np.reshape(x,[self.batch_size,-1])
        print('X', x.shape)
        print('Y', y.shape)
        return x,y


    def generate_batch(self):
        start_idx = self.cur_idx
        end_idx = self.cur_idx + self.batch_size-1
        if end_idx > len(self.df) :
            self.reset()
            return self.generate_batch()

        tmp_df = self.df.loc[start_idx:end_idx]
        tmp_df = pd.DataFrame(tmp_df,copy=True)
        print('generate_batch >>> ',len(tmp_df))
        return self.format_batch_data(tmp_df)

    def __next__(self):
        return self.generate_batch()

class TrainIterator(BaseDataIterator):
    file_path = None
    def __init__(
            self,
            functype,
            batch_size,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            seqlen,
            featuretype,
            autoreset
        )
        self.read_data()
        self.format_x()
        return

    def read_data(self):
        file_name = ''.join([str(self.functype),'_','train.pkl'])
        TrainIterator.file_path = os.path.join(BaseDataIterator.data_loc,file_name)
        self.df = pd.read_pickle(TrainIterator.file_path)



class TestIterator(BaseDataIterator):
    file_path = None
    def __init__(
            self,
            functype,
            batch_size,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            seqlen,
            featuretype,
            autoreset)
        return

    def read_data(self):
        file_name = ''.join(str(function),'_','test.pkl')
        TestIterator.file_path = os.path.join(BaseDataIterator.data_loc,file_name)
        self.df = pd.read_pickle(TestIterator.file_path)

        # filter data by sequence length
        self.df = self.filter_by_seq_len(self.df)

class ValIterator(BaseDataIterator):
    file_path = None
    def __init__(
            self,
            functype,
            batch_size,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            seqlen,
            featuretype,
            autoreset)
        return

    def read_data(self):
        file_name = ''.join(str(function),'_','val.pkl')
        ValIterator.file_path = os.path.join(BaseDataIterator.data_loc,file_name)
        self.df = pd.read_pickle(ValIterator.file_path)


ti = TrainIterator('MF',100)
x, y = ti.__next__()
print(x.shape, y.shape)
x, y = ti.__next__()