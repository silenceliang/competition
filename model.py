
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from collections import OrderedDict
import parameters as p

''' store all fund Data with dictionary '''
class fund():

    def __init__(self, fund_id):
        self.fund_id = fund_id
        self.fund = dict()
        self.attr = ['open', 'high', 'low', 'close', 'volume']
        for n in fund_id:
            self.fund[n] = OrderedDict()

    def _put(self, id, date, name, open, high, low, close, volume):
        '''
            put Data inside
        '''
        self.fund[id][date] = {'open': open, 'high': high, 'low': low, 'close': close, 'volume': float(volume.strip().replace(',', '.'))}

    def _history(self):
        '''
        the length is different from each kind .
        :return:
        '''
        history_arr = []
        for id in self.fund_id:
            temp = []
            for d in sorted(self.fund[id]):
                x = [self.fund[id][d][attr] for attr in self.attr]
                temp.append(x)

            history_arr.append(np.array(temp))

        return np.array(history_arr)

    def combine_SP500(self, sp500):

        self.attr.append('sp500')

        for id in self.fund.keys():
            x = 0
            for date in self.fund[id].keys():
                try:
                    x = sp500[str(date)]
                    self.fund[id][date]['sp500'] = x

                except:
                    self.fund[id][date]['sp500'] = x

class dataLoader():

    def __init__(self, data):

        self.data = data
        # nn hyper-parameters
        self.partition = p.partition
        self.lr = p.lr
        self.num_epochs = p.num_epochs
        self.span = p.span
        self.hidden_size = p.hidden_size
        self.num_layers = p.num_layers
        self.batch_size = p.batch_size

        self.sample_num = data.shape[0]
        self.feature_num = data.shape[1]

        self.output_size = self.span
        self.time_steps, self.input_size = self.span, self.feature_num
        self.validation = int(self.partition * self.sample_num)  # not yet

        self.norm_data_X, _ = self.normalize_l2(self.data)
        self.norm_data_X = self.data
        # self.norm_data_Y, self.pf = self.normalize_l2(self.data[:, 3])
        self.norm_data_Y = self.data[:, 3]  # it should be 'close'.

        training_data_X = []
        training_data_Y = []

        for n in range(self.sample_num - 2*self.span):
            training_data_X.append(self.norm_data_X[n: n+self.span])
            training_data_Y.append(self.norm_data_Y[n+self.span: n + 2*self.span])

        ''' We want to predict the unknown price '''
        self.training_data_X = torch.FloatTensor(np.array(training_data_X)[:-1])
        self.training_data_Y = torch.FloatTensor(np.array(training_data_Y)[:-1])

        self.testing_data_X = torch.FloatTensor(np.array(training_data_X)[-1:])
        self.testing_data_Y = torch.FloatTensor(np.array(training_data_Y)[-1:])

        torch_dataset = TensorDataset(data_tensor=self.training_data_X, target_tensor=self.training_data_Y)
        torch_test_dataset = TensorDataset(data_tensor=self.testing_data_X, target_tensor=self.testing_data_Y)

        self.loader = DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.testloader = DataLoader(dataset=torch_test_dataset, batch_size=1, shuffle=False, num_workers=4)

    def normalize(self):

        tdata = self.data.copy()
        '''
        [[ 46.57   47.13   46.49   46.92   16.487]
         [ 47.35   47.48   47.13   47.31   29.02 ]
         [ 47.31   47.31   46.92   47.      9.837]
         ...
         [ 82.25   82.35   81.8    82.1     4.099]
         [ 82.65   83.05   82.65   82.85    4.994]
         [ 82.85   83.05   82.75   82.95  878.   ]]
        '''
        mean = np.mean(tdata, axis=0)
        std = np.std(tdata, axis=0)
        tdata = (tdata-mean) / std

        return tdata

    def normalize_l2(self, training_feature):

        tdata = training_feature.copy()
        '''
        [[ 46.57   47.13   46.49   46.92   16.487]
         [ 47.35   47.48   47.13   47.31   29.02 ]
         [ 47.31   47.31   46.92   47.      9.837]
         ...
         [ 82.25   82.35   81.8    82.1     4.099]
         [ 82.65   83.05   82.65   82.85    4.994]
         [ 82.85   83.05   82.75   82.95  878.   ]]
        '''
        partition = np.sqrt(np.sum(np.power(tdata, 2), axis=0))
        l2_norm = tdata / partition
        '''
        [[0.02071839 0.02087869 0.0207747  0.0208696  0.00892746]
         [0.02106541 0.02103374 0.0210607  0.02104307 0.01571388]
         [0.02104761 0.02095843 0.02096686 0.02090519 0.00532658]
         ...
         [0.03659197 0.03648122 0.03655347 0.03651736 0.00221955]
         [0.03676992 0.03679132 0.0369333  0.03685095 0.00270417]
         [0.0368589  0.03679132 0.03697799 0.03689543 0.4754235 ]]
        '''
        return l2_norm, partition
