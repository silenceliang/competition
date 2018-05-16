import os
from os.path import join
import csv
import pandas as pd
import numpy as np
from model import dataLoader, fund
from train import Trainer
from matplotlib import pyplot as plt
from collections import OrderedDict
from path import ETF_DIRECTORY
from SP500_cov import sp500_date_close
from parameters import ANS_SHEET
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler('record.log', 'w', 'utf-8')])

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

''' read Data by csv '''
def file_reader(dir):
    f = os.listdir(dir)
    for data in f:
        with open(join(dir, data), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)

''' read Data by pandas '''
def file_pd_reader(dir):

    f = os.listdir(dir)
    result = 0
    for data in f:
        pd_file = pd.read_csv(join(dir, data))

        if type(result) == type(pd_file):
            result = pd.concat([result, pd_file])
        else:
            result = pd_file
        # print(pd.unique(pd_file[list(pd_file)[0]]), len(pd.unique(pd_file[list(pd_file)[0]])))
    return result

def init():

    fund_pd = file_pd_reader(dir=ETF_DIRECTORY)
    fund_id = pd.unique(fund_pd[list(fund_pd)[0]])
    ALLFUND = fund(fund_id)
    price_matrix = fund_pd.as_matrix()

    for row in price_matrix:
        ALLFUND._put(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        # here we have the all fund Data from official .csv

    ''' append sp500 as feature '''
    ALLFUND.combine_SP500(sp500_date_close())

    return ALLFUND

def main():

    fund_obj = init()
    naive_fund = fund_obj._history()

    for n, i in enumerate(naive_fund):
        logging.info('len(fund_{})={}'.format(fund_obj.fund_id[n], len(i)))

    answer_sheet = open(ANS_SHEET, 'w')
    real = open('real_data.csv', 'w')

    accumulation_trends = []

    for i, fund in enumerate(naive_fund):

        dl = dataLoader(np.asarray(fund))
        trainer = Trainer(dl, fund_obj.fund_id[i], logging=logging)
        trainer._train()
        predict_arr, trend, real_arr, real_trend = trainer._valuate()

        my_dict = OrderedDict()
        my_dict['id'] = fund_obj.fund_id[i]
        close_arr = list(predict_arr[0])
        real_arr = list(real_arr[0])
        x = my_dict.copy()
        date = ['1 / 22', '1 / 23', '1 / 24', '1 / 25', '1 / 26']
        tt = ['t1', 't2', 't3', 't4', 't5']

        for (d, close, t1, t) in zip(date, real_arr[0], tt, real_trend):
            x[t1] = t
            x[d] = close
        w = csv.DictWriter(real, x.keys())
        if i == 0: w.writeheader()
        w.writerow(x)

        for (d, close, t1, t) in zip(date, close_arr, tt, trend):
            # if t == 0.5:
            #     accumulation_trends.append(1)
            # else:
            #     accumulation_trends.append(0)
            if t == 1:
                accumulation_trends.append(1)
            else:
                accumulation_trends.append(0)

            my_dict[t1] = t
            my_dict[d] = close

        ''' write to answer sheet along with different funds '''
        w = csv.DictWriter(answer_sheet, my_dict.keys())
        if i == 0: w.writeheader()
        w.writerow(my_dict)

    logging.info('The accuracy about rise/down : %.4f' % (sum(accumulation_trends[0:len(accumulation_trends)]) / (len(accumulation_trends) + 0.01)))
    answer_sheet.close()

    # print('--- plotting ---')
    # plt.title('loss graph')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # print(loss)
    # plt.plot(range(len(loss)), loss, label='loss curve')
    # plt.legend(loc='upper right')
    # plt.show()


if __name__ == '__main__':
    main()
