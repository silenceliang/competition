import pandas as pd
from path import SP500_DIRECTORY
from datetime import datetime

def sp500_date_close():

    sp = pd.read_csv(SP500_DIRECTORY)

    d = sp['Date'].copy()

    for i, date in enumerate(d):
        # x = datetime.strptime(d[i], '%m/%d/%y')
        x = datetime.strptime(d[i], '%Y-%m-%d')
        sp['Date'][i] = x.strftime('%Y%m%d')

    sp500_dic = dict()
    for i in range(0, len(sp)):
        sp500_dic[sp['Date'][i]] = sp['Close'][i]

    return sp500_dic

