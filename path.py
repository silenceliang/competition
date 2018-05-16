import os
from os.path import join


''' defined the file path '''
WORK_DIRECTORY = os.getcwd()
DATA_DIRECTORY = join(WORK_DIRECTORY, "Data")
MODEL_DIRECTORY = join(WORK_DIRECTORY, "Model")
MODEL_SP500_DIRECTORY = join(MODEL_DIRECTORY, "model_sp500")
RNN_MODEL_DIRECTORY = join(MODEL_DIRECTORY, "rnn_model")
ETF_DIRECTORY = join(DATA_DIRECTORY, "ETF")
SP500_DIRECTORY = join(DATA_DIRECTORY, "SP500/S&P500.csv")

