
from os.path import join
import torch
from RNN import BiRNN, RNN
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import SmoothL1Loss, MSELoss
from path import MODEL_DIRECTORY

class Trainer():

    def __init__(self, DataLoader, fund_id, logging):

        self.DataLoader = DataLoader
        self.fund_id = fund_id
        self.logging = logging

    def _train(self):

        self.logging.debug('--- training ---')

        rnn = BiRNN(self.DataLoader.input_size, self.DataLoader.hidden_size, self.DataLoader.num_layers, self.DataLoader.output_size).cuda()
        criterion = MSELoss().cuda()
        optimizer = Adam(rnn.parameters(), lr=self.DataLoader.lr)

        pre_loss, after_loss = 1, 0

        for epoch in range(self.DataLoader.num_epochs):
            for i, (material, close) in enumerate(self.DataLoader.loader):

                material = Variable(material.view(-1, self.DataLoader.time_steps, self.DataLoader.input_size)).cuda()
                close = Variable(close.float()).cuda()

                optimizer.zero_grad()
                outputs = rnn(material)
                loss = criterion(outputs, close)
                loss.backward()
                optimizer.step()

                after_loss = loss.data[0]

                if (i + 1) % 8 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'
                          % (epoch + 1, self.DataLoader.num_epochs, i + 1, len(self.DataLoader.training_data_X) // self.DataLoader.batch_size, loss.data[0]))

            if after_loss < pre_loss:
                self.logging.debug('save new model ---> {}, loss={}'.format(self.fund_id, after_loss))
                torch.save(rnn, join(MODEL_DIRECTORY, 'model_{}.pkl'.format(self.fund_id)))

            pre_loss = after_loss



    def _valuate(self):

        self.logging.debug('--- testing ---')

        predict_close, predict_trend = [], []
        real_close, real_trend = [], []

        Model = torch.load(join(MODEL_DIRECTORY, 'model_{}.pkl').format(self.fund_id))

        for material, close in self.DataLoader.testloader:

            material = Variable(material.view(-1,  self.DataLoader.time_steps, self.DataLoader.input_size)).cuda()
            outputs = Model(material)
            pred = outputs.cpu().data.numpy()[0]  # result

            # trend : up/down from train data and test data
            # real data
            ' considering 1,0,-1'
            # pre_trend = [1 if self.DataLoader.testing_data_Y[0][i+1] - self.DataLoader.testing_data_Y[0][i] > 0 else 0 if self.DataLoader.testing_data_Y[0][i+1] - self.DataLoader.testing_data_Y[0][i] == 0 else -1 for i in range(self.DataLoader.span-1)]
            # pre_trend = [1 if self.DataLoader.testing_data_Y[0][0] - self.DataLoader.training_data_Y[-1, -1] > 0 else 0 if self.DataLoader.testing_data_Y[0][0] - self.DataLoader.training_data_Y[-1, -1] == 0 else -1] + pre_trend

            ' considering 1,-1'
            pre_trend = [
                1 if self.DataLoader.testing_data_Y[0][i + 1] - self.DataLoader.testing_data_Y[0][i] >= 0 else -1 for i in
                range(self.DataLoader.span - 1)]
            pre_trend = [1 if self.DataLoader.testing_data_Y[0][0] - self.DataLoader.training_data_Y[
                -1, -1] > 0 else -1] + pre_trend

            real_close.append(close.numpy())
            real_trend = pre_trend

            # predict data
            ' considering 1,0,-1'
            # trend = [1 if pred[i+1] - pred[i] > 0 else 0 if pred[i+1] - pred[i] == 0 else -1 for i in range(self.DataLoader.span-1)]
            # trend = [1 if pred[0] - self.DataLoader.training_data_Y[-1, -1] > 0 else -1 if pred[0] - self.DataLoader.training_data_Y[-1, -1] == 0 else 0] + trend

            ' considering 1,-1'
            trend = [1 if pred[i + 1] - pred[i] > 0 else -1 for i in range(self.DataLoader.span - 1)]
            trend = [1 if pred[0] - self.DataLoader.training_data_Y[-1, -1] > 0 else -1] + trend

            self.logging.debug('the final value of training data ={}'.format(self.DataLoader.training_data_Y[-1, -1]))
            self.logging.debug('the trend ={}'.format(trend))
            self.logging.debug('predict ={}'.format(pred))

            # trend : compare the real trend to predict trend
            pre_trend = trend
            # predict_trend = [0.5 if i == j else 0 for i, j in zip(pre_trend, trend)]

            ''' predict the true value '''
            predict_close.append(pred)
            # predict_close.append((close.numpy() - abs(close.numpy() - pred)) / close.numpy() * 0.5)

        return predict_close, pre_trend, real_close, real_trend
