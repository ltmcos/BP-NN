'''
Neural Network(NN) for Iris Datas using Backward Propagation(BP)
------------------------
Li Tianming
Institue of Physics, CAS
2022.3.31
------------------------
Cao Zhendong
Institue of Physics, CAS
2022.4.02
'''

import pandas as pd
from pandas.plotting import radviz
import numpy as np
import datetime
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmod(input):

    # avoid large input for exp
    # if input >= 0:
    #     return 1.0 / (1 + np.exp(-input))
    # else:
    #     return np.exp(input) / (1 + np.exp(input))
    # ans = np.where(input<0, np.exp(input)/(1+np.exp(input)), 1.0/(1+np.exp(-input)))
    # return np.where(input<0, np.exp(input)/(1+np.exp(input)), 1.0/(1+np.exp(-input)))
    return 1.0 / (1 + np.exp(-input))

# frompyfunc(func, cin_num, cout_num)
# sigmod_np =  np.frompyfunc(sigmod, 1, 1)

def dsigmod(input):

    # ans = sigmod(input) * (1 - sigmod(input))
    return sigmod(input) * (1 - sigmod(input))

def relu(input):
    return np.where(input<0, 0, input)

def drelu(input):
    return np.where(input<0, 0, 1.0)

def lrelu(input):
    return np.where(input<0, 0.01*input, input)

def dlrelu(input):
    return np.where(input<0, 0.01, 1.0)

def change_label(data_y):

    labels = {'0':[1,0,0], '1':[0,1,0], '2':[0,0,1]}
    data_new = []
    for idx in range(len(data_y)):
        data_new.append(labels[str(data_y[idx])])
    data_new = np.array(data_new)
    return data_new

class nn_paras(object):

    def __init__(self, n_paras, func='sigmod', Type='float32', ini_seed=2):

        # def nn_model size
        self.ni = n_paras['ni']
        self.nh = n_paras['nh']
        self.no = n_paras['no']

        # def net_paras
        np.random.seed(ini_seed)
        # randn: loc=0, sigma=1; same as .normal(loc, scale, size)
        self.wh = np.random.randn(self.nh, self.ni).astype(Type)
        self.wo = np.random.randn(self.no, self.nh).astype(Type)
        
        # def cache_data
        self.ai = np.zeros(shape=(self.ni, 1)).astype(Type)
        self.ah = np.zeros(shape=(self.nh, 1)).astype(Type)
        self.ao = np.zeros(shape=(self.no, 1)).astype(Type)

        # def excitation function
        # func in ['sigmod', 'relu', 'lrelu']
        self.func = eval(func)
        self.dfunc = eval('d'+func)
        self.Type = Type

    def forward_propagation(self, data_x):

        # '''for codes''' are same as the below np.array expression
        # However, np.array ones 10 times faster
        '''
        for i in range(self.ni):
            self.ai[i] = data_x[i]
        '''
        self.ai = data_x.astype(self.Type)

        '''
        for j in range(self.nh):
            # sum = 0.0
            # for i in range(self.ni):
            #     sum += self.wh[j][i] * self.ai[i]
            sum = np.matmul(self.wh[j,:], self.ai)
            self.ah[j] = self.func(sum)
        '''
        sum = np.matmul(self.wh, self.ai).astype(self.Type)
        self.ah = self.func(sum)
        
        '''
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += self.wo[k][j] * self.ah[j]
            self.ao[k] = self.dfunc(sum)
        '''
        sum = np.matmul(self.wo, self.ah).astype(self.Type)
        self.ao = self.func(sum)

    def compute_error(self, label_y):

        '''
        error = 0.0
        for k in range(self.no):
            error += 0.5 * (label_y[k] - self.ao[k])**2
        '''
        error = np.sum(0.5*(label_y-self.ao)**2)
        return error
    
    def backward_propagation(self, label_y, learning_rate=0.4):

        # error for output
        '''
        output_deltas = np.zeros(shape=(self.no, 1))
        for k in range(self.no):
            error = label_y[k] - self.ao[k]
            output_deltas[k] = error * self.dfunc(self.ao[k])
        '''
        label_y = np.array(label_y).astype(self.Type)
        error = label_y - self.ao
        output_deltas = (error * self.dfunc(self.ao)).reshape((self.no, 1))

        # error for hidden
        '''
        hidden_deltas = np.zeros(shape=(self.nh, 1))
        for j in range(self.nh):
            # error = 0.0
            # for k in range(self.no):
            #     error += output_deltas[k] * self.wo[k][j]
            error = np.matmul(output_deltas.T, self.wo[:,j].reshape((self.no, 1)))
            hidden_deltas[j] = error * self.dfunc(self.ah[j])
        '''
        error = np.matmul(self.wo.T, output_deltas).astype(self.Type)
        hidden_deltas = error * self.dfunc(self.ah).reshape((self.nh, 1))

        # update grads
        '''
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[k][j] += learning_rate * change
        '''
        change = output_deltas * self.ah.T
        self.wo += learning_rate * change
        '''
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wh[j][i] += learning_rate * change
        '''
        change = hidden_deltas * self.ai.T
        self.wh += learning_rate * change 

def predict(paras, datas):

    num = datas.shape[0]
    result = []
    for idx in range(num):
        data_x = datas[idx]
        paras.forward_propagation(data_x)
        # print('Test data {} prediction: {}'.format(idx, paras.ao))

        # method 1: maximum
        max = -100
        label = 3 # 3 for unknown
        for i in range(paras.no):
            if paras.ao[i] > max:
                max = paras.ao[i]
                label = i
        # result.append(label)

        # method 2: half, contain unknown
        ans = []
        label = 3
        labels = {'0':[1,0,0], '1':[0,1,0], '2':[0,0,1]}
        for i in range(paras.no):
            if paras.ao[i] > 0.5:
                ans.append(1)
            else:
                ans.append(0)
        for i in range(len(labels)):
            if ans == labels[str(i)]:
                label = i
        result.append(label)

        # print('Test data {} prediction: {}'.format(idx, label))
    return np.array(result)

def result_visualization(datas_x, label_y, predict_y, headers):

    labels = ['setosa','versicolor','virginica','unknown']
    color_box = ['blue', 'green', 'red', 'yellow']

    y_label = []
    y_predict = []
    for i in range(len(label_y)):
        y_label.append([labels[label_y[i]]])
        y_predict.append([labels[predict_y[i]]])

    # combine datas and label as DataFrame
    x_data = pd.DataFrame(datas_x, index=None)
    y_label = pd.DataFrame(y_label, index=None)
    y_predict = pd.DataFrame(y_predict, index=None)
    
    df_real = pd.concat([x_data, y_label], axis=1)
    df_predict = pd.concat([x_data, y_predict], axis=1)
    df_real.columns = headers
    df_predict.columns = headers

    plt.figure(num='real')
    radviz(df_real, 'Species', color=color_box, alpha=0.5)
    # plt.legend(loc='upper right')

    plt.figure(num='predict')
    radviz(df_predict, 'Species', color=color_box, alpha=0.5)
    # plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":

    # automatic file path
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

    # load datasets
    dataset_all = pd.read_csv(dirname + '/iris.csv') # header=None means file has no header
    dataset_all.sort_values(by='Species', ascending=True)
    headers = dataset_all.columns.tolist()

    '''
    adjustable paras:
        seed: for generate net_paras wi & wh
        state: for randomly split datasets
        iterations: for the times iterating all train_data
        print_error: print error or not
        n_paras: structure for 3-layer NN_model
        restore_type: 'float16', 'float32', 'float64'
        excit_func: 'sigmod', 'relu', 'lrelu'
    '''
    # random paras
    seed = 2
    state = 14
    learning_rate = 0.01
    iterations = 1000
    print_error = True
    restore_type = 'float32'
    excit_func = 'lrelu'
    print('Excited func: {}'.format(excit_func))

    # define n_paras
    n_paras = {'ni':4, 'nh':10, 'no':3}

    # sklearn: split dataset
    # random split: random_state, int; test_size, 0-1 rate; shuffle; stratify
    train_data, test_data = train_test_split(dataset_all, test_size=0.2, random_state=state, stratify=dataset_all.loc[:,'Species'])

    # change data_like
    train_data_x = train_data.iloc[:,0:4].values # iloc for index
    train_data_y = train_data.loc[:,'Species'].values # loc for label name
    train_data_y = change_label(train_data_y)

    test_data_x = test_data.iloc[:,0:4].values
    test_data_y = test_data.loc[:,'Species'].values

    # build nn_model
    start_time = datetime.datetime.now()
    nn_model = nn_paras(n_paras, func=excit_func, Type=restore_type, ini_seed=seed)

    for time in range(iterations):
        num = train_data.shape[0]

        for idx in range(num):
            data_x = train_data_x[idx]
            label_y = train_data_y[idx]
            nn_model.forward_propagation(data_x)
            nn_model.backward_propagation(label_y,learning_rate)

        if print_error and (time+1)%(iterations/10) == 0:
            error = 0.0
            for idx in range(num):
                data_x = train_data_x[idx]
                label_y = train_data_y[idx]
                nn_model.forward_propagation(data_x)
                error += nn_model.compute_error(label_y)
            error = error / num
            print('Iterated %4d times, error is %f'%((time+1), error))
    
    end_time = datetime.datetime.now()
    print("Time cost: " + str((end_time-start_time).seconds) + 's' + str(round((end_time-start_time).microseconds/1000)) + 'ms')

    # test NN model
    result = predict(nn_model, test_data_x)
    print('Predict:', result)
    print('Real   :', test_data_y)

    error = 0
    num = result.shape[0]
    for i in range(num):
        if result[i] != test_data_y[i]:
            error += 1
    print('Error rate: %d/%d = %.2f%%'%(error, num, (error/num)*100))
    result_visualization(test_data_x, test_data_y, result, headers)
