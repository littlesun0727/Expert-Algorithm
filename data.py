import numpy as np
from sklearn.model_selection import train_test_split

def load_spambase():
    """load digit1 data"""
    f = open('./spambase/spambase.data', 'r')
    x, y= [], []
    for line in f.readlines():
        line = line.strip('\n')
        line_list = line.split(',')
        
        if line_list[-1]=='0':
            label = 0
        elif line_list[-1]=='1':
            label = 1
        y.append(label)
        
        data = [float(elem) for elem in line_list[:-1]]
        x.append(data)
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train,y_train,x_test,y_test