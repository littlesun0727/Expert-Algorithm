import numpy as np 
import os
from sklearn import metrics
from data import load_spambase
from options import parser
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from FCN import FCNet


def expert(args, x_train, y_train, x_test, y_test, expert_model="Logistic Regression"):
    '''
    create experts: Logistic Regression, SVM(Linear), Fully-connected NN, Bayes Classification, 
    '''
    if expert_model == "Logistic Regression":

        logisticRegr = LogisticRegression(penalty='none',max_iter=10000)
        logisticRegr.fit(x_train, y_train)

        score = logisticRegr.score(x_test, y_test)
        result = "Accuracy for logistic regression: {}".format(score)
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)

        return logisticRegr

    if expert_model == "Logistic Regression(l2)":

        logisticRegr = LogisticRegression(penalty="l2",max_iter=1000)
        logisticRegr.fit(x_train, y_train)

        score = logisticRegr.score(x_test, y_test)
        result = "Accuracy for logistic regression(l2): {}".format(score)
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)

        return logisticRegr
    
    if expert_model == "SVM(Linear)":
        
        clf = svm.SVC(kernel='linear') # Linear Kernel
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result = "Accuracy for SVM(Linear): {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)

        return clf
    
    if expert_model == "SVM(rbf)":
        
        clf = svm.SVC(kernel="rbf", C=2.8, gamma=.0073,verbose=10) # rbf Kernel
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result = "Accuracy for SVM(rbf): {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)

        return clf

        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        #              'C': [1, 10, 100, 1000]},
        #             {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        # clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='precision_macro')
        # clf.fit(x_train, y_train)

        # y_pred = clf.predict(x_test)
        # print("Accuracy for SVM(rbf):",metrics.accuracy_score(y_test, y_pred))

        # return clf

    
    if expert_model == "Bayes":
        clf = GaussianNB()
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result = "Accuracy for Bayes: {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)
        return clf
    
    if expert_model == "Decision Tree":
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result="Accuracy for Decision Tree: {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)
        return clf       
    
    if expert_model == "Fully-connected NN(15)":
        clf = MLPClassifier([15,2],learning_rate_init= 0.001,activation='relu',solver='adam', alpha=0.0001,max_iter=30000)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result="Accuracy for Fully-connected NN(15): {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)
        return clf

    
    if expert_model == "Fully-connected NN(10)":
        clf = MLPClassifier([10,2],learning_rate_init= 0.001,activation='relu',solver='adam', alpha=0.0001,max_iter=30000)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result="Accuracy for Fully-connected NN(10): {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)
        return clf
    
    if expert_model == "Fully-connected NN(5)":
        clf = MLPClassifier([5,2],learning_rate_init= 0.001,activation='relu',solver='adam', alpha=0.0001,max_iter=30000)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        result="Accuracy for Fully-connected NN(5): {}".format(metrics.accuracy_score(y_test, y_pred))
        filename="results/experts.txt"
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')
        print(result)
        return clf

def train(args, alpha, experts, X, y, p, Theta, epoch):
    global Loss, Loss_experts, ps

    window,eta = args.batch_size, args.eta
    n_obs, n_experts = len(X), len(experts)

    for i in range(0,n_obs,window):
        data, target = X[i:i+window,:], y[i:i+window]

        y_preds = np.zeros((n_experts, data.shape[0]))
        for i_exp, expert in enumerate(experts):
            y_preds[i_exp] = expert.predict(data)
        
        y_pred = y_preds.transpose().dot(p)
        
        loss_experts = np.linalg.norm(y_preds-target,axis=1) ##MSE_loss
        loss = np.linalg.norm(y_pred-target)
        Loss.append(loss); Loss_experts.append(loss_experts)

        p = Theta.dot(p*np.exp(-eta*loss_experts))
        p = p/p.sum()
        
        y_hat = y_pred>0.5
        accu = np.mean(y_hat==target)

        result = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccu: {:.2f}\tExpertsLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                epoch, i+len(data), len(X),100. * (i+len(data)) / len(X), loss,accu*100.,loss_experts[0],loss_experts[1],loss_experts[2],loss_experts[3],loss_experts[4],loss_experts[5])
        filename = 'results/train_{}_{}.txt'.format(alpha,args.eta)
        with open(filename, 'a') as fp: 
            fp.write(result+'\n')

        result_p = 'Train Epoch: {} [{}/{} ({:.0f}%)] {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                epoch, i+len(data), len(X),100. * (i+len(data)) / len(X),p[0],p[1],p[2],p[3],p[4],p[5])
        filename = 'results/p_{}_{}.txt'.format(alpha,args.eta)
        with open(filename, 'a') as fp: 
            fp.write(result_p+'\n')

        if i%(window*10) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccu:{:.2f}'.format(
                epoch, i+len(data), len(X),100. * (i+len(data)) / len(X), loss,accu*100.))
        ps.append(p)

    return p

def test(args, alpha, experts, X, y, p, epoch):
    print(p, p.sum())
    y_preds = np.zeros((len(experts), X.shape[0]))
    for i_exp, expert in enumerate(experts):
        y_preds[i_exp] = expert.predict(X)
    
    y_pred = y_preds.transpose().dot(p)
    loss = np.linalg.norm(y_pred-y)
    y_hat = y_pred>0.5
    accu = np.mean(y_hat==y)
    result = 'Test Epoch: {} \tLoss: {:.6f}\tAccu:{:.2f}'.format(epoch, loss,accu*100.)
    filename = 'results/test_{}_{}.txt'.format(alpha,args.eta)
    with open(filename, 'a') as fp: 
        fp.write(result+'\n')
    print(result)



if __name__=="__main__":
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_cuda = use_cuda
    args.device = torch.device("cuda" if use_cuda else "cpu")

    x_train,y_train,x_test,y_test = load_spambase()

    try:
        filename = 'results/experts.txt'
        os.remove(filename)
        print('Removed previous experts!')
    except:
        pass

    expert_models = ["Logistic Regression", "Logistic Regression(l2)", "Bayes", "Fully-connected NN(15)","Fully-connected NN(10)","Fully-connected NN(5)"]
    experts = []
    for expert_model in expert_models:
        exp = expert(args, x_train, y_train, x_test, y_test, expert_model)
        experts.append(exp)
    
    alphas = [0, 0.001, 0.01, 0.05, 0.1]
    for alpha in alphas:
        try:
            filename = 'results/train_{}_{}.txt'.format(alpha,args.eta)
            os.remove(filename)
            filename = 'results/p_{}_{}.txt'.format(alpha,args.eta)
            os.remove(filename)
            filename = 'results/test_{}_{}.txt'.format(alpha,args.eta)
            os.remove(filename)
            print('Removed previous results!')
        except:
            pass
        
        n_models = len(experts)
        p = np.ones(n_models)/n_models
        Theta = (1-alpha*n_models/(n_models-1))*np.eye(n_models)+alpha/(n_models-1)

        global Loss, loss_experts
        Loss, Loss_experts, ps = [], [], []

        for epoch in range(1, args.epochs + 1):
            p = train(args, alpha, experts, x_train, y_train, p, Theta, epoch)
            test(args, alpha, experts, x_test, y_test, p, epoch)
        

        


    

