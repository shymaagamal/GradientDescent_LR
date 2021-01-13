
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def computeCost(X,Y,weights):
    m=len(X)
    cost= (1/(2*m))*np.sum(np.power((X.dot(weights)-Y),2))
    return cost

def gradientDescent(X, Y, alpha,weights):
     m=len(X)
     
     h=X.dot(weights)
     loss=h-Y
     g=(X.T.dot(loss))/m
     weights=weights-alpha * g
     
     return weights  


def fit(X_train,Y_train,alpha=0.0001,iteration=1000): 

    weights=np.zeros((X_train.shape[1],1))
    cost=np.zeros(iteration) 
    for i in range(iteration):
        weights=gradientDescent(X_train,Y_train,alpha,weights)
        cost[i]=computeCost(X_train,Y_train,weights)
    return weights ,cost

def predict(weights,X_test):
    Y_predict=X_test.dot(weights)
    return Y_predict


def EvaluatePerformance(Y,Y_pred):
    def R2(Y,Y_pred):
        mean_y = np.mean(Y)
        ss_tot = np.sum(np.power((Y - mean_y) , 2))
        ss_res = np.sum(np.power((Y - Y_pred),  2))
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def MAE(Y,Y_pred):
        mae=np.sum(np.power((Y-Y_pred),2))/len(Y)
        return mae    

    return R2(Y,Y_pred), MAE(Y,Y_pred) 

def preprocessing(data):
    data.insert(0,'Bias',1)
    col=data.shape[1]
    X=data.iloc[:,:col-1]
    Y=data.iloc[:,col-1:col]
    X=np.matrix(X)
    Y=np.matrix(Y)

    return X , Y

def plot_TT_Curves(X,Y,learningRate,iteration):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        weight,cost=fit(X_train[:m], y_train[:m],learningRate,iteration)
        y_train_predict = predict(weight,X_train[:m])
        y_test_predict =predict(weight,X_test)
        train_errors.append(EvaluatePerformance(y_train[:m], y_train_predict)[1])
        test_errors.append(EvaluatePerformance(y_test, y_test_predict)[1])
    plt.xlabel('Train Size')
    plt.ylabel('Error')    
    plt.plot(train_errors, "r-+", linewidth=2, label="train")
    plt.plot(test_errors, "b-", linewidth=3, label="test")
    plt.legend()   

data=pd.read_csv('univariateData.dat',header=None,names=['X','Y'])
data.head(1)

data.describe()


plt.scatter(data['X'],data['Y']);
plt.title('Relation Between Input and Output ');
plt.xlabel('X')
plt.ylabel('Y');


X1, Y1= preprocessing(data)

X_train1, X_test1, y_train1, y_test1 = train_test_split( X1, Y1, test_size=0.2, random_state=42)
weight1 ,cost1=fit(X_train1,y_train1,.001,2000)
y_p1=predict(weight1,X_test1)
costTrain1=computeCost(X_train1,y_train1,weight1)

r21,mae1=EvaluatePerformance(y_test1,y_p1);
print('costTrain  =',costTrain1)
print('\nr2       =',r21)
print('\nmae       =',mae1)
print('\nWeights =>',weight1.T)

plot_TT_Curves(X1,Y1,0.001,2000)

iteration1=np.array(range(0, 2000))
cost1=cost1
plt.plot(iteration1,cost1)
plt.title('Relation between Iteration and Cost Function')
plt.xlabel('iteration')
plt.ylabel('cost function');

x = np.linspace(data['X'].min(), data['X'].max(), 100)
f =weight1[0, 0] + (weight1[1, 0] * x)


fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction Data')
ax.scatter(data['X'], data['Y'], label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('X')
ax.set_ylabel('Y');

LR = LinearRegression()
LR = LR.fit(X_train1, y_train1)
Y_pred1 = LR.predict(X_test1)
mse1 = mean_squared_error(y_test1, Y_pred1)
r21 = r2_score(y_test1, Y_pred1)
print('r2  =',r21)
print('\nmsa =',mse1)


data2=pd.read_csv('multivariateData.dat',header=None,names=['X1','X2','Y'])
data2.head(1)

data2=(data2-data2.mean())/data2.std()
data2.head(1)

features = ['X1','X2'] 
scatter_matrix(data2[features]) 
plt.show() 


X2, Y2= preprocessing(data2)


X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size=0.2, random_state=42)
weight ,cost=fit(X_train,y_train,.001,5000)
y_p=predict(weight,X_test)
costTrain=computeCost(X_train,y_train,weight)
r2,mae=EvaluatePerformance(y_test,y_p);
print('costTrain  =',costTrain)
print('\nr2       =',r2)
print('\nmae      =',mae)
print('\nWeights =>',weight.T)

plot_TT_Curves(X2,Y2,0.001,5000);
iteration=np.array(range(0, 5000))
cost=cost
plt.plot(iteration,cost)
plt.title('Relation between Iteration and Cost Function')
plt.xlabel('iteration')
plt.ylabel('cost function');

LR2 = LinearRegression()
LR2 = LR2.fit(X_train, y_train)
Y_pred2 = LR2.predict(X_test)
mse = mean_squared_error(y_test, Y_pred2)
r2 = r2_score(y_test, Y_pred2)
print('r2  =',r2)
print('\nmsa =',mse)


