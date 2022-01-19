import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxopt
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score #used for sklearn
from sklearn.metrics import mean_squared_error #used for sklearn

data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+")
data = data.values
X= data[:,0:13]
Y= data[:,13].reshape((506,1))

def min_max_scaler(x_train,x_test):
    mi = np.min(x_train, axis=0)
    ma = np.max(x_train,axis= 0)
    x_train = (x_train-mi)/(ma-mi)
    x_test = (x_test-mi)/(ma-mi)
    return x_train,x_test 

def rsquared_score(Y_test, Y_pred):
    data_var = np.sum((Y_test-np.mean(Y_test,axis=0))**2)
    model_var = np.sum((Y_pred-Y_test)**2)
    expl_var = data_var-model_var
    return float(expl_var/data_var)

def mse_error(Y_pred,Y_test):
    return np.sum((Y_pred-Y_test)**2)/(len(Y_pred))

def linear(x,y,c=0):
    return np.dot(x,y)

def poly(x,y,d):
    return (1+(np.dot(x.T,y))**d)
    
def rbf(x,y,gamma):
    return np.exp(-(np.sum((x-y)**2)*gamma))
   
def eps_svr(X_train,Y_train,X_test,kernel,epsilon, c,kernel_param):
    """implements the CVXOPT version of epsilon SVR"""
    m, n = X_train.shape      #m is num samples, n is num features
    #Finding the kernels i.e. k(x,x')
    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernel(X_train[i,:], X_train[j,:], kernel_param)
    
    P= np.hstack((k,-1*k))
    P= np.vstack((P,-1*P))
    q= epsilon*np.ones((2*m,1))
    qadd=np.vstack((-1*Y_train,Y_train))
    q=q+qadd
    A=np.hstack((np.ones((1,m)),-1*(np.ones((1,m)))))
    
    #define matrices for optimization problem       
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.zeros((1,1)))
    
    c= float(c)
    temp=np.vstack((np.eye(2*m),-1*np.eye(2*m)))
    G=cvxopt.matrix(temp)
    
    temp=np.vstack((c*np.ones((2*m,1)),np.zeros((2*m,1))))
    h = cvxopt.matrix(temp)
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    #lagrange multipliers
    l = np.ravel(sol['x'])
    #extracting support vectors i.e. non-zero lagrange multiplier
    alpha=l[0:m]
    alpha_star=l[m:]
    
    bias= sol['y']
    print("bias="+str(bias))
    #find weight vector and predict y
    Y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        res=0
        for u_,v_,z in zip(alpha,alpha_star,X_train):
            res+=(u_ - v_)*kernel(X_test[i],z,kernel_param)
        Y_pred[i]= res
    Y_pred = Y_pred+bias[0,0]
    
    return Y_pred

def sklearn_svr(X_train, Y_train,X_test, ker_index, eps,reg_param, kernel_param): 
    """implements the SKlearn version of SVR
    ker_index is 1 for linear
    ker_index is 2 for polynimial
    ker_index is 3 for rbf"""
    if(ker_index ==1 ):
        regressor = SVR(kernel='linear', epsilon=eps, C=reg_param)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    elif(ker_index == 2):
        regressor = SVR(kernel='poly', epsilon=eps,degree=kernel_param, C=reg_param, gamma=1)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    else :
        regressor= SVR(kernel = 'rbf', epsilon = eps, C= reg_param, gamma=kernel_param)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    return y_pred

def rh_svr(X_train,Y_train,X_test,kernel,epsilon,c, kernel_param):
    """implements the RH_SVR algorithm """
    m, n = X_train.shape      #m is num samples, n is num features
    #scale Y_train as it is a feature as well
    Y_train_max=Y_train.max(axis=0)
    Y_train_min=Y_train.min(axis=0)
    Y_train =(Y_train-Y_train_min)/(Y_train_max-Y_train_min) 
    #Finding the kernels i.e. k(x,x')
    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernel(X_train[i,:], X_train[j,:], kernel_param)
    
    temp = k+np.dot(Y_train,Y_train.T)
    P_temp= np.hstack((temp,-1*temp))
    P= np.vstack((P_temp,-1*P_temp))
    q= 2*epsilon*np.vstack((Y_train,-1*Y_train))
    
    temp1=np.hstack((np.ones((1,m)),np.zeros((1,m))))
    temp2=np.hstack((np.zeros((1,m)),np.ones((1,m))))
    A=np.vstack((temp1,temp2)) 
    #define matrices for optimization problem  
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.ones((2,1)))
    c= float(c)
    temp=np.vstack((np.eye(2*m),-1*np.eye(2*m)))
    G=cvxopt.matrix(temp)
    
    temp1=np.vstack((c*np.ones((2*m,1)),np.zeros((2*m,1))))
    h = cvxopt.matrix(temp1)
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    #lagrange multipliers
    l = np.ravel(sol['x'])
    #print(np.sort(l,axis=0))
    #extracting support vectors i.e. non-zero lagrange multiplier
    u_cap=l[0:m]
    v_cap=l[m:]
    delta=np.dot((u_cap-v_cap).T,Y_train)+2*epsilon
    u_bar=u_cap/delta
    v_bar=v_cap/delta
    #print(u_cap)
    #print(v_cap)
    #u1=u_bar > 1e-5
    #v1=v_bar > 1e-5
    #SV=np.logical_or(u1, v1)
    #indices = np.arange(len(l)/2)[SV]
    #u=u_bar[indices.astype(int)]
    #v=v_bar[indices.astype(int)]
    #support_vectors_x = X_train[SV]
    #support_vectors_y = Y_train[SV]
    #calculate intercept
    bias= np.dot((u_cap-v_cap).T,k)
    bias= np.dot(bias,(u_cap+v_cap))
    bias = bias/(2*delta)+np.dot((u_cap+v_cap).T,Y_train)/2
    #print('bias is ', bias)
    #find weight vector and predict y
    Y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        res=0
        for u_,v_,z in zip(u_bar,v_bar,X_train):
            res+=(v_ -u_)*kernel(X_test[i],z,kernel_param)
        Y_pred[i]= res
    Y_pred = Y_pred+bias
    Y_pred=Y_pred*(Y_train_max-Y_train_min)+Y_train_min
    return Y_pred



def compare_model(X,Y,model,kernel,ker_index,epsilon,C,kernel_param): #ker_index is 1 for linear, 2 for poly, 3 for rbf
    """to compare the performance of the models(primarily epsilon svr CVXOPT and sklearn)
    used to generate other plots for RH-SVR as well""" 
    kf = KFold(n_splits = 5,random_state=None, shuffle = False)
    scores_1 = []
    mse_scores1=[]
    scores_2  = []
    mse_scores2=[]
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index].reshape((len(X_test),1))
        #scale the data by min-max scaler
        X_train,X_test = min_max_scaler(X_train,X_test)
        #get predictions from cvxopt model
        pred=model(X_train,Y_train,X_test,kernel,epsilon,C,kernel_param).reshape((len(Y_test),1))
        scores_1.append(rsquared_score(Y_test,pred))
        mse_scores1.append(mse_error(Y_test,pred))
        #get predictions from sklearn model
        y_pred = sklearn_svr(X_train,Y_train,X_test, ker_index, epsilon, C, kernel_param).reshape((len(Y_test),1))
        scores_2.append(r2_score(Y_test,y_pred))
        mse_scores2.append(mean_squared_error(Y_test,y_pred))
    print("MSE")
    print("cvxopt===>"+str(np.around((mse_scores1),decimals=5)))
    print("sklearn===>" +str(np.around((mse_scores2), decimals=5)))
    print("R2")
    print("cvxopt===>"+str(np.around((scores_1),decimals=5)))
    print("sklearn===>" +str(np.around((scores_2), decimals=5)))
    return float(np.around(np.mean(scores_1),decimals=5)),float(np.around(np.mean(scores_2),decimals=5)),float(np.around(np.mean(mse_scores1),decimals=5)),float(np.around(np.mean(mse_scores2),decimals=5))

def plot_eps_c_3D(eps_vals,c_vals,model, ker, ker_index):
    """generates 3D plot for R2 ves eps and c for the given model and parameter arrays"""
    ax = plt.axes(projection='3d')
    plt.figure()
    for c in c_vals:
            r1=[]
            r2=[]
            a1=[]
            a2=[]
            for eps in eps_vals:
                t1,t2,m1,m2=compare_model(X,Y,model,linear,1,eps, c,1)
                r1.append(t1)
                r2.append(t2)
                a1.append(m1)
                a2.append(m2)
            ax.scatter(eps_vals, [c]*len(eps_vals), r1,s=25, marker='o',color='g')
            #ax.scatter(vals1, [c]*len(eps_vals), r1, s=25,marker='x',color='r')
            ax.set_xlabel('epsilon')
            ax.set_ylabel('c values')
            ax.set_zlabel('R2 coeff')
            ax.set_title('R2 for RH_SVR CVXOPT')
    plt.legend()
    plt.show()

"""uncomment to get a 3D plot"""
#plot_eps_c_3D(np.arange(0.05,1,0.05),np.arange(0.005,0.1,0.005) ,rh_svr,linear, 1)


"""the code below is used to generate the various plots
use eps_svr and rh_svr for models
linear, poly, rbf for kernels
1         2     3    as kernel index respectively"""
vals1= [0.1,0.5,1,1.5,2,2.5,3,5]
vals2 = [4.5]
for eps in vals2:
    r1=[]
    r2=[]
    a1=[]
    a2=[]
    for c in vals1:
        t1,t2,m1,m2=compare_model(X,Y,eps_svr,poly,2,eps,c,3)
        r1.append(t1)
        r2.append(t2)
        a1.append(m1)
        a2.append(m2)
    print(r1)
    print(a1)    
    plt.figure()
    plt.plot(vals1,r1,label="eps-SVR poly", marker='o')
    """comment if plotting only for RH-SVR"""
    plt.plot(vals1,r2,label="sklern poly", marker='x')#comment this line
    plt.xlabel("C")
    plt.ylabel("R-squared coeff")
    #plt.title("eps="+str(eps))
    plt.legend()
    plt.show()        
        
    plt.figure()
    plt.plot(vals1,a1,label="epsilon-SVR poly", marker='o')
    """comment if plotting only for RH-SVR"""
    plt.plot(vals1,a2,label="SKLearn poly", marker='x')#comment this line
    plt.xlabel("C")
    plt.ylabel("MSE")
    #plt.title("eps="+str(eps))
    plt.legend()
    plt.show()