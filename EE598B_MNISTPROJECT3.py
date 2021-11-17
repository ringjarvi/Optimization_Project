import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mlxtend.data import loadlocal_mnist
from autograd import grad
import scipy.optimize as opt
from scipy.special import softmax

X, y = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-labels-idx1-ubyte')

size = 200
test_size = 100
X = X[0:size,:]
X = X/255
y = y[0:size,]
y = np.equal.outer(y,np.unique(y)).astype(int)

y_0 = y[:,0]
y_0 = y_0.reshape(size,1)

y_1 = y[:,1]
y_1 = y_1.reshape(size,1)

y_2 = y[:,2]
y_2 = y_2.reshape(size,1)

y_3 = y[:,3]
y_3 = y_3.reshape(size,1)

y_4 = y[:,4]
y_4 = y_4.reshape(size,1)

y_5 = y[:,5]
y_5 = y_5.reshape(size,1)

y_6 = y[:,6]
y_6 = y_6.reshape(size,1)

y_7 = y[:,7]
y_7 = y_7.reshape(size,1)

y_8 = y[:,8]
y_8 = y_8.reshape(size,1)

y_9 = y[:,9]
y_9 = y_9.reshape(size,1)

X_test = X_test[0:test_size,:]
X_test = X_test/255
y_test = y_test[0:test_size,]
y_test = np.equal.outer(y_test,np.unique(y_test)).astype(int)

def sigmoid(x):
    epsl = 1e-18
    return 1 / (1 + np.exp(-x + epsl))

def h(X,w):
    #The logistic function
    #X : Data
    #w : Weightsdef softmax(x):
    return sigmoid(X@w)

def cross_entropy(X, w, y):
    #The Cross Entropy Function
    #X : Data
    #w : Weights
    #y : labels
    #lam : regularization coefficient for Tikonoff Regularization
    a = h(X, w)
    classes = 2
    epsl = 1e-18
    return (1/classes)*np.sum(-y*np.log(a + epsl) - (1 - y)*np.log(1 - a + epsl))


def cross_entropy_grad(X,w,y):
    #First uses correct function to zero out any values of 1's that match y one-hot encoded
    a=h(X,w)
    grad = X.T@(a - y)
    return grad

def backtracking(xk, dk, feval, grad, alpha0, rho, c):
    fvk = feval(xk)
    gk = grad(xk)
    alpha = alpha0
    while feval(xk + alpha*dk) > fvk + c*alpha*gk.T@dk:
        alpha = rho*alpha
    return alpha

def eval(X,w,y):

    a=np.argmax(X@w,axis=1).tolist()
    b=np.argmax(y,axis=1).tolist()
    total_percent=0
    o=0
    t=0
    th=0
    f=0
    fi=0
    s=0
    se=0
    e=0
    n=0
    z=0
    for i in range(X.shape[0]):
            if a[i]==b[i]:
                total_percent+=1
                if a[i]==1:
                    o+=1
                if a[i]==2:
                    t+=1
                if a[i]==3:
                    th+=1
                if a[i]==4:
                    f+=1
                if a[i]==5:
                    fi+=1
                if a[i]==6:
                    s+=1
                if a[i]==7:
                    se+=1
                if a[i]==8:
                    e+=1
                if a[i]==9:
                    n+=1
                if a[i]==0:
                    z+=1
    print("Total percent correct=",(total_percent/X.shape[0])*100)
    print("Percent of 0's correct=", z/b.count(0)*100)
    print("Percent of 1's correct=", o/b.count(1)*100)
    print("Percent of 2's correct=", t/b.count(2)*100)
    print("Percent of 3's correct=", th/b.count(3)*100)
    print("Percent of 4's correct=", f/b.count(4)*100)
    print("Percent of 5's correct=", fi/b.count(5)*100)
    print("Percent of 6's correct=", s/b.count(6)*100)
    print("Percent of 7's correct=", se/b.count(7)*100)
    print("Percent of 8's correct=", e/b.count(8)*100)
    print("Percent of 9's correct=", n/b.count(9)*100)

def L_BFGS(w0, feval, grad, m, options = {'gtol':1e-6,'MaxIter':1e3,'disp':True, 'disp_interval':100}):

    if 'gtol' in options:
        gtol = options['gtol']
    else:
        gtol = 1e-6

    if 'disp' in options:
        disp = options['disp']
    else:
        disp = True

    if 'MaxIter' in options:
        MaxIter = options['MaxIter']
    else:
        MaxIter = 1e3

    if 'disp_interval' in options:
        disp_interval = options['disp_interval']
    else:
        disp_interval = 100
    w = w0
    fv = feval(w)
    g = grad(w)
    ng = np.linalg.norm(g)
    alpha = backtracking(w, -g, feval, grad, 1, 0.5, 1e-4)
    s = -alpha*g
    y = grad(w+s) - g
    Memory = [[s, y]]
    history = []
    iter_count = 0

    def two_loop(q0, Memory):
        q = np.concatenate(q0,axis = 0).ravel(order = 'F')
        q = q.reshape(784,1)
        m = len(Memory)
        eps = 1e-18
        alpha = np.zeros((m,1))
        for i, (s, y) in enumerate(reversed(Memory)):
            if np.abs(y.T@s) > 1e-8:
                rho = 1/(y.T@s + eps)
            else:
                rho = 1e5
            alpha[-(i+1)] = rho*(s.T@q)
            q = q - alpha[-(i+1)]*y
        s, y = Memory[-1]
        gamma = (s.T@y)/(y.T@y + eps)
        r = gamma*q
        for i, (s, y) in enumerate(Memory):
            if np.abs(y.T@s) > 1e-8:
                rho = 1/(y.T@s + eps)
            else:
                rho = 1e5
            beta = rho*(y.T@r)
            r = r + s*(alpha[i]-beta)
        return r

    if disp:
        print('{:15s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm', 'Step Size'))

    while ng > eps and iter_count <= MaxIter:
        d = -two_loop(g, Memory)
        alpha = backtracking(w, d, feval, grad, 1, 0.5, 1e-4)
        w_prev = w
        w = w_prev + alpha*d
        s = w - w_prev
        g_prev = g
        g = grad(w)
        ng = LA.norm(g)
        y = g - g_prev
        if len(Memory) >= m:
            Memory.pop(0)
        Memory.append([s,y])
        fv = feval(w)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, alpha))
    return w, fv, g, history, iter_count

w0 = np.random.normal(0,1,(X.shape[1],1))
m = 20
eps = 1e-3
MaxIter = 2000
disp_interval = 1
options = {'gtol':eps, 'MaxIter':MaxIter, 'disp_interval':disp_interval}

feval = lambda w: cross_entropy(X, w, y_0)
grad = lambda w: cross_entropy_grad(X, w, y_0)

w_0, fv_0, g_0, history_0, iter_count_0 = L_BFGS(w0, feval, grad, m, options)

history_fv1_0 = [item[0] for item in history_0]
history_ng1_0 = [item[1] for item in history_0]

feval = lambda w: cross_entropy(X, w, y_1)
grad = lambda w: cross_entropy_grad(X, w, y_1)

w_1, fv_1, g_1, history_1, iter_count_1 = L_BFGS(w0, feval, grad, m, options)

history_fv1_1 = [item[0] for item in history_1]
history_ng1_1 = [item[1] for item in history_1]

feval = lambda w: cross_entropy(X, w, y_2)
grad = lambda w: cross_entropy_grad(X, w, y_2)

w_2, fv_2, g_2, history_2, iter_count_2 = L_BFGS(w0, feval, grad, m, options)

history_fv1_2 = [item[0] for item in history_2]
history_ng1_2 = [item[1] for item in history_2]

feval = lambda w: cross_entropy(X, w, y_3)
grad = lambda w: cross_entropy_grad(X, w, y_3)

w_3, fv_3, g_3, history_3, iter_count_3 = L_BFGS(w0, feval, grad, m, options)

history_fv1_3 = [item[0] for item in history_3]
history_ng1_3 = [item[1] for item in history_3]

feval = lambda w: cross_entropy(X, w, y_4)
grad = lambda w: cross_entropy_grad(X, w, y_4)

w_4, fv_4, g_4, history_4, iter_count_4 = L_BFGS(w0, feval, grad, m, options)

history_fv1_4 = [item[0] for item in history_4]
history_ng1_4 = [item[1] for item in history_4]

feval = lambda w: cross_entropy(X, w, y_5)
grad = lambda w: cross_entropy_grad(X, w, y_5)

w_5, fv_5, g_5, history_5, iter_count_5 = L_BFGS(w0, feval, grad, m, options)

history_fv1_5 = [item[0] for item in history_5]
history_ng1_5 = [item[1] for item in history_5]

feval = lambda w: cross_entropy(X, w, y_6)
grad = lambda w: cross_entropy_grad(X, w, y_6)

w_6, fv_6, g_6, history_6, iter_count_6 = L_BFGS(w0, feval, grad, m, options)

history_fv1_6 = [item[0] for item in history_6]
history_ng1_6 = [item[1] for item in history_6]

feval = lambda w: cross_entropy(X, w, y_7)
grad = lambda w: cross_entropy_grad(X, w, y_7)

w_7, fv_7, g_7, history_7, iter_count_7 = L_BFGS(w0, feval, grad, m, options)

history_fv1_7 = [item[0] for item in history_7]
history_ng1_7 = [item[1] for item in history_7]

feval = lambda w: cross_entropy(X, w, y_8)
grad = lambda w: cross_entropy_grad(X, w, y_8)

w_8, fv_8, g_8, history_8, iter_count_8 = L_BFGS(w0, feval, grad, m, options)

history_fv1_8 = [item[0] for item in history_8]
history_ng1_8 = [item[1] for item in history_8]

feval = lambda w: cross_entropy(X, w, y_9)
grad = lambda w: cross_entropy_grad(X, w, y_9)

w_9, fv_9, g_9, history_9, iter_count_9 = L_BFGS(w0, feval, grad, m, options)

history_fv1_9 = [item[0] for item in history_9]
history_ng1_9 = [item[1] for item in history_9]

w = np.concatenate((w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9)).reshape((-1, 10), order='F')


evaluation = eval(X,w,y)
print('The training model evaluation is:', evaluation)

testeval = eval(X_test,w,y_test)
print('The test model evaluation is:', testeval)


plt.plot(np.log10(history_ng1_0),label = '0')
plt.plot(np.log10(history_ng1_1),label = '1')
plt.plot(np.log10(history_ng1_2),label = '2')
plt.plot(np.log10(history_ng1_3),label = '3')
plt.plot(np.log10(history_ng1_4),label = '4')
plt.plot(np.log10(history_ng1_5),label = '5')
plt.plot(np.log10(history_ng1_6),label = '6')
plt.plot(np.log10(history_ng1_7),label = '7')
plt.plot(np.log10(history_ng1_8),label = '8')
plt.plot(np.log10(history_ng1_9),label = '9')

plt.title('L-BFGS')
plt.legend(loc = 'upper right')
plt.xlabel('Iteration $k$', fontsize = 20)
plt.ylabel(r'$\log_{10}\|\nabla f(x_k)|_2^2$', fontsize = 20)
plt.show()
