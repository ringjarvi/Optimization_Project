import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mlxtend.data import loadlocal_mnist
from scipy.special import softmax
from autograd import grad


X, y = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-labels-idx1-ubyte')


X = X[0:60000,:]
X = X/255
y = y[0:60000,]
y = np.equal.outer(y,np.unique(y)).astype(int)


X_test = X_test[0:10000,:]
X_test = X_test/255
y_test = y_test[0:10000,]
y_test = np.equal.outer(y_test,np.unique(y_test)).astype(int)
'''
def softmax(x):
    ps = np.empty(x.shape)
    for i in range(x.shape[0]):
        ps[i,:] = np.exp(x[i,:] - np.max(x[i,:]))
        ps[i,:]/= np.sum(ps[i,:])
    return ps
'''
def h(X,w):
    #The logistic function
    #X : Data
    #w : Weightsdef softmax(x):
    return softmax(X@w, axis=1)

def cross_entropy(X, w, y):
    #The Cross Entropy Function
    #X : Data
    #w : Weights
    #y : labels
    #lam : regularization coefficient for Tikonoff Regularization
    eps = 1e-18
    classes = 10
    a = h(X, w)
    return (1/classes)*np.sum(-(y*np.log(a+eps)))

def cross_entropy_grad(X,w,y):
    #First uses correct function to zero out any values of 1's that match y one-hot encoded
    a=h(X,w)
    grad = X.T@(a - y)
    return grad

def backtracking(xk, dk, feval, grad, alpha0, rho, c):
    fvk = feval(xk)
    gk = grad(xk)
    alpha = alpha0
    while feval(xk + alpha*dk) > fvk + c*alpha*LA.norm(gk.T@dk):
        alpha = rho*alpha
    return alpha

def eval(X,w,y):

    a=np.argmax(softmax(X@w),axis=1).tolist()
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


def steepest_descent(w0, feval, grad, stepsize, options = {'gtol':1e-6,'MaxIter':1e6,'disp':True, 'disp_interval':100}):

    """
    x0: initialization for the algorithm
    feval: function handle that evaluates the objective function value at a given point
    grad: function handle that evaluates the gradient value at a given point
    stepsize: a function handle to compute stepsize given the current point and the direction.
              Using a function handle allows us to deal with different stepsize schemes such
              as constant, exact line search, and Wolfes' cnoditions
    options: options for the algorithms that provides the gradient tolerance for stopping, display, etc.
    """
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
    ng = LA.norm(g)
    history = []
    iter_count = 0
    if disp:
        print('{:15s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm w', 'Step Size'))
    while ng > gtol and iter_count <= MaxIter: # Use the gradient as a stopping criteria
        alpha = stepsize(w, g, iter_count)
        w = w - alpha*g
        # Record the history
        fv = feval(w)
        g = grad(w)
        ng = LA.norm(g)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, alpha))

    return w, fv, g, history, iter_count
'''
w0 = np.random.normal(0,.1, (X.shape[1], y.shape[1]))

eps = 1e-3
MaxIter = 20000
disp_interval = 10
options = {'gtol':eps, 'MaxIter':MaxIter, 'disp_interval':disp_interval}

lam = 0

feval = lambda w: cross_entropy(X, w, y)
grad = lambda w: cross_entropy_grad(X, w, y)

stepsize = lambda wk, dk, i: 1e-1
#stepsize = lambda wk, dk, i : backtracking(wk, dk, feval, grad, 1, 0.4, 1e-4)
w, fv, g, history, iter_count1 = steepest_descent(w0, feval, grad, stepsize, options)

# extract the function values and gradient norms
history_fv1 = [item[0] for item in history]
history_ng1 = [item[1] for item in history]

evaluation = eval(X,w,y)
print('The model evaluation is:', evaluation)

testeval = eval(X_test,w,y_test)
print('The test model evaluation is:', testeval)


plt.plot(np.log10(history_ng1))
plt.title('Steepest Descent')
plt.xlabel('Iteration $k$', fontsize = 20)
plt.ylabel(r'$\log_{10}\|\nabla f(x_k)\|_2^2$', fontsize = 20)
plt.show()
'''

def SGD(X, w0, y, batch_size, feval, grad, stepsize, options = {'gtol':1e-6,'MaxIter':1e3,'disp':True, 'disp_interval':100}):

    """
    x0: initialization for the algorithm
    feval: function handle that evaluates the objective function value at a given point
    grad: function handle that evaluates the gradient value at a given point
    stepsize: a function handle to compute stepsize given the current point and the direction.
              Using a function handle allows us to deal with different stepsize schemes such
              as constant, exact line search, and Wolfes' cnoditions
    options: options for the algorithms that provides the gradient tolerance for stopping, display, etc.
    """
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
    a = np.random.randint(0, X.shape[0])
    Xs = X[a, :].reshape(1,784)
    ys = y[a, :].reshape(1,10)
    fv = feval(Xs, w, ys)
    g = grad(Xs, w, ys)
    ng = np.linalg.norm(g)
    history = []
    iter_count = 0
    if disp:
        print('{:15s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm w', 'Step Size'))
    while ng > gtol and iter_count <= MaxIter:
        #g = 0
        fv = 0
        for i in np.arange(0,batch_size):
            a = np.random.randint(0, X.shape[0])
            Xs = X[a, :].reshape(1,784)
            ys = y[a, :].reshape(1,10)
            fv += feval(Xs, w, ys)/batch_size
            g += grad(Xs, w, ys)/batch_size
        alpha = stepsize(w, -g, iter_count)
        w = w - alpha*g
        ng = np.linalg.norm(g)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, alpha))

    return w, fv, g, history, iter_count

w0 = np.random.normal(0,1, (X.shape[1], y.shape[1]))
batch_size = 4
eps = 1e-6
MaxIter = 200000
disp_interval = 10
options = {'gtol':eps, 'MaxIter':MaxIter, 'disp_interval':disp_interval}

feval = lambda X, w, y: cross_entropy(X, w, y)
grad = lambda X, w, y: cross_entropy_grad(X, w, y)

stepsize = lambda wk, dk, i: 1/(i+1)
w, fv, g, history, iter_count1 = SGD(X, w0, y, batch_size, feval, grad, stepsize, options)

# extract the function values and gradient norms
history_fv1 = [item[0] for item in history]
history_ng1 = [item[1] for item in history]

evaluation = eval(X,w,y)
print('The training model evaluation is:', evaluation)

testeval = eval(X_test,w,y_test)
print('The test model evaluation is:', testeval)

plt.plot(np.log10(history_ng1))
plt.title('SGD')
plt.xlabel('Iteration $k$', fontsize = 20)
plt.ylabel(r'$\log_{10}\|\nabla f(x_k)\|_2^2$', fontsize=20)
plt.show()


