import sympy as sp
import numpy as np 
import scipy.optimize

def jacobian(f, x0, u0, epsilon=0.001):
    
    # Get A
    Aj = []
    for i in range(len(f(x0,u0))):
        
        def f_scalar(x,u,i):
            x_new = f(x, u)
            return np.ravel(x_new)[i]
        
        j = scipy.optimize.approx_fprime(x0, f_scalar, epsilon, u0, i)
        Aj.append(j)
        
    # Get B
    Bj = []
    for i in range(len(f(x0,u0))):
        
        def f_scalar(u,x,i):
            x_new = f(x, u)
            return np.ravel(x_new)[i]
        
        j = scipy.optimize.approx_fprime(u0, f_scalar, epsilon, x0, i)
        Bj.append(j)
    
    return np.array(np.vstack(Aj)), np.array(np.vstack(Bj))

def __extended_kalman_forward_update__(xhat_fm, P_fm, y, u, A, B, C, D, R, Q, h, f):
    """
    Linear kalman update equations

    :param xhat_fm:
    :param P_fm:
    :param y:
    :param u:
    :param A:
    :param B:
    :param C:
    :param R:
    :param Q:
    :return:
    """


    I = np.array(np.eye(A.shape[0]))
    gammaW = np.array(np.eye(A.shape[0]))

    K_f = P_fm@C.T@np.linalg.inv(C@P_fm@C.T + R)

    #xhat_fp = xhat_fm + K_f@(y - (C@xhat_fm+D@u))
    xhat_fp = xhat_fm + K_f@(y - h(xhat_fm, u))
    
    #xhat_fm = A@xhat_fp + B@u
    xhat_fm = f(xhat_fp, u)

    P_fp = (I - K_f@C)@P_fm
    P_fm = A@P_fp@A.T + gammaW@Q@gammaW.T

    return xhat_fp, xhat_fm, P_fp, P_fm

def ekf(y, x0, f, h, Q, R, u, P0=None):
    '''
    y -- 2D array of measurements, rows = measurements; columns = time points
    x0 -- 2D array of initial state (guess), rows = states; 1 column
    f -- discrete dynamics function that takes (state, control) as an input and returns a 2D (1 column) array
    h  -- discrete measurement function that takes (state, control) as an input and returns a 2D (1 column) array
    Q, R -- 2D square arrays corresponding to process and measurement covariance matrices, respectively
    u -- 2D array of controls, rows = measurements; columns = time points
    P0 -- optional, 2D array of initial error covariance (guess)
    '''

    nx = x0.shape[0]
    if P0 is None:
        P0 = np.array(np.eye(nx)*100)

    xhat_fp = None
    P_fp = []
    P_fm = [P0]
    xhat_fm = x0

    for i in range(y.shape[1]):
        
        A, B = jacobian(f, np.ravel(xhat_fm[:, -1:]), np.ravel(u[:, i:i+1]))
        C, D = jacobian(h, np.ravel(xhat_fm[:, -1:]), np.ravel(u[:, i:i+1]))

        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __extended_kalman_forward_update__(xhat_fm[:, -1:], P_fm[-1], y[:, i:i+1], u[:, i:i+1],
                                                                              A, B, C, D, R, Q, h, f)
        if xhat_fp is None:
            xhat_fp = _xhat_fp
        else:
            xhat_fp = np.hstack((xhat_fp, _xhat_fp))
        xhat_fm = np.hstack((xhat_fm, _xhat_fm))
        
        P_fp.append(_P_fp)
        P_fm.append(_P_fm)

    s = np.zeros([nx,y.shape[1]]);
    for i in range(nx):
        s[i,:] = [np.sqrt( P_fm[j][i,i].squeeze() ) for j in range(y.shape[1])]

    return xhat_fm[:,0:-1], np.dstack(P_fm[0:-1]), s

