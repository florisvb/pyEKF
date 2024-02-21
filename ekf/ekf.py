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
    
    return np.matrix(np.vstack(Aj)), np.matrix(np.vstack(Bj))

def __kalman_forward_update__(xhat_fm, P_fm, y, u, A, B, C, R, Q):
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

    xhat_fp = xhat_fm + K_f@(y - C@xhat_fm)
    
    xhat_fm = A@xhat_fp + B@u

    P_fp = (I - K_f@C)@P_fm
    P_fm = A@P_fp@A.T + gammaW@Q@gammaW.T

    return xhat_fp, xhat_fm, P_fp, P_fm

def ekf_numerical(y, x0, f, h, Q, R, u, P0=None):
    '''
    everything should be a matrix 
    '''

    nx = x0.shape[0]
    if P0 is None:
        P0 = np.matrix(np.eye(nx)*100)

    xhat_fp = None
    P_fp = []
    P_fm = [P0]
    xhat_fm = x0

    for i in range(y.shape[1]):
        A, B = jacobian(f, np.ravel(xhat_fm[:, -1:]), np.ravel(u[:, i:i+1]))
        C, D = jacobian(h, np.ravel(xhat_fm[:, -1:]), np.ravel(u[:, i:i+1]))
        
        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __kalman_forward_update__(xhat_fm[:, -2:], P_fm[-1], y[:, i:i+1], u[:, i:i+1],
                                                                     A, B, C, R, Q)
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

    return xhat_fm[:,0:-1], P_fm[0:-1], s

def ekf_symbolic(Y, x0, f, h, X_s, U_s, Q, R, U, P0=None):
    '''
    y -- measurements
    x0 -- initial state guess
    f -- function f(X,U) that returns symbolic discrete time dynamics
    h -- function h(X,U) that returns symbolic discrete time measurements 
    X -- list of states, symbolic (sympy) variables
    U -- list of control inputs, symbolic (sympy) variables
    Q, R -- covariance matrices
    u -- control inputs
    P0 -- initial covariance guess
    '''

    nx = x0.shape[0]
    if P0 is None:
        P0 = np.matrix(np.eye(nx)*100)

    xhat_fp = None
    P_fp = []
    P_fm = [P0]
    xhat_fm = x0

    # Jacobian of dynamics with respect to state (symbolic A matrix)
    A_s = f_sd(X_s, U_s).jacobian(X_s)

    # Jacobian of dynamics with respect to controls (symbolic B matrix)
    B_s = f_sd(X_s, U_s).jacobian(U_s)

    # Jacobian of measurements with respect to state (symbolic C matrix)
    C_s = h_sd(X_s, U_s).jacobian(X_s)

    # Jacobian of measurements with respect to controls (symbolic D matrix)
    D_s = h_sd(X_s, U_s).jacobian(U_s)

    for i in range(Y.shape[1]):
        sub_x = {X_s[n]: xhat_fm[n, -1] for n in range(xhat_fm.shape[0])}
        sub_u = {U_s[m]: U[m, i] for m in range(U.shape[0])}

        A = np.array( A_s.subs(sub_x).subs(sub_u) ).astype(float)
        B = np.array( B_s.subs(sub_x).subs(sub_u) ).astype(float)
        C = np.array( C_s.subs(sub_x).subs(sub_u) ).astype(float)
        D = np.array( D_s.subs(sub_x).subs(sub_u) ).astype(float)
        
        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __kalman_forward_update__(xhat_fm[:, -1:], 
                                                                     P_fm[-1], 
                                                                     Y[:, i:i+1], 
                                                                     U[:, i:i+1],
                                                                     A, B, C, D, 
                                                                     R, Q)
        if xhat_fp is None:
            xhat_fp = _xhat_fp
        else:
            xhat_fp = np.hstack((xhat_fp, _xhat_fp))
        xhat_fm = np.hstack((xhat_fm, _xhat_fm))
        
        P_fp.append(_P_fp)
        P_fm.append(_P_fm)

    s = np.zeros([nx,Y.shape[1]]);
    for i in range(nx):
        s[i,:] = [np.sqrt( P_fm[j][i,i].squeeze() ) for j in range(Y.shape[1])]

    return xhat_fm[:,0:-1], P_fm[0:-1], s