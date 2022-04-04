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
    I = np.matrix(np.eye(A.shape[0]))
    gammaW = np.matrix(np.eye(A.shape[0]))

    K_f = P_fm*C.T*(C*P_fm*C.T + R).I

    xhat_fp = xhat_fm + K_f*(y - C*xhat_fm)
    
    xhat_fm = A*xhat_fp + B*u

    P_fp = (I - K_f*C)*P_fm
    P_fm = A*P_fp*A.T + gammaW*Q*gammaW.T

    return xhat_fp, xhat_fm, P_fp, P_fm

def ekf(y, x0, f, h, Q, R, u, P0=None):
    '''
    everything should be a matrix 
    '''

    nx = len(x0)
    if P0 is None:
        P0 = np.matrix(np.eye(nx)*100)

    xhat_fp = None
    P_fp = []
    P_fm = [P0]
    xhat_fm = x0

    for i in range(y.shape[1]):
        A, B = jacobian(f, np.ravel(xhat_fm[:, -1]), np.ravel(u[:, i]))
        C, D = jacobian(h, np.ravel(xhat_fm[:, -1]), np.ravel(u[:, i]))
        
        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __kalman_forward_update__(xhat_fm[:, -1], P_fm[-1], y[:, i], u[:, i],
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