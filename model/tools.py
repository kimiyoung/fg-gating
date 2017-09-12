import numpy as np

def sub_sample(m_d, m_c, N):
    """
    mask everything except N tokens before and after the candidates
    """
    m = np.copy(m_c)
    for i in range(N):
        m += np.pad(m_c, ((0,0),(i+1,0)), mode='constant')[:,:-(i+1)] + \
                np.pad(m_c, ((0,0),(0,i+1)), mode='constant')[:,(i+1):]
    m[m.nonzero()] = 1
    return m_d*m

