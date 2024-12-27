import numpy as np

P0 = np.zeros((9, 9))
np.fill_diagonal(P0, 0.01**2)
print(P0)

F1k=np.eye((len(P0)))

F1k[0,0]=4
print(F1k)

Qk=np.array([[0.01 ,      0. ,        0.      ],[0.,         0.01   ,    0.        ],[0.  ,       0.    ,     0.00761544]])
print(len(Qk))
F2k=np.zeros((len(Qk),1))
print(F2k)
Qsk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2])) 
print(Qsk)
