import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import matplotlib.patches as mpatches
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS

"""                 functions"""               
"""                 preprocessing"""
def preproc(data):
    kernel_lenght=100
    (T,N)=data.shape
    mu = np.zeros(shape=(T-kernel_lenght+1,N)) 
    stand=np.zeros(shape=(T-kernel_lenght+1,N))
    for t in range( 0,(T - kernel_lenght + 1)):
        for j in range(0,N):
            dd=data[t :t+kernel_lenght - 1,j]
            mu[t,j] = np.mean(dd)
            stand[t,j] = np.std(dd)
    ff =int(kernel_lenght /2)
    ddd=data[ ff:T-ff+1,0:N]
    proc_data=(ddd-mu)/stand
    return proc_data
"""           eigen vector finder """
def eigen(A,k):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    idx_smallest=idx[0:k]
    eigenVectors = eigenVectors[:,idx_smallest]
    return eigenVectors
"""             computing T matrix"""
def Tmatrix(S,k):
    d=np.sum(S,axis=0)
    D=np.diag(d)
    L=D-S
    inv_D=np.diag(d**(-0.5))
    L_rw=np.matmul(inv_D,  L)
    L_sym=np.matmul(L_rw,  inv_D)
    U=eigen(L_sym,k)
    T = normalize(U, axis=1, norm='l2')
    return T
"""               map clusters to colors"""
def createcolor(C):
    strs = ["" for x in range(len(C))]
    for p in range(len(C)):
        if C[p]==0:
            strs[p]='r'
        if C[p]==1:
            strs[p]='b'
        if C[p]==2:
            strs[p]='y'
        if C[p]==3:
            strs[p]='palegreen'
        if C[p]==4:
            strs[p]='k'
        if C[p]==5:
            strs[p]='c'
        if C[p]==6:
            strs[p]='pink'
        if C[p]==7:
            strs[p]='salmon'
        if C[p]==8:
            strs[p]='darkgreen'
        if C[p]==9:
            strs[p]='skyblue'
        if C[p]==10:
            strs[p]='saddlebrown'
    return strs
         
######################################  
"""                  main part"""
              
"""                  number of clusters""" 
k=11   

"""                  reading data"""
       
mydata = read_csv('data.csv',header=None, engine='python')
GR = read_csv('gr.csv',header=None, engine='python')
GR = GR.values
dataset = mydata.values

"""                  calculating log returns"""
        
returns=dataset[1:1509]/dataset[0:1508]
log_returns=np.log10(returns)
proc_data=preproc(log_returns)

"""                Constructing similarity matrix by using corelations"""
            
similarities=np.abs(np.corrcoef(proc_data,rowvar=False))
(C,C)=similarities.shape
similarities=similarities-np.identity(C)

"""                   Spectral clustering by scikit learn"""
           
clustering = SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels="kmeans",random_state=0).fit(similarities)
labels=clustering.labels_

"""                    using MDS to decreasing dimension"""
         
T=Tmatrix(similarities,k)
embedding = MDS(n_components=2)
X_t = embedding.fit_transform(T)

"""                       plot legends creator"""
        
r_patch = mpatches.Patch(color='r', label='Health Care')
b_patch = mpatches.Patch(color='b', label='Materials')
y_patch = mpatches.Patch(color='y', label='Consumer Discretionary')
p_patch = mpatches.Patch(color='palegreen', label='Information Technology')
k_patch = mpatches.Patch(color='k', label='Financials')
c_patch = mpatches.Patch(color='c', label='Industrials')
pi_patch = mpatches.Patch(color='pink', label='Energy')
sa_patch = mpatches.Patch(color='salmon', label='Real Estate')
da_patch = mpatches.Patch(color='darkgreen', label='Consumer Staples')
sk_patch = mpatches.Patch(color='skyblue', label='Utilities')
sad_patch = mpatches.Patch(color='saddlebrown', label='Telecommunications Services')

"""                          Plots """
       
plt.figure(figsize=(15,8))
plt.scatter(X_t[:,0],X_t[:,1])
plt.savefig('plt2.pdf',dpi=400)
plt.show()
colors=createcolor(labels)
plt.figure(figsize=(15,8))
plt.scatter(X_t[:,0],X_t[:,1],c=colors)
plt.savefig('plt5.pdf',dpi=400)
plt.show()
colors2=createcolor(GR)
plt.figure(figsize=(15,8))
plt.scatter(X_t[:,0],X_t[:,1],c=colors2)
plt.legend(handles=[r_patch,b_patch,y_patch,p_patch,k_patch,c_patch,pi_patch,sa_patch,sk_patch,sad_patch],prop={'size': 9})
plt.savefig('plt6.pdf',dpi=400)
plt.show()
