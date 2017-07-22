# coding: UTF-8
import os
import numpy as np
import matplotlib.pyplot as plt

K = 2   # Class
N_total = 6000

#   Unknown
mu = np.array([1.0,-1.0]) 
sigma = np.array([1.0,1.0]) #[1.0,1.0], [1.0,0.5]
alpha = np.array([0.5, 0.5]) #[0.5, 0.5], [0.3, 0.7] 
N = [int(N_total*alpha[0]), int(N_total*alpha[1])] 
s0 = np.random.normal(mu[0],sigma[0],N[0]) 
s1 = np.random.normal(mu[1],sigma[1],N[1]) 

#   Known.  Variables, which we want to infer.
data = np.append(s0,s1)
data_size = np.sum(N) 
identity = np.ones(data_size)

def gauss(x,mean,std):
    exp = np.exp( - ( x - mean )**2 / (2.0 * std**2) ) 
    normal = 1.0 / ( std * np.sqrt(2.0 * np.pi) ) 
    return exp * normal

def visualize_s(mu,sigma,alpha):
    pi2 = 2 * np.pi
    
    # plot of s0
    count0, bins0, ignored0 = plt.hist(s0, 30, normed=True, label='dist0')
    plot0, = plt.plot(bins0, gauss(bins0, mu[0], sigma[0]), linewidth=2, color='r')
     
    # plot of s1
    count1, bins1, ignored1= plt.hist(s1, 30, normed=True, label='dist1')
    plot1, = plt.plot(bins1, gauss(bins1, mu[1], sigma[1]), linewidth=2, color='b')
    
    # plot of s0+s1
    count01, bins01, ignored01 = plt.hist(s0+s1, 30, normed=True, alpha=0.5, label='dist0 + dist1')
     
    plt.legend()
    #plt.legend([plot0,plot1],['dist0','dist1'])
    plt.savefig("image.png")
    plt.show()

def init_variables():
    global mu_hat, sigma_hat, alpha_hat
    
    mu_hat = np.array([-2.0,1.5]) #np.random.uniform(-1.0, 1.0 , K)
    sigma_hat = np.array([2.0,0.5]) #np.random.uniform(0.1, 2.0, K)
    alpha_hat = np.array([0.7,0.3]) #np.random.uniform(0.0, 1.0, K) ;alpha_hat = alpha_hat / np.sum(alpha_hat)

def EM():
    global mu_hat, sigma_hat, alpha_hat 
    p0_hat = alpha_hat[0] * gauss(data, mu_hat[0], sigma_hat[0]) 
    p1_hat = alpha_hat[1] * gauss(data, mu_hat[1], sigma_hat[1]) 
    
    sum_p0_p1 = p0_hat+p1_hat
    w0 = np.copy(p0_hat) / np.copy(sum_p0_p1)
    w1 = np.copy(p1_hat) / np.copy(sum_p0_p1)
    
    #-------- Expectation step --------#  and  #-------- Maximization step --------#   
    N0, N1 = np.sum(w0), np.sum(w1)
    alpha_hat[0], alpha_hat[1] = N0/data_size, N1/data_size
    sigma_hat[0]= np.sqrt( np.sum( (data - mu_hat[0]*identity) * (data - mu_hat[0]*identity) * w0 ) / N0 )
    sigma_hat[1]= np.sqrt( np.sum( (data - mu_hat[1]*identity) * (data - mu_hat[1]*identity) * w1 ) / N1 )
    mu_hat[0], mu_hat[1] = np.sum( data * w0  ) / N0  , np.sum( data * w1 ) / N1

if __name__ == "__main__":
    global mu_hat, sigma_hat, alpha_hat 
    
    #visualize_s(mu,sigma,alpha)    # True parameter.
    init_variables()
    converge_step = 300 
    stoping_criteria = 1e-2
    for t in range(converge_step):
        EM()
        error = np.sum(abs(mu_hat-mu)) + np.sum(abs(sigma_hat-sigma)) \
                +np.sum(abs(alpha_hat-alpha))
        print t," ", error
        if(error < stoping_criteria):
            print "Difference between true parameter and estimate parameter is sufficiently small."
            break
     
    print "mu_hat = \n", mu_hat
    print "sigma_hat = \n", sigma_hat
    print "alpha_hat = \n", alpha_hat
    visualize_s(mu_hat,sigma_hat,alpha_hat)    # Estimated parameter.

