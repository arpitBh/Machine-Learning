import json
import random
import numpy as np

def findPosterior(X,K,pi,mu,cov):
    posterior = np.zeros((len(X),K))
    for i in range(len(X)):
        for j in range(K):
            n = X[i].shape[0]

            posterior[i,j] = pi[j]*dmvnorm(X[i],mu[j],cov[j])
        posterior[i] /= (np.sum(posterior[i]))
    return posterior

def det(cov_list):
    return cov_list[0]*cov_list[3] - cov_list[1]*cov_list[2]

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []

    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    # Run 100 iterations of EM updates
    prob = np.zeros((len(X),K))
    for t in range(100):
        new_pi = [0 for q in range(K)]
        new_mu = [0 for q in range(K)]
        new_cov = [[]for q in range(K)]
        for i in range(len(X)):
            for j in range(K):
                diff = (np.array(X[i]) - np.array(mu[j]))
                diff = diff.reshape(diff.shape[0],1)
                prob[i][j] = pi[j]*((1/np.sqrt((2*np.pi)**len(X[i])*det(cov[j]))) * np.exp(-0.5*diff.T.dot(np.linalg.inv(np.array(cov[j]).reshape(2,2))).dot(diff)))
            tot = np.sum(prob[i])
            prob[i]/=tot
        ppp = np.sum(prob, axis = 0)

        for j in range(K):
            new_pi[j] = ppp[j]/np.sum(ppp)
            new_mu[j] = [0.0, 0.0]
            new_cov[j] = [0.0,0.0,0.0,0.0]
            for i in range(len(X)):
                new_mu[j] = list(np.array(new_mu[j]) + np.array(prob[i][j]) * np.array(X[i]))
            new_mu[j] = [q/ppp[j] for q in new_mu[j]] 
            for i in range(len(X)):
                diff = (np.array(X[i]) - np.array(new_mu[j]))
                diff = diff.reshape(diff.shape[0],1)
                new_cov[j] = list(np.array(new_cov[j]) + (prob[i][j] * np.outer(diff.T,diff)).reshape(-1))
            new_cov[j] = [q/ppp[j] for q in new_cov[j]] 
        pi = new_pi
        mu = new_mu
        cov = new_cov
    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()