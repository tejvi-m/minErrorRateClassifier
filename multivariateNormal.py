import numpy as np

def likelihood(xvec, meanvec, covmat, dims = 7):
    d = xvec - meanvec
    a = 1 / (((2 * np.pi) ** (dims/2)) * (np.linalg.det(covmat)**(1/2)))
    expon = -(1/2) * np.matmul(np.transpose(d), np.matmul(np.linalg.inv(covmat), d))
    expon =  np.exp(expon)

    likelihood = a * expon

    return likelihood

def getPrior(reqClass, class1, class2):
    #for binary classifiers
    return len(reqClass[0])/(len(class1[0]) + len(class2[0]))


def action(p1, p2):
    #for binary classifiers
    if(p1 >= p2):
        return 1
    else:
        return 0

def getmeanVec(train):
    n = len(train)
    meanvec = np.zeros(n)
    for i in range(len(train)):
        meanvec[i] = np.mean(train[i])
    return meanvec

def getProbabilities(row, t1, t2):

    train1 = np.asarray(t1.values.tolist()).transpose()
    train2 = np.asarray(t2.values.tolist()).transpose()

    cov1 = np.cov(train1)
    cov2 = np.cov(train2)

    meanvec1 = getmeanVec(train1)
    meanvec2 = getmeanVec(train2)

    prior1 = getPrior(train1, train1, train2)
    prior2 = getPrior(train2, train1, train2)

    p1 = likelihood(row, meanvec1, cov1)
    p2 = likelihood(row, meanvec2, cov2)

    evidence = p1 * prior1 + p2 * prior2

    prob1 = p1 * prior1 / evidence
    prob2 = p2 * prior2 / evidence

    return prob1, prob2
