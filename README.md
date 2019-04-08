# Bayesian Minimum Error Rate Classifier


## Running the model ##
Create a new directory `Data` and place the csv files containing the data of the two classes (separately) in it.

Add the relevant column names to the list `features` in the `binClassifier.py`.
Assign the split values to `split1` and `split2` in `binClassifier.py`.

On running `binClassifier.py`, the dataset is shuffled and sampled 100 times. the mean, minimum and maximum accuracies are printed(classwise and overall).

## Mathematical Background ##

### Likelihood ###

The probability density function used is the multivariate normal distribution. 
the likelihood p(x|w<sub>i</sub>) is given by 

<img src = "https://github.com/tejvi-m/minErrorRateClassifier/blob/master/equations/multivariatenorm.png" width="400">

x is the d-dimensional feature vector, μ is the mean vector, &Sigma; is the covariance matrix,|&Sigma;| is the determinant of the covariance matrix, &Sigma;<sup>−1</sup> is the determinant and (x − μ)<sup>t</sup> is the transpose of the (x - μ) vector.
p(x) is calculated using the covariance matrix of the data of a class w<sub>i</sub>.

p(x) for each of the classes is computed given the equation for multivariate normal distribution.
This would be p(x|w<sub>i</sub> ) for i = 1, 2 (being a binary classifier).

### Apriori Probabilities ###
the apriori probabilities P(w<sub>1</sub>) and P(w<sub>2</sub>) are calculated by using

P(w<sub>i</sub>) = (numberof data points in w<sub>i</sub>) / (total number of datapoints)

### Evidence ###
the evidence for each data point(in the test set) is calculated using the equation

p(x) = P (w<sub>1</sub>) ∗ p(x|w<sub>1</sub>) + P (w<sub>2</sub>) ∗ p(x|w<sub>2</sub>) (being a two category case)

### Posterior Probability ###
Using Bayes rule, the posterior probability (Conditional probability) is found for each of the two classes.

<img src = "https://github.com/tejvi-m/minErrorRateClassifier/blob/master/equations/posterior.png" width="300">

Now with the conditional probabilities computed for each of the two classes, we can make a prediction based off of the values of 
P(w<sub>1</sub>|x) and P (w<sub>2</sub>|x).

### Prediction ###
And being a minimum error rate classifier, we define the discriminant function g<sub>i</sub>(x) as P (w<sub>i</sub>|x). if
P(w<sub>1</sub>|x) ≥ P (w<sub>2</sub>|x) (i.e., g<sub>1</sub>(x) ≥ g<sub>2</sub>(x)) then we predict the class to be w<sub>1</sub>
and predict w<sub>2</sub> otherwise.
