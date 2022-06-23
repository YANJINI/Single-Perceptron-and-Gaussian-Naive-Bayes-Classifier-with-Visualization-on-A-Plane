from Perceptron import twoD_coordinates_Perceptron
from NaiveBayes import twoD_coordinates_GNB

a = twoD_coordinates_Perceptron(n_iters=20)
a.labeled_coordinates()

a = twoD_coordinates_GNB()