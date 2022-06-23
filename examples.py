sys.path.extend(['/Users/Yeongjin/PycharmProjects/Practice/git_project/Single-Perceptron-and-Naive-Bayes-Classifier-with-Visualization-on-A-Plane'])
from Perceptron import twoD_coordinates_Perceptron
from NaiveBayes import twoD_coordinates_GNB

a = twoD_coordinates_Perceptron(n_iters=20)
a.labeled_coordinates()

a = twoD_coordinates_GNB()
