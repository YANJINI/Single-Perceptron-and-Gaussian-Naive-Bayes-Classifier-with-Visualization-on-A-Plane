# Single-Perceptron-and-Gaussian-Naive-Bayes-Classifier-with-Visualization-on-A-Plane
Comparison between single Perceptron classifier and, linear and non-linear Gaussian navie Bayes classifier over labeled scatterplots on a plane and its visualization with matplotlib <br />


## Bacic Idea
From an inspiration by [Perceptron demo](https://youtu.be/wl7gVvI-HuY?t=1331) on lecture 4 and [Naive Bayes demo](https://youtu.be/rqB0XWoMreU?t=2498) on lecture 10 of Kilian Weinberger's [Machine Learning for Intelligent Systems course](https://www.cs.cornell.edu/courses/cs4780/2018fa/) at Cornell University, I have built a conceptually same interactive single Perceptron and, linear and non-linear Gaussian naive Bayes demo in Python with matplotlib event handling. <br />


## How it works
### Single Perceptron Classifier

on the figure that pops up, you click to plot from class 1 and 2 as follows. <br />

![click to plot from class 1](/images/click%20to%20plot%20from%20class%201.gif)

Press enter to switch to plotting from class 2. <br />
![click to plot from class 2](/images/click%20to%20plot%20from%20class%202.gif)

Pressing enters makes the program iterate to find a right line to seperate two classes based on Perceptron algorithm. <br />
![Perceptron_lseperable](/images/Perceptron_lseperable.gif)

In a typical non-linearly seperable case (XOR), it goes as below when the number of iterations hits the number you set. <br />
![Perceptron_nlseperable](/images/Perceptron_nlseperable.gif) <br />
<br />
<br />

### Linear and Gaussian Naive Bayes Classifier
When each of those decomposed one-dimension feature distributions, P(x_i | y) follows Gaussian distribution and their variances are assumed to be the same across all labels (in this case when y=1 and y=-1), we can derive the closed form of a line that represents 50:50 case of falling into the two labels. In other words, Gaussian Naive Bayes classifier's decision boundary is linear in this case. <br />

![linear_GNB](/images/linear_GNB.gif) <br />
<br />
<br />

### Non-linear Gaussian Naive Bayes Classifier
When the assumption about variances given label is violated, the decision boundary is not linear and there is no closed form of this.

![nonlinear_GNB](/images/nonlinear_GNB.gif) <br />


## Setup

### git clone
git clone to have this repository on your local machine as follows.
```ruby
git clone git@github.com:YANJINI/Single-Perceptron-and-Naive-Bayes-Classifier-with-Visualization-on-A-Plane.git
```

### path control
To import modules written in this repository on your local macine, you need control path to this clone, which could be done as below. (Mac OS)
```ruby
import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/Practice/git_project/Single-Perceptron-and-Gaussian-Naive-Bayes-Classifier-with-Visualization-on-A-Plane'])
```

### import 
Import these two classifiers in another py project as below.
```ruby
from Perceptron import twoD_coordinates_Perceptron
from NaiveBayes import twoD_coordinates_GNB
```

### Others
Check examples.py to see how to use the programes
