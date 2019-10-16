# What is Machine Learning?
Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications: Supervised learning and Unsupervised learning.

# What is Supervised learning?

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

Example 1:

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.
We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2:

(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

<p align="center">
<img src="images/supervisedlearning.png" width="40%" height="40%">
</p>

# Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

Example:

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

<p align="center">
<img src="images/unsupervisedlearning.png" width="40%" height="40%">
</p>

# Model Representation

To establish notation for future use, we’ll use $x^{(i)}x(i)$ to denote the “input” variables (living area in this example), also called input features, and $y^{(i)}y(i)$ to denote the “output” or target variable that we are trying to predict (price).

A pair $(x^{(i)}, y^{(i)})$ is called a training example, and the dataset that we’ll be using to learn—a list of m training examples $(x^{(i)}, y^{(i)});i=1,...,m$—is called a training set.

Note that the superscript “$(i)$” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, $X = Y = ℝ$.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X → Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function $h$ is called a hypothesis. Seen pictorially, the process is therefore like this:

<p align="center">
<img src="images/hypothesis.png">
</p>

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

# Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

\begin{align*}
    J(\theta_0,\theta_1)=\frac{1}{2m}\sum\limits_{i=1}^{m}(\hat{y}_i−y_i)^2=\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)^2
\end{align*}

To break it apart, it is $\frac{1}{2}\bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta(x_{i}) - y$, or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\frac{1}{2}$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$.

The idea is to choose the $\theta_0,\theta_1$ so that $h_\theta(x)$ is close to $y$ for our training examples $(x,y)$

# Cost Function - Intuition I

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by $h_\theta(x)$ which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ will be 0. The following example shows the ideal situation where we have a cost function of 0.

<p align="center">
<img src="images/costfunction1.png">
</p>

When $\theta_1 = 1$ , we get a slope of 1 which goes through every single data point in our model. Conversely, when $\theta_1 = 0.5$, we see the vertical distance from our fit to the data points increase.

<p align="center">
<img src="images/costfunction2.png">
</p>

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

<p align="center">
<img src="images/costfunction3.png">
</p>

Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1 = 1$ is our global minimum.

# Cost Function - Intuition II

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

<p align="center">
<img src="images/costfunction4.png">
</p>

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(\theta_0,\theta_1)$ and as a result, they are found along the same line. The circled $x$ displays the value of the cost function for the graph on the left when $\theta_0= 800_$ and $\theta_1 = -0.15$. Taking another $h(x)$ and plotting its contour plot, one gets the following graphs:

<p align="center">
<img src="images/costfunction5.png">
</p>

When $\theta_0 = 360$ and $\theta_1 = 0$, the value of $J(\theta_0,\theta_1)$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

<p align="center">
<img src="images/costfunction6.png">
</p>

The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put $\theta_0$ on the x axis and $\theta_1$ on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

<p align="center">
<img src="images/gradient.png">
</p>

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $\alpha$, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter $\alpha$,. A smaller $\alpha$ would result in a smaller step and a larger $\alpha$, results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

\begin{align*}
  \theta_j := \theta_j - \alpha\frac{d}{d\theta_j}J(\theta_0,\theta_1)
\end{align*}

where:

$j=0,1$ represents the feature index number, $:=$ is the assigment ("update") math symbol and $\alpha$ is the learning rate.

At each iteration j, one should simultaneously update the parameters $\theta_1, \theta_2,...,\theta_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation:

<p align="center">
<img src="images/correct_gradient.png">
</p>

# Gradient Descent Intuition

We are going to explore the scenario where we used one parameter $\theta_1$ and plotted its cost function to implement a gradient descent.

<p align="center">
<img src="images/gradient_intuition_1.png" width="50%" height="50%">
</p>

- The derivate term ($\frac{d}{d\theta_1}$)

We start at a random point on the function $J(\tetha_1)$, e.g $\theta_1$ in the x axis. We compute the derivative \frac{d}{d\theta_1}, that is the tangent line to the point $\theta_1$. We discover that it is positive, now the function know that the point is a in positive slope (given that the slope is the derivative of $\theta_1$). So, the update is going to be $\theta_1$ minus $\alpha$ times some positive number:

<p align="center">
<img src="images/gradient_intuition_2.png" width="60%" height="60%">
</p>

With the update, the gradient descent drives $\theta_1$ to the left, closer to the minimum.

It can happen the opposite, the slope of the tangent line is negative given that the derivative of $\theta_1$ is negative. With the formula we see that negative $\alpha$ times the negative derivative makes the $\theta_1$ bigger, driving the updated $\theta_1$ to the right.

<p align="center">
<img src="images/gradient_intuition_3.png" width="60%" height="60%">
</p>

In any case, regardless of the slope's sign for $\frac{d}{d\theta_1}J(\theta_1), \theta_1$ eventually converges to its minimum value.

- Learning rate ($\alpha$)

We should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

<p align="center">
<img src="images/gradient_intuition_4.png" width="60%" height="60%">
</p>

Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed. As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease $\alpha$ over time.

Note that if you are already at the local optimum it leaves $\theta_1$ unchanged cause its updates as $\theta_j := \theta_j - \alpha\times0$.

<p align="center">
<img src="images/gradient_intuition_5.png" width="60%" height="60%">
</p>

# Gradient Descent For Linear Regression

When specifically applied to the case of linear regression (the "OLS" cost function), a new form of the gradient descent equation can be derived. We substitute the gradient descent algorithm:

\begin{align*}
  \theta_j := \theta_j - \alpha\frac{d}{d\theta_j}J(\theta_0,\theta_1)
\end{align*}

With our actual hypothesis function and our actual cost function:

\begin{align*}
  h_{\theta}(x)= \theta_0 + \theta_1x
\end{align*}
\begin{align*}
  J(\theta_0,\theta_1)= \frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)^2
\end{align*}


For the partial derivate, we derivate with respect of $\theta_0$ and $\theta_1$:

* For $\theta_0: \frac{d}{d\theta_0}J(\theta_0,\theta_1)=\frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)$

* For $\theta_1: \frac{d}{d\theta_1}J(\theta_0,\theta_1)=\frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)x_i$

Resulting in:

repeat until convergence {

\begin{align*}
\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)
\end{align*}

\begin{align*}
\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}x_i - y_i)x_i
\end{align*}

}

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent.

Note that, while gradient descent can be susceptible to local minimum in general, the optimization problem we have posed here for linear regression has only one global, and no other local,thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum.

# Multiple Features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

\begin{align*}
h_{\theta}(x)= \theta_0 + \theta_{1}x_1 + \theta_{2}x_2 + \theta_{3}x_3 + ... + \theta_{n}x_n
\end{align*}

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$\[
h_{\theta}(x)= [\theta_0 \theta_1 ... \theta_n]
\begin{bmatrix}
x_0 \\
x_1 \\
. \\
. \\
. \\
x_n \\
\end{bmatrix} = \theta^{T}x
\]$

# Gradient Descent for Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

\begin{align*}
& \text{repeat until convergence:} \; \\
\; & \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)} \\

\; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\

\; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \\

\end{align*}

In other words:

\begin{align*}
& \text{repeat until convergence:} \;
\newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}
\newline
\end{align*}

# Gradient Descent in Practice I - Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally: $-1 <= x_i <= 1$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are feature scaling and mean normalization.
* Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.

* Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:
\begin{align*}
x_i := \frac{x_i - \mu_i}{s_i}
\end{align*}

Where $\mu_i$ is the average of $x_i$ in the training set, and $s_i$
is a measure of dispersion, either the range of values (max - min), or the standard deviation.

# Gradient Descent in Practice II - Learning Rate

* Debugging.

How to make sure that the gradient descent is working correctly?

Make a plot with number of iterations on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If learning rate $\alpha$ is sufficiently small, then $J(\theta)$ will decrease on every iteration.

<p align="center">
<img src="images/gradient_iteration1.png" width="40%" height="40%">
</p>

If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

<p align="center">
<img src="images/gradient_iteration2.png">
</p>

* Automatic convergence test. Declare convergence if $J(\theta)$ decreases by less than $E$ in one iteration, where $E$ is some small value such as $10^−3$. However in practice it's difficult to choose this threshold value, it's usually clear when you graph it.

Try with a scale factors of alpha: $\alpha = ...,0.001,0.003,0.01,0.03,0.1,0.3,1,...$

# Features and Polynomial Regression

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form). For example: $h_{\theta}(x)=\theta_0 + \theta_1x_1 + \theta_2x_2^2 + \theta_3x_3^3$

<p align="center">
<img src="images/polynomial.png" width="60%" height="60%">
</p>

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

For example: if $x_1$ has range $1 - 1000$, then range of $x_1^2$ becomes $1 - 1000000$, and the range of $x_1^3$ becomes $1 - 1000000000$

# Normal Equation

Gradient descent gives one way of minimizing $J(\theta)$. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm.

In the "Normal Equation" method, we will minimize $J(\theta)$ by explicitly taking its derivatives, and setting them to zero. This allows us to find the optimum theta without iteration.

The normal equation formula is: $\theta=(X^{T}X)^{-1}X^{T}y$ . It finds the $\theta$ that minimizes the cost function $J(\theta)$.

With the normal equation method, there is no need to choose alpha, since there is no iteration or learning rate.

The main drawback is that it is slow if the sample is very large as it needs to compute the inverse of very large matrix $n \times n$ and it doesn't work for more complex models than linear regression.

What if the matrix $X^{T}X$ is non-invertible (singular matrices)?

If $X^{T}X$ is noninvertible, the common causes might be having :

* Redundant features, where two features are very closely related (i.e. they are linearly dependent).

* Too many features (e.g. $m <= n$). In this case, delete some features or use regularization.
