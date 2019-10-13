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

To establish notation for future use, we’ll use <img src="/tex/839bd530124f302b99b3bf7c247a4bb6.svg?invert_in_darkmode&sanitize=true" align=middle width=52.985455049999985pt height=29.190975000000005pt/> to denote the “input” variables (living area in this example), also called input features, and <img src="/tex/f1ff827dd1819f91fd40229eac1ce7b7.svg?invert_in_darkmode&sanitize=true" align=middle width=51.493891349999984pt height=29.190975000000005pt/> to denote the “output” or target variable that we are trying to predict (price).

A pair <img src="/tex/9a9ed1968ddefaa8d6f1635e03f6c72b.svg?invert_in_darkmode&sanitize=true" align=middle width=69.62915025pt height=29.190975000000005pt/> is called a training example, and the dataset that we’ll be using to learn—a list of m training examples <img src="/tex/7725b3ed04e52ec88cb35b792c1cb11c.svg?invert_in_darkmode&sanitize=true" align=middle width=155.4786387pt height=29.190975000000005pt/>—is called a training set.

Note that the superscript “<img src="/tex/945cfdab316c27e0a9475969788be662.svg?invert_in_darkmode&sanitize=true" align=middle width=18.44865989999999pt height=24.65753399999998pt/>” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, <img src="/tex/1b20a5d9f7a35a8a4fd6a5063afe967f.svg?invert_in_darkmode&sanitize=true" align=middle width=67.37420909999999pt height=22.465723500000017pt/>.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function <img src="/tex/bba86056bdbd914e17b3a73cfac216cd.svg?invert_in_darkmode&sanitize=true" align=middle width=51.27458819999998pt height=22.831056599999986pt/> so that <img src="/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> is a “good” predictor for the corresponding value of <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>. For historical reasons, this function <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> is called a hypothesis. Seen pictorially, the process is therefore like this:

<p align="center">
<img src="images/hypothesis.png">
</p>

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

# Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

<img src="/tex/5124250c58027b14c7d2f110b5db9b9d.svg?invert_in_darkmode&sanitize=true" align=middle width=315.4630330499999pt height=41.14169729999998pt/>

To break it apart, it is <img src="/tex/d6b19f68aacb4461f4427e1489098827.svg?invert_in_darkmode&sanitize=true" align=middle width=17.920126949999997pt height=27.77565449999998pt/> where <img src="/tex/33717a96ef162d4ca3780ca7d161f7ad.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=18.666631500000015pt/> is the mean of the squares of <img src="/tex/f0e008f4b5c3add6125c62fca6b420a1.svg?invert_in_darkmode&sanitize=true" align=middle width=73.30191659999998pt height=24.65753399999998pt/>, or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved <img src="/tex/47d54de4e337a06266c0e1d22c9b417b.svg?invert_in_darkmode&sanitize=true" align=middle width=6.552545999999997pt height=27.77565449999998pt/> as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the <img src="/tex/47d54de4e337a06266c0e1d22c9b417b.svg?invert_in_darkmode&sanitize=true" align=middle width=6.552545999999997pt height=27.77565449999998pt/>.

The idea is to choose the <img src="/tex/17fde55b60bb334afb4a2b303a1ea335.svg?invert_in_darkmode&sanitize=true" align=middle width=36.66667454999999pt height=22.831056599999986pt/> so that <img src="/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> is close to <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> for our training examples <img src="/tex/7392a8cd69b275fa1798ef94c839d2e0.svg?invert_in_darkmode&sanitize=true" align=middle width=38.135511149999985pt height=24.65753399999998pt/>

# Cost Function - Intuition I

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by <img src="/tex/b687e9cb7f5356da0e24f1b1cac73585.svg?invert_in_darkmode&sanitize=true" align=middle width=39.088702949999984pt height=24.65753399999998pt/> which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of <img src="/tex/dde9bb45048690d27e2b6c199faf1f5d.svg?invert_in_darkmode&sanitize=true" align=middle width=60.970370999999986pt height=24.65753399999998pt/> will be 0. The following example shows the ideal situation where we have a cost function of 0.

<p align="center">
<img src="images/costfunction1.png">
</p>

When <img src="/tex/a9c6a444f1f65977995aeafe59acd891.svg?invert_in_darkmode&sanitize=true" align=middle width=45.22819289999998pt height=22.831056599999986pt/> , we get a slope of 1 which goes through every single data point in our model. Conversely, when <img src="/tex/2bcbe5c0630d86ce638154ddcb6dc655.svg?invert_in_darkmode&sanitize=true" align=middle width=58.013625449999985pt height=22.831056599999986pt/>, we see the vertical distance from our fit to the data points increase.

<p align="center">
<img src="images/costfunction2.png">
</p>

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

<p align="center">
<img src="images/costfunction3.png">
</p>

Thus as a goal, we should try to minimize the cost function. In this case, <img src="/tex/a9c6a444f1f65977995aeafe59acd891.svg?invert_in_darkmode&sanitize=true" align=middle width=45.22819289999998pt height=22.831056599999986pt/> is our global minimum.

# Cost Function - Intuition II

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

<p align="center">
<img src="images/costfunction4.png">
</p>

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for <img src="/tex/8fe4007120c3df67440d00ff258ffcda.svg?invert_in_darkmode&sanitize=true" align=middle width=60.970370999999986pt height=24.65753399999998pt/> and as a result, they are found along the same line. The circled <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> displays the value of the cost function for the graph on the left when <img src="/tex/5cd8e0abe46d1d38b0144343ff33af27.svg?invert_in_darkmode&sanitize=true" align=middle width=61.66661159999999pt height=22.831056599999986pt/> and <img src="/tex/e0cc6478e5b19281c2d782203319dd99.svg?invert_in_darkmode&sanitize=true" align=middle width=79.01826899999998pt height=22.831056599999986pt/>. Taking another <img src="/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> and plotting its contour plot, one gets the following graphs:

<p align="center">
<img src="images/costfunction5.png">
</p>

When <img src="/tex/112e0823e05422da9985658c829beed2.svg?invert_in_darkmode&sanitize=true" align=middle width=61.66661159999999pt height=22.831056599999986pt/> and <img src="/tex/a85bf85e8a0643297472863cdc9c7506.svg?invert_in_darkmode&sanitize=true" align=middle width=45.22819289999998pt height=22.831056599999986pt/>, the value of <img src="/tex/8fe4007120c3df67440d00ff258ffcda.svg?invert_in_darkmode&sanitize=true" align=middle width=60.970370999999986pt height=24.65753399999998pt/> in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

<p align="center">
<img src="images/costfunction6.png">
</p>

The graph above minimizes the cost function as much as possible and consequently, the result of <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> and <img src="/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields <img src="/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> and <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put <img src="/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> on the x axis and <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

<p align="center">
<img src="images/gradient.png">
</p>

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>,. A smaller <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> would result in a smaller step and a larger <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>, results in a larger step. The direction in which the step is taken is determined by the partial derivative of <img src="/tex/8fe4007120c3df67440d00ff258ffcda.svg?invert_in_darkmode&sanitize=true" align=middle width=60.970370999999986pt height=24.65753399999998pt/>. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

<p align="center"><img src="/tex/85f6fd126fe0d1fcb9290362a9b81370.svg?invert_in_darkmode&sanitize=true" align=middle width=174.55293899999998pt height=38.5152603pt/></p>

where

<img src="/tex/a8f306a6f9035370006c5350c9aa4aa8.svg?invert_in_darkmode&sanitize=true" align=middle width=53.37235034999999pt height=21.68300969999999pt/> represents the feature index number, <img src="/tex/5fc6094a9c29537af5f99e0fceb76364.svg?invert_in_darkmode&sanitize=true" align=middle width=17.35165739999999pt height=14.15524440000002pt/> is the assigment ("update") math symbol and <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the learning rate.

At each iteration j, one should simultaneously update the parameters <img src="/tex/1edafae90942e6f441406ef59f6d1774.svg?invert_in_darkmode&sanitize=true" align=middle width=81.64194059999998pt height=22.831056599999986pt/>. Updating a specific parameter prior to calculating another one on the <img src="/tex/f9027510b2e4d1837e4ea831d8c89be6.svg?invert_in_darkmode&sanitize=true" align=middle width=30.64626839999999pt height=29.190975000000005pt/> iteration would yield to a wrong implementation:

<p align="center">
<img src="images/correct_gradient.png">
</p>

# Gradient Descent Intuition

We are going to explore the scenario where we used one parameter <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> and plotted its cost function to implement a gradient descent.

<p align="center">
<img src="images/gradient_intuition_1.png" width="50%" height="50%">
</p>

- The derivate term (<img src="/tex/5cd4bd489ca10340bf4bc49aeb256ed9.svg?invert_in_darkmode&sanitize=true" align=middle width=19.520070899999997pt height=28.92634470000001pt/>)

We start at a random point on the function <img src="/tex/8386f4ee1bd33b1859e566a2277e0342.svg?invert_in_darkmode&sanitize=true" align=middle width=30.856244099999987pt height=24.65753399999998pt/>, e.g <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> in the x axis. We compute the derivative \frac{d}{d\theta_1}, that is the tangent line to the point <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>. We discover that it is positive, now the function know that the point is a in positive slope (given that the slope is the derivative of <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>). So, the update is going to be <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> minus <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> times some positive number:

<p align="center">
<img src="images/gradient_intuition_2.png" width="60%" height="60%">
</p>

With the update, the gradient descent drives <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> to the left, closer to the minimum.

It can happen the opposite, the slope of the tangent line is negative given that the derivative of <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> is negative. With the formula we see that negative <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> times the negative derivative makes the <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> bigger, driving the updated <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> to the right.

<p align="center">
<img src="images/gradient_intuition_3.png" width="60%" height="60%">
</p>

In any case, regardless of the slope's sign for <img src="/tex/9f5988a44f8908cc670ea5db12559946.svg?invert_in_darkmode&sanitize=true" align=middle width=81.64112714999999pt height=28.92634470000001pt/> eventually converges to its minimum value.

- Learning rate (<img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>)

We should adjust our parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

<p align="center">
<img src="images/gradient_intuition_4.png" width="60%" height="60%">
</p>

Gradient descent can converge to a local minimum, even with the learning rate <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> fixed. As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> over time.

Note that if you are already at the local optimum it leaves <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> unchanged cause its updates as <img src="/tex/7341b0dc745871a6165cd40beb5d7e8f.svg?invert_in_darkmode&sanitize=true" align=middle width=114.74853555pt height=22.831056599999986pt/>.

<p align="center">
<img src="images/gradient_intuition_5.png" width="60%" height="60%">
</p>

# Gradient Descent For Linear Regression
