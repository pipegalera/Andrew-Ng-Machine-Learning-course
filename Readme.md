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

<p align="center"><img src="/tex/6b7a5ddc51751a8f4e79e04d8dc7f549.svg?invert_in_darkmode&sanitize=true" align=middle width=342.9304362pt height=44.89738935pt/></p>

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

where:

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

When specifically applied to the case of linear regression (the "OLS" cost function), a new form of the gradient descent equation can be derived. We substitute the gradient descent algorithm:

<p align="center"><img src="/tex/85f6fd126fe0d1fcb9290362a9b81370.svg?invert_in_darkmode&sanitize=true" align=middle width=174.55293899999998pt height=38.5152603pt/></p>

With our actual hypothesis function and our actual cost function:

<p align="center"><img src="/tex/bdf46c49ed58c9399ae3ade9f03789e6.svg?invert_in_darkmode&sanitize=true" align=middle width=120.67521674999999pt height=16.438356pt/></p>
<p align="center"><img src="/tex/44138a72fe0641526fcfea4accaec310.svg?invert_in_darkmode&sanitize=true" align=middle width=220.70696504999998pt height=44.89738935pt/></p>


For the partial derivate, we derivate with respect of <img src="/tex/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> and <img src="/tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/>:

* For <img src="/tex/7115e3452676ee02b49fd2c47efcbf49.svg?invert_in_darkmode&sanitize=true" align=middle width=252.97245270000002pt height=41.14169729999998pt/>

* For <img src="/tex/8bc270de8a690302a6373df25d85f11c.svg?invert_in_darkmode&sanitize=true" align=middle width=267.0183384pt height=41.14169729999998pt/>

Resulting in:

repeat until convergence {

<p align="center"><img src="/tex/261c0715c01032b853090ebd78e8145f.svg?invert_in_darkmode&sanitize=true" align=middle width=210.3814614pt height=44.89738935pt/></p>

<p align="center"><img src="/tex/1c5df8a0f3bd77299db5d4e7921782b0.svg?invert_in_darkmode&sanitize=true" align=middle width=224.42734874999996pt height=44.89738935pt/></p>

}

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent.

Note that, while gradient descent can be susceptible to local minimum in general, the optimization problem we have posed here for linear regression has only one global, and no other local,thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum.

# Multiple Features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

<p align="center"><img src="/tex/903ad592c758452b120565e499e8a18b.svg?invert_in_darkmode&sanitize=true" align=middle width=320.02055415pt height=16.438356pt/></p>

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

<img src="/tex/ad7aaa0380129cc039b848209d9a1210.svg?invert_in_darkmode&sanitize=true" align=middle width=223.52716484999996pt height=126.57653249999998pt/>

# Gradient Descent for Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

<p align="center"><img src="/tex/a0ed4b7032900d9c4a61b4403b26c7c4.svg?invert_in_darkmode&sanitize=true" align=middle width=1115.7356974499999pt height=44.89738935pt/></p>

In other words:

<p align="center"><img src="/tex/a180dd432697d7c533214cc45e243358.svg?invert_in_darkmode&sanitize=true" align=middle width=619.8554867999999pt height=44.89738935pt/></p>

# Gradient Descent in Practice I - Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because <img src="/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally: <img src="/tex/e00ad96489e7a2a70015b06f0a6ffa24.svg?invert_in_darkmode&sanitize=true" align=middle width=113.4977646pt height=21.18721440000001pt/>

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are feature scaling and mean normalization.
* Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.

* Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:
<p align="center"><img src="/tex/0f68d69725e19cf853b26b7f06c185cf.svg?invert_in_darkmode&sanitize=true" align=middle width=93.6608838pt height=34.45133834999999pt/></p>

Where <img src="/tex/ce9c41bf6906ffd46ac330f09cacc47f.svg?invert_in_darkmode&sanitize=true" align=middle width=14.555823149999991pt height=14.15524440000002pt/> is the average of <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> in the training set, and <img src="/tex/4fa3ac8fe93c68be3fe7ab53bdeb2efa.svg?invert_in_darkmode&sanitize=true" align=middle width=12.35637809999999pt height=14.15524440000002pt/>
is a measure of dispersion, either the range of values (max - min), or the standard deviation.

# Gradient Descent in Practice II - Learning Rate

* Debugging.

How to make sure that the gradient descent is working correctly?

Make a plot with number of iterations on the x-axis. Now plot the cost function, <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> over the number of iterations of gradient descent. If learning rate <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is sufficiently small, then <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> will decrease on every iteration.

<p align="center">
<img src="images/gradient_iteration1.png" width="40%" height="40%">
</p>

If <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> ever increases, then you probably need to decrease <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>.

<p align="center">
<img src="images/gradient_iteration2.png">
</p>

* Automatic convergence test. Declare convergence if <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> decreases by less than <img src="/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/> in one iteration, where <img src="/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/> is some small value such as <img src="/tex/7ecaa0ca1b65792148153cac2f19940d.svg?invert_in_darkmode&sanitize=true" align=middle width=22.990966349999994pt height=26.76175259999998pt/>. However in practice it's difficult to choose this threshold value, it's usually clear when you graph it.

Try with a scale factors of alpha: <img src="/tex/81b1dd92b992840109f36f368864460c.svg?invert_in_darkmode&sanitize=true" align=middle width=301.90086134999996pt height=21.18721440000001pt/>

# Features and Polynomial Regression

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form). For example: <img src="/tex/1c9e8b1095e196ef8e2f5baf80e50513.svg?invert_in_darkmode&sanitize=true" align=middle width=231.13174589999994pt height=26.76175259999998pt/>

<p align="center">
<img src="images/polynomial.png" width="60%" height="60%">
</p>

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

For example: if <img src="/tex/277fbbae7d4bc65b6aa601ea481bebcc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> has range <img src="/tex/3443d8653bc83e0e9c94974dbf7cb485.svg?invert_in_darkmode&sanitize=true" align=middle width=61.18723874999999pt height=21.18721440000001pt/>, then range of <img src="/tex/1b18c546a3e04cc5aa0fbf9774cc8b71.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> becomes <img src="/tex/c85ecc9776f2d6db597d6ccfc9e9144f.svg?invert_in_darkmode&sanitize=true" align=middle width=85.84486679999998pt height=21.18721440000001pt/>, and the range of <img src="/tex/2c801c1cd04b6e9d689d152b32873b42.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> becomes <img src="/tex/94f648cfeaa6d02e00571e786a0ab684.svg?invert_in_darkmode&sanitize=true" align=middle width=110.50249484999998pt height=21.18721440000001pt/>

# Normal Equation

Gradient descent gives one way of minimizing <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm.

In the "Normal Equation" method, we will minimize <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/> by explicitly taking its derivatives, and setting them to zero. This allows us to find the optimum theta without iteration.

The normal equation formula is: <img src="/tex/fbf03f24d68fac3a50be0c675f521fcd.svg?invert_in_darkmode&sanitize=true" align=middle width=134.61145829999998pt height=27.6567522pt/> . It finds the <img src="/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> that minimizes the cost function <img src="/tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.655311049999987pt height=24.65753399999998pt/>.

With the normal equation method, there is no need to choose alpha, since there is no iteration or learning rate.

The main drawback is that it is slow if the sample is very large as it needs to compute the inverse of very large matrix <img src="/tex/3add1221abfa79cb14021bc2dacd5725.svg?invert_in_darkmode&sanitize=true" align=middle width=39.82494449999999pt height=19.1781018pt/> and it doesn't work for more complex models than linear regression.

What if the matrix <img src="/tex/1de8c04d4724ca165e5e49104c88f89a.svg?invert_in_darkmode&sanitize=true" align=middle width=40.17294764999998pt height=27.6567522pt/> is non-invertible (singular matrices)?

If <img src="/tex/1de8c04d4724ca165e5e49104c88f89a.svg?invert_in_darkmode&sanitize=true" align=middle width=40.17294764999998pt height=27.6567522pt/> is noninvertible, the common causes might be having :

* Redundant features, where two features are very closely related (i.e. they are linearly dependent).

* Too many features (e.g. <img src="/tex/3e46e3d0547ca10739be98de8389c8a3.svg?invert_in_darkmode&sanitize=true" align=middle width=59.00304134999999pt height=17.723762100000005pt/>). In this case, delete some features or use regularization.
