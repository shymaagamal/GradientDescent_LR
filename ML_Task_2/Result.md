# Build Model
* **Used This Cost Function Equation** 

![first equation ](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20J%28w%29%3D%5Cfrac%7B1%7D%7B2m%7D%20%5Csum_%7B%5C%20i%7D%5Em%7B%28h%7B%20%28x%20%5Ei%29%7D-y%5Ei%29%5E2%7D)  

![](Cost.png)

* **Gradient Decent**

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20h%28x%5Ei%29%3D%5Cbeta%5ET%20X)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20Loss%3Dh%28x%5Ei%29-y%5Ei)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20g%3D%28h%28x%5Ei%29-y%5Ei%29x%5Ei)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Ctheta%3D%5Ctheta-%5Calpha%20%5Ctimes%20g)

![](gradientDescent.png)

* **Fit Function**

![](Fit.png)

* **Predict Function**

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20h%7B%20%28x%20%5Ei%29%7D%3D%5Cbeta%5ET%20X)

![](Predict.png)

* **Evaluate Performance**
> **``I Found that these metrics are the best to evaluate my model and tried to use the two metrics just to practice more``**

1. by Mean absolute error (MAE) 

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20MAE%3D%5Cfrac%7B%5Csum_%7Bi%7D%5En%7B%28Y_%7Bactual%7D-Y_%7Bpred%7D%29%5E2%7D%7D%7Bn%7D)

2. by Coefficient of Determination or R^2

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20R_%7B2%7D%3D%5Cfrac%7BSSR%7D%7BSST%7D%3D%5Cfrac%7B%5Csum_%7Bi%7D%5En%7B%28Y_%7Bactual%7D-Y_%7Bmean%7D%29%5E2%7D%7D%7B%5Csum_%7Bi%7D%5En%7B%28Y_%7Bpred%7D-Y_%7Bmean%7D%29%5E2%7D%7D)

![](Evaluate.png)

# Explore Data
* ## UniVariate DataSet

* **Plot Data X and Y**

![](Vis_data.png)

* **Result of My model** 

> * r2  = > Coefficient of Determination 
> * mae = >  Mean absolute error
  
![](R_UniVariate.png)

* **Result When used Sicit_learn Linear Regression Model**

![](SIc_R.png)

* **I Tried to see at any iteration must stop because the cost function doesn't decrease more**
> 2000 iteration sounds good

![](Cost_Iterate_uni.png)

* **I Tried to Visualize Error of Train and Test data to Compare between them and see the Performance of my model**

![](Error_uni.png)

* **Plot best fit line on data**

![](Best_fit.png)


* ## MuliVariate DataSet
>**``at the first, I didn't do scaling but it gives ``nan`` Values in the prediction of output and subsequently in Metrics to evaluate the performance``**
* **Feature Scaling** 
> **``Due to range of each Features are differ from each other we must do Standardization``**

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20X%3D%5Cfrac%7BX-X_%7Bmean%7D%7D%7B%5Csigma%7D)

* **Result of My model** 


> * r2  = > Coefficient of Determination 
> * mae = >  Mean absolute error


![](R_multi.png)

* **Result When used Sicit_learn Linear Regression Model**

![](Sic_R_multi.png)

* **I Tried to see at any iteration must stop because the cost function doesn't decrease more**
> 5000 iteration sounds good

![](cost_iterate_multi.png)

* **I Tried to Visualize Error of Train and Test data to Compare between them and see the Performance of my model**

![](Error_multi.png)