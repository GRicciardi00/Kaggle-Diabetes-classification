# Kaggle-Diabetes-classification

## Dataset
The dataset is made of 70,692 survey responses to the CDC's BRFSS2015. It has an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is balanced. <br />
Soruce: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## Data analysis
The function "data_analysis" allow the user to choose which report choose: 0 is for ''detailed report'' and 1 for "general report".
There *label* is "Diabetes binary", there are 3 numerical features and 18 categorical features (for an accurate description of each features visit the kaggle page). <br />
The **general report** is a matplotlib subplot, a 6x4 matrix: each cell is an histogram. <br />
The **detailed report** make a plot of each feature and  calculates the mean, variance, mode and standard deviation <br />

## Dimensionality reduction
"dimensionality reduction()" function implement two algorithms from *scikitlearn*: **PCA** and **TSN-E** for the dimensionality reduction, then plot a 3D graphic for visualize the results: green points are positive to diabetes while yellow negative.
These two algorithms are *unsupervisioned*, so they are not important for machine learning purposes.
## Preprocessing: correlation matrix & feature selection
The ''dimensionality_reduction()" function generate a **correlation matrix** that exploit correlations between each feature and export it on a excel file. <br />
*Please note* that high correlation value between two feautres doesn't mean there is not necessarily a *causal relationship* between the two.<br />
After plotting the correlation matrix, the function proceeds with the **features selection** of the 7 best features using the SelectKmodel method of the Sklearn library, the unit of measurement used is *mutual information*


## Neural network approach
For neural network approach the program make 2 different neural network, for both models the number of epochs is 10 to avoid *overfitting* and and the metric for evaluating performance is **accuracy**.
 1. **Neural Network without hidden layer**, the *input layer* has seven neurons, one for each feature while the *output layer* has only one neuron. The activation function is a *sigmoid* because the task is a *binary classification*.
 2.  **Neural Network without hidden layer**, the input layer and output layer are the same as the first model, there is also a *hidden layer* consisting of 5 neurons to exploit the *embedding* of the neural network.

## Model comparison
"model_comparison()" execute various classification algorithms with standard values and calculate its accuracy:
 - Logistic regression
 - Decision Tree Classifier
 - Gradient Boosting 
 - RandomForest
 - XBoosting
after several program runs, the best average accuracy value is that of **Gradient Boosting**.
## Model tuning
"gradient_model_tuning()" implement two algorithms: **halving random search** and **Bayes search**. The hyperparameters to be optimized are *learning rate*, *number of estimators* and *max depth*.
## Final test
The final test is made on the test_set, a **confusion matrix** is plotted in the end of the program: it shows the number of true positive, true negative,false positive and false negative predictions. <br />
The final accuracy value is approximately 0.75 (=75%), that's quite good for  our purpose.

