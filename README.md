# NY-Property-Rolling-Sales-Data

Data Preprocessing
Encoding categorical variables to numeric and finding correlation matrix: 
Since most of the columns had categorical values, I encoded them with numeric values to be able to find correlation between variables. Variables with high multi-collinearity are handled by dropping one of the columns.
Also I calculated variables that affect the value of SALE PRICE the most from the correlation matrix. The least significant variables relative to target variable are dropped from the model.

Handling Category variables by creating dummies:
Since the category variables like BOROUGH and TAX CLASS AT TIME OF SALE are significant variables, to use them in Regression models I had to create dummy variables for each level and used them in the model. 

Checking the missing values:
I observed that GROSS SQUARE FEET and LAND SQUARE FEET had some missing values. Plotted a box plot to check the spread of values.
It was observed that 90% of the values lie in a particular range and the data was skewed, therefore the missing values were replaced by median value of that column.
Splitting data into train and test:
The processed and clean data was split into train and test where test contains 20% of the data selected at random each time the algorithm runs. 

Approach
LINEAR REGRESSION
Regression analysis is a basic method to be used in cases where we need to predict a continuous target variable. However I observed that many of the independent variables were of object type with categorical values. Regardless, I wanted to try linear regression method to check its accuracy and see if the algorithm can be improved by handling the categorical variables.
One way can be to create dummy variables for each category column and use them in the linear regression model. This technique has proved to be beneficial in cases where the number of levels were limited to 8-10 but in this particular case for some variables the levels can go above 200. However, this approach gave a very poor accuracy score with Linear regression which might be because of less continuous variables and more categorical variables.
The score of linear regression was 0.24472183797977276

RANDOM FOREST REGRESSION
Random forest regressor is known to work better for discrete as well as continuous variables. Using the random forest regressor on same train and test data gave the accuracy sore of 0.88282475878125111.

FUTURE SCOPE
•	More complex techniques like gradient booster and PCA for feature selection can be implemented for higher accuracy. 
•	Ensemble methods can be used where two different algorithms can be used to handle discrete variables and continuous variables, and both can be stacked to obtain higher accuracy.


CHALLENGES

•	Large number of categorical variables in the dataset
•	Missing values in the Target variable

