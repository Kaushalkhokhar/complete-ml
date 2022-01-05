## Work Flow

- Basic EDA
    - Identifying the types of features
- Categorizing the columns. 
    - categorical, numerical(continuos, discrete, integer, float etc), datetime, labels, ids etc
- Checking the missing values
    - fill missing values with all the features. May be we can fill the missing values here or later on.
    - we can use different imputer techniques available in sklearn for numerical features
- Checking the Target
    - if it is a regression problem then we analyze the distribution of the target and check whether there is necessary preprocessing based on this. If it is a classification problem We need to check target imbalance.
    - for regression we can use the log-transformation of label
        - Logarithm function increases the spacing between small numbers and reduces the spacing between large numbers. When certain features are dense with values in small values, by increasing these intervals, our models increase the intervals for small values, and we can improve the performance of the model when training and testing using these values.
- EDA for numerical features
    - Separate the feature like numerical or categorical or date-time or any other
    - inspect discrete and continuos features separately
    - detecting the outliers
    - adding New Derived Features using Numerical Feature.
        - Good derivative features come from good questions. Good questions come from a lot of domain-knowledge(knowledge about the dataset domain. i.e here we need to have knowledge about factors affecting the house price).
    - scaling
        - There are various scaling methods for numerical features. if we do a log scaling with dependent feature then same should be apply to independent feature as well.
- EDA for categorical features
    - classify the features into categorical and nominal features
        - In the case of an ordinal type, there is a difference in importance for each level. This value plays an important role in the case of regression, so encode it with care.
    - filling missing values
        - A good way to fill in the missing values of categorical features in the absence of domain-knowledge is to take the most-frequent strategy.
    - checking ordinal features
        - In some cases, it is easy to judge that there is an order on a commonsense level. However, there are many cases where it is difficult to judge that there is an order. The method used in this notebook to determine whether the features are ordinal or not was determined to have a certain order through visualization. However, if you have real estate knowledge, you will be able to determine the order of each level by classifying ordinal features smarter than me
    - Making Derived Features for Categorical Data
        - process for making derived features will be same as like numerical features
- Checking dataset before modeling
    - Missing values check
    - selecting features
        - by correlation
    - Encoding nominal data using one-hot encoding
    - Encoding Target using Log Scaling

Ref: https://www.kaggle.com/ohseokkim/house-price-all-about-house-price#Doing-EDA-for-Categorical-Features
    
    