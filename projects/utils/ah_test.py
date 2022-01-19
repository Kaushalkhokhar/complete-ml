import pandas as pd
from scipy.stats import chi2_contingency

def chi2_test(df, attrs_list, with_):
    """
    The Chi-Square test of independence is used to determine if there 
    is a significant relationship between two categorical (nominal) variables.
    
    Null Hypothesis (H0): There is no relationship between the variables
    Alternative Hypothesis (H1): There is a relationship between variables
    
    If we choose our p-value level to 0.05, as the p-value test result is more than 
    0.05 we fail to reject the Null Hypothesis. This means, there is no relationship 
    between based on the Chi-Square test of independence.
    
    ref: https://towardsdatascience.com/categorical-feature-selection-via-chi-square-fc558b09de43
    
    Note: chisquare test can only be apply to find relationship between categorical variables
    """
    ral_attrs = [] # relative attributes
    for attr in attrs_list: 
        chi_res = chi2_contingency(pd.crosstab(df[with_], df[attr]))
#         print(f"Chi2 Statistic: {chi_res[0]}, p-value: {chi_res[1]}")
        if chi_res[1] < 0.05: ral_attrs.append(attr)
    return ral_attrs


def post_hoc_test(df, attrs_list, with_):
    """
    If we have multiple classes within a category, we would not be able to 
    easily tell which class of the features are responsible for the 
    relationship if the Chi-square table is larger than 2×2. To pinpoint which 
    class is responsible, we need a post hoc test. o do this, we could apply 
    OneHotEncoding to each class and create a new cross-tab table against the other feature
    
    However, there is something to remember. Comparing multiple classes against 
    each other would means that the error rate of a false positive compound 
    with each test. For example, if we choose our first test at p-value level 
    0.05 means there is a 5% chance of a false positive; if we have multiple classes, 
    the test after that would compounding the error with the chance become 
    10% of a false positive, and so forth. With each subsequent test, 
    the error rate would increase by 5%. Let's consider we had 3 pairwise 
    comparisons. This means that our Chi-square test would have an error rate of 
    15%. Meaning our p-value being tested at would equal 0.15, which is quite high.
    
    In this case, we could use the Bonferroni-adjusted method for correcting the p-value 
    we use. We adjust our P-value by the number of pairwise comparisons we want to do. 
    The formula is p/N, where p= the p-value of the original test and N= the number of 
    planned pairwise comparisons. For example, in our case, above we have 3 class 
    within the categorical feature; which means we would have 3 pairwise comparisons 
    if we test all the class against the labels(or categorical) feature. Our P-value 
    would be 0.05/3 = 0.0167
    """
    attrs =  [] # relative attributes
    for attr in attrs_list:
        dummies = pd.get_dummies(data=df[attr], columns=[attr]) # creates onehotencoding
        for i in range(len(dummies.columns)):
            p_value = 0.05 / len(dummies.columns)
            chi_res = chi2_contingency(pd.crosstab(df[with_], dummies[dummies.columns[i]]))
#             print(dummies.columns[i])
#             print(f"Chi2 Statistic: {chi_res[0]}, p-value: {chi_res[1]}")
            if chi_res[1] < p_value: attrs.append((attr, dummies.columns[i]))
    return attrs