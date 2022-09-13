# Example 2.9 in Woodridge "Introductory Econometrics: A Modern Approach"

# Import the dataset
import wooldridge as woo

# Import Panda. 
# Pandas is an open source Python package that is most widely used for data science/data analysis and machine learning tasks.
# User guide: https://pandas.pydata.org/docs/user_guide/index.html
import pandas as pd

# statsmodels.formula.api: specify models using formula strings and DataFrames.
#  information: https://www.statsmodels.org/dev/api.html
import statsmodels.formula.api as smf

#dataset
vote1 = woo.dataWoo('vote1')


# OLS regression:
reg = smf.ols(formula='voteA ~ shareA', data=vote1)
results = reg.fit()

# print results using summary:
print(f'results.summary(): \n{results.summary()}\n')

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

