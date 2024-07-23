# %% [markdown]
# ### Import Libraries

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import seaborn as sns
sns.set()

# %% [markdown]
# ### Load the data

# %%
raw_data = pd.read_csv("1.04. Real-life example.csv")

# %% [markdown]
# ### Explore the data

# %%
raw_data.head()

# %%
raw_data.describe(include='all')

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Deal with missing values

# %%
raw_data.isnull().sum()

# %%
#Deleting entries is not always recommended, its ok though if the deleted entries are less than 5% of the overall data
#Delete Rows with Null Values 
data_no_null = raw_data.dropna(axis= 0) 

# %% [markdown]
# ### Deal with outliers

# %% [markdown]
# #### Check the outliers in the numerical data

# %%
# Reshape data for FacetGrid
melted_data = data_no_null.melt(var_name='variable', value_name='value', value_vars=['Price', 'Mileage', 'EngineV' , 'Year'])

# Create FacetGrid
g = sns.FacetGrid(melted_data, col="variable", sharex=False, sharey=False)
g.map(sns.histplot, "value")

# Show plot
plt.show()


# %%
# Outlier detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(data_no_null[['Price' , 'Mileage' , 'EngineV' , 'Year']])
data_no_outliers = data_no_null[outliers != -1]  # Remove outliers

# %%
#Plot the data after removing the outliers

melted_data = data_no_outliers.melt(var_name='variable', value_name='value', value_vars=['Price', 'Mileage', 'EngineV' , 'Year'])

g = sns.FacetGrid(melted_data, col="variable", sharex=False, sharey=False)
g.map(sns.histplot, "value")

plt.show()


# %%
data_no_outliers.describe(include="all")

# %%
data_cleaned = data_no_outliers.reset_index(drop=True)

# %% [markdown]
# ## Check the OLS assumptions

# %% [markdown]
# ### Linearity

# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()

# %% [markdown]
# As shown in the plots, the relationship between the "Price" and the other variables is not linear
# 
# to achieve the linearity, we can transform the data using "log transformation"

# %%
# Let's transform 'Price' with a log transformation
log_price = np.log(data_cleaned['Price'])

# Then we add it to our data frame
data_cleaned['log_price'] = log_price
data_cleaned.head()

# %%
# Let's check the three scatters once again
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()

# %%
# Since we will be using the log price variable, we can drop the old 'Price' one
data_cleaned = data_cleaned.drop(['Price'],axis=1)

# %% [markdown]
# ### Multicollinearity

# %%
data_cleaned.columns.values

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

# %%
vif

# %% [markdown]
#  if VIF is greater than 5, then the explanatory variable is highly collinear with the other explanatory variables, and the parameter estimates will have large standard errors because of this.
# 
#  Dropping the "Year" variable will drive the VIF of other variables down.

# %%
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_no_multicollinearity[['Mileage','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

# %%
vif

# %% [markdown]
# ## Working with categorical data

# %% [markdown]
# ### Create dummy variables

# %%
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

# %% [markdown]
# It is extremely important that we drop one of the dummies, alternatively we will introduce multicollinearity

# %%
data_with_dummies.head()

# %% [markdown]
# ## Regression Model

# %% [markdown]
# ### Target and Input

# %%
target = data_with_dummies['log_price']

input = data_with_dummies.drop(['log_price'] , axis= 1) #axis = 1 -> drop the column

# %% [markdown]
# ### Scale the data

# %%
from sklearn.preprocessing import StandardScaler

scale  = StandardScaler()

scale.fit(input)

# %%
scaled_input = scale.transform(input)

# %% [markdown]
# ### Split the data

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_input , target , test_size= 0.2 , random_state=47)

# %% [markdown]
# ### Create the regression

# %%
reg = LinearRegression()

reg.fit(x_train,y_train)

# %%
y_hat = reg.predict(x_train)

# %%
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.ylim(6,13)
plt.show()

# %%
sns.displot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)

# %%
reg.score(x_train,y_train)

# %% [markdown]
# ### Wights and Bias

# %%
reg.intercept_

# %%
reg.coef_

# %%
reg_summary = pd.DataFrame(input.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

# %%
data_cleaned['Brand'].unique()

# %% [markdown]
# ### Testing

# %%
y_hat_test = reg.predict(x_test)

# %%
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

# %%
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()

# %%
y_test = y_test.reset_index(drop=True)

# Check the result
y_test.head()

# %%
df_pf['Target'] = np.exp(y_test)
df_pf

# %%
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# %%
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf

# %%
df_pf.describe()

# %%
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])

# %%



