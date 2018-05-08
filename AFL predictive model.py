
# coding: utf-8

# # Let's build an AFL sports betting model

# # Contents
# 
# #### 0. Introduction
# - 0.1 Data
# - 0.2 Model inputs
# - 0.3 Model outputs
# 
# #### 1. Setup
# #### 2. Feature engineering
# - 2.1 Further features
# - 2.2 Aggregate team stats from player stats
# - 2.3 Create features
# - 2.4 Choose the lags and stats
# 
# #### 3. Train the model
# - 3.1 Setup the training and test data
# - 3.2 Feature selection
# - 3.3 Model selection
# 
# #### 4. Model evaluation
# - 4.1 Evaluate odds' predictions
# - 4.2 Evaluate model's predictions
# - 4.3 Evaluate model by simulating betting
# 
# #### 5. Next steps

# # 0. Introduction
# We are asked to build a classification model that predicts the winner of an AFL match
# ### 0.1 Data
# We have 2012-2016 data to train the model, which we will evaluate on 2017 data. The data includes player stats for every match.
# ### 0.2 Model inputs
# Our model will be indifferent to which teams are playing, and only consider the home and away teams' recent stats
# ### 0.3 Model outputs
# Our model output will be the predicted probability the home team winning
# - We chose this because we have been given odds for match win/loss. Hence we can compare the model output to the given odds to see if we can do better than the odds.
# - We chose this output over classification (win/lose) because this output would not be informative for betting. If the model predicts a home win, but so do the odds, should the user bet or not?
# - A model that predicts home win margin would also by definition predict home win/loss. If we were interested in betting in margin betting markets and had the odds data, that would have been a useful output.

# # 1. Setup

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Import the results table
results = pd.read_csv("source_data/afl_match_results.csv")

# Add some basic columns
results["Year"] = results["Date"].map(lambda x: int(x[:4]))
results["home_win"] = np.nan
results.loc[results.Margin > 0, "home_win"] = 1
results.loc[results.Margin == 0, "home_win"] = 0.5
results.loc[results.Margin < 0, "home_win"] = 0

results.head()


# # 2. Feature engineering
# The main information we have is past **player stats**. 
# 
# When predicting any particular game, only stats for past games are available. Therefore we should use **lagged variables**.
# 
# This model will primarily use past games' **team stats**, which are created by aggregating the player stats for each match.
# 
# ### 2.1 Further features
# Other potential features absent in the model that could be investigated further include:
# - non-linear variables
# - player-level data (what is the effect of a key player being absent in the next match?)
# - previous *match* stats (e.g. previous match's margin compared to previous match's points)
# - venue (our model treats away games equally regardless of venue and travel distance; our model treats home-ground advantage as equal for all teams, rather than assigning greater advantage to e.g. Perth-based teams) 
# - interaction variables involving specific teams and matchups (i.e. consider play-style; our model account for what teams are playing, only past stats)
# - round type
# - advanced stats (lacking data)
# - change in the metagame of football over time (i.e. changing relationships between features and outcome over time)
# - additional data such as weather conditions

# ### 2.2 Aggregate team stats from player stats

# In[3]:


# Import the player stats table
player_stats = pd.read_csv("source_data/afl_player_stats.csv")
player_stats.head()


# In[4]:


# Drop advanced stats
print(player_stats.shape)
player_stats.dropna(axis=1, how='any', inplace=True)
print(player_stats.shape)
print("no rows dropped")
player_stats.head()


# In[5]:


# Aggregate to team-level stats
team_stats = player_stats.groupby(["Date", "Team"], as_index=False).sum()
print(team_stats.shape)
team_stats.head()


# In[6]:


# Select the most informative stats based on game knowledge
columns = ["Date", "Team", "CP", "D", "CL", "I50", "MI5"]

# Merge home stats by match
flat = pd.merge(results, team_stats[columns], how="left", left_on=["Date", "Home.Team"], right_on=["Date", "Team"])
# Merge away stats by match
flat = pd.merge(flat, team_stats[columns], how="left", left_on=["Date", "Away.Team"], right_on=["Date", "Team"], suffixes=("_Home", "_Away"))
# Rename points to make naming convention consistent with other stats later
flat.rename(columns={"Home.Points":"P_Home", "Away.Points":"P_Away"}, inplace=True)

print(flat.shape)
print(flat.isnull().sum()) # check if merge was successful
flat.head()


# In[7]:


# Investigating the null rows
flat[flat.isnull().any(axis=1)]


# In[8]:


# It appears that Footscray and Brisbane Lions did not merge successfully
# Let's try building the team_stats table again

# Appropriately rename Bulldogs and Lions
team_stats['Team'] = team_stats['Team'].map(lambda x: "Footscray" if x == "Western Bulldogs" else x)
team_stats['Team'] = team_stats['Team'].map(lambda x: "Brisbane Lions" if x == "Brisbane" else x)


# In[9]:


# Try again
# Merge home stats
flat = pd.merge(results, team_stats[columns], how="left", left_on=["Date", "Home.Team"], right_on=["Date", "Team"])
# Merge away stats
flat = pd.merge(flat, team_stats[columns], how="left", left_on=["Date", "Away.Team"], right_on=["Date", "Team"], suffixes=("_Home", "_Away"))
# Rename points to make naming convention consistent with other stats
flat.rename(columns={"Home.Points":"P_Home", "Away.Points":"P_Away"}, inplace=True)

print(flat.shape)
print(flat.isnull().sum()) # check if merge was successful


# In[10]:


# Investigating the null rows
flat[flat.isnull().any(axis=1)]
# appears to be actual missing data


# In[11]:


# drop rows with missing team stats
print(flat.shape)
flat.dropna(inplace=True)
flat = flat.reset_index(drop=True)
print(flat.shape)


# ### 2.3 Create features
# The approach here is to create for each match, features that relate to:
# - The home team's home form
# - The away team's away form
# 
# We do this by:
# 1. Select stats that we believe are predictive of a team's performance in the next match
# 2. Find the average value for each stat in the team's previous x home games for home team, away games for away team
# 3. Compute the difference between the home team and away team's averages for each stat
# 
# These features should capture:
# - Each team's form (by incorporating recent match stats), as well as
# - Each team's relative strength in the league (by incorporating the difference in the two teams' stats)

# In[12]:


# function for creating lagged stats for a team's recent form (home form or away form) inside a loop
def create_stat_byteam(df, i, stat, lags, team):
    colname = stat+"_"+team+"_"+str(lags)
    counter = 1
    sum = 0
    
    # look backwards from current game
    for j in reversed(range(0,i)):
        
        if df.loc[i, team+".Team"] == df.loc[j, team+".Team"]:
            sum += df.loc[j, stat+"_"+team]
            counter += 1
            
            if counter > lags:
                df.loc[i, colname] = sum / lags
                break

# wrapper function to execute for both teams
def create_stat(df, i, stat, lags):
    create_stat_byteam(df, i, stat, lags, "Home")
    create_stat_byteam(df, i, stat, lags, "Away")
    
    home_col = stat+"_Home_"+str(lags)
    away_col = stat+"_Away_"+str(lags)
    df[stat+"_diff_"+str(lags)] = df[home_col] - df[away_col]

# create the new feature columns by team, return the column names that relate to the difference in stats
def create_cols(df, cols, lags):
    new_cols = []
    for col in cols:
        home_col = col+"_Home_"+str(lags)
        away_col = col+"_Away_"+str(lags)
        diff_col = col+"_diff_"+str(lags)
        df[home_col] = np.nan
        df[away_col] = np.nan
#         df[diff_col] = np.nan
        
        new_cols.append(diff_col)
    return new_cols

# drop unneeded columns
def drop_cols(df, cols, lags):
    for col in cols:
        home_col = col+"_Home_"+str(lags)
        away_col = col+"_Away_"+str(lags)
        df.drop([home_col, away_col], axis=1, inplace=True)


# ### 2.4 Choose the lags and stats
# ###### Parameter choice: How many past home matches would be indicative of recent form for the home team?
# - We choose 5, but this could be varied
# 
# ###### Parameter choice: Should they all be weighted equally, or should recent games be weighted more heavily?
# - We choose equal weighting, but this could be varied
# 
# ###### Parameter choice: Should the strength of the teams played against be factored?
# - Our model does not, but this could be predictive
# - Increasing the lags could reduce the noise, but also cause the form features to capture less recent data

# In[13]:


lags = 5


# In[14]:


# Creating the features

feature_stats = ["P", "CP", "D", "CL", "I50", "MI5"]

features = create_cols(flat, feature_stats, lags)

# for i in range(0, flat.shape[0]):
#     for stat in feature_stats:
#         create_stat(flat, i, stat, lags)

# drop_cols(flat, feature_stats, lags)    

# flat.to_csv("working_data/flat.csv")

flat = pd.read_csv("working_data/flat.csv", index_col=0)
flat.tail()


# In[15]:


# Investigating null rows
flat[flat.isnull().any(axis=1)]
# These games do not have enough historical information to create the features


# In[16]:


# drop those rows
flat.dropna(inplace=True)
# flat = flat.reset_index(drop=True)
print(flat.shape)


# In[17]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(flat, x_vars=['P_diff_5', 'CP_diff_5'], y_vars='Margin', size=5, aspect=2.5, kind='reg')
sns.pairplot(flat, x_vars=['D_diff_5', 'CL_diff_5'], y_vars='Margin', size=5, aspect=2.5, kind='reg')
sns.pairplot(flat, x_vars=['I50_diff_5', 'MI5_diff_5'], y_vars='Margin', size=5, aspect=2.5, kind='reg')


# In[18]:


flat.iloc[:,28:].corr(method="pearson")
# Unsurprisingly, the features / regressors are fairly correlated with one another (collinearity)
# This will tend increase the standard errors of regression coefficients and affect their interpretability
# It may be worthwhile to remove less informative features later


# In[19]:


sns.heatmap(flat.iloc[:,28:].corr(method="pearson"))


# In[20]:


flat.iloc[:,28:].describe()


# # 3. Train the model
# The model we will use is the **Logistic Regression** model, also known as the **Logit** model, which is a classification model.
# This model is particularly suited for this application because its output is a probability that an observation will be in one class over another. Therefore we will take this output to compare against the odds for a particular match.
# 
# - If we were only interested in classification, we could have considered models such as kNN, SVM or a Random Forest.
# - If we were interested in predicting a margin between teams, a regression model would have been a better alternative.

# ### 3.1 Setup the training and test data

# In[46]:


print(flat.shape)

# drop tied games
model_data = flat[flat.home_win != 0.5]
print(model_data.shape)

# drop finals games
model_data = model_data[~model_data.Round.isin(["QF", "EF", "SF", "PF", "GF"])]
print(model_data.shape)


# In[22]:


# split data into train and test sets
train = model_data[model_data.Year != 2017]
print(train.shape)
test = model_data[model_data.Year == 2017]
print(test.shape)


# ### 3.2 Feature selection

# In[23]:


# To select the best combination of features, we start by creating a list of all the combinations of features
import itertools
feature_combos = []
for i in range(1, len(features)+1):
    for combo in itertools.combinations(features, i):
        feature_combos.append(list(combo))

# show the first 10 combos
feature_combos[:10]


# In[ ]:


# import scikit-learn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.cross_validation import cross_val_score


# In[24]:


# search for the best performing combo of features using accuracy score from 10-fold cross-validation
# no need for precision, recall, accuracy as wins/losses relatively balanced

y = train["home_win"]
# X = train[features]
combo_accuracies = []

for combo in feature_combos:
    X = train[combo]
    accuracies = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
    combo_accuracies.append([", ".join(combo), len(combo), accuracies.mean()])

combo_accuracies = pd.DataFrame(combo_accuracies, columns=["combo", "feature_count", "accuracy"])

# plot the 10-fold mean accuracy scores for each feature combo
sns.pairplot(combo_accuracies.reset_index(), x_vars="index", y_vars="accuracy", hue="feature_count", size=5, aspect=2.5)


# In[25]:


# From the plot, we observe minimal improvement in accuracy as features are added past 2
# Remembering the high correlation between features from before, we suspect overfitting
# Let's evaluate the best 4, 3 and 2 feature models


# ### 3.3 Model selection
# Evaluate the best 4, 3 and 2 feature models

# In[26]:


combo_accuracies.sort_values("accuracy", ascending=False).head(15)


# In[27]:


# Best 4 features
features = ["P_diff_5", "CP_diff_5", "CL_diff_5", "MI5_diff_5"]
X = train[features]

# use statsmodels to run the regression to show summary
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.Logit(y, X2).fit()
est.summary2()


# In[28]:


# Best 3 features
features = ["CP_diff_5", "I50_diff_5", "MI5_diff_5"]
X = train[features]

X2 = sm.add_constant(X)
est = sm.Logit(y, X2).fit()
est.summary2()


# In[29]:


# Best 2 features
features = ["CP_diff_5", "I50_diff_5"]
X = train[features]

X2 = sm.add_constant(X)
est = sm.Logit(y, X2).fit()
est.summary2()


# In[30]:


# Finally, select 3 feature model to continue (on the basis of pseudo-R^2, AIC and BIC)
best_features = ["CP_diff_5", "I50_diff_5", "MI5_diff_5"]
X = train[best_features]

logreg.fit(X, y)
print(logreg.intercept_)
print(logreg.coef_)


# ## 4. Model evaluation
# To evaluate the model, simulate a betting strategy over the 2017 season

# ### 4.1 Evaluate odds' predictions

# In[31]:


# Import the odds table
odds = pd.read_csv("source_data/afl_odds.csv")
odds["prob"] = 1/odds["odds"]
odds.head()


# In[32]:


# Used Excel to quickly manually match odds and results team names in a lookup table
team_lookup = pd.read_csv("working_data/team_lookup.csv")
team_lookup.head()


# In[33]:


# Merge in the standard team names to the odds table
odds = pd.merge(odds, team_lookup, how="left", left_on="team_name", right_on="team_odds")
odds.head()


# In[34]:


# Merge in the implied probability of home win
test = pd.merge(test, odds[["date", "team_std", "prob"]], how="left", left_on=["Date", "Home.Team"], right_on=["date", "team_std"])
test.drop(["date", "team_std"], axis=1, inplace=True)
test.rename(columns={"prob":"home_prob"}, inplace=True)

# Merge in the implied probability of away win
test = pd.merge(test, odds[["date", "team_std", "prob"]], how="left", left_on=["Date", "Away.Team"], right_on=["date", "team_std"])
test.drop(["date", "team_std"], axis=1, inplace=True)
test.rename(columns={"prob":"away_prob"}, inplace=True)

test.head()


# In[35]:


# Check for missing odds
test[test.isnull().any(axis=1)]
# No matches missing odds


# In[36]:


# create a column for the outcome the odds predict
test.loc[test["home_prob"] >= 0.5, "odds_predict"] = 1
test.loc[test["home_prob"] < 0.5, "odds_predict"] = 0


# In[37]:


from sklearn import metrics
odds_accuracy = metrics.accuracy_score(test["home_win"], test["odds_predict"])
print("Odds predict the correct outcome %.2f%% of the time" % (100*odds_accuracy))


# ### 4.2 Evaluate model's predictions

# In[38]:


# Predictions (win/loss) from model
X_test = test[best_features]
y_test = test["home_win"]
y_pred = logreg.predict(X_test)


# In[39]:


model_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Model predicts the correct outcome %.2f%% of the time" % (100*model_accuracy))


# ### 4.3 Evaluate model by simulating betting

# In[40]:


# Probabilities from model
y_prob = logreg.predict_proba(X_test)
y_prob = pd.DataFrame(y_prob[:,1], columns=["model_prob"])
y_prob.head()


# In[41]:


# Merge in probabilities from model
simulation = pd.concat([test, y_prob], axis=1)

# Compare the predicted probabilities for a home win between our model and the odds
simulation["predict_diff"] = simulation.model_prob - simulation.home_prob
simulation.head()


# In[ ]:


# set up betting strategy

# bet only if model gives a prediction with at least 5% implied odds difference to betfair odds
predict_significance = 0.05


# In[42]:


def bet_strategy(predict_diff):
    if abs(predict_diff) < predict_significance:
        return -1
    elif predict_diff > 0:
        return 1
    else:
        return 0

# place the bets!
simulation["bet"] = simulation["predict_diff"].map(bet_strategy)


# In[43]:


# bet a constant amount per match (if any). (could consider varying the amount based on predict_diff)
bet_amount = 100


# In[44]:


# calculate value if bet is won/lost (or no bet)
simulation["bet_val_w"] = np.where(simulation.bet == 1,
                                   bet_amount*(1/simulation.home_prob - 1), # bet home team to win
                                   bet_amount*(1/simulation.away_prob - 1)) # bet away team to win
simulation["bet_val_w"] = np.where(simulation.bet == -1, 0, simulation.bet_val_w) # if no bet, set value = 0
simulation["bet_val_l"] = np.where(simulation.bet == -1, 0, -bet_amount)        # for losing bet, lose bet_amount

# calculate outcome of the bet after each match
simulation["outcome_val"] = np.where(simulation.bet == simulation.home_win, # if bet successful
                                     simulation.bet_val_w, # win value
                                     simulation.bet_val_l) # lose value

# calculate running profit/(loss) balance over the 2017 season
simulation["balance"] = simulation["outcome_val"].cumsum()

simulation.iloc[:,35:].tail()


# In[45]:


# calculate profit/(loss) over the 2017 season
print("Bet amount per match: $%d" % bet_amount)
print("Total bets placed: %d of %d matches" % (simulation.loc[simulation.bet != -1, "bet"].count(), len(simulation.index)))
print("Total profit over season: $%.2f" % simulation.outcome_val.sum())
print("Return on Investment: %.2f%%" % (100 * simulation.outcome_val.sum() / (100 * simulation.loc[simulation.bet != -1, "bet"].count())))
print("Max positive balance over season: $%.2f" % simulation.balance.max())
print("Max negative balance over season: $%.2f" % -simulation.balance.min())


# ### 5. Next steps
# Now improve the model further!
# - Consider more features
# - Tune hyperparameters in the model
# - Tune betting strategy parameters
# - Using further visualisation to communicate the models to users/customers
# - Check for bias and heteroskedasticity in the error term
