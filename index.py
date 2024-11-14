# Step01: Import of packages


# Import packages for data manipulation
import pandas as pd
import numpy as np
# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer
# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance



# Step 02 : Examination of data --> summary info, and descriptive stats

# Load dataset into dataframe
data = pd.read_csv("tiktokdataset.csv")
# Display first few rows
print(data.head())
# Get number of rows and columns
print(data.shape)
# Get basic information
print(data.info())
# Generate basic descriptive stats
print(data.describe())
# Check for missing values
print(data.isna().sum())
# Drop rows with missing values
data = data.dropna(axis=0)
# Check for duplicates
data.duplicated().sum()

# Check class balance (claims vs Opinions)
print(data["claim_status"].value_counts(normalize=True))
# Approximately 50.3% of the dataset represents claims and 49.7% represents opinions, so the outcome variable is balanced.

#Extract the length (character count) of each video_transcription_text and add this to the dataframe as a new column called text_length so that it can be used as a feature in the model.
# Create `text_length` column
data['text_length'] = data['video_transcription_text'].str.len()
print("After adding Text Length column")
print(data.head())

#Calculate the average text_length for claims and opinions
print("Mean of Text length of Claim vs Opinion")
print(data[['claim_status', 'text_length']].groupby('claim_status').mean())

# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot
sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()

#Step 03: Feature selection and transformation (Encode target and catgorical variables)
X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis=1)


# Encode target variable (0/1)
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
# "dummy encoding" or "one-hot encoding" on two categorical columns: 'verified_status' and 'author_ban_status'
X = pd.get_dummies(X,columns=['verified_status', 'author_ban_status'], drop_first=True)  #The drop_first=True means it will drop one of the categories to avoid what's called the "dummy variable trap"
#print(X.head())

#Step 05: seperation of variables
# Isolate target variable
y = X['claim_status']
# Isolate features (axis=1 means you're dropping a column (if it was axis=0, it would drop rows))
X = X.drop(['claim_status'], axis=1)

#Step 06: Data Splitting
# Split the data into training and testing sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
# Get shape of each training, validation, and testing set
print("\n The Shape of Training, testing and validation data")
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)




#Tokenization (of video transcript)
#each video's transcription text are broken into both 2-grams and 3-grams, then takes the 15 most frequently occurring tokens from the entire dataset to use as features.
# Set up a `CountVectorizer` object, which converts a collection of text to a matrix of token counts
count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
#print(count_vec)

#Fit the vectorizer to the training data (generate the n-grams) and transform it
# Extract numerical features from `video_transcription_text` in the training set
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
#print(count_data)



# Place the numerical representation of `video_transcription_text` from training set into a dataframe
count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())
# Display first few rows
#print(count_df.head())



# Concatenate `X_train` and `count_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `video_transcription_text`,
# so that the indices align with those in `X_train` and `count_df`
X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)
# Display first few rows
#print("\n\n Final Training Data Set: ")
#print(X_train_final.head()) 


# Extract numerical features from `video_transcription_text` in the testing set
validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()


# Place the numerical representation of `video_transcription_text` from validation set into a dataframe
validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
#print(validation_count_df.head())

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_val after dropping `video_transcription_text`,
# so that the indices align with those in `validation_count_df`
X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)



#Repeat the process to get n-gram counts for the test data. Again, don't refit the vectorizer to the test data. Just transform it
# Extract numerical features from `video_transcription_text` in the testing set
test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()
# Place the numerical representation of `video_transcription_text` from test set into a dataframe
test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())
# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']).reset_index(drop=True), test_count_df], axis=1)
#print(X_test_final.head())


#STEP:07 Model Training

# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=0)
# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall',
    n_jobs=-1, )
rf_cv.fit(X_train_final, y_train)
# Examine best recall score
print("random_forest_cv.best_score_")
print(rf_cv.best_score_)
# Examine best parameters
print(rf_cv.best_params_)


#Building an XGBoost model
# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall',n_jobs=-1,)
xgb_cv.fit(X_train_final, y_train)
print("XGBoost.best_score_ and parameters")
print(xgb_cv.best_score_)
print(xgb_cv.best_params_)



# Use the random forest "best estimator" model to get predictions on the validation set
y_pred = rf_cv.best_estimator_.predict(X_val_final)
# Display the predictions on the validation set
#print(y_pred)
# Display the true labels of the validation set
#print(y_val)



# Create a confusion matrix to visualize the results of the classification model
# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)
# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.show()



# Create a classification report
# Create classification report for random forest model
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


#XGBoost
#Now, evaluate the XGBoost model on the validation set.
#Evaluate XGBoost model
y_pred = xgb_cv.best_estimator_.predict(X_val_final)
y_pred

# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)
# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.title('XGBoost - validation set')
plt.show()
# Create a classification report
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


#Both random forest and XGBoost model architectures resulted in nearly perfect models. Nonetheless, in this case random forest performed a little bit better, so it is the champion model.
#Now, use the champion model to predict on the test data.

# Use champion model to predict on test data
y_pred = rf_cv.best_estimator_.predict(X_test_final)
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.title('Random forest - test set');
plt.show()
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()