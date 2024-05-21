import openai
from sk import my_sk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import data
df_resume = pd.read_csv('resumes/resumes_train.csv')
df_resume.head()

# Generate text embeddings
def generate_embeddings(text, my_sk):
    # set credentials
    client = openai.OpenAI(api_key = my_sk)
    
    # make api call
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    # return text embedding
    return response.data

# generate embeddings
text_embeddings = generate_embeddings(df_resume['resume'], my_sk)
# extract embeddings
text_embedding_list = [text_embeddings[i].embedding for i in range(len(text_embeddings))]

# Store embeddings in df
# define df column names
column_names = ["embedding_" + str(i) for i in range(len(text_embedding_list[0]))]

# store text embeddings in dataframe
df_train = pd.DataFrame(text_embedding_list, columns=column_names)

# create target variable
df_train['is_data_scientist'] = df_resume['role']=="Data Scientist"
df_train.to_csv('resumes/embeddings_train.csv', index=False)
df_train.head()

# Visualize embeddings with PCA
# split variables by predictors and target
X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]

# apply PCA to predictors (i.e. embeddings)
pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_ratio_)

# plot data along PCA components
c_arr = np.array(["k"] * len(y))
c_arr[y] = "r"

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})
plt.scatter(pca.transform(X)[:,0], pca.transform(X)[:,1], c=c_arr)
plt.legend(["Data Scientist"])
plt.xticks(rotation = 45)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# Train random forest model
# train rf model
clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
clf.score(X,y) # model accuracy for training data

# AUC value for training data
auc_val = roc_auc_score(y, clf.predict_proba(X)[:,1])
print("AUC = ", auc_val)

# print feature importance ranking
feature_importances = pd.Series(clf.feature_importances_, index=column_names)
feature_importances_sorted = feature_importances.sort_values(ascending=False)
print("feature importance ranking: ",feature_importances_sorted[:25])

# plot data along top 2 most important features
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})
plt.scatter(X[feature_importances_sorted.index[0]], X[feature_importances_sorted.index[1]], c=c_arr)
plt.legend(["Data Scientist"])
plt.xticks(rotation = 45)
plt.xlabel(feature_importances_sorted.index[0])
plt.ylabel(feature_importances_sorted.index[1])
plt.show()

# Evaluate model on test data

# import testing data
df_resume = pd.read_csv('resumes/resumes_test.csv')

# generate embeddings
text_embedding_list = generate_embeddings(df_resume['resume'], my_sk)
text_embedding_list = [text_embedding_list[i].embedding for i in range(len(text_embedding_list))]

# store text embeddings in dataframe
df_test = pd.DataFrame(text_embedding_list, columns=column_names)

# create target variable
df_test['is_data_scientist'] = df_resume['role']=="Data Scientist"
df_test.to_csv('resumes/embeddings_test.csv', index=False)
df_test.head()

# define predictors and target
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]

# accuracy
clf.score(X_test,y_test)

# auc
auc_val = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print("AUC test data = ", auc_val)

cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

# look at errors
df_errors = df_resume[y_test != clf.predict(X_test)]
print(df_errors.iloc[0,0])