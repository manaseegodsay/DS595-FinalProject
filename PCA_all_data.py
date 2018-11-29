from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load in the data
train_data = pd.read_csv("./Data/train.csv")

# Split into training and testing DataFrames
x_train = train_data[["id", "qid1", "qid2", "question1", "question2"]]
y_train = train_data["is_duplicate"]

# Separate the questions into their own DataFrames for vectorization
train_question1s = train_data["question1"].astype(str).tolist()
train_question2s = train_data["question2"].astype(str).tolist()

# Combine the two for vectorization and dimension reduction
train_questions = train_question1s + train_question2s

# Declare the vectorizer
vectorizer = TfidfVectorizer(max_features=257)

# Fit and transform the data
train_tfidfs = vectorizer.fit_transform(train_questions)
print(train_tfidfs.shape)

# Declare the "PCA"
#    NOTE: We can't use PCA because our matrix is very sparse
pca = TruncatedSVD(n_components=256)

# Fir and transform the data
reduced_tfidfs = pca.fit_transform(train_tfidfs, pd.concat([y_train, y_train]))

# Take some statistics from the PCA
explained_variance = pca.explained_variance_ratio_
singular_values = pca.singular_values_

# Plot the explained variance
plt.figure()
explained_variance_plt = plt.plot(explained_variance)
plt.title('Variance Explained by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.savefig("./Figures/ExplainedVariance.png", dpi=700)

# Plot the singular values
plt.figure()
singular_values_plt = plt.plot(singular_values)
plt.title('Singular Values by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Singular Values')
plt.savefig("./Figures/SingularValues.png", dpi=700)

plt.show()

# Split the questions back into separate matrices
[train_question1_tfidfs, train_question2_tfidfs] = np.split(reduced_tfidfs, 2)
