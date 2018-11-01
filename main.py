from gensim.models import word2vec
import sklearn as skl
import pandas as pd
import numpy as np


train_data = pd.read_csv("./Data/train.csv")

print(train_data.columns)

