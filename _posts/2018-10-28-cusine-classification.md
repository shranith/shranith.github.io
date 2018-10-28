---
layout: post
title: Cusine Classification
feature-img: "assets/img/quantization/cover.png"
tags: [kaggle, word-embeddings, logistic-regressions]
---
# Classify Cusine based on ingredients

This is one of the kaggle
challenges , refer  [this link](https://www.kaggle.com/c/whats-cooking/data) for
more info on challenge

Run [Ipynb](https://colab.research.google.com/github/shranith/Colabintro/blob/master/Cuisine_classification.ipynb) on Google-colab

Details

Use recipe ingredients to categorize the cuisine.
Training data consists of id, cuisine and ingredients

```json
{
"id": 24717,
"cuisine": "indian",
"ingredients": [
    "tumeric",
    "vegetable stock",
    "tomatoes",
    "garam masala",
    "naan",
    "red lentils",
    "red chili peppers",
    "onions",
    "spinach",
    "sweet potatoes"
 ]
 }

 ```

Test data consistst of id and ingredients and we are expected to predict the
cuisine

``` json
{
"id": 41580, 
"ingredients": [
    "sausage links",
    "fennel bulb",
    "fronds",
    "olive oil",
    "cuban peppers",
    "onions"
]
}

```

**Install the requirements **

Most of the pacakges are pre installed in the
google colab.

Installed packages does not include `gensim`.  

Installing
gensim using `pip3`

```bash
!pip3 install gensim
```

```python
import os
from os import listdir
import gensim
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
```

**Mount google drive** 

Add [this
folder](https://drive.google.com/drive/folders/1jMTbMH0qwoiU64JWO8I_4OXVLTtD7pl6?usp=sharing)
to your google drive to load the datasets from your google drive

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```python
# Print contents of mount dataset

path = '/content/gdrive/My Drive/cusines dataset/'
l = [f for f in listdir(path)]
print(l)


# Inspect contents 
inspect_test = json.load(open(path+'/'+'test.json','r'))
print(inspect_test[2])
```

```python
# Load train and test sets using pandas

data = pd.read_json(path+'train.json')
test = pd.read_json(path+'test.json')

print('Training data shape: {}'.format(data.shape))
print('Test data shape: {}'.format(test.shape))
print('Dataset Keys {}'.format(data.keys()))


index = 1
print('id: {}'.format(data['id'].iloc[index]))
print('ingredients: {}'.format(data['ingredients'].iloc[index]))
print('cuisine: {}'.format(data['cuisine'].iloc[index]))

```

```python
#load target labels to predict
target = data.cuisine
```

```python
# Assign a new column to have the counts of each ingredients
data['ingredient_count'] = data.ingredients.apply(lambda x: len(x))

def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]
```

```python
# Plot figures 

f = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs[0, :])
data.ingredient_count.value_counts().hist(ax=ax1)
ax1.set_title('Recipe richness', fontsize=12)

ax2 = plt.subplot(gs[1, 0])
pd.Series(flatten_lists(list(data['ingredients']))).value_counts()[:20].plot(kind='barh', ax=ax2)
ax2.set_title('Most popular ingredients', fontsize=12)

ax3 = plt.subplot(gs[1, 1])
data.groupby('cuisine').mean()['ingredient_count'].sort_values(ascending=False).plot(kind='barh', ax=ax3)
ax3.set_title('Average number of ingredients in cuisines', fontsize=12)

plt.show()
```

```python
# load word embeddings for all of the ingredients 

w2v = gensim.models.Word2Vec(list(data.ingredients), size=350, window=10, min_count=2, iter=20)

```

```python
# most similar word
w2v.most_similar(['meat'])
```

```python
w2v.most_similar(['salt'])
```

```python
#Inspect Vocab

print(w2v.wv.vocab.keys())
print(w2v.wv['romaine lettuce'])
print(type(data.ingredients))

print(len(data.ingredients))
print(len(test.ingredients))
```

```python
def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in w2v.wv.vocab]
    return np.mean(w2v[doc], axis=0)
```

```python
# Adding another column to store the document embeddings
data['doc_vector'] = data.ingredients.apply(document_vector)
test['doc_vector'] = test.ingredients.apply(document_vector)
```

```python
print(data['doc_vector'].iloc[1])
print(data['cuisine'].iloc[1])
```

```python

lb = LabelEncoder() # Encode labels with value between 0 and n_classes-1.

y = lb.fit_transform(target) # Fit label encoder and return encoded labels
print(y)
```

```python
X = list(data['doc_vector'])
X_test = list(test['doc_vector'])
```

```python
# Intialize a Logistic Regression Classifier
clf = LogisticRegression(C=100) 
# C is regularization strength

#Follow this link for a primer on Logistic Regression
#https://www.kdnuggets.com/2016/08/primer-logistic-regression-part-1.html
```

```python
# Train the classifier
clf.fit(X, y)

```

```python
# save the model to disk
filename = 'finalized_model.sav'
# pickle.dump(clf, open(path + '/'+filename, 'wb'))
 
 
# load the model from disk
clf = pickle.load(open(path + '/'+ filename, 'rb'))
```

```python
def predict(ingredient_list):
  """Predict cusine based on ingredient list"""
  doc_vector = document_vector(ingredient_list)
  y_test = clf.predict([doc_vector])
  y_pred = lb.inverse_transform(y_test)
  return y_pred
  
  
# ingredient_list = ['sausage links','fennel bulb','fronds','olive oil','cuban peppers','onions', 'salt']
ingredient_list = ['plain flour', 'cheese', 'ground pepper', 'salt', 'tomatoes', 'ground black pepper', 'thyme', 'eggs', 'green tomatoes', 'yellow corn meal', 'milk', 'vegetable oil']
print('Cusine ',predict(ingredient_list))
```

```bash
# !ls -alh
!ls gdrive/My\ Drive/
```

```python
# X_test1 = X_test[:1]
# 
y_test = clf.predict(X_test)
y_pred = lb.inverse_transform(y_test)
print(y_pred)
```

```python
test_id = [id_ for id_ in test.id]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv(path + '/'+'clf_output.csv', index=False)
```
