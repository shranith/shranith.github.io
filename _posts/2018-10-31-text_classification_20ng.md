---
layout: post
title: Text Classification on 20 ng dataset
feature-img: "assets/img/national_park/jackson_hole.jpg"
tags: [Google-colab, 20ng, tensorflow, estimators]
---
### Building a text classification model with TF Hub
{: style="text-align: center"}

Before jumping into the content the above photo is from Jackson Hole, Wyoming. If you are around Colorado, Wyoming, Utah then this one of the nice place for skiing ⛷

**NOTE** You can run the following code as a python notebook on Google Colab, If you are not familiar with Google colab here is a link to an intro [tutorial](https://shranith.github.io/2018/10/28/google_colab_intro.htm) I made on how to use it.

<a href="https://colab.research.google.com/github/shranith/ML-notebooks/blob/master/text_classification_20ng.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

What are we gonna talk about today ?

* Text Classification
* 20 NG Dataset
* TF Hub and TF Estimators

**Text Classification** is essentially classifying a piece of text in to classes. There are broadly two ways one can approach the problem, supervised vs unsupervised depending on the availability of dataset. In both the approaches there are many ways one can classify the text. Today we are gonna look at one possiible approach using the tensorflow hub. The emphasis here is not on accuracy, but instead how to use TF Hub and TF Estimators in a text classification model.

**20 NG** or 20 Newsgroups data set, is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular dataset for experiments in text applications of machine learning techniques, such as text classification and text clustering.

**TensorFlow Hub** is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning.
Transfer learning can

* Train a model with a smaller dataset,
* Improve generalization, and
* Speed up training.

**Estimators** are highlevel abstractions provided by tensorflow and  simplifies training, evaluation, prediction, export for serving. One can use the prebuilt estimators or create their own custom estimator

Estimators provide the following benefits:

* You can run Estimator-based models on a local host or on a distributed multi-server environment without changing your model. Furthermore, you can run Estimator-based models on CPUs, GPUs, or TPUs without recoding your model.
* Estimators simplify sharing implementations between model developers.
* You can develop a state of the art model with high-level intuitive code. In short, it is generally much easier to create models with Estimators than with the low-level TensorFlow APIs.
* Estimators are themselves built on tf.keras.layers, which simplifies customization.
* Estimators build the graph for you.
* Estimators provide a safe distributed training loop that controls how and when to:
  * build the graph
  * initialize variables
  * load data
  * handle exceptions
  * create checkpoint files and recover from failures
  * save summaries for TensorBoard

**NOTE** When writing an application with Estimators, you must separate the data input pipeline from the model. This separation simplifies experiments with different data sets.

## Lets get to the code

To start, import the necessary
dependencies for this project.

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import json
import pickle
import urllib

from sklearn.preprocessing import LabelEncoder

print(tf.__version__)
```

One can download the 20NG dataset from [here](http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz) however I pre-processed and saved it as csv file in a shared [google-drive](https://drive.google.com/drive/folders/1xaQS8KsGWu7eQSZVcVYkNjFmmTaSXgpr?usp=sharing)

Mount the google drive folder and load the csv file

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```python
data = pd.read_csv('/content/gdrive/My Drive/20ng/20news-bydate-train.csv')
print(data.head())

descriptions = data['text']
category = data['category']
category[:10]
```

```python
descriptions[:10]
print(type(category[1]))
type(category)

```

### Splitting our data

When we train our model, we'll use 80% of the data for
training and set aside 20% of the data to evaluate how our model performed.

```python
train_size = int(len(descriptions) * .8)

train_descriptions = descriptions[:train_size].astype('str')
train_category = category[:train_size]

test_descriptions = descriptions[train_size:].astype('str')
test_category = category[train_size:]
```

```python
print(test_category)
```

```python
encoder = LabelEncoder()
encoder.fit_transform(train_category)
train_encoded = encoder.transform(train_category)
test_encoded = encoder.transform(test_category)
num_classes = len(encoder.classes_)

# Print all possible classes and the labels for the first movie in our training dataset
print(encoder.classes_)
print(train_encoded[0])
```

### Create our TF Hub embedding layer
[TF Hub]() provides a library of existing
pre-trained model checkpoints for various kinds of models (images, text, and
more) In this model we'll use the TF Hub `universal-sentence-encoder` module for
our pre-trained word embeddings. We only need one line of code to instantiate
module. When we train our model, it'll convert our array of movie description
strings to embeddings. When we train our model, we'll use this as a feature
column.

```python
description_embeddings = hub.text_embedding_column("descriptions", module_spec="https://tfhub.dev/google/universal-sentence-encoder/3", trainable=False)

```

## Instantiating our DNNEstimator Model
The first parameter we pass to our
DNNEstimator is called a head, and defines the type of labels our model should
expect. Since we want our model to output one of the multiple labels, we’ll use
multi_class_head here. Then we'll convert our features and labels to numpy
arrays and instantiate our Estimator. `batch_size` and `num_epochs` are
hyperparameters - you should experiment with different values to see what works
best on your dataset.

```python
multi_label_head = tf.contrib.estimator.multi_class_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)
```

```python
features = {
  "descriptions": np.array(train_descriptions).astype(np.str)
}
labels = np.array(train_encoded).astype(np.int32)
train_input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True, batch_size=32, num_epochs=25)
estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings])
```

## Training and serving our model 
To train our model, we simply call `train()`
passing it the input function we defined above. Once our model is trained, we'll
define an evaluation input function similar to the one above and call
`evaluate()`. When this completes we'll get a few metrics we can use to evaluate
our model's accuracy.

```python
estimator.train(input_fn=train_input_fn)
```

```python
# Define our eval input_fn and run eval
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(test_descriptions).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)
estimator.evaluate(input_fn=eval_input_fn)
```

## Generating predictions on new data
Now for the most fun part! Let's generate
predictions on random descriptions our model hasn't seen before. We'll define an
array of 3 new description strings (the comments indicate the correct classes)
and create a `predict_input_fn`. Then we'll display the top 2 categories along
with their confidence percentages for each of the 3 descriptions

```python
# Test our model on some raw description data
raw_test = [
    "The attacking midfielder came on as a substitute in the 1-0 defeat to Pep Guardiola's side having not played since September's Carabao Cup win against Watford because of a hamstring injury.", # sports
    "On Twitter on Tuesday, West said he supports prison reform, common-sense gun laws and compassion for people seeking asylum, then denied that he had designed a logo for a branding exercise known as “Blexit,” which urges African Americans to leave the Democratic party. The concept, originated by Owens, claimed that West had designed the group’s merchandise.", # Politics
    "From: ahmeda@McRCIM.McGill.EDU (Ahmed Abu-Abed)\nSubject: Re: Desertification of the Negev\nOriginator: ahmeda@ice.mcrcim.mcgill.edu\nNntp-Posting-Host: ice.mcrcim.mcgill.edu\nOrganization: McGill Research Centre for  Intelligent Machines\nLines: 23\n\n\nIn article <1993Apr26.021105.25642@cs.brown.edu>, dzk@cs.brown.edu (Danny Keren) writes:\n|> This is nonsense. I lived in the Negev for many years and I can say\n|> for sure that no Beduins were \"moved\" or harmed in any way. On the\n|> contrary, their standard of living has climbed sharply; many of them\n|> now live in rather nice, permanent houses, and own cars. There are\n|> quite a few Beduin students in the Ben-Gurion university. There are\n|> good, friendly relations between them and the rest of the population.\n|> \n|> All the Beduins I met would be rather surprised to read Mr. Davidson's\n|> poster, I have to say.\n|> \n|> -Danny Keren.\n|> \n\nIt is nonsense, Danny, if you can refute it with proof. If you are citing your\nexperience then you should have been there in the 1940's (the article is\ncomparing the condition then with that now).\n\nOtherwise, it is you who is trying to change the facts.\n\n-Ahmed.\n", # politics.middleeast
]


```

```python
# Generate predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn({"descriptions": np.array(raw_test).astype(np.str)}, shuffle=False)
results = estimator.predict(predict_input_fn)
```

```python
# Display predictions
for categories in results:
  top_2 = categories['probabilities'].argsort()[-2:][::-1]
  for category in top_2:
    text_category = encoder.classes_[category]
    print(text_category + ': ' + str(round(categories['probabilities'][category] * 100, 2)) + '%')
  print('')
```
