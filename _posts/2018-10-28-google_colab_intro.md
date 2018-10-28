---
layout: post
title: Introduction to Google Colab
feature-img: "assets/img/quantization/cover.png"
tags: [quantization, tf-lite]
---
### Introduction to Google Colab Notebooks
{: style="text-align: center"}



*Hello, Welcome to google colab intro tutorial*

# **What is Google Colab ?**
Google Colab is a free cloud service which comes with pre installed machine
learning frameworks like tensorflow and free gpus to run on

Google colaboratory
currently offers the computing services of a Tesla K80 GPU for free. The only
catch here is that you can use the computing services for a maximum of 12 hours
at a time (you can think of it in terms of a session). Basically, when you train
your models on the colaboratory, you are connected to a GPU-based virtual
machine where you are given a maximum of 12 hours at a time, after which you
lose access to that particular virtual machine instance (all data, that is,
models parameters as well as datasets that aren’t saved to the Google drive
before this period will be lost, so make sure to save snapshots of your model
parameters at regular intervals, else you will have to start training your
models from scratch again). After 12 hours you are assigned a different virtual
machine (for free of course) and the cycle repeats.

**Contents**:
1.   Running
Basic Python code in Colab
1.   How to run shell commands in cells
2.   CPU and
GPU configurations   
1.   How to mount google drive
2.   Run any ipython
notebook from github on Colab
1.   A simple cusine classification problem based
on ingredients

# ***1. Running Basic Python Code in Colab***

The cell below shows some python
examples

```python
x = 10

print(x)

y = 20

print(x + y)

```

# ***2. How to run shell commands ***

Inorder to run shell commands add ***!***
in front of the command

Example: **!ls** will list the contents of the folder

```python
!ls
!pip freeze #list of installed libraries
```

# ***3. Memory, CPU and GPU Configurations***

Memory - !cat /proc/meminfo

CPU
- !cat /proc/cpuinfo

GPU 

        import tensorflow as tf
tf.test.gpu_device_name()

```python
!cat /proc/meminfo

```

```python
!cat /proc/cpuinfo
```

```python
import tensorflow as tf
from tensorflow.python.client import device_lib

tf.test.gpu_device_name()
device_lib.list_local_devices()
```

# ***4. How to mount google drive***

The example below shows how to mount your
Google Drive in your virtual machine using an authorization code, and shows a
couple of ways to write & read files there. Once executed, observe the new file
(foo.txt) is visible in https://drive.google.com/

Note this only supports
reading and writing files; to programmatically change sharing settings etc use
one of the other options below.



Please refer this link to know how to load
files into Colab from local file system, google drive in three different forms
*
Using the native REST API;
* Using a wrapper around the API such as PyDrive; or
* Mounting your Google Drive in the runtime's virtual machine.
* From Google
Cloud Storage bucket

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```python
with open('/content/gdrive/My Drive/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
!cat /content/gdrive/My\ Drive/foo.txt
```

# ***5. How to run Pynb from github***

To load a specific notebook from github,
append the github path to http://colab.research.google.com/github/.

For example
to load 

https://github.com/shranith/Colab-intro/blob/master/Colab_intro.ipynb
Append `shranith/Colab-intro/blob/master/Colab_intro.ipynb` to
`http://colab.research.google.com/github/`

Link after appending:
http://colab.research.google.com/github/shranith/Colab-
intro/blob/master/Colab_intro.ipynb
