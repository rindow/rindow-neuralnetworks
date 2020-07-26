Rindow Neural networks
======================
Master: [![Build Status](https://travis-ci.com/rindow/rindow-neuralnetworks.png?branch=master)](https://travis-ci.com/rindow/rindow-neuralnetworks)

The Rindow Neural networks is a high-level neural networks library for deep learning.

The goal is to be able to describe a network model in PHP as well as Python
using a description method similar to Keras.

To find out more please visit our website now!

- [Rindow projects](https://rindow.github.io/)


If you use the rindow_openblas php extension,
you can calculate at speed close to CPU version of tensorflow.
The trained model trained on your laptop is available on general web hosting.
You can also benefit from deep learning on popular PHP web hosting services.

It has the following features.

- A high-level neural networks description
- Cooperation with high-speed operation library
- Designing for scalability of operation library
- Heterogeneous model distribution

Rindow Neural networks usually work with:

- Rindow Math Matrix: scientific matrix operation library
- Rindow OpenBLAS extension: PHP extension of OpenBLAS
- Rindow Math Plot: Visualize machine learning results

Requires
========

- PHP 7.2 or later.

Install
=======

## Install the Rindow Neural networks

Please set up with composer.

```shell
$ composer require rindow/rindow-neuralnetworks
$ composer require rindow/rindow-math-plot
```

### Download and setup the rindow_openblas extension

Training will take a lot of time if left untouched. It is **strongly recommended** that you set up the **rindow_openblas extension** in php for speed.

- [Pre-build binaries](https://github.com/rindow/rindow-openblas-binaries)
- [Build from source](https://github.com/rindow/rindow-openblas)

## Expansion of memory used
You need to increase the maximum amount of memory used by PHP depending on the amount of data used.

When training image data, do not be surprised that the amount of sample data is so large that it will be the maximum memory capacity that you cannot usually imagine.

For example, change the memory_limit item of php.ini as follows.

```shell
memory_limit = 8G
```

Describing the model
====================
Source code for simple image learning is provided in the sample directory.

Execute as follows.
```shell
$ RINDOW_MATH_PLOT_VIEWER=/some/bin/dir/png-file-viewer
$ export RINDOW_MATH_PLOT_VIEWER
$ mkdir samples
$ cd samples
$ cp ../vendor/rindow/rindow-neuralnetworks/samples/mnist-basic-clasification.php .
$ php mnist-basic-clasification.php
```

If done correctly, a graph of the learning process will be displayed.
