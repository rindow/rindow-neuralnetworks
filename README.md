Rindow Neural networks
======================
Master: [![Build Status](https://app.travis-ci.com/rindow/rindow-neuralnetworks.svg?branch=master)](https://app.travis-ci.com/github/rindow/rindow-neuralnetworks)

The Rindow Neural networks is a high-level neural networks library for deep learning.

The goal is to make it easy to write network models in PHP, just like Keras on Python.

To find out more please visit our website now!

- [Rindow projects](https://rindow.github.io/)
- [Rindow NeuralNetworks](https://rindow.github.io/neuralnetworks)

If you use the rindow_openblas php extension,
you can calculate at speed close to CPU version of tensorflow.
The trained model trained on your laptop is available on general web hosting.
You can also benefit from deep learning on popular PHP web hosting services.

It supports GPU acceleration using OpenCL. This is an experimental attempt. The speed is not very fast yet. Only compatible with the Windows version.

It has the following features.

- A high-level neural networks description
- Cooperation with high-speed operation library
- Designing for scalability of operation library
- Heterogeneous model distribution

Rindow Neural networks usually work with:

- Rindow Math Matrix: scientific matrix operation library
- Rindow OpenBLAS extension: PHP extension of OpenBLAS
- Rindow Math Plot: Visualize machine learning results
- Rindow OpenCL extension: PHP extension of OpenCL
- Rindow CLBlast extension: PHP extension of CLBlast PHP binding.

Requires
========

- PHP 7.2, 7.3, 7.4 and 8.0.

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

- [Pre-build binaries](https://github.com/rindow/rindow-openblas/releases)
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
$ cp ../vendor/rindow/rindow-neuralnetworks/samples/basic-image-clasification.php .
$ php basic-image-clasification.php
```

If done correctly, a graph of the learning process will be displayed.

GPU/OpenCL support
==================

Download binaries and setup PHP extension and libraries.

- [Rindow OpenCL extension](https://github.com/rindow/rindow-opencl/releases)
- [Rindow CLBlast extension](https://github.com/rindow/rindow-clblast/releases)
- [CLBlast library](https://github.com/CNugteren/CLBlast/releases)

Set environment variable.

```shell
$ RINDOW_NEURALNETWORKS_BACKEND=rindowclblast
$ export RINDOW_NEURALNETWORKS_BACKEND
$ cd samples
$ php basic-image-clasification.php
```
