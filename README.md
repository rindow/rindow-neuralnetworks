Rindow Neural networks
======================
Status:
[![Build Status](https://github.com/rindow/rindow-neuralnetworks/workflows/tests/badge.svg)](https://github.com/rindow/rindow-neuralnetworks/actions)
[![Downloads](https://img.shields.io/packagist/dt/rindow/rindow-neuralnetworks)](https://packagist.org/packages/rindow/rindow-neuralnetworks)
[![Latest Stable Version](https://img.shields.io/packagist/v/rindow/rindow-neuralnetworks)](https://packagist.org/packages/rindow/rindow-neuralnetworks)
[![License](https://img.shields.io/packagist/l/rindow/rindow-neuralnetworks)](https://packagist.org/packages/rindow/rindow-neuralnetworks)


Rindow Neural Network Library is a high-level neural network library for deep learning.

Overview
--------

Like Keras in Python, you can easily write network models in PHP.

Website:
- Rindow project: https://rindow.github.io/
- Rindow Neural Networks: https://rindow.github.io/neuralnetworks

Speeding up
-----------

The external libraries rindow-matlib and OpenBLAS can be used to perform calculations at speeds close to CPU versions of TensorFlow.
Models trained on laptops are available on popular web hosting.
Deep learning is also available on popular PHP web hosting services.

GPU acceleration
----------------
It supports GPU acceleration using OpenCL.
You can also use GPUs other than n-vidia if they support OpenCL. It can also be used with an integrated GPU installed in your laptop.


Linked library
--------------
- Rindow Math Matrix: Scientific calculation library
- Rindow Matlib: A fast matrix calculation library suitable for machine learning
- OpenBLAS: Fast Matrix Arithmetic Library
- Rindow Math Plot: Visualize machine learning results
- OpenCL: GPU computational programming interface
- CLBlast: High-speed matrix calculation library using OpenCL

Required environment
--------------------

- PHP 8.1, 8.2, 8.3, 8.4
- For PHP 7.x, 8.0 environments, use Release 1.x.

Install
-------

> Click [here](https://rindow.github.io/neuralnetworks/install.html) for detailed instructions.

Please prepare in advance:

- Install php-cli, php-gd, and php-sqlite3 in advance using the method appropriate for each operating system.
- Install composer


Please install using Composer.
```shell
$ composer require rindow/rindow-neuralnetworks
$ composer require rindow/rindow-math-plot
```

If you use it as is, it will take time to learn. In order to increase speed, we strongly recommend that you install a high-speed calculation library.

Please set up an external library.

Prebuilt binaries:
- Rindow-matlib: https://github.com/rindow/rindow-matlib/releases
- OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/releases

Please set up according to your environment. Click [here](https://github.com/rindow/rindow-math-matrix-matlibffi) for detailed instructions.

```shell
$ composer require rindow/rindow-math-matlibffi
```

memory expansion:

Depending on the amount of data you use, you may need to increase the maximum amount of memory that PHP uses.

Especially when dealing with image data, the amount of sample data becomes enormous and requires more memory capacity than expected.

For example, change memory_limit in php.ini as follows.

memory_limit = 8G

Model description
-----------------
The sample directory provides source code for simple image learning.

Please run as follows:

```shell
$ RINDOW_MATH_PLOT_VIEWER=/path/to/viewer
$ export RINDOW_MATH_PLOT_VIEWER
$ mkdir samples
$ cd samples
$ cp ../vendor/rindow/rindow-neuralnetworks/samples/basic-image-clasification.php .
$ php basic-image-clasification.php
```

*Please specify an appropriate viewer for RINDOW_MATH_PLOT_VIEWER.
(ex. viewnior)

If done correctly, a graph of the learning process will be displayed.


GPU/OpenCL support
------------------
Please download the binary.
- CLBlast: https://github.com/CNugteren/CLBlast/releases

Set up the binary files according to your environment.
Detailed instructions here https://github.com/rindow/rindow-math-matrix-matlibffi/

Please set environment variables.

```shell
$ RINDOW_NEURALNETWORKS_BACKEND=rindowclblast::GPU
$ export RINDOW_NEURALNETWORKS_BACKEND
$ cd samples
$ php basic-image-classification.php
```

*For RINDOW_NEURALNETWORKS_BACKEND, you can specify not only a name such as rindowclblast, but also the OpenCL device type and a set of Platform-ID and Device-ID. For example, "rindowclblast::GPU" or "rindowclblast::0,0"
