<?php
define('COMPOSER_LIBRARY_PATH', getenv('COMPOSER_LIBRARY_PATH'));
if(COMPOSER_LIBRARY_PATH && file_exists(COMPOSER_LIBRARY_PATH.'/vendor/autoload.php')) {
    $loader = include COMPOSER_LIBRARY_PATH.'/vendor/autoload.php';
} else {
    throw new \Exception("Loader is not found.");
}
$loader->addPsr4('Rindow\\NeuralNetworks\\',__DIR__.'/../src');
$loader->addPsr4('Rindow\\Math\\Matrix\\', __DIR__.'/../../rindow-math-matrix/src');
$loader->addPsr4('Rindow\\Math\\Matrix\\Drivers\\MatlibFFI\\',__DIR__.'/../../rindow-math-matrix-matlibffi/src');
$loader->addPsr4('Rindow\\Math\\Buffer\\FFI\\', __DIR__.'/../../rindow-math-buffer-ffi/src');
$loader->addPsr4('Rindow\\Math\\Plot\\', __DIR__.'/../../rindow-math-plot/src');
$loader->addPsr4('Rindow\\Matlib\\FFI\\', __DIR__.'/../../rindow-matlib-ffi/src');
$loader->addPsr4('Rindow\\OpenBLAS\\FFI\\', __DIR__.'/../../rindow-openblas-ffi/src');
$loader->addPsr4('Rindow\\OpenCL\\FFI\\', __DIR__.'/../../rindow-opencl-ffi/src');
$loader->addPsr4('Rindow\\CLBlast\\FFI\\', __DIR__.'/../../rindow-clblast-ffi/src');
$loader->addPsr4('Interop\\Polite\\Math\\', __DIR__.'/../../../interop-phpobjects/polite-math/src');

return $loader;
