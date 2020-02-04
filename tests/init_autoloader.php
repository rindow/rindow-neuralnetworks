<?php
if (file_exists(__DIR__.'/../vendor/autoload.php')) {
    $loader = include __DIR__.'/../vendor/autoload.php';
} else {
    define('COMPOSER_LIBRARY_PATH', getenv('COMPOSER_LIBRARY_PATH'));
    if(COMPOSER_LIBRARY_PATH && file_exists(COMPOSER_LIBRARY_PATH.'/vendor/autoload.php')) {
        $loader = include COMPOSER_LIBRARY_PATH.'/vendor/autoload.php';
    } else {
        throw new \Exception("Loader is not found.");
    }
}
//$loader->add('Rindow\\Module\\Monolog\\', __DIR__.'/../../module-monolog/src');
