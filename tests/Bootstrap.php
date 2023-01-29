<?php
ini_set('short_open_tag', '1');

date_default_timezone_set('UTC');
#ini_set('short_open_tag',true);
if(file_exists(__DIR__.'/../vendor/autoload.php')) {
    $loader = require_once __DIR__.'/../vendor/autoload.php';
} else {
    $loader = require_once __DIR__.'/init_autoloader.php';
}

define('RINDOWTEST_TEMP_DIR',__DIR__.'/tmp');
if(!file_exists(RINDOWTEST_TEMP_DIR)) {
    mkdir(RINDOWTEST_TEMP_DIR);
}
#if(!class_exists('PHPUnit\Framework\TestCase')) {
#    include __DIR__.'/travis/patch55.php';
#}
