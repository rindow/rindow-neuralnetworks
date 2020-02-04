<?php
ini_set('short_open_tag', '1');

date_default_timezone_set('UTC');
#ini_set('short_open_tag',true);
include 'init_autoloader.php';
if(!file_exists(__DIR__.'/tmp')) {
    mkdir(__DIR__.'/tmp');
}

#if(!class_exists('PHPUnit\Framework\TestCase')) {
#    include __DIR__.'/travis/patch55.php';
#}
