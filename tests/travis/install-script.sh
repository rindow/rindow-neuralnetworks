#!/bin/sh

composer install
composer require "phpunit/phpunit"

#if [ "$TRAVIS_PHP_VERSION" = "7.2" ]; then
#	composer require --dev "phpunit/phpunit 6.*"
#fi
