#!/bin/sh

if [ -e vendor/bin/phpunit ]; then
	PHPUNITDIR="vendor/bin/"
else
	PHPUNITDIR=""
fi
${PHPUNITDIR}phpunit -c tests
