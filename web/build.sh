#!/bin/bash

echo ">>>>>>>>ARGS: $@"

set -e

if [ "$1" != "--quick" ]; then
    rm -rf web-build
    mkdir -p web-build/_static
    rm -rf web/.build
    mkdir web/.build
    ./run web-docs
    mkdir -p web-build/docs
    cp -r docs/.web-build/* web-build/docs/
fi

echo -n "var DOC_VERSIONS_RAW = [" >web-build/_static/versions.js
for V in `ls web-build/docs/v`; do
	echo "Patching version $V"
	cp -r web/_static web-build/docs/v/$V/

	echo "\"$V\", " >>web-build/_static/versions.js
done
echo "]" >>web-build/_static/versions.js

cp web/docs/index.html web-build/docs

cp -r web/* web/.build

_WEB_DOCS=1 sphinx-build --keep-going -n -W -b html web/.build web-build
