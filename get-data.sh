#!/usr/bin/env bash

if [[ ! -d data/wikitext-2 ]]; then
    echo 'Downloading WikiText-2'
    mkdir -p data
    wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    unzip wikitext-2-v1.zip
    rm wikitext-2-v1.zip
    mv wikitext-2 data
fi
