#!/usr/bin/env bash
wiki_dump_file=$1
directory=$2
log_file=$directory/log.txt

if [ ! -d $2 ]; then
   mkdir -p $2;
fi

./create_resource_from_corpus.sh $1 $2 $3 $4 >> "$log_file" 2>> "$log_file"
