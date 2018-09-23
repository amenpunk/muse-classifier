#!/bin/bash
# usage: [tag] [page-from] [page-to] [char-label]
[ ! -e "$4.txt" ] && (python3 scrape.py $1 $2 $3 | tee $4.txt)
mkdir -p "images-$4" && \
cd "images-$4" && \
for i in `cat ../$4.txt`; do wget "$i"; done
#for i in `cat ../$4.txt`; do wget "$i" & disown; done
