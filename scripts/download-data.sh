#!/usr/bin/env sh 

# download wine data from kaggle

kaggle datasets download -d zynicide/wine-reviews
mv wine-reviews.zip ../data
cd ../data
unzip wine-reviews.zip
rm -rf wine-reviews.zip


