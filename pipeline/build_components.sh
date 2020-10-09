#!/bin/sh

echo "\nBuild and push preprocess component"
./Preprocess/build_image.sh

echo "\nBuild and push train component"
./Train/build_image.sh

echo "\nBuild and push test component"
./Test/build_image.sh

echo "\nBuild and push deploy component"
./Deploy/build_image.sh
