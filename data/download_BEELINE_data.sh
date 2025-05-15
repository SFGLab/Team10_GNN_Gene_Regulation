#!/bin/bash

if [ ! -f "BEELINE-data.zip" ]; then
	wget -O BEELINE-data.zip https://zenodo.org/records/3701939/files/BEELINE-data.zip?download=1
fi
unzip -o BEELINE-data.zip

mkdir -p BEELINE-Networks
cd BEELINE-Networks
if [ ! -f "BEELINE-Networks.zip" ]; then
	wget -O BEELINE-Networks.zip https://zenodo.org/records/3701939/files/BEELINE-Networks.zip?download=1
fi
unzip -o BEELINE-Networks.zip

