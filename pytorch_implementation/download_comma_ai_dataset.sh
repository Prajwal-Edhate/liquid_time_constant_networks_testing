#!/bin/bash

# mkdir data

# Download and extract driving dataset from comma.ai
curl -L "https://public.roboflow.com/ds/lPaKtJTeEb?key=ppuQGPa9vG" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
