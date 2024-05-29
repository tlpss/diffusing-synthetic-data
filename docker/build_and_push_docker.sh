#!bin/bash

cd .. && docker build -f docker/Dockerfile . -t dsd:ral
docker tag dsd:ral tlpss/dsd:ral-v1
docker push tlpss/dsd:ral-v1
