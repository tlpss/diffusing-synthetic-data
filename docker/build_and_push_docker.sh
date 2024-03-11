#!bin/bash

cd .. && docker build -f docker/Dockerfile . -t dsd:cvpr-syntagen
docker tag dsd:cvpr-syntagen tlpss/dsd:cvpr-syntagen
docker push tlpss/dsd:cvpr-syntagen
