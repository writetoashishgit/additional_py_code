#!/usr/bin/env bash

nohup spark2-submit HDS-340.py \
--master yarn \
--deploy-mode client \
--executor-memory 4G \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.dynamicAllocation.maxExecutors=4 \
--conf spark.executor.cores=1 \
&
