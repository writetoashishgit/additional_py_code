#!/usr/bin/env bash

nohup spark2-submit HDS-351.py \
--master yarn \
--deploy-mode client \
--executor-memory 4G \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.dynamicAllocation.maxExecutors=4 \
--conf spark.executor.cores=1 \
--files HDS-351-stage-1.sql \
&
