#!/bin/bash

for index in {0..9}; do
    for year in {0..15}; do
        python3 MCMC_Finance.py $index $year
    done
done