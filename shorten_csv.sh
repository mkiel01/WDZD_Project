#!/usr/bin/env bash

set -e

head -n 1 tweets16m.csv > test.csv
tail -n +2 tweets16m.csv | shuf -n "$1" >> test.csv
