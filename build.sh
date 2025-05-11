#! /bin/bash

# make clean
set -e 
set -o pipefail
#
# update the auton.mk file
# build the project
pros build-compile-commands --no-analytics
#upload to the right file
