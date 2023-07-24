#!/bin/bash

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$JULIA_192 --project=$BASEPATH $BASEPATH/format.jl