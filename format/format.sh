#!/bin/bash

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$JULIA_194 --project=$BASEPATH $BASEPATH/format.jl