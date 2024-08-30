#!/bin/bash

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$JULIA_1105 --project=$BASEPATH --interactive $BASEPATH/revise.jl