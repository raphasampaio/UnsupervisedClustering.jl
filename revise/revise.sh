#!/bin/bash

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia +1.11 --project=$BASEPATH --interactive $BASEPATH/revise.jl