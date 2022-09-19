#!/bin/bash

BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REVISE_PATH="$BASE_PATH/revise"

$JULIA_167 --project=$REVISE_PATH --load=$REVISE_PATH/revise.jl