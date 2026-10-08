#!/bin/bash

BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia --project=$BASE_PATH --interactive $BASE_PATH/revise.jl