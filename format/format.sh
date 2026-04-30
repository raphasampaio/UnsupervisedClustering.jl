#!/bin/bash

BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia +1.12 --project=$BASE_PATH $BASE_PATH/format.jl