#!/bin/bash

FORMATTER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia --project=$FORMATTER_DIR $FORMATTER_DIR/format.jl
