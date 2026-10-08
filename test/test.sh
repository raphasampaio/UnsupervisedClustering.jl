#!/bin/bash

BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia --project=$BASE_PATH/.. -e "import Pkg; Pkg.test()"