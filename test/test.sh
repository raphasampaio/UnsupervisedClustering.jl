#!/bin/bash

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$JULIA_1104 --project=$BASEPATH/.. -e "import Pkg; Pkg.test()"