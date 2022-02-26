

DOCUMENTER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

julia --project=$DOCUMENTER_DIR $DOCUMENTER_DIR/make.jl
