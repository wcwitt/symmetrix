#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage:    ./install.sh path/to/gromacs"
    echo "Patches the GROMACS 2026.4 source tree for the Symmetrix NNPot models (GMX_NNPOT=SYMMETRIX)."
    exit 1
fi

# absolute path to gromacs, then work from the directory containing this script
gromacs=$(cd "$1" && pwd)
cd "$(dirname "$0")"
patch=$(pwd)/gromacs-2026.4.patch

if ! grep -q '^project(Gromacs VERSION 2026.4)' ${gromacs}/CMakeLists.txt; then
    echo "${gromacs} is not GROMACS 2026.4, which the patch is written for." 1>&2
    exit 2
fi

# a GROMACS tree without its own .git must not be taken for part of an enclosing repository, where git apply would
# skip the paths outside the current directory
export GIT_CEILING_DIRECTORIES=$(dirname ${gromacs})

if git -C ${gromacs} apply --unidiff-zero --reverse --check ${patch} 2>/dev/null; then
    echo "The Symmetrix patch is already applied to ${gromacs}." 1>&2
elif git -C ${gromacs} apply --unidiff-zero --check ${patch}; then
    echo git apply --unidiff-zero ${patch} 1>&2
         git -C ${gromacs} apply --unidiff-zero ${patch}
else
    echo "The Symmetrix patch does not apply to ${gromacs}; restore its NNPot sources and re-run." 1>&2
    exit 2
fi

echo "
Configure GROMACS with
    -DGMX_NNPOT=SYMMETRIX -DSYMMETRIX_GROMACS_DIR=$(pwd)
and, to build for other GPUs than this machine's, -DSYMMETRIX_CUDA_ARCHITECTURES=\"90;100\"." 1>&2
