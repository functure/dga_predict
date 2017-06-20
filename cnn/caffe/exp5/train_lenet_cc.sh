#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=dga/lenet_solver_cc.prototxt $@
