#! /bin/bash
export CMAKE_PREFIX_PATH=$AMENT_PREFIX_PATH
rm -rf _build _install

mkdir -p _build
mkdir -p _install

pushd _build
cmake -DCMAKE_INSTALL_PREFIX:PATH=../_install ..
make install
popd