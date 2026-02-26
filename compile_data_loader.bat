cmake -S . -B build -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build --config PGO_Generate

./build/training_data_loader .pgo/small.binpack

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=PGO_Use \
    -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config PGO_Use --target install