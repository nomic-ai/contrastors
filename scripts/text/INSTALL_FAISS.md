# Building FAISS from Source For Cuda 12+
* `source env/bin/activate`
* `sudo apt purge --autoremove`
* `sudo apt-get install libssl-dev`
* `sudo apt-get install libblas-dev liblapack-dev`
* `wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc4/cmake-3.28.0-rc4.tar.gz`
* `tar -xvf cmake-3.28.0-rc4.tar.gz`
* `cd cmake-3.28.0-rc4/`
* `./bootstrap`
* `make -j$(nproc)`
* `sudo make install`
* `cmake -B build .`
* `make -C build -j faiss`
   * Make sure you're in the faiss directory now
* `make -C build -j swigfaiss`
* `(cd build/faiss/python && python setup.py install)`