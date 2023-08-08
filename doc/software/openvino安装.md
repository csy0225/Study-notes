安装软件：
    ```
        apt-get install cpio
        apt-get install pciutils
        apt-get install libusb-dev
    ```

编译：
    ```
    mkdir -p build/Debug && \
    cd build/Debug 
    
    cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_CLANG_FORMAT=ON \
    -DCMAKE_INSTALL_PREFIX="../../openvino_dist/Debug" \
    -DPYTHON_EXECUTABLE=/usr/local/python3.7.5/bin/python3 \
    -DPYTHON_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so \
    -DPYTHON_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m \
    -DENABLE_MKL_DNN=ON \
    -DENABLE_CLDNN=OFF \
    ../..

    cd /src/bindings/python/src/compatibility/openvino
    pip3 install -r requirements-dev.txt
    cmake --build . --parallel 8 && cmake --install .
    ```

Paddle-Lite编译：
```
    ./lite/tools/build_linux.sh --arch=x86 --with_nnadapter=ON --nnadapter_with_intel_openvino=ON  --nnadapter_intel_openvino_sdk_root=/work/test1/openvino/openvino_dist/Debug
```


    cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_CLANG_FORMAT=ON \
    -DCMAKE_INSTALL_PREFIX="../../openvino_dist/Debug" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.7 \
    -DENABLE_MKL_DNN=ON \
    -DENABLE_CLDNN=OFF \
    ../..

    -Wdelete-non-virtual-dtor

    https://github.com/openvinotoolkit/openvino/blob/master/docs/MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md

    benchmark_app

    https://docs.openvino.ai/2020.2/_tools_accuracy_checker_README.html

    https://docs.openvino.ai/latest/omz_tools_accuracy_checker.html

    MobileNet v2/v3; BiSeNet v2; OCRNET; ResNet-50 can support by GPU

安装步骤：
    1. 操作系统：仅支持 Ubuntu 18.04 long-term support (LTS), 64-bit,
                     Ubuntu 20.04 long-term support (LTS), 64-bit
       Since the OpenVINO™ 2022.1 release, CentOS 7.6, 64-bit is not longer supported.
    2. 安装步骤：
        链接：https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
        wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18617/l_openvino_toolkit_p_2022.1.0.643_offline.sh

        2.1 Run the installer in silent mode 
        l_openvino_toolkit_p_2022.1.0.643_offline.sh -a -s --eula accept