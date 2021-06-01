sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
vulkaninfo | grep deviceType


# https://github.com/Tencent/ncnn/tree/master/python
# 

sudo mkdir -p /usr/local/c++
sudo chmod -R 777 /usr/local/c++
cd /usr/local/c++/
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule init && git submodule update

mkdir -p build && cd build

# set DNCNN_VULKAN=ON on GPU
sudo cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_PYTHON=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
sudo make -j$(nproc)
sudo make install

# -------------------------------------------------
# python
# -------------------------------------------------
#     Install

#         cd /pathto/ncnn/python
#         pip install .

#     if you use conda or miniconda, you can also install as following:

#         cd /pathto/ncnn/python
#         python3 setup.py install


cd ../python
pip3 install .

# ------------------------------------------------------
# Model Zoo
# -----------------------------------------------------

# install requirements

#     pip install -r requirements.txt

# then you can import ncnn.model_zoo and get model list as follow:

#     import ncnn
#     import ncnn.model_zoo as model_zoo

#     print(model_zoo.get_model_list())

pip3 install -r requirements.txt

# ---------------------------------------------------
# cmd usage
# ---------------------------------------------------
# python examples/retinaface.py  "path/to/image"
