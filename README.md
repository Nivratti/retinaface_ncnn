# retinaface_ncnn
Retinaface face detector in cpp and python using ncnn

Features:
 * Added support to select retinaface R50 model for face detection in python
 
## Build retinaface ncnn cpp

```sh
$ git clone https://github.com/Nivratti/retinaface_ncnn.git
$ cd retinaface_ncnn/cpp
$ mkdir -p "build"
$ cd "build"
```
```sh
$ cmake ..
$ make -j$(nproc)
```

To test face detection run 
```sh
$ ./retinaface [image_path]
```
