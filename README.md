# retinaface_ncnn
Retinaface face detector in cpp and python using ncnn

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
