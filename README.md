# drml
Org link from Kaili Zhao: https://github.com/zkl20061823/DRML

I changed these files to make them fit the new caffe version. (It's OK now for caffe version: https://github.com/BVLC/caffe 2017/8/19)

Please compare caffe.proto-org and caffe.proto to see differences.(You can use a tool named Beyond Compare.)



Usage:
1, Download caffe from https://github.com/BVLC/caffe. Be sure to install caffe correctly.

2, Copy files.

Copy all *\*.cpp/\**.cu  files into *src/caffe/layers/ 


Copy all *\*.hpp*  files into *include/caffe/layers/


Replace the old *src/caffe/proto/caffe.proto* with the new one given.

3, Recompile caffe

*make clean

*make  all
