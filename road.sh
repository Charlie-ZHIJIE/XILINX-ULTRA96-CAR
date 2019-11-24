CXX=${CXX:-g++}

$CXX -std=c++11 -g -I. -o road road.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpssd -pthread -lglog 