#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "demo1.hpp"
#include <xilinx/ai/ssd.hpp>
#include <xilinx/ai/nnpp/ssd.hpp>
#include "./process_result.hpp" 

using namespace std;

int main(int argc, char *argv[]) {

	std::string output =  "result.mp4";
//	cv::VideoCapture cap(argv[1]);
  LOG(INFO) << "argv[1] = " << argv[1];
	std::string str = argv[1];
  bool is_camera =
        str.size() == 1 && str[0] >= '0' && str[0] <= '9';
    if (is_camera) {
        LOG(INFO) << "Using camera";
      } 
  int n = atoi(str.c_str());
	cv::VideoCapture cap(n);
  const int width = 640 ;
	const int height = 360;
  cv::VideoWriter writer;
	const int fps = 25;
//	const int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//	const int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  cap.set(cv::CAP_PROP_FRAME_WIDTH,width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT,height);
  LOG(INFO) << "width = " << width;
  LOG(INFO) << " height = " << height;
	cv::Size size = cv::Size(width, height);
	std::vector<cv::Point2i> vertices;
	vertices.push_back(cv::Point2i(size.width*0.15, size.height*0.88));
	vertices.push_back(cv::Point2i(size.width*0.38, int(size.height*0.65)));
	vertices.push_back(cv::Point2i(size.width - size.width*0.48, int(size.height*0.65)));
	vertices.push_back(cv::Point2i(size.width - size.width*0.15, size.height*0.88));
	double low_thres = 30.0;
	double high_thres = 80.0;
	cv::Mat frame;
	cv::Mat result;
//	writer.open(output, CAP_OPENCV_MJPEG, fps, size);
// 	writer.open(output, CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
//  判断是否创建成功	
//	if (!writer.isOpened()) {
//		cout << "could not creat video file" << endl;
//		return -1;
//	}	
//	Mat frame(size, CV_8UC3); //视频帧
	cv::Mat gray;
	cap >> frame;
  cap.release();
//	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
  cv::imwrite("frame.jpg", frame);
  mask = cv::Mat::zeros(gray.size(), gray.type());
	if (gray.channels() == 1) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255));
		//		imshow("mask", mask);
	}
	else if (gray.channels() == 3) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));
	}
    xilinx::ai::main_for_video_demo(
      argc, argv,
      [] {return xilinx::ai::SSD::create(xilinx::ai::SSD_TRAFFIC_480x360);},
      process_result);
    return 0;
}


