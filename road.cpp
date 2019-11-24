#include <glog/logging.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
using namespace std;
using namespace cv;

using std::cout;
typedef std::vector<cv::Vec4i> linesType;
const double PI = 3.1415926535;
bool show_steps = false;

// m: 斜率, b: 截距, norm: 线长
struct hough_pts {
	double m, b, norm;
	hough_pts(double m, double b, double norm) :
		m(m), b(b), norm(norm) {};
};

void drawLines(cv::Mat& img, linesType lines, cv::Scalar color) {
	for (int i = 0; i < lines.size(); ++i) {
		cv::Point pt1 = cv::Point(lines[i][0], lines[i][1]);
		cv::Point pt2 = cv::Point(lines[i][2], lines[i][3]);
		cv::line(img, pt1, pt2, color, 20, 8);
	}
}

void averageLines(cv::Mat, linesType lines, double y_min, double y_max, linesType& output) {
	std::vector<hough_pts> left_lane, right_lane;
	for (int i = 0; i < lines.size(); ++i) {
		cv::Point2f pt1 = cv::Point2f(lines[i][0], lines[i][1]);
		cv::Point2f pt2 = cv::Point2f(lines[i][2], lines[i][3]);
		double m = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		double b = -m * pt1.x + pt1.y;
		//double norm = sqrt((pt2.x - pt1.x)*(pt2.x - pt1.x) + (pt2.y - pt1.y)*(pt2.y - pt1.y));
		double norm = 0;
		if (m < 0) { // left lane
			left_lane.push_back(hough_pts(m, b, norm));
		}
		if (m > 0) { // right lane
			right_lane.push_back(hough_pts(m, b, norm));
		}
	}

	double b_avg_left = 0.0, m_avg_left = 0.0, xmin_left, xmax_left;
	for (int i = 0; i < left_lane.size(); ++i) {
		b_avg_left += left_lane[i].b;
		m_avg_left += left_lane[i].m;
	}
	b_avg_left /= left_lane.size();
	m_avg_left /= left_lane.size();
	xmin_left = int((y_min - b_avg_left) / m_avg_left);
	xmax_left = int((y_max - b_avg_left) / m_avg_left);
	// left lane
	output.push_back(cv::Vec4i(xmin_left, y_min, xmax_left, y_max));

	double b_avg_right = 0.0, m_avg_right = 0.0, xmin_right, xmax_right;
	for (int i = 0; i < right_lane.size(); ++i) {
		b_avg_right += right_lane[i].b;
		m_avg_right += right_lane[i].m;
	}
	b_avg_right /= right_lane.size();
	m_avg_right /= right_lane.size();
	xmin_right = int((y_min - b_avg_right) / m_avg_right);
	xmax_right = int((y_max - b_avg_right) / m_avg_right);
	// right lane
	output.push_back(cv::Vec4i(xmin_right, y_min, xmax_right, y_max));

	// left lane: output[0]
	// right lane: output[1]
}

void bypassAngleFilter(linesType lines, double low_thres, double high_thres, linesType& output) {
	for (int i = 0; i < lines.size(); ++i) {
		double x1 = lines[i][0], y1 = lines[i][1];
		double x2 = lines[i][2], y2 = lines[i][3];
		double angle = abs(atan((y2 - y1) / (x2 - x1)) * 180 / PI);
		if (angle > low_thres && angle < high_thres) {
			output.push_back(cv::Vec4i(x1, y1, x2, y2));
		}
	}
}

//void pipeline(cv::Mat img, std::vector<cv::Point>vertices, double low_thres, double high_thres, cv::Mat& img_all_lines) {
void pipeline(cv::Mat img, cv::Mat mask, double low_thres, double high_thres, cv::Mat& img_all_lines) {
    cv::Mat gray;
	cv::Mat gray_blur;
	cv::Mat masked;	
	cv::Mat edges;	
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::bitwise_and(gray, gray, masked, mask);
	cv::Canny(masked, edges, 50, 180);
//	cv::imshow("edges", edges);
//	getROI(edges, vertices, masked);
	std::vector<cv::Vec4i>lines;
	cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 26, 5, 50);
//	drawLines(edges, lines, cv::Scalar(255, 0, 0));
//	cv::imshow("Lines image", edges);
	cv::Mat hlines_img = cv::Mat::zeros(img.size(), CV_8UC3);
	linesType filtered_lines;
	bypassAngleFilter(lines, low_thres, high_thres, filtered_lines);
	cv::Mat avg_img = cv::Mat::zeros(img.size(), CV_8UC3);
	linesType avg_hlines;
	averageLines(img, filtered_lines, int(img.rows * 0.65), img.rows, avg_hlines);
//	drawLines(hlines_img, avg_hlines, cv::Scalar(255, 0, 0));
	drawLines(avg_img, avg_hlines, cv::Scalar(255, 0, 0));
//	cv::imshow("avg_hlines", avg_img);
//	cv::fillConvexPoly(colorMask, lineVertices, cv::Scalar(0, 255, 0));
//	cv::imshow("colorMask", colorMask);
	cv::addWeighted(img, 1.0, avg_img, 1, 0.0, img_all_lines);
}

int main(int argc, char* argv[]) {
	std::string output =  "streetr.mp4";
//	cv::VideoCapture cap(dir_video + "test.mp4");
	  std::string str = argv[1];  	
    bool is_camera =
        str.size() == 1 && str[0] >= '0' && str[0] <= '9';
    cv::VideoCapture cap(0);
//    cv::VideoCapture cap(argv[1]);
    const int width = is_camera?640:cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = is_camera?320:cap.get(cv::CAP_PROP_FRAME_HEIGHT); 
//    cv::VideoCapture cap(is_camera?atoi(str.c_str()):argv[1]);
    if (is_camera) {
        LOG(INFO) << "Using camera";
//      int n = atoi(str.c_str());
//      width = 640 ;
//      height = 360;
    }


//	 cv::VideoWriter writer;
  LOG(INFO) << "width = " << width;
  LOG(INFO) << " height = " << height;
	const int fps = 25;
	cv::Size size = cv::Size(width, height);
	std::vector<cv::Point2i> vertices;
	vertices.push_back(cv::Point2i(size.width*0.15, size.height*0.88));
	vertices.push_back(cv::Point2i(size.width*0.38, int(size.height*0.65)));
	vertices.push_back(cv::Point2i(size.width - size.width*0.38, int(size.height*0.65)));
	vertices.push_back(cv::Point2i(size.width - size.width*0.15, size.height*0.88));
	double low_thres = 30.0;
	double high_thres = 80.0;
	cv::Mat frame;
	cv::Mat result;
//	writer.open(output, cv::CAP_OPENCV_MJPEG, fps, size);
// 	writer.open(output, cv::CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
//  判断是否创建成功	
//	if (!writer.isOpened()) {
//		cout << "could not creat video file" << endl;
//		return -1;
//	}	
//	Mat frame(size, CV_8UC3); //视频帧
	cv::Mat gray;
	cap >> frame;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	cv::Mat mask = cv::Mat::zeros(gray.size(), gray.type());
	if (gray.channels() == 1) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255));
		//		imshow("mask", mask);
	}
	else if (gray.channels() == 3) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));
	}
//	cv::imshow("mask",mask);
	while (1) {
		cap >> frame;
		if (frame.empty()) break;
		pipeline(frame, mask, low_thres, high_thres, result);
//		writer << result;
    	cv::imshow("roadline", result);
//	    cv::imshow("result", frame);
    if (cv::waitKey(1) == 27)
		  {
			cap.release();
//      writer.release();
			break;
		  }
	}
	return 0;
}


