#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
typedef std::vector<cv::Vec4i> linesType;
const double PI = 3.1415926535;
const	double low_thres = 30.0;
const	double high_thres = 80.0;
// m: Ð±ÂÊ, b: ½Ø¾à, norm: Ïß³¤
static cv::VideoWriter writer;
//static cv::Mat mask;
struct hough_pts {
	double m, b, norm;
	hough_pts(double m, double b, double norm) :
		m(m), b(b), norm(norm) {};
};

/*
 *   The color loops every 27 times,
 *    because each channel of RGB loop in sequence of "0, 127, 254"
 */
static cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
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
}

void drawLines(cv::Mat& img, linesType lines, cv::Scalar color) {
	for (int i = 0; i < lines.size(); ++i) {
		cv::Point pt1 = cv::Point(lines[i][0], lines[i][1]);
		cv::Point pt2 = cv::Point(lines[i][2], lines[i][3]);
		cv::line(img, pt1, pt2, color, 20, 8);
	}
}

static cv::Mat pipeline(cv::Mat &image) {
//    cv::putText(image, "Star", cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2);
    cv::Mat gray;
  	cv::Mat masked;	
  	cv::Mat edges;	
//  	cv::bitwise_and(img, img, masked, mask);
    image.copyTo(masked, mask);
	  cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
  	cv::Canny(gray, edges, 50, 180);
   	std::vector<cv::Vec4i>lines;
	  cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 26, 5, 50);
  	cv::Mat hlines_img = cv::Mat::zeros(image.size(), CV_8UC3);
	  linesType filtered_lines;
	  bypassAngleFilter(lines, low_thres, high_thres, filtered_lines);
  	linesType avg_hlines;
  	averageLines(image, filtered_lines, int(image.rows * 0.65), image.rows, avg_hlines);
    cv::Mat avg_img = cv::Mat::zeros(image.size(), CV_8UC3);
    drawLines(avg_img, avg_hlines, cv::Scalar(255, 0, 0));
//    drawLines(edges, lines, cv::Scalar(255, 0, 0));
	  cv::addWeighted(image, 1, avg_img, 1, 0.0, image);

    return image;
    }

static cv::Mat process_result(cv::Mat &image,
                              const xilinx::ai::SSDResult &result,
                              bool is_jpeg) {
     //auto x = 10;
     //auto y = 20;

  for (const auto bbox : result.bboxes) {
    int label = bbox.label;
    float xmin = bbox.x * image.cols;
    float ymin = bbox.y * image.rows;
    float xmax = xmin + bbox.width * image.cols;
    float ymax = ymin + bbox.height * image.rows;
    float confidence = bbox.score;
    if (xmax > image.cols)
      xmax = image.cols;
    if (ymax > image.rows)
      ymax = image.rows;
    LOG_IF(INFO, is_jpeg) << "RESULT: " << label << "\t" << xmin << "\t" << ymin
                          << "\t" << xmax << "\t" << ymax << "\t" << confidence
                          << "\n";

    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  getColor(label), 1, 1, 0);
//    writer << image;
//    cv::putText(image,"BOX", cv::Point(xmin,ymin), cv::FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2);
  }
//  image = pipeline(image); 
  return image;
}
