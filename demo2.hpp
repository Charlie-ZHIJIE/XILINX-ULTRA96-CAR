#pragma once
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <signal.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <xilinx/ai/bounded_queue.hpp>
#include <xilinx/ai/env_config.hpp>
#include <opencv2/videoio.hpp>
#include<typeinfo>
#ifndef USE_DRM
#define USE_DRM 0
#endif
#if USE_DRM
#include "./dpdrm.hpp"
#endif
using namespace std;
static cv::Mat mask;
DEF_ENV_PARAM(DEBUG_DEMO, "1");
// set the layout
inline std::vector<cv::Rect> &gui_layout() {
  static std::vector<cv::Rect> rects;
  return rects;
}
// set the wallpaper
inline cv::Mat &gui_background() {
  static cv::Mat img;
  return img;
}

namespace xilinx {
namespace ai {
// Read a video without doing anything
struct VideoByPass {
public:
  int run(const cv::Mat &input_image) { return 0; }
};

// Do nothing after after excuting
inline cv::Mat process_none(cv::Mat image, int fake_result, bool is_jpeg) {
  return image;
}

// A struct that can storage data and info for each frame
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  float max_fps;
  float fps;
};

using queue_t = xilinx::ai::BoundedQueue<FrameInfo>;
struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread *> &all_threads() {
    static std::vector<MyThread *> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }
  static void stop_all() {
    for (auto &th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto &th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    for (auto &th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread *me) { return me->main(); }
  void main() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is started";
    while (!stop_) {
    
    if (cv::waitKey(1) == 27)
		{
//			cap.release();
			break;
		}
   
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;

  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    all_threads().push_back(this);
  }

  virtual ~MyThread() { //
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is starting";
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "waiting for [" << name() << "] ended";
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};

// std::vector<MyThread *> MyThread::all_threads_;
struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string &video_file, queue_t *queue)
      : MyThread{}, channel_id_{channel_id},
        video_file_{video_file}, frame_id_{0}, video_stream_{}, queue_{queue} {

    open_stream();
    auto &cap = *video_stream_.get();
    if (is_camera_) {
      cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
      cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    }
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    auto &cap = *video_stream_.get();
    cv::Mat image;
    cap >> image;
    auto video_ended = image.empty();
    if (video_ended) {
      // loop the video
      open_stream();
      return 0;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    if (queue_->size() > 0 && is_camera_ == true) {
      return 0;
    }
    while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image},
                         std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
  }

  void open_stream() {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
     LOG(INFO) << "is_camera_ = " << is_camera_;             
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
                   : new cv::VideoCapture(video_file_));
    if (!video_stream_->isOpened()) {
      LOG(ERROR) << "cannot open file " << video_file_;
      stop();
    }
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  std::unique_ptr<cv::VideoCapture> video_stream_;
  queue_t *queue_;
  bool is_camera_;
};

struct GuiThread : public MyThread {
  static std::shared_ptr<GuiThread> instance() {
    static std::weak_ptr<GuiThread> the_instance;
    std::shared_ptr<GuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<GuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
#if USE_DRM
    xilinx::ai::imshow_open();
    cv::Mat img = gui_background();
    imshow_set_background(img);
#endif
    return ret;
  }

  GuiThread()
      : MyThread{},
        queue_{
            new queue_t{10} // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0} {}
  virtual ~GuiThread() { //
#if USE_DRM
    xilinx::ai::imshow_close();
#endif
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    while (!queue_->empty()) {
      queue_->pop(frame_info);
      frames_[frame_info.channel_id].frame_info = frame_info;
      frames_[frame_info.channel_id].dirty = true;
    }
  }
  virtual int run() override {
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 10) {
        // inactive for 5 second, stop
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
        return 1;
      } else {
        return 0;
      }
    }
    inactive_counter_ = 0;
    frames_[frame_info.channel_id].frame_info = frame_info;
    frames_[frame_info.channel_id].dirty = true;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
    clean_up_queue();
#if USE_DRM
    bool all_dirty = true;
    for (auto &f : frames_) {
      all_dirty = all_dirty && f.second.dirty;
    }
    if (!all_dirty) {
      // only show frames until all channels are dirty
      return 0;
    }
    auto width = modeset_get_fb_width();
    auto height = modeset_get_fb_height();
    auto screen_size = cv::Size{width, height};
    auto sizes = std::vector<cv::Size>(frames_.size());
    std::transform(frames_.begin(), frames_.end(), sizes.begin(),
                   [](const decltype(frames_)::value_type &a) {
                     return a.second.frame_info.mat.size();
                   });
    std::vector<cv::Rect> rects;
    rects = gui_layout();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "rects size is  " << rects.size();

    for (const auto &rect : rects) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "screen " << screen_size << "; r = " << rect;
      if ((rect.x + rect.width > width) || (rect.y + rect.height > height) ||
          (rect.x + rect.width < 1) || (rect.y + rect.height < 1)) {
        LOG(FATAL) << "out of boundary";
      }
    }
    int c = 0;
    for (auto &f : frames_) {
      xilinx::ai::imshow(rects[c], f.second.frame_info.mat);
      f.second.dirty = false;
      c++;
    }
    xilinx::ai::imshow_update();
#else
    bool any_dirty = false;
    for (auto &f : frames_) {
      if (f.second.dirty) {
        cv::imshow(std::string{"CH-"} +
                       std::to_string(f.second.frame_info.channel_id),
                   f.second.frame_info.mat);
        f.second.dirty = false;
        any_dirty = true;
      }
    }
    if (any_dirty) {
      auto key = cv::waitKey(1);
      if (key == 27) {
        return 1;
      }
    }
#endif
    clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t *getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_;
  struct FrameCache {
    bool dirty;
    FrameInfo frame_info;
  };
  std::map<int, FrameCache> frames_;
};

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual cv::Mat run(cv::Mat &input) = 0;
};

// Execute each lib run function and processor your implement
template <typename dpu_model_type_t, typename ProcessResult>
struct DpuFilter : public Filter {

  DpuFilter(std::unique_ptr<dpu_model_type_t> &&dpu_model,
            const ProcessResult &processor)
      : Filter{}, dpu_model_{std::move(dpu_model)}, processor_{processor} {}
  virtual ~DpuFilter() {}
  cv::Mat run(cv::Mat &image) override {
    auto result = dpu_model_->run(image);
    return processor_(image, result, false);
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  const ProcessResult &processor_;
};
template <typename FactoryMethod, typename ProcessResult>
std::unique_ptr<Filter> create_dpu_filter(const FactoryMethod &factory_method,
                                          const ProcessResult &process_result) {
  using dpu_model_type_t = typename decltype(factory_method())::element_type;
  return std::unique_ptr<Filter>(new DpuFilter<dpu_model_type_t, ProcessResult>(
      factory_method(), process_result));
}

// Execute dpu filter
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<Filter> &&filter, queue_t *queue_in,
            queue_t *queue_out, const std::string &suffix)
      : MyThread{}, filter_{std::move(filter)}, queue_in_{queue_in},
        queue_out_{queue_out}, suffix_{suffix} {}
  virtual ~DpuThread() {}

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }
    if (filter_) {
      frame.mat = filter_->run(frame.mat);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t *queue_in_;
  queue_t *queue_out_;
  std::string suffix_;
};

// Implement sorting thread
struct SortingThread : public MyThread {
  SortingThread(queue_t *queue_in, queue_t *queue_out,
                const std::string &suffix)
      : MyThread{}, queue_in_{queue_in}, queue_out_{queue_out}, frame_id_{0},
        suffix_{suffix}, fps_{0.0f}, max_fps_{0.0f} {}
  virtual ~SortingThread() {}
  virtual int run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond =
        std::function<bool(const FrameInfo &)>{[frame_id](const FrameInfo &f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    if (!queue_in_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto now = std::chrono::steady_clock::now();
    float fps = -1.0f;
    long duration = 0;
    if (!points_.empty()) {
      auto end = points_.back();
      duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;
      auto x = 10;
      auto y = 20;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
 //     cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
 //                 cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
 //                 cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) { // sliding window for 2 seconds.
      points_.pop_back();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t *queue_in_;
  queue_t *queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
};
inline void usage_video(const char *progname) {
  std::cout << "usage: " << progname << "      -t <num_of_threads>\n"
            << "      <video file name>\n"
            << std::endl;
  return;
}
/*
  global command line options
 */
static std::vector<int> g_num_of_threads;
static std::vector<std::string> g_avi_file;

inline void parse_opt(int argc, char *argv[]) {
  int opt = 0;
  optind = 1;
  while ((opt = getopt(argc, argv, "c:t:")) != -1) {
    switch (opt) {
    case 't':
      g_num_of_threads.emplace_back(std::stoi(optarg));
      break;
    case 'c': // how many channels
      break;  // do nothing. parse it in outside logic.
    default:
      usage_video(argv[0]);
      exit(1);
    }
  }
  for (int i = optind; i < argc; ++i) {
    g_avi_file.push_back(std::string(argv[i]));
  }
  if (g_avi_file.empty()) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_num_of_threads.empty()) {
    // by default, all channels has at least one thread
    g_num_of_threads.emplace_back(1);
  }
  return;
}
//void main_for_video_road() 
//    { cout << "show space1" << endl; } 
// Entrance of single channel video demo
template <typename FactoryMethod, typename ProcessResult>
int main_for_video_demo(int argc, char *argv[],
                        const FactoryMethod &factory_method,
                        const ProcessResult &process_result) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  {
    cv::VideoCapture video_cap(g_avi_file[0]);
    std::string file_name(g_avi_file[0]);
    bool is_camera =
        file_name.size() == 1 && file_name[0] >= '0' && file_name[0] <= '9';
    if (is_camera) {
      gui_layout() = {{0, 0, 640, 480}};
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "Using camera";
  
      video_cap.set(cv::CAP_PROP_FRAME_WIDTH,640);
      video_cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "Using file";
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "width " << video_cap.get(3);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "height " << video_cap.get(4);
      gui_layout() = {{0, 0, (int)video_cap.get(3), (int)video_cap.get(4)}};
    }
    video_cap.release();
    auto channel_id = 0;
    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    auto decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{channel_id, g_avi_file[0], decode_queue.get()});
    auto dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    auto sorting_queue =
        std::unique_ptr<queue_t>(new queue_t(5 * g_num_of_threads[0]));
    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    for (int i = 0; i < g_num_of_threads[0]; ++i) {
      dpu_thread.emplace_back(new DpuThread(
          create_dpu_filter(factory_method, process_result), decode_queue.get(),
          sorting_queue.get(), std::to_string(i)));
    }
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));
    // start everything
    MyThread::start_all();
    gui_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

// A class can create a video channel
struct Channel {
  Channel(size_t ch, const std::string &avi_file,
          const std::function<std::unique_ptr<Filter>()> &filter,
          int n_of_threads) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "create channel " << ch << " for " << avi_file;
    auto channel_id = ch;
    decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)channel_id, avi_file, decode_queue.get()});
    dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(5 * n_of_threads));
    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix =
          avi_file + "-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
};

// Entrance of multi-channel video demo
inline int main_for_video_demo_multiple_channel(
    int argc, char *argv[],
    const std::vector<std::function<std::unique_ptr<Filter>()>> &filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  auto gui_thread = GuiThread::instance();
  std::vector<Channel> channels;
  channels.reserve(filters.size());
  for (auto ch = 0u; ch < filters.size(); ++ch) {
    channels.emplace_back(ch, g_avi_file[ch % g_avi_file.size()], filters[ch],
                          g_num_of_threads[ch % g_num_of_threads.size()]);
  }
  // start everything
  MyThread::start_all();
  gui_thread->wait();
  MyThread::stop_all();
  MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

static inline void usage_jpeg(const char *progname) {
  std::cout << "usage : " << progname << " <img_url> [<img_url> ...]"
            << std::endl;
}

// Entrance of jpeg demo
template <typename FactoryMethod, typename ProcessResult>
int main_for_jpeg_demo(int argc, char *argv[],
                       const FactoryMethod &factory_method,
                       const ProcessResult &process_result) {
  if (argc <= 1) {
    usage_jpeg(argv[0]);
    exit(1);
  }
  auto model = factory_method();
  for (int i = 1; i < argc; ++i) {
    auto image_file_name = std::string{argv[i]};
    auto image = cv::imread(image_file_name);
    if (image.empty()) {
      LOG(FATAL) << "cannot load " << image_file_name << std::endl;
      abort();
    }
    auto result = model->run(image);
    image = process_result(image, result, true);
    auto out_file =
        image_file_name.substr(0, image_file_name.size() - 4) + "_result.jpg";
    cv::imwrite(out_file, image);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "result image write to " << out_file;
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}
} // namespace ai
} // namespace xilinx
