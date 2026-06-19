// Copyright Axelera AI, 2025
#include "AxOpenCVRender.hpp"
#include <chrono>
#include <iostream>
#include "AxStreamerUtils.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

static void
draw_bounding_boxes(const AxMetaBbox &bboxes, cv::Mat &buffer,
    const Ax::OpenCV::RenderOptions &options)
{
  for (auto i = size_t{}; i < bboxes.num_elements(); ++i) {
    auto box = bboxes.get_box_xyxy(i);
    if (options.render_labels) {
      auto label = std::string{};
      if (bboxes.has_class_id()) {
        const auto id = bboxes.class_id(i);
        label = id >= 0 && id < options.labels.size() ? options.labels[id] :
                                                        "cls=" + std::to_string(id);
        label += " ";
      }
      auto msg = label + std::to_string(int(bboxes.score(i) * 100)) + "%";
      cv::putText(buffer, msg, cv::Point(box.x1, box.y1 - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(224, 190, 24), 2);
    }
    if (options.render_bboxes) {
      cv::rectangle(buffer,
          cv::Rect(cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2)),
          cv::Scalar(224, 190, 24), 2);
    }
  }
}

void
Ax::OpenCV::render(const AxMetaObjDetection &detections, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(detections, buffer, options);
}


void
Ax::OpenCV::render(const AxMetaKptsDetection &detections, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(detections, buffer, options);
  auto kpts = detections.get_kpts();
  constexpr auto num_keypoints = 17;
  std::vector<cv::Point *> polylines;
  std::vector<int> npts;
  std::vector<cv::Point> points;
  for (auto i = size_t{}; i < detections.num_elements(); ++i) {
    if (options.render_keypoint_lines) {
      points.clear();
      npts.clear();
      polylines.clear();
      for (auto &&idx : options.keypoint_lines) {
        if (idx > -1) {
          points.emplace_back(kpts[idx].x, kpts[idx].y);
        }
      }
      auto *first_point = &points[0];
      auto *current_point = first_point;
      for (auto &&idx : options.keypoint_lines) {
        if (idx == -1) {
          polylines.push_back(first_point);
          npts.push_back(current_point - first_point);
          first_point = current_point;
        } else {
          ++current_point;
        }
      }
      cv::polylines(buffer, polylines.data(), npts.data(), polylines.size(),
          false, cv::Scalar(0, 0xff, 0));
    }
    if (options.render_keypoints) {
      for (auto k = 0; k != num_keypoints; ++k) {
        auto &kpt = kpts[k + i * num_keypoints];
        cv::rectangle(buffer, cv::Point(kpt.x - 1, kpt.y - 1),
            cv::Point(kpt.x + 1, kpt.y + 1), cv::Scalar(0, 0xff, 0));
      }
    }
  }
}

static box_xyxy
translate_image_space_rect(const auto &bbox, const box_xyxy &input_roi, int mask_width)
{
  // note this assumes a square mask shape, which will not be true for a rect model
  int input_w = input_roi.x2 - input_roi.x1;
  int input_h = input_roi.y2 - input_roi.x1;
  int longest_edge = std::max(input_w, input_h);
  float scale_factor = static_cast<float>(longest_edge) / mask_width;
  int xoffset = static_cast<int>((longest_edge - input_w) / 2.0);
  int yoffset = static_cast<int>((longest_edge - input_h) / 2.0);
  return {
    static_cast<int>(bbox.x1 * scale_factor - xoffset + input_roi.x1),
    static_cast<int>(bbox.y1 * scale_factor - yoffset + input_roi.y1),
    static_cast<int>(bbox.x2 * scale_factor - xoffset + input_roi.x1),
    static_cast<int>(bbox.y2 * scale_factor - yoffset + input_roi.y1),
  };
}

void
Ax::OpenCV::render(const AxMetaSegmentsDetection &segs, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(segs, buffer, options);
  if (!options.render_segments) {
    return;
  }
  const auto shape = segs.get_segments_shape();
  for (auto i = size_t{}; i < segs.num_elements(); ++i) {
    auto seg = segs.get_segment(i);
    const auto height = seg.y2 - seg.y1;
    const auto width = seg.x2 - seg.x1;
    assert(seg.map.size() == height * width);
    auto dest = translate_image_space_rect(
        seg, { 0, 0, buffer.cols, buffer.rows }, shape[1]);
    auto scale = static_cast<float>(dest.x2 - dest.x1) / width;
    auto fmask = cv::Mat(height, width, CV_32F, seg.map.data());
    for (auto y = dest.y1; y < dest.y2; ++y) {
      for (auto x = dest.x1; x < dest.x2; ++x) {
        const auto b = buffer.at<cv::Vec3b>(y, x)[0];
        const auto g = buffer.at<cv::Vec3b>(y, x)[1];
        const auto r = buffer.at<cv::Vec3b>(y, x)[2];
        const auto gray = (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
        const auto grayness
            = options.segments_in_grayscale ?
                  fmask.at<float>((y - dest.y1) / scale, (dest.x1 - x) / scale) :
                  0.0;
        buffer.at<cv::Vec3b>(y, x)[0] = (grayness * gray) + ((1.0f - grayness) * b);
        buffer.at<cv::Vec3b>(y, x)[1] = (grayness * gray) + ((1.0f - grayness) * g);
        buffer.at<cv::Vec3b>(y, x)[2] = (grayness * gray) + ((1.0f - grayness) * r);
      }
    }
  }
}

void
Ax::OpenCV::render(const AxMetaBase &detections, cv::Mat &buffer, const RenderOptions &options)
{
  const auto *bboxes = dynamic_cast<const AxMetaBbox *>(&detections);
  if (const auto *obj = dynamic_cast<const AxMetaObjDetection *>(&detections)) {
    render(*obj, buffer, options);
  } else if (const auto *kpts = dynamic_cast<const AxMetaKptsDetection *>(&detections)) {
    render(*kpts, buffer, options);
  } else if (const auto *segs = dynamic_cast<const AxMetaSegmentsDetection *>(&detections)) {
    render(*segs, buffer, options);
  }

  if (!options.render_submeta) {
    return;
  }
  const auto submeta_names = detections.submeta_names();
  for (auto &&submeta_name : submeta_names) {
    auto submetas = detections.get_submetas(submeta_name);
    for (auto &&[n, sub] : Ax::Internal::enumerate(submetas)) {
      if (!sub) {
        continue;
      }
      auto roi = bboxes ? bboxes->get_box_xyxy(n) :
                          BboxXyxy{ 0, 0, buffer.cols, buffer.rows };
      cv::Mat subbuffer(
          buffer, cv::Rect(roi.x1, roi.y1, roi.x2 - roi.x1, roi.y2 - roi.y1));
      render(*sub, subbuffer, options);
    }
  }
}

namespace
{
auto
micro_duration(high_resolution_clock::time_point start, high_resolution_clock::time_point now)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start);
}

class TimeKeeper
{
  public:
  explicit TimeKeeper(high_resolution_clock::time_point start)
      : start_(start),
        last_update_(start),
        last_render_(start)
  {
  }

  float fps() const
  {
    return fps_;
  }
  int frame_count() const
  {
    return count_;
  }
  float total_time() const
  {
    return total_time_;
  }
  bool should_render() const
  {
    return should_render_;
  }

  void update(high_resolution_clock::time_point now, const Ax::OpenCV::RenderOptions &options)
  {
    ++count_;
    if (micro_duration(last_update_, now).count() > update_rate_) {
      const auto frames = count_ - last_count_;
      last_count_ = count_;
      fps_ = static_cast<float>(frames) / (micro_duration(last_update_, now).count() / 1e6);
      last_update_ = now;
    }
    total_time_ = static_cast<double>(micro_duration(start_, now).count()) / 1e6;
    const auto interval = options.render_rate == 0 ? 0 : 1'000'000 / options.render_rate;
    if ((should_render_ = (micro_duration(last_render_, now).count() >= interval))) {
      last_render_ = now;
    }
  }

  private:
  const high_resolution_clock::time_point start_;
  high_resolution_clock::time_point last_update_;
  high_resolution_clock::time_point last_render_;
  int count_ = 0;
  int last_count_ = 0;
  double fps_ = 0.0;
  double total_time_ = 0.0;
  static constexpr int update_rate_ = 500'000;
  bool should_render_ = false;
};

class OpenCVDisplay : public Ax::OpenCV::Display
{
  public:
  explicit OpenCVDisplay(const std::string &name)
      : wndname(name)
  {
    cv::namedWindow(wndname, cv::WINDOW_AUTOSIZE);
    cv::setWindowProperty(wndname, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Executed 0 frames...\r" << std::flush;
  }

  void show(cv::Mat &image, const Ax::MetaMap &meta, AxVideoFormat format,
      const Ax::OpenCV::RenderOptions &options, int stream_id) override
  {
    (void) stream_id; // Unused in this implementation
    tk.update(high_resolution_clock::now(), options);

    if (tk.should_render()) {
      if (format == AxVideoFormat::RGB) {
        cv::cvtColor(image, converted, cv::COLOR_RGB2BGR);
      } else if (format == AxVideoFormat::I420) {
        cv::cvtColor(image, converted, cv::COLOR_YUV2BGR_I420);
      } else if (format == AxVideoFormat::NV12) {
        cv::cvtColor(image, converted, cv::COLOR_YUV2BGR_NV12);
      } else {
        converted = image;
      }
      for (auto &&[name, m] : meta) {
        Ax::OpenCV::render(*m, converted, options);
      }

      cv::imshow(wndname, converted);
      cv::waitKey(1);
    }
    std::cout << "Executed " << tk.frame_count() << " frames in "
              << tk.total_time() << "s : " << tk.fps() << "fps\r" << std::flush;
  }

  ~OpenCVDisplay() override
  {
    cv::destroyWindow(wndname);
    std::cout << std::endl;
  }

  private:
  TimeKeeper tk{ high_resolution_clock::now() };
  std::string wndname;
  cv::Mat converted;
};

class AnsiDisplay : public Ax::OpenCV::Display
{
  public:
  AnsiDisplay(std::ostream &os)
      : f(os)
  {
    const auto [term_cols, term_rows] = get_terminal_size();
    f << "\033[2J\033[H"; // Clear console and move cursor to top left
    f << std::string(term_rows - 2, '\n');
    f << std::fixed << std::setprecision(2);
    f << "Executed 0 frames...\r" << std::flush;
  }

  ~AnsiDisplay() override
  {
    f << "\033[0m" << std::endl; // Reset color
  }

  void show(cv::Mat &image, const Ax::MetaMap &meta, AxVideoFormat format,
      const Ax::OpenCV::RenderOptions &options, int stream_id) override
  {
    (void) stream_id; // Unused in this implementation
    const auto now = high_resolution_clock::now();
    tk.update(now, options);

    if (tk.should_render()) {
      for (auto &&[name, m] : meta) {
        Ax::OpenCV::render(*m, image, options);
      }
      const auto [term_cols, term_rows] = get_terminal_size();
      show_as_ansi(image, 0, 2, term_cols, term_rows - 2, format);
      std::cout << "Executed " << tk.frame_count() << " frames in "
                << tk.total_time() << "s : " << tk.fps() << "fps\r" << std::flush;
    }
  }

  private:
  std::pair<int, int> get_terminal_size()
  {
    struct winsize w;
    ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return { w.ws_col, w.ws_row };
  }

  void show_as_ansi(const cv::Mat &image, int posx, int posy, int cols,
      int rows, AxVideoFormat format)
  {
    cv::resize(image, resized, { cols, rows * 2 });
    char buf[64];
    f << "\033[" << posy << ";" << posx << "H";
    const auto R = (format == AxVideoFormat::RGB) ? 0 : 2;
    const auto G = (format == AxVideoFormat::RGB) ? 1 : 0;
    const auto B = (format == AxVideoFormat::RGB) ? 2 : 1;
    for (int y = 0; y < (resized.rows & ~1); y += 2) {
      for (int x = 0; x < resized.cols; ++x) {
        const auto &top = resized.at<cv::Vec3b>(y, x);
        const auto &bot = resized.at<cv::Vec3b>(y + 1, x);
        auto sz = std::snprintf(buf, std::size(buf), "\033[48;2;%d;%d;%dm\033[38;2;%d;%d;%dm\u2584",
            top[R], top[G], top[B], bot[R], bot[G], bot[B]);
        f.write(buf, sz);
      }
      f << "\n";
    }
    f << "\033[0m\n" << std::flush; // Reset color
    f << "\033[" << 0 << ";" << 0 << "H";
  }

  TimeKeeper tk{ high_resolution_clock::now() };
  std::ostream &f;
  cv::Mat resized;
};

class NullDisplay : public Ax::OpenCV::Display
{
  public:
  NullDisplay()
  {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Executed 0 frames...\r" << std::flush;
  }

  void show(cv::Mat &image, const Ax::MetaMap &meta, AxVideoFormat format,
      const Ax::OpenCV::RenderOptions &options, int stream_id) override
  {
    (void) image; // Unused in this implementation
    (void) meta; // Unused in this implementation
    (void) format; // Unused in this implementation
    (void) stream_id; // Unused in this implementation
    tk.update(high_resolution_clock::now(), options);
    std::cout << "Executed " << tk.frame_count() << " frames in "
              << tk.total_time() << "s : " << tk.fps() << "fps\r" << std::flush;
  }

  ~NullDisplay() override
  {
    std::cout << std::endl;
  }

  private:
  TimeKeeper tk{ high_resolution_clock::now() };
};
} // namespace

std::unique_ptr<Ax::OpenCV::Display>
Ax::OpenCV::create_cv_display(const std::string &name)
{
  return std::make_unique<OpenCVDisplay>(name);
}

std::unique_ptr<Ax::OpenCV::Display>
Ax::OpenCV::create_ansi_display()
{
  return std::make_unique<AnsiDisplay>(std::cout);
}

std::unique_ptr<Ax::OpenCV::Display>
Ax::OpenCV::create_null_display()
{
  return std::make_unique<NullDisplay>();
}
