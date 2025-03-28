#include <stdexcept>

#include "MultiObjTracker.hpp"
#include "TrackerFactory.h"

namespace ax
{
std::string
to_string(TrackState &state)
{
  switch (state) {
    case kUndefined:
      return "Undefined";
    case kNew:
      return "New";
    case kTracked:
      return "Tracked";
    case kLost:
      return "Lost";
    case kRemoved:
      return "Removed";
    default:
      return "Unknown State";
  }
}
} // namespace ax

// Helper function to get a parameter value or a default value if not found
template <typename T>
T
GetParamOrDefault(const TrackerParams &params, const std::string &key, T defaultValue)
{
  auto it = params.find(key);
  if (it != params.end() && std::holds_alternative<T>(it->second)) {
    return std::get<T>(it->second);
  }
  return defaultValue;
}

std::unique_ptr<ax::MultiObjTracker>
CreateMultiObjTracker(const std::string &tracker_type_str, const TrackerParams &params)
{
  static const std::unordered_map<std::string, std::function<std::unique_ptr<ax::MultiObjTracker>(const TrackerParams &)>> tracker_factory = {
    { "scalarmot",
        [](const TrackerParams &p) {
          return std::make_unique<ScalarMOTWrapper>(p);
        } },
    { "sort",
        [](const TrackerParams &p) { return std::make_unique<SORTWrapper>(p); } },
#ifdef HAVE_BYTETRACK
    { "bytetrack",
        [](const TrackerParams &p) {
          return std::make_unique<BytetrackWrapper>(p);
        } },
#endif
#ifdef HAVE_OC_SORT
    { "oc-sort",
        [](const TrackerParams &p) { return std::make_unique<OCSortWrapper>(p); } },
    { "oc_sort",
        [](const TrackerParams &p) { return std::make_unique<OCSortWrapper>(p); } },
#endif
  };

  std::string type_lower = tracker_type_str;
  std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);
  auto it = tracker_factory.find(type_lower);
  if (it != tracker_factory.end()) {
    return it->second(params);
  }

  throw std::runtime_error("Unknown tracker type: " + tracker_type_str);
}

//************** Wrappers of Axelera Multiple Object Trackers ***********
ScalarMOTWrapper::ScalarMOTWrapper(const TrackerParams &params)
    : tracker_(GetParamOrDefault<int>(params, "maxLostFrames", 30))
{
}

const std::vector<ax::TrackedObject>
ScalarMOTWrapper::Update(const std::vector<ax::ObservedObject> &detections)
{
  std::vector<axtracker::BboxXyxyRelative> inputs;
  for (const auto &det : detections) {
    axtracker::BboxXyxyRelative bbox;
    bbox.x1 = det.bbox.x1;
    bbox.y1 = det.bbox.y1;
    bbox.x2 = det.bbox.x2;
    bbox.y2 = det.bbox.y2;
    bbox.class_id = det.class_id;
    bbox.score = det.score;
    inputs.push_back(bbox);
  }
  tracker_.update(inputs);

  std::vector<ax::TrackedObject> objects;
  const auto trks = tracker_.getTrackers();
  for (const auto &trk : trks) {
    auto result = trk.get_state();
    ax::TrackedObject obj(result.x1, result.y1, result.x2, result.y2,
        trk.getTrackId(), trk.getClassId());
    objects.push_back(obj);
  }

  return objects;
}

SORTWrapper::SORTWrapper(const TrackerParams &params)
    : tracker_(GetParamOrDefault<int>(params, "maxAge", 30),
        GetParamOrDefault<int>(params, "minHits", 3),
        GetParamOrDefault<float>(params, "iouThreshold", 0.3))
{
}

const std::vector<ax::TrackedObject>
SORTWrapper::Update(const std::vector<ax::ObservedObject> &detections)
{
  std::vector<axtracker::BboxXyxyRelative> inputs;
  for (const auto &det : detections) {
    axtracker::BboxXyxyRelative bbox;
    bbox.x1 = det.bbox.x1;
    bbox.y1 = det.bbox.y1;
    bbox.x2 = det.bbox.x2;
    bbox.y2 = det.bbox.y2;
    bbox.class_id = det.class_id;
    bbox.score = det.score;
    inputs.push_back(bbox);
  }
  tracker_.update(inputs);

  std::vector<ax::TrackedObject> objects;
  const auto &trks = tracker_.getTrackers();
  for (const auto &trk : trks) {
    auto result = trk.get_state();
    ax::TrackedObject obj(result.x1, result.y1, result.x2, result.y2,
        trk.getTrackId(), trk.getClassId());
    objects.push_back(obj);
  }
  return objects;
}

// unlink axtracker

//************** Wrappers of Third-party Multiple Object Trackers ***********
#ifdef HAVE_BYTETRACK

ax::TrackState
MapSTrackTrackState(int state)
{
  using namespace ax;
  switch (state) {
    case 0: // STrack's 'New'
      return kNew;
    case 1: // STrack's 'Tracked'
      return kTracked;
    case 2: // STrack's 'Lost'
      return kLost;
    case 3: // STrack's 'Removed'
      return kRemoved;
    default:
      return kUndefined;
  }
}

BytetrackWrapper::BytetrackWrapper(const TrackerParams &params)
    : tracker_(GetParamOrDefault<int>(params, "frame_rate", 30),
        GetParamOrDefault<int>(params, "track_buffer", 30))
{
}

const std::vector<ax::TrackedObject>
BytetrackWrapper::Update(const std::vector<ax::ObservedObject> &detections)
{
  std::vector<Object> inputs;
  std::vector<ax::TrackedObject> outputs;
  for (const auto &det : detections) {
    Object obj;
    obj.rect = cv::Rect_<float>(det.bbox.x1, det.bbox.y1,
        det.bbox.x2 - det.bbox.x1, det.bbox.y2 - det.bbox.y1);
    obj.label = det.class_id;
    obj.prob = det.score;
    inputs.push_back(obj);
  }
  vector<STrack> output_stracks = tracker_.update(inputs);

  for (const auto &trk : output_stracks) {
    ax::TrackedObject obj(trk.tlbr[0], trk.tlbr[1], trk.tlbr[2], trk.tlbr[3],
        trk.track_id, trk.label, trk.score, MapSTrackTrackState(trk.state));
    outputs.push_back(obj);
  }

  return outputs;
}
#endif

#ifdef HAVE_OC_SORT
Eigen::Matrix<float, Eigen::Dynamic, 6>
Vector2Matrix(const std::vector<std::array<float, 6>> &data)
{
  Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), 6);
  for (int i = 0; i < data.size(); ++i) {
    for (int j = 0; j < 6; ++j) {
      matrix(i, j) = data[i][j];
    }
  }
  return matrix;
}


OCSortWrapper::OCSortWrapper(const TrackerParams &params)
    : tracker_(GetParamOrDefault<float>(params, "det_thresh", 0),
        GetParamOrDefault<int>(params, "max_age", 50),
        GetParamOrDefault<int>(params, "min_hits", 1),
        GetParamOrDefault<float>(params, "iou_threshold", 0.22136877277096445),
        GetParamOrDefault<int>(params, "delta", 1),
        GetParamOrDefault<std::string>(params, "asso_func", "giou"),
        GetParamOrDefault<float>(params, "inertia", 0.3941737016672115),
        GetParamOrDefault<bool>(params, "use_byte", true),
        // default as 0 for measurement which never reset id; 999 for demo
        GetParamOrDefault<int>(params, "max_id", 999))
{
}

const std::vector<ax::TrackedObject>
OCSortWrapper::Update(const std::vector<ax::ObservedObject> &detections)
{
  std::vector<ax::TrackedObject> outputs;
  std::vector<std::array<float, 6>> inputs;
  inputs.reserve(detections.size());

  for (const auto &det : detections) {
    inputs.emplace_back(std::array<float, 6>{ det.bbox.x1, det.bbox.y1,
        det.bbox.x2, det.bbox.y2, det.score, static_cast<float>(det.class_id) });
  }
  std::vector<Eigen::RowVectorXf> res = tracker_.update(Vector2Matrix(inputs));
  outputs.reserve(res.size()); // Reserve memory based on the expected size

  for (const auto &det : res) {
    if (det.size() != 8) {
      throw std::runtime_error("Invalid output from OC-SORT");
    }
    int class_id = static_cast<int>(det[5]);
    int latest_detection_id = static_cast<int>(det[7]);
    outputs.emplace_back(
        det[0], det[1], det[2], det[3], det[4], class_id, det[6], ax::kTracked);
    outputs.back().latest_detection_id = latest_detection_id;
  }
  return outputs;
}
#endif
