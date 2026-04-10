// Copyright Axelera AI, 2024
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
#ifdef HAVE_TRACKTRACK
    { "tracktrack",
        [](const TrackerParams &p) {
          return std::make_unique<TrackTrackWrapper>(p);
        } },
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
ScalarMOTWrapper::Update(const std::vector<ax::ObservedObject> &detections,
    const std::vector<std::vector<float>> &embeddings,
    const std::optional<Eigen::Matrix<float, 2, 3>> &transform)
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
          GetParamOrDefault<float>(params, "iouThreshold", 0.3)),
      min_hits_(GetParamOrDefault<int>(params, "minHits", 3))
{
}

const std::vector<ax::TrackedObject>
SORTWrapper::Update(const std::vector<ax::ObservedObject> &detections,
    const std::vector<std::vector<float>> &embeddings,
    const std::optional<Eigen::Matrix<float, 2, 3>> &transform)
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

    // Determine track state based on SORT algorithm logic:
    // - New: hit_streak < min_hits (unconfirmed track)
    // - Tracked: hit_streak >= min_hits and time_since_update == 0
    // - Lost: time_since_update > 0 (not matched in current frame)
    ax::TrackState state;
    if (trk.time_since_update > 0) {
      state = ax::kLost;
    } else if (trk.hit_streak < min_hits_) {
      state = ax::kNew;
    } else {
      state = ax::kTracked;
    }

    ax::TrackedObject obj(result.x1, result.y1, result.x2, result.y2,
        trk.getTrackId(), trk.getClassId(), result.score, state);
    obj.latest_detection_id = trk.getLatestDetectionId();
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
          GetParamOrDefault<int>(params, "track_buffer", 30)),
      return_all_states_(GetParamOrDefault<bool>(params, "return_all_states", false))
{
}

const std::vector<ax::TrackedObject>
BytetrackWrapper::Update(const std::vector<ax::ObservedObject> &detections,
    const std::vector<std::vector<float>> &embeddings,
    const std::optional<Eigen::Matrix<float, 2, 3>> &transform)
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

  if (!return_all_states_) {
    // Default mode: return only active tracks (backward compatible)
    for (const auto &trk : output_stracks) {
      ax::TrackedObject obj(trk.tlbr[0], trk.tlbr[1], trk.tlbr[2], trk.tlbr[3],
          trk.track_id, trk.label, trk.score, MapSTrackTrackState(trk.state));
      obj.latest_detection_id = trk.latest_det_id;
      outputs.push_back(obj);
    }
  } else {
    // Enhanced mode: return all track states (tracked, lost, removed)

    // Helper lambda to convert STrack to TrackedObject
    auto strack_to_obj = [](const STrack &trk) -> ax::TrackedObject {
      ax::TrackedObject obj(trk.tlbr[0], trk.tlbr[1], trk.tlbr[2], trk.tlbr[3],
          trk.track_id, trk.label, trk.score, MapSTrackTrackState(trk.state));
      obj.latest_detection_id = trk.latest_det_id;
      return obj;
    };

    // Add all tracked tracks
    for (const auto &trk : tracker_.tracked_stracks) {
      outputs.push_back(strack_to_obj(trk));
    }

    // Add all lost tracks
    for (const auto &trk : tracker_.lost_stracks) {
      outputs.push_back(strack_to_obj(trk));
    }

    // Deduplication: track removed tracks to avoid emitting duplicates
    std::unordered_set<int> current_removed_ids;
    for (const auto &trk : tracker_.removed_stracks) {
      current_removed_ids.insert(trk.track_id);
    }

    // Emit only newly removed tracks (set difference: current - previous)
    for (const auto &trk : tracker_.removed_stracks) {
      if (previous_removed_ids_.find(trk.track_id) == previous_removed_ids_.end()) {
        outputs.push_back(strack_to_obj(trk));
      }
    }

    // Update previous_removed_ids_ for next iteration
    previous_removed_ids_ = std::move(current_removed_ids);
  }

  return outputs;
}
#endif

#ifdef HAVE_OC_SORT
template <int Cols>
Eigen::Matrix<float, Eigen::Dynamic, Cols>
VectorOfArrays2Matrix(const std::vector<std::array<float, Cols>> &data)
{
  Eigen::Matrix<float, Eigen::Dynamic, Cols> matrix(data.size(), Cols);
  for (int i = 0; i < data.size(); ++i) {
    for (int j = 0; j < Cols; ++j) {
      matrix(i, j) = data[i][j];
    }
  }
  return matrix;
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
VectorOfVectors2Matrix(const std::vector<std::vector<float>> &data)
{
  if (data.empty()) {
    return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();
  }
  int rows = data.size();
  int cols = data[0].size();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix(rows, cols);
  for (int i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      throw std::runtime_error("All rows must have the same number of columns");
    }
    for (int j = 0; j < cols; ++j) {
      matrix(i, j) = data[i][j];
    }
  }
  return matrix;
}

OCSortWrapper::OCSortWrapper(const TrackerParams &params)
    : tracker_(GetParamOrDefault<float>(params, "det_thresh", 0),
          GetParamOrDefault<int>(params, "max_age", 30),
          GetParamOrDefault<int>(params, "min_hits", 3),
          GetParamOrDefault<float>(params, "iou_threshold", 0.3),
          GetParamOrDefault<int>(params, "delta", 3),
          GetParamOrDefault<float>(params, "inertia", 0.2),
          GetParamOrDefault<float>(params, "w_assoc_emb", 0.75),
          GetParamOrDefault<float>(params, "alpha_fixed_emb", 0.95),
          // default as 0 for measurement which never reset id; 999 for demo
          GetParamOrDefault<int>(params, "max_id", 999),
          // Deep-OC-SORT parameters
          !GetParamOrDefault<bool>(params, "aw_enabled", false),
          GetParamOrDefault<float>(params, "aw_param", 0.5),
          !GetParamOrDefault<bool>(params, "cmc_enabled", false),
          GetParamOrDefault<bool>(params, "enable_id_recovery", false),
          GetParamOrDefault<int>(params, "img_width", 0),
          GetParamOrDefault<int>(params, "img_height", 0),
          GetParamOrDefault<int>(params, "rec_image_rect_margin", 20),
          GetParamOrDefault<int>(params, "rec_track_min_time_since_update_at_boundary", 6),
          GetParamOrDefault<int>(params, "rec_track_min_time_since_update_inside", 300),
          GetParamOrDefault<int>(params, "rec_track_min_age", 30),
          GetParamOrDefault<float>(params, "rec_track_merge_lap_thresh", 0.09f),
          GetParamOrDefault<int>(params, "rec_track_memory_capacity", 1000),
          GetParamOrDefault<int>(params, "rec_track_memory_max_age", 54000)), // 30min at 30fps
      return_all_states_(GetParamOrDefault<bool>(params, "return_all_states", false))
{
}

const std::vector<ax::TrackedObject>
OCSortWrapper::Update(const std::vector<ax::ObservedObject> &detections,
    const std::vector<std::vector<float>> &embeddings,
    const std::optional<Eigen::Matrix<float, 2, 3>> &transform)
{
  std::vector<ax::TrackedObject> outputs;
  std::vector<std::array<float, 6>> inputs;
  inputs.reserve(detections.size());

  for (const auto &det : detections) {
    inputs.emplace_back(std::array<float, 6>{ det.bbox.x1, det.bbox.y1,
        det.bbox.x2, det.bbox.y2, det.score, static_cast<float>(det.class_id) });
  }
  // Use CMC transform directly - no conversion needed!
  Eigen::Matrix<float, 2, 3> cmc_transform = Eigen::Matrix<float, 2, 3>::Identity();
  if (transform.has_value()) {
    cmc_transform = transform.value();
  }

  std::vector<Eigen::RowVectorXf> res = tracker_.update(VectorOfArrays2Matrix<6>(inputs),
      VectorOfVectors2Matrix(embeddings), cmc_transform);

  if (!return_all_states_) {
    // Default mode: return only active tracks from update() output (backward compatible)
    outputs.reserve(res.size());
    for (const auto &det : res) {
      if (det.size() != 8) {
        throw std::runtime_error("Invalid output from OC-SORT");
      }
      int class_id = static_cast<int>(det[5]);
      int latest_detection_id = static_cast<int>(det[7]);
      outputs.emplace_back(det[0], det[1], det[2], det[3], det[4], class_id,
          det[6], ax::kTracked);
      outputs.back().latest_detection_id = latest_detection_id;
    }
  } else {
    // Enhanced mode: return all track states by inspecting internal tracker state

    // Build a map from track_id to tracker for quick lookup
    std::unordered_map<int, const ocsort::KalmanBoxTracker *> id_to_tracker;

    // Add active trackers
    for (const auto &trk : tracker_.active_trackers) {
      id_to_tracker[trk.id] = &trk;
    }

    // Add not_active trackers
    for (const auto &trk : tracker_.not_active_trackers) {
      id_to_tracker[trk.id] = &trk;
    }

    // Add lost trackers
    for (const auto &trk : tracker_.lost_trackers) {
      id_to_tracker[trk.id] = &trk;
    }

    // Helper lambda to infer state from KalmanBoxTracker fields
    auto infer_state = [](const ocsort::KalmanBoxTracker &trk) -> ax::TrackState {
      // State inference logic for OC-SORT:
      // - New: hit_streak < min_hits (unconfirmed tracks)
      // - Tracked: hit_streak >= min_hits and time_since_update == 0
      // - Lost: time_since_update > 0 (not matched in current frame)
      // - Removed: handled separately (not in active/not_active/lost lists)

      if (trk.time_since_update > 0) {
        return ax::kLost;
      } else if (trk.hit_streak < 3) { // Using default min_hits=3
        return ax::kNew;
      } else {
        return ax::kTracked;
      }
    };

    // Helper lambda to convert KalmanBoxTracker to TrackedObject
    auto tracker_to_obj
        = [&infer_state](const ocsort::KalmanBoxTracker &trk) -> ax::TrackedObject {
      Eigen::VectorXf state = const_cast<ocsort::KalmanBoxTracker &>(trk).get_state();
      // state is [x1, y1, x2, y2, score]
      ax::TrackedObject obj(state[0], state[1], state[2], state[3], trk.id,
          trk.cls, trk.conf, infer_state(trk));
      obj.latest_detection_id = trk.latest_detection_id;
      return obj;
    };

    // Add all active trackers
    for (const auto &trk : tracker_.active_trackers) {
      outputs.push_back(tracker_to_obj(trk));
    }

    // Add all not_active trackers (unconfirmed tracks)
    for (const auto &trk : tracker_.not_active_trackers) {
      outputs.push_back(tracker_to_obj(trk));
    }

    // Add all lost trackers
    for (const auto &trk : tracker_.lost_trackers) {
      outputs.push_back(tracker_to_obj(trk));
    }

    // Note: Removed tracks are not accessible from OC-SORT's public API
    // They are permanently removed from all lists once max_age is exceeded
  }

  return outputs;
}
#endif

#ifdef HAVE_TRACKTRACK
TrackTrackWrapper::TrackTrackWrapper(const TrackerParams &params)
{
  // Convert TrackerParams to TrackTrack::Params
  tracktrack::TrackTrack::Params tt_params;

  // Detection thresholds
  tt_params.det_thr = GetParamOrDefault<float>(params, "det_thr", 0.6f);
  tt_params.init_thr = GetParamOrDefault<float>(params, "init_thr", 0.6f);

  // Matching parameters
  tt_params.match_thr = GetParamOrDefault<float>(params, "match_thr", 0.8f);
  tt_params.tai_thr = GetParamOrDefault<float>(params, "tai_thr", 0.55f);
  tt_params.penalty_p = GetParamOrDefault<float>(params, "penalty_p", 0.20f);
  tt_params.penalty_q = GetParamOrDefault<float>(params, "penalty_q", 0.40f);
  tt_params.reduce_step = GetParamOrDefault<float>(params, "reduce_step", 0.05f);

  // Track management
  tt_params.max_time_lost = GetParamOrDefault<int>(params, "max_time_lost", 30);
  tt_params.min_len = GetParamOrDefault<int>(params, "min_len", 3);
  tt_params.min_box_area = GetParamOrDefault<int>(params, "min_box_area", 100);

  // Feature parameters
  tt_params.alpha = GetParamOrDefault<float>(params, "alpha", 0.95f);

  // Optional components
  tt_params.use_cmc = GetParamOrDefault<bool>(params, "use_cmc", true);
  tt_params.use_aflink = GetParamOrDefault<bool>(params, "use_aflink", false);
  tt_params.aflink_model = GetParamOrDefault<std::string>(params, "aflink_model", "");

  // Dataset type
  tt_params.dataset_type = GetParamOrDefault<std::string>(params, "dataset_type", "MOT");

  // State exposure control
  return_all_states_ = GetParamOrDefault<bool>(params, "return_all_states", false);

  // AFLink validation - raise error if enabled (not implemented)
  if (tt_params.use_aflink) {
    throw std::runtime_error(
        "AFLink is not yet implemented for TrackTrack. Set use_aflink=false.");
  }

  // Create tracker
  tracker_ = std::make_unique<tracktrack::TrackTrack>(tt_params);
}

const std::vector<ax::TrackedObject>
TrackTrackWrapper::Update(const std::vector<ax::ObservedObject> &detections,
    const std::vector<std::vector<float>> &embeddings,
    const std::optional<Eigen::Matrix<float, 2, 3>> &transform)
{

  // Convert ObservedObject to tracktrack::Detection
  std::vector<tracktrack::Detection> tt_detections;
  tt_detections.reserve(detections.size());

  for (size_t i = 0; i < detections.size(); ++i) {
    tracktrack::Detection det;
    det.bbox = Eigen::Vector4f(detections[i].bbox.x1, detections[i].bbox.y1,
        detections[i].bbox.x2, detections[i].bbox.y2);
    det.score = detections[i].score;
    det.class_id = detections[i].class_id;
    det.original_index = static_cast<int>(i);

    // Debug disabled
    // if (i < 2) {
    //   std::cout << "  Wrapper: ObservedObject score=" << detections[i].score
    //             << " -> Detection score=" << det.score << std::endl;
    // }

    // Add features if available
    if (i < embeddings.size() && !embeddings[i].empty()) {
      det.features = Eigen::Map<const Eigen::VectorXf>(
          embeddings[i].data(), embeddings[i].size());
    }

    tt_detections.push_back(det);
  }

  // For TrackTrack, detections_95 represents detections with score >= 0.95
  // before NMS Since we only receive post-NMS detections, we'll approximate by:
  // Using all detections as detections_95 (this means no deleted detections)
  std::vector<tracktrack::Detection> tt_detections_95 = tt_detections;

  // Get CMC transform
  Eigen::Matrix<float, 2, 3> cmc_transform = Eigen::Matrix<float, 2, 3>::Identity();
  if (transform.has_value()) {
    cmc_transform = transform.value();
  }

  // Update tracker
  tracker_->update(tt_detections, tt_detections_95, cmc_transform);

  // Get tracks based on return_all_states setting
  std::vector<tracktrack::Track *> tracks;
  if (return_all_states_) {
    tracks = tracker_->get_tracks(); // All tracks including lost/removed
  } else {
    tracks = tracker_->get_active_tracks(); // Only Tracked state
  }

  // Convert results to TrackedObject with proper state mapping
  std::vector<ax::TrackedObject> outputs;
  outputs.reserve(tracks.size());

  for (const auto &track : tracks) {
    Eigen::Vector4f bbox = track->get_bbox();

    // Map tracktrack::TrackState to ax::TrackState
    ax::TrackState ax_state;
    switch (track->get_state()) {
      case tracktrack::TrackState::New:
        ax_state = ax::kNew;
        break;
      case tracktrack::TrackState::Tracked:
        ax_state = ax::kTracked;
        break;
      case tracktrack::TrackState::Lost:
        ax_state = ax::kLost;
        break;
      case tracktrack::TrackState::Removed:
        ax_state = ax::kRemoved;
        break;
      default:
        ax_state = ax::kUndefined;
        break;
    }

    ax::TrackedObject obj(bbox[0], bbox[1], bbox[2], bbox[3],
        track->get_track_id(), track->get_class_id(), track->get_score(), ax_state);
    obj.latest_detection_id = track->get_latest_detection_id();
    outputs.push_back(obj);
  }

  // Store current detections for next frame
  prev_detections_ = tt_detections;

  return outputs;
}
#endif
