# TrackTrack C++ Implementation

C++ implementation of the TrackTrack multi-object tracking algorithm for the Axelera tracking framework.

## Building

The tracker is automatically built when the `algorithms/tracktrack` directory exists:

```bash
cd /path/to/trackers
mkdir build && cd build
cmake .. -G Ninja
ninja
```

## Usage

### Command Line
```bash
./track video.mp4 --algo tracktrack

# With custom parameters
./track video.mp4 --algo tracktrack --det_thr 0.7 --match_thr 0.75

# Disable CMC
./track video.mp4 --algo tracktrack --use_cmc false
```

### C++ API
```cpp
#include "TrackerFactory.h"

// Create tracker with default parameters
auto tracker = CreateMultiObjTracker("tracktrack", {});

// Or with custom parameters
TrackerParams params = {
    {"det_thr", 0.7f},
    {"match_thr", 0.75f},
    {"use_cmc", true}
};
auto tracker = CreateMultiObjTracker("tracktrack", params);

// Update with detections
std::vector<ax::ObservedObject> detections = ...;
std::vector<ax::TrackedObject> tracks = tracker->Update(detections);
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `det_thr` | float | 0.6 | Detection confidence threshold |
| `init_thr` | float | 0.6 | Track initialization threshold |
| `match_thr` | float | 0.8 | Initial matching threshold |
| `tai_thr` | float | 0.55 | Track-aware NMS threshold |
| `penalty_p` | float | 0.20 | Low confidence penalty |
| `penalty_q` | float | 0.40 | Deleted detection penalty |
| `reduce_step` | float | 0.05 | Threshold reduction step |
| `max_time_lost` | int | 30 | Max frames before removing track |
| `min_len` | int | 3 | Min frames for Tracked state |
| `min_box_area` | int | 100 | Minimum box area |
| `use_cmc` | bool | true | Enable camera motion compensation |
| `use_aflink` | bool | false | Enable AFLink post-processing |
| `dataset_type` | string | "MOT" | Dataset type ("MOT" or "DanceTrack") |

## AFLink Support (Not Yet Available)

**Note:** AFLink post-processing requires the TrackTrack Python reference implementation,
which is not currently included in this repository. The infrastructure is in place but
the feature is not yet usable.

To enable AFLink in the future:

1. Obtain the TrackTrack Python reference implementation
2. Place it at `algorithms/TrackTrack_python/`
3. Convert the PyTorch model to ONNX:
```bash
cd algorithms/tracktrack/scripts
python convert_aflink_to_onnx.py
```

4. Enable AFLink when running:
```bash
./track video.mp4 --algo tracktrack --use_aflink true --aflink_model models/aflink_epoch20.onnx
```

Note: AFLink requires ONNX Runtime to be installed.

## Known Limitations

1. Currently uses greedy assignment instead of Hungarian algorithm
2. AFLink post-processing infrastructure exists but requires external Python components
3. Deleted detections are simulated (not from actual NMS)
