# `axelera.runtime.op.tracker`

Tracker operator wrapping C++ multi-object trackers (ByteTrack, OC-SORT).

## Summary

| Name | Description |
|------|-------------|
| [Tracker](#tracker) | Multi-object tracker operator wrapping C++ implementations. |

---

### Tracker

Multi-object tracker operator wrapping C++ implementations.

Tracks detected objects across video frames, maintaining persistent identity over time.
Select the algorithm via the `algo` parameter; all other parameters are algorithm-specific.

**Args:**

- **algo**: Algorithm name: 'bytetrack' (default), 'oc-sort', 'sort', or 'tracktrack'.
- **return_all_states**: If False (default), only return active tracks. If True, also return lost/removed tracks for visualization. WARNING: Enabling adds ~7% overhead and breaks MOT metrics.
- **ByteTrack parameters**: (Zhang et al., 2022 -- https://arxiv.org/abs/2110.06864)
- **frame_rate**: Video frame rate used to compute track buffer duration (default: 30).
- **track_buffer**: Number of frames to keep a lost track alive (default: 30).
- **SORT parameters**: (Bewley et al., 2016 -- https://arxiv.org/abs/1602.00763)
- **max_age**: Max frames a track can be lost before removal (default: 30).
- **min_hits**: Minimum detections before a track is confirmed (default: 3).
- **iou_threshold**: IoU threshold for matching detections to tracks (default: 0.3).
- **OC-SORT parameters**: (Cao et al., 2022 -- https://arxiv.org/abs/2203.14360). Extends SORT with observation-centric re-update and virtual trajectory.
- **det_thresh**: Minimum detection confidence (default: 0.0).
- **max_age**: Max frames a track can be lost before removal (default: 30).
- **min_hits**: Minimum detections before a track is confirmed (default: 3).
- **iou_threshold**: IoU threshold for matching (default: 0.3).
- **delta**: Frames used for velocity direction estimation (default: 3).
- **inertia**: Weight of motion inertia in association (default: 0.2).
- **w_assoc_emb**: Appearance embedding weight in cost matrix (default: 0.75).
- **alpha_fixed_emb**: EMA coefficient for embedding updates (default: 0.95).
- **max_id**: Max track ID before counter resets (0 = no reset, default: 0).
- **aw_enabled**: Enable adaptive appearance weighting (default: False).
- **aw_param**: Adaptive weight scaling parameter (default: 0.5).
- **cmc_enabled**: Enable camera motion compensation (default: False).
- **enable_id_recovery**: Enable boundary-based track ID recovery (default: False).
- **rec_image_rect_margin**: Pixel margin defining the boundary zone (default: 20).
- **rec_track_min_time_since_update_at_boundary**: Min lost frames at boundary for recovery (default: 6).
- **rec_track_min_time_since_update_inside**: Min lost frames inside image for recovery (default: 300).
- **rec_track_min_age**: Minimum track age (frames) to qualify for recovery (default: 30).
- **rec_track_merge_lap_thresh**: LAP cost threshold for merging recovered tracks (default: 0.09).
- **rec_track_memory_capacity**: Max number of tracks stored in recovery memory (default: 1000).
- **rec_track_memory_max_age**: Max frames a track stays in recovery memory (default: 54000).
- **TrackTrack parameters**: (internal tracker with appearance-based association)
- **det_thr**: Detection confidence threshold for high-confidence detections (default: 0.6).
- **init_thr**: Score threshold for initializing new tracks (default: 0.6).
- **match_thr**: IoU threshold for first-stage association (default: 0.8).
- **tai_thr**: Track appearance index threshold for second-stage association (default: 0.55).
- **penalty_p**: Penalty parameter p in motion cost function (default: 0.20).
- **penalty_q**: Penalty parameter q in motion cost function (default: 0.40).
- **reduce_step**: Confidence reduction per lost frame (default: 0.05).
- **max_time_lost**: Max frames before a lost track is removed (default: 30).
- **min_len**: Minimum track length (frames) before outputting (default: 3).
- **min_box_area**: Minimum bounding box area in pixels to keep a track (default: 100).
- **alpha**: EMA smoothing factor for track state updates (default: 0.95).
- **use_cmc**: Enable camera motion compensation (default: True).
- **use_aflink**: Enable AFLink appearance feature linking (default: False, not implemented).
- **aflink_model**: Path to AFLink model file (default: '', not implemented).
- **dataset_type**: Evaluation dataset format ('MOT' or 'dance') (default: 'MOT').

**Note:**

TrackTrack deleted detection recovery is disabled (requires pre-NMS detections).
TrackTrack AFLink is not implemented (raises error in C++ if use_aflink=True).

**Examples:**

```python
pipeline = op.seq(
    op.load('yolov8n-coco'),
    op.decode_detections(algo='yolov8'),
    op.nms(),
    op.to_image_space(),
    op.axdetection(class_id_type=op.CocoClasses),
    op.tracker(),  # Default: ByteTrack
)

# Or use TrackTrack:
pipeline = op.seq(
    op.load('yolov8n-coco'),
    op.decode_detections(algo='yolov8'),
    op.nms(),
    op.to_image_space(),
    op.axdetection(class_id_type=op.CocoClasses),
    op.tracker(algo='tracktrack'),
)
```

**Constructor:**

```python
__init__(algo: str = 'bytetrack', frame_rate: int = 30, track_buffer: int = 30, det_thresh: float = 0.0, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3, delta: int = 3, inertia: float = 0.2, w_assoc_emb: float = 0.75, alpha_fixed_emb: float = 0.95, max_id: int = 0, aw_enabled: bool = False, aw_param: float = 0.5, cmc_enabled: bool = False, enable_id_recovery: bool = False, rec_image_rect_margin: int = 20, rec_track_min_time_since_update_at_boundary: int = 6, rec_track_min_time_since_update_inside: int = 300, rec_track_min_age: int = 30, rec_track_merge_lap_thresh: float = 0.09, rec_track_memory_capacity: int = 1000, rec_track_memory_max_age: int = 54000, det_thr: float = 0.6, init_thr: float = 0.6, match_thr: float = 0.8, tai_thr: float = 0.55, penalty_p: float = 0.2, penalty_q: float = 0.4, reduce_step: float = 0.05, max_time_lost: int = 30, min_len: int = 3, min_box_area: int = 100, alpha: float = 0.95, use_cmc: bool = True, use_aflink: bool = False, aflink_model: str = '', dataset_type: str = 'MOT', return_all_states: bool = False, tracker: Any = field(default=None, init=False, repr=False))
```
