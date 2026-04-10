#!/usr/bin/env python
# Copyright Axelera AI, 2025

import argparse
from collections import defaultdict, deque
import os
import threading
import time
from typing import Optional

import cv2

# Defer wx import until we know we need a GUI
wx = None

# Try to import CLIP, install if missing
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip

import numpy as np
import torch

from axelera import types
from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.meta.segmentation import InstanceSegmentationMeta
from axelera.app.meta.tracker import TrackerMeta
from axelera.app.stream import create_inference_stream
import axelera.trackers as axtracker

W = 1280
H = 720
SURFACE_SIZE = (W, H)

SURFACE_REFRESH_INTERVAL = 1000 // 30  # ~30 FPS


def _update_prompt(model, prompt):
    device = next(model.parameters()).device
    text_input = clip.tokenize(prompt).to(device)
    text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


DEFAULT_PROMPT = 'fruits and vegetables'
DEFAULT_FEATURES = None
DEFAULT_TOPK = 1
model = None
preprocess = None

TRACK_HISTORY_LEN = 30
SORT_PARAMS = {
    "det_thresh": 0.0,
    "maxAge": 30,
    "minHits": 3,
    "iouThreshold": 0.3,
}


def build_wx_viewer(prompt: str, topk: int):
    class WxViewer(wx.Frame):
        def __init__(
            self,
            app: display.App,
            size,
            stop: threading.Event,
            surface: display.Surface,
        ):
            super().__init__(parent=None, title="FastSAM demo", size=size)
            self._app = app
            self._stop = stop
            self._surface = surface
            self._sizer = wx.BoxSizer(wx.VERTICAL)
            self._prompt = prompt
            self._topk = topk

            self._bitmap_panel = wx.Panel(self)
            self._bitmap_sizer = wx.BoxSizer(wx.VERTICAL)
            self._bmp_ctrl = wx.StaticBitmap(self._bitmap_panel, -1, size=SURFACE_SIZE)
            self._bitmap_sizer.Add(self._bmp_ctrl, 0, wx.ALIGN_CENTER)
            self._bitmap_panel.SetSizer(self._bitmap_sizer)
            black = wx.Bitmap.FromBuffer(W, H, np.zeros((H, W, 3), dtype=np.uint8))
            self._bmp_ctrl.SetBitmap(black)

            self._sizer.Add(self._bitmap_panel, 1, wx.ALL | wx.EXPAND, 5)

            controls = wx.BoxSizer(wx.HORIZONTAL)

            prompt_label = wx.StaticText(self, label="Text prompt:")
            controls.Add(prompt_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.prompt_entry = wx.TextCtrl(self, value=self._prompt)
            controls.Add(self.prompt_entry, 3, wx.ALL | wx.EXPAND, 5)

            topk_label = wx.StaticText(self, label="Top-K:")
            controls.Add(topk_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.topk_entry = wx.TextCtrl(self, value=str(self._topk))
            controls.Add(self.topk_entry, 1, wx.ALL | wx.EXPAND, 5)

            update_btn = wx.Button(self, label="Update")
            update_btn.Bind(wx.EVT_BUTTON, self.on_update)
            controls.Add(update_btn, 0, wx.ALL, 5)

            self._sizer.Add(controls, 0, wx.ALL | wx.EXPAND, 5)
            self.SetSizer(self._sizer)
            self.Bind(wx.EVT_CLOSE, self._on_close)
            self.Bind(wx.EVT_SIZE, self._on_resize)
            wx.CallAfter(self._on_timer, None)

            self.Fit()
            self.Show()

        def _get_bitmap_size(self):
            client_size = self.GetClientSize()
            controls_height = 0
            if self._sizer.GetItemCount() > 1:
                controls_item = self._sizer.GetItem(1)
                if controls_item:
                    controls_height = controls_item.GetMinSize().height

            w = max(1, client_size.width)
            h = max(1, client_size.height - controls_height)

            return w, h

        def _on_resize(self, event):
            self._bitmap_panel.Layout()
            event.Skip()

        def _on_timer(self, _):
            start = time.time()
            if self._stop.is_set():
                self._on_close(wx.CloseEvent())
                return

            if new := self._surface.pop_latest():
                np_img = new.asarray(types.ColorFormat.RGB).astype(np.uint8)
                buf = np_img.tobytes()
                bmp = wx.Image(new.width, new.height, buf).ConvertToBitmap()
                self._bmp_ctrl.SetBitmap(bmp)
            delay = max(1, SURFACE_REFRESH_INTERVAL - int((time.time() - start) * 1000))
            wx.CallLater(delay, self._on_timer, None)

        def _on_close(self, evt):
            self._stop.set()
            self.Destroy()

        def on_update(self, event):
            self._prompt = self.prompt_entry.GetValue()
            self._topk = self.topk_entry.GetValue()
            try:
                self._topk = int(self.topk_entry.GetValue())
            except ValueError:
                print("[Warning] Invalid topk value, must be integer.")
            print(f"[Prompt Updated] -> {self._prompt} | topk = {self._topk}")

        @property
        def prompt(self) -> str:
            return self._prompt

        @property
        def topk(self) -> int:
            return self._topk

    return WxViewer


def run_sort_on_meta(meta, text_features, topk, tracker, track_histories):
    """Run SORT on current detections, returning updated metas and tracks.

    Mapping masks->tracks:
    - SORT outputs include `latest_detection_id`, the index of the detection
      used for the current track update.
    - We keep the detection order from FastSAM postprocess; masks/boxes arrays
      are indexed consistently, so we can attach the matching mask/box by that
      index. SORT itself tracks boxes only; we propagate masks via this index.
    """
    det_idxs = meta.secondary_frame_indices.get('detections', [])
    using_secondary = len(det_idxs) > 0

    topk = min(len(det_idxs), topk) if using_secondary else topk
    similarity = None

    if using_secondary and len(det_idxs) > 1:
        emb_tensors = []
        for idx in det_idxs:
            emb = meta.get_secondary_meta('detections', idx).embedding
            emb_tensor = torch.tensor(emb, device=text_features.device)
            emb_tensors.append(emb_tensor.squeeze())

        img_features = torch.stack(emb_tensors)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        img_features = img_features.to(text_features.dtype)
        similarity = 100.0 * img_features @ text_features.T

        idxs = torch.argsort(similarity.squeeze(), descending=True)
        top_idxs = idxs[0:topk]

        newmasks = [meta.masks[det_idxs[idx]] for idx in top_idxs]
        newboxes = [meta.boxes[det_idxs[idx]] for idx in top_idxs]
        newscores = (
            similarity.squeeze()[top_idxs].detach().cpu().numpy().tolist()
            if similarity is not None
            else [1.0] * len(newboxes)
        )

    else:
        # Fallback to all detections/masks (common for FastSAM segmentation)
        newmasks = list(meta.masks)
        newboxes = list(meta.boxes)
        scores_arr = getattr(meta, "scores", np.array([]))
        newscores = scores_arr.tolist() if len(scores_arr) else [1.0] * len(newboxes)

    # Track objects with SORT to stabilize IDs/colors
    boxes_np = np.array(newboxes).reshape(-1, 4) if len(newboxes) else np.zeros((0, 4))
    scores_np = np.array(newscores).reshape(-1) if len(newscores) else np.zeros((0,))

    observations = []
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = box.tolist()
        score_val = float(scores_np[i]) if scores_np.size else 1.0
        observations.append(
            axtracker.ObservedObject.from_xyxy(x1, y1, x2, y2, class_id=0, score=score_val)
        )

    tracked_objects = tracker.update(observations)
    active_ids = set()
    tracked_boxes = []
    tracked_ids = []
    tracked_scores = []
    track_id_to_mask = {}

    for i, t in enumerate(tracked_objects):
        active_ids.add(t.track_id)
        bbox = np.array([t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2], dtype=np.float32)
        track_histories[t.track_id].append(bbox)

        det_idx = getattr(t, 'latest_detection_id', -1)
        # Fallback: if tracker didn’t set latest_detection_id, assume same order as detections
        if det_idx is None or det_idx < 0 or det_idx >= len(newmasks):
            det_idx = i if i < len(newmasks) else -1

        score_val = float(t.score)
        if det_idx >= 0 and det_idx < len(newmasks):
            mask = newmasks[det_idx]
            score_val = float(scores_np[det_idx]) if scores_np.size > det_idx else float(t.score)
            track_id_to_mask[t.track_id] = mask

        tracked_boxes.append(bbox)
        tracked_ids.append(t.track_id)
        tracked_scores.append(score_val)

    # Remove histories for tracks no longer active
    for tid in list(track_histories.keys()):
        if tid not in active_ids:
            track_histories.pop(tid, None)

    tracker_history_np = {tid: np.stack(hist) for tid, hist in track_histories.items()}
    tracker_meta = TrackerMeta(
        tracking_history=tracker_history_np,
        class_ids=[0] * len(tracker_history_np),
    )

    tracked_seg_meta = InstanceSegmentationMeta(seg_shape=meta.seg_shape, labels=meta.labels)
    masks_for_tracks = [
        track_id_to_mask.get(tid) for tid in tracked_ids if tid in track_id_to_mask
    ]
    boxes_for_tracks = [
        box for tid, box in zip(tracked_ids, tracked_boxes) if tid in track_id_to_mask
    ]
    ids_for_tracks = [tid for tid in tracked_ids if tid in track_id_to_mask]
    scores_for_tracks = [
        s for tid, s in zip(tracked_ids, tracked_scores) if tid in track_id_to_mask
    ]

    tracked_seg_meta.add_results(
        masks_for_tracks,
        np.array(boxes_for_tracks) if len(boxes_for_tracks) else np.array([]).reshape(0, 4),
        np.array(ids_for_tracks) if len(ids_for_tracks) else np.array([]),
        np.array(scores_for_tracks) if len(scores_for_tracks) else np.array([]),
    )

    return tracked_seg_meta, tracker_meta, tracked_objects, track_id_to_mask


def _overlay_mask(np_img: np.ndarray, mask_tuple, color: tuple[int, int, int], alpha: float = 0.5):
    """Overlay a single segmentation mask onto the RGB image."""
    if mask_tuple is None or len(mask_tuple) < 9:
        return
    img_x0, img_y0, img_x1, img_y1 = mask_tuple[4:8]
    mask = mask_tuple[-1]
    if mask.size == 0:
        return
    x0, y0 = max(0, img_x0), max(0, img_y0)
    x1, y1 = min(np_img.shape[1], img_x1), min(np_img.shape[0], img_y1)
    if x1 <= x0 or y1 <= y0:
        return
    dst_h, dst_w = y1 - y0, x1 - x0
    if mask.shape != (dst_h, dst_w):
        mask_resized = cv2.resize(mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask
    region = np_img[y0:y1, x0:x1]
    mask_bool = mask_resized.astype(bool)
    if mask_bool.shape[:2] != region.shape[:2]:
        return
    blended = region.copy()
    blended[mask_bool] = (
        alpha * np.array(color, dtype=np.uint8) + (1 - alpha) * blended[mask_bool]
    ).astype(np.uint8)
    region[mask_bool] = blended[mask_bool]
    np_img[y0:y1, x0:x1] = region


def _main(stream, stop, surface: display.Surface, window):
    nr_boxes_to_process = 15
    last_prompt = DEFAULT_PROMPT
    text_features = DEFAULT_FEATURES
    tracker = axtracker.create_tracker('sort', SORT_PARAMS)
    track_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
    update_msg = surface.text(
        "50%, 50%", "Updating...", font_size=24, anchor_x="center", anchor_y="center"
    )
    update_msg.hide()

    surface.options(0, bbox_label_format="{scorep:.0f}{scoreunit}")

    update_at_end = False
    for frame_result in stream:
        if stop.is_set():
            return
        current_prompt = window.prompt

        if current_prompt != last_prompt:
            # show msg here - but update after the frame updates so the message
            # is actually visible during the slow op...
            update_msg.show()
            last_prompt = current_prompt
            update_at_end = True

        meta = frame_result.meta['master_detections']
        nr_boxes = meta.boxes.shape[0]
        nr_boxes = min(nr_boxes, nr_boxes_to_process)

        tracked_seg_meta, tracker_meta, _, _ = run_sort_on_meta(
            meta, text_features, window.topk, tracker, track_histories
        )

        frame_result.meta.add_instance('tracking', tracker_meta)
        frame_result.meta.delete_instance('master_detections')
        frame_result.meta.add_instance('master_detections', tracked_seg_meta)

        surface.push(frame_result.image, frame_result.meta)
        if update_at_end:
            text_features = _update_prompt(model, current_prompt)
            update_msg.hide()
            update_at_end = False


def main(stream, stop: threading.Event, surface: display.Surface, window):
    try:
        _main(stream, stop, surface, window)
    finally:
        stop.set()


def _headless(stream, prompt: str, topk: int, max_frames: int):
    video_writer: Optional[cv2.VideoWriter] = None
    writer_fps = 7
    nr_boxes_to_process = 15
    tracker = axtracker.create_tracker('sort', SORT_PARAMS)
    track_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
    text_features = _update_prompt(model, prompt)

    for idx, frame_result in enumerate(stream, start=1):
        meta = frame_result.meta['master_detections']
        nr_boxes = meta.boxes.shape[0]
        nr_boxes = min(nr_boxes, nr_boxes_to_process)
        if args.debug_masks:
            print(
                f"    pre-run meta: masks={len(getattr(meta, 'masks', []))} "
                f"boxes={meta.boxes.shape if hasattr(meta, 'boxes') else 'n/a'}"
            )

        tracked_seg_meta, tracker_meta, tracked_objects, track_id_to_mask = run_sort_on_meta(
            meta, text_features, topk, tracker, track_histories
        )
        frame_result.meta.add_instance('tracking', tracker_meta)
        frame_result.meta.delete_instance('master_detections')
        frame_result.meta.add_instance('master_detections', tracked_seg_meta)

        summary = ", ".join(
            f"id={t.track_id} bbox=({int(t.bbox.x1)},{int(t.bbox.y1)},"
            f"{int(t.bbox.x2)},{int(t.bbox.y2)}) score={t.score:.2f}"
            for t in tracked_objects
        )
        print(f"[frame {idx}] tracks: {summary or 'none'}")
        if args.debug_masks:
            print(
                f"    masks: total={len(tracked_seg_meta.masks)} "
                f"track_id_to_mask={list(track_id_to_mask.keys())}"
            )

        if args.save_video:
            # Lazy init writer based on first frame size
            np_img = frame_result.image.asarray(types.ColorFormat.RGB).astype('uint8')
            h, w, _ = np_img.shape
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.save_video, fourcc, writer_fps, (w, h))
            # Draw tracks
            for t in tracked_objects:
                x1, y1, x2, y2 = map(int, (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2))
                color = ((37 * t.track_id) % 255, (17 * t.track_id) % 255, (97 * t.track_id) % 255)
                cv2.rectangle(np_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    np_img,
                    f"id:{t.track_id}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                mask = track_id_to_mask.get(t.track_id)
                if mask is not None:
                    _overlay_mask(np_img, mask, color, alpha=0.4)
            video_writer.write(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

        if max_frames and idx >= max_frames:
            break

    if video_writer is not None:
        video_writer.release()


def _load_clip(prompt: str):
    global model, preprocess
    if model is None or preprocess is None:
        print("Loading CLIP model...")
        model, preprocess = clip.load('RN50x4')
        print("CLIP model loaded.")
    print(f"Initializing with prompt: '{prompt}'")
    features = _update_prompt(model, prompt)
    print("Prompt initialized.")
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="FastSAM demo with SORT tracking")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless: skip wx display and print tracking to console",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for CLIP filtering"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Top-K detections to keep after CLIP scoring",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source (defaults to bowl-of-fruit demo video)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit the number of frames processed (0 = all)",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional path to save an annotated MP4 when running with --no-display",
    )
    parser.add_argument(
        "--debug-masks",
        action="store_true",
        help="Print per-frame mask/debug info (headless only)",
    )
    parser.add_argument(
        "--network",
        default="fastsams-rn50x4-onnx",
        choices=["fastsams-rn50x4-onnx", "fastsamx-rn50x4-onnx"],
        help="Network pipeline to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    HEADLESS = args.no_display

    DEFAULT_PROMPT = args.prompt
    DEFAULT_FEATURES = _load_clip(args.prompt)

    source = args.source or str(config.env.framework / "media/bowl-of-fruit.mp4@auto")

    tracers = inf_tracers.create_tracers('cpu_usage', 'end_to_end_fps')
    stream = create_inference_stream(
        network=args.network,
        sources=[source],
        pipe_type='gst',
        log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
        hardware_caps=config.HardwareCaps(
            vaapi=config.HardwareEnable.detect,
            opencl=config.HardwareEnable.detect,
            opengl=config.HardwareEnable.detect,
        ),
        tracers=tracers,
        specified_frame_rate=7,
    )

    if HEADLESS:
        try:
            _headless(stream, args.prompt, args.topk, args.max_frames)
        finally:
            stream.stop()
    else:
        # Import wx lazily only when GUI is requested
        try:
            import wx as _wx
        except ImportError:
            print(
                "[ERROR] The 'wxPython' module is not installed.\n"
                "Please install it using pip.\n"
                "First ensure libgtk-3-dev is installed. Run: sudo apt install libgtk-3-dev\n"
                "Run: pip install wxpython\n"
                "Exiting."
            )
            stream.stop()
            sys.exit(1)

        wx = _wx
        WxViewer = build_wx_viewer(args.prompt, args.topk)

        with display.App(renderer='opencv') as app:
            surface = app.create_surface(SURFACE_SIZE)
            stop = threading.Event()
            wx_app = wx.App(False)
            wx_wnd = WxViewer(app, (W, H), stop, surface)
            app.start_thread(main, (stream, stop, surface, wx_wnd), name="InferenceThread")
            wx_app.MainLoop()
        stream.stop()
