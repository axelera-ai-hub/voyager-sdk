# Copyright Axelera AI, 2026
import os
import argparse
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

#   The script builds a ReID dataset from DanceTrack. It crops person boxes from
#   the val sequences, scores each crop using visibility, aspect ratio, blur, and
#   overlap penalties, then keeps the best-scoring query image per identity. The
#   remaining crops are routed to the gallery split, with optional downsampling
#   via the gallery stride. It also samples a fixed number of random train crops
#   from the DanceTrack train split and saves them to bounding_box_train for
#   calibration.


def make_parser():
    parser = argparse.ArgumentParser("dancetrack reid dataset")

    parser.add_argument("--data_path", default="datasets", help="path to dancetrack data")
    parser.add_argument(
        "--save_path", default=".", help="Path to save the dancetrack-reid dataset"
    )
    parser.add_argument(
        "--query_target_aspect",
        type=float,
        default=2.0,
        help="Desired height/width ratio when using the score-based query strategy.",
    )
    parser.add_argument(
        "--query_aspect_weight",
        type=float,
        default=0.3,
        help="Penalty weight for aspect-ratio deviations when using the score strategy.",
    )
    parser.add_argument(
        "--gallery_stride",
        type=int,
        default=10,
        help="Keep only every Nth crop for the gallery (test) split. Use 1 to keep all.",
    )
    parser.add_argument(
        "--query_blur_penalty",
        type=float,
        default=1.0,
        help="Scale for blur penalty (higher penalizes blur more).",
    )
    parser.add_argument(
        "--query_overlap_penalty",
        type=float,
        default=1.0,
        help="Penalty (scaled by max IoU) applied to score-based queries when overlapping others.",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="Number of random train crops to save into bounding_box_train. Use 0 to skip.",
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=42,
        help="Random seed for sampling train crops.",
    )

    return parser


# ============================ for dancetrack ============================
def generate_trajectories(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')  # list of [n_lines] or [n_objs]
    values = []
    for l in lines:
        split = l.split(
            ','
        )  # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <active>, <category>, <visible_ratio>
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]  # int to float
        values.append(numbers)

    values = np.array(values)
    values[:, 4] += values[:, 2]  # tlwh to tlbr
    values[:, 5] += values[:, 3]

    return values


def _ensure_split_path(root_dir, split):
    data_path = os.path.join(root_dir, 'dancetrack', split)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Cannot find DanceTrack split '{split}' at {data_path}")
    return data_path


def _score_detection(
    visible_ratio,
    width,
    height,
    target_ratio=2.0,
    aspect_weight=0.3,
    blur_penalty_value=0.0,
    max_iou=0.0,
    overlap_penalty=1.0,
):
    ratio = height / max(1.0, width)
    aspect_penalty = abs(ratio - target_ratio)
    score = float(visible_ratio) - aspect_weight * aspect_penalty
    score -= blur_penalty_value
    if max_iou > 0:
        score -= overlap_penalty * max_iou
    return score


def _blur_penalty(patch, scale):
    if scale <= 0:
        return 0.0
    if patch.size == 0:
        return scale
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return scale / (variance + 1.0)


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _generate_split(args, test_save_path, query_save_path):
    data_path = _ensure_split_path(args.data_path, 'val')
    seqs = sorted(os.listdir(data_path))
    query_records = defaultdict(list)
    gallery_counts = defaultdict(int)
    pid_map = {}
    best_scores = {}
    best_paths = {}
    best_indices = {}
    pid_frame_idx = defaultdict(int)

    for seq in seqs:  # iteration over seqs
        print("current val seq", seq)

        ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
        gt = generate_trajectories(
            ground_truth_path
        )  # frame, id, x_tl, y_tl, x_br, y_br, active, category, visible_ratio

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = sorted(os.listdir(images_path))

        num_frames = len(img_files)
        for f in tqdm(range(num_frames)):  # iteration over frames
            img = cv2.imread(os.path.join(images_path, img_files[f]))
            if img is None:
                print("ERROR: Receive empty frame")
                continue
            H, W, _ = np.shape(img)
            frame_mask = f + 1 == gt[:, 0]
            dets = gt[
                frame_mask, 1:
            ]  # dets in current frame. [id, x_tl, y_tl, x_br, y_br, active, category, visible_ratio]
            frame_entries = []
            for det in dets:
                id_ = int(det[0]) + 1  # 0-index to 1-index
                x1 = int(det[1])
                y1 = int(det[2])
                x2 = int(det[3])
                y2 = int(det[4])
                # clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)
                if x1 >= x2 or y1 >= y2:
                    continue
                frame_entries.append((id_, x1, y1, x2, y2, det))

            # Ensure deterministic iteration even if gt ordering changes.
            frame_entries.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))

            for idx, (id_, x1, y1, x2, y2, det) in enumerate(frame_entries):
                patch = img[y1:y2, x1:x2, :]  # crop image

                pid_key = (seq, id_)
                if pid_key not in pid_map:
                    pid_map[pid_key] = len(pid_map) + 1
                pid = pid_map[pid_key]
                pid_frame_idx[pid] += 1
                frame_idx = pid_frame_idx[pid]
                file_name = (
                    (str(pid)).zfill(7)
                    + '_'
                    + seq[-4:]
                    + '_'
                    + (str(f + 1)).zfill(7)
                    + '_acc_data.jpg'
                )

                max_iou = 0.0
                for j, (_, ox1, oy1, ox2, oy2, _) in enumerate(frame_entries):
                    if j == idx:
                        continue
                    iou = _bbox_iou((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
                    max_iou = max(max_iou, iou)
                blur_penalty = _blur_penalty(patch, args.query_blur_penalty)
                visible_ratio = det[7] if det.size > 7 else 1.0
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                score = _score_detection(
                    visible_ratio,
                    width,
                    height,
                    target_ratio=args.query_target_aspect,
                    aspect_weight=args.query_aspect_weight,
                    blur_penalty_value=blur_penalty,
                    max_iou=max_iou,
                    overlap_penalty=args.query_overlap_penalty,
                )
                prev_best = best_scores.get(pid, -float('inf'))
                if score > prev_best:
                    prev_path = best_paths.get(pid)
                    prev_idx = best_indices.get(pid)
                    query_path = os.path.join(query_save_path, file_name)
                    cv2.imwrite(query_path, patch)
                    best_scores[pid] = score
                    best_paths[pid] = query_path
                    best_indices[pid] = frame_idx
                    query_records[pid] = [query_path]
                    if prev_path:
                        keep_prev = args.gallery_stride <= 1 or (
                            prev_idx % args.gallery_stride == 0
                        )
                        if keep_prev:
                            os.replace(
                                prev_path,
                                os.path.join(test_save_path, os.path.basename(prev_path)),
                            )
                            gallery_counts[pid] += 1
                        else:
                            os.remove(prev_path)
                else:
                    keep_in_gallery = args.gallery_stride <= 1 or (
                        frame_idx % args.gallery_stride == 0
                    )
                    if keep_in_gallery:
                        cv2.imwrite(os.path.join(test_save_path, file_name), patch)
                        gallery_counts[pid] += 1

    for pid, query_paths in query_records.items():
        if gallery_counts[pid] == 0 and query_paths:
            os.replace(
                query_paths[0],
                os.path.join(test_save_path, os.path.basename(query_paths[0])),
            )


def _generate_train_split(args, train_save_path):
    train_samples = args.train_samples
    train_seed = args.train_seed
    if train_samples <= 0:
        return

    data_path = _ensure_split_path(args.data_path, 'train1')
    seqs = sorted(os.listdir(data_path))
    rng = random.Random(train_seed)

    pid_map = {}
    samples = []
    seen = 0

    for seq in seqs:
        print("current train seq", seq)

        ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
        if not os.path.isfile(ground_truth_path):
            print("WARNING: Missing gt for", seq)
            continue
        gt = generate_trajectories(ground_truth_path)

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = sorted(os.listdir(images_path))

        num_frames = len(img_files)
        for f in tqdm(range(num_frames)):
            img_path = os.path.join(images_path, img_files[f])
            frame_mask = f + 1 == gt[:, 0]
            dets = gt[frame_mask, 1:]

            for det in dets:
                id_ = int(det[0]) + 1
                x1 = int(det[1])
                y1 = int(det[2])
                x2 = int(det[3])
                y2 = int(det[4])
                if x2 <= x1 or y2 <= y1:
                    continue

                pid_key = (seq, id_)
                if pid_key not in pid_map:
                    pid_map[pid_key] = len(pid_map) + 1
                pid = pid_map[pid_key]

                candidate = (img_path, (x1, y1, x2, y2), pid, seq, f + 1)
                seen += 1
                if len(samples) < train_samples:
                    samples.append(candidate)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < train_samples:
                        samples[j] = candidate

    for img_path, (x1, y1, x2, y2), pid, seq, frame_idx in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W, _ = np.shape(img)
        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, W)
        y2 = min(y2, H)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = img[y1:y2, x1:x2, :]
        if patch.size == 0:
            continue
        file_name = (
            (str(pid)).zfill(7)
            + '_'
            + seq[-4:]
            + '_'
            + (str(frame_idx)).zfill(7)
            + '_acc_data.jpg'
        )
        cv2.imwrite(os.path.join(train_save_path, file_name), patch)


def main_dancetrack(args):
    # Create folder for outputs
    save_path = os.path.join(args.save_path, 'dancetrack-reid')
    os.makedirs(save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    query_save_path = os.path.join(save_path, 'query')
    os.makedirs(query_save_path, exist_ok=True)

    _generate_split(args, test_save_path, query_save_path)
    _generate_train_split(args, train_save_path)

    return save_path


if __name__ == "__main__":
    args = make_parser().parse_args()
    main_dancetrack(args)
