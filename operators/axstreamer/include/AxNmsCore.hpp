// Copyright Axelera AI, 2026
#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <span>
#include <unordered_map>
#include <vector>

///
/// @brief Corner box geometry, and the two suppression rules over any measure.
///
/// This file deliberately exists twice, under operators/axstreamer/include and
/// under axelera_runtime2/include/axstreamer, because runtime2 does not take a
/// build dependency on the operators tree. The two are meant to stay identical,
/// so a change to one belongs in the other. Nothing on the GStreamer side
/// includes it yet: AxNms.cpp still carries its own geometry and its own greedy
/// loop, so over there everything here is available and tested rather than
/// wired.
///
/// Plain spans and PODs only, so no metadata class and no OpenCV comes with
/// them.
///
namespace ax_nms
{

/// Corner box. The GStreamer metadata holds integral pixel coordinates, which
/// float represents exactly, so the same type also serves float model space
/// boxes.
struct box_xyxy {
  float x1;
  float y1;
  float x2;
  float y2;
};

/// Added to each side length before an area is taken. Metadata boxes count
/// x2/y2 as the last covered pixel, so a box from 0 to 9 is 10 wide and needs
/// inclusive_pixels. Half open float boxes, as used by runtime2, OpenCV and
/// torchvision, need exclusive_pixels.
inline constexpr float inclusive_pixels = 1.0F;
inline constexpr float exclusive_pixels = 0.0F;

inline float
area(const box_xyxy &box, float box_offset)
{
  return (box_offset + box.x2 - box.x1) * (box_offset + box.y2 - box.y1);
}

inline float
intersection(const box_xyxy &lhs, const box_xyxy &rhs, float box_offset)
{
  const float ix = box_offset + std::min(lhs.x2, rhs.x2) - std::max(lhs.x1, rhs.x1);
  const float iy = box_offset + std::min(lhs.y2, rhs.y2) - std::max(lhs.y1, rhs.y1);
  return std::max(0.0F, ix) * std::max(0.0F, iy);
}

/// Overload for a loop that holds one box fixed, so its area is computed once.
inline float
intersection_over_union(const box_xyxy &lhs, float lhs_area, const box_xyxy &rhs, float box_offset)
{
  const float inter = intersection(lhs, rhs, box_offset);
  return inter / (lhs_area + area(rhs, box_offset) - inter);
}

inline float
intersection_over_union(const box_xyxy &lhs, const box_xyxy &rhs, float box_offset)
{
  return intersection_over_union(lhs, area(lhs, box_offset), rhs, box_offset);
}

///
/// @brief Which of the two suppression rules to apply.
///
/// Greedy takes the best box left, drops everything it overlaps, and repeats,
/// so a dropped box never gets its turn to suppress: where the best box reaches
/// a second box but not a third, and the second reaches the third, the third
/// survives.
///
/// Fast decides every removal against the original ranking, so a box goes
/// whenever ANY better scoring box overlaps it, including one that was itself
/// removed. On the case above it drops the third box. These are different
/// algorithms rather than approximations of each other, and which one a model
/// wants is fixed by whatever it was evaluated against.
///
/// Greedy is also the cheaper of the two here, which inverts the usual
/// expectation. Fast suppression earns its name where the whole overlap matrix
/// is one vectorised op; this walks the pairs in scalar code, and greedy erases
/// a dropped candidate, which skips a whole row of that walk.
///
enum class rule { greedy, fast };

///
/// @brief Rank the candidates, best score first.
///
/// Stable, so which of a tied group survives follows the input order rather
/// than the sort's internals. Quantised scores tie routinely.
///
/// @return order[rank] = index of that candidate in the caller's arrays
///
inline std::vector<int>
rank_by_score(std::span<const float> scores)
{
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
      [scores](int a, int b) { return scores[a] > scores[b]; });
  return order;
}

///
/// @brief The exact corner box intersection over union, as a measure for suppress().
///
/// An overlap of exactly iou_threshold keeps the box, matching torchvision and
/// cv2.dnn.NMSBoxes. AxNms.cpp drops it, and that is the one place this copy
/// does not follow it: quantised scores put boxes on the boundary often enough
/// that agreeing with the references runtime2 is checked against is worth more.
///
class CornerIou
{
  public:
  CornerIou(std::span<const box_xyxy> boxes, std::span<const int> order,
      float iou_threshold, float box_offset)
      : boxes_(boxes),
        order_(order),
        threshold_(iou_threshold),
        offset_(box_offset)
  {
  }

  void suppresses(int reference, std::span<const int> candidates, std::span<bool> out) const
  {
    //  Held in locals so the fixed box is not reloaded per candidate.
    const box_xyxy best = boxes_[order_[reference]];
    const float best_area = area(best, offset_);
    for (size_t k = 0; k != candidates.size(); ++k) {
      out[k] = intersection_over_union(best, best_area, boxes_[order_[candidates[k]]], offset_)
               > threshold_;
    }
  }

  private:
  std::span<const box_xyxy> boxes_;
  std::span<const int> order_;
  float threshold_;
  float offset_;
};

///
/// @brief Apply a suppression rule over any overlap measure.
///
/// This is the only suppression loop. A measure supplies
///
///     void suppresses(int reference, std::span<const int> candidates,
///                     std::span<bool> out) const;
///
/// where `reference` and every entry of `candidates` are ranks in the
/// `rank_by_score` order the measure was built with, `candidates` ascends, and
/// `out[k]` says whether the reference box suppresses `candidates[k]`. Checked
/// by use rather than by a concept declaration, so a measure needs no base
/// class.
///
/// The batch is a row rather than a pair so a measure can hoist its
/// per-reference work out of it, and can settle a candidate on arithmetic of its
/// own: CornerIou above is the trivial case, and the probabilistic-IoU measure
/// runtime2 adds precomputes a covariance per rank and skips a log per pair.
///
/// The threshold belongs to the measure, because the comparison is the measure's
/// own. CornerIou keeps a box at exactly the threshold and ProbIou suppresses
/// it, each following its own upstream reference; those references disagree, so
/// the boundary cannot be unified without breaking one of them.
///
/// @param order          rank_by_score output over the same candidates
/// @param class_ids      indexed by the caller's own index, not by rank
/// @param class_agnostic Compare across classes as well as within them
/// @param top_k          Positive bounds the candidates ranked per class; one
///                       past the cap is neither kept nor able to suppress. Not
///                       positive keeps every candidate
/// @param max_boxes      Cap on the number kept. 0 keeps nothing; only a
///                       negative value means no cap
/// @return Indices into the caller's arrays to keep, highest score first
///
template <class Measure>
std::vector<int>
suppress(std::span<const int> order, std::span<const int> class_ids,
    const Measure &measure, bool class_agnostic, int top_k, int max_boxes, rule which)
{
  //  Ranks still in play, ascending. top_k is a candidate cap rather than a
  //  suppression rule, so it is applied once here instead of inside the walk.
  std::vector<int> live;
  live.reserve(order.size());
  if (top_k > 0) {
    std::unordered_map<int, int> ranked_of_class;
    for (int rank = 0; rank != static_cast<int>(order.size()); ++rank) {
      if (++ranked_of_class[class_ids[order[rank]]] <= top_k) {
        live.push_back(rank);
      }
    }
  } else {
    live.resize(order.size());
    std::iota(live.begin(), live.end(), 0);
  }

  const bool fast = which == rule::fast;
  //  Greedy erases a dropped candidate from `live`, so it never pays for that
  //  candidate again. Fast cannot, because a dropped box still suppresses, so it
  //  records the drop instead and measures every pair. That is the whole
  //  difference between the two rules; everything else below is shared.
  std::vector<char> dropped(fast ? live.size() : 0, 0);
  auto row = std::make_unique_for_overwrite<bool[]>(live.size());

  std::vector<int> keep;
  keep.reserve(max_boxes < 0 ? live.size() : std::min(size_t(max_boxes), live.size()));

  int kept = 0;
  for (size_t i = 0; i != live.size() && kept != max_boxes; ++i) {
    const int best = live[i];
    if (!fast || dropped[i] == 0) {
      keep.push_back(order[best]);
      ++kept;
    }

    const size_t count = live.size() - i - 1;
    if (count == 0) {
      break;
    }
    measure.suppresses(best, std::span<const int>(live.data() + i + 1, count),
        std::span<bool>(row.get(), count));

    const int best_class = class_ids[order[best]];
    if (fast) {
      for (size_t k = 0; k != count; ++k) {
        const int candidate = live[i + 1 + k];
        if (row[k] && (class_agnostic || best_class == class_ids[order[candidate]])) {
          dropped[i + 1 + k] = 1;
        }
      }
    } else {
      //  Compacted in place, forward, so the write cursor never passes the read
      //  cursor and the span handed to the measure above stays valid until here.
      size_t write = i + 1;
      for (size_t k = 0; k != count; ++k) {
        const int candidate = live[i + 1 + k];
        const bool suppressed
            = row[k] && (class_agnostic || best_class == class_ids[order[candidate]]);
        if (!suppressed) {
          live[write++] = candidate;
        }
      }
      live.resize(write);
    }
  }
  return keep;
}

///
/// @brief Remove corner boxes that overlap a better scoring box of the same class.
///
/// @param class_agnostic  Compare across classes as well as within them
/// @param max_boxes       Cap on the number kept. 0 keeps nothing; only a
///                        negative value means no cap
/// @param box_offset      inclusive_pixels or exclusive_pixels
/// @return Indices into boxes to keep, highest score first
///
inline std::vector<int>
greedy_indices(std::span<const box_xyxy> boxes, std::span<const float> scores,
    std::span<const int> class_ids, float iou_threshold, bool class_agnostic,
    int max_boxes, float box_offset)
{
  const std::vector<int> order = rank_by_score(scores);
  return suppress(order, class_ids, CornerIou(boxes, order, iou_threshold, box_offset),
      class_agnostic, 0, max_boxes, rule::greedy);
}

///
/// @brief Remove corner boxes using fast suppression, as YOLACT defines it.
///
/// Suppression is always per class, since ranking within a class is what the
/// algorithm is built on, so there is no class agnostic form.
///
/// @param top_k      Candidates ranked per class before that class is compared;
///                   not positive keeps every candidate of the class
/// @param max_boxes  Cap on the total kept. Not positive means no cap, unlike
///                   greedy_indices, where 0 keeps nothing
/// @param box_offset inclusive_pixels or exclusive_pixels
/// @return Indices into boxes to keep, highest score first
///
inline std::vector<int>
fast_indices(std::span<const box_xyxy> boxes, std::span<const float> scores,
    std::span<const int> class_ids, float iou_threshold, int top_k,
    int max_boxes, float box_offset)
{
  const std::vector<int> order = rank_by_score(scores);
  //  Both spellings of "no cap" are already in use at this boundary, so the one
  //  suppress() carries is the greedy one and this edge normalises to it.
  return suppress(order, class_ids, CornerIou(boxes, order, iou_threshold, box_offset),
      false, top_k, max_boxes > 0 ? max_boxes : -1, rule::fast);
}

} // namespace ax_nms
