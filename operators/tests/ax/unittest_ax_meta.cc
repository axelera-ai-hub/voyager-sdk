// Copyright Axelera AI, 2025
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "AxMeta.hpp"
#include "AxMetaClassification.hpp"
#include "AxMetaObjectDetection.hpp"
#include "GstAxMeta.hpp"

TEST(ax_meta_base, insert_submeta)
{
  auto meta = AxMetaBase();
  auto submeta1 = std::make_shared<AxMetaBase>();
  auto submeta2 = std::make_shared<AxMetaBase>();
  auto submeta3 = std::make_shared<AxMetaBase>();

  EXPECT_NO_THROW(meta.insert_submeta("submeta", 1, 2, submeta1));
  EXPECT_NO_THROW(meta.insert_submeta("submeta", 0, 2, submeta2));
}

TEST(ax_meta_base, get_submeta)
{
  auto meta = AxMetaBase();
  auto submeta = std::make_shared<AxMetaObjDetection>();

  meta.insert_submeta("submeta", 0, 2, submeta);

  EXPECT_THROW(meta.get_submeta<AxMetaObjDetection>("submeta", 0, 1), std::runtime_error);
  EXPECT_THROW(meta.get_submeta<AxMetaObjDetection>("submeta", 1, 2), std::runtime_error);
  EXPECT_THROW(meta.get_submeta<AxMetaObjDetection>("non_existent", 0, 2), std::runtime_error);
  EXPECT_THROW(meta.get_submeta<AxMetaClassification>("submeta", 0, 2), std::runtime_error);
  EXPECT_NO_THROW(meta.get_submeta<AxMetaObjDetection>("submeta", 0, 2));
}

TEST(ax_meta_base, get_submetas)
{
  auto meta = AxMetaBase();
  auto submeta = std::make_shared<AxMetaObjDetection>();
  meta.insert_submeta("submeta", 1, 3, submeta);

  EXPECT_THROW(meta.get_submetas<AxMetaClassification>("submeta"), std::runtime_error);
  EXPECT_EQ(meta.get_submetas<AxMetaObjDetection>("submeta").size(), 3);
  EXPECT_EQ(meta.get_submetas<AxMetaObjDetection>("non_existent").size(), 0);
  EXPECT_EQ(meta.get_submetas<AxMetaObjDetection>("submeta")[0], nullptr);
  EXPECT_EQ(meta.get_submetas<AxMetaObjDetection>("submeta")[1], submeta.get());
  EXPECT_EQ(meta.get_submetas<AxMetaObjDetection>("submeta")[2], nullptr);
}

namespace
{

std::shared_ptr<AxMetaObjDetection>
make_detection(int num_boxes)
{
  std::vector<box_xyxy> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  for (int i = 0; i != num_boxes; ++i) {
    boxes.push_back({ i, i, i + 10, i + 10 });
    scores.push_back(0.5F);
    classes.push_back(1);
  }
  return std::make_shared<AxMetaObjDetection>(
      std::move(boxes), std::move(scores), std::move(classes));
}

} // namespace

//  get_submeta_from_buffer() allocates an extern_meta array per subframe on top of
//  the arrays of models, submodels and subframes. free_submeta() once released only
//  the outer three, so every buffer carrying submetas leaked its per-subframe arrays
//  - which, on a cascaded network, is every frame. The release build cannot observe
//  that, so this test earns its keep under LSan (make operators CFG=asan).
TEST(gst_ax_meta, free_submeta_releases_per_subframe_meta)
{
  gst_init(nullptr, nullptr);
  constexpr int subframes = 3;

  GstBuffer *buffer = gst_buffer_new_allocate(nullptr, 16, nullptr);
  ASSERT_NE(buffer, nullptr);
  auto &meta_map = *gst_buffer_get_general_meta(buffer)->meta_map_ptr;

  auto master = std::make_unique<AxMetaObjDetection>();
  for (int i = 0; i != subframes; ++i) {
    master->insert_submeta("classifier", i, subframes, make_detection(2));
  }
  meta_map["detections"] = std::move(master);

  auto all_models = get_submeta_from_buffer(buffer);
  ASSERT_EQ(all_models.num_meta, 1);
  ASSERT_EQ(all_models.meta_models[0].num_submeta, 1);

  const auto &submodel = all_models.meta_models[0].meta_submodels[0];
  ASSERT_EQ(submodel.subframe_number, subframes);
  for (int i = 0; i != subframes; ++i) {
    EXPECT_EQ(submodel.meta_indices[i].subframe_index, i);
    EXPECT_GT(submodel.meta_indices[i].num_extern_meta, 0);
    EXPECT_NE(submodel.meta_indices[i].meta_vector, nullptr);
  }

  free_submeta(&all_models);
  gst_buffer_unref(buffer);
}

//  A model without submetas must still round-trip cleanly: meta_indices and
//  meta_submodels are left null there, and free_submeta() has to tolerate that.
TEST(gst_ax_meta, free_submeta_handles_model_without_submetas)
{
  gst_init(nullptr, nullptr);

  GstBuffer *buffer = gst_buffer_new_allocate(nullptr, 16, nullptr);
  ASSERT_NE(buffer, nullptr);
  auto &meta_map = *gst_buffer_get_general_meta(buffer)->meta_map_ptr;
  meta_map["detections"] = std::make_unique<AxMetaObjDetection>();

  auto all_models = get_submeta_from_buffer(buffer);
  ASSERT_EQ(all_models.num_meta, 1);
  EXPECT_EQ(all_models.meta_models[0].num_submeta, 0);
  EXPECT_EQ(all_models.meta_models[0].meta_submodels, nullptr);

  free_submeta(&all_models);
  gst_buffer_unref(buffer);
}
