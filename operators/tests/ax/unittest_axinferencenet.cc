#include <gmock/gmock.h>
#include "AxInference.hpp"
#include "AxInferenceNet.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxStreamerUtils.hpp"
#include "unittest_ax_common.h"

using namespace std::string_literals;


TEST(ax_streamer_utils, output_drop)
{
  Ax::InferenceProperties props{};
  EXPECT_EQ(0, output_drop(props));
  props.num_children = 0;
  props.double_buffer = true;
  EXPECT_EQ(2, output_drop(props));
  props.num_children = 4;
  props.double_buffer = false;
  EXPECT_EQ(0, output_drop(props));
  props.num_children = 4;
  props.double_buffer = true;
  EXPECT_EQ(8, output_drop(props));
}

TEST(ax_streamer_utils, pipeline_pre_fill)
{
  Ax::InferenceProperties props{};
  EXPECT_EQ(1, pipeline_pre_fill(props));
  props.num_children = 0;
  props.double_buffer = true;
  EXPECT_EQ(1, pipeline_pre_fill(props));
  props.num_children = 2;
  props.double_buffer = true;
  EXPECT_EQ(2, pipeline_pre_fill(props));
  props.num_children = 4;
  props.double_buffer = false;
  EXPECT_EQ(4, pipeline_pre_fill(props));
  props.num_children = 4;
  props.double_buffer = true;
  EXPECT_EQ(4, pipeline_pre_fill(props));
}

TEST(ax_streamer_utils, split)
{
  using strings = std::vector<std::string_view>;
  using Ax::Internal::split;
  EXPECT_EQ((strings{}), split("", ','));
  EXPECT_EQ((strings{ "", "" }), split(",", ','));
  EXPECT_EQ((strings{ "a" }), split("a", ','));
  EXPECT_EQ((strings{ "a", "" }), split("a,", ','));
  EXPECT_EQ((strings{ "", "a" }), split(",a", ','));
  EXPECT_EQ((strings{ "", "a", "" }), split(",a,", ','));
  EXPECT_EQ((strings{ "a", "b" }), split("a,b", ','));
  EXPECT_EQ((strings{ "a", "", "b" }), split("a,,b", ','));
  EXPECT_EQ((strings{ "a", "b", "" }), split("a,b,", ','));
  EXPECT_EQ((strings{ "a", "b", "c" }), split("a,b,c", ','));
  EXPECT_EQ((strings{ "", "a", "b", "c" }), split(",a,b,c", ','));
  EXPECT_EQ((strings{ "", "a", "b", "c", "" }), split(",a,b,c,", ','));
}

TEST(ax_streamer_utils, extract_options)
{
  using Ax::Internal::extract_options;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, nullptr };
  using opts = std::unordered_map<std::string, std::string>;
  EXPECT_EQ((opts{}), extract_options(logger, ""));
  EXPECT_EQ((opts{ { "bob", "1" } }), extract_options(logger, "bob:1"));
  EXPECT_EQ((opts{ { "bob", "1" }, { "jane", "2" } }),
      extract_options(logger, "bob:1;jane:2"));
}

TEST(ax_streamer_utils, parse_skip_rate)
{
  EXPECT_EQ((Ax::SkipRate{ 0, 0 }), Ax::parse_skip_rate(""));
  EXPECT_EQ((Ax::SkipRate{ 1, 2 }), Ax::parse_skip_rate("1/2"));

  EXPECT_THROW(Ax::parse_skip_rate("1"), std::invalid_argument);
  EXPECT_THROW(Ax::parse_skip_rate("1/"), std::invalid_argument);
  EXPECT_THROW(Ax::parse_skip_rate("/1"), std::invalid_argument);
  EXPECT_THROW(Ax::parse_skip_rate("1:2"), std::invalid_argument);
  EXPECT_THROW(Ax::parse_skip_rate("1/0"), std::invalid_argument);
  EXPECT_THROW(Ax::parse_skip_rate("4/3"), std::invalid_argument);
}

TEST(ax_streamer_utils, interface_to_string)
{
  EXPECT_EQ("empty", Ax::to_string(AxDataInterface{}));
  EXPECT_EQ("RGB/640x480",
      Ax::to_string(AxVideoInterface{ { 640, 480, 0, 0, AxVideoFormat::RGB } }));
  EXPECT_EQ("empty", Ax::to_string(AxTensorsInterface{}));
  EXPECT_EQ("400,300,3[1 byte]", Ax::to_string(AxTensorsInterface{
                                     AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
                                 }));
  EXPECT_EQ("tensors/400,300,3[1 byte]",
      Ax::to_string(AxDataInterface{ AxTensorsInterface{
          AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
      } }));
  EXPECT_EQ("400,300,3[1 byte];600,400,4[4 byte]",
      Ax::to_string(AxTensorsInterface{
          AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
          AxTensorInterface{ { 600, 400, 4 }, 4, nullptr },
      }));
}

TEST(axinferencenet, heap_allocator_empty)
{
  auto allocator = Ax::create_heap_allocator();
  auto managed = allocator->allocate({});
  EXPECT_EQ("empty", Ax::to_string(managed.data()));
  EXPECT_TRUE(managed.buffers().empty());
  EXPECT_TRUE(managed.fds().empty());
}

TEST(axinferencenet, heap_allocator_video)
{
  auto allocator = Ax::create_heap_allocator();
  auto managed = allocator->allocate(
      AxVideoInterface{ { 640, 480, 0, 0, AxVideoFormat::RGB } });
  EXPECT_EQ("video/RGB/640x480", Ax::to_string(managed.data()));
  ASSERT_EQ(1, managed.buffers().size());
  EXPECT_TRUE(managed.fds().empty());
  auto &video = std::get<AxVideoInterface>(managed.data());
  EXPECT_EQ(video.data, managed.buffers()[0].get());
}

TEST(axinferencenet, heap_allocator_tensors1)
{
  auto allocator = Ax::create_heap_allocator();
  auto managed = allocator->allocate(AxTensorsInterface{
      AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
  });
  EXPECT_EQ("tensors/400,300,3[1 byte]", Ax::to_string(managed.data()));
  ASSERT_EQ(1, managed.buffers().size());
  EXPECT_TRUE(managed.fds().empty());
  auto &tensors = std::get<AxTensorsInterface>(managed.data());
  ASSERT_EQ(1, tensors.size());
  EXPECT_EQ(tensors[0].data, managed.buffers()[0].get());
}

TEST(axinferencenet, heap_allocator_tensors2)
{
  auto allocator = Ax::create_heap_allocator();
  auto managed = allocator->allocate(AxTensorsInterface{
      AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
      AxTensorInterface{ { 600, 400, 4 }, 4, nullptr },
  });
  EXPECT_EQ("tensors/400,300,3[1 byte];600,400,4[4 byte]", Ax::to_string(managed.data()));
  ASSERT_EQ(2, managed.buffers().size());
  EXPECT_TRUE(managed.fds().empty());
  auto &tensors = std::get<AxTensorsInterface>(managed.data());
  ASSERT_EQ(2, tensors.size());
  auto align = [](void *p) {
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) & ~0x7f);
  };
  EXPECT_EQ(align(tensors[0].data), align(managed.buffers()[0].get()));
  EXPECT_EQ(align(tensors[1].data), align(managed.buffers()[1].get()));
  EXPECT_EQ(align(tensors[0].data), tensors[0].data); // check they are aligned
  EXPECT_EQ(align(tensors[1].data), tensors[1].data); // check they are aligned
}

void
check_dmabuf_mapped_consistency(const Ax::ManagedDataInterface &managed)
{
  EXPECT_EQ(managed.fds().size(), managed.buffers().size());
  if (const auto *video = std::get_if<AxVideoInterface>(&managed.data())) {
    EXPECT_EQ(managed.fds().size(), 1);
    EXPECT_EQ(managed.buffers()[0].get(), video->data);
  } else if (const auto *tensors = std::get_if<AxTensorsInterface>(&managed.data())) {
    EXPECT_EQ(managed.fds().size(), tensors->size());
    size_t n = 0;
    for (const auto &tensor : *tensors) {
      EXPECT_EQ(managed.buffers()[n].get(), tensor.data);
      ++n;
    }
  }
}

void
check_dmabuf_unmapped_consistency(const Ax::ManagedDataInterface &managed)
{
  EXPECT_EQ(0, managed.buffers().size());
  if (const auto *video = std::get_if<AxVideoInterface>(&managed.data())) {
    EXPECT_EQ(managed.fds().size(), 1);
    EXPECT_EQ(nullptr, video->data);
  } else if (const auto *tensors = std::get_if<AxTensorsInterface>(&managed.data())) {
    EXPECT_EQ(managed.fds().size(), tensors->size());
    for (const auto &tensor : *tensors) {
      EXPECT_EQ(nullptr, tensor.data);
    }
  } else {
    EXPECT_TRUE(managed.fds().empty());
  }
}

TEST(axinferencenet, dmabuf_allocator_empty)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }

  auto allocator = Ax::create_dma_buf_allocator();
  auto managed = allocator->allocate({});
  EXPECT_EQ("empty", Ax::to_string(managed.data()));
  EXPECT_TRUE(managed.buffers().empty());
  EXPECT_TRUE(managed.fds().empty());
  allocator->map(managed);
  check_dmabuf_mapped_consistency(managed);
  allocator->unmap(managed);
  check_dmabuf_unmapped_consistency(managed);
}

TEST(axinferencenet, dmabuf_allocator_video)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }

  auto allocator = Ax::create_dma_buf_allocator();
  auto managed = allocator->allocate(
      AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr });

  EXPECT_EQ("video/RGB/640x480", Ax::to_string(managed.data()));
  check_dmabuf_unmapped_consistency(managed);
  for (int n = 0; n != 2; ++n) {
    allocator->map(managed);
    check_dmabuf_mapped_consistency(managed);
    allocator->unmap(managed);
    check_dmabuf_unmapped_consistency(managed);
  }
}

TEST(axinferencenet, dmabuf_allocator_tensors1)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }

  auto allocator = Ax::create_dma_buf_allocator();
  auto managed = allocator->allocate(AxTensorsInterface{
      AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
  });
  EXPECT_EQ("tensors/400,300,3[1 byte]", Ax::to_string(managed.data()));
  check_dmabuf_unmapped_consistency(managed);
  for (int n = 0; n != 2; ++n) {
    allocator->map(managed);
    check_dmabuf_mapped_consistency(managed);

    allocator->unmap(managed);
    check_dmabuf_unmapped_consistency(managed);
  }
}
TEST(axinferencenet, dmabuf_allocator_tensors2)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }

  auto allocator = Ax::create_dma_buf_allocator();
  auto managed = allocator->allocate(AxTensorsInterface{
      AxTensorInterface{ { 400, 300, 3 }, 1, nullptr },
      AxTensorInterface{ { 600, 400, 4 }, 4, nullptr },
  });
  EXPECT_EQ("tensors/400,300,3[1 byte];600,400,4[4 byte]", Ax::to_string(managed.data()));
  check_dmabuf_unmapped_consistency(managed);
  for (int n = 0; n != 2; ++n) {
    allocator->map(managed);
    check_dmabuf_mapped_consistency(managed);
    allocator->unmap(managed);
    check_dmabuf_unmapped_consistency(managed);
  }
}

TEST(axinferencenet, batched_buffer_pool)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }

  auto allocator = Ax::create_dma_buf_allocator();
  const int batch_size = 5; // in case there are any assumptions of 4 in the code
  auto desired = AxTensorsInterface{
    AxTensorInterface{ { 1, 400, 300, 3 }, 1, nullptr },
  };
  Ax::BatchedBufferPool pool(batch_size, desired, *allocator);

  auto batched = pool.new_batched_buffer();
  const auto &managed = batched->get_batched();
  EXPECT_EQ("tensors/5,400,300,3[1 byte]", Ax::to_string(managed.data()));
  check_dmabuf_mapped_consistency(managed);
  batched->get_batched(true);
  check_dmabuf_unmapped_consistency(managed);
  batched->map();
  check_dmabuf_mapped_consistency(managed);
}

namespace
{
void
init_gvalue(GValue *value, std::string * = nullptr)
{
  g_value_init(value, G_TYPE_STRING);
}

void
set_gvalue(GValue *value, std::string val)
{
  g_value_init(value, G_TYPE_STRING);
  g_value_set_string(value, val.c_str());
}
std::string
get_gvalue(GValue *value, std::string * = nullptr)
{
  EXPECT_TRUE(G_VALUE_HOLDS_STRING(value));
  return g_value_get_string(value);
}

void
init_gvalue(GValue *value, bool * = nullptr)
{
  g_value_init(value, G_TYPE_BOOLEAN);
}

void
set_gvalue(GValue *value, bool val)
{
  g_value_init(value, G_TYPE_BOOLEAN);
  g_value_set_boolean(value, val);
}
bool
get_gvalue(GValue *value, bool * = nullptr)
{
  EXPECT_TRUE(G_VALUE_HOLDS_BOOLEAN(value));
  return g_value_get_boolean(value);
}

void
init_gvalue(GValue *value, int * = nullptr)
{
  g_value_init(value, G_TYPE_INT);
}
void
set_gvalue(GValue *value, int val)
{
  g_value_init(value, G_TYPE_INT);
  g_value_set_int(value, val);
}
int
get_gvalue(GValue *value, int * = nullptr)
{
  EXPECT_TRUE(G_VALUE_HOLDS_INT(value));
  return g_value_get_int(value);
}
} // namespace

template <typename T>
void
test_inference_property(Ax::InferenceProperties &props, int prop_id, T &prop,
    const T &new_value, const T &default_value = {})
{
  GValue old_val{};
  init_gvalue(&old_val, static_cast<T *>(nullptr));
  EXPECT_TRUE(get_inference_property(props, prop_id, &old_val));
  EXPECT_EQ(default_value, get_gvalue(&old_val, static_cast<T *>(nullptr)));

  GValue new_val{};
  set_gvalue(&new_val, new_value);
  EXPECT_TRUE(Ax::set_inference_property(props, prop_id, &new_val));
  EXPECT_EQ(new_value, prop);

  GValue changed_value{};
  init_gvalue(&changed_value, static_cast<T *>(nullptr));
  EXPECT_TRUE(get_inference_property(props, prop_id, &changed_value));
  EXPECT_EQ(new_value, get_gvalue(&changed_value, static_cast<T *>(nullptr)));
}


TEST(axstreamer_utils, set_inference_property)
{
  Ax::InferenceProperties props;
  test_inference_property(props, Ax::AXINFERENCE_PROP_MODEL, props.model, "model"s);
  test_inference_property(props, Ax::AXINFERENCE_PROP_DMABUF_INPUTS,
      props.dmabuf_inputs, true, false);
  test_inference_property(props, Ax::AXINFERENCE_PROP_DMABUF_OUTPUTS,
      props.dmabuf_outputs, true, false);
  test_inference_property(props, Ax::AXINFERENCE_PROP_DOUBLE_BUFFER,
      props.double_buffer, true, false);
  test_inference_property(
      props, Ax::AXINFERENCE_PROP_NUM_CHILDREN, props.num_children, 4, 0);

  auto type_string = static_cast<std::string *>(nullptr);

  GValue skip_old{};
  init_gvalue(&skip_old, type_string);
  Ax::get_inference_property(props, Ax::AXINFERENCE_PROP_INFERENCE_SKIP_RATE, &skip_old);
  EXPECT_EQ("0/1", get_gvalue(&skip_old, type_string));

  GValue skip_new{};
  set_gvalue(&skip_new, "3/5"s);
  EXPECT_TRUE(Ax::set_inference_property(
      props, Ax::AXINFERENCE_PROP_INFERENCE_SKIP_RATE, &skip_new));
  EXPECT_EQ(3, props.skip_count);
  EXPECT_EQ(5, props.skip_stride);

  GValue skip_changed{};
  init_gvalue(&skip_changed, static_cast<std::string *>(nullptr));
  EXPECT_TRUE(get_inference_property(
      props, Ax::AXINFERENCE_PROP_INFERENCE_SKIP_RATE, &skip_changed));
  EXPECT_EQ("3/5"s, get_gvalue(&skip_changed, static_cast<std::string *>(nullptr)));

  GValue val;
  EXPECT_FALSE(Ax::set_inference_property(props, Ax::AXINFERENCE_PROP_NEXT_AVAILABLE, &val));
  EXPECT_FALSE(get_inference_property(props, Ax::AXINFERENCE_PROP_NEXT_AVAILABLE, &val));
}

static GstTensorsConfig
string_to_gst_tensors_config(std::string s)
{
  GstTensorsConfig cfg{};
  for (auto &&t : Ax::Internal::split(s, ',')) {
    auto &tensor = cfg.info.info[cfg.info.num_tensors++];
    int n = 0;
    for (auto &&d : Ax::Internal::split(t, ':')) {
      tensor.dimension[n++] = std::stoi(std::string{ d });
    }
    tensor.type = tensor_type::INT8;
  }
  return cfg;
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_ok_1_tensor)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1");
  AxTensorInterface ax_tensor = { { 1024 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor };
  Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_ok_2_tensors)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1,3:640:480:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1024 }, 1, nullptr, 0 };
  AxTensorInterface ax_tensor1 = { { 480, 640, 3 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0, ax_tensor1 };
  Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_ok_similar_shape)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1,3:640:480:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1, 1024 }, 1, nullptr, 0 }; // extra 1 should be ignored
  AxTensorInterface ax_tensor1 = { { 480, 640, 3 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0, ax_tensor1 };
  Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
}


TEST(axstreamer_utils, ensure_input_tensors_compatible_wrong_num)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1,3:640:480:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1024 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0 };
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()), "Input num tensors 2 != Model num tensors 1");
  }
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_too_many_tensors)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { {
                                       1,
                                   },
    1, nullptr, 0 };
  AxTensorsInterface ax_tensors(33, ax_tensor0);
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()), "Model has more inputs (33) than supported (32)");
  }
}


TEST(axstreamer_utils, ensure_input_tensors_compatible_too_many_dimensions)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0 };
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()),
        "Model input #0 (1,2,3,4,5,6,7,8,9) has more dimensions than supported (8)");
  }
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_wrong_first_shape)
{
  auto cfg = string_to_gst_tensors_config("1000:1:1:1:1:1:1:1,3:640:480:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1024 }, 1, nullptr, 0 };
  AxTensorInterface ax_tensor1 = { { 480, 640, 3 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0, ax_tensor1 };
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()), "Model input #0 shape (1024) != given input shape (1000)");
  }
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_wrong_second_shape)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1,3:640:480:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1024 }, 1, nullptr, 0 };
  AxTensorInterface ax_tensor1 = { { 480 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0, ax_tensor1 };
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()),
        "Model input #1 shape (480) != given input shape (480,640,3)");
  }
}

TEST(axstreamer_utils, ensure_input_tensors_compatible_wrong_second_shape2)
{
  auto cfg = string_to_gst_tensors_config("1024:1:1:1:1:1:1:1,640:1:1:1:1:1:1:1");
  AxTensorInterface ax_tensor0 = { { 1024 }, 1, nullptr, 0 };
  AxTensorInterface ax_tensor1 = { { 480, 640, 3 }, 1, nullptr, 0 };
  AxTensorsInterface ax_tensors = { ax_tensor0, ax_tensor1 };
  try {
    Ax::ensure_input_tensors_compatible(cfg, ax_tensors);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    ASSERT_EQ(std::string(e.what()),
        "Model input #1 shape (480,640,3) != given input shape (1,1,640)");
  }
}


TEST(axinferencenet, properties_from_string)
{
  Ax::Logger logger{ Ax::Severity::trace, nullptr, nullptr };
  auto props = Ax::properties_from_string(
      "model=build/ces2025-ls/yolov8sseg-coco-onnx/1/model.json\n"
      "devices=metis-0:3:0\n"
      "double_buffer=True\n"
      "dmabuf_inputs=1\n"
      "dmabuf_outputs=true\n"
      "num_children=3\n"
      "options=blahblah\n"
      "meta=master_detections\n"
      "preprocess0_lib=libtransform_roicrop.so\n"
      "preprocess0_options=meta_key:master_detections\n"
      "preprocess1_lib=libtransform_resize_cl.so\n"
      "preprocess1_options=width:640;height:640;blah\n"
      "preprocess1_batch=1\n"
      "postprocess0_lib=libdecode_yolov8seg.so\n"
      "postprocess0_options=meta_key:segmentations;\n"
      "postprocess0_mode=read\n"
      "postprocess1_lib=libinplace_nms.so\n"
      "postprocess1_options=nms_threshold:0.45;class_agnostic:0;location:CPU;\n",
      logger);

  EXPECT_EQ(props.model, "build/ces2025-ls/yolov8sseg-coco-onnx/1/model.json");
  EXPECT_EQ(props.devices, "metis-0:3:0");
  EXPECT_EQ(props.double_buffer, true);
  EXPECT_EQ(props.dmabuf_inputs, true);
  EXPECT_EQ(props.dmabuf_outputs, true);
  EXPECT_EQ(props.skip_stride, 1);
  EXPECT_EQ(props.skip_count, 0);
  EXPECT_EQ(props.num_children, 3);
  EXPECT_EQ(props.options, "blahblah");
  EXPECT_EQ(props.meta, "master_detections");
  EXPECT_EQ(props.preproc[0].lib, "libtransform_roicrop.so");
  EXPECT_EQ(props.preproc[0].options, "meta_key:master_detections");
  EXPECT_EQ(props.preproc[0].mode, "");
  EXPECT_EQ(props.preproc[1].lib, "libtransform_resize_cl.so");
  EXPECT_EQ(props.preproc[1].options, "width:640;height:640;blah");
  EXPECT_EQ(props.preproc[1].batch, "1");
  EXPECT_EQ(props.postproc[0].lib, "libdecode_yolov8seg.so");
  EXPECT_EQ(props.postproc[0].options, "meta_key:segmentations;");
  EXPECT_EQ(props.postproc[0].mode, "read");
  EXPECT_EQ(props.postproc[1].lib, "libinplace_nms.so");
  EXPECT_EQ(props.postproc[1].options, "nms_threshold:0.45;class_agnostic:0;location:CPU;");
}
