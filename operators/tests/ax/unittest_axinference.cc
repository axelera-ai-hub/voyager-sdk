// Copyright Axelera AI, 2024
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include "unittest_ax_common.h"

const auto statechange_timeout = 2000U;

using namespace std::string_literals;

class AxInferenceFixture : public ::testing::TestWithParam<bool>
{
  protected:
  GstElement *gstpipe = nullptr;
  GstElement *sink_handle = nullptr;

  static void CheckOutput(GstElement *element, GstBuffer *buffer, gpointer user_data)
  {
    GstMapInfo info_res{};
    auto mem_res = gst_buffer_peek_memory(buffer, 0);
    ASSERT_TRUE(gst_memory_map(mem_res, &info_res, GST_MAP_READ));
    auto output = static_cast<const uint8_t *>(info_res.data);
    for (guint i = 0; i < 10; i++) {
      EXPECT_EQ(0xa0, output[i]); // 0xa0 is magic value returned by mock when no crc matched
    }
    gst_memory_unmap(mem_res, &info_res);
  }

  void SetUp() override
  {
    auto *current_path = getenv("GST_PLUGIN_PATH");
    std::string plugin_path = g_get_current_dir();
    plugin_path = plugin_path + ":" + plugin_path + "/axstreamer";
    if (current_path == NULL) {
      setenv("GST_PLUGIN_PATH", plugin_path.c_str(), 1);
      std::cerr << "GST_PLUGIN_PATH not set, setting to current directory: " << plugin_path
                << std::endl;
    } else {
      std::string path = plugin_path + ":" + current_path;
      setenv("GST_PLUGIN_PATH", path.c_str(), 1);
    }
    gst_init(NULL, NULL);
  }

  void TearDown() override
  {
    if (gstpipe)
      g_clear_object(&gstpipe);
    if (sink_handle)
      g_clear_object(&sink_handle);
  }
};

int
setPipelineStateSync(GstElement *pipeline, GstState state, uint32_t timeout_ms)
{
  GstState cur_state = GST_STATE_VOID_PENDING;
  GstStateChangeReturn ret;
  guint counter = 0;
  ret = gst_element_set_state(pipeline, state);

  if (ret == GST_STATE_CHANGE_FAILURE)
    return -1;

  do {
    ret = gst_element_get_state(pipeline, &cur_state, NULL, 10 * GST_MSECOND);
    if (ret == GST_STATE_CHANGE_FAILURE)
      return -2;
    if (cur_state == state)
      return 0;
    g_usleep(10000);
  } while ((timeout_ms / 20) > counter++);
  return -ETIME;
}

TEST_P(AxInferenceFixture, HappyPathTest)
{
  if (not has_dma_heap()) {
    GTEST_SKIP() << "Skipping test as dma heap is not available";
  }
  const auto use_dmabuf_param = GetParam();
  const auto use_dmabuf = use_dmabuf_param ? "true" : "false";
  const auto pipeline = "videotestsrc num-buffers=1 "
                        "! video/x-raw,format=RGBA,width=480,height=640 "
                        "! axtransform lib=libtransform_resize.so options=to_tensor:1 "
                        "! axinference dmabuf_inputs="s
                        + use_dmabuf + " model="s + __FILE__
                        + " options=mock-load:1;mock-shapes:1x640x480x4,1x640x480x4 "
                          "! appsink name=sink"s;
  gstpipe = gst_parse_launch(pipeline.c_str(), nullptr);
  EXPECT_NE(gstpipe, nullptr);

  sink_handle = gst_bin_get_by_name(GST_BIN(gstpipe), "sink");
  EXPECT_NE(sink_handle, nullptr);
  g_signal_connect(sink_handle, "new-sample",
      (GCallback) AxInferenceFixture::CheckOutput, NULL);

  EXPECT_EQ(setPipelineStateSync(gstpipe, GST_STATE_PLAYING, statechange_timeout), 0);
  EXPECT_EQ(setPipelineStateSync(gstpipe, GST_STATE_NULL, statechange_timeout), 0);
}

// Define a test suite with parameters, dmabuf flag
// On CPU models we don't use dmabuf
INSTANTIATE_TEST_SUITE_P(AxInferenceTestSuite, AxInferenceFixture, ::testing::Values(false));
