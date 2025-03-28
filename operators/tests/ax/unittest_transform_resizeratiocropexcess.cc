#include "unittest_transform_common.h"

namespace
{
const auto resize_lib = "libtransform_resizeratiocropexcess.so";

TEST(resizeratiocropexcess, output_size)
{
  std::unordered_map<std::string, std::string> input = {
    { "resize_size", "256" },
    { "final_size_after_crop", "224" },
  };
  Transformer resizer(resize_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  auto out_interface = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, 224);
  EXPECT_EQ(info.height, 224);
}

TEST(resizeratiocropexcess, cropping)
{
  std::unordered_map<std::string, std::string> input = {
    { "resize_size", "4" },
    { "final_size_after_crop", "2" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
    114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(2 * 2 * 4, 255);
  AxVideoInterface in_info{ { 4, 6, 4 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 2, 2, 2 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>(2 * 2 * 4, 33);
  ASSERT_EQ(out_buf, expected);
}

} // namespace
