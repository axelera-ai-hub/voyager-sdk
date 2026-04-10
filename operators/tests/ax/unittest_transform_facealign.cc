// Copyright Axelera AI, 2025
#include "AxMetaKptsDetection.hpp"
#include "AxStreamerUtils.hpp"
#include "unittest_ax_common.h"

namespace
{
const auto facealign_lib = "facealign";

// Create a simple keypoints meta class for testing
class TestMetaKpts : public AxMetaKpts
{
  public:
  TestMetaKpts(const std::vector<KptXyv> &kpts, int num_subframes = 1)
      : AxMetaKpts(kpts),
        num_subframes_(num_subframes)
  {
    std::cout << "TestMetaKpts created with " << kpts.size() << " keypoints" << std::endl;
  }

  size_t get_number_of_subframes() const override
  {
    return num_subframes_;
  }

  private:
  int num_subframes_;
};

// Create a simple bbox meta class that includes keypoints
class TestMetaBbox : public AxMetaBbox
{
  public:
  TestMetaBbox(const std::vector<box_xyxy> &boxes)
      : AxMetaBbox(),
        boxes_(boxes) // Call the base class constructor properly
  {
    std::cout << "TestMetaBbox created with " << boxes.size() << " boxes" << std::endl;

    // Initialize the base class's bboxvec with our boxes
    for (const auto &box : boxes_) {
      bboxvec.push_back(box);
    }
  }

  // Version that takes a map of submetas by value and moves them
  TestMetaBbox(const std::vector<box_xyxy> &boxes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> submetas)
      : AxMetaBbox(),
        boxes_(boxes) // Call the base class constructor properly
  {
    std::cout << "TestMetaBbox created with " << boxes.size()
              << " boxes and submetas" << std::endl;

    // Initialize the base class's bboxvec with our boxes
    for (const auto &box : boxes_) {
      bboxvec.push_back(box);
    }

    // Move each element from submetas into our own map
    for (auto &pair : submetas) {
      std::cout << "Adding submeta: " << pair.first << " of type "
                << typeid(*(pair.second)).name() << std::endl;
      // For each keypoint submeta, we need to insert it properly
      try {
        auto *kpts_meta = dynamic_cast<AxMetaKpts *>(pair.second.get());
        if (kpts_meta) {
          std::cout << "Keypoints meta has " << kpts_meta->num_elements()
                    << " elements" << std::endl;
        }
        // Use the base class method to insert submeta
        insert_submeta(pair.first, 0, boxes_.size(),
            std::shared_ptr<AxMetaBase>(pair.second.release()));
      } catch (const std::exception &e) {
        std::cerr << "Exception during submeta insertion: " << e.what() << std::endl;
        throw;
      }
    }
  }

  size_t get_number_of_subframes() const override
  {
    std::cout << "Getting number of subframes: " << boxes_.size() << std::endl;
    return boxes_.size();
  }

  int get_id(size_t i) const
  {
    if (i >= boxes_.size()) {
      std::cerr << "Box ID index out of range: " << i
                << " (max: " << boxes_.size() - 1 << ")" << std::endl;
      throw std::out_of_range("Box ID index out of range");
    }
    return static_cast<int>(i);
  }

  private:
  std::vector<box_xyxy> boxes_;
};

// Test basic output interface setting
TEST(transform_facealign, output_interface_set)
{
  int out_width = 112;
  int out_height = 112;
  std::unordered_map<std::string, std::string> input
      = { { "width", std::to_string(out_width) },
          { "height", std::to_string(out_height) }, { "master_meta", "face_boxes" } };

  auto xform = Ax::LoadTransform(facealign_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_interface = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_interface).info;

  EXPECT_EQ(info.width, out_width);
  EXPECT_EQ(info.height, out_height);
}

// Test self-normalizing mode with 5-point landmarks
TEST(transform_facealign, self_normalizing_five_points)
{
  int out_width = 112;
  int out_height = 112;
  std::unordered_map<std::string, std::string> input = { { "width", std::to_string(out_width) },
    { "height", std::to_string(out_height) }, { "master_meta", "face_boxes" },
    { "keypoints_submeta_key", "face_landmarks" }, { "use_self_normalizing", "1" } };

  try {
    auto xform = Ax::LoadTransform(facealign_lib, input);

    // Create a 100x100 test image (gray for simplicity)
    auto in_buf = std::vector<uint8_t>(100 * 100, 100); // Gray background
    auto out_buf = std::vector<uint8_t>(out_width * out_height, 0); // Black output

    // Skip drawing features - they don't affect the transformation itself

    AxVideoInterface in_info{ { 100, 100, 100, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
    AxVideoInterface out_info{ { out_width, out_height, out_width, 0, AxVideoFormat::GRAY8 },
      out_buf.data() };

    // Create keypoints for the 5 landmarks - use simpler values
    std::vector<KptXyv> kpts;
    // Left eye
    kpts.push_back(KptXyv(30.0f, 30.0f, 1.0f));
    // Right eye
    kpts.push_back(KptXyv(70.0f, 30.0f, 1.0f));
    // Nose
    kpts.push_back(KptXyv(50.0f, 50.0f, 1.0f));
    // Left mouth corner
    kpts.push_back(KptXyv(30.0f, 70.0f, 1.0f));
    // Right mouth corner
    kpts.push_back(KptXyv(70.0f, 70.0f, 1.0f));

    std::cout << "Created " << kpts.size() << " keypoints" << std::endl;

    // Create a bounding box
    std::vector<box_xyxy> boxes;
    boxes.push_back(box_xyxy(10.0f, 10.0f, 90.0f, 90.0f)); // Face bounding box

    // Create keypoint meta with the same number of subframes as boxes
    auto kpts_meta = std::make_unique<TestMetaKpts>(kpts, boxes.size());

    // Create the metadata
    Ax::MetaMap metadata;

    // Add submeta to a temporary map
    Ax::MetaMap submetas;
    submetas.emplace("face_landmarks", std::move(kpts_meta));

    // Create the bbox meta with submeta - we move submetas here
    auto bbox_meta = std::make_unique<TestMetaBbox>(boxes, std::move(submetas));
    metadata.emplace("face_boxes", std::move(bbox_meta));

    std::cout << "About to call transform" << std::endl;

    // Perform the transformation
    xform->transform(in_info, out_info, 0, 1, metadata);

    std::cout << "Transform completed" << std::endl;

    // Check that there are non-zero pixels in the output (transformation occurred)
    bool has_nonzero = false;
    for (auto val : out_buf) {
      if (val > 0) {
        has_nonzero = true;
        break;
      }
    }

    EXPECT_TRUE(has_nonzero) << "Output image should not be all zeros";
    std::cout << "Test completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    FAIL() << "Exception thrown: " << e.what();
  }
}

// Test template-based alignment
TEST(transform_facealign, template_based)
{
  int out_width = 112;
  int out_height = 112;
  float padding = 0.2f;
  std::unordered_map<std::string, std::string> input
      = { { "width", std::to_string(out_width) },
          { "height", std::to_string(out_height) }, { "master_meta", "face_boxes" },
          { "keypoints_submeta_key", "face_landmarks" },
          { "padding", std::to_string(padding) }, { "use_self_normalizing", "0" } };

  try {
    auto xform = Ax::LoadTransform(facealign_lib, input);

    // Create a 100x100 test image (gray for simplicity)
    auto in_buf = std::vector<uint8_t>(100 * 100, 100); // Gray background
    auto out_buf = std::vector<uint8_t>(out_width * out_height, 0); // Black output

    // Simplified drawing - just fill the buffer with gray

    AxVideoInterface in_info{ { 100, 100, 100, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
    AxVideoInterface out_info{ { out_width, out_height, out_width, 0, AxVideoFormat::GRAY8 },
      out_buf.data() };

    // Create keypoints for the 5 landmarks
    std::vector<KptXyv> kpts;
    // Add 5 keypoints with simple values
    kpts.push_back(KptXyv(30.0f, 30.0f, 1.0f));
    kpts.push_back(KptXyv(40.0f, 40.0f, 1.0f));
    kpts.push_back(KptXyv(50.0f, 50.0f, 1.0f));
    kpts.push_back(KptXyv(60.0f, 60.0f, 1.0f));
    kpts.push_back(KptXyv(70.0f, 70.0f, 1.0f));

    std::cout << "Created " << kpts.size()
              << " keypoints for template-based test" << std::endl;

    // Create a bounding box
    std::vector<box_xyxy> boxes;
    boxes.push_back(box_xyxy(10.0f, 10.0f, 90.0f, 90.0f)); // Face bounding box

    // Create keypoint meta with the same number of subframes as boxes
    auto kpts_meta = std::make_unique<TestMetaKpts>(kpts, boxes.size());

    // Create the metadata
    Ax::MetaMap metadata;

    // Add submeta to a temporary map
    Ax::MetaMap submetas;
    submetas.emplace("face_landmarks", std::move(kpts_meta));

    // Create the bbox meta with submeta
    auto bbox_meta = std::make_unique<TestMetaBbox>(boxes, std::move(submetas));
    metadata.emplace("face_boxes", std::move(bbox_meta));

    std::cout << "About to call transform for template-based test" << std::endl;

    // Perform the transformation
    xform->transform(in_info, out_info, 0, 1, metadata);

    std::cout << "Template-based transform completed" << std::endl;

    // Check that there are non-zero pixels in the output (transformation occurred)
    bool has_nonzero = false;
    for (auto val : out_buf) {
      if (val > 0) {
        has_nonzero = true;
        break;
      }
    }

    EXPECT_TRUE(has_nonzero) << "Output image should not be all zeros";
    std::cout << "Template-based test completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception caught in template-based test: " << e.what() << std::endl;
    FAIL() << "Exception thrown in template-based test: " << e.what();
  }
}

// Test with custom template keypoints
TEST(transform_facealign, custom_template)
{
  int out_width = 112;
  int out_height = 112;
  float padding = 0.2f;

  // Create custom template keypoints (5 points)
  std::vector<float> template_x = { 0.2f, 0.4f, 0.5f, 0.3f, 0.7f };
  std::vector<float> template_y = { 0.3f, 0.3f, 0.5f, 0.7f, 0.7f };

  // Convert vectors to comma-separated strings
  std::string template_x_str;
  std::string template_y_str;

  for (size_t i = 0; i < template_x.size(); ++i) {
    if (i > 0) {
      template_x_str += ",";
      template_y_str += ",";
    }
    template_x_str += std::to_string(template_x[i]);
    template_y_str += std::to_string(template_y[i]);
  }

  std::unordered_map<std::string, std::string> input = { { "width", std::to_string(out_width) },
    { "height", std::to_string(out_height) }, { "master_meta", "face_boxes" },
    { "keypoints_submeta_key", "face_landmarks" },
    { "padding", std::to_string(padding) }, { "template_keypoints_x", template_x_str },
    { "template_keypoints_y", template_y_str }, { "use_self_normalizing", "0" } };

  try {
    auto xform = Ax::LoadTransform(facealign_lib, input);

    // Create a 100x100 test image (gray for simplicity)
    auto in_buf = std::vector<uint8_t>(100 * 100, 100); // Gray background
    auto out_buf = std::vector<uint8_t>(out_width * out_height, 0); // Black output

    AxVideoInterface in_info{ { 100, 100, 100, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
    AxVideoInterface out_info{ { out_width, out_height, out_width, 0, AxVideoFormat::GRAY8 },
      out_buf.data() };

    // Create keypoints for the 5 landmarks
    std::vector<KptXyv> kpts;
    kpts.push_back(KptXyv(30.0f, 30.0f, 1.0f));
    kpts.push_back(KptXyv(40.0f, 40.0f, 1.0f));
    kpts.push_back(KptXyv(50.0f, 50.0f, 1.0f));
    kpts.push_back(KptXyv(60.0f, 60.0f, 1.0f));
    kpts.push_back(KptXyv(70.0f, 70.0f, 1.0f));

    std::cout << "Created " << kpts.size()
              << " keypoints for custom template test" << std::endl;

    // Create a bounding box
    std::vector<box_xyxy> boxes;
    boxes.push_back(box_xyxy(10.0f, 10.0f, 90.0f, 90.0f)); // Face bounding box

    // Create keypoint meta with the same number of subframes as boxes
    auto kpts_meta = std::make_unique<TestMetaKpts>(kpts, boxes.size());

    // Create the metadata
    Ax::MetaMap metadata;

    // Add submeta to a temporary map
    Ax::MetaMap submetas;
    submetas.emplace("face_landmarks", std::move(kpts_meta));

    // Create the bbox meta with submeta
    auto bbox_meta = std::make_unique<TestMetaBbox>(boxes, std::move(submetas));
    metadata.emplace("face_boxes", std::move(bbox_meta));

    std::cout << "About to call transform for custom template test" << std::endl;

    // Perform the transformation
    xform->transform(in_info, out_info, 0, 1, metadata);

    std::cout << "Custom template transform completed" << std::endl;

    // Check that there are non-zero pixels in the output (transformation occurred)
    bool has_nonzero = false;
    for (auto val : out_buf) {
      if (val > 0) {
        has_nonzero = true;
        break;
      }
    }

    EXPECT_TRUE(has_nonzero) << "Output image should not be all zeros";
    std::cout << "Custom template test completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception caught in custom template test: " << e.what() << std::endl;
    FAIL() << "Exception thrown in custom template test: " << e.what();
  }
}

// Test error handling when no keypoints are found
TEST(transform_facealign, no_keypoints)
{
  int out_width = 112;
  int out_height = 112;
  std::unordered_map<std::string, std::string> input
      = { { "width", std::to_string(out_width) },
          { "height", std::to_string(out_height) }, { "master_meta", "face_boxes" },
          { "keypoints_submeta_key", "nonexistent_landmarks" }, // Nonexistent key
          { "use_self_normalizing", "1" } };

  try {
    auto xform = Ax::LoadTransform(facealign_lib, input);

    // Create a 100x100 test image
    auto in_buf = std::vector<uint8_t>(100 * 100, 100);
    auto out_buf = std::vector<uint8_t>(out_width * out_height, 0);

    AxVideoInterface in_info{ { 100, 100, 100, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
    AxVideoInterface out_info{ { out_width, out_height, out_width, 0, AxVideoFormat::GRAY8 },
      out_buf.data() };

    // Create a bounding box without keypoints
    std::vector<box_xyxy> boxes;
    boxes.push_back(box_xyxy(10.0f, 10.0f, 90.0f, 90.0f));

    // Create the metadata
    Ax::MetaMap metadata;

    // Create the bbox meta without submeta
    auto bbox_meta = std::make_unique<TestMetaBbox>(boxes);
    metadata.emplace("face_boxes", std::move(bbox_meta));

    std::cout << "About to call transform for NoKeypoints test (expecting exception)"
              << std::endl;

    // The transform should throw an exception
    bool exception_thrown = false;
    try {
      xform->transform(in_info, out_info, 0, 1, metadata);
    } catch (const std::exception &e) {
      exception_thrown = true;
      std::cout << "Exception caught as expected: " << e.what() << std::endl;
    }

    EXPECT_TRUE(exception_thrown) << "Expected an exception when no keypoints are found";
    std::cout << "NoKeypoints test completed successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Unexpected exception in test setup: " << e.what() << std::endl;
    FAIL() << "Unexpected exception in test setup: " << e.what();
  }
}

} // namespace
