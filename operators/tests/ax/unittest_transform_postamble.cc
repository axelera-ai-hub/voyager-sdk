// Copyright Axelera AI, 2025
#include <algorithm>
#include <gmock/gmock.h>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include "unittest_ax_common.h"

/*
 * IMPORTANT: Test ONNX Model Constraints
 * =====================================
 * The add_one.onnx model expects fixed input dimensions: [batch, 3, 20, 20]
 * The add_tensors.onnx model expects two inputs with dimensions: [batch, 3, 20, 20] each
 *
 * Tests that use ONNX processing MUST provide tensors matching these exact dimensions,
 * otherwise they will fail with dimension mismatch errors.
 *
 * For tests requiring flexible dimensions, use passthrough mode (no onnx_path).
 */

class TransformPostambleTest : public ::testing::Test
{
  protected:
  std::string get_onnx_file()
  {
    std::filesystem::path current_path = std::filesystem::current_path();
    return current_path / "assets/add_one.onnx";
  }

  std::string get_multi_input_onnx_file()
  {
    std::filesystem::path current_path = std::filesystem::current_path();
    return current_path / "assets/add_tensors.onnx";
  }
};

// =============================================================================
// Basic Initialization and Error Handling Tests
// =============================================================================

TEST_F(TransformPostambleTest, valid_onnx_file_initialization)
{
  std::string onnx_path = get_onnx_file();

  // Should initialize successfully with valid ONNX file
  EXPECT_NO_THROW(Ax::LoadTransform("postamble", { { "onnx_path", onnx_path } }));
}

TEST_F(TransformPostambleTest, initialization_parameter_validation_errors)
{
  std::string onnx_path = get_onnx_file();

  // Test mismatch between dequant_scale and dequant_zeropoint sizes
  std::unordered_map<std::string, std::string> input_mismatch = {
    { "onnx_path", onnx_path }, { "dequant_scale", "0.5,0.25" },
    { "dequant_zeropoint", "1" } // Only one value, scale has two
  };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_mismatch), std::logic_error);

  // Test only scale provided (missing zeropoint)
  std::unordered_map<std::string, std::string> input_only_scale = {
    { "onnx_path", onnx_path }, { "dequant_scale", "0.5" }
    // Missing dequant_zeropoint
  };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_only_scale), std::logic_error);

  // Test only zeropoint provided (missing scale)
  std::unordered_map<std::string, std::string> input_only_zeropoint = {
    { "onnx_path", onnx_path }, { "dequant_zeropoint", "10" }
    // Missing dequant_scale
  };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_only_zeropoint), std::logic_error);

  // Test no dequant params (should be fine)
  std::unordered_map<std::string, std::string> input_no_dequant
      = { { "onnx_path", onnx_path } };
  EXPECT_NO_THROW(Ax::LoadTransform("postamble", input_no_dequant));
}

TEST_F(TransformPostambleTest, tensor_selection_plan_validation)
{
  std::string onnx_path = get_onnx_file(); // Expects 1 input

  // Test tensor selection plan with wrong size (too many)
  std::unordered_map<std::string, std::string> input_wrong_plan_size
      = { { "onnx_path", onnx_path },
          { "tensor_selection_plan", "0,1" }, // ONNX model only has 1 input
          { "dequant_scale", "1.0,1.0" }, { "dequant_zeropoint", "0,0" } };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_wrong_plan_size), std::runtime_error);

  // Test missing dequant params for selected tensor
  std::unordered_map<std::string, std::string> input_missing_dequant
      = { { "onnx_path", onnx_path }, { "tensor_selection_plan", "1" }, // Selects tensor index 1
          { "dequant_scale", "1.0" }, // Only provides params for tensor 0
          { "dequant_zeropoint", "0" } };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_missing_dequant), std::runtime_error);
}

TEST_F(TransformPostambleTest, multi_input_onnx_validation)
{
  std::string onnx_path = get_multi_input_onnx_file(); // Expects 2 inputs

  // Test correct configuration
  std::unordered_map<std::string, std::string> input_correct
      = { { "onnx_path", onnx_path }, { "tensor_selection_plan", "0,1" },
          { "dequant_scale", "1.0,1.0" }, { "dequant_zeropoint", "0,0" } };
  EXPECT_NO_THROW(Ax::LoadTransform("postamble", input_correct));

  // Test insufficient dequant params for multi-input
  std::unordered_map<std::string, std::string> input_insufficient
      = { { "onnx_path", onnx_path }, { "tensor_selection_plan", "0,1" },
          { "dequant_scale", "1.0" }, // Only 1 param but 2 tensors selected
          { "dequant_zeropoint", "0" } };
  EXPECT_THROW(Ax::LoadTransform("postamble", input_insufficient), std::runtime_error);
}

TEST(transform_postamble, null_data_pointer_error)
{
  // Test error handling during transform with null data pointer
  auto xform = Ax::LoadTransform("postamble", {});

  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, nullptr } }; // null data pointer
  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));

  auto out_data = std::vector<int8_t>(1 * 2 * 3 * 4);
  out[0].data = out_data.data();

  // Should fail during transform due to null input data pointer
  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(inp, out, 0, 1, metadata), std::runtime_error);
}

// =============================================================================
// Passthrough Mode Tests
// =============================================================================

TEST(transform_postamble, passthrough_mode_empty_config)
{
  std::unordered_map<std::string, std::string> input = {};
  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 2 * 3 * 4;
  auto inp_data = std::vector<int8_t>(size);
  std::iota(inp_data.begin(), inp_data.end(), 0);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 2, 3, 4 }));
  EXPECT_EQ(out[0].bytes, 1); // Remains int8 when no dequant params provided

  auto out_data = std::vector<int8_t>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Should copy input data unchanged
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(out_data[i], inp_data[i]);
  }
}

TEST(transform_postamble, passthrough_mode_no_onnx_with_dequant)
{
  // True passthrough but with dequantization
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "0.5" }, { "dequant_zeropoint", "10" } };
  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 2 * 2 * 2;
  auto inp_data = std::vector<int8_t>(size, 20); // All values = 20
  AxTensorsInterface inp{ { { 1, 2, 2, 2 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  EXPECT_EQ(out[0].bytes, 4); // Should be float after dequantization

  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Expected: 0.5 * (20 - 10) = 5.0
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(5.0f)));
}

TEST(transform_postamble, passthrough_float_input)
{
  // Passthrough with float input (no dequantization parameters)
  std::unordered_map<std::string, std::string> input = {};
  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 2 * 3 * 4;
  auto inp_data = std::vector<float>(size, 3.5f);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 4, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  EXPECT_EQ(out[0].bytes, 4); // Should remain float

  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Should copy float data unchanged
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(3.5f)));
}

TEST(transform_postamble, mixed_input_types_passthrough)
{
  // Mixed int8 and float inputs in passthrough mode
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "1.0" }, // Only dequant first tensor
          { "dequant_zeropoint", "0" } };
  auto xform = Ax::LoadTransform("postamble", input);

  auto inp1_data = std::vector<int8_t>(1 * 2 * 2 * 2, 5); // Will be dequantized
  auto inp2_data = std::vector<float>(1 * 3 * 3 * 3, 7.5f); // Float passthrough (no dequant)

  AxTensorsInterface inp{ { { 1, 2, 2, 2 }, 1, inp1_data.data() },
    { { 1, 3, 3, 3 }, 4, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].bytes, 4); // Dequantized to float
  EXPECT_EQ(out[1].bytes, 4); // Remains float

  auto out1_data = std::vector<float>(1 * 2 * 2 * 2);
  auto out2_data = std::vector<float>(1 * 3 * 3 * 3);
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // First tensor: 1.0 * (5 - 0) = 5.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(5.0f)));
  // Second tensor: unchanged (float passthrough, no dequant)
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(7.5f)));
}

// =============================================================================
// Basic ONNX Processing Tests
// =============================================================================

TEST_F(TransformPostambleTest, basic_onnx_processing)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Use correct dimensions for ONNX model: [1, 3, 20, 20]
  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<int8_t>(size, 1);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_GE(out.size(), 1);
  EXPECT_EQ(out[0].bytes, 4); // Float output

  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // add_one.onnx should add 1 to each input value
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(2.0f)));
}

TEST_F(TransformPostambleTest, onnx_with_float_input)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Use correct dimensions for ONNX model: [1, 3, 20, 20]
  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<float>(size, 5.5f);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 4, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // add_one.onnx should add 1 to each input value
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(6.5f)));
}

TEST_F(TransformPostambleTest, onnx_int8_input_without_dequant_should_fail)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }
    // No dequant params
  };

  auto xform = Ax::LoadTransform("postamble", input);

  // int8 input but no dequantization parameters
  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<int8_t>(size, 1);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  // Should fail because ONNX needs float but we have int8 without dequant
  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(inp, out, 0, 1, metadata), std::runtime_error);
}

// =============================================================================
// Padding and Validation Tests
// =============================================================================

TEST_F(TransformPostambleTest, padding_and_dequant_fixed)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }, { "tensor_selection_plan", "0" },
    { "dequant_scale", "0.5" }, { "dequant_zeropoint", "0" },
    // Padding format: [N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after]
    { "padding", "0,0,0,0,0,0,0,-1" } // Remove 1 from last dimension (depad from 21 to 20)
  };

  auto xform = Ax::LoadTransform("postamble", input);

  // Create padded input tensor [1,3,20,21] that should become [1,3,20,20] after depadding
  int padded_size = 1 * 3 * 20 * 21;
  auto inp_data = std::vector<int8_t>(padded_size, 2); // All values = 2

  AxTensorsInterface inp{ { { 1, 3, 20, 21 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].bytes, 4); // Float output
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 3, 20, 20 })); // Depadded to match ONNX

  int out_size = 1 * 3 * 20 * 20;
  auto out_data = std::vector<float>(out_size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Expected: 0.5 * (2 - 0) + 1 (from ONNX) = 2.0
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(2.0f)));
}

TEST_F(TransformPostambleTest, invalid_padding_dimensions)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }, { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" },
    { "padding", "0,0,0" } // Invalid - should have 8 values for 4D tensor
  };

  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<int8_t>(size, 1);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  // Should fail due to invalid padding dimensions
  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(inp, out, 0, 1, metadata), std::runtime_error);
}

// =============================================================================
// Transpose Tests
// =============================================================================

TEST_F(TransformPostambleTest, transpose_with_dequant_fixed)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" }, { "transpose", "1" } };

  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 20 * 20 * 3; // NHWC
  auto inp_data = std::vector<int8_t>(size, 5);
  AxTensorsInterface inp{ { { 1, 20, 20, 3 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 3, 20, 20 })); // NCHW
  EXPECT_EQ(out[0].bytes, 4);

  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // All values should be 5 + 1 = 6 after ONNX processing
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(6.0f)));
}

TEST_F(TransformPostambleTest, selective_transpose)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "dequant_scale", "1.0,1.0,1.0" }, { "dequant_zeropoint", "0,0,0" },
    { "transpose", "1,0,1" }, // First: transpose, Second: no transpose, Third: transpose
    { "tensor_selection_plan", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp1_data = std::vector<int8_t>(1 * 20 * 20 * 3, 1); // NHWC -> NCHW
  auto inp2_data = std::vector<int8_t>(1 * 3 * 10 * 10, 2); // NCHW -> NCHW
  auto inp3_data = std::vector<int8_t>(1 * 15 * 15 * 3, 3); // NHWC -> NCHW

  AxTensorsInterface inp{ { { 1, 20, 20, 3 }, 1, inp1_data.data() },
    { { 1, 3, 10, 10 }, 1, inp2_data.data() },
    { { 1, 15, 15, 3 }, 1, inp3_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 3);

  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 3, 20, 20 })); // Transposed
  EXPECT_EQ(out[1].sizes, std::vector<int>({ 1, 3, 10, 10 })); // Not transposed
  EXPECT_EQ(out[2].sizes, std::vector<int>({ 1, 3, 15, 15 })); // Transposed

  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out2_data = std::vector<float>(1 * 3 * 10 * 10);
  auto out3_data = std::vector<float>(1 * 15 * 15 * 3);
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();
  out[2].data = out3_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(2.0f))); // 1 + 1 from ONNX
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(2.0f))); // Just dequantized
  EXPECT_THAT(out3_data, testing::Each(testing::FloatEq(3.0f))); // Just dequantized
}

// =============================================================================
// Dequantization Tests
// =============================================================================

TEST_F(TransformPostambleTest, dequant_with_lut)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }, { "dequant_scale", "0.25" },
    { "dequant_zeropoint", "128" }, { "dequant_lut", "1" } // Use LUT
  };

  auto xform = Ax::LoadTransform("postamble", input);

  // Use correct dimensions for ONNX model: [1, 3, 20, 20]
  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<int8_t>(size, 0); // All zeros
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // With LUT: dequantized value for 0 should be 0.25 * (0 - 128) = -32.0, then +1 from ONNX = -31.0
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(-31.0f)));
}

TEST_F(TransformPostambleTest, dequant_without_lut)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }, { "dequant_scale", "0.25" },
    { "dequant_zeropoint", "128" }, { "dequant_lut", "0" } // Don't use LUT
  };

  auto xform = Ax::LoadTransform("postamble", input);

  // Use correct dimensions for ONNX model: [1, 3, 20, 20]
  int size = 1 * 3 * 20 * 20;
  auto inp_data = std::vector<int8_t>(size, 0);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Without LUT: 0.25 * (0 - 128) = -32.0, then +1 from ONNX = -31.0
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(-31.0f)));
}

TEST_F(TransformPostambleTest, different_dequant_params_per_tensor)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input
      = { { "onnx_path", onnx_path }, { "dequant_scale", "1.0,2.0" },
          { "dequant_zeropoint", "0,10" }, { "tensor_selection_plan", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // First tensor goes to ONNX (must match [1,3,20,20])
  auto inp1_data = std::vector<int8_t>(1 * 3 * 20 * 20, 5);
  // Second tensor is passthrough (can be different size)
  auto inp2_data = std::vector<int8_t>(1 * 2 * 10 * 10, 15);

  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp1_data.data() },
    { { 1, 2, 10, 10 }, 1, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2);

  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out2_data = std::vector<float>(1 * 2 * 10 * 10);
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // First tensor: 1.0 * (5 - 0) + 1 (ONNX) = 6.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(6.0f)));

  // Second tensor: 2.0 * (15 - 10) = 10.0 (passthrough, no ONNX)
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(10.0f)));
}

// =============================================================================
// Multi-Input ONNX Tests
// =============================================================================

TEST_F(TransformPostambleTest, multi_input_onnx)
{
  std::string onnx_path = get_multi_input_onnx_file();
  std::unordered_map<std::string, std::string> input = {
    { "onnx_path", onnx_path }, { "dequant_scale", "1.0,1.0,1.0" },
    { "dequant_zeropoint", "0,0,0" },
    { "tensor_selection_plan", "0,1" } // Use first two tensors for ONNX
  };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp1_data = std::vector<int8_t>(1 * 3 * 20 * 20, 1);
  auto inp2_data = std::vector<int8_t>(1 * 3 * 20 * 20, 2);
  auto inp3_data = std::vector<int8_t>(1 * 3 * 3 * 3, 5); // Unused

  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp1_data.data() },
    { { 1, 3, 20, 20 }, 1, inp2_data.data() }, { { 1, 3, 3, 3 }, 1, inp3_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2); // One ONNX output + one passthrough

  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out2_data = std::vector<float>(1 * 3 * 3 * 3);
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // add_tensors.onnx should add the two inputs: 1 + 2 = 3
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(3.0f)));
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(5.0f))); // Passthrough
}

// =============================================================================
// Error Condition Tests
// =============================================================================

TEST_F(TransformPostambleTest, tensor_selection_plan_out_of_bounds_runtime)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "tensor_selection_plan", "1" }, // Selects tensor index 1
    { "dequant_scale", "1.0,1.0" }, { "dequant_zeropoint", "0,0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Only provide one tensor (index 0), but selection plan wants tensor 1
  auto inp_data = std::vector<int8_t>(1 * 3 * 20 * 20, 1);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(1 * 3 * 20 * 20);
  out[0].data = out_data.data();

  // Should fail during transform due to insufficient tensors
  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(inp, out, 0, 1, metadata), std::runtime_error);
}

TEST_F(TransformPostambleTest, dequant_params_out_of_bounds_runtime)
{
  std::unordered_map<std::string, std::string> input = { { "dequant_scale", "1.0" }, // Only one parameter
    { "dequant_zeropoint", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Provide two tensors but only one set of dequant params
  auto inp1_data = std::vector<int8_t>(1 * 2 * 3 * 4, 5);
  auto inp2_data = std::vector<int8_t>(1 * 2 * 3 * 4, 10);

  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, inp1_data.data() },
    { { 1, 2, 3, 4 }, 1, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out1_data = std::vector<float>(1 * 2 * 3 * 4);
  auto out2_data = std::vector<int8_t>(1 * 2 * 3 * 4); // No dequant for second tensor
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // First tensor should be dequantized: 1.0 * (5 - 0) = 5.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(5.0f)));
  // Second tensor should be passed through unchanged
  EXPECT_THAT(out2_data, testing::Each(testing::Eq(10)));
}

TEST_F(TransformPostambleTest, wrong_tensor_type_for_dequant)
{
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Provide float tensor but it's marked for dequantization (should be int8)
  int size = 1 * 2 * 3 * 4;
  auto inp_data = std::vector<float>(size, 1.0f);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 4, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  // Should fail because tensor is marked for dequantization but not int8
  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(inp, out, 0, 1, metadata), std::runtime_error);
}

// =============================================================================
// Thread Configuration Tests
// =============================================================================

TEST_F(TransformPostambleTest, custom_thread_config)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "ort_intra_op_num_threads", "2" }, { "ort_inter_op_num_threads", "1" },
    { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" } };

  // Should initialize successfully with custom thread counts
  EXPECT_NO_THROW(auto xform = Ax::LoadTransform("postamble", input));

  auto xform = Ax::LoadTransform("postamble", input);

  // Use correct dimensions for ONNX model
  auto inp_data = std::vector<int8_t>(1 * 3 * 20 * 20, 1);
  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(1 * 3 * 20 * 20);
  out[0].data = out_data.data();

  // Should work normally
  Ax::MetaMap metadata;
  EXPECT_NO_THROW(xform->transform(inp, out, 0, 1, metadata));

  // Should get expected result: 1.0 * (1 - 0) + 1 = 2.0
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(2.0f)));
}

// =============================================================================
// Edge Cases and Boundary Tests (Using Passthrough Mode)
// =============================================================================

TEST_F(TransformPostambleTest, single_element_tensor_passthrough)
{
  // Use passthrough mode to avoid ONNX dimension constraints
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "1.0" }, { "dequant_zeropoint", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp_data = std::vector<int8_t>({ 42 });
  AxTensorsInterface inp{ { { 1, 1, 1, 1 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(1);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);
  EXPECT_FLOAT_EQ(out_data[0], 42.0f); // 42 - 0 = 42 (no ONNX processing)
}

TEST_F(TransformPostambleTest, large_tensor_passthrough)
{
  // Use passthrough mode for flexibility with dimensions
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "0.1" }, { "dequant_zeropoint", "0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  int size = 1 * 3 * 224 * 224; // Large tensor
  auto inp_data = std::vector<int8_t>(size, 10);
  AxTensorsInterface inp{ { { 1, 3, 224, 224 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(size);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Should be 0.1 * 10 = 1.0 (no ONNX processing)
  EXPECT_THAT(out_data, testing::Each(testing::FloatEq(1.0f)));
}

TEST_F(TransformPostambleTest, extreme_dequant_values_passthrough)
{
  // Use passthrough mode
  std::unordered_map<std::string, std::string> input
      = { { "dequant_scale", "1000.0" }, { "dequant_zeropoint", "-100" } };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp_data = std::vector<int8_t>({ 127 }); // Max int8 value
  AxTensorsInterface inp{ { { 1, 1, 1, 1 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  auto out_data = std::vector<float>(1);
  out[0].data = out_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Should be 1000.0 * (127 - (-100)) = 227000
  EXPECT_FLOAT_EQ(out_data[0], 227000.0f);
}

// =============================================================================
// Default Tensor Selection Plan Tests
// =============================================================================

TEST_F(TransformPostambleTest, empty_tensor_selection_plan_default_behavior)
{
  std::string onnx_path = get_onnx_file();
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    // No tensor_selection_plan - should default to first tensor
    { "dequant_scale", "1.0,2.0" }, { "dequant_zeropoint", "0,10" } };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp1_data = std::vector<int8_t>(1 * 3 * 20 * 20, 5); // Should go to ONNX
  auto inp2_data = std::vector<int8_t>(1 * 2 * 3 * 4, 15); // Should be passthrough

  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp1_data.data() },
    { { 1, 2, 3, 4 }, 1, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2);

  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out2_data = std::vector<float>(1 * 2 * 3 * 4);
  out[0].data = out1_data.data();
  out[1].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // First tensor: 1.0 * (5 - 0) + 1 (ONNX) = 6.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(6.0f)));

  // Second tensor: 2.0 * (15 - 10) = 10.0 (passthrough)
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(10.0f)));
}

// =============================================================================
// Non-Sequential Tensor Selection with Padding Tests (Bug Exposure)
// =============================================================================

TEST_F(TransformPostambleTest, non_sequential_selection_with_padding_onnx_inputs)
{
  std::string onnx_path = get_onnx_file();

  // Test case: Use tensor 1 for ONNX (skip tensor 0)
  // Each tensor has DIFFERENT padding that must be correctly applied
  // Bug: Code uses ONNX input index `i` instead of actual tensor index `tensor_idx`
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "tensor_selection_plan", "1" }, // Use tensor index 1 for ONNX input 0
    { "dequant_scale", "1.0,1.0" }, { "dequant_zeropoint", "0,0" },
    // padding format: [N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after]
    // Tensor 0: no padding (will be unused/passthrough)
    // Tensor 1: remove 1 pixel from right (depad from 21 to 20)
    { "padding", "0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,-1" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // Tensor 0: [1,3,20,20] - no padding, will be unused/passthrough
  auto inp0_data = std::vector<int8_t>(1 * 3 * 20 * 20, 10);
  // Tensor 1: [1,3,20,21] - padded, will go to ONNX, should be depadded to [1,3,20,20]
  auto inp1_data = std::vector<int8_t>(1 * 3 * 20 * 21, 5);

  AxTensorsInterface inp{ { { 1, 3, 20, 20 }, 1, inp0_data.data() },
    { { 1, 3, 20, 21 }, 1, inp1_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2); // 1 ONNX output + 1 unused input

  // Output 0: ONNX result from tensor 1 (depadded)
  // Output 1: passthrough from tensor 0 (unused)
  auto out0_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  out[0].data = out0_data.data();
  out[1].data = out1_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // BUG EXPOSURE:
  // If padding lookup uses `i` (ONNX input index 0) instead of `tensor_idx` (actual tensor 1),
  // it will use paddings[0] (no padding) instead of paddings[1] (depad right),
  // and the ONNX input will have wrong dimensions [1,3,20,21] instead of [1,3,20,20],
  // causing ONNX to fail with dimension mismatch.

  // Expected: tensor 1 depadded to 20x20, then 1.0 * (5 - 0) + 1 (ONNX) = 6.0
  EXPECT_THAT(out0_data, testing::Each(testing::FloatEq(6.0f)));

  // Expected: tensor 0 passthrough, 1.0 * (10 - 0) = 10.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(10.0f)));
}

TEST_F(TransformPostambleTest, non_sequential_selection_with_different_padding_multi_input)
{
  std::string onnx_path = get_multi_input_onnx_file();

  // Test case: Use tensors 2 and 0 for ONNX (in that order), skip tensor 1
  // Each tensor has DIFFERENT padding configuration
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "tensor_selection_plan", "2,0" }, // ONNX input 0 = tensor 2, ONNX input 1 = tensor 0
    { "dequant_scale", "1.0,1.0,1.0" }, { "dequant_zeropoint", "0,0,0" },
    // Tensor 0: remove 1 from right (21 -> 20)
    // Tensor 1: remove 1 from bottom (21 -> 20)
    // Tensor 2: remove 1 from left (21 -> 20)
    { "padding", "0,0,0,0,0,0,0,-1|0,0,0,0,0,-1,0,0|0,0,0,0,0,0,-1,0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  // All tensors padded differently
  auto inp0_data = std::vector<int8_t>(1 * 3 * 20 * 21, 3); // Tensor 0: padded right
  auto inp1_data = std::vector<int8_t>(1 * 3 * 21 * 20, 7); // Tensor 1: padded bottom (unused)
  auto inp2_data = std::vector<int8_t>(1 * 3 * 20 * 21, 5); // Tensor 2: padded left

  AxTensorsInterface inp{ { { 1, 3, 20, 21 }, 1, inp0_data.data() },
    { { 1, 3, 21, 20 }, 1, inp1_data.data() },
    { { 1, 3, 20, 21 }, 1, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 2); // 1 ONNX output + 1 unused input

  auto out0_data = std::vector<float>(1 * 3 * 20 * 20);
  auto out1_data = std::vector<float>(1 * 3 * 20 * 20);
  out[0].data = out0_data.data();
  out[1].data = out1_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // BUG EXPOSURE:
  // If padding lookup for ONNX input 0 uses `i=0` instead of `tensor_idx=2`,
  // it will use paddings[0] (remove right) instead of paddings[2] (remove left).
  // If padding lookup for ONNX input 1 uses `i=1` instead of `tensor_idx=0`,
  // it will use paddings[1] (remove bottom) instead of paddings[0] (remove right).
  // This will cause wrong depadding and ONNX dimension mismatches.

  // add_tensors.onnx adds two inputs: 5 + 3 = 8
  EXPECT_THAT(out0_data, testing::Each(testing::FloatEq(8.0f)));

  // Tensor 1 passthrough (with correct depadding): 1.0 * (7 - 0) = 7.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(7.0f)));
}

TEST_F(TransformPostambleTest, unused_tensors_with_padding_non_sequential)
{
  std::string onnx_path = get_onnx_file();

  // Test case: Multiple unused tensors with different padding
  // Only tensor 2 goes to ONNX, tensors 0 and 1 are unused but need depadding
  std::unordered_map<std::string, std::string> input = { { "onnx_path", onnx_path },
    { "tensor_selection_plan", "2" }, // Only use tensor 2 for ONNX
    { "dequant_scale", "1.0,1.0,1.0" }, { "dequant_zeropoint", "0,0,0" },
    // Tensor 0: remove 2 from right (10 -> 8)
    // Tensor 1: remove 1 from bottom (9 -> 8)
    // Tensor 2: no padding (already 20x20 for ONNX)
    { "padding", "0,0,0,0,0,0,0,-2|0,0,0,0,0,-1,0,0|0,0,0,0,0,0,0,0" } };

  auto xform = Ax::LoadTransform("postamble", input);

  auto inp0_data = std::vector<int8_t>(1 * 2 * 8 * 10, 11); // Tensor 0: padded right by 2
  auto inp1_data = std::vector<int8_t>(1 * 2 * 9 * 8, 13); // Tensor 1: padded bottom by 1
  auto inp2_data = std::vector<int8_t>(1 * 3 * 20 * 20, 17); // Tensor 2: no padding

  AxTensorsInterface inp{ { { 1, 2, 8, 10 }, 1, inp0_data.data() },
    { { 1, 2, 9, 8 }, 1, inp1_data.data() }, { { 1, 3, 20, 20 }, 1, inp2_data.data() } };

  auto out = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out.size(), 3); // 1 ONNX output + 2 unused inputs

  auto out0_data = std::vector<float>(1 * 3 * 20 * 20); // ONNX output from tensor 2
  auto out1_data = std::vector<float>(1 * 2 * 8 * 8); // Tensor 0 depadded
  auto out2_data = std::vector<float>(1 * 2 * 8 * 8); // Tensor 1 depadded
  out[0].data = out0_data.data();
  out[1].data = out1_data.data();
  out[2].data = out2_data.data();

  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // BUG EXPOSURE FOR UNUSED TENSORS:
  // Unused tensors use index `i` to look up padding, which should be correct
  // BUT if paddings array is sized only for ONNX inputs (size=1 in this case),
  // then tensor 0 (i=0) gets paddings[0] and tensor 1 (i=1) would be out of
  // bounds! This exposes whether paddings is per-tensor or per-ONNX-input.

  // ONNX output from tensor 2: 1.0 * (17 - 0) + 1 = 18.0
  EXPECT_THAT(out0_data, testing::Each(testing::FloatEq(18.0f)));

  // Tensor 0 depadded and dequantized: 1.0 * (11 - 0) = 11.0
  EXPECT_THAT(out1_data, testing::Each(testing::FloatEq(11.0f)));

  // Tensor 1 depadded and dequantized: 1.0 * (13 - 0) = 13.0
  EXPECT_THAT(out2_data, testing::Each(testing::FloatEq(13.0f)));
}
