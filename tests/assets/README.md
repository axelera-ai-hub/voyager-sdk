# Test Assets

This directory contains test assets for the framework test suite.

## ONNX Focus Layer Test Assets

The following ONNX models are used for testing Focus layer replacement:

- **`focus_preprocess_graph.onnx`**: Focus-only pattern (4 Slices + 1 Concat, no Conv)
- **`focus_conv_test.onnx`**: Direct Focus+Conv pattern (4 Slices directly on input)
- **`focus_conv_nested_test.onnx`**: Nested Focus+Conv pattern (YOLOX-style, 6 Slices)
