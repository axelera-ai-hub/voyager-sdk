// Copyright Axelera AI, 2025
// An example program showing how to use axruntime class to run inference on one
// or more images with an imagenet classification network such as ResNet50.

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

// these values are an artefact of the model
const std::array<float, 3> mean = { { 0.485, 0.456, 0.406 } };
const std::array<float, 3> stddev = { { 0.229, 0.224, 0.225 } };

using namespace std::string_literals;

constexpr auto DEFAULT_LABELS = "ax_datasets/labels/imagenet1000_clsidx_to_labels.txt";

void
color_convert_norm_quant_and_pad(cv::Mat input, std::int8_t *out_rgba, const axrTensorInfo &info)
{
  const auto height = info.dims[1];
  const auto width = info.dims[2];
  const auto channels = info.dims[3];
  const auto [y_pad_left, y_pad_right] = info.padding[1];
  const auto [x_pad_left, x_pad_right] = info.padding[2];
  const auto [c_pad_left, c_pad_right] = info.padding[3];
  const auto unpadded_height = height - y_pad_left - y_pad_right;
  const auto unpadded_width = width - x_pad_left - x_pad_right;

  cv::Mat resized;
  // Proper imagenet preproc for the dataset would include a resize and a centercrop
  // here, but to simplify the example we just resize to the model input size
  cv::resize(input, resized, cv::Size(unpadded_width, unpadded_height), 0, 0, cv::INTER_LINEAR);

  const auto *in_bgr = static_cast<std::uint8_t *>(resized.data);

  const std::array<float, 3> mul
      = { { static_cast<float>(1.0f / (info.scale * stddev[0])),
          static_cast<float>(1.0f / (info.scale * stddev[1])),
          static_cast<float>(1.0f / (info.scale * stddev[2])) } };
  const std::array<float, 3> add = { { info.zero_point - (mul[0] * mean[0]),
      info.zero_point - (mul[1] * mean[1]), info.zero_point - (mul[2] * mean[2]) } };
  const auto pad_value
      = static_cast<std::int8_t>(std::clamp(info.zero_point, -128, 127));
  out_rgba = std::fill_n(out_rgba, y_pad_left * width * 4, pad_value);
  for (size_t y = 0; y != unpadded_height; ++y) {
    out_rgba = std::fill_n(out_rgba, x_pad_left * 4, pad_value);
    for (size_t x = 0; x != unpadded_width; ++x) {
      out_rgba = std::fill_n(out_rgba, c_pad_left, pad_value);
      for (size_t c = 0; c != 3; ++c) {
        const auto val = static_cast<float>(in_bgr[2 - c]);
        const auto norm = std::clamp(val / 255.0f * mul[c] + add[c], -128.0f, 127.0f);
        *out_rgba++ = static_cast<std::int8_t>(norm);
      }
      in_bgr += 3;
      out_rgba = std::fill_n(out_rgba, c_pad_right, pad_value);
    }
    out_rgba = std::fill_n(out_rgba, x_pad_right * channels, pad_value);
  }
  out_rgba = std::fill_n(out_rgba, y_pad_right * width * channels, pad_value);
}

std::string
tolower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return std::tolower(static_cast<int>(c)); });
  return s;
}

auto
read_labels(const std::string &path)
{
  std::vector<std::string> labels;
  std::ifstream file(path);
  for (std::string line; std::getline(file, line);) {
    labels.push_back(line);
  }
  return labels;
}

std::tuple<std::string, std::vector<std::string>, std::vector<std::string>>
parse_args(int argc, char **argv)
{
  std::string model_path;
  std::vector<std::string> labels;
  std::vector<std::string> images;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (s.ends_with(".json")) {
      model_path = s;
    } else if (s.ends_with(".txt") || s.ends_with(".names")) {
      labels = read_labels(s);
    } else if (std::filesystem::is_directory(s)) {
      for (const auto &entry : std::filesystem::directory_iterator(s)) {
        const auto ext = tolower(entry.path().extension().string());
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
          images.push_back(entry.path().string());
        }
      }
    } else if (!std::filesystem::exists(s)) {
      std::cerr << "Warning: Path does not exist: " << s << std::endl;
    } else {
      images.push_back(s);
    }
  }
  if (model_path.empty() || images.empty()) {
    std::cerr
        << "Usage: " << argv[0] << " model.json [labels.txt] images-or-dirs...\n"
        << "  model.json: path to the model file\n"
        << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
        << "  images-or-dirs: paths to images or directories containing images\n"
        << "\n"
        << "A prebuilt model can be downloaded using \n"
        << "\n"
        << "  ./download_prebuilt.py resnet50-imagenet\n"
        << "\n"
        << "The model path would then be \n"
        << "  build/resnet50-imagenet/resnet50-imagenet/1/model.json\n"
        << std::endl;
    std::exit(1);
  }
  if (labels.empty()) {
    const auto root = std::getenv("AXELERA_FRAMEWORK");
    labels = read_labels(root[0] == '\0' ? DEFAULT_LABELS : root + "/"s + DEFAULT_LABELS);
  }
  return { model_path, labels, images };
}

void
logger(void *arg, axrLogLevel level, const char *msg)
{
  (void) arg;
  (void) level;
  puts(msg);
}

int
main(int argc, char **argv)
{
  const auto [model_path, labels, images] = parse_args(argc, argv);
  auto ctx = axr::to_ptr(axr_create_context());
  axr_set_logger(ctx.get(), AXR_LOG_WARNING, logger, nullptr);
  auto model = axr_load_model(ctx.get(), model_path.c_str());
  if (!model) {
    std::cerr << axr_last_error_string(AXR_OBJECT(ctx.get())) << std::endl;
    return 1;
  }

  std::vector<axrTensorInfo> input_infos, output_infos;
  auto inputs = axr_num_model_inputs(model);
  for (size_t n = 0; n != inputs; ++n) {
    input_infos.push_back(axr_get_model_input(model, n));
  }
  auto outputs = axr_num_model_outputs(model);
  for (size_t n = 0; n != outputs; ++n) {
    output_infos.push_back(axr_get_model_output(model, n));
  }
  if (inputs != 1) {
    std::cerr << "This example only supports one input, but this model has "
              << inputs << std::endl;
    return 1;
  }
  if (outputs != 1) {
    std::cerr << "This example only supports one output, but this model has "
              << outputs << std::endl;
    return 1;
  }

  const auto batch_size = input_infos[0].dims[0];
  if (batch_size != 1) {
    std::cerr << "This example only supports batch size 1, but this model has "
              << input_infos[0].dims[0]
              << " Please re-deploy with --aipu-cores=1" << std::endl;
    return 1;
  }

  // use 1 subdevice from first available device
  auto connection = axr_device_connect(ctx.get(), nullptr, batch_size, nullptr);
  if (!connection) {
    std::cerr << axr_last_error_string(AXR_OBJECT(ctx.get())) << std::endl;
    return 1;
  }
  const auto props = "input_dmabuf=0;num_sub_devices=" + std::to_string(batch_size)
                     + ";aipu_cores=" + std::to_string(batch_size);
  auto properties = axr_create_properties(ctx.get(), props.c_str());
  auto instance = axr_load_model_instance(connection, model, properties);
  if (!instance) {
    std::cerr << axr_last_error_string(AXR_OBJECT(ctx.get())) << std::endl;
    return 1;
  }


  std::vector<axrArgument> input_args(inputs);
  std::vector<axrArgument> output_args(outputs);
  std::vector<std::unique_ptr<std::int8_t[]>> input_data;
  std::vector<std::unique_ptr<std::int8_t[]>> output_data;
  for (int n = 0; n != inputs; ++n) {
    input_data.emplace_back(new std::int8_t[axr_tensor_size(&input_infos[n])]);
    input_args[n].ptr = input_data[n].get();
    input_args[n].fd = 0;
    input_args[n].offset = 0;
  }
  for (int n = 0; n != outputs; ++n) {
    output_data.emplace_back(new std::int8_t[axr_tensor_size(&output_infos[n])]);
    output_args[n].ptr = output_data[n].get();
    output_args[n].fd = 0;
    output_args[n].offset = 0;
  }
  for (auto &&image : images) {
    auto in = cv::imread(image);
    if (in.empty()) {
      std::cerr << "Failed to read image: " << image << std::endl;
      return 1;
    }

    color_convert_norm_quant_and_pad(
        in, static_cast<std::int8_t *>(input_args[0].ptr), input_infos[0]);

    if (axr_run_model_instance(instance, input_args.data(), input_args.size(),
            output_args.data(), output_args.size())
        != AXR_SUCCESS) {
      std::cerr << "Failed to run model instance" << std::endl;
      std::cerr << axr_last_error_string(AXR_OBJECT(ctx.get())) << std::endl;
      return 1;
    }

    // depad and dequantize the result
    const auto info = output_infos[0]; // for classification the output is shape (1, 1, 1, N)
    const auto [out_pad_left, out_pad_right] = info.padding[3];
    const auto out_size = info.dims[info.ndims - 1] - out_pad_left - out_pad_right;
    const auto out
        = std::span<const std::int8_t>(output_data[0].get() + out_pad_left, out_size);
    std::vector<float> floats;
    floats.reserve(out_size);
    std::transform(out.begin(), out.end(), std::back_inserter(floats),
        [info](auto v) { return (v - info.zero_point) * info.scale; });

    // argmax
    const auto max = std::max_element(floats.begin(), floats.end());
    const auto cls = std::distance(floats.begin(), max);
    const auto label = cls < labels.size() ? labels[cls] : " (no label)";
    const auto score = static_cast<int>(*max);
    std::cout << image << " : classified as class=" << cls << " " << label
              << " " << std::to_string(score) << "%" << std::endl;
  }
}
