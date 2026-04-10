// Copyright Axelera AI, 2025
#pragma once
#include <AxMeta.hpp>
#include <variant>
#include <vector>

using VectorType = std::variant<std::vector<float>, std::vector<uint8_t>>;
class AxMetaImage : public AxMetaBase
{
  public:
  AxMetaImage(VectorType data, int channels, int width, int height)
      : datavec(std::move(data)),
        imagechannels(channels),
        imagewidth(width),
        imageheight(height)
  {
    if (std::get_if<std::vector<uint8_t>>(&data) != nullptr) {
      is_float = false;
    } else if (std::get_if<std::vector<float>>(&data) != nullptr) {
      is_float = true;
    } else {
      throw std::runtime_error("image_decode_to_meta: Invalid variant type, internal error");
    }
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "ImageMeta";
    auto results = std::vector<extern_meta>();
    if (auto *vec = std::get_if<std::vector<uint8_t>>(&datavec)) {
      results.push_back({ class_meta, "data", int(vec->size() * sizeof(uint8_t)),
          reinterpret_cast<const char *>(vec->data()) });
    } else if (auto *vec = std::get_if<std::vector<float>>(&datavec)) {
      results.push_back({ class_meta, "data", int(vec->size() * sizeof(float)),
          reinterpret_cast<const char *>(vec->data()) });
    }
    results.push_back({ class_meta, "float_datatype", int(sizeof(bool)),
        reinterpret_cast<const char *>(&is_float) });
    results.push_back({ class_meta, "width", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&imagewidth) });

    results.push_back({ class_meta, "height", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&imageheight) });

    results.push_back({ class_meta, "channels", static_cast<int>(sizeof(int)),
        reinterpret_cast<const char *>(&imagechannels) });

    return results;
  }

  private:
  VectorType datavec;
  int imagechannels;
  int imagewidth;
  int imageheight;
  int is_float;
};
