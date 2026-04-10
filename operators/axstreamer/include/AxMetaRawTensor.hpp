#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "AxDataInterface.h"
#include "AxMeta.hpp"

// Core C++ tensor container. No meta/export logic here.
class AxTensorContainer
{
  public:
  AxTensorContainer() = default;

  // Add a tensor to the internal list, copying the data and owning it
  void add_tensor(const void *data_ptr, size_t size_elements, int bytes,
      const std::vector<int64_t> &dims_vec)
  {
    // Allocate and copy data
    size_t total_bytes = size_elements * bytes;
    auto owned = std::make_unique_for_overwrite<std::byte[]>(total_bytes);
    std::memcpy(owned.get(), data_ptr, total_bytes);

    // Prepare AxTensorInterface
    AxTensorInterface tensor;
    tensor.sizes.clear();
    for (auto d : dims_vec)
      tensor.sizes.push_back(static_cast<int>(d));
    tensor.bytes = bytes;
    tensor.data = owned.get();
    tensor.fd = -1;

    tensors_.push_back(tensor);
    owned_data_.push_back(std::move(owned));
  }

  // --- Accessors ---
  size_t num_tensors() const
  {
    return tensors_.size();
  }
  const std::vector<AxTensorInterface> &as_tensors() const
  {
    return tensors_;
  }
  // Convenience: dims(i) returns the shape as int64_t for export or C++ users who need it.
  std::vector<int64_t> dims(size_t tensor_idx) const
  {
    const auto &sizes = tensors_.at(tensor_idx).sizes;
    return std::vector<int64_t>(sizes.begin(), sizes.end());
  }

  private:
  std::vector<AxTensorInterface> tensors_;
  std::vector<std::unique_ptr<std::byte[]>> owned_data_;
};

// Meta/export wrapper for Python/C API. Owns an AxTensorContainer.
class AxMetaRawTensor : public AxMetaBase
{
  public:
  AxMetaRawTensor()
      : tensor_(std::make_unique<AxTensorContainer>())
  {
  }

  // Add tensor (generic, supports any type)
  void add_tensor(const void *data_ptr, size_t size_elements, int bytes,
      const std::vector<int64_t> &dims)
  {
    tensor_->add_tensor(data_ptr, size_elements, bytes, dims);
  }

  // Expose as_tensors for C++ users
  const std::vector<AxTensorInterface> &as_tensors() const
  {
    return tensor_->as_tensors();
  }

  // Access the underlying tensor
  AxTensorContainer *get_tensor()
  {
    return tensor_.get();
  }
  const AxTensorContainer *get_tensor() const
  {
    return tensor_.get();
  }

  // get_extern_meta prepares export buffers for Python/C API.
  // dims_storage_ and dtype_storage_ are only for export lifetime management, not persistent state.
  std::vector<extern_meta> get_extern_meta() const override
  {
    attr_names_.clear();
    dims_storage_.clear();
    dtype_storage_.clear();
    std::vector<extern_meta> result;
    size_t num_tensors = tensor_->num_tensors();
    result.reserve(num_tensors * 3);
    attr_names_.reserve(num_tensors * 3);
    dims_storage_.reserve(num_tensors);
    dtype_storage_.reserve(num_tensors);

    const char *meta_type = "TensorMeta";

    for (size_t i = 0; i < num_tensors; ++i) {
      auto data_attr_name = "data_" + std::to_string(i);
      auto dims_attr_name = "dims_" + std::to_string(i);
      auto dtype_attr_name = "dtype_" + std::to_string(i);
      attr_names_.push_back(data_attr_name);
      attr_names_.push_back(dims_attr_name);
      attr_names_.push_back(dtype_attr_name);

      // Add tensor data
      const auto &tensors = tensor_->as_tensors();
      size_t data_size_bytes = tensors[i].total_bytes();
      const char *data_ptr = reinterpret_cast<const char *>(tensors[i].data);
      result.push_back({ meta_type, attr_names_[i * 3].c_str(),
          static_cast<int>(data_size_bytes), data_ptr });

      // Add tensor dimensions (dims_storage_ ensures pointer validity for export)
      dims_storage_.emplace_back(tensor_->dims(i));
      const auto &dims = dims_storage_.back();
      size_t dims_size_bytes = dims.size() * sizeof(int64_t);
      const char *dims_ptr = reinterpret_cast<const char *>(dims.data());
      result.push_back({ meta_type, attr_names_[i * 3 + 1].c_str(),
          static_cast<int>(dims_size_bytes), dims_ptr });

      // Add dtype string (generated from bytes, dtype_storage_ ensures pointer validity)
      dtype_storage_.push_back(get_dtype_string(tensors[i].bytes));
      const std::string &dtype = dtype_storage_.back();
      result.push_back({ meta_type, attr_names_[i * 3 + 2].c_str(),
          static_cast<int>(dtype.size()), dtype.data() });
    }

    return result;
  }

  private:
  std::string get_dtype_string(int bytes) const
  {
    switch (bytes) {
      case 4:
        return "f4"; // float32
      case 2:
        return "i2"; // int16
      case 1:
        return "i1"; // int8
      case 8:
        return "f8"; // float64
      default:
        return "u1"; // fallback to uint8
    }
  }

  std::unique_ptr<AxTensorContainer> tensor_;
  // Storage for generated attribute names (data_0, dims_0, dtype_0, ...)
  // Needs mutable because get_extern_meta is const but needs to generate/store these.
  mutable std::vector<std::string> attr_names_;
  // Temporary export buffer for dims (int64_t), not persistent state.
  mutable std::vector<std::vector<int64_t>> dims_storage_;
  // Temporary export buffer for dtype strings, not persistent state.
  mutable std::vector<std::string> dtype_storage_;
};

// --- User-facing helpers for extracting tensor data and shape ---
// Usage:
//   const float* data = get_tensor_data<float>(meta);
//   std::vector<int64_t> shape = get_tensor_shape(meta);

template <typename T>
const T *
get_tensor_data(const AxMetaRawTensor &meta, size_t tensor_idx = 0)
{
  return static_cast<const T *>(meta.get_tensor()->as_tensors()[tensor_idx].data);
}

inline std::vector<int64_t>
get_tensor_shape(const AxMetaRawTensor &meta, size_t tensor_idx = 0)
{
  return meta.get_tensor()->dims(tensor_idx);
}
