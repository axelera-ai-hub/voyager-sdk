#include "GstAxDataUtils.hpp"
#include <algorithm>
#include <span>
#include "AxStreamerUtils.hpp"

#include <gst/allocators/gstfdmemory.h>
#include <gst/video/video.h>
#include <unordered_map>
#include <unordered_set>
#include "AxLog.hpp"

std::string
gst_format_to_string(GstVideoFormat fmt)
{
  switch (fmt) {
    case GST_VIDEO_FORMAT_UNKNOWN:
      return "UNKNOWN";
    case GST_VIDEO_FORMAT_GRAY8:
      return "GRAY8";
    case GST_VIDEO_FORMAT_RGB:
      return "RGB";
    case GST_VIDEO_FORMAT_BGR:
      return "BGR";
    case GST_VIDEO_FORMAT_RGBx:
      return "RGBx";
    case GST_VIDEO_FORMAT_BGRx:
      return "BGRx";
    case GST_VIDEO_FORMAT_xRGB:
      return "xRGB";
    case GST_VIDEO_FORMAT_xBGR:
      return "xBGR";
    case GST_VIDEO_FORMAT_RGBA:
      return "RGBA";
    case GST_VIDEO_FORMAT_BGRA:
      return "BGRA";
    case GST_VIDEO_FORMAT_ARGB:
      return "ARGB";
    case GST_VIDEO_FORMAT_ABGR:
      return "ABGR";
    case GST_VIDEO_FORMAT_I420:
      return "I420";
    case GST_VIDEO_FORMAT_YUY2:
      return "YUY2";
    case GST_VIDEO_FORMAT_NV12:
      return "NV12";
    default:
      return "<unknown " + std::to_string(fmt) + ">";
  }
}

#define AX_GENERATE_FROM_GST_VIDEO_FORMAT(x) \
  case GST_VIDEO_FORMAT_##x:                 \
    return AxVideoFormat::x;

static AxVideoFormat
ax_video_format_from_gst(GstVideoFormat gst_format)
{
  switch (gst_format) {
    GST_AX_VIDEO_FORMATS(AX_GENERATE_FROM_GST_VIDEO_FORMAT)
    default:
      return AxVideoFormat::UNDEFINED;
  }
}

#define AX_GENERATE_TO_GST_VIDEO_FORMAT(x) \
  case AxVideoFormat::x:                   \
    return GST_VIDEO_FORMAT_##x;

static GstVideoFormat
ax_video_format_to_gst(AxVideoFormat format)
{
  switch (format) {
    GST_AX_VIDEO_FORMATS(AX_GENERATE_TO_GST_VIDEO_FORMAT)
    case AxVideoFormat::UNDEFINED:
      return GST_VIDEO_FORMAT_UNKNOWN;
      // not using default to catch missing cases, which keeps the MACRO
      // GST_AX_VIDEO_FORMATS consistent with the one in AxStreamerUtils.cpp
  }
  return GST_VIDEO_FORMAT_UNKNOWN;
}

static std::vector<int>
tensor_sizes_from_nnstreamer(std::span<uint32_t> dimensions, int minimum_rank = 4)
{
  auto first = dimensions.rbegin();
  auto last = dimensions.rend();
  auto pos = std::find_if(first, last, [](int x) { return x != 1; });

  std::vector<int> result(pos, last);
  if (result.size() < minimum_rank) {
    result.insert(result.begin(), minimum_rank - result.size(), 1);
  }
  return result;
}

static AxVideoInterface
video_interface_from_caps(GstCaps *caps)
{
  GstVideoInfo *info = gst_video_info_new();
  AxVideoInterface video_interface = AxVideoInterface();
  if (gst_video_info_from_caps(info, caps)) {
    video_interface.info.width = GST_VIDEO_INFO_WIDTH(info);
    video_interface.info.height = GST_VIDEO_INFO_HEIGHT(info);
    video_interface.info.format = ax_video_format_from_gst(GST_VIDEO_INFO_FORMAT(info));

    //  Need to add strides and offsets for multi-plane formats
    video_interface.strides = std::vector<size_t>(GST_VIDEO_INFO_N_PLANES(info));
    video_interface.offsets = std::vector<size_t>(GST_VIDEO_INFO_N_PLANES(info));
    for (int i = 0; i < GST_VIDEO_INFO_N_PLANES(info); ++i) {
      video_interface.strides[i] = GST_VIDEO_INFO_PLANE_STRIDE(info, i);
      video_interface.offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(info, i);
    }
    //  For backwards comaptibility
    video_interface.info.stride = info->stride[0];
    video_interface.info.offset = info->offset[0];
  }
  gst_video_info_free(info);
  video_interface.fd = -1;
  return video_interface;
}

static AxDataInterface
interface_from_caps(GstCaps *caps)
{
  GstStructure *structure = gst_caps_get_structure(caps, 0);
  if (gst_structure_has_name(structure, "video/x-raw")) {
    return video_interface_from_caps(caps);
  } else if (structure_is_tensor_stream(structure)) {
    GstTensorsConfig config;
    if (!gst_tensors_config_from_structure(config, structure)) {
      throw std::runtime_error("num_tensors not found in caps");
    }

    if (!validate_tensors_config(config)) {
      auto *caps_str = gst_caps_to_string(caps);
      auto str_caps = std::string(caps_str);
      g_free(caps_str);
      throw std::runtime_error("Cannot decode tensor caps: " + str_caps);
    }

    AxTensorsInterface tensors = AxTensorsInterface(config.info.num_tensors);
    for (int i = 0; i < (int) tensors.size(); ++i) {
      tensors[i].bytes = tensor_type_size(config.info.info[i].type);
      tensors[i].sizes = tensor_sizes_from_nnstreamer(
          { (uint32_t *) config.info.info[i].dimension.data(), AX_TENSOR_RANK_LIMIT });
      tensors[i].fd = -1;
    }

    return tensors;

  } else {
    auto *caps_str = gst_caps_to_string(caps);
    auto str_caps = std::string(caps_str);
    g_free(caps_str);

    throw std::logic_error("interface_from_caps must have either a video or tensor caps, actually has: "
                           + str_caps);
  }
}

AxDataInterface
interface_from_caps_and_meta(GstCaps *caps, GstBuffer *buffer)
{
  auto interface = interface_from_caps(caps);

  if (std::holds_alternative<AxTensorsInterface>(interface) || !buffer) {
    return interface;
  }

  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto &video_interface = std::get<AxVideoInterface>(interface);
    GstVideoMeta *meta = gst_buffer_get_video_meta(buffer);
    if (!meta) {
      return interface;
    }
    auto num_planes = meta->n_planes;
    video_interface.offsets.assign(meta->offset, meta->offset + num_planes);
    video_interface.strides.assign(meta->stride, meta->stride + num_planes);
    video_interface.info.stride = meta->stride[0];
    video_interface.info.offset = meta->offset[0];

    if (auto *crop_meta = gst_buffer_get_video_crop_meta(buffer)) {
      video_interface.info.x_offset = crop_meta->x;
      video_interface.info.y_offset = crop_meta->y;
      video_interface.info.cropped = true;
    } else {
      video_interface.info.x_offset = 0;
      video_interface.info.y_offset = 0;
      video_interface.info.cropped = false;
    }
  }
  return interface;
}

size_t
size_from_interface(const AxDataInterface &interface)
{
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    const auto &video_info = std::get<AxVideoInterface>(interface).info;
    return video_info.stride * video_info.height;
  } else if (std::holds_alternative<AxTensorsInterface>(interface)) {
    const auto &tensors_interface = std::get<AxTensorsInterface>(interface);
    if (tensors_interface.size() > 1) {
      return 0;
    }
    return tensors_interface[0].total_bytes();
  } else {
    return 0;
  }
}

void
copy_or_fixate_framerate(GstCaps *from, GstCaps *to)
{
  gint framerate_numerator = 0;
  gint framerate_denominator = 1;
  if (!gst_structure_has_field(gst_caps_get_structure(from, 0), "framerate")) {
    return;
  }
  gst_structure_get_fraction(gst_caps_get_structure(from, 0), "framerate",
      &framerate_numerator, &framerate_denominator);

  for (int i = 0; i < gst_caps_get_size(to); ++i) {
    if (gst_structure_has_field(gst_caps_get_structure(to, i), "framerate")) {
      gst_structure_fixate_field_nearest_fraction(gst_caps_get_structure(to, i),
          "framerate", framerate_numerator, framerate_denominator);
    } else {
      gst_structure_set(gst_caps_get_structure(to, i), "framerate",
          GST_TYPE_FRACTION, framerate_numerator, framerate_denominator, NULL);
    }
  }
}

void
assign_data_ptrs_to_interface(const std::vector<GstMapInfo> &info, AxDataInterface &interface)
{
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    std::get<AxVideoInterface>(interface).data = info[0].data;
  } else if (std::holds_alternative<AxTensorsInterface>(interface)) {
    auto &tensors = std::get<AxTensorsInterface>(interface);
    if (tensors.size() != info.size()) {
      throw std::runtime_error("Size mismatch in assign_data_ptrs_to_interface");
    }
    for (int i = 0; i < tensors.size(); ++i) {
      tensors[i].data = info[i].data;
    }
  } else {
    throw std::logic_error(
        "No video or tensor interface passed to assign_data_ptrs_to_interface.");
  }
}

void
assign_fds_to_interface(AxDataInterface &input, GstBuffer *buffer)
{
  if (auto *video = std::get_if<AxVideoInterface>(&input)) {
    auto *mem = gst_buffer_peek_memory(buffer, 0);
    video->fd = gst_fd_memory_get_fd(mem);
  } else if (auto *tensors = std::get_if<AxTensorsInterface>(&input)) {
    for (int i = 0; i < tensors->size(); ++i) {
      auto &tensor = (*tensors)[i];
      auto *mem = gst_buffer_peek_memory(buffer, i);
      tensor.fd = gst_fd_memory_get_fd(mem);
    }
  } else {
    throw std::logic_error("No tensor or video interface in assign_fds_to_interface");
  }
}

void
add_video_meta_from_interface_to_buffer(GstBuffer *buffer, const AxDataInterface &interface)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    return;
  }
  auto &info = std::get<AxVideoInterface>(interface).info;
  if (!info.cropped) {
    return;
  }
  if (!gst_buffer_is_writable(buffer)) {
    throw std::runtime_error("Buffer is not writable in add_video_meta_from_interface_to_buffer");
  }

  GstVideoMeta *meta = gst_buffer_get_video_meta(buffer);
  if (!meta) {
    gint strides[1] = { info.stride };
    gsize offsets[1] = { static_cast<gsize>(info.offset) };
    meta = gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
        ax_video_format_to_gst(info.format), info.width, info.height, 1, offsets, strides);
  } else {
    meta->width = info.width;
    meta->height = info.height;
  }

  int original_x = 0;
  int original_y = 0;
  auto *crop_meta = gst_buffer_get_video_crop_meta(buffer);
  if (crop_meta) {
    original_x = crop_meta->x;
    original_y = crop_meta->y;
  } else {
    crop_meta = gst_buffer_add_video_crop_meta(buffer);
  }
  crop_meta->x = original_x + info.x_offset;
  crop_meta->y = original_y + info.y_offset;
  crop_meta->width = info.width;
  crop_meta->height = info.height;
}

static std::string
caps_string_from_tensors_interface(const AxTensorsInterface &tensors)
{
  std::string result = "other/tensors,format=static";
  if (tensors.empty()) {
    return result;
  }
  result += ",num_tensors=" + std::to_string(tensors.size());
  int num_has_dimension = 0;
  int num_has_type = 0;
  std::string string_dimensions;
  std::string string_types;
  for (const auto &tensor : tensors) {

    if (!tensor.sizes.empty()) {
      ++num_has_dimension;
      for (auto itr = tensor.sizes.rbegin(); itr != tensor.sizes.rend(); ++itr) {
        string_dimensions += std::to_string(*itr) + ":";
      }
      for (int i = 0; i < 8 - tensor.sizes.size(); i++) {
        string_dimensions += "1:";
      }
      string_dimensions.back() = '.';
    }

    if (tensor.bytes) {
      ++num_has_type;
      if (tensor.bytes == 1) {
        string_types += "int8.";
      } else if (tensor.bytes == 4) {
        string_types += "float32.";
      } else {
        throw std::runtime_error("Only int8 and float32 are supported in caps_string_from_tensors_interface, given tensor.bytes of "
                                 + std::to_string(tensor.bytes));
      }
    }
  }

  if (!string_dimensions.empty()) {
    if (num_has_dimension != tensors.size()) {
      throw std::runtime_error("Either all or none of the tensors should provide dimensions ");
    }
    string_dimensions.pop_back();
    result += ",dimensions=" + string_dimensions;
  }
  if (!string_types.empty()) {
    if (num_has_type != tensors.size()) {
      throw std::runtime_error("Either all or none of the tensors should provide types");
    }
    string_types.pop_back();
    result += ",types=" + string_types;
  }
  return result;
}

GstCaps *
caps_from_interface(const AxDataInterface &interface)
{
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto &info = std::get<AxVideoInterface>(interface).info;
    std::string result = "video/x-raw";
    if (AxVideoFormat::UNDEFINED != info.format) {
      result += std::string(",format=")
                + gst_video_format_to_string(ax_video_format_to_gst(info.format));
    }
    if (info.width) {
      result += ",width=" + std::to_string(info.width);
    }
    if (info.height) {
      result += ",height=" + std::to_string(info.height);
    }
    return gst_caps_from_string(result.c_str());
  } else if (std::holds_alternative<AxTensorsInterface>(interface)) {
    return gst_caps_from_string(
        caps_string_from_tensors_interface(std::get<AxTensorsInterface>(interface))
            .c_str());
  } else {
    throw std::logic_error("caps_from_interface provided with uninitialized interface");
  }
}

std::vector<GstMapInfo>
get_mem_map(GstBuffer *buffer, GstMapFlags flags, GObject *self)
{
  std::vector<GstMapInfo> mapInfoVec(gst_buffer_n_memory(buffer));

  for (guint i = 0; i < mapInfoVec.size(); ++i) {
    GstMemory *mem = gst_buffer_peek_memory(buffer, i);
    if ((flags & GST_MAP_WRITE) && !gst_memory_is_writable(mem)) {
      GST_WARNING_OBJECT(self, "Memory to be mapped is not writable.");
      if (1 == mapInfoVec.size()) {
        gst_buffer_map(buffer, &mapInfoVec[0], flags);
        return mapInfoVec;
      }
      throw std::runtime_error("Memory to be mapped is not writable.");
    }
    if (!gst_memory_map(gst_buffer_peek_memory(buffer, i), &mapInfoVec[i], flags)) {
      GST_ERROR_OBJECT(self, "Failed to map memory");
      throw std::runtime_error("Failed to map memory");
    }
  }
  return mapInfoVec;
}

void
unmap_mem(std::vector<GstMapInfo> &mapInfoVec)
{
  std::for_each(mapInfoVec.begin(), mapInfoVec.end(),
      [](GstMapInfo &info) { gst_memory_unmap(info.memory, &info); });
  mapInfoVec = std::vector<GstMapInfo>();
}

static std::vector<std::string_view>
split(std::string_view s, char delim)
{
  return Ax::Internal::split(s, delim);
}

static std::string_view
trim(std::string_view s)
{
  return Ax::Internal::trim(s);
}

static std::unordered_map<std::string, std::string>
extract_options(GObject *self, const std::string &opts)
{
  std::unordered_map<std::string, std::string> properties;

  auto options = split(opts, ';');
  properties.reserve(options.size());
  for (const auto &op : options) {
    if (op.empty()) {
      continue;
    }
    auto option = split(op, ':');
    if (option.size() != 2) {
      GST_ELEMENT_ERROR(self, LIBRARY, FAILED, (NULL),
          ("Options must be specified as a semicolon separated list of colon separated pairs. Given: %s",
              opts.c_str()));
    } else {
      auto key = std::string(trim(option[0]));
      if (properties.count(key) != 0) {
        GST_ELEMENT_ERROR(self, LIBRARY, FAILED, (NULL),
            ("Option is specified more than once."));
      }
      auto value = std::string(trim(option[1]));
      properties[key] = value;
    }
  }
  return properties;
}

static std::unordered_map<std::string, std::string>
parse_and_validate_options(GObject *self, const std::string &options_string,
    const Ax::V1Plugin::Base &fns)
{
  std::unordered_map<std::string, std::string> opts{};
  if (options_string.empty()) {
    return opts;
  }

  if (!fns.allowed_properties) {
    throw std::runtime_error("Module symbol: allowed_properties, could not be loaded: "
                             + std::string(g_module_error()));
  }

  opts = extract_options(self, options_string);
  for (const auto &opt : opts) {
    if (fns.allowed_properties().count(opt.first) == 0) {
      GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
          ("Property not allowed - %s.", opt.first.c_str()), (NULL));
    }
  }
  return opts;
}

void
init_options(GObject *self, const std::string &options_string,
    const Ax::V1Plugin::Base &fns, Ax::Logger &logger,
    std::shared_ptr<void> &options, bool &initialized)
{
  if (initialized) {
    return;
  }
  if (!fns.init_and_set_static_properties) {
    if (options_string.empty()) {
      return;
    }
    throw std::runtime_error("Function -init_and_set_static_properties- not found but assigned following options: "
                             + options_string);
  }
  auto opts = parse_and_validate_options(self, options_string, fns);
  options = fns.init_and_set_static_properties(opts, logger);
  if (fns.set_dynamic_properties) {
    fns.set_dynamic_properties(opts, options.get(), logger);
  }
  initialized = true;
}

void
update_options(GObject *self, const std::string &options_string,
    const Ax::V1Plugin::Base &fns, Ax::Logger &logger,
    std::shared_ptr<void> &options, bool &initialized)
{
  if (!initialized) {
    return;
  }
  if (fns.set_dynamic_properties) {
    auto opts = parse_and_validate_options(self, options_string, fns);
    fns.set_dynamic_properties(opts, options.get(), logger);
  }
}

namespace
{

struct tensor_type_details {
  tensor_type type;
  const char *name;
  int size;
};

tensor_type_details type_details[] = {
  { tensor_type::INT8, "int8", 1 },
  { tensor_type::UINT8, "uint8", 1 },
  { tensor_type::INT16, "int16", 2 },
  { tensor_type::UINT16, "uint16", 2 },
  { tensor_type::FLOAT16, "float16", 2 },
  { tensor_type::INT32, "int32", 4 },
  { tensor_type::UINT32, "uint32", 4 },
  { tensor_type::FLOAT32, "float32", 4 },
  { tensor_type::INT64, "int64", 8 },
  { tensor_type::UINT64, "uint64", 8 },
  { tensor_type::FLOAT64, "float64", 8 },
};

tensor_type
get_tensor_type(const std::string_view type)
{
  auto pos = std::find_if(std::begin(type_details), std::end(type_details),
      [type](const tensor_type_details &t) { return type == t.name; });
  return pos != std::end(type_details) ? pos->type : tensor_type::LAST;
}

std::string
get_tensor_type_string(tensor_type type)
{
  auto pos = std::find_if(std::begin(type_details), std::end(type_details),
      [type](const tensor_type_details &t) { return type == t.type; });
  return pos != std::end(type_details) ? pos->name : "unknown";
}

tensor_format
get_tensor_format(const std::string &format)
{
  if (format == "static") {
    return tensor_format::STATIC;
  } else if (format == "flexible") {
    return tensor_format::FLEXIBLE;
  }
  return tensor_format::LAST;
}

guint
parse_tensor_dimension(std::string_view dimstr, tensor_dim &dim)
{
  auto vals = Ax::Internal::split(dimstr, ':');
  std::transform(vals.begin(), vals.end(), dim.begin(), [](std::string_view s) {
    return std::stoi(std::string(Ax::Internal::trim(s)));
  });
  return vals.size();
}

bool
tensor_dimension_is_valid(const tensor_dim &dim)
{
  return std::find(dim.begin(), dim.end(), 0) == dim.end();
}

bool
structures_have_same_dimensions(GstStructure *s1, GstStructure *s2)
{
  if (!s1 || !s2) {
    return FALSE;
  }
  if (!gst_structure_has_field(s1, "dimension") || !gst_structure_has_field(s2, "dimension")) {
    return FALSE;
  }

  tensor_dim dim1;
  std::string dim_str1 = gst_structure_get_string(s1, "dimension");
  parse_tensor_dimension(dim_str1, dim1);

  tensor_dim dim2;
  std::string dim_str2 = gst_structure_get_string(s2, "dimension");
  parse_tensor_dimension(dim_str2, dim2);

  return dim1 == dim2;
}

guint
tensors_info_parse_dimensions_string(GstTensorsInfo &info, std::string_view dim_string)
{
  auto str_dims = Ax::Internal::split(dim_string, ",.");
  auto num_dims = str_dims.size();

  if (num_dims > AX_TENSOR_SIZE_LIMIT) {
    num_dims = AX_TENSOR_SIZE_LIMIT;
  }

  for (uint32_t i = 0; i < num_dims; i++) {
    parse_tensor_dimension(str_dims[i], info.info[i].dimension);
  }
  return num_dims;
}

guint
tensors_info_parse_types_string(GstTensorsInfo &info, const std::string &type_string)
{
  auto str_types = Ax::Internal::split(type_string, ",.");
  auto num_types = str_types.size();

  if (num_types > AX_TENSOR_SIZE_LIMIT) {
    //  Log warning
    num_types = AX_TENSOR_SIZE_LIMIT;
  }

  for (uint32_t i = 0; i < num_types; i++) {
    info.info[i].type = get_tensor_type(str_types[i]);
  }
  return num_types;
}

std::string
gst_tensor_get_dimension_string(const tensor_dim &dim)
{
  return Ax::Internal::join(dim, ":");
}

std::string
tensors_info_get_dimensions_string(const GstTensorsInfo &info)
{
  std::vector<std::string> dims;
  dims.reserve(info.num_tensors);
  for (int i = 0; i < info.num_tensors; ++i) {
    auto dim_str = gst_tensor_get_dimension_string(info.info[i].dimension);
    dims.push_back(dim_str);
  }
  return Ax::Internal::join(dims, ",");
}

std::string
tensor_get_rank_dimension_string(const tensor_dim &dim, const unsigned int rank)
{
  auto actual_rank
      = (rank == 0 || rank > AX_TENSOR_RANK_LIMIT ? AX_TENSOR_RANK_LIMIT : rank);

  std::vector<std::string> dims;
  dims.reserve(actual_rank);
  for (uint32_t i = 0; i < actual_rank; i++) {
    dims.push_back(std::to_string(dim[i]));
  }
  return Ax::Internal::join(dims, ":");
}

std::string
tensors_info_get_rank_dimensions_string(const GstTensorsInfo &info, const unsigned int rank)
{
  std::vector<std::string> dims;
  dims.reserve(info.num_tensors);
  for (uint32_t i = 0; i < info.num_tensors; i++) {
    auto dim_str = tensor_get_rank_dimension_string(info.info[i].dimension, rank);
    dims.push_back(dim_str);
  }
  return Ax::Internal::join(dims, ",");
}

std::string
tensors_info_get_types_string(const GstTensorsInfo &info)
{
  std::vector<std::string> type_strs;
  type_strs.reserve(info.num_tensors);
  for (uint32_t i = 0; i != info.num_tensors; ++i) {
    if (info.info[i].type != tensor_type::LAST) {
      type_strs.push_back(get_tensor_type_string(info.info[i].type));
    }
  }
  return Ax::Internal::join(type_strs, ",");
}

bool
validate_tensor_info(const GstTensorInfo &info)
{
  return info.type != tensor_type::LAST && tensor_dimension_is_valid(info.dimension);
}

bool
validate_tensors_info(const GstTensorsInfo &info)
{
  /* tensor stream format */
  if (info.format == tensor_format::LAST) {
    return false;
  }

  /* cannot check tensor info when tensor is not static */
  if (info.format != tensor_format::STATIC) {
  }

  if (info.num_tensors < 1) {
    return false;
  }

  for (uint32_t i = 0; i < info.num_tensors; i++) {
    if (!validate_tensor_info(info.info[i])) {
      return false;
    }
  }
  return true;
}


bool
have_same_dimensions(GstStructure *s1, GstStructure *s2)
{
  if (!s1 || !s2) {
    return FALSE;
  }

  GstTensorsInfo info1;
  std::string dim_str1 = gst_structure_get_string(s1, "dimensions");
  auto num_dim1 = tensors_info_parse_dimensions_string(info1, dim_str1);

  GstTensorsInfo info2;
  std::string dim_str2 = gst_structure_get_string(s2, "dimensions");
  auto num_dim2 = tensors_info_parse_dimensions_string(info2, dim_str2);

  if (num_dim1 != num_dim2) {
    return false;
  }
  return std::equal(info1.info.begin(), std::next(info1.info.begin(), num_dim1),
      info2.info.begin(),
      [](const auto &a, const auto &b) { return a.dimension == b.dimension; });
}


GstCaps *
get_tensors_caps(const GstTensorsConfig &config)
{
  auto *caps = gst_caps_from_string(GST_TENSORS_CAP_DEFAULT);

  /* structure for backward compatibility */
  auto *structure = gst_structure_new_empty(TENSORS_MIME_TYPE);

  if (config.info.num_tensors > 0) {
    auto dim_str = tensors_info_get_dimensions_string(config.info);
    auto type_str = tensors_info_get_types_string(config.info);

    gst_caps_set_simple(caps, "num_tensors", G_TYPE_INT, config.info.num_tensors, NULL);
    gst_caps_set_simple(caps, "dimensions", G_TYPE_STRING, dim_str.c_str(), NULL);
    gst_caps_set_simple(caps, "types", G_TYPE_STRING, type_str.c_str(), NULL);

    dim_str = tensors_info_get_rank_dimensions_string(config.info, 4);

    gst_structure_set(structure, "num_tensors", G_TYPE_INT, config.info.num_tensors, NULL);
    gst_structure_set(structure, "dimensions", G_TYPE_STRING, dim_str.c_str(), NULL);
    gst_structure_set(structure, "types", G_TYPE_STRING, type_str.c_str(), NULL);
  }

  if (config.rate_n >= 0 && config.rate_d > 0) {
    gst_caps_set_simple(caps, "framerate", GST_TYPE_FRACTION, config.rate_n,
        config.rate_d, NULL);
    gst_structure_set(structure, "framerate", GST_TYPE_FRACTION, config.rate_n,
        config.rate_d, NULL);
  }

  gst_caps_append_structure(caps, structure);
  return caps;
}

static GstCaps *
get_tensor_caps(const GstTensorsConfig &config)
{
  //  other/tensor can only have one tensor
  if (config.info.num_tensors > 1)
    return nullptr;
  ;

  auto &_info = config.info.info[0];
  auto *caps = gst_caps_from_string(GST_TENSOR_CAP_DEFAULT);

  /* structure for backward compatibility */
  auto *structure = gst_structure_new_empty(TENSOR_MIME_TYPE);

  if (tensor_dimension_is_valid(_info.dimension)) {
    auto dim_str = gst_tensor_get_dimension_string(_info.dimension);
    gst_caps_set_simple(caps, "dimension", G_TYPE_STRING, dim_str.c_str(), NULL);

    dim_str = tensor_get_rank_dimension_string(_info.dimension, 4);
    gst_structure_set(structure, "dimension", G_TYPE_STRING, dim_str.c_str(), NULL);
  }

  if (_info.type != tensor_type::LAST) {
    gst_caps_set_simple(caps, "type", G_TYPE_STRING,
        get_tensor_type_string(_info.type).c_str(), NULL);
    gst_structure_set(structure, "type", G_TYPE_STRING,
        get_tensor_type_string(_info.type).c_str(), NULL);
  }

  if (config.rate_n >= 0 && config.rate_d > 0) {
    gst_caps_set_simple(caps, "framerate", GST_TYPE_FRACTION, config.rate_n,
        config.rate_d, NULL);
    gst_structure_set(structure, "framerate", GST_TYPE_FRACTION, config.rate_n,
        config.rate_d, NULL);
  }

  gst_caps_append_structure(caps, structure);
  return caps;
}


static GstCaps *
get_flexible_caps(const GstTensorsConfig &config)
{
  GstCaps *caps = gst_caps_from_string(GST_TENSORS_FLEX_CAP_DEFAULT);
  if (config.rate_n >= 0 && config.rate_d > 0) {
    gst_caps_set_simple(caps, "framerate", GST_TYPE_FRACTION, config.rate_n,
        config.rate_d, NULL);
  }

  return caps;
}

} // namespace

bool
validate_tensors_config(const GstTensorsConfig &config)
{
  return 0 <= config.rate_n && 0 < config.rate_d && validate_tensors_info(config.info);
}

guint
get_buffer_n_tensor(GstBuffer *buffer)
{
  return buffer ? gst_buffer_n_memory(buffer) : 0;
}

bool
structure_is_tensor_stream(const GstStructure *structure)
{
  const gchar *name = gst_structure_get_name(structure);
  return name && (g_str_equal(name, TENSOR_MIME_TYPE) || g_str_equal(name, TENSORS_MIME_TYPE));
}

bool
gst_tensors_config_from_structure(GstTensorsConfig &config, const GstStructure *structure)
{
  if (!structure) {
    return false;
  }

  tensor_format format = tensor_format::STATIC;

  std::string name = gst_structure_get_name(structure);
  if (name == TENSOR_MIME_TYPE) {
    config.info.num_tensors = 1;
    if (gst_structure_has_field(structure, "dimension")) {
      std::string dim_str = gst_structure_get_string(structure, "dimension");
      parse_tensor_dimension(dim_str, config.info.info[0].dimension);
    }

    if (gst_structure_has_field(structure, "type")) {
      std::string type_str = gst_structure_get_string(structure, "type");
      config.info.info[0].type = get_tensor_type(type_str);
    }
  } else if (name == TENSORS_MIME_TYPE) {
    if (gst_structure_has_field(structure, "format")) {
      std::string format_str = gst_structure_get_string(structure, "format");
      format = get_tensor_format(format_str);

      //  If format is not valid, use default format
      if (format != tensor_format::LAST) {
        config.info.format = format;
      }
    }

    if (config.info.format == tensor_format::STATIC) {
      gst_structure_get_int(structure, "num_tensors", (gint *) (&config.info.num_tensors));

      /* parse dimensions */
      if (gst_structure_has_field(structure, "dimensions")) {
        auto *dims_str = gst_structure_get_string(structure, "dimensions");
        auto num_dims = tensors_info_parse_dimensions_string(config.info, dims_str);
        if (config.info.num_tensors != num_dims) {
          //  Log warning if number of tensors and dimensions are not same
        }
      }

      /* parse types */
      if (gst_structure_has_field(structure, "types")) {
        const gchar *types_str;
        guint num_types;

        types_str = gst_structure_get_string(structure, "types");
        num_types = tensors_info_parse_types_string(config.info, types_str);

        if (config.info.num_tensors != num_types) {
          //  Log warning if number of tensors and types are not sameL
        }
      }
    }
  } else {
    //  Log warning for unsupported type
    return FALSE;
  }

  if (gst_structure_has_field(structure, "framerate")) {
    gst_structure_get_fraction(structure, "framerate", &config.rate_n, &config.rate_d);
  }

  return TRUE;
}

int
tensor_type_size(tensor_type type)
{
  switch (type) {
    case tensor_type::INT8:
    case tensor_type::UINT8:
      return 1;
    case tensor_type::INT16:
    case tensor_type::UINT16:
    case tensor_type::FLOAT16:
      return 2;
    case tensor_type::INT32:
    case tensor_type::UINT32:
    case tensor_type::FLOAT32:
      return 4;
    case tensor_type::INT64:
    case tensor_type::UINT64:
    case tensor_type::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

GstCaps *
get_possible_pad_caps_from_config(GstPad *pad, const GstTensorsConfig &config)
{
  auto *caps = gst_caps_new_empty();
  auto *templ = gst_pad_get_pad_template_caps(pad);

  /* append caps for static tensor */
  if (config.info.format == tensor_format::STATIC) {
    /* other/tensor */
    if (auto *tmp = get_tensor_caps(config)) {
      if (gst_caps_can_intersect(tmp, templ))
        gst_caps_append(caps, tmp);
      else
        gst_caps_unref(tmp);
    }

    /* other/tensors */
    if (auto *tmp = get_tensors_caps(config)) {
      if (gst_caps_can_intersect(tmp, templ))
        gst_caps_append(caps, tmp);
      else
        gst_caps_unref(tmp);
    }
  }

  /* caps for flexible tensor */
  if (auto *tmp = get_flexible_caps(config)) {
    if (gst_caps_can_intersect(tmp, templ))
      gst_caps_append(caps, tmp);
    else
      gst_caps_unref(tmp);
  }

  /* if no possible caps for given config, return null. */
  if (gst_caps_is_empty(caps)) {
    gst_caps_unref(caps);
    caps = nullptr;
  }

  gst_caps_unref(templ);
  return caps;
}

void
update_tensor_dimensions(GstCaps *caps, GstCaps *peer_caps)
{
  if (!caps || !peer_caps) {
    return;
  }
  for (uint32_t i = 0; i < gst_caps_get_size(caps); i++) {
    auto *structure = gst_caps_get_structure(caps, i);

    for (uint32_t j = 0; j < gst_caps_get_size(peer_caps); j++) {
      auto *structure_peer = gst_caps_get_structure(peer_caps, j);

      /* other/tensor */
      if (gst_structure_has_field(structure, "dimension")
          && gst_structure_has_field(structure_peer, "dimension")) {
        /* update dimensions for negotiation */
        if (structures_have_same_dimensions(structure, structure_peer)) {
          gst_structure_set(structure, "dimension", G_TYPE_STRING,
              gst_structure_get_string(structure_peer, "dimension"), NULL);
        }
      }
      /* other/tensors */
      else if (gst_structure_has_field(structure, "dimensions")
               && gst_structure_has_field(structure_peer, "dimensions")) {
        /* update dimensions for negotiation */
        if (have_same_dimensions(structure, structure_peer)) {
          gst_structure_set(structure, "dimensions", G_TYPE_STRING,
              gst_structure_get_string(structure_peer, "dimensions"), NULL);
        }
      }
    }
  }
}
