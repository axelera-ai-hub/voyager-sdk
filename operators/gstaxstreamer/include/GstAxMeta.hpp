#pragma once

#include <gst/gst.h>
#include <memory>
#include <string>
#include <unordered_map>
#include "AxMeta.hpp"

struct GstMetaGeneral {
  GstMeta meta;
  unsigned int subframe_index;
  unsigned int subframe_number;
  // At this point we don't have visibilty to AxMetaBase child classes to create and put object to meta_map_ptr
  std::unordered_map<std::string, std::vector<extern_meta_container>> extern_data;
  std::shared_ptr<std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>> meta_map_ptr;
};

GstMetaGeneral *gst_buffer_get_general_meta(GstBuffer *buffer);
gboolean gst_buffer_has_general_meta(GstBuffer *buffer);
