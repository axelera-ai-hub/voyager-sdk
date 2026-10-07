// Copyright Axelera AI, 2024
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

//  The C ABI below is what axelera/app/meta/gst.py reaches through ctypes. The
//  ctypes structures there mirror these by hand, so the two must be kept in step.
struct extern_named_meta {
  const char *name{}; // The name used to access the metadata
  extern_meta meta{}; // The metaddata chunk
};

struct all_extern_meta {
  int num_meta{}; // The total number of buffers returned
  extern_named_meta *meta{}; // The array of buffers
};

struct extern_indexed_submeta {
  extern_meta *meta_vector{}; // The metadata chunk
  int subframe_index{}; // The subframe index of the metadata chunk
  int num_extern_meta{}; // The size of the extern_meta vector
};

struct extern_named_submeta_collection {
  const char *name_submodel{}; // The name of the submodel
  extern_indexed_submeta *meta_indices{}; // The array of buffers for each index
  int subframe_number{}; // The total number of subframes
};

struct all_extern_submeta {
  const char *name_model{}; // The name of the model
  int num_submeta{}; // The total number of submodels
  extern_named_submeta_collection *meta_submodels{}; // The array of buffers for each submodel
};

struct all_extern_submeta_all_models {
  int num_meta{}; // The total number of models
  all_extern_submeta *meta_models{}; // The array of buffers for each model
};

extern "C" {
void set_meta_to_buffer(GstBuffer *buffer, const char *meta_key, const extern_meta *meta);
all_extern_meta get_meta_from_buffer(GstBuffer *buffer);
void free_meta(extern_named_meta *all_meta);
all_extern_submeta_all_models get_submeta_from_buffer(GstBuffer *buffer);
void free_submeta(all_extern_submeta_all_models *all_models);
}

GType gst_vaapi_video_meta_api_get_type(void);

#define GST_MAP_VAAPI (GST_MAP_FLAG_LAST << 1)
