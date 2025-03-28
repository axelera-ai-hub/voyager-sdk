#include <gst/gst.h>
#include <gst/video/video.h>
#include "AxMeta.hpp"
#include "GstAxMeta.hpp"
#include "GstAxMetaToHailo.hpp"

#include "AxMetaClassification.hpp"
#include "AxMetaObjectDetection.hpp"

#include "gsthailometa/gst_hailo_meta.hpp" // get_hailo_main_roi
#include "hailo_common.hpp" // add_detections


GST_ELEMENT_REGISTER_DEFINE(axmetatohailo, "axmetatohailo", GST_RANK_NONE, GST_TYPE_AXMETATOHAILO);

GST_DEBUG_CATEGORY_STATIC(gst_axmetatohailo_debug_category);
#define GST_CAT_DEFAULT gst_axmetatohailo_debug_category

G_DEFINE_TYPE_WITH_CODE(GstAxmetatohailo, gst_axmetatohailo, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_axmetatohailo_debug_category, "axmetatohailo",
        0, "debug category for axmetatohailo element"));

static void
gst_axmetatohailo_init(GstAxmetatohailo *axmetatohailo)
{
  gst_debug_category_get_threshold(gst_axmetatohailo_debug_category);
}

static GstFlowReturn
gst_axmetatohailo_transform_ip(GstBaseTransform *trans, GstBuffer *buffer)
{
  GstCaps *caps = gst_pad_get_current_caps(trans->sinkpad);
  GstVideoInfo *info = gst_video_info_new();
  if (!gst_video_info_from_caps(info, caps)) {
    throw std::runtime_error("Cannot get video info from caps in gst_axmetatohailo_transform_ip");
  }
  float inv_width = 1.0 / GST_VIDEO_INFO_WIDTH(info);
  float inv_height = 1.0 / GST_VIDEO_INFO_HEIGHT(info);
  gst_video_info_free(info);
  gst_caps_unref(caps);

  GstMetaGeneral *gst_meta = gst_buffer_get_general_meta(buffer);
  if (gst_meta->subframe_index != 0 || gst_meta->subframe_number != 1) {
    throw std::runtime_error("axmetatohailo: does not work with subframes");
  }
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map
      = *gst_meta->meta_map_ptr;
  HailoROIPtr hailo_roi = get_hailo_main_roi(buffer, true);
  for (const auto &[name, meta_p] : meta_map) {
    if (AxMetaObjDetection *ax_meta = dynamic_cast<AxMetaObjDetection *>(meta_p.get());
        ax_meta != nullptr) {
      std::vector<HailoDetection> detections;
      for (int i = 0; i < ax_meta->num_elements(); ++i) {
        box_ltxywh box = ax_meta->get_box_ltxywh(i);
        HailoBBox h_box(inv_width * box.x, inv_height * box.y,
            inv_width * box.w, inv_height * box.h);
        int class_id = ax_meta->class_id(i);
        HailoDetection detection(h_box, class_id,
            std::string("Detection: ") + std::to_string(class_id), ax_meta->score(i));
        detections.push_back(detection);
      }
      hailo_common::add_detections(hailo_roi, detections);
    }
    if (AxMetaClassification *ax_meta
        = dynamic_cast<AxMetaClassification *>(meta_p.get());
        ax_meta != nullptr) {
      auto labels = ax_meta->get_labels();
      auto scores = ax_meta->get_scores();
      auto classes = ax_meta->get_classes();
      hailo_common::add_classification(hailo_roi, std::string("imagenet"),
          std::string("Classification: ") + labels[0][0], scores[0][0], classes[0][0]);
    }
  }
  meta_map.clear();
  return GST_FLOW_OK;
}

static void
gst_axmetatohailo_class_init(GstAxmetatohailoClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
      "axmetatohailo", "Effect", "description", "axelera.ai");

  GST_BASE_TRANSFORM_CLASS(klass)->transform_ip
      = GST_DEBUG_FUNCPTR(gst_axmetatohailo_transform_ip);

  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_CAPS_ANY));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_CAPS_ANY));
}

static gboolean
axmetatohailo_plugin_init(GstPlugin *plugin)
{
  return GST_ELEMENT_REGISTER(axmetatohailo, plugin);
}

#ifndef PACKAGE
#define PACKAGE "axmetatohailo"
#endif
#define VERSION "0.1.0"

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, axmetatohailo, "axmetatohailo",
    axmetatohailo_plugin_init, VERSION, "LGPL", "axstreamer", "https://axelera.ai");
