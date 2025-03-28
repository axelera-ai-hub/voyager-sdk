#include <gst/gst.h>


#include "GstAxDecode.hpp"
#include "GstAxDistributor.hpp"
#include "GstAxFunnel.hpp"
#include "GstAxInPlace.hpp"
#include "GstAxInference.hpp"
#include "GstAxInferenceNet.hpp"
#include "GstAxTransform.hpp"

struct PluginInfo {
  const char *name;
  GType (*get_type)(void);
};

static const PluginInfo plugins[] = {
  { "axinferencenet", gst_axinferencenet_get_type },
  { "axinference", gst_axinference_get_type },
  { "axinplace", gst_axinplace_get_type },
  { "axtransform", gst_axtransform_get_type },
  { "axfunnel", gst_axfunnel_get_type },
  { "decode_muxer", gst_axdecoder_get_type },
  { "distributor", gst_distributor_get_type },
};

static gboolean
axstreamer_plugin_init(GstPlugin *plugin)
{
  for (const PluginInfo &info : plugins) {
    if (!gst_element_register(plugin, info.name, GST_RANK_NONE, info.get_type())) {
      GST_ERROR("Failed to register plugin : %s", info.name);
      return FALSE;
    }
  }
  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "axstreamer"
#endif
#define VERSION "0.1.0"

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, axstreamer, "axstreamer",
    axstreamer_plugin_init, VERSION, "LGPL", "axstreamer", "https://axelera.ai");
