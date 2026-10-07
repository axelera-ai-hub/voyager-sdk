// Copyright Axelera AI, 2024
#include "GstAxMeta.hpp"

#include <numeric>
#include <string>

#define GST_GENERAL_META_API_TYPE (gst_general_meta_api_get_type())
#define GST_GENERAL_META_INFO (gst_general_meta_get_info())


GType
gst_general_meta_api_get_type(void)
{
  static const gchar *tags[] = { NULL };
  static GType type;
  if (g_once_init_enter(&type)) {
    GType _type = gst_meta_api_type_register("GstMetaGeneralAPI", tags);
    g_once_init_leave(&type, _type);
  }
  return type;
}


static gboolean
gst_general_meta_init(GstMeta *meta, gpointer params, GstBuffer *buffer)
{
  new (meta) GstMetaGeneral;
  return TRUE;
}


static void
gst_general_meta_free(GstMeta *meta, GstBuffer *buffer)
{
  reinterpret_cast<GstMetaGeneral *>(meta)->~GstMetaGeneral();
}


static gboolean gst_general_meta_transform(GstBuffer *transbuf, GstMeta *meta,
    GstBuffer *buffer, GQuark type, gpointer data);


const GstMetaInfo *
gst_general_meta_get_info(void)
{
  static const GstMetaInfo *gst_general_meta_info = NULL;

  if (g_once_init_enter(&gst_general_meta_info)) {
    const GstMetaInfo *meta = gst_meta_register(GST_GENERAL_META_API_TYPE, /* api type */
        "GstMetaGeneral", /* implementation type */
        sizeof(GstMetaGeneral), /* size of the structure */
        gst_general_meta_init, gst_general_meta_free, gst_general_meta_transform);
    g_once_init_leave(&gst_general_meta_info, meta);
  }
  return gst_general_meta_info;
}


static gboolean
gst_general_meta_transform(GstBuffer *transbuf, GstMeta *meta,
    GstBuffer *buffer, GQuark type, gpointer data)
{
  GstMetaGeneral *old_meta = reinterpret_cast<GstMetaGeneral *>(meta);

  GstMetaGeneral *new_meta = reinterpret_cast<GstMetaGeneral *>(
      gst_buffer_add_meta(transbuf, GST_GENERAL_META_INFO, NULL));
  if (!new_meta) {
    GST_ERROR("gst_general_meta_transform: failed to transform meta");
    return FALSE;
  }

  new_meta->subframe_index = old_meta->subframe_index;
  new_meta->subframe_number = old_meta->subframe_number;
  new_meta->meta_map_ptr = old_meta->meta_map_ptr;
  new_meta->extern_data = std::move(old_meta->extern_data);

  return TRUE;
}


// THE MAP IS CONSTRUCTED HERE! ON THE FIRST CALL OF THE GET FUNCTION
GstMetaGeneral *
gst_buffer_get_general_meta(GstBuffer *buffer)
{
  GstMetaGeneral *meta = reinterpret_cast<GstMetaGeneral *>(
      gst_buffer_get_meta((buffer), GST_GENERAL_META_API_TYPE));

  if (!meta) {
    meta = reinterpret_cast<GstMetaGeneral *>(
        gst_buffer_add_meta(buffer, GST_GENERAL_META_INFO, NULL));
    if (!meta) {
      GST_ERROR("gst_buffer_get_general_meta: failed to add meta");
      return NULL;
    }
    meta->meta_map_ptr
        = std::make_shared<std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>>();
    meta->meta_map_ptr->reserve(16);
    meta->subframe_index = 0;
    meta->subframe_number = 1;
    meta->extern_data.clear();
  }
  return meta;
}


gboolean
gst_buffer_has_general_meta(GstBuffer *buffer)
{
  GstMetaGeneral *meta = reinterpret_cast<GstMetaGeneral *>(
      gst_buffer_get_meta((buffer), GST_GENERAL_META_API_TYPE));
  return meta != NULL;
}


extern "C" void
set_meta_to_buffer(GstBuffer *buffer, const char *meta_key, const extern_meta *meta)
{
  auto *metadata = gst_buffer_get_general_meta(buffer);
  auto key = std::string(meta_key);
  metadata->extern_data[key].push_back(
      extern_meta_container(meta->type, meta->subtype, meta->meta_size, meta->meta));
}

extern "C" all_extern_meta
get_meta_from_buffer(GstBuffer *buffer)
{

  all_extern_meta all_meta;
  if (!gst_buffer_has_general_meta(buffer)) {
    return all_meta;
  }

  auto *metadata = gst_buffer_get_general_meta(buffer);

  size_t total_size = std::accumulate(metadata->meta_map_ptr->begin(),
      metadata->meta_map_ptr->end(), (size_t) 0, [](size_t sum, const auto &pair) {
        if (!pair.second->enable_extern) {
          return sum;
        }
        return sum + pair.second->get_extern_meta().size();
      });

  for (const auto &pair : metadata->extern_data) {
    total_size += pair.second.size();
  }
  all_meta.meta = new extern_named_meta[total_size];
  int i = 0;
  for (const auto &[name, values] : *metadata->meta_map_ptr) {
    if (!values->enable_extern) {
      continue;
    }
    auto meta = values->get_extern_meta();
    for (const auto &m : meta) {
      all_meta.meta[i].name = name.c_str();
      all_meta.meta[i].meta = m;
      ++i;
    }
  }
  for (const auto &[key, value] : metadata->extern_data) {
    for (const auto &m : value) {
      all_meta.meta[i].name = key.c_str();
      all_meta.meta[i].meta.type = m.type.c_str();
      all_meta.meta[i].meta.subtype = m.subtype.c_str();
      all_meta.meta[i].meta.meta_size = m.meta.size();
      all_meta.meta[i].meta.meta = m.meta.data();
      ++i;
    }
  }
  all_meta.num_meta = i;
  return all_meta;
}

extern "C" void
free_meta(extern_named_meta *all_meta)
{
  delete[] all_meta;
}


extern "C" all_extern_submeta_all_models
get_submeta_from_buffer(GstBuffer *buffer)
{
  all_extern_submeta_all_models all_models;
  auto *metadata = gst_buffer_get_general_meta(buffer);
  const auto &models_map = *metadata->meta_map_ptr;
  all_models.num_meta = std::count_if(models_map.begin(), models_map.end(),
      [](const auto &kv) { return kv.second->enable_extern; });
  if (!all_models.num_meta) {
    return all_models;
  }
  all_models.meta_models = new all_extern_submeta[all_models.num_meta];
  int i_model = -1;
  for (const auto &[name_model, axmeta] : models_map) {
    if (!axmeta->enable_extern) {
      continue;
    }
    ++i_model;
    all_models.meta_models[i_model].name_model = name_model.c_str();
    const auto submodel_names = axmeta->submeta_names();
    all_models.meta_models[i_model].num_submeta = submodel_names.size();
    if (submodel_names.empty()) {
      all_models.meta_models[i_model].meta_submodels = nullptr;
      continue;
    }
    all_models.meta_models[i_model].meta_submodels
        = new extern_named_submeta_collection[submodel_names.size()];
    for (int i_submodel = 0; i_submodel < submodel_names.size(); ++i_submodel) {
      const auto submeta_vector = axmeta->get_submetas(submodel_names[i_submodel]);
      int subframe_number = submeta_vector.size();
      all_models.meta_models[i_model].meta_submodels[i_submodel]
          = { submodel_names[i_submodel], nullptr, subframe_number };
      if (subframe_number == 0) {
        continue;
      }
      all_models.meta_models[i_model].meta_submodels[i_submodel].meta_indices
          = new extern_indexed_submeta[subframe_number];
      for (int i_subframe = 0; i_subframe < subframe_number; ++i_subframe) {
        auto *axmeta = submeta_vector[i_subframe];
        int extern_meta_size = 0;
        extern_meta *p_meta = nullptr;
        if (axmeta && axmeta->enable_extern) {
          auto extern_meta_vector = axmeta->get_extern_meta();
          extern_meta_size = extern_meta_vector.size();
          p_meta = new extern_meta[extern_meta_size];
          std::copy(extern_meta_vector.begin(), extern_meta_vector.end(), p_meta);
        }
        all_models.meta_models[i_model].meta_submodels[i_submodel].meta_indices[i_subframe]
            = { p_meta, i_subframe, extern_meta_size };
      }
    }
  }
  return all_models;
}

extern "C" void
free_submeta(all_extern_submeta_all_models *all_models)
{
  for (int i_model = 0; i_model < all_models->num_meta; ++i_model) {
    auto &model = all_models->meta_models[i_model];
    for (int i_submodel = 0; i_submodel < model.num_submeta; ++i_submodel) {
      auto &submodel = model.meta_submodels[i_submodel];
      //  get_submeta_from_buffer() allocates one extern_meta array per subframe,
      //  so each has to be released before the array of subframes that holds them.
      for (int i_subframe = 0; i_subframe < submodel.subframe_number; ++i_subframe) {
        delete[] submodel.meta_indices[i_subframe].meta_vector;
      }
      delete[] submodel.meta_indices;
    }
    delete[] model.meta_submodels;
  }
  delete[] all_models->meta_models;
}


GType
gst_vaapi_video_meta_api_get_type(void)
{
  const GstMetaInfo *info = gst_meta_get_info("GstVaapiVideoMeta");
  return info ? info->api : 0;
}
