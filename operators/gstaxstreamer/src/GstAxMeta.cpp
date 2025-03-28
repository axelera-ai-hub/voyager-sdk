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


struct extern_named_meta {
  const char *name{}; // The name used to access the metadata
  extern_meta meta{}; // The metaddata chunk
};

struct all_extern_meta {
  int num_meta{}; // The total number of buffers returned
  extern_named_meta *meta{}; // The array of buffers
};

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

  int total_size = std::accumulate(metadata->meta_map_ptr->begin(),
      metadata->meta_map_ptr->end(), 0, [](int sum, const auto &pair) {
        return sum + pair.second->get_extern_meta().size();
      });

  for (const auto &pair : metadata->extern_data) {
    total_size += pair.second.size();
  }
  all_meta.meta = new extern_named_meta[total_size];
  int i = 0;
  for (const auto &[name, values] : *metadata->meta_map_ptr) {
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
free_meta(extern_meta *all_meta)
{
  delete[] all_meta;
}


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

extern "C" all_extern_submeta_all_models
get_submeta_from_buffer(GstBuffer *buffer)
{
  all_extern_submeta_all_models all_models;
  auto *metadata = gst_buffer_get_general_meta(buffer);

  const auto &models_map = *metadata->meta_map_ptr;
  all_models.num_meta = models_map.size();
  all_models.meta_models = new all_extern_submeta[all_models.num_meta];
  int i_model = -1;
  for (const auto &[name_model, axmeta] : models_map) {
    ++i_model;
    all_models.meta_models[i_model].name_model = name_model.c_str();
    const auto &submeta_map = axmeta->submeta_map;
    if (!submeta_map) {
      all_models.meta_models[i_model].num_submeta = 0;
      all_models.meta_models[i_model].meta_submodels = nullptr;
      continue;
    }
    const auto [submodel_names, subframe_numbers] = submeta_map->contents();
    all_models.meta_models[i_model].num_submeta = submodel_names.size();
    if (submodel_names.empty()) {
      all_models.meta_models[i_model].meta_submodels = nullptr;
      continue;
    }
    all_models.meta_models[i_model].meta_submodels
        = new extern_named_submeta_collection[submodel_names.size()];
    for (int i_submodel = 0; i_submodel < submodel_names.size(); ++i_submodel) {
      all_models.meta_models[i_model].meta_submodels[i_submodel]
          = { submodel_names[i_submodel], nullptr, subframe_numbers[i_submodel] };
      if (subframe_numbers[i_submodel] == 0) {
        continue;
      }
      all_models.meta_models[i_model].meta_submodels[i_submodel].meta_indices
          = new extern_indexed_submeta[subframe_numbers[i_submodel]];
      for (int i_subframe = 0; i_subframe < subframe_numbers[i_submodel]; ++i_subframe) {
        auto *axmeta = submeta_map->get(submodel_names[i_submodel], i_subframe,
            subframe_numbers[i_submodel]);
        auto extern_meta_vector = axmeta->get_extern_meta();
        auto *p_meta = new extern_meta[extern_meta_vector.size()];
        std::copy(extern_meta_vector.begin(), extern_meta_vector.end(), p_meta);
        all_models.meta_models[i_model].meta_submodels[i_submodel].meta_indices[i_subframe]
            = { p_meta, i_subframe, static_cast<int>(extern_meta_vector.size()) };
      }
    }
  }
  return all_models;
}

extern "C" void
free_submeta(all_extern_submeta_all_models *all_models)
{
  for (int i_model = 0; i_model < all_models->num_meta; ++i_model) {
    for (int i_submodel = 0;
         i_submodel < all_models->meta_models[i_model].num_submeta; ++i_submodel) {
      delete[] all_models->meta_models[i_model].meta_submodels[i_submodel].meta_indices;
    }
    delete[] all_models->meta_models[i_model].meta_submodels;
  }
  delete[] all_models->meta_models;
}
