// Copyright Axelera AI, 2025
#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

class AxMetaStreamId : public AxMetaBase
{
  public:
  int stream_id = 0;
  std::uint64_t timestamp{};
  std::atomic<int> inference_count = 0;
  // The copy of the inference count is necessary to access it in a
  // thread-safe manner, and prevent it going out of scope after
  // reading before it can be read by Python.
  mutable int inferences{};

  explicit AxMetaStreamId(int stream_id)
      : stream_id{ stream_id }
  {
    auto now = std::chrono::high_resolution_clock::now();
    timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
                    .count();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "stream_meta";
    inferences = inference_count.load();
    auto results = std::vector<extern_meta>{
      { class_meta, "stream_id", int(sizeof(stream_id)),
          reinterpret_cast<const char *>(&stream_id) },
      { class_meta, "timestamp", int(sizeof(timestamp)),
          reinterpret_cast<const char *>(&timestamp) },
      { class_meta, "inferences", int(sizeof(inferences)),
          reinterpret_cast<const char *>(&inferences) },
    };
    return results;
  }
};
