// Copyright Axelera AI, 2025
#pragma once
#include <barrier>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

namespace Ax
{
/// @brief A very simple thread pool for running a number of tasks in parallel
/// @tparam Arg
/// Construct with the number of threads to use.
template <typename Arg> class threaded_runner
{
  public:
  using Task = void(Arg &);

  struct packaged_task {
    Task *task;
    Arg arg;
  };

  explicit threaded_runner(size_t num_threads)
      : num_workers(num_threads),
        workers(num_threads),
        tasks(num_threads)
  {
    for (size_t i = 0; i < num_workers; ++i) {
      workers[i]
          = std::jthread([this, i](std::stop_token st) { worker_loop(i, st); });
    }
  }

  ~threaded_runner()
  {
    for (auto &w : workers) {
      w.request_stop();
    }
    cv.notify_all();
  }

  /// @brief Run a set of tasks in parrallel
  /// @param tasks - The tasks to execute
  /// Blocks until all tasks are complete
  void run(std::span<packaged_task> tasks_to_run)
  {
    auto num_tasks = tasks_to_run.size();
    auto first = tasks_to_run.begin();
    auto end = tasks_to_run.end();
    while (num_tasks > num_workers) {
      auto last = first + num_workers;
      run(std::span<packaged_task>(first, last));
      first = last;
      num_tasks -= num_workers;
    }
    {
      std::unique_lock<std::mutex> lock(m);
      auto it = std::copy(first, end, tasks.begin());
      std::fill(it, tasks.end(), packaged_task{ dummy, Arg{} });
      completed = 0;
      ++generation;
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lock(m);
    cv_done.wait(lock, [this] { return completed >= num_workers; });
  }

  private:
  size_t num_workers;
  std::vector<std::jthread> workers;
  std::vector<packaged_task> tasks;

  std::mutex m;
  std::condition_variable cv;
  std::condition_variable cv_done;
  std::atomic<uint64_t> generation{ 0 };
  std::atomic<int> completed{ 0 };

  static void dummy(Arg & /*unused*/)
  {
  }

  void worker_loop(size_t id, std::stop_token st)
  {
    uint64_t last_generation = 0;
    while (!st.stop_requested()) {
      Task *task;
      Arg *arg;
      uint64_t current_generation;
      {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this, &st, &last_generation] {
          return generation > last_generation || st.stop_requested();
        });
        if (st.stop_requested()) {
          return;
        }
        current_generation = generation;
        task = tasks[id].task;
        arg = &tasks[id].arg;
      }
      task(*arg);
      {
        std::unique_lock<std::mutex> lock(m);
        last_generation = current_generation;
        completed++;
        if (completed == num_workers) {
          cv_done.notify_one();
        }
      }
    }
  }
};

} // namespace Ax
