// Copyright 2024 The JAX SC Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_test_utils.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/concurrency/chain.h"  // from @xla
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/test.h"  // from @tsl

namespace jax_sc_embedding {
namespace rpc {
namespace {

struct Event {
  uint64_t time;
  std::function<void()> run;

  bool operator>(const Event& other) const { return time > other.time; }
};

class Simulator {
 public:
  Simulator(int num_tasks, int threads_per_task, uint32_t seed)
      : num_tasks_(num_tasks),
        threads_per_task_(threads_per_task),
        fake_env_(std::make_shared<FakeEnv>()),
        rng_(seed) {
    services_.reserve(num_tasks);
    for (int i = 0; i < num_tasks; ++i) {
      services_.push_back(std::make_unique<AllReduceServiceImpl>(
          /*task_id=*/i, /*num_tasks=*/num_tasks, threads_per_task,
          fake_env_.get()));
    }
  }

  void Schedule(uint64_t delay, std::function<void()> run) {
    events_.push({current_time_ + delay, std::move(run)});
  }

  bool Run(uint64_t max_ticks = 1000000) {
    while (!events_.empty() && events_.top().time < max_ticks) {
      Event event = events_.top();
      events_.pop();
      current_time_ = event.time;
      event.run();
    }
    return events_.empty();
  }

  uint64_t current_time() const { return current_time_; }
  int num_tasks() const { return num_tasks_; }
  int threads_per_task() const { return threads_per_task_; }

  AllReduceServiceImpl* service(int task_id) {
    return services_[task_id].get();
  }

  uint64_t NextRandomDelay(uint64_t min = 1, uint64_t max = 100) {
    std::uniform_int_distribution<uint64_t> dist(min, max);
    return dist(rng_);
  }

  std::mt19937& rng() { return rng_; }

 private:
  int num_tasks_;
  int threads_per_task_;
  std::shared_ptr<FakeEnv> fake_env_;
  std::vector<std::unique_ptr<AllReduceServiceImpl>> services_;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events_;
  uint64_t current_time_ = 0;
  std::mt19937 rng_;
};

// Unified helper to schedule sequential local contributions with optional
// chaos.
void ScheduleSequentialLocalContributions(
    Simulator& sim, int task_id, int key_index,
    const std::vector<int>& sync_keys,
    const std::vector<std::vector<bool>>& inputs,
    std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>>& results,
    bool inject_duplicates = false, int delay_task_id = -1,
    uint64_t extra_delay = 0) {
  if (key_index >= sync_keys.size()) return;
  int sync_key = sync_keys[key_index];
  bool value = inputs[key_index][task_id];

  sim.Schedule(sim.NextRandomDelay(), [&sim, task_id, key_index, sync_key,
                                       value, &sync_keys, &inputs, &results,
                                       inject_duplicates, delay_task_id,
                                       extra_delay]() {
    AllReduceData data;
    data.set_sync_key(sync_key);
    data.set_src_rank(task_id);
    data.set_bool_val(value);

    bool is_last_local =
        sim.service(task_id)->InitializeOrUpdateState(sync_key, data);
    results[key_index][task_id] = sim.service(task_id)->GetResult(sync_key);

    // Schedule next key when this one completes (sequentially).
    results[key_index][task_id].AndThen([&sim, task_id, key_index, &sync_keys,
                                         &inputs, &results, inject_duplicates,
                                         delay_task_id, extra_delay]() {
      ScheduleSequentialLocalContributions(
          sim, task_id, key_index + 1, sync_keys, inputs, results,
          inject_duplicates, delay_task_id, extra_delay);
    });

    if (is_last_local) {
      tsl::AsyncValueRef<AllReduceData> local_reduced_av =
          sim.service(task_id)->GetLocalReducedValue(sync_key);

      local_reduced_av.AndThen([&sim, task_id, sync_key, local_reduced_av,
                                inject_duplicates, delay_task_id,
                                extra_delay]() {
        AllReduceData local_data = local_reduced_av.get();
        for (int peer = 0; peer < sim.num_tasks(); ++peer) {
          if (peer == task_id) continue;

          uint64_t delay = sim.NextRandomDelay();
          if (task_id == delay_task_id) {
            delay += extra_delay;
          }

          sim.Schedule(delay, [&sim, peer, local_data]() {
            absl::Status s =
                sim.service(peer)->ContributeDataInternal(local_data);
            EXPECT_TRUE(s.ok()) << "ContributeData failed: " << s.message();
          });

          if (inject_duplicates && sim.rng()() % 2 == 0) {
            sim.Schedule(delay + sim.NextRandomDelay(10, 50), [&sim, peer,
                                                               local_data]() {
              absl::Status s =
                  sim.service(peer)->ContributeDataInternal(local_data);
              EXPECT_TRUE(s.ok())
                  << "Duplicate ContributeData failed: " << s.message();
            });
          }
        }
      });
    }
  });
}

TEST(AllReduceSimulationTest, RandomInterleaving) {
  const int num_tasks = 4;
  const int threads_per_task = 1;
  const std::vector<int> sync_keys = {100};
  const uint32_t seed = 12345;

  Simulator sim(num_tasks, threads_per_task, seed);
  std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> results(
      sync_keys.size(),
      std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));

  std::vector<std::vector<bool>> inputs = {{false, true, false, false}};
  bool expected_result = true;  // OR of inputs

  for (int i = 0; i < num_tasks; ++i) {
    ScheduleSequentialLocalContributions(sim, i, 0, sync_keys, inputs, results);
  }

  ASSERT_TRUE(sim.Run());

  for (int i = 0; i < num_tasks; ++i) {
    ASSERT_TRUE(results[0][i].IsAvailable());
    EXPECT_EQ(results[0][i]->bool_val(), expected_result);
    EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
  }
}

TEST(AllReduceSimulationTest, DuplicateInjection) {
  const int num_tasks = 4;
  const int threads_per_task = 1;
  const std::vector<int> sync_keys = {100};
  const uint32_t seed = 54321;

  Simulator sim(num_tasks, threads_per_task, seed);
  std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> results(
      sync_keys.size(),
      std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));

  std::vector<std::vector<bool>> inputs = {{false, false, true, false}};
  bool expected_result = true;

  for (int i = 0; i < num_tasks; ++i) {
    ScheduleSequentialLocalContributions(sim, i, 0, sync_keys, inputs, results,
                                         /*inject_duplicates=*/true);
  }

  ASSERT_TRUE(sim.Run());

  for (int i = 0; i < num_tasks; ++i) {
    ASSERT_TRUE(results[0][i].IsAvailable());
    EXPECT_EQ(results[0][i]->bool_val(), expected_result);
    EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
  }
}

TEST(AllReduceSimulationTest, LateRpcInjection) {
  const int num_tasks = 4;
  const int threads_per_task = 1;
  const std::vector<int> sync_keys = {100};
  const uint32_t seed = 98765;

  Simulator sim(num_tasks, threads_per_task, seed);
  std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> results(
      sync_keys.size(),
      std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));
  std::vector<std::vector<bool>> inputs = {{false, false, false, false}};

  for (int i = 0; i < num_tasks; ++i) {
    ScheduleSequentialLocalContributions(sim, i, 0, sync_keys, inputs, results);
  }

  ASSERT_TRUE(sim.Run());

  // Verify first run completed and cleaned up.
  for (int i = 0; i < num_tasks; ++i) {
    ASSERT_TRUE(results[0][i].IsAvailable());
    EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
  }

  // Inject a late RPC for sync_key = 100 to Task 0 (claiming to be from Task 1)
  AllReduceData late_data;
  late_data.set_sync_key(100);
  late_data.set_src_rank(1);
  late_data.set_bool_val(false);

  absl::Status s = sim.service(0)->ContributeDataInternal(late_data);
  EXPECT_TRUE(s.ok());

  // Verify that it was ignored and didn't create a state.
  EXPECT_EQ(sim.service(0)->GetActiveStatesCount(), 0);

  // Now run a new sync_key = 101 and verify it still works.
  const std::vector<int> new_sync_keys = {101};
  std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> new_results(
      new_sync_keys.size(),
      std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));
  for (int i = 0; i < num_tasks; ++i) {
    ScheduleSequentialLocalContributions(sim, i, 0, new_sync_keys, inputs,
                                         new_results);
  }

  ASSERT_TRUE(sim.Run());

  for (int i = 0; i < num_tasks; ++i) {
    ASSERT_TRUE(new_results[0][i].IsAvailable());
    EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
  }
}

TEST(AllReduceSimulationTest, StarvationProgress) {
  const int num_tasks = 4;
  const int threads_per_task = 1;
  const std::vector<int> sync_keys = {100};
  const uint32_t seed = 11111;

  Simulator sim(num_tasks, threads_per_task, seed);
  std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> results(
      sync_keys.size(),
      std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));
  std::vector<std::vector<bool>> inputs = {{true, false, false, false}};

  for (int i = 0; i < num_tasks; ++i) {
    // Delay Task 0's outgoing RPCs by a huge amount.
    ScheduleSequentialLocalContributions(sim, i, 0, sync_keys, inputs, results,
                                         /*inject_duplicates=*/false,
                                         /*delay_task_id=*/0,
                                         /*extra_delay=*/10000);
  }

  // Run for 5000 ticks (Task 0's RPCs should not have arrived yet).
  // Other tasks should be blocked.
  ASSERT_FALSE(sim.Run(5000));

  for (int i = 0; i < num_tasks; ++i) {
    if (i == 0) {
      EXPECT_TRUE(results[0][i].IsAvailable());
    } else {
      EXPECT_FALSE(results[0][i].IsAvailable());
      EXPECT_GT(sim.service(i)->GetActiveStatesCount(), 0);
    }
  }

  // Now run until completion (allow Task 0's delayed RPCs to execute).
  ASSERT_TRUE(sim.Run(20000));

  for (int i = 0; i < num_tasks; ++i) {
    ASSERT_TRUE(results[0][i].IsAvailable());
    EXPECT_TRUE(results[0][i]->bool_val());
    EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
  }
}

TEST(AllReduceSimulationTest, FuzzingSimulation) {
  const int num_tasks = 4;
  const int threads_per_task = 1;

  // Run 100 iterations with different random seeds and random inputs.
  for (int iter = 0; iter < 10000; ++iter) {
    uint32_t seed = 20000 + iter;
    Simulator sim(num_tasks, threads_per_task, seed);

    // We run 5 interleaved sync keys sequentially per task.
    std::vector<int> sync_keys = {100, 101, 102, 103, 104};
    std::vector<std::vector<tsl::AsyncValueRef<AllReduceData>>> results(
        sync_keys.size(),
        std::vector<tsl::AsyncValueRef<AllReduceData>>(num_tasks));

    // Random inputs for each key.
    std::vector<std::vector<bool>> inputs(sync_keys.size(),
                                          std::vector<bool>(num_tasks));
    std::vector<bool> expected_results(sync_keys.size());

    std::mt19937 local_rng(seed);
    for (int k = 0; k < sync_keys.size(); ++k) {
      bool expected = false;
      for (int i = 0; i < num_tasks; ++i) {
        inputs[k][i] = local_rng() % 2 == 1;
        expected = expected || inputs[k][i];
      }
      expected_results[k] = expected;
    }

    for (int i = 0; i < num_tasks; ++i) {
      ScheduleSequentialLocalContributions(sim, i, 0, sync_keys, inputs,
                                           results,
                                           /*inject_duplicates=*/true);
    }

    ASSERT_TRUE(
        sim.Run(100000));  // Larger timeout since we have 3 keys sequential

    for (int k = 0; k < sync_keys.size(); ++k) {
      for (int i = 0; i < num_tasks; ++i) {
        ASSERT_TRUE(results[k][i].IsAvailable())
            << "Key " << sync_keys[k] << " at task " << i
            << " failed to complete in iter " << iter;
        EXPECT_EQ(results[k][i]->bool_val(), expected_results[k]);
        EXPECT_EQ(sim.service(i)->GetActiveStatesCount(), 0);
      }
    }
  }
}

}  // namespace
}  // namespace rpc
}  // namespace jax_sc_embedding
