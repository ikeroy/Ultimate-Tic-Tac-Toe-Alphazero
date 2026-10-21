py_mod = import('python')
py3_install = py_mod.find_installation('python3', pure: false)
py3_dep = py3_install.dependency()
pybind11_proj = subproject('pybind11')
pybind11_dep = pybind11_proj.get_variable('pybind11_dep')

absl_proj = subproject('abseil-cpp')
absl_container_dep = absl_proj.get_variable('absl_container_dep')
absl_hash_dep = absl_proj.get_variable('absl_hash_dep')

# eigen uses meson.override_dependency in newer versions
eigen_dep = dependency('eigen3')

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')

gbench_proj = subproject('google-benchmark')
gbench_main_dep = gbench_proj.get_variable('google_benchmark_main_dep')
gbench_dep = gbench_proj.get_variable('google_benchmark_dep')

thread_dep = dependency('threads')

# Tracy profiler (opt-in via -Dtracy_enabled=true)
tracy_enabled = get_option('tracy_enabled')
tracy_args = []
tracy_dep = []
if tracy_enabled
  tracy_proj = subproject('tracy')
  tracy_dep = tracy_proj.get_variable('tracy_dep')
  tracy_args = ['-DTRACY_ENABLE']
endif

lib_args = ['-DBUILDLIB'] + tracy_args

uttt_gs = library(
    'uttt_gs',
    'uttt_gs.cc',
    dependencies: [eigen_dep, absl_hash_dep],
    cpp_args: lib_args,
)

uttt_gs_test = executable(
  'uttt_gs_test',
  'uttt_gs_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep],
  link_with: [uttt_gs],
)
test('gtest tests', uttt_gs_test)

mcts = library(
    'mcts',
    'mcts.cc',
    dependencies: [eigen_dep, absl_hash_dep, tracy_dep],
    cpp_args: lib_args,
)

game_state_lib = library(
  'game_state',
  'game_state.cc',
  dependencies: [eigen_dep, absl_hash_dep],
  cpp_args: lib_args,
)

mcts_test = executable(
  'mcts_test',
  'mcts_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep],
  link_with: [mcts, uttt_gs, game_state_lib],
)
test('gtest tests', mcts_test)

concurrent_queue_test = executable(
  'concurrent_queue_test',
  'concurrent_queue_test.cc',
  dependencies: [gtest_dep],
  link_with: [],
)
test('gtest tests', concurrent_queue_test)

s3fifo_cache_test = executable(
  's3fifo_cache_test',
  's3fifo_cache_test.cc',
  dependencies: [gtest_dep, absl_container_dep, absl_hash_dep],
  link_with: [],
)
test('gtest tests', s3fifo_cache_test)

play_manager = library(
  'play_manager',
  'play_manager.cc',
  dependencies: [eigen_dep, absl_container_dep, absl_hash_dep, thread_dep, tracy_dep],
  link_with: [mcts, game_state_lib],
  cpp_args: lib_args,
)

play_manager_test = executable(
  'play_manager_test',
  'play_manager_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_container_dep, absl_hash_dep],
  link_with: [play_manager, uttt_gs],
)
test('gtest tests', play_manager_test)

play_manager_bench = executable(
  'play_manager_bench',
  'play_manager_bench.cc',
  dependencies: [gbench_main_dep, gbench_dep, thread_dep, eigen_dep, absl_container_dep, absl_hash_dep],
  link_with: [play_manager, uttt_gs],
)

py3_install.extension_module(
  'alphazero',
  sources: ['py_wrapper.cc'],
  link_with: [uttt_gs, play_manager, mcts, game_state_lib],
  dependencies : [pybind11_dep, py3_dep, eigen_dep, absl_container_dep, absl_hash_dep, tracy_dep],
  cpp_args: lib_args,
  install: true,
)
