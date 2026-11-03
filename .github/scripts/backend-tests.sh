#!/usr/bin/env bash
# Build + tests of one backend, one phase per workflow step: each step has its own timeout and publishes its own
# annotations, so a slow or hanging test cannot lose the results of the others. Run at the repository root.
#   backend-tests.sh cpu|metal|vulkan|cuda|rocm <out-dir> build|probe|candle-core|candle-nn|candle-flash-attn|reports|types|stopped
# Env: REQUIRE_DEVICE=1 : the job exists for this device, so a failed probe fails it instead of being NOT RUN.
set -euo pipefail
BACKEND="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; export OUT; PHASE="$3"
. "$(dirname "$0")/lib.sh"

# hip-sys, hip-runtime and candle-rocm link ROCm, so they are only built where it is installed. candle-flash-attn has no
# `metal` feature. `--features default` for cpu: an empty array breaks `set -u` in the bash 3.2 of macOS.
case "$BACKEND" in
  cpu)    FEAT=(--features default); PKGS=(candle-core candle-nn candle-flash-attn) ;;
  metal)  FEAT=(--features metal);   PKGS=(candle-core candle-nn) ;;
  vulkan) FEAT=(--features vulkan);  PKGS=(candle-core candle-nn candle-flash-attn) ;;
  cuda)   FEAT=(--features cuda);    PKGS=(candle-core candle-nn candle-flash-attn) ;;
  rocm)   FEAT=(--features rocm);    PKGS=(candle-core candle-nn candle-flash-attn candle-rocm hip-sys hip-runtime) ;;
  *) echo "unknown backend $BACKEND"; exit 2 ;;
esac
SEL=(); for p in "${PKGS[@]}"; do SEL+=(-p "$p"); done

# The tests of accel_kernels_host.
HOST_KERNELS=(
  fa_gqa_packed_rows_bit_exact fa_row_kernel fa_tile_v2_kernels fa_tile_v3_kernels fa_varlen_kernels
  fa_half_kv_with_f32_queries fa_paged_row_kernel fa_combine_matches_previous_formula element_decoders_exhaustive
  tq_attention_fp8_fast_and_paged tq_attention_gqa_packed_bit_exact tq_quantize_matches_cpu_codec
  tq_quantize_half_inputs tq_attention_matches_reference
)
# Host-only tests that compile the same with every feature: the cpu job runs them, a GPU job would only repeat them.
HOST_ONLY=("${HOST_KERNELS[@]}" cpu_matches_f64_reference host_emulated_kernels)

# suite <package> <libtest args...> : a GPU job compiles the CPU tests too, the doc tests would only repeat the cpu job.
suite() {
  local p="$1"; shift
  if [ "$BACKEND" = cpu ]; then
    run "suite-$p" cargo test --release --no-fail-fast -p "$p" "${FEAT[@]}" -- --nocapture "$@"
  else
    run "suite-$p" cargo test --release --lib --tests --no-fail-fast -p "$p" "${FEAT[@]}" -- --nocapture --test-threads 1 "$@"
  fi
}

# device_ok <backend> : the probe runs a small matmul on the device, it does not only enumerate.
device_ok() { grep -q "DEVICE_PROBE $1 OK" "$OUT/probe-$1.log"; }

# kernel_images : the build script writes an empty image for a skipped kernel, and a job that embeds those proves nothing.
kernel_images() {
  local dir; dir="$(ls -dt target/debug/build/candle-core-*/out 2>/dev/null | head -1)"
  if [ -z "$dir" ] || [ -z "$(ls "$dir"/*.co 2>/dev/null)" ]; then
    fail_step kernel-images "no kernel images under target/debug/build/candle-core-*/out"
    return 0
  fi
  local empty; empty="$(find "$dir" -maxdepth 1 -name '*.co' -size 0 -exec basename {} \; | tr '\n' ' ')"
  if [ -n "$empty" ]; then fail_step kernel-images "empty images: $empty"; return 0; fi
  echo "| kernel-images | PASS ($(ls "$dir"/*.co | wc -l | tr -d ' ') images) | |" >> "$SUMMARY"
  echo "::notice title=kernel images::$(cd "$dir" && wc -c ./*.co | sed 's|\./||' | awk '{printf "%s %.1f MB ;; ", $2, $1 / 1048576}')"
}

# runtime_kernels : kernels/*.hip are compiled by hip-runtime at run time, so nothing else in CI ever compiles them.
runtime_kernels() {
  local root="$ROCM_PATH" tmp f; tmp="$(mktemp -d)"
  if [ "${RUNNER_OS:-}" = Windows ]; then root="$(cygpath -u "$root")"; fi
  local hipcc="$root/bin/hipcc"
  # The rule of toolchain::device_lib_dir: the Windows tarball has no <root>/amdgcn.
  [ -d "$root/amdgcn/bitcode" ] || export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
  for f in kernels/*.hip; do
    run "hipcc-$(basename "$f" .hip)" "$hipcc" --genco --offload-arch=gfx1030 -O3 -o "$tmp/$(basename "$f" .hip).co" "$f"
  done
}

case "$PHASE" in
build)
  case "$BACKEND" in
  cuda)
    # Links nothing: no NVIDIA runtime here. The kernels of candle-kernels are compiled by the real nvcc.
    run check-cuda cargo check "${SEL[@]}" "${FEAT[@]}" --tests
    ;;
  rocm)
    # Debug profile: nothing here is executed, and the kernels are -O3 either way.
    run build-rocm cargo test --no-run "${SEL[@]}" "${FEAT[@]}"
    kernel_images
    runtime_kernels
    ;;
  *)
    for p in "${PKGS[@]}"; do run "build-$p" cargo test --release --no-run -p "$p" "${FEAT[@]}"; done
    ;;
  esac
  ;;
probe)
  if [ "$BACKEND" = cpu ]; then
    run probe-cpu cargo run --release -p candle-core --example device_probe
    # Row results independent of the thread count: the digest of this 1-thread run is asserted by the all-threads run in suite-candle-core.
    run m-invariance-1thread env RAYON_NUM_THREADS=1 cargo test --release -p candle-core --test matmul_m_invariance_tests -- --nocapture --test-threads 1
    grep -oE 'digest [0-9a-f]{16}' "$OUT/m-invariance-1thread.log" | head -1 | cut -d' ' -f2 > "$OUT/digest" || true
  else
    run "probe-$BACKEND" cargo run --release -p candle-core --features "$BACKEND" --example device_probe
    if ! device_ok "$BACKEND"; then
      why="$(grep -E "DEVICE_PROBE $BACKEND" "$OUT/probe-$BACKEND.log" | cut -c1-300 | escape)"
      if [ "${REQUIRE_DEVICE:-0}" = 1 ]; then fail_step "probe-$BACKEND" "no $BACKEND device: $why"
      else not_run "tests on $BACKEND" "no $BACKEND device on this runner: $why"; fi
    fi
  fi
  ;;
candle-core)
  if [ "$BACKEND" = cpu ]; then
    if [ -s "$OUT/digest" ]; then
      CANDLE_MATMUL_INVARIANCE_DIGEST="$(cat "$OUT/digest")"; export CANDLE_MATMUL_INVARIANCE_DIGEST
    else
      not_run m-invariance-allthreads "no digest from the 1-thread run"
    fi
    # The same kernel sources hipcc and NVRTC compile, on the host emulator (no GPU needed). Every workgroup is 128 host
    # threads at a barrier: 30 s on 4 Linux cores, over 55 minutes on the 3 cores of the Mac runner without finishing.
    if [ "${RUNNER_OS:-}" = macOS ]; then
      not_run accel_kernels_host "too slow on the 3-core Mac runner; the same sources run in linux-cpu and windows-cpu"
      skips=(--exact); for t in "${HOST_KERNELS[@]}"; do skips+=(--skip "$t"); done
      suite candle-core "${skips[@]}"
    else
      suite candle-core
    fi
    refuse_skip suite-candle-core "skipping (accel|delta rule) host kernel tests"
  else
    skips=(--exact); for t in "${HOST_ONLY[@]}"; do skips+=(--skip "$t"); done
    suite candle-core "${skips[@]}"
    refuse_skip suite-candle-core "no GPU, skipping|skipping vulkan accel tests"
  fi
  ;;
candle-nn)
  suite candle-nn
  refuse_skip suite-candle-nn "skipping: "
  ;;
candle-flash-attn)
  suite candle-flash-attn
  ;;
reports)
  # Report-only tests, they never fail: the measurement line is what proves they ran.
  case "$BACKEND" in
  metal)
    run m-invariance-metal cargo test --release -p candle-core --features metal --test matmul_m_invariance_tests metal_ -- --ignored --nocapture
    require_measured m-invariance-metal "\\[METAL-M-INV\\] 측정 결과"
    ;;
  vulkan)
    run m-invariance-vulkan cargo test --release -p candle-core --features vulkan --test matmul_m_invariance_tests vulkan_ -- --ignored --nocapture
    require_measured m-invariance-vulkan "M-INV\\] 측정 결과"
    ;;
  esac
  ;;
types)
  # Feature sets no other job builds, and the benches, which no test run compiles. The OS does not change any of them.
  run check-all-targets cargo check "${SEL[@]}" --all-targets
  run check-no-default-features cargo check "${SEL[@]}" --no-default-features --all-targets
  run check-opencl cargo check -p candle-core --features opencl --all-targets
  ;;
stopped)
  # A step killed by its timeout never reached `run`'s summary row. With --nocapture its log ends at the test that ran.
  for f in "$OUT"/*.log; do
    label="$(basename "$f" .log)"
    if [ -f "$f" ] && ! grep -q "^| $label |" "$SUMMARY"; then
      echo "::error title=$label did not finish::$(tail -n 12 "$f" | sed 's/$/ |/' | tr -d '\n' | cut -c1-2500 | escape)"
    fi
  done
  exit 0
  ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
finish
