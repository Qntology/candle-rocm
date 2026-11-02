#!/usr/bin/env bash
# Build + tests of one backend, one phase per workflow step (each step has its own timeout and publishes its own
# annotations, so a slow later phase cannot lose the earlier results). Run at the repository root.
#   backend-tests.sh cpu|metal|vulkan|cuda|rocm <out-dir> build|suite
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

# A GPU job compiles the CPU tests too, the doc tests would only repeat what the cpu job ran.
if [ "$BACKEND" = cpu ]; then TARGETS=(--release); else TARGETS=(--release --lib --tests); fi

# suite <libtest args...> : every package on its own, so that a failure in one still reports the others.
suite() {
  for p in "${PKGS[@]}"; do
    run "suite-$p" cargo test "${TARGETS[@]}" --no-fail-fast -p "$p" "${FEAT[@]}" -- --nocapture "$@"
  done
}

# device_ok <backend> <labels of what depends on it...> : the probe runs a small matmul on the device, not only enumerates.
device_ok() {
  local b="$1"; shift
  grep -q "DEVICE_PROBE $b OK" "$OUT/probe-$b.log" && return 0
  if [ "${REQUIRE_DEVICE:-0}" = 1 ]; then
    fail_step "probe-$b" "no $b device: $(grep -E "DEVICE_PROBE $b" "$OUT/probe-$b.log" | cut -c1-300 | escape)"
  else
    for s in "$@"; do not_run "$s" "no $b device on this runner (see probe-$b.log)"; done
  fi
  return 1
}

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
  local hipcc="$ROCM_PATH/bin/hipcc" tmp; tmp="$(mktemp -d)"
  [ "${RUNNER_OS:-}" = Windows ] && hipcc="$ROCM_PATH/bin/hipcc.exe"
  local f
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
    if [ "$BACKEND" = cpu ] && [ "${RUNNER_OS:-}" = Linux ]; then
      run check-no-default-features cargo check "${SEL[@]}" --no-default-features --all-targets
      run check-opencl cargo check -p candle-core --features opencl --all-targets
    fi
    ;;
  esac
  ;;
suite)
  case "$BACKEND" in
  cpu)
    run probe-cpu cargo run --release -p candle-core --example device_probe
    # Row results independent of the thread count: the digest of this 1-thread run is asserted by the run in suite-candle-core.
    run m-invariance-1thread env RAYON_NUM_THREADS=1 cargo test --release -p candle-core --test matmul_m_invariance_tests -- --nocapture --test-threads 1
    digest=$(grep -oE 'digest [0-9a-f]{16}' "$OUT/m-invariance-1thread.log" | head -1 | cut -d' ' -f2 || true)
    if [ -n "$digest" ]; then
      export CANDLE_MATMUL_INVARIANCE_DIGEST="$digest"
    else
      not_run m-invariance-allthreads "no digest from the 1-thread run"
    fi
    suite
    # Run on every OS: the same kernel sources hipcc and NVRTC compile, on the host emulator (no GPU needed).
    refuse_skip suite-candle-core "skipping (accel|delta rule) host kernel tests"
    ;;
  metal)
    run probe-metal cargo run --release -p candle-core --features metal --example device_probe
    if device_ok metal suite-metal m-invariance-metal; then
      # Skipped here: the CPU-only cases run in the cpu job.
      suite --test-threads 1
      refuse_skip suite-candle-core "no GPU, skipping"
      run m-invariance-metal cargo test --release -p candle-core --features metal --test matmul_m_invariance_tests metal_ -- --ignored --nocapture
      require_measured m-invariance-metal "\\[METAL-M-INV\\] 측정 결과"
    fi
    ;;
  vulkan)
    run probe-vulkan cargo run --release -p candle-core --features vulkan --example device_probe
    if device_ok vulkan suite-vulkan m-invariance-vulkan; then
      suite --test-threads 1
      refuse_skip suite-candle-core "no GPU, skipping|skipping vulkan accel tests"
      refuse_skip suite-candle-nn "skipping: "
      run m-invariance-vulkan cargo test --release -p candle-core --features vulkan --test matmul_m_invariance_tests vulkan_ -- --ignored --nocapture
      require_measured m-invariance-vulkan "M-INV\\] 측정 결과"
    fi
    ;;
  *) echo "backend $BACKEND has no suite phase"; exit 2 ;;
  esac
  ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
finish
