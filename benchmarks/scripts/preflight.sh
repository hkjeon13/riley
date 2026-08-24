#!/usr/bin/env bash

set -euo pipefail

expected_gpu="${RUSTINFER_EXPECTED_GPU:-NVIDIA GeForce RTX 4090}"
expected_compute_cap="${RUSTINFER_EXPECTED_COMPUTE_CAP:-8.9}"
max_idle_memory_mib="${RUSTINFER_MAX_IDLE_MEMORY_MIB:-256}"
max_start_temperature_c="${RUSTINFER_MAX_START_TEMPERATURE_C:-50}"
expected_driver_version='580.173.02'
expected_persistence_mode='Disabled'
expected_cpu_governor='powersave'
expected_cpu_governor_policy_count='24'
expected_memory_total_mib='24564'
expected_environment_id='rtx4090-ubuntu22-driver580-v1'
expected_os_id='ubuntu'
expected_os_version_id='22.04'
expected_kernel_release='6.8.0-138-generic'
expected_machine='x86_64'
expected_cpu_model='Intel Core i7-13700K'
expected_physical_cpu_cores='16'
expected_logical_cpu_threads='24'
expected_mem_total_kib='65610936'
expected_ram_bytes='67185598464'
minimum_staging_available_bytes=21474836480
cpu_governor_root="${RUSTINFER_CPU_GOVERNOR_ROOT:-/sys/devices/system/cpu/cpufreq}"
host_root="${RUSTINFER_HOST_ROOT:-}"
staging_output_root="${RUSTINFER_PREFLIGHT_OUTPUT_ROOT:-}"
os_release_path="${host_root}/etc/os-release"
cpuinfo_path="${host_root}/proc/cpuinfo"
meminfo_path="${host_root}/proc/meminfo"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "preflight: nvidia-smi is required" >&2
  exit 2
fi

gpu_count="$(nvidia-smi --list-gpus | wc -l | tr -d '[:space:]')"
if [[ "${gpu_count}" != "1" ]]; then
  echo "preflight: expected exactly one visible GPU, found ${gpu_count:-unknown}" >&2
  exit 2
fi

gpu_row="$(nvidia-smi \
  --query-gpu=name,compute_cap,memory.total,memory.used,driver_version,persistence_mode,temperature.gpu,power.limit,clocks.applications.graphics,clocks.applications.memory \
  --format=csv,noheader,nounits)"

IFS=',' read -r gpu_name compute_cap memory_total_mib memory_used_mib driver_version persistence_mode temperature_c power_limit_w graphics_clock_mhz memory_clock_mhz <<<"${gpu_row}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

if [[ ! -r "${os_release_path}" || ! -r "${cpuinfo_path}" || ! -r "${meminfo_path}" ]]; then
  echo "preflight: canonical host identity files are not readable" >&2
  exit 2
fi
os_id="$(sed -n 's/^ID=//p' "${os_release_path}" | tr -d '"' | head -n 1)"
os_version_id="$(sed -n 's/^VERSION_ID=//p' "${os_release_path}" | tr -d '"' | head -n 1)"
kernel_release="$(uname -r)"
machine="$(uname -m)"
cpu_model_raw="$(awk -F: '/^model name[[:space:]]*:/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' "${cpuinfo_path}")"
if [[ "${cpu_model_raw}" != *"i7-13700K"* ]]; then
  echo "preflight: expected CPU containing i7-13700K, found ${cpu_model_raw:-unknown}" >&2
  exit 2
fi
cpu_model="${expected_cpu_model}"
physical_cpu_cores="$(awk -F: '
  /^physical id[[:space:]]*:/ { gsub(/[[:space:]]/, "", $2); package=$2 }
  /^core id[[:space:]]*:/ { gsub(/[[:space:]]/, "", $2); seen[package ":" $2]=1 }
  END { for (key in seen) count++; print count+0 }
' "${cpuinfo_path}")"
logical_cpu_threads="$(awk -F: '/^processor[[:space:]]*:/ { count++ } END { print count+0 }' "${cpuinfo_path}")"
mem_total_kib="$(awk '$1 == "MemTotal:" { print $2; exit }' "${meminfo_path}")"
if [[ ! "${mem_total_kib}" =~ ^[0-9]+$ ]]; then
  echo "preflight: cannot parse MemTotal from /proc/meminfo" >&2
  exit 2
fi
ram_bytes="$((mem_total_kib * 1024))"

if [[ "${os_id}" != "${expected_os_id}" || "${os_version_id}" != "${expected_os_version_id}" ]]; then
  echo "preflight: expected Ubuntu ${expected_os_version_id}, found ${os_id:-unknown} ${os_version_id:-unknown}" >&2
  exit 2
fi
if [[ "${kernel_release}" != "${expected_kernel_release}" || "${machine}" != "${expected_machine}" ]]; then
  echo "preflight: expected kernel/machine ${expected_kernel_release}/${expected_machine}, found ${kernel_release}/${machine}" >&2
  exit 2
fi
if [[ "${physical_cpu_cores}" != "${expected_physical_cpu_cores}" || "${logical_cpu_threads}" != "${expected_logical_cpu_threads}" ]]; then
  echo "preflight: expected CPU topology ${expected_physical_cpu_cores} cores/${expected_logical_cpu_threads} threads, found ${physical_cpu_cores}/${logical_cpu_threads}" >&2
  exit 2
fi
if [[ "${mem_total_kib}" != "${expected_mem_total_kib}" || "${ram_bytes}" != "${expected_ram_bytes}" ]]; then
  echo "preflight: expected RAM ${expected_ram_bytes} bytes, found ${ram_bytes}" >&2
  exit 2
fi

gpu_name="$(trim "${gpu_name}")"
compute_cap="$(trim "${compute_cap}")"
memory_total_mib="$(trim "${memory_total_mib}")"
memory_used_mib="$(trim "${memory_used_mib}")"
driver_version="$(trim "${driver_version}")"
persistence_mode="$(trim "${persistence_mode}")"
temperature_c="$(trim "${temperature_c}")"
power_limit_w="$(trim "${power_limit_w}")"
graphics_clock_mhz="$(trim "${graphics_clock_mhz}")"
memory_clock_mhz="$(trim "${memory_clock_mhz}")"

if [[ "${gpu_name}" != "${expected_gpu}" ]]; then
  echo "preflight: expected GPU '${expected_gpu}', found '${gpu_name}'" >&2
  exit 2
fi

if [[ "${compute_cap}" != "${expected_compute_cap}" ]]; then
  echo "preflight: expected compute capability ${expected_compute_cap}, found ${compute_cap}" >&2
  exit 2
fi

if [[ "${memory_total_mib}" != "${expected_memory_total_mib}" ]]; then
  echo "preflight: expected GPU memory ${expected_memory_total_mib} MiB, found ${memory_total_mib}" >&2
  exit 2
fi

if [[ "${driver_version}" != "${expected_driver_version}" ]]; then
  echo "preflight: expected NVIDIA driver ${expected_driver_version}, found ${driver_version}" >&2
  exit 2
fi

if [[ "${persistence_mode}" != "${expected_persistence_mode}" ]]; then
  echo "preflight: expected persistence mode ${expected_persistence_mode}, found ${persistence_mode}" >&2
  exit 2
fi

if (( memory_used_mib > max_idle_memory_mib )); then
  echo "preflight: idle GPU memory ${memory_used_mib} MiB exceeds ${max_idle_memory_mib} MiB" >&2
  exit 2
fi

if (( temperature_c > max_start_temperature_c )); then
  echo "preflight: start temperature ${temperature_c} C exceeds ${max_start_temperature_c} C" >&2
  exit 2
fi

if ! compute_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "preflight: failed to query active CUDA compute processes" >&2
  exit 2
fi
if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
  echo "preflight: another CUDA compute process is active" >&2
  printf '%s\n' "${compute_processes}" >&2
  exit 2
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_revision="$(git rev-parse HEAD)"
  if [[ -n "$(git status --porcelain=v1 --untracked-files=normal -- . ':(exclude)benchmarks/results')" ]]; then
    echo "preflight: benchmark requires a clean Git revision" >&2
    exit 2
  fi
else
  git_revision="not-a-git-worktree"
fi

governor_paths=("${cpu_governor_root}"/policy*/scaling_governor)
if [[ ! -e "${governor_paths[0]}" ]]; then
  echo "preflight: no CPU frequency policy governor files under ${cpu_governor_root}" >&2
  exit 2
fi
for governor_path in "${governor_paths[@]}"; do
  if ! cpu_governor="$(<"${governor_path}")"; then
    echo "preflight: cannot read CPU governor ${governor_path}" >&2
    exit 2
  fi
  if [[ "${cpu_governor}" != "${expected_cpu_governor}" ]]; then
    echo "preflight: expected CPU governor ${expected_cpu_governor}, found ${cpu_governor} at ${governor_path}" >&2
    exit 2
  fi
done
cpu_governor_policy_count="${#governor_paths[@]}"
cpu_governor="${expected_cpu_governor}"
if [[ "${cpu_governor_policy_count}" != "${expected_cpu_governor_policy_count}" ]]; then
  echo "preflight: expected ${expected_cpu_governor_policy_count} CPU governor policies, found ${cpu_governor_policy_count}" >&2
  exit 2
fi

if ! command -v timedatectl >/dev/null 2>&1; then
  echo "preflight: timedatectl is required to verify clock synchronization" >&2
  exit 2
fi
if ! clock_synchronized="$(timedatectl show -p NTPSynchronized --value 2>/dev/null)"; then
  echo "preflight: timedatectl could not determine clock synchronization" >&2
  exit 2
fi
if [[ "${clock_synchronized}" != "yes" ]]; then
  echo "preflight: system clock is not NTP synchronized (${clock_synchronized:-unknown})" >&2
  exit 2
fi

if [[ -z "${staging_output_root}" || ! -d "${staging_output_root}" ]]; then
  echo "preflight: RUSTINFER_PREFLIGHT_OUTPUT_ROOT must name the existing staging directory" >&2
  exit 2
fi
staging_available_kib="$(df -Pk -- "${staging_output_root}" | awk 'NR == 2 { print $4 }')"
if [[ ! "${staging_available_kib}" =~ ^[0-9]+$ ]]; then
  echo "preflight: could not determine staging filesystem available bytes" >&2
  exit 2
fi
staging_available_bytes="$((staging_available_kib * 1024))"
if (( staging_available_bytes < minimum_staging_available_bytes )); then
  echo "preflight: staging filesystem has ${staging_available_bytes} bytes available; ${minimum_staging_available_bytes} required" >&2
  exit 2
fi

printf 'environment_id=%s\n' "${expected_environment_id}"
printf 'os_id=%s\n' "${os_id}"
printf 'os_version_id=%s\n' "${os_version_id}"
printf 'kernel_release=%s\n' "${kernel_release}"
printf 'machine=%s\n' "${machine}"
printf 'cpu_model=%s\n' "${cpu_model}"
printf 'physical_cpu_cores=%s\n' "${physical_cpu_cores}"
printf 'logical_cpu_threads=%s\n' "${logical_cpu_threads}"
printf 'ram_bytes=%s\n' "${ram_bytes}"
printf 'git_revision=%s\n' "${git_revision}"
printf 'gpu_name=%s\n' "${gpu_name}"
printf 'compute_capability=%s\n' "${compute_cap}"
printf 'memory_total_mib=%s\n' "${memory_total_mib}"
printf 'memory_used_mib=%s\n' "${memory_used_mib}"
printf 'driver_version=%s\n' "${driver_version}"
printf 'persistence_mode=%s\n' "${persistence_mode}"
printf 'temperature_c=%s\n' "${temperature_c}"
printf 'power_limit_w=%s\n' "${power_limit_w}"
printf 'graphics_clock_mhz=%s\n' "${graphics_clock_mhz}"
printf 'memory_clock_mhz=%s\n' "${memory_clock_mhz}"
printf 'cpu_governor=%s\n' "${cpu_governor}"
printf 'cpu_governor_policy_count=%s\n' "${cpu_governor_policy_count}"
printf 'clock_synchronized=%s\n' "${clock_synchronized}"
printf 'staging_available_bytes=%s\n' "${staging_available_bytes}"
printf 'staging_minimum_bytes=%s\n' "${minimum_staging_available_bytes}"
