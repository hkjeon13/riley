#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
host_profile_path="${script_dir}/host_profiles/rtx4090-ubuntu22-driver580-host-v3.env"
requested_environment_id="${RILEY_PREFLIGHT_ENVIRONMENT_ID:-rtx4090-ubuntu22-driver580-v1}"

# Host policy is checked in and recorded below.  Do not permit an ambient
# threshold or identity override to make a run appear comparable.
for forbidden_override in \
  RILEY_EXPECTED_GPU RILEY_EXPECTED_COMPUTE_CAP \
  RILEY_MAX_IDLE_MEMORY_MIB RILEY_MAX_START_TEMPERATURE_C; do
  if declare -p "${forbidden_override}" >/dev/null 2>&1; then
    echo "preflight: ${forbidden_override} is not supported; use a checked-in host profile revision" >&2
    exit 2
  fi
done

if [[ ! -f "${host_profile_path}" || -L "${host_profile_path}" || ! -r "${host_profile_path}" ]]; then
  echo "preflight: checked-in host profile is not a readable regular file" >&2
  exit 2
fi

profile_keys=(
  schema_version profile_id profile_version environment_ids
  gpu_name compute_capability memory_total_mib driver_version_prefix
  persistence_mode os_id os_version_id kernel_release machine
  cpu_model cpu_model_substring physical_cpu_cores logical_cpu_threads
  cpu_governor cpu_governor_policy_count ram_bytes_target
  ram_bytes_tolerance_bytes max_idle_memory_mib max_start_temperature_c
)
profile_values=()
profile_key_index=0
while IFS= read -r profile_line || [[ -n "${profile_line}" ]]; do
  [[ -z "${profile_line}" || "${profile_line}" == \#* ]] && continue
  if [[ "${profile_line}" != *=* ]]; then
    echo "preflight: host profile has a malformed line" >&2
    exit 2
  fi
  profile_key="${profile_line%%=*}"
  profile_value="${profile_line#*=}"
  if (( profile_key_index >= ${#profile_keys[@]} )) || [[ "${profile_key}" != "${profile_keys[${profile_key_index}]}" ]]; then
    echo "preflight: host profile keys must be complete and in reviewed order" >&2
    exit 2
  fi
  profile_values[${profile_key_index}]="${profile_value}"
  profile_key_index=$((profile_key_index + 1))
done < "${host_profile_path}"
if (( profile_key_index != ${#profile_keys[@]} )); then
  echo "preflight: host profile is incomplete" >&2
  exit 2
fi
for profile_value in "${profile_values[@]}"; do
  if [[ -z "${profile_value}" ]]; then
    echo "preflight: host profile has an empty value" >&2
    exit 2
  fi
done
profile_schema_version="${profile_values[0]}"
profile_id="${profile_values[1]}"
profile_version="${profile_values[2]}"
profile_environment_ids="${profile_values[3]}"
expected_gpu="${profile_values[4]}"
expected_compute_cap="${profile_values[5]}"
expected_memory_total_mib="${profile_values[6]}"
driver_version_prefix="${profile_values[7]}"
expected_persistence_mode="${profile_values[8]}"
expected_os_id="${profile_values[9]}"
expected_os_version_id="${profile_values[10]}"
expected_kernel_release="${profile_values[11]}"
expected_machine="${profile_values[12]}"
expected_cpu_model="${profile_values[13]}"
expected_cpu_model_substring="${profile_values[14]}"
expected_physical_cpu_cores="${profile_values[15]}"
expected_logical_cpu_threads="${profile_values[16]}"
expected_cpu_governor="${profile_values[17]}"
expected_cpu_governor_policy_count="${profile_values[18]}"
ram_bytes_target="${profile_values[19]}"
ram_bytes_tolerance_bytes="${profile_values[20]}"
max_idle_memory_mib="${profile_values[21]}"
max_start_temperature_c="${profile_values[22]}"
if [[ "${profile_schema_version}" != "riley.host-profile.v1" || "${profile_version}" != "3" ]]; then
  echo "preflight: unsupported checked-in host profile version" >&2
  exit 2
fi
numeric_profile_keys=(
  memory_total_mib physical_cpu_cores logical_cpu_threads cpu_governor_policy_count
  ram_bytes_target ram_bytes_tolerance_bytes max_idle_memory_mib max_start_temperature_c
)
numeric_profile_values=(
  "${expected_memory_total_mib}" "${expected_physical_cpu_cores}" "${expected_logical_cpu_threads}" "${expected_cpu_governor_policy_count}"
  "${ram_bytes_target}" "${ram_bytes_tolerance_bytes}" "${max_idle_memory_mib}" "${max_start_temperature_c}"
)
for numeric_profile_index in "${!numeric_profile_keys[@]}"; do
  numeric_profile_key="${numeric_profile_keys[${numeric_profile_index}]}"
  numeric_profile_value="${numeric_profile_values[${numeric_profile_index}]}"
  if [[ ! "${numeric_profile_value}" =~ ^[0-9]+$ ]]; then
    echo "preflight: host profile ${numeric_profile_key} must be a nonnegative integer" >&2
    exit 2
  fi
done
if [[ ! "${driver_version_prefix}" =~ ^[0-9]+\.$ ]]; then
  echo "preflight: host profile driver_version_prefix must name a driver branch" >&2
  exit 2
fi
environment_supported=false
IFS=',' read -r -a supported_environment_ids <<< "${profile_environment_ids}"
for supported_environment_id in "${supported_environment_ids[@]}"; do
  if [[ "${requested_environment_id}" == "${supported_environment_id}" ]]; then
    environment_supported=true
    break
  fi
done
if [[ "${environment_supported}" != true ]]; then
  echo "preflight: unsupported environment ID" >&2
  exit 2
fi
minimum_staging_available_bytes=21474836480
cpu_governor_root="${RILEY_CPU_GOVERNOR_ROOT:-/sys/devices/system/cpu/cpufreq}"
host_root="${RILEY_HOST_ROOT:-}"
staging_output_root="${RILEY_PREFLIGHT_OUTPUT_ROOT:-}"
os_release_path="${host_root}/etc/os-release"
cpuinfo_path="${host_root}/proc/cpuinfo"
meminfo_path="${host_root}/proc/meminfo"

if [[ ! -x /usr/bin/nvidia-smi || -L /usr/bin/nvidia-smi ]]; then
  echo "preflight: nvidia-smi is required" >&2
  exit 2
fi

gpu_count="$(/usr/bin/nvidia-smi --list-gpus | /usr/bin/wc -l | /usr/bin/tr -d '[:space:]')"
if [[ "${gpu_count}" != "1" ]]; then
  echo "preflight: expected exactly one visible GPU, found ${gpu_count:-unknown}" >&2
  exit 2
fi

gpu_row="$(/usr/bin/nvidia-smi \
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
os_id="$(/usr/bin/sed -n 's/^ID=//p' "${os_release_path}" | /usr/bin/tr -d '"' | /usr/bin/head -n 1)"
os_version_id="$(/usr/bin/sed -n 's/^VERSION_ID=//p' "${os_release_path}" | /usr/bin/tr -d '"' | /usr/bin/head -n 1)"
kernel_release="$(/usr/bin/uname -r)"
machine="$(/usr/bin/uname -m)"
cpu_model_raw="$(/usr/bin/mawk -F: '/^model name[[:space:]]*:/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' "${cpuinfo_path}")"
if [[ "${cpu_model_raw}" != *"${expected_cpu_model_substring}"* ]]; then
  echo "preflight: expected CPU containing ${expected_cpu_model_substring}, found ${cpu_model_raw:-unknown}" >&2
  exit 2
fi
cpu_model="${expected_cpu_model}"
physical_cpu_cores="$(/usr/bin/mawk -F: '
  /^physical id[[:space:]]*:/ { gsub(/[[:space:]]/, "", $2); package=$2 }
  /^core id[[:space:]]*:/ { gsub(/[[:space:]]/, "", $2); seen[package ":" $2]=1 }
  END { for (key in seen) count++; print count+0 }
' "${cpuinfo_path}")"
logical_cpu_threads="$(/usr/bin/mawk -F: '/^processor[[:space:]]*:/ { count++ } END { print count+0 }' "${cpuinfo_path}")"
mem_total_kib="$(/usr/bin/mawk '$1 == "MemTotal:" { print $2; exit }' "${meminfo_path}")"
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
ram_bytes_delta="$((ram_bytes - ram_bytes_target))"
if (( ram_bytes_delta < 0 )); then
  ram_bytes_delta="$((-ram_bytes_delta))"
fi
if (( ram_bytes_delta > ram_bytes_tolerance_bytes )); then
  echo "preflight: observed RAM ${ram_bytes} bytes differs from profile target ${ram_bytes_target} by more than ${ram_bytes_tolerance_bytes} bytes" >&2
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

if [[ ! "${memory_total_mib}" =~ ^[0-9]+$ || ! "${memory_used_mib}" =~ ^[0-9]+$ || ! "${temperature_c}" =~ ^[0-9]+$ ]]; then
  echo "preflight: nvidia-smi returned a nonnumeric memory or temperature value" >&2
  exit 2
fi

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

if [[ "${driver_version}" != "${driver_version_prefix}"* ]]; then
  echo "preflight: expected NVIDIA driver branch ${driver_version_prefix}*, found ${driver_version}" >&2
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

if ! compute_processes="$(/usr/bin/nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "preflight: failed to query active CUDA compute processes" >&2
  exit 2
fi
if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
  echo "preflight: another CUDA compute process is active" >&2
  printf '%s\n' "${compute_processes}" >&2
  exit 2
fi

if /usr/bin/git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_revision="$(/usr/bin/git rev-parse HEAD)"
  if [[ -n "$(/usr/bin/git status --porcelain=v1 --untracked-files=normal -- . ':(exclude)benchmarks/results')" ]]; then
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

if [[ ! -x /usr/bin/timedatectl || -L /usr/bin/timedatectl ]]; then
  echo "preflight: timedatectl is required to verify clock synchronization" >&2
  exit 2
fi
if ! clock_synchronized="$(/usr/bin/timedatectl show -p NTPSynchronized --value 2>/dev/null)"; then
  echo "preflight: timedatectl could not determine clock synchronization" >&2
  exit 2
fi
if [[ "${clock_synchronized}" != "yes" ]]; then
  echo "preflight: system clock is not NTP synchronized (${clock_synchronized:-unknown})" >&2
  exit 2
fi

if [[ -z "${staging_output_root}" || ! -d "${staging_output_root}" ]]; then
  echo "preflight: RILEY_PREFLIGHT_OUTPUT_ROOT must name the existing staging directory" >&2
  exit 2
fi
staging_available_kib="$(/usr/bin/df -Pk -- "${staging_output_root}" | /usr/bin/mawk 'NR == 2 { print $4 }')"
if [[ ! "${staging_available_kib}" =~ ^[0-9]+$ ]]; then
  echo "preflight: could not determine staging filesystem available bytes" >&2
  exit 2
fi
staging_available_bytes="$((staging_available_kib * 1024))"
if (( staging_available_bytes < minimum_staging_available_bytes )); then
  echo "preflight: staging filesystem has ${staging_available_bytes} bytes available; ${minimum_staging_available_bytes} required" >&2
  exit 2
fi

printf 'environment_id=%s\n' "${requested_environment_id}"
printf 'host_profile_schema_version=%s\n' "${profile_schema_version}"
printf 'host_profile_id=%s\n' "${profile_id}"
printf 'host_profile_version=%s\n' "${profile_version}"
printf 'os_id=%s\n' "${os_id}"
printf 'os_version_id=%s\n' "${os_version_id}"
printf 'kernel_release=%s\n' "${kernel_release}"
printf 'machine=%s\n' "${machine}"
printf 'cpu_model=%s\n' "${cpu_model}"
printf 'cpu_model_observed=%s\n' "${cpu_model_raw}"
printf 'physical_cpu_cores=%s\n' "${physical_cpu_cores}"
printf 'logical_cpu_threads=%s\n' "${logical_cpu_threads}"
printf 'ram_bytes=%s\n' "${ram_bytes}"
printf 'ram_bytes_target=%s\n' "${ram_bytes_target}"
printf 'ram_bytes_tolerance_bytes=%s\n' "${ram_bytes_tolerance_bytes}"
printf 'git_revision=%s\n' "${git_revision}"
printf 'gpu_name=%s\n' "${gpu_name}"
printf 'compute_capability=%s\n' "${compute_cap}"
printf 'memory_total_mib=%s\n' "${memory_total_mib}"
printf 'memory_used_mib=%s\n' "${memory_used_mib}"
printf 'driver_version=%s\n' "${driver_version}"
printf 'driver_version_prefix=%s\n' "${driver_version_prefix}"
printf 'persistence_mode=%s\n' "${persistence_mode}"
printf 'idle_memory_limit_mib=%s\n' "${max_idle_memory_mib}"
printf 'temperature_c=%s\n' "${temperature_c}"
printf 'start_temperature_limit_c=%s\n' "${max_start_temperature_c}"
printf 'power_limit_w=%s\n' "${power_limit_w}"
printf 'graphics_clock_mhz=%s\n' "${graphics_clock_mhz}"
printf 'memory_clock_mhz=%s\n' "${memory_clock_mhz}"
printf 'cpu_governor=%s\n' "${cpu_governor}"
printf 'cpu_governor_policy_count=%s\n' "${cpu_governor_policy_count}"
printf 'clock_synchronized=%s\n' "${clock_synchronized}"
printf 'staging_available_bytes=%s\n' "${staging_available_bytes}"
printf 'staging_minimum_bytes=%s\n' "${minimum_staging_available_bytes}"
