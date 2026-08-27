# BuildKit cannot portably resolve a raw local sha256 image ID in FROM. The
# launcher therefore creates a collision-closed local alias from the reviewed
# candidate revision and full image ID, verifies it before and after this
# build, and verifies the resulting RootFS prefix. There is no default base.
ARG RILEY_RELEASE_IMAGE_REF
FROM ${RILEY_RELEASE_IMAGE_REF} AS reliability-soak-test-layer

ARG RILEY_RELEASE_IMAGE_ID
ARG RILEY_SOURCE_REVISION
ARG RILEY_SOURCE_ARCHIVE_SHA256
ARG RILEY_RELEASE_BINARY_SHA256

LABEL org.riley.reliability-soak.release-image-id="${RILEY_RELEASE_IMAGE_ID}" \
      org.riley.reliability-soak.source-revision="${RILEY_SOURCE_REVISION}" \
      org.riley.reliability-soak.source-archive-sha256="${RILEY_SOURCE_ARCHIVE_SHA256}" \
      org.riley.reliability-soak.release-binary-sha256="${RILEY_RELEASE_BINARY_SHA256}"

USER 0:0
ENV DEBIAN_FRONTEND=noninteractive

# Capture the actual loader resolution before package installation, then make
# the install fail closed unless every resolution state, resolved path,
# canonical regular-file target, and target byte digest remains identical.
# Build-time unresolved entries (for example a runtime-injected libcuda.so.1)
# are retained as NAME/NOT_FOUND/-/- rows. Address tokens emitted by ldd are
# deliberately ignored. The retained TSV is an auditable build receipt.
RUN set -eu; \
    release_binary_path=/opt/riley/bin/riley; \
    closure_before=/opt/riley-soak/release-runtime-closure.tsv; \
    closure_after=/tmp/release-runtime-closure.after.tsv; \
    capture_runtime_closure() { \
        closure_output=$1; \
        closure_raw="${closure_output}.ldd"; \
        closure_unsorted="${closure_output}.unsorted"; \
        unresolved_count=0; \
        LC_ALL=C ldd "$release_binary_path" >"$closure_raw"; \
        : >"$closure_unsorted"; \
        while IFS=' ' read -r dependency_name dependency_relation dependency_resolution _; do \
            test -n "$dependency_name" || continue; \
            test "$dependency_name" != linux-vdso.so.1 || continue; \
            if test "$dependency_relation" = '=>'; then \
                if test "$dependency_resolution" = not; then \
                    test "$dependency_name" = libcuda.so.1 || { \
                        echo "unreviewed unresolved release dependency: $dependency_name" >&2; \
                        exit 1; \
                    }; \
                    unresolved_count=$((unresolved_count + 1)); \
                    printf '%s\tNOT_FOUND\t-\t-\n' \
                        "$dependency_name" >>"$closure_unsorted"; \
                    continue; \
                fi; \
                dependency_path=$dependency_resolution; \
            else \
                dependency_path=$dependency_name; \
            fi; \
            case "$dependency_path" in \
                /*) ;; \
                *) echo "non-absolute release runtime dependency: $dependency_name" >&2; exit 1 ;; \
            esac; \
            dependency_target=$(readlink -f -- "$dependency_path"); \
            case "$dependency_target" in \
                /*) ;; \
                *) echo "non-absolute release runtime target: $dependency_path" >&2; exit 1 ;; \
            esac; \
            test -f "$dependency_target" && test ! -L "$dependency_target"; \
            dependency_sha256=$(sha256sum "$dependency_target"); \
            dependency_sha256=${dependency_sha256%% *}; \
            test "${#dependency_sha256}" -eq 64; \
            case "$dependency_sha256" in *[!0-9a-f]*) exit 1 ;; esac; \
            printf '%s\t%s\t%s\t%s\n' \
                "$dependency_name" "$dependency_path" "$dependency_target" \
                "$dependency_sha256" >>"$closure_unsorted"; \
        done <"$closure_raw"; \
        test "$unresolved_count" -eq 1; \
        test -s "$closure_unsorted"; \
        LC_ALL=C sort -u "$closure_unsorted" >"$closure_output"; \
        rm -f "$closure_raw" "$closure_unsorted"; \
    }; \
    for closure_command in ldd readlink sha256sum sort cmp; do \
        command -v "$closure_command" >/dev/null 2>&1 || { \
            echo "missing release runtime closure utility: $closure_command" >&2; \
            exit 1; \
        }; \
    done; \
    mkdir -p /opt/riley-soak; \
    capture_runtime_closure "$closure_before"; \
    apt-get update; \
    apt-get install -y --no-install-recommends --no-upgrade \
        bash \
        coreutils \
        curl \
        findutils \
        gawk \
        grep \
        jq \
        procps \
        util-linux; \
    capture_runtime_closure "$closure_after"; \
    cmp --silent "$closure_before" "$closure_after" || { \
        echo 'release runtime dependency closure changed during observation-tool installation' >&2; \
        exit 1; \
    }; \
    chmod 0444 "$closure_before"; \
    rm -f "$closure_after"; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/riley-soak/ci \
        /opt/riley-soak/benchmarks/soak \
        /evidence \
        /model \
        /run-input
COPY ci/run_release_soak.sh /opt/riley-soak/ci/run_release_soak.sh
COPY benchmarks/soak/reliability-soak-v1.json /opt/riley-soak/benchmarks/soak/reliability-soak-v1.json
RUN chmod 0555 /opt/riley-soak/ci/run_release_soak.sh \
    && chmod 0444 /opt/riley-soak/benchmarks/soak/reliability-soak-v1.json

ENV LC_ALL=C
ENV TZ=UTC

# The derivative may add observation tools, but it must preserve the release
# binary byte-for-byte and remain Python-free. The materialized manifest and
# model are supplied later as read-only mounts.
RUN test -x /opt/riley/bin/riley \
    && test "$(sha256sum /opt/riley/bin/riley | awk '{print $1}')" = "${RILEY_RELEASE_BINARY_SHA256}" \
    && for command_name in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do \
        if command -v "${command_name}" >/dev/null 2>&1; then \
            echo "forbidden reliability soak executable: ${command_name}" >&2; \
            exit 1; \
        fi; \
    done \
    && for required_command in bash jq curl sha256sum awk ps flock readlink find sort grep wc date env; do \
        command -v "${required_command}" >/dev/null 2>&1 || { \
            echo "missing reliability soak utility: ${required_command}" >&2; \
            exit 1; \
        }; \
    done \
    && if find / -xdev -type f \( \
        -name '*.py' -o -name '*.pyc' -o -name '*.whl' \
        -o -name '*.pkl' -o -name '*.pickle' \) | grep -q .; then \
        echo 'forbidden Python artifact in reliability soak test layer' >&2; \
        exit 1; \
    fi

USER 65532:65532
ENTRYPOINT ["/opt/riley-soak/ci/run_release_soak.sh"]
CMD []
