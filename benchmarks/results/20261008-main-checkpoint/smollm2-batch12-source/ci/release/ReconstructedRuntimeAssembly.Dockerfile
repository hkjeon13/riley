# This recipe is a reviewed assembly tool, not an RC2 source artifact.  Its
# caller must provide a closed context containing only this recipe plus
# input/riley and input/riley.tar.gz.  The later capture receipt binds those
# raw context bytes and build arguments; labels alone are not provenance.

# NVIDIA-published linux/amd64 index digest for
# nvidia/cuda:12.8.1-runtime-ubuntu22.04, resolved 2026-08-26.
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd AS verify-input

ARG RILEY_RECONSTRUCTION_ID
ARG RILEY_SOURCE_REVISION
ARG RILEY_SOURCE_ARCHIVE_SHA256
ARG RILEY_REPRO_BUILD_INPUTS_SHA256
ARG RILEY_RELEASE_BINARY_SHA256
ARG RILEY_RELEASE_BUNDLE_SHA256
ARG RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256

COPY input/riley /assembly-input/riley
COPY input/riley.tar.gz /assembly-input/riley.tar.gz

# Keep extraction of caller-supplied archive bytes out of a root build shell.
# The final stage starts from a fresh base and receives only the verified tree.
RUN /usr/bin/chmod 0644 /assembly-input/riley /assembly-input/riley.tar.gz \
    && /usr/bin/mkdir -p /opt/riley \
    && /usr/bin/chown 65532:65532 /opt/riley

USER 65532:65532

# Verify both selected PR16 artifacts before the final stage can receive any
# runtime bytes.  The bundle's embedded binary must be exactly the separately
# captured reconstruction binary.
RUN (test "${RILEY_RECONSTRUCTION_ID}" = a || test "${RILEY_RECONSTRUCTION_ID}" = b) \
    && printf '%s\n' "${RILEY_SOURCE_REVISION}" | grep -Ex '[0-9a-f]{40}' >/dev/null \
    && printf '%s\n' "${RILEY_SOURCE_ARCHIVE_SHA256}" | grep -Ex '[0-9a-f]{64}' >/dev/null \
    && printf '%s\n' "${RILEY_REPRO_BUILD_INPUTS_SHA256}" | grep -Ex '[0-9a-f]{64}' >/dev/null \
    && printf '%s\n' "${RILEY_RELEASE_BINARY_SHA256}" | grep -Ex '[0-9a-f]{64}' >/dev/null \
    && printf '%s\n' "${RILEY_RELEASE_BUNDLE_SHA256}" | grep -Ex '[0-9a-f]{64}' >/dev/null \
    && printf '%s\n' "${RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256}" | grep -Ex '[0-9a-f]{64}' >/dev/null \
    && test "$(sha256sum /assembly-input/riley | cut -d ' ' -f 1)" = "${RILEY_RELEASE_BINARY_SHA256}" \
    && test "$(sha256sum /assembly-input/riley.tar.gz | cut -d ' ' -f 1)" = "${RILEY_RELEASE_BUNDLE_SHA256}" \
    && tar --extract --gzip --file /assembly-input/riley.tar.gz \
        --no-same-owner --no-same-permissions --no-overwrite-dir \
        --strip-components=1 --directory /opt/riley \
    && (cd /opt/riley && sha256sum --strict --check SHA256SUMS) \
    && test -z "$(find /opt/riley -xdev \( -type l -o -type b -o -type c -o -type p -o -type s \) -print -quit)" \
    && cmp --silent /assembly-input/riley /opt/riley/bin/riley \
    && test -x /opt/riley/bin/riley

# The final stage starts from the same immutable runtime base and copies only
# the verified, extracted bundle tree.  It never receives source or a build
# toolchain stage.
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd AS runtime

ARG RILEY_RECONSTRUCTION_ID
ARG RILEY_SOURCE_REVISION
ARG RILEY_SOURCE_ARCHIVE_SHA256
ARG RILEY_REPRO_BUILD_INPUTS_SHA256
ARG RILEY_RELEASE_BINARY_SHA256
ARG RILEY_RELEASE_BUNDLE_SHA256
ARG RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256

LABEL org.riley.reconstructed-runtime-assembly.version="v1" \
      org.riley.reconstructed-runtime-assembly.reconstruction-id="${RILEY_RECONSTRUCTION_ID}" \
      org.riley.reconstructed-runtime-assembly.source-revision="${RILEY_SOURCE_REVISION}" \
      org.riley.reconstructed-runtime-assembly.source-archive-sha256="${RILEY_SOURCE_ARCHIVE_SHA256}" \
      org.riley.reconstructed-runtime-assembly.repro-build-inputs-sha256="${RILEY_REPRO_BUILD_INPUTS_SHA256}" \
      org.riley.reconstructed-runtime-assembly.release-binary-sha256="${RILEY_RELEASE_BINARY_SHA256}" \
      org.riley.reconstructed-runtime-assembly.release-bundle-sha256="${RILEY_RELEASE_BUNDLE_SHA256}" \
      org.riley.reconstructed-runtime-assembly.recipe-normalized-instructions-sha256="${RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256}"

COPY --from=verify-input --chown=65532:65532 /opt/riley/ /opt/riley/

# The pinned NVIDIA base carries six debugger/apport Python source hooks even
# though it has no Python interpreter. Remove that exact reviewed inventory so
# the effective runtime filesystem remains Python-artifact-free.
RUN /usr/bin/rm \
        /usr/share/apport/package-hooks/source_shadow.py \
        /usr/share/gcc/python/libstdcxx/__init__.py \
        /usr/share/gcc/python/libstdcxx/v6/__init__.py \
        /usr/share/gcc/python/libstdcxx/v6/printers.py \
        /usr/share/gcc/python/libstdcxx/v6/xmethods.py \
        /usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30-gdb.py

# This check runs in the image build, before a future capture creates (but
# does not start) a container from it. It never executes the input ELF and is
# not service or GPU evidence.
RUN export PATH=/usr/bin:/bin \
    && test -x /opt/riley/bin/riley \
    && test -s /opt/riley/SHA256SUMS \
    && test -s /opt/riley/manifest/native-dependencies.txt \
    && test -s /opt/riley/manifest/release.json \
    && (cd /opt/riley && /usr/bin/sha256sum --strict --check SHA256SUMS) \
    && test -z "$(/usr/bin/find /opt/riley -xdev \( \
        -type l -o -type b -o -type c -o -type p -o -type s -o -perm /6000 \
        -o \( -type f -links +1 \) \) -print -quit)" \
    && for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do \
        if test -e "/opt/riley/bin/${command}" || command -v "${command}" >/dev/null 2>&1; then \
            echo "forbidden runtime executable: ${command}" >&2; exit 1; \
        fi; \
    done \
    && test ! -e /assembly-input \
    && test ! -e /workspace \
    && if /usr/bin/find / -xdev -type f \( \
        -name '*.py' -o -name '*.pyc' -o -name '*.whl' \
        -o -name '*.pkl' -o -name '*.pickle' \) | /usr/bin/grep -q .; then \
        echo 'forbidden Python artifact in runtime image' >&2; exit 1; \
    fi

ENV PATH=/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/opt/riley/bin/riley"]
CMD ["--help"]
