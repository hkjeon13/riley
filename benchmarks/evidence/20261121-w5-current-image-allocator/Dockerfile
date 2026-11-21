ARG RILEY_PARENT
FROM ${RILEY_PARENT}
ARG RILEY_PARENT_ID
ARG JEMALLOC_PACKAGE_SHA256
LABEL io.riley.test.parent-image="${RILEY_PARENT_ID}" \
      io.riley.test.jemalloc-package-sha256="${JEMALLOC_PACKAGE_SHA256}"
COPY riley-with-allocator /opt/riley/bin/riley-with-allocator
COPY libjemalloc.so.2 /opt/riley/lib/libjemalloc.so.2
COPY jemalloc-copyright /opt/riley/licenses/jemalloc/copyright
ENTRYPOINT ["/opt/riley/bin/riley-with-allocator"]
CMD ["help"]
