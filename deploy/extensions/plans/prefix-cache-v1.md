# Prefix-cache v1 admission plan

## Admission scope

prefix-cache-v1 is admitted as an experimental, default-off reference extension
for the tenant-safe exact prefix-cache track. This admission adds no runtime
implementation, production runtime-flag source literal, CUDA execution,
benchmark result, or performance claim.

## Frozen boundaries

- Reuse is permitted only for an exact model execution identity and an allowed
  sharing domain.
- The default sharing domain is tenant-private; only the server or operator may
  assign a shared domain, and a request payload cannot select an arbitrary one.
- Raw tokens, prompt text, and tenant IDs must not appear as log, metric,
  debug-output, or debug-dump values.
- Only complete immutable KV blocks may be published or attached.
- Any identity, ownership, lifetime, or validation failure takes the existing
  exact prefill fallback.
- The future implementation must preserve behavioral and generated-token parity,
  stable fallback, and lifetime/resource regression coverage.

## Later implementation gate

A separate implementation change must link this admission through the extension
manifest, keep RILEY_EXPERIMENTAL_PREFIX_CACHE_V1 default-off, add direct
integration coverage, and run the frozen cache-off/cache-on end-to-end
comparison. Its contract fixes SmolLM2-135M bf16 on one RTX 4090 for
concurrency 1/8/32, prompt lengths 128/1024/4096, a 16-token fixed greedy
output, cold and warm states, five independent processes, five measured
iterations per process, and median plus p95 disclosure.

## Rollback

Operational rollback is disabling the experimental flag. Code rollback must
remove the later cache metadata and scheduler integration together and retain
the existing exact paged-KV prefill path.
