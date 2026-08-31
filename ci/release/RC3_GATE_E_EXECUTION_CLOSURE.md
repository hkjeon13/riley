# RC3 Gate E execution-closure manifest v1

## Status and scope

`gate_e_execution_closure_contract_v1.py` is a **pure canonical-byte
declaration contract** for a future Gate E interpreter, dynamic loader, and
runtime dependency closure. It consumes caller-supplied bytes only. It has no
CLI, filesystem reader, `/opt` probe, subprocess, ELF parser, `ldd`, loader,
Python execution, service, root installation, lock, cgroup, GPU/Docker,
evidence, receipt, freeze, rollback, or qualification operation.

The companion schema is
`benchmarks/release/candidates/gate-e-execution-closure-manifest-v1.schema.json`.
It is deliberately a separate sidecar declaration, not an extension or
replacement for `gate-e-root-bundle-manifest-v1.schema.json`. The existing
root-bundle v1 grammar remains exactly bootstrap plus core; neither that C
authenticator nor its checked diagnostic is changed by this contract.

The separate C11 library in tools/native/gate-e-execution-closure-manifest-parser
now consumes the same fixed v1 byte grammar before a future native guardian
would cross the Python loader boundary. It returns only copied declaration
metadata and the raw manifest SHA-256 in a caller-owned fixed-size structure.
It has no CLI, filesystem, descriptor, ELF, loader, execution, or privilege
operation, and does not make the Python parser, schema, root-bundle
authenticator, or guardian lease model an execution authority.

The companion C11 library in tools/native/gate-e-execution-closure-held-fds
now composes that parser with an already-borrowed, role-ordered set of dynamic
loader, interpreter, and runtime FDs. It privately duplicates each descriptor,
requires a linked regular CLOEXEC read-only object of the exact declared
length, hashes it with pre/post full-identity checks, rejects numeric and
device/inode aliases, and retains only its duplicate plus the copied
declaration and raw sidecar SHA. The caller serializes input/output FD-table
ownership for each call. It never opens or authenticates the sidecar path or
any declared path; its retained result is a point-in-time binding precursor,
not an authenticated or complete execution closure.

## Exact bytes and closure identity

The only accepted raw bytes are:

```text
json.dumps(value, ensure_ascii=True, sort_keys=True,
           separators=(",", ":"), allow_nan=False) + "\n"
```

There must be exactly one terminal newline. Duplicate JSON keys, non-UTF-8,
non-finite numbers, noncanonical spacing/key order, unknown fields, and
unbounded shape drift fail closed. The returned `runtime_closure_sha256` is
SHA-256 of those exact raw manifest bytes **including** the terminal newline.
The manifest has no self-digest field.

The fixed v1 document shape is:

```json
{
  "dynamic_loader": {
    "audit_path": "/absolute/canonical/path",
    "byte_length": 1,
    "sha256": "64-lowercase-nonzero-hex"
  },
  "interpreter": {
    "audit_path": "/absolute/canonical/path",
    "byte_length": 1,
    "sha256": "64-lowercase-nonzero-hex"
  },
  "runtime_leaves": [
    {
      "audit_path": "/absolute/canonical/path",
      "byte_length": 1,
      "sha256": "64-lowercase-nonzero-hex"
    }
  ],
  "schema_version": "riley.rc3-gate-e-execution-closure-manifest.v1"
}
```

Each audit path is bounded ASCII, absolute, and canonical: no empty component,
dot component, traversal, trailing slash, space, or non-ASCII spelling is
accepted. Runtime leaves are nonempty, bytewise strictly sorted by their ASCII
path, unique, and cannot repeat the declared interpreter or loader path. v1
also bounds each leaf, number of leaves, and total declared closure bytes. It
is a declaration of bytes and labels only; paths are never reopened by this
module.

## Native parser parity precursor

The C11 parser accepts a raw byte pointer and length only. It recognizes the
fixed canonical member order directly rather than providing a permissive JSON
surface, rejects strings that would require escapes, and returns a bounded
copy of every accepted label, declared byte length, decoded SHA-256, and the
raw newline-inclusive closure SHA-256. It performs no allocation. On a failed
parse, a valid initialized output is cleared to prevent stale declaration
reuse.

The parser itself remains parser parity, not object acquisition. The held-FD
binder separately verifies only caller-borrowed objects against the parsed
declaration and retains no raw sidecar bytes. Neither component reads a
sidecar from a path or FD, verifies sidecar owner/ACL/filesystem policy,
discovers a dependency, parses ELF, establishes an interpreter/loader closure,
or arranges FD 31/32. A future guardian must bind the raw digest and
independently authenticated held objects to the session under a later review.

## Boundaries that remain unestablished

The manifest alone does **not** prove any declared path exists; that the file
type, owner, ACL, filesystem, namespace, device/inode, digest, length, ELF
type, `PT_INTERP`, `DT_NEEDED`, RPATH/RUNPATH, loader search, Python
standard-library tree, or runtime dependency list is complete; or that a path
is safe to open or execute. The held-FD binder adds only a point-in-time
linked-regular-FD length/digest/identity match for inputs supplied by its
caller; zero-link objects fail closed, but multiple links remain allowed. Its
caller must serialize FD-table ownership during each bind/recheck/close call.
It does not authenticate provenance, the sidecar, or any later mutation, and
does not make the multi-object result atomic. It cannot establish
same-object `execveat`, secure dynamic-loader behavior, interpreter/runtime
closure, a pre-Python trust root, or a launch authority.

The future guardian lease model already treats `runtime_closure_sha256` as an
opaque value bound through its synthetic session inputs. This new pure parser
does not modify that model or turn its synthetic value into a static manifest
guarantee. A later native-acquisition review must bind this sidecar's raw hash
to the guardian session, authenticate held interpreter/loader/runtime objects,
reject unsafe ELF resolution, and separately prove a same-object launch.

`tools/native/gate-e-sealed-leaf-snapshot/` remains only a `MFD_NOEXEC_SEAL`
data-copy primitive for Python bootstrap/core leaves. It must never be used for
the interpreter or loader execution object. FD 31/32 placement, old v3 FD
7/8/9/10 templates, `/opt` installation, and the root-bundle authenticator's
read-only descriptor lifecycle remain separate.

## Qualification denial and checks

This manifest is not a result or qualification input. Its terminal-newline raw
form is rejected by the RC3 qualification policy as noncanonical before schema
dispatch. A canonical reserialization without that newline receives the exact
`execution-closure-manifest-not-qualification` denial reason. No allow-list is
added.

Run the CPU-only checks without writing bytecode:

```bash
/usr/bin/python3.10 -I -S -E -B \
  ci/release/test_gate_e_execution_closure_contract_v1.py
/usr/bin/python3.10 -I -S -E -B \
  ci/release/test_rc3_qualification_input_policy_v2.py
```

The tests use only in-memory manifest fixtures plus a read of the checked-in
schema. They do not inspect the current host's interpreter/loader/runtime.
The native parser has its own source-only C11 checks in
tools/native/gate-e-execution-closure-manifest-parser: make test and
make analyze. Those checks compile only the parser and in-memory hostile
fixtures; they do not inspect a host interpreter, loader, or runtime leaf.
Neither suite creates `/opt`, requests sudo, or runs GPU/Docker work.
