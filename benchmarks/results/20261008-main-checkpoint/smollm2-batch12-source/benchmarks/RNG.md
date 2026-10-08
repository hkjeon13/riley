# Deterministic RNG contract

`riley.philox4x32-10.v1` is the portable random-number contract used by
fixtures, the native runtime, and any Python reference tooling.  A fixed seed
is reproducible only when the algorithm ID, stream ID, domain, and draw order
are also fixed.  Reproducibility under this contract does **not** claim that a
different engine or a distribution-preserving algorithm consumes the same
number of random words.

All integers below are unsigned.  Integer byte encodings are little-endian.
Arithmetic on 32-bit words wraps modulo `2^32`; multiplication first produces
the full 64-bit product.  Text supplied by a public API is encoded as strict
UTF-8 before applying this contract.  An implementation must reject invalid
lengths and integers rather than truncate them.

## Philox4x32-10 core

The counter is four `u32` words `(c0, c1, c2, c3)` and the key is two `u32`
words `(k0, k1)`.  Version 1 fixes these Random123 constants:

```text
M0 = 0xd2511f53    M1 = 0xcd9e8d57
W0 = 0x9e3779b9    W1 = 0xbb67ae85
rounds = 10
```

One round computes the following, where `hi32` and `lo32` select halves of a
64-bit product:

```text
(hi0, lo0) = mulhilo32(M0, c0)
(hi1, lo1) = mulhilo32(M1, c2)

(c0, c1, c2, c3) = (
    hi1 XOR c1 XOR k0,
    lo1,
    hi0 XOR c3 XOR k1,
    lo0,
)
(k0, k1) = (k0 + W0, k1 + W1) modulo 2^32
```

Apply exactly ten rounds.  The key bump after the tenth round has no effect on
that block, but performing it is permitted.  Words are consumed from the
result in `(c0, c1, c2, c3)` order.

## Seed, stream, and domain derivation

Version 1 accepts a master seed in `0..2^64-1` and arbitrary byte strings for
the stream and domain.  Length-prefixed bytes are encoded as
`LE32(byte_length) || bytes`; inputs longer than `2^32-1` bytes are rejected.
The trailing NUL in every tag is part of the hash input.

```text
SEED_TAG   = b"riley.philox4x32-10.v1/seed\0"
STREAM_TAG = b"riley.philox4x32-10.v1/stream\0"
DOMAIN_TAG = b"riley.philox4x32-10.v1/domain\0"
FORK_TAG   = b"riley.philox4x32-10.v1/fork\0"

seed_digest   = SHA256(SEED_TAG || LE64(master_seed))
stream_digest = SHA256(STREAM_TAG || seed_digest || LP(stream))
digest        = SHA256(DOMAIN_TAG || stream_digest || LP(domain))
```

`digest[0:8]` is decoded as `(k0, k1)`, two little-endian `u32` key words.
`digest[8:16]` is decoded as `(nonce0, nonce1)`, also two little-endian `u32`
words.  The remaining digest bytes are retained as part of the derivation
identity and are not silently discarded when forking.

The stream ID isolates logical requests.  It must be stable and unique within
the lifetime of a master seed; a request ID is the usual choice.  The domain
isolates consumers within one request, for example `token-sampling`,
`speculative-acceptance`, and `speculative-residual`.  Reusing a tuple
`(master_seed, stream, domain)` deliberately recreates the same stream.

## State and draw order

The 128-bit Philox counter for block number `b` is:

```text
(low32(b), high32(b), nonce0, nonce1),  0 <= b < 2^64
```

The logical state points to the **next** word and consists of the algorithm
ID, the full 32-byte derivation digest, a 64-bit block number, a word offset in
`0..3`, and an exhausted flag.  A draw evaluates the current block, returns
the word at the offset, then advances the offset.  Advancing past offset 3
moves to the next block.  After the fourth word of block `2^64-1`, the state is
marked exhausted; another draw must fail and must never wrap to block zero.

There is no hidden process-global generator.  Each request owns its state, and
moving, batching, pausing, or resuming a request cannot cause another request
to consume its words.  Greedy decoding consumes zero RNG words.

### Snapshot and restore

A canonical JSON snapshot has this shape:

```json
{
  "algorithm_id": "riley.philox4x32-10.v1",
  "derivation_digest_hex": "64 lowercase hexadecimal characters",
  "block": "unsigned decimal u64 encoded as a JSON string",
  "word_offset": 0,
  "exhausted": false
}
```

The snapshot is taken between draws and identifies the next word.  `block` is
a string so JSON consumers cannot lose precision.  Restore validates every
field, re-derives the key and nonce from the digest, and yields the same suffix
bit-for-bit.  Unknown algorithm IDs, non-canonical hex, out-of-range values,
and inconsistent exhausted states are errors.  Snapshotting and restoring do
not consume a word.  The sole canonical exhausted representation preserves
`block = "18446744073709551615"` and `word_offset = 3` with `exhausted = true`;
the offset denotes the last consumed word because no next word exists.

### Fork

`fork(label)` creates a child at block zero and word offset zero using:

```text
child_digest = SHA256(FORK_TAG || parent_digest || LP(label))
```

The child key and nonce are decoded from `child_digest` in the same way as the
root digest.  Forking does not inspect or change the parent's block or offset,
so the child is identical whether it is created before or after parent draws.
The same parent identity and label intentionally produce the same child;
callers requiring distinct children must use distinct labels.  Nested forks
are supported because the child digest becomes the next parent digest.

## Mapping one word to a uniform value

The portable scalar mapping is an IEEE-754 binary64 value in the open interval
`(0, 1)`:

```text
uniform_open01(x) = (binary64(x) + 0.5) * 2^-32
```

Both the addition and multiplication are exact in binary64 for every `u32`,
so this definition does not depend on fused operations or rounding modes.  In
particular, `0` maps to `2^-33` and `0xffffffff` maps to `1 - 2^-33`.
Implementations needing `f32` must first compute this binary64 value and then
perform an explicitly documented conversion; `f32` values are not part of the
version-1 cross-language bitwise contract.

## Known-answer vectors

Hex words are written in numeric order, not as a byte dump.  These vectors are
merge gates for every implementation.

### Raw Philox core

```text
counter = [00000000, 00000000, 00000000, 00000000]
key     = [00000000, 00000000]
output  = [6627e8d5, e169c58d, bc57ac4c, 9b00dbd8]

counter = [ffffffff, ffffffff, ffffffff, ffffffff]
key     = [ffffffff, ffffffff]
output  = [408f276d, 41c83b0e, a20bc7c6, 6d5451fd]
```

### SHA-256 derivation and first block

```text
master_seed = 42
stream      = UTF-8 "request-0001"
domain      = UTF-8 "token-sampling"

seed_digest   = 5593ba984817f52bc0241ca4da04b0ecfee2fe30bd75567c36a8196f66c08418
stream_digest = c98613181da8edadd6bf2e27d23576268a3ba2542e514fdfdb9f2f18c23dff96
digest        = c9826cff0d3267e8597dd2f16928ca083e76df3b8fc4af9810c747236deb517a
key           = [ff6c82c9, e867320d]
nonce         = [f1d27d59, 08ca2869]
block 0       = [c875248d, 7d889ea1, d6887282, 6daf5198]
```

After two draws the snapshot has `block = "0"`, `word_offset = 2`, and
`exhausted = false`; restoring it returns `d6887282` next.  After all four
draws the next state has `block = "1"`, `word_offset = 0`, and
`exhausted = false`.

The corresponding four binary64 uniforms, printed with 17 significant
decimal digits, are:

```text
0.78303745703306049
0.49036590044852346
0.83801952062640339
0.42845640156883746
```

Forking that state with the UTF-8 label `draft` gives:

```text
child_digest = c266fe1f4647900fd7c88a368c5b9dde897d4d0c49bc5378dbb66dda9c1c84f1
child key    = [1ffe66c2, 0f904746]
child nonce  = [368ac8d7, de9d5b8c]
child block0 = [f58f7119, 0cbf7dc4, 8531e25f, 50267c0d]
```

## Versioning rule

Changing a constant, hash tag, byte encoding, digest slice, counter layout,
draw order, fork rule, or uniform mapping requires a new algorithm ID.  Adding
a faster implementation under this ID is allowed only when all known-answer,
snapshot/restore, request-isolation, and fork tests remain bit-for-bit equal.
