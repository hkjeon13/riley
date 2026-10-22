# Rejected lossless BF16 operator experiment

Sibling ZIP `20261022-lossless-projection-v19-r2.zip`:32130bytes,21entries, SHA256 `a181dbbe835bbb14df33e7fe1dedc061d9fdc7504972e996a35367b9bbb8e0d3`. Every ZIP CRC and SHA were checked locally. Standalone source includes exact bit encoding/decoding, WMMA pipeline, original native/Lt controls, raw cases/pairs, observer samples and static compiler resource usage. No model bytes or ELF files are committed.

EFS `/mnt/efs/nlp/psyche/riley-projection-layout-v19-20261022/lossless`. Three actual checkpoint layers; four R values; six finite activation states.360 exact comparisons include216 compressed cases and144 uncompressed controls. All compressed variants are slower. Original accuracy/tail criteria remain unchanged. This screen is separate from normal HTTP performance and is not adopted.

The display copy of resource-usage.txt strips trailing whitespace; the ZIP retains the raw compiler output.
