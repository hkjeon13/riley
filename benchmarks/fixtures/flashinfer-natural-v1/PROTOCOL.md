# Natural-language numerical screen v1

Inputs and rules fixed before inspecting results. Eight newly authored English prose passages; not an external benchmark or a representative quality dataset. Use the first 32 tokens as prompt and next 32 as target, with the local model tokenizer and no added special tokens. Evaluate identical target histories with existing exact Riley, experimental consistent-decode v2, and offline HF eager FP32 (TF32 off).

Predeclared screen: all logits finite, valid output counts and clean resource release; aggregate target NLL(candidate) <= NLL(baseline), and aggregate KL(FP32 || candidate) <= KL(FP32 || baseline). No tolerance adjustment after observing results. Report every passage and aggregate, including adverse differences and argmax agreements. Failure of either relative metric means this screen does not pass. Passing this small screen is not general quality acceptance, exact token equivalence, or serving qualification. No vLLM performance claim comes from it.

Rust produces logits without calling Python. Token preparation and the independent FP32 reference are offline Python tools. Preserve tokenizer/input/source/binary hashes and raw logits remotely. A subsequent serving benchmark must report the experimental numerical profile explicitly and compare matched workloads at a meaningful integration milestone.
