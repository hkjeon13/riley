"""Reference server KV for a single sequential workflow, published only after a real preview POST."""

from agent_workflow import fingerprint


class HTTPPreview:
    def __init__(self, model, tokenizer, budget):
        self.model = model
        self.tokenizer = tokenizer
        self.budget = budget
        self.cache = None
        self.ids = []
        self.records = []

    def serve(self, request, ids, emit):
        import torch

        assert len(ids) <= self.budget and request["max_tokens"] == 1
        assert self.cache is None
        with torch.inference_mode():
            result = self.model(
                input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1
            )
            token = int(result.logits[0, -1].argmax().item())
        self.cache = result.past_key_values
        self.ids = ids
        assert self.cache.get_seq_length() == len(ids)
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": 1,
            "total_tokens": len(ids) + 1,
        }
        ident = f"preview-{len(self.records)}"
        self.records.append(
            {
                "preview_request": request,
                "preview_ids_sha256": fingerprint(ids),
                "preview_usage": usage,
                "preview_output": token,
            }
        )
        content = self.tokenizer.decode([token], skip_special_tokens=True)
        emit({"id": ident, "choices": [{"index": 0, "delta": {"content": content}}]})
        emit(
            {
                "id": ident,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": usage,
            }
        )

    def take(self, ids):
        assert self.cache is not None
        cache, self.cache = self.cache, None
        shared = next(
            (i for i, (a, b) in enumerate(zip(ids, self.ids)) if a != b),
            min(len(ids), len(self.ids)),
        )
        reused = min(shared, len(ids) - 1) // 16 * 16
        self.records[-1].update(common=shared, reused=reused)
        if reused:
            if reused < len(self.ids):
                cache.crop(reused - len(self.ids))
            assert cache.get_seq_length() == reused
            return cache
        return None
