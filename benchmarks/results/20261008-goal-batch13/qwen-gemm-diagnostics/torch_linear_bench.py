"""Diagnostic: time torch F.linear (the call vLLM's model makes) on cold BF16 weights for the Qwen projection shapes."""
import time, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False
print(torch.__version__, torch.version.cuda)
for K, N in ((2048, 2560), (2048, 2048), (2048, 22016), (11008, 2048)):
    R = (700 << 20) // (K * N * 2) + 2
    Ws = [torch.full((N, K), 0.01, dtype=torch.bfloat16, device='cuda') for _ in range(R)]
    bias = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
    for M in (1, 8, 32):
        x = torch.full((M, K), 0.01, dtype=torch.bfloat16, device='cuda'); out = {}
        for name, fn in (('linear', lambda w: F.linear(x, w)), ('linear+bias', lambda w: F.linear(x, w, bias)), ('matmul', lambda w: x @ w.t())):
            for i in range(20): fn(Ws[i % R])
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(300): fn(Ws[i % R])
            torch.cuda.synchronize(); out[name] = (time.perf_counter() - t0) / 300 * 1e6
        print(f"K={K} N={N} M={M}: " + "  ".join(f"{k} {v:.1f} us" for k, v in out.items()), flush=True)
    del Ws
