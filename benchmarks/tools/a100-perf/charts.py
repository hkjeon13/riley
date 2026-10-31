"""charts.py A100-PERF.json OUT_DIR: per-model Riley vs vLLM charts for docs/images/a100-perf (Korean) and en/ (English).

A100-PERF.json comes from extract.py, run on the measurement host. Model order is the JSON order.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RILEY, VLLM = '#2f74d0', '#e8673a'
CONCURRENCY = [1, 8, 16, 32]
METRICS = [  # key, file, Korean title, English title
    ('output_tok_per_s', 'throughput', '처리량 (출력 tok/s, 높을수록 좋음)', 'throughput (output tok/s, higher is better)'),
    ('ttft_p50', 'ttft', 'TTFT p50 (ms, 낮을수록 좋음)', 'TTFT p50 (ms, lower is better)'),
    ('tpot_p50', 'tpot', 'TPOT p50 (ms, 낮을수록 좋음)', 'TPOT p50 (ms, lower is better)'),
    ('e2e_p50', 'e2e', 'E2E p50 (ms, 낮을수록 좋음)', 'E2E p50 (ms, lower is better)'),
]
LANG = {'ko': ('A100 모델별 {} — Riley vs vLLM', '동시 요청 수'), 'en': ('A100 per-model {} — Riley vs vLLM', 'concurrent requests')}

def draw(data, key, title, lang, path):
    names = list(data); columns = 4; rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.7 * rows + 1.0), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // columns][index % columns]
        for engine, color in (('riley-scheduler', RILEY), ('vllm', VLLM)):
            for workload, style, marker in (('fixed', '-', 'o'), ('natural', '--', 's')):
                ys = [data[name]['cells'][f'c{c}-{workload}'][engine][key] for c in CONCURRENCY]
                ax.plot(CONCURRENCY, ys, style, color=color, marker=marker, markersize=5, linewidth=2)
        ax.set_title(name, fontsize=11); ax.set_xticks(CONCURRENCY); ax.set_ylim(bottom=0)
        ax.grid(True, color='#e6e6e6', linewidth=0.8); ax.spines[['top', 'right']].set_visible(False)
        if index + columns >= len(names): ax.set_xlabel(LANG[lang][1], fontsize=9)
    for index in range(len(names), rows * columns): axes[index // columns][index % columns].axis('off')
    handles = [Line2D([], [], color=RILEY, linewidth=2), Line2D([], [], color=VLLM, linewidth=2),
               Line2D([], [], color='#555', marker='o', linewidth=2), Line2D([], [], color='#555', marker='s', linestyle='--', linewidth=2)]
    fig.legend(handles, ['Riley', 'vLLM 0.27.1', 'fixed', 'natural'], loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1 - 0.55 / fig.get_figheight()))
    fig.suptitle(LANG[lang][0].format(title), fontsize=15, y=1 - 0.12 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / fig.get_figheight())); fig.savefig(path, dpi=100); plt.close(fig)

if __name__ == '__main__':
    data = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); (out / 'en').mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans']
    for key, stem, ko, en in METRICS:
        draw(data, key, ko, 'ko', out / f'{stem}.png'); draw(data, key, en, 'en', out / 'en' / f'{stem}.png')
