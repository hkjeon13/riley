from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=(r/'mixed_phase_v1.py').read_text();assert s.count('concurrency <= 8')==1;(r/'mixed_phase_v34.py').write_text(s.replace('concurrency <= 8','concurrency <= 32'))
s=(r/'run_fill_v34.py').read_text().replace('natural-fill-v34','natural-fill-v34-high').replace('from mixed_phase_v1 import','from mixed_phase_v34 import').replace('for concurrency in (4,8,16,32):','for concurrency in (16,32):');(r/'run_fill_v34_high.py').write_text(s)
p=r/'analyze_fill_v34.py';s=p.read_text().replace("(r/f'natural-fill-v34/c{concurrency}.log')","(r/('natural-fill-v34' if concurrency<=8 else 'natural-fill-v34-high')/f'c{concurrency}.log')");p.write_text(s)
