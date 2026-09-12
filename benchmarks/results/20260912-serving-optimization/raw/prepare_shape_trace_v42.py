from pathlib import Path
r=Path('/tmp/riley-opt-260912')
s=(r/'run_native_trace_v41.py').read_text().replace('native-trace-v41','native-shape-v42').replace('variable-candidate-v41','variable-candidate-v42')
s=s.replace('for cap in (16,32):','for cap in (128,512):').replace("'--max-active-sequences',str(cap)","'--max-active-sequences','16'").replace("'--kv-blocks',str(cap*64)","'--kv-blocks',str(16*64)").replace("'--prefill-chunk-tokens','512'","'--prefill-chunk-tokens',str(cap)").replace('concurrency=cap','concurrency=16').replace('records=[]','corpus=corpus[1:2];refs=refs[1:2]\nassert corpus[0]["max_tokens"]==64 and len(corpus[0]["prompt_token_ids"])==128\nrecords=[]').replace("{'capacity':cap,","{'prefill_capacity':cap,'client_concurrency':16,")
(r/'run_shape_trace_v42.py').write_text(s)
s=(r/'analyze_trace_v41.py').read_text().replace('native-trace-v41','native-shape-v42').replace('for cap in [16,32]:','for cap in [128,512]:').replace('V41 V4 natural mixed96 requests each atC16/C32','V42 V4 fixedP128/O64, 96 requests each atC16; keys128/512 are prepared prefill capacities, all other serving settings match')
(r/'analyze_shape_trace_v42.py').write_text(s)
