"""Independent checks of captured SSE content, reference identity and arrival bounds.

The terminal [DONE] marker is not stored in frames; transport completion still
relies on the hash-bound client, which requires that marker before valid=True.
"""
import hashlib,json

def validate_row(row,fixture):
    assert row['valid'] and row['id']==fixture['id']
    tokens=[];texts=[];prompt=None;finish=None;usage=None
    for event in row['frames']:
        if event.get('usage') is not None:usage=event['usage']
        for choice in event.get('choices',[]):
            assert choice['index']==0
            ids=choice.get('token_ids') or []
            assert all(type(token) is int and 0<=token<49152 for token in ids)
            tokens.extend(ids);texts.append(choice.get('text') or '')
            if choice.get('prompt_token_ids') is not None:
                assert prompt is None;prompt=choice['prompt_token_ids']
            if choice.get('finish_reason') is not None:
                assert finish is None;finish=choice['finish_reason']
    assert tokens==row['token_ids'] and prompt==row['prompt_token_ids']
    assert ''.join(texts)==row['text'] and finish==row['finish_reason'] and usage==row['usage']
    assert tokens and finish is not None and usage is not None
    assert usage['completion_tokens']==len(tokens) and usage['prompt_tokens']==len(prompt)
    arrivals=row['arrivals_ns']
    assert len(arrivals)==len(tokens) and row['started_ns']<=arrivals[0]<=arrivals[-1]<=row['ended_ns']
    assert all(a<=b for a,b in zip(arrivals,arrivals[1:]))
    text_hash=hashlib.sha256(json.dumps(row['text'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    expected={'prompt':prompt==fixture['prompt_token_ids'],'tokens':tokens==fixture['token_ids'],'text':text_hash==fixture['text_sha256'],'finish':finish==fixture['finish_reason']}
    assert row['checks']==expected
    return expected
