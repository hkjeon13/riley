"""Observe the real attention backend at its existing custom-op boundary."""
from pathlib import Path
import json
ROOT=Path('/tmp/riley-g04-serving-trace-260911')
class ServingTrace:
    def install_trace(self):
        import types
        self.trace_originals=[];self.trace_records=[];self.trace_counters={}
        model=self.get_model()
        for i,layer in enumerate(model.model.layers):
            impl=layer.self_attn.attn.impl
            original=impl.forward
            self.trace_originals.append((impl,original))
            self.trace_counters[i]=0
            def observed(_impl, layer, query,key,value,kv_cache,attn_metadata,*args,_i=i,_original=original,**kwargs):
                index=self.trace_counters[_i];self.trace_counters[_i]+=1
                capture=index<=31
                tensors={}
                if capture:
                    tensors={k:t.detach().clone() for k,t in [('q',query),('k',key),('v',value)]}
                result=_original(layer,query,key,value,kv_cache,attn_metadata,*args,**kwargs)
                if capture:
                    tensors['context']=kwargs['output'].detach().clone()
                    record={'layer':_i,'call':index,'backend':type(_impl).__name__,'tensors':{}}
                    for name,tensor in tensors.items():
                        tensor=tensor.contiguous()
                        filename=f'layer{_i}-call{index}-{name}.bf16'
                        (ROOT/filename).write_bytes(tensor.view(__import__('torch').uint8).cpu().numpy().tobytes())
                        record['tensors'][name]={'file':filename,'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
                    self.trace_records.append(record)
                return result
            impl.forward=types.MethodType(observed,impl)
        return {'layers':len(self.trace_originals),'boundary':'actual backend impl.forward','cuda_graphs':'NONE'}
    def finish_trace(self):
        for impl,original in self.trace_originals: impl.forward=original
        result={'records':self.trace_records,'calls':self.trace_counters,'restored':True,'performance_trials':0}
        (ROOT/'serving-tensors.json').write_text(json.dumps(result,indent=2)+'\n')
        return {'records':len(self.trace_records),'calls':self.trace_counters,'restored':True}
