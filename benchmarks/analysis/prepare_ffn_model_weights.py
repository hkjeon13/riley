"""Offline BF16 fixture extraction; not a Python serving dependency."""
import hashlib,json,pathlib,struct,sys
import numpy as np
source=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir()
with source.open('rb') as f:
 size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));base=8+size
 records=[]
 for layer in range(30):
  for kind,shape in [('gate',(1536,576)),('up',(1536,576)),('down',(576,1536))]:
   name=f'model.layers.{layer}.mlp.{kind}_proj.weight';meta=header[name]
   assert meta['dtype']=='BF16' and tuple(meta['shape'])==shape
   begin,end=meta['data_offsets'];assert end-begin==np.prod(shape)*2
   f.seek(base+begin);raw=f.read(end-begin);a=np.frombuffer(raw,dtype='<u2').reshape(shape);n,k=shape
   packed=a.reshape(n//8,8,k//16,2,8).transpose(0,2,3,1,4).copy()
   restored=packed.transpose(0,3,1,2,4).reshape(shape);assert np.array_equal(a,restored)
   dest=out/f'{layer}-{kind}.bin';dest.write_bytes(packed.tobytes())
   records.append({'tensor':name,'shape':shape,'file':dest.name,'source_tensor_sha256':hashlib.sha256(raw).hexdigest(),'packed_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'layout':'N/8,K/16,K-half,N%8,K%8','tensors':records},indent=2)+'\n')
print('90 actual BF16 tensors packed and inverse layout verified')
