import importlib.util,sys,json,socket,threading,time,hashlib
from pathlib import Path
code=Path('/data/riley-serving-261007/kernel-batch12-stability-controller-draft-attempt01')
out=code/'loopback-tests';out.mkdir()
sys.dont_write_bytecode=True;sys.path.insert(0,str(code))
spec=importlib.util.spec_from_file_location('draft_stability',code/'stability.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
body={'model':'g04-smol','prompt':'fixed prompt','max_tokens':32,'temperature':0,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True}}
valid={'model':'g04-smol','object':'text_completion','choices':[{'token_ids':[42],'text':'가'}]}
frame=lambda x:('data: '+json.dumps(x,ensure_ascii=False)+'\n\n').encode()
tests=[('plain-fragmented',200,frame(valid),False,True,None),('chunked-fragmented',200,frame(valid),True,True,None),('http500',500,frame(valid),False,True,'AssertionError'),('error200',200,frame({'error':{'message':'test'}}),True,True,'AssertionError'),('done-before-token',200,b'data: [DONE]\n\n',True,True,'AssertionError'),('EOF-without-token',200,frame({'model':'g04-smol','object':'text_completion','choices':[]}),True,True,'RuntimeError'),('early-close',200,frame(valid),False,False,None)]
tests.append(('wrong-prefix',200,frame({'model':'g04-smol','object':'text_completion','choices':[{'token_ids':[43],'text':'wrong'}]}),True,True,'AssertionError'))
results=[]
for name,status,payload,chunked,after,expected in tests:
 listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);port=listener.getsockname()[1];received={};server_errors=[]
 def serve():
  conn=None
  try:
   conn,_=listener.accept();conn.settimeout(5);raw=b''
   while b'\r\n\r\n' not in raw:raw+=conn.recv(4096)
   head,data=raw.split(b'\r\n\r\n',1);length=int(next(x.split(b':',1)[1] for x in head.split(b'\r\n') if x.lower().startswith(b'content-length:')))
   while len(data)<length:data+=conn.recv(4096)
   received['body']=data[:length]
   headers=f'HTTP/1.1 {status} test\r\nContent-Type: text/event-stream\r\nConnection: close\r\n'.encode()
   headers+=(b'Transfer-Encoding: chunked\r\n' if chunked else f'Content-Length: {len(payload)}\r\n'.encode())+b'\r\n'
   conn.sendall(headers)
   for i in range(0,len(payload),3):
    b=payload[i:i+3];conn.sendall((f'{len(b):x}\r\n'.encode()+b+b'\r\n') if chunked else b);time.sleep(.001)
   if chunked:conn.sendall(b'0\r\n\r\n')
  except (BrokenPipeError,ConnectionResetError):pass
  except BaseException as e:server_errors.append(repr(e))
  finally:
   if conn:conn.close()
   listener.close()
 t=threading.Thread(target=serve);t.start();failure=None;path=out/(name+'-raw.json')
 try:module.cancel(port,body,after,path,expected_prefix=[42])
 except BaseException as e:failure=type(e).__name__
 t.join(6);assert not t.is_alive() and not server_errors,(name,server_errors)
 assert failure==expected,(name,failure,expected)
 raw=json.loads(path.read_text());assert raw['encoded_request_sha256']==hashlib.sha256(received['body']).hexdigest()
 assert json.loads(received['body'])==body
 assert raw['closed_after_observed_token']==(expected is None and after)
 results.append({'case':name,'expected_exception':expected,'observed_exception':failure,'preserved_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pass':True})
(out/'result.json').write_text(json.dumps({'GPU_executed':False,'production_server_executed':False,'cases':results,'all_passed':True,'scope':'HTTP cancellation driver transport and failure-evidence unit checks only; no serving correctness/stability qualification'},indent=2)+'\n')
print(json.dumps({'cases':len(results),'all_passed':True,'result':str(out/'result.json')}))
