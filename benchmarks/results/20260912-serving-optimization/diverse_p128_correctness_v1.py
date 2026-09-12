import concurrent.futures,hashlib,http.client,json,os,pathlib,signal,socket,subprocess,sys
r=pathlib.Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r))
from tokenizers import Tokenizer
from batch7_http_check import wait_ready
from serving_token_client_v2 import TokenHttpClient,TokenReference,overlap_peak
out=r/'diverse-p128-correctness-v1';out.mkdir()
plan=json.loads((r/'token-serving-round16-plan-v2.json').read_text());base=plan['lanes']['baseline'];tok=Tokenizer.from_file('/data/riley-benchmark/20260827T051948Z-d7ad713a/model/tokenizer.json')
texts=[
('summary','Summarize the following project update in two sentences. The library team replaced an aging database, migrated the catalog in stages, and checked each imported record. A small pilot found several duplicate entries. The team corrected these records before opening the new search service to all readers. Staff training continues next week. '),
('python','Explain what this Python function computes and describe one edge case. def count_words(text): counts = {}; for word in text.split(): counts[word] = counts.get(word, 0) + 1; return counts. Consider repeated words, punctuation, and an empty input string. A student wants to adapt the function for a small document collection. '),
('arithmetic','Solve this problem step by step. A school buys twelve boxes of pencils. Each box contains twenty four pencils. Five classes receive thirty pencils each, and the remaining pencils are kept in a supply cupboard. How many pencils remain? Explain your calculation clearly and check the result by adding the distributed and remaining pencils. '),
('planning','Write a practical plan for a community garden. Volunteers can meet on Saturday mornings. The site receives six hours of sunlight each day and has one water tap. The group wants to grow beans, tomatoes, and herbs while keeping a clear walking path. Include preparation, planting, watering, and a simple way to assign responsibilities. '),
('comparison','Compare bicycles and buses for a short daily commute. Consider travel time, weather, cost, flexibility, and accessibility. The route is five kilometers long and includes two busy intersections. A bus arrives every fifteen minutes. A protected cycling lane covers most of the route, but there is limited indoor bicycle storage at the destination. '),
('extraction','Extract the date, location, and action items from these meeting notes. The workshop will take place on October twelve at the Riverside Learning Center. Maya will reserve the room, Omar will prepare the handouts, and Lin will test the projector. Registration closes three days before the event. The organizers expect approximately twenty participants. '),
('science','Explain why water evaporates from a shallow bowl even when it is not boiling. Use simple language for a curious student. Mention the motion of molecules, temperature, and the surrounding air. The student placed identical bowls near a sunny window and in a cool cupboard, then observed different changes in water level. '),
('editing','Rewrite this message in a polite and concise tone. I have asked about the delivery twice and still have no update. We need the equipment for a training session next Wednesday. Please confirm whether the shipment has left your warehouse and provide an estimated arrival date. If there is a delay, explain the available alternatives. '),
('logic','Three friends arrive at a cafe at different times. Alice arrives before Ben. Clara arrives after Alice but before Ben. List their arrival order and explain which statements determine it. Then describe whether the order would still be uniquely determined if the second statement only said that Clara arrives after Alice. '),
('sql','Describe a SQL query that lists each customer and the number of orders they placed. The customers table contains customer_id and name. The orders table contains order_id, customer_id, and order_date. Include customers with no orders and explain why the choice of join matters. The report should contain one row per customer. '),
('story','Continue this short story in a calm descriptive style. The old station clock stopped just as Nora stepped onto the empty platform. In her pocket was a postcard with a drawing of the same clock, sent by someone whose name she did not recognize. A gardener across the tracks waved and pointed toward a narrow footbridge. '),
('classification','Classify these customer comments as positive, negative, or mixed and explain briefly. The package arrived early, but one item was missing. The instructions were clear and the assembly took only ten minutes. The support agent was friendly, although it took three days to receive a reply. Focus on the meaning of each complete comment. ')]
corpus=[]
for i,(name,text) in enumerate(texts):
 ids=tok.encode(text*4).ids[:128];prompt=tok.decode(ids,skip_special_tokens=False)
 assert len(ids)==128 and tok.encode(prompt).ids==ids
 corpus.append({'id':name,'prompt':prompt,'prompt_token_ids':ids,'max_tokens':[8,16,24,32][i%4]})
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
write('corpus.json',corpus)
def launch(binary,capacity,prefix):
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 args=[v.format(port=port) for v in base['argv']];args[0]=str(binary)
 for k,v in [('--max-active-sequences',capacity),('--kv-blocks',10*capacity)]:args[args.index(k)+1]=str(v)
 args+=['--shutdown-on-stdin'];env=os.environ.copy();env.update(plan['base_environment']);env.update(base['env'])
 log=(out/(prefix+'.log')).open('w');p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.PIPE,start_new_session=True)
 write(prefix+'-launch.json',{'argv':args,'binary_sha256':hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()})
 return p,port,log
def stop(p,log):
 if p.poll() is None:p.stdin.write(b'\n');p.stdin.flush()
 try:code=p.wait(timeout=30)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);code=p.wait(timeout=20)
 log.close();return code
refs=[];p,port,log=launch(base['argv'][0],1,'baseline')
try:
 wait_ready(p,port)
 for item in corpus:
  conn=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
  conn.request('POST','/v1/completions',json.dumps({'model':'g04-smol','prompt':item['prompt'],'max_tokens':item['max_tokens'],'temperature':0,'return_token_ids':True,'stream':False}),{'Content-Type':'application/json'})
  response=conn.getresponse();body=json.loads(response.read());assert response.status==200,body;conn.close()
  choice=body['choices'][0];assert choice['prompt_token_ids']==item['prompt_token_ids'];refs.append(body)
finally:assert stop(p,log)==0
write('references.json',refs)
records=[]
for capacity in [1,4]:
 p,port,log=launch(r/'multisequence-candidate-v4/riley',capacity,f'candidate-c{capacity}')
 try:
  wait_ready(p,port)
  with TokenHttpClient() as client:
   def one(i):
    item=corpus[i%12];choice=refs[i%12]['choices'][0]
    ref=TokenReference('g04-smol',tuple(item['prompt_token_ids']),tuple(choice['token_ids']),choice['text'],choice['finish_reason'])
    row=client.request(port,{'model':'g04-smol','prompt':item['prompt'],'max_tokens':item['max_tokens'],'temperature':0},ref,streaming=bool(i%2));row['corpus_id']=item['id'];return row
   with concurrent.futures.ThreadPoolExecutor(max_workers=1 if capacity==1 else 8) as pool:rows=list(pool.map(one,range(12 if capacity==1 else 48)))
  write(f'candidate-c{capacity}-rows.json',rows)
  assert all(x['reference_match'] and x['protocol_valid'] and x['transport_complete'] for x in rows)
  records.append({'capacity':capacity,'requests':len(rows),'strict_matches':len(rows),'observed_overlap':overlap_peak(rows)})
 finally:assert stop(p,log)==0
 print(json.dumps(records[-1]),flush=True)
write('completion.json',{'records':records,'corpus_cases':12,'synthetic_fixed_p128_only':True,'performance_claim':False})
