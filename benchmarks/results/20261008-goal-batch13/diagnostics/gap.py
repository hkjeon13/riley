import sqlite3,sys,statistics as st
db=sqlite3.connect(sys.argv[1]); lo,hi=int(sys.argv[2]),int(sys.argv[3])
names=dict(db.execute("select id,value from StringIds"))
# runtime API calls in window
api=[(s,e,names[n]) for s,e,n in db.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? order by start",(lo,hi))]
launch=[(s,e) for s,e,n in api if 'GraphLaunch' in n]; sync=[(s,e) for s,e,n in api if 'StreamSynchronize' in n]
k=list(db.execute("select start,end,graphId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start",(lo,hi)))
m=list(db.execute("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? order by start",(lo,hi)))
print('launches',len(launch),'syncs',len(sync),'kernels',len(k),'memcpy',len(m))
us=lambda x:round(x/1e3,1)
import bisect
ks=[x[0] for x in k]
rows=[]
for i in range(len(launch)-1):
    ls,le=launch[i]; nls=launch[i+1][0]
    a=bisect.bisect_left(ks,ls); b=bisect.bisect_left(ks,nls)
    if b<=a: continue
    kk=k[a:b]; first=kk[0][0]; last=max(x[1] for x in kk); work=sum(x[1]-x[0] for x in kk)
    sy=[x for x in sync if x[0]>=ls and x[0]<nls]
    send=sy[0][1] if sy else last
    rows.append(dict(n=len(kk),period=nls-ls,launch_api=le-ls,to_first=first-ls,span=last-first,work=work,intra_gap=(last-first)-work,sync_after_last=send-last,host=nls-send))
def stat(key,sel):
    v=[r[key] for r in sel]; return f"{key}: med {us(st.median(v))} mean {us(st.mean(v))} sum_ms {round(sum(v)/1e6,1)}"
for label,sel in [('all',rows),('decode-only(n<=300)',[r for r in rows if r['n']<=300]),('mixed',[r for r in rows if r['n']>300])]:
    if not sel: continue
    print('==',label,len(sel),'kernels/iter med',st.median([r['n'] for r in sel]))
    for key in ['period','launch_api','to_first','span','work','intra_gap','sync_after_last','host']: print('  ',stat(key,sel))
