import sqlite3,sys,collections
db=sqlite3.connect(sys.argv[1]);lo,hi=int(sys.argv[2]),int(sys.argv[3])
names=dict(db.execute("select id,value from StringIds"))
tids={t:names[n] for n,t in db.execute("select nameId,globalTid from ThreadNames")}
eng=[t for t,n in tids.items() if n=='riley-engine'][0]
ev=dict(db.execute("select id,start from COMPOSITE_EVENTS where globalTid=? and start>=? and start<=?",(eng,lo,hi)))
stacks=collections.defaultdict(list)
for i,sym,mod,depth in db.execute("select id,symbol,module,stackDepth from SAMPLING_CALLCHAINS order by id,stackDepth"):
    if i in ev: stacks[i].append((names.get(sym,'?'),names.get(mod,'?').split('/')[-1]))
print('engine samples',len(stacks))
incl=collections.Counter();leaf=collections.Counter();host=0
for st in stacks.values():
    syms=[s for s,m in st]
    insync=any('StreamSynchronize' in s or 'GraphLaunch' in s for s in syms)
    # fallback: classify by whether any riley frame is replay_transfer
    inreplay=any('replay_transfer' in s for s in syms)
    key='replay_transfer(launch+sync)' if inreplay else 'host'
    if not inreplay:
        host+=1
        seen=set()
        for s,m in st:
            if m.startswith('riley') and s not in seen: seen.add(s);incl[s[:150]]+=1
        leaf[(st[0][0][:110],st[0][1])]+=1
print('host samples',host,'of',len(stacks))
print('-- inclusive (riley frames) among host samples');[print(f'{100*c/host:5.1f}% {s}') for s,c in incl.most_common(45)]
print('-- leaf');[print(f'{100*c/host:5.1f}% {s}') for s,c in leaf.most_common(25)]
