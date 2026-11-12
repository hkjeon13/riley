import sqlite3,sys,re,collections
for path in sys.argv[1:]:
    db=sqlite3.connect(path)
    rows=db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
    first=max((i for i in range(1,len(rows)) if rows[i][0]-rows[i-1][1]>5e9),default=0); rows=rows[first:]
    reqs=[];cur=[rows[0]]
    for r in rows[1:]:
        if r[0]-cur[-1][1]>3e6: reqs.append(cur);cur=[]
        cur.append(r)
    reqs.append(cur)
    print("==",path,len(reqs),"groups")
    for g in reqs[-4:]:
        span=(g[-1][1]-g[0][0])/1e6; busy=sum(e-s for s,e,_ in g)/1e6
        gaps=sorted(((g[i][0]-g[i-1][1])/1e3, re.sub(r"\(.*","",g[i-1][2])[:50], re.sub(r"\(.*","",g[i][2])[:50]) for i in range(1,len(g)))[-4:]
        print(f" n={len(g)} span {span:.2f} busy {busy:.2f} idle {span-busy:.2f}; biggest gaps us:",[(round(a),b,c) for a,b,c in gaps])
