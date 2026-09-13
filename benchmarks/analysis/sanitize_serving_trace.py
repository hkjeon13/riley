"""Create a fresh SQLite with only numeric CUDA events and public CUDA/kernel names."""
import argparse,hashlib,re,sqlite3
from pathlib import Path

def sanitize(source,destination):
    assert not destination.exists()
    columns={
        'CUPTI_ACTIVITY_KIND_RUNTIME':['start','end','correlationId','globalTid','nameId','returnValue'],
        'CUPTI_ACTIVITY_KIND_KERNEL':['start','end','correlationId','globalPid','contextId','streamId','demangledName'],
        'CUPTI_ACTIVITY_KIND_MEMCPY':['start','end','correlationId','globalPid','contextId','streamId','bytes','copyKind'],
        'CUPTI_ACTIVITY_KIND_MEMSET':['start','end','correlationId','globalPid','contextId','streamId','bytes'],
    }
    rows={};names={}
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        for table,fields in columns.items():
            rows[table]=list(db.execute('select '+','.join(fields)+' from '+table)) if table in tables else []
            assert all(all(isinstance(v,int) for v in row) for row in rows[table])
        assert rows['CUPTI_ACTIVITY_KIND_KERNEL'] and rows['CUPTI_ACTIVITY_KIND_RUNTIME']
        for table,field in [('CUPTI_ACTIVITY_KIND_RUNTIME','nameId'),('CUPTI_ACTIVITY_KIND_KERNEL','demangledName')]:
            index=columns[table].index(field)
            for identifier in {r[index] for r in rows[table]}:
                name,=db.execute('select value from StringIds where id=?',(identifier,)).fetchone()
                if field=='nameId':assert re.fullmatch(r'cu[A-Za-z0-9_]{1,160}',name)
                else:
                    assert len(name)<2048 and re.fullmatch(r'[A-Za-z0-9_ :<>,*&().\[\]-]+',name)
                    assert name.startswith(('riley_','void riley_','void gemm_','gemm_','mixed_','void cutlass::','<unnamed>::rope_table_sin_kernel(','<unnamed>::rope_table_cos_in_place_kernel(')), 'unexpected kernel namespace'
                names[identifier]=name
    with sqlite3.connect(destination) as out:
        for table,fields in columns.items():
            out.execute('create table '+table+'('+','.join(f+' integer' for f in fields)+')')
            out.executemany('insert into '+table+' values('+','.join('?' for f in fields)+')',rows[table])
        out.execute('create table StringIds(id integer primary key,value text)')
        out.executemany('insert into StringIds values(?,?)',sorted(names.items()))
        out.execute('create table RILEY_TRACE_REDACTION(version integer, source_sha256 text, policy text)')
        hasher=hashlib.sha256()
        with source.open('rb') as raw:
            for block in iter(lambda:raw.read(1024*1024),b''):hasher.update(block)
        digest=hasher.hexdigest()
        out.execute('insert into RILEY_TRACE_REDACTION values(1,?,?)',(digest,'Numeric CUDA events and public API/kernel names only; no environment or process metadata'))
    print({k:len(v) for k,v in rows.items()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('destination',type=Path);a=ap.parse_args();sanitize(a.source,a.destination)
