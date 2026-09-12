from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text()
s=s.replace('pub block_ownership:Vec<BlockOwnership>,','pub block_ownership:Vec<BlockOwnership>,\n    /// Explicit V6 capability, bound to the retained native graph by the session.\n    pub packed_prefill:bool,')
s=s.replace('pub fn validate<const ROWS:usize>', '''fn result_magic<const ROWS:usize>(e:&Expectation<ROWS>)->u32 {if e.packed_prefill{0x36524d52}else{Layout::<ROWS>::RESULT_MAGIC}}
fn compact_magic<const ROWS:usize>(e:&Expectation<ROWS>)->u32 {result_magic(e)^0x8000_0000}
pub fn validate<const ROWS:usize>''')
s=s.replace('check(e.stage!=InputStage::Prefill || e.rows.len()==1,"stage","prefill must describe one request")?;', '''check(!e.packed_prefill || ROWS==32,"capacity","packed prefill requires V6 capacity")?;
    check(e.stage!=InputStage::Prefill || e.rows.len()<=if e.packed_prefill{4}else{1},"stage","unsupported prefill owner count")?;
    let total=e.rows.iter().try_fold(0usize,|n,r|n.checked_add(r.input_tokens.len()).ok_or_else(||overflow("tokens")))?;
    check(e.stage!=InputStage::Prefill || total<=1024,"tokens","packed token capacity exceeded")?;''')
s=s.replace('(0,Layout::<ROWS>::MAGIC),(4,Layout::<ROWS>::VERSION)', '(0,if e.packed_prefill{0x36444d52}else{Layout::<ROWS>::MAGIC}),(4,if e.packed_prefill{6}else{Layout::<ROWS>::VERSION})')
s=s.replace('    for (i,row) in e.rows.iter().enumerate() {\n        let b=', '    let mut token_offset=0usize;\n    for (i,row) in e.rows.iter().enumerate() {\n        let b=',1)
s=s.replace('        u64_at(packet,b+48,row.sequence_tag);u64_at(packet,b+56,row.cookie);','''        u64_at(packet,b+48,row.sequence_tag);u64_at(packet,b+56,row.cookie);
        if e.stage==InputStage::Prefill {
            if e.packed_prefill {u32_at(packet,b+64,token_offset as u32);}
            for &t in &row.input_tokens {u32_at(packet,Layout::<ROWS>::TOKENS_OFFSET+token_offset*4,t);token_offset+=1;}
        }''')
s=s.replace('    if e.stage==InputStage::Prefill {for (i,&t) in e.rows[0].input_tokens.iter().enumerate(){u32_at(packet,Layout::<ROWS>::TOKENS_OFFSET+i*4,t);}}','    if e.packed_prefill && e.stage==InputStage::Prefill {u32_at(packet,36,token_offset as u32);}')
s=s.replace('(124,Layout::<ROWS>::RESULT_MAGIC)', '(124,result_magic(e))')
s=s.replace('u32_at(&mut expected,124,Layout::<ROWS>::COMPACT_RESULT_MAGIC)', 'u32_at(&mut expected,124,compact_magic(e))')
s=s.replace('u32_at(&mut h,124,Layout::<ROWS>::COMPACT_RESULT_MAGIC)', 'u32_at(&mut h,124,compact_magic(e))')
s=s.replace('rows:vec![],block_ownership:vec![]','rows:vec![],block_ownership:vec![],packed_prefill:false')
p.write_text(s)
p=r/'crates/riley-scheduler/src/authority.rs';s=p.read_text().replace('let e = Expectation { owner_generation:', 'let e = Expectation { packed_prefill:false, owner_generation:');p.write_text(s)
p=r/'crates/riley-runtime/src/llama/variable_session.rs';s=p.read_text().replace('if e.rows.is_empty() || e.rows.len()>if self.shared', 'if e.packed_prefill || e.rows.is_empty() || e.rows.len()>if self.shared');p.write_text(s)
p=r/'kernels/src/prefill_shape_packet.hpp';s=p.read_text().replace('template<uint32_t Rows>', 'template<uint32_t Rows,bool Packed=false>')
s=s.replace('constexpr uint32_t magic=Rows==8?0x33444d52:(Rows==16?0x34444d52:0x35444d52),version=Rows==8?3:(Rows==16?4:5);', 'static_assert(!Packed||Rows==32,"packed prefill requires V6 capacity");\n constexpr uint32_t magic=Packed?0x36444d52:(Rows==8?0x33444d52:(Rows==16?0x34444d52:0x35444d52)),version=Packed?6:(Rows==8?3:(Rows==16?4:5));')
s=s.replace('!zero(36,40)', '(Packed&&stage==0?u32(36)==0||u32(36)>1024:!zero(36,40))')
s=s.replace('(stage==0&&active!=1)', '(stage==0&&active>(Packed?4U:1U))')
s=s.replace('!zero(b+64,b+128)', '!zero(b+(Packed&&stage==0?68:64),b+128)')
s=s.replace('prefill_count=count;', '''if(Packed){if(u32(b+64)!=prefill_count||count>1024-prefill_count)return false;}
   if(u32(tokens+prefill_count*4)!=u32(b))return false;
   prefill_count+=count;''')
s=s.replace(' if(!zero(header+active*stride,tokens))', ' if(Packed&&stage==0&&u32(36)!=prefill_count)return false;\n if(!zero(header+active*stride,tokens))')
s+='\ninline bool valid_v6_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {return valid_variable_shape_packet<32,true>(p,bytes,physical,capacity);}\n'
p.write_text(s)
print('V6 wire and native parser added; existing sessions reject unrecorded V6 capability')
