from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor')
f=p/'variable_wire.rs';s=f.read_text()
s=s.replace('//! V3 request encoding. Packet bytes are checked against external live authority;\n//! they neither create ownership nor demonstrate GPU completion. Not yet dispatched.','//! Versioned request encoding checked against external live authority.\n//! V3 retains eight rows; V4 uses sixteen. Packet bytes never establish ownership.')
s=s.replace('pub struct Expectation {','pub struct Expectation<const ROWS:usize=8> {')
s=s.replace('pub fn validate(e:&Expectation)', 'pub fn validate<const ROWS:usize>(e:&Expectation<ROWS>)')
s=s.replace('    check(e.owner_generation', '    check(matches!(ROWS,8|16),"capacity","unsupported wire capacity")?;\n    check(e.owner_generation',1)
s=s.replace('matches!(e.max_active_rows,1|2|4|8) &&', 'matches!(e.max_active_rows,1|2|4|8|16) && e.max_active_rows as usize<=ROWS &&')
s=s.replace('pub fn encode_into(packet:&mut [u8],e:&Expectation)', 'pub fn encode_into<const ROWS:usize>(packet:&mut [u8],e:&Expectation<ROWS>)')
s=s.replace('pub fn validate_packet(packet:&[u8],scratch:&mut [u8],e:&Expectation)', 'pub fn validate_packet<const ROWS:usize>(packet:&[u8],scratch:&mut [u8],e:&Expectation<ROWS>)')
s=s.replace('pub fn validate_result<\'a>(bytes:&\'a[u8],e:&Expectation)', "pub fn validate_result<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>)")
s=s.replace('fn validate_result_row<\'a>(bytes:&\'a[u8],e:&Expectation,index:usize)', "fn validate_result_row<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>,index:usize)")
s=s.replace('fn result_identity_into(out:&mut[u8;128],e:&Expectation,token:u32)', 'fn result_identity_into<const ROWS:usize>(out:&mut[u8;128],e:&Expectation<ROWS>,token:u32)')
s=s.replace('fn result_row_identity_into(out:&mut[u8;128],e:&Expectation,index:usize,token:u32)', 'fn result_row_identity_into<const ROWS:usize>(out:&mut[u8;128],e:&Expectation<ROWS>,index:usize,token:u32)')
s=s.replace('pub fn validate_batch_result<\'a>(bytes:&\'a[u8],e:&Expectation)', "pub fn validate_batch_result<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>)")
s=s.replace('packet.len()==REQUEST_BYTES', 'packet.len()==Layout::<ROWS>::REQUEST_BYTES')
s=s.replace('(0,MAGIC),(4,3),(8,REQUEST_BYTES as u32)', '(0,Layout::<ROWS>::MAGIC),(4,Layout::<ROWS>::VERSION),(8,Layout::<ROWS>::REQUEST_BYTES as u32)')
s=s.replace('u32_at(packet,TOKENS_OFFSET+i*4,t)', 'u32_at(packet,Layout::<ROWS>::TOKENS_OFFSET+i*4,t)')
s=s.replace('(124,0x33524d52)', '(124,Layout::<ROWS>::RESULT_MAGIC)')
s=s.replace('bytes.len()==BATCH_RESULT_BYTES', 'bytes.len()==Layout::<ROWS>::BATCH_RESULT_BYTES')
s=s.replace('exact V3 request extent required','exact versioned request extent required').replace('V3 bytes differ','versioned bytes differ').replace('exact eight-record extent required','exact capacity result extent required')
pos=s.index('#[derive(Clone, Debug)]')
s=s[:pos]+'''/// Compile-time capacities. Public operations reject unsupported ROWS before writing.
pub struct Layout<const ROWS:usize>;
impl<const ROWS:usize> Layout<ROWS> {
    pub const VERSION:u32=if ROWS==8 {3} else {4};
    pub const MAGIC:u32=if ROWS==8 {0x33444d52} else {0x34444d52};
    pub const RESULT_MAGIC:u32=if ROWS==8 {0x33524d52} else {0x34524d52};
    pub const TOKENS_OFFSET:usize=HEADER_BYTES+ROWS*ROW_BYTES;
    pub const REQUEST_BYTES:usize=Self::TOKENS_OFFSET+1024*4;
    pub const BATCH_RESULT_BYTES:usize=ROWS*RESULT_BYTES;
    pub const OUTPUT_OFFSET:usize=Self::BATCH_RESULT_BYTES;
    pub const STAGING_BYTES:usize=2*Self::BATCH_RESULT_BYTES;
}
'''+s[pos:]
f.write_text(s)
(p/'variable_wire16.rs').write_text('''//! V4 sixteen-row contract. This does not enable sixteen-row serving by itself.
pub use super::variable_wire::{Row, RowResult, encode_into, validate, validate_packet, validate_result, validate_batch_result, HEADER_BYTES, ROW_BYTES, RESULT_BYTES};
pub type Expectation=super::variable_wire::Expectation<16>;
pub type Layout=super::variable_wire::Layout<16>;
pub const MAGIC:u32=Layout::MAGIC;
pub const RESULT_MAGIC:u32=Layout::RESULT_MAGIC;
pub const VERSION:u32=Layout::VERSION;
pub const TOKENS_OFFSET:usize=Layout::TOKENS_OFFSET;
pub const REQUEST_BYTES:usize=Layout::REQUEST_BYTES;
pub const BATCH_RESULT_BYTES:usize=Layout::BATCH_RESULT_BYTES;
pub const OUTPUT_OFFSET:usize=Layout::OUTPUT_OFFSET;
pub const STAGING_BYTES:usize=Layout::STAGING_BYTES;
''')
m=p/'mod.rs';s=m.read_text().replace('pub mod variable_wire;','pub mod variable_wire;\npub mod variable_wire16;');m.write_text(s)
