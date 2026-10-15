//! Reject model-identity dispatch, while permitting explicit numerical fixtures.
//! Operator/profile names are not model identity. Parse Rust so comments and
//! calibration labels cannot accidentally replace the architectural guarantee.
use syn::{visit::Visit, ExprField, ExprMethodCall, ExprPath, LitStr, Macro, Member};
const HOT_PATHS: [(&str,&str);3] = [
 ("llama/forward.rs",include_str!("../src/llama/forward.rs")),
 ("llama/decode.rs",include_str!("../src/llama/decode.rs")),
 ("llama/generation.rs",include_str!("../src/llama/generation.rs")),
];
fn identity_name(name:&str)->bool {
 matches!(name.to_ascii_lowercase().as_str(),"model_name"|"model_id"|"source_architecture"|"source_model"|"family"|"architecture"|"modelfamily"|"qwen"|"qwen2"|"qwen3"|"qwen2forcausallm"|"qwen3forcausallm") || name.to_ascii_lowercase().starts_with("is_qwen")
}
#[derive(Default)] struct IdentityReferences { found:Vec<String> }
impl<'a> Visit<'a> for IdentityReferences {
 fn visit_expr_method_call(&mut self, node:&'a ExprMethodCall) {
  if identity_name(&node.method.to_string()) {self.found.push(format!("method {}",node.method));}
  syn::visit::visit_expr_method_call(self,node);
 }
 fn visit_expr_field(&mut self,node:&'a ExprField) {
  if let Member::Named(name)=&node.member {if identity_name(&name.to_string()) {self.found.push(format!("field {name}"));}}
  syn::visit::visit_expr_field(self,node);
 }
 fn visit_expr_path(&mut self,node:&'a ExprPath) {
  for part in &node.path.segments {if identity_name(&part.ident.to_string()) {self.found.push(format!("value {}",part.ident));}}
  syn::visit::visit_expr_path(self,node);
 }
 fn visit_path(&mut self,node:&'a syn::Path) {
  for part in &node.segments {if matches!(part.ident.to_string().as_str(),"ModelFamily"|"Qwen2"|"Qwen3") {self.found.push(format!("type/family {}",part.ident));}}
  syn::visit::visit_path(self,node);
 }
 fn visit_lit_str(&mut self,node:&'a LitStr) {
  if identity_name(&node.value()) {self.found.push(format!("identity literal {:?}",node.value()));}
 }
 fn visit_macro(&mut self,node:&'a Macro) {
  // syn does not visit arbitrary macro input as Rust expressions. Fail closed
  // on the same identity APIs/variants inside matches!, assert! and other input.
  let rendered=node.tokens.to_string().to_ascii_lowercase();let compact:String=rendered.chars().filter(|c|!c.is_whitespace()).collect();
  for token in ["modelfamily","model_name","model_id","source_architecture","source_model",".family(",".architecture(","is_qwen(","\"qwen\"","\"qwen2\"","\"qwen3\"","qwen2forcausallm","qwen3forcausallm"] {
   if compact.contains(token) {self.found.push(format!("macro identity reference {token}"));}
  }
  syn::visit::visit_macro(self,node);
 }
}
fn references(source:&str)->Vec<String> {
 let parsed=syn::parse_file(source).expect("hot-path source must parse as Rust");let mut visitor=IdentityReferences::default();visitor.visit_file(&parsed);visitor.found
}
#[test] fn hot_execution_paths_do_not_dispatch_on_qwen_identity() {
 for (path,source) in HOT_PATHS {let violations=references(source);assert!(violations.is_empty(),"{path} contains model identity references: {violations:?}");}
}
#[test] fn guard_rejects_real_identity_dispatch_including_macro_conditions() {
 for source in [
  "fn dispatch(m: &Model) { if m.family() == ModelFamily::Qwen3 { run(); } }",
  "fn dispatch(m: &Model) { if m.source_architecture() == \"Qwen2ForCausalLM\" { run(); } }",
  "fn dispatch(m: &Model) { if matches!(m . family (), ModelFamily :: Qwen2) { run(); } }",
  "fn dispatch(m: &Model) { if m.model_name.contains(\"Qwen\") { run(); } }",
  "fn dispatch(m: &Model) { if is_qwen(m) { run(); } }",
 ] {assert!(!references(source).is_empty(),"identity mutation escaped guard: {source}");}
}
#[test] fn guard_accepts_explicit_fixture_profiles_without_model_identity() {
 let source="// Qwen fixture names describe arithmetic, not model metadata.\nfn dispatch(profile: Profile, params: Params) { if profile.is_hf_eager_qwen_p2048_cache_on_probe() { hugging_face_qwen_p2051_rms_norm(params); } let label = \"hf-eager-qwen-p2048-probe-v1\"; }";
 assert!(references(source).is_empty());
}
