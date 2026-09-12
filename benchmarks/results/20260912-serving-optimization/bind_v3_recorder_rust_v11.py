from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-cuda/src/ffi.rs';s=p.read_text();s+='''
unsafe extern "C" {
    fn riley_cuda_graph_resources_record_v3_prefill(owner:*mut RawGraphResources,devices:*const *mut RawDeviceBuffer,weights:*const *mut RawDeviceBuffer,weight_count:u64,head:*mut RawGemmPlan,staging:*mut RawPinnedHostBuffer,capacity:u32,physical:u32,error:*mut ErrorInfo)->i32;
}
impl GraphResourcesHandle {
    pub(super) fn record_v3_prefill(&mut self,devices:&[&DeviceBufferHandle],workspace:Option<&DeviceBufferHandle>,weights:&[&DeviceBufferHandle],head:&GemmPlanHandle,staging:&PinnedHostBufferHandle,capacity:u32,physical:u32)->CudaResult<()> {
        if devices.len()!=22||weights.len()!=273 {return Err(CudaError::invalid_argument("record V3 prefill","parent count mismatch"));}
        let mut raw=[ptr::null_mut();23];for (i,d) in devices.iter().enumerate(){raw[i]=d.as_ptr();}raw[22]=workspace.map_or(ptr::null_mut(),DeviceBufferHandle::as_ptr);
        let weights:Vec<_>=weights.iter().map(|w|w.as_ptr()).collect();let mut error=ErrorInfo::new();
        // SAFETY: fixed descriptor sizes checked above; retained parent handles outlive capture.
        let status=unsafe{riley_cuda_graph_resources_record_v3_prefill(self.pointer.map_or(ptr::null_mut(),NonNull::as_ptr),raw.as_ptr(),weights.as_ptr(),273,head.as_ptr(),staging.as_ptr(),capacity,physical,&mut error)};
        status_result(status,"record V3 prefill",&error)
    }
}
''';p.write_text(s)
p=r/'crates/riley-cuda/src/graph_resources.rs';s=p.read_text();s+='''
impl BorrowedGraphResourceReservation<'_> {
    /// Records V3 prefill and its canonical head under this retained ledger.
    /// This does not authorize scheduler publication of the returned result.
    pub fn record_v3_prefill(&mut self,devices:&[usize;22],workspace:Option<usize>,weights:&[usize],head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V3 prefill","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v3_prefill(&devices,workspace,&weights,head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V3 prefill"))}
    }
}
''';p.write_text(s)
