//! Bounded immutable full-page cache. Scheduler completion supplies quiescence.
use super::*;
use riley_runtime::paged_kv::{BlockId,KvIdentity,PrefixDescriptor,PrefixExport};

struct Entry { tokens:Vec<u32>, page_ids:Vec<u32>, export:PrefixExport, tag:RequestId }
#[derive(Clone,Default)]
struct CachePage { block:Option<BlockId>, leases:usize }
pub(super) struct PrefixCache {
    identity:KvIdentity, entries:VecDeque<Entry>, max_entries:usize, max_pages:usize,
    pages:usize, lease_pages:usize, residency:Vec<CachePage>, hits:u64, reused_tokens:u64,
}
impl PrefixCache {
    fn evict(&mut self,pool:&mut KvBlockPool)->SchedulerResult<()> {
        if let Some(entry)=self.entries.front_mut() {
            pool.release_prefix(&mut entry.export)?;
            // Release succeeds before changing accounting: failed releases retain ownership.
            for &id in &entry.page_ids {
                let page=&mut self.residency[id as usize];
                page.leases-=1;
                if page.leases==0 {page.block=None;self.pages-=1;}
            }
            self.lease_pages-=entry.page_ids.len();
            self.entries.pop_front();
        }
        Ok(())
    }
    pub(super) fn clear(&mut self,pool:&mut KvBlockPool)->SchedulerResult<()> {
        while !self.entries.is_empty(){self.evict(pool)?;} Ok(())
    }
    pub(super) fn reserve_room(&mut self,pool:&mut KvBlockPool,promised:usize)->SchedulerResult<()> {
        let spare=pool.layout().physical_block_count().saturating_sub(promised);
        while self.pages>spare {self.evict(pool)?;} Ok(())
    }
    pub(super) fn import(&mut self,pool:&mut KvBlockPool,sequence:&mut SequenceState,prompt:&[u32])->SchedulerResult<usize> {
        let found=self.entries.iter().enumerate().map(|(index,e)| {
            let common=e.tokens.iter().zip(prompt).take_while(|(a,b)|a==b).count();
            (index,common.min(prompt.len().saturating_sub(1))/KV_BLOCK_SIZE*KV_BLOCK_SIZE)
        }).filter(|(_,count)|*count!=0).max_by_key(|(_,count)|*count);
        let Some((index,count))=found else{return Ok(0)};
        let entry=&self.entries[index];
        sequence.import_prefix_range(pool,&entry.export,&entry.tokens,count)?;
        let entry=self.entries.remove(index).expect("selected cache entry");self.entries.push_back(entry);
        self.hits=self.hits.saturating_add(1);self.reused_tokens=self.reused_tokens.saturating_add(count as u64);
        Ok(count)
    }
    pub(super) fn publish(&mut self,pool:&mut KvBlockPool,source:&SequenceState,prompt:&[u32],promised:usize,next_id:&mut u64)->SchedulerResult<()> {
        // Keep at least one prompt token for logits and never lease a write tail.
        let count=prompt.len().saturating_sub(1)/KV_BLOCK_SIZE*KV_BLOCK_SIZE;
        let pages=count/KV_BLOCK_SIZE;
        let limit=self.max_pages.min(pool.layout().physical_block_count().saturating_sub(promised));
        if count==0 || pages>limit {return Ok(())}
        if let Some(index)=self.entries.iter().position(|e|e.tokens==prompt[..count]) {
            let entry=self.entries.remove(index).expect("existing cache entry");self.entries.push_back(entry);return Ok(())
        }
        let tokens=copy_tokens(&prompt[..count],"prefix cache tokens")?;
        let descriptor=PrefixDescriptor::new(self.identity.clone(),0,&tokens)?;
        let following=next_id.checked_add(1).ok_or(SchedulerError::IdentifierExhausted{kind:"cache owner"})?;
        let tag=RequestId::new(*next_id).ok_or(SchedulerError::IdentifierExhausted{kind:"cache owner"})?;
        let table=source.block_table()?;
        let page_ids=copy_tokens(&table.physical_block_ids()[..pages],"prefix cache page indices")?;
        // Source ownership keeps these pages alive across eviction. Recompute after
        // each eviction because a page can lose its last cache lease.
        loop {
            let added=page_ids.iter().filter(|&&id|self.residency[id as usize].leases==0).count();
            if self.entries.len()<self.max_entries && self.pages<=limit-added {break;}
            self.evict(pool)?;
        }
        let export=pool.export_prefix(source,descriptor)?;
        for &block in export.blocks() {
            let page=&mut self.residency[block.physical_index() as usize];
            if page.leases==0 {page.block=Some(block);self.pages+=1;}
            else {debug_assert_eq!(page.block,Some(block));}
            page.leases+=1;
        }
        *next_id=following;self.lease_pages+=pages;
        self.entries.push_back(Entry{tokens,page_ids,export,tag});Ok(())
    }
    pub(super) fn owner_entries(&self)->usize {self.lease_pages}
    pub(super) fn append_owners(&self,owners:&mut Vec<(u32,RequestId)>) {
        for entry in &self.entries {owners.extend(entry.export.blocks().iter().map(|block|(block.physical_index(),entry.tag)));}
    }
}

impl Scheduler {
    /// Enables a bounded full-page cache before the first request. The execution
    /// adapter must supply identity from the actual retained shared-prefix model.
    /// Reuse always leaves a nonempty prefill suffix to obtain fresh logits.
    pub fn enable_prefix_cache(&mut self,identity:KvIdentity,max_entries:usize,max_pages:usize)->SchedulerResult<()> {
        if self.prefix_cache.is_some() || self.next_request_id!=1 || self.inflight.is_some()
            || self.execution_shape_policy!=ExecutionShapePolicy::MixedPrefillDecode32
            || identity.layout!=self.pool.layout().into() || max_entries==0 || max_pages==0
            || max_entries>max_pages || max_pages>self.pool.layout().physical_block_count() {
            return Err(SchedulerError::InvalidConfiguration{field:"prefix cache",reason:"requires unused mixed32 scheduler and bounded matching identity"});
        }
        let _=PrefixDescriptor::new(identity.clone(),0,&[0])?;
        let mut entries=VecDeque::new();entries.try_reserve_exact(max_entries).map_err(|_|SchedulerError::HostAllocation{resource:"prefix cache entries",requested_elements:max_entries})?;
        let count=self.pool.layout().physical_block_count();
        let mut residency=Vec::new();
        residency.try_reserve_exact(count).map_err(|_|SchedulerError::HostAllocation{resource:"prefix cache residency",requested_elements:count})?;
        residency.resize(count,CachePage::default());
        self.prefix_cache=Some(PrefixCache{identity,entries,max_entries,max_pages,pages:0,lease_pages:0,residency,hits:0,reused_tokens:0});Ok(())
    }
    /// Cache entries, unique retained physical pages, hits and reused tokens.
    pub fn prefix_cache_stats(&self)->(usize,usize,u64,u64) {
        self.prefix_cache.as_ref().map(|c|(c.entries.len(),c.pages,c.hits,c.reused_tokens)).unwrap_or_default()
    }
    /// Allocated entry, token and immutable block-handle payload capacity.
    /// Allocator bookkeeping and the inline scheduler fields are excluded.
    pub fn prefix_cache_host_bytes(&self)->usize {
        self.prefix_cache.as_ref().map_or(0,|cache|cache.entries.capacity()*std::mem::size_of::<Entry>()+
            cache.residency.capacity()*std::mem::size_of::<CachePage>()+
            cache.entries.iter().map(|entry|(entry.tokens.capacity()+entry.page_ids.capacity())*std::mem::size_of::<u32>()+
                entry.export.blocks().len()*std::mem::size_of::<riley_runtime::paged_kv::BlockId>()).sum::<usize>())
    }
    pub(super) fn reserve_cache_room(&mut self,requested:usize)->SchedulerResult<()> {
        let total=self.promised_kv_blocks.checked_add(requested).ok_or(SchedulerError::ArithmeticOverflow{field:"cache admission budget"})?;
        if let Some(cache)=self.prefix_cache.as_mut(){cache.reserve_room(&mut self.pool,total)?;}Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scheduler()->Scheduler {
        let layout=KvLayout::checked(30,6,3,64).unwrap();
        let mut s=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:4,max_waiting_prompt_tokens:256,max_active_sequences:2,
            max_sequence_tokens:64,iteration_token_budget:64,max_prefill_chunk_tokens:64,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:6,metrics_window_samples:4,
        },layout,ExecutionShapePolicy::MixedPrefillDecode32).unwrap();
        s.enable_prefix_cache(KvIdentity{model_revision:[1;32],numerical_profile:[2;32],
            position_encoding:[3;32],partition:[4;32],layout:layout.into()},1,2).unwrap();s
    }
    fn complete(s:&mut Scheduler,plan:&IterationPlan,now:u64) {
        let outputs=plan.output_slots().iter().map(|&slot|IterationOutput::new(slot,7,false)).collect();
        let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
        assert!(s.complete_iteration(&result,now).unwrap().settlement_failures().is_empty());
    }
    fn warm(s:&mut Scheduler) {
        s.submit(RequestDescriptor::new(vec![17;33],1),0).unwrap();
        let p=s.plan_iteration(1).unwrap().into_parts().0.unwrap();complete(s,&p,2);
        assert_eq!(s.prefix_cache_stats(),(1,2,0,0));assert_eq!(s.pool_stats().allocated_block_count(),2);
    }
    #[test]
    fn shared_prefix_capacity_counts_unique_pages_and_keeps_all_lease_owners() {
        let layout=KvLayout::checked(30,16,3,64).unwrap();
        let mut s=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:4,max_waiting_prompt_tokens:256,max_active_sequences:2,
            max_sequence_tokens:64,iteration_token_budget:64,max_prefill_chunk_tokens:64,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:16,metrics_window_samples:4,
        },layout,ExecutionShapePolicy::MixedPrefillDecode32).unwrap();
        s.enable_prefix_cache(KvIdentity{model_revision:[1;32],numerical_profile:[2;32],position_encoding:[3;32],partition:[4;32],layout:layout.into()},4,6).unwrap();
        for i in 0..3 {
            let mut prompt=vec![17;49];prompt[32]=18+i;
            s.submit(RequestDescriptor::new(prompt,1),i as u64*3).unwrap();
            let plan=s.plan_iteration(i as u64*3+1).unwrap().into_parts().0.unwrap();complete(&mut s,&plan,i as u64*3+2);
        }
        let stats=s.prefix_cache_stats();let physical=s.pool_stats().allocated_block_count();
        println!("cache-capacity entries={} page_charge={} allocated_physical={} hits={} reused_tokens={}",stats.0,stats.1,physical,stats.2,stats.3);
        assert_eq!((stats.0,stats.1,physical),(3,5,5));
        let mut prompt=vec![17;49];prompt[32]=18;
        let id=s.submit(RequestDescriptor::new(prompt,1),9).unwrap().request_id();
        assert_eq!(s.prefix_cache_stats().3,112); // two partial hits plus all 48 cached tokens
        s.cancel(id,9).unwrap();
        let cache=s.prefix_cache.as_ref().unwrap();
        assert_eq!(cache.owner_entries(),9);
        let mut owners=Vec::new();cache.append_owners(&mut owners);assert_eq!(owners.len(),9);
        let cache=s.prefix_cache.as_mut().unwrap();cache.evict(&mut s.pool).unwrap();
        assert_eq!((cache.pages,cache.lease_pages),(4,6));
        assert_eq!(s.pool_stats().allocated_block_count(),4);
        s.shutdown(10).unwrap();assert_eq!(s.pool_stats().allocated_block_count(),0);
        let cache=s.prefix_cache.as_ref().unwrap();assert_eq!((cache.pages,cache.lease_pages),(0,0));
        assert!(cache.residency.iter().all(|p|p.leases==0 && p.block.is_none()));
        s.close(11,None).unwrap();
    }
    #[test]
    fn cleared_residency_accepts_reallocated_page_generations() {
        let mut s=scheduler();warm(&mut s);
        let old=s.prefix_cache.as_ref().unwrap().entries[0].export.blocks().to_vec();
        s.prefix_cache.as_mut().unwrap().clear(&mut s.pool).unwrap();
        s.submit(RequestDescriptor::new(vec![18;33],1),3).unwrap();
        let plan=s.plan_iteration(4).unwrap().into_parts().0.unwrap();complete(&mut s,&plan,5);
        let cache=s.prefix_cache.as_ref().unwrap();let new=cache.entries[0].export.blocks();
        let reused:Vec<_>=new.iter().filter_map(|b|old.iter().find(|a|a.physical_index()==b.physical_index()).map(|a|(a,b))).collect();
        assert!(!reused.is_empty());
        for (a,b) in reused {assert_ne!(a.generation(),b.generation());}
        assert_eq!((cache.pages,cache.lease_pages),(2,2));
        for &b in new {assert_eq!(cache.residency[b.physical_index() as usize].block,Some(b));}
        s.close(6,None).unwrap();
    }
    #[test]
    fn automatic_cache_hit_retries_suffix_and_authorizes_cache_only_owner() {
        let mut s=scheduler();warm(&mut s);
        let id=s.submit(RequestDescriptor::new(vec![17;33],1),3).unwrap().request_id();
        assert_eq!(s.prefix_cache_stats(),(1,2,1,32));
        for (now,abort) in [(4,true),(6,false)] {
            let p=s.plan_iteration(now).unwrap().into_parts().0.unwrap();
            assert_eq!(p.prefill_items()[0].input_tokens(),&[17]);
            assert_eq!(p.prefill_items()[0].request_id(),id);
            let a=s.authorize_execution(&p).unwrap();
            assert_eq!(a.block_owners().len(),5); // 3 request pages + 2 cache leases
            let owner=crate::authority::VariableOwnerGeometry{generation:1,last_accepted_replay:0,catalog_digest:[9;32],
                physical_block_count:6,max_active_rows:32,context_tokens:64,packed_prefill:true,mixed_execution:true,shared_prefixes:true};
            let e=a.variable_descriptor_expectation_rows::<32>(&owner,1,&[1],crate::descriptor::ResultMode::FullLogits).unwrap();
            assert_eq!(e.rows[0].progress.committed_tokens,32);drop(a);
            if abort {s.abort_iteration(p.iteration_id(),ExecutionAbort::NotDispatched,now+1).unwrap();}
            else {complete(&mut s,&p,now+1);}
        }
        assert_eq!(s.pool_stats().allocated_block_count(),2);
        s.shutdown(8).unwrap();assert_eq!(s.pool_stats().allocated_block_count(),0);s.close(9,None).unwrap();
    }
    #[test]
    fn admission_pressure_evicts_cache_without_releasing_live_consumer_pages() {
        let mut s=scheduler();warm(&mut s);
        let first=s.submit(RequestDescriptor::new(vec![17;33],1),3).unwrap().request_id();
        let second=s.submit(RequestDescriptor::new(vec![18;33],1),3).unwrap().request_id();
        assert_eq!(s.prefix_cache_stats().0,0);
        assert_eq!(s.pool_stats().allocated_block_count(),2); // still owned by first
        let p=s.plan_iteration(4).unwrap().into_parts().0.unwrap();
        let a=s.authorize_execution(&p).unwrap();assert!(a.block_owners().len()<=6);drop(a);
        s.abort_iteration(p.iteration_id(),ExecutionAbort::NotDispatched,5).unwrap();
        s.cancel(first,6).unwrap();s.cancel(second,6).unwrap();
        assert_eq!(s.pool_stats().allocated_block_count(),0);s.close(7,None).unwrap();
    }
    #[test]
    fn quiesced_unknown_mutation_invalidates_cached_prefixes() {
        let mut s=scheduler();warm(&mut s);
        s.submit(RequestDescriptor::new(vec![17;33],1),3).unwrap();
        let p=s.plan_iteration(4).unwrap().into_parts().0.unwrap();
        s.abort_iteration(p.iteration_id(),ExecutionAbort::DeviceQuiescedMutationUnknown,5).unwrap();
        assert_eq!(s.prefix_cache_stats().0,0);assert_eq!(s.pool_stats().allocated_block_count(),0);
        s.close(6,None).unwrap();
    }
    #[test]
    fn shorter_prompt_reuses_only_its_matching_full_pages() {
        let mut s=scheduler();warm(&mut s);
        let mut prompt=vec![17;25];prompt[20]=18;
        s.submit(RequestDescriptor::new(prompt.clone(),1),3).unwrap();
        assert_eq!(s.prefix_cache_stats(),(1,2,1,16));
        let p=s.plan_iteration(4).unwrap().into_parts().0.unwrap();
        assert_eq!(p.prefill_items()[0].input_tokens(),&prompt[16..]);
        complete(&mut s,&p,5);s.close(6,None).unwrap();
    }
    #[test]
    fn different_prefix_misses_and_lru_replacement_reclaims_old_pages() {
        let mut s=scheduler();warm(&mut s);
        s.submit(RequestDescriptor::new(vec![18;33],1),3).unwrap();
        let p=s.plan_iteration(4).unwrap().into_parts().0.unwrap();assert_eq!(p.total_tokens(),33);complete(&mut s,&p,5);
        assert_eq!(s.prefix_cache_stats(),(1,2,0,0));assert_eq!(s.pool_stats().allocated_block_count(),2);
        s.submit(RequestDescriptor::new(vec![17;33],1),6).unwrap();
        let p=s.plan_iteration(7).unwrap().into_parts().0.unwrap();assert_eq!(p.total_tokens(),33);
        s.abort_iteration(p.iteration_id(),ExecutionAbort::NotDispatched,8).unwrap();s.close(9,None).unwrap();
    }
}
