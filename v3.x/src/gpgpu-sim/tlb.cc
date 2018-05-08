// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "stat-tool.h"
#include <assert.h>
#include "tlb.h"
#include "dram.h"
#include "gpu-sim.h"
#include <stdlib.h>
#include "mem_latency_stat.h"
#include "l2cache.h"
#include <map>

#define COALESCE_DEBUG 0
#define COALESCE_DEBUG_SHORT 0
#define PWCACHE_DEBUG 0
#define INVALIDATION_DEBUG 0
#define MERGE_DEBUG 0
#define HUGEPAGE_DEBUG 0
#define APPID_SHIFT_DEBUG 0


extern int gpu_sms;

tlb_tag_array::tlb_tag_array(const memory_config * config, shader_core_stats * stat,
    mmu * page_manager, tlb_tag_array * shared_tlb, int core_id) {

  ready_cycle = 0;

  m_config = config;
  m_stat = stat;
  m_access = 0;
  m_miss = 0;
  m_page_manager = page_manager;
  tag_array = new std::list<new_addr_type>;
  large_tag_array = new std::list<new_addr_type>;
  tlb_level = 0;
  m_shared_tlb = shared_tlb;
  m_mem_stats = shared_tlb->get_mem_stat();
  if (shared_tlb != NULL)
    printf("Assigned Shared TLB at %x\n", m_shared_tlb);
  isL2TLB = false;
  root = page_manager->get_page_table_root();
  //root = new page_table(m_config, m_stat, 0);

  m_core_id = core_id;

  total_tokens = (float) m_config->tlb_bypass_initial_tokens / 100;
  for (std::map<appid_t, App*>::iterator apps = App::get_apps().begin();
      apps != App::get_apps().end(); apps++) {
    App* app = apps->second;

    app->tokens = (float) m_config->tlb_bypass_initial_tokens / 100;

    for (int j = 0; j < 4000; j++) {
      app->wid_tokens[j] = true; //In the beginning, everyone has a token
      app->wid_epoch_hit[j] = 0;
      app->wid_epoch_miss[j] = 0;
      app->wid_epoch_previous_miss_rate[j] = 0;
      app->wid_epoch_previous2_miss_rate[j] = 0;
    }
  }
  previous_bypass_enabled = false;

  bypass_done_queue = new std::list<new_addr_type>();
  bypass_done_REG = new std::list<new_addr_type>();
  bypass_done_REG_time = new std::list<unsigned long long>();

  prev_stride = 0;
  stride = 0;

  for (int i = 0; i < 256; i++) {
    prefetch_queue[i] = new std::list<new_addr_type>();
    prefetch_buffer[i] = new std::list<new_addr_type>();
    prefetch_queue_time[i] = new std::list<unsigned long long>();
    if (i > m_config->tlb_prefetch_set)
      break;
  }

  //bypass_done_queue = new std::set<new_addr_type>();
  tlb_entry_sharer = new std::map<new_addr_type, std::set<unsigned>>();

  replacement_hashed_map = new bool[m_config->tlb_replacement_hash_size];
  for (int i = 0; i < m_config->tlb_replacement_hash_size; i++)
    replacement_hashed_map[i] = false;

  tlb_occupancy_last_updated = 0;
  tlb_demotion_last_checked = 0;
  hotness_track = NULL;
  warpID_track = NULL;
  locality_track = NULL;
}



tlb_tag_array::tlb_tag_array(const memory_config * config, shader_core_stats * stat,
    mmu * page_manager, bool isL2TLB, memory_stats_t * mem_stat,
    memory_partition_unit ** memory_partition) {

  ready_cycle = 0;

  m_memory_partition = memory_partition;

  for (int i = 0; i < 200; i++)
    L1_ready_cycle[i] = 0;
  L2_ready_cycle = 0;
  bypass_cache_ready_cycle = 0;

  m_config = config;
  m_stat = stat;
  m_access = 0;
  m_miss = 0;
  m_ways = m_config->l2_tlb_ways;
  m_ways_large = m_config->l2_tlb_ways_large;
  m_entries = m_config->l2_tlb_ways == 0 ? 0 : m_config->l2_tlb_entries / m_config->l2_tlb_ways;
  m_entries_large = m_config->l2_tlb_entries / m_config->l2_tlb_ways;
  m_page_manager = page_manager;
  l2_tag_array = new std::list<new_addr_type>*[m_entries];
  large_l2_tag_array = new std::list<new_addr_type>*[m_entries_large];
  for (int i = 0; i < m_entries; i++) {
    l2_tag_array[i] = new std::list<new_addr_type>;
  }
  for (int i = 0; i < m_entries_large; i++) {
    large_l2_tag_array[i] = new std::list<new_addr_type>;
  }

  l1_tlb = new tlb_tag_array*[gpu_sms];

  tlb_level = 0;
  isL2TLB = true;
  miss_queue = new std::set<new_addr_type>();
  bypass_dir = true;
  bypass_done_queue = new std::list<new_addr_type>();
  large_bypass_done_queue = new std::list<new_addr_type>();
  bypass_done_REG = new std::list<new_addr_type>();
  bypass_done_REG_time = new std::list<unsigned long long>();
  //bypass_done_queue = new std::set<new_addr_type>();

  total_tokens = (float) m_config->tlb_bypass_initial_tokens / 100;
  for (std::map<appid_t, App*>::iterator apps = App::get_apps().begin();
      apps != App::get_apps().end(); apps++) {
    App* app = apps->second;

    app->tokens = (float) m_config->tlb_bypass_initial_tokens / 100;

    for (int j = 0; j < 4000; j++) {
      app->wid_tokens[j] = true; //In the beginning, everyone has a token
      app->wid_epoch_hit[j] = 0;
      app->wid_epoch_miss[j] = 0;
      app->wid_epoch_previous_miss_rate[j] = 0;
      app->wid_epoch_previous2_miss_rate[j] = 0;
    }
  }
  previous_bypass_enabled = false;

  pw_cache_entries = m_config->tlb_pw_cache_entries;
  pw_cache_ways = m_config->tlb_pw_cache_ways;
  pw_cache = new std::list<new_addr_type>*[pw_cache_entries];
  for (int i = 0; i < pw_cache_entries; i++)
    pw_cache[i] = new std::list<new_addr_type>;

  pw_cache_lat_queue = new std::list<mem_fetch*>;
  pw_cache_lat_time = new std::list<unsigned long long>;

  m_mem_stats = mem_stat;

  tlb_entry_sharer = new std::map<new_addr_type, std::set<unsigned>>();
  replacement_hashed_map = new bool[m_config->tlb_replacement_hash_size];
  for (int i = 0; i < m_config->tlb_replacement_hash_size; i++)
    replacement_hashed_map[i] = false;

  if (m_config->tlb_fixed_latency_enabled) {
    static_queue = new std::list<mem_fetch*>();
    static_queue_time = new std::list<unsigned long long>();
  }

  hotness_track = new std::map<new_addr_type, unsigned long long>();
  warpID_track = new std::map<new_addr_type, std::set<int>*>();
  locality_track = new std::map<new_addr_type, std::set<new_addr_type>*>();

  tlb_occupancy_last_updated = 0;
  tlb_demotion_last_checked = 0;

  //promoted_pages = new std::set<new_addr_type>();

  for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
    printf("Initializing promoted page list inside the shared TLB for appID = %d\n", App::get_app_id(i));
    promoted_pages[App::get_app_id(i)] = new std::set<new_addr_type>();
  }



  last_shifted_page_stat = 0;

  m_page_manager->set_L2_tlb(this);

}

tlb_tag_array::tlb_tag_array(tlb_tag_array * tlb, unsigned level) {

  ready_cycle = 0;
  m_config = tlb->m_config;
  m_stat = tlb->m_stat;
  m_access = 0;
  m_miss = 0;
  m_page_manager = tlb->m_page_manager;
  tag_array = new std::list<new_addr_type>;
  large_tag_array = new std::list<new_addr_type>;
  tlb_level = level + 1;
  if (tlb_level < m_config->max_tlb_cache_depth)
    m_next_tlb = new tlb_tag_array(this, tlb_level);

  total_tokens = (float) m_config->tlb_bypass_initial_tokens / 100;
  for (std::map<appid_t, App*>::iterator apps = App::get_apps().begin();
      apps != App::get_apps().end(); apps++) {
    App* app = apps->second;

    app->tokens = (float) m_config->tlb_bypass_initial_tokens / 100;

    for (int j = 0; j < 4000; j++) {
      app->wid_tokens[j] = true; //In the beginning, everyone has a token
      app->wid_epoch_hit[j] = 0;
      app->wid_epoch_miss[j] = 0;
      app->wid_epoch_previous_miss_rate[j] = 0;
      app->wid_epoch_previous2_miss_rate[j] = 0;
    }
  }
  previous_bypass_enabled = false;
  large_bypass_done_queue = new std::list<new_addr_type>();
  bypass_done_REG = new std::list<new_addr_type>();
  bypass_done_REG_time = new std::list<unsigned long long>();
  //bypass_done_queue = new std::set<new_addr_type>();
  tlb_entry_sharer = new std::map<new_addr_type, std::set<unsigned>>();
  replacement_hashed_map = new bool[m_config->tlb_replacement_hash_size];
  for (int i = 0; i < m_config->tlb_replacement_hash_size; i++)
    replacement_hashed_map[i] = false;


  tlb_occupancy_last_updated = 0;

  hotness_track = NULL;
  warpID_track = NULL;
  locality_track = NULL;
}

void tlb_tag_array::add_mf_to_done(mem_fetch *mf) {
  remove_list.push_back(mf);
}

// delete everything in the remove_list
void tlb_tag_array::remove_done_mf() {
  while (!remove_list.empty()) {
    delete (mem_fetch *) (remove_list.front());
    remove_list.pop_front();
  }
}


void tlb_tag_array::handle_pending_prefetch(){
    for(int i=0;i<256;i++){
        while(!prefetch_queue_time[i]->empty())
        {
            unsigned long long temp = prefetch_queue_time[i]->front();
            if((temp + 2000) < gpu_sim_cycle + gpu_tot_sim_cycle){
                new_addr_type entry = prefetch_queue[i]->front();
                if(m_config->tlb_prefetch == 1){
                    std::list<new_addr_type>::iterator pref_itr = std::find(prefetch_buffer[i]->begin(), prefetch_buffer[i]->end(), entry);
                    if(pref_itr == prefetch_buffer[i]->end()) //If this is a new prefetch entry
                    {
                        prefetch_buffer[i]->push_back(entry);
                        if(prefetch_buffer[i]->size() > m_config->tlb_prefetch_buffer_size)
                            prefetch_buffer[i]->pop_front();
                    }
                }
                if(m_config->tlb_prefetch == 2){
                    std::list<new_addr_type>::iterator pref_itr = std::find(tag_array->begin(), tag_array->end(), entry);
                    if(pref_itr == tag_array->end()) //If this is a new prefetch entry, add it to the tag array
                    {
                        tag_array->push_back(entry);
                        if(tag_array->size() > m_config->tlb_size)
                            tag_array->pop_front();
                    }
                }
                prefetch_queue[i]->pop_front();
                prefetch_queue_time[i]->pop_front();
            }
            else break;
        }
        if(i>m_config->tlb_prefetch_set) break;
    }
}


// For static TLB latency
void tlb_tag_array::put_mf_to_static_queue(mem_fetch * mf)
{
    if(static_queue == NULL)
    {
        printf("ERROR: Static TLB latency queue is not initiated\n");
    }
    else
    {
        static_queue->push_back(mf);
        static_queue_time->push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
    }
}

// For static latency, remove anything that are done. Called before TLB probe
void tlb_tag_array::clear_static_queue()
{
    mem_fetch * top;
    while(!static_queue->empty())
    {
        if((static_queue_time->front() + m_config->tlb_fixed_latency) < (gpu_sim_cycle + gpu_tot_sim_cycle))
        {
            mem_fetch *mf = static_queue->front();
            //done_tlb_req already handle this
            //mf->get_tlb()->fill(mf->get_addr(),mf);
            //mf->get_tlb()->l2_fill(mf->get_addr(), mf->get_appID(),mf);
            //mf->get_tlb()->remove_miss_queue(mf->get_addr());
            mf->done_tlb_req(mf); 
            static_queue->pop_front();
            static_queue_time->pop_front();
        }
        else
        {
            break;
        }
    }
}

void tlb_tag_array::remove_done_REG()
{
    while(!bypass_done_REG->empty())
    {

        unsigned factor = (m_config->page_transfer_time > 10000)? m_config->page_transfer_time :10000;

        unsigned long long temp = bypass_done_REG_time->front();
        if((temp + factor) < gpu_sim_cycle + gpu_tot_sim_cycle){
            bypass_done_REG->pop_front();
            bypass_done_REG_time->pop_front();
        }
        else break;
    }
}

void tlb_tag_array::remove_pw_cache_lat_queue()
{
    int ports = 0;
    while(!pw_cache_lat_queue->empty())
    {
        unsigned long long temp = pw_cache_lat_time->front();
        if((ports < m_config->pw_cache_num_ports) && ((temp + m_config->pw_cache_latency) < gpu_sim_cycle + gpu_tot_sim_cycle)){
            mem_fetch * mf = pw_cache_lat_queue->front();

            // Remove mf from the list
            pw_cache_lat_time->pop_front();
            pw_cache_lat_queue->pop_front();

            // Finish up the current pw cache hit routine, call the next mf
            mf->pwcache_done = true;
            mf->done_tlb_req(mf); //Note that the actual done_tlb_req is called in mem_fetch, at this point we just need to mark pwcache as done. (Otherwise this we will call parent request on both PW cache hit and PW cache miss requests
            //mf->done_tlb_req(mf->get_parent_tlb_request()); //This is wrong

            ports++;
        }
        else break;
    }
}

new_addr_type tlb_tag_array::process_pwcache_key(mem_fetch * mf)
{
    //unsigned key = (mf->get_addr() / (*(m_config->page_sizes))[m_config->page_sizes->size()-1] )* (*(m_config->page_sizes))[m_config->page_sizes->size()-1]; //Mask the physical page location
    //Then add the associated offset

    unsigned key = mf->get_original_addr(); //get mf->VA

    unsigned bitmask = m_page_manager->get_bitmask(mf->get_tlb_depth_count());
    key = key & bitmask; //Process the offset, bitmask should be increasingly longer as depth increases
    //Then need to shift the bitmask

    unsigned temp_mask = bitmask;
    while((bitmask > 0) &&  ((bitmask & 1) == 0)){
        if(PWCACHE_DEBUG) printf("[original_addr = %x, key = %x, original_bitmask = %x] new value = %x\n", mf->get_original_addr(), key, bitmask, key >> 1);
        key = key >> 1;
        bitmask = bitmask >> 1;
    }

    key = (key << 4)  | mf->get_appID(); //Embedd appID

    if(PWCACHE_DEBUG) printf("Processing pw_cache key, mf depth = %d, bitmask = %x, mf_original_addr = %x, mf tlb addr = %x, processed key = %x\n",mf->get_tlb_depth_count(), temp_mask, mf->get_original_addr(), mf->get_addr(), key);

    return key;
}


void tlb_tag_array::pw_cache_fill(mem_fetch * mf)
{
    unsigned key = process_pwcache_key(mf);
    //unsigned key = mf->get_addr() >> (m_config->page_size); //Old
    unsigned index = (mf->get_addr() >> (m_config->page_size) ) & (pw_cache_entries-1);

     pw_cache[index]->remove(key); // remove the current entry, put this in the front of the queue
     while(pw_cache[index]->size() >= pw_cache_ways)
     //if(pw_cache[index]->size() >= pw_cache_ways)
     {
         pw_cache[index]->pop_back();
     }
     pw_cache[index]->push_front(key); //LRU
}

bool tlb_tag_array::pw_cache_access(mem_fetch *mf)
{
    unsigned key = process_pwcache_key(mf);
    //unsigned key = mf->get_addr() >> (m_config->page_size); //Old

    unsigned index = (mf->get_addr() >> (m_config->page_size) ) & (pw_cache_entries-1);

    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_appID()));

    std::list<new_addr_type>::iterator findIter = std::find(pw_cache[index]->begin(), pw_cache[index]->end(), key);
    if(findIter != pw_cache[index]->end())
    {
        pw_cache[index]->remove(key);
        pw_cache[index]->push_front(key); // insert at MRU
        m_stat->pw_cache_hit++;
        app->pw_cache_hit_app++;
        return true;
    }
    else
    {
//        pw_cache[index]->remove(key);
//        pw_cache[index]->push_front(key); // insert at MRU
        m_stat->pw_cache_miss++;
        app->pw_cache_miss_app++;
        pw_cache_fill(mf);
        return false;
    }
}

//tlb_tag_array::tlb_tag_array( const memory_config  *config,
//                      int core_id,
//                      int type_id, memory_stats_t * stat)
//    : m_config( config )
//{
//    //assert( m_config->m_write_policy == READ_ONLY ); Old assert
//    m_lines = new tlb_entry_t[ config.get_num_TLB_lines()];
//    m_stat = stat;
//    init( core_id, type_id );
//}


// Given an access, get the addess of the localtion of the tlb
new_addr_type tlb_tag_array::get_tlbreq_addr(mem_fetch * mf)
{
    //return root->parse_pa(mf);
    new_addr_type return_addr =  root->parse_pa(mf);
    //new_addr_type return_addr =  root->parse_pa(mf)->addr;
    if(COALESCE_DEBUG) printf("Generating TLBreq for mf = %x, level = %d, return addr = %x\n", mf->get_addr(), mf->get_tlb_depth_count(), return_addr);
    return return_addr;
}


// Check the size of new_addr_type and do we want warpID here as well? More policies
new_addr_type tlb_tag_array::parse_tlb_access(mem_fetch *mf)
{
//    new_addr_type base = m_config->page_base_addr;
    if(m_config->page_mapping_policy == 0) //Dummy test mapping
        return 0xe0000000 | (mf->get_addr() + (mf->get_tlb_depth_count()<<5));
    else
    {
        //new_addr_type assigned_pa = root->parse_pa(mf);
        new_addr_type assigned_pa = root->parse_pa(mf);
        //new_addr_type assigned_pa = root->parse_pa(mf)->addr;
        return assigned_pa;
    }
}


//Do not need this at the moment. Obsolete
// Check the size of new_addr_type and do we want warpID here as well? More policies
new_addr_type tlb_tag_array::assign_pa(new_addr_type original, unsigned tlb_level, unsigned appID)
{
   return original+(tlb_level<<8)+(appID<<20);
}

int tlb_tag_array::remove_miss_queue(new_addr_type addr, int appID)
{
     unsigned key;
     key = ((addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2])  | appID; //Get the correct base page | appID to search in large page list
     if(m_shared_tlb->promoted_pages[appID]->find(key)!=m_shared_tlb->promoted_pages[appID]->end()) //Large page
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2];
     else
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-1];
     int num_erased = m_shared_tlb->miss_queue->erase(key);
     return num_erased;
}

// Right now multi-page-size fill only support baseline and MASK
void tlb_tag_array::fill( new_addr_type addr, mem_fetch * mf)
{

     if(!m_config->tlb_enable) return; //Does not have to fill if we always return TLB HIT(speeding things up)

     measure_sharer_stats();

     if(HUGEPAGE_DEBUG || COALESCE_DEBUG) printf("Filling into L1 tlb for address = %x\n", mf->get_addr());

     //temporary key to search in the superpage list, should be [base_large_pa_addr | appID]
     unsigned key = ((addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
     //unsigned key = ((addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | mf->get_appID();
     bool to_large_tlb = false;

     //First thing first, remove the key from the tag array, if present (should not be)
     if(m_shared_tlb->promoted_pages[mf->get_appID()]->find(key) != m_shared_tlb->promoted_pages[mf->get_appID()]->end()) //Large page
     //if(m_shared_tlb->promoted_pages->find(addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) != m_shared_tlb->promoted_pages->end()) //Large page
     {
         to_large_tlb = true;
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2];
         if(HUGEPAGE_DEBUG || COALESCE_DEBUG) printf("Filling into large page L1 tlb for address = %x, search_key = %x\n", mf->get_addr(), key);
         if(HUGEPAGE_DEBUG) printf("Content of the large L1 TLB (before fill) is {");
         for(std::list<new_addr_type>::iterator findIter = large_tag_array->begin(); findIter != large_tag_array->end();findIter++)
         {
             if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
         }
         if(HUGEPAGE_DEBUG) printf("}\n");
         large_tag_array->remove(key); // remove the current entry, put this in the front of the queue
     }
     else
     {
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-1];

         if(HUGEPAGE_DEBUG || COALESCE_DEBUG) printf("Filling into small page L1 tlb for address = %x, search_key = %x\n", mf->get_addr(), key);
         if(HUGEPAGE_DEBUG) printf("Content of the small L1 TLB (before fill) is {");
         for(std::list<new_addr_type>::iterator findIter = tag_array->begin(); findIter != tag_array->end();findIter++)
         {
             if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
         }
         if(HUGEPAGE_DEBUG) printf("}\n");


    //     unsigned index = addr;
         tag_array->remove(key); // remove the current entry, put this in the front of the queue
    //     printf("Rachat-debug: Filling %x to TLB\n",addr);
     }


     int num_erased = m_shared_tlb->miss_queue->erase(key);
     App* app = App::get_app(mf->get_appID());
     //App* app = App::get_app(App::get_app_id(mf->get_appID()));
     app->concurrent_tracker-= num_erased;

     if(m_config->tlb_prefetch)
     {
         prev_stride = stride;
         stride = key;
         stride_set.insert(stride-prev_stride);
         int prefetch_index = 0;
         if(m_config->tlb_prefetch == 3) prefetch_index = (stride-prev_stride) % m_config->tlb_prefetch_set;
         prefetch_queue[prefetch_index]->push_back(key+ (stride-prev_stride));
         prefetch_queue_time[prefetch_index]->push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
     }


     //This part is needed to emulate the MSHR of the TLB. Because how GPGPU sim keep creating new mf on any RESERVATION_FAIL,
     //we can't track any mem_fetch that has TLB_miss because mem_fetch has not even arrive in the cache yet (if we keep track
     //of these mem_fetch, shader.cc will eventually delete them after RESERVATION_FAIL or TLB is returned.
     if((m_config->tlb_lookup_bypass > 0) && m_shared_tlb->bypass_TLB(mf, mf->get_appID()))
     {
//         bypass_done_queue->insert(key);
         bypass_done_REG->push_back(key);
         bypass_done_REG_time->push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
         if(m_config->tlb_lookup_bypass < 6)
             return; //Bypass the fill into TLB
     }

     std::list<new_addr_type> * active_tag_array = to_large_tlb?large_tag_array:tag_array;

     if(active_tag_array->size() >= (to_large_tlb?m_config->tlb_size_large:m_config->tlb_size))
     {
         if(m_config->tlb_replacement_policy == 0)
         {
            active_tag_array->pop_back();
            //if(to_large_tlb) large_tag_array->pop_back();
            //else tag_array->pop_back(); // LRU policy
         }
         else if(m_config->tlb_replacement_policy == 1) //WID-based #1, probably will suck because it probably will delete new entries
         {
             std::list<new_addr_type>::const_iterator min_itr = active_tag_array->begin();
             unsigned min_count = 999999;
             for(std::list<new_addr_type>::const_iterator itr = active_tag_array->begin(); itr !=active_tag_array->end(); itr++)
             {
                 new_addr_type temp = *itr;
                 if((*tlb_entry_sharer)[temp].size() > 0 && (*tlb_entry_sharer)[temp].size() < min_count)
                 {
                     min_itr = itr;
                     min_count = (*tlb_entry_sharer)[temp].size();
                 }
             }
             new_addr_type removed_entry = *min_itr;
             active_tag_array->remove(removed_entry);
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }
         else if(m_config->tlb_replacement_policy == 2) //WID-based #2, inverse of 1, which should also suck because this time it removes the highest sharer
         {
             std::list<new_addr_type>::const_iterator min_itr = active_tag_array->begin();
             unsigned min_count = 0;
             for(std::list<new_addr_type>::const_iterator itr = active_tag_array->begin(); itr !=active_tag_array->end(); itr++)
             {
                 new_addr_type temp = *itr;
                 if((*tlb_entry_sharer)[temp].size() > 0 && (*tlb_entry_sharer)[temp].size() > min_count)
                 {
                     min_itr = itr;
                     min_count = (*tlb_entry_sharer)[temp].size();
                 }
             }
             new_addr_type removed_entry = *min_itr;
             active_tag_array->remove(removed_entry);
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }
         else if(m_config->tlb_replacement_policy == 3) //WID-based #3, get the most LRU from low and high pool. Always pick the one from low pool
         {
             new_addr_type target_entry_low = *(active_tag_array->begin());
             new_addr_type target_entry_high = *(active_tag_array->begin());
             for(std::list<new_addr_type>::const_iterator itr = active_tag_array->begin(); itr !=active_tag_array->end(); itr++)
             {
                 new_addr_type temp = *itr;
                 if(replacement_hashed_map[temp%m_config->tlb_replacement_hash_size]) //Found on the high list, mark this entry for high_list_LRU (more LRU are in the back)
                 {
                     target_entry_high = temp;
                 }
                 else
                 {
                     target_entry_low = temp;
                 }
             }
             new_addr_type removed_entry = target_entry_low;
             active_tag_array->remove(removed_entry);
             if((*tlb_entry_sharer).size() > m_config->tlb_replacement_high_threshold) replacement_hashed_map[removed_entry%m_config->tlb_replacement_hash_size] = true;
             else replacement_hashed_map[removed_entry%m_config->tlb_replacement_hash_size] = false;
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }
         else
             active_tag_array->pop_back(); // LRU policy by default
     }
     active_tag_array->push_front(key);

     if(HUGEPAGE_DEBUG) printf("Content of the associated L1 TLB after the fill is {");
     for(std::list<new_addr_type>::iterator findIter = active_tag_array->begin(); findIter != active_tag_array->end();findIter++)
     {
         if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
     }
     if(HUGEPAGE_DEBUG) printf("}\n");


//     if(to_large_tlb)
//         large_tag_array->push_front(key);
//     else
//         tag_array->push_front(key);
}

void tlb_tag_array::shift_page_stats()
{
//    int time_shifted = (int)(((gpu_sim_cycle + gpu_tot_sim_cycle) - last_shifted_page_stat) / m_config->page_stat_update_cycle); //Figure out how much each counter should be shifted
//
//    for(std::map<new_addr_type,int>::iterator itr = hotness_track->begin(); itr!=hotness_track->end();itr++)
//        (*itr).second = (*itr).second >> time_shifted;
    for(std::map<new_addr_type,std::set<int>*>::iterator itr = warpID_track->begin(); itr!=warpID_track->end();itr++)
        (*itr).second->clear();
    for(std::map<new_addr_type,std::set<new_addr_type>*>::iterator itr = locality_track->begin(); itr!=locality_track->end();itr++)
        (*itr).second->clear();

}


// demotion routine, called by the mmu
bool tlb_tag_array::demote_page(new_addr_type va, int appID)
{
    unsigned large_page_size;
    if(m_config->page_sizes->size() > 1)//Need to make sure it happens only if the config has multiple page size
        large_page_size = (*(m_config->page_sizes))[m_config->page_sizes->size()-2];
    else
        large_page_size = (*(m_config->page_sizes))[m_config->page_sizes->size()-1];

    //Note, promote_pages has the folloing format: [base_addr | appID]
    new_addr_type base_addr = (va / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the base address
    if(MERGE_DEBUG || COALESCE_DEBUG || COALESCE_DEBUG_SHORT) printf("Demoting page with entry %x (page index is %x) for app = %d from the promoted page list. Key for searching is %x\n", va,base_addr, appID, base_addr | appID);

    int demote_id = m_page_manager->demote_page(m_page_manager->find_physical_page(base_addr ,appID, (*(m_config->page_sizes))[m_config->page_sizes->size()-2]));

    // Invalidate all the entries in existing small page TLBs (Both L1 and L2, but this is called from L2. invalidate entries will automatically invalidate L1 entries as well)
    // Note that the tag use shifted version of the based address
    if(demote_id != 1)
        invalidate_entries(appID, base_addr /  (*(m_config->page_sizes))[m_config->page_sizes->size()-2]);

    promoted_pages[appID]->erase(base_addr); //Note that this move itr to the next address. Use itr++ to prevent the case where c++98 is used instead of c++11 (only c++11 return the iterator value)

    return true;

}


//This is no longer needed with the new allocator. Page stats are now collected through mmu
void tlb_tag_array::update_page_stats(mem_fetch * mf)
{
    unsigned large_page_size;
    if(m_config->page_sizes->size() > 1)//Need to make sure it happens only if the config has multiple page size
        large_page_size = (*(m_config->page_sizes))[m_config->page_sizes->size()-2];
    else
        large_page_size = (*(m_config->page_sizes))[m_config->page_sizes->size()-1];

    new_addr_type key;

//    //shift all page stats if it has not been updated for a while
//    if( (last_shifted_page_stat + m_config->page_stat_update_cycle) < (gpu_sim_cycle + gpu_tot_sim_cycle))
//    {
//        shift_page_stats(); //All saturation counter perform the shifting
//        last_shifted_page_stat = ((gpu_sim_cycle + gpu_tot_sim_cycle) / m_config->page_stat_update_cycle) * m_config->page_stat_update_cycle; //Set the new timer
//    }


//    if(COALESCE_DEBUG) printf("Updating page stats for mf VA = %x, PA = %x, at cycle = %lld\n", mf->get_addr(), m_page_manager->get_pa(mf->get_addr(),mf->get_appID())/large_page_size,gpu_sim_cycle + gpu_tot_sim_cycle);

    //If use VA
    key = mf->get_addr()/large_page_size;
    //If use PA. Assume that this page exist
    //key = m_page_manager->get_pa(mf->get_addr(), mf->get_appID())/large_page_size;

    if(COALESCE_DEBUG) printf("Update page statistics for page %x (va = %x), page size = %d\n", key, mf->get_addr()/large_page_size,large_page_size);

    (*hotness_track)[key] = gpu_sim_cycle + gpu_tot_sim_cycle;
    if((*warpID_track)[key] == NULL)
    {
        (*warpID_track)[key] = new std::set<int>();
    }
    (*warpID_track)[key]->insert(mf->get_wid());
    if((*locality_track)[key] == NULL)
    {
        (*locality_track)[key] = new std::set<new_addr_type>();
    }
    (*locality_track)[key]->insert(mf->get_addr()/(*(m_config->page_sizes))[m_config->page_sizes->size()-1]);


    //Demote page as needed after updating all the page statistics
    if(m_config->enable_page_coalescing && (gpu_sim_cycle + gpu_tot_sim_cycle > tlb_demotion_last_checked + m_config->demotion_check_cycle ))
    {
        if(COALESCE_DEBUG_SHORT || COALESCE_DEBUG || INVALIDATION_DEBUG) printf("Triggering demotion check. Previous check at %lld cycles, current cycle is %lld. Comparing if %lld is greater than %lld\n", tlb_demotion_last_checked, gpu_sim_cycle+gpu_tot_sim_cycle,gpu_sim_cycle + gpu_tot_sim_cycle,tlb_demotion_last_checked + m_config->demotion_check_cycle);

        tlb_demotion_last_checked = gpu_sim_cycle + gpu_tot_sim_cycle;

        for(std::set<new_addr_type>::iterator itr = promoted_pages[mf->get_appID()]->begin(); itr!=promoted_pages[mf->get_appID()]->end();)
        {
            bool need_demote = false;

            //Setting up the searched key
            float num_pages = (float)(*(m_config->page_sizes))[m_config->page_sizes->size()-2] / (float)(*(m_config->page_sizes))[m_config->page_sizes->size()-1];
            unsigned base_va = (*itr);

            float percent_thres = (float)m_config->page_coalesce_lower_thres_offset / (float)100;
            // Check if this huge page need a demotion or not
            if(m_config->enable_page_coalescing == 1 && (*locality_track)[base_va] != NULL &&// (rand()%20 < 10))
              ( (((float)(*locality_track)[base_va]->size() / num_pages  ) * 100) < (m_config->page_coalesce_locality_thres * percent_thres))) //Policy 1: Only check locality
            {
                if(COALESCE_DEBUG) printf("Trigger demotion for page at %x, ratio = %f\n",base_va,((float)(*locality_track)[base_va]->size() / num_pages  ) * 100);
                need_demote = true;
            }
            else if(m_config->enable_page_coalescing == 2 &&// (rand()%20 < 10))
               //Policy 2: Only consider hot pages, hotness_track tracks the last time this page is touched
              ( ( (gpu_sim_cycle + gpu_tot_sim_cycle) - (*hotness_track)[base_va]  ) > ( m_config->page_coalesce_hotness_thres * percent_thres)))
            {
                if(COALESCE_DEBUG) printf("Trigger demotion for page at %x\n",base_va);
                need_demote = true;
            }
            else if(m_config->enable_page_coalescing == 3 && (*locality_track)[base_va] != NULL &&// (rand()%20 < 10))
              ( (((float)(*locality_track)[base_va]->size() / num_pages  ) * 100) < (m_config->page_coalesce_locality_thres * percent_thres)) && //Policy 3: Check both locality and hotness. Only consider hot pages
              ( ( (gpu_sim_cycle + gpu_tot_sim_cycle) - (*hotness_track)[base_va]  ) > (m_config->page_coalesce_hotness_thres * percent_thres)))
            {
                if(COALESCE_DEBUG) printf("Trigger demotion for page at %x\n",base_va);
                need_demote = true;
            }
            else //Always resize
                need_demote = true;


            // If demotion is needed, perform demotion
            if(need_demote)
            {
                //Note, promote_pages has the folloing format: [base_addr | appID]
                new_addr_type base_addr = (*itr) / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the base address
                int appID = (*itr) & 0xFFFF; //Then get an appID
                if(COALESCE_DEBUG || COALESCE_DEBUG_SHORT) printf("Demoting page with entry %x (page index is %x) for app = %d from the promoted page list\n", (*itr),base_addr, appID);
                demote_page(base_addr,appID);
                promoted_pages[appID]->erase(itr++); //Note that this move itr to the next address. Use itr++ to prevent the case where c++98 is used instead of c++11 (only c++11 return the iterator value)
            }
            else itr++;
        }
    }


}

void tlb_tag_array::clear_page_stats()
{
    //if(COALESCE_DEBUG) printf("Resetting page stats at cycle = %lld\n", gpu_sim_cycle + gpu_tot_sim_cycle);
    printf("Resetting page stats at cycle = %lld\n", gpu_sim_cycle + gpu_tot_sim_cycle);
    for(std::map<new_addr_type,unsigned long long>::iterator itr = hotness_track->begin(); itr!=hotness_track->end();itr++)
        (*itr).second = 0;
    for(std::map<new_addr_type,std::set<int>*>::iterator itr = warpID_track->begin(); itr!=warpID_track->end();itr++)
        (*itr).second->clear();
    for(std::map<new_addr_type,std::set<new_addr_type>*>::iterator itr = locality_track->begin(); itr!=locality_track->end();itr++)
        (*itr).second->clear();


}

// This is obsolete, all the commands are handled in the mmu (memory_owner.cc)
void tlb_tag_array::send_dram_cmd(new_addr_type from_addr, new_addr_type to_addr, int command, int appID)
{
    if(m_shared_tlb!= NULL) //L1 tlb, should not get called, call from shared tlb instead
    {
        m_shared_tlb->send_dram_cmd(from_addr, to_addr, command, appID);
    }
    else //L2 shared TLB send DRAM command
    {
        addrdec_t m_raw_src_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)
        addrdec_t m_raw_target_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)

        int from_bank,to_bank,from_channel,to_channel,from_sa,to_sa;
        m_config->m_address_mapping.addrdec_tlx(from_addr,&m_raw_src_addr, appID, 0,1);
        m_config->m_address_mapping.addrdec_tlx(to_addr,&m_raw_target_addr, appID, 0,1);
        //Parse src and target
        from_bank = (int)(m_raw_src_addr.bk);
        from_sa = (int)(m_raw_src_addr.subarray);
        from_channel = (int)(m_raw_src_addr.chip);
        to_bank = (int)(m_raw_target_addr.bk);
        to_sa = (int)(m_raw_target_addr.subarray);
        to_channel = (int)(m_raw_target_addr.chip);

        dram_cmd * cmd = new dram_cmd(command,from_bank,to_bank,from_channel,to_channel,from_sa,to_sa,appID,(m_config->page_size/m_config->dram_row_size),m_config);
        m_memory_partition[from_channel]->insert_dram_command(cmd);
    }
}

// This routinely call resize page to coalesce pages whenever the page statistics associated with mem_fetch are over the threshold
// This is exclusively called from the L2 TLB
void tlb_tag_array::check_threshold(mem_fetch *mf)
{
    //Only do this if we enable page coalescing. To trigger coalescing, set need_resize to true
    unsigned base_va = mf->get_addr() / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the large page stats
    unsigned search_key = ((mf->get_addr() / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]); //key to search promoted_page [base_va]
    //unsigned search_key = ((mf->get_addr() / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | mf->get_appID(); //key to search promoted_page [base_va | appID]
    //First, if mf is not from a large page, do we want to promote?
    if(COALESCE_DEBUG) printf("Searching for page %x in the promoted page list\n", search_key);
    if(m_config->enable_page_coalescing && (promoted_pages[mf->get_appID()]->find(search_key) == promoted_pages[mf->get_appID()]->end()))
    {

        bool need_resize = false;
        float num_pages = (float)(*(m_config->page_sizes))[m_config->page_sizes->size()-2] / (float)(*(m_config->page_sizes))[m_config->page_sizes->size()-1];


        if(COALESCE_DEBUG) printf("Checking if we need to coalesce page for mem_fetch with VA = %x, base page = %x, base_page denominator = %u, next page size = %u, calculated base_va = %x, calculated num_pages = %d\n", mf->get_addr(),mf->get_addr() / (*(m_config->page_sizes))[m_config->page_sizes->size()-1],(*(m_config->page_sizes))[m_config->page_sizes->size()-1],(*(m_config->page_sizes))[m_config->page_sizes->size()-2], base_va, num_pages);
        // First, check if there are any threshold that trigger page coalescing

        //if(COALESCE_DEBUG) printf("Checking the threshold for coalescing for mf VA = %x, PA = %x. Base VA = %x, number of pages = %d, current number of pages = %d, at cycle = %lld. Measured Locality ratio = %f, threshold (in percentage) = %d\n", mf->get_addr(), m_page_manager->get_pa(mf->get_addr(),mf->get_appID())/(*(m_config->page_sizes))[0], base_va, num_pages, (*locality_track)[base_va]->size(),gpu_sim_cycle + gpu_tot_sim_cycle, (((float)(*locality_track)[base_va]->size() / num_pages  ) * 100), m_config->page_coalesce_locality_thres);
        if(m_config->enable_page_coalescing == 1 && (*locality_track)[base_va] != NULL &&// (rand()%20 < 10))
          ( (((float)(*locality_track)[base_va]->size() / num_pages  ) * 100) > m_config->page_coalesce_locality_thres )) //Policy 1: Only check locality
        {
            if(COALESCE_DEBUG) printf("Trigger coalescing for page at %x, ratio = %f",base_va,((float)(*locality_track)[base_va]->size() / num_pages  ) * 100);
            need_resize = true;
        }
        else if(m_config->enable_page_coalescing == 2 &&// (rand()%20 < 10))
           //Policy 2: Only consider hot pages
          ( ( (gpu_sim_cycle + gpu_tot_sim_cycle) - (*hotness_track)[base_va]  ) < m_config->page_coalesce_hotness_thres ))
        {
            if(COALESCE_DEBUG) printf("Trigger coalescing for page at %x\n",base_va);
            need_resize = true;
        }
        else if(m_config->enable_page_coalescing == 3 && (*locality_track)[base_va] != NULL &&// (rand()%20 < 10))
          ( (((float)(*locality_track)[base_va]->size() / num_pages  ) * 100) > m_config->page_coalesce_locality_thres ) && //Policy 3: Check both locality and hotness. Only consider hot pages
          ( ( (gpu_sim_cycle + gpu_tot_sim_cycle) - (*hotness_track)[base_va]  ) < m_config->page_coalesce_hotness_thres ))
        {
            if(COALESCE_DEBUG) printf("Trigger coalescing for page at %x\n",base_va);
            need_resize = true;
        }

        else //Always resize
            need_resize = true;
        //Call resize page as needed
        if(need_resize)
            resize_page(mf->get_appID(), base_va, mf->get_addr()); //This cause segfault for policy 1

    }
    else if((COALESCE_DEBUG || COALESCE_DEBUG_SHORT ) && m_config->enable_page_coalescing)
    {
        printf("Page with key %x in already in the promoted page list\n", search_key);
    }

}

// New coalesce routine called from Allocator Hub. Note that coalescing is decided from the allocator hence it does not have to
// handle data movements from here. 
int tlb_tag_array::promotion(new_addr_type va, int appID) {
  if(MERGE_DEBUG) printf("Got a promotion call for appID = %d, va = %x, num_apps = %d\n", appID, va, ConfigOptions::n_apps);
  App* app = App::get_app(appID);

  new_addr_type base_addr = va / (*(m_config->page_sizes))[m_config->page_sizes->size() - 2];

  if (MERGE_DEBUG || COALESCE_DEBUG || COALESCE_DEBUG_SHORT)
    printf(
        "Putting the base page %x in the promoted_pages list. (Shifted base_VA with appID = %x).\n",
        base_addr,
        (base_addr * (*(m_config->page_sizes))[m_config->page_sizes->size() - 2]));
  m_mem_stats->coalesced_tried++;
  app->coalesced_tried_app++;


  promoted_pages[appID]->insert(
        (base_addr * (*(m_config->page_sizes))[m_config->page_sizes->size() - 2]));

//    invalidate_entries(appID, base_addr); //Not needed

  if (MERGE_DEBUG || COALESCE_DEBUG || COALESCE_DEBUG_SHORT) {
    printf("Current content of the promoted pages for app %d are: {", appID);
    for (std::set<new_addr_type>::iterator itr = promoted_pages[appID]->begin();
        itr != promoted_pages[appID]->end(); itr++)
      printf("%x, ", *itr);
    printf("}\n");
  }
}


//Old coalesce routine
//Might want to keep track of huge page utilization tracking when L2 TLB fill?
void tlb_tag_array::resize_page(int appID, new_addr_type base_addr, new_addr_type original_addr)
{
  App* app = App::get_app(App::get_app_id(appID));

    if(COALESCE_DEBUG || COALESCE_DEBUG_SHORT) printf("Checking if we need to coalesce page for app = %d, VA = %x, at cycle = %lld\n", appID, base_addr,gpu_sim_cycle + gpu_tot_sim_cycle);

    if(COALESCE_DEBUG || COALESCE_DEBUG_SHORT) printf("Coalesce page for app = %d, VA = %x, at cycle = %lld\n", appID, base_addr,gpu_sim_cycle + gpu_tot_sim_cycle);

    //Just simply grab the physical page associating with base_addr
    int merge_done = m_page_manager->coalesce_page(m_page_manager->find_physical_page(original_addr,appID, (*(m_config->page_sizes))[m_config->page_sizes->size()-1]));
    //bool merge_done = true;

    // This keeps track so probe and fill can fill into the right tlbs
    // Note that we do not put the entries in the TLB, this has to be done afterward
    // Format of entries in promoted_pages = [base_va | appID]
    if(COALESCE_DEBUG || COALESCE_DEBUG_SHORT) printf("Putting the base page %x in the promoted_pages list. (Shifted base_VA with appID = %x). Coalescing return status = %d\n",base_addr, (base_addr *  (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | appID, merge_done);
    m_mem_stats->coalesced_tried++;
    app->coalesced_tried_app++;
    if(merge_done == 1)
    {
        m_mem_stats->coalesced_noinval_succeed++;
        m_mem_stats->coalesced_succeed++;
        app->coalesced_noinval_succeed_app++;
        app->coalesced_succeed_app++;
        promoted_pages[appID]->insert( (base_addr *  (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | appID);
        // No need to invalidate in this case, page tables are the same
    }
    else if(merge_done == 2)
    {
        m_mem_stats->coalesced_succeed++;
        app->coalesced_succeed_app++;
        promoted_pages[appID]->insert( (base_addr *  (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | appID);

        // Invalidate all the entries in existing small page TLBs (Both L1 and L2, but this is called from L2. invalidate entries will automatically invalidate L1 entries as well)
        invalidate_entries(appID, base_addr );
    }
    else
    {
        m_mem_stats->coalesced_fail++;
        app->coalesced_fail_app++;
    }
    if(COALESCE_DEBUG || COALESCE_DEBUG_SHORT){
        printf("Current content of the promoted pages are: {");
        for(std::set<new_addr_type>::iterator itr = promoted_pages[appID]->begin();itr != promoted_pages[appID]->end(); itr++)
           printf("%x, ", *itr);
        printf("\n");
    }



}

//Use by both L1 TLBs and L2 TLB
void tlb_tag_array::invalidate_entries(int appID, new_addr_type base_entry)
{
    if(INVALIDATION_DEBUG) printf("During invalidation for %x\n",base_entry);
    if(m_shared_tlb == NULL) //If L2 TLB
    {
        if(INVALIDATION_DEBUG) printf("During invalidation for L2 shared TLB %x\n",base_entry);
        for(unsigned index = 0; index < m_entries-1 ; index ++)
        {
            if(INVALIDATION_DEBUG) printf("Content of the L2 TLB set %d is {",index);
            for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end();findIter++)
            {
                if(INVALIDATION_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            }
            if(INVALIDATION_DEBUG) printf("}\n");

            for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end();)
            //for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end(); findIter ++)
            {
                //Found, invalidate this entry
                if(INVALIDATION_DEBUG) printf("Invalidate check for L2 entry %x: key = %x, want to invalidate = %x\n",*findIter,((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2],base_entry);
                //if(INVALIDATION_DEBUG) printf("Invalidate check for L2 entry %x: key = %x, want to invalidate = %x\n",*findIter,(*findIter)/(*(m_config->page_sizes))[m_config->page_sizes->size()-2],base_entry);
                //if((*findIter)/(*(m_config->page_sizes))[m_config->page_sizes->size()-2] == base_entry)
                if(((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2] == base_entry)
                {
                    findIter = l2_tag_array[index]->erase(findIter); //Note to use erase instead of remove, as it is an iterator, also, this is incremented right after, so no need to increment the iterator here
                }
                else findIter++;
            }
            if(INVALIDATION_DEBUG) printf("After L2 TLB invalidation for set = %d, content of the L2 TLB set # is {",index);
            for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end();findIter++)
            {
                if(INVALIDATION_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            }
            if(INVALIDATION_DEBUG) printf("}\n");


        }
        // Invalidate the TLB entries
        ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->l2_tlb_invalidate_latency;
        if(INVALIDATION_DEBUG) printf("Invalidated shared L2 TLB for %x, setting ready cycle to %lld, num shaders = %d\n",base_entry, ready_cycle, gpu_sms);
        for(int i = 0 ; i < (gpu_sms); i++)
        {
            // Only invalidate SMs that belong to appID
            unsigned this_appID;
            // Get AppID for each core i
            int group_size = gpu_sms / m_config->gpgpu_num_groups;
            this_appID = i / group_size;

            if(INVALIDATION_DEBUG) printf("Figure out if this core need invalidation. Core ID = %d, app = %d, invalidating app = %d\n",i, this_appID, appID);

            if(this_appID == appID) //If the same core as invalidating app
                l1_tlb[i]->invalidate_entries(appID, base_entry);
        }
    }
    else
    {
        ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->l1_tlb_invalidate_latency;
        if(INVALIDATION_DEBUG) printf("During invalidation for L1 TLB for core %d, invalidating base %x, setting ready cycle to %lld\n",m_core_id,base_entry, ready_cycle);
        if(INVALIDATION_DEBUG) printf("Content of the L1 TLB is {");
        for(std::list<new_addr_type>::iterator findIter = tag_array->begin(); findIter != tag_array->end();findIter++)
        {
            if(INVALIDATION_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
        }
        if(INVALIDATION_DEBUG) printf("}\n");

        // Invalidate the TLB entries
        for(std::list<new_addr_type>::iterator findIter = tag_array->begin(); findIter != tag_array->end();)
        {
            //Found, invalidate this entry
            if(INVALIDATION_DEBUG) printf("During invalidate, searching L1 small TLB for %x, current entry address is %x, compared key is %x, shifted_value is %u, val2 is %u\n",base_entry,*findIter,((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2],(*(m_config->page_sizes))[m_config->page_sizes->size()-2],(*(m_config->page_sizes))[m_config->page_sizes->size()-1]);
            //if(INVALIDATION_DEBUG) printf("During invalidate, searching L1 small TLB for %x, current entry address is %x, compared key is %x, shifted_value is %u\n",base_entry,*findIter,((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2],(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            if(((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2] == base_entry)
            //if((*findIter)/(*(m_config->page_sizes))[m_config->page_sizes->size()-2] == base_entry)
            {
                findIter = tag_array->erase(findIter); //Same here, need to handle itr
            }
            else findIter ++ ;
        }
        if(INVALIDATION_DEBUG) printf("After invalidation, content of the L1 TLB is {");
        for(std::list<new_addr_type>::iterator findIter = tag_array->begin(); findIter != tag_array->end();findIter++)
        {
            if(INVALIDATION_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
        }
        if(INVALIDATION_DEBUG) printf("}\n");

    }
}

// Only called from L1 TLB to L2 TLB
void tlb_tag_array::set_l1_tlb(int coreID, tlb_tag_array * l1)
{
    printf("Setting L1 TLB pointer for the shared TLB for SM %d\n",coreID);
    l1_tlb[coreID] = l1;
}

void tlb_tag_array::l2_fill( new_addr_type addr, unsigned accessor, mem_fetch * mf)
{
     m_shared_tlb->fill(addr,accessor,mf);
}

void tlb_tag_array::fill( new_addr_type addr, unsigned accessor, mem_fetch * mf ) // For L2 TLB only
{

     if(!m_config->tlb_enable) return; //Does not have to fill if we always return TLB HIT(speeding things up)

     measure_sharer_stats();

     if(COALESCE_DEBUG) printf("In L2 fill. Prior to checking the coalesce threshold for VA = %x, appID = %d\n", mf->get_addr(), accessor);


// No longer need to do this two: MMU handles this
//     update_page_stats(mf); //Do this up here
//     check_threshold(mf);

     //temporary key to search into the promoted page list
     unsigned key =( (addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) *  (*(m_config->page_sizes))[m_config->page_sizes->size()-2] );
     //unsigned key =( (addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) *  (*(m_config->page_sizes))[m_config->page_sizes->size()-2] )| mf->get_appID();
     //unsigned key = addr >> (m_config->page_size);
     bool to_large_tlb = false;


     //First thing first, remove the key from the tag array, if present (should not be)
     if(promoted_pages[mf->get_appID()]->find(key) != promoted_pages[mf->get_appID()]->end()) //Large page
     {
         to_large_tlb = true;
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2];
     }
     else
     {
         key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-1];
     }

     if(HUGEPAGE_DEBUG || COALESCE_DEBUG) printf("Fill into shared TLBs for VA = %x, appID = %d. to_large_tlb = %d. Searched key = %x\n", mf->get_addr(), accessor, to_large_tlb, key);

     unsigned index;
     if(m_config->tlb_core_index)
         index = to_large_tlb?(mf->get_sid() ^ (key)) & (m_entries_large-1):(mf->get_sid() ^ (key)) & (m_entries-1);
         //index = (mf->get_sid() ^ (addr >> (m_config->page_size) ) ) & (m_entries-1);
     else
         //index = (addr >> (m_config->page_size) )& (m_entries-1);
         index = to_large_tlb?(key)& (m_entries_large-1):(key)& (m_entries-1);

     bool evicted = false;

     if(
          (m_config->TLB_flush_enable && ((gpu_sim_cycle+gpu_tot_sim_cycle) < L2_ready_cycle)) || //During Flush
          (m_config->tlb_lookup_bypass > 0) && bypass_TLB(mf, mf->get_appID()) // Normal condition
       )
     {
         if(COALESCE_DEBUG) printf("Bypass this particular fill for VA = %x, appID = %d\n", mf->get_addr(), accessor);

//         bypass_done_queue->insert(key);
         if(to_large_tlb)
         {
             large_bypass_done_queue->remove(key); //For LRU
             if(large_bypass_done_queue->size() >= m_config->tlb_victim_size_large)
                 large_bypass_done_queue->pop_back();
             large_bypass_done_queue->push_front(key);
         }
         else
         {
             bypass_done_queue->remove(key); //For LRU
             if(bypass_done_queue->size() >= m_config->tlb_victim_size)
                 bypass_done_queue->pop_back();
             bypass_done_queue->push_front(key);
         }

         return; //Bypass the fill into TLB
     }

//     unsigned index = addr;
     std::list<new_addr_type> ** correct_tag_array;
     if(to_large_tlb)
         correct_tag_array = large_l2_tag_array;
     else
         correct_tag_array = l2_tag_array;


     if(HUGEPAGE_DEBUG) printf("Content of the associated tag array in the shared TLB (before fill) set %d is {", index);
     for(std::list<new_addr_type>::iterator findIter = correct_tag_array[index]->begin(); findIter != correct_tag_array[index]->end();findIter++)
     {
         if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
     }
     if(HUGEPAGE_DEBUG) printf("}\n");

     correct_tag_array[index]->remove(key); // remove the current entry, put this in the front of the queue
     if(COALESCE_DEBUG) printf("Rachat-debug: Filling %x to TLB, current size for this way = %d\n",addr, l2_tag_array[index]->size());
     new_addr_type removed_entry;
     if(correct_tag_array[index]->size() >= (to_large_tlb?m_ways_large:m_ways))
     {
         evicted = true;
         if(m_config->tlb_replacement_policy == 0)
         {
             removed_entry = correct_tag_array[index]->back(); // Track app TLB footprint
             correct_tag_array[index]->pop_back(); // LRU policy
         }
         else if(m_config->tlb_replacement_policy == 1) //WID-based #1, probably will suck because it probably will delete new entries
         {
             std::list<new_addr_type>::const_iterator min_itr = correct_tag_array[index]->begin();
             unsigned min_count = 999999;
             for(std::list<new_addr_type>::const_iterator itr = correct_tag_array[index]->begin(); itr !=correct_tag_array[index]->end(); itr++)
             {
                 new_addr_type temp = *itr;
                 if((*tlb_entry_sharer)[temp].size() > 0 && (*tlb_entry_sharer)[temp].size() < min_count)
                 {
                     min_itr = itr;
                     min_count = (*tlb_entry_sharer)[temp].size();
                 }
             }
             removed_entry = *min_itr;
             correct_tag_array[index]->remove(removed_entry);
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }
         else if(m_config->tlb_replacement_policy == 2) //WID-based #2, inverse of 1, which should also suck because this time it removes the highest sharer
         {
             std::list<new_addr_type>::const_iterator min_itr = correct_tag_array[index]->begin();
             unsigned min_count = 0;
             for(std::list<new_addr_type>::const_iterator itr = correct_tag_array[index]->begin(); itr !=correct_tag_array[index]->end(); itr++)
             {
                 new_addr_type temp = *itr;
                 if((*tlb_entry_sharer)[temp].size() > 0 && (*tlb_entry_sharer)[temp].size() > min_count)
                 {
                     min_itr = itr;
                     min_count = (*tlb_entry_sharer)[temp].size();
                 }
             }
             removed_entry = *min_itr;
             correct_tag_array[index]->remove(removed_entry);
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }
         else if(m_config->tlb_replacement_policy == 3) //WID-based #3, get the most LRU from low and high pool. Always pick the one from low pool
         {
             new_addr_type target_entry_low = *(correct_tag_array[index]->begin());
             new_addr_type target_entry_high = *(correct_tag_array[index]->begin());
//             printf("Rachat-debug: Finding replacement candidate, current addr = {");
             for(std::list<new_addr_type>::const_iterator itr = correct_tag_array[index]->begin(); itr !=correct_tag_array[index]->end(); itr++)
             {
                 new_addr_type temp = *itr;
//                 printf("%x ", temp);
                 if(replacement_hashed_map[temp%m_config->tlb_replacement_hash_size]) //Found on the high list, mark this entry for high_list_LRU (more LRU are in the back)
                 {
//                     printf("(New High)");
                     target_entry_high = temp;
                 }
                 else
                 {
//                     printf("(New Low)");
                     target_entry_low = temp;
                 }
//                 printf(", ");
             }
//             printf("}\n");
             removed_entry = target_entry_low;
//             printf("Rachat-debug: Finding replacement candidate is addr = %x, size = %d\n",removed_entry, l2_tag_array[index]->size());
             correct_tag_array[index]->remove(removed_entry);
             if((*tlb_entry_sharer).size() > m_config->tlb_replacement_high_threshold) replacement_hashed_map[removed_entry%m_config->tlb_replacement_hash_size] = true;
             else replacement_hashed_map[removed_entry%m_config->tlb_replacement_hash_size] = false;
             tlb_entry_sharer->erase(removed_entry); // Removed evicted entry from our sharer map (that contains how many wid are sharing this entry)
         }

         else
         {
             removed_entry = correct_tag_array[index]->back(); // Track app TLB footprint
             correct_tag_array[index]->pop_back(); // LRU policy by default
         }
     }


     //Codes below might not work with two TLBs for large and small pages

     App* app = App::get_app(mf->get_appID());
     //App* app = App::get_app(App::get_app_id(mf->get_appID()));
     app->tlb_occupancy++;
     app->addr_mapping.insert(key);
     app->evicted = false;

     if (evicted) {
       for (std::map<appid_t, App*>::iterator i = App::get_apps().begin(); i != App::get_apps().end();
           i++) {
         App* app = i->second;
         if (app->addr_mapping.find(removed_entry) != app->addr_mapping.end()) {
           app->addr_mapping.erase(removed_entry);
           app->evicted = true;
           app->tlb_occupancy--;
         }
       }
     }

     unsigned long long epoch = gpu_tot_sim_cycle + gpu_sim_cycle - tlb_occupancy_last_updated;

     unsigned l2_tlb_size = m_ways * m_entries;
     for (std::map<appid_t, App*>::iterator i = App::get_apps().begin(); i != App::get_apps().end();
         i++) {
       App* app = i->second;
       app->tlb_occupancy_end = (float) app->tlb_occupancy / (float) l2_tlb_size;
       if (app->tlb_occupancy_peak < app->tlb_occupancy_end) {
         app->tlb_occupancy_peak = app->tlb_occupancy_end;
       }
       // This doesn't look right, but it was what was here before...
       app->tlb_occupancy_avg = app->tlb_occupancy_avg + ((float) app->tlb_occupancy /
           (float) l2_tlb_size * epoch);
     }
     tlb_occupancy_last_updated = gpu_tot_sim_cycle + gpu_sim_cycle;

     // Lastly, pushed the filled entry back in at the MRU position
     if(to_large_tlb)
         large_l2_tag_array[index]->push_front(key); //LRU
     else
         l2_tag_array[index]->push_front(key); //LRU
     //Then stats for accessor


     if(MERGE_DEBUG) printf("After L2 TLB fill for set = %d, content of the L2 TLB set # is {",index);
     for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end();findIter++)
     {
         if(MERGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
     }
     if(MERGE_DEBUG) printf("}\n");


     if(HUGEPAGE_DEBUG) printf("Content of the associated tag array (set = %d) in the shared TLB (after fill) is {", index);
     for(std::list<new_addr_type>::iterator findIter = correct_tag_array[index]->begin(); findIter != correct_tag_array[index]->end();findIter++)
     {
         if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
     }
     if(HUGEPAGE_DEBUG) printf("}\n");



}

bool tlb_tag_array::lots_of_sharer(mem_fetch * mf)
{
     new_addr_type temp = (mf->get_addr()) >> (m_config->page_size);
     return replacement_hashed_map[temp%m_config->tlb_replacement_hash_size];
}

//Update L1 TLB ready cycle for all slices associated with appID
void tlb_tag_array::update_L1_ready_cycle(int appID, unsigned long long time)
{
    for(int i=0;i<gpu_sms;i++)
    {
        unsigned this_appID;
        // Get AppID for each core
        int group_size = gpu_sms / m_config->gpgpu_num_groups;
        this_appID = m_core_id / group_size;
        if(this_appID == appID ) //Check if this L1 slice belong to this app
        {
            L1_ready_cycle[i] = gpu_sim_cycle+gpu_tot_sim_cycle+m_config->tlb_L1_flush_cycles;
        }
    }
}

//This is for non-triggering L1 TLB slice to flush once it detect it needs to flush
unsigned tlb_tag_array::flush()
{
    unsigned num_flushed = tag_array->size();
    m_mem_stats->TLB_L1_flush_stalled[m_core_id]+=m_config->tlb_L1_flush_cycles;
    tag_array->clear(); //Flush this tag arrays
}

unsigned tlb_tag_array::flush(int appID)
{
  App* app = App::get_app(App::get_app_id(appID));
  unsigned num_flushed = 0;
  if (m_shared_tlb != NULL) { //L1 TLB Flush, either no flush or all flush
    m_shared_tlb->update_L1_ready_cycle(appID,
        gpu_sim_cycle + gpu_tot_sim_cycle + m_config->tlb_L1_flush_cycles);

    num_flushed = tag_array->size();

    m_mem_stats->TLB_L1_flush_stalled[appID] += m_config->tlb_L1_flush_cycles;
    tag_array->clear(); //Flush this tag arrays

  } else { //L2 TLB Flush -> go through all the entries
    app->flush_count++;
    //Go through all the entries and invalide (remove) any
    for (int i = 0; i < m_entries; i++) {
      l2_tag_array[i]->clear(); //Somehow this cause segfault...thanks for that.
    }

    //Set ready cycle
    L2_ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle + m_config->tlb_L2_flush_cycles;
    m_mem_stats->TLB_L2_flush_stalled += m_config->tlb_L2_flush_cycles;

    // Invalidate bypass cache as well
    bypass_done_queue->clear();
    large_bypass_done_queue->clear();
    //Set ready cycle for the bypass cache
    m_mem_stats->TLB_bypass_cache_flush_stalled += m_config->tlb_bypass_cache_flush_cycles;
    bypass_cache_ready_cycle = gpu_sim_cycle + gpu_tot_sim_cycle
        + m_config->tlb_bypass_cache_flush_cycles;

  }
  return num_flushed;
}

void tlb_tag_array::print_statistics()
{
    printf("Resetting TLB Statistics\nStride set content = {");
    for(std::set<unsigned>::iterator itr = stride_set.begin();itr!=stride_set.end();itr++)
    {
        printf("%d, ",*itr);
    }
    printf("}\n");
}

void tlb_tag_array::reset_bypass_stats(unsigned appID) {
  App* app = App::get_app(App::get_app_id(appID));

  //Reset stats
  app->epoch_accesses = 0;
  app->epoch_previous2_miss_rate = app->epoch_previous_miss_rate;
  app->epoch_previous_miss_rate = (float) app->epoch_miss
      / ((float) app->epoch_miss + (float) app->epoch_hit + (float) app->epoch_bypass_hit);
  app->epoch_hit = 0;
  app->epoch_bypass_hit = 0;
  app->epoch_miss = 0;

  unsigned total_warp_counts = 0;

  for (int i = 0; i < 4000; i++) {
    app->wid_epoch_previous2_miss_rate[i] = app->wid_epoch_previous_miss_rate[i];
    if ((app->wid_epoch_miss[i] + app->wid_epoch_hit[i]) == 0) {
      if (m_config->tlb_lookup_bypass == 3)
        app->wid_epoch_previous_miss_rate[i] = -1;
      else
        app->wid_epoch_previous_miss_rate[i] = 101.0;
    } else {
      total_warp_counts++;
      app->wid_epoch_previous_miss_rate[i] = (float) app->wid_epoch_miss[i]
          / ((float) app->wid_epoch_miss[i] + (float) app->wid_epoch_hit[i]);
    }
    app->wid_epoch_hit[i] = 0;
    app->wid_epoch_miss[i] = 0;
  }

  if (m_stat->total_warp_counts < total_warp_counts)
    m_stat->total_warp_counts = total_warp_counts;

  previous_bypass_enabled = true;

  float jump_dist = 0.1;
  if (app->epoch_previous2_miss_rate > app->epoch_previous_miss_rate) {
    //This epoch is better, if we are more aggressive (true), be even more aggressive (fewer tokens)
    app->tokens =
        bypass_dir == true ? (float) app->tokens - jump_dist : (float) app->tokens + jump_dist;
  } else { //This epoch is worse, if we are more aggressive (true), be less aggressive (more tokens)
    app->tokens =
        bypass_dir == true ? (float) app->tokens + jump_dist : (float) app->tokens - jump_dist;
  }

  if (app->tokens > 1.0)
    app->tokens = 1.0;
  if (app->tokens < 0)
    app->tokens = 0;

  unsigned handout_tokens = total_warp_counts * app->tokens;

  m_stat->tlb_bypassed_token_size = handout_tokens;

  //Rank all the warpID hit rate if needed
  unsigned index[4000];
  float copy[4000];
  for (int ct = 0; ct < 4000; ct++) {
    index[ct] = ct;
    copy[ct] = app->wid_epoch_previous_miss_rate[ct];
  }

  if (m_config->tlb_lookup_bypass == 2 || m_config->tlb_lookup_bypass == 3) {
    // Stupid sort :( ... For now, whatever
    for (int i = 0; i < 4000; i++) {
      for (int j = i; j < 4000; j++) {
        if (m_config->tlb_lookup_bypass == 2) {
          if (copy[i] > copy[j]) //Sorted by miss rate, lowest first
              {
            float temp = copy[i];
            unsigned temp2 = index[i];
            copy[i] = copy[j];
            index[i] = index[j];
            copy[j] = temp;
            index[j] = temp2;
          }
        }
        if (m_config->tlb_lookup_bypass == 3) {
          if (copy[i] < copy[j]) //Sorted by miss rate, lowest first
              {
            float temp = copy[i];
            unsigned temp2 = index[i];
            copy[i] = copy[j];
            index[i] = index[j];
            copy[j] = temp;
            index[j] = temp2;
          }
        }
      }
    }
  }

  //Release tokens
  if (m_config->tlb_lookup_bypass == 4) {
    int i = 0;
    while (handout_tokens < 4000) {
      for (; i < handout_tokens; i++) {
        app->wid_tokens[i] = true;
      }
    }
    for (; i < 4000; i++) {
      app->wid_tokens[i] = false;
    }
  }

  //Release tokens
  if (m_config->tlb_lookup_bypass == 6) {
    int i = 0;
    for (i = 0; i < 4000; i++) {
      app->wid_tokens[i] = false;
    }
    i = 0;
    while (handout_tokens > 0) {
      if (i == 4000)
        break;
      if ((app->wid_epoch_miss[i] + app->wid_epoch_hit[i]) > 0) {
        app->wid_tokens[i] = true;
        handout_tokens--;
      }
      i++;
    }
  }

  //Release tokens
  if (m_config->tlb_lookup_bypass == 5) {
    int i = 0;
    for (i = 0; i < 4000; i++) {
      app->wid_tokens[i] = false;
    }
    i = 0;
    while (handout_tokens > 0) {
      for (int j = 0; j < 30; j++) {
        if ((app->wid_epoch_miss[i] + app->wid_epoch_hit[j * 100 + i]) > 0) {
          app->wid_tokens[j * 100 + i] = true;
          handout_tokens--;
        }
      }
      if (i > 100)
        break;
      i++;
    }
  }

  //Release tokens
  if (m_config->tlb_lookup_bypass == 1) {
    int i = 0;
    for (i = 0; i < 4000; i++) {
      app->wid_tokens[i] = false;
    }
    i = 0;
    while (handout_tokens > 0) {
      if (i == 4000)
        break;
      if ((app->wid_epoch_miss[i] + app->wid_epoch_hit[i]) > 0) {
        app->wid_tokens[i] = true;
        handout_tokens--;
      }
      i++;
    }
  }
  //Release tokens
  if (m_config->tlb_lookup_bypass == 2 || m_config->tlb_lookup_bypass == 3) {
    int i = 0;
    for (i = 0; i < 4000; i++) {
      app->wid_tokens[i] = false;
    }
    for (i = 0; i < handout_tokens; i++) {
      if (i == 4000)
        break;
      app->wid_tokens[index[i]] = true; //Assign tokens to top warps
    }
  }
}


//Should only be called from the L2 (shared) TLB
bool tlb_tag_array::bypass_TLB(mem_fetch * mf, unsigned appID) {
  App* app = App::get_app(App::get_app_id(appID));
  if (app->epoch_accesses > m_config->tlb_stat_resets) {
    reset_bypass_stats(appID);
  }
  if (app->epoch_previous_miss_rate > (float) m_config->max_tlb_miss / 100) {
    return (previous_bypass_enabled && !app->wid_tokens[(mf->get_sid() * 100) + mf->get_wid()]);
  }
  return false;
}

unsigned tlb_tag_array::get_concurrent_tlb_app(int appID) {
  App* app = App::get_app(App::get_app_id(appID));
  return app->concurrent_tracker;
}

// Get the sharer statistics for each TLBs
void tlb_tag_array::measure_sharer_stats()
{
    unsigned total = 0;
    unsigned size = 0;
    unsigned max = 0;
    unsigned min = 9999999999;
    std::list<int> tracker;
    for(std::map<new_addr_type,std::set<unsigned>>::iterator itr = tlb_entry_sharer->begin(); itr!=tlb_entry_sharer->end(); itr++)
    {
        unsigned sharer = (itr->second).size();
        total = total + sharer;
        size++;
        if(sharer > max) max = sharer;
        if(sharer < min) min = sharer;
        tracker.push_back((int)sharer);
    }

    float average;
    if(size == 0) average = 0.0;
    else average = (float)total/(float)size;
    float variance = 0.0;
    for(std::list<int>::iterator itr = tracker.begin(); itr != tracker.end(); itr++)
    {
        float tmp = (float)(*itr)-average;
        if(tmp > 0) variance += tmp;
        else variance = variance - tmp;
    }
    if(size == 0) variance = 0;
    else variance = variance / (float)size;


    if(m_shared_tlb!=NULL) //L1 TLB statistics
    {
        m_mem_stats->TLBL1_sharer_avg[m_core_id] = average;
        m_mem_stats->TLBL1_total_unique_addr[m_core_id] = size;
        m_mem_stats->TLBL1_sharer_var[m_core_id] = variance;
        if(m_mem_stats->TLBL1_sharer_max[m_core_id] < max)
            m_mem_stats->TLBL1_sharer_max[m_core_id] = max;
        if(m_mem_stats->TLBL1_sharer_max[m_core_id] > min)
            m_mem_stats->TLBL1_sharer_max[m_core_id] = min;
    }
    else //L2 TLB statistics
    {
        m_mem_stats->TLBL2_sharer_avg = average;
        m_mem_stats->TLBL2_total_unique_addr = size;
        m_mem_stats->TLBL2_sharer_var = variance;
        if(m_mem_stats->TLBL2_sharer_max < max)
            m_mem_stats->TLBL2_sharer_max = max;
        if(m_mem_stats->TLBL2_sharer_max > min)
            m_mem_stats->TLBL2_sharer_max = min;

    }
}

enum tlb_request_status tlb_tag_array::probe( new_addr_type addr, unsigned accessor, mem_fetch * mf) {
    if(APPID_SHIFT_DEBUG) printf("Accessor ID = %d, mf->appID = %d. Probing for %p (%p) Size = %d. Uint64 format = %p (%p) size = %d. Hacked addr = %p (%p)\n",  accessor, mf->get_appID(), addr, mf->get_addr(), sizeof(new_addr_type), (uint64_t) addr, (uint64_t) mf->get_addr(), sizeof(uint64_t), ((uint64_t)accessor << (uint64_t)48) | addr, ((uint64_t)accessor << (uint64_t)48) | mf->get_addr());
  App* app = App::get_app(mf->get_appID());
  //App* app = App::get_app(App::get_app_id(accessor));
    if(!m_config->tlb_enable) return TLB_HIT;


//    if(COALESCE_DEBUG) printf("Probe into TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);

    bool to_large_page = false;
    //temporary, so we can search promoted page set
    unsigned searched_key = ((addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
    //unsigned searched_key = ((addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2] ) * (*(m_config->page_sizes))[m_config->page_sizes->size()-2]) | mf->get_appID();
    if(m_shared_tlb != NULL) //Pick the right key to search (small vs. large page), based on whether it is an L1 or L2 TLB
    {
        if(COALESCE_DEBUG) printf("Probe L1 TLBs for VA = %x, appID = %d, searched addr = %x\n", mf->get_addr(), accessor, searched_key);
        if(m_shared_tlb->promoted_pages[mf->get_appID()]->find(searched_key) != m_shared_tlb->promoted_pages[mf->get_appID()]->end()) //Large page
        {
            to_large_page = true;
            searched_key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the searched key
            if(HUGEPAGE_DEBUG || MERGE_DEBUG || COALESCE_DEBUG) printf("Probe into Large page L1 TLBs for VA = %x, appID = %d. Searched key = %x\n", mf->get_addr(), accessor, searched_key);
        }
        else
        {
            if(COALESCE_DEBUG) printf("Probe into Small page L1 TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);
            searched_key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-1]; //Get the searched key
        }
    }
    else
    {
        if(COALESCE_DEBUG) printf("Probe L2 TLBs for VA = %x, appID = %d, searched addr = %x\n", mf->get_addr(), accessor, searched_key);
        if(promoted_pages[mf->get_appID()]->find(searched_key) != promoted_pages[mf->get_appID()]->end()) //Large page
        {
            to_large_page = true;
            searched_key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the searched key
            if(MERGE_DEBUG || COALESCE_DEBUG) printf("Probe into Large page L2 TLBs for VA = %x, appID = %d. Searched key = %x\n", mf->get_addr(), accessor, searched_key);
        }
        else
        {
            if(COALESCE_DEBUG) printf("Probe into Small page L2 TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);
            searched_key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size()-1]; //Get the searched key
        }
    }

//Check if this access is to an active faulting page
    if(m_page_manager->get_DRAM_layout()->is_active_fault(addr)) //Check if a request is going a page with active FAULT
    {
//        if(COALESCE_DEBUG) printf("Active fault when probing into TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);
        return TLB_FAULT;
    }

//    if(bypass_done_queue->find(searched_key)!=bypass_done_queue->end())
//    {
//        m_stat->tlb_bypassed++;
//        m_stat->tlb_bypassed_app[accessor]++;
//        return TLB_HIT;
//    }

    //L1_TLB
    if(m_shared_tlb!= NULL)
    {
//        if(COALESCE_DEBUG) printf("Probe into L1 TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);
        app->addr_mapping.insert(searched_key);

        //While flushing
        if(m_config->TLB_flush_enable && (gpu_sim_cycle+gpu_tot_sim_cycle < m_shared_tlb->L1_ready_cycle[m_core_id]))
        {
            return TLB_MISS;
        }

        if((gpu_sim_cycle + gpu_tot_sim_cycle) < ready_cycle)
        {
            return TLB_FLUSH;
        }

        // If fixed TLB latency is used, clear out any finished requests
        if(m_config->tlb_fixed_latency_enabled)
        {
            m_shared_tlb->clear_static_queue();
        }

        remove_done_REG();
        if(m_config->tlb_prefetch)handle_pending_prefetch();

        std::list<new_addr_type>::iterator REG_itr = std::find(bypass_done_REG->begin(), bypass_done_REG->end(), searched_key);
        if(REG_itr != bypass_done_REG->end())
        {
//            m_stat->tlb_concurrent_serviced++;
//            m_stat->tlb_concurrent_serviced_app[accessor]++;
            return TLB_HIT;
        }

        if(m_access%100000==0)
            print_statistics();

        if(m_access%1000000==0)
        {
            stride_set.clear();
        }

        if(m_config->tlb_prefetch){
            prev_stride = stride;
            stride = searched_key;
            stride_set.insert(stride-prev_stride);
            int prefetch_index = 0;
            if(m_config->tlb_prefetch == 3) prefetch_index = (stride-prev_stride) % m_config->tlb_prefetch_set;
            std::list<new_addr_type>::iterator pref_itr = std::find(prefetch_buffer[prefetch_index]->begin(), prefetch_buffer[prefetch_index]->end(), searched_key);
            if(pref_itr != prefetch_buffer[prefetch_index]->end())
            {
                prefetch_queue[prefetch_index]->push_back(searched_key+ (stride-prev_stride));
                prefetch_queue_time[prefetch_index]->push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
                m_stat->tlb_prefetch_hit++;
                app->tlb_prefetch_hit_app++;
                return TLB_HIT;
            }
        }

        //While flushing, return TLB miss if we cannot access bypass cache in shared TLB
        if(m_config->TLB_flush_enable && (gpu_sim_cycle+gpu_tot_sim_cycle < m_shared_tlb->bypass_cache_ready_cycle))
        {
            return TLB_MISS;
        }


        // Add support for victim TLB for large page
        if(to_large_page)
        {
            std::list<new_addr_type>::iterator bypass_find_itr = std::find(m_shared_tlb->large_bypass_done_queue->begin(), m_shared_tlb->large_bypass_done_queue->end(), searched_key);
            if(bypass_find_itr != m_shared_tlb->large_bypass_done_queue->end())
            {
                if(HUGEPAGE_DEBUG) printf("Found TLB entry in the lastpage victim TLB (should not be here if MASK is not enabled. key = %x\n", searched_key);

                (*tlb_entry_sharer)[searched_key].insert(mf->get_wid()); //Put this wid as one of the sharer of this TLB entry

                m_stat->tlb_bypassed++;
                m_stat->large_tlb_bypassed++;
                app->tlb_bypassed_app++;
                app->large_tlb_bypassed_app++;
                m_shared_tlb->large_bypass_done_queue->remove(searched_key); //For LRU, put this in the MRU position
                m_shared_tlb->large_bypass_done_queue->push_front(searched_key);
                app->epoch_bypass_hit++;
                return TLB_HIT;
            }

        }
        else
        {
            std::list<new_addr_type>::iterator bypass_find_itr = std::find(m_shared_tlb->bypass_done_queue->begin(), m_shared_tlb->bypass_done_queue->end(), searched_key);
            if(bypass_find_itr != m_shared_tlb->bypass_done_queue->end())
            {

                (*tlb_entry_sharer)[searched_key].insert(mf->get_wid()); //Put this wid as one of the sharer of this TLB entry

                m_stat->tlb_bypassed++;
                m_stat->small_tlb_bypassed++;
                app->tlb_bypassed_app++;
                app->small_tlb_bypassed_app++;
                m_shared_tlb->bypass_done_queue->remove(searched_key); //For LRU, put this in the MRU position
                m_shared_tlb->bypass_done_queue->push_front(searched_key);
                app->epoch_bypass_hit++;
                return TLB_HIT;
            }
        }



        unsigned key = searched_key;

        m_access++;
        m_stat->tlb_access++;
        app->tlb_access_app++;

        // Hit in the TLB MSHR
        if(m_shared_tlb->miss_queue->find(key)!=m_shared_tlb->miss_queue->end())
        {

            if(HUGEPAGE_DEBUG) printf("Searched key = %x is in the MSHR\n", key);
            int old_size = (*tlb_entry_sharer)[searched_key].size();
            (*tlb_entry_sharer)[searched_key].insert(mf->get_wid()); //Put this wid as one of the sharer of this TLB entry
            int new_size = (*tlb_entry_sharer)[searched_key].size();



//            unsigned new_count = miss_tracker_count[mf->get_appID()]->find(key)->second + 1;
//            miss_tracker_count[mf->get_appID()]->erase(key);
//            miss_tracker_count[mf->get_appID()]->insert(std::pair<new_addr_type,unsigned>(key,new_count));

//            for(std::set<new_addr_type>::for(iterator itr = m_shared_tlb->miss_queue->begin(); itr!=m_shared_tlb->miss_queue->end();itr++)
//            {
//                printf("%x, ",*itr);
//            }
//            printf("}\n");
            return TLB_HIT_RESERVED;
        }

        // If this is in the large page list
        if(to_large_page)
        {
            //This should already be set up there
            //key = searched_key / (*(m_config->page_sizes))[1]; //Get the searched key
            if(HUGEPAGE_DEBUG) printf("Searched key = %x, Content of the large L1 TLB (before probing) is {", key);
            for(std::list<new_addr_type>::iterator findIter = large_tag_array->begin(); findIter != large_tag_array->end();findIter++)
            {
                if(HUGEPAGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            }
            if(HUGEPAGE_DEBUG) printf("}\n");


            std::list<new_addr_type>::iterator findIter = std::find(large_tag_array->begin(), large_tag_array->end(), key);
            if(findIter != large_tag_array->end()) //Found in L1 TLB
            {
                if(HUGEPAGE_DEBUG) printf("Large page L1 TLB hit for VA = %x, key = %x\n", mf->get_addr(), key);
                large_tag_array->remove(key);
                large_tag_array->push_front(key); // insert at MRU
                m_stat->tlb_hit++;
                app->tlb_hit_app++;
                app->tlb_hit_app_epoch++;

                m_stat->large_tlb_hit++;
                app->large_tlb_hit_app++;


                app->epoch_accesses++;
                app->epoch_hit++;

                app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
                app->wid_epoch_hit[(mf->get_sid()*100)+mf->get_wid()]++;

                return TLB_HIT;
            }
            else //Not in L1 TLB, searched shared TLB
            {
                //Just return TLB MISS is l2 is enabled. No need to probe
                if(m_config->l2_tlb_entries == 0) return TLB_MISS;
                m_stat->large_tlb_miss++;
                m_stat->tlb_miss++;
                app->large_tlb_miss_app++;
                app->tlb_miss_app++;
                m_miss++;

                app->epoch_accesses++;
                app->epoch_miss++;

                app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
                app->wid_epoch_miss[(mf->get_sid()*100)+mf->get_wid()]++;

    //            return TLB_MISS;
                tlb_request_status tlb_status = m_shared_tlb->probe(addr,accessor, mf);
                //These should be the same
                if(tlb_status == TLB_MISS)
                {

                    //New miss, add to the tracker
    //                miss_tracker[mf->get_appID()]->push_front(key);
    //                miss_tracker_count[mf->get_appID()]->insert(std::pair<new_addr_type,unsigned>(key,(unsigned)1));
    //                miss_tracker_timestamp[mf->get_appID()]->push_front(gpu_sim_cycle + gpu_tot_sim_cycle);


                    m_shared_tlb->miss_queue->insert(key);
                    app->concurrent_tracker++;

                    //Statistics
                    m_stat->tlb_current_concurrent_serviced = m_shared_tlb->miss_queue->size();
                    //Update fraction of time core spent on this concurrent TLB misses
                    if(m_shared_tlb->miss_queue->size() > 199)
                        m_stat->tlb_concurrent_total_time[199] += (gpu_sim_cycle+gpu_tot_sim_cycle -
                            last_update_concurrent_stat);
                    else
                        m_stat->tlb_concurrent_total_time[m_shared_tlb->miss_queue->size()] +=
                            (gpu_sim_cycle+gpu_tot_sim_cycle - last_update_concurrent_stat);

                    if(m_shared_tlb->miss_queue->size() > 199) {
                      assert(mf->get_appID() == accessor && "modified data on wrong app, should be mf->get_appID() app");
                      app->tlb_concurrent_total_time_app[199] += (gpu_sim_cycle+gpu_tot_sim_cycle -
                          last_update_concurrent_stat);
                    } else {
                      assert(mf->get_appID() == accessor && "modified data on wrong app, should be mf->get_appID() app");
                      app->tlb_concurrent_total_time_app[m_shared_tlb->miss_queue->size()] +=
                          (gpu_sim_cycle+gpu_tot_sim_cycle - last_update_concurrent_stat);
                    }
                    last_update_concurrent_stat = gpu_sim_cycle+gpu_tot_sim_cycle;

                    if(m_shared_tlb->miss_queue->size() > m_stat->tlb_concurrent_serviced) //Update max concurrent TLB
                        m_stat->tlb_concurrent_serviced = m_shared_tlb->miss_queue->size();


                    if(m_stat->tlb_concurrent_max < m_shared_tlb->miss_queue->size())
                      m_stat->tlb_concurrent_max = m_shared_tlb->miss_queue->size();
                    if(app->tlb_concurrent_max_app < m_shared_tlb->miss_queue->size())
                       app->tlb_concurrent_max_app = m_shared_tlb->miss_queue->size();

                    app->tlb_miss_app_epoch++;
                }
                else
                {
                    app->tlb_hit_app_epoch++;
                }
                return tlb_status;
            }
        } //End of large page L1 TLB search
        else{
            std::list<new_addr_type>::iterator findIter = std::find(tag_array->begin(), tag_array->end(), key);
        //    if(m_page_manager->check_wrong_ownership(addr, accessor)){
        //        m_stat->tlb_fault++;
        //        m_stat->tlb_fault_app[accessor]++;
        //        return TLB_FAULT;
        //    }
        //    else if(findIter != tag_array->end())
            if(findIter != tag_array->end())
            {
                tag_array->remove(key);
                tag_array->push_front(key); // insert at MRU
                m_stat->tlb_hit++;
                m_stat->small_tlb_hit++;
                app->tlb_hit_app++;
                app->small_tlb_hit_app++;
                app->tlb_hit_app_epoch++;

                app->epoch_accesses++;
                app->epoch_hit++;

                app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
                app->wid_epoch_hit[(mf->get_sid()*100)+mf->get_wid()]++;

                return TLB_HIT;
            }
            else
            {
                m_stat->tlb_miss++;
                app->tlb_miss_app++;
                m_miss++;

                app->epoch_accesses++;
                app->epoch_miss++;

                app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
                app->wid_epoch_miss[(mf->get_sid()*100)+mf->get_wid()]++;

    //            return TLB_MISS;
                tlb_request_status tlb_status = m_shared_tlb->probe(addr,accessor, mf);
                if(tlb_status == TLB_MISS)
                {

                    //New miss, add to the tracker
    //                miss_tracker[mf->get_appID()]->push_front(key);
    //                miss_tracker_count[mf->get_appID()]->insert(std::pair<new_addr_type,unsigned>(key,(unsigned)1));
    //                miss_tracker_timestamp[mf->get_appID()]->push_front(gpu_sim_cycle + gpu_tot_sim_cycle);


                    m_shared_tlb->miss_queue->insert(key);
                    app->concurrent_tracker++;

                    //Statistics
                    m_stat->tlb_current_concurrent_serviced = m_shared_tlb->miss_queue->size();
                    //Update fraction of time core spent on this concurrent TLB misses
                    if(m_shared_tlb->miss_queue->size() > 199)
                        m_stat->tlb_concurrent_total_time[199] += (gpu_sim_cycle+gpu_tot_sim_cycle - last_update_concurrent_stat);
                    else
                        m_stat->tlb_concurrent_total_time[m_shared_tlb->miss_queue->size()] += (gpu_sim_cycle+gpu_tot_sim_cycle - last_update_concurrent_stat);

                    if(m_shared_tlb->miss_queue->size() > 199) {
                      assert(mf->get_appID() == accessor && "modified data on wrong app, should be mf->get_appID() app");
                      app->tlb_concurrent_total_time_app[199] += (gpu_sim_cycle+gpu_tot_sim_cycle -
                          last_update_concurrent_stat);
                    } else {
                      assert(mf->get_appID() == accessor && "modified data on wrong app, should be mf->get_appID() app");
                      app->tlb_concurrent_total_time_app[m_shared_tlb->miss_queue->size()] +=
                          (gpu_sim_cycle+gpu_tot_sim_cycle - last_update_concurrent_stat);
                    }

                    last_update_concurrent_stat = gpu_sim_cycle+gpu_tot_sim_cycle;

                    if(m_shared_tlb->miss_queue->size() > m_stat->tlb_concurrent_serviced) //Update max concurrent TLB
                        m_stat->tlb_concurrent_serviced = m_shared_tlb->miss_queue->size();


                    if(m_stat->tlb_concurrent_max < m_shared_tlb->miss_queue->size())
                        m_stat->tlb_concurrent_max = m_shared_tlb->miss_queue->size();
                    if(app->tlb_concurrent_max_app < m_shared_tlb->miss_queue->size())
                        app->tlb_concurrent_max_app = m_shared_tlb->miss_queue->size();

                    app->tlb_miss_app_epoch++;
                }
                else
                {
                    app->tlb_hit_app_epoch++;
                }
                return tlb_status;
            }
        } //End of small page TLB search
    }
    else//L2 TLB
    {



//        if(COALESCE_DEBUG) printf("Probe into L2 TLBs for VA = %x, appID = %d\n", mf->get_addr(), accessor);
        app->addr_mapping.insert(searched_key); 
        //While flushing
        if(m_config->TLB_flush_enable && (gpu_sim_cycle+gpu_tot_sim_cycle < L2_ready_cycle))
        {
            return TLB_MISS;
        }
        // While invalidating
        if((gpu_sim_cycle + gpu_tot_sim_cycle) < ready_cycle)
        {
            return TLB_FLUSH;
        }

        m_access++;
        m_stat->tlb2_access++;
        app->tlb2_access_app++;

        //First probe
        //unsigned index = addr & (m_entries-1);
        // Set key and search index
        unsigned key;
        unsigned index;
        // Probe large vs. small page TLB
        if(to_large_page) //Large page
        {
            key = searched_key / (*(m_config->page_sizes))[m_config->page_sizes->size()-2]; //Get the searched key
        }
        else
        {
            key = searched_key; //small page
        }
        if(m_config->tlb_core_index)
            index = to_large_page?(mf->get_sid() ^ key ) & (m_entries_large-1):(mf->get_sid() ^ key ) & (m_entries-1);
        else
            index = to_large_page?key & (m_entries_large-1):key & (m_entries-1);

        if(to_large_page){
            if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("L2 TLB probe for set = %d, Finding for addr = %x (key = %x) content of the L2 TLB set # is {",index, addr, key);
            for(std::list<new_addr_type>::iterator findIter = large_l2_tag_array[index]->begin(); findIter != large_l2_tag_array[index]->end();findIter++)
            {
                if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            }
            if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("}\n");
        }
        else{
            if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("L2 TLB probe for set = %d, Finding for addr = %x (key = %x) content of the large page L2 TLB set # is {",index, addr, key);
            for(std::list<new_addr_type>::iterator findIter = l2_tag_array[index]->begin(); findIter != l2_tag_array[index]->end();findIter++)
            {
                if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("%x (%x), ", *findIter, ((*findIter)*(*(m_config->page_sizes))[m_config->page_sizes->size()-1])/(*(m_config->page_sizes))[m_config->page_sizes->size()-2]);
            }
            if(HUGEPAGE_DEBUG || MERGE_DEBUG) printf("}\n");
        }


        std::list<new_addr_type>::iterator findIter;
        if(to_large_page) findIter = std::find(large_l2_tag_array[index]->begin(), large_l2_tag_array[index]->end(), key);
        else findIter = std::find(l2_tag_array[index]->begin(), l2_tag_array[index]->end(), key);
        if(to_large_page && findIter != large_l2_tag_array[index]->end()) //Found in the large page TLB
        {
            (*tlb_entry_sharer)[key].insert(mf->get_wid()); //Put this wid as one of the sharer of this TLB entry

            large_l2_tag_array[index]->remove(key);
            large_l2_tag_array[index]->push_front(key); // insert at MRU
            m_stat->tlb2_hit++;
            m_stat->large_tlb2_hit++;
            app->tlb2_hit_app++;
            app->large_tlb2_hit_app++;

            app->epoch_accesses++;
            app->epoch_hit++;

            app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
            app->wid_epoch_hit[(mf->get_sid()*100)+mf->get_wid()]++;

            if(MERGE_DEBUG && (m_stat->large_tlb2_hit + m_stat->large_tlb2_miss)>0) printf("L2 Large TLB HIT for set = %d, Finding for addr = %x (key = %x). Large TLB hit rate = %f\n",index, addr, key, (float)m_stat->large_tlb2_hit/(float)(m_stat->large_tlb2_hit + m_stat->large_tlb2_miss));



            return TLB_HIT;


        }
        else if(!to_large_page && findIter != l2_tag_array[index]->end()) //Found in the small page TLB
        {
            (*tlb_entry_sharer)[key].insert(mf->get_wid()); //Put this wid as one of the sharer of this TLB entry



            l2_tag_array[index]->remove(key);
            l2_tag_array[index]->push_front(key); // insert at MRU
            m_stat->tlb2_hit++;
            m_stat->small_tlb2_hit++;
            app->tlb2_hit_app++;
            app->small_tlb2_hit_app++;

            app->epoch_accesses++;
            app->epoch_hit++;

            app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
            app->wid_epoch_hit[(mf->get_sid()*100)+mf->get_wid()]++;
            if(MERGE_DEBUG) printf("L2 Small TLB HIT for set = %d, Finding for addr = %x (key = %x)\n",index, addr, key);

            return TLB_HIT;
        }
        else //Not found
        {
            m_stat->tlb2_miss++;
            app->tlb2_miss_app++;
            m_miss++;

            if(to_large_page) m_stat->large_tlb2_miss++;
            else m_stat->small_tlb2_miss++;
            if(to_large_page) app->large_tlb2_miss_app++;
            else app->small_tlb2_miss_app++;

            app->epoch_accesses++;
            app-> epoch_miss++;

            app->wid_epoch_accesses[(mf->get_sid()*100)+mf->get_wid()]++;
            app->wid_epoch_miss[(mf->get_sid()*100)+mf->get_wid()]++;


            //fill(addr,accessor); // Fill in the case of a miss
            if(MERGE_DEBUG && (m_stat->tlb2_hit + m_stat->tlb2_miss) > 0) printf("L2 TLB MISS for set = %d, Finding for addr = %x (key = %x) Current Miss Rate = %f\n",index, addr, key, (float)m_stat->tlb2_miss/(float)(m_stat->tlb2_hit + m_stat->tlb2_miss));
            return TLB_MISS;
        }
    }
}


