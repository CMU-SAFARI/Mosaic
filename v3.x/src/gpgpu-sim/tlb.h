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

#ifndef GPU_TLB_H
#define GPU_TLB_H

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <set>

#include "App.h"

#define GLOBAL_SPACE 0
#define TEXTURE_SPACE 1
#define SHARED_SPACE 2
#define OTHER_SPACE 3

class mem_fetch;

class mmu;

class shader_core_stats;

class memory_stats_t;

class memory_partition_unit;

struct memory_config;

class dram_cmd;

enum tlb_request_status {
  TLB_HIT,
  TLB_HIT_RESERVED,
  TLB_PENDING,
  TLB_MISS,
  TLB_FLUSH,
  TLB_FAULT,
  TLB_PAGE_FAULT,
  TLB_NOFAULT,
  TLB_VIOLATION,
  TLB_MEMORY_RESERVATION_FAIL,
  COALESCED_TLB
};

class page_table;

class tlb_tag_array {
public:
  tlb_tag_array(const memory_config * config, shader_core_stats * stat, mmu* page_manager,
      tlb_tag_array * shared_tlb, int core_id);
  tlb_tag_array(const memory_config * config, shader_core_stats * stat, mmu* page_manager,
      bool isL2TLB, memory_stats_t * mem_stat, memory_partition_unit ** mem_partition);
  tlb_tag_array(tlb_tag_array * tlb, unsigned level);
  ~tlb_tag_array();

  std::map<new_addr_type, std::set<unsigned>> * tlb_entry_sharer; //Track how many warpID are using this entry, the set contains unique warpID

  memory_partition_unit ** m_memory_partition;

  std::list<new_addr_type> * tag_array;
  std::list<new_addr_type> * large_tag_array;
  std::list<new_addr_type> ** l2_tag_array;
  std::list<new_addr_type> ** large_l2_tag_array;
  std::set<new_addr_type> * miss_queue;
  std::list<new_addr_type> * bypass_done_queue;
  std::list<new_addr_type> * large_bypass_done_queue;
  std::list<new_addr_type> * prefetch_queue[256];
  std::list<new_addr_type> * prefetch_buffer[256];
  std::list<unsigned long long> * prefetch_queue_time[256];
  std::set<unsigned> stride_set;

  std::map<appid_t, std::set<new_addr_type>*> promoted_pages; //List of coalesced pages, key = page_base_addr | appID

  //For multiple page size metadata tracking
  std::map<new_addr_type, unsigned long long> * hotness_track;
  std::map<new_addr_type, std::set<int>*> * warpID_track;
  std::map<new_addr_type, std::set<new_addr_type>*> * locality_track;

  //Update page-related statistics used by resize_page
  void update_page_stats(mem_fetch *mf);
  void shift_page_stats();
  //This gets called once in a while to clear statistics
  void clear_page_stats();

  unsigned prev_stride;
  unsigned stride;
  unsigned long long last_update_concurrent_stat;

  unsigned long long tlb_occupancy_last_updated; //Don't think we need this
  unsigned long long tlb_demotion_last_checked;

  std::list<mem_fetch*> remove_list; //list of mf* that is done with pt_walk routine and wait to be deleted

  void set_l1_tlb(int coreID, tlb_tag_array * l1); //Called from L1 TLB to L2 TLB. So that L2 TLB can send invalidate commands to L1 TLBs
  tlb_tag_array ** l1_tlb;

  void invalidate_entries(appid_t appID, new_addr_type base_entry);

  unsigned long long ready_cycle;

  void check_threshold(mem_fetch * mf);

  unsigned get_concurrent_tlb_app(appid_t appID);

  void add_mf_to_done(mem_fetch * mf); //Add to remove_list for deletion later

  void remove_done_mf();

  int promotion(new_addr_type va, appid_t appID); //Called from the MMU

  void handle_pending_prefetch();

  page_table * root;

  mmu * m_page_manager;

  shader_core_stats * m_stat;

  unsigned long long last_shifted_page_stat;

  const memory_config * m_config;
  enum tlb_request_status probe(new_addr_type addr, appid_t accessor, mem_fetch * mf);


  bool bypass_dir; //True == more aggressive, false = less aggressive

  void fill(new_addr_type addr, mem_fetch * mf);
  int remove_miss_queue(new_addr_type addr, appid_t appID);
  void fill(new_addr_type addr, appid_t accessor, mem_fetch * mf); // L2 TLB fill
  void l2_fill(new_addr_type addr, appid_t accessor, mem_fetch * mf); // L2 TLB fill

  bool bypass_TLB(mem_fetch * mf, appid_t appID);

  new_addr_type get_tlbreq_addr(mem_fetch * mf);

  // Get the actual physical address of a TLB access
  new_addr_type assign_pa(new_addr_type original, unsigned tlb_level, appid_t appID);
  // Parse the assiciated memory access (location of the page dir/table)
  new_addr_type parse_tlb_access(mem_fetch * mf);

  unsigned flush(appid_t appID); // flash invalidate all entries
  unsigned flush(); // flash invalidate all entries, for non-triggering slice

  unsigned m_access;
  unsigned m_miss;
  unsigned m_ways;
  unsigned m_ways_large;
  unsigned m_entries;
  unsigned m_entries_large;
  bool isL2TLB;

  tlb_tag_array * m_next_tlb;
  tlb_tag_array * m_shared_tlb;
  unsigned tlb_level;

  //Resize the page belong to base_addr, appID. MMU will automatically detect the size
  void resize_page(appid_t appID, new_addr_type base_addr, new_addr_type original_addr);
  bool demote_page(new_addr_type va, appid_t appID);

  tlb_tag_array * get_shared_tlb() {
    return m_shared_tlb;
  }
  memory_stats_t * get_mem_stat() {
    return m_mem_stats;
  }

  // For TLB flush, only set on shared TLB (so that all L1 TLB can sync)
  unsigned long long L1_ready_cycle[200];
  unsigned long long L2_ready_cycle;
  unsigned long long bypass_cache_ready_cycle;

  void update_L1_ready_cycle(appid_t appID, unsigned long long time);

  bool previous_bypass_enabled;

  bool * replacement_hashed_map;

  float total_tokens;
  int m_core_id; //For L1 private TLB

  void print_statistics();


  bool pw_cache_access(mem_fetch * mf);
  void pw_cache_fill(mem_fetch * mf);
  new_addr_type process_pwcache_key(mem_fetch * mf); //Get the pwcache fill and access tags

  unsigned pw_cache_entries;
  unsigned pw_cache_ways;

  std::list<new_addr_type> ** pw_cache;

  std::list<mem_fetch*> * pw_cache_lat_queue;
  std::list<unsigned long long> * pw_cache_lat_time;

  void remove_pw_cache_lat_queue();

  memory_stats_t * m_mem_stats;

  std::list<mem_fetch*> * static_queue;
  std::list<unsigned long long> * static_queue_time;
  void put_mf_to_static_queue(mem_fetch * mf);
  void clear_static_queue();
  void measure_sharer_stats();

  bool lots_of_sharer(mem_fetch *mf); //Return true if the page MF is accessing has a lot of WID accessing it

};

#endif
