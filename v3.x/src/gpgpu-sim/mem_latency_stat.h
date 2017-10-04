// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>
#include <zlib.h>
#include <map>

class memory_stats_t {
public:
   memory_stats_t( unsigned n_shader, 
                   const struct shader_core_config *shader_config, 
                   const struct memory_config *mem_config );

   void init();
   uint64_t memlatstat_done( class mem_fetch *mf );
   void memlatstat_read_done( class mem_fetch *mf );
   void memlatstat_dram_access( class mem_fetch *mf );
   void memlatstat_icnt2mem_pop( class mem_fetch *mf);
   void memlatstat_lat_pw();
   void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

   void visualizer_print( gzFile visualizer_file );
   //new
   void memlatstat_print_file( unsigned n_mem, unsigned gpu_mem_n_bk, FILE *fout );
   //int get_latency(int s_id);

   unsigned m_n_shader;

   const struct shader_core_config *m_shader_config;
   const struct memory_config *m_memory_config;

   uint64_t max_mrq_latency;
   uint64_t max_dq_latency;
   uint64_t max_mf_latency;
   uint64_t tlb_max_mf_latency;
   uint64_t max_icnt2mem_latency;
   uint64_t max_icnt2sh_latency;
   uint64_t mrq_lat_table[32];
   uint64_t dq_lat_table[32];
   uint64_t mf_lat_table[32];
   uint64_t icnt2mem_lat_table[24];
   uint64_t icnt2sh_lat_table[24];
   uint64_t mf_lat_pw_table[32]; //table storing values of mf latency Per Window
   uint64_t mf_total_num_lat_pw = 0;
   uint64_t tlb_mf_total_num_lat_pw = 0;
   uint64_t dram_app_switch;
   uint64_t max_warps;
   uint64_t mf_total_tot_lat_pw = 0;

   uint64_t tlb_mf_total_tot_lat_pw = 0;
   uint64_t mf_total_total_lat = 0;
   uint64_t tlb_mf_total_total_lat = 0;
   uint64_t high_prio_queue_count;

   uint64_t coalesced_tried;
   uint64_t coalesced_succeed;
   uint64_t coalesced_noinval_succeed;
   uint64_t coalesced_fail;

   //Number of pages not being used when being coalesced
   uint64_t max_bloat;
   uint64_t num_coalesce;
   uint64_t pt_space_size;

   uint64_t tlb_bypassed;
   uint64_t tlb_bypassed_level[10];
   uint64_t tlb_bypassed_core[200];
   uint64_t tlb_level_accesses[10];
   uint64_t tlb_level_hits[10];
   uint64_t tlb_level_misses[10];
   uint64_t tlb_level_fails[10];


   uint64_t l2_cache_accesses = 0;
   uint64_t l2_cache_hits = 0;
   uint64_t l2_cache_misses = 0;

   float TLBL1_sharer_avg[200];
   uint64_t TLBL1_total_unique_addr[200];
   float TLBL1_sharer_var[200];
   uint64_t TLBL1_sharer_max[200];
   uint64_t TLBL1_sharer_min[200];

   uint64_t TLB_bypass_cache_flush_stalled;
   uint64_t TLB_L1_flush_stalled[200];
   uint64_t TLB_L2_flush_stalled;

   float TLBL2_sharer_avg;
   uint64_t TLBL2_total_unique_addr;
   float TLBL2_sharer_var;
   uint64_t TLBL2_sharer_max;
   uint64_t TLBL2_sharer_min;

   uint64_t ** mf_total_lat_table; //mf latency sums[dram chip id][bank id]
   uint64_t ** mf_max_lat_table; //mf latency sums[dram chip id][bank id]
   uint64_t total_num_mfs = 0;
   uint64_t tlb_total_num_mfs = 0;
   uint64_t ***bankwrites; //bankwrites[shader id][dram chip id][bank id]
   uint64_t ***bankreads; //bankreads[shader id][dram chip id][bank id]
   uint64_t **totalbankblocked; //number of cycles banks are blocked [dram chip id][bank id]
   uint64_t **totalbankwrites; //bankwrites[dram chip id][bank id]
   uint64_t **totalbankreads; //bankreads[dram chip id][bank id]
   uint64_t **totalbankaccesses; //bankaccesses[dram chip id][bank id]
   uint64_t *num_MCBs_accessed; //tracks how many memory controllers are accessed whenever any thread in a warp misses in cache
   uint64_t *position_of_mrq_chosen; //position of mrq in m_queue chosen
   
   uint64_t ***mem_access_type_stats; // dram access type classification

   // TLB-related

   uint64_t totalL1TLBMissesAll;
   uint64_t totalL2TLBMissesAll;

   uint64_t sched_from_normal_prio;
   uint64_t sched_from_high_prio;
   uint64_t sched_from_special_prio;
   uint64_t DRAM_normal_prio;
   uint64_t DRAM_high_prio;
   uint64_t DRAM_special_prio;
   uint64_t drain_reset;
   uint64_t total_combo;

   uint64_t ** totalL1TLBMisses; //totalL1TLBMisses[shader id][app id]
   uint64_t ** totalL2TLBMisses; //totalL2TLBMisses[shader id][app id]
   uint64_t ** totalTLBMissesCausedAccess; //totalTLBMissesCausedAccess[shader id][app id]

   // L2 cache stats
   uint64_t *L2_cbtoL2length;
   uint64_t *L2_cbtoL2writelength;
   uint64_t *L2_L2tocblength;
   uint64_t *L2_dramtoL2length;
   uint64_t *L2_dramtoL2writelength;
   uint64_t *L2_L2todramlength;

   // DRAM access row locality stats 
   uint64_t **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
   
   uint64_t ** num_activates;
   uint64_t ** row_access;

   uint64_t ** num_activates_w;
   uint64_t ** row_access_w;

   uint64_t **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
   uint64_t **max_servicetime2samerow; //max_servicetime2samerow[dram chip id][bank id]

   // Power stats
   uint64_t total_n_access;
   uint64_t total_n_reads;
   uint64_t total_n_writes;
};

#endif /*MEM_LATENCY_STAT_H*/
