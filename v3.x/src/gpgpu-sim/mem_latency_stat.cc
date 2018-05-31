// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan
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

#include "../abstract_hardware_model.h"
#include "mem_latency_stat.h"
#include "gpu-sim.h"
#include "gpu-misc.h"
#include "gpu-cache.h"
#include "shader.h"
#include "mem_fetch.h"
#include "stat-tool.h"
#include "../cuda-sim/ptx-stats.h"
#include "visualizer.h"
#include "dram.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

memory_stats_t::memory_stats_t( unsigned n_shader, const struct shader_core_config *shader_config,
    const struct memory_config *mem_config ) {
   assert( mem_config->m_valid );
   assert( shader_config->m_valid );

   concurrent_row_access = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   num_activates = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   num_activates_w = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   row_access = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   row_access_w = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   max_conc_access2samerow = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   max_servicetime2samerow = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));

   for (unsigned i = 0; i < mem_config->m_n_mem; i++) {
      concurrent_row_access[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      num_activates[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
	    row_access[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      row_access_w[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      num_activates_w[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      max_conc_access2samerow[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      max_servicetime2samerow[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
   }


   high_prio_queue_count=0;
   dram_app_switch = 0;

   tlb_bypassed = 0;
   max_bloat = 0;
   coalesced_tried = 0;
   coalesced_succeed = 0;
   coalesced_noinval_succeed = 0;
   coalesced_fail = 0;
   num_coalesce = 0;
   pt_space_size = 0;

   for(int i=0; i<10; i++){
      tlb_bypassed_level[i] = 0;
      tlb_level_accesses[i] = 0;
      tlb_level_hits[i] = 0;
      tlb_level_misses[i] = 0;
      tlb_level_fails[i] = 0;
   }
   for(int i=0; i<200; i++){
      tlb_bypassed_core[i] = 0;
      TLBL1_sharer_avg[i] = 0.0;
      TLBL1_total_unique_addr[i] = 0;
      TLBL1_sharer_var[i] = 0.0;
      TLBL1_sharer_max[i] = 0;
      TLBL1_sharer_min[i] = 0;
      TLB_L1_flush_stalled[i] = 0;
   }

   TLBL2_sharer_avg = 0.0;
   TLBL2_total_unique_addr = 0;
   TLBL2_sharer_var = 0.0;
   TLBL2_sharer_max = 0;
   TLBL2_sharer_min = 0;
   TLB_L2_flush_stalled = 0;
   TLB_bypass_cache_flush_stalled = 0;

   m_n_shader=n_shader;
   m_memory_config=mem_config;
   total_n_access=0;
   total_n_reads=0;
   total_n_writes=0;
   max_mrq_latency = 0;
   max_dq_latency = 0;
   max_mf_latency = 0;
   tlb_max_mf_latency = 0;
   max_icnt2mem_latency = 0;
   max_icnt2sh_latency = 0;
   memset(mrq_lat_table, 0, sizeof(unsigned)*32);
   memset(dq_lat_table, 0, sizeof(unsigned)*32);
   memset(mf_lat_table, 0, sizeof(unsigned)*32);
   memset(icnt2mem_lat_table, 0, sizeof(unsigned)*24);
   memset(icnt2sh_lat_table, 0, sizeof(unsigned)*24);
   memset(mf_lat_pw_table, 0, sizeof(unsigned)*32);

   DRAM_normal_prio = 0;
   DRAM_high_prio = 0;
   DRAM_special_prio = 0;
   sched_from_normal_prio = 0;
   sched_from_high_prio = 0;
   sched_from_special_prio = 0;
   drain_reset = 0;
   total_combo = 0;

   max_warps = n_shader * (shader_config->n_thread_per_shader / shader_config->warp_size+1);

//   printf("*** Initializing Memory Statistics ***\n");
   totalbankreads = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   totalbankblocked = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   totalbankwrites = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   totalbankaccesses = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
   mf_total_lat_table = (uint64_t **) calloc(mem_config->m_n_mem, sizeof(uint64_t *));
   mf_max_lat_table = (uint64_t **) calloc(mem_config->m_n_mem, sizeof(uint64_t *));
   bankreads = (uint64_t***) calloc(n_shader, sizeof(uint64_t**));
   bankwrites = (uint64_t***) calloc(n_shader, sizeof(uint64_t**));
   num_MCBs_accessed = (uint64_t*) calloc(mem_config->m_n_mem*mem_config->nbk, sizeof(uint64_t));
   if (mem_config->gpgpu_frfcfs_dram_sched_queue_size) {
      position_of_mrq_chosen = (uint64_t*) calloc(mem_config->gpgpu_frfcfs_dram_sched_queue_size, sizeof(uint64_t));
   } else
      position_of_mrq_chosen = (uint64_t*) calloc(1024, sizeof(uint64_t));
   for (int i=0;i<n_shader ;i++ ) {
      bankreads[i] = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
      bankwrites[i] = (uint64_t**) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
      for (int j=0;j<mem_config->m_n_mem ;j++ ) {
         bankreads[i][j] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
         bankwrites[i][j] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      }
   }

   for (int i=0;i<mem_config->m_n_mem ;i++ ) {
      totalbankreads[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      totalbankblocked[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      totalbankwrites[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      totalbankaccesses[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      mf_total_lat_table[i] = (uint64_t*) calloc(mem_config->nbk, sizeof(uint64_t));
      mf_max_lat_table[i] = (uint64_t *) calloc(mem_config->nbk, sizeof(uint64_t));
   }

   mem_access_type_stats = (uint64_t ***) malloc(NUM_MEM_ACCESS_TYPE * sizeof(uint64_t **));
   for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
      int j;
      mem_access_type_stats[i] = (uint64_t **) calloc(mem_config->m_n_mem, sizeof(uint64_t*));
      for (j=0; (uint64_t) j< mem_config->m_n_mem; j++) {
         mem_access_type_stats[i][j] = (uint64_t *) calloc((mem_config->nbk+1), sizeof(uint64_t*));
      }
   }

   L2_cbtoL2length = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
   L2_cbtoL2writelength = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
   L2_L2tocblength = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
   L2_dramtoL2length = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
   L2_dramtoL2writelength = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
   L2_L2todramlength = (uint64_t*) calloc(mem_config->m_n_mem, sizeof(uint64_t));
}

void memory_stats_t::init() {
   for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
     App* app = App::get_app(App::get_app_id(i));
     app->num_activates_ = (uint64_t**) calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
     app->num_activates_w_ = (uint64_t**) calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
     app->row_access_ = (uint64_t**) calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
     app->row_access_w_ = (uint64_t**) calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
   }

   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      for (int j = 0; j < ConfigOptions::n_apps; j++) {
        App* app = App::get_app(App::get_app_id(j));
        app->num_activates_[i] = (uint64_t*) calloc(m_memory_config->nbk, sizeof(uint64_t));
        app->num_activates_w_[i] = (uint64_t*) calloc(m_memory_config->nbk, sizeof(uint64_t));
        app->row_access_[i] = (uint64_t*) calloc(m_memory_config->nbk, sizeof(uint64_t));
	      app->row_access_w_[i] = (uint64_t*) calloc(m_memory_config->nbk, sizeof(uint64_t));
      }
   }
}

// record the total latency
uint64_t memory_stats_t::memlatstat_done(mem_fetch *mf )
{
  unsigned mf_latency;
  mf_latency = (gpu_sim_cycle + gpu_tot_sim_cycle) - mf->get_timestamp();
  mf_total_num_lat_pw++;
  mf_total_tot_lat_pw += mf_latency;

  if (mf->get_tlb_depth_count() > 0) {
    tlb_mf_total_num_lat_pw++;
    tlb_mf_total_tot_lat_pw += mf_latency;
  } // TODO remove zeroth entry

  if (mf->get_sid() != -1) {
    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_sid()));
    app->mf_num_lat_pw++; // the first entry is reserved for something else?
    app->mf_tot_lat_pw++;
    if (mf->get_tlb_depth_count() > 0) {
      app->tlb_mf_num_lat_pw++;
      app->tlb_mf_tot_lat_pw += mf_latency;
    }
  }

  for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
    App* app = App::get_app(App::get_app_id(i));
    app->mflatency = (float) app->mf_total_lat / app->num_mfs;
    app->tlb_num_mfs = (float) app->tlb_mf_total_lat / app->tlb_num_mfs;
  }

  unsigned idx = LOGB2(mf_latency);
  assert(idx < 32);
  mf_lat_table[idx]++;
  shader_mem_lat_log(mf->get_sid(), mf_latency);
  mf_total_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] += mf_latency;

  if (mf_latency > max_mf_latency)
    max_mf_latency = mf_latency;
  if ((mf_latency > tlb_max_mf_latency) && (mf->get_tlb_depth_count() > 0))
    tlb_max_mf_latency = mf_latency;
  return mf_latency;
}

void memory_stats_t::memlatstat_read_done(mem_fetch *mf)
{
   if (m_memory_config->gpgpu_memlatency_stat) {
      unsigned mf_latency = memlatstat_done(mf);
      if (mf_latency > mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk])
         mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] = mf_latency;
      unsigned icnt2sh_latency;
      icnt2sh_latency = (gpu_tot_sim_cycle+gpu_sim_cycle) - mf->get_return_timestamp();
      icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
      if (icnt2sh_latency > max_icnt2sh_latency)
         max_icnt2sh_latency = icnt2sh_latency;
   }
}

void memory_stats_t::memlatstat_dram_access(mem_fetch *mf)
{

   unsigned dram_id = mf->get_tlx_addr().chip;
   unsigned bank = mf->get_tlx_addr().bk;
   if (m_memory_config->gpgpu_memlatency_stat) {
      if (mf->get_is_write()) {
         if ( mf->get_sid() < m_n_shader  ) {   //do not count L2_writebacks here
            bankwrites[mf->get_sid()][dram_id][bank]++;
            shader_mem_acc_log( mf->get_sid(), dram_id, bank, 'w');
         }
         totalbankwrites[dram_id][bank]++;
      } else {
         bankreads[mf->get_sid()][dram_id][bank]++;
         shader_mem_acc_log( mf->get_sid(), dram_id, bank, 'r');
         totalbankreads[dram_id][bank]++;
      }
      mem_access_type_stats[mf->get_access_type()][dram_id][bank]++;
   }
   if (mf->get_pc() != (unsigned)-1)
      ptx_file_line_stats_add_dram_traffic(mf->get_pc(), mf->get_data_size());

}

void memory_stats_t::memlatstat_icnt2mem_pop(mem_fetch *mf)
{
   if (m_memory_config->gpgpu_memlatency_stat) {
      unsigned icnt2mem_latency;
      icnt2mem_latency = (gpu_tot_sim_cycle+gpu_sim_cycle) - mf->get_timestamp();
      icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
      if (icnt2mem_latency > max_icnt2mem_latency)
         max_icnt2mem_latency = icnt2mem_latency;
   }
}

void memory_stats_t::memlatstat_lat_pw() {
  if (mf_total_num_lat_pw > 0 && m_memory_config->gpgpu_memlatency_stat) {
    mf_total_total_lat += mf_total_tot_lat_pw;
    total_num_mfs += mf_total_num_lat_pw;
    mf_lat_pw_table[LOGB2(mf_total_tot_lat_pw / mf_total_num_lat_pw)]++;
    mf_total_tot_lat_pw = 0;
    mf_total_num_lat_pw = 0;
  }
  if (m_memory_config->gpgpu_memlatency_stat) {
    for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
      App* app = App::get_app(App::get_app_id(i));
      app->mf_total_lat += app->mf_tot_lat_pw;
      app->num_mfs += app->mf_num_lat_pw;
      app->mf_tot_lat_pw = 0;
      app->mf_num_lat_pw = 0;
    }
  }

  if (tlb_mf_total_num_lat_pw && m_memory_config->gpgpu_memlatency_stat) {
    tlb_mf_total_total_lat += tlb_mf_total_tot_lat_pw;
    tlb_total_num_mfs += tlb_mf_total_num_lat_pw;
    tlb_mf_total_tot_lat_pw = 0;
    tlb_mf_total_num_lat_pw = 0;
  }
  if (m_memory_config->gpgpu_memlatency_stat) {
    for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
      App* app = App::get_app(App::get_app_id(i));
      app->tlb_mf_total_lat += app->tlb_mf_tot_lat_pw;
      app->tlb_num_mfs += app->tlb_mf_num_lat_pw;
      app->tlb_mf_tot_lat_pw = 0;
      app->tlb_mf_num_lat_pw = 0;
    }
  }
}


void memory_stats_t::memlatstat_print_file( unsigned n_mem, unsigned gpu_mem_n_bk, FILE *fout  )
{
   unsigned i,j,k,l,m;
   unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses, min_chip_accesses;

   if (m_memory_config->gpgpu_memlatency_stat) {
      fprintf(fout,"maxmrqlatency = %d \n", max_mrq_latency);
      fprintf(fout,"maxdqlatency = %d \n", max_dq_latency);
      fprintf(fout,"maxmflatency = %d \n", max_mf_latency);
      fprintf(fout,"high_prio_queue_drain_reset = %d \n", drain_reset);
      fprintf(fout,"average_combo_count = %f \n", (float)total_combo/drain_reset);
      fprintf(fout,"sched_from_normal_prio = %d, DRAM_normal_prio = %d\n",  sched_from_normal_prio, DRAM_normal_prio);
      fprintf(fout,"sched_from_high_prio = %d, DRAM_high_prio = %d\n",  sched_from_high_prio, DRAM_high_prio);
      fprintf(fout,"sched_from_special_prio = %d, DRAM_special_prio = %d\n",  sched_from_special_prio, DRAM_special_prio);



      fprintf(fout,"l1_tlb_stalled_cycles_due_to_flush = {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%ld, ",TLB_L1_flush_stalled[i]);
      fprintf(fout,"}\n");


      fprintf(fout,"l1_tlb_average_number_of_warps_per_entry= {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%f, ",TLBL1_sharer_avg[i]);
      fprintf(fout,"}\n");


      fprintf(fout,"l1_tlb_variance_number_of_warps_per_entry= {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%f, ",TLBL1_sharer_var[i]);
      fprintf(fout,"}\n");


      fprintf(fout,"l1_tlb_max_number_of_warps_per_entry= {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%d, ",TLBL1_sharer_max[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"l1_tlb_min_number_of_warps_per_entry= {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%d, ",TLBL1_sharer_min[i]);
      fprintf(fout,"}\n");


      fprintf(fout,"l1_tlb_number_of_unique_entries= {");
      for(int i=0;i<200;i++)
          fprintf(fout,"%d, ",TLBL1_total_unique_addr[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"l2_tlb_average_number_of_warps_per_entry = %f\n", TLBL2_sharer_avg);
      fprintf(fout,"l2_tlb_variance_number_of_warps_per_entry = %f\n", TLBL2_sharer_var);
      fprintf(fout,"l2_tlb_max_number_of_warps_per_entry = %d\n", TLBL2_sharer_max);
      fprintf(fout,"l2_tlb_min_number_of_warps_per_entry = %d\n", TLBL2_sharer_min);
      fprintf(fout,"l2_tlb_number_of_unique_entries = %d\n", TLBL2_total_unique_addr);

      fprintf(fout,"l2_tlb_stalled_cycles_due_to_flush = %lu\n", TLB_L2_flush_stalled);
      fprintf(fout,"tlb_bypass_cache_stalled_cycles_due_to_flush = %lu\n", TLB_bypass_cache_flush_stalled);

      fprintf(fout,"l2_cache_accesses = %lu\n", l2_cache_accesses);
      fprintf(fout,"l2_cache_accesses_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->l2_cache_accesses_app);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"l2_cache_hits = %lu\n", l2_cache_hits);
      fprintf(fout,"l2_cache_hits_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->l2_cache_hits_app);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"l2_cache_misses = %lu\n", l2_cache_misses);
      fprintf(fout,"l2_cache_misses_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ", app->l2_cache_misses_app);
      }
      fprintf(fout,"}\n");


      fprintf(fout,"number_of_coalesced_attempts = %lu\n", coalesced_tried );
      fprintf(fout,"number_of_coalseced_attempts_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->coalesced_tried_app);
      }
      fprintf(fout,"}\n");


      fprintf(fout,"number_of_coalesced_noinval_successes = %lu\n", coalesced_noinval_succeed );
      fprintf(fout,"number_of_coalseced_noinval_successes_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->coalesced_noinval_succeed_app);
      }
      fprintf(fout,"}\n");



      fprintf(fout,"number_of_coalesced_successes = %d\n", coalesced_succeed );
      fprintf(fout,"number_of_coalseced_successes_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->coalesced_succeed_app);
      }
      fprintf(fout,"}\n");


      fprintf(fout,"number_of_coalesced_fails = %d\n", coalesced_fail );
      fprintf(fout,"number_of_coalseced_fails_app= {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->coalesced_fail_app);
      }
      fprintf(fout,"}\n");


      fprintf(fout,"number_of_coalesced_pages = %lu\n", num_coalesce );
      fprintf(fout,"peak_bloated_pages = %lu\n", max_bloat );
      fprintf(fout,"size_of_page_tables = %lu\n", pt_space_size );

      fprintf(fout,"tlb_bypassed_data_cache = %lu\n", tlb_bypassed);

      fprintf(fout,"tlb_bypassed_data_cache_app = {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%lu, ",app->tlb_bypassed_app);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_peak_occupancy_app = {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%f, ",app->tlb_occupancy_peak);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_end_occupancy_app = {");
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%f, ",app->tlb_occupancy_end);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_avg_occupancy_app = {");
      for(unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"%f, ",app->tlb_occupancy_avg);
      }
      fprintf(fout,"}\n");

      fprintf(fout,"dram_req_high_queue_count = %d \n", high_prio_queue_count);
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++)  {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"dram_req_high_queue_count_app%u = %lu \n", i, app->high_prio_queue_count_app);
      }

      fprintf(fout,"dram_priority_switch_triggered = %d \n", dram_app_switch);
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++)  {
        App* app = App::get_app(App::get_app_id(i));
        fprintf(fout,"dram_priority_for_app%u = %lu \n", i, app->dram_prioritized_cycles_app);
      }

      fprintf(fout,"tlb_level_cache_accesses = {");
      for(int i=0;i<10;i++)
          fprintf(fout,"%lu, ",tlb_level_accesses[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_level_cache_hits = {");
      for(int i=0;i<10;i++)
          fprintf(fout,"%lu, ",tlb_level_hits[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_level_cache_misses = {");
      for(int i=0;i<10;i++)
          fprintf(fout,"%lu, ",tlb_level_misses[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_level_cache_fails = {");
      for(int i=0;i<10;i++)
          fprintf(fout,"%lu, ",tlb_level_fails[i]);
      fprintf(fout,"}\n");

      fprintf(fout,"tlb_level_cache_hit_rate = {");
      for(int i=0;i<10;i++)
          fprintf(fout,"%f, ",(float)tlb_level_hits[i]/(float)(tlb_level_hits[i]+tlb_level_misses[i]));
      fprintf(fout,"}\n");


      if (total_num_mfs) {
         fprintf(fout,"averagemflatency = %lld \n", mf_total_total_lat / total_num_mfs);
      }
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        if (app->num_mfs) // avoid divide by zero
          fprintf(fout,"averagemflatency_%u = %lu \n", i, app->mf_total_lat / app->num_mfs);
      }

      if (tlb_total_num_mfs) {
         fprintf(fout,"averageTLBmflatency = %lld \n", tlb_mf_total_total_lat / tlb_total_num_mfs);
      }
      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        if (app->tlb_num_mfs) // avoid divide by zero
          fprintf(fout,"averageTLBmflatency_%u = %lu \n", i, app->tlb_mf_total_lat / app->tlb_num_mfs);
      }

      for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        if (app->mrq_num) {
          fprintf(fout, "averagemrqlatency_%u = %lu \n", i, app->mrqs_latency / app->mrq_num);
        }
      }


      fprintf(fout,"max_icnt2mem_latency = %d \n", max_icnt2mem_latency);
      fprintf(fout,"max_icnt2sh_latency = %d \n", max_icnt2sh_latency);
      fprintf(fout,"mrq_lat_table:");
      for (i=0; i< 32; i++) {
         fprintf(fout,"%d \t", mrq_lat_table[i]);
      }
      fprintf(fout,"\n");
      fprintf(fout,"dq_lat_table:");
      for (i=0; i< 32; i++) {
         fprintf(fout,"%d \t", dq_lat_table[i]);
      }
      fprintf(fout,"\n");
      fprintf(fout,"mf_lat_table:");
      for (i=0; i< 32; i++) {
         fprintf(fout,"%d \t", mf_lat_table[i]);
      }
      fprintf(fout,"\n");
      fprintf(fout,"icnt2mem_lat_table:");
      for (i=0; i< 24; i++) {
         fprintf(fout,"%d \t", icnt2mem_lat_table[i]);
      }
      fprintf(fout,"\n");
      fprintf(fout,"icnt2sh_lat_table:");
      for (i=0; i< 24; i++) {
         fprintf(fout,"%d \t", icnt2sh_lat_table[i]);
      }
      fprintf(fout,"\n");
      fprintf(fout,"mf_lat_pw_table:");
      for (i=0; i< 32; i++) {
         fprintf(fout,"%d \t", mf_lat_pw_table[i]);
      }
      fprintf(fout,"\n");

      /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
      fprintf(fout,"maximum concurrent accesses to same row:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            fprintf(fout,"%9d ",max_conc_access2samerow[i][j]);
         }
         fprintf(fout,"\n");
      }

      /*MAXIMUM SERVICE TIME TO SAME ROW*/
      fprintf(fout,"maximum service time to same row:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            fprintf(fout,"%9d ",max_servicetime2samerow[i][j]);
         }
         fprintf(fout,"\n");
      }

      /*AVERAGE ROW ACCESSES PER ACTIVATE*/
      int total_row_accesses = 0;
      int total_num_activates = 0;
      std::vector<int> total_row_accesses_(ConfigOptions::n_apps, 0);
      std::vector<int> total_num_activates_(ConfigOptions::n_apps, 0);


      fprintf(fout,"average row accesses per activate:\n");
      for (i = 0; i < n_mem; i++) {
        fprintf(fout, "dram[%d]: ", i);
        for (j = 0; j < gpu_mem_n_bk; j++) {
          total_row_accesses += row_access[i][j];
          total_num_activates += num_activates[i][j];
          for (int k = 0; k < ConfigOptions::n_apps; k++) {
            App* app = App::get_app(App::get_app_id(k));
            total_row_accesses_[k] += app->row_access_[i][j];
            total_num_activates_[k] += app->num_activates_[i][j];
          }
          fprintf(fout, "%9f ", (float) row_access[i][j] / num_activates[i][j]);
        }
        fprintf(fout, "\n");
      }


      fprintf(fout,"average row locality = %d/%d = %f\n", total_row_accesses, total_num_activates,
          (float)total_row_accesses/total_num_activates);
      for (int i = 0; i < ConfigOptions::n_apps; i++) {
        fprintf(fout,"average row locality_1 = %d/%d = %f\n", total_row_accesses_[i],
            total_num_activates_[i], (float) total_row_accesses_[i] / total_num_activates_[i]);
      }

	  /*MEMORY ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      fprintf(fout,"number of total memory accesses made:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            l = totalbankaccesses[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            fprintf(fout,"%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         fprintf(fout,"\n");
      }
      fprintf(fout,"total accesses: %d\n", k);
      if (min_bank_accesses)
         fprintf(fout,"bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         fprintf(fout,"min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         fprintf(fout,"chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         fprintf(fout,"min_chip_accesses = 0!\n");

      /*READ ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      fprintf(fout,"number of total read accesses:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout, "dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            l = totalbankreads[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            fprintf(fout,"%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         fprintf(fout,"\n");
      }
      fprintf(fout,"total reads: %d\n", k);
      if (min_bank_accesses)
         fprintf(fout,"bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         fprintf(fout,"min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         fprintf(fout,"chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         fprintf(fout,"min_chip_accesses = 0!\n");

      fprintf(fout,"number of cycles banks are stalled = {");
      for (int temp1=0;temp1<n_mem ;temp1++ ) {
          for (int temp2=0;temp2<gpu_mem_n_bk;temp2++ ) {
              fprintf(fout,"%d,",totalbankblocked[temp1][temp2]);
          }
      }
      fprintf(fout,"}\n");

      /*WRITE ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      fprintf(fout,"number of total write accesses:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            l = totalbankwrites[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            fprintf(fout,"%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         fprintf(fout,"\n");
      }
      fprintf(fout,"total reads: %d\n", k);
      if (min_bank_accesses)
         fprintf(fout,"bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         fprintf(fout,"min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         fprintf(fout,"chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         fprintf(fout,"min_chip_accesses = 0!\n");


      /*AVERAGE MF LATENCY PER BANK*/
      fprintf(fout,"average mf latency per bank:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            k = totalbankwrites[i][j] + totalbankreads[i][j];
            if (k)
               fprintf(fout,"%10lld", mf_total_lat_table[i][j] / k);
            else
               fprintf(fout,"    none  ");
         }
        fprintf(fout,"\n");
      }

      /*MAXIMUM MF LATENCY PER BANK*/
      fprintf(fout,"maximum mf latency per bank:\n");
      for (i=0;i<n_mem ;i++ ) {
         fprintf(fout,"dram[%d]: ", i);
         for (j=0;j<gpu_mem_n_bk;j++ ) {
            fprintf(fout,"%10d", mf_max_lat_table[i][j]);
         }
         fprintf(fout,"\n");
      }
   }

   if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
      fprintf(fout,"\nNumber of Memory Banks Accessed per Memory Operation per Warp (from 0):\n");
      uint64_t accum_MCBs_accessed = 0;
      uint64_t tot_mem_ops_per_warp = 0;
      for (i=0;i< n_mem*gpu_mem_n_bk ; i++ ) {
         accum_MCBs_accessed += i*num_MCBs_accessed[i];
         tot_mem_ops_per_warp += num_MCBs_accessed[i];
         fprintf(fout,"%d\t", num_MCBs_accessed[i]);
      }

      fprintf(fout,"\nAverage # of Memory Banks Accessed per Memory Operation per Warp=%f\n", (float)accum_MCBs_accessed/tot_mem_ops_per_warp);

      //printf("\nAverage Difference Between First and Last Response from Memory System per warp = ");


      fprintf(fout,"\nposition of mrq chosen\n");

      if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
         j = 1024;
      else
         j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
      k=0;l=0;
      for (i=0;i< j; i++ ) {
         fprintf(fout,"%d\t", position_of_mrq_chosen[i]);
         k += position_of_mrq_chosen[i];
         l += i*position_of_mrq_chosen[i];
      }
      fprintf(fout,"\n");
      fprintf(fout,"\naverage position of mrq chosen = %f\n", (float)l/k);
   }
}

void memory_stats_t::memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk) {
  memlatstat_print_file(n_mem, gpu_mem_n_bk, stdout);
}
