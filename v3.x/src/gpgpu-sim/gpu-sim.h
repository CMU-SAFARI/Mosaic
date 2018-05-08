// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include "../option_parser.h"
#include "../abstract_hardware_model.h"
#include "../trace.h"
#include "addrdec.h"
#include "shader.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <stdio.h>
#include <vector>
#include "memory_owner.h"
#include <map>
#include "App.h"


// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT  0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP  0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

enum dram_ctrl_t {
   DRAM_FIFO=0,
   DRAM_FRFCFS=1
};

struct power_config {
	power_config()
	{
		m_valid = true;
	}
	void init()
	{

        // initialize file name if it is not set
        time_t curr_time;
        time(&curr_time);
        char *date = ctime(&curr_time);
        char *s = date;
        while (*s) {
            if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
            if (*s == '\n' || *s == '\r' ) *s = 0;
            s++;
        }
        char buf1[1024];
        snprintf(buf1,1024,"gpgpusim_power_report__%s.log",date);
        g_power_filename = strdup(buf1);
        char buf2[1024];
        snprintf(buf2,1024,"gpgpusim_power_trace_report__%s.log.gz",date);
        g_power_trace_filename = strdup(buf2);
        char buf3[1024];
        snprintf(buf3,1024,"gpgpusim_metric_trace_report__%s.log.gz",date);
        g_metric_trace_filename = strdup(buf3);
        char buf4[1024];
        snprintf(buf4,1024,"gpgpusim_steady_state_tracking_report__%s.log.gz",date);
        g_steady_state_tracking_filename = strdup(buf4);

        if(g_steady_power_levels_enabled){
            sscanf(gpu_steady_state_definition,"%lf:%lf", &gpu_steady_power_deviation,&gpu_steady_min_period);
        }

        //NOTE: After changing the nonlinear model to only scaling idle core,
        //NOTE: The min_inc_per_active_sm is not used any more
		if (g_use_nonlinear_model)
		    sscanf(gpu_nonlinear_model_config,"%lf:%lf", &gpu_idle_core_power,&gpu_min_inc_per_active_sm);

	}
	void reg_options(class OptionParser * opp);

	char *g_power_config_name;

	bool m_valid;
    bool g_power_simulation_enabled;
    bool g_power_trace_enabled;
    bool g_steady_power_levels_enabled;
    bool g_power_per_cycle_dump;
    bool g_power_simulator_debug;
    char *g_power_filename;
    char *g_power_trace_filename;
    char *g_metric_trace_filename;
    char * g_steady_state_tracking_filename;
    int g_power_trace_zlevel;
    char * gpu_steady_state_definition;
    double gpu_steady_power_deviation;
    double gpu_steady_min_period;

    //Nonlinear power model
    bool g_use_nonlinear_model;
    char * gpu_nonlinear_model_config;
    double gpu_idle_core_power;
    double gpu_min_inc_per_active_sm;


};



struct memory_config {

   memory_config()
   {
       m_valid = false;
       gpgpu_dram_timing_opt=NULL;
       gpgpu_dram_subarray_timing_opt=NULL;
       gpgpu_L2_queue_config=NULL;
       page_size_list = NULL;
   }
   void init()
   {
      assert(gpgpu_dram_timing_opt);
      if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
         // dram timing option in ordered variables (legacy)
         // Disabling bank groups if their values are not specified
         nbkgrp = 1;
         tCCDL = 0;
         tRTPL = 0;
         sscanf(gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                &nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tCDLR,&tWR,&nbkgrp,&tCCDL,&tRTPL);
      } else {
         // named dram timing options (unordered)
         option_parser_t dram_opp = option_parser_create(); 

         option_parser_register(dram_opp, "nbk",  OPT_UINT32, &nbk,   "number of banks", ""); 
         option_parser_register(dram_opp, "CCD",  OPT_UINT32, &tCCD,  "column to column delay", ""); 
         option_parser_register(dram_opp, "RRD",  OPT_UINT32, &tRRD,  "minimal delay between activation of rows in different banks", ""); 
         option_parser_register(dram_opp, "RCD",  OPT_UINT32, &tRCD,  "row to column delay", ""); 
         option_parser_register(dram_opp, "RAS",  OPT_UINT32, &tRAS,  "time needed to activate row", ""); 
         option_parser_register(dram_opp, "RP",   OPT_UINT32, &tRP,   "time needed to precharge (deactivate) row", ""); 
         option_parser_register(dram_opp, "RC",   OPT_UINT32, &tRC,   "row cycle time", ""); 
         option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR, "switching from write to read (changes tWTR)", ""); 
         option_parser_register(dram_opp, "WR",   OPT_UINT32, &tWR,   "last data-in to row precharge", ""); 

         option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency", ""); 
         option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency", ""); 

         //Disabling bank groups if their values are not specified
         option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp, "number of bank groups", "1"); 
         option_parser_register(dram_opp, "CCDL",   OPT_UINT32, &tCCDL,  "column to column delay between accesses to different bank groups", "0"); 
         option_parser_register(dram_opp, "RTPL",   OPT_UINT32, &tRTPL,  "read to precharge delay between accesses to different bank groups", "0"); 

         option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;"); 
         fprintf(stdout, "DRAM Timing Options:\n"); 
         option_parser_print(dram_opp, stdout); 
         option_parser_destroy(dram_opp); 
      }


      assert(gpgpu_dram_subarray_timing_opt);
      if (strchr(gpgpu_dram_subarray_timing_opt, '=') == NULL) {
         // dram timing option in ordered variables (legacy)
         // Disabling bank groups if their values are not specified
         num_subarray = 1;
         sCCDL = 0;
         sRTPL = 0;
         sscanf(gpgpu_dram_subarray_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                &num_subarray,&sCCD,&sRRD,&sRCD,&sRAS,&sRP,&sRC,&sCL,&sWL,&sCDLR,&sWR,&sCCDL,&sRTPL);
      } else {
         // named dram timing options (unordered)
         option_parser_t dram_opp = option_parser_create(); 

         option_parser_register(dram_opp, "nsa",  OPT_UINT32, &num_subarray,   "number of subarrays", ""); 
         option_parser_register(dram_opp, "sCCD",  OPT_UINT32, &sCCD,  "column to column delay", ""); 
         option_parser_register(dram_opp, "sRRD",  OPT_UINT32, &sRRD,  "minimal delay between activation of rows in different banks", ""); 
         option_parser_register(dram_opp, "sRCD",  OPT_UINT32, &sRCD,  "row to column delay", ""); 
         option_parser_register(dram_opp, "sRAS",  OPT_UINT32, &sRAS,  "time needed to activate row", ""); 
         option_parser_register(dram_opp, "sRP",   OPT_UINT32, &sRP,   "time needed to precharge (deactivate) row", ""); 
         option_parser_register(dram_opp, "sRC",   OPT_UINT32, &sRC,   "row cycle time", ""); 
         option_parser_register(dram_opp, "sCDLR", OPT_UINT32, &sCDLR, "switching from write to read (changes tWTR)", ""); 
         option_parser_register(dram_opp, "sWR",   OPT_UINT32, &sWR,   "last data-in to row precharge", ""); 

         option_parser_register(dram_opp, "sCL", OPT_UINT32, &sCL, "CAS latency", ""); 
         option_parser_register(dram_opp, "sWL", OPT_UINT32, &sWL, "Write latency", ""); 
         option_parser_register(dram_opp, "sCCDL",   OPT_UINT32, &sCCDL,  "column to column delay between accesses to different bank groups", "0"); 
         option_parser_register(dram_opp, "sRTPL",   OPT_UINT32, &sRTPL,  "read to precharge delay between accesses to different bank groups", "0"); 

         option_parser_delimited_string(dram_opp, gpgpu_dram_subarray_timing_opt, "=:;"); 
         fprintf(stdout, "Subarray Timing Options:\n"); 
         option_parser_print(dram_opp, stdout); 
         fprintf(stdout, "Done parsing Subarray Timing Options:\n"); 
         option_parser_destroy(dram_opp); 
      }

      int nbkt = nbk/nbkgrp;
      unsigned i;
      for (i=0; nbkt>0; i++) {
          nbkt = nbkt>>1;
      }
      bk_tag_length = i;
      assert(nbkgrp>0 && "Number of bank groups cannot be zero");
      tRCDWR = tRCD-(WL+1);
      tRTW = (CL+(BL/data_command_freq_ratio)+2-WL);
      tWTR = (WL+(BL/data_command_freq_ratio)+tCDLR); 
      tWTP = (WL+(BL/data_command_freq_ratio)+tWR);
      dram_atom_size = BL * busW * gpu_n_mem_per_ctrlr; // burst length x bus width x # chips per partition 

      assert( m_n_sub_partition_per_memory_channel > 0 ); 
      assert( (nbk % m_n_sub_partition_per_memory_channel == 0) 
              && "Number of DRAM banks must be a perfect multiple of memory sub partition"); 
      m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel; 
      fprintf(stdout, "Total number of memory sub partition = %u\n", m_n_mem_sub_partition); 

      m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel, this);
      m_L2_config.init(&m_address_mapping);


      m_valid = true;
      icnt_flit_size = 32; // Default 32


      //If we want tlb_level tied to va_mask
      tlb_levels = 0;
      std::string mask(va_mask);
      for(int i=0;i<mask.size();i++)
      {
          if(tlb_levels < (mask[i] - '0'))
              tlb_levels = mask[i] - '0';
      }

      page_sizes = new std::vector<unsigned>;
      page_sizes->push_back(DRAM_size); //Set DRAM size as the biggest page
      std::stringstream strm(page_size_list);
      std::string temp;
      while(std::getline(strm, temp, ':')){
          page_sizes->push_back(std::stoul(temp));
      }
      //tlb_levels = page_sizes->size(); If we want to have page table walk strictly tied to page_size list
      //parse_page_size;
      printf("Done parsing page_size list, the list is: [");
      for(std::vector<unsigned>::const_iterator itr = page_sizes->begin(); itr != page_sizes->end(); itr++)
      {
          printf("%u, ",*itr);
          base_page_size = *itr;
          page_size = LOGB2(*itr);
      }
      printf("]. Number of TLB levels = %d, base_page_size = %d\n", tlb_levels, base_page_size);
      

      printf("Done parsing DRAM options");
   }
   void reg_options(class OptionParser * opp);

   void set_mmu(mmu * page_manager)
   {
       m_address_mapping.set_mmu(page_manager);
   }

   bool m_valid;
   mutable l2_cache_config m_L2_config;
   bool m_L2_texure_only;

   char *gpgpu_dram_timing_opt;
   char *gpgpu_dram_subarray_timing_opt;
   char *gpgpu_L2_queue_config;
   bool l2_ideal;
   unsigned gpgpu_frfcfs_dram_sched_queue_size;
   unsigned gpgpu_dram_return_queue_size;
   
   int gpgpu_num_groups; //new
   int gpu_sms; //Number of SMs, taken from gpu-sim->config
   int gpu_char;
  //int count_tlp; 
   enum dram_ctrl_t scheduler_type;
   bool gpgpu_memlatency_stat;
   unsigned m_n_mem;
   unsigned m_n_sub_partition_per_memory_channel;
   unsigned m_n_mem_sub_partition;
   unsigned gpu_n_mem_per_ctrlr;

   unsigned rop_latency;
   unsigned dram_latency;

   // DRAM parameters

   unsigned tCCDL;  //column to column delay when bank groups are enabled
   unsigned tRTPL;  //read to precharge delay when bank groups are enabled for GDDR5 this is identical to RTPS, if for other DRAM this is different, you will need to split them in two

   unsigned tCCD;   //column to column delay
   unsigned tRRD;   //minimal time required between activation of rows in different banks
   unsigned tRCD;   //row to column delay - time required to activate a row before a read
   unsigned tRCDWR; //row to column delay for a write command
   unsigned tRAS;   //time needed to activate row
   unsigned tRP;    //row precharge ie. deactivate row
   unsigned tRC;    //row cycle time ie. precharge current, then activate different row
   unsigned tCDLR;  //Last data-in to Read command (switching from write to read)
   unsigned tWR;    //Last data-in to Row precharge 

   unsigned CL;     //CAS latency
   unsigned WL;     //WRITE latency
   unsigned BL;     //Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
   unsigned tRTW;   //time to switch from read to write
   unsigned tWTR;   //time to switch from write to read 
   unsigned tWTP;   //time to switch from write to precharge in the same bank
   unsigned busW;

   //Subarrays -- should be the same as DRAM timing except MASA
   unsigned sCCDL;  //column to column delay when bank groups are enabled
   unsigned sRTPL;  //read to precharge delay when bank groups are enabled for GDDR5 this is identical to RTPS, if for other DRAM this is different, you will need to split them in two

   unsigned sCCD;   //column to column delay
   unsigned sRRD;   //minimal time required between activation of rows in different banks
   unsigned sRCD;   //row to column delay - time required to activate a row before a read
   unsigned sRCDWR; //row to column delay for a write command
   unsigned sRAS;   //time needed to activate row
   unsigned sRP;    //row precharge ie. deactivate row
   unsigned sRC;    //row cycle time ie. precharge current, then activate different row
   unsigned sCDLR;  //Last data-in to Read command (switching from write to read)
   unsigned sWR;    //Last data-in to Row precharge 

   unsigned sCL;     //CAS latency
   unsigned sWL;     //WRITE latency
   unsigned sRTW;   //time to switch from read to write
   unsigned sWTR;   //time to switch from write to read 
   unsigned sWTP;   //time to switch from write to precharge in the same bank


   unsigned nbkgrp; // number of bank groups (has to be power of 2)
   unsigned bk_tag_length; //number of bits that define a bank inside a bank group

   unsigned nbk;

   unsigned data_command_freq_ratio; // frequency ratio between DRAM data bus and command bus (2 for GDDR3, 4 for GDDR5)
   unsigned dram_atom_size; // number of bytes transferred per read or write command 

   linear_to_raw_address_translation m_address_mapping;

   unsigned icnt_flit_size;

   // Page and TLB related config

   bool capture_VA;
   char *va_trace_file;
   bool capture_VA_map;
   char *pt_file;

   char * va_mask;
   unsigned page_mapping_policy;

   //Page coalescing paramenters
   unsigned enable_page_coalescing;
   unsigned enable_compaction;
   unsigned enable_rctest; //If true, randomly send Rowclone commands for debugging
   unsigned compaction_probe_cycle;
   unsigned compaction_probe_additional_latency;
   unsigned enable_costly_coalesce;
   unsigned page_coalesce_locality_thres;
   unsigned page_coalesce_hotness_thres;
   unsigned page_coalesce_lower_thres_offset;

   unsigned page_stat_update_cycle;
   unsigned demotion_check_cycle;
   unsigned l1_tlb_invalidate_latency;
   unsigned l2_tlb_invalidate_latency;

   bool pw_cache_enable;
   //Copy and zero timing
   unsigned interSA_latency;
   unsigned intraSA_latency;
   unsigned lisa_latency;
   unsigned RCintraSA_latency;
   unsigned RCzero_latency;
   unsigned zero_latency;
   unsigned interBank_latency;
   unsigned RCpsm_latency;
   unsigned RC_enabled;
   unsigned LISA_enabled;
   unsigned MASA_enabled;
   unsigned SALP_enabled;

   unsigned num_subarray;
   unsigned enable_subarray;
   unsigned channel_partition;
   unsigned bank_partition;
   unsigned app1_channel;
   unsigned app2_channel;
   unsigned app3_channel;
   unsigned app1_bank;
   unsigned app2_bank;
   unsigned app3_bank;
   unsigned subarray_partition;
   unsigned get_shader_avail_warp_stat;
   unsigned pw_cache_latency;
   unsigned pw_cache_num_ports;
   unsigned TLB_flush_enable;
   unsigned TLB_flush_freq;
   unsigned tlb_pw_cache_entries;
   unsigned tlb_L1_flush_cycles;
   unsigned tlb_L2_flush_cycles;
   unsigned tlb_bypass_cache_flush_cycles;
   unsigned tlb_replacement_policy;
   unsigned tlb_replacement_hash_size;
   unsigned tlb_replacement_high_threshold;
   unsigned tlb_pw_cache_ways;
   unsigned enable_PCIe;
   unsigned page_queue_size;
   unsigned tlb_size;
   unsigned tlb_size_large;
   unsigned tlb_prefetch;
   unsigned tlb_prefetch_set;
   unsigned tlb_prefetch_buffer_size;
   unsigned tlb_core_index;
   unsigned tlb_victim_size;
   unsigned tlb_victim_size_large;
   unsigned tlb_lookup_bypass;
   unsigned tlb_miss_rate_ratio;
   unsigned tlb_bypass_initial_tokens;
   unsigned tlb_stat_resets;
   unsigned max_tlb_miss;
   unsigned tlb_prio_max_level;
   unsigned tlb_enable;
   unsigned tlb_bypass_enabled;
   unsigned tlb_bypass_level;
   unsigned data_cache_bypass_threshold;
   unsigned l2_tlb_entries;
   unsigned use_old_Alloc;
   unsigned l2_tlb_entries_large;
   unsigned l2_tlb_ways;
   unsigned l2_tlb_ways_large;
   unsigned l2_tlb_way_reset;
   unsigned tlb_cache_part;
   unsigned max_tlb_cache_depth;
   unsigned tlb_levels;
   unsigned tlb_fixed_latency_enabled;
   unsigned tlb_fixed_latency;
   unsigned page_size;
   unsigned base_page_size;
   char * page_size_list;
   std::vector<unsigned> * page_sizes;
   unsigned dram_row_size;
   unsigned DRAM_size;
   float DRAM_fragmentation;
   unsigned DRAM_fragmentation_pages_per_frame;
   unsigned tlb_dram_aware;
   unsigned dram_switch_factor;
   unsigned dram_switch_max;
   unsigned dram_switch_threshold;
   unsigned dram_high_prio_chance;
   
   unsigned dram_scheduling_policy;
   unsigned dram_always_prioritize_app;
   unsigned tlb_high_prio_level;
   unsigned max_DRAM_high_prio_wait;
   bool dram_batch;
   unsigned max_DRAM_high_prio_combo;

   unsigned epoch_length;
   char * epoch_file;
   bool epoch_enabled;

//   unsigned tlb_template_bits;
//   unsigned page_appID_bits;
//   unsigned page_tlb_level_bits;

   unsigned page_evict_policy;
   unsigned page_partition_policy;

   unsigned PCIe_queue_size;
   unsigned page_transfer_time;
//   unsigned row_hit_time; // This should be set automatically, not through the config file

};

// global counters and flags (please try not to add to this list!!!)
extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;
extern int count_tlp;
extern bool g_interactive_debugger_enabled;
class gpgpu_sim_config : public power_config, public gpgpu_functional_sim_config {
public:
    gpgpu_sim_config() { m_valid = false; }
    void reg_options(class OptionParser * opp);
    void init() 
    {
        gpu_stat_sample_freq = 10000;
        gpu_runtime_stat_flag = 0;
        sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq, &gpu_runtime_stat_flag);
        m_shader_config.init();
        ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
        m_memory_config.init();
        init_clock_domains(); 
        power_config::init();
        Trace::init();


        // initialize file name if it is not set 
        time_t curr_time;
        time(&curr_time);
        char *date = ctime(&curr_time);
        char *s = date;
        while (*s) {
            if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
            if (*s == '\n' || *s == '\r' ) *s = 0;
            s++;
        }
        char buf[1024];
        snprintf(buf,1024,"gpgpusim_visualizer__%s.log.gz",date);
        g_visualizer_filename = strdup(buf);

        m_valid=true;
    }

    void set_mmu(mmu * page_manager){
        m_memory_config.set_mmu(page_manager);
    }

    memory_config * get_mem_config(){return &m_memory_config;}

    unsigned num_shader() const { return m_shader_config.num_shader(); }
    unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
    unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }

private:
    void init_clock_domains(void ); 


    bool m_valid;
    shader_core_config m_shader_config;
    memory_config m_memory_config;
    // clock domains - frequency
    double core_freq;
    double icnt_freq;
    double dram_freq;
    double l2_freq;
    double core_period;
    double icnt_period;
    double dram_period;
    double l2_period;

    // GPGPU-Sim timing model options
    unsigned gpu_max_cycle_opt;
    unsigned gpu_max_insn_opt;
    unsigned gpu_max_cta_opt;
    char *gpgpu_runtime_stat;
    bool  gpgpu_flush_l1_cache;
    bool  gpgpu_flush_l2_cache;
    bool  gpu_deadlock_detect;
    int   gpgpu_frfcfs_dram_sched_queue_size; 
    int   gpgpu_cflog_interval;
    char * gpgpu_clock_domains;
    unsigned max_concurrent_kernel;

    // visualizer
    bool  g_visualizer_enabled;
    char *g_visualizer_filename;
    int   g_visualizer_zlevel;


    // statistics collection
    int gpu_stat_sample_freq;
    int gpu_runtime_stat_flag;



    unsigned long long liveness_message_freq; 

    friend class gpgpu_sim;
};

class gpgpu_sim : public gpgpu_t {
public:
   gpgpu_sim( const gpgpu_sim_config &config );

   mmu * m_page_manager;
   tlb_tag_array * m_shared_tlb;

   void set_prop( struct cudaDeviceProp *prop );

   void launch( kernel_info_t *kinfo );
   bool can_start_kernel();
   unsigned finished_kernel();
   void set_kernel_done( kernel_info_t *kernel );

   void init();
   void cycle();
   bool active(); 
   void print_stats();
   void update_stats();
   void deadlock_check();

   void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc );

   int shared_mem_size() const;
   int num_registers_per_core() const;
   int wrp_size() const;
   int shader_clock() const;
   const struct cudaDeviceProp *get_prop() const;
   enum divergence_support_t simd_model() const; 

   unsigned threads_per_core() const;
   bool get_more_cta_left() const;
   kernel_info_t *select_kernel();

   const gpgpu_sim_config &get_config() const { return m_config; }
   void gpu_print_stat_file(FILE* file);
   void app_cache_flush(int i);
   
   unsigned long long get_gpu_insn_max();
   void dump_pipeline( int mask, int s, int m ) const;

   //The next three functions added to be used by the functional simulation function
   
   //! Get shader core configuration
   /*!
    * Returning the configuration of the shader core, used by the functional simulation only so far
    */
   const struct shader_core_config * getShaderCoreConfig();
   
   
   //! Get shader core Memory Configuration
    /*!
    * Returning the memory configuration of the shader core, used by the functional simulation only so far
    */
   const struct memory_config * getMemoryConfig();
   
   
   //! Get shader core SIMT cluster
   /*!
    * Returning the cluster of of the shader core, used by the functional simulation so far
    */
    simt_core_cluster * getSIMTCluster();


private:
   // clocks
   void reinit_clock_domains(void);
   int  next_clock_domain(void);
   void issue_block2core();
   void L2c_print_cache_stat(FILE *fout) const;
   void L2c_print_cache_stat_periodic(unsigned long long gpu_ins,
       std::vector<uint64_t> gpu_ins_s) const;
   void print_dram_stats(FILE *fout) const;
   void shader_print_runtime_stat( FILE *fout );
   void shader_print_l1_miss_stat( FILE *fout ) const;
   void shader_print_cache_stats( FILE *fout ) const;
   void shader_print_scheduler_stat( FILE* fout, bool print_dynamic_info ) const;
   void visualizer_printstat();
   void print_shader_cycle_distro( FILE *fout ) const;

   void gpgpu_debug();

///// data /////

   class simt_core_cluster **m_cluster;
   class memory_partition_unit **m_memory_partition_unit;
   class memory_sub_partition **m_memory_sub_partition;

   std::vector<kernel_info_t*> m_running_kernels;
   unsigned m_last_issued_kernel;

   std::list<unsigned> m_finished_kernel;
   unsigned m_total_cta_launched;
   unsigned m_last_cluster_issue;
   float * average_pipeline_duty_cycle;
   float * active_sms;
   // time of next rising edge 
   double core_time;
   double icnt_time;
   double dram_time;
   double l2_time;

   // debug
   bool gpu_deadlock;
   bool max_insn_struck; // new

   //// configuration parameters ////
   const gpgpu_sim_config &m_config;
  
   const struct cudaDeviceProp     *m_cuda_properties;
   const struct shader_core_config *m_shader_config;
   const struct memory_config      *m_memory_config;

   // stats
   class shader_core_stats  *m_shader_stats;
   class memory_stats_t     *m_memory_stats;
   class power_stat_t *m_power_stats;
   class gpgpu_sim_wrapper *m_gpgpusim_wrapper;
   unsigned long long  gpu_tot_issued_cta;
   unsigned long long  last_gpu_sim_insn;

   unsigned long long  last_liveness_message_time; 

   std::map<std::string, FuncCache> m_special_cache_config;

   std::vector<std::string> m_executed_kernel_names; //< names of kernel for stat printout 
   std::vector<unsigned> m_executed_kernel_uids; //< uids of kernel launches for stat printout
   std::string executed_kernel_info_string(); //< format the kernel information into a string for stat printout
   void clear_executed_kernel_info(); //< clear the kernel information after stat printout

public:
   unsigned long long  gpu_sim_insn;
   unsigned long long  gpu_tot_sim_insn;
   unsigned long long  gpu_sim_insn_last_update;
   unsigned gpu_sim_insn_last_update_sid;
 
   unsigned long long  gpu_sim_insn_per_core[1000]; //why 1000

   FuncCache get_cache_config(std::string kernel_name);
   void set_cache_config(std::string kernel_name, FuncCache cacheConfig );
   bool has_special_cache_config(std::string kernel_name);
   void change_cache_config(FuncCache cache_config);
   void set_cache_config(std::string kernel_name);

};

#endif
