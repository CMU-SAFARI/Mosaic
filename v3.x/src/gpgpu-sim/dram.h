// Copyright (c) 2009-2011, Tor M. Aamodt, Ivan Sham, Ali Bakhoda, 
// George L. Yuan, Wilson W.L. Fung
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

#ifndef DRAM_H
#define DRAM_H

#include "delayqueue.h"
#include <set>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include "memory_owner.h"
#include "App.h"

#define READ 'R'  //define read and write states
#define WRITE 'W'
#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'
#define BANK_BLOCKED 'B' //LISA modify data within the bank
#define BANK_BLOCKED_PENDING 'P' //Sent inter-bank command, waiting for both banks to be idle
#define BANK_TRANSFER 'T' //Row-clone PSM Mode, blocking two banks (source, target)

#define LISA_COPY 0
#define InterSA_COPY 1
#define IntraSA_COPY 2
#define RC_PSM 3
#define RC_IntraSA 4
#define RC_zero 5
#define Channel_copy 6
#define COPY 7 //Generic copy, will figure the best timing from from/to_bank and subarrays
#define TARGET 8 //Target bank, got the command from other bank
#define ZERO 9 //Baseline zeroing a page
#define SCAN 10 //SCAN request (works similar to a read request, but does not queue in DRAM request queue from the GPU core side

#define DRAM_CMD 12345678 //Special ID for DRAM command, so that addrdec can parse dram command address

class page;

class tlb_tag_array;

class Hub;

//class virtual_address_translation;

class dram_cmd{
public: 
   dram_cmd(int cmd, int from_bk, int to_bk, int from_ch, int to_ch, int from_subarray, int to_sa, int pk_size, int app_ID, const struct memory_config * config);
   dram_cmd(int cmd, page * from_page, page * to_page, const struct memory_config * config);
   int command;
   int from_bank;
   int from_channel;
   int from_sa;
   int to_bank;
   int to_channel;
   int to_sa;
   int size;
   int appID;
   const struct memory_config *m_config;
};



class dram_req_t {
public:
   dram_req_t( class mem_fetch *data );

   unsigned int row;
   unsigned int col;
   unsigned int bk;
   unsigned int nbytes;
   unsigned int txbytes;
   unsigned int dqbytes;
   unsigned int age;
   unsigned int timestamp;
   unsigned char rw;    //is the request a read or a write?
   unsigned long long int addr;
   unsigned int insertion_time;
   class mem_fetch * data;
};

struct bankgrp_t
{
	unsigned int CCDLc;
	unsigned int RTPLc;
};

struct bank_t
{
   unsigned int RCDc;
   unsigned int RCDWRc;
   unsigned int RASc;
   unsigned int RPc;
   unsigned int RCc;
   unsigned int WTPc; // write to precharge
   unsigned int RTPc; // read to precharge

   unsigned char rw;    //is the bank reading or writing?
   unsigned char state; //is the bank active or idle?
   unsigned int curr_row;
   unsigned int curr_subarray;

   dram_req_t *mrq;

   unsigned int n_access;
   unsigned int n_writes;
   unsigned int n_idle;

   //Counter for BANK_BLOCKED and BANK_TRANSFER, if these are zeros, banks should become idle
   unsigned int blocked;
   unsigned int transfer;

   std::list<dram_cmd*> * cmd_queue; 

   unsigned int bkgrpindex;
};

struct mem_fetch;


class dram_t 
{
public:
   dram_t( unsigned int parition_id, const struct memory_config *config, class memory_stats_t *stats, 
           class memory_partition_unit *mp, mmu * page_manager, tlb_tag_array * shared_tlb);


   mmu * m_page_manager;

   tlb_tag_array * m_shared_tlb;

   bool full() const;
   void print( FILE* simFile ) const;
   void visualize() const;
   void print_stat( FILE* simFile );
   unsigned que_length() const; 
   bool returnq_full() const;
   unsigned int queue_limit() const;
   void visualizer_print( gzFile visualizer_file );
   
   unsigned long long m_compaction_last_probed;

   //channel ID is not needed, done in parallel
   unsigned compaction_bank_id;

   //new
   unsigned data_bus_busy; //If data are being copied over channel (from another channel)

   unsigned dram_bwutil();
   unsigned dram_bwutil(appid_t);

   unsigned dram_bwutil_data();
   unsigned dram_bwutil_data(appid_t);
 
   unsigned dram_bwutil_tlb();
   unsigned dram_bwutil_tlb(appid_t);
 
   void set_miss(float m);
   void set_miss_r(appid_t, float m);

   void set_miss_core(float m, unsigned which_core);

   float get_miss();
   float get_miss_core(unsigned i);


   float get_rbl();

   // till here


   class mem_fetch* return_queue_pop();
   class mem_fetch* return_queue_top();
   void push( class mem_fetch *data );
   void cycle();
   void dram_log (int task);

   class memory_partition_unit *m_memory_partition_unit;
   unsigned int id;

   void insert_dram_command(dram_cmd * cmd);

   // Power Model
   void set_dram_power_stats(unsigned &cmd,
								unsigned &activity,
								unsigned &nop,
								unsigned &act,
								unsigned &pre,
								unsigned &rd,
								unsigned &wr,
								unsigned &req) const;

private:
   void scheduler_fifo();
   void scheduler_frfcfs();

   const struct memory_config *m_config;
   
   bankgrp_t **bkgrp;

   bank_t **bk;
   unsigned int prio;

   unsigned int RRDc;
   unsigned int CCDc;
   unsigned int RTWc;   //read to write penalty applies across banks
   unsigned int WTRc;   //write to read penalty applies across banks

   unsigned char rw; //was last request a read or write? (important for RTW, WTR)

   unsigned int pending_writes;

   fifo_pipeline<dram_req_t> *rwq;
   fifo_pipeline<dram_req_t> *mrqq;
   //buffer to hold packets when DRAM processing is over
   //should be filled with dram clock and popped with l2or icnt clock 
   fifo_pipeline<mem_fetch> *returnq;

   std::list<mem_fetch*> wait_list; //Rachata: A queue for TLB-related requests

   unsigned int dram_util_bins[10];
   unsigned int dram_eff_bins[10];
   unsigned int last_n_cmd, last_n_activity, last_bwutil;

   unsigned int n_cmd;
   unsigned int n_activity;
   unsigned int n_nop;
   unsigned int n_act;
   unsigned int n_pre;
   unsigned int n_rd;
   unsigned int n_wr;
   unsigned int n_req;
   unsigned int max_mrqs_temp;

   unsigned int bwutil;
   unsigned int bwutil_data;
   unsigned int bwutil_tlb;
   unsigned int max_mrqs;
   unsigned int ave_mrqs;
   
   
   
   unsigned int bwutil_periodic;
   unsigned int bwutil_periodic_data;
   unsigned int bwutil_periodic_tlb;
   unsigned int n_cmd_blp;
   unsigned int  mem_state_blp;
   unsigned int  mem_state_blp_alarm[32];
   unsigned int  mem_state_blp_ncmd[32];
   unsigned int sanity_read;
   unsigned int sanity_write;
   
   float miss_rate_d;
   float miss_rate_d_core[64];
   
   unsigned int dram_cycles;
   unsigned int dram_cycles_active;

   class frfcfs_scheduler* m_frfcfs_scheduler;

   unsigned int n_cmd_partial;
   unsigned int n_activity_partial;
   unsigned int n_nop_partial; 
   unsigned int n_act_partial; 
   unsigned int n_pre_partial; 
   unsigned int n_req_partial;
   unsigned int ave_mrqs_partial;
   unsigned int bwutil_partial;
   
   

   struct memory_stats_t *m_stats;
   class Stats* mrqq_Dist; //memory request queue inside DRAM  

   friend class frfcfs_scheduler;
};

#endif /*DRAM_H*/
