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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "addrdec.h"
#include "../abstract_hardware_model.h"
#include <bitset>
#include "App.h"

enum cache_event {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT
};



enum mf_type {
   READ_REQUEST = 0,
   WRITE_REQUEST,
   READ_REPLY, // send to shader
   WRITE_ACK
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

//extern int gpu_sms;
//extern int gpu_mode3;
//extern int gpu_sms_app1;

class dram_t;


class tlb_tag_array;

class memory_config;

class data_cache;

class memory_partition_unit;

class mem_fetch {
public:
    mem_fetch( const mem_access_t &access, 
               const warp_inst_t *inst,
               unsigned ctrl_size, 
               unsigned wid,
               unsigned sid, 
               unsigned tpc, 
               const class memory_config *config );
    mem_fetch( mem_fetch * mf );


   ~mem_fetch();

   void set_status( enum mem_fetch_status status, unsigned long long cycle );
   void set_reply() 
   { 
       assert( m_access.get_type() != L1_WRBK_ACC && m_access.get_type() != L2_WRBK_ACC );
       if( m_type==READ_REQUEST ) {
           assert( !get_is_write() );
           m_type = READ_REPLY;
       } else if( m_type == WRITE_REQUEST ) {
           assert( get_is_write() );
           m_type = WRITE_ACK;
       }
   }
   void do_atomic();

   dram_t * m_dram_slice;

   dram_t * get_DRAM(){return m_dram_slice;}
   void set_DRAM(dram_t * dram){m_dram_slice = dram;}

   void print( FILE *fp, bool print_inst = true ) const;

   const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
   unsigned get_data_size() const { return m_data_size; }
   void     set_data_size( unsigned size ) { m_data_size=size; }
   unsigned get_ctrl_size() const { return m_ctrl_size; }
   unsigned size() const { return m_data_size+m_ctrl_size; }
   bool is_write();
//   bool is_write() {return m_access.is_write();}
   void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
   //void set_addr(new_addr_type addr) { m_access.set_addr(((uint64_t)m_appID << (uint64_t)48) | addr); }
   new_addr_type get_addr() const { return ((uint64_t)m_appID << (uint64_t)48 ) | (uint64_t)m_access.get_addr(); }
   //new_addr_type get_addr() const { return m_access.get_addr(); }
   void set_original_addr(new_addr_type addr) { m_original_addr = addr; }
   new_addr_type get_original_addr() const { return m_original_addr; }
   new_addr_type get_partition_addr() const { return m_partition_addr; }
   unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
   bool     get_is_write() const;
//   bool     get_is_write() const { return m_access.is_write(); }
   unsigned get_request_uid() const { return m_request_uid; }
   unsigned get_sid() const { return m_sid; }
   unsigned get_tpc() const { return m_tpc; }
   unsigned get_wid() const { return m_wid; }
   bool istexture() const;
   bool isconst() const;
   enum mf_type get_type() const { return m_type; }
   bool isatomic() const;

   void set_page_fault(bool val);
   void set_tlb_miss(bool val){m_tlb_miss = val;}
   bool get_page_fault(){return m_page_fault;}
   bool get_tlb_miss(){return m_tlb_miss;}

   void probe_pw_cache(mem_fetch * mf);

   unsigned get_appID(){return m_appID;}

   mem_access_t get_access(){return m_access;}

   unsigned get_subarray();

   void set_return_timestamp( unsigned t ) { m_timestamp2=t; }
   void set_icnt_receive_time( unsigned t ) { m_icnt_receive_time=t; }
   unsigned get_timestamp() const { return m_timestamp; }
   unsigned get_return_timestamp() const { return m_timestamp2; }
   unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

   enum mem_access_type get_access_type() const { return m_access.get_type(); }
   const active_mask_t& get_access_warp_mask() const { return m_access.get_warp_mask(); }
   mem_access_byte_mask_t get_access_byte_mask() const { return m_access.get_byte_mask(); }

   address_type get_pc() const { return m_inst.empty()?-1:m_inst.pc; }
   const warp_inst_t &get_inst() { return m_inst; }
   enum mem_fetch_status get_status() const { return m_status; }

   const memory_config *get_mem_config(){return m_mem_config;}

   mem_fetch * get_parent_tlb_request(){return m_parent_tlb_request;}
   void set_parent_tlb_request(mem_fetch * mf){m_parent_tlb_request = mf;}


   mem_fetch * get_child_tlb_request(){return m_child;}
   void set_child_tlb_request(mem_fetch * mf){m_child = mf;}

   unsigned get_tlb_depth_count(){return m_tlb_depth_count;}
   void set_tlb_depth_count(unsigned depth){m_tlb_depth_count = depth;}

   void done_tlb_req(mem_fetch * mf);
   void pt_walk(mem_fetch * mf);

   void set_tlb(tlb_tag_array * tlb){m_tlb = tlb;}
   tlb_tag_array * get_tlb(){return m_tlb;}
   
   unsigned accum_dram_access; //If this is a TLB-miss, how many DRAM request it generates

   unsigned get_num_flits(bool simt_to_mem);

   bool isL2TLBMiss;

   bool beenThroughL1; //Used to identify L2 requests in gpu-cache.cc and gpu-cache.h

   bool m_tlb_related_req;

   bool been_through_tlb;

   void set_cache(data_cache * cache);
   void set_memory_partition_unit(memory_partition_unit * m_unit){m_memory_partition_unit = m_unit;}
   data_cache * get_cache();
   std::list<cache_event> & get_events(){return m_events;}
   void set_events(std::list<cache_event> &ev){m_events = ev;}

   bool check_bypass(mem_fetch *mf);

   bool bypass_L2;

   bool pwcache_hit; 
   bool pwcache_done; // For PW cache hit request, is this done through the latency queue

private:
   // request source information
   std::list<cache_event> m_events;
   unsigned m_request_uid;
   unsigned m_sid;
   unsigned m_tpc;
   unsigned m_wid;


   bool m_page_fault;
   bool m_tlb_miss;

   appid_t m_appID;
   unsigned m_tlb_depth_count;

   // where is this request now?
   enum mem_fetch_status m_status;
   unsigned long long m_status_change;

   // request type, address, size, mask
   mem_access_t m_access;
   unsigned m_data_size; // how much data is being written
   unsigned m_ctrl_size; // how big would all this meta data be in hardware (does not necessarily match actual size of mem_fetch)
   new_addr_type m_partition_addr; // linear physical address *within* dram partition (partition bank select bits squeezed out)
   addrdec_t m_raw_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)
   enum mf_type m_type;

   // statistics
   unsigned m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
   unsigned m_timestamp2; // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed onto icnt to shader; only used for reads
   unsigned m_icnt_receive_time; // set to gpu_sim_cycle + interconnect_latency when fixed icnt latency mode is enabled

   // requesting instruction (put last so mem_fetch prints nicer in gdb)
   warp_inst_t m_inst;

   static unsigned sm_next_mf_request_uid;

   const class memory_config *m_mem_config;
   unsigned icnt_flit_size;

   new_addr_type m_original_addr;

   mem_fetch * m_parent_tlb_request;
   mem_fetch * m_child;

   tlb_tag_array * m_tlb;

   data_cache * m_cache;

   memory_partition_unit * m_memory_partition_unit;

};

#endif
