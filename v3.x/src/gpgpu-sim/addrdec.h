// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../option_parser.h"

#ifndef PT_SPACE
#define PT_SPACE 987654321
#endif

#ifndef ADDRDEC_H
#define ADDRDEC_H

#include "../abstract_hardware_model.h"


class mmu;
struct memory_config;

struct addrdec_t {
   void print( FILE *fp ) const;
    
   unsigned chip;
   unsigned bk;
   unsigned row;
   unsigned col;
   unsigned burst;

   unsigned subarray;

   unsigned sub_partition; 
};

class linear_to_raw_address_translation {
public:
   linear_to_raw_address_translation();
   void addrdec_setoption(option_parser_t opp);
   void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel, memory_config * config); 

   // accessors
//   void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const; 
   bool addrdec_tlx(new_addr_type in_addr, addrdec_t *tlx, unsigned appID, unsigned level, bool isRead) const; 
//   new_addr_type partition_address( new_addr_type addr ) const;
   new_addr_type partition_address( new_addr_type in_addr, unsigned appID, unsigned level, bool isRead) const;

   unsigned m_app1_channels, m_app2_channels, m_app3_channels;
   unsigned m_app1_banks, m_app2_banks, m_app3_banks;

   void set_mmu(mmu * page_manager);

private:
   void addrdec_parseoption(const char *option);
   void sweep_test() const; // sanity check to ensure no overlapping

   enum {
      CHIP  = 0,
      BK    = 1,
      ROW   = 2,
      COL   = 3,
      BURST = 4,
      SUBARRAY = 5,
      N_ADDRDEC
   };

   const char *addrdec_option;
   int gpgpu_mem_address_mask;
   bool run_test; 

   mmu * m_mmu;

   int ADDR_CHIP_S;
   memory_config * m_config;
   unsigned char addrdec_mklow[N_ADDRDEC];
   unsigned char addrdec_mkhigh[N_ADDRDEC];
   new_addr_type addrdec_mask[N_ADDRDEC];
   new_addr_type sub_partition_id_mask; 

   unsigned int m_subarray_bits;

   unsigned int gap;
   int m_n_channel;
   int m_n_sub_partition_in_channel; 
   int m_n_bk; //number of banks
   int m_n_bkgrp; //number of bank groups
};

#endif
