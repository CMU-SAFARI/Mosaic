// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan
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

#ifndef dram_sched_h_INCLUDED
#define dram_sched_h_INCLUDED

#include "dram.h"
#include "shader.h"
#include "gpu-sim.h"
#include "gpu-misc.h"
#include <list>
#include <map>
#include <vector>

class frfcfs_scheduler {
public:
  frfcfs_scheduler(const memory_config *config, dram_t *dm, memory_stats_t *stats,
      tlb_tag_array * shared_tlb);
  void add_req(dram_req_t *req);

  void data_collection(unsigned bank);
  dram_req_t *schedule(unsigned bank, unsigned curr_row);
  void print(FILE *fp);
  unsigned num_pending() const {
    return m_num_pending;
  }

private:
  unsigned set_req_prio_criteria(mem_fetch * mf);

  const memory_config *m_config;
  unsigned *last_core_id; //new
  dram_t *m_dram;
  unsigned m_num_pending;
  std::list<dram_req_t*> *m_queue;
  std::list<dram_req_t*> *m_queue_high;
  std::list<dram_req_t*> *m_queue_special;
  std::map<unsigned, std::list<std::list<dram_req_t*>::iterator> > *m_bins;
  std::list<std::list<dram_req_t*>::iterator> **m_last_row;
  unsigned *curr_row_service_time; //one set of variables for each bank.
  unsigned *row_service_timestamp; //tracks when scheduler began servicing current row
  dram_req_t *pick(std::list<dram_req_t*> * buffers, unsigned curr_row, unsigned bank);

  memory_stats_t *m_stats;

  tlb_tag_array * m_shared_tlb;

  unsigned current_prioritize;
  unsigned current_prioritize_total;
  unsigned epoch_total_concurrent;

  unsigned long long last_switch;

  unsigned m_high_pq_wait_timestamp;

  bool m_high_prio_queue_drain;

  unsigned m_high_pq_combo_count;
  float avg_combo_count;

  void check_high_queue_drain();

};

#endif
