// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan,
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

#include "gpu-sim.h"
#include "gpu-misc.h"
#include "dram.h"
#include "tlb.h"
#include "mem_latency_stat.h"
#include "dram_sched.h"
#include "mem_fetch.h"
#include "l2cache.h"
#include "../cuda-sim/cuda-sim.h"

extern Hub * gpu_alloc;

extern int gpu_sms; // how many sms

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

#define RC_DEBUG 1
#define MERGE_DEBUG 0

template class fifo_pipeline<mem_fetch> ;
template class fifo_pipeline<dram_req_t> ;

dram_cmd::dram_cmd(int cmd, int from_bk, int to_bk, int from_ch, int to_ch, int from_subarray,
    int to_subarray, int pk_size, appid_t app_ID, const struct memory_config * config) {
  command = cmd;
  from_bank = from_bk;
  from_channel = from_ch;
  from_sa = from_subarray;
  to_sa = to_subarray;
  to_bank = to_bk;
  to_channel = to_ch;
  size = pk_size;
  appID = app_ID;
  m_config = config;
}

dram_cmd::dram_cmd(int cmd, page * from_page, page * to_page, const struct memory_config * config) {
  command = cmd;
  appID = from_page->appID;
  size = from_page->size;
  new_addr_type from_addr = from_page->starting_addr;
  addrdec_t from_raw_addr;

  m_config = config;

  m_config->m_address_mapping.addrdec_tlx(from_addr, &from_raw_addr, appID, DRAM_CMD, 0);
  from_channel = from_raw_addr.chip;
  from_bank = from_raw_addr.bk;
  from_sa = from_raw_addr.subarray;
  if (to_page != NULL) //Zero command
  {
    new_addr_type to_addr = to_page->starting_addr;
    addrdec_t to_raw_addr;
    m_config->m_address_mapping.addrdec_tlx(to_addr, &to_raw_addr, appID, DRAM_CMD, 0);
    to_channel = to_raw_addr.chip;
    to_bank = to_raw_addr.bk;
    to_sa = to_raw_addr.subarray;
  } else {
    to_channel = from_channel;
    to_bank = from_bank;
    to_sa = from_sa;
  }
}

dram_t::dram_t(unsigned int partition_id, const struct memory_config *config, memory_stats_t *stats,
    memory_partition_unit *mp, mmu * page_manager, tlb_tag_array * shared_tlb) {
  id = partition_id;
  m_memory_partition_unit = mp;
  m_stats = stats;
  m_config = config;

  compaction_bank_id = 0;

  m_shared_tlb = shared_tlb;

  m_page_manager = page_manager;

// Should be created in gpu-sim
//   page_manager = new virtual_address_translation(m_config);

  CCDc = 0;
  RRDc = 0;
  RTWc = 0;
  WTRc = 0;

  data_bus_busy = 0;

  rw = READ; //read mode is default

  bkgrp = (bankgrp_t**) calloc(sizeof(bankgrp_t*), m_config->nbkgrp);
  bkgrp[0] = (bankgrp_t*) calloc(sizeof(bank_t), m_config->nbkgrp);
  for (unsigned i = 1; i < m_config->nbkgrp; i++) {
    bkgrp[i] = bkgrp[0] + i;
  }
  for (unsigned i = 0; i < m_config->nbkgrp; i++) {
    bkgrp[i]->CCDLc = 0;
    bkgrp[i]->RTPLc = 0;
  }

  for (unsigned i = 0; i <= 31; i++) {
    mem_state_blp_alarm[i] = 0; // new
    mem_state_blp_ncmd[i] = 0;
  }

  bk = (bank_t**) calloc(sizeof(bank_t*), m_config->nbk);
  bk[0] = (bank_t*) calloc(sizeof(bank_t), m_config->nbk);
  for (unsigned i = 1; i < m_config->nbk; i++)
    bk[i] = bk[0] + i;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    bk[i]->state = BANK_IDLE;
    bk[i]->bkgrpindex = i / (m_config->nbk / m_config->nbkgrp);
    bk[i]->blocked = 0;
    bk[i]->transfer = 0;
  }
  prio = 0;
  rwq = new fifo_pipeline<dram_req_t>("rwq", m_config->CL, m_config->CL + 1);
  mrqq = new fifo_pipeline<dram_req_t>("mrqq", 0, 2);
  returnq = new fifo_pipeline<mem_fetch>("dramreturnq", 0,
      m_config->gpgpu_dram_return_queue_size == 0 ? 1024 : m_config->gpgpu_dram_return_queue_size);
  m_frfcfs_scheduler = NULL;
  if (m_config->scheduler_type == DRAM_FRFCFS)
    m_frfcfs_scheduler = new frfcfs_scheduler(m_config, this, stats, m_shared_tlb);

  n_cmd = 0;
  n_activity = 0;
  n_nop = 0;
  n_act = 0;
  n_pre = 0;
  n_rd = 0;
  n_wr = 0;

  max_mrqs_temp = 0;

  max_mrqs = 0;
  ave_mrqs = 0;
  n_cmd_blp = 0;
  n_req = 0;
  bwutil = 0;
  bwutil_data = 0;
  bwutil_tlb = 0;
  n_cmd_blp = 0;
  mem_state_blp = 0;
  dram_cycles_active = 0;

  // till here
  for (unsigned i = 0; i < 10; i++) {
    dram_util_bins[i] = 0;
    dram_eff_bins[i] = 0;
  }
  last_n_cmd = last_n_activity = last_bwutil = 0;

  n_cmd_partial = 0;
  n_activity_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;
  ave_mrqs_partial = 0;
  bwutil_partial = 0;

  if (queue_limit())
    mrqq_Dist = StatCreate("mrqq_length", 1, queue_limit());
  else
    //queue length is unlimited;
    mrqq_Dist = StatCreate("mrqq_length", 1, 64); //track up to 64 entries

//subarray
  for (unsigned i = 0; i < m_config->nbk; i++) {
    bk[i]->cmd_queue = new std::list<dram_cmd*>();
  }

}

bool dram_t::full() const {
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    if (m_config->gpgpu_frfcfs_dram_sched_queue_size == 0)
      return false;
    return m_frfcfs_scheduler->num_pending() >= m_config->gpgpu_frfcfs_dram_sched_queue_size;
  } else
    return mrqq->full();
}

unsigned dram_t::que_length() const {
  unsigned nreqs = 0;
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    nreqs = m_frfcfs_scheduler->num_pending();
  } else {
    nreqs = mrqq->get_length();
  }
  return nreqs;
}

bool dram_t::returnq_full() const {
  return returnq->full();
}

unsigned int dram_t::queue_limit() const {
  return m_config->gpgpu_frfcfs_dram_sched_queue_size;
}

dram_req_t::dram_req_t(class mem_fetch *mf) {
  txbytes = 0;
  dqbytes = 0;
  data = mf;

  const addrdec_t &tlx = mf->get_tlx_addr();

  bk = tlx.bk;
  row = tlx.row;
  col = tlx.col;
  nbytes = mf->get_data_size();

  timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
  addr = mf->get_addr();
  insertion_time = (unsigned) gpu_sim_cycle;
  rw = data->get_is_write() ? WRITE : READ;
}

void dram_t::push(class mem_fetch *data) {
  if (mrqq->full()) { //make sure that we have enough room to fill into the queue
    wait_list.push_back(data);
  } else {
    if (data->get_page_fault()) // Page not in DRAM
    {
      // This part is obsolete. TLB.cc handles this part
      data->set_page_fault(false);
      data->set_tlb_miss(false);
      data->set_DRAM(this);
    } else { // TLB hit and page in DRAM
      if (data->get_tlb_depth_count() == 0 && data->get_wid() != -1) { //Only check if it is the actual requests, not TLB related request
        if (MERGE_DEBUG)
          printf(
              "Got a memory request in DRAM before issuing. VA = %llx, channelID = %d, DRAM ID = %d\n",
              data->get_addr(), data->get_tlx_addr().chip, id);
        assert(id == data->get_tlx_addr().chip); // Ensure request is in correct memory partition
      }

      dram_req_t *mrq = new dram_req_t(data);
      data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);

      mrqq->push(mrq);

      // stats...

      n_req_partial += 1;

      if (data->get_sid() != -1) {
        n_req += 1;
        App* app = App::get_app(data->get_appID());
        app->n_req++;
      }

      if (m_config->scheduler_type == DRAM_FRFCFS) {
        unsigned nreqs = m_frfcfs_scheduler->num_pending();
        if (nreqs > max_mrqs_temp)
          max_mrqs_temp = nreqs;
      } else {
        max_mrqs_temp = (max_mrqs_temp > mrqq->get_length()) ? max_mrqs_temp : mrqq->get_length();
      }
      m_stats->memlatstat_dram_access(data);
    }
  }
}

void dram_t::insert_dram_command(dram_cmd * cmd) {
  bk[cmd->to_bank]->cmd_queue->push_back(cmd);
}

void dram_t::scheduler_fifo() {
  if (!mrqq->empty()) {
    unsigned int bkn;
    dram_req_t *head_mrqq = mrqq->top();
    head_mrqq->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
    bkn = head_mrqq->bk;
    if (!bk[bkn]->mrq)
      bk[bkn]->mrq = mrqq->pop();
  }
}

#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void dram_t::cycle() {
  // Check if we need to trigger the scanner
  if (m_config->enable_compaction
      && (m_config->compaction_probe_cycle
          < (gpu_sim_cycle + gpu_tot_sim_cycle - m_page_manager->m_compaction_last_probed))) {
    if (RC_DEBUG)
      printf(
          "Compaction read probe triggered by bankid = %d, channelid = %d. Current cycles = %lld\n",
          compaction_bank_id, id, gpu_sim_cycle + gpu_tot_sim_cycle);
    m_page_manager->m_compaction_last_probed = gpu_sim_cycle + gpu_tot_sim_cycle;

    m_page_manager->send_scan_request(); //Trigger the scan request

  }

  while (!mrqq->full() && !wait_list.empty()) {
    mem_fetch *temp = wait_list.front();
    wait_list.pop_front();
    push(temp);
  }

  if (!returnq->full()) {
    dram_req_t *cmd = rwq->pop();
    if (cmd) {
#ifdef DRAM_VIEWCMD
      printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row, cmd->col + cmd->dqbytes);
#endif
      cmd->dqbytes += m_config->dram_atom_size;
      if (cmd->dqbytes >= cmd->nbytes) {
        mem_fetch *data = cmd->data;
        data->set_status(IN_PARTITION_MC_RETURNQ, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (data->get_access_type() != L1_WRBK_ACC && data->get_access_type() != L2_WRBK_ACC) {
          if (data->get_parent_tlb_request() == NULL) {
            data->set_reply();
          }
          returnq->push(data);
        } else {
          m_memory_partition_unit->set_done(data);
          delete data;
        }
        delete cmd;
      }
#ifdef DRAM_VIEWCMD
      printf("\n");
#endif
    }
  }

  /* check if the upcoming request is on an idle bank */
  /* Should we modify this so that multiple requests are checked? */

  switch (m_config->scheduler_type) {
    case DRAM_FIFO:
      scheduler_fifo();
      break;
    case DRAM_FRFCFS:
      scheduler_frfcfs();
      break;
    default:
      printf("Error: Unknown DRAM scheduler type\n");
      assert(0);
  }
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();
    if (nreqs > max_mrqs) {
      max_mrqs = nreqs;
    }
    ave_mrqs += nreqs;
    ave_mrqs_partial += nreqs;
  } else {
    if (mrqq->get_length() > max_mrqs) {
      max_mrqs = mrqq->get_length();
    }
    ave_mrqs += mrqq->get_length();
    ave_mrqs_partial += mrqq->get_length();
  }

  //blp
  unsigned blp;
  unsigned k_app;

  unsigned k = m_config->nbk;

  bool issued = false;

  // check if any bank is ready to issue a new read
  for (unsigned i = 0; i < m_config->nbk; i++) {
    unsigned j = (i + prio) % m_config->nbk;
    unsigned grp = j >> m_config->bk_tag_length;

    if (bk[i]->blocked > 0) //Blocked due to copying
        {
      bk[i]->blocked--;
      if (bk[i]->blocked == 0) {
        if (RC_DEBUG)
          printf("Unblock bankid = %d, channelid = %d, Current cycles = %lld\n", i, id,
              gpu_sim_cycle + gpu_tot_sim_cycle);
        bk[i]->state = BANK_IDLE;
      }
      m_stats->totalbankblocked[id][i]++; //Collect how many cycles this bank has been blocked. Format is[channel][bankID]
      return;
    }

//     //1) other DRAM command here, decrement the counter for each copy command
//     // Note that multiple copies are handled in the MMU
    if ((bk[i]->state == BANK_IDLE) && (!bk[i]->cmd_queue->empty())) //Add cmd_queue
        {
      if (RC_DEBUG && bk[i]->blocked > 0)
        printf("BANK IS BLOCKED but trying to issue new request. Should not happen!!!\n");

      dram_cmd * command = bk[i]->cmd_queue->front();
      bool issued = false;
      if (command->command == TARGET) //Got a copy command from other bank
      {
        //Set states and timing for both banks
        bk[i]->state = BANK_TRANSFER;
        bk[i]->blocked =
            m_config->RC_enabled ? m_config->RCpsm_latency : m_config->interBank_latency;
        bk[command->from_bank]->state = BANK_BLOCKED;
        bk[command->from_bank]->blocked =
            m_config->RC_enabled ? m_config->RCpsm_latency : m_config->interBank_latency;
        issued = true;
      } else if (command->command == LISA_COPY) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->lisa_latency;
        issued = true;
      } else if (command->command == InterSA_COPY) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->interSA_latency;
        issued = true;
      } else if (command->command == IntraSA_COPY) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->intraSA_latency;
        issued = true;
      } else if (command->command == RC_IntraSA) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->RCintraSA_latency;
        issued = true;
      } else if (command->command == RC_zero) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->RCzero_latency;
        issued = true;
      } else if (command->command == ZERO) {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->zero_latency;
        issued = true;
      } else if (command->command == SCAN) //Scan request from the Falcon chip
      {
        bk[i]->state = BANK_BLOCKED;
        bk[i]->blocked = m_config->tRC;
        issued = true;
      } else if (command->command == RC_PSM) {
        if (bk[command->to_bank]->state == BANK_IDLE) //Can issue inter-bank copy, both banks idle
        {
          bk[i]->state = BANK_BLOCKED;
          bk[i]->blocked =
              m_config->RC_enabled ? m_config->RCpsm_latency : m_config->interBank_latency;
          bk[command->to_bank]->state = BANK_TRANSFER;
          bk[command->to_bank]->blocked =
              m_config->RC_enabled ? m_config->RCpsm_latency : m_config->interBank_latency;
          issued = true;
        } else {
          bk[i]->state = BANK_BLOCKED_PENDING; //Can only become other state when target bank change the state of this bank
          bk[command->to_bank]->cmd_queue->push_back(command);
        }
      } else if (command->command == Channel_copy) 
      {
      }
      bk[i]->cmd_queue->pop_front(); //Always remove the command
      if (issued) {
        delete command; //Delete command once it is being issued
      }
    }

    if (bk[j]->mrq) { //if currently servicing a read/write memory request
      bk[j]->mrq->data->set_status(IN_PARTITION_DRAM, gpu_sim_cycle + gpu_tot_sim_cycle);

      // correct row activated for a READ

      blp++;
      k_app++;

      if (bk[j]->mrq->data->get_sid() != -1) { //new
        App* app = App::get_app(bk[j]->mrq->data->get_appID());
        app->blp++;
      }

      //Row hit, push the request to the read write queue (bank is still active)
      if (!issued && !CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc)
          && (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) && (WTRc == 0)
          && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
        if (rw == WRITE) {
          rw = READ;
          rwq->set_min_length(m_config->CL);
        }

        bool same_subarray = (
            m_config->enable_subarray ?
                bk[j]->mrq->data->get_subarray() == bk[j]->curr_subarray : false);
        bk[j]->curr_subarray = bk[j]->mrq->data->get_subarray();

        rwq->push(bk[j]->mrq);
        bk[j]->mrq->txbytes += m_config->dram_atom_size;
        CCDc = (same_subarray ? m_config->sCCD : m_config->tCCD);
        bkgrp[grp]->CCDLc = (same_subarray ? m_config->sCCDL : m_config->tCCDL);
        RTWc = (same_subarray ? m_config->sRTW : m_config->tRTW);
        bk[j]->RTPc = m_config->BL / m_config->data_command_freq_ratio;
        bkgrp[grp]->RTPLc = (same_subarray ? m_config->sRTPL : m_config->tRTPL);
        issued = true;
        n_rd++;
        bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
        bk[j]->n_access++;

        if (bk[j]->mrq->data->get_sid() == -1) {
          sanity_read++;
        }
        if (bk[j]->mrq->data->get_sid() != -1) { //new
          bwutil += m_config->BL / m_config->data_command_freq_ratio;
          bwutil_periodic += m_config->BL / m_config->data_command_freq_ratio;

          if (bk[j]->mrq->data->get_tlb_depth_count() > 0) { //TLB-related data
            bwutil_tlb += m_config->BL / m_config->data_command_freq_ratio;
            bwutil_periodic_tlb += m_config->BL / m_config->data_command_freq_ratio;
          } else {
            bwutil_data += m_config->BL / m_config->data_command_freq_ratio;
            bwutil_periodic_data += m_config->BL / m_config->data_command_freq_ratio;
          }
          App* app = App::get_app(bk[j]->mrq->data->get_appID());
          app->bwutil += m_config->BL / m_config->data_command_freq_ratio;
          app->bwutil_periodic += m_config->BL / m_config->data_command_freq_ratio;
          if (bk[j]->mrq->data->get_tlb_depth_count() > 0) { //TLB-related data
            app->bwutil_tlb += m_config->BL / m_config->data_command_freq_ratio;
            app->bwutil_periodic_tlb += m_config->BL / m_config->data_command_freq_ratio;
          } else {
            app->bwutil_data += m_config->BL / m_config->data_command_freq_ratio;
            app->bwutil_periodic_data += m_config->BL / m_config->data_command_freq_ratio;
          }
        }
#ifdef DRAM_VERIFY
        PRINT_CYCLE=1;
        printf("\tRD  Bk:%d Row:%03x Col:%03x \n",
            j, bk[j]->curr_row,
            bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
        // transfer done
        if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
          bk[j]->mrq = NULL;
        }
      } else
      // correct row activated for a WRITE
      if (!issued && !CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc)
          && (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) && (RTWc == 0)
          && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
        if (rw == READ) {
          rw = WRITE;
          rwq->set_min_length(m_config->WL);
        }
        rwq->push(bk[j]->mrq);
        bool same_subarray = (
            m_config->enable_subarray ?
                bk[j]->mrq->data->get_subarray() == bk[j]->curr_subarray : false);
        bk[j]->curr_subarray = bk[j]->mrq->data->get_subarray();

        bk[j]->mrq->txbytes += m_config->dram_atom_size;
        CCDc = (same_subarray ? m_config->sCCD : m_config->tCCD);
        bkgrp[grp]->CCDLc = (same_subarray ? m_config->sCCDL : m_config->tCCDL);
        WTRc = (same_subarray ? m_config->sWTR : m_config->tWTR);
        bk[j]->WTPc = (same_subarray ? m_config->sWTP : m_config->tWTP);
        issued = true;
        n_wr++;
        if (bk[j]->mrq->data->get_sid() != -1) {            //new
          bwutil += m_config->BL / m_config->data_command_freq_ratio;
          if (bk[j]->mrq->data->get_tlb_depth_count() > 0) {           //TLB-related data
            bwutil_tlb += m_config->BL / m_config->data_command_freq_ratio;
          } else {
            bwutil_data += m_config->BL / m_config->data_command_freq_ratio;
          }
          App* app = App::get_app(bk[j]->mrq->data->get_appID());
          app->bwutil += m_config->BL / m_config->data_command_freq_ratio;
          if (bk[j]->mrq->data->get_tlb_depth_count() > 0) {			//TLB-related data
            app->bwutil_tlb += m_config->BL / m_config->data_command_freq_ratio;
          } else {
            app->bwutil_data += m_config->BL / m_config->data_command_freq_ratio;
          }
        }
#ifdef DRAM_VERIFY
        PRINT_CYCLE=1;
        printf("\tWR  Bk:%d Row:%03x Col:%03x \n",
            j, bk[j]->curr_row,
            bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
        // transfer done
        if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
          bk[j]->mrq = NULL;
        }
      }

      else
      // bank is idle
      if (!issued && !RRDc && (bk[j]->state == BANK_IDLE) && !bk[j]->RPc && !bk[j]->RCc) {
#ifdef DRAM_VERIFY
        PRINT_CYCLE=1;
        printf("\tACT BK:%d NewRow:%03x From:%03x \n",
            j,bk[j]->mrq->row,bk[j]->curr_row);
#endif
        // activate the row with current memory request
        bk[j]->curr_row = bk[j]->mrq->row;
        bk[j]->state = BANK_ACTIVE;

        bool same_subarray = (
            m_config->enable_subarray ?
                bk[j]->mrq->data->get_subarray() == bk[j]->curr_subarray : false);
        bk[j]->curr_subarray = bk[j]->mrq->data->get_subarray();

        RRDc = (same_subarray ? m_config->sRRD : m_config->tRRD);
        bk[j]->RCDc = (same_subarray ? m_config->sRCD : m_config->tRCD);
        bk[j]->RCDWRc = (same_subarray ? m_config->sRCDWR : m_config->tRCDWR);
        bk[j]->RASc = (same_subarray ? m_config->sRAS : m_config->tRAS);
        bk[j]->RCc = (same_subarray ? m_config->sRC : m_config->tRC);
        prio = (j + 1) % m_config->nbk;
        issued = true;
        n_act_partial++;
        n_act++;
      }

      else
      // different row activated
      if ((!issued) && (bk[j]->curr_row != bk[j]->mrq->row) && (bk[j]->state == BANK_ACTIVE)
          && (!bk[j]->RASc && !bk[j]->WTPc && !bk[j]->RTPc && !bkgrp[grp]->RTPLc)) {
        // make the bank idle again
        bk[j]->state = BANK_IDLE;
        bool same_subarray = (
            m_config->enable_subarray ?
                bk[j]->mrq->data->get_subarray() == bk[j]->curr_subarray : false);
        bk[j]->curr_subarray = bk[j]->mrq->data->get_subarray();

        bk[j]->RPc = (same_subarray ? m_config->sRP : m_config->tRP);
        prio = (j + 1) % m_config->nbk;
        issued = true;
        n_pre++;
        n_pre_partial++;
#ifdef DRAM_VERIFY
        PRINT_CYCLE=1;
        printf("\tPRE BK:%d Row:%03x \n", j,bk[j]->curr_row);
#endif
      }
    } else {
      if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc && !bk[j]->RCc
          && !bk[j]->RPc && !bk[j]->RCDWRc) {
        k--;
      } else {
        k_app++;
      }
      bk[j]->n_idle++;
    }
  }
  if (!issued) {
    n_nop++;
    n_nop_partial++;
#ifdef DRAM_VIEWCMD
    printf("\tNOP                        ");
#endif
  }
  if (k) {
    n_activity++;
    n_activity_partial++;
  }
  if (k_app) {
    dram_cycles_active++;
  }
  std::map<appid_t, App*> apps = App::get_apps();
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    if (i->second->k_app) {
      i->second->dram_cycles_active++;
    }
  }

  n_cmd++;
  n_cmd_partial++;

  if (blp) {
    n_cmd_blp++;
    mem_state_blp += blp;
  }

  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    if (i->second->blp) {
      i->second->n_cmd_blp++;
      i->second->mem_state_blp += i->second->blp;
    }
  }

  // decrements counters once for each time dram_issueCMD is called
  DEC2ZERO(RRDc);
  DEC2ZERO(CCDc);
  DEC2ZERO(RTWc);
  DEC2ZERO(WTRc);
  for (unsigned j = 0; j < m_config->nbk; j++) {
    DEC2ZERO(bk[j]->RCDc);
    DEC2ZERO(bk[j]->RASc);
    DEC2ZERO(bk[j]->RCc);
    DEC2ZERO(bk[j]->RPc);
    DEC2ZERO(bk[j]->RCDWRc);
    DEC2ZERO(bk[j]->WTPc);
    DEC2ZERO(bk[j]->RTPc);
  }
  for (unsigned j = 0; j < m_config->nbkgrp; j++) {
    DEC2ZERO(bkgrp[j]->CCDLc);
    DEC2ZERO(bkgrp[j]->RTPLc);
  }

#ifdef DRAM_VISUALIZE
  visualize();
#endif
}

//if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch* dram_t::return_queue_pop() {
  return returnq->pop();
}

class mem_fetch* dram_t::return_queue_top() {
  return returnq->top();
}

void dram_t::print(FILE* simFile) const {
  fprintf(simFile, "DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", id, m_config->nbk, m_config->busW,
      m_config->BL, m_config->CL);
  fprintf(simFile, "tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n", m_config->tCCD,
      m_config->tRRD, m_config->tRCD, m_config->tRAS, m_config->tRP, m_config->tRC);
  fprintf(simFile, "n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d ", n_cmd, n_nop, n_act, n_pre,
      n_req);
  std::map<appid_t, App*> apps = App::get_apps();
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "n_req_%d=%d ", i->first, i->second->n_req);
  }
  fprintf(simFile, "n_rd=%d n_write=%d bw_util=%.4g ", n_rd, n_wr, (float) bwutil / n_cmd);
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "bw_util_%d=%.4g ", i->first, (float) i->second->bwutil / n_cmd);
  }
  fprintf(simFile, "bw_util_tlb=%.4g ", (float) bwutil_tlb / n_cmd);
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "bw_util_tlb_%d=%.4g ", i->first, (float) i->second->bwutil_tlb / n_cmd);
  }
  fprintf(simFile, "bw_util_data=%.4g ", (float) bwutil_data / n_cmd);
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "bw_util_data_%d=%.4g ", i->first, (float) i->second->bwutil_data / n_cmd);
  }
  fprintf(simFile, "blp=%f ", (float) mem_state_blp / n_cmd_blp);
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "blp_%d=%f ", i->first,
        (float) i->second->mem_state_blp / i->second->n_cmd_blp);
  }
  fprintf(simFile, "\n");
  fprintf(simFile, "n_activity=%d dram_eff=%.4g ", n_activity, (float) bwutil / n_activity);
  for (std::map<appid_t, App*>::iterator i = apps.begin(); i != apps.end(); i++) {
    fprintf(simFile, "dram_eff_%d=%.4g ", i->first, (float) i->second->bwutil / n_activity);
  }
  fprintf(simFile, "\n");

  for (int j = 0; j < m_config->nbk; j++) {
    fprintf(simFile, "bk%d: %da %di ", j, bk[j]->n_access, bk[j]->n_idle);
  }
  fprintf(simFile, "\n");
  fprintf(simFile, "bw_dist = ");
  fprintf(simFile, "%4.3f\t%4.3f\n", ((float) dram_cycles_active / n_cmd - (float) bwutil / n_cmd),
      (1 - (float) dram_cycles_active / n_cmd));

  fprintf(simFile, "bw_dist_detailed (tlb,data)= ");
  fprintf(simFile, "%4.3f\t%4.3f", ((float) dram_cycles_active / n_cmd - (float) bwutil / n_cmd),
      (1 - (float) dram_cycles_active / n_cmd));

  fprintf(simFile, "\n");

  fprintf(simFile, "mrqq: %d %.4g mrqsmax=%d ", max_mrqs, (float) ave_mrqs / n_cmd, max_mrqs_temp);
  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (int i = 0; i < 10; i++)
    fprintf(simFile, " %d", dram_util_bins[i]);
  fprintf(simFile, "\ndram_eff_bins:");
  for (int i = 0; i < 10; i++)
    fprintf(simFile, " %d", dram_eff_bins[i]);
  fprintf(simFile, "\n");
  if (m_config->scheduler_type == DRAM_FRFCFS)
    fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs, (float) ave_mrqs / n_cmd);
}

void dram_t::visualize() const {
  printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", RRDc, CCDc, mrqq->get_length(),
      rwq->get_length());
  for (unsigned i = 0; i < m_config->nbk; i++) {
    printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", i, bk[i]->state, bk[i]->curr_row,
        bk[i]->RCDc, bk[i]->RASc, bk[i]->RPc, bk[i]->RCc, bk[i]->mrq);
    if (bk[i]->mrq)
      printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
    printf("\n");
  }
  if (m_frfcfs_scheduler)
    m_frfcfs_scheduler->print(stdout);
}

void dram_t::print_stat(FILE* simFile) {
  return print(simFile);
}

void dram_t::visualizer_print(gzFile visualizer_file) {
  // dram specific statistics
  gzprintf(visualizer_file, "dramncmd: %u %u\n", id, n_cmd_partial);
  gzprintf(visualizer_file, "dramnop: %u %u\n", id, n_nop_partial);
  gzprintf(visualizer_file, "dramnact: %u %u\n", id, n_act_partial);
  gzprintf(visualizer_file, "dramnpre: %u %u\n", id, n_pre_partial);
  gzprintf(visualizer_file, "dramnreq: %u %u\n", id, n_req_partial);
  gzprintf(visualizer_file, "dramavemrqs: %u %u\n", id,
      n_cmd_partial ? (ave_mrqs_partial / n_cmd_partial) : 0);

  // utilization and efficiency
  gzprintf(visualizer_file, "dramutil: %u %u\n", id,
      n_cmd_partial ? 100 * bwutil_partial / n_cmd_partial : 0);
  gzprintf(visualizer_file, "drameff: %u %u\n", id,
      n_activity_partial ? 100 * bwutil_partial / n_activity_partial : 0);

  // reset for next interval
  bwutil_partial = 0;
  n_activity_partial = 0;
  ave_mrqs_partial = 0;
  n_cmd_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;

  // dram access type classification
  for (unsigned j = 0; j < m_config->nbk; j++) {
    gzprintf(visualizer_file, "dramglobal_acc_r: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramglobal_acc_w: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_r: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_w: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramconst_acc_r: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramtexture_acc_r: %u %u %u\n", id, j,
        m_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
  }
}

void dram_t::set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop, unsigned &act,
    unsigned &pre, unsigned &rd, unsigned &wr, unsigned &req) const {

  // Point power performance counters to low-level DRAM counters
  cmd = n_cmd;
  activity = n_activity;
  nop = n_nop;
  act = n_act;
  pre = n_pre;
  rd = n_rd;
  wr = n_wr;
  req = n_req;
}

void dram_t::set_miss(float m) {
  miss_rate_d = m;
}

void dram_t::set_miss_r(appid_t appid, float m) {
  App* app = App::get_app(appid);
  app->miss_rate_d = m;
}

void dram_t::set_miss_core(float m, unsigned i) {
  miss_rate_d_core[i] = m;
}

float dram_t::get_miss() {
  return miss_rate_d;
}

unsigned dram_t::dram_bwutil_data() {
  unsigned temp = bwutil_periodic_data;
  bwutil_periodic_data = 0;
  return temp;
}

unsigned dram_t::dram_bwutil_data(appid_t appid) {
  App* app = App::get_app(appid);
  unsigned temp = app->bwutil_periodic_data;
  app->bwutil_periodic_data = 0;
  return temp;
}

unsigned dram_t::dram_bwutil_tlb() {
  unsigned temp = bwutil_periodic_tlb;
  bwutil_periodic_tlb = 0;
  return temp;
}

unsigned dram_t::dram_bwutil_tlb(appid_t appid) {
  App* app = App::get_app(appid);
  unsigned temp = app->bwutil_periodic_tlb;
  app->bwutil_periodic_tlb = 0;
  return temp;
}

unsigned dram_t::dram_bwutil() {
  unsigned temp = bwutil_periodic;
  bwutil_periodic = 0;
  return temp;
}

unsigned dram_t::dram_bwutil(appid_t appid) {
  App* app = App::get_app(appid);
  unsigned temp = app->bwutil_periodic;
  app->bwutil_periodic = 0;
  return temp;
}

