// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
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

#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "../abstract_hardware_model.h"
#include "mem_latency_stat.h"
extern int gpu_sms;

frfcfs_scheduler::frfcfs_scheduler(const memory_config *config, dram_t *dm, memory_stats_t *stats,
    tlb_tag_array * shared_tlb) {
  m_config = config;
  m_shared_tlb = shared_tlb;
  m_stats = stats;
  m_num_pending = 0;
  m_dram = dm;
//   chain_counter = 0;
  last_core_id = new unsigned[m_config->nbk]; //new
  m_queue = new std::list<dram_req_t*>[m_config->nbk];
  m_queue_high = new std::list<dram_req_t*>[m_config->nbk];
  m_queue_special = new std::list<dram_req_t*>[m_config->nbk];
  m_bins = new std::map<unsigned, std::list<std::list<dram_req_t*>::iterator> >[m_config->nbk];
//   m_bins_high = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ m_config->nbk ];
  m_last_row = new std::list<std::list<dram_req_t*>::iterator>*[m_config->nbk];
  curr_row_service_time = new unsigned[m_config->nbk];
  row_service_timestamp = new unsigned[m_config->nbk];
  for (unsigned i = 0; i < m_config->nbk; i++) {
    m_queue[i].clear();
    m_bins[i].clear();
    m_last_row[i] = NULL;
    curr_row_service_time[i] = 0;
    row_service_timestamp[i] = 0;
  }

  m_high_pq_wait_timestamp = 0;
  m_high_pq_combo_count = 0;
  epoch_total_concurrent = 1;

}

unsigned frfcfs_scheduler::set_req_prio_criteria(mem_fetch * mf) {
  if (mf->get_mem_config()->tlb_dram_aware == 1 && (mf->get_tlb_depth_count() > 0)
      && ((mf->get_mem_config()->tlb_levels - mf->get_tlb_depth_count())
          >= mf->get_mem_config()->tlb_high_prio_level)) {
    return 1;
  } else if (mf->get_mem_config()->tlb_dram_aware == 2 && (mf->get_tlb_depth_count() > 0)) {
    return 1;
  } else if (mf->get_mem_config()->tlb_dram_aware == 3 && (mf->get_tlb_depth_count() == 0)) {
    return 1;
  } else if (mf->get_mem_config()->tlb_dram_aware == 4
      && (mf->get_appID() == (mf->get_mem_config()->dram_always_prioritize_app))) {
    return 1;
  } else if (mf->get_mem_config()->tlb_dram_aware == 5) {
    if (mf->get_tlb_depth_count() > 0)
      return 1; //For TLB-related requests, always prioritize
    unsigned return_val;

    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_appID()));

    if (mf->get_appID() == current_prioritize) {

      current_prioritize_total++;
      m_stats->high_prio_queue_count++;
      app->high_prio_queue_count_app++;
      return_val = 1;
    } else
      return_val = 0;

    //Switch app
    //float total_sched = m_stats->schedulable_avg[0]+m_stats->schedulable_avg[1]+m_stats->schedulable_avg[2];
    //float bw_factor = 1 + m_stats->schedulable_avg[mf->get_appID()]/total_sched;
    //Figure out how to tune this bw_factor
    //These two lines below cause segfault
    unsigned this_concurrent = app->concurrent_tracker;
    unsigned total_concurrent = 0;
    for (std::map<appid_t, App*>::iterator i = App::get_apps().begin(); i != App::get_apps().end();
        i++) {
      total_concurrent += i->second->concurrent_tracker;
    }

    float bw_factor =
        (total_concurrent == 0) ? 1 : (float) this_concurrent / (float) total_concurrent;
    float rand_factor = ((float) (mf->get_mem_config()->dram_switch_factor) / 100 * bw_factor); //The more schedulable, the more likely dram sched switch to the other app
    int decay = (int) (current_prioritize_total * rand_factor);
    int max = mf->get_mem_config()->dram_switch_max;
    int chance = rand() % ((max > decay) ? max - decay : 1);

    if (chance < mf->get_mem_config()->dram_switch_threshold) {
      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;
      current_prioritize_total = 0;
      current_prioritize++;
      current_prioritize %= ConfigOptions::n_apps;
    }

    return return_val;
  } else if (mf->get_mem_config()->tlb_dram_aware == 6) {
    if (mf->get_tlb_depth_count() > 0)
      return 1; //For TLB-related requests, always prioritize
    unsigned return_val;

    //Switch app
    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_appID()));
    unsigned this_concurrent = app->concurrent_tracker;
    unsigned total_concurrent = 0;
    for (std::map<appid_t, App*>::iterator i = App::get_apps().begin(); i != App::get_apps().end();
        i++) {
      total_concurrent += i->second->concurrent_tracker;
    }

    if (mf->get_appID() == current_prioritize) {
      current_prioritize_total++;
      if (total_concurrent == 0)
        return 0;
      else if ((rand() % mf->get_mem_config()->dram_high_prio_chance) < this_concurrent) {
        m_stats->high_prio_queue_count++;
        app->high_prio_queue_count_app++;
        return_val = 1;
      } else
        return_val = 0;
    } else
      return_val = 0;


    if (total_concurrent > 0
        && ((rand() % total_concurrent) < (total_concurrent - this_concurrent))) {
      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;

      current_prioritize_total = 0;
      current_prioritize++;
      current_prioritize %= ConfigOptions::n_apps;
    }

    return return_val;
  } else if (mf->get_mem_config()->tlb_dram_aware == 7) {
    if (mf->get_tlb_depth_count() > 0)
      return 1; //For TLB-related requests, always prioritize
    unsigned return_val;

    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_appID()));

    unsigned threshold = mf->get_mem_config()->dram_switch_threshold
        * ((float) (app->epoch_app_concurrent) / (float) (epoch_total_concurrent));

    // If equal to batch size, switch app
    if (mf->get_appID() == current_prioritize) {
      m_stats->high_prio_queue_count++;
      app->high_prio_queue_count_app++;

      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;

      current_prioritize_total++;

      return_val = 1;
    } else {
      return_val = 0;
    }

    if (current_prioritize_total > threshold) {
      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;

      current_prioritize_total = 0;
      current_prioritize++;
      current_prioritize = current_prioritize % ConfigOptions::n_apps;

      //update threshold
      for (int i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        app->epoch_app_concurrent =app->concurrent_tracker;
        epoch_total_concurrent += app->concurrent_tracker;
      }
      if (epoch_total_concurrent == 0)
        epoch_total_concurrent = 1; //so that we do not have a divide by zero - OMG
    }

    return return_val;
  }

  else if (mf->get_mem_config()->tlb_dram_aware == 8) {
    if (mf->get_tlb_depth_count() > 0)
      return 2; //For TLB-related requests, always prioritize
    unsigned return_val;

    App* app = App::get_app(mf->get_appID());
    //App* app = App::get_app(App::get_app_id(mf->get_appID()));

    unsigned threshold = mf->get_mem_config()->dram_switch_threshold
        * ((float) (app->epoch_app_concurrent) / (float) (epoch_total_concurrent));

    // If equal to batch size, switch app
    if (mf->get_appID() == current_prioritize) {
      m_stats->high_prio_queue_count++;
      app->high_prio_queue_count_app++;

      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;

      current_prioritize_total++;

      return_val = 1;
    } else
      return_val = 0;


    if (current_prioritize_total > threshold) {
      m_stats->dram_app_switch++;
      app = App::get_app(current_prioritize);
      //app = App::get_app(App::get_app_id(current_prioritize));
      app->dram_prioritized_cycles_app = gpu_sim_cycle + gpu_tot_sim_cycle - last_switch;
      last_switch = gpu_sim_cycle + gpu_tot_sim_cycle;

      current_prioritize_total = 0;
      current_prioritize = current_prioritize % ConfigOptions::n_apps;

      //update threshold
      for (int i = 0; i < ConfigOptions::n_apps; i++) {
        App* app = App::get_app(App::get_app_id(i));
        app->epoch_app_concurrent = app->concurrent_tracker;
        epoch_total_concurrent += app->concurrent_tracker;
      }

      if (epoch_total_concurrent == 0)
        epoch_total_concurrent = 1; //so that we do not have a divide by zero
    }
    return return_val;
  } else {
    return 0;
  }
}

void frfcfs_scheduler::add_req(dram_req_t *req) {
  m_num_pending++;
  if (m_config->dram_scheduling_policy == 0) //If FR-FCFS-based
      {
    if (set_req_prio_criteria(req->data) == 2) //Special priority (higher than high)
        {
      m_stats->DRAM_special_prio++;
      m_queue_special[req->bk].push_front(req);
      std::list<dram_req_t*>::iterator ptr = m_queue_special[req->bk].begin();
      m_bins[req->bk][req->row].push_front(ptr); //newest reqs to the front
    } else if (set_req_prio_criteria(req->data) == 1) {
      m_stats->DRAM_high_prio++;
      if (m_queue_high[req->bk].empty()) {
        m_high_pq_wait_timestamp = gpu_sim_cycle;
      }
      m_queue_high[req->bk].push_front(req);
      std::list<dram_req_t*>::iterator ptr = m_queue_high[req->bk].begin();
      m_bins[req->bk][req->row].push_front(ptr); //newest reqs to the front
    } else {
      m_stats->DRAM_normal_prio++;
      m_queue[req->bk].push_front(req);
      std::list<dram_req_t*>::iterator ptr = m_queue[req->bk].begin();
      m_bins[req->bk][req->row].push_front(ptr); //newest reqs to the front
    }
  } else if (m_config->dram_scheduling_policy == 1) {
    m_queue[req->bk].push_front(req);
  }
}

void frfcfs_scheduler::data_collection(unsigned int bank) {
  if (gpu_sim_cycle > row_service_timestamp[bank]) {
    curr_row_service_time[bank] = gpu_sim_cycle - row_service_timestamp[bank];
    if (curr_row_service_time[bank] > m_stats->max_servicetime2samerow[m_dram->id][bank])
      m_stats->max_servicetime2samerow[m_dram->id][bank] = curr_row_service_time[bank];
  }
  curr_row_service_time[bank] = 0;
  row_service_timestamp[bank] = gpu_sim_cycle;
  if (m_stats->concurrent_row_access[m_dram->id][bank]
      > m_stats->max_conc_access2samerow[m_dram->id][bank]) {
    m_stats->max_conc_access2samerow[m_dram->id][bank] =
        m_stats->concurrent_row_access[m_dram->id][bank];
  }
  m_stats->concurrent_row_access[m_dram->id][bank] = 0;
  m_stats->num_activates[m_dram->id][bank]++;
  m_stats->num_activates_w[m_dram->id][bank]++;

  App* app = App::get_app(App::get_app_id_from_sm(last_core_id[bank]));
  app->num_activates_w_[m_dram->id][bank]++;
  app->num_activates_[m_dram->id][bank]++;
}


// Mechanism to batch requests in the high prio queue, set m_high_prio_drain to false to batch, true to drain batched requests
void frfcfs_scheduler::check_high_queue_drain() {
  if (m_high_pq_wait_timestamp + m_config->max_DRAM_high_prio_wait < gpu_sim_cycle) {
    m_high_prio_queue_drain = true;
    m_high_pq_wait_timestamp = gpu_sim_cycle;
    m_high_pq_combo_count = 0;
  }
}

// Rachata: Pick a request from the request buffers. Can overwrite FR-FCFS policy
dram_req_t *frfcfs_scheduler::pick(std::list<dram_req_t*> * buffers, unsigned curr_row,
    unsigned bank) {
  //FR-FCFS
  if (m_config->dram_scheduling_policy == 1)
    return buffers->back(); //In - front, out - back
}

//From MeDiC
dram_req_t *frfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {

  dram_req_t *req;
  bool fromHighPool = false;
  std::list<dram_req_t*>::iterator next;
  if (m_config->dram_scheduling_policy == 0) {

    if (m_last_row[bank] == NULL) {
      if (m_config->dram_batch && !m_high_prio_queue_drain && !m_queue_high[bank].empty()) // Not draining High prio queue if the low_prio_queue is not empty
          {
        // Check when to start draining high prio queue
        check_high_queue_drain();
      }
      // If there are anything in the high prio queue
      if (!m_queue_special[bank].empty()) {
        std::map<unsigned, std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr =
            m_bins[bank].find(curr_row);
        if (bin_ptr == m_bins[bank].end()) {
          dram_req_t *tmp = m_queue_special[bank].back();
          bin_ptr = m_bins[bank].find(tmp->row);
          assert(bin_ptr != m_bins[bank].end()); // where did the request go???
          m_last_row[bank] = &(bin_ptr->second);
          data_collection(bank);
          // 		 last_core_id[bank] = req->data->get_sid(); //new --> This cause a segfault somehow .....
        } else {
          m_last_row[bank] = &(bin_ptr->second);
        }
        m_stats->sched_from_special_prio++;
      }

      else if (!m_queue_high[bank].empty() && (!m_config->dram_batch || m_high_prio_queue_drain)) {
        std::map<unsigned, std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr =
            m_bins[bank].find(curr_row);
        if (bin_ptr == m_bins[bank].end()) {
          dram_req_t *tmp = m_queue_high[bank].back();
          bin_ptr = m_bins[bank].find(tmp->row);
          assert(bin_ptr != m_bins[bank].end()); // where did the request go???
          m_last_row[bank] = &(bin_ptr->second);
          data_collection(bank);
          // 		 last_core_id[bank] = req->data->get_sid(); //new --> This cause a segfault somehow .....
        } else {
          m_last_row[bank] = &(bin_ptr->second);
        }
        fromHighPool = true;
        m_stats->sched_from_high_prio++;
        //Disable high PQ draining if the combo count is above the threshole. The next drain will happen after this cycle + max_DRAM_high_prio_wait
        m_high_pq_combo_count++;
        if (m_high_pq_combo_count > m_config->max_DRAM_high_prio_combo) {
          m_stats->total_combo += m_high_pq_combo_count;
          avg_combo_count = (float) (m_stats->total_combo) / m_stats->drain_reset;
          m_high_pq_combo_count = 0;
          m_high_pq_wait_timestamp = gpu_sim_cycle;
          m_high_prio_queue_drain = false;
          m_stats->drain_reset++;
        }
      } else {
        if (m_config->dram_batch) //Finish draining requests from high prio queue
        {
          if (m_high_prio_queue_drain) // In case high_prio queue has no more request, finish draining and collect stat
            m_stats->drain_reset++;
          m_high_prio_queue_drain = false;
          m_stats->total_combo += m_high_pq_combo_count;
          avg_combo_count = (float) (m_stats->total_combo) / m_stats->drain_reset;
          m_high_pq_combo_count = 0;
        }
        if (m_queue[bank].empty())
          return NULL;
        std::map<unsigned, std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr =
            m_bins[bank].find(curr_row);

        m_stats->sched_from_normal_prio++;
        if (bin_ptr == m_bins[bank].end()) {
          dram_req_t *tmp = m_queue[bank].back();
          bin_ptr = m_bins[bank].find(tmp->row);
          assert(bin_ptr != m_bins[bank].end()); // where did the request go???
          m_last_row[bank] = &(bin_ptr->second);
          data_collection(bank);
        } else {
          m_last_row[bank] = &(bin_ptr->second);
        }
      }
    }
    next = m_last_row[bank]->back();
    req = (*next);

  } else if (m_config->dram_scheduling_policy == 1) {
    req = pick(&m_queue[bank], curr_row, bank);
  }

  m_stats->concurrent_row_access[m_dram->id][bank]++;
  m_stats->row_access[m_dram->id][bank]++;
  m_last_row[bank]->pop_back();

  App* app = App::get_app(App::get_app_id_from_sm(last_core_id[bank]));
  app->row_access_w_[m_dram->id][bank]++;
  app->row_access_[m_dram->id][bank]++;

  if (fromHighPool)
    m_queue_high[bank].erase(next);
  else
    m_queue[bank].erase(next);
  if (m_last_row[bank]->empty()) {
    m_bins[bank].erase(req->row);
    m_last_row[bank] = NULL;
  }
#ifdef DEBUG_FAST_IDEAL_SCHED
  if ( req )
  printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n",
      (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif
  assert(req != NULL && m_num_pending != 0);
  m_num_pending--;

  return req;
}

void frfcfs_scheduler::print(FILE *fp) {
  for (unsigned b = 0; b < m_config->nbk; b++) {
    printf(" %u: queue length = %u\n", b, (unsigned) m_queue[b].size());
  }
}

void dram_t::scheduler_frfcfs() {
  unsigned mrq_latency;
  frfcfs_scheduler *sched = m_frfcfs_scheduler;
  while (!mrqq->empty()
      && (!m_config->gpgpu_frfcfs_dram_sched_queue_size
          || sched->num_pending() < m_config->gpgpu_frfcfs_dram_sched_queue_size)) {
    dram_req_t *req = mrqq->pop();

    // Power stats
    //if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
    m_stats->total_n_access++;

    if (req->data->get_type() == WRITE_REQUEST) {
      m_stats->total_n_writes++;
    } else if (req->data->get_type() == READ_REQUEST) {
      m_stats->total_n_reads++;
    }

    req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
    sched->add_req(req);
  }

  dram_req_t *req;
  unsigned i;
  for (i = 0; i < m_config->nbk; i++) {
    unsigned b = (i + prio) % m_config->nbk;
    if (!bk[b]->mrq) {

      req = sched->schedule(b, bk[b]->curr_row);

      if (req) {
        req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE, gpu_sim_cycle + gpu_tot_sim_cycle);
        prio = (prio + 1) % m_config->nbk;
        bk[b]->mrq = req;
        if (m_config->gpgpu_memlatency_stat) {
          mrq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - bk[b]->mrq->timestamp;

          App* app = App::get_app(req->data->get_appID());
          //App* app = App::get_app(App::get_app_id(req->data->get_sid()));
          app->mrqs_latency += mrq_latency;
          app->mrq_num++;

          bk[b]->mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
          m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
          if (mrq_latency > m_stats->max_mrq_latency) {
            m_stats->max_mrq_latency = mrq_latency;
          }
        }

        break;
      }
    }
  }
}
