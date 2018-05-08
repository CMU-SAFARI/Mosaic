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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"
#include "l2cache_trace.h"
#include "App.h"

extern int gpu_sms;


mem_fetch * partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr, unsigned sid) const 
{
    assert( wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   WRITE_PACKET_SIZE, 
                                   -1, 
                                   sid, 
                                   -1,
                                   m_memory_config );
    return mf;
}

memory_partition_unit::memory_partition_unit( unsigned partition_id, 
                                              const struct memory_config *config,
                                              class memory_stats_t *stats, mmu * page_manager, tlb_tag_array * shared_tlb)
: m_id(partition_id), m_config(config), m_stats(stats), m_arbitration_metadata(config) 
{

    m_page_manager = page_manager;

    m_shared_tlb = shared_tlb;

    m_dram = new dram_t(m_id,m_config,m_stats,this, m_page_manager, shared_tlb);

    page_manager->set_DRAM_channel(m_dram,partition_id);

    m_sub_partition = new memory_sub_partition*[m_config->m_n_sub_partition_per_memory_channel]; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned sub_partition_id = m_id * m_config->m_n_sub_partition_per_memory_channel + p; 
        m_sub_partition[p] = new memory_sub_partition(sub_partition_id, m_config, stats); 
    }

//    m_tlb_request_queue = new std::list<mem_fetch*>;

}

memory_partition_unit::~memory_partition_unit() 
{
    delete m_dram; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        delete m_sub_partition[p]; 
    } 
    delete[] m_sub_partition; 
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(const struct memory_config *config) 
: m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1), 
  m_private_credit(config->m_n_sub_partition_per_memory_channel, 0), 
  m_shared_credit(0) 
{
    // each sub partition get at least 1 credit for forward progress 
    // the rest is shared among with other partitions 
    m_private_credit_limit = 1; 
    m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size 
                            + config->gpgpu_dram_return_queue_size 
                            - (config->m_n_sub_partition_per_memory_channel - 1); 
    if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 
        or config->gpgpu_dram_return_queue_size == 0) 
    {
        m_shared_credit_limit = 0; // no limit if either of the queue has no limit in size 
    }
    assert(m_shared_credit_limit >= 0); 
}

bool memory_partition_unit::arbitration_metadata::has_credits(int inner_sub_partition_id) const 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        return true; 
    } else if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        return true; 
    } else {
        return false; 
    }
}


void memory_partition_unit::insert_dram_command(dram_cmd * cmd)
{
    m_dram->insert_dram_command(cmd);
}


void memory_partition_unit::arbitration_metadata::borrow_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] < m_private_credit_limit) {
        m_private_credit[spid] += 1; 
    } else if (m_shared_credit_limit == 0 || m_shared_credit < m_shared_credit_limit) {
        m_shared_credit += 1; 
    } else {
        assert(0 && "DRAM arbitration error: Borrowing from depleted credit!"); 
    }
    m_last_borrower = spid; 
}

void memory_partition_unit::arbitration_metadata::return_credit(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    if (m_private_credit[spid] > 0) {
        m_private_credit[spid] -= 1; 
    } else {
        m_shared_credit -= 1; 
    } 
    assert((m_shared_credit >= 0) && "DRAM arbitration error: Returning more than available credits!"); 
}

void memory_partition_unit::arbitration_metadata::print( FILE *fp ) const 
{
    fprintf(fp, "private_credit = "); 
    for (unsigned p = 0; p < m_private_credit.size(); p++) {
        fprintf(fp, "%d ", m_private_credit[p]); 
    }
    fprintf(fp, "(limit = %d)\n", m_private_credit_limit); 
    fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit, m_shared_credit_limit); 
}

bool memory_partition_unit::busy() const 
{
    bool busy = false; 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        if (m_sub_partition[p]->busy()) {
            busy = true; 
        }
    }
    return busy; 
}

void memory_partition_unit::cache_cycle(unsigned cycle) 
{
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->cache_cycle(cycle); 
    }
}

void memory_partition_unit::visualizer_print( gzFile visualizer_file ) const 
{
    m_dram->visualizer_print(visualizer_file);
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->visualizer_print(visualizer_file); 
    }
}

// determine whether a given subpartition can issue to DRAM 
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) 
{
    int spid = inner_sub_partition_id; 
    bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full(); 
    bool has_dram_resource = m_arbitration_metadata.has_credits(spid); 

    MEMPART_DPRINTF("sub partition %d sub_partition_contention=%c has_dram_resource=%c\n", 
                    spid, (sub_partition_contention)? 'T':'F', (has_dram_resource)? 'T':'F'); 

    return (has_dram_resource && !sub_partition_contention); 
}

int memory_partition_unit::global_sub_partition_id_to_local_id(int global_sub_partition_id) const
{
    return (global_sub_partition_id - m_id * m_config->m_n_sub_partition_per_memory_channel); 
}

void memory_partition_unit::dram_cycle() 
{ 

//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle() from memory_partition_unit\n");

    // pop completed memory request from dram and push it to dram-to-L2 queue 
    // of the original sub partition 
    mem_fetch* mf_return = m_dram->return_queue_top();


//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), got the return mem_fetch\n");

//    if(mf_return && (mf_return->get_parent_tlb_request()!=NULL))
//    {
//        printf("In l2cache.cc, Cycleing DRAM in dram_cycle(), return mem_fetch is a TLB-related request for addr %x, level %d\n", mf_return->get_addr(), mf_return->get_tlb_depth_count());
//        mf_return->done_tlb_req(mf_return->get_parent_tlb_request());
//    }
//    // If not
//    else if (mf_return) {
    if (mf_return) {
        unsigned dest_global_spid = mf_return->get_sub_partition_id(); 
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid); 
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid); 
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
            if( mf_return->get_access_type() == L1_WRBK_ACC ) {
                m_sub_partition[dest_spid]->set_done(mf_return); 
                delete mf_return;
            } else {
                m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
                mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.return_credit(dest_spid); 
                MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf_return, dest_spid); 
            }
            m_dram->return_queue_pop(); 
        }
    } else {
//        printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), return mem_fetch is NULL, nothing to return?\n");
        m_dram->return_queue_pop(); 
    }
    
//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), A call to cycle DRAM banks\n");

    m_dram->cycle(); 
//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Done cycling DRAM banks\n");
    m_dram->dram_log(SAMPLELOG);   

//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Before adding a new request to DRAM\n");

    if( !m_dram->full() ) {
        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions 
        int last_issued_partition = m_arbitration_metadata.last_borrower(); 
        for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
            int spid = (p + last_issued_partition + 1) % m_config->m_n_sub_partition_per_memory_channel; 
            if (!m_sub_partition[spid]->L2_dram_queue_empty() && can_issue_to_dram(spid)) {
//                printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), picking a mem_fetch to be sent to DRAM\n");
                mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
                m_sub_partition[spid]->L2_dram_queue_pop();
                MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n", mf, spid); 
                dram_delay_t d;
                d.req = mf;
                d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
                m_dram_latency_queue.push_back(d);
                mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_arbitration_metadata.borrow_credit(spid); 
                break;  // the DRAM should only accept one request per cycle 
            }
        }
    }

//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Done adding any requests to DRAM, before stat collection\n");
    // DRAM latency queue
    if( !m_dram_latency_queue.empty() && ( (gpu_sim_cycle+gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) && !m_dram->full() ) {
//        printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Adding a requests to DRAM, doing a dram.push\n");
        mem_fetch* mf = m_dram_latency_queue.front().req;
        m_dram_latency_queue.pop_front();
//        printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Adding a requests to DRAM, grabbing a request to send the latency queue dram.push\n");
        m_dram->push(mf);
//        printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Adding a requests to DRAM, after dram.push\n");
    }
//    printf("In l2cahce.cc, Cycleing DRAM in dram_cycle(), Done adding any requests to DRAM, done with stat collection\n");
}

void memory_partition_unit::set_done( mem_fetch *mf )
{


    unsigned global_spid = mf->get_sub_partition_id(); 
    int spid = global_sub_partition_id_to_local_id(global_spid); 
    if(mf->get_wid() != -1) assert(m_sub_partition[spid]->get_id() == global_spid); 
    if (mf->get_access_type() == L1_WRBK_ACC || mf->get_access_type() == L2_WRBK_ACC) {
        m_arbitration_metadata.return_credit(spid); 
        MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, spid); 
    }
    m_sub_partition[spid]->set_done(mf); 
}

void memory_partition_unit::set_dram_power_stats(unsigned &n_cmd,
                                                 unsigned &n_activity,
                                                 unsigned &n_nop,
                                                 unsigned &n_act,
                                                 unsigned &n_pre,
                                                 unsigned &n_rd,
                                                 unsigned &n_wr,
                                                 unsigned &n_req) const
{
    m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd, n_wr, n_req);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses, unsigned &misses,
    std::vector<uint64_t>& accesses_s, std::vector<uint64_t>& misses_s, FILE *fp) const
{
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->print(fp, accesses, misses, accesses_s, misses_s);
}

void memory_partition_unit::print( FILE *fp ) const
{
    /*fprintf(fp, "Memory Partition %u: \n", m_id); 
    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel; p++) {
        m_sub_partition[p]->print(fp); 
    }
    fprintf(fp, "In Dram Latency Queue (total = %zd): \n", m_dram_latency_queue.size()); 
    for (std::list<dram_delay_t>::const_iterator mf_dlq = m_dram_latency_queue.begin(); 
         mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
        mem_fetch *mf = mf_dlq->req; 
        fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle); 
        if (mf) 
            mf->print(fp); 
        else 
            fprintf(fp, " <NULL mem_fetch?>\n"); 
    }*/
    m_dram->print(fp); 
}

memory_sub_partition::memory_sub_partition( unsigned sub_partition_id, 
                                            const struct memory_config *config,
                                            class memory_stats_t *stats )
{
    m_id = sub_partition_id;
    m_config=config;
    m_stats=stats;

    assert(m_id < m_config->m_n_mem_sub_partition); 

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);

    if(!m_config->m_L2_config.disabled())
       m_L2cache = new l2_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE);

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;
    sscanf(m_config->gpgpu_L2_queue_config,"%u:%u:%u:%u", &icnt_L2,&L2_dram,&dram_L2,&L2_icnt );
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2",0,icnt_L2); 
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram",0,L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2",0,dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt",0,L2_icnt);
    wb_addr=-1;


    m_tlb_bypassed_dram = new std::list<mem_fetch*>; // queue to collect bypassed dram request, need in case L2_to_DRAM_queue is full
}

memory_sub_partition::~memory_sub_partition()
{
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
}

void memory_sub_partition::cache_cycle( unsigned cycle )
{

    // L2 fill responses
    if( !m_config->m_L2_config.disabled()) {
       if ( m_L2cache->access_ready() && !m_L2_icnt_queue->full() ) {
           mem_fetch *mf = m_L2cache->next_access();

           //if(mf->get_tlb_depth_count() > 1){ //Done with this request in L2
           if(mf->get_parent_tlb_request()!=NULL){ //Done with this request in L2
                      //added TLB-related latency stats
                      m_stats->memlatstat_read_done(mf);
                      mf->done_tlb_req(mf->get_parent_tlb_request());
                      delete mf;
           }
//           else if(mf->get_tlb_depth_count() == 1){
//                      mf->get_tlb()->fill(mf->get_original_addr());
//                      mf->get_tlb()->l2_fill(mf->get_original_addr(),mf->get_appID());
//                      delete mf;
//           }
           else if(mf->get_access_type() != L2_WR_ALLOC_R){ // Don't pass write allocate read request back to upper level cache

		mf->set_reply();
		mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
		m_L2_icnt_queue->push(mf);
           }else{

				m_request_tracker.erase(mf);
				delete mf;
           }
       }
    }



    // DRAM to L2 (texture) and icnt (not texture)
    if ( !m_dram_L2_queue->empty() ) {
        mem_fetch *mf = m_dram_L2_queue->top();

//        if(!mf->bypass_L2)
//        {
//           m_stats->memlatstat_read_done(mf);
//           mf->done_tlb_req(mf->get_parent_tlb_request());
//           m_dram_L2_queue->pop();
//           delete mf;
//        }

 

        if ( !m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf) ) {
            if (m_L2cache->fill_port_free()) {
                mf->set_status(IN_PARTITION_L2_FILL_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
//                if(!mf->bypass_L2)
                    m_L2cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                m_dram_L2_queue->pop();
            }
        } else if ( !m_L2_icnt_queue->full() ) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);

           m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
        }
    }


    // prior L2 misses inserted into m_L2_dram_queue here
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->cycle();



    while(!m_icnt_L2_queue->full() && !m_tlb_bypassed_dram->empty())
    {
        mem_fetch * mf = m_tlb_bypassed_dram->front();
        m_icnt_L2_queue->push(mf); //OLD bypassing
        //if ( m_L2_dram_queue->full() ){
        //    break;
        //}
        //m_L2_dram_queue->push(mf);
        //mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        m_tlb_bypassed_dram->pop_front();
    }





    // new L2 texture accesses and/or non-texture accesses
    if ( !m_L2_dram_queue->full() && !m_icnt_L2_queue->empty() ) {
        mem_fetch *mf = m_icnt_L2_queue->top();

        if ( !m_config->m_L2_config.disabled() &&
              ( (m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only) )
           ) {
            // L2 is enabled and access is for L2
            bool output_full = m_L2_icnt_queue->full(); 
            bool port_free = m_L2cache->data_port_free(); 
            if ( !output_full && port_free ) {
                std::list<cache_event> events;
                 
                enum cache_request_status status = m_L2cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);


                if(mf->get_tlb_depth_count() > 0) //Status collection for each levels if pt-walk-related accesses
                {
                    m_stats->tlb_level_accesses[mf->get_tlb_depth_count()]++;
                    if(status == HIT)
                        m_stats->tlb_level_hits[mf->get_tlb_depth_count()]++;
                    else if(status == MISS)
                        m_stats->tlb_level_misses[mf->get_tlb_depth_count()]++;
                    else
                        m_stats->tlb_level_fails[mf->get_tlb_depth_count()]++;
                }
                else
                {
                  App* app = App::get_app(mf->get_appID());
                  //App* app = App::get_app(App::get_app_id(mf->get_appID()));
                  m_stats->l2_cache_accesses++;
                  app->l2_cache_accesses_app++;
                }

                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);

                if ( status == HIT ) {
//                 if(mf->get_tlb_depth_count() > 1){ //Hit in L2 cache for tlb-related request
                 if(mf->get_parent_tlb_request()!=NULL){ //Hit in L2 cache for tlb-related request
                      m_stats->memlatstat_read_done(mf);
                      mf->done_tlb_req(mf->get_parent_tlb_request());
                      m_icnt_L2_queue->pop(); // Get rid of the content of this queue
                      delete mf;
                 }
//                 else if(mf->get_tlb_depth_count() == 1){
//                      mf->get_tlb()->fill(mf->get_original_addr());
//                      mf->get_tlb()->l2_fill(mf->get_original_addr(),mf->get_appID());
//                      m_icnt_L2_queue->pop(); // Get rid of the content of this queue
//                      delete mf;
//                 }
                 else {
                   App* app = App::get_app(mf->get_appID());
                   //App* app = App::get_app(App::get_app_id(mf->get_appID()));
                   m_stats->l2_cache_hits++;
                   app->l2_cache_hits_app++;

                    // Done pt_walk, fill the result into TLB entries
                    if(mf->get_tlb_miss())
                    {
                        mf->get_tlb()->fill(mf->get_addr(),mf);
                        mf->get_tlb()->l2_fill(mf->get_addr(), mf->get_appID(),mf);
                        mf->get_tlb()->remove_miss_queue(mf->get_addr(), (int)mf->get_appID());
                        //mf->get_tlb()->remove_miss_queue(mf->get_addr());
                    }

                    if( !write_sent ) {
                        // L2 cache replies
                        assert(!read_sent);
                        if( mf->get_access_type() == L1_WRBK_ACC ) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                            m_L2_icnt_queue->push(mf);
                        }
                        m_icnt_L2_queue->pop();
                    } else {
                        assert(write_sent);
                        m_icnt_L2_queue->pop();
                    }
                  }
                } else if ( status != RESERVATION_FAIL ) {

                    if(mf->get_tlb_depth_count() == 0){
                      App* app = App::get_app(mf->get_appID());
                      //App* app = App::get_app(App::get_app_id(mf->get_appID()));
                      m_stats->l2_cache_misses++;
                      app->l2_cache_misses_app++;
                    }
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
                } else {
                    assert(!write_sent);
                    assert(!read_sent);
                    // L2 cache lock-up: will try again next cycle
                }
            }
        } else {
            // L2 is disabled or non-texture access to texture-only L2
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
            m_icnt_L2_queue->pop();
        }
    }


    // ROP delay queue
    if( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full() ) {
        mem_fetch* mf = m_rop.front().req;
        m_rop.pop();

        m_icnt_L2_queue->push(mf);
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        
    }


}

bool memory_sub_partition::full() const
{
    return m_icnt_L2_queue->full();
}

bool memory_sub_partition::L2_dram_queue_empty() const
{
   return m_L2_dram_queue->empty(); 
}

class mem_fetch* memory_sub_partition::L2_dram_queue_top() const
{
   return m_L2_dram_queue->top(); 
}

void memory_sub_partition::L2_dram_queue_pop() 
{
   m_L2_dram_queue->pop(); 
}

bool memory_sub_partition::dram_L2_queue_full() const
{
   return m_dram_L2_queue->full(); 
}

void memory_sub_partition::dram_L2_queue_push( class mem_fetch* mf )
{
   m_dram_L2_queue->push(mf); 
}

/*void memory_sub_partition::print_cache_stat(unsigned &accesses, unsigned &misses) const
{
    FILE *fp = stdout;
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->print(fp,accesses,misses);
}*/

void memory_sub_partition::print( FILE *fp ) const
{
    if ( !m_request_tracker.empty() ) {
        fprintf(fp,"Memory Sub Parition %u: pending memory requests:\n", m_id);
        for ( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
            mem_fetch *mf = *r;
            if ( mf )
                mf->print(fp);
            else
                fprintf(fp," <NULL mem_fetch?>\n");
        }
    }
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print( gzFile visualizer_file )
{
   // gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
   // gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_access-L2_write_miss);
   // gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
   // gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_access-L2_read_miss);
   if (total_num_mfs)
      gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_total_lat / total_num_mfs);
}

void gpgpu_sim::print_dram_stats(FILE *fout) const
{
	unsigned cmd=0;
	unsigned activity=0;
	unsigned nop=0;
	unsigned act=0;
	unsigned pre=0;
	unsigned rd=0;
	unsigned wr=0;
	unsigned req=0;
	unsigned tot_cmd=0;
	unsigned tot_nop=0;
	unsigned tot_act=0;
	unsigned tot_pre=0;
	unsigned tot_rd=0;
	unsigned tot_wr=0;
	unsigned tot_req=0;

	for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
		m_memory_partition_unit[i]->set_dram_power_stats(cmd,activity,nop,act,pre,rd,wr,req);
		tot_cmd+=cmd;
		tot_nop+=nop;
		tot_act+=act;
		tot_pre+=pre;
		tot_rd+=rd;
		tot_wr+=wr;
		tot_req+=req;
	}
    fprintf(fout,"gpgpu_n_dram_reads = %d\n",tot_rd );
    fprintf(fout,"gpgpu_n_dram_writes = %d\n",tot_wr );
    fprintf(fout,"gpgpu_n_dram_activate = %d\n",tot_act );
    fprintf(fout,"gpgpu_n_dram_commands = %d\n",tot_cmd);
    fprintf(fout,"gpgpu_n_dram_noops = %d\n",tot_nop );
    fprintf(fout,"gpgpu_n_dram_precharges = %d\n",tot_pre );
    fprintf(fout,"gpgpu_n_dram_requests = %d\n",tot_req );
}


void gpgpu_sim::L2c_print_cache_stat(FILE *fout) const {
  std::vector<uint64_t> accesses_s(ConfigOptions::n_apps, 0);
  std::vector<uint64_t> misses_s(ConfigOptions::n_apps, 0);
  unsigned accesses = 0;
  unsigned misses = 0;
  for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
    m_memory_sub_partition[i]->print_cache_stat(accesses, misses, accesses_s, misses_s, fout);
  }
	
	std::vector<uint64_t> misses_core(gpu_sms, 0);
	
	for (unsigned j = 0; j < gpu_sms; j++) {
		for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
			misses_core[j] += m_memory_sub_partition[i]->get_misses_core(j);
		}
	}
	
	fprintf(fout, "L2 Cache Total Miss Rate = %0.3f\n", (float) misses / accesses);
	for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
	  fprintf(fout, "Stream %ul: L2 Cache Miss Rate = %0.3f\n", i + 1, (float) misses_s[i] /
	      accesses_s[i]);
    fprintf(fout, "Stream %ul: Accesses  = %d\n", accesses_s[i]);
    fprintf(fout, "Stream %ul: Misses  = %d\n", misses_s[i]);
	}
	
	std::vector<float> sums(ConfigOptions::n_apps, 0.0);
	fprintf(fout, "MPKI-CORES\n");
	for (int j = 0; j < gpu_sms; j++) {
		fprintf(fout, "CORE_L2MPKI_%d\t%4.3f\n", j, (float) misses_core[j] * 1000 /
		    gpu_sim_insn_per_core[j]);
	}
	
	unsigned cluster_size = gpu_sms / ConfigOptions::n_apps;
	for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
	  for (unsigned j = 0; j < cluster_size; j++) {
	    sums[i] += (float) misses_core[j] * 1000 / gpu_sim_insn_per_core[j];
	  }
    fprintf(fout, "Avg_MPKI_Stream%ul= %4.3f\n", i + 1, (float) sums[i] / cluster_size);
	}
	
	unsigned long total_misses_core = 0;
	
	fprintf(fout, "MISSES-CORES\n");
	for (int j=0; j<gpu_sms; j++) {
		fprintf(fout, "CORE_MISSES_%d\t%ld\n", j, misses_core[j]);
		total_misses_core += misses_core[j];
	}	
	
	fprintf(fout, "L2_MISSES = %ld\n", total_misses_core);
	//fprintf(fout, "TOTAL_INSN = %ld\n", total_misses_insn);
    //fprintf(fout, "TOTAL_CTAs = %ld\n", total_ctas);
}

void gpgpu_sim::L2c_print_cache_stat_periodic(unsigned long long gpu_ins,
    std::vector<uint64_t> gpu_ins_s) const
{
    std::vector<uint64_t> accesses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> misses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> periodic_misses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> current_misses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> periodic_accesses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> current_accesses_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> periodic_inst_s(ConfigOptions::n_apps, 0);
    std::vector<uint64_t> current_inst_s(ConfigOptions::n_apps, 0);

    unsigned accesses = 0;
    unsigned misses = 0;
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
        m_memory_sub_partition[i]->print_cache_stat(accesses, misses, accesses_s, misses_s, stdout);
    }

    unsigned periodic_accesses = 0;
    unsigned current_accesses = 0;
    unsigned periodic_misses = 0;
    unsigned current_misses = 0;
    unsigned periodic_inst = 0;
    unsigned current_inst = 0;
 
    for (int i = 0; i < ConfigOptions::n_apps; i++) {
      App* app = App::get_app(App::get_app_id(i));

      periodic_misses_s[i] = misses_s[i] - current_misses_s[i];
      periodic_accesses_s[i] = accesses_s[i] - current_accesses_s[i];
      periodic_misses = misses_s[i] - current_misses;
      periodic_accesses = accesses - current_accesses;
      current_misses = misses;
      current_accesses = accesses;
      current_misses_s[i] = misses_s[i];
      current_accesses_s[i] = accesses_s[i];

      periodic_inst = gpu_ins - current_inst;
      current_inst = gpu_ins;
      periodic_inst_s[i] = gpu_ins_s[i] - current_inst_s[i];
      current_inst_s[i] = gpu_ins_s[i];


      app->periodic_miss_rate = (float) periodic_misses_s[i] / periodic_accesses_s[i];
      app->periodic_l2mpki = (float) periodic_misses_s[i] * 1000 / periodic_inst_s[i];
    }
} 


float memory_sub_partition::periodic_miss_rate(unsigned long long  gpu_sim_insn_window)
{ 
	unsigned accesses = 0;
	unsigned misses = 0;
	
	if( !m_config->m_L2_config.disabled() ) {
		m_L2cache->get_missr(accesses,misses);
	}
	
	if (gpu_sim_insn_window) {
		//miss_rate = (float) misses/accesses;
		miss_rate = (float) misses*1000/gpu_sim_insn_window;
	}
        
	m_dram->set_miss(miss_rate);	
	
	return miss_rate;
}
	
float memory_sub_partition::periodic_miss_rate_stream(unsigned long long  gpu_sim_insn_window,
    unsigned stream_idx)
{ 
	unsigned accesses = 0;
	unsigned misses = 0;
	if(!m_config->m_L2_config.disabled() ) {
		m_L2cache->get_missr_stream(accesses, misses, stream_idx);
	}
	if (gpu_sim_insn_window) {
		miss_rate_stream[stream_idx] = (float) misses * 1000 / gpu_sim_insn_window;
	}
	m_dram->set_miss_r(stream_idx, miss_rate_stream[stream_idx]);
	return miss_rate_stream[stream_idx];
}

unsigned memory_sub_partition::get_misses_core(unsigned which_core)
{ 	
	unsigned accesses = 0;
	unsigned misses = 0;
	
	if( !m_config->m_L2_config.disabled() ) {
		m_L2cache->get_misses_cum_core(accesses,misses,which_core);
	}

	return misses;
}

float memory_sub_partition::periodic_mpki_core( unsigned long long insn_data[], unsigned which_core)
{ 
	unsigned accesses;
	unsigned misses;
	
	if( !m_config->m_L2_config.disabled() ) {
		m_L2cache->get_missr_core(accesses,misses,which_core);
	}	

	if (insn_data[which_core]) {
		miss_rate_core[which_core] = (float) misses*1000/insn_data[which_core];
	}

	m_dram->set_miss_core(miss_rate_core[which_core], which_core);	
	
	return miss_rate_core[which_core];
}

unsigned memory_partition_unit::dram_bwutil() {
	return m_dram->dram_bwutil();
}
unsigned memory_sub_partition::flushL2() 
{ 
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->flush(); 
    }
    return 0; // L2 is read only in this version
}

bool memory_sub_partition::busy() const 
{
    return !m_request_tracker.empty();
}

void memory_sub_partition::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {


        m_request_tracker.insert(req);
        m_stats->memlatstat_icnt2mem_pop(req);
        if( req->istexture() ) {
            m_icnt_L2_queue->push(req);
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
        else if(req->bypass_L2)
        {
            m_request_tracker.erase(req); // This should not really matter
            m_stats->tlb_bypassed++;
            App* app = App::get_app(App::get_app_id(req->get_appID()));
            app->tlb_bypassed_app++;
            m_stats->tlb_bypassed_core[req->get_sid()]++;
            m_stats->tlb_bypassed_level[req->get_tlb_depth_count()]++;
            m_tlb_bypassed_dram->push_back(req);
        }
        else {
            rop_delay_t r;
            req->beenThroughL1 = true;
            r.req = req;
            r.ready_cycle = cycle + m_config->rop_latency;
            m_rop.push(r);
            req->set_status(IN_PARTITION_ROP_DELAY,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
    }
}

mem_fetch* memory_sub_partition::pop() 
{
    mem_fetch* mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if ( mf && mf->isatomic() )
        mf->do_atomic();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        delete mf;
        mf = NULL;
    } 
    return mf;
}

mem_fetch* memory_sub_partition::top() 
{
    mem_fetch *mf = m_L2_icnt_queue->top();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    } 
    return mf;
}

void memory_sub_partition::set_done( mem_fetch *mf )
{

    if(mf->get_wid() !=-1) m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(class cache_stats &l2_stats) const {
    if (!m_config->m_L2_config.disabled()) {
        l2_stats += m_L2cache->get_stats();
    }
}

void memory_sub_partition::get_L2cache_sub_stats(struct cache_sub_stats &css) const{
    if (!m_config->m_L2_config.disabled()) {
        m_L2cache->get_sub_stats(css);
    }
}

void memory_sub_partition::visualizer_print( gzFile visualizer_file )
{
    // TODO: Add visualizer stats for L2 cache 
}

