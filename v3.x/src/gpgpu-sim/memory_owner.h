//// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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


#ifndef MEMORY_OWNER_H
#define MEMORY_OWNER_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <list>
#include <set>
#include <queue>
#include <map>
#include <utility>
#include "App.h"
#include "../abstract_hardware_model.h"

class Hub;
class dram_t;
class tlb_tag_array;
struct mem_fetch;
class memory_stats_t;
class page_table;
class mmu;
class page;
class dram_cmd;

class page_metadata{
  public:
    page_metadata(unsigned long long parent_last_accessed_time, unsigned long long child_last_accessed_time, appid_t appID, float util, float actual_util);
    unsigned long long child_last_accessed;
    unsigned long long parent_last_accessed;
    appid_t accessed_app; //If multiple apps are in this range, return App::mixapp
    float utilization; //number of pages touched / (total possible num pages within the large page)
    float actual_utilization; //number of pages touched / (allocated pages within the large page)
};

class page_table_entry{
  public:
    page_table_entry(new_addr_type key, new_addr_type address, page_table * parent);
    new_addr_type key;
    new_addr_type addr;
    bool isLeaf;
    appid_t appID;
    bool isRead;
    bool inDRAM;
    page_table* parent_pt;
};

class page_table{
  public:
    page_table(const class memory_config * config, unsigned parent_level, /*appid_t appID, */ mmu * main_mmu);
    ~page_table();

    void set_leaf_list(std::list<new_addr_type> * the_list);

    void parse_bitmask();
    new_addr_type parse_pa(mem_fetch * mf); //Need to make sure parse_pa is aware of tlb-related addr

    //Update the page table with the corresponding new leaf node.
    void update_leaf_node(page * this_page, bool value);
    void update_leaf_node(page_table_entry * pte, bool value, appid_t appID);
    // Used by DRAM_layout, whenever a page is grabbed from a free-page list, mark this page as in-DRAM
    void set_page_in_DRAM(page * this_page);

    //Called from memory.cc
    page_table_entry * add_entry(new_addr_type address, appid_t appID, bool isRead);

    std::list<new_addr_type> * all_page_table_leaf;

    //Update page table after a DRAM copy
    void update_swapped_pages(page * swap_in, page * swap_out);
    std::map<new_addr_type, page_table_entry*> entries;
    page_table * next_level;
    unsigned current_level;
    const memory_config * m_config;
    unsigned m_bitmask;
    unsigned m_bitmask_pw;
    unsigned get_bitmask(int level);
    unsigned m_size; //Current size of the page table entries for this level
    new_addr_type current_fillable_address; //Current entries that a new entry can fill into
    unsigned address_space_ID;
    appid_t m_appID;
    unsigned addr_count;
    mmu* m_mmu;
};

class page_fault_info{
  public:
    page_fault_info(new_addr_type addr, appid_t in_appID)
    {
      appID = in_appID;
      key = addr;
    }
    new_addr_type key;
    appid_t appID;
};

//Physical page structure
class page{
  public:
    page();
    ~page(){};
    new_addr_type starting_addr; //Starting physical address
    new_addr_type va_page_addr; //VA page address. Used to check when we want to find the page
    unsigned size;
    bool dataPresent; //True means not a free page, actual data present
    bool used; //Actual page node is here, false means the page is not active page (a subpage of an active superpage)
    appid_t appID;

    std::list<page*> * sub_pages;
    page * parent_page;
    //Point to the page table level that represent this page size
    //Help make it a bit easier to coalesce a page and when we want to update the content of page table
    //This is initialized in the beginning
    page_table * pt_level;
    // For stat collection, keep track of how many of the super page are actually being used
    float utilization;

    //Easy to track these here instead of calling a function to find the mapping everytime.
    int channel_id;
    int bank_id;
    int sa_id;
    unsigned long long last_accessed_time;
    page_table_entry * leaf_pte;
};

// DRAM layout object takes care of the physical mapping of pages in DRAM
class DRAM_layout{
  public:
    std::map<unsigned, std::list<page*>*> free_pages; //Set of free page, each entries are for differing page size
    std::map<appid_t, std::list<page*>*> occupied_pages; //List of occupied pages, each entry denote the appID. Note that occupied_pages[App::noapp] means free pages within the coalesce range (bloat)

    std::map<unsigned,std::list<page*>*> all_page_list; //This is a map that contains an ordered list for all the pages in DRAM.
    std::map<unsigned,page*> all_page_map; //This is a map for PA_base to all small pages.

    std::list<new_addr_type> * all_page_table_leaf; //Keep track of the addresses of all the leaf of the page table

    float check_utilization();
    void check_utilization(page * this_page);

    void send_scan_request(); //Sending reads to DRAM for scan requests.

    const memory_config * m_config;
    // Initialize free page list and occupied page
    DRAM_layout(const class memory_config * config, page_table * root);

    void initialize_pages(page * this_page, unsigned size_index, page_table * pt);

    // This map point to the current free huge block for each app (so that they hand out pages in huge block region for different apps.
    std::map<appid_t, page*> active_huge_block;

    // Handle a page fault, return a physical page of a new page that can hold mf
    page * handle_page_fault(appid_t appID, new_addr_type key);

    // Helper function to find the target swapped page for compaction
    page* compaction_find_swapped_page(page * searched_parent_page, page * source_page);
    // Helper function that move souce_small page to target_huge range
    dram_cmd * compact_routine(page * target_huge, page * source_small);
    // Perform compaction (merge page 2 to page 1
    bool compaction(page * page1, page* page2);

    // Find a page using PA
    page * find_page_from_pa(new_addr_type pa);

    // Search parent for a free page
    page * get_free_base_page(page * parent);

    //When there is a page fault, insert them into the current active page fault queue
    void insert_page_fault(new_addr_type key, appid_t appID);
    //Return true of the address is to the location (VA) that has active page fault
    bool is_active_fault(new_addr_type address);
    void remove_page_fault();

    //Used in coalesce page, copy original page into the free page "free_page"
    void allocate_free_page(page * original_page, page * free_page);
    // Used as helper functions to coalesce page, when it has to swap pages of other apps out
    page * find_swapped_page(appid_t appID, page * searched_parent_page, page * source_page);
    page * find_free_page_for_coalesce(page * searched_parent_page, page * source_page);
    void swap_page(page * swap_in, page * swap_out);

    // Grab a free page of a certain size
    // If there is no more free page, then send DRAM command based on eviction policy.
    page * allocate_free_page(unsigned size, appid_t appID);
    // Release a page, put this page into the free page
    void release_occupied_page(page * this_page); //Done except sending DRAM commands to zero page

    //When allocate a free page, mark all the subpages (if it is a super page) as not free
    void propagate_sub_page_as_used(page * this_page);
    void propagate_parent_page_as_used(page * this_page);

    page_table * m_pt_root;
    std::vector<unsigned> * m_page_size; //Array containing list of possible page sizes
    int m_page_size_count;

    page * m_page_root;
    page * evict_page(appid_t demander, appid_t target, new_addr_type addr);

    page * get_target_page(appid_t appID_target, appid_t appID_requestor);
    appid_t get_evict_target(appid_t requestor_appID);

    unsigned DRAM_size;

    //Support for coalesce and demotion updates of metadata
    int demote_page(page *this_page);
    int coalesce_page(page * this_page); //Done except sending DRAM commands, and the two helper functions below

    //These two are helper functions
    //Helper function for coalesce page, this remove all pages in the remove list from the free list
    void remove_free_page(page* remove_page);

    //Update parent's metadata whenever there is an access (both huge and small page)
    void update_parent_metadata(page * triggered_page);

    page * get_page_from_va(new_addr_type va, appid_t appID, unsigned size); //Return the page correspond to the virtual address

    unsigned long long get_total_size(); //Return the current allocated DRAM space
    unsigned long long get_current_size(appid_t appID); //Return the current allocated DRAM space for appID


    unsigned long long page_fault_last_service_time;
    std::set<new_addr_type> page_fault_set;
    std::list<page_fault_info*> page_fault_list;

    void set_DRAM_channel(dram_t * dram_channel, int channel_id);

    void set_stat(memory_stats_t * stat);

    memory_stats_t * m_stats;

    page * allocate_PA(new_addr_type va_base, new_addr_type pa_base, appid_t appID);
    bool free_up_page(page * this_page);
    bool update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va, new_addr_type new_pa, appid_t appID);

  private:
    dram_t ** dram_channel_interface;

};


// MMU object take care of page replacement policy as well as keeping track of all the virtual pages in all address spaces.
class mmu {
  public:
    mmu();

    void init(const class memory_config * config);
    void init2(const class memory_config * config);
    void set_ready(); //Called when gpgpu-sim is done initialization

    std::list<std::pair<new_addr_type, appid_t>> promoted_list;
    std::list<std::pair<new_addr_type, appid_t>> demoted_list;

    float check_utilization();
    void check_utilization(page * this_page);

    page * find_physical_page(mem_fetch *mf); //Find if mf accesses data already in DRAM, return the page if found, otherwise, return null
    page * find_physical_page(new_addr_type va, appid_t appID, unsigned size); //Find if mf accesses data already in DRAM, return the page if found, otherwise, return null

    //Combine multiple pages into a superpage, return the new page size
    int coalesce_page(page * this_page); //Call DRAM_layout->coalesce_page


    void send_scan_request(); //Sending reads to DRAM for scan requests.

    //Assign entries in page table structures (and initialize page table entries associated with the VA), return if there is a page fault (so that the request can be handled later)

    page_table * get_page_table_root(){return m_pt_root;}

    bool check_integrity(); //Check the integrity of DRAM. No overlapping pages, etc.

    bool is_active_fault(mem_fetch * mf);
    bool is_active_fault(new_addr_type addr);


    int demote_page(page *this_page);


    //This is called from addrdec.cc, to get the physical location (row, channel, bank and subarray)
    new_addr_type get_pa(new_addr_type addr, appid_t appID); //This is called from tlb. For stats
    new_addr_type get_pa(new_addr_type addr, appid_t appID, bool isRead);
    new_addr_type get_pa(new_addr_type addr, appid_t appID, bool * fault, bool isRead);

    void clearAll();

    tlb_tag_array * l2_tlb;

    // Old, maybe model a PCIe queue using FIFO, and sending PCIe DRAM commands
    // List of pair that contain all mf that goes to access that page --> can be coalesced
    std::map<new_addr_type, new_addr_type> * va_to_pa_mapping;
    std::map<new_addr_type, page*> * va_to_page_mapping;

    DRAM_layout * get_DRAM_layout(){return m_DRAM;}

    const memory_config * get_mem_config(){return m_config;}

    unsigned get_bitmask(int level);

    void set_DRAM_channel(dram_t * dram_channel, int channel_id);

    // Get the actual dataPresent bit for all subpages in the large page (over allocated bits)
    float get_actual_util(page * large_page);

    bool need_init;

    unsigned long long m_compaction_last_probed;

    memory_stats_t * m_stats;

    int promote_page(new_addr_type va, appid_t appID);
    int demote_page(new_addr_type va, appid_t appID);

    void set_stat(memory_stats_t * stat);

    void set_L2_tlb(tlb_tag_array * L2TLB);

    //Return page's metadata given a VA.
    page_metadata * update_metadata(new_addr_type va, appid_t appID);
    bool update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va, new_addr_type new_pa, appid_t appID);
    bool allocate_PA(new_addr_type va_base, new_addr_type pa_base, appid_t appID);

    // Create a metadata to return for a given page. API for the allocator
    // NEED TO ALWAYS DELETE THIS OBJECT ONCE DONE USING
    page_metadata * create_metadata(page * this_page);


  private:

    DRAM_layout * m_DRAM;

    page_table * m_pt_root;

    const memory_config * m_config;

};

#endif
