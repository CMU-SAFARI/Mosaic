// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt, Ali Bakhoda,
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

#include <string.h>
#include "memory_owner.h"
#include "gpu-sim.h"
#include "../option_parser.h"
#include <math.h>
#include "dram.h"
#include "tlb.h"
#include "mem_latency_stat.h"
#include "../cuda-sim/cuda-sim.h"

#define DEBUG_ENABLE 0
#define ALLOC_DEBUG if(DEBUG_ENABLE) std::cout << "ALLOC_DEBUG: "
#define MERGE_DEBUG if(DEBUG_ENABLE) std::cout << "MERGE_DEBUG: "
#define PROMOTE_DEBUG if(DEBUG_ENABLE) std::cout << "PROMOTE_DEBUG: "
#define COMPACTION_DEBUG if(DEBUG_ENABLE) std::cout << "COMPACTION_DEBUG: "
#define RC_DEBUG if(DEBUG_ENABLE) std::cout << "RC_DEBUG: "
#define COALESCE_DEBUG if(DEBUG_ENABLE) std::cout << "COALESCE_DEBUG: "
#define PT_DEBUG if(DEBUG_ENABLE) std::cout << "PT_DEBUG: "
#define FAULT_DEBUG if(DEBUG_ENABLE) std::cout << "FAULT_DEBUG: "
#define UTIL_DEBUG if(DEBUG_ENABLE) std::cout << "UTIL_DEBUG: "

//Enable this will print fault queue entries


extern Hub * gpu_alloc;

page_metadata::page_metadata(unsigned long long parent_last_accessed_time,
    unsigned long long child_last_accessed_time, appid_t appID, float util, float actual_util) {
  parent_last_accessed = parent_last_accessed_time;
  child_last_accessed = child_last_accessed_time;
  accessed_app = appID;
  utilization = util;
  actual_utilization = actual_util;
}

float mmu::get_actual_util(page * large_page) {
  int current = 0;
  int max = 0;
  std::list<page*> * pages = large_page->sub_pages;
  for (std::list<page*>::iterator itr = pages->begin(); itr != pages->end(); itr++) {
    if ((*itr)->utilization > 0) {
      max++;
      if ((*itr)->dataPresent)
        current++;
    }
  }
  if (max == 0)
    return 0.0;
  else
    return (float) current / (float) max;
}

// Update page metadata based on the information of this_page. Called with this_page is accessed
page_metadata * mmu::create_metadata(page * this_page) {
  page * temp;
  float actual_util;
  //Should always fall here
  if (this_page->size == m_config->base_page_size) {
    temp = this_page->parent_page;
    check_utilization(temp); //Update the parent page's utilization
    actual_util = get_actual_util(temp);
  } else
    actual_util = temp->utilization;
  return new page_metadata(temp->last_accessed_time, this_page->last_accessed_time, temp->appID,
      temp->utilization, actual_util);
}

page_table_entry::page_table_entry(new_addr_type key, new_addr_type address, page_table * parent) {
  addr = address;
  isLeaf = false;
  appID = App::noapp.appid;
  isRead = true;
  inDRAM = false;
  parent_pt = parent; //So that leaf nodes (hence, page object) can access every entry/level of the page table if needed
}

page_table::page_table(const memory_config * config, unsigned parent_level, mmu * main_mmu) {
  printf("Initializing page_table of level %d\n", parent_level);
  m_config = config;
  m_mmu = main_mmu;
  m_size = 0;
  if (parent_level < m_config->tlb_levels - 1) {
    printf("Setting pointers current level %d to the next page_table of level %d\n", parent_level,
        parent_level + 1);
    next_level = new page_table(m_config, parent_level + 1, main_mmu);
  }
  current_level = parent_level + 1;
  parse_bitmask();
}

void page_table::set_leaf_list(std::list<new_addr_type> * the_list) {
  all_page_table_leaf = the_list;
  if (next_level != NULL)
    next_level->set_leaf_list(the_list);
}

// Parse bitmask for page table walk. Can use the page_sizes instead (above), but this seems more flexible. Downside is the two config (va_mask), tlb_levels and page_sizes has to match.
void page_table::parse_bitmask() {
  std::string mask(m_config->va_mask);
  std::string mask2(m_config->va_mask);
  for (unsigned i = 1; i <= m_config->tlb_levels; i++) {
    if (i <= current_level - 1) {
      std::replace(mask.begin(), mask.end(), (char) (i + '0'), '0');
    } else {
      std::replace(mask.begin(), mask.end(), (char) (i + '0'), '1');
    }
    if (i == current_level)
      std::replace(mask2.begin(), mask2.end(), (char) (i + '0'), '1');
    else
      std::replace(mask2.begin(), mask2.end(), (char) (i + '0'), '0');
  }
  std::bitset<32> temp(mask);
  std::bitset<32> temp2(mask2);
  m_bitmask = temp.to_ulong();
  m_bitmask_pw = temp2.to_ulong();
  printf(
      "Converting VA bitmask for page_table translation for level = %d, original string = %s, results = %x, mask_string = %s, pwcache_offset_mask = %x, mask_string = %s\n",
      current_level, m_config->va_mask, m_bitmask, mask.c_str(), m_bitmask_pw, mask2.c_str());
}

unsigned page_table::get_bitmask(int level) {
  if (level == current_level) {
    return m_bitmask;
  } else if (next_level == NULL) {
    return m_bitmask_pw;
  } else {
    return next_level->get_bitmask(level);
  }
}

void page_table::update_leaf_node(page * this_page, bool value) {
  std::map<new_addr_type, page_table_entry*>::iterator itr = entries.find(this_page->starting_addr);
  itr->second->isLeaf = value;
  itr->second->appID = this_page->appID;
}

//Called as the subroutine when page is swapped.
void page_table::update_swapped_pages(page * swap_in, page * swap_out) {
  new_addr_type key_in = (swap_in->va_page_addr) & m_bitmask;
  new_addr_type key_out = (swap_out->va_page_addr) & m_bitmask;
  page_table_entry * entry_in = entries[key_in];
  page_table_entry * entry_out = entries[key_out];
  new_addr_type temp = entry_in->key;
  entry_in->key = entry_out->key;
  entry_out->key = temp;
  temp = entry_in->addr;
  entry_in->addr = entry_out->addr;
  entry_out->addr = temp;
  appid_t temp2 = entry_in->appID;
  entry_in->appID = entry_out->appID;
  entry_out->appID = temp2;
  bool temp3 = entry_in->isLeaf;
  entry_in->isLeaf = entry_out->isLeaf;
  entry_out->isLeaf = temp3;
  temp3 = entry_in->isRead;
  entry_in->isRead = entry_out->isRead;
  entry_out->isRead = temp3;
  temp3 = entry_in->inDRAM;
  entry_in->inDRAM = entry_out->inDRAM;
  entry_out->inDRAM = temp3;
}

void page_table::update_leaf_node(page_table_entry * pte, bool value, appid_t appID) {
  pte->isLeaf = value;
  pte->appID = appID;
}

void page_table::set_page_in_DRAM(page * this_page) {
  //Only mark it as inDRAM when used for non-TLB-related pages
  if (this_page->appID != App::pt_space.appid) {
    std::map<new_addr_type, page_table_entry*>::iterator itr = entries.find(
        this_page->starting_addr & m_bitmask);
    if (itr != entries.end()) {
      itr->second->inDRAM = true;
      itr->second->appID = this_page->appID;
    }
    if (next_level != NULL)
      next_level->set_page_in_DRAM(this_page);
  }
}

//This is called from memory.cc, add a mapping between virtual address to physical address
//i.e, this populate the entries (map of addr and actual entries) of each level of page table
//Each entry contain whether the page is in DRAM or not, whether this is a leaf (in case of
//a superpage, leaf node would be at an earlier level, valid flag (nodes below leaf nodes should
//not be valid.
// Once this mapping is set, parse_pa, should be able to simply return the page_table_entry
// back to mem_fetch to process its request.
page_table_entry* page_table::add_entry(new_addr_type address, appid_t appID, bool isRead) {
  new_addr_type key = address & m_bitmask;
  page_table_entry * temp;
  std::map<new_addr_type, page_table_entry*>::iterator itr = entries.find(key);
  PT_DEBUG << "Adding page table entry for address = " << address << ", current level key = " << key
    << ", bitmask = " << m_bitmask << std::endl;
  if (itr == entries.end()) //Entry is not in the page table, page fault
  {
    PT_DEBUG << "Allocating space for page of size " << m_config->base_page_size
      << " for page table entry, current entry size = " << m_size << std::endl;
    if (m_size % (m_config->base_page_size / 64) == 0) {
      page * new_page = m_mmu->get_DRAM_layout()->allocate_free_page(m_config->base_page_size,
          App::pt_space.appid);
      if (new_page == NULL) {
        PT_DEBUG << "Not enough space to allocate for page of size " << m_config->base_page_size
          << " for page table entry" << std::endl;
        current_fillable_address = rand();
      } else
        current_fillable_address = new_page->starting_addr;
      PT_DEBUG << "Acquiring a new page for this entry. new page base addr = "
        << new_page->starting_addr << "Current fillable address = " << current_fillable_address
        << std::endl;
      temp = new page_table_entry(key, current_fillable_address, this);
      if (next_level == NULL)
        all_page_table_leaf->push_front(current_fillable_address); //If this is the leaf, add it to the address list for the scanner to probe
    } else {
      PT_DEBUG << "Appending to an existing page for this entry. Current fillable address = "
        << current_fillable_address << ", added index = "
        << m_size % (m_config->base_page_size / 64) << ", actual entry address = "
        << current_fillable_address + (m_size % (m_config->base_page_size / 64)) << std::endl;
      temp = new page_table_entry(key,
          current_fillable_address + (m_size % (m_config->base_page_size / 64)), this);
      if (next_level == NULL)
        all_page_table_leaf->push_front(
            current_fillable_address + (m_size % (m_config->base_page_size / 64))); //If this is the leaf, add it to the address list for the scanner to probe
    }
    m_size++;
    temp->isLeaf = next_level == NULL ? true : false; //Only last level is a leaf
    temp->appID = appID;
    temp->isRead = true;
    temp->inDRAM = false; //Not in DRAM. All checkings happen when get_pa is called
    PT_DEBUG << "Creating new page table entry for address = " << address
      << ", current level key = " << key << ", entry address = " << current_fillable_address
      << ", bitmask = " << m_bitmask << std::endl;
    entries.insert(std::pair<unsigned, page_table_entry*>(key, temp));
  } else {
    PT_DEBUG << "Found the page table entry for address = " << address << ", current level key = "
      << key << ", key = " << itr->second->key << ", address = " << itr->second->addr
      << std::endl;
    temp = itr->second;
  }
  if (next_level != NULL) //Propagate entires across multiple levels
    return next_level->add_entry(address, appID, isRead);
  else
    update_leaf_node(temp, true, appID); //If this is the last level, mark PTE as the leaf node
  return temp;
}

// Find the address for tlb-related data by going through the page table entries of each level
new_addr_type page_table::parse_pa(mem_fetch * mf) {
  unsigned key = mf->get_original_addr() & m_bitmask;
  PT_DEBUG << "Parsing PA for address = " << mf->get_original_addr() << ", current level = "
    << mf->get_tlb_depth_count() << ", key = " << key << ", mf->addr = " << mf->get_addr()
    << std::endl;
  std::map<new_addr_type, page_table_entry*>::iterator itr = entries.find(key);
  if (itr == entries.end()) //This should never happen. add_entry should have cover this part
  {
    //cuda-sim/memory.cc should have already add these entries
    PT_DEBUG << "Entry not found: Adding new page table entry for mf = " <<  mf->get_addr() <<
      ", level = " << mf->get_tlb_depth_count() << std::endl;
    add_entry(mf->get_addr(), mf->get_appID(), !mf->is_write());
    return parse_pa(mf);
  }
  else //Found the entry, should fall into this category most of the time
  {
    if((mf->get_tlb_depth_count()+1) == current_level)
    {
      return itr->second->addr;
    }

    if(itr->second->isLeaf && itr->second->inDRAM) //No more requests, return this node and mark it as a leaf node
    {
      return itr->second->addr;
    }
    else if(itr->second->isLeaf && !itr->second->inDRAM)
    {
      return itr->second->addr;
    }
    else //Not a leaf
    {
      return next_level->parse_pa(mf);
    }
  }
}

////////////// Physical DRAM layout (used be MMU) ///////////////

page::page() {
  dataPresent = false;
  size = 0;
  starting_addr = 0;
  used = false;
  appID = App::noapp.appid;
  sub_pages = NULL;
  parent_page = NULL;
  pt_level = NULL;
  utilization = 0.0;

  channel_id = -1;
  bank_id = -1;
  sa_id = -1;

  last_accessed_time = 0;

}

DRAM_layout::DRAM_layout(const class memory_config * config, page_table * root)
{
  //Parse page size into m_size_count and page_size
  ALLOC_DEBUG << "Initialing DRAM physical structure" << std::endl;
  m_config = config;
  DRAM_size = (unsigned)m_config->DRAM_size;

  m_page_size = m_config->page_sizes;

  m_pt_root = root;

  m_page_root = new page();
  m_page_root->starting_addr = 0;
  m_page_root->used = false;
  m_page_root->size = m_config->DRAM_size;
  m_page_root->dataPresent = false;
  m_page_root->appID = App::noapp.appid;
  m_page_root->sub_pages = new std::list<page*>();
  m_page_root->pt_level = m_pt_root;
  m_page_root->parent_page = NULL;
  m_page_root->leaf_pte = NULL;//This is just the root node, should not have PTE associated with this node

  // define channel/bank/sa mapping
  addrdec_t from_raw_addr;

  m_config->m_address_mapping.addrdec_tlx(0,&from_raw_addr, App::noapp.appid, DRAM_CMD, 0);//m_page_root
  m_page_root->channel_id = from_raw_addr.chip;
  m_page_root->bank_id = from_raw_addr.bk;
  m_page_root->sa_id = from_raw_addr.subarray;

  //  These are FIFO that take care of page fault list
  page_fault_last_service_time = 0;
  page_fault_list.clear();
  page_fault_set.clear();

  //Initilize interface to DRAM memory channels
  dram_channel_interface = new dram_t*[m_config->m_n_mem];
  all_page_table_leaf = new std::list<new_addr_type>();

  m_pt_root->set_leaf_list(all_page_table_leaf);

  for(int i=0;i<m_page_size->size();i++)
  {
    free_pages[(*m_page_size)[i]] = new std::list<page*>();
    all_page_list[(*m_page_size)[i]] = new std::list<page*>();
  }

  free_pages[DRAM_size] = new std::list<page*>();
  all_page_list[DRAM_size] = new std::list<page*>();
  free_pages[DRAM_size]->push_front(m_page_root);
  all_page_list[DRAM_size]->push_front(m_page_root);

  float utilization = 0.0;

  //Populate all the possible mapping
  initialize_pages(m_page_root, 1, m_pt_root);

  for(int i = 0; i < ConfigOptions::n_apps; i++) {
    appid_t appid = App::get_app_id(i);
    occupied_pages[appid] = new std::list<page*>();
    active_huge_block[appid] = free_pages[(*m_page_size)[m_page_size->size()-2]]->front();
    free_pages[(*m_page_size)[m_page_size->size()-2]]->pop_front();
    //Grant the first n huge blocks to each app
  }
  active_huge_block[App::pt_space.appid] = free_pages[(*m_page_size)[m_page_size->size()-2]]->front();
  occupied_pages[App::pt_space.appid] = new std::list<page*>();
  // List of bloated pages
  occupied_pages[App::noapp.appid] = new std::list<page*>();
  occupied_pages[App::mixapp.appid] = new std::list<page*>();

  ALLOC_DEBUG << "Done initialing DRAM physical structure" << std::endl;

}

//Update the utilization of this_page so that it represent the correct value. Called when metadata is needed
void DRAM_layout::check_utilization(page * this_page)
{
  float total_util = 0;
  int num_pages = 0;

  std::list<page*> * the_list2 = this_page->sub_pages;
  for(std::list<page*>::iterator itr2 = the_list2->begin(); itr2 != the_list2->end(); itr2++)
  {
    total_util = total_util + (*itr2)->utilization;
    num_pages++;
    UTIL_DEBUG << "Updating utilization page PA = " << this_page->starting_addr << ", VA = " <<
      this_page->va_page_addr << ", app = " << this_page->appID << ", used = " <<
      this_page->used << ", size = " << this_page->size << ", utilization value = " <<
      this_page->utilization << " (" << (*itr2)->utilization <<
      " was added from the previous page)" << std::endl;
  }
  this_page->utilization = total_util/(float)num_pages;
}

//Return current DRAM utilization. Also can be used to check utilization (for sanity check) if
//UTIL_DEBUG is set to true and size is 0
float DRAM_layout::check_utilization()
{
  //First, do the sanity check to make sure that utilization are consistent across the board
  float total_util = 0;
  int num_pages = 0;
  std::list<page*> * the_list = all_page_list[(*m_page_size)[m_page_size->size()-1]];
  for (std::list<page*>::iterator itr = the_list->begin(); itr != the_list->end(); itr++) {
    UTIL_DEBUG << "Checking utilization page PA = " << (*itr)->starting_addr << ", VA = " <<
      (*itr)->va_page_addr << ", app = " << (*itr)->appID << ", used = " << (*itr)->used <<
      ", size = " << (*itr)->size << ", utilization value = " << (*itr)->utilization << std::endl;
    total_util = total_util + (*itr)->utilization;
    if((*itr)->utilization > 0) num_pages++;
  }
  UTIL_DEBUG << "Current page size (size = " << (*m_page_size)[m_page_size->size()-1] <<
    ") utilization value = " << total_util << std::endl;

  //Now, update parent page's utilization.
  the_list = all_page_list[(*m_page_size)[m_page_size->size()-2]];
  for(std::list<page*>::iterator itr = the_list->begin(); itr != the_list->end(); itr++)
  {
    std::list<page*> * the_list2 = (*itr)->sub_pages;
    int num_pages2 = 0;
    for(std::list<page*>::iterator itr2 = the_list2->begin(); itr2 != the_list2->end(); itr2++)
    {
      (*itr)->utilization = (*itr)->utilization + (*itr2)->utilization;
      num_pages2++;
      UTIL_DEBUG << "Updating utilization page PA = " << (*itr)->starting_addr << ", VA = " <<
        (*itr)->va_page_addr << ", app = " << (*itr)->appID << ", used = " << (*itr)->used <<
        ", size = " << (*itr)->size << ", utilization value = " << (*itr)->utilization << " (" <<
        (*itr2)->utilization << " was added from the previous page)" << std::endl;
    }
    UTIL_DEBUG << "Done updating utilization page PA = " << (*itr)->starting_addr << ", VA = " <<
      (*itr)->va_page_addr << ", app = " << (*itr)->appID << ", used = " << (*itr)->used <<
      ", size = " << (*itr)->size << ", utilization value = " << (*itr)->utilization << std::endl;
    (*itr)->utilization = (*itr)->utilization / num_pages2;
  }
  return total_util / (float)num_pages;
}

void DRAM_layout::initialize_pages(page * this_page, unsigned size_index, page_table * pt)
{
  // Propagate this sub_page
  page * temp;
  for(unsigned i=0;i<((*m_page_size)[size_index-1]/(*m_page_size)[size_index]);i++)//Initialize all sub-pages under this page
  {
    temp = new page();
    // Physical data -- Can't be changed
    temp->starting_addr = this_page->starting_addr + (i*((*m_page_size)[size_index]));
    // Metadata, can be changed
    temp->used = false;
    temp->size = (*m_page_size)[size_index];
    temp->dataPresent = false;
    temp->appID = App::noapp.appid;
    temp->parent_page = this_page;
    temp->sub_pages = new std::list<page*>();
    temp->pt_level = pt->next_level;
    temp->leaf_pte = NULL;

    // define channel/bank/sa mapping -- Also physical data -- Can't be changed
    addrdec_t from_raw_addr;
    m_config->m_address_mapping.addrdec_tlx(temp->starting_addr,&from_raw_addr, App::noapp.appid,
        DRAM_CMD, 0);//m_page_root
    temp->channel_id = from_raw_addr.chip;
    temp->bank_id = from_raw_addr.bk;
    temp->sa_id = from_raw_addr.subarray;

    //Add this page to the free page list
    ALLOC_DEBUG << "Initialing free page list of page of size " << (*m_page_size)[size_index] <<
      ", starting address = " << temp->starting_addr << ", parent_page = " <<
      this_page->starting_addr << std::endl;
    free_pages[(*m_page_size)[size_index]]->push_front(temp);
    all_page_list[(*m_page_size)[size_index]]->push_front(temp);
    if(temp->size == (*m_page_size)[m_page_size->size()-1])//If this is the smallest page, add this page to a map of all small pages
    {
      ALLOC_DEBUG << "Adding a leaf page of size " << (*m_page_size)[size_index] <<
        ", starting address = " << temp->starting_addr << ", parent_page = " <<
        this_page->starting_addr << " to the all page map" << std::endl;
      all_page_map[temp->starting_addr] = temp; //Used by gpgpu-sim to find Page * based on PA
    }
    this_page->sub_pages->push_front(temp);
    this_page->utilization = 0.0;
    if(size_index >= (m_page_size->size()-1)) //If this is a leaf node
    {
      temp->used = true;
      temp->utilization = 0;
    }
    else
      initialize_pages(temp,size_index +1,pt->next_level);
  }
}

page * DRAM_layout::find_page_from_pa(new_addr_type pa)
{
  new_addr_type pa_base = (pa / (*m_page_size)[m_page_size->size()-1]) * (*m_page_size)[m_page_size->size()-1];
  page * res = all_page_map[pa_base];
  if (res == NULL)
    ALLOC_DEBUG << "Searching for a page using PA: Cannot find the page for PA = " <<  pa << ", searched key = " << pa_base << std::endl;
  else
    ALLOC_DEBUG << "Searching for a page using PA: Found the page for PA = " << pa <<
      ", searched key = " << pa_base << ", VA = " << res->va_page_addr << ", appID = " <<
      res->appID << ", size = " << res->size << std::endl;
  return res;
}

// Demotion
int DRAM_layout::demote_page(page * this_page)
{
  if(this_page == NULL)
  {
    printf("Somehow trying to demote a null page, current free_page size = %lld\n",free_pages[m_config->base_page_size]->size());
    return 1;
  }
  else if(this_page->size == 4096)
  {
    printf("Somehow trying to demote a small page\n");
    return 1;
  }
  else if(m_page_size->size() == 1)
  {
    printf("Somehow trying to demote when there is only one page size\n");
    return 1;
  }

  std::list<page*> add_free_list;
  std::list<page*> occupied_list;

  COALESCE_DEBUG << "Demoting a superpage PA = " <<  this_page->starting_addr << ", base_va = " <<
    this_page->va_page_addr << ", size = " << this_page->size << std::endl;
  for(std::list<page*>::iterator itr = this_page->sub_pages->begin();itr != this_page->sub_pages->end();itr++)
  {
    //        (*itr)->appID = this_page->appID;
    COALESCE_DEBUG << "Breaking subpage PA = " << (*itr)->starting_addr << ", base_va = " <<
      (*itr)->va_page_addr << ", size = " << (*itr)->size << ", appID = " << (*itr)->appID <<
      std::endl;
    (*itr)->used = true;//Mark subpage as a leaf node
    (*itr)->dataPresent = true;
    if((*itr)->appID == App::noapp.appid)//Add to free page list if this is originally a free page
      add_free_list.push_front(*itr);
    else
      occupied_list.push_front(*itr);
  }
  //Set the current node parameters
  this_page->dataPresent = false;
  this_page->used = false;
  //this_page->appID = App::noapp; //Note that the parent page still should have the same appID as its subpages, only that it is not used

  //Push all free pages to a free page list
  while(!add_free_list.empty())
  {
    page * temp = add_free_list.front();
    free_pages[temp->size]->push_back(temp);
    //Remove this page from the occupy_pages[App::noapp] list;
    occupied_pages[App::noapp.appid]->remove(temp);
    add_free_list.pop_front();
  }
  //Push all non-free pages to occupied page list
  while(!occupied_list.empty())
  {
    page * temp = occupied_list.front();
    occupied_pages[temp->appID]->push_back(temp);
    occupied_list.pop_front();
    //Note that we do not have to push these subpages back to the occupied list, coalesce_page never remove them, only just mark these pages as used
  }

  return 1;
}

//When allocate a free page, mark all the subpages (if it is a super page) as not free
void DRAM_layout::propagate_sub_page_as_used(page * this_page)
{
  if(this_page->sub_pages != NULL && !this_page->sub_pages->empty())
  {
    for(std::list<page*>::iterator itr = this_page->sub_pages->begin();itr != this_page->sub_pages->end();itr++)
    {
      (*itr)->dataPresent = false;
      (*itr)->used = false;
      remove_free_page(*itr);
      propagate_sub_page_as_used(*itr); //Depth first
    }
  }
  else return;
}

void DRAM_layout::remove_free_page(page * this_page)
{

  COALESCE_DEBUG << "Removing a page at " << this_page->starting_addr << ", va_base = " <<
    this_page->va_page_addr << ", app = " << this_page->appID << " of size = " <<
    this_page->size << " from the free page list" << std::endl;

  std::list<page*> * toSearch = free_pages[this_page->size];

  for(std::list<page*>::iterator itr = toSearch->begin(); itr!= toSearch->end(); itr++)
  {
    //Match
    if((*itr)->starting_addr == this_page->starting_addr)
    {
      toSearch->erase(itr);
      occupied_pages[this_page->appID]->push_back(*itr);
      // Update bloat status
      if(this_page->appID == App::noapp.appid && (occupied_pages[App::noapp.appid]->size() > m_stats->max_bloat))
        m_stats->max_bloat = occupied_pages[App::noapp.appid]->size();
      return;
    }
  }
}

void DRAM_layout::set_DRAM_channel(dram_t * dram_channel, int channel_id)
{
  printf("Setting DRAM interface in the MMU for channel id %d\n", channel_id);
  dram_channel_interface[channel_id] = dram_channel;
}

void DRAM_layout::set_stat(memory_stats_t * stat)
{
  printf("Setting stat object in DRAM layout\n");
  m_stats = stat;
}

void DRAM_layout::propagate_parent_page_as_used(page * this_page)
{
  if(this_page->parent_page == NULL) return;
  else
  {
    free_pages[this_page->parent_page->size]->remove(this_page->parent_page);
    // No app uses this huge range, remove this large page from the free list
    if(this_page->parent_page->appID == App::noapp.appid)
    {
      free_pages[this_page->parent_page->size]->remove(this_page->parent_page);
      occupied_pages[this_page->appID]->push_back(this_page->parent_page);
    }
    //If this page consist of mixed apps
    else if(this_page->parent_page->appID != this_page->appID)
    {
      occupied_pages[this_page->parent_page->appID]->remove(this_page->parent_page);
      occupied_pages[App::mixapp.appid]->push_back(this_page->parent_page);
    }
    propagate_parent_page_as_used(this_page->parent_page);
  }
}

// Return a free page within this huge block. Called from allocate_free_page
page * DRAM_layout::get_free_base_page(page * parent)
{
  page * free = NULL;
  std::list<page*> * temp = parent->sub_pages;
  if(temp == NULL) return NULL; //Should not reach this line
  for(std::list<page*>::iterator itr = temp->begin(); itr!=temp->end(); itr++)
  {
    if((*itr)->appID == App::noapp.appid)
    {
      free = *itr;
      break;
    }
  }
  return free;
}

// Be aware of appID, and pick the correct page (don't mix two apps together in the same huge block
//Return a free page if a certain size
//This part now cause wierd syscall error
page * DRAM_layout::allocate_free_page(unsigned size, appid_t appID)
{
  page * return_page = NULL;
  ALLOC_DEBUG << "Allocating a page of size = " << size << ", for appID = " << appID <<
    ", free page exist, free page size = " << free_pages[size]->size() << std::endl;
  if((free_pages[size]->size()) > 0) {
    ALLOC_DEBUG << "Trying to grab the front of free page list, size = " <<
      free_pages[size]->size() << ", front entry is at " <<
      free_pages[size]->front()->starting_addr << ", back is at " <<
      free_pages[size]->back()->starting_addr << std::endl;
    //Grab the free page of a certain size
    if(appID == App::pt_space.appid)//To reduce coalescing conflict
    {
      m_stats->pt_space_size = m_stats->pt_space_size + m_config->base_page_size;
    }
    //For a normal request, first get the current parent page that handle this appID current huge block range
    page * parent = active_huge_block[appID];
    //Then grab a free page within this huge block
    return_page = get_free_base_page(parent);
    if(return_page == NULL)// If there is no more free page in this huge range
    {
      assert(free_pages[parent->size]->size());
      active_huge_block[appID] = free_pages[parent->size]->front(); //Update current free huge block for this app. Note that get_pa will update this page as used in the line below (at propagate_parent_page_as_used(return_page)
      parent = active_huge_block[appID];
      return_page = get_free_base_page(parent);//Get the free page
    }
    //Remove this page from the free page list
    free_pages[size]->pop_front();
    ALLOC_DEBUG << "Returning a page of size = " << size << ", page starting address = " <<
      return_page->starting_addr << ", free page size is now at = " << free_pages[size]->size() <<
      ", appID = " << appID << ", freepage_list_front is " <<
      free_pages[size]->front()->starting_addr << ", back is " <<
      free_pages[size]->back()->starting_addr << std::endl;

    //Add this page to app
    return_page->used = true;
    return_page->dataPresent = true;
    return_page->appID = appID;
    return_page->utilization = 1.0;
    occupied_pages[appID]->push_back(return_page);
    //Remove the parent page from the free page list too
    propagate_parent_page_as_used(return_page);

    ALLOC_DEBUG << "Setting return page as used and data present" << std::endl;
    // Add all the subpages of this free page to the non_free_list
    propagate_sub_page_as_used(return_page);
    // Mark the entry in page table that the page is in DRAM
    m_pt_root->set_page_in_DRAM(return_page);//Config parameters can have discrepancy between va_mask and page_sizes
  }
  return return_page; //Return null if there are no more free page of this size
}

void DRAM_layout::swap_page(page * swap_in, page * swap_out)
{
  //Change page tables to reflect this copy. This is done first while swap_in and swap_out retain their original metadata
  swap_in->pt_level->update_swapped_pages(swap_in, swap_out);

  appid_t temp = swap_in->appID;
  swap_in->appID = swap_out->appID;
  swap_out->appID = temp;
  new_addr_type temp2 = swap_in->va_page_addr;
  swap_in->va_page_addr = swap_out->va_page_addr;
  swap_out->va_page_addr = temp2;
  float temp4 = swap_in->utilization;
  swap_in->utilization = swap_out->utilization;
  swap_out->utilization = temp4;
  //Note that we do not have to swap parent pages, SA, bank, channel IDs. The physical page structures are the same, only appID and VA page are different

  // Then, propagate the swap to all the sub_pages
  //    if(swap_in->sub_pages!=NULL)
  //    {
  //        std::list<page*>::iterator itr_in = swap_in->sub_pages->begin();
  //        std::list<page*>::iterator itr_out = swap_out->sub_pages->begin();
  //        for(;itr_in!=swap_in->sub_pages->end();)
  //        {
  //            swap_page(*itr_in,*itr_out); //Do the swapping for all the sub_pages
  //            itr_in++;
  //            itr_out++;
  //        }
  //    }

}

void DRAM_layout::release_occupied_page(page * this_page)
{
  if(this_page == NULL)
  {
    printf("Trying to releasce a NULL page!!!\n");
    return;
  }
  COALESCE_DEBUG << "Releasing occupied page from app " << this_page->appID << " with VA = " <<
    this_page->va_page_addr << ", size = " << this_page->size << ", PA = " <<
    this_page->starting_addr << std::endl;
  for(std::list<page*>::iterator itr = occupied_pages[this_page->appID]->begin();
      itr != occupied_pages[this_page->appID]->end(); itr++) {
    if((*itr) == this_page) {
      COALESCE_DEBUG << "Found the match to release in the occupied page list from app " <<
        this_page->appID << " with VA = " << this_page->va_page_addr << ", size = " <<
        this_page->size << ", PA = " << this_page->starting_addr << ", PA in occupied list = " <<
        (*itr)->starting_addr << ", this_page_address = " << this_page <<
        ", occupied_page_matched_pointer_addr = " << *itr << std::endl;
      //            //Add to the free page list
      free_pages[this_page->size]->push_back(*itr);
      //            //Remove this from occupied page
      occupied_pages[this_page->appID]->erase(itr++);
      break;//No need to go further, the element is already found and removed
    }
  }
  // Send a DRAM command to zero out the page
  dram_cmd * zero_cmd = new dram_cmd(ZERO, this_page, NULL, m_config);
  COALESCE_DEBUG << "Then, send ZERO command to DRAM to free up page at address " <<
    this_page->starting_addr << ", channel = " << zero_cmd->to_channel << ", bank = " <<
    zero_cmd->to_bank << ", subarray = " << zero_cmd->to_sa << std::endl;
  dram_channel_interface[zero_cmd->from_channel]->insert_dram_command(zero_cmd);
}

page* DRAM_layout::get_page_from_va(new_addr_type va, appid_t appID, unsigned size) {
  std::list<page*> * page_list = occupied_pages[appID];
  // This gets printed too often
  ALLOC_DEBUG << "Trying to find the physical page for va = " <<  va << ", appID = " << appID <<
    ", size = " << size << std::endl;
  for (std::list<page*>::iterator itr = page_list->begin(); itr != page_list->end();itr++) {
    ALLOC_DEBUG << "During the exhausive search (subset of bits) to find the physical page for va \
      = " << va << ", appID = " << appID << ", iterator point at " << (*itr)->va_page_addr <<
      "lx, compared value (from VA) is = " << (va & ~((*itr)->size - 1)) << std::endl;
    if((*itr)->size > 4096 && (*itr)->used)
      COALESCE_DEBUG << "During the exhausive search (subset of bits) to find the huge physical \
        page for va = " << va << ", appID = " << appID << ", iterator point at " <<
        (*itr)->va_page_addr << "lx, compared value (from VA) is = " <<
        (va & ~((*itr)->size - 1)) << std::endl;
    if((*itr)->va_page_addr == (va & ~((*itr)->size-1)) && ((*itr)->size == size)) {
      //need to check if the page is used too (otherwise it might return a subpage instead of superpage)
      COALESCE_DEBUG << "Physical page is found for VA = " << va << ", app = " << appID <<
        ", VA page addr = " << (*itr)->va_page_addr << "lx, mask = " << ~((*itr)->size - 1) <<
        ", computed mask key = " << (va & ~((*itr)->size - 1)) <<
        ", Physical page base address = " << (*itr)->starting_addr << ", size = " <<
        (*itr)->size <<  std::endl;
      return *itr;
    }
  }
  COALESCE_DEBUG << "Trying to find the physical page for va = " << va << ", appID = " << appID <<
    ", size = " << size << ", not found --  occupy page list contains:{";
  for (std::list<page*>::iterator itr = page_list->begin(); itr != page_list->end();itr++)
    COALESCE_DEBUG << "VA:" << (*itr)->va_page_addr << ", (PA:" << (*itr)->starting_addr <<
      ", Comparing " << (*itr)->va_page_addr << " == " << (va & ~((*itr)->size - 1)) <<
      "? use = " << (*itr)->used << ", size = " << (*itr)->size << ")";
  COALESCE_DEBUG << std::endl;
  return NULL;
}

// Find a page (appID) to swap into the new coalesced page, hopefullt to increase utilization
// Source_page is the location of the intended swap, so the code below can try to find good page from the same SA/bank
page * DRAM_layout::find_free_page_for_coalesce(page * searched_parent_page, page * source_page)
{
  std::list<page*> * page_list = free_pages[source_page->size];

  page * highest_prio = NULL;
  page * mid_prio = NULL;
  page * low_prio = NULL;
  page * lowest_prio = NULL;

  for(std::list<page*>::iterator itr = page_list->begin(); itr !=page_list->end(); itr++)
  {
    //If done searching, break;
    if((highest_prio != NULL) && (mid_prio != NULL) && (low_prio != NULL) && (lowest_prio !=NULL)) break;
    //Get SA and bank index for these pages

    //Prioritize pages with the same bank same SA first
    if(highest_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && (source_page->sa_id == (*itr)->sa_id) && ((*itr)->parent_page != searched_parent_page))
    {
      highest_prio = *itr;
    }
    //Then prioritize pages with the same bank
    if(mid_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && ((*itr)->parent_page != searched_parent_page))
    {
      mid_prio = *itr;
    }
    //Then any page of the same size
    if(low_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && ((*itr)->parent_page != searched_parent_page))
    {
      low_prio = *itr;
    }
    if(low_prio == NULL && ((*itr)->parent_page != searched_parent_page))
    {
      lowest_prio = *itr;
    }
  }
  // Can add other policy as needed, but the DRAM config (LISA/RC enabled) should be able to take care of latency modeling correctly
  if(highest_prio != NULL) return highest_prio;
  else if(mid_prio != NULL) return mid_prio;
  else if(low_prio != NULL) return low_prio;
  else return lowest_prio;
  //If return null, it means this app occupy no other page to swap into
}

// Find a page (appID) to swap into the new coalesced page, hopefullt to increase utilization
// Source_page is the location of the intended swap, so the code below can try to find good page from the same SA/bank
page * DRAM_layout::compaction_find_swapped_page(page * searched_parent_page, page * source_page)
{
  std::list<page*> * page_list = searched_parent_page->sub_pages; //Only difference is now it searches occupied page

  page * highest_prio = NULL;
  page * mid_prio = NULL;
  page * low_prio = NULL;
  page * lowest_prio = NULL;

  for(std::list<page*>::iterator itr = page_list->begin(); itr !=page_list->end(); itr++)
  {
    //If done searching, break;
    if((highest_prio != NULL) && (mid_prio != NULL) && (low_prio != NULL) && (lowest_prio !=NULL)) break;
    //Get SA and bank index for these pages

    //Prioritize pages with the same bank same SA first
    if(highest_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && (source_page->sa_id == (*itr)->sa_id) && ((*itr)->appID == App::noapp.appid))
    {
      highest_prio = *itr;
    }
    //Then prioritize pages with the same bank
    if(mid_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && ((*itr)->appID == App::noapp.appid))
    {
      mid_prio = *itr;
    }
    //Then any page of the same size
    if(low_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && ((*itr)->appID == App::noapp.appid))
    {
      low_prio = *itr;
    }
    if(low_prio == NULL && ((*itr)->appID == App::noapp.appid))
    {
      lowest_prio = *itr;
    }
  }
  // Can add other policy as needed, but the DRAM config (LISA/RC enabled) should be able to take care of latency modeling correctly
  if(highest_prio != NULL) return highest_prio;
  else if(mid_prio != NULL) return mid_prio;
  else if(low_prio != NULL) return low_prio;
  else return lowest_prio;
  //If return null, it means this app occupy no other page to swap into
}

// Find a page (appID) to swap into the new coalesced page, hopefullt to increase utilization
// Source_page is the location of the intended swap, so the code below can try to find good page from the same SA/bank
page * DRAM_layout::find_swapped_page(appid_t appID, page * searched_parent_page, page * source_page)
{
  std::list<page*> * page_list = occupied_pages[appID]; //Only difference is now it searches occupied page

  page * highest_prio = NULL;
  page * mid_prio = NULL;
  page * low_prio = NULL;
  page * lowest_prio = NULL;

  for(std::list<page*>::iterator itr = page_list->begin(); itr !=page_list->end(); itr++)
  {
    //If done searching, break;
    if((highest_prio != NULL) && (mid_prio != NULL) && (low_prio != NULL) && (lowest_prio !=NULL)) break;
    //Get SA and bank index for these pages

    //Prioritize pages with the same bank same SA first
    if(highest_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && (source_page->sa_id == (*itr)->sa_id) && ((*itr)->parent_page != searched_parent_page))
    {
      highest_prio = *itr;
    }
    //Then prioritize pages with the same bank
    if(mid_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && ((*itr)->parent_page != searched_parent_page))
    {
      mid_prio = *itr;
    }
    //Then any page of the same size
    if(low_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && ((*itr)->parent_page != searched_parent_page))
    {
      low_prio = *itr;
    }
    if(low_prio == NULL && ((*itr)->parent_page != searched_parent_page))
    {
      lowest_prio = *itr;
    }
  }
  // Can add other policy as needed, but the DRAM config (LISA/RC enabled) should be able to take care of latency modeling correctly
  if(highest_prio != NULL) return highest_prio;
  else if(mid_prio != NULL) return mid_prio;
  else if(low_prio != NULL) return low_prio;
  else return lowest_prio;
  //If return null, it means this app occupy no other page to swap into
}

dram_cmd * DRAM_layout::compact_routine(page * target_huge, page * source_small)
{
  dram_cmd * return_cmd;
  COMPACTION_DEBUG << "Compaction routine for merging small page into a huge page called at cycle " << gpu_sim_cycle << std::endl;
  //Find a free page within page 1 range to swap into
  page * target_page = compaction_find_swapped_page(target_huge,source_small);

  //Set the metadata for both pages, setup DRAM commands
  return_cmd = new dram_cmd(COPY,source_small,target_page,m_config);
  target_page->appID = target_huge->appID;
  target_page->used = false;
  target_page->dataPresent = source_small->dataPresent;
  target_page->utilization = source_small->utilization;
  target_page->va_page_addr = source_small->va_page_addr;

  source_small->appID = App::noapp.appid;
  source_small->used = false;
  source_small->dataPresent = false;
  source_small->utilization = 0.0;
  source_small->va_page_addr = 0;

  // Remove this free page from occupied_pages[App::noapp].
  occupied_pages[App::noapp.appid]->remove(target_page);
  // Free up the new page
  occupied_pages[target_huge->appID]->remove(source_small);
  free_pages[source_small->size]->push_back(source_small);
  // Put this new allocated page to the occupied page list
  occupied_pages[target_huge->appID]->push_back(target_page);
  return return_cmd;
}

//Sending scan requests to DRAM
void DRAM_layout::send_scan_request()
{
  //Goes through all the addresses and send scan (essentially this works similar to a read command)
  for(std::list<new_addr_type>::iterator itr = all_page_table_leaf->begin(); itr!= all_page_table_leaf->end(); itr++)
  {
    //Get the associated channel and bank ID, need to check how sending App::pt_space would behave in addrdec.cc, should be fine.

    addrdec_t raw_addr;
    m_config->m_address_mapping.addrdec_tlx((*itr),&raw_addr, App::pt_space.appid, 0, 0);
    int channel_from = raw_addr.chip;
    int bank_from = raw_addr.bk;
    int sa_from = raw_addr.subarray;
    dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(SCAN,
        bank_from,bank_from,channel_from,channel_from,sa_from,sa_from,64,App::pt_space.appid,m_config));
  }
}

//Perform compaction by merging pages from page 2 to pages from page 1, free up page 2
bool DRAM_layout::compaction(page * page1, page* page2)
{
  COMPACTION_DEBUG << "Compaction routine called at cycle " << gpu_sim_cycle << std::endl;
  if(page1 == NULL || page2 == NULL) {
    printf("Trying to perform compaction on a NULL page\n");
    return false;
  }
  if(page1->sub_pages == NULL) {
    printf("Trying to perform compaction into a NULL sub pages\n");
    return false;
  }
  std::list<dram_cmd*> command_list;
  //If page2 is the sub-block size. Note, copied from the routine below
  if(page1->size != page2->size) {
    if(page1->size > page2->size)
    {
      compact_routine(page1,page2);
    }
    else
    {
      printf("Trying to perform compaction on different page sizes\n");
      return false;
    }
  }
  else {
    //Count number of free sub-pages to make sure compaction is possible
    if(page1->utilization + page2->utilization > 1) return false;

    std::list<page*> * sub_pg = page2->sub_pages;
    for(std::list<page*>::iterator itr = sub_pg->begin(); itr!=sub_pg->end();itr++)
    {
      if((*itr)->appID != page1->appID) //The page to be moved
      {
        compact_routine(page1,*itr);
        // compact_routine perfrom these tasks
        //                //Find a free page within page 1 range to swap into
        //                page * target_page = compaction_find_swapped_page(page1,(*itr));
        //
        //                //Set the metadata for both pages, setup DRAM commands
        //                command_list.push_back(new dram_cmd(COPY,(*itr),target_page,m_config));
        //                target_page->appID = page1->appID;
        //                target_page->used = false;
        //                target_page->dataPresent = (*itr)->dataPresent;
        //                target_page->utilization = (*itr)->utilization;
        //                target_page->va_page_addr = (*itr)->va_page_addr;
        //
        //                (*itr)->appID = App::noapp;
        //                (*itr)->used = false;
        //                (*itr)->dataPresent = false;
        //                (*itr)->utilization = 0.0;
        //                (*itr)->va_page_addr = 0;
        //
        //
        //                // Remove this free page from occupied_pages[App::noapp].
        //                occupied_pages[App::noapp]->remove(target_page);
        //                // Free up the new page
        //                occupied_pages[page1->appID]->remove(*itr);
        //                free_pages[(*itr)->size]->push_back(*itr);
        //                // Put this new allocated page to the occupied page list
        //                occupied_pages[page1->appID]->push_back(target_page);
      }
    }
  }
  // Once done, update utilization
  float util_sum = 0.0;
  float count = 0.0;
  for(std::list<page*>::iterator itr = page1->sub_pages->begin(); itr!=page1->sub_pages->end();itr++)
  {
    util_sum+= (*itr)->utilization;
    count += 1;
  }
  page1->utilization = util_sum/count;
  // Add the new free huge page to the free page list
  free_pages[page2->size]->push_back(page2);
  //Free up page2
  page2->used = false;
  page2->appID = App::noapp.appid;
  page2->va_page_addr = 0;
  page2->utilization = 0.0;
  page2->dataPresent = false;

  //Send DRAM commands
  while(!command_list.empty())
  {
    dram_cmd * front = command_list.front();
    // Note that DRAM will automatically block to_channel if from_channel is different from to_channel
    dram_channel_interface[front->from_channel]->insert_dram_command(front);
    command_list.pop_front();
  }
  return true;
}

// Coalsece, return coalescing result ID: 0 = no coalesce happened (similar to return false in earlier versions, 1 = coalsece succeed. No need to invalidate, 2 = coalesce succeed but need invalidation
int DRAM_layout::coalesce_page(page * this_page)
{

  //return true;
  //return false;

  if(this_page == NULL) {
    printf("Somehow trying to coalesce a null page (O_O ) , current free_page size = %lld\n",free_pages[m_config->base_page_size]->size());
    return 0;
  }
  else if(this_page->size != 4096) {
    printf("Somehow trying to coalesce superpage together\n");
    return 0;
  }
  else if(m_page_size->size() == 1)
  {
    printf("Somehow trying to coalesce when there is only one page size\n");
    return 0;
  } else {
    COALESCE_DEBUG << "In coalescing routine, Coalescing the page " << this_page->starting_addr <<
      ", base VA = " << this_page->va_page_addr << ", size = " << this_page->size << std::endl;
  }

  m_stats->num_coalesce++;

  //First, check if adjacent pages can be combined
  std::list<dram_cmd*> dram_cmd_list;
  std::list<page*> remove_free_list;
  //std::list<page*> combined_list;
  dram_cmd_list.empty();
  remove_free_list.empty();
  //combined_list.empty();

  std::list<page*> swap_index;//This is a list containing pages that have to be swapped out for coalescing to happen
  std::list<page*> move_to_free_list;//This is a list containing pages that have to be swapped out if we were to move all pages under the same appID to a new bigger free page

  bool can_merge = true;
  //Go through all the parent's sub_pages_list, see if they are from the same appID
  page * parent = this_page->parent_page;
  //VA base address of the huge page
  new_addr_type va_base = ((this_page->va_page_addr) / parent->size) * parent->size;
  if(parent == NULL) {
    printf("For some reason, trying to coalesce a page with no parent page. BAD\n");
    assert(false);
  } else  {
    COALESCE_DEBUG << "Parent of the coalesced page is " << parent->starting_addr <<
      ", base VA = " << parent->va_page_addr << ", size = " << parent->size << std::endl;
  }
  float avg_util = 0.0;
  int count = 0;
  COALESCE_DEBUG << "in MMU, coalescing page with base addr = " <<  this_page->starting_addr <<
    " (VA = " << this_page->va_page_addr << "), parent page is " << parent->starting_addr <<
    std::endl;

  //this out by searching for the parent page in the occupied_pages[APP]. Can
  //coalesce right away when it exists. If multiple pages are there, then
  //this parent page should be in occupied_pages[App::mixapp]

  for(std::list<page*>::iterator itr = parent->sub_pages->begin();itr != parent->sub_pages->end();itr++)
  {
    count++;
    if (((*itr)->appID != this_page->appID) && ((*itr)->appID != App::noapp.appid)) {
      COALESCE_DEBUG << "A page from other app is within the coalesce range, can merge = " <<
        can_merge << ", conflicting appID = " << (*itr)->appID << ", page base PA = " <<
        (*itr)->starting_addr << "lx, page base VA = " << (*itr)->va_page_addr << std::endl;
      if (!m_config->enable_costly_coalesce) return 0;//Usually if it reach this point, can return right away because we detect there is a page of other app within the huge block range
      can_merge = false;//Mark this as not coalesable, need to find another free location
      swap_index.push_back((*itr));//Mark this page as it has to be swapped out
    }
    else if((*itr)->appID == App::noapp.appid) { //Free page within the coalesced range
      COALESCE_DEBUG << "A free page within the coalesce range, can merge = " << can_merge <<
        ", conflicting appID = " << (*itr)->appID << ", page base PA = " <<
        (*itr)->starting_addr << "lx, page base VA = " << (*itr)->va_page_addr << std::endl;
      remove_free_list.push_back(*itr);
    } else { //Used page in the coalesced range
      COALESCE_DEBUG << "An occupied page of the same app in coalesce range, can merge = " <<
        can_merge << ", conflicting appID = " << (*itr)->appID << ", page base PA = " <<
        (*itr)->starting_addr << "lx, page base VA = " << (*itr)->va_page_addr << std::endl;
      move_to_free_list.push_back((*itr));
      avg_util += (*itr)->utilization;
      //combined_list.insert(itr);
    }
  }

  COALESCE_DEBUG << "Checking if no copy required (should be the case for 1 app, can merge = " <<
    can_merge << std::endl;

  // At this point, done with checking if we can just promote the entire block to a superpage.
  if(can_merge) {
    COALESCE_DEBUG << "Can directly coalesce the page " << this_page->starting_addr <<
      ", base VA = " << this_page->va_page_addr << ", size = " << this_page->size <<
      ", free page size = " << free_pages[parent->size] << std::endl;
    // Mark parent as a new used superpage
    parent->used = true;
    parent->dataPresent = true;
    parent->appID = this_page->appID;
    parent->utilization = avg_util/(float)count;

    parent->va_page_addr = va_base;

    // Update all sub-page table with the coalesced page
    for(std::list<page*>::iterator it = parent->sub_pages->begin(); it != parent->sub_pages->end();
        it++) {
      (*it)->used = false; //Mark the page as not used (because superpage is the active node)
    }

    // Remove all free page under this super page range
    while (!remove_free_list.empty()) {
      page* remove_page = remove_free_list.front();
      COALESCE_DEBUG << "Removing sub page pa = " << remove_page->starting_addr << ", va_base = " << remove_page->va_page_addr << ", size = " << remove_page->size << std::endl;
      remove_free_page(remove_page);
      remove_free_list.pop_front();
    }

    //Update the page table with the corresponding new leaf node (parent).
    parent->pt_level->update_leaf_node(parent,true);
    this_page->pt_level->update_leaf_node(parent,false);

    return 1;
  }

  // Parts below are what we should avoid (i.e., we can't coalesce the page right aware as there are pages of other app within the huge block range)
  // Implementations is there so we can test and compare as needed

  //Need to use some other page, can't coalesce without copy
  else {
    COALESCE_DEBUG << "Need to coalesce by moving to free page for " << this_page->starting_addr <<
      ", base VA = " << this_page->va_page_addr << ", size = " << this_page->size <<
      ", free page size = " << free_pages[parent->size] << std::endl;

    //Step2: See if there is a free large page, if so, move all subpages there and free this old page
    if(free_pages[parent->size] > 0)//Check if there is a large page
      can_merge = true;
    else
      can_merge = false;

    if (can_merge)
      COALESCE_DEBUG << "Cannot just merge, but free huge page is available need to migrate to a \
        new free page, pa_base = " << this_page->starting_addr << ", va_base = " <<
        this_page->va_page_addr << ", size = " << this_page->size <<
        ". Target free page is at " << free_pages[parent->size]->front() << ", size " <<
        parent->size << std::endl;
    else
      COALESCE_DEBUG << "Cannot just merge, and no free huge page is available need to migrate to a new free page, pa_base = " << this_page->starting_addr << ", va_base = " << this_page->va_page_addr << ", size = " << this_page->size << std::endl;

    if(can_merge) {
      //Move everything from app1 to the large free page
      // Need to make sure that this superpage is really free ... Whenever a page is allocated, the superpage range need to also be updated
      page * target_free_page = free_pages[parent->size]->front();
      appid_t this_appID = this_page->appID;
      int count = move_to_free_list.size();

      COALESCE_DEBUG << "Attempting to move set of sub pages with huge page va_base at " <<
        va_base << " to free_page at PA = " << target_free_page->starting_addr << " (VA = " <<
        target_free_page->va_page_addr << "), appID = " << this_appID <<
        ", free page list size = " << free_pages[parent->size]->size() << std::endl;

      while (!move_to_free_list.empty()) { //This list contain pages with the same appID as this_page
        //Populate the sub pages of target_free_page with pages in the move_to_free_list
        page * swap_out = move_to_free_list.front();
        page * swap_in = find_free_page_for_coalesce(target_free_page,swap_out);
        //Update the free page list/occupied page list and send the zero DRAM command
        COALESCE_DEBUG << "Moving sub page pa = " << swap_out->starting_addr << ", va_base = " <<
          swap_out->va_page_addr << ", size = " << swap_out->size << ", with free page pa = " <<
          swap_in->starting_addr << ", va = " << swap_in->va_page_addr << ", size = " <<
          swap_in->size << std::endl;

        release_occupied_page(swap_out);
        swap_page(swap_in,swap_out);//This seems fine as long as we do not recursively swap its subpages (no need to do this with two page sizes I think)

        //Generate all corresponding DRAM commands for copy, always goes in front
        dram_cmd_list.push_back(new dram_cmd(COPY,swap_out,swap_in,m_config));
        //Lastly, remove this from the list
        move_to_free_list.pop_front();
      }


      //Set the new parent page: VA_base_page, appID, isleaf
      target_free_page->appID = this_appID;
      target_free_page->va_page_addr = va_base;
      target_free_page->dataPresent = true;
      target_free_page->used = true;
      target_free_page->utilization = (float) count / (float) target_free_page->sub_pages->size();
      //Update the occupied page list. Also, note that this huge page range (old location) does not become free because other apps' pages are still in there, no need to add this huge page to the free page
      COALESCE_DEBUG << "Pushing huge page at PA = " << target_free_page->starting_addr <<
        ", VA = " << target_free_page->va_page_addr << ", app = " << target_free_page->appID <<
        ", size = " << target_free_page->size << " to the occupied page list" << std::endl;
      occupied_pages[this_appID]->push_back(target_free_page);
      //target_free_page->pt_level->update_leaf_node(target_free_page,true);

      //Fine up to here
      return 2;
    } //End of step 2
    else//Begin step 3: No free large page available, copy pages from other apps out
    {
      COALESCE_DEBUG << "No more huge free page for coalesceing page " <<
        this_page->starting_addr << ", base VA = " << this_page->va_page_addr << ", size = " <<
        this_page->size << ", free page size = " << free_pages[parent->size] << std::endl;
      return 0;
      // Do we want to copy from other location to fill in this superpage?
      // If this is the case, setup DRAM commands to send

      //These two list keep track of which pages are moved
      std::list<page *> taken_free_pages;
      std::list<page *> moved_to_free_pages;
      std::list<page *> taken_swapped_pages;
      std::list<page *> moved_to_occupied_pages;
      while(!swap_index.empty()) {
        bool found = false;
        page * source = swap_index.front();
        //Try to find another free page of this size for need-to-be-swapped pages
        page * target = find_free_page_for_coalesce(parent, source);
        if(target != NULL)
        {
          found = true;
          taken_free_pages.push_back(target);
          moved_to_free_pages.push_back(swap_index.front());
          swap_index.pop_front();
        }
        else
        {
          target = find_swapped_page(source->appID, parent, source);
          found = target==NULL?false:true;
          taken_swapped_pages.push_back(target);
          moved_to_occupied_pages.push_back(swap_index.front());
          swap_index.pop_front();
        }
        if(found == false) break; //No other pages from the same app to swap in, and no free page exist. Cannot coalesce
      }
      if(swap_index.empty())//Can be coalesce, commit all the copies and send DRAM commands
      {
        //If coalescable, copy old pages to new free pages
        while(!taken_free_pages.empty())
        {
          //Zero out the old page location, swap the page out
          allocate_free_page(moved_to_free_pages.front(),taken_free_pages.front());
          release_occupied_page(moved_to_free_pages.front());
          dram_cmd_list.push_back(new dram_cmd(ZERO,moved_to_free_pages.front(),NULL,m_config));

          //Setup DRAM commands for the copy
          dram_cmd_list.push_back(new dram_cmd(COPY,moved_to_free_pages.front(),taken_free_pages.front(),m_config));
          taken_free_pages.pop_front();
          moved_to_free_pages.pop_front();
        }
        //If coalescable, swap new page into the old page
        while(!taken_swapped_pages.empty())
        {
          page * swap_in = taken_swapped_pages.front();
          page * swap_out = moved_to_occupied_pages.front();
          swap_page(swap_in, swap_out);//Swap the page metadata
          //Then create the actual DRAM command to be send
          //The first command to reflect the latency for copying swap_in to temporary DRAM cache
          dram_cmd_list.push_back(new dram_cmd(COPY,swap_in,swap_in,m_config));
          //Then copy from swap_in to swap_out
          dram_cmd_list.push_back(new dram_cmd(COPY,swap_in,swap_out,m_config));
          //Then another latency to copy from temporary DRAM cache back
          dram_cmd_list.push_back(new dram_cmd(COPY,swap_out,swap_out,m_config));
          //Then pop the queue
          taken_swapped_pages.pop_front();
          moved_to_occupied_pages.pop_front();
        }
      }
    } //End of step 3
  }

  //Send DRAM commands to coalesce page, for timing model
  while(!dram_cmd_list.empty())
  {
    dram_channel_interface[dram_cmd_list.front()->from_channel]->insert_dram_command(dram_cmd_list.front());
    dram_cmd_list.pop_front();
  }

}

////////////// MMU ///////////////

mmu::mmu() {
  need_init = true;
}

void mmu::init(const class memory_config * config) {
  printf("Initializing MMU object\n");
  m_config = config;
}

void mmu::init2(const class memory_config * config) {
  printf("Initializing MMU object\n");

  m_config = config;

  m_compaction_last_probed = 0;

  //Initialize the page table object
  printf("Setting up page tables objects\n");
  m_pt_root = new page_table(m_config, 0, this);
  printf("Done setting up page tables, setting up DRAM physical layout\n");

  va_to_pa_mapping = new std::map<new_addr_type, new_addr_type>();
  va_to_page_mapping = new std::map<new_addr_type, page*>();
  //   va_to_metadata_mapping = new std::map<new_addr_type,page_metadata*>();

  //Initilize page mapping. All pages are now free, all leaf are at the base page size
  m_DRAM = new DRAM_layout(m_config, m_pt_root);

  printf("Done setting up MMU\nSending pending promotion/demotion requests to TLBs\n");

}

void mmu::set_ready() {
  printf("Setting the MMU object as ready\n");
  need_init = false;
}

// Done grabbing data for the page, now need to evict a page for appID and virtual page key
page * DRAM_layout::handle_page_fault(appid_t appID, new_addr_type key) {
  //Evict a page and grab a free page for mf's data
  ALLOC_DEBUG << "Handling page fault (with data arrived at the GPU, on virtual address = " <<
    key << ", app = " << appID << std::endl;
  return evict_page(appID, get_evict_target(appID), key);
}

// A helper function for evict page. Return a free page, if there is no free page, evict a page from appID
page * DRAM_layout::get_target_page(appid_t appID_target, appid_t appID_requestor) {
  //LRU as the default policy, can be changed later
  ALLOC_DEBUG<< "Trying to allocate a free page for application " << appID_requestor << std::endl;
  page * return_page = allocate_free_page(m_config->base_page_size,appID_requestor); //In case there is a free page available, no eviction occur
  if(return_page !=NULL)//If there is a free page
  {
    ALLOC_DEBUG << "Free page exist, grabbing a free page for application " << appID_requestor << ", free page PA = " << return_page->starting_addr << std::endl;
    return_page->appID = appID_requestor;
    return return_page;
  }
  else
  {
    ALLOC_DEBUG << "No free page exist, Releasing a page from application " << appID_target << std::endl;
    page * released_page = occupied_pages[appID_target]->front(); //FIXME: fix get_evicted_target when we have page replacement policy. Always assume this app has an occupied page
    ALLOC_DEBUG << "Got a target page (PA = " <<  released_page->starting_addr << ", original Virtual page addr = " << released_page->va_page_addr << ") from application " << appID_target << std::endl;
    release_occupied_page(released_page);//Release the page from the occupied page
    ALLOC_DEBUG << "Finish moving the occupied page to the free page list, before grabbing this free page from the free page list for " << appID_requestor << std::endl;
    return_page = allocate_free_page(return_page->size, appID_requestor);
    return_page->appID = appID_requestor;
  }
  return return_page;
}

page * DRAM_layout::evict_page(appid_t demander, appid_t target, new_addr_type va_key) {
  // This also handles insertion into the occupied list. It takes a page from the target (or from a free page), and give it to the demander
  page * evicted_page = get_target_page(target, demander);
  if (evicted_page == NULL) {
    printf("NOOOOOOOOO ... This should not happen. Cannot evict a page for some reason\n");
    assert(false);
  }
  else //Set page info
  {
    evicted_page->appID = demander;
    ALLOC_DEBUG << "Adding a new page for app = " << demander << ", VA page = " << va_key <<
      ", page_size = " << evicted_page->size << std::endl;
    evicted_page->va_page_addr = va_key * (evicted_page->size);
  }
  return evicted_page; //Allocate this page as it belongs to the appID
}

appid_t DRAM_layout::get_evict_target(appid_t requestor_appID) {
  //FIXME: Always evict page from app 0, for now -- This is not supported atm
  assert(false);
}

bool mmu::is_active_fault(mem_fetch * mf) {
  if (m_config->page_transfer_time == 0 || m_config->enable_PCIe == 0)
    return false; //If PCIe modeling is disabled
  else
    return m_DRAM->is_active_fault(mf->get_addr());
}

//Clear page fault list when
void DRAM_layout::remove_page_fault() {
  if (m_config->page_transfer_time == 0 || m_config->enable_PCIe == 0)
    return;

  // Retire any fault that is done
  ALLOC_DEBUG << "Checking if there are any page fault resolved, cycle = " << gpu_sim_cycle <<
    ". Current ongoing fault size = " << page_fault_list.size() << std::endl;

  if (!page_fault_list.empty()
      && ((gpu_sim_cycle + gpu_tot_sim_cycle)
        > (page_fault_last_service_time + m_config->page_transfer_time))) {
    page_fault_info * done = page_fault_list.front();
    page * new_page = handle_page_fault(done->appID, done->key); //appID is the page that cause the fault, key is the virtual page of the fault. After calling this, occupied_page[appID] must contains key

    ALLOC_DEBUG << "Page fault resolved for page = " << done->key << ", appID = " << done->appID <<
      ", handling it at the moment. Page fault size = " << page_fault_set.size() << std::endl;

    page_fault_set.erase(done->key);
    page_fault_list.pop_front();
    FAULT_DEBUG << "[FAULT_Q_STAT] size = " << page_fault_set.size() << ": {";
    for (std::set<new_addr_type>::iterator itr = page_fault_set.begin();
        itr != page_fault_set.end(); itr++)
      FAULT_DEBUG << *itr << ", ";
    FAULT_DEBUG << "}" << std::endl;

    delete done; //Delete the page fault request
    page_fault_last_service_time = gpu_sim_cycle + gpu_tot_sim_cycle;
  }

}

bool DRAM_layout::is_active_fault(new_addr_type address) {
  new_addr_type key = address / m_config->base_page_size;
  ALLOC_DEBUG << "Is active fault check routine for address = " << address << std::endl;
  remove_page_fault();
  std::set<new_addr_type>::iterator itr = page_fault_set.find(key);
  if (itr == page_fault_set.end()) {
    ALLOC_DEBUG << "Checking if addr = " << address <<
      " is an access in the fault region, not a fault" << std::endl;
    return false;
  } else {
    ALLOC_DEBUG << "Checking if addr = " << address <<
      " is an access in the fault region, this a fault. page that fault = " << *itr << std::endl;
    return true;
  }
}

//Ongoing page fault, do nothing. is_active_fault will handle the rest
//New fault, handle page fault and add this page to the associated fault set and list
void DRAM_layout::insert_page_fault(new_addr_type key, appid_t appID) {
  remove_page_fault();
  //Reset the timing for faults
  if (page_fault_list.empty())
    page_fault_last_service_time = gpu_sim_cycle + gpu_tot_sim_cycle;

  if (page_fault_set.find(key) == page_fault_set.end()) {
    ALLOC_DEBUG << "Inserting page fault to page = " << key << " from app = " << appID <<
      " to the FIFO, current size = " << page_fault_set.size() << std::endl;
    FAULT_DEBUG << "[FAULT_Q_STAT] size = " << page_fault_set.size() << ": {";
    for (std::set<new_addr_type>::iterator itr = page_fault_set.begin();
        itr != page_fault_set.end(); itr++)
      FAULT_DEBUG << *itr << ", ";
    FAULT_DEBUG << "}" << std::endl;

    page_fault_set.insert(key);
    page_fault_list.push_back(new page_fault_info(key, appID)); //Add this to the pending fault FIFO queue
  }
}

//Return true if the request is resolving a page fault. This is called from tlb.cc, which will block accesses to this page when fault happens
bool mmu::is_active_fault(new_addr_type address) {
  m_DRAM->is_active_fault(address);
}

//This is used from tlb.cc
new_addr_type mmu::get_pa(new_addr_type addr, appid_t appID) {

  return get_pa(addr, appID, true);
}

//Return physical address given a mem fetch. This is called from addrdec when it only want to check which memory partition a request go to
new_addr_type mmu::get_pa(new_addr_type addr, appid_t appID, bool isRead) {
  bool fault;
  return get_pa(addr, appID, &fault, isRead);
}

void mmu::send_scan_request() {
  m_DRAM->send_scan_request();
}

//This function return page metadata of va (with its appID so that it gives the correct page)
page_metadata * mmu::update_metadata(new_addr_type va, appid_t appID) {
  //Get the page from VA
  new_addr_type searchID = ((va >> m_config->page_size) << m_config->page_size); // | appID;
  assert(false);
  // this was originally ored with appID. WHY???
  page * the_page = (*va_to_page_mapping)[searchID];

  //Whoever call this please do not forget to delete the return pointer
  return the_page == NULL ? NULL : create_metadata(the_page);
}

bool mmu::allocate_PA(new_addr_type va_base, new_addr_type pa_base, appid_t appID) {
  page * target_page = m_DRAM->allocate_PA(va_base, pa_base, appID);

  if (target_page != NULL) {
    //Add this page to VA to page mapping so that it can be easily search
    unsigned searchID = va_base; // | appID;
    ALLOC_DEBUG << "Physical page for VA = " << va_base << ", app = " << appID <<
      " not in DRAM. Allocating a free page for this VA at PA = " << target_page->starting_addr <<
      ", VA base is " << target_page->va_page_addr <<
      " (Occupy page for this app should contain this page), cycle = " << gpu_sim_cycle +
      gpu_tot_sim_cycle << ". use_value = " << target_page->used << std::endl;
    (*va_to_page_mapping)[searchID] = target_page;
    return true;
  }
  ALLOC_DEBUG << "Physical page for VA = " << va_base << ", app = " << appID << "either is not in DRAM (Impossible) or clash with page table space." << std::endl;
  return false;
}

// This set flags in physical page pa_base as used by appID. Initilize all the page metdata
// Also create the page table entries as needed. Return false if for some reason this pa_base
// conflict with App::pt_space
page * DRAM_layout::allocate_PA(new_addr_type va_base, new_addr_type pa_base, appid_t appID) {
  //Grab this physical page, check if it is occupied by PT
  ALLOC_DEBUG << "Trying to get physical page for VA = " << va_base << " using PA " << pa_base << "  as the key. App creation ID = " << appID << std::endl;
  page * target_page = find_page_from_pa(pa_base); 

  if (target_page == NULL) {
    ALLOC_DEBUG << printf("Cannot find physical page for VA = %llx using PA %llx as the key. App = %d.\n",
        va_base, pa_base, appID);
    return NULL;
  }

  //If so, return false, do nothing
  if (target_page->appID == App::pt_space.appid)
    return NULL;

  //Otherwise, allocate the page, update page metadata, create the PTE entry for this page

  target_page->va_page_addr = va_base;
  target_page->appID = appID;
  target_page->utilization = 0.0;
  target_page->dataPresent = false;
  target_page->used = true;
  target_page->last_accessed_time = gpu_sim_cycle + gpu_tot_sim_cycle;

  //Take this page out from the free page list. Add it to the occupied page list
  //Set this page's parent page as used by the appID. This can also detect the case when multiple apps are in the same huge page range, and will mark appID as App::mixapp

  //Create PTE entry for this page 
  target_page->leaf_pte = m_pt_root->add_entry(va_base, appID, true);

  //Return the page to MMU so it can get the mapping between VA and page
  return target_page;

}

bool DRAM_layout::free_up_page(page * this_page) {
  this_page->va_page_addr = 0;
  this_page->appID = App::noapp.appid;
  this_page->utilization = 0.0;
  this_page->dataPresent = false;
  this_page->used = true;
  this_page->last_accessed_time = 0;
  occupied_pages[this_page->appID]->remove(this_page);
  free_pages[this_page->size]->remove(this_page);
}

bool mmu::update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va,
    new_addr_type new_pa, appid_t appID) {
  return m_DRAM->update_mapping(old_va, old_pa, new_va, new_pa, appID);
}

// Update mapping for this VA page
bool DRAM_layout::update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va,
    new_addr_type new_pa, appid_t appID) {
  page * original = find_page_from_pa(old_pa); // Get the original page, before updating

  if (original->va_page_addr != old_va)
    return false; // Something happen. The page does not have the same VA

  //Allocate this in the new page location
  allocate_PA(new_va, new_pa, appID);
  //Free the old page location
  free_up_page(original);

}

void DRAM_layout::update_parent_metadata(page * triggered_page) {
  page * parent = triggered_page;
  parent->last_accessed_time = triggered_page->last_accessed_time; //update parent's last_accessed bit
  check_utilization(parent); //update parent's utilization
}

// This is called by the address decoder in main gpgpu-sim
new_addr_type mmu::get_pa(new_addr_type addr, appid_t appID, bool * fault, bool isRead) {

  page * this_page = NULL;

  new_addr_type searchID = ((addr >> m_config->page_size) << m_config->page_size); // | appID;
  assert(false);
  // this was originally ored with appID...

  //Check if this page has their own PT entries setup or not. (VA page seen before, VA not seen)
  if (va_to_page_mapping->find(searchID) != va_to_page_mapping->end()) {
    ALLOC_DEBUG << "Searching the map (searchID = " <<  searchID <<
      ") for physical page for VA = " << addr << ", app = " << appID <<
      ". Not the first time access." << std::endl;
    this_page = (*va_to_page_mapping)[searchID];
  } else {
    new_addr_type result = (new_addr_type) (gpu_alloc->translate(appID,
          (void*) ((App::get_app(appID)->addr_offset << 48) | addr)));
    //Note that at this point, addr should exist because addrdec.cc should have already handle any additional allocations
    ALLOC_DEBUG << "Searching the map (searchID = " <<  searchID <<
      "). Cannot find the physical page for VA = " << addr << ", app = " << appID <<
      ". First time access. Allcating this page in the mmu to keep track of metadata" <<
      std::endl;
    //Allocate this page
    allocate_PA((addr >> m_config->page_size) << m_config->page_size,
        (result >> m_config->page_size) << m_config->page_size, appID);
  }

  this_page = (*va_to_page_mapping)[searchID];
  if (this_page == NULL) {
    printf("This should never happen! Page is already allocated but not found\n");
    assert(false);
  }

  // Updates the base page's metadata
  this_page->last_accessed_time = gpu_sim_cycle + gpu_tot_sim_cycle; //Update last accessed time
  this_page->dataPresent = true;
  this_page->utilization = 1.0; //Might want to use a bit vector in the future to represent each cache line in the small page range. Might be an overkiltar

  // Update parent's metadata
  m_DRAM->update_parent_metadata(this_page);

  new_addr_type result = (new_addr_type) (gpu_alloc->translate(appID,
        (void*) ((App::get_app(appID)->addr_offset << 48) | addr)));
  MERGE_DEBUG << "Requesting the PA for VA = " <<  addr << ", appID = " << appID << ". Got " <<
    result << ", at address " << (void*) &result << std::endl;

  return result; //Get the correct physical address
}

page * mmu::find_physical_page(mem_fetch * mf) {
  return m_DRAM->get_page_from_va(mf->get_addr(), mf->get_appID(), m_config->base_page_size);
}

void mmu::check_utilization(page * this_page) {
  m_DRAM->check_utilization(this_page);
}

float mmu::check_utilization() {
  return m_DRAM->check_utilization();
}

page * mmu::find_physical_page(new_addr_type va, appid_t appID, unsigned size) {
  return m_DRAM->get_page_from_va(va, appID, size);
}

int mmu::demote_page(page * this_page) {
  return m_DRAM->demote_page(this_page);
}

// Return true if succeeded
int mmu::coalesce_page(page * this_page) {
  return m_DRAM->coalesce_page(this_page);
}

void mmu::set_L2_tlb(tlb_tag_array * L2TLB) {
  printf("Setting L2 TLB for the MMU at address = %x\n", (void*) L2TLB);
  l2_tlb = L2TLB;
  //Sending all the promote/demote requests
  while (!promoted_list.empty()) {
    PROMOTE_DEBUG << "Send a pending promotion call for VA = " <<
      (void*) promoted_list.front().first << ", appID = " << promoted_list.front().second << std::endl;
    if(m_config->enable_page_coalescing) l2_tlb->promotion(promoted_list.front().first, promoted_list.front().second);
    promoted_list.pop_front();
  }
  while (!demoted_list.empty()) {
    PROMOTE_DEBUG << "Send a pending demotion call for VA = " <<  demoted_list.front().first << ", appID = " << demoted_list.front().second << std::endl;
    if(m_config->enable_page_coalescing) l2_tlb->demote_page(demoted_list.front().first, demoted_list.front().second);
    demoted_list.pop_front();
  }
}

int mmu::promote_page(new_addr_type va, appid_t appID) {
  if (need_init || l2_tlb == NULL) {
    PROMOTE_DEBUG << "MMU got a promotion call during INIT for VA = " <<  va << ", appID = " << appID << std::endl;
    promoted_list.push_back(std::pair<new_addr_type, appid_t>(va, appID));
  }
  else {
    // Mark the page metadata by calling coalesce_page(page)
    // Same sa demote page, tlb will handle this call so that it can check MMU's promotion/demotion return status

    // Then update the promoted page list in tlb.cc
    PROMOTE_DEBUG << "MMU got a promotion call for VA = " <<  va << ", appID = " << appID << std::endl;
    if(m_config->enable_page_coalescing) return l2_tlb->promotion(va, appID);
    else return 0;
  }
}

int mmu::demote_page(new_addr_type va, appid_t appID) {
  if (need_init || l2_tlb == NULL) {
    PROMOTE_DEBUG << "MMU got a demotion call during INIT for VA = " <<  va << ", appID = " << appID << std::endl;
    demoted_list.push_back(std::pair<new_addr_type, appid_t>(va, appID));
  }
  else
  {
    // Mark the page metadata by call demote_page

    // Then update the promoted page list in tlb.cc
    PROMOTE_DEBUG << "MMU got a demotion call for VA = " <<  va << ", appID = " << appID << std::endl;
    if(m_config->enable_page_coalescing) return l2_tlb->demote_page(va, appID);
    else return 0;
  }
}

unsigned mmu::get_bitmask(int level) {
  return m_pt_root->get_bitmask(level);
}

void mmu::set_stat(memory_stats_t * stat) {
  m_stats = stat;
  printf("Setting stat object in MMU\n");
  m_DRAM->set_stat(stat);
}

void mmu::set_DRAM_channel(dram_t * dram_channel, int channel_id) {
  m_DRAM->set_DRAM_channel(dram_channel, channel_id);
}

