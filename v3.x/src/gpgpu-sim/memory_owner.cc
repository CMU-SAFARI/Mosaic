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


#define ALLOC_DEBUG_SHORT 0
#define ALLOC_DEBUG 0
#define MERGE_DEBUG 0
#define MERGE_DEBUG_SHORT 0
#define PROMOTE_DEBUG 1
#define ALLOC_DEBUG_LONG 0
//Enable this will print fault queue entries
#define GET_FAULT_STATUS 1

#define COMPACTION_DEBUG 0

#define RC_DEBUG 1

//Print to check that page->utilization is correct
#define UTIL_CHECK 0

#define COALESCE_DEBUG 0
#define COALESCE_DEBUG_SMALL 0
#define PT_DEBUG 0

extern Hub * gpu_alloc;


//static long int powli( long int x, long int y );
//static unsigned int LOGB2_32( unsigned int v );
//static new_addr_type addrdec_packbits( new_addr_type mask, new_addr_type val, unsigned char high, unsigned char low);
//static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high, unsigned char *low); 


page_metadata::page_metadata(unsigned long long parent_last_accessed_time, unsigned long long child_last_accessed_time, int appID, float util, float actual_util)
{
    parent_last_accessed = parent_last_accessed_time;
    child_last_accessed = child_last_accessed_time;
    accessed_app = appID;
    utilization = util;
    actual_utilization = actual_util;
}



float mmu::get_actual_util(page * large_page)
{
    int current=0;
    int max=0;
    std::list<page*> * pages = large_page->sub_pages;
    for(std::list<page*>::iterator itr = pages->begin(); itr !=pages->end(); itr++)
    {
        if((*itr)->utilization > 0) 
        {
            max++;
            if((*itr)->dataPresent) current++;
        }
    }
    if(max == 0) return 0.0;
    else return (float)current/(float)max;
}


// Update page metadata based on the information of this_page. Called with this_page is accessed
page_metadata * mmu::create_metadata(page * this_page)
{
    page * temp;
    float actual_util;
    //Should always fall here
    if(this_page->size == m_config->base_page_size)
    {
        temp = this_page->parent_page;
        check_utilization(temp); //Update the parent page's utilization
        actual_util = get_actual_util(temp);
    }
    else
        actual_util = temp->utilization;
    return new page_metadata(temp->last_accessed_time,this_page->last_accessed_time,temp->appID,temp->utilization,actual_util);
}

page_table_entry::page_table_entry(new_addr_type key, new_addr_type address, page_table * parent)
//page_table_entry::page_table_entry(new_addr_type key, new_addr_type address, mmu * main_mmu, new_addr_type VA)
{
    addr = address;
    isLeaf = false;
    appID = NOAPP;
    isRead = true;
    inDRAM = false;
    parent_pt = parent; //So that leaf nodes (hence, page object) can access every entry/level of the page table if needed
//    metadata = new page_metadata(this);

//    VA_addr = (VA/m_config->base_page_size)*m_config->base_page_size;
    //VA should be the masked (small page range) VA
//    main_mmu->add_metadata_entry(VA_addr, metadata); 
}


page_table::page_table(const memory_config * config, unsigned parent_level, mmu * main_mmu)
{
    //entries = new std::map<unsigned,new_addr_type>();
    printf("Initializing page_table of level %d\n", parent_level);
    m_config = config;
    m_mmu = main_mmu;
    m_size = 0;
    // m_appID = appID;
    if(parent_level < m_config->tlb_levels - 1)
    {
        printf("Setting pointers current level %d to the next page_table of level %d\n", parent_level,parent_level + 1);
        next_level = new page_table(m_config, parent_level + 1, main_mmu);
    }
    current_level = parent_level + 1;
    parse_bitmask();
}


void page_table::set_leaf_list(std::list<new_addr_type> * the_list)
{
    all_page_table_leaf = the_list;
    if(next_level!=NULL)
    next_level->set_leaf_list(the_list);
}


////new, use page_sizes
//void page_table::parse_bitmask()
//{
//    m_bitmask = sizeof(m_bitmask) - ((*(m_config->page_sizes))[current_level]-1);
//    printf("Converting VA bitmask for page_table translation for level = %d, page_size = %d, results = %x\n", current_level, (*(m_config->page_sizes))[current_level], m_bitmask);
//}

// Parse bitmask for page table walk. Can use the page_sizes instead (above), but this seems more flexible. Downside is the two config (va_mask), tlb_levels and page_sizes has to match.
void page_table::parse_bitmask()
{
    std::string mask(m_config->va_mask);
    std::string mask2(m_config->va_mask);
    //for(int i = m_config->tlb_levels; i > 0;i--)
    for(unsigned i = 1; i <= m_config->tlb_levels;i++)
    {
        if(i<=current_level-1)
        //if(i>current_level)
        //if(i!=current_level)
        {
//            printf("converting mask string for char %c, to char %c\n", (char)(i+'0'),'0');
            std::replace(mask.begin(),mask.end(),(char)(i+'0'),'0');
        }
        else
        {
//            printf("converting mask string for char %c, to char %c\n", (char)(i+'0'),'1');
            std::replace(mask.begin(),mask.end(),(char)(i+'0'),'1');
        }
        if(i==current_level)
            std::replace(mask2.begin(),mask2.end(),(char)(i+'0'),'1');
        else
            std::replace(mask2.begin(),mask2.end(),(char)(i+'0'),'0');
    }
    std::bitset<32> temp(mask);
    std::bitset<32> temp2(mask2);
    m_bitmask = temp.to_ulong();
    m_bitmask_pw = temp2.to_ulong();
    printf("Converting VA bitmask for page_table translation for level = %d, original string = %s, results = %x, mask_string = %s, pwcache_offset_mask = %x, mask_string = %s\n", current_level, m_config->va_mask, m_bitmask, mask.c_str(), m_bitmask_pw, mask2.c_str());
}

unsigned page_table::get_bitmask(int level)
{
//    if(next_level == NULL)
//        printf("Getting bitmask, this is the last level, current level = %d, bitmask = %x, mftlb level = %d\n", current_level, m_bitmask_pw, level);
//    else
//        printf("Getting bitmask, this is NOT the last level, current level = %d, bitmask = %x, mftlb level = %d\n", current_level, m_bitmask_pw, level);
    
    if(level==current_level)
    {
        return m_bitmask;
        //return m_bitmask_pw;
    }
    else if(next_level == NULL)
    {
        return m_bitmask_pw;
        //return m_bitmask;
    }
    else
    {
        return next_level->get_bitmask(level);
    }
}

void page_table::update_leaf_node(page * this_page, bool value)
{
    std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(this_page->starting_addr);
    itr->second->isLeaf = value;
    itr->second->appID = this_page->appID;
}

//Called as the subroutine when page is swapped. 
void page_table::update_swapped_pages(page * swap_in, page * swap_out)
{
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
    int temp2 = entry_in->appID;
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


void page_table::update_leaf_node(page_table_entry * pte, bool value, int appID)
{
    pte->isLeaf = value;
    pte->appID = appID;
}

void page_table::set_page_in_DRAM(page * this_page)
{
    //Only mark it as inDRAM when used for non-TLB-related pages
    if(this_page->appID !=PT_SPACE){
        std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(this_page->starting_addr & m_bitmask); 
        if(itr != entries.end())
        {
            itr->second->inDRAM = true;
            itr->second->appID = this_page->appID;
        }
        if(next_level!=NULL) next_level->set_page_in_DRAM(this_page);
    }
}


//This is called from memory.cc, add a mapping between virtual address to physical address
//i.e, this populate the entries (map of addr and actual entries) of each level of page table
//Each entry contain whether the page is in DRAM or not, whether this is a leaf (in case of
//a superpage, leaf node would be at an earlier level, valid flag (nodes below leaf nodes should
//not be valid.
// Once this mapping is set, parse_pa, should be able to simply return the page_table_entry
// back to mem_fetch to process its request.
page_table_entry * page_table::add_entry(new_addr_type address, int appID, bool isRead)
{
    new_addr_type key = address & m_bitmask;
    page_table_entry * temp;
    std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(key);
    if(PT_DEBUG) printf("Adding page table entry for address = %x, current level key = %x, bitmask = %x\n",address, key, m_bitmask);
    if(itr == entries.end()) //Entry is not in the page table, page fault
    {
        if(PT_DEBUG) printf("Allocating space for page of size %d for page table entry, current entry size = %d\n",m_config->base_page_size, m_size); // Remove this if we want a cleaner stdout
        if(m_size % (m_config->base_page_size/64) == 0)
        {
            page * new_page = m_mmu->get_DRAM_layout()->allocate_free_page(m_config->base_page_size,PT_SPACE);
            if(new_page == NULL)
            {
                if(PT_DEBUG) printf("Not enough space to allocate for page of size %d for page table entry\n",m_config->base_page_size); // Remove this if we want a cleaner stdout
                current_fillable_address = rand();
            }
            else current_fillable_address = new_page->starting_addr;
            if(PT_DEBUG) printf("Acquiring a new page for this entry. new page base addr = %x, Current fillable address = %x\n",new_page->starting_addr,current_fillable_address);
            temp = new page_table_entry(key,current_fillable_address, this);
            //temp = new page_table_entry(key,current_fillable_address,m_mmu,address);
            if(next_level==NULL) all_page_table_leaf->push_front(current_fillable_address); //If this is the leaf, add it to the address list for the scanner to probe
        }
        else
        {
            if(PT_DEBUG) printf("Appending to an existing page for this entry. Current fillable address = %x, added index = %x, actual entry address = %x\n",current_fillable_address, m_size%(m_config->base_page_size/64),current_fillable_address + (m_size%(m_config->base_page_size/64)));
            temp = new page_table_entry(key,current_fillable_address + (m_size%(m_config->base_page_size/64)), this);
            //temp = new page_table_entry(key,current_fillable_address + (m_size%(m_config->base_page_size/64)),m_mmu, address);
            if(next_level==NULL) all_page_table_leaf->push_front(current_fillable_address + (m_size%(m_config->base_page_size/64))); //If this is the leaf, add it to the address list for the scanner to probe
        }
        m_size++;
        temp->isLeaf = next_level==NULL?true:false; //Only last level is a leaf
        temp->appID = appID;
        temp->isRead = true;
        temp->inDRAM = false; //Not in DRAM. All checkings happen when get_pa is called
        if(PT_DEBUG) printf("Creating new page table entry for address = %x, current level key = %x, entry address = %x, bitmask = %x\n",address, key, current_fillable_address,m_bitmask);
        entries.insert(std::pair<unsigned,page_table_entry*>(key,temp));
    }
    else
    {
        if(PT_DEBUG) printf("Found the page table entry for address = %x, current level key = %x, key = %x, address = %x\n",address, key, itr->second->key, itr->second->addr);
        temp = itr->second;
    }
    if(next_level !=NULL) //Propagate entires across multiple levels
        return next_level->add_entry(address,appID,isRead);
    else
        update_leaf_node(temp,true,appID); //If this is the last level, mark PTE as the leaf node
    return temp;
}

// Find the address for tlb-related data by going through the page table entries of each level
//page_table_entry * page_table::parse_pa(mem_fetch * mf)
new_addr_type page_table::parse_pa(mem_fetch * mf)
{
    unsigned key = mf->get_original_addr() & m_bitmask;
    if(PT_DEBUG) printf("Parsing PA for address = %x, current level = %d, key = %x, mf->addr = %x\n",mf->get_original_addr(),mf->get_tlb_depth_count(),key,mf->get_addr());
    std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(key);
    if(itr == entries.end()) //This should never happen. add_entry should have cover this part
    {
//        printf("Entry not found!");
        //cuda-sim/memory.cc should have already add these entries
        if(PT_DEBUG) printf("Entry not found: Adding new page table entry for mf = %x, level = %d\n", mf->get_addr(), mf->get_tlb_depth_count());
        add_entry(mf->get_addr(), mf->get_appID(), !mf->is_write());
        return parse_pa(mf);
    }
    else //Found the entry, should fall into this category most of the time
    {
//        printf("Entry found!, isLeaf = %d, inDRAM = %d, current_level_address = %x, entry key = %x, current level = %d, mf level = %d\n", itr->second->isLeaf, itr->second->inDRAM, itr->second->addr, itr->second->key, current_level, mf->get_tlb_depth_count());
//        printf("Searching for TLBreqAddress for mf level = %d, current level = %d\n",mf->get_tlb_depth_count(),current_level);
//        if((mf->get_tlb_depth_count()-1) == current_level)
//            return itr->second;
//        else //Not a matching level
//            return next_level->parse_pa(mf);

        if((mf->get_tlb_depth_count()+1) == current_level)
        {
//            printf("Returning!, isLeaf = %d, inDRAM = %d, current_level_address = %x, entry key = %x, current level = %d, mf level = %d\n", itr->second->isLeaf, itr->second->inDRAM, itr->second->addr, itr->second->key, current_level, mf->get_tlb_depth_count());
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

page::page()
{
    dataPresent = false;
    size = 0;
    starting_addr = 0;
    used = false;
    appID = NOAPP;
    sub_pages = NULL;
    //sub_pages = new std::list<page*>(); This gets initialized later during DRAM layout initialization
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
    if(ALLOC_DEBUG)  printf("Initialing DRAM physical structure\n");
    m_config = config;
    DRAM_size = (unsigned)m_config->DRAM_size;


    m_page_size = m_config->page_sizes;

    m_pt_root = root;

    m_page_root = new page();
    m_page_root->starting_addr = 0;
    m_page_root->used = false;
    m_page_root->size = m_config->DRAM_size; 
    m_page_root->dataPresent = false;
    m_page_root->appID = NOAPP;
    m_page_root->sub_pages = new std::list<page*>();
    m_page_root->pt_level = m_pt_root;
    m_page_root->parent_page = NULL;
    m_page_root->leaf_pte = NULL; //This is just the root node, should not have PTE associated with this node

    // define channel/bank/sa mapping
    addrdec_t from_raw_addr;

    m_config->m_address_mapping.addrdec_tlx(0,&from_raw_addr, NOAPP, DRAM_CMD, 0); //m_page_root
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



    for(int i=0;i<3/*FIXME: m_config->num_apps*/;i++)
    {
        occupied_pages[i] = new std::list<page*>();
        active_huge_block[i] = free_pages[(*m_page_size)[m_page_size->size()-2]]->front();
        free_pages[(*m_page_size)[m_page_size->size()-2]]->pop_front();
        //Grant the first n huge blocks to each app
    }
    active_huge_block[PT_SPACE] = free_pages[(*m_page_size)[m_page_size->size()-2]]->front();
    occupied_pages[PT_SPACE] = new std::list<page*>();
    // List of bloated pages
    occupied_pages[NOAPP] = new std::list<page*>();
    occupied_pages[MIXAPP] = new std::list<page*>();

    if(ALLOC_DEBUG)  printf("Done initialing DRAM physical structure\n");


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
        if(UTIL_CHECK) printf("Updating utilization page PA = %x, VA = %x, app = %d, used = %d, size = %d, utilization value = %f (%f was added from the previous page)\n",this_page->starting_addr, this_page->va_page_addr, this_page->appID, this_page->used, this_page->size, this_page->utilization, (*itr2)->utilization);
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
    for(std::list<page*>::iterator itr = the_list->begin(); itr != the_list->end(); itr++)
    {
        if(UTIL_CHECK) printf("Checking utilization page PA = %x, VA = %x, app = %d, used = %d, size = %d, utilization value = %f\n",(*itr)->starting_addr, (*itr)->va_page_addr, (*itr)->appID, (*itr)->used, (*itr)->size, (*itr)->utilization);
        total_util = total_util + (*itr)->utilization;
        if((*itr)->utilization > 0) num_pages++;
    }
    if(UTIL_CHECK) printf("Current page size (size = %u) utilization value = %f\n",(*m_page_size)[m_page_size->size()-1], total_util);

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
            if(UTIL_CHECK) printf("Updating utilization page PA = %x, VA = %x, app = %d, used = %d, size = %d, utilization value = %f (%f was added from the previous page)\n",(*itr)->starting_addr, (*itr)->va_page_addr, (*itr)->appID, (*itr)->used, (*itr)->size, (*itr)->utilization, (*itr2)->utilization);
        }
        if(UTIL_CHECK) printf("Done updating utilization page PA = %x, VA = %x, app = %d, used = %d, size = %d, utilization value = %f\n",(*itr)->starting_addr, (*itr)->va_page_addr, (*itr)->appID, (*itr)->used, (*itr)->size, (*itr)->utilization);
        (*itr)->utilization = (*itr)->utilization / num_pages2;
    }
    return total_util/(float)num_pages;

//    float total_util = all_page_list[DRAM_size]->front()->utilization;
//    float check_total_util;
//    for(int i=0;i<m_page_size->size();i++)
//    {
//        check_total_util = 0.0;
//        std::list<page*> * the_list = all_page_list[(*m_page_size)[i]];
//        for(std::list<page*>::iterator itr = the_list->begin(); itr != the_list->end(); itr++)
//        {
//            check_total_util = check_total_util + (*itr)->utilization;
//        }
//        if(UTIL_CHECK) printf("Current page size (size = %u) utilization value = %f, checked value = %f\n",(*m_page_size)[i],checked_total_util, total_util);
//        if(check_total_util != total_util)
//        {
//            printf("Total utilzation is not consistent!. Diff = %f", total_util - check_total_util);
//        }
//    }
//    return checked_total_util;
}

void DRAM_layout::initialize_pages(page * this_page, unsigned size_index, page_table * pt)
{
    // Propagate this sub_page
    page * temp;
    for(unsigned i=0;i<((*m_page_size)[size_index-1]/(*m_page_size)[size_index]);i++) //Initialize all sub-pages under this page
    {
        temp = new page(); 
        // Physical data -- Can't be changed
        temp->starting_addr = this_page->starting_addr + (i*((*m_page_size)[size_index]));
        // Metadata, can be changed
        temp->used = false;
        temp->size = (*m_page_size)[size_index];
        temp->dataPresent = false;
        temp->appID = NOAPP;
        temp->parent_page = this_page;
        temp->sub_pages = new std::list<page*>();
        temp->pt_level = pt->next_level;
        temp->leaf_pte = NULL;

        // define channel/bank/sa mapping -- Also physical data -- Can't be changed
        addrdec_t from_raw_addr;
        m_config->m_address_mapping.addrdec_tlx(temp->starting_addr,&from_raw_addr, NOAPP, DRAM_CMD, 0); //m_page_root
        temp->channel_id = from_raw_addr.chip;
        temp->bank_id = from_raw_addr.bk;
        temp->sa_id = from_raw_addr.subarray;

        //Add this page to the free page list
        if(ALLOC_DEBUG)  printf("Initialing free page list of page of size %d, starting address = %x, parent_page = %x\n",(*m_page_size)[size_index],temp->starting_addr, this_page->starting_addr);
        free_pages[(*m_page_size)[size_index]]->push_front(temp);
        all_page_list[(*m_page_size)[size_index]]->push_front(temp);
        if(temp->size == (*m_page_size)[m_page_size->size()-1]) //If this is the smallest page, add this page to a map of all small pages
        {
            if(MERGE_DEBUG || ALLOC_DEBUG)  printf("Adding a leaf page of size %d, starting address = %x, parent_page = %x to the all page map\n",(*m_page_size)[size_index],temp->starting_addr, this_page->starting_addr);
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


//void DRAM_layout::initialize_pages(page * this_page, unsigned size_index, page_table * pt)
//{
//    if(size_index < page_size.size())
//    {
//        // Propagate this sub_page
//        page * temp;
//        for(unsigned i=0;i<(m_config->DRAM_size/page_size[size_index]);i++)
//        {
//            temp = new page(); 
//            temp->starting_addr = this_page->starting_addr + (i*page_size[size_index]);
//            temp->used = false;
//            temp->size = page_size[size_index];
//            temp->dataPresent = false;
//            temp->appID = NOAPP;
//            temp->parent_page = this_page;
//            temp->sub_pages = new std::list<page*>();
//            temp->pt_level = pt->next_level;
//            //Add this page to the free page list
////            printf("Initialing free page list of page of size %d, starting address = %x\n",page_size[size_index],temp->starting_addr);
//            free_pages[page_size[size_index]]->push_front(temp);
//            this_page->sub_pages->push_front(temp);
//            this_page->utilization = 0.0;
//            initialize_pages(temp,size_index +1,pt->next_level);
//        }
//    }
//    else 
//    {
//        this_page->used = true;
//        this_page->utilization = 0.0;
//        return;
//    }
//}

page * DRAM_layout::find_page_from_pa(new_addr_type pa)
{
    new_addr_type pa_base = (pa / (*m_page_size)[m_page_size->size()-1]) * (*m_page_size)[m_page_size->size()-1];
    page * res = all_page_map[pa_base];
    if(ALLOC_DEBUG || COALESCE_DEBUG){
        if(res == NULL) printf("Searching for a page using PA: Cannot find the page for PA = %x, searched key = %x\n", pa, pa_base);    
        else printf("Searching for a page using PA: Found the page for PA = %x, searched key = %x, VA = %x, appID = %d, size = %d\n", pa, pa_base, res->va_page_addr, res->appID, res->size);    
    }

    return res;
}

// Demotion
int DRAM_layout::demote_page(page * this_page)
{
    if(UTIL_CHECK) check_utilization();

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

    if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("Demoting a superpage PA = %x, base_va = %x, size = %u\n", this_page->starting_addr, this_page->va_page_addr, this_page->size);
    for(std::list<page*>::iterator itr = this_page->sub_pages->begin();itr != this_page->sub_pages->end();itr++)
    {
//        (*itr)->appID = this_page->appID;
        if(COALESCE_DEBUG) printf("Breaking subpage PA = %x, base_va = %x, size = %u, appID = %d\n", (*itr)->starting_addr, (*itr)->va_page_addr, (*itr)->size, (*itr)->appID);
        (*itr)->used = true; //Mark subpage as a leaf node
        (*itr)->dataPresent = true; 
        if((*itr)->appID == NOAPP) //Add to free page list if this is originally a free page
            add_free_list.push_front(*itr);
        else
            occupied_list.push_front(*itr);
    }
    //Set the current node parameters
    this_page->dataPresent = false;
    this_page->used = false;
    //this_page->appID = NOAPP; //Note that the parent page still should have the same appID as its subpages, only that it is not used

    //Push all free pages to a free page list
    while(!add_free_list.empty())
    {
        page * temp = add_free_list.front();
        free_pages[temp->size]->push_back(temp);
        //Remove this page from the occupy_pages[NOAPP] list;
        occupied_pages[NOAPP]->remove(temp);
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
//            if(ALLOC_DEBUG_SHORT) printf("For parent page at PA = %x, size = %d: Marking page at %x VA = %x use val = %d to false\n",this_page->starting_addr,this_page->size,(*itr)->starting_addr,(*itr)->va_page_addr, (*itr)->used);
            (*itr)->dataPresent = false;
            (*itr)->used = false;
            remove_free_page(*itr);
            propagate_sub_page_as_used(*itr); //Depth first
        }
    }
    else return;
}

//void DRAM_layout::remove_free_page(std::list<page*> * remove_list)
//{
//    while(!remove_list->empty())
//    {
//        remove_free_page(remove_list->front());
//        remove_list->pop_front();
//    }
//}

void DRAM_layout::remove_free_page(page * this_page)
{

    if(COALESCE_DEBUG) printf("Removing a page at %x, va_base = %x, app = %d of size = %d from the free page list\n",this_page->starting_addr,this_page->va_page_addr,this_page->appID,this_page->size);

    std::list<page*> * toSearch = free_pages[this_page->size];


    for(std::list<page*>::iterator itr = toSearch->begin(); itr!= toSearch->end(); itr++)
    {
        //Match
        if((*itr)->starting_addr == this_page->starting_addr)
        {
            toSearch->erase(itr);
            occupied_pages[this_page->appID]->push_back(*itr);
            // Update bloat status
            if(this_page->appID == NOAPP && (occupied_pages[NOAPP]->size() > m_stats->max_bloat))
                m_stats->max_bloat = occupied_pages[NOAPP]->size();
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

// Make sure all parent pages are removed from the free page list when this_page is allocated, and mark this parent page to belong to the app
void DRAM_layout::propagate_parent_page_as_used(page * this_page)
{
    if(this_page->parent_page == NULL) return;
    else
    {
        free_pages[this_page->parent_page->size]->remove(this_page->parent_page); 
        // No app uses this huge range, remove this large page from the free list
        if(this_page->parent_page->appID == NOAPP)
        {
            free_pages[this_page->parent_page->size]->remove(this_page->parent_page);
            occupied_pages[this_page->appID]->push_back(this_page->parent_page);
        }
        //If this page consist of mixed apps
        else if(this_page->parent_page->appID != this_page->appID)
        {
            occupied_pages[this_page->parent_page->appID]->remove(this_page->parent_page);
            occupied_pages[MIXAPP]->push_back(this_page->parent_page);
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
        if((*itr)->appID == NOAPP)
        {
            free = *itr;
            break;
        }
        //TODO: Check if this page is in the free page list, make sure the list is consistent
    }
    return free;
}

//Move original page to free page. ASSUMPTION: free page exist
//               This is used during coalescing, when we are swapping pages around
//               so I guess both original_page and free_page are also under the
//               same huge block. Ideally, we might want to just avoid calling this function.
void DRAM_layout::allocate_free_page(page * original_page, page * free_page)
{
    free_page->va_page_addr = original_page->va_page_addr;
    free_page->appID = original_page->appID;
    free_page->used = true;
    free_page->dataPresent = true;
    // Remove free_page from the free page list. Add it to the occupied page list
    for(std::list<page*>::iterator itr = free_pages[free_page->size]->begin();itr != free_pages[free_page->size]->end();itr++)
    {
        if((*itr)->starting_addr == free_page->starting_addr) //Found
        {
            free_pages[free_page->size]->erase(itr);
            break;
        }
    }
    //Remove the parent page from the free page list too
    propagate_parent_page_as_used(free_page);

    occupied_pages[original_page->appID]->push_back(free_page);
    propagate_sub_page_as_used(free_page);
    m_pt_root->set_page_in_DRAM(free_page);
}



//Return a free page if a certain size
//This part now cause wierd syscall error
page * DRAM_layout::allocate_free_page(unsigned size, int appID)
{
    page * return_page = NULL;
    if(ALLOC_DEBUG || COALESCE_DEBUG_SMALL) printf("Allocating a page of size = %d, for appID = %d, free page exist, free page size = %d\n",size,appID,free_pages[size]->size());
    if((free_pages[size]->size()) > 0){
        if(ALLOC_DEBUG) printf("Trying to grab the front of free page list, size = %d, front entry is at %x, back is at %x\n", free_pages[size]->size(), free_pages[size]->front()->starting_addr, free_pages[size]->back()->starting_addr);
        //Grab the free page of a certain size
        if(appID == PT_SPACE) //To reduce coalescing conflict
        {
            m_stats->pt_space_size = m_stats->pt_space_size + m_config->base_page_size;
        }
        //For a normal request, first get the current parent page that handle this appID current huge block range
        page * parent = active_huge_block[appID];
        //Then grab a free page within this huge block
        return_page = get_free_base_page(parent);
        if(return_page == NULL) // If there is no more free page in this huge range
        {
            active_huge_block[appID] = free_pages[parent->size]->front(); //Update current free huge block for this app. Note that get_pa will update this page as used in the line below (at propagate_parent_page_as_used(return_page)
            parent = active_huge_block[appID];
            return_page = get_free_base_page(parent); //Get the free page
        }
        //Remove this page from the free page list
        free_pages[size]->pop_front();
        if(ALLOC_DEBUG || PT_DEBUG) printf("Returning a page of size = %d, page starting address = %x, free page size is now at = %d, appID = %d, freepage_list_front is %x, back is %x\n",size,return_page->starting_addr,free_pages[size]->size(), appID, free_pages[size]->front()->starting_addr, free_pages[size]->back()->starting_addr);

        //Add this page to app
        return_page->used = true;
        return_page->dataPresent = true;
        return_page->appID = appID;
        return_page->utilization = 1.0;
        occupied_pages[appID]->push_back(return_page);
        //Remove the parent page from the free page list too
        propagate_parent_page_as_used(return_page);


        if(ALLOC_DEBUG) printf("Setting return page as used and data present\n");
        // Add all the subpages of this free page to the non_free_list
        propagate_sub_page_as_used(return_page); 
        // Mark the entry in page table that the page is in DRAM
        m_pt_root->set_page_in_DRAM(return_page); //Config parameters can have discrepancy between va_mask and page_sizes
    }
    return return_page;//Return null if there are no more free page of this size



// Not as old, but still old
//    page * return_page = NULL;
//    if(ALLOC_DEBUG || COALESCE_DEBUG_SMALL) printf("Allocating a page of size = %d, for appID = %d, free page exist, free page size = %d\n",size,appID,free_pages[size]->size());
//    if((free_pages[size]->size()) > 0){
//        if(ALLOC_DEBUG) printf("Trying to grab the front of free page list, size = %d, front entry is at %x, back is at %x\n", free_pages[size]->size(), free_pages[size]->front()->starting_addr, free_pages[size]->back()->starting_addr);
//        //Grab the free page of a certain size
//        if(appID == PT_SPACE) //To reduce coalescing conflict
//        {
//            return_page = free_pages[size]->back();
//            free_pages[size]->pop_back();
//            m_stats->pt_space_size = m_stats->pt_space_size + m_config->base_page_size;
//        }
//        else
//        {
//            //For a normal request, first get the current parent page that handle this appID current huge block range
//            page * parent = active_huge_block[appID];
//            //Then grab a free page within this huge block
//            return_page = get_free_base_page(parent);
//            if(return_page == NULL) // If there is no more free page in this huge range
//            {
//                active_huge_block[appID] = free_pages[parent->size]->front(); //Update current free huge block for this app. Note that get_pa will update this page as used in the line below (at propagate_parent_page_as_used(return_page)
//                parent = active_huge_block[appID];
//                return_page = get_free_base_page(parent); //Get the free page
//            }
//            //Remove this page from the free page list
//            free_pages[size]->pop_front();
//        }
//        if(ALLOC_DEBUG || PT_DEBUG) printf("Returning a page of size = %d, page starting address = %x, free page size is now at = %d, appID = %d, freepage_list_front is %x, back is %x\n",size,return_page->starting_addr,free_pages[size]->size(), appID, free_pages[size]->front()->starting_addr, free_pages[size]->back()->starting_addr);
//
//        //Add this page to app
//        return_page->used = true;
//        return_page->dataPresent = true;
//        return_page->appID = appID;
//        return_page->utilization = 1.0;
//        occupied_pages[appID]->push_back(return_page);
//        //Remove the parent page from the free page list too
//        propagate_parent_page_as_used(return_page);
//
//
//        if(ALLOC_DEBUG) printf("Setting return page as used and data present\n");
//        // Add all the subpages of this free page to the non_free_list
//        propagate_sub_page_as_used(return_page); 
//        // Mark the entry in page table that the page is in DRAM
//        m_pt_root->set_page_in_DRAM(return_page); //Config parameters can have discrepancy between va_mask and page_sizes
//    }
//    return return_page;//Return null if there are no more free page of this size


//Old
//    page * return_page = NULL;
//    if(ALLOC_DEBUG) printf("Allocating a page of size = %d, for appID = %d, free page exist, free page size = %d\n",size,appID,free_pages[size]->size());
//    if((free_pages[size]->size()) > 0){
//        if(ALLOC_DEBUG) printf("Trying to grab the front of free page list, size = %d\n", free_pages[size]->size());
//        //Grab the free page of a certain size
//        if(appID == PT_SPACE) //To reduce coalescing conflict
//        {
//            return_page = free_pages[size]->back();
//            free_pages[size]->pop_back();
//            m_stats->pt_space_size = m_stats->pt_space_size + m_config->base_page_size;
//        }
//        else
//        {
//            return_page = free_pages[size]->front();
//            free_pages[size]->pop_front();
//        }
//        if(ALLOC_DEBUG) printf("Returning a page of size = %d, page starting address = %x, free page size is now at = %d\n",size,return_page->starting_addr,free_pages[size]->size());
//
//        //Add this page to app
//        occupied_pages[appID]->push_back(return_page);
//        return_page->used = true;
//        return_page->dataPresent = true;
//        return_page->appID = appID;
//        //Remove the parent page from the free page list too
//        propagate_parent_page_as_used(return_page);
//
//
//        if(ALLOC_DEBUG) printf("Setting return page as used and data present\n");
//        // Add all the subpages of this free page to the non_free_list
//        propagate_sub_page_as_used(return_page); 
//        // Mark the entry in page table that the page is in DRAM
//        m_pt_root->set_page_in_DRAM(return_page); //Config parameters can have discrepancy between va_mask and page_sizes
//    }
//    return return_page;//Return null if there are no more free page of this size
}

void DRAM_layout::swap_page(page * swap_in, page * swap_out)
{
    //Change page tables to reflect this copy. This is done first while swap_in and swap_out retain their original metadata
    swap_in->pt_level->update_swapped_pages(swap_in, swap_out);

    int temp = swap_in->appID;
    swap_in->appID = swap_out->appID;
    swap_out->appID = temp;    
    new_addr_type temp2 = swap_in->va_page_addr;
    swap_in->va_page_addr = swap_out->va_page_addr;
    swap_out->va_page_addr = temp2;
    float temp4 = swap_in->utilization;
    swap_in->utilization = swap_out->utilization;
    swap_out->utilization = temp4;
    //Note that we do not have to swap parent pages, SA, bank, channel IDs. The physical page structures are the same, only appID and VA page are different


}

void DRAM_layout::release_occupied_page(page * this_page)
{
    if(this_page == NULL)
    {
        printf("Trying to releasce a NULL page!!!\n");
        return;
    }
    if(COALESCE_DEBUG) printf("Releasing occupied page from app %d with VA = %x, size = %u, PA = %x\n",this_page->appID, this_page->va_page_addr, this_page->size, this_page->starting_addr);
    for(std::list<page*>::iterator itr = occupied_pages[this_page->appID]->begin();
        itr != occupied_pages[this_page->appID]->end(); itr++)
    {
        if((*itr) == this_page)
        //if((*itr)->starting_addr == this_page->starting_addr) //No need to actually check the PA, it should be the same pointer
        {
            if(COALESCE_DEBUG) printf("Found the match to release in the occupied page list from app %d with VA = %x, size = %u, PA = %x, PA in occupied list = %x, this_page_address = %x, occupied_page_matched_pointer_addr = %x\n",this_page->appID, this_page->va_page_addr, this_page->size, this_page->starting_addr, (*itr)->starting_addr,this_page, *itr);
//            //Add to the free page list
            free_pages[this_page->size]->push_back(*itr);
//            //Remove this from occupied page
            occupied_pages[this_page->appID]->erase(itr++);
            break; //No need to go further, the element is already found and removed
        }
    }
    // Send a DRAM command to zero out the page
    dram_cmd * zero_cmd = new dram_cmd(ZERO, this_page, NULL, m_config);
    if(COALESCE_DEBUG) printf("Then, send ZERO command to DRAM to free up page at address %x, channel = %d, bank = %d, subarray = %d\n",this_page->starting_addr, zero_cmd->to_channel, zero_cmd->to_bank, zero_cmd->to_sa);
    dram_channel_interface[zero_cmd->from_channel]->insert_dram_command(zero_cmd);
}

page * DRAM_layout::get_page_from_va(new_addr_type va, int appID, unsigned size)
{
    std::list<page*> * page_list = occupied_pages[appID];
// This gets printed too often
    if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Trying to find the physical page for va = %x, appID = %d, size = %d\n", va, appID, size);
    for(std::list<page*>::iterator itr = page_list->begin(); itr != page_list->end();itr++)
    {
        if(ALLOC_DEBUG_LONG)
//        if((((*itr)->va_page_addr)>>24) == (va >> 24))
            printf("During the exhausive search (subset of bits) to find the physical page for va = %x, appID = %d, iterator point at %x, compared value (from VA) is = %x\n", va, appID, (*itr)->va_page_addr,(va & ~((*itr)->size-1)));
        if(COALESCE_DEBUG && (*itr)->size > 4096 && (*itr)->used)
            printf("During the exhausive search (subset of bits) to find the huge physical page for va = %x, appID = %d, iterator point at %x, compared value (from VA) is = %x\n", va, appID, (*itr)->va_page_addr,(va & ~((*itr)->size-1)));
        if((*itr)->va_page_addr == (va & ~((*itr)->size-1)) && ((*itr)->size == size)) //need to check if the page is used too (otherwise it might return a subpage instead of superpage)
        {
            if(COALESCE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT)
                printf("Physical page is found for VA = %x, app = %d, VA page addr = %x, mask = %x, computed mask key = %x, Physical page base address = %x, size = %d\n",va, appID, (*itr)->va_page_addr,~((*itr)->size-1),(va & ~((*itr)->size-1)),(*itr)->starting_addr, (*itr)->size);
            return *itr;
        }
    }
    if(COALESCE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT){
        printf("Trying to find the physical page for va = %x, appID = %d, size = %d, not found --  occupy page list contains:{", va, appID, size);
        for(std::list<page*>::iterator itr = page_list->begin(); itr != page_list->end();itr++)
            printf("VA:0x%x, (PA:0x%x, Comparing %x == %x? use = %d, size = %d)",(*itr)->va_page_addr, (*itr)->starting_addr, (*itr)->va_page_addr,(va & ~((*itr)->size-1)), (*itr)->used, (*itr)->size);
        printf("}\n");
    }

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
        if(highest_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && (source_page->sa_id == (*itr)->sa_id) && ((*itr)->appID == NOAPP))
        {
            highest_prio = *itr;
        }
        //Then prioritize pages with the same bank
        if(mid_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && (source_page->bank_id == (*itr)->bank_id) && ((*itr)->appID == NOAPP))
        {
            mid_prio = *itr;
        }
        //Then any page of the same size
        if(low_prio == NULL && (source_page->channel_id == (*itr)->channel_id) && ((*itr)->appID == NOAPP))
        {
            low_prio = *itr;
        }
        if(low_prio == NULL && ((*itr)->appID == NOAPP))
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
page * DRAM_layout::find_swapped_page(int appID, page * searched_parent_page, page * source_page)
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
    if(COMPACTION_DEBUG) printf("Compaction routine for merging small page into a huge page called at cycle %lld\n",gpu_sim_cycle, gpu_tot_sim_cycle);
    //Find a free page within page 1 range to swap into
    page * target_page = compaction_find_swapped_page(target_huge,source_small);

    //Set the metadata for both pages, setup DRAM commands
    return_cmd = new dram_cmd(COPY,source_small,target_page,m_config);
    target_page->appID = target_huge->appID;
    target_page->used = false;
    target_page->dataPresent = source_small->dataPresent;
    target_page->utilization = source_small->utilization;
    target_page->va_page_addr = source_small->va_page_addr;

    source_small->appID = NOAPP;
    source_small->used = false;
    source_small->dataPresent = false;
    source_small->utilization = 0.0;
    source_small->va_page_addr = 0;
    

    // Remove this free page from occupied_pages[NOAPP]. 
    occupied_pages[NOAPP]->remove(target_page);
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
        //Get the associated channel and bank ID, need to check how sending PT_SPACE would behave in addrdec.cc, should be fine.
        
        addrdec_t raw_addr;
        m_config->m_address_mapping.addrdec_tlx((*itr),&raw_addr, PT_SPACE, 0, 0);
        int channel_from = raw_addr.chip;
        int bank_from = raw_addr.bk;
        int sa_from = raw_addr.subarray;
        dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(SCAN,bank_from,bank_from,channel_from,channel_from,sa_from,sa_from,64,0,m_config));
    }
}

bool DRAM_layout::RC_test()
{
    int channel_from = rand()%m_config->m_n_mem;
    int bank_from = rand()%m_config->nbk;
    int channel_to = rand()%m_config->m_n_mem;
    int bank_to = rand()%m_config->nbk;
    int sa_from = rand()%m_config->num_subarray;
    int sa_to = rand()%m_config->num_subarray;

    if(RC_DEBUG) printf("Inserting copy and zero commands from (ch,bk,sa) (%d, %d, %d) to (ch,bk,sa) (%d, %d, %d) Current cycle = %lld",channel_from,bank_from,sa_from,channel_to,bank_to,sa_to, gpu_sim_cycle + gpu_tot_sim_cycle);

    dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(COPY,bank_from,bank_to,channel_from,channel_to,sa_from,sa_to,64,0,m_config));
    dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(COPY,bank_from,bank_to,channel_from,channel_from,sa_from,sa_to,64,0,m_config)); //Same channel copy
    dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(COPY,bank_from,bank_from,channel_from,channel_from,sa_from,sa_to,64,0,m_config)); //Same channel same bank copy
    dram_channel_interface[channel_from]->insert_dram_command(new dram_cmd(ZERO,bank_from,bank_from,channel_from,channel_from,sa_from,sa_to,64,0,m_config)); //Same channel same bank copy
    return true;
}

//Perform compaction by merging pages from page 2 to pages from page 1, free up page 2
bool DRAM_layout::compaction(page * page1, page* page2)
{
    if(COMPACTION_DEBUG) printf("Compaction routine called at cycle %lld\n",gpu_sim_cycle, gpu_tot_sim_cycle);
    if(page1 == NULL || page2 == NULL){
        printf("Trying to perform compaction on a NULL page\n");
        return false;
    }
    if(page1->sub_pages == NULL){
        printf("Trying to perform compaction into a NULL sub pages\n");
        return false;
    }
    std::list<dram_cmd*> command_list;
    //If page2 is the sub-block size. Note, copied from the routine below
    if(page1->size != page2->size){
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
    else{
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
//                (*itr)->appID = NOAPP;
//                (*itr)->used = false;
//                (*itr)->dataPresent = false;
//                (*itr)->utilization = 0.0;
//                (*itr)->va_page_addr = 0;
//                
//    
//                // Remove this free page from occupied_pages[NOAPP]. 
//                occupied_pages[NOAPP]->remove(target_page);
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
    page2->appID = NOAPP;
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

    if(this_page == NULL)
    {
        printf("Somehow trying to coalesce a null page (O_O ) , current free_page size = %lld\n",free_pages[m_config->base_page_size]->size());
        return 0;
    }
    else if(this_page->size != 4096)
    {
        printf("Somehow trying to coalesce superpage together\n");
        return 0;
    }
    else if(m_page_size->size() == 1)
    {
        printf("Somehow trying to coalesce when there is only one page size\n");
        return 0;
    }
    else
    {
        if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("In coalescing routine, Coalescing the page %x, base VA = %x, size = %u\n",this_page->starting_addr,this_page->va_page_addr,this_page->size);
    }


    m_stats->num_coalesce++;

    //First, check if adjacent pages can be combined
    std::list<dram_cmd*> dram_cmd_list;
    std::list<page*> remove_free_list; 
    //std::list<page*> combined_list;
    dram_cmd_list.empty();
    remove_free_list.empty();
    //combined_list.empty();

    std::list<page*> swap_index; //This is a list containing pages that have to be swapped out for coalescing to happen
    std::list<page*> move_to_free_list; //This is a list containing pages that have to be swapped out if we were to move all pages under the same appID to a new bigger free page


    bool can_merge = true;
    //Go through all the parent's sub_pages_list, see if they are from the same appID
    page * parent = this_page->parent_page;
    //VA base address of the huge page
    new_addr_type va_base = ((this_page->va_page_addr) / parent->size) * parent->size;
    if(parent == NULL)
    {
        printf("For some reason, trying to coalesce a page with no parent page. BAD\n");
        return 0;
    }
    else
    {
        if(COALESCE_DEBUG) printf("Parent of the coalesced page is %x, base VA = %x, size = %u\n",parent->starting_addr,parent->va_page_addr,parent->size);
    }
    float avg_util = 0.0;
    int count = 0;
    if(COALESCE_DEBUG) printf("in MMU, coalescing page with base addr = %x (VA = %x), parent page is %x\n", this_page->starting_addr, this_page->va_page_addr, parent->starting_addr);


    //this out by searching for the parent page in the occupied_pages[APP]. Can
    //coalesce right away when it exists. If multiple pages are there, then
    //this parent page should be in occupied_pages[MIXAPP]

    for(std::list<page*>::iterator itr = parent->sub_pages->begin();itr != parent->sub_pages->end();itr++)
    {
        count++;
        if(!(((*itr)->appID) == this_page->appID)  && (( (*itr)->appID) != NOAPP))
        {
            if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("A page from other app is within the coalesce range, can merge = %d, conflicting appID = %d, page base PA = %x, page base VA = %x\n",can_merge, (*itr)->appID, (*itr)->starting_addr, (*itr)->va_page_addr);
            if(!m_config->enable_costly_coalesce) return 0; //Usually if it reach this point, can return right away because we detect there is a page of other app within the huge block range

            can_merge = false; //Mark this as not coalesable, need to find another free location
            swap_index.push_back((*itr)); //Mark this page as it has to be swapped out

        }
        else if((( (*itr)->appID) == NOAPP)) //Free page within the coalesced range
        {
            if(COALESCE_DEBUG) printf("A free page within the coalesce range, can merge = %d, conflicting appID = %d, page base PA = %x, page base VA = %x\n",can_merge, (*itr)->appID, (*itr)->starting_addr, (*itr)->va_page_addr);
            remove_free_list.push_back(*itr);
        }
        else //Used page in the coalesced range
        { 
            if(COALESCE_DEBUG) printf("An occupied page of the same app in coalesce range, can merge = %d, conflicting appID = %d, page base PA = %x, page base VA = %x\n",can_merge, (*itr)->appID, (*itr)->starting_addr, (*itr)->va_page_addr);
            move_to_free_list.push_back((*itr));
            avg_util += (*itr)->utilization;
            //combined_list.insert(itr);
        }
    }



    if(COALESCE_DEBUG) printf("Checking if no copy required (should be the case for 1 app, can merge = %d\n",can_merge);




    // At this point, done with checking if we can just promote the entire block to a superpage. 
    if(can_merge)
    {
        if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("Can directly coalesce the page %x, base VA = %x, size = %u, free page size = %d\n",this_page->starting_addr,this_page->va_page_addr,this_page->size, free_pages[parent->size]);
        // Mark parent as a new used superpage
        parent->used = true;
        parent->dataPresent = true;
        parent->appID = this_page->appID;
        parent->utilization = avg_util/(float)count;

        parent->va_page_addr = va_base;


        // Update all sub-page table with the coalesced page
        for(std::list<page*>::iterator itr = parent->sub_pages->begin();itr != parent->sub_pages->end();itr++)
        {
            (*itr)->used = false; //Mark the page as not used (because superpage is the active node)
        }

        // Remove all free page under this super page range
        while(!remove_free_list.empty())
        {
            page * remove_page = remove_free_list.front();
            if(COALESCE_DEBUG) printf("Removing sub page pa = %x, va_base = %x, size = %u\n",remove_page->starting_addr,remove_page->va_page_addr,remove_page->size);
            remove_free_page(remove_page);
            remove_free_list.pop_front();
        }
       
        return 1;

        //Update the page table with the corresponding new leaf node (parent).
        parent->pt_level->update_leaf_node(parent,true);
        this_page->pt_level->update_leaf_node(parent,false);

        return 1;
    }

    // Parts below are what we should avoid (i.e., we can't coalesce the page right aware as there are pages of other app within the huge block range)
    // Implementations is there so we can test and compare as needed

    //Need to use some other page, can't coalesce without copy
    else
    {
        if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("Need to coalesce by moving to free page for %x, base VA = %x, size = %u, free page size = %d\n",this_page->starting_addr,this_page->va_page_addr,this_page->size, free_pages[parent->size]);

        //Step2: See if there is a free large page, if so, move all subpages there and free this old page
        if(free_pages[parent->size] > 0) //Check if there is a large page 
            can_merge = true;
        else
            can_merge = false;



        if(COALESCE_DEBUG && can_merge) printf("Cannot just merge, but free huge page is available need to migrate to a new free page, pa_base = %x, va_base = %x, size = %u. Target free page is at %x, size %u\n",this_page->starting_addr,this_page->va_page_addr,this_page->size, free_pages[parent->size]->front(), parent->size);
        if(COALESCE_DEBUG && !can_merge) printf("Cannot just merge, and no free huge page is available need to migrate to a new free page, pa_base = %x, va_base = %x, size = %u\n",this_page->starting_addr,this_page->va_page_addr,this_page->size);

       

        if(can_merge){
            //Move everything from app1 to the large free page
            page * target_free_page = free_pages[parent->size]->front();
            int this_appID = this_page->appID;
            int count = move_to_free_list.size();

            if(COALESCE_DEBUG) printf("Attempting to move set of sub pages with huge page va_base at %x to free_page at PA = %x (VA = %x), appID = %d, free page list size = %d\n", va_base, target_free_page->starting_addr, target_free_page->va_page_addr, this_appID, free_pages[parent->size]->size());



            while(!move_to_free_list.empty()) //This list contain pages with the same appID as this_page
            {
                //Populate the sub pages of target_free_page with pages in the move_to_free_list
                page * swap_out = move_to_free_list.front();
                page * swap_in = find_free_page_for_coalesce(target_free_page,swap_out);
                //Update the free page list/occupied page list and send the zero DRAM command
                if(COALESCE_DEBUG) printf("Moving sub page pa = %x, va_base = %x, size = %u, with free page pa = %x, va = %x, size = %u\n",swap_out->starting_addr,swap_out->va_page_addr,swap_out->size, swap_in->starting_addr, swap_in->va_page_addr, swap_in->size);


                release_occupied_page(swap_out);  
                swap_page(swap_in,swap_out); //This seems fine as long as we do not recursively swap its subpages (no need to do this with two page sizes I think)
                
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
            if(COALESCE_DEBUG) printf("Pushing huge page at PA = %x, VA = %x, app = %d, size = %u to the occupied page list\n",target_free_page->starting_addr, target_free_page->va_page_addr, target_free_page->appID, target_free_page->size);
            occupied_pages[this_appID]->push_back(target_free_page);
        

            return 2; 

        } //End of step 2
        else //Begin step 3: No free large page available, copy pages from other apps out
        {
            if(COALESCE_DEBUG || COALESCE_DEBUG_SMALL) printf("No more huge free page for coalesceing page %x, base VA = %x, size = %u, free page size = %d\n",this_page->starting_addr,this_page->va_page_addr,this_page->size, free_pages[parent->size] > 0);
        return 0; 
            // Do we want to copy from other location to fill in this superpage?
            // If this is the case, setup DRAM commands to send
    
            //These two list keep track of which pages are moved
            std::list<page *> taken_free_pages;
            std::list<page *> moved_to_free_pages;
            std::list<page *> taken_swapped_pages;
            std::list<page *> moved_to_occupied_pages;
            while(!swap_index.empty())
            {
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
            if(swap_index.empty()) //Can be coalesce, commit all the copies and send DRAM commands
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
                    swap_page(swap_in, swap_out); //Swap the page metadata
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

mmu::mmu() 
{
   need_init = true;
   promoted_list = new std::list<std::pair<new_addr_type,int>>();
   demoted_list = new std::list<std::pair<new_addr_type,int>>();


}

void mmu::init(const class memory_config * config)
{
   printf("Initializing MMU object\n");

   //need_init = false;
   m_config = config;

//   m_compaction_last_probed = 0;
//
//   //Initialize the page table object
//   printf("Setting up page tables objects\n");
//   m_pt_root = new page_table(m_config, 0, this);   
//   printf("Done setting up page tables, setting up DRAM physical layout\n");
//
//   va_to_pa_mapping = new std::map<new_addr_type,new_addr_type>();
//   va_to_page_mapping = new std::map<new_addr_type,page*>();
////   va_to_metadata_mapping = new std::map<new_addr_type,page_metadata*>();
//
//   //Initilize page mapping. All pages are now free, all leaf are at the base page size
//   m_DRAM = new DRAM_layout(m_config, m_pt_root);
//
//   printf("Done setting up MMU\n");

}

void mmu::init2(const class memory_config * config)
{
   printf("Initializing MMU object\n");

   //need_init = false;
   m_config = config;


   m_compaction_last_probed = 0;

   //Initialize the page table object
   printf("Setting up page tables objects\n");
   m_pt_root = new page_table(m_config, 0, this);   
   printf("Done setting up page tables, setting up DRAM physical layout\n");

   va_to_pa_mapping = new std::map<new_addr_type,new_addr_type>();
   va_to_page_mapping = new std::map<new_addr_type,page*>();
//   va_to_metadata_mapping = new std::map<new_addr_type,page_metadata*>();

   //Initilize page mapping. All pages are now free, all leaf are at the base page size
   m_DRAM = new DRAM_layout(m_config, m_pt_root);

   printf("Done setting up MMU\nSending pending promotion/demotion requests to TLBs\n");


}

void mmu::set_ready()
{
   printf("Setting the MMU object as ready\n");
   need_init = false;
}

// Done grabbing data for the page, now need to evict a page for appID and virtual page key
page * DRAM_layout::handle_page_fault(int appID, new_addr_type key)
{
    //Evict a page and grab a free page for mf's data
    if(ALLOC_DEBUG) 
        printf("Handling page fault (with data arrived at the GPU, on virtual address = %x, app = %d\n", key, appID);
    return evict_page(appID, get_evict_target(appID), key);
}

// A helper function for evict page. Return a free page, if there is no free page, evict a page from appID
page * DRAM_layout::get_target_page(int appID_target, int appID_requestor)
{
    //LRU as the default policy, can be changed later
    if(ALLOC_DEBUG) printf("Trying to allocate a free page for application %d\n",appID_requestor);
    page * return_page = allocate_free_page(m_config->base_page_size,appID_requestor); //In case there is a free page available, no eviction occur
    if(return_page !=NULL) //If there is a free page
    {
        if(ALLOC_DEBUG) printf("Free page exist, grabbing a free page for application %d, free page PA = %x\n",appID_requestor, return_page->starting_addr);
        return_page->appID = appID_requestor;
        return return_page;
    }
    else
    { 
        if(ALLOC_DEBUG) printf("No free page exist, Releasing a page from application %d\n",appID_target);
        page * released_page = occupied_pages[appID_target]->front(); //FIXME: fix get_evicted_target when we have page replacement policy. Always assume this app has an occupied page
        if(ALLOC_DEBUG) printf("Got a target page (PA = %x, original Virtual page addr = %x) from application %d\n", released_page->starting_addr,released_page->va_page_addr, appID_target);
        release_occupied_page(released_page); //Release the page from the occupied page
        if(ALLOC_DEBUG) printf("Finish moving the occupied page to the free page list, before grabbing this free page from the free page list for %d\n",appID_requestor);
        return_page = allocate_free_page(return_page->size, appID_requestor);
        return_page->appID = appID_requestor;
    }
    return return_page;
}

page * DRAM_layout::evict_page(int demander,int target,new_addr_type va_key)
{
    // This also handles insertion into the occupied list. It takes a page from the target (or from a free page), and give it to the demander
    page * evicted_page = get_target_page(target, demander);
    if(evicted_page==NULL)
        printf("NOOOOOOOOO ... This should not happen. Cannot evict a page for some reason\n");
    else //Set page info
    {
        evicted_page->appID = demander;
        if(ALLOC_DEBUG)
            printf("Adding a new page for app = %d, VA page = %x, page_size = %x\n", demander, va_key, evicted_page->size);
        evicted_page->va_page_addr = va_key * (evicted_page->size);
    }
    return evicted_page; //Allocate this page as it belongs to the appID
}

int DRAM_layout::get_evict_target(int requestor_appID)
{
    //FIXME: Always evict page from app 0, for now
    return 0;
}

bool mmu::is_active_fault(mem_fetch * mf)
{
    if(m_config->page_transfer_time == 0 || m_config->enable_PCIe == 0) return false; //If PCIe modeling is disabled
    else return m_DRAM->is_active_fault(mf->get_addr());
}

//Clear page fault list when 
void DRAM_layout::remove_page_fault()
{
    if(m_config->page_transfer_time == 0 || m_config->enable_PCIe == 0) return;

    // Retire any fault that is done
    if(ALLOC_DEBUG) printf("Checking if there are any page fault resolved, cycle = %lld. Current ongoing fault size = %d\n", gpu_sim_cycle, gpu_tot_sim_cycle, page_fault_list.size());

    if(!page_fault_list.empty() && ((gpu_sim_cycle + gpu_tot_sim_cycle) > (page_fault_last_service_time + m_config->page_transfer_time)))
    {
        page_fault_info * done = page_fault_list.front();
        page * new_page = handle_page_fault(done->appID, done->key); //appID is the page that cause the fault, key is the virtual page of the fault. After calling this, occupied_page[appID] must contains key
        
        //if(ALLOC_DEBUG) 
            printf("Page fault resolved for page = %x, appID = %d, handling it at the moment. Page fault size = %d\n", done->key, done->appID, page_fault_set.size());

        page_fault_set.erase(done->key);
        page_fault_list.pop_front();
        if(GET_FAULT_STATUS)
        {
            printf("[FAULT_Q_STAT] size = %d: {",page_fault_set.size());
            for(std::set<new_addr_type>::iterator itr = page_fault_set.begin(); itr!=page_fault_set.end();itr++)
                printf("%x, ", *itr);
            printf("}\n");
        }

        delete done; //Delete the page fault request
        page_fault_last_service_time = gpu_sim_cycle + gpu_tot_sim_cycle;
    }

}

bool DRAM_layout::is_active_fault(new_addr_type address)
{
    new_addr_type key = address / m_config->base_page_size;
    if(ALLOC_DEBUG) printf("Is active fault check routine for address = %x\n",address);

//    if(GET_FAULT_STATUS)
//    {
//        printf("During the routine is_active_fault check: [FAULT_Q_STAT] size = %d: {",page_fault_set.size());
//        for(std::set<new_addr_type>::iterator itr = page_fault_set.begin(); itr!=page_fault_set.end();itr++)
//            printf("%x, ", *itr);
//        printf("}\n");
//    }

    remove_page_fault();
    std::set<new_addr_type>::iterator itr = page_fault_set.find(key);
    if(itr == page_fault_set.end())
    {
        if(ALLOC_DEBUG)
            printf("Checking if addr = %x is an access in the fault region, not a fault\n", address);
        return false;
    }
    else
    {
        if(ALLOC_DEBUG)
            printf("Checking if addr = %x is an access in the fault region, this a fault. page that fault = %x\n", address, *itr);
        return true;
    }
}

//Ongoing page fault, do nothing. is_active_fault will handle the rest
//New fault, handle page fault and add this page to the associated fault set and list
void DRAM_layout::insert_page_fault(new_addr_type key, int appID)
{
    remove_page_fault();
    //Reset the timing for faults
    if(page_fault_list.empty())
        page_fault_last_service_time = gpu_sim_cycle + gpu_tot_sim_cycle;
        
    if(page_fault_set.find(key) == page_fault_set.end()){
        //if(ALLOC_DEBUG) 
            printf("Inserting page fault to page = %x from app = %d to the FIFO, current size = %d\n",key, appID, page_fault_set.size());
        if(GET_FAULT_STATUS)
        {
            printf("[FAULT_Q_STAT] size = %d: {",page_fault_set.size());
            for(std::set<new_addr_type>::iterator itr = page_fault_set.begin(); itr!=page_fault_set.end();itr++)
                printf("%x, ", *itr);
            printf("}\n");
        }
        page_fault_set.insert(key);
        page_fault_list.push_back(new page_fault_info(key,appID)); //Add this to the pending fault FIFO queue
    }
}

//Return true if the request is resolving a page fault. This is called from tlb.cc, which will block accesses to this page when fault happens
bool mmu::is_active_fault(new_addr_type address)
{
    m_DRAM->is_active_fault(address);
}

//This is used from tlb.cc
new_addr_type mmu::get_pa(new_addr_type addr, int appID){

    return get_pa(addr,appID,true);
}

//Return physical address given a mem fetch. This is called from addrdec when it only want to check which memory partition a request go to
new_addr_type mmu::get_pa(new_addr_type addr, int appID, bool isRead){


    bool fault;
    return get_pa(addr, appID, &fault, isRead);
    
//
////new_addr_type mmu::get_pa(mem_fetch * mf){
//    assign_va_to_pa(addr, appID,isRead);
//    //mf->set_page_fault(!(assign_va_to_pa(mf))); //assign_va_to_pa return false if there is a page fault
//    return (((*va_to_pa_mapping)[addr >> m_config->page_size]) << m_config->page_size) | (addr & (m_config->base_page_size-1));
//    //return (*va_to_pa_mapping)[addr];
}


////Return physical address given a mem fetch. This is called from addrdec when it only want to check which memory partition a request go to
new_addr_type mmu::old_get_pa(new_addr_type addr, int appID, bool isRead){


//    if(COALESCE_DEBUG) printf("Getting PA for VA = %x, app = %d, actual PA is %x\n", addr, appID, (((*va_to_pa_mapping)[addr >> m_config->page_size]) << m_config->page_size) | (addr & (m_config->base_page_size-1)));

    bool fault;
    return old_get_pa(addr, appID, &fault, isRead);
    
}



void mmu::send_scan_request()
{
    m_DRAM->send_scan_request();
}


//void mmu::add_metadata_entry(new_addr_type address, page_metadata * metadata)
//{
//    va_to_metadata_mapping[address] = metadata;
//}

//This function return page metadata of va (with its appID so that it gives the correct page)
page_metadata * mmu::update_metadata(new_addr_type va, int appID)
{
    //Get the page from VA
    unsigned searchID = ((va>>m_config->page_size) << m_config->page_size) | appID;
    page * the_page = (*va_to_page_mapping)[searchID];

    //Whoever call this please do not forget to delete the return pointer
    return the_page==NULL?NULL:create_metadata(the_page);
}


bool mmu::allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID)
{
    page * target_page = m_DRAM->allocate_PA(va_base, pa_base, appID);

    if(target_page != NULL){
        //Add this page to VA to page mapping so that it can be easily search
        // SearchID = [base page VA | appID ]
        unsigned searchID = va_base | appID;
        if(((MERGE_DEBUG || MERGE_DEBUG_SHORT) && ((gpu_sim_cycle + gpu_tot_sim_cycle) > 0)) || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Physical page for VA = %x, app = %d not in DRAM. Allocating a free page for this VA at PA = %x, VA base is %x (Occupy page for this app should contain this page), cycle = %lld. use_value = %d.\n",va_base, appID, target_page->starting_addr,target_page->va_page_addr, gpu_sim_cycle + gpu_tot_sim_cycle, target_page->used);
        (*va_to_page_mapping)[searchID] = target_page; 
        return true;
    }
    if(MERGE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Physical page for VA = %x, app = %d either is not in DRAM (Impossible) or clash with page table space.\n",va_base, appID);
    return false;
}

// This set flags in physical page pa_base as used by appID. Initilize all the page metdata
// Also create the page table entries as needed. Return false if for some reason this pa_base
// conflict with PT_SPACE
page * DRAM_layout::allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID)
{
    //Grab this physical page, check if it is occupied by PT
    if((MERGE_DEBUG && ((gpu_sim_cycle+gpu_tot_sim_cycle) > 0)) || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Trying to get physical page for VA = %x using PA %x as the key. App creation ID = %d,\n",va_base, pa_base, appID);
    page * target_page = find_page_from_pa(pa_base);

    if(target_page == NULL)
    {
        if(MERGE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Cannot find physical page for VA = %x using PA %x as the key. App = %d.\n",va_base, pa_base, appID);
        return NULL;
    }

    //If so, return false, do nothing
    if(target_page->appID == PT_SPACE) return NULL;

    //Otherwise, allocate the page, update page metadata, create the PTE entry for this page

    target_page->va_page_addr = va_base;
    target_page->appID = appID;
    target_page->utilization = 0.0;
    target_page->dataPresent = false;
    target_page->used = true;
    target_page->last_accessed_time = gpu_sim_cycle + gpu_tot_sim_cycle;

    //Take this page out from the free page list. Add it to the occupied page list
//    return target_page;
//    free_pages[target_page->size]->remove(target_page);
//    occupied_pages[appID]->push_back(target_page); 
//    //Set this page's parent page as used by the appID. This can also detect the case when multiple apps are in the same huge page range, and will mark appID as MIXAPP
//    propagate_parent_page_as_used(target_page);    
    
    target_page->leaf_pte = m_pt_root->add_entry(va_base, appID, true);

    //Return the page to MMU so it can get the mapping between VA and page
    return target_page;

}

bool DRAM_layout::free_up_page(page * this_page)
{
    this_page->va_page_addr = 0;
    this_page->appID = NOAPP;
    this_page->utilization = 0.0;
    this_page->dataPresent = false;
    this_page->used = true;
    this_page->last_accessed_time = 0;
    occupied_pages[this_page->appID]->remove(this_page);
    free_pages[this_page->size]->remove(this_page);
}

bool mmu::update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va, new_addr_type new_pa, int appID)
{
    return m_DRAM->update_mapping(old_va, old_pa, new_va, new_pa, appID);
}


// Update mapping for this VA page
bool DRAM_layout::update_mapping(new_addr_type old_va, new_addr_type old_pa, new_addr_type new_va, new_addr_type new_pa, int appID)
{
    page * original = find_page_from_pa(old_pa); // Get the original page, before updating

    if(original->va_page_addr != old_va) return false; // Something happen. The page does not have the same VA
    
    //Allocate this in the new page location
    allocate_PA(new_va, new_pa, appID);
    //Free the old page location
    free_up_page(original);

}

void DRAM_layout::update_parent_metadata(page * triggered_page)
{
    page * parent = triggered_page;
    parent->last_accessed_time = triggered_page->last_accessed_time; //update parent's last_accessed bit
    check_utilization(parent); //update parent's utilization
}

// This is called by the address decoder in main gpgpu-sim
new_addr_type mmu::get_pa(new_addr_type addr, int appID, bool * fault, bool isRead){

    page * this_page =  NULL;
    
    unsigned searchID = ((addr>>m_config->page_size) << m_config->page_size) | appID;


    //Check if this page has their own PT entries setup or not. (VA page seen before, VA not seen)
    if(va_to_page_mapping->find(searchID)!=va_to_page_mapping->end())
    {
        if(MERGE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Searching the map (searchID = %x) for physical page for VA = %x, app = %d. Not the first time access.\n",searchID, addr, appID, gpu_sim_cycle + gpu_tot_sim_cycle);
        this_page = (*va_to_page_mapping)[searchID];
    }
    else
    {
        new_addr_type result = (new_addr_type)(gpu_alloc->translate(appID,  (void*)((appID << 48) | addr))); //Note that at this point, addr should exist because addrdec.cc should have already handle any additional allocations
        if(MERGE_DEBUG || ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Searching the map (searchID = %x). Cannot find the physical page for VA = %x, app = %d. First time access. Allcating this page in the mmu to keep track of metadata\n",searchID, addr, appID, gpu_sim_cycle + gpu_tot_sim_cycle);
        //Allocate this page
        allocate_PA((addr>>m_config->page_size) << m_config->page_size, (result >> m_config->page_size) << m_config->page_size, appID);
        //return result;
    }

    this_page = (*va_to_page_mapping)[searchID];
    if(this_page == NULL)
    {
        printf("This should never happen! Page is already allocated but not found\n");
        exit(0);
    }

    // Updates the base page's metadata
    this_page->last_accessed_time = gpu_sim_cycle + gpu_tot_sim_cycle; //Update last accessed time
    this_page->dataPresent = true;
    this_page->utilization = 1.0; //Might want to use a bit vector in the future to represent each cache line in the small page range. Might be an overkiltar

    // Update parent's metadata
    m_DRAM->update_parent_metadata(this_page);


    new_addr_type result = (new_addr_type)(gpu_alloc->translate(appID,  (void*)((appID << 48) | addr)));
    if(MERGE_DEBUG) printf("Requesting the PA for VA = %x, appID = %d. Got %x, at address %x", appID, addr, result, &result);

    return result; //Get the correct physical address

}

new_addr_type mmu::old_get_pa(new_addr_type addr, int appID, bool * fault, bool isRead){

    //page * this_page = m_DRAM->get_page_from_va(addr,appID); This is slow, use the map to prevent this search
    page * this_page = NULL;
    // SearchID = [base page VA | appID ]
    unsigned searchID = ((addr>>m_config->page_size) << m_config->page_size) | appID;

    //Check if this page has their own PT entries setup or not. (VA page seen before, VA not seen)
    if(va_to_page_mapping->find(searchID)!=va_to_page_mapping->end())
    {
        if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Searching the map (searchID = %x) for physical page for VA = %x, app = %d not in DRAM. Allocating a free page for this VA page next, cycle = %lld\n",searchID, addr, appID, gpu_sim_cycle + gpu_tot_sim_cycle);
        this_page = (*va_to_page_mapping)[searchID];
    }



    if(this_page == NULL) //Cannot find the page. Setup the page table entries
    {
        *fault = true;
        //Grab a free page
        if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Physical page for VA = %x, app = %d not in DRAM. Allocating a free page for this VA page next, cycle = %lld. Adding searchID = %x to the map\n",addr, appID, gpu_sim_cycle + gpu_tot_sim_cycle, searchID);
        this_page = m_DRAM->allocate_free_page(m_config->base_page_size,appID);
        this_page->va_page_addr = (addr >> m_config->page_size) << m_config->page_size;
        if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Physical page for VA = %x, app = %d not in DRAM. Got a free page for this VA at PA = %x, VA base is %x (Occupy page for this app should contain this page), cycle = %lld. use_value = %d \n",addr, appID, this_page->starting_addr,this_page->va_page_addr, gpu_sim_cycle + gpu_tot_sim_cycle, this_page->used);
        // Assign the page to this VA, so that the next time this is VA is searched, lookup time is significantly faster
        (*va_to_page_mapping)[searchID] = this_page; 
    }
    else if(this_page->used = false) //Page is not being used (invalid page). Set fault and mark the page as valid (note that this fault will be handled later when the mem_fetch reach DRAM.
    {
        if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Physical page for VA = %x, app = %d is in DRAM, but not used. Cycle = %lld\n",addr, appID, gpu_sim_cycle + gpu_tot_sim_cycle);
        *fault = true;
        bool found_page = false;
        page * temp;
        // What if this is a huge block?
        while(this_page->parent_page != NULL)
        {
            if(this_page->parent_page->used == true)
            {
                found_page = true;
                temp = this_page->parent_page; //Set the correct level
                break;
            }
        }
        if(!found_page) //Found that the huge block is not active. Re-activate this page
        {
            this_page->used = true;
            this_page->appID = appID;
            this_page->va_page_addr = (addr >> m_config->page_size) << m_config->page_size;
            this_page->utilization = 1.0;
        }
        else this_page = temp;
    }
    else
    {
        if(ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Found physical page for VA = %x, app = %d in DRAM. Physical page is at 0x%x. Cycle = %lld\n",addr, appID, this_page->starting_addr, gpu_sim_cycle + gpu_tot_sim_cycle);
        *fault = false;
    }

    if(this_page == NULL) //If we run out of space
    {
        //TODO-low priority: Eviction and then use the new acquired free page
    }
    else
    {
        this_page->last_accessed_time = gpu_sim_cycle + gpu_tot_sim_cycle; //Update last accessed time
    }

    if(ALLOC_DEBUG) printf("Requesting the page with va = 0x%x, appID = %d, fault = %d, returning physical address 0x%x, starting page = 0x%x\n",addr, appID, *fault, this_page->starting_addr | (addr & this_page->size-1), this_page->starting_addr);
    return (this_page->starting_addr | (addr & (this_page->size-1))); 

    

//Old
//    *fault = !(assign_va_to_pa(addr, appID,isRead)); //assign_va_to_pa return false if there is a new page fault
//
//    new_addr_type key = addr >> m_config->page_size;
//    if(ALLOC_DEBUG)
//        printf("Translating the virtual address = %x, app = %d, fault = %d. VA Page = 0x%x (%d), base_page_size = 0x%x (%d)\n", addr, appID, *fault, key, key, m_config->base_page_size, m_config->base_page_size);
//
//    if((*fault) == true && (m_config->page_transfer_time > 0 || m_config->enable_PCIe != 0)) //If PCIe modeling is enabled
//    {
//        m_DRAM->insert_page_fault(key,appID);
//    }
//
//    if(ALLOC_DEBUG) printf("Address %x is mapped to %x\n", addr, (((*va_to_pa_mapping)[addr >> m_config->page_size]) << m_config->page_size) | (addr & (m_config->base_page_size-1)) );
//    return (((*va_to_pa_mapping)[addr >> m_config->page_size]) << m_config->page_size) | (addr & (m_config->base_page_size-1));
    //mf->set_page_fault(!(assign_va_to_pa(mf))); //assign_va_to_pa return false if there is a page fault
    //return (*va_to_pa_mapping)[addr];
}

page * mmu::find_physical_page(mem_fetch * mf)
{
    return m_DRAM->get_page_from_va(mf->get_addr(), mf->get_appID(), m_config->base_page_size);
}

void mmu::check_utilization(page * this_page)
{
    m_DRAM->check_utilization(this_page);
}

float mmu::check_utilization()
{
    return m_DRAM->check_utilization();
}

page * mmu::find_physical_page(new_addr_type va, int appID, unsigned size)
{
    return m_DRAM->get_page_from_va(va, appID, size);
}

int mmu::demote_page(page * this_page)
{
    return m_DRAM->demote_page(this_page);
}

unsigned long long get_total_size()
{
    unsigned long long total_size = 0;
    for(int i=0;i<3/*FIXME:m_config->num_apps*/;i++)
    {
        //Goes through occupied page to get the total size
    }
    return total_size;
}

// Return true if succeeded
int mmu::coalesce_page(page * this_page)
{
    return m_DRAM->coalesce_page(this_page);
}

void mmu::set_L2_tlb(tlb_tag_array * L2TLB)
{
   printf("Setting L2 TLB for the MMU at address = %x\n", L2TLB);
   l2_tlb = L2TLB;
   //Sending all the promote/demote requests 
   while(!promoted_list->empty())
   {
        if(PROMOTE_DEBUG) printf("Send a pending promotion call for VA = %x, appID = %d.\n", promoted_list->front().first, promoted_list->front().second);
        if(m_config->enable_page_coalescing) l2_tlb->promotion(promoted_list->front().first, promoted_list->front().second);
        //if(m_config->enable_page_coalescing) l2_tlb->promotion(promoted_list->front().first, App::get_app_id(promoted_list->front().second));
        promoted_list->pop_front();
   }
   while(!demoted_list->empty())
   {
        if(PROMOTE_DEBUG) printf("Send a pending demotion call for VA = %x, appID = %d\n", demoted_list->front().first, demoted_list->front().second);
        if(m_config->enable_page_coalescing) l2_tlb->demote_page(demoted_list->front().first, demoted_list->front().second);
        demoted_list->pop_front();
   }
}

int mmu::promote_page(new_addr_type va, int appID)
{
    if(need_init || l2_tlb == NULL)
    {
        if(PROMOTE_DEBUG) printf("MMU got a promotion call during INIT for VA = %x, appID = %d.\n", va, appID);
        promoted_list->push_back(std::pair<new_addr_type,int>(va,appID));
    }
    else{
    // Mark the page metadata by calling coalesce_page(page)
    // Same sa demote page, tlb will handle this call so that it can check MMU's promotion/demotion return status
//    unsigned searchID = ((va>>m_config->page_size) << m_config->page_size) | appID;
//    coalesce_page((*va_to_page_mapping)[searchID]);

        // Then update the promoted page list in tlb.cc
        if(PROMOTE_DEBUG) printf("MMU got a promotion call for VA = %x, appID = %d\n", va, appID);
        if(m_config->enable_page_coalescing) return l2_tlb->promotion(va, appID);
        else return 0;
    }
}

int mmu::demote_page(new_addr_type va, int appID)
{
    if(need_init || l2_tlb == NULL)
    {
        if(PROMOTE_DEBUG) printf("MMU got a demotion call during INIT for VA = %x, appID = %d\n", va, appID);
        demoted_list->push_back(std::pair<new_addr_type,int>(va,appID));
    }
    else
    {
    // Mark the page metadata by call demote_page
//    unsigned searchID = ((va>>m_config->page_size) << m_config->page_size) | appID;

//    demote_page((*va_to_page_mapping)[searchID]); //Note: No need to do this as tlb will handle it. So, just call tlb->demote page

    // Then update the promoted page list in tlb.cc
        if(PROMOTE_DEBUG) printf("MMU got a demotion call for VA = %x, appID = %d.\n", va, appID);
        if(m_config->enable_page_coalescing) return l2_tlb->demote_page(va, appID);
        else return 0;
    }
}

unsigned mmu::get_bitmask(int level)
{
    return m_pt_root->get_bitmask(level);
}


void mmu::set_stat(memory_stats_t * stat)
{
    m_stats = stat;
    printf("Setting stat object in MMU\n");
    m_DRAM->set_stat(stat);
}


void mmu::set_DRAM_channel(dram_t * dram_channel, int channel_id)
{
    m_DRAM->set_DRAM_channel(dram_channel,channel_id);
}

