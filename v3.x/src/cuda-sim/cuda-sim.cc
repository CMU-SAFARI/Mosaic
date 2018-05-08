// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Jimmy Kwa
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

#include "cuda-sim.h"

#include "instructions.h"
#include "ptx_ir.h"
#include "ptx.tab.h"
#include "ptx_sim.h"
#include <stdio.h>
#include <numeric>

#include "opcodes.h"
#include "../statwrapper.h"
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "../abstract_hardware_model.h"
#include "memory.h"
#include "ptx-stats.h"
#include "ptx_loader.h"
#include "ptx_parser.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "ptx_sim.h"
#include "../gpgpusim_entrypoint.h"
#include "decuda_pred_table/decuda_pred_table.h"
#include "../stream_manager.h"
#include <pthread.h>
#include "../common.h"
//#include "../gpgpu-sim/App.h"

extern mmu * g_mmu;

extern thread_to_appID_struct* thread_to_appID;

int gpgpu_ptx_instruction_classification;
void ** g_inst_classification_stat = NULL;
void ** g_inst_op_classification_stat= NULL;
int g_ptx_kernel_count = -1; // used for classification stat collection purposes
int g_debug_execution = 0;
int g_debug_thread_uid = 0;
addr_t g_debug_pc = 0xBEEF1518;
// Output debug information to file options

unsigned g_ptx_sim_num_insn = 0;
unsigned gpgpu_param_num_shaders = 0;

char *opcode_latency_int, *opcode_latency_fp, *opcode_latency_dp;
char *opcode_initiation_int, *opcode_initiation_fp, *opcode_initiation_dp;

void ptx_opcocde_latency_options (option_parser_t opp) {
	option_parser_register(opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
			"Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV>"
			"Default 1,1,19,25,145",
			"1,1,19,25,145");
	option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR, &opcode_latency_fp,
			"Opcode latencies for single precision floating points <ADD,MAX,MUL,MAD,DIV>"
			"Default 1,1,1,1,30",
			"1,1,1,1,30");
	option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR, &opcode_latency_dp,
			"Opcode latencies for double precision floating points <ADD,MAX,MUL,MAD,DIV>"
			"Default 8,8,8,8,335",
			"8,8,8,8,335");
	option_parser_register(opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
			"Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV>"
			"Default 1,1,4,4,32",
			"1,1,4,4,32");
	option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR, &opcode_initiation_fp,
			"Opcode initiation intervals for single precision floating points <ADD,MAX,MUL,MAD,DIV>"
			"Default 1,1,1,1,5",
			"1,1,1,1,5");
	option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR, &opcode_initiation_dp,
			"Opcode initiation intervals for double precision floating points <ADD,MAX,MUL,MAD,DIV>"
			"Default 8,8,8,8,130",
			"8,8,8,8,130");
}

static address_type get_converge_point(address_type pc);

void gpgpu_t::gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref, int dim, int readmode, int ext)
{
   std::string texname(name);
   m_NameToTextureRef[texname] = texref;
   const textureReferenceAttr *texAttr = new textureReferenceAttr(texref, dim, (enum cudaTextureReadMode)readmode, ext);
   m_TextureRefToAttribute[texref] = texAttr;
}

const char* gpgpu_t::gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref)
{
   std::map<std::string, const struct textureReference*>::iterator itr = m_NameToTextureRef.begin();
   while (itr != m_NameToTextureRef.end()) {
      if ((*itr).second == texref) {
         const char *p = ((*itr).first).c_str();
         return p;
      }
      itr++;
   }
   return NULL;
}

unsigned int intLOGB2( unsigned int v ) {
   unsigned int shift;
   unsigned int r;

   r = 0;

   shift = (( v & 0xFFFF0000) != 0 ) << 4; v >>= shift; r |= shift;
   shift = (( v & 0xFF00    ) != 0 ) << 3; v >>= shift; r |= shift;
   shift = (( v & 0xF0      ) != 0 ) << 2; v >>= shift; r |= shift;
   shift = (( v & 0xC       ) != 0 ) << 1; v >>= shift; r |= shift;
   shift = (( v & 0x2       ) != 0 ) << 0; v >>= shift; r |= shift;

   return r;
}

void gpgpu_t::gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array)
{
   m_TextureRefToCudaArray[texref] = array;
   unsigned int texel_size_bits = array->desc.w + array->desc.x + array->desc.y + array->desc.z;
   unsigned int texel_size = texel_size_bits/8;
   unsigned int Tx, Ty;
   int r;

   printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);
   printf("GPGPU-Sim PTX:   texture cache linesize = %d\n", m_function_model_config.get_texcache_linesize());
   //first determine base Tx size for given linesize
   switch (m_function_model_config.get_texcache_linesize()) {
   case 16:  Tx = 4; break;
   case 32:  Tx = 8; break;
   case 64:  Tx = 8; break;
   case 128: Tx = 16; break;
   case 256: Tx = 16; break;
   default:
      printf("GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n", m_function_model_config.get_texcache_linesize());
      assert(0);
      break;
   }
   r = texel_size >> 2;
   //modify base Tx size to take into account size of each texel in bytes
   while (r != 0) {
      Tx = Tx >> 1;
      r = r >> 2;
   }
   //by now, got the correct Tx size, calculate correct Ty size
   Ty = m_function_model_config.get_texcache_linesize()/(Tx*texel_size);

   printf("GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n", Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));
   printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n", texel_size, intLOGB2(texel_size));
   printf("GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n", array->devPtr32);
   printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);
   struct textureInfo* texInfo = (struct textureInfo*) malloc(sizeof(struct textureInfo));
   texInfo->Tx = Tx;
   texInfo->Ty = Ty;
   texInfo->Tx_numbits = intLOGB2(Tx);
   texInfo->Ty_numbits = intLOGB2(Ty);
   texInfo->texel_size = texel_size;
   texInfo->texel_size_numbits = intLOGB2(texel_size);
   m_TextureRefToTexureInfo[texref] = texInfo;
}

unsigned g_assemble_code_next_pc=0;
std::map<unsigned,function_info*> g_pc_to_finfo;
std::vector<ptx_instruction*> function_info::s_g_pc_to_insn;

#define MAX_INST_SIZE 8 /*bytes*/

void function_info::ptx_assemble()
{
   if( m_assembled ) {
      return;
   }

   // get the instructions into instruction memory...
   unsigned num_inst = m_instructions.size();
   m_instr_mem_size = MAX_INST_SIZE*(num_inst+1);
   m_instr_mem = new ptx_instruction*[ m_instr_mem_size ];

   printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ", m_name.c_str() );
   fflush(stdout);
   std::list<ptx_instruction*>::iterator i;

   addr_t PC = g_assemble_code_next_pc; // globally unique address (across functions)
   // start function on an aligned address
   for( unsigned i=0; i < (PC%MAX_INST_SIZE); i++ )
      s_g_pc_to_insn.push_back((ptx_instruction*)NULL);
   PC += PC%MAX_INST_SIZE;
   m_start_PC = PC;

   addr_t n=0; // offset in m_instr_mem
   s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() + MAX_INST_SIZE*m_instructions.size());
   for ( i=m_instructions.begin(); i != m_instructions.end(); i++ ) {
      ptx_instruction *pI = *i;
      if ( pI->is_label() ) {
         const symbol *l = pI->get_label();
         labels[l->name()] = n;
      } else {
         g_pc_to_finfo[PC] = this;
         m_instr_mem[n] = pI;
         s_g_pc_to_insn.push_back(pI);
         assert(pI == s_g_pc_to_insn[PC]);
         pI->set_m_instr_mem_index(n);
         pI->set_PC(PC);
         assert( pI->inst_size() <= MAX_INST_SIZE );
         for( unsigned i=1; i < pI->inst_size(); i++ ) {
            s_g_pc_to_insn.push_back((ptx_instruction*)NULL);
            m_instr_mem[n+i]=NULL;
         }
         n  += pI->inst_size();
         PC += pI->inst_size();
      }
   }
   g_assemble_code_next_pc=PC;
   for ( unsigned ii=0; ii < n; ii += m_instr_mem[ii]->inst_size() ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      if ( pI->get_opcode() == BRA_OP || pI->get_opcode() == BREAKADDR_OP  || pI->get_opcode() == CALLP_OP) {
         operand_info &target = pI->dst(); //get operand, e.g. target name
         if ( labels.find(target.name()) == labels.end() ) {
            printf("GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not appear in assembly code.",
                   pI->source_file(),pI->source_line(), target.name().c_str() );
            abort();
         }
         unsigned index = labels[ target.name() ]; //determine address from name
         unsigned PC = m_instr_mem[index]->get_PC();
         m_symtab->set_label_address( target.get_symbol(), PC );
         target.set_type(label_t);
      }
   }

   printf("  done.\n");
   fflush(stdout);
   printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n", m_name.c_str() );

   create_basic_blocks();
   connect_basic_blocks();
   bool modified = false;
   do {
      find_dominators();
      find_idominators();
      modified = connect_break_targets();
   } while (modified == true);

   if ( g_debug_execution>=50 ) {
      print_basic_blocks();
      print_basic_block_links();
      print_basic_block_dot();
   }
   if ( g_debug_execution>=2 ) {
      print_dominators();
   }
   find_postdominators();
   find_ipostdominators();
   if ( g_debug_execution>=50 ) {
      print_postdominators();
      print_ipostdominators();
   }

   printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n", m_name.c_str() );
   for ( unsigned ii=0; ii < n; ii += m_instr_mem[ii]->inst_size() ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      pI->pre_decode();
   }
   printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n", m_name.c_str() );
   fflush(stdout);

   m_assembled = true;
}

addr_t shared_to_generic( unsigned smid, addr_t addr )
{
   assert( addr < SHARED_MEM_SIZE_MAX );
   return SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX + addr;
}

addr_t global_to_generic( addr_t addr )
{
   return addr;
}

bool isspace_shared( unsigned smid, addr_t addr )
{
   addr_t start = SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX;
   addr_t end = SHARED_GENERIC_START + (smid+1)*SHARED_MEM_SIZE_MAX;
   if( (addr >= end) || (addr < start) )
      return false;
   return true;
}

bool isspace_global( addr_t addr )
{
   return (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
}

memory_space_t whichspace( addr_t addr )
{
   if( (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT) ) {
      return global_space;
   } else if( addr >= SHARED_GENERIC_START ) {
      return shared_space;
   } else {
      return local_space;
   }
}

addr_t generic_to_shared( unsigned smid, addr_t addr )
{
   assert(isspace_shared(smid,addr));
   return addr - (SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX);
}

addr_t local_to_generic( unsigned smid, unsigned hwtid, addr_t addr )
{
   assert(addr < LOCAL_MEM_SIZE_MAX);
   return LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
}

bool isspace_local( unsigned smid, unsigned hwtid, addr_t addr )
{
   addr_t start = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid);
   addr_t end   = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * (hwtid+1));
   if( (addr >= end) || (addr < start) )
      return false;
   return true;
}

addr_t generic_to_local( unsigned smid, unsigned hwtid, addr_t addr )
{
   assert(isspace_local(smid,hwtid,addr));
   return addr - (LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid));
}

addr_t generic_to_global( addr_t addr )
{
   return addr;
}


#define max_uint64 (~uint64_t{0})

// interface for physical memory manager
class Abstract_PMM {
   public:
   void allocate(uint64_t ID, uint64_t first_page_num, uint64_t num_pages);
   void free(uint64_t ID, uint64_t first_page_num, uint64_t num_pages);
   uint64_t translate(uint64_t ID, uint64_t page_num);
   void print();
};

// interface for virtual memory manager
class Abstract_VMM {
   public:
   void* allocate(uint64_t bytes);
   void free(void* a);
   void print();
};

class PMM;
class VMM;


void Hub::modify_mapping(bool adding, bool promotion, uint64_t ID, uint64_t vaddr, uint64_t paddr) {
      if (ID = max_uint64) return;
      void* vaddr_ptr = (void*)vaddr;
      void* paddr_ptr = (void*)paddr;
      if (adding) {
         if (promotion) {
            // huge page region at vaddr is being promoted to a huge page
             g_mmu->promote_page((new_addr_type)vaddr, ID);
         } else {
            // mapping being added from base page at vaddr to paddr
             g_mmu->allocate_PA((new_addr_type)vaddr, (new_addr_type) paddr, ID);
         }
      } else {
         if (promotion) {
            // huge page at vadding is being demoted to base pages
             g_mmu->demote_page((new_addr_type)vaddr, ID);
         } else {
            // mapping from vaddr to paddr is being removed
         }
         // mmu->flush(ID)?
      }
}
void Hub::transfer_page(uint64_t ID, uint64_t vaddr, uint64_t old_paddr, uint64_t new_paddr) {
      void* vaddr_ptr = (void*)vaddr;
      void* old_paddr_ptr = (void*)old_paddr;
      void* new_paddr_ptr = (void*)new_paddr;
      printf("page at %p being moved from %p to %p\n", vaddr_ptr, old_paddr_ptr, new_paddr_ptr);
      // base page at vaddr is being moved from old_paddr to new_paddr
      g_mmu->update_mapping((new_addr_type)vaddr, (new_addr_type) old_paddr, (new_addr_type) vaddr, (new_addr_type) new_paddr, ID);
}



class PMM : public Abstract_PMM {
   typedef struct Log_Info {
      uint64_t ID;
      Hub* hub;
      Log_Info(uint64_t _ID, Hub* h) {
         ID = _ID;
         hub = h;
      }
   } Log_Info;

   class Virtual_Chunk;
   class Shared_Chunk;

   class Virtual_Chunk {
      uint64_t paddr1(Log_Info info, uint64_t page_num) { // use if reservation / huge page
         return (chunk_num*info.hub->huge_page_size + page_num)*info.hub->page_size;
      }
      uint64_t paddr2(Log_Info info, uint64_t page_num) { // use if fragmented
         uint64_t cn = pages[page_num].first->chunk_num;
         return (cn*info.hub->huge_page_size + pages[page_num].second)*info.hub->page_size;
      }
      uint64_t vaddr1(Log_Info info, uint64_t page_num) {
         return vaddr + page_num*info.hub->page_size;
      }
      public:
      const uint64_t vaddr;
      uint64_t unused_pages; // 0 if full
      uint64_t chunk_num; // used if chunk is reserved or huge page
      std::vector<std::pair<Shared_Chunk*,uint64_t>> pages;
      Virtual_Chunk(uint64_t huge_page_size, uint64_t va, uint64_t cn = max_uint64) : pages(huge_page_size, std::make_pair(nullptr, max_uint64)), vaddr(va) {
         unused_pages = huge_page_size;
         chunk_num = cn;
      }
      bool full() {
         return unused_pages == 0;
      }
      bool fragmented() {
         return chunk_num == max_uint64;
      }
      uint64_t getPPN(Log_Info info, uint64_t page_num) {
         if (pages[page_num].second == max_uint64) return max_uint64;
         return fragmented() ? paddr2(info, page_num) : paddr1(info, page_num);
      }
      void add_page(Log_Info info, uint64_t page_num) {
         assert(!fragmented() && "allocation error");
         assert(!full() && "allocation error");
         --unused_pages;
         pages[page_num] = std::make_pair(nullptr, page_num);
         info.hub->modify_mapping(true, false, info.ID, vaddr1(info, page_num), paddr1(info, page_num));
         if (full()) info.hub->modify_mapping(true, true, info.ID, vaddr, paddr1(info, 0));
      }
      void add_page(Log_Info info, uint64_t page_num, Shared_Chunk* schunk) {
         assert(fragmented() && "allocation error");
         assert(!full() && "allocation error");
         --unused_pages;
         const uint64_t physical_offset = schunk->add_page(this);
         pages[page_num] = std::make_pair(schunk, physical_offset);
         info.hub->modify_mapping(true, false, info.ID, vaddr1(info, page_num), paddr2(info, page_num));
      }
      Shared_Chunk* remove_page(Log_Info info, uint64_t page_num) {
         ++unused_pages;
         Shared_Chunk* ret_val = nullptr;
         uint64_t paddr;
         if (fragmented()) {
            Shared_Chunk* schunk;
            uint64_t physical_offset;
            std::tie(schunk, physical_offset) = pages[page_num];
            ret_val = schunk->remove_page(this, physical_offset);
            paddr = paddr2(info, page_num);
         } else {
            paddr = paddr1(info, page_num);
            if (unused_pages == 1) {
               info.hub->modify_mapping(false, true, info.ID, vaddr, paddr1(info, 0));
            }
         }
         info.hub->modify_mapping(false, false, info.ID, vaddr1(info, page_num), paddr);
         pages[page_num] = std::make_pair(nullptr, max_uint64);
         return ret_val;
      }
      std::set<Shared_Chunk*> defragment(Log_Info info, uint64_t cn) {
         assert(fragmented() && "can't defragment reservation / huge page");
         std::set<Shared_Chunk*> schunks;
         chunk_num = cn;
         for (uint64_t i = 0; i < pages.size(); ++i) {
            if (pages[i] != std::make_pair((Shared_Chunk*)nullptr, max_uint64)) {
               Shared_Chunk* schunk;
               uint64_t physical_offset;
               std::tie(schunk, physical_offset) = pages[i];
               if (schunk->remove_page(this, physical_offset) != nullptr) {
                  schunks.insert(schunk);
               }
               info.hub->modify_mapping(false, false, info.ID, vaddr1(info, i), paddr2(info, i));
               info.hub->modify_mapping(true, false, info.ID, vaddr1(info, i), paddr1(info, i));
               pages[i] = std::make_pair(nullptr, i);
            }
         }
         if (full()) info.hub->modify_mapping(true, true, info.ID, vaddr, paddr1(info, 0));
         return schunks;
      }
      Shared_Chunk* preempt() {
         assert(!full() && "can't preempt huge page");
         return new Shared_Chunk(this);
      }
      bool transfer_pages(Log_Info info, Shared_Chunk* from, Shared_Chunk* to) {
         for (uint64_t i = 0; i < pages.size() && from->pages_per_vchunk[this] > 0 && to->unused_pages > 0; ++i) {
            Shared_Chunk* schunk;
            uint64_t physical_offset;
            std::tie(schunk, physical_offset) = pages[i];
            if (schunk == from) {
               schunk->remove_page(this, physical_offset);
               uint64_t old_paddr = paddr2(info, i);
               //info.hub->modify_mapping(false, false, info.ID, vaddr1(info, i), paddr2(info, i));
               pages[i] = std::make_pair(to, to->add_page(this));
               //info.hub->modify_mapping(true, false, info.ID, vaddr1(info, i), paddr2(info, i));
               info.hub->transfer_page(info.ID, vaddr1(info, i), old_paddr, paddr2(info, i));
            }
         }
         /*
         if (to->pages_per_vchunk[this] == info.hub->huge_page_size) {
            // make huge page?
         }*/
         bool ret_val = from->pages_per_vchunk[this] == 0;
         if (ret_val) from->pages_per_vchunk.erase(this);
         return ret_val;
      }
   };

   class Shared_Chunk {
      public:
      const uint64_t chunk_num;
      uint64_t unused_pages;
      std::map<Virtual_Chunk*,uint64_t> pages_per_vchunk;
      std::vector<uint64_t> free_offsets;
      Shared_Chunk(Virtual_Chunk* vchunk) : chunk_num(vchunk->chunk_num), pages_per_vchunk(), free_offsets() {
         // move pages from reservation into shared chunk
         assert(!vchunk->fragmented() && "memory error");
         for (uint64_t i = 0; i < vchunk->pages.size(); ++i) {
            if (vchunk->pages[i] == std::make_pair((Shared_Chunk*)nullptr, i)) {
               vchunk->pages[i] = std::make_pair(this, i);
               ++pages_per_vchunk[vchunk];
            } else {
               free_offsets.push_back(i);
            }
         }
         vchunk->chunk_num = max_uint64;
         unused_pages = vchunk->unused_pages;
      }
      Shared_Chunk(uint64_t huge_page_size, uint64_t cn) : chunk_num(cn), pages_per_vchunk(), free_offsets(huge_page_size) {
         unused_pages = huge_page_size;
         std::iota(free_offsets.begin(), free_offsets.end(), 0);
      }
      bool full() {
         return unused_pages == 0;
      }
      bool empty() {
         return pages_per_vchunk.empty();
      }
      uint64_t add_page(Virtual_Chunk* vchunk) {
         assert(unused_pages != 0 && "allocation error");
         --unused_pages;
         ++pages_per_vchunk[vchunk];
         const uint64_t physical_offset = free_offsets.back();
         free_offsets.pop_back();
         return physical_offset;
      }
      Shared_Chunk* remove_page(Virtual_Chunk* vchunk, uint64_t physical_offset) {
         Shared_Chunk* ret_val = nullptr;
         if (unused_pages == 0) ret_val = this;
         ++unused_pages;
         if (pages_per_vchunk[vchunk] == 1) {
            pages_per_vchunk.erase(vchunk);
            if (empty()) {
               ret_val = this;
            }
         } else {
            --pages_per_vchunk[vchunk];
         }
         free_offsets.push_back(physical_offset);
         return ret_val;
      }
   };

   class Vspace_Data {
      public:
      std::unordered_map<uint64_t,Virtual_Chunk*> virt_to_phys;
      std::unordered_set<Virtual_Chunk*> reserved_chunks;
      std::unordered_set<Shared_Chunk*> fragmented_chunks;
      Vspace_Data() : virt_to_phys(), reserved_chunks(), fragmented_chunks() {}
   };

   Hub* hub;
   uint64_t huge_page_size;
   uint64_t max_chunks;
   std::unordered_map<uint64_t,Vspace_Data*> ID_to_data;
   std::vector<uint64_t> free_chunks, special_chunks;
   uint64_t pages_allocated, pages_freed;

   // attempts to free up one chunk
   uint64_t coalesce(uint64_t ID = max_uint64) {
      uint64_t ret_val = max_uint64;
      if (ID != max_uint64) ret_val = coalesce2(ID);
      for (auto it = ID_to_data.begin(); ret_val == max_uint64 && it != ID_to_data.end(); ++it) {
         if (it->first == ID) continue;
         ret_val = coalesce2(it->first);
      }
      return ret_val;
   }
   uint64_t coalesce2(uint64_t ID) {
      Vspace_Data* data = ID_to_data[ID];
      // check if coalescing shared chunks will work
      Shared_Chunk* most_unused_schunk = nullptr;
      uint64_t total_free_pages = 0;
      if (data->fragmented_chunks.size() > 0) {
         most_unused_schunk = *(data->fragmented_chunks.begin());
         for (Shared_Chunk* schunk : data->fragmented_chunks) {
            total_free_pages += schunk->unused_pages;
            if (schunk->unused_pages > most_unused_schunk->unused_pages) {
               most_unused_schunk = schunk;
            }
         }
      }

      // not enough free space, fall back to reservations
      while (data->reserved_chunks.size() > 0 && (total_free_pages < huge_page_size)) {
         // check if coalescing reservations will work and find most empty reservation
         Virtual_Chunk* most_unused_vchunk = *(data->reserved_chunks.begin());
         for (Virtual_Chunk* vchunk : data->reserved_chunks) {
            if (vchunk->unused_pages > most_unused_vchunk->unused_pages) {
               most_unused_vchunk = vchunk;
            }
         }
         // preempt most empty reservation
         Shared_Chunk* schunk = most_unused_vchunk->preempt();
         data->reserved_chunks.erase(most_unused_vchunk);
         data->fragmented_chunks.insert(schunk);
         total_free_pages += most_unused_vchunk->unused_pages;
         if (most_unused_schunk == nullptr || schunk->unused_pages > most_unused_schunk->unused_pages) {
            most_unused_schunk = schunk;
         }
      }

      // not enough free space
      if (total_free_pages < huge_page_size) return max_uint64;

      // coalesce shared chunks
      Log_Info info = Log_Info(ID, hub);
      data->fragmented_chunks.erase(most_unused_schunk);
      while (!most_unused_schunk->empty()) {
         Virtual_Chunk* vchunk = most_unused_schunk->pages_per_vchunk.begin()->first;
         while (!vchunk->transfer_pages(info, most_unused_schunk, *(data->fragmented_chunks.begin()))) {
            assert((*(data->fragmented_chunks.begin()))->full() && "code error");
            data->fragmented_chunks.erase(data->fragmented_chunks.begin());
         }
      }
      assert(most_unused_schunk->empty() && "counting error");
      const uint64_t chunk_num = most_unused_schunk->chunk_num;
      delete most_unused_schunk;
      return chunk_num;
   }
   float fmfi() {
      /* This doesn't make any sense
      if (free_chunks.size() == 0) return 0.0;
      uint64_t total_free = max_chunks * huge_page_size - (pages_allocated - pages_freed);
      return 1 - (total_free / huge_page_size / free_chunks.size());*/
      if (free_chunks.size() == 0) return 1.0;
      uint64_t total_free = max_chunks * huge_page_size - (pages_allocated - pages_freed);
      return 1.0 - ((float)free_chunks.size() / (float)(total_free / huge_page_size));
      
   }
   void prefragment(float target_fmfi, uint64_t fppf) {
      assert(target_fmfi >= 0.0 && target_fmfi <= 1.0 && "invalid target fmfi");
      assert(target_fmfi == 0.0 || fppf > 0 && fppf < huge_page_size && "invalid pages per frame");
      if (target_fmfi == 0.0f) return;
      const uint64_t hps = huge_page_size;
      //uint64_t appID;
      //if(App::is_registered(max_uint64)) appID = App::get_app_id(max_uint64);
      //else appID = App::register_app(max_uint64);
      //if (ID_to_data[appID] == nullptr) ID_to_data[appID] = new Vspace_Data();
      //Vspace_Data* data = ID_to_data[appID];
      if (ID_to_data[max_uint64] == nullptr) ID_to_data[max_uint64] = new Vspace_Data();
      Vspace_Data* data = ID_to_data[max_uint64];
      uint64_t vcn = 1 + hub->start_vaddr / hub->page_size / hps;
      while (fmfi() < target_fmfi && !free_chunks.empty()) {
         Virtual_Chunk* chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size, free_chunks.back());
         free_chunks.pop_back();
         data->virt_to_phys[vcn] = chunk;
         data->reserved_chunks.insert(chunk);
         //Log_Info info = Log_Info(appID, hub);
         Log_Info info = Log_Info(max_uint64, hub);
         for (uint64_t i = 0; i < fppf; ++i) {
            chunk->add_page(info, i);
         }
         pages_allocated += fppf;
         ++vcn;
      }
      if (fmfi() < target_fmfi && free_chunks.empty()) printf("could not acheive desired fragmentation");
   }

   public:
   PMM(Hub* h, uint64_t hps, uint64_t total_pages, float tf = 0.0f, uint64_t fppf = 1) : free_chunks(total_pages / hps - 100), special_chunks(100), ID_to_data() {
      hub = h;
      huge_page_size = hps;
      max_chunks = total_pages / huge_page_size - 100;
      std::iota(free_chunks.rbegin(), free_chunks.rend(), 0);
      std::iota(special_chunks.rbegin(), special_chunks.rend(), max_chunks);
      pages_allocated = 0; pages_freed = 0;
      prefragment(tf, fppf);
   }

   uint64_t translate(uint64_t ID, uint64_t vpn) {
      Vspace_Data* data = ID_to_data[ID];
      if (data == nullptr) {
         ID_to_data.erase(ID);
         return max_uint64;
      }
      const uint64_t vcn = vpn / huge_page_size; // virtual chunk number
      const uint64_t virtual_offset = vpn % huge_page_size;
      Virtual_Chunk* chunk = data->virt_to_phys[vcn];
      if (chunk == nullptr) {
         data->virt_to_phys.erase(vcn);
         return max_uint64;
      }
      Log_Info info = Log_Info(ID, hub);
      return chunk->getPPN(info, virtual_offset);
   }

   void print() {
      uint64_t huge_pages = 0, eligible_regions = 0, reserved_regions = 0;
      uint64_t reserved_unused_pages = 0;
      std::set<Shared_Chunk*> compacted_regions;
      for (auto itt = ID_to_data.begin(); itt != ID_to_data.end(); ++itt) {
         Vspace_Data* data = itt->second;
         for (auto it = data->virt_to_phys.begin(); it != data->virt_to_phys.end(); ++it) {
            Virtual_Chunk* vchunk = it->second;
            if (vchunk->full()) ++eligible_regions;
            if (vchunk->full() && !vchunk->fragmented()) ++huge_pages;
            if (!vchunk->full() && !vchunk->fragmented()) {
               ++reserved_regions;
               reserved_unused_pages += vchunk->unused_pages;
            }
            for (auto page : vchunk->pages) {
               if (page.first != nullptr) compacted_regions.insert(page.first);
            }
         }
      }
      uint64_t compacted_unused_pages = 0;
      for (auto schunk : compacted_regions) {
         compacted_unused_pages += schunk->unused_pages;
      }
      printf("%llu reserved regions (%g%% utilized)\n", reserved_regions, 100.0*double(huge_page_size*reserved_regions-reserved_unused_pages)/double(huge_page_size*reserved_regions));
      printf("%zu compacted regions (%g%% utilized)\n", compacted_regions.size(), 100.0*double(huge_page_size*compacted_regions.size()-compacted_unused_pages)/double(huge_page_size*compacted_regions.size()));
      printf("%llu total pages freed / %llu total pages allocated\n", pages_freed, pages_allocated);
      printf("%llu huge pages / %llu eligible regions\n", huge_pages, eligible_regions);
   }

   void allocate(uint64_t ID, uint64_t first_vpn, uint64_t num_pages) {
      if (ID_to_data[ID] == nullptr) ID_to_data[ID] = new Vspace_Data();
      Vspace_Data* data = ID_to_data[ID];
      // vpn = virtual page number
      for (uint64_t vpn = first_vpn; vpn < first_vpn + num_pages; ++vpn) {
         const uint64_t vcn = vpn / huge_page_size; // virtual chunk number
         const uint64_t virtual_offset = vpn % huge_page_size;
         const uint64_t hps = huge_page_size;
         Virtual_Chunk* chunk = data->virt_to_phys[vcn];
         Log_Info info = Log_Info(ID, hub);
         if (chunk == nullptr) {
            if (!free_chunks.empty()) {
               // reserve empty physical chunk if possible
               chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size, free_chunks.back());
               free_chunks.pop_back();
               data->reserved_chunks.insert(chunk);
            } else if (virtual_offset == 0 && vpn + hps <= first_vpn + num_pages) {
               // try extra hard to make continuous physical region because it will be nice
               const uint64_t chunk_num = coalesce(ID);
               if (chunk_num != max_uint64) {
                  chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size, chunk_num);
                  data->reserved_chunks.insert(chunk); // will be removed later
               } else {
                  chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size);
               }

            } else {
               chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size);
            }
            data->virt_to_phys[vcn] = chunk;
         } else if (chunk->fragmented() &&
                    chunk->unused_pages + virtual_offset == hps &&
                    vpn + chunk->unused_pages <= first_vpn + num_pages) {
            // try extra hard to make continuous physical region because it will be nice
            const uint64_t chunk_num = coalesce(ID);
            if (chunk_num != max_uint64) {
               std::set<Shared_Chunk*> schunks = chunk->defragment(info, chunk_num);
               for (Shared_Chunk* schunk : schunks) {
                  if (schunk->empty()) {
                     free_chunks.push_back(schunk->chunk_num);
                     data->fragmented_chunks.erase(schunk);
                     delete schunk;
                  } else {
                     data->fragmented_chunks.insert(schunk);
                  }
               }
            }
         }
         if (chunk->fragmented()) {
            // allocate out of already fragmented chunk
            if (!data->fragmented_chunks.empty()) {
               chunk->add_page(info, vpn % hps, *(data->fragmented_chunks.begin()));
               if ((*(data->fragmented_chunks.begin()))->full()) {
                  data->fragmented_chunks.erase(data->fragmented_chunks.begin());
               }
            } else if (!data->reserved_chunks.empty()) {
               // preempt smallest reservation to create new fragmented chunk
               // find smallest reservation
               Virtual_Chunk* vchunk = (*(data->reserved_chunks.begin()));
               for (Virtual_Chunk* vchunk2 : data->reserved_chunks) {
                  if (vchunk2->unused_pages > vchunk->unused_pages) {
                     vchunk = vchunk2;
                  }
               }
               // update references (physical addresses should stay the same)
               Shared_Chunk* schunk = vchunk->preempt();
               data->reserved_chunks.erase(vchunk);
               // add page to new shared chunk
               chunk->add_page(info, vpn % hps, schunk);
               if (!schunk->full()) {
                  data->fragmented_chunks.insert(schunk);
               }
            } else if (!free_chunks.empty()) {
               HAVE_FREE_CHUNK:
               // use free chunk
               const uint64_t chunk_num = free_chunks.back();
               free_chunks.pop_back();
               Shared_Chunk* schunk = new Shared_Chunk(huge_page_size, chunk_num);
               /*
               // find largest fragmented chunk
               Virtual_Chunk* vchunk = (*(data->reserved_chunks.begin()));
               uint64_t min_unused_pages = vchunk->unused_pages;
               for (Virtual_Chunk* vchunk2 : data->reserved_chunks) {
                  if (vchunk2->unused_pages < min_unused_pages) {
                     vchunk = vchunk2;
                     min_unused_pages = vchunk2->unused_pages;
                  }
               }
               // move fragmented pages to new reservation
               std::set<Shared_Chunk*> schunks = vchunk->defragment(info, chunk_num);
               Shared_Chunk* schunk = *schunks.begin();
               schunks.erase(schunks.begin());
               for (Shared_Chunk* schunk : schunks) {
                  if (schunk->empty()) {
                     free_chunks.push_back(schunk->chunk_num);
                     data->fragmented_chunks.erase(schunk);
                     delete schunk;
                  } else {
                     data->fragmented_chunks.insert(schunk);
                  }
               }*/
               // allocate out of shared chunk(s) that have now been freed up
               chunk->add_page(info, virtual_offset, schunk);
               if (!schunk->full()) {
                  data->fragmented_chunks.insert(schunk);
               }
            } else {
               uint64_t chunk_num = coalesce(ID);
               if (chunk_num != max_uint64) {
                  free_chunks.push_back(chunk_num);
                  goto HAVE_FREE_CHUNK;
               } else {
                  printf("Out of useable memory\n");
                  exit(EXIT_FAILURE);
               }
            }
         } else {
            chunk->add_page(info, virtual_offset);
         }
         if (chunk->full()) {
            data->reserved_chunks.erase(chunk);
            if (chunk->fragmented()) {
               // defragment and promote to huge page
               uint64_t chunk_num = max_uint64;
               if (!free_chunks.empty()) {
                  chunk_num = free_chunks.back();
                  free_chunks.pop_back();
               } else {
                  chunk_num = coalesce(ID);
               }
               if (chunk_num != max_uint64) {
                  std::set<Shared_Chunk*> schunks = chunk->defragment(info, chunk_num);
                  for (Shared_Chunk* schunk : schunks) {
                     if (schunk->empty()) {
                        free_chunks.push_back(schunk->chunk_num);
                        data->fragmented_chunks.erase(schunk);
                        delete schunk;
                     } else {
                        data->fragmented_chunks.insert(schunk);
                     }
                  }
               }
            }
         }
      }
      pages_allocated += num_pages;
   }

   void free(uint64_t ID, uint64_t first_vpn, uint64_t num_pages) {
      Vspace_Data* data = ID_to_data[ID];
      for (uint64_t vpn = first_vpn; vpn < first_vpn + num_pages; ++vpn) {
         const uint64_t vcn = vpn / huge_page_size;
         Log_Info info = Log_Info(ID, hub);
         Virtual_Chunk* vchunk = data->virt_to_phys[vcn];
         if (vchunk->full() && !vchunk->fragmented()) {
            data->reserved_chunks.insert(vchunk);
         }
         Shared_Chunk* schunk = vchunk->remove_page(info, vpn % huge_page_size);
         if (schunk != nullptr) {
            if (schunk->empty()) {
               free_chunks.push_back(schunk->chunk_num);
               data->fragmented_chunks.erase(schunk);
               delete schunk;
            } else {
               data->fragmented_chunks.insert(schunk);
            }
         }
         if (vchunk->unused_pages == huge_page_size) {
            if (!vchunk->fragmented()) {
               free_chunks.push_back(vchunk->chunk_num);
            }
            data->virt_to_phys.erase(vcn);
            data->reserved_chunks.erase(vchunk);
            delete vchunk;
         }
      }
      pages_freed += num_pages;
   }
   
   void special_allocate(uint64_t ID, uint64_t vcn) {
      const uint64_t hps = huge_page_size;
      if (ID_to_data[ID] == nullptr) ID_to_data[ID] = new Vspace_Data();
      Vspace_Data* data = ID_to_data[ID];
      assert(!special_chunks.empty() && "no more reserved huge frames for system addresses");
      Virtual_Chunk* chunk = new Virtual_Chunk(hps, vcn*hps*hub->page_size, special_chunks.back());
      special_chunks.pop_back();
      data->virt_to_phys[vcn] = chunk;
      Log_Info info = Log_Info(ID, hub);
      for (uint64_t virtual_offset = 0; virtual_offset < hps; ++virtual_offset) {
         chunk->add_page(info, virtual_offset);
      }
      pages_allocated += hps;
   }
};

class VMM : public Abstract_VMM {
   class Range {
      public:
      uint64_t start, end;

      Range() {}
      Range(uint64_t s, uint64_t e) {
         start = s;
         end = e;
      }
      uint64_t size() const {
         return end-start;
      }
      bool contains(uint64_t num) const {
         return (start <= num) && (num < end);
      }
   };
   struct RangeStartCompare {
      bool operator()(const Range& a, const Range& b) const {
         return a.start < b.start;
      }
   };
   struct RangeSizeCompare {
      bool operator()(const Range& a, const Range& b) const {
         if (a.size() != b.size()) {
            return a.size() < b.size();
         } else {
            return a.start < b.start;
         }
      }
   };

   uint64_t end_addr;
   uint64_t in_use;
   uint64_t ID;
   Hub* hub;
   std::set<Range, RangeStartCompare> used_ranges;
   std::set<Range, RangeSizeCompare> free_ranges;

   public:
   VMM(Hub* h, uint64_t _ID) : used_ranges(), free_ranges() {
      hub = h;
      end_addr = hub->start_vaddr;
      in_use = 0;
      ID = _ID;
   }

   void print() {
      printf("%llu non-continuous bytes used / %llu continuous bytes used\n", in_use, end_addr - hub->start_vaddr);
   }

   void* allocate(uint64_t bytes) {
      if (bytes == 0) return (void*)end_addr;
      if ((hub->round_malloc_to != 0) && (bytes % hub->round_malloc_to != 0)) {
         bytes += hub->round_malloc_to - (bytes % hub->round_malloc_to);
      }
      in_use += bytes;

      Range allocated_range;
      // try to allocate out of existing free address space
      auto it = free_ranges.upper_bound(Range(0, bytes-1));
      if (it != free_ranges.end()) {
         const Range& r = *it;
         // add unused space back to free pool
         if (r.size() != bytes) {
            Range new_r = r;
            new_r.start += bytes;
            free_ranges.insert(new_r);
         }
         allocated_range = Range(r.start, r.start + bytes);
         free_ranges.erase(it);
      } else {
         // fall back to using new virtual addresses
         allocated_range = Range(end_addr, end_addr + bytes);
         end_addr += bytes;
      }
      it = used_ranges.insert(allocated_range).first;
      // request physical backings for new pages
      int64_t first_page = allocated_range.start / hub->page_size;
      int64_t last_page = (allocated_range.end-1) / hub->page_size;
      if (it != used_ranges.begin() && (*std::prev(it)).end > first_page * hub->page_size) ++first_page;
      if (it != used_ranges.end() && std::next(it) != used_ranges.end() && (*std::next(it)).start < (last_page+1) * hub->page_size) --last_page;
      if (first_page <= last_page) {
         hub->pmm->allocate(ID, first_page, last_page - first_page + 1);
      }
      return (void*)allocated_range.start;
   }

   void free(void* a) {
      uint64_t addr = (uint64_t)a;
      auto it = used_ranges.find(Range(addr, addr));
      assert(it != used_ranges.end() && "invalid address to free");
      assert((*it).start == addr && "address freeing error");
      in_use -= (*it).size();
      int64_t first_page = (*it).start / hub->page_size;
      int64_t last_page = ((*it).end-1) / hub->page_size;
      if (it != used_ranges.begin() && (*std::prev(it)).end > first_page * hub->page_size) ++first_page;
      auto it2 = std::next(it);
      if (it != used_ranges.end() && std::next(it) != used_ranges.end() && (*std::next(it)).start < (last_page+1) * hub->page_size) --last_page;

      Range new_free_range = *it;
      if (it != used_ranges.begin() || (*it).start != hub->start_vaddr) {
         uint64_t preceding_free_space;
         if (it != used_ranges.begin()) preceding_free_space = (*it).start - (*std::prev(it,1)).end;
         else preceding_free_space = (*it).start - hub->start_vaddr;
         if (preceding_free_space != 0) {
            // coalesce adjacent regions of free space
            auto its = free_ranges.lower_bound(Range(0, preceding_free_space));
            while (its != free_ranges.end() && its->size() == preceding_free_space) {
               if ((*its).end == (*it).start) {
                  new_free_range.start = (*its).start;
                  free_ranges.erase(its);
                  break;
               }
               ++its;
            }
         }
      }
      if (std::next(it,1) != used_ranges.end()) {
         const uint64_t following_free_space = (*std::next(it,1)).start - (*it).end;
         if (following_free_space != 0) {
            // coalesce adjacent regions of free space
            auto its = free_ranges.lower_bound(Range(0, following_free_space));
            while (its != free_ranges.end() && its->size() == following_free_space) {
               if ((*its).start == (*it).end) {
                  new_free_range.end = (*its).end;
                  free_ranges.erase(its);
                  break;
               }
               ++its;
            }
         }
         free_ranges.insert(new_free_range);
      } else {
         end_addr = new_free_range.start;
      }
      used_ranges.erase(*it);
      // return physical allocation for whole pages
      if (first_page <= last_page) {
         hub->pmm->free(ID, first_page, last_page - first_page + 1);
      }
   }
};

Hub::Hub(uint64_t rmt, uint64_t ps, uint64_t hps, uint64_t tp, uint64_t sv = 0, float tf = 0.0, uint64_t fppf = 1) {
   // tp is total size of memory, in base pages
   // PMM currently rounds that down to the nearest size in huge pages
   round_malloc_to = rmt;
   page_size = ps;
   huge_page_size = hps;
   start_vaddr = sv | (uint64_t{1} << 63);
   pmm = new PMM(this, huge_page_size, tp, tf, fppf);
}

void* Hub::allocate(uint64_t ID, uint64_t bytes) {
   if (vmms[ID] == nullptr) {
      vmms[ID] = new VMM(this, ID);
   }
   uint64_t addr = (ID << 48) | uint64_t(vmms[ID]->allocate(bytes));
   print();
   return (void*)addr;
}

void Hub::free(uint64_t ID, void* addr) {
   assert(vmms[ID] != nullptr && "bad appID");
   const uint64_t mask = (max_uint64 >> 16) | (uint64_t{1} << 63);
   vmms[ID]->free((void*)(uint64_t(addr) & mask));
   print();
}

void Hub::print() {
   for (auto it = vmms.begin(); it != vmms.end(); ++it) {
      printf("Virtual memory stats for ID %llu\n", it->first);
      it->second->print();
   }
   printf("Physical memory stats\n");
   pmm->print();
}

//New translate from Josh's update
void* Hub::translate(uint64_t ID, void* vaddr) {
   const uint64_t mask = (max_uint64 >> 16) | (uint64_t{1} << 63);
   const uint64_t fixed_vaddr = uint64_t(vaddr) & mask;
   const uint64_t vpn = uint64_t(fixed_vaddr) / page_size;
   uint64_t phys_page_addr;
   if (uint64_t(vaddr) & (uint64_t{1} << 63)) {
      phys_page_addr = pmm->translate(ID, vpn);
   } else {
      //printf("translating non-malloc virtual address %p\n", vaddr);
      //if (fixed_vaddr >= start_vaddr) printf("non-malloc vaddr %p in m_dev_malloc range at cycle %llu\n", vaddr, gpu_sim_cycle + gpu_tot_sim_cycle);
      phys_page_addr = pmm->translate(ID, vpn);
      if (phys_page_addr == max_uint64) pmm->special_allocate(ID, vpn / huge_page_size);
      phys_page_addr = pmm->translate(ID, vpn);
      assert(phys_page_addr != max_uint64 && "failed to register non-malloc virtual address");
   }
   return (void*)(phys_page_addr + uint64_t(fixed_vaddr) % page_size);
}

Hub* gpu_alloc;

void gpgpu_t::gpu_malloc_init() {
   uint64_t DRAM_size = 0;
   if(g_mmu == NULL){
// Should not really happen
       printf("GPU alloc tries to use MMU before initialized!!\n");
       exit(0);
//       g_mmu = new mmu();
//       // Need to be able to access mem_config here if we want to initialize mmu now
//       g_mmu->init(g_the_gpu_config.get_mem_config());
   }
   DRAM_size = g_mmu->get_mem_config()->DRAM_size;
   float DRAM_frag = g_mmu->get_mem_config()->DRAM_fragmentation;
   uint64_t DRAM_fppf = g_mmu->get_mem_config()->DRAM_fragmentation_pages_per_frame;
   //if (DRAM_size == 0) DRAM_size = (uint64_t)3 * (uint64_t)1024 * (uint64_t)1024 * (uint64_t)1024; // default to 1GB
   gpu_alloc = new Hub(256, 4096, 512, DRAM_size / 4096, m_dev_malloc, DRAM_frag, DRAM_fppf);
   printf("using DRAM size: %lu\n", DRAM_size);
   printf("using starting DRAM fragmentation: %g\n", DRAM_frag);
   printf("using DRAM fragmentation pages per frame: %lu\n", DRAM_fppf);
   printf("gpu_malloc starting pointer: %p\n", (void*)m_dev_malloc);
}

void* gpgpu_t::gpu_malloc( size_t size )
{
   int appID = thread_to_appID->lookup((void*)pthread_self());
   assert(appID != -1 && "thread not registered");
   void* ptr = gpu_alloc->allocate(appID, size);
   printf("malloc returning pointer %p\n", ptr);
   printf("translate test for VA %p (appID %d): %p\n", ptr, appID, gpu_alloc->translate(appID, ptr));
   return ptr;
}

void* gpgpu_t::gpu_mallocarray( size_t size )
{
   return gpu_malloc(size);
}

// Old version v2
/*
class Allocator {
public:
   virtual void* malloc(size_t size);
};

class Page_Allocator: public Allocator {
   unsigned long long& m_dev_malloc;
   std::map<void*,unsigned> tid_to_idx;
   struct partial_page {
      unsigned free_space;   // useable space left in bytes
      char* end_ptr;   // pointer to end of page
   };
   struct stream {
      unsigned full_pages;   // number of full pages
      std::vector<partial_page> partial_pages;
   };
   std::vector<stream> streams;
   unsigned last_allocation_stream_idx;   // which stream allocated the last page
   bool last_allocation_partial_page;   // is there space left on the last page
   const size_t page_size, data_alignment;

   void add_stream(void* tid) {
      const unsigned stream_idx = streams.size();
      tid_to_idx[tid] = stream_idx;
      streams.push_back(stream());
      streams[stream_idx].full_pages = 0;
      streams[stream_idx].partial_pages = std::vector<partial_page>(0);
   }

   public:
   Page_Allocator(unsigned long long& mdm) : m_dev_malloc(mdm), tid_to_idx(), streams(1), last_allocation_stream_idx(1),
   last_allocation_partial_page(false), page_size(1<<21), data_alignment(256) {
      if (m_dev_malloc%page_size) m_dev_malloc += (page_size - m_dev_malloc%page_size); //align to page size
   }

   void* malloc(size_t size) {
      void* tid = (void*)pthread_self();
      printf("Thread %p calling malloc for %zu bytes\n", tid, size);

      if (tid_to_idx[tid] == 0) add_stream(tid);   // map returns 0 when key not found
      const unsigned stream_idx = tid_to_idx[tid];
      // round up allocations so everything is always aligned
      size_t aligned_size = (size + data_alignment - 1) / data_alignment * data_alignment;

      // if possible, allocate out of existing, partially-filled page
      for(auto it = streams[stream_idx].partial_pages.begin(); it != streams[stream_idx].partial_pages.end(); ++it) {
         partial_page& page = (*it);
         if (page.free_space > aligned_size) {
            char* ptr = page.end_ptr - page.free_space;
            page.free_space -= aligned_size;
            return ptr;
         } else if (page.free_space == aligned_size) {
            char* ptr = page.end_ptr - page.free_space;
            streams[stream_idx].full_pages += 1;
            if (last_allocation_stream_idx == stream_idx && last_allocation_partial_page && (it+1) == streams[stream_idx].partial_pages.end()) {
               last_allocation_partial_page = false;
            }
            streams[stream_idx].partial_pages.erase(it);
            return ptr;
         }
      }

      char* ptr = NULL;
      if (last_allocation_stream_idx == stream_idx && last_allocation_partial_page) {
         // continue on previously-allocated page if no new pages have been allocated since then
         partial_page& page = streams[stream_idx].partial_pages.back();
         ptr = page.end_ptr - page.free_space;
         aligned_size -= page.free_space;
         streams[stream_idx].partial_pages.pop_back();
         streams[stream_idx].full_pages += 1;
      } else {
         ptr = (char*)m_dev_malloc;
      }
      const unsigned new_full_pages = aligned_size / page_size;
      streams[stream_idx].full_pages += new_full_pages;
      m_dev_malloc += page_size * new_full_pages;
      last_allocation_stream_idx = stream_idx;
      if (aligned_size % page_size != 0) {
         m_dev_malloc += page_size;
         partial_page new_partial_page;
         new_partial_page.free_space = page_size - (aligned_size % page_size);
         new_partial_page.end_ptr = (char*)m_dev_malloc;
         streams[stream_idx].partial_pages.push_back(new_partial_page);
         last_allocation_partial_page = true;
      } else {
         last_allocation_partial_page = false;
      }
      return ptr;
   }

   void print() {
      size_t total_pages = 0, total_bloat = 0;
      for (unsigned i = 1; i < streams.size(); ++i) {
         const unsigned full_pages = streams[i].full_pages, partial_pages = streams[i].partial_pages.size();
         printf("stream %u has allocated %u full pages and %u partially-used pages\n", i, full_pages, partial_pages);
         total_pages += full_pages + partial_pages;
         for (unsigned j = 0; j < partial_pages; ++j) {
            total_bloat += streams[i].partial_pages[j].free_space;
         }
      }
      printf("total memory usage: %zu bytes\n", total_pages * page_size);
      printf("total bloat: %zu bytes\n", total_bloat);
   }
};

class Basic_Allocator: public Allocator {
   unsigned long long& m_dev_malloc;
public:
   Basic_Allocator(unsigned long long& mdm) : m_dev_malloc(mdm) {}

   void* malloc(size_t size) {
      unsigned long long result = m_dev_malloc;
      if(g_debug_execution >= 3) {
         printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, m_dev_malloc);
         fflush(stdout);
      }
      m_dev_malloc += size;
      if (size%256) m_dev_malloc += (256 - size%256); //align to 256 byte boundaries
      return(void*) result;
   }
};

Allocator* gpu_alloc;

void gpgpu_t::gpu_malloc_init() {
   gpu_alloc = new Page_Allocator(m_dev_malloc);
   printf("gpu_malloc starting pointer: %p\n", (void*)m_dev_malloc);
}

void* gpgpu_t::gpu_malloc( size_t size )
{
   void* ptr = gpu_alloc->malloc(size);
   printf("malloc returning pointer %p\n", ptr);
   return ptr;
}*/

// Old version
//void* gpgpu_t::gpu_malloc( size_t size )
//{
//   unsigned long long result = m_dev_malloc;
//   if(g_debug_execution >= 3) {
//      printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, m_dev_malloc );
//      fflush(stdout);
//   }
//      printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, m_dev_malloc );
//   m_dev_malloc += size;
//   if (size%256) m_dev_malloc += (256 - size%256); //align to 256 byte boundaries
//   return(void*) result;
//}

//void* gpgpu_t::gpu_mallocarray( size_t size )
//{
//   return gpu_alloc->malloc(size);
//}


// Old version
//void* gpgpu_t::gpu_mallocarray( size_t size )
//{
//   unsigned long long result = m_dev_malloc;
//   if(g_debug_execution >= 3) {
//      printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, m_dev_malloc );
//      fflush(stdout);
//   }
//      printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, m_dev_malloc );
//   m_dev_malloc += size;
//   if (size%256) m_dev_malloc += (256 - size%256); //align to 256 byte boundaries
//   return(void*) result;
//}


void gpgpu_t::memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count , int appID)
{
   if(g_debug_execution >= 3) {
      printf("GPGPU-Sim PTX: copying %zu bytes from CPU[0x%Lx] to GPU[0x%Lx] ... ", count, (unsigned long long) src, (unsigned long long) dst_start_addr );
      fflush(stdout);
   }
   char *src_data = (char*)src;
   for (unsigned n=0; n < count; n ++ )
      m_global_mem->write(dst_start_addr+n,1, src_data+n,NULL,NULL,appID);
   if(g_debug_execution >= 3) {
      printf( " done.\n");
      fflush(stdout);
   }
}

void gpgpu_t::memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count, int appID)
{
   if(g_debug_execution >= 3) {
      printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to CPU[0x%Lx] ...", count, (unsigned long long) src_start_addr, (unsigned long long) dst );
      fflush(stdout);
   }
   unsigned char *dst_data = (unsigned char*)dst;
   for (unsigned n=0; n < count; n ++ )
      m_global_mem->read(src_start_addr+n,1,dst_data+n,appID);
   if(g_debug_execution >= 3) {
      printf( " done.\n");
      fflush(stdout);
   }
}

void gpgpu_t::memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count, int appID)
{
   if(g_debug_execution >= 3) {
      printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to GPU[0x%Lx] ...", count,
          (unsigned long long) src, (unsigned long long) dst );
      fflush(stdout);
   }
   for (unsigned n=0; n < count; n ++ ) {
      unsigned char tmp;
      m_global_mem->read(src+n,1,&tmp,appID);
      m_global_mem->write(dst+n,1, &tmp,NULL,NULL,appID);
   }
   if(g_debug_execution >= 3) {
      printf( " done.\n");
      fflush(stdout);
   }
}

void gpgpu_t::gpu_memset( size_t dst_start_addr, int c, size_t count, int appID)
{
   if(g_debug_execution >= 3) {
      printf("GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at 0x%Lx... ",
          count, (unsigned char) c, (unsigned long long) dst_start_addr );
      fflush(stdout);
   }
   unsigned char c_value = (unsigned char)c;
   for (unsigned n=0; n < count; n ++ )
      m_global_mem->write(dst_start_addr+n,1,&c_value,NULL,NULL,appID);
   if(g_debug_execution >= 3) {
      printf( " done.\n");
      fflush(stdout);
   }
}

void ptx_print_insn( address_type pc, FILE *fp )
{
   std::map<unsigned,function_info*>::iterator f = g_pc_to_finfo.find(pc);
   if( f == g_pc_to_finfo.end() ) {
       fprintf(fp,"<no instruction at address 0x%x>", pc );
       return;
   }
   function_info *finfo = f->second;
   assert( finfo );
   finfo->print_insn(pc,fp);
}

std::string ptx_get_insn_str( address_type pc )
{
   std::map<unsigned,function_info*>::iterator f = g_pc_to_finfo.find(pc);
   if( f == g_pc_to_finfo.end() ) {
       #define STR_SIZE 255
       char buff[STR_SIZE];
       buff[STR_SIZE - 1] = '\0';
       snprintf(buff, STR_SIZE,"<no instruction at address 0x%x>", pc );
       return std::string(buff);
   }
   function_info *finfo = f->second;
   assert( finfo );
   return finfo->get_insn_str(pc);
}

void ptx_instruction::set_fp_or_int_archop(){
    oprnd_type=UN_OP;
	if((m_opcode == MEMBAR_OP)||(m_opcode == SSY_OP )||(m_opcode == BRA_OP) || (m_opcode == BAR_OP) || (m_opcode == RET_OP) || (m_opcode == RETP_OP) || (m_opcode == NOP_OP) || (m_opcode == EXIT_OP) || (m_opcode == CALLP_OP) || (m_opcode == CALL_OP)){
			// do nothing
	}else if((m_opcode == CVT_OP || m_opcode == SET_OP || m_opcode == SLCT_OP)){
		if(get_type2()==F16_TYPE || get_type2()==F32_TYPE || get_type2() == F64_TYPE || get_type2() == FF64_TYPE){
		    oprnd_type= FP_OP;
		}else oprnd_type=INT_OP;

	}else{
		if(get_type()==F16_TYPE || get_type()==F32_TYPE || get_type() == F64_TYPE || get_type() == FF64_TYPE){
		    oprnd_type= FP_OP;
		}else oprnd_type=INT_OP;
	}
}
void ptx_instruction::set_mul_div_or_other_archop(){
    sp_op=OTHER_OP;
	if((m_opcode != MEMBAR_OP) && (m_opcode != SSY_OP) && (m_opcode != BRA_OP) && (m_opcode != BAR_OP) && (m_opcode != EXIT_OP) && (m_opcode != NOP_OP) && (m_opcode != RETP_OP) && (m_opcode != RET_OP) && (m_opcode != CALLP_OP) && (m_opcode != CALL_OP)){
		if(get_type()==F32_TYPE || get_type() == F64_TYPE || get_type() == FF64_TYPE){
			switch(get_opcode()){
				case MUL_OP:
				case MAD_OP:
				    sp_op=FP_MUL_OP;
					break;
				case DIV_OP:
				    sp_op=FP_DIV_OP;
					break;
				case LG2_OP:
				    sp_op=FP_LG_OP;
					break;
				case RSQRT_OP:
				case SQRT_OP:
				    sp_op=FP_SQRT_OP;
					break;
				case RCP_OP:
				    sp_op=FP_DIV_OP;
					break;
				case SIN_OP:
				case COS_OP:
				    sp_op=FP_SIN_OP;
					break;
				case EX2_OP:
				    sp_op=FP_EXP_OP;
					break;
				default:
					if(op==ALU_OP)
					    sp_op=FP__OP;
					break;

			}
		}else {
			switch(get_opcode()){
				case MUL24_OP:
				case MAD24_OP:
				    sp_op=INT_MUL24_OP;
				break;
				case MUL_OP:
				case MAD_OP:
					if(get_type()==U32_TYPE || get_type()==S32_TYPE || get_type()==B32_TYPE)
					    sp_op=INT_MUL32_OP;
					else
					    sp_op=INT_MUL_OP;
				break;
				case DIV_OP:
				    sp_op=INT_DIV_OP;
				break;
				default:
					if(op==ALU_OP)
					    sp_op=INT__OP;
					break;
			}
		}
	}

}
void ptx_instruction::set_opcode_and_latency()
{
	unsigned int_latency[5];
	unsigned fp_latency[5];
	unsigned dp_latency[5];
	unsigned int_init[5];
	unsigned fp_init[5];
	unsigned dp_init[5];
	/*
	 * [0] ADD,SUB
	 * [1] MAX,Min
	 * [2] MUL
	 * [3] MAD
	 * [4] DIV
	 */
	sscanf(opcode_latency_int, "%u,%u,%u,%u,%u",
			&int_latency[0],&int_latency[1],&int_latency[2],
			&int_latency[3],&int_latency[4]);
	sscanf(opcode_latency_fp, "%u,%u,%u,%u,%u",
			&fp_latency[0],&fp_latency[1],&fp_latency[2],
			&fp_latency[3],&fp_latency[4]);
	sscanf(opcode_latency_dp, "%u,%u,%u,%u,%u",
			&dp_latency[0],&dp_latency[1],&dp_latency[2],
			&dp_latency[3],&dp_latency[4]);
	sscanf(opcode_initiation_int, "%u,%u,%u,%u,%u",
			&int_init[0],&int_init[1],&int_init[2],
			&int_init[3],&int_init[4]);
	sscanf(opcode_initiation_fp, "%u,%u,%u,%u,%u",
			&fp_init[0],&fp_init[1],&fp_init[2],
			&fp_init[3],&fp_init[4]);
	sscanf(opcode_initiation_dp, "%u,%u,%u,%u,%u",
			&dp_init[0],&dp_init[1],&dp_init[2],
			&dp_init[3],&dp_init[4]);

	if(!m_operands.empty()){
		std::vector<operand_info>::iterator it;
	   	for(it=++m_operands.begin();it!=m_operands.end();it++){
	   		num_operands++;
	   		if((it->is_reg() || it->is_vector())){
	   			   num_regs++;
	   		}
	   	 }
	}
   op = ALU_OP;
   mem_op= NOT_TEX;
   initiation_interval = latency = 1;
   switch( m_opcode ) {
   case MOV_OP:
       assert( !(has_memory_read() && has_memory_write()) );
       if ( has_memory_read() ) op = LOAD_OP;
       if ( has_memory_write() ) op = STORE_OP;
       break;
   case LD_OP: op = LOAD_OP; break;
   case LDU_OP: op = LOAD_OP; break;
   case ST_OP: op = STORE_OP; break;
   case BRA_OP: op = BRANCH_OP; break;
   case BREAKADDR_OP: op = BRANCH_OP; break;
   case TEX_OP: op = LOAD_OP; mem_op=TEX; break;
   case ATOM_OP: op = LOAD_OP; break;
   case BAR_OP: op = BARRIER_OP; break;
   case MEMBAR_OP: op = MEMORY_BARRIER_OP; break;
   case CALL_OP:
   {
       if(m_is_printf)
           op = ALU_OP;
       else
           op = CALL_OPS;
       break;
   }
   case CALLP_OP:
   {
       if(m_is_printf)
               op = ALU_OP;
           else
               op = CALL_OPS;
           break;
   }
   case RET_OP: case RETP_OP:  op = RET_OPS;break;
   case ADD_OP: case ADDP_OP: case ADDC_OP: case SUB_OP: case SUBC_OP:
	   //ADD,SUB latency
	   switch(get_type()){
	   case F32_TYPE:
		   latency = fp_latency[0];
		   initiation_interval = fp_init[0];
		   break;
	   case F64_TYPE:
	   case FF64_TYPE:
		   latency = dp_latency[0];
		   initiation_interval = dp_init[0];
		   break;
	   case B32_TYPE:
	   case U32_TYPE:
	   case S32_TYPE:
	   default: //Use int settings for default
		   latency = int_latency[0];
		   initiation_interval = int_init[0];
		   break;
	   }
	   break;
   case MAX_OP: case MIN_OP:
	   //MAX,MIN latency
	   switch(get_type()){
	   case F32_TYPE:
		   latency = fp_latency[1];
		   initiation_interval = fp_init[1];
		   break;
	   case F64_TYPE:
	   case FF64_TYPE:
		   latency = dp_latency[1];
		   initiation_interval = dp_init[1];
		   break;
	   case B32_TYPE:
	   case U32_TYPE:
	   case S32_TYPE:
	   default: //Use int settings for default
		   latency = int_latency[1];
		   initiation_interval = int_init[1];
		   break;
	   }
	   break;
   case MUL_OP:
	   //MUL latency
	   switch(get_type()){
	   case F32_TYPE:
		   latency = fp_latency[2];
		   initiation_interval = fp_init[2];
		   op = ALU_SFU_OP;
		   break;
	   case F64_TYPE:
	   case FF64_TYPE:
		   latency = dp_latency[2];
		   initiation_interval = dp_init[2];
		   op = ALU_SFU_OP;
		   break;
	   case B32_TYPE:
	   case U32_TYPE:
	   case S32_TYPE:
	   default: //Use int settings for default
		   latency = int_latency[2];
		   initiation_interval = int_init[2];
		   op = SFU_OP;
		   break;
	   }
	   break;
   case MAD_OP: case MADP_OP:
	   //MAD latency
	   switch(get_type()){
	   case F32_TYPE:
		   latency = fp_latency[3];
		   initiation_interval = fp_init[3];
		   break;
	   case F64_TYPE:
	   case FF64_TYPE:
		   latency = dp_latency[3];
		   initiation_interval = dp_init[3];
		   break;
	   case B32_TYPE:
	   case U32_TYPE:
	   case S32_TYPE:
	   default: //Use int settings for default
		   latency = int_latency[3];
		   initiation_interval = int_init[3];
		   op = SFU_OP;
		   break;
	   }
	   break;
   case DIV_OP:
	   // Floating point only
	   op = SFU_OP;
	   switch(get_type()){
	   case F32_TYPE:
		   latency = fp_latency[4];
		   initiation_interval = fp_init[4];
		   break;
	   case F64_TYPE:
	   case FF64_TYPE:
		   latency = dp_latency[4];
		   initiation_interval = dp_init[4];
		   break;
	   case B32_TYPE:
	   case U32_TYPE:
	   case S32_TYPE:
	   default: //Use int settings for default
		   latency = int_latency[4];
		   initiation_interval = int_init[4];
		   break;
	   }
	   break;
   case SQRT_OP: case SIN_OP: case COS_OP: case EX2_OP: case LG2_OP: case RSQRT_OP: case RCP_OP:
	   //Using double to approximate those
	  latency = dp_latency[2];
	  initiation_interval = dp_init[2];
      op = SFU_OP;
      break;
   default:
       break;
   }
	set_fp_or_int_archop();
	set_mul_div_or_other_archop();

}

void ptx_thread_info::ptx_fetch_inst( inst_t &inst ) const
{
   addr_t pc = get_pc();
   const ptx_instruction *pI = m_func_info->get_instruction(pc);
   inst = (const inst_t&)*pI;
   assert( inst.valid() );
}

static unsigned datatype2size( unsigned data_type )
{
   unsigned data_size;
   switch ( data_type ) {
      case B8_TYPE:
      case S8_TYPE:
      case U8_TYPE:
         data_size = 1; break;
      case B16_TYPE:
      case S16_TYPE:
      case U16_TYPE:
      case F16_TYPE:
         data_size = 2; break;
      case B32_TYPE:
      case S32_TYPE:
      case U32_TYPE:
      case F32_TYPE:
         data_size = 4; break;
      case B64_TYPE:
      case BB64_TYPE:
      case S64_TYPE:
      case U64_TYPE:
      case F64_TYPE:
      case FF64_TYPE:
         data_size = 8; break;
      case BB128_TYPE:
         data_size = 16; break;
      default: assert(0); break;
   }
   return data_size;
}

void ptx_instruction::pre_decode()
{
   pc = m_PC;
   isize = m_inst_size;
   for( unsigned i=0; i<4; i++) {
       out[i] = 0;
       in[i] = 0;
   }
   is_vectorin = 0;
   is_vectorout = 0;
   std::fill_n(arch_reg.src, MAX_REG_OPERANDS, -1);
   std::fill_n(arch_reg.dst, MAX_REG_OPERANDS, -1);
   pred = 0;
   ar1 = 0;
   ar2 = 0;
   space = m_space_spec;
   memory_op = no_memory_op;
   data_size = 0;
   if ( has_memory_read() || has_memory_write() ) {
      unsigned to_type = get_type();
      data_size = datatype2size(to_type);
      memory_op = has_memory_read() ? memory_load : memory_store;
   }

   bool has_dst = false ;

   switch ( get_opcode() ) {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) case OP: has_dst = (DST!=0); break;
#include "opcodes.def"
#undef OP_DEF
   default:
      printf( "Execution error: Invalid opcode (0x%x)\n", get_opcode() );
      break;
   }

   switch( m_cache_option ) {
   case CA_OPTION: cache_op = CACHE_ALL; break;
   case CG_OPTION: cache_op = CACHE_GLOBAL; break;
   case CS_OPTION: cache_op = CACHE_STREAMING; break;
   case LU_OPTION: cache_op = CACHE_LAST_USE; break;
   case CV_OPTION: cache_op = CACHE_VOLATILE; break;
   case WB_OPTION: cache_op = CACHE_WRITE_BACK; break;
   case WT_OPTION: cache_op = CACHE_WRITE_THROUGH; break;
   default:
      if( m_opcode == LD_OP || m_opcode == LDU_OP )
         cache_op = CACHE_ALL;
      else if( m_opcode == ST_OP )
         cache_op = CACHE_WRITE_BACK;
      else if( m_opcode == ATOM_OP )
         cache_op = CACHE_GLOBAL;
      break;
   }

   set_opcode_and_latency();

   // Get register operands
   int n=0,m=0;
   ptx_instruction::const_iterator opr=op_iter_begin();
   for ( ; opr != op_iter_end(); opr++, n++ ) { //process operands
      const operand_info &o = *opr;
      if ( has_dst && n==0 ) {
         // Do not set the null register "_" as an architectural register
         if ( o.is_reg() && !o.is_non_arch_reg() ) {
            out[0] = o.reg_num();
            arch_reg.dst[0] = o.arch_reg_num();
         } else if ( o.is_vector() ) {
            is_vectorin = 1;
            unsigned num_elem = o.get_vect_nelem();
            if( num_elem >= 1 ) out[0] = o.reg1_num();
            if( num_elem >= 2 ) out[1] = o.reg2_num();
            if( num_elem >= 3 ) out[2] = o.reg3_num();
            if( num_elem >= 4 ) out[3] = o.reg4_num();
            for (int i = 0; i < num_elem; i++)
               arch_reg.dst[i] = o.arch_reg_num(i);
         }
      } else {
         if ( o.is_reg() && !o.is_non_arch_reg() ) {
            int reg_num = o.reg_num();
            arch_reg.src[m] = o.arch_reg_num();
            switch ( m ) {
            case 0: in[0] = reg_num; break;
            case 1: in[1] = reg_num; break;
            case 2: in[2] = reg_num; break;
            default: break;
            }
            m++;
         } else if ( o.is_vector() ) {
            //assert(m == 0); //only support 1 vector operand (for textures) right now
            is_vectorout = 1;
            unsigned num_elem = o.get_vect_nelem();
            if( num_elem >= 1 ) in[0] = o.reg1_num();
            if( num_elem >= 2 ) in[1] = o.reg2_num();
            if( num_elem >= 3 ) in[2] = o.reg3_num();
            if( num_elem >= 4 ) in[3] = o.reg4_num();
            for (int i = 0; i < num_elem; i++)
               arch_reg.src[i] = o.arch_reg_num(i);
            m+=4;
         }
      }
   }

   // Get predicate
   if(has_pred()) {
	   const operand_info &p = get_pred();
	   pred = p.reg_num();
   }

   // Get address registers inside memory operands.
   // Assuming only one memory operand per instruction,
   //  and maximum of two address registers for one memory operand.
   if( has_memory_read() || has_memory_write() ) {
      ptx_instruction::const_iterator op=op_iter_begin();
      for ( ; op != op_iter_end(); op++, n++ ) { //process operands
         const operand_info &o = *op;

         if(o.is_memory_operand()) {
             // We do not support the null register as a memory operand
             assert( !o.is_non_arch_reg() );

            // Check PTXPlus-type operand
            // memory operand with addressing (ex. s[0x4] or g[$r1])
            if(o.is_memory_operand2()) {

               // memory operand with one address register (ex. g[$r1+0x4] or s[$r2+=0x4])
               if(o.get_double_operand_type() == 0 || o.get_double_operand_type() == 3){
                  ar1 = o.reg_num();
                  arch_reg.src[4] = o.arch_reg_num();
               }
               // memory operand with two address register (ex. s[$r1+$r1] or g[$r1+=$r2])
               else if(o.get_double_operand_type() == 1 || o.get_double_operand_type() == 2) {
                  ar1 = o.reg1_num();
                  arch_reg.src[4] = o.arch_reg_num();
                  ar2 = o.reg2_num();
                  arch_reg.src[5] = o.arch_reg_num();
               }
            }
            else if(o.is_immediate_address()){

            }
            // Regular PTX operand
            else if (o.get_symbol()->type()->get_key().is_reg()) { // Memory operand contains a register
              ar1 = o.reg_num();
              arch_reg.src[4] = o.arch_reg_num();
            }

         }
      }
   }

   // get reconvergence pc
   reconvergence_pc = get_converge_point(pc);

   m_decoded=true;
}

void function_info::add_param_name_type_size( unsigned index, std::string name, int type, size_t size, bool ptr, memory_space_t space )
{
   unsigned parsed_index;
   char buffer[2048];
   snprintf(buffer,2048,"%s_param_%%u", m_name.c_str() );
   int ntokens = sscanf(name.c_str(),buffer,&parsed_index);
   if( ntokens == 1 ) {
      assert( m_ptx_kernel_param_info.find(parsed_index) == m_ptx_kernel_param_info.end() );
      m_ptx_kernel_param_info[parsed_index] = param_info(name, type, size, ptr, space);
   } else {
      assert( m_ptx_kernel_param_info.find(index) == m_ptx_kernel_param_info.end() );
      m_ptx_kernel_param_info[index] = param_info(name, type, size, ptr, space);
   }
}

void function_info::add_param_data( unsigned argn, struct gpgpu_ptx_sim_arg *args )
{
   const void *data = args->m_start;

   bool scratchpad_memory_param = false; // Is this parameter in CUDA shared memory or OpenCL local memory

   std::map<unsigned,param_info>::iterator i=m_ptx_kernel_param_info.find(argn);
   if( i != m_ptx_kernel_param_info.end() ) {
      if (i->second.is_ptr_shared()) {
         assert(args->m_start == NULL && "OpenCL parameter pointer to local memory must have NULL as value");
         scratchpad_memory_param = true;
      } else {
         param_t tmp;
         tmp.pdata = args->m_start;
         tmp.size = args->m_nbytes;
         tmp.offset = args->m_offset;
         tmp.type = 0;
         i->second.add_data(tmp);
         i->second.add_offset((unsigned) args->m_offset);
      }
   } else {
      scratchpad_memory_param = true;
   }

   if (scratchpad_memory_param) {
      // This should only happen for OpenCL:
      //
      // The LLVM PTX compiler in NVIDIA's driver (version 190.29)
      // does not generate an argument in the function declaration
      // for __constant arguments.
      //
      // The associated constant memory space can be allocated in two
      // ways. It can be explicitly initialized in the .ptx file where
      // it is declared.  Or, it can be allocated using the clCreateBuffer
      // on the host. In this later case, the .ptx file will contain
      // a global declaration of the parameter, but it will have an unknown
      // array size.  Thus, the symbol's address will not be set and we need
      // to set it here before executing the PTX.

      char buffer[2048];
      snprintf(buffer,2048,"%s_param_%u",m_name.c_str(),argn);

      symbol *p = m_symtab->lookup(buffer);
      if( p == NULL ) {
         printf("GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot bind buffer\n", buffer);
         abort();
      }
      if( data )
         p->set_address((addr_t)*(size_t*)data);
      else {
         // clSetKernelArg was passed NULL pointer for data...
         // this is used for dynamically sized shared memory on NVIDIA platforms
         bool is_ptr_shared = false;
         if( i != m_ptx_kernel_param_info.end() ) {
            is_ptr_shared = i->second.is_ptr_shared();
         }

         if( !is_ptr_shared and !p->is_shared() ) {
            printf("GPGPU-Sim PTX: ERROR ** clSetKernelArg passed NULL but arg not shared memory\n");
            abort();
         }
         unsigned num_bits = 8*args->m_nbytes;
         printf("GPGPU-Sim PTX: deferred allocation of shared region for \"%s\" from 0x%x to 0x%x (shared memory space)\n",
                p->name().c_str(),
                m_symtab->get_shared_next(),
                m_symtab->get_shared_next() + num_bits/8 );
         fflush(stdout);
         assert( (num_bits%8) == 0  );
         addr_t addr = m_symtab->get_shared_next();
         addr_t addr_pad = num_bits ? (((num_bits/8) - (addr % (num_bits/8))) % (num_bits/8)) : 0;
         p->set_address( addr+addr_pad );
         m_symtab->alloc_shared( num_bits/8 + addr_pad );
      }
   }
}

void function_info::finalize( memory_space *param_mem )
{
   unsigned param_address = 0;
   for( std::map<unsigned,param_info>::iterator i=m_ptx_kernel_param_info.begin(); i!=m_ptx_kernel_param_info.end(); i++ ) {
      param_info &p = i->second;
      if (p.is_ptr_shared()) continue; // Pointer to local memory: Should we pass the allocated shared memory address to the param memory space?
      std::string name = p.get_name();
      int type = p.get_type();
      param_t param_value = p.get_value();
      param_value.type = type;
      symbol *param = m_symtab->lookup(name.c_str());
      unsigned xtype = param->type()->get_key().scalar_type();
      assert(xtype==(unsigned)type);
      size_t size;
      size = param_value.size; // size of param in bytes
      // assert(param_value.offset == param_address);
      if( size != p.get_size() / 8) {
         printf("GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. formal size = %zu (using smaller of two)\n",
                size, p.get_size()/8);
         size = (size<(p.get_size()/8))?size:(p.get_size()/8);
      }
      // copy the parameter over word-by-word so that parameter that crosses a memory page can be copied over
      const size_t word_size = 4;
      for (size_t idx = 0; idx < size; idx += word_size) {
         const char *pdata = reinterpret_cast<const char*>(param_value.pdata) + idx; // cast to char * for ptr arithmetic
         param_mem->write(param_address + idx, word_size, pdata,NULL,NULL, 0);
      }
      param->set_address(param_address);
      param_address += size;
   }
}

void function_info::param_to_shared( memory_space *shared_mem, symbol_table *symtab )
{
   extern gpgpu_sim* g_the_gpu;
   if (not g_the_gpu->get_config().convert_to_ptxplus()) return;

   // copies parameters into simulated shared memory
   for( std::map<unsigned,param_info>::iterator i=m_ptx_kernel_param_info.begin(); i!=m_ptx_kernel_param_info.end(); i++ ) {
      param_info &p = i->second;
      if (p.is_ptr_shared()) continue; // Pointer to local memory: Should we pass the allocated shared memory address to the param memory space?
      std::string name = p.get_name();
      int type = p.get_type();
      param_t value = p.get_value();
      value.type = type;
      symbol *param = symtab->lookup(name.c_str());
      unsigned xtype = param->type()->get_key().scalar_type();
      assert(xtype==(unsigned)type);

      int tmp;
      size_t size;
      unsigned offset = p.get_offset();
      type_info_key::type_decode(xtype,size,tmp);

      // Write to shared memory - offset + 0x10
      shared_mem->write(offset+0x10,size/8,value.pdata,NULL,NULL,/*fixme add AppID instead of 0*/0);
   }
}


void function_info::list_param( FILE *fout ) const
{
   for( std::map<unsigned,param_info>::const_iterator i=m_ptx_kernel_param_info.begin(); i!=m_ptx_kernel_param_info.end(); i++ ) {
      const param_info &p = i->second;
      std::string name = p.get_name();
      symbol *param = m_symtab->lookup(name.c_str());
      addr_t param_addr = param->get_address();
      fprintf(fout, "%s: %#08x\n", name.c_str(), param_addr);
   }
   fflush(fout);
}

template<int activate_level>
bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc)
{
   if (g_debug_execution >= activate_level) {
      // check each type of debug dump constraint to filter out dumps
      if ( (g_debug_thread_uid != 0) && (thd_uid != (unsigned)g_debug_thread_uid) ) {
         return false;
      }
      if ( (g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc) ) {
         return false;
      }

      return true;
   }

   return false;
}

void init_inst_classification_stat()
{
   static std::set<unsigned> init;
   if( init.find(g_ptx_kernel_count) != init.end() )
      return;
   init.insert(g_ptx_kernel_count);

   #define MAX_CLASS_KER 1024
   char kernelname[MAX_CLASS_KER] ="";
   if (!g_inst_classification_stat) g_inst_classification_stat = (void**)calloc(MAX_CLASS_KER, sizeof(void*));
   snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",g_ptx_kernel_count  );
   assert( g_ptx_kernel_count < MAX_CLASS_KER ) ; // a static limit on number of kernels increase it if it fails!
   g_inst_classification_stat[g_ptx_kernel_count] = StatCreate(kernelname,1,20);
   if (!g_inst_op_classification_stat) g_inst_op_classification_stat = (void**)calloc(MAX_CLASS_KER, sizeof(void*));
   snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",g_ptx_kernel_count  );
   g_inst_op_classification_stat[g_ptx_kernel_count] = StatCreate(kernelname,1,100);
}

static unsigned get_tex_datasize( const ptx_instruction *pI, ptx_thread_info *thread )
{
   const operand_info &src1 = pI->src1(); //the name of the texture
   std::string texname = src1.name();

   gpgpu_t *gpu = thread->get_gpu();
   const struct textureReference* texref = gpu->get_texref(texname);
   const struct textureInfo* texInfo = gpu->get_texinfo(texref);

   unsigned data_size = texInfo->texel_size;
   return data_size;
}

void ptx_thread_info::ptx_exec_inst( warp_inst_t &inst, unsigned lane_id)
{

   bool skip = false;
   int op_classification = 0;
   addr_t pc = next_instr();
   assert( pc == inst.pc ); // make sure timing model and functional model are in sync
   const ptx_instruction *pI = m_func_info->get_instruction(pc);
   set_npc( pc + pI->inst_size() );


   try {

   clearRPC();
   m_last_set_operand_value.u64 = 0;

   if(is_done())
   {
      printf("attempted to execute instruction on a thread that is already done.\n");
      assert(0);
   }

   if ( g_debug_execution >= 6 || m_gpu->get_config().get_ptx_inst_debug_to_file()) {
      if ( (g_debug_thread_uid==0) || (get_uid() == (unsigned)g_debug_thread_uid) ) {

          clear_modifiedregs();
         enable_debug_trace();
      }
   }


   if( pI->has_pred() ) {
      const operand_info &pred = pI->get_pred();
      ptx_reg_t pred_value = get_operand_value(pred, pred, PRED_TYPE, this, 0);
      if(pI->get_pred_mod() == -1) {
            skip = (pred_value.pred & 0x0001) ^ pI->get_pred_neg(); //ptxplus inverts the zero flag
      } else {
            skip = !pred_lookup(pI->get_pred_mod(), pred_value.pred & 0x000F);
      }
   }

   if( skip ) {
      inst.set_not_active(lane_id);
   } else {
      const ptx_instruction *pI_saved = pI;
      ptx_instruction *pJ = NULL;
      if( pI->get_opcode() == VOTE_OP ) {
         pJ = new ptx_instruction(*pI);
         *((warp_inst_t*)pJ) = inst; // copy active mask information
         pI = pJ;
      }
      switch ( pI->get_opcode() ) {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) case OP: FUNC(pI,this); op_classification = CLASSIFICATION; break;
#include "opcodes.def"
#undef OP_DEF
      default: printf( "Execution error: Invalid opcode (0x%x)\n", pI->get_opcode() ); break;
      }
      delete pJ;
      pI = pI_saved;

      // Run exit instruction if exit option included
      if(pI->is_exit())
         exit_impl(pI,this);
   }



   const gpgpu_functional_sim_config &config = m_gpu->get_config();

   // Output instruction information to file and stdout
   if( config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 || config.get_ptx_inst_debug_thread_uid() == get_uid()) ) {
      fprintf(m_gpu->get_ptx_inst_debug_file(),
             "[thd=%u] : (%s:%u - %s)\n",
             get_uid(),
             pI->source_file(), pI->source_line(), pI->get_source() );
      //fprintf(ptx_inst_debug_file, "has memory read=%d, has memory write=%d\n", pI->has_memory_read(), pI->has_memory_write());
      fflush(m_gpu->get_ptx_inst_debug_file());
   }

   if ( ptx_debug_exec_dump_cond<5>(get_uid(), pc) ) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      printf("%u [thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u [pc=%u] (%s:%u - %s)  [0x%llx]\n",
             g_ptx_sim_num_insn,
             get_uid(),
             pI->uid(), ctaid.x,ctaid.y,ctaid.z,tid.x,tid.y,tid.z,
             get_icount(),
             pc, pI->source_file(), pI->source_line(), pI->get_source(),
             m_last_set_operand_value.u64 );
      fflush(stdout);
   }

   addr_t insn_memaddr = 0xFEEBDAED;
   memory_space_t insn_space = undefined_space;
   _memory_op_t insn_memory_op = no_memory_op;

   unsigned insn_data_size = 0;
   if ( (pI->has_memory_read()  || pI->has_memory_write()) ) {
      insn_memaddr = last_eaddr();
      insn_space = last_space();
      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
      insn_memory_op = pI->has_memory_read() ? memory_load : memory_store;
   }

   if ( pI->get_opcode() == ATOM_OP ) {
      insn_memaddr = last_eaddr();
      insn_space = last_space();
      inst.add_callback( lane_id, last_callback().function, last_callback().instruction, this );
      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
   }

   if (pI->get_opcode() == TEX_OP) {
      inst.set_addr(lane_id, last_eaddr() );
      assert( inst.space == last_space() );
      insn_data_size = get_tex_datasize(pI, this); // texture obtain its data granularity from the texture info
   }

   // Output register information to file and stdout
   if( config.get_ptx_inst_debug_to_file()!=0 &&
       (config.get_ptx_inst_debug_thread_uid()==0||config.get_ptx_inst_debug_thread_uid()==get_uid()) ) {
      dump_modifiedregs(m_gpu->get_ptx_inst_debug_file());
      dump_regs(m_gpu->get_ptx_inst_debug_file());
   }

   if ( g_debug_execution >= 6 ) {
      if ( ptx_debug_exec_dump_cond<6>(get_uid(), pc) )
         dump_modifiedregs(stdout);
   }
   if ( g_debug_execution >= 10 ) {
      if ( ptx_debug_exec_dump_cond<10>(get_uid(), pc) )
         dump_regs(stdout);
   }
   update_pc();
   g_ptx_sim_num_insn++;

   //not using it with functional simulation mode
   if(!(this->m_functionalSimulationMode))
       ptx_file_line_stats_add_exec_count(pI);

   if ( gpgpu_ptx_instruction_classification ) {
      init_inst_classification_stat();
      unsigned space_type=0;
      switch ( pI->get_space().get_type() ) {
      case global_space: space_type = 10; break;
      case local_space:  space_type = 11; break;
      case tex_space:    space_type = 12; break;
      case surf_space:   space_type = 13; break;
      case param_space_kernel:
      case param_space_local:
                         space_type = 14; break;
      case shared_space: space_type = 15; break;
      case const_space:  space_type = 16; break;
      default:
         space_type = 0 ;
         break;
      }
      StatAddSample( g_inst_classification_stat[g_ptx_kernel_count],  op_classification);
      if (space_type) StatAddSample( g_inst_classification_stat[g_ptx_kernel_count], ( int )space_type);
      StatAddSample( g_inst_op_classification_stat[g_ptx_kernel_count], (int)  pI->get_opcode() );
   }
   if ( (g_ptx_sim_num_insn % 100000) == 0 ) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      printf("GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) tid=(%u,%u,%u)\n",
             g_ptx_sim_num_insn, ctaid.x,ctaid.y,ctaid.z,tid.x,tid.y,tid.z );
      fflush(stdout);
   }

   // "Return values"
   if(!skip) {
      inst.space = insn_space;
      inst.set_addr(lane_id, insn_memaddr);
      inst.data_size = insn_data_size; // simpleAtomicIntrinsics
      assert( inst.memory_op == insn_memory_op );
   }

   } catch ( int x  ) {
      printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x, pI->source_file(), pI->source_line() );
      printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source() );
      abort();
   }

}

void set_param_gpgpu_num_shaders(int num_shaders)
{
   gpgpu_param_num_shaders = num_shaders;
}

const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info(const function_info *kernel)
{
   return kernel->get_kernel_info();
}

const warp_inst_t *ptx_fetch_inst( address_type pc )
{
    return function_info::pc_to_instruction(pc);
}

unsigned ptx_sim_init_thread( kernel_info_t &kernel,
                              ptx_thread_info** thread_info,
                              int sid,
                              unsigned tid,
                              unsigned threads_left,
                              unsigned num_threads,
                              core_t *core,
                              unsigned hw_cta_id,
                              unsigned hw_warp_id,
                              gpgpu_t *gpu,
                              bool isInFunctionalSimulationMode)
{
   std::list<ptx_thread_info *> &active_threads = kernel.active_threads();

   static std::map<unsigned,memory_space*> shared_memory_lookup;
   static std::map<unsigned,ptx_cta_info*> ptx_cta_lookup;
   static std::map<unsigned,std::map<unsigned,memory_space*> > local_memory_lookup;

   if ( *thread_info != NULL ) {
      ptx_thread_info *thd = *thread_info;
      assert( thd->is_done() );
      if ( g_debug_execution==-1 ) {
         dim3 ctaid = thd->get_ctaid();
         dim3 t = thd->get_tid();
         printf("GPGPU-Sim PTX simulator:  thread exiting ctaid=(%u,%u,%u) tid=(%u,%u,%u) uid=%u\n",
                ctaid.x,ctaid.y,ctaid.z,t.x,t.y,t.z, thd->get_uid() );
         fflush(stdout);
      }
      thd->m_cta_info->register_deleted_thread(thd);
      delete thd;
      *thread_info = NULL;
   }

   if ( !active_threads.empty() ) {
      assert( active_threads.size() <= threads_left );
      ptx_thread_info *thd = active_threads.front();
      active_threads.pop_front();
      *thread_info = thd;
      thd->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid, isInFunctionalSimulationMode ,0);
      return 1;
   }

   if ( kernel.no_more_ctas_to_run() ) {
      return 0; //finished!
   }

   if ( threads_left < kernel.threads_per_cta() ) {
      return 0;
   }

   if ( g_debug_execution==-1 ) {
      printf("GPGPU-Sim PTX simulator:  STARTING THREAD ALLOCATION --> \n");
      fflush(stdout);
   }

   //initializing new CTA
   ptx_cta_info *cta_info = NULL;
   memory_space *shared_mem = NULL;

   unsigned cta_size = kernel.threads_per_cta();
   unsigned max_cta_per_sm = num_threads/cta_size; // e.g., 256 / 48 = 5
   assert( max_cta_per_sm > 0 );

   unsigned sm_idx = (tid/cta_size)*gpgpu_param_num_shaders + sid;

   if ( shared_memory_lookup.find(sm_idx) == shared_memory_lookup.end() ) {
      if ( g_debug_execution >= 1 ) {
         printf("  <CTA alloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n",
                sm_idx, sid, max_cta_per_sm );
      }
      char buf[512];
      snprintf(buf,512,"shared_%u", sid);
      shared_mem = new memory_space_impl<16*1024>(buf,4,shared_space);
      shared_memory_lookup[sm_idx] = shared_mem;
      cta_info = new ptx_cta_info(sm_idx);
      ptx_cta_lookup[sm_idx] = cta_info;
   } else {
      if ( g_debug_execution >= 1 ) {
         printf("  <CTA realloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n",
                sm_idx, sid, max_cta_per_sm );
      }
      shared_mem = shared_memory_lookup[sm_idx];
      cta_info = ptx_cta_lookup[sm_idx];
      cta_info->check_cta_thread_status_and_reset();
   }

   std::map<unsigned,memory_space*> &local_mem_lookup = local_memory_lookup[sid];
   while( kernel.more_threads_in_cta() ) {
      dim3 ctaid3d = kernel.get_next_cta_id();
      unsigned new_tid = kernel.get_next_thread_id();
      dim3 tid3d = kernel.get_next_thread_id_3d();
      kernel.increment_thread_id();
      new_tid += tid;
      ptx_thread_info *thd = new ptx_thread_info(kernel);

      memory_space *local_mem = NULL;
      std::map<unsigned,memory_space*>::iterator l = local_mem_lookup.find(new_tid);
      if ( l != local_mem_lookup.end() ) {
         local_mem = l->second;
      } else {
         char buf[512];
         snprintf(buf,512,"local_%u_%u", sid, new_tid);
         local_mem = new memory_space_impl<32>(buf,32,local_space);
         local_mem_lookup[new_tid] = local_mem;
      }
      thd->set_info(kernel.entry());
      thd->set_nctaid(kernel.get_grid_dim());
      thd->set_ntid(kernel.get_cta_dim());
      thd->set_ctaid(ctaid3d);
      thd->set_tid(tid3d);
      if( kernel.entry()->get_ptx_version().extensions() )
         thd->cpy_tid_to_reg(tid3d);
      thd->set_valid();
      thd->m_shared_mem = shared_mem;
      function_info *finfo = thd->func_info();
      symbol_table *st = finfo->get_symtab();
      thd->func_info()->param_to_shared(thd->m_shared_mem,st);
      thd->m_cta_info = cta_info;
      cta_info->add_thread(thd);
      thd->m_local_mem = local_mem;
      if ( g_debug_execution==-1 ) {
         printf("GPGPU-Sim PTX simulator:  allocating thread ctaid=(%u,%u,%u) tid=(%u,%u,%u) @ 0x%Lx\n",
                ctaid3d.x,ctaid3d.y,ctaid3d.z,tid3d.x,tid3d.y,tid3d.z, (unsigned long long)thd );
         fflush(stdout);
      }
      active_threads.push_back(thd);
   }
   if ( g_debug_execution==-1 ) {
      printf("GPGPU-Sim PTX simulator:  <-- FINISHING THREAD ALLOCATION\n");
      fflush(stdout);
   }

   kernel.increment_cta_id();

   assert( active_threads.size() <= threads_left );
   *thread_info = active_threads.front();
   (*thread_info)->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,isInFunctionalSimulationMode  ,0);
   active_threads.pop_front();
   return 1;
}

size_t get_kernel_code_size( class function_info *entry )
{
   return entry->get_function_size();
}


kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                             gpgpu_ptx_sim_arg_list_t args,
                                             struct dim3 gridDim,
                                             struct dim3 blockDim,
                                             gpgpu_t *gpu )
{
   kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
   unsigned argcount=args.size();
   unsigned argn=1;
   for( gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++ ) {
      entry->add_param_data(argcount-argn,&(*a));
      argn++;
   }
   entry->finalize(result->get_param_memory());
   g_ptx_kernel_count++;
   fflush(stdout);

   return result;
}

#include "../../version"

void print_splash()
{
   static int splash_printed=0;
   if ( !splash_printed ) {
      unsigned build=0;
      sscanf(g_gpgpusim_build_string, "$Change"": %u $", &build);
      fprintf(stdout, "\n\n        *** %s [build %u] ***\n\n\n", g_gpgpusim_version_string, build );
      splash_printed=1;
   }
}

std::map<const void*,std::string>   g_const_name_lookup; // indexed by hostVar
std::map<const void*,std::string>   g_global_name_lookup; // indexed by hostVar
std::set<std::string>   g_globals;
std::set<std::string>   g_constants;

void gpgpu_ptx_sim_register_const_variable(void *hostVar, const char *deviceName, size_t size )
{
   printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n", deviceName, size );
   g_const_name_lookup[hostVar] = deviceName;
}

void gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size )
{
   printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n", deviceName );
   g_global_name_lookup[hostVar] = deviceName;
}

void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, gpgpu_t *gpu )
{
   printf("GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n", hostVar);
   bool found_sym = false;
   memory_space_t mem_region = undefined_space;
   std::string sym_name;

   std::map<const void*,std::string>::iterator c=g_const_name_lookup.find(hostVar);
   if ( c!=g_const_name_lookup.end() ) {
      found_sym = true;
      sym_name = c->second;
      mem_region = const_space;
   }
   std::map<const void*,std::string>::iterator g=g_global_name_lookup.find(hostVar);
   if ( g!=g_global_name_lookup.end() ) {
      if ( found_sym ) {
         printf("Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared both const and global?\n",
                sym_name.c_str(), (unsigned long long)hostVar );
         abort();
      }
      found_sym = true;
      sym_name = g->second;
      mem_region = global_space;
   }
   if( g_globals.find(hostVar) != g_globals.end() ) {
      found_sym = true;
      sym_name = hostVar;
      mem_region = global_space;
   }
   if( g_constants.find(hostVar) != g_constants.end() ) {
      found_sym = true;
      sym_name = hostVar;
      mem_region = const_space;
   }

   if ( !found_sym ) {
      printf("Execution error: No information for PTX symbol w/ hostVar=0x%Lx\n", (unsigned long long)hostVar );
      abort();
   } else printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ hostVar=0x%Lx\n", (unsigned long long)hostVar );
   const char *mem_name = NULL;
   memory_space *mem = NULL;

   std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
   assert( st != g_sym_name_to_symbol_table.end() );
   symbol_table *symtab = st->second;

   symbol *sym = symtab->lookup(sym_name.c_str());
   assert(sym);
   unsigned dst = sym->get_address() + offset;
   switch (mem_region.get_type()) {
   case const_space:
      mem = gpu->get_global_memory();
      mem_name = "const";
      break;
   case global_space:
      mem = gpu->get_global_memory();
      mem_name = "global";
      break;
   default:
      abort();
   }
   printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu bytes %s symbol %s+%zu @0x%x ...\n",
          mem_name, count, (to?" to ":"from"), sym_name.c_str(), offset, dst );
   for ( unsigned n=0; n < count; n++ ) {
      if( to ) mem->write(dst+n,1,((char*)src)+n,NULL,NULL,0);
      else mem->read(dst+n,1,((char*)src)+n,0);
   }
   fflush(stdout);
}

int g_ptx_sim_mode; // if non-zero run functional simulation only (i.e., no notion of a clock cycle)

extern int ptx_debug;

bool g_cuda_launch_blocking = false;

void read_sim_environment_variables()
{
   ptx_debug = 0;
   g_debug_execution = 0;
   g_interactive_debugger_enabled = false;

   char *mode = getenv("PTX_SIM_MODE_FUNC");
   if ( mode )
      sscanf(mode,"%u", &g_ptx_sim_mode);
   printf("GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC environment variable:\n", g_ptx_sim_mode);
   printf("               1=functional simulation only, 0=detailed performance simulator)\n");
   char *dbg_inter = getenv("GPGPUSIM_DEBUG");
   if ( dbg_inter && strlen(dbg_inter) ) {
      printf("GPGPU-Sim PTX: enabling interactive debugger\n");
      fflush(stdout);
      g_interactive_debugger_enabled = true;
   }
   char *dbg_level = getenv("PTX_SIM_DEBUG");
   if ( dbg_level && strlen(dbg_level) ) {
      printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level );
      fflush(stdout);
      sscanf(dbg_level,"%d", &g_debug_execution);
   }
   char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");
   if ( dbg_thread && strlen(dbg_thread) ) {
      printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n", dbg_thread );
      fflush(stdout);
      sscanf(dbg_thread,"%d", &g_debug_thread_uid);
   }
   char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");
   if ( dbg_pc && strlen(dbg_pc) ) {
      printf("GPGPU-Sim PTX: printing debug information for instruction with PC = %s\n", dbg_pc );
      fflush(stdout);
      sscanf(dbg_pc,"%d", &g_debug_pc);
   }

#if CUDART_VERSION > 1010
    g_override_embedded_ptx = false;
    char *usefile = getenv("PTX_SIM_USE_PTX_FILE");
    if (usefile && strlen(usefile)) {
        printf("GPGPU-Sim PTX: overriding embedded ptx with ptx file (PTX_SIM_USE_PTX_FILE is set)\n");
        fflush(stdout);
        g_override_embedded_ptx = true;
    }
    char *blocking = getenv("CUDA_LAUNCH_BLOCKING");
    if( blocking && !strcmp(blocking,"1") ) {
        g_cuda_launch_blocking = true;
    }
#else
   g_cuda_launch_blocking = true;
   g_override_embedded_ptx = true;
#endif

   if ( g_debug_execution >= 40 ) {
      ptx_debug = 1;
   }
}

ptx_cta_info *g_func_cta_info = NULL;

#define MAX(a,b) (((a)>(b))?(a):(b))

/*!
This function simulates the CUDA code functionally, it takes a kernel_info_t parameter
which holds the data for the CUDA kernel to be executed
!*/
void gpgpu_cuda_ptx_sim_main_func( kernel_info_t &kernel, bool openCL )
{
     printf("GPGPU-Sim: Performing Functional Simulation, executing kernel %s...\n",kernel.name().c_str());

     //using a shader core object for book keeping, it is not needed but as most function built for performance simulation need it we use it here
    extern gpgpu_sim *g_the_gpu;

    //we excute the kernel one CTA (Block) at the time, as synchronization functions work block wise
    while(!kernel.no_more_ctas_to_run()){
        functionalCoreSim cta(
            &kernel,
            g_the_gpu,
            g_the_gpu->getShaderCoreConfig()->warp_size
        );
        cta.execute();
    }

   //registering this kernel as done
   extern stream_manager *g_stream_manager;

   //openCL kernel simulation calls don't register the kernel so we don't register its exit
   if(!openCL)
   g_stream_manager->register_finished_kernel(kernel.get_uid());

   //******PRINTING*******
   printf( "GPGPU-Sim: Done functional simulation (%u instructions simulated).\n", g_ptx_sim_num_insn );
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp ( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }

   //time_t variables used to calculate the total simulation time
   //the start time of simulation is hold by the global variable g_simulation_starttime
   //g_simulation_starttime is initilized by gpgpu_ptx_sim_init_perf() in gpgpusim_entrypoint.cc upon starting gpgpu-sim
   time_t end_time, elapsed_time, days, hrs, minutes, sec;
   end_time = time((time_t *)NULL);
   elapsed_time = MAX(end_time - g_simulation_starttime, 1);


   //calculating and printing simulation time in terms of days, hours, minutes and seconds
   days    = elapsed_time/(3600*24);
   hrs     = elapsed_time/3600 - 24*days;
   minutes = elapsed_time/60 - 60*(hrs + 24*days);
   sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));

   fflush(stderr);
   printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
          (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec, (unsigned)elapsed_time );
   printf("gpgpu_simulation_rate = %u (inst/sec)\n", (unsigned)(g_ptx_sim_num_insn / elapsed_time) );
   fflush(stdout);
}

void functionalCoreSim::initializeCTA()
{
    int ctaLiveThreads=0;

    for(int i=0; i< m_warp_count; i++){
        m_warpAtBarrier[i]=false;
        m_liveThreadCount[i]=0;
    }
    for(int i=0; i< m_warp_count*m_warp_size;i++)
        m_thread[i]=NULL;

    //get threads for a cta
    for(unsigned i=0; i<m_kernel->threads_per_cta();i++) {
        ptx_sim_init_thread(*m_kernel,&m_thread[i],0,i,m_kernel->threads_per_cta()-i,m_kernel->threads_per_cta(),this,0,i/m_warp_size,(gpgpu_t*)m_gpu, true);
        assert(m_thread[i]!=NULL && !m_thread[i]->is_done());
        ctaLiveThreads++;
    }

    for(int k=0;k<m_warp_count;k++)
        createWarp(k);
}

void  functionalCoreSim::createWarp(unsigned warpId)
{
   simt_mask_t initialMask;
   unsigned liveThreadsCount=0;
   initialMask.set();
    for(int i=warpId*m_warp_size; i<warpId*m_warp_size+m_warp_size;i++){
        if(m_thread[i]==NULL) initialMask.reset(i-warpId*m_warp_size);
        else liveThreadsCount++;
    }

   assert(m_thread[warpId*m_warp_size]!=NULL);
   m_simt_stack[warpId]->launch(m_thread[warpId*m_warp_size]->get_pc(),initialMask);
   m_liveThreadCount[warpId]= liveThreadsCount;
}

void functionalCoreSim::execute()
 {
    initializeCTA();

    //start executing the CTA
    while(true){
        bool someOneLive= false;
        bool allAtBarrier = true;
        for(unsigned i=0;i<m_warp_count;i++){
            executeWarp(i,allAtBarrier,someOneLive);
        }
        if(!someOneLive) break;
        if(allAtBarrier){
             for(unsigned i=0;i<m_warp_count;i++)
                 m_warpAtBarrier[i]=false;
        }
    }
 }

void functionalCoreSim::executeWarp(unsigned i, bool &allAtBarrier, bool & someOneLive)
{
    if(!m_warpAtBarrier[i] && m_liveThreadCount[i]!=0){
        warp_inst_t inst =getExecuteWarp(i);
        execute_warp_inst_t(inst,i);
        if(inst.isatomic()) inst.do_atomic(true);
        if(inst.op==BARRIER_OP || inst.op==MEMORY_BARRIER_OP ) m_warpAtBarrier[i]=true;
        updateSIMTStack( i, &inst );
    }
    if(m_liveThreadCount[i]>0) someOneLive=true;
    if(!m_warpAtBarrier[i]&& m_liveThreadCount[i]>0) allAtBarrier = false;
}

unsigned translate_pc_to_ptxlineno(unsigned pc)
{
   // this function assumes that the kernel fits inside a single PTX file
   // function_info *pFunc = g_func_info; // assume that the current kernel is the one in query
   const ptx_instruction *pInsn = function_info::pc_to_instruction(pc);
   unsigned ptx_line_number = pInsn->source_line();

   return ptx_line_number;
}

// ptxinfo parser

int g_ptxinfo_error_detected;

static char *g_ptxinfo_kname = NULL;
static struct gpgpu_ptx_sim_kernel_info g_ptxinfo_kinfo;

const char *get_ptxinfo_kname()
{
    return g_ptxinfo_kname;
}

void print_ptxinfo()
{
    printf ("GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
            get_ptxinfo_kname(),
            g_ptxinfo_kinfo.regs,
            g_ptxinfo_kinfo.lmem,
            g_ptxinfo_kinfo.smem,
            g_ptxinfo_kinfo.cmem );
}


struct gpgpu_ptx_sim_kernel_info get_ptxinfo_kinfo()
{
    return g_ptxinfo_kinfo;
}

void ptxinfo_function(const char *fname )
{
    clear_ptxinfo();
    g_ptxinfo_kname = strdup(fname);
}

void ptxinfo_regs( unsigned nregs )
{
    g_ptxinfo_kinfo.regs=nregs;
}

void ptxinfo_lmem( unsigned declared, unsigned system )
{
    g_ptxinfo_kinfo.lmem=declared+system;
}

void ptxinfo_smem( unsigned declared, unsigned system )
{
    g_ptxinfo_kinfo.smem=declared+system;
}

void ptxinfo_cmem( unsigned nbytes, unsigned bank )
{
    g_ptxinfo_kinfo.cmem+=nbytes;
}

void clear_ptxinfo()
{
    free(g_ptxinfo_kname);
    g_ptxinfo_kname=NULL;
    g_ptxinfo_kinfo.regs=0;
    g_ptxinfo_kinfo.lmem=0;
    g_ptxinfo_kinfo.smem=0;
    g_ptxinfo_kinfo.cmem=0;
    g_ptxinfo_kinfo.ptx_version=0;
    g_ptxinfo_kinfo.sm_target=0;
}


void ptxinfo_opencl_addinfo( std::map<std::string,function_info*> &kernels )
{
   if( !strcmp("__cuda_dummy_entry__",g_ptxinfo_kname) ) {
      // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
      clear_ptxinfo();
      return;
   }
   std::map<std::string,function_info*>::iterator k=kernels.find(g_ptxinfo_kname);
   if( k==kernels.end() ) {
      printf ("GPGPU-Sim PTX: ERROR ** implementation for '%s' not found.\n", g_ptxinfo_kname );
      abort();
   } else {
      printf ("GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
              g_ptxinfo_kname,
              g_ptxinfo_kinfo.regs,
              g_ptxinfo_kinfo.lmem,
              g_ptxinfo_kinfo.smem,
              g_ptxinfo_kinfo.cmem );
      function_info *finfo = k->second;
      assert(finfo!=NULL);
      finfo->set_kernel_info( g_ptxinfo_kinfo );
   }
   clear_ptxinfo();
}

struct rec_pts {
   gpgpu_recon_t *s_kernel_recon_points;
   int s_num_recon;
};

struct std::map<function_info*,rec_pts> g_rpts;

struct rec_pts find_reconvergence_points( function_info *finfo )
{
   rec_pts tmp;
   std::map<function_info*,rec_pts>::iterator r=g_rpts.find(finfo);

   if( r==g_rpts.end() ) {
      int num_recon = finfo->get_num_reconvergence_pairs();

      gpgpu_recon_t *kernel_recon_points = (struct gpgpu_recon_t*) calloc(num_recon, sizeof(struct gpgpu_recon_t));
      finfo->get_reconvergence_pairs(kernel_recon_points);
      printf("GPGPU-Sim PTX: reconvergence points for %s...\n", finfo->get_name().c_str() );
      for (int i=0;i<num_recon;i++) {
         printf("GPGPU-Sim PTX: %2u (potential) branch divergence @ ", i+1 );
         kernel_recon_points[i].source_inst->print_insn();
         printf("\n");
         printf("GPGPU-Sim PTX:    immediate post dominator      @ " );
         if( kernel_recon_points[i].target_inst )
            kernel_recon_points[i].target_inst->print_insn();
         printf("\n");
      }
      printf("GPGPU-Sim PTX: ... end of reconvergence points for %s\n", finfo->get_name().c_str() );

      tmp.s_kernel_recon_points = kernel_recon_points;
      tmp.s_num_recon = num_recon;
      g_rpts[finfo] = tmp;
   } else {
      tmp = r->second;
   }
   return tmp;
}

address_type get_return_pc( void *thd )
{
    // function call return
    ptx_thread_info *the_thread = (ptx_thread_info*)thd;
    assert( the_thread != NULL );
    return the_thread->get_return_PC();
}

address_type get_converge_point( address_type pc )
{
   // the branch could encode the reconvergence point and/or a bit that indicates the
   // reconvergence point is the return PC on the call stack in the case the branch has
   // no immediate postdominator in the function (i.e., due to multiple return points).

   std::map<unsigned,function_info*>::iterator f=g_pc_to_finfo.find(pc);
   assert( f != g_pc_to_finfo.end() );
   function_info *finfo = f->second;
   rec_pts tmp = find_reconvergence_points(finfo);

   int i=0;
   for (; i < tmp.s_num_recon; ++i) {
      if (tmp.s_kernel_recon_points[i].source_pc == pc) {
          if( tmp.s_kernel_recon_points[i].target_pc == (unsigned) -2 ) {
              return RECONVERGE_RETURN_PC;
          } else {
              return tmp.s_kernel_recon_points[i].target_pc;
          }
      }
   }
   return NO_BRANCH_DIVERGENCE;
}

void functionalCoreSim::warp_exit( unsigned warp_id )
{
    for(int i=0;i<m_warp_count*m_warp_size;i++){
        if(m_thread[i]!=NULL){
             m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]);
             delete m_thread[i];
        }
    }
}
