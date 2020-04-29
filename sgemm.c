//gcc -march=skylake-avx512 --shared -fPIC -O2 sgemm.c -o sgemm.so
/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 for k_count, %5 for c_store */
/* r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#define KERNEL_DEBUG //enable the generation of sgemm caller function (serial mode)

#if defined KERNEL_DEBUG
 #include <stdio.h>
 #include <stdlib.h>
 #include <immintrin.h>
 #define BLASLONG int
#else
 #include "common.h"
 #include <stdint.h>
#endif

/* m = 16 */ /* zmm8-zmm31 for accumulators, zmm1-zmm7 for temporary use, zmm0 for alpha */
#define KERNEL_k1m16n1 \
    "vmovups (%0),%%zmm4; addq $64,%0;"\
    "vbroadcastss (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8;"\
    "addq $4,%1;"
#define KERNEL_h_k1m16n2 \
    "vmovsldup (%0),%%zmm4; vmovshdup (%0),%%zmm5; prefetcht0 512(%0); addq $64,%0;"\
    "vbroadcastsd (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8; vfmadd231ps %%zmm5,%%zmm6,%%zmm9;"
#define KERNEL_k1m16n2 KERNEL_h_k1m16n2 "addq $8,%1;"
#define KERNEL_h_k1m16n4 KERNEL_h_k1m16n2 "vbroadcastsd 8(%1),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,%%zmm10; vfmadd231ps %%zmm5,%%zmm7,%%zmm11;"
#define KERNEL_k1m16n4 KERNEL_h_k1m16n4 "addq $16,%1;"
#define unit_kernel_k1m16n4(c1,c2,c3,c4, ...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,"#c1"; vfmadd231ps %%zmm5,%%zmm6,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,"#c3"; vfmadd231ps %%zmm5,%%zmm7,"#c4";"
#define KERNEL_h_k1m16n8 KERNEL_h_k1m16n4 unit_kernel_k1m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15,%1,%%r12,1)
#define KERNEL_k1m16n8 KERNEL_h_k1m16n8 "addq $16,%1;"
#define KERNEL_h_k1m16n12 KERNEL_h_k1m16n8 unit_kernel_k1m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19,%1,%%r12,2)
#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $16,%1;"
#define KERNEL_h_k1m16n16 KERNEL_k1m16n12 unit_kernel_k1m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23,%%r15)
#define KERNEL_k1m16n16 KERNEL_h_k1m16n16 "addq $16,%%r15;"
#define KERNEL_h_k1m16n20 KERNEL_h_k1m16n16 unit_kernel_k1m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27,%%r15,%%r12,1)
#define KERNEL_k1m16n20 KERNEL_h_k1m16n20 "addq $16,%%r15;"
#define KERNEL_h_k1m16n24 KERNEL_h_k1m16n20 unit_kernel_k1m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31,%%r15,%%r12,2)
#define KERNEL_k1m16n24 KERNEL_h_k1m16n24 "addq $16,%%r15;"
#define INIT_m16n1 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m16n2 INIT_m16n1 "vpxorq %%zmm9,%%zmm9,%%zmm9;"
#define INIT_m16n4 INIT_m16n2 "vpxorq %%zmm10,%%zmm10,%%zmm10;vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define unit_init_m16n4(c1,c2,c3,c4) \
    "vpxorq "#c1","#c1","#c1";vpxorq "#c2","#c2","#c2";vpxorq "#c3","#c3","#c3";vpxorq "#c4","#c4","#c4";"
#define INIT_m16n8 INIT_m16n4 unit_init_m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define INIT_m16n12 INIT_m16n8 unit_init_m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define INIT_m16n16 INIT_m16n12 unit_init_m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23)
#define INIT_m16n20 INIT_m16n16 unit_init_m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27)
#define INIT_m16n24 INIT_m16n20 unit_init_m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31)
#define SAVE_h_m16n1 "vfmadd213ps (%2),%%zmm0,%%zmm8; vmovups %%zmm8,(%2);"
#define unit_save_m16n2(c1,c2) \
    "vunpcklps "#c2","#c1",%%zmm6; vunpckhps "#c2","#c1",%%zmm7; vunpcklpd %%zmm7,%%zmm6,%%zmm4; vunpckhpd %%zmm7,%%zmm6,%%zmm5;"\
    "vfmadd213ps (%5),%%zmm0,%%zmm4; vfmadd213ps (%5,%3,1),%%zmm0,%%zmm5;"\
    "prefetcht1 127(%5); prefetcht1 127(%5,%3,1);"\
    "vmovups %%zmm4,(%5); vmovups %%zmm5,(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_h_m16n2 "movq %2,%5;" unit_save_m16n2(%%zmm8,%%zmm9)
#define SAVE_h_m16n4  SAVE_h_m16n2  unit_save_m16n2(%%zmm10,%%zmm11)
#define SAVE_h_m16n8  SAVE_h_m16n4  unit_save_m16n2(%%zmm12,%%zmm13) unit_save_m16n2(%%zmm14,%%zmm15)
#define SAVE_h_m16n12 SAVE_h_m16n8  unit_save_m16n2(%%zmm16,%%zmm17) unit_save_m16n2(%%zmm18,%%zmm19)
#define SAVE_h_m16n16 SAVE_h_m16n12 unit_save_m16n2(%%zmm20,%%zmm21) unit_save_m16n2(%%zmm22,%%zmm23)
#define SAVE_h_m16n20 SAVE_h_m16n16 unit_save_m16n2(%%zmm24,%%zmm25) unit_save_m16n2(%%zmm26,%%zmm27)
#define SAVE_h_m16n24 SAVE_h_m16n20 unit_save_m16n2(%%zmm28,%%zmm29) unit_save_m16n2(%%zmm30,%%zmm31)
#define SAVE_m16(ndim) SAVE_h_m16n##ndim "addq $64,%2;"
#define COMPUTE_m16(ndim) \
    INIT_m16n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15;"\
    "cmpq $4,%4; jb "#ndim"016162f;"\
    #ndim"016161:\n\t"\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    "subq $4,%4; cmpq $4,%4; jnb "#ndim"016161b;"\
    #ndim"016162:\n\t"\
    "testq %4,%4; jz "#ndim"016163f;"\
    KERNEL_k1m16n##ndim\
    "decq %4; jmp "#ndim"016162b;"\
    #ndim"016163:\n\t"\
    SAVE_m16(ndim)

/* m = 8 *//* ymm0 for alpha, ymm1-ymm3 for temporary use, ymm4-ymm15 for accumulators */
#define KERNEL_k1m8n1(b_addr) \
    "vmovups (%0),%%ymm1; addq $32,%0;"\
    "vbroadcastss ("#b_addr"),%%ymm2; vfmadd231ps %%ymm1,%%ymm2,%%ymm4;"\
    "addq $4,"#b_addr";"
#define KERNEL_h_k1m8n2(b_addr) \
    "vmovsldup (%0),%%ymm1; vmovshdup (%0),%%ymm2; addq $32,%0;"\
    "vbroadcastsd ("#b_addr"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm4; vfmadd231ps %%ymm2,%%ymm3,%%ymm5;"
#define KERNEL_k1m8n2(b_addr) KERNEL_h_k1m8n2(b_addr) "addq $8,"#b_addr";"
#define KERNEL_h_k1m8n4(b_addr) \
    KERNEL_h_k1m8n2(b_addr) "vbroadcastsd 8("#b_addr"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm6; vfmadd231ps %%ymm2,%%ymm3,%%ymm7;"
#define KERNEL_k1m8n4(b_addr) KERNEL_h_k1m8n4(b_addr) "addq $16,"#b_addr";"
#define unit_kernel_k1m8n4(c1,c2,c3,c4,...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c1"; vfmadd231ps %%ymm2,%%ymm3,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c3"; vfmadd231ps %%ymm2,%%ymm3,"#c4";"
#define KERNEL_h_k1m8n8(b_addr) KERNEL_h_k1m8n4(b_addr) unit_kernel_k1m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11,b_addr,%%r12,1)
#define KERNEL_k1m8n8(b_addr) KERNEL_h_k1m8n8(b_addr) "addq $16,"#b_addr";"
#define KERNEL_h_k1m8n12(b_addr) KERNEL_h_k1m8n8(b_addr) unit_kernel_k1m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15,b_addr,%%r12,2)
#define KERNEL_k1m8n12(b_addr) KERNEL_h_k1m8n12(b_addr) "addq $16,"#b_addr";"
#define INIT_m8n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m8n2 INIT_m8n1 "vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m8n4 INIT_m8n2 "vpxor %%ymm6,%%ymm6,%%ymm6;vpxor %%ymm7,%%ymm7,%%ymm7;"
#define unit_init_m8n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m8n8  INIT_m8n4 unit_init_m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11)
#define INIT_m8n12 INIT_m8n8 unit_init_m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15)
#define SAVE_L_m8n1 "vfmadd213ps (%2),%%ymm0,%%ymm4; vmovups %%ymm4,(%2);"
#define unit_save_m8n2(c1,c2) \
    "vunpcklps "#c2","#c1",%%ymm2; vunpckhps "#c2","#c1",%%ymm3;"\
    "vunpcklpd %%ymm3,%%ymm2,%%ymm1;vfmadd213ps (%5),     %%ymm0,%%ymm1;vmovups %%ymm1,(%5);"\
    "vunpckhpd %%ymm3,%%ymm2,%%ymm1;vfmadd213ps (%5,%3,1),%%ymm0,%%ymm1;vmovups %%ymm1,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m8n2 "movq %2,%5;" unit_save_m8n2(%%ymm4,%%ymm5)
#define SAVE_L_m8n4  SAVE_L_m8n2  unit_save_m8n2(%%ymm6,%%ymm7)
#define SAVE_L_m8n8  SAVE_L_m8n4  unit_save_m8n2(%%ymm8,%%ymm9)   unit_save_m8n2(%%ymm10,%%ymm11)
#define SAVE_L_m8n12 SAVE_L_m8n8  unit_save_m8n2(%%ymm12,%%ymm13) unit_save_m8n2(%%ymm14,%%ymm15)
#define SAVE_R_m8n4               unit_save_m8n2(%%ymm4,%%ymm5)   unit_save_m8n2(%%ymm6,%%ymm7)
#define SAVE_R_m8n8  SAVE_R_m8n4  unit_save_m8n2(%%ymm8,%%ymm9)   unit_save_m8n2(%%ymm10,%%ymm11)
#define SAVE_R_m8n12 SAVE_R_m8n8  unit_save_m8n2(%%ymm12,%%ymm13) unit_save_m8n2(%%ymm14,%%ymm15)
#define COMPUTE_L_m8(ndim,sim) \
    INIT_m8n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"882:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"883f;"\
    KERNEL_k1m8n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"882b;"\
    #ndim""#sim"883:\n\t"\
    SAVE_L_m8n##ndim "addq $32,%2;"
#define COMPUTE_R_m8(ndim,sim) \
    "subq %%r12,%0; subq %%r12,%0;"\
    INIT_m8n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"882:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"883f;"\
    KERNEL_k1m8n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"882b;"\
    #ndim""#sim"883:\n\t"\
    SAVE_R_m8n##ndim
#define COMPUTE_m8_n1  COMPUTE_L_m8(1,33833)
#define COMPUTE_m8_n2  COMPUTE_L_m8(2,33833)
#define COMPUTE_m8_n4  COMPUTE_L_m8(4,33833)
#define COMPUTE_m8_n8  COMPUTE_L_m8(8,33833)
#define COMPUTE_m8_n12 COMPUTE_L_m8(12,33833)
#define COMPUTE_m8_n16 COMPUTE_L_m8(12,33733) COMPUTE_R_m8(4,33933)
#define COMPUTE_m8_n20 COMPUTE_L_m8(12,33633) COMPUTE_R_m8(8,33933)
#define COMPUTE_m8_n24 COMPUTE_L_m8(12,33533) COMPUTE_R_m8(12,33933)
#define COMPUTE_m8(ndim) COMPUTE_m8_n##ndim

/* m = 4 *//* xmm0 for alpha, xmm1-xmm3 for temporary use, xmm4-xmm15 for accumulators */
#define KERNEL_k1m4n1(b_addr) \
    "vmovups (%0),%%xmm1; addq $16,%0;"\
    "vbroadcastss ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,"#b_addr";"
#define KERNEL_h_k1m4n2(b_addr) \
    "vmovsldup (%0),%%xmm1; vmovshdup (%0),%%xmm2; addq $16,%0;"\
    "vmovddup ("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm4; vfmadd231ps %%xmm2,%%xmm3,%%xmm5;"
#define KERNEL_k1m4n2(b_addr) KERNEL_h_k1m4n2(b_addr) "addq $8,"#b_addr";"
#define KERNEL_h_k1m4n4(b_addr) \
    KERNEL_h_k1m4n2(b_addr) "vmovddup 8("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm6; vfmadd231ps %%xmm2,%%xmm3,%%xmm7;"
#define KERNEL_k1m4n4(b_addr) KERNEL_h_k1m4n4(b_addr) "addq $16,"#b_addr";"
#define unit_kernel_k1m4n4(c1,c2,c3,c4,...) \
    "vmovddup  ("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c1"; vfmadd231ps %%xmm2,%%xmm3,"#c2";"\
    "vmovddup 8("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c3"; vfmadd231ps %%xmm2,%%xmm3,"#c4";"
#define KERNEL_h_k1m4n8(b_addr) KERNEL_h_k1m4n4(b_addr) unit_kernel_k1m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11,b_addr,%%r12,1)
#define KERNEL_k1m4n8(b_addr) KERNEL_h_k1m4n8(b_addr) "addq $16,"#b_addr";"
#define KERNEL_h_k1m4n12(b_addr) KERNEL_h_k1m4n8(b_addr) unit_kernel_k1m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15,b_addr,%%r12,2)
#define KERNEL_k1m4n12(b_addr) KERNEL_h_k1m4n12(b_addr) "addq $16,"#b_addr";"
#define INIT_m4n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m4n2 INIT_m4n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%xmm6,%%xmm6,%%xmm6;vpxor %%xmm7,%%xmm7,%%xmm7;"
#define unit_init_m4n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m4n8  INIT_m4n4 unit_init_m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11)
#define INIT_m4n12 INIT_m4n8 unit_init_m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15)
#define SAVE_L_m4n1 "vfmadd213ps (%2),%%xmm0,%%xmm4; vmovups %%xmm4,(%2);"
#define unit_save_m4n2(c1,c2) \
    "vunpcklps "#c2","#c1",%%xmm2; vunpckhps "#c2","#c1",%%xmm3;"\
    "vunpcklpd %%xmm3,%%xmm2,%%xmm1;vfmadd213ps (%5),     %%xmm0,%%xmm1;vmovups %%xmm1,(%5);"\
    "vunpckhpd %%xmm3,%%xmm2,%%xmm1;vfmadd213ps (%5,%3,1),%%xmm0,%%xmm1;vmovups %%xmm1,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m4n2 "movq %2,%5;" unit_save_m4n2(%%xmm4,%%xmm5)
#define SAVE_L_m4n4  SAVE_L_m4n2  unit_save_m4n2(%%xmm6,%%xmm7)
#define SAVE_L_m4n8  SAVE_L_m4n4  unit_save_m4n2(%%xmm8,%%xmm9)   unit_save_m4n2(%%xmm10,%%xmm11)
#define SAVE_L_m4n12 SAVE_L_m4n8  unit_save_m4n2(%%xmm12,%%xmm13) unit_save_m4n2(%%xmm14,%%xmm15)
#define SAVE_R_m4n4               unit_save_m4n2(%%xmm4,%%xmm5)   unit_save_m4n2(%%xmm6,%%xmm7)
#define SAVE_R_m4n8  SAVE_R_m4n4  unit_save_m4n2(%%xmm8,%%xmm9)   unit_save_m4n2(%%xmm10,%%xmm11)
#define SAVE_R_m4n12 SAVE_R_m4n8  unit_save_m4n2(%%xmm12,%%xmm13) unit_save_m4n2(%%xmm14,%%xmm15)
#define COMPUTE_L_m4(ndim,sim) \
    INIT_m4n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"442:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"443f;"\
    KERNEL_k1m4n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"442b;"\
    #ndim""#sim"443:\n\t"\
    SAVE_L_m4n##ndim "addq $16,%2;"
#define COMPUTE_R_m4(ndim,sim) \
    "subq %%r12,%0;"\
    INIT_m4n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"442:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"443f;"\
    KERNEL_k1m4n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"442b;"\
    #ndim""#sim"443:\n\t"\
    SAVE_R_m4n##ndim
#define COMPUTE_m4_n1  COMPUTE_L_m4(1,55855)
#define COMPUTE_m4_n2  COMPUTE_L_m4(2,55855)
#define COMPUTE_m4_n4  COMPUTE_L_m4(4,55855)
#define COMPUTE_m4_n8  COMPUTE_L_m4(8,55855)
#define COMPUTE_m4_n12 COMPUTE_L_m4(12,55855)
#define COMPUTE_m4_n16 COMPUTE_L_m4(12,55755) COMPUTE_R_m4(4,55955)
#define COMPUTE_m4_n20 COMPUTE_L_m4(12,55655) COMPUTE_R_m4(8,55955)
#define COMPUTE_m4_n24 COMPUTE_L_m4(12,55555) COMPUTE_R_m4(12,55955)
#define COMPUTE_m4(ndim) COMPUTE_m4_n##ndim

/* m = 2 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm9 for accumulators */
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m2n1(b_addr) \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,"#b_addr";"
#define SAVE_L_m2n1 "vmovsd (%2),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm4; vmovsd %%xmm4,(%2);"
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define KERNEL_k1m2n2(b_addr) \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss  ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "vbroadcastss 4("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm5;"\
    "addq $8,"#b_addr";"
#define SAVE_L_m2n2 SAVE_L_m2n1 "vmovsd (%2,%3,1),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm5; vmovsd %%xmm5,(%2,%3,1);"
#define INIT_m2n4  INIT_m2n2
#define INIT_m2n8  INIT_m2n4 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m2n12 INIT_m2n8 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define KERNEL_k1m2n4(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "vbroadcastss 4(%0),%%xmm2; vfmadd231ps %%xmm3,%%xmm2,%%xmm5;"\
    "addq $8,%0;"
#define KERNEL_k1m2n8(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm6;"\
    "vbroadcastss 4(%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm5; vfmadd231ps %%xmm2,%%xmm1,%%xmm7;"\
    "addq $8,%0;"
#define KERNEL_k1m2n12(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; vmovups ("#b_addr",%%r12,2),%%xmm1; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm6; vfmadd231ps %%xmm1,%%xmm10,%%xmm8;"\
    "vbroadcastss 4(%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm5; vfmadd231ps %%xmm2,%%xmm10,%%xmm7; vfmadd231ps %%xmm1,%%xmm10,%%xmm9;"\
    "addq $8,%0;"
#define unit_save_m2n4(c1,c2) \
    "vunpcklps "#c2","#c1",%%xmm1; vunpckhps "#c2","#c1",%%xmm2;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm1; vmovsd %%xmm1,(%5); vmovhpd %%xmm1,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm2; vmovsd %%xmm2,(%5); vmovhpd %%xmm2,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m2n4  "movq %2,%5;" unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_L_m2n8  SAVE_L_m2n4   unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_L_m2n12 SAVE_L_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define SAVE_R_m2n4                unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_R_m2n8  SAVE_R_m2n4   unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_R_m2n12 SAVE_R_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define COMPUTE_L_m2(ndim,sim) \
    INIT_m2n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"222:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"223f;"\
    KERNEL_k1m2n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"222b;"\
    #ndim""#sim"223:\n\t"\
    SAVE_L_m2n##ndim "addq $8,%2;"
#define COMPUTE_R_m2(ndim,sim) \
    "salq $3,%%r13;subq %%r13,%0;sarq $3,%%r13;"\
    INIT_m2n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"222:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"223f;"\
    KERNEL_k1m2n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"222b;"\
    #ndim""#sim"223:\n\t"\
    SAVE_R_m2n##ndim
#define COMPUTE_m2_n1  COMPUTE_L_m2(1,77877)
#define COMPUTE_m2_n2  COMPUTE_L_m2(2,77877)
#define COMPUTE_m2_n4  COMPUTE_L_m2(4,77877)
#define COMPUTE_m2_n8  COMPUTE_L_m2(8,77877)
#define COMPUTE_m2_n12 COMPUTE_L_m2(12,77877)
#define COMPUTE_m2_n16 COMPUTE_L_m2(12,77777) COMPUTE_R_m2(4,77977)
#define COMPUTE_m2_n20 COMPUTE_L_m2(12,77677) COMPUTE_R_m2(8,77977)
#define COMPUTE_m2_n24 COMPUTE_L_m2(12,77577) COMPUTE_R_m2(12,77977)
#define COMPUTE_m2(ndim) COMPUTE_m2_n##ndim

/* m = 1 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm6 for accumulators */
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m1n1(b_addr) \
    "vmovss ("#b_addr"),%%xmm3; addq $4,"#b_addr";"\
    "vmovss (%0),%%xmm1; vfmadd231ss %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_L_m1n1 "vfmadd213ss (%2),%%xmm0,%%xmm4; vmovss %%xmm4,(%2);"
#define INIT_m1n2 INIT_m1n1
#define KERNEL_k1m1n2(b_addr) \
    "vmovsd ("#b_addr"),%%xmm3; addq $8,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_L_m1n2 \
    "vmovss (%2),%%xmm3; vinsertps $16,(%2,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm4;"\
    "vmovss %%xmm4,(%2); vextractps $1,%%xmm4,(%2,%3,1);"
#define INIT_m1n4  INIT_m1n2
#define INIT_m1n8  INIT_m1n4 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n12 INIT_m1n8 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define KERNEL_k1m1n4(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define KERNEL_k1m1n8(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm5;"\
    "addq $4,%0;"
#define KERNEL_k1m1n12(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; vmovups ("#b_addr",%%r12,2),%%xmm1; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm5; vfmadd231ps %%xmm1,%%xmm10,%%xmm6;"\
    "addq $4,%0;"
#define unit_save_m1n4(c1) \
    "vpxor %%xmm10,%%xmm10,%%xmm10; vmovsd "#c1",%%xmm10,%%xmm2; vmovhlps "#c1",%%xmm10,%%xmm1;"\
    "vmovss (%5),%%xmm3; vinsertps $16,(%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm2;"\
    "vmovss %%xmm2,(%5); vextractps $1,%%xmm2,(%5,%3,1); leaq (%5,%3,2),%5;"\
    "vmovss (%5),%%xmm3; vinsertps $16,(%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm1;"\
    "vmovss %%xmm1,(%5); vextractps $1,%%xmm1,(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_L_m1n4 "movq %2,%5;" unit_save_m1n4(%%xmm4)
#define SAVE_L_m1n8  SAVE_L_m1n4  unit_save_m1n4(%%xmm5)
#define SAVE_L_m1n12 SAVE_L_m1n8  unit_save_m1n4(%%xmm6)
#define SAVE_R_m1n4               unit_save_m1n4(%%xmm4)
#define SAVE_R_m1n8  SAVE_R_m1n4  unit_save_m1n4(%%xmm5)
#define SAVE_R_m1n12 SAVE_R_m1n8  unit_save_m1n4(%%xmm6)
#define COMPUTE_L_m1(ndim,sim) \
    INIT_m1n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"112:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"113f;"\
    KERNEL_k1m1n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"112b;"\
    #ndim""#sim"113:\n\t"\
    SAVE_L_m1n##ndim "addq $4,%2;"
#define COMPUTE_R_m1(ndim,sim) \
    "salq $2,%%r13;subq %%r13,%0;sarq $2,%%r13;"\
    INIT_m1n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"112:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"113f;"\
    KERNEL_k1m1n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"112b;"\
    #ndim""#sim"113:\n\t"\
    SAVE_R_m1n##ndim
#define COMPUTE_m1_n1  COMPUTE_L_m1(1,99899)
#define COMPUTE_m1_n2  COMPUTE_L_m1(2,99899)
#define COMPUTE_m1_n4  COMPUTE_L_m1(4,99899)
#define COMPUTE_m1_n8  COMPUTE_L_m1(8,99899)
#define COMPUTE_m1_n12 COMPUTE_L_m1(12,99899)
#define COMPUTE_m1_n16 COMPUTE_L_m1(12,99799) COMPUTE_R_m1(4,99999)
#define COMPUTE_m1_n20 COMPUTE_L_m1(12,99699) COMPUTE_R_m1(8,99999)
#define COMPUTE_m1_n24 COMPUTE_L_m1(12,99599) COMPUTE_R_m1(12,99999)
#define COMPUTE_m1(ndim) COMPUTE_m1_n##ndim

/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 = "+r"(K), %5 = "+r"(ctemp) */
/* %6 = "+r"(&alpha), %7 = "+r"(M) */
/* r11 = m(const), r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#define COMPUTE(ndim) {\
    __asm__ __volatile__(\
    "vbroadcastss (%6),%%zmm0;"\
    "movq %4,%%r13; movq %4,%%r12; salq $4,%%r12; movq %1,%%r14; movq %7,%%r11;"\
    "cmpq $16,%7;jb 33101"#ndim"f;"\
    "33109"#ndim":\n\t"\
    COMPUTE_m16(ndim)\
    "subq $16,%7;cmpq $16,%7;jnb 33109"#ndim"b;"\
    "33101"#ndim":\n\t"\
    "cmpq $8,%7;jb 33102"#ndim"f;"\
    COMPUTE_m8(ndim)\
    "subq $8,%7;"\
    "33102"#ndim":\n\t"\
    "cmpq $4,%7;jb 33103"#ndim"f;"\
    COMPUTE_m4(ndim)\
    "subq $4,%7;"\
    "33103"#ndim":\n\t"\
    "cmpq $2,%7;jb 33104"#ndim"f;"\
    COMPUTE_m2(ndim)\
    "subq $2,%7;"\
    "33104"#ndim":\n\t"\
    "testq %7,%7;jz 33105"#ndim"f;"\
    COMPUTE_m1(ndim)\
    "33105"#ndim":\n\t"\
    "movq %%r13,%4; movq %%r14,%1; movq %%r11,%7;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ldc_in_bytes),"+r"(K),"+r"(ctemp),"+r"(alp),"+r"(M)\
    ::"r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14",\
    "zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",\
    "cc","memory");\
    a_pointer -= M * K; b_pointer += ndim * K;c_pointer += LDC * ndim - M;\
}
static inline void// __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alpha, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, BLASLONG LDC)
{
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(float);float ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    float *a_pointer = A,*b_pointer = B,*c_pointer = C,*ctemp = C,*alp = &ALPHA;
    for(;n_count>23;n_count-=24) COMPUTE(24)
    for(;n_count>19;n_count-=20) COMPUTE(20)
    for(;n_count>15;n_count-=16) COMPUTE(16)
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
}

/* test zone */
#ifdef KERNEL_DEBUG
static inline void SCALE_MULT(float *dat, float *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0) return;
    float scale = *sca;float *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale;current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
static inline void sgemm_tcopy_16(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 16;
      for(count_first=dim_first;count_first>15;count_first-=16){
        _mm512_storeu_ps(todst,_mm512_loadu_ps(tosrc));
        tosrc+=16;todst+=16*dim_second;
      }
      todst -= count_second * 8;
      for(;count_first>7;count_first-=8){
        _mm256_storeu_ps(todst,_mm256_loadu_ps(tosrc));
        tosrc+=8;todst+=8*dim_second;
      }
      todst -= count_second * 4;
      for(;count_first>3;count_first-=4){
        _mm_storeu_ps(todst,_mm_loadu_ps(tosrc));
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        *todst=*tosrc;todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static inline void sgemm_ncopy_16(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8,*tosrc9,*tosrc10,*tosrc11,*tosrc12,*tosrc13,*tosrc14,*tosrc15,*tosrc16;
    float *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;tosrc7=tosrc6+lead_dim;tosrc8=tosrc7+lead_dim;
    tosrc9=tosrc8+lead_dim;tosrc10=tosrc9+lead_dim;tosrc11=tosrc10+lead_dim;tosrc12=tosrc11+lead_dim;
    tosrc13=tosrc12+lead_dim;tosrc14=tosrc13+lead_dim;tosrc15=tosrc14+lead_dim;tosrc16=tosrc15+lead_dim;
    tosrc_inc=16*lead_dim-dim_first;
    for(count_second=dim_second;count_second>15;count_second-=16){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst[8]=*tosrc9;tosrc9++;todst[9]=*tosrc10;tosrc10++;
        todst[10]=*tosrc11;tosrc11++;todst[11]=*tosrc12;tosrc12++;
        todst[12]=*tosrc13;tosrc13++;todst[13]=*tosrc14;tosrc14++;
        todst[14]=*tosrc15;tosrc15++;todst[15]=*tosrc16;tosrc16++;
        todst+=16;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
      tosrc9+=tosrc_inc;tosrc10+=tosrc_inc;tosrc11+=tosrc_inc;tosrc12+=tosrc_inc;
      tosrc13+=tosrc_inc;tosrc14+=tosrc_inc;tosrc15+=tosrc_inc;tosrc16+=tosrc_inc;
    }
    tosrc_inc-=8*lead_dim;
    for(;count_second>7;count_second-=8){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst+=8;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
    }
    tosrc_inc-=4*lead_dim;
    for(;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static inline void sgemm_tcopy_4(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 4;
      for(count_first=dim_first;count_first>3;count_first-=4){
        _mm_storeu_ps(todst,_mm_loadu_ps(tosrc));
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        *todst=*tosrc;todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static inline void sgemm_ncopy_4(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4;
    float *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc_inc=4*lead_dim-dim_first;
    for(count_second=dim_second;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
#define BLOCKDIM_K 192 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 384 //GEMM_P in OpenBLAS
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
void inline SGEMM_(char *transa,char *transb,BLASLONG *m,BLASLONG *n,BLASLONG *k,float *alpha,float *a,BLASLONG *lda,float *b,BLASLONG *ldb,float *beta,float *c,BLASLONG *ldc){
    if((*m)==0||(*n)==0) return;
    if((*beta)!=1.0) SCALE_MULT(c,beta,*ldc,*m,*n);
    if((*alpha)==0.0||(*k)==0) return;
/* start main calculation here */
    //if((*m)==91 && (*n)==45 && (*k)==31) c[101]*=2.0;
    float *b_buffer = (float *)aligned_alloc(64,BLOCKDIM_K*(*n)*sizeof(float));
    float *a_buffer = (float *)aligned_alloc(4096,BLOCKDIM_K*BLOCKDIM_M*sizeof(float));
    float *a_current_pos,*b_current_pos;
    BLASLONG m_count,k_count,k_subdim,m_subdim;//,n_count;
    b_current_pos = b;
    for(k_count=0;k_count<(*k);k_count+=BLOCKDIM_K){
      k_subdim = (*k)-k_count;
      if(k_subdim > BLOCKDIM_K) k_subdim = BLOCKDIM_K;
      if(NOTRANSB) { sgemm_ncopy_4(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { sgemm_tcopy_4(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { sgemm_tcopy_16(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { sgemm_ncopy_16(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        CNAME(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
#endif
