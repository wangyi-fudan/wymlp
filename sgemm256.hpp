/* %0=a_ptr, %1=b_ptr, %2=c_ptr, %3=c_tmp, %4=ldc(bytes), %5=&alpha */

#define KERNEL_k1m1n8 \
  "vbroadcastss (%0),%%ymm1; addq $4,%0; vfmadd231ps (%1),%%ymm1,%%ymm4; addq $32,%1;"
#define KERNEL_k1m1n16 \
  "vbroadcastss (%0),%%ymm1; addq $4,%0; vfmadd231ps (%1),%%ymm1,%%ymm4; vfmadd231ps 32(%1),%%ymm1,%%ymm5; addq $64,%1;"
#define KERNEL_k1m2n8 \
  "vbroadcastss (%0),%%ymm1; vbroadcastss 4(%0),%%ymm2; addq $8,%0;"\
  "vmovups (%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm4; vfmadd231ps %%ymm2,%%ymm0,%%ymm5; addq $32,%1;"
#define KERNEL_k1m2n16 \
  "vbroadcastss (%0),%%ymm1; vbroadcastss 4(%0),%%ymm2; addq $8,%0;"\
  "vmovups (%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm4; vfmadd231ps %%ymm2,%%ymm0,%%ymm5;"\
  "vmovups 32(%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm6; vfmadd231ps %%ymm2,%%ymm0,%%ymm7; addq $64,%1;"
#define unit_acc_m4n8(c1_no,c2_no,c3_no,c4_no,boff) \
  "vmovsldup "#boff"(%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm"#c1_no"; vfmadd231ps %%ymm2,%%ymm0,%%ymm"#c3_no";"\
  "vmovshdup "#boff"(%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm"#c2_no"; vfmadd231ps %%ymm2,%%ymm0,%%ymm"#c4_no";"
#define KERNEL_k1m4n8 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; addq $16,%0;"\
  unit_acc_m4n8(4,5,6,7,0) "addq $32,%1;"
#define KERNEL_k1m4n16 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; addq $16,%0;"\
  unit_acc_m4n8(4,5,6,7,0) unit_acc_m4n8(8,9,10,11,32) "addq $64,%1;"
#define unit_acc_m6n8(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no,boff) \
  "vmovsldup "#boff"(%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm"#c1_no"; vfmadd231ps %%ymm2,%%ymm0,%%ymm"#c3_no"; vfmadd231ps %%ymm3,%%ymm0,%%ymm"#c5_no";"\
  "vmovshdup "#boff"(%1),%%ymm0; vfmadd231ps %%ymm1,%%ymm0,%%ymm"#c2_no"; vfmadd231ps %%ymm2,%%ymm0,%%ymm"#c4_no"; vfmadd231ps %%ymm3,%%ymm0,%%ymm"#c6_no";"
#define KERNEL_k1m6n8 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; vbroadcastsd 16(%0),%%ymm3; addq $24,%0;"\
  unit_acc_m6n8(4,5,6,7,8,9,0) "addq $32,%1;"
#define KERNEL_k1m6n16 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; vbroadcastsd 16(%0),%%ymm3; addq $24,%0;"\
  unit_acc_m6n8(4,5,6,7,8,9,0) unit_acc_m6n8(10,11,12,13,14,15,32) "addq $64,%1;"
#define KERNEL_k2m6n16 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; vbroadcastsd 16(%0),%%ymm3; prefetcht0 256(%1);"\
  unit_acc_m6n8(4,5,6,7,8,9,0) unit_acc_m6n8(10,11,12,13,14,15,32)\
  "vbroadcastsd 24(%0),%%ymm1; vbroadcastsd 32(%0),%%ymm2; vbroadcastsd 40(%0),%%ymm3; prefetcht0 320(%1); addq $48,%0; prefetcht0 384(%0);"\
  unit_acc_m6n8(4,5,6,7,8,9,64) unit_acc_m6n8(10,11,12,13,14,15,96) "addq $128,%1;"
#define save_init_m1 "vbroadcastss (%5),%%ymm0; movq %2,%3; addq $4,%2;"
#define save_init_m2 "vbroadcastss (%5),%%ymm0; movq %2,%3; addq $8,%2;"
#define save_init_m4 "vbroadcastss (%5),%%ymm0; movq %2,%3; addq $16,%2;"
#define save_init_m6 "vbroadcastss (%5),%%ymm0; movq %2,%3; addq $24,%2;"
#define unit_save_m1_dn1_m1(c0_no) \
  "vmovss (%3),%%xmm2; vinsertps $16,(%3,%4,1),%%xmm2,%%xmm2; vfmadd231ps %%xmm"#c0_no",%%xmm0,%%xmm2;"\
  "vmovss %%xmm2,(%3); vextractps $1,%%xmm2,(%3,%4,1);"
#define unit_save_m1n8(c1_no) \
  "vextractf128 $1,%%ymm"#c1_no",%%xmm3;" unit_save_m1_dn1_m1(c1_no) "leaq (%3,%4,2),%3;"\
  "vunpckhpd %%xmm"#c1_no",%%xmm"#c1_no",%%xmm"#c1_no";" unit_save_m1_dn1_m1(c1_no) "leaq (%3,%4,2),%3;"\
  unit_save_m1_dn1_m1(3) "leaq (%3,%4,2),%3;"\
  "vunpckhpd %%xmm3,%%xmm3,%%xmm3;" unit_save_m1_dn1_m1(3) "leaq (%3,%4,2),%3;"
#define SAVE_m1n8 save_init_m1 unit_save_m1n8(4)
#define SAVE_m1n16 SAVE_m1n8 unit_save_m1n8(5)
#define unit_save_m4_dn4_m4(c0_no) \
  "vmovups (%3),%%xmm2; vinsertf128 $1,(%3,%4,4),%%ymm2,%%ymm2;"\
  "vfmadd213ps %%ymm2,%%ymm0,%%ymm"#c0_no"; vmovups %%xmm"#c0_no",(%3); vextractf128 $1,%%ymm"#c0_no",(%3,%4,4);"
#define unit_save_m2_dn1_m2(c0_no,off) \
  "vmovsd "#off"(%3),%%xmm2; vmovhpd "#off"(%3,%4,1),%%xmm2,%%xmm2; vfmadd231ps %%xmm"#c0_no",%%xmm0,%%xmm2;"\
  "vmovsd %%xmm2,"#off"(%3); vmovhpd %%xmm2,"#off"(%3,%4,1);"
#define unit_save_m2_dn2_m2(c0_no,off) \
  "vmovsd "#off"(%3),%%xmm2; vmovhpd "#off"(%3,%4,2),%%xmm2,%%xmm2; vfmadd231ps %%xmm"#c0_no",%%xmm0,%%xmm2;"\
  "vmovsd %%xmm2,"#off"(%3); vmovhpd %%xmm2,"#off"(%3,%4,2);"
#define unit_save_dn2_m2_dn2_m2(c0_no) \
  "vmovsd 16(%3,%4,2),%%xmm2; vmovhpd 16(%3,%4,4),%%xmm2,%%xmm2; vfmadd231ps %%xmm"#c0_no",%%xmm0,%%xmm2;"\
  "vmovsd %%xmm2,16(%3,%4,2); vmovhpd %%xmm2,16(%3,%4,4);"
#define unit_save_m2n8(c1_no,c2_no) \
  "vunpcklps %%ymm"#c2_no",%%ymm"#c1_no",%%ymm1; vunpckhps %%ymm"#c2_no",%%ymm"#c1_no",%%ymm3;"\
  unit_save_m2_dn1_m2(1,0) "leaq (%3,%4,2),%3; vextractf128 $1,%%ymm1,%%xmm1;"\
  unit_save_m2_dn1_m2(3,0) "leaq (%3,%4,2),%3; vextractf128 $1,%%ymm3,%%xmm3;"\
  unit_save_m2_dn1_m2(1,0) "leaq (%3,%4,2),%3;"\
  unit_save_m2_dn1_m2(3,0) "leaq (%3,%4,2),%3;"
#define SAVE_m2n8 save_init_m2 unit_save_m2n8(4,5)
#define SAVE_m2n16 SAVE_m2n8 unit_save_m2n8(6,7)
#define unit_save_m4n8(c1_no,c2_no,c3_no,c4_no) \
  "vunpcklpd %%ymm"#c3_no",%%ymm"#c1_no",%%ymm1;" unit_save_m4_dn4_m4(1) "addq %4,%3;"\
  "vunpcklpd %%ymm"#c4_no",%%ymm"#c2_no",%%ymm1;" unit_save_m4_dn4_m4(1) "addq %4,%3;"\
  "vunpckhpd %%ymm"#c3_no",%%ymm"#c1_no",%%ymm1;" unit_save_m4_dn4_m4(1) "addq %4,%3;"\
  "vunpckhpd %%ymm"#c4_no",%%ymm"#c2_no",%%ymm1;" unit_save_m4_dn4_m4(1) "addq %4,%3; leaq (%3,%4,4),%3;"
#define SAVE_m4n8 save_init_m4 unit_save_m4n8(4,5,6,7)
#define SAVE_m4n16 SAVE_m4n8 unit_save_m4n8(8,9,10,11)
#define unit_save_m6n8(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no) \
  "vunpcklpd %%ymm"#c3_no",%%ymm"#c1_no",%%ymm1;" unit_save_m4_dn4_m4(1) unit_save_m2_dn2_m2(c5_no,16) "addq %4,%3;"\
  "vunpcklpd %%ymm"#c4_no",%%ymm"#c2_no",%%ymm1;" unit_save_m4_dn4_m4(1) unit_save_m2_dn2_m2(c6_no,16) "addq %4,%3;"\
  "vextractf128 $1,%%ymm"#c5_no",%%xmm"#c5_no"; vextractf128 $1,%%ymm"#c6_no",%%xmm"#c6_no";"\
  "vunpckhpd %%ymm"#c3_no",%%ymm"#c1_no",%%ymm1;" unit_save_m4_dn4_m4(1) unit_save_dn2_m2_dn2_m2(c5_no) "addq %4,%3;"\
  "vunpckhpd %%ymm"#c4_no",%%ymm"#c2_no",%%ymm1;" unit_save_m4_dn4_m4(1) unit_save_dn2_m2_dn2_m2(c6_no) "addq %4,%3; leaq (%3,%4,4),%3;"
#define SAVE_m6n8 save_init_m6 unit_save_m6n8(4,5,6,7,8,9)
#define SAVE_m6n16 SAVE_m6n8 unit_save_m6n8(10,11,12,13,14,15)
#define INIT_m1n8 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m1n16 INIT_m1n8 "vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m2n8 INIT_m1n16
#define INIT_m2n16 INIT_m2n8 "vpxor %%ymm6,%%ymm6,%%ymm6; vpxor %%ymm7,%%ymm7,%%ymm7;"
#define INIT_m4n8 INIT_m2n16
#define INIT_m4n16 INIT_m4n8\
  "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9; vpxor %%ymm10,%%ymm10,%%ymm10; vpxor %%ymm11,%%ymm11,%%ymm11;"
#define INIT_m6n8 INIT_m4n8 "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9;"
#define INIT_m6n16 INIT_m6n8\
  "vpxor %%ymm10,%%ymm10,%%ymm10; vpxor %%ymm11,%%ymm11,%%ymm11; vpxor %%ymm12,%%ymm12,%%ymm12;"\
  "vpxor %%ymm13,%%ymm13,%%ymm13; vpxor %%ymm14,%%ymm14,%%ymm14; vpxor %%ymm15,%%ymm15,%%ymm15;"

#define KERNEL_k1m4n1 \
  "vbroadcastss (%1),%%xmm1; addq $4,%1;  vfmadd231ps (%0),%%xmm1,%%xmm4; addq $16,%0;"
#define KERNEL_k1m4n2 \
  "vmovups (%0),%%xmm0; addq $16,%0;"\
  "vbroadcastss (%1),%%xmm1; vfmadd231ps %%xmm0,%%xmm1,%%xmm4;"\
  "vbroadcastss 4(%1),%%xmm2; vfmadd231ps %%xmm0,%%xmm2,%%xmm5; addq $8,%1;"
#define KERNEL_k1m4n4 \
  "vmovsldup (%1),%%xmm1; vmovshdup (%1),%%xmm2; addq $16,%1;"\
  "vmovddup (%0),%%xmm0; vfmadd231ps %%xmm1,%%xmm0,%%xmm4; vfmadd231ps %%xmm2,%%xmm0,%%xmm5;"\
  "vmovddup 8(%0),%%xmm0; vfmadd231ps %%xmm1,%%xmm0,%%xmm6; vfmadd231ps %%xmm2,%%xmm0,%%xmm7; addq $16,%0;"
#define SAVE_m4n1 save_init_m4 "vfmadd213ps (%3),%%xmm0,%%xmm4; vmovups %%xmm4,(%3);"
#define SAVE_m4n2 SAVE_m4n1 "vfmadd213ps (%3,%4,1),%%xmm0,%%xmm5; vmovups %%xmm5,(%3,%4,1);"
#define SAVE_m4n4 save_init_m4\
  "vunpcklpd %%xmm6,%%xmm4,%%xmm1; vfmadd213ps (%3),%%xmm0,%%xmm1; vmovups %%xmm1,(%3);"\
  "vunpcklpd %%xmm7,%%xmm5,%%xmm2; vfmadd213ps (%3,%4,1),%%xmm0,%%xmm2; vmovups %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"\
  "vunpckhpd %%xmm6,%%xmm4,%%xmm1; vfmadd213ps (%3),%%xmm0,%%xmm1; vmovups %%xmm1,(%3);"\
  "vunpckhpd %%xmm7,%%xmm5,%%xmm2; vfmadd213ps (%3,%4,1),%%xmm0,%%xmm2; vmovups %%xmm2,(%3,%4,1);"
#define INIT_m4n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m4n2 INIT_m4n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define KERNEL_k1m6n1 \
  "vbroadcastss (%1),%%xmm1; addq $4,%1;"\
  "vfmadd231ps (%0),%%xmm1,%%xmm4; vfmadd231ps 8(%0),%%xmm1,%%xmm5; addq $24,%0;"
#define KERNEL_k1m6n2 \
  "vbroadcastss (%1),%%xmm1; vbroadcastss 4(%1),%%xmm2; addq $8,%1;"\
  "vmovups (%0),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm4; vfmadd231ps %%xmm2,%%xmm3,%%xmm6;"\
  "vmovsd 16(%0),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm5; vfmadd231ps %%xmm2,%%xmm3,%%xmm7; addq $24,%0;"
#define KERNEL_k1m6n4 \
  "vmovsldup (%1),%%xmm1; vmovshdup (%1),%%xmm2; addq $16,%1;"\
  "vmovddup (%0),%%xmm0; vfmadd231ps %%xmm1,%%xmm0,%%xmm4; vfmadd231ps %%xmm2,%%xmm0,%%xmm5;"\
  "vmovddup 8(%0),%%xmm0; vfmadd231ps %%xmm1,%%xmm0,%%xmm6; vfmadd231ps %%xmm2,%%xmm0,%%xmm7;"\
  "vmovddup 16(%0),%%xmm0; vfmadd231ps %%xmm1,%%xmm0,%%xmm8; vfmadd231ps %%xmm2,%%xmm0,%%xmm9; addq $24,%0;"
#define SAVE_m6n1 save_init_m6 "vfmadd213ps (%3),%%xmm0,%%xmm4; vmovsd %%xmm4,(%3); vfmadd213ps 8(%3),%%xmm0,%%xmm5; vmovups %%xmm5,8(%3);"
#define SAVE_m6n2 save_init_m6\
  "vfmadd213ps (%3),%%xmm0,%%xmm4; vmovups %%xmm4,(%3); vmovsd 16(%3),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm5; vmovsd %%xmm5,16(%3);"\
  "vfmadd213ps (%3,%4,1),%%xmm0,%%xmm6; vmovups %%xmm6,(%3,%4,1); vmovsd 16(%3,%4,1),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm7; vmovsd %%xmm7,16(%3,%4,1);"
#define SAVE_m6n4 save_init_m6\
  "vunpcklpd %%xmm6,%%xmm4,%%xmm1; vfmadd213ps (%3),%%xmm0,%%xmm1; vmovups %%xmm1,(%3);"\
  "vunpckhpd %%xmm6,%%xmm4,%%xmm3; vfmadd213ps (%3,%4,2),%%xmm0,%%xmm3; vmovups %%xmm3,(%3,%4,2);" unit_save_m2_dn2_m2(8,16) "addq %4,%3;"\
  "vunpcklpd %%xmm7,%%xmm5,%%xmm1; vfmadd213ps (%3),%%xmm0,%%xmm1; vmovups %%xmm1,(%3);"\
  "vunpckhpd %%xmm7,%%xmm5,%%xmm3; vfmadd213ps (%3,%4,2),%%xmm0,%%xmm3; vmovups %%xmm3,(%3,%4,2);" unit_save_m2_dn2_m2(9,16)
#define INIT_m6n1 "vpxor %%xmm4,%%xmm4,%%xmm4; vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m6n2 INIT_m6n1 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m6n4 INIT_m6n2 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define KERNEL_k1m2n4 \
  "vmovups (%1),%%xmm0; addq $16,%1;"\
  "vbroadcastss (%0),%%xmm1; vfmadd231ps %%xmm0,%%xmm1,%%xmm4;"\
  "vbroadcastss 4(%0),%%xmm2; vfmadd231ps %%xmm0,%%xmm2,%%xmm5; addq $8,%0;"
#define KERNEL_k1m2n2 \
  "vmovsd (%0),%%xmm0; addq $8,%0;"\
  "vbroadcastss (%1),%%xmm1; vfmadd231ps %%xmm0,%%xmm1,%%xmm4;"\
  "vbroadcastss 4(%1),%%xmm2; vfmadd231ps %%xmm0,%%xmm2,%%xmm5; addq $8,%1;"
#define KERNEL_k1m2n1 \
  "vbroadcastss (%1),%%xmm1; addq $4,%1;"\
  "vmovsd (%0),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4; addq $8,%0;"
#define SAVE_m2n1 save_init_m2 "vmovsd (%3),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm4; vmovsd %%xmm4,(%3);"
#define SAVE_m2n2 SAVE_m2n1 "vmovsd (%3,%4,1),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm5; vmovsd %%xmm5,(%3,%4,1);"
#define SAVE_m2n4 save_init_m2\
  "vunpcklps %%xmm5,%%xmm4,%%xmm1;" unit_save_m2_dn1_m2(1,0) "leaq (%3,%4,2),%3;"\
  "vunpckhps %%xmm5,%%xmm4,%%xmm1;" unit_save_m2_dn1_m2(1,0)
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m2n4 INIT_m2n2
#define KERNEL_k1m1n1 "vmovss (%1),%%xmm1; addq $4,%1; vfmadd231ss (%0),%%xmm1,%%xmm4; addq $4,%0;"
#define KERNEL_k1m1n2 "vmovsd (%1),%%xmm1; addq $8,%1; vbroadcastss (%0),%%xmm2; vfmadd231ps %%xmm2,%%xmm1,%%xmm4; addq $4,%0;"
#define KERNEL_k1m1n4 "vbroadcastss (%0),%%xmm2; addq $4,%0; vfmadd231ps (%1),%%xmm2,%%xmm4; addq $16,%1;"
#define SAVE_m1n1 save_init_m1 "vfmadd213ss (%3),%%xmm0,%%xmm4; vmovss %%xmm4,(%3);"
#define SAVE_m1n2 save_init_m1 unit_save_m1_dn1_m1(4)
#define SAVE_m1n4 SAVE_m1n2 "leaq (%3,%4,2),%3; vunpckhpd %%xmm4,%%xmm4,%%xmm4;" unit_save_m1_dn1_m1(4)
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n2 INIT_m1n1
#define INIT_m1n4 INIT_m1n1

/* %6=k_counter, %7=b_pref */
/* r12=tmp, r13=k, r14=b_head */
#define COMPUTE_SIMPLE(mdim,ndim) \
  "testq %%r13,%%r13; jz 5"#mdim"55"#ndim"5f;"\
  "movq %%r13,%6; movq %%r14,%1;" INIT_m##mdim##n##ndim\
  "5"#mdim"55"#ndim"7:\n\t"\
  KERNEL_k1m##mdim##n##ndim "decq %6; jnz 5"#mdim"55"#ndim"7b;"\
  "5"#mdim"55"#ndim"5:\n\t"\
  SAVE_m##mdim##n##ndim
#define COMPUTE_m6n1 COMPUTE_SIMPLE(6,1)
#define COMPUTE_m6n2 COMPUTE_SIMPLE(6,2)
#define COMPUTE_m6n4 COMPUTE_SIMPLE(6,4)
#define COMPUTE_m6n8 COMPUTE_SIMPLE(6,8)
#define COMPUTE_m6n16 \
  "movq %%r13,%6; movq %%r14,%1;" INIT_m6n16\
  "cmpq $16,%6; jb 5655165f; movq %2,%3; testq %%r12,%%r12;"\
  "5655167:\n\t"\
  KERNEL_k2m6n16 "cmpq $46,%%r12; movq $46,%%r12; cmoveq %4,%%r12;"\
  KERNEL_k2m6n16 "prefetcht1 (%3); subq $23,%3; addq %%r12,%3;"\
  KERNEL_k2m6n16 "prefetcht1 (%7); addq $16,%7;"\
  KERNEL_k2m6n16 "subq $8,%6; cmpq $16,%6; jnb 5655167b;"\
  "5655165:\n\t"\
  "movq %2,%3; prefetcht0 (%5); testq %6,%6; jz 5655169f;"\
  "5655163:\n\t"\
  "prefetcht0 (%3); prefetcht0 23(%3); prefetcht0 (%3,%4,1); prefetcht0 23(%3,%4,1);"\
  KERNEL_k1m6n16 "leaq (%3,%4,2),%3; decq %6; jnz 5655163b;"\
  "5655169:\n\t"\
  "prefetcht0 (%%r14); prefetcht0 64(%%r14); prefetcht0 128(%%r14); prefetcht0 192(%%r14);" SAVE_m6n16

/* r11=m_counter */
#define COMPUTE(ndim) {\
  b_pref=b_ptr+ndim*ldc;\
  __asm__ __volatile__(\
  "movq %1,%%r14; movq %6,%%r13; movq %8,%%r11;"\
  "cmpq $6,%%r11; jb 99301f;"\
  "99300:\n\t"\
  COMPUTE_m6n##ndim "subq $6,%%r11; cmpq $6,%%r11; jnb 99300b;"\
  "99301:\n\t"\
  "cmpq $4,%%r11; jb 99302f;"\
  COMPUTE_SIMPLE(4,ndim) "subq $4,%%r11;"\
  "99302:\n\t"\
  "cmpq $2,%%r11; jb 99303f;"\
  COMPUTE_SIMPLE(2,ndim) "subq $2,%%r11;"\
  "99303:\n\t"\
  "testq %%r11,%%r11; jz 99304f;"\
  COMPUTE_SIMPLE(1,ndim)\
  "99304:\n\t"\
  "movq %%r13,%6; movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(alp),"+r"(K),"+r"(b_pref)\
  :"m"(M):"xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7",\
  "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15",\
  "r11","r12","r13","r14","cc","memory");\
  a_ptr-=M*K; b_ptr+=K*ndim; c_ptr+=ldc*ndim-M;\
}

//#include "common.h"
#include <stdint.h>
#include <stdio.h>//debug
#include <stdlib.h>//debug
#include <string.h>
#define BLASLONG int//debug
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alpha, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, BLASLONG ldc)
{
    if(m==0||n==0||k==0||alpha==(float)0.0) return 0;
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(float); float ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    float *a_ptr = A,*b_ptr = B,*c_ptr = C,*c_tmp = C,*alp = &ALPHA,*b_pref = B;
    for(;n_count>15;n_count-=16) COMPUTE(16)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}
/* test zone */
static void sgemm_tcopy_6(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 6;
      for(count_first=dim_first;count_first>5;count_first-=6){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];
        tosrc+=6;todst+=6*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>3;count_first-=4){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void sgemm_ncopy_6(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6;
    float *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;
    tosrc_inc=6*lead_dim-dim_first;
    for(count_second=dim_second;count_second>5;count_second-=6){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst+=6;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
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
static void sgemm_tcopy_16(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 16;
      for(count_first=dim_first;count_first>15;count_first-=16){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];todst[6]=tosrc[6];todst[7]=tosrc[7];
        todst[8]=tosrc[8];todst[9]=tosrc[9];todst[10]=tosrc[10];todst[11]=tosrc[11];
        todst[12]=tosrc[12];todst[13]=tosrc[13];todst[14]=tosrc[14];todst[15]=tosrc[15];
        tosrc+=16;todst+=16*dim_second;
      }
      todst -= count_second * 8;
      for(;count_first>7;count_first-=8){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];todst[6]=tosrc[6];todst[7]=tosrc[7];
        tosrc+=8;todst+=8*dim_second;
      }
      todst -= count_second * 4;
      for(;count_first>3;count_first-=4){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void sgemm_ncopy_16(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8;
    float *tosrc9,*tosrc10,*tosrc11,*tosrc12,*tosrc13,*tosrc14,*tosrc15,*tosrc16;
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
/*
static void SCALE_MULT(float *dat,float *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || (*sca)==(float)1.0) return;
    float scale = *sca; float *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
*/
template<unsigned	transa,	unsigned	transb,	unsigned	m,	unsigned	n,	unsigned	k,	unsigned	lda,	unsigned	ldb,	unsigned	ldc,	unsigned	beta>
void inline sgemm(float alpha,float *a,float *b,float *c){
	const	unsigned	BLOCKDIM=256;
	float *b_buffer = (float *)aligned_alloc(64,BLOCKDIM*n*sizeof(float));
    float *a_buffer = (float *)aligned_alloc(4096,BLOCKDIM*BLOCKDIM*sizeof(float));
	float *a_current_pos,*b_current_pos=b;
    unsigned m_count,k_count,k_subdim,m_subdim;
	if(beta==0)	memset(c,0,m*n*sizeof(float));
    for(k_count=0;k_count<k;k_count+=BLOCKDIM){
      k_subdim = k-k_count;
      if(k_subdim > BLOCKDIM) k_subdim = BLOCKDIM;
      if(!transb) { sgemm_ncopy_16(b_current_pos,b_buffer,ldb,k_subdim,n); b_current_pos += BLOCKDIM; }
      else { sgemm_tcopy_16(b_current_pos,b_buffer,ldb,n,k_subdim); b_current_pos += (int64_t)(ldb) * BLOCKDIM; }
      if(!transa) a_current_pos = a + (int64_t)k_count * (int64_t)(lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<m;m_count+=BLOCKDIM){
        m_subdim = m-m_count;
        if(m_subdim > BLOCKDIM) m_subdim = BLOCKDIM;
        if(!transa) { sgemm_tcopy_6(a_current_pos,a_buffer,lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM; }
        else { sgemm_ncopy_6(a_current_pos,a_buffer,lda,k_subdim,m_subdim); a_current_pos += (int64_t)(lda) * BLOCKDIM; }
        CNAME(m_subdim,n,k_subdim,alpha,a_buffer,b_buffer,c+m_count,ldc);
      }
    }
    free(a_buffer);	free(b_buffer);
}

