#ifndef TSMJ_H_INCLUDED
#define TSMJ_H_INCLUDED

void CSE_init(int aam,int bam,int cam,int dam);
int CSE_setup_delta_list(int ipq, int index, int ia,int ib,int ia2,int ib2);
void CSE_generate_delta_list(int ipq,int aam,int bam);
void CSE_finish();

void MD_set_QR_table(int ipq,int aam, int bam, int cam, int dam);
void MD_zero_QR_table();

void TSMJ_declare_aPin(FILE * fp, int ipq, int aan, int ban);

void MD_d_declare(int ipq,FILE * fp,int aan, int ban,bool b_dformat);
void TSMJ_d_expression_0(int ipq,FILE * fp,int aan, int ban,bool b_smalld);
void MD_d_expression_0(int ipq,FILE * fp,int aam, int bam,bool b_smalld,bool b_CSE);

void TSMJ_D_expression_1(int ipq,FILE * fp,int aan, int ban);

void TSMJ_delta_declare_0(int ipq,FILE * fp,int type,int delta_type);
void TSMJ_delta_declare_1(int ipq,FILE * fp,int type,int delta_type,bool only_q);

void TSMJ_delta_expression_0(int ipq,FILE * fp,int delta_type,int type);
void TSMJ_delta_expression_1(int ipq,FILE * fp,int delta_type,int type,bool only_q,bool b_smalld);

void MD_R_init(int aan, int ban, int can, int dan);
void TSMJ_R_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,bool b_smalld,bool b_fullr);

void MD_QR_declare(FILE * fp,int aan, int ban, int can, int dan,int ipq);
void TSMJ_QR_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,\
						bool outer_zero,bool b_declare,bool summation,bool b_delta,\
						bool b_smalld,bool b_fullr,bool b_PmQR,bool b_aPinQR,bool b_QRsum);
void TSMJ_QR_declare(FILE * fp, int aam, int bam, int cam, int dam,int ipq,bool b_JME);

void TSMJ_ans_expression(FILE * fp,bool J_type, int aam, int bam, int cam, int dam,int ipq,\
						bool inner_zero,int output_P,bool b_delta,bool b_smalld,bool b_fullr,bool b_PmQR,bool b_aPinQR,bool b_QRsum);
void zero_R_mark_table();

void set_d_mark_table();
void check_d_mark_table(int ipq, int ixyz,int np, int na, int nb);
void check_d_mark_table(int aam,int bam,int cam,int dam);
void zero_d_mark_table();

void TSMJ_output_J(FILE * fp,\
                     int aam, int bam, int cam, int dam,\
                     bool bDform,bool bNRR, bool bCSE,\
                     bool b_Tex,bool b_JME,\
                     char * tail,char * ft_tail,int ipq_inner_loop);
void HGP_gpu(FILE * fp,int aam,int bam,int cam,int dam);
void TSMJ_output_K(FILE * fp,\
                     int aam, int bam, int cam, int dam,\
                     bool b_Dform,bool b_NRR, bool b_CSE,\
                     char * tail,char * ft_tail,int ipq_inner_loop);

void MD_set_d_mark_table(int ipq,int aam,int bam);
void MD_output_J(FILE * fp,int aam, int bam, int cam, int dam,char * tail,int ipq_inner_loop);
void MD_delta_expression(int ipq,FILE * fp,int aam,int bam,bool b_smalld);
void MD_QR_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,bool outer_zero,bool b_declare,bool summation,bool b_delta,bool b_smalld,bool b_fullr);
void MD_ans_expression(FILE * fp,bool J_type, int aam, int bam, int cam, int dam,int ipq,bool inner_zero,int output_P,bool b_delta,bool b_smalld,bool b_fullr);
void MD_output_K(FILE * fp,\
                     int aam, int bam, int cam, int dam,char * tail,int ipq_inner_loop);
void output_include(FILE * fp);
void output_define_texture_memory(FILE * fp,bool b_Jmtrx, int ipq,int aam,int bam);
void output_define_texture_memory_id(FILE * fp,int ipq);
void output_texture_function(FILE * fp,bool b_Jmtrx,char * head, int ipq,int aam,int bam);
void output_texture_function_K(FILE * fp,bool b_Jmtrx,char * head,int ipq,int aam,int bam);
void output_kernel_define(FILE * fp,char * name_string,char * tail_string,int aam, int bam, int cam, int dam);
#endif // TSMJ_H_INCLUDED
