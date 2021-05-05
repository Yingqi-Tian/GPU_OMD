#include<stdio.h>
#include<memory.h>
#include"TSMJ.h"
#include"global.h"
#include"expression.h"
#define SUBLEN 150
#define ONLY_FT true

extern char shell_name[5];
extern delta_mark * P_delta_mark_array;
extern delta_mark * Q_delta_mark_array;
extern int P_delta_mark_sum;
extern int Q_delta_mark_sum;
extern expression_encode * P_expression_encode_list;
extern expression_encode * Q_expression_encode_list;
extern int P_expression_encode_list_sum;
extern int Q_expression_encode_list_sum;
extern int P_d_mark_table[3][MAX_P][MAX_AMP][MAX_AMP];
extern int Q_d_mark_table[3][MAX_P][MAX_AMP][MAX_AMP];
extern int QR_table[MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_P][MAX_P][MAX_P];
extern int MD_R_mark_table[MAX_J][MAX_J][MAX_J][MAX_J];
extern char c_ABCD[4];
extern char c_PQ[2];
extern char c_ZE[2];
extern char c_AC[2];
extern char c_BD[2];
extern char c_XYZ[3];
char c_PQs[2]={'p','q'};
char s_bk[2][4]={"bra","ket"};
char s_ii[2][3]={"ii","jj"};
char c_ij[2]={'i','j'};
char c_xy[2]={'x','y'};

int get_id_j(int aam,int bam);
int get_id_k(int aam,int bam);

void get_MD_nlm(int aam, int ia, int * na, int * la, int * ma);

int shell_len[5]={1,3,6,10,15};


void output_include(FILE * fp){
	fprintf(fp,"\
#include<math.h>\n\
#include\"Boys_gpu.h\"\n\
#define PI 3.1415926535897932\n\
#define P25 17.4934183276248620\n\
#define NTHREAD_32 32\n\
#define NTHREAD_64 64\n\
#define NTHREAD_128 128\n");
}

void output_define_texture_memory(FILE * fp,bool b_Jmtrx,int ipq,int aam,int bam){
	fprintf(fp,"\
texture<int2,1,cudaReadModeElementType> tex_%c;\n\
texture<int2,1,cudaReadModeElementType> tex_%cta;\n\
texture<int2,1,cudaReadModeElementType> tex_p%c;\n\
texture<float,1,cudaReadModeElementType> tex_K2_%c;\n",c_PQ[ipq],c_ZE[ipq],c_PQs[ipq],c_PQs[ipq]);
    if(aam>0) fprintf(fp,"\
texture<int2,1,cudaReadModeElementType> tex_%c%c;\n",c_PQ[ipq],c_AC[ipq]);
    if(bam>0) fprintf(fp,"\
texture<int2,1,cudaReadModeElementType> tex_%c%c;\n",c_PQ[ipq],c_BD[ipq]);
    if(b_Jmtrx){
        fprintf(fp,"\
texture<int2,1,cudaReadModeElementType> tex_Pmtrx;\n");
    }
}

void output_define_texture_memory_id(FILE * fp,int ipq){
    fprintf(fp,"\
texture<unsigned int,1,cudaReadModeElementType> tex_id_%s;\n",s_bk[ipq]);
}

void output_texture_function(FILE * fp,bool b_Jmtrx,char * head,int ipq,int aam,int bam){
	fprintf(fp,"\n\
void %stexture_binding_%c%c(double * %c_d,double * %c%c_d,double * %c%c_d,\\\n\
        double * alpha%c_d,double * p%c_d,float * K2_%c_d,",head,\
         shell_name[aam],shell_name[bam],c_PQ[ipq],c_PQ[ipq],c_AC[ipq],c_PQ[ipq],c_BD[ipq],c_PQ[ipq],c_PQs[ipq],c_PQs[ipq]);
	if(b_Jmtrx) fprintf(fp,"double * Pmtrx_d,\\\n");
	else fprintf(fp,"\\\n");
	fprintf(fp,"\
        unsigned int contrc_ket_start_pr,unsigned int primit_ket_len,unsigned int contrc_Pmtrx_start_pr){\n\
    cudaBindTexture(0, tex_%c, %c_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);\n\
    cudaBindTexture(0, tex_%cta, alpha%c_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));\n\
    cudaBindTexture(0, tex_p%c, p%c_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));\n\
    cudaBindTexture(0, tex_K2_%c, K2_%c_d, sizeof(float)*(contrc_ket_start_pr+primit_ket_len));\n",\
         c_PQ[ipq],c_PQ[ipq],\
         c_ZE[ipq],c_PQ[ipq],\
         c_PQs[ipq],c_PQs[ipq],\
         c_PQs[ipq],c_PQs[ipq]);
    if(aam>0) fprintf(fp,"\
    cudaBindTexture(0, tex_%c%c, %c%c_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);\n",\
         c_PQ[ipq],c_AC[ipq],c_PQ[ipq],c_AC[ipq]);
    if(bam>0) fprintf(fp,"\
    cudaBindTexture(0, tex_%c%c, %c%c_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);\n",\
         c_PQ[ipq],c_BD[ipq],c_PQ[ipq],c_BD[ipq]);
	if(b_Jmtrx) fprintf(fp,"\
    cudaBindTexture(0, tex_Pmtrx, Pmtrx_d, sizeof(double)*(contrc_Pmtrx_start_pr+primit_ket_len)*%d);\n",shell_len[aam]*shell_len[bam]);
	else fprintf(fp,"\\\n");
	fprintf(fp,"\
}\n\
\n");
fprintf(fp,"\
void %stexture_unbind_%c%c(){\n\
    cudaUnbindTexture(tex_%c);\n\
    cudaUnbindTexture(tex_%cta);\n\
    cudaUnbindTexture(tex_p%c);\n\
    cudaUnbindTexture(tex_K2_%c);\n",head,\
         shell_name[aam],shell_name[bam],\
         c_PQ[ipq],\
         c_ZE[ipq],\
         c_PQs[ipq],\
         c_PQs[ipq]);
    if(aam>0) fprintf(fp,"\
    cudaUnbindTexture(tex_%c%c);\n",\
         c_PQ[ipq],c_AC[ipq]);
    if(bam>0) fprintf(fp,"\
    cudaUnbindTexture(tex_%c%c);\n",\
         c_PQ[ipq],c_BD[ipq]);
	if(b_Jmtrx) fprintf(fp,"\
    cudaUnbindTexture(tex_Pmtrx);\n");
	else fprintf(fp,"\\\n");
	fprintf(fp,"\
\n\
}\n");
}


void output_texture_function_K(FILE * fp,bool b_Jmtrx,char * head,int ipq,int aam,int bam){
	fprintf(fp,"\n\
void %stexture_binding_%s_%c%c(double * %c_d,double * %c%c_d,double * %c%c_d,\\\n\
        double * alpha%c_d,double * p%c_d,float * K2_%c_d,unsigned int * id_%s_d,",head,\
         s_bk[ipq],shell_name[aam],shell_name[bam],\
         c_PQ[ipq],c_PQ[ipq],c_AC[ipq],c_PQ[ipq],c_BD[ipq],\
         c_PQ[ipq],c_PQs[ipq],c_PQs[ipq],s_bk[ipq]);
	if(b_Jmtrx) fprintf(fp,"double * Pmtrx_d,\\\n");
	else fprintf(fp,"\\\n");
	fprintf(fp,"\
        unsigned int primit_len){\n\
    cudaBindTexture(0, tex_%c, %c_d, sizeof(double)*primit_len*3);\n\
    cudaBindTexture(0, tex_%cta, alpha%c_d, sizeof(double)*primit_len);\n\
    cudaBindTexture(0, tex_p%c, p%c_d, sizeof(double)*primit_len);\n\
    cudaBindTexture(0, tex_K2_%c, K2_%c_d, sizeof(float)*primit_len);\n",\
         c_PQ[ipq],c_PQ[ipq],\
         c_ZE[ipq],c_PQ[ipq],\
         c_PQs[ipq],c_PQs[ipq],\
         c_PQs[ipq],c_PQs[ipq]);
    fprintf(fp,"\
    cudaBindTexture(0, tex_%c%c, %c%c_d, sizeof(double)*primit_len*3);\n",\
         c_PQ[ipq],c_AC[ipq],c_PQ[ipq],c_AC[ipq]);
    fprintf(fp,"\
    cudaBindTexture(0, tex_%c%c, %c%c_d, sizeof(double)*primit_len*3);\n",\
         c_PQ[ipq],c_BD[ipq],c_PQ[ipq],c_BD[ipq]);
    fprintf(fp,"\
    cudaBindTexture(0, tex_id_%s, id_%s_d, sizeof(unsigned int)*primit_len);\n",\
         s_bk[ipq],s_bk[ipq]);
    fprintf(fp,"\n}\n");

	fprintf(fp,"\
void %stexture_unbind_%s_%c%c(){\n\
    cudaUnbindTexture(tex_%c);\n\
    cudaUnbindTexture(tex_%cta);\n\
    cudaUnbindTexture(tex_p%c);\n\
    cudaUnbindTexture(tex_K2_%c);\n",head,\
         s_bk[ipq],shell_name[aam],shell_name[bam],\
         c_PQ[ipq],\
         c_ZE[ipq],\
         c_PQs[ipq],\
         c_PQs[ipq]);
    fprintf(fp,"\
    cudaUnbindTexture(tex_%c%c);\n",\
         c_PQ[ipq],c_AC[ipq]);
    fprintf(fp,"\
    cudaUnbindTexture(tex_%c%c);\n",\
         c_PQ[ipq],c_BD[ipq]);
    fprintf(fp,"\
    cudaUnbindTexture(tex_id_%s);\n",\
         s_bk[ipq]);
    fprintf(fp,"\n}\n");
}

void output_kernel_define(FILE * fp,char * name_string,char * tail_string,int aam, int bam, int cam, int dam){
	fprintf(fp,"\
__global__ void %s_%c%c%c%c%s(unsigned int contrc_bra_num,unsigned int primit_ket_len,\\\n\
                unsigned int contrc_bra_start_pr,\\\n\
                unsigned int contrc_ket_start_pr,\\\n\
                unsigned int contrc_Pmtrx_start_pr,\\\n\
                unsigned int * contrc_bra_id,\\\n\
                double * P,\\\n\
                double * PA,\\\n\
                double * PB,\\\n\
                double * Zta,\\\n\
                double * pp,\\\n\
                double * Q,\\\n\
                double * QC,\\\n\
                double * QD,\\\n\
                double * Eta,\\\n\
                double * pq,\\\n\
                double * K2_p, double * K2_q,\\\n\
                double * Pmtrx_in,\\\n\
                double * ans)",name_string,\
                shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail_string);
}

void output_J_kernel_define_tex(FILE * fp,char * name_string,char * tail_string,int aam, int bam, int cam, int dam,bool b_Tex){
	fprintf(fp,"\
__global__ void %s_%c%c%c%c%s(unsigned int contrc_bra_num,unsigned int primit_ket_len,\\\n\
                unsigned int contrc_bra_start_pr,\\\n\
                unsigned int contrc_ket_start_pr,\\\n\
                unsigned int contrc_Pmtrx_start_pr,\\\n\
                unsigned int * contrc_bra_id,\\\n\
                double * P,\\\n\
                double * PA,\\\n\
                double * PB,\\\n\
                double * Zta_in,\\\n\
                double * pp_in,\\\n\
                float * K2_p,\\\n",name_string,\
                shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail_string);
	if(!b_Tex) fprintf(fp,"\
                double * Q,\\\n\
                double * QC,\\\n\
                double * QD,\\\n\
                double * Eta_in,\\\n\
                double * pq_in,\\\n\
                float * K2_q_in,\\\n\
                double * Pmtrx_in,\\\n");
	fprintf(fp,"\
                double * ans)");
}

void output_K_kernel_define_tex(FILE * fp,char * name_string,char c_kpq,char * tail_string,int aam, int bam, int cam, int dam){
    int ipq=0;

	fprintf(fp,"\
__global__ void %s%c_%c%c%c%c%s",name_string,c_kpq,\
                shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail_string);
	fprintf(fp,"(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\\\n\
                unsigned int * contrc_bra_id,\\\n\
                unsigned int * contrc_ket_id,\\\n\
                unsigned int mtrx_len,\\\n\
                double * Pmtrx_in,\\\n\
                double * P,\\\n\
                double * PA,\\\n\
                double * PB,\\\n\
                double * Zta_in,\\\n\
                double * pp_in,\\\n\
                float * K2_p_in,\\\n\
                unsigned int * id_bra_in,\\\n\
                double * Q,\\\n\
                double * QC,\\\n\
                double * QD,\\\n\
                double * Eta_in,\\\n\
                double * pq_in,\\\n\
                float * K2_q_in,\\\n\
                unsigned int * id_ket_in,\\\n\
                double * ans)");
}

void output_id_init(FILE * fp){
	fprintf(fp,"{\n\n\
    unsigned int tId_x = threadIdx.x;\n\
    unsigned int bId_x = blockIdx.x;\n\
    unsigned int tdis = blockDim.x;\n\
    unsigned int bdis = gridDim.x;\n\
    unsigned int ans_id=tId_x;\n");
}

void output_K_id_init(FILE * fp){
	fprintf(fp,"{\n\n\
    unsigned int tId_x = threadIdx.x;\n\
    unsigned int bId_x = blockIdx.x;\n\
    unsigned int bId_y = blockIdx.y;\n\
    unsigned int tdis = blockDim.x;\n\
    unsigned int bdis_x = gridDim.x;\n\
    unsigned int bdis_y = gridDim.y;\n\
    unsigned int ans_id=tId_x;\n");
}

void output_define_Pmtrx(FILE * fp,int len){
	fprintf(fp,"\
    double Pmtrx[%d]={0.0};\n",len);
}

void output_define_ans_temp(FILE * fp,int nthread,int len){
	fprintf(fp,"\
\n\
    __shared__ double ans_temp[NTHREAD_%d*%d];\n\
    for(int i=0;i<%d;i++){\n\
        ans_temp[i*tdis+tId_x]=0.0;\n\
    }\n\
\n",nthread,len,len);
}
void output_contrc_bra_loop(FILE * fp){
	fprintf(fp,"\
\n\
    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){\n\
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];\n\
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];\n");
}
void output_primit_ket_loop(FILE * fp){
	fprintf(fp,"\
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){\n\
            unsigned int jj=contrc_ket_start_pr+j;\n");
}
void output_K_get_Pmtrx(FILE * fp,int bam,int dam){
    if(bam<=2 && dam <=2){
	fprintf(fp,"\
            double P_max=0.0;\n\
            for(int p_j=0;p_j<%d;p_j++){\n\
            for(int p_i=0;p_i<%d;p_i++){\n\
                Pmtrx[p_i*%d+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];\n\
                double temp_P=fabsf(Pmtrx[p_i*%d+p_j]);\n\
                if(temp_P>P_max) P_max=temp_P;\n\
            }\n\
            }\n",shell_len[dam],shell_len[bam],shell_len[dam],shell_len[dam],shell_len[dam]);
    }
    else{
	fprintf(fp,"\
            double P_max=1.0;\n\
            for(int p_i=0;p_i<%d;p_i++){\n\
            for(int p_j=0;p_j<%d;p_j++){\n\
                Pmtrx[p_i*%d+p_j]=1.0;\n\
            }\n\
            }\n",shell_len[bam],shell_len[dam],shell_len[dam],shell_len[dam],shell_len[dam]);
    }
}

void output_Pmtrx_screening(FILE * fp){
	fprintf(fp,"\
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;\n");
}
void output_primit_bra_loop(FILE * fp){
	fprintf(fp,"\
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){\n");
}
void output_schwartz_screening(FILE * fp,bool b_direct,bool b_break,int ipq){
    if(b_direct){
        fprintf(fp,"\
                if(fabsf(K2_p%s*K2_q%s)<1.0E-14){\n",(ipq==0)?"":"[ii]",(ipq==0)?"[jj]":"");
    }
    else{
        fprintf(fp,"\
                if(fabsf(K2_p*K2_q)<1.0E-14){\n");
    }
	if(b_break) fprintf(fp,"\
                    break;\n");
    else fprintf(fp,"\
                    primit_bra_end=ii;\n\
                    continue;\n");
	fprintf(fp,"\
                }\n");
}

void output_define_aPin(FILE * fp,int ipq,int aam,int bam,bool b_MDQR){
    if(aam>0 || bam>0) fprintf(fp,"\t\t\t\tdouble a%cin1=1/(2*%cta);\n",c_PQ[ipq],c_ZE[ipq]);
	if(!b_MDQR) TSMJ_declare_aPin(fp,ipq,aam,bam);
}

void output_getK2_tex(FILE * fp,int ipq,int i_ij,bool b_tex){
    if(b_tex){
    fprintf(fp,"\
            float K2_%c=tex1Dfetch(tex_K2_%c,%s);\n",
                c_PQs[ipq],c_PQs[ipq],s_ii[i_ij]);
    }
    else{
    fprintf(fp,"\
            float K2_%c=K2_%c_in[%s];\n",
                c_PQs[ipq],c_PQs[ipq],s_ii[i_ij]);
    }
}

void output_getPQ_tex(FILE * fp,int ipq,int i_ij,int aam,int bam,bool b_tex,bool b_smalld){
    if(b_tex){
        fprintf(fp,"\
            int2 temp_int2;\n\
            temp_int2=tex1Dfetch(tex_%cta,%s);\n\
            double %cta=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_p%c,%s);\n\
            double p%c=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c,%s*3+0);\n\
            double %cX=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c,%s*3+1);\n\
            double %cY=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c,%s*3+2);\n\
            double %cZ=__hiloint2double(temp_int2.y,temp_int2.x);\n",\
                c_ZE[ ipq],s_ii[i_ij],c_ZE[ ipq],\
                c_PQs[ipq],s_ii[i_ij],c_PQs[ipq],\
                c_PQ[ipq],s_ii[i_ij],c_PQ[ipq] ,\
                c_PQ[ipq],s_ii[i_ij],c_PQ[ipq] ,\
                c_PQ[ipq],s_ii[i_ij],c_PQ[ipq] );
        if(aam>0){
            fprintf(fp,"\t\t\t\tdouble %c%c_010[3];\n",c_PQ[ipq],(b_smalld)?'d':'D');
        fprintf(fp,"\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+0);\n\
            %c%c_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+1);\n\
            %c%c_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+2);\n\
            %c%c_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);\n",\
                c_PQ[ipq],c_AC[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' ,\
                c_PQ[ipq],c_AC[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' ,\
                c_PQ[ipq],c_AC[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' );
        }
        if(bam>0){
            fprintf(fp,"\t\t\t\tdouble %c%c_001[3];\n",c_PQ[ipq],(b_smalld)?'d':'D');
        fprintf(fp,"\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+0);\n\
            %c%c_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+1);\n\
            %c%c_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);\n\
            temp_int2=tex1Dfetch(tex_%c%c,%s*3+2);\n\
            %c%c_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);\n",\
                c_PQ[ipq],c_BD[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' ,\
                c_PQ[ipq],c_BD[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' ,\
                c_PQ[ipq],c_BD[ipq],s_ii[i_ij],c_PQ[ipq],(b_smalld)?'d':'D' );
        }
    }
    else{
        fprintf(fp,"\
\t\t\t\tdouble %cX=%c[%s*3+0];\n\
\t\t\t\tdouble %cY=%c[%s*3+1];\n\
\t\t\t\tdouble %cZ=%c[%s*3+2];\n",\
                c_PQ[ipq],c_PQ[ipq],s_ii[i_ij],\
                c_PQ[ipq],c_PQ[ipq],s_ii[i_ij],\
                c_PQ[ipq],c_PQ[ipq],s_ii[i_ij]);
    if(aam>0){
        fprintf(fp,"\
\t\t\t\tdouble %c%c_010[3];\n\
\t\t\t\t%c%c_010[0]=%c%c[%s*3+0];\n\
\t\t\t\t%c%c_010[1]=%c%c[%s*3+1];\n\
\t\t\t\t%c%c_010[2]=%c%c[%s*3+2];\n",\
                c_PQ[ipq],(b_smalld)?'d':'D',\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_AC[ipq],s_ii[i_ij],\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_AC[ipq],s_ii[i_ij],\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_AC[ipq],s_ii[i_ij]);

    }
    if(bam>0){
        fprintf(fp,"\
\t\t\t\tdouble %c%c_001[3];\n\
\t\t\t\t%c%c_001[0]=%c%c[%s*3+0];\n\
\t\t\t\t%c%c_001[1]=%c%c[%s*3+1];\n\
\t\t\t\t%c%c_001[2]=%c%c[%s*3+2];\n",\
                c_PQ[ipq],(b_smalld)?'d':'D',\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_BD[ipq],s_ii[i_ij],\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_BD[ipq],s_ii[i_ij],\
                c_PQ[ipq],(b_smalld)?'d':'D',c_PQ[ipq],c_BD[ipq],s_ii[i_ij]);
    }
        fprintf(fp,"\
\t\t\t\tdouble %cta=%cta_in[%s];\n\
\t\t\t\tdouble p%c=p%c_in[%s];\n",\
                c_ZE[ ipq],c_ZE[ ipq],s_ii[i_ij],\
                c_PQs[ipq],c_PQs[ipq],s_ii[i_ij]);
    }

}

void output_getPmtrx_tex(FILE * fp,int ipq,int i_ij,int aam,int bam,bool b_tex,int len){
    if(b_tex){
	fprintf(fp,"\
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*%d;\n",len);
    if(get_id_j(aam,bam)<=get_id_j(2,2)){
        fprintf(fp,"\
        double P_max=0.0;\n\
        for(int p_i=0;p_i<%d;p_i++){\n\
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_%s+p_i);\n\
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);\n\
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];\n\
            }\n",shell_len[aam]*shell_len[bam],s_ii[i_ij]);
    }
    else{
        fprintf(fp,"\
        double P_max=1.0;\n\
        for(int p_i=0;p_i<%d;p_i++){\n\
            Pmtrx[p_i]=1.0;\n\
            }\n",shell_len[aam]*shell_len[bam]);
    }
    }
    else{
	fprintf(fp,"\
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*%d;\n",len);
    if(get_id_j(aam,bam)<=get_id_j(2,2)){
        fprintf(fp,"\
        double P_max=0.0;\n\
        for(int p_i=0;p_i<%d;p_i++){\n\
            Pmtrx[p_i]=Pmtrx_in[p_%s+p_i];\n\
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];\n\
            }\n",shell_len[aam]*shell_len[bam],s_ii[i_ij]);
    }
    else{
        fprintf(fp,"\
        double P_max=1.0;\n\
        for(int p_i=0;p_i<%d;p_i++){\n\
            Pmtrx[p_i]=1.0;\n\
            }\n",shell_len[aam]*shell_len[bam]);
    }
    }
}

void output_getid_tex(FILE * fp,int ipq,int i_ij,bool b_tex){
    if(b_tex){
	fprintf(fp,"\
            unsigned int id_%s=tex1Dfetch(tex_id_%s,%s);\n",s_bk[ipq],s_bk[ipq],s_ii[i_ij]);
    }
    else{
	fprintf(fp,"\
            unsigned int id_%s=id_%s_in[%s];\n",s_bk[ipq],s_bk[ipq],s_ii[i_ij]);
    }

}

void output_loop_end(FILE * fp,int n){
    for(int i=0;i<n;i++) fprintf(fp,"\t");
    fprintf(fp,"}\n");
}

void output_gather_ans(FILE * fp,int aam,int bam,bool b_jmtrx){
    int len=shell_len[aam]*shell_len[bam];
    fprintf(fp,"\
        __syncthreads();\n\
        int num_thread=tdis/2;\n\
        while (num_thread!=0){\n\
            __syncthreads();\n\
            if(tId_x<num_thread){\n\
                for(int ians=0;ians<%d;ians++){\n\
                    ans_temp[tId_x*%d+ians]+=ans_temp[(tId_x+num_thread)*%d+ians];\n\
                }\n\
            }\n\
            num_thread/=2;\n\
        }\n",len,len,len);
    fprintf(fp,"\
        if(tId_x==0){\n\
            for(int ians=0;ians<%d;ians++){\n\
                ans[%s*%d+ians]=ans_temp[(tId_x)*%d+ians];\n\
            }\n\
        }\n",len,(b_jmtrx)?"i_contrc_bra":"(i_contrc_bra*contrc_ket_num+j_contrc_ket)",len,len);
}

void output_FT(FILE * fp,int j, int output_P,char * tail,bool b_two,bool only_ft){
    fprintf(fp,"\
                double alphaT=rsqrt(Eta+Zta);\n\
                double lmd=%d*P25*pp*pq*alphaT;\n\
                alphaT=Eta*Zta*alphaT*alphaT;\n",(b_two)?2:4);
    if(output_P==0) fprintf(fp,"\
                lmd*=Pmtrx[0];\n");
    if(j!=0){
        fprintf(fp,"\
                double TX=PX-QX;\n\
                double TY=PY-QY;\n\
                double TZ=PZ-QZ;\n\
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);\n");
    }
    else{
        fprintf(fp,"\
                double T=alphaT*((PX-QX)*(PX-QX)+(PY-QY)*(PY-QY)+(PZ-QZ)*(PZ-QZ));\n");
    }
    fprintf(fp,"\
                double R_000[%d];\n",j+1);
    char ftj[3]="";
    if(tail[1]=='f' && tail[2]=='s'){
        if(j==0) sprintf(ftj,"_0");
        else if(j<=2){
            sprintf(ftj,"_1");
        }
        else{
            sprintf(ftj,"_%d",j);
        }
    }
    if(tail[1]=='p' && tail[2]=='o'){
        sprintf(ftj,"_%d",j);
    }
    fprintf(fp,"\
                Ft%s%s(%d,T,R_000);\n",tail,ftj,j);
    int temp=1;
    for(int i=0;i<=j;i++){
        fprintf(fp,"\
                R_000[%d]*=",i);
        if(i>0) fprintf(fp,"%d*",temp);
        for(int ii=0;ii<i;ii++) fprintf(fp,"alphaT*");
        fprintf(fp,"lmd;\n");
        temp*=-2;
    }
    if(only_ft){
    fprintf(fp,"\
                for(int i=0;i<=%d;i++) ans_temp[ans_id]+=R_000[i];\n",j);
    }
}

void TSMJ_J_loop_start(FILE * fp,int aam, int bam, int cam, int dam,int ipq_inner_loop,bool inner_zero,bool outer_zero,bool bsmalld, bool b_QRsum,bool b_MDQR,bool b_Tex){
    output_contrc_bra_loop(fp);//outer loop: loop contracted bra pair
    output_primit_bra_loop(fp);//loop primitive bra pair within the contracted bra pair

    output_getPQ_tex(fp,(ipq_inner_loop+1)%2,0,aam,bam,false,(bsmalld|| (!inner_zero)));//load P coefficients from global memory without using texture memory
    output_define_aPin(fp,0,aam,bam,b_MDQR);
    if(b_QRsum) TSMJ_QR_declare(fp,aam,bam,cam,dam,ipq_inner_loop,b_QRsum);
    output_primit_ket_loop(fp);//inner loop: loop primitive ket pair

    output_getK2_tex(fp,ipq_inner_loop,1,b_Tex);
    output_schwartz_screening(fp,true,true,ipq_inner_loop);

    output_getPQ_tex(fp,ipq_inner_loop,1,cam,dam,b_Tex,(bsmalld|| (!outer_zero)));
    output_getPmtrx_tex(fp,ipq_inner_loop,1,cam,dam,b_Tex,shell_len[cam]*shell_len[dam]);
}
void output_K_contrc_loop(FILE * fp,int ipq){
    fprintf(fp,"\
    for(unsigned int %c_contrc_%s=bId_%c;%c_contrc_%s<contrc_%s_num;%c_contrc_%s+=bdis_%c){\n",\
        c_ij[ipq],s_bk[ipq],c_xy[ipq],\
        c_ij[ipq],s_bk[ipq],s_bk[ipq],\
        c_ij[ipq],s_bk[ipq],c_xy[ipq]);
}
void output_K_get_prim_pair_id(FILE * fp,int ipq){
    fprintf(fp,"\
    unsigned int primit_%s_start = contrc_%s_id[%c_contrc_%s  ];\n\
    unsigned int primit_%s_end   = contrc_%s_id[%c_contrc_%s+1];\n",\
        s_bk[ipq],s_bk[ipq],c_ij[ipq],s_bk[ipq],\
        s_bk[ipq],s_bk[ipq],c_ij[ipq],s_bk[ipq]);
}
void output_K_prim_loop_0(FILE * fp,int ipq){
    fprintf(fp,"\
        for(unsigned int ii=primit_%s_start;ii<primit_%s_end;ii++){\n",\
        s_bk[ipq],s_bk[ipq]);
}
void output_K_prim_loop_1(FILE * fp,int ipq){
    fprintf(fp,"\
        for(unsigned int j=tId_x;j<primit_%s_end-primit_%s_start;j+=tdis){\n\
            unsigned int jj=primit_%s_start+j;\n",\
        s_bk[ipq],s_bk[ipq],s_bk[ipq]);
}
void output_K_block_break(FILE * fp,int len){
    fprintf(fp,"\
        if(i_contrc_bra>j_contrc_ket){\n\
            if(tId_x==0){\n\
                for(int ians=0;ians<%d;ians++){\n\
                    ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*%d+ians]=0.0;\n\
                }\n\
            }\n\
            continue;\n\
        }\n",len,len);
}
void TSMJ_K_loop_start(FILE * fp,int aam, int bam, int cam, int dam,int ipq_inner_loop,bool b_smalld,bool b_MDQR){
    output_K_contrc_loop(fp,0);//outer loop: loop contracted bra pair
    output_K_contrc_loop(fp,1);//outer loop: loop contracted ket pair
    output_K_get_prim_pair_id(fp,0);
    output_K_get_prim_pair_id(fp,1);
    if(aam==cam && bam==dam) output_K_block_break(fp,shell_len[aam]*shell_len[cam]);

    output_K_prim_loop_0(fp,(ipq_inner_loop+1)%2);//loop primitive bra pair within the contracted bra pair
    output_getid_tex(fp,(ipq_inner_loop+1)%2,0,false);
    output_getPQ_tex(fp,(ipq_inner_loop+1)%2,0,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,false,(b_MDQR || b_smalld));//load P coefficients from global memory without using texture memory

    output_getK2_tex(fp,(ipq_inner_loop+1)%2,0,false);
    output_define_aPin(fp,(ipq_inner_loop+1)%2,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_MDQR);

    output_K_prim_loop_1(fp,ipq_inner_loop);//inner loop: loop primitive ket pair
    output_getid_tex(fp,ipq_inner_loop,1,true);

    output_K_get_Pmtrx(fp,bam,dam);

    output_getK2_tex(fp,ipq_inner_loop,1,true);
    output_schwartz_screening(fp,false,true,ipq_inner_loop);
    output_Pmtrx_screening(fp);

    output_getPQ_tex(fp,ipq_inner_loop,1,(ipq_inner_loop==1)?cam:aam,(ipq_inner_loop==1)?dam:bam,true,(b_MDQR || b_smalld));
    //output_getPmtrx_tex(fp,ipq_inner_loop,1,cam,dam,true,shell_len[cam]*shell_len[dam]);
}

void TSMJ_output_J(FILE * fp,\
                     int aam, int bam, int cam, int dam,\
                     bool b_Dform,bool b_NRR, bool b_CSE,\
                     bool b_Tex,bool b_JME,\
                     char * tail,char * ft_tail,int ipq_inner_loop){
    ipq_inner_loop=1;
    bool outer_zero=false;
    bool inner_zero=false;
    if(ipq_inner_loop==0){
        if(aam+bam==0) inner_zero=true;
        if(cam+dam==0) outer_zero=true;
    }
    else{
        if(aam+bam==0) outer_zero=true;
        if(cam+dam==0) inner_zero=true;
    }
    int ipq_r=ipq_inner_loop;
    if(inner_zero) ipq_r=(ipq_inner_loop+1)%2;
    bool b_smalld=true;
    if(b_Dform){
        if(ipq_r==ipq_inner_loop){
            b_smalld=!outer_zero;
        }
        else{
            b_smalld=!inner_zero;
        }
    }

    bool b_fullr=true;
    bool b_delta=true;
    bool b_QRsum=b_JME;
    bool b_PmQR=false;
    bool b_aPinQR=true;
    bool b_MDQR=false;

    if(b_smalld) b_aPinQR=false;
    if(cam+dam==0){
        b_PmQR=false;
        //b_QRsum=false;
    }
    output_J_kernel_define_tex(fp,"TSMJ",tail,aam,bam,cam,dam,b_Tex);//declare and define kernel function

    output_id_init(fp);//get CUDA thread and block id

    output_define_Pmtrx(fp,shell_len[cam]*shell_len[dam]);//define temporary variable: Pmtrx
    output_define_ans_temp(fp,(aam+bam+cam+dam>=4 && (cam>=1 && cam==dam))?32:128,shell_len[aam]*shell_len[bam]);//define temporary variable: ans_temp

    TSMJ_J_loop_start(fp,aam,bam,cam,dam,ipq_inner_loop,inner_zero,outer_zero,b_smalld,b_QRsum,b_MDQR,b_Tex);
    output_FT(fp,aam+bam+cam+dam,cam+dam,ft_tail,true,ONLY_FT);

    if(!ONLY_FT){

    output_define_aPin(fp,ipq_inner_loop,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_MDQR);
	TSMJ_R_expression(fp,aam,bam,cam,dam,ipq_r,b_smalld,b_fullr);

	MD_d_declare(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
	if(b_NRR) TSMJ_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
	else MD_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld,b_CSE);
    if(b_delta){
        if(b_CSE){
            TSMJ_delta_declare_1(ipq_inner_loop,fp,0,0,false);
            TSMJ_delta_expression_1(ipq_inner_loop,fp,0,0,false,b_smalld);
        }
        else{
            MD_delta_expression(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
        }
    }

	if(b_CSE) TSMJ_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,!b_QRsum,b_QRsum,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,b_QRsum);
    else MD_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr);
	//TSMJ_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr);
    if(!b_QRsum){
	MD_d_declare((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
	if(b_NRR) TSMJ_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
	else MD_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld,b_CSE);

    if(b_delta){
        if(b_CSE){
            TSMJ_delta_declare_1((ipq_inner_loop+1)%2,fp,0,0,false);
            TSMJ_delta_expression_1((ipq_inner_loop+1)%2,fp,0,0,false,b_smalld);
        }
        else{
            MD_delta_expression((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
        }
    }

	if(b_CSE) TSMJ_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,b_QRsum);
	else MD_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam,b_delta,b_smalld,b_fullr);
    }
    }

	output_loop_end(fp,3);

    if(!ONLY_FT){
	if(b_QRsum){
	MD_d_declare((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,!inner_zero);
	if(b_NRR) TSMJ_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
	else MD_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,!inner_zero,b_CSE);

    if(b_delta){
        if(b_CSE){
            TSMJ_delta_declare_1((ipq_inner_loop+1)%2,fp,0,0,false);
            TSMJ_delta_expression_1((ipq_inner_loop+1)%2,fp,0,0,false,b_smalld);
        }
        else{
            MD_delta_expression((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
        }
    }

	if(b_CSE) TSMJ_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,b_QRsum);
	else MD_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam,b_delta,b_smalld,b_fullr);
	}
    }

	output_loop_end(fp,2);
	output_gather_ans(fp,aam,bam,true);
	output_loop_end(fp,1);
	output_loop_end(fp,0);
}

void MD_output_J(FILE * fp,\
                     int aam, int bam, int cam, int dam,char * tail,int ipq_inner_loop){
    ipq_inner_loop=1;
    bool outer_zero=false;
    bool inner_zero=false;
    int ipq_r=ipq_inner_loop;
    bool b_smalld=true;

    bool b_fullr=true;
    bool b_delta=true;
    bool b_MDQR=true;
    bool b_CSE=false;
    //if(b_include) output_include(fp);
    output_J_kernel_define_tex(fp,"MD",tail,aam,bam,cam,dam,true);//declare and define kernel function
    output_id_init(fp);//get CUDA thread and block id

    output_define_Pmtrx(fp,shell_len[cam]*shell_len[dam]);

    output_define_ans_temp(fp,64,shell_len[aam]*shell_len[bam]);

    TSMJ_J_loop_start(fp,aam,bam,cam,dam,ipq_inner_loop,inner_zero,outer_zero,b_smalld,false,b_MDQR,true);

    output_FT(fp,aam+bam+cam+dam,cam+dam,tail,true,ONLY_FT);

    if(!ONLY_FT){
    //output_define_aPin(fp,0,aam,bam);
    output_define_aPin(fp,ipq_inner_loop,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_MDQR);

	//TSMJ_R_expression(fp,aam,bam,cam,dam,0,false,outer_zero,false);
	TSMJ_R_expression(fp,aam,bam,cam,dam,ipq_r,b_smalld,b_fullr);

	MD_d_declare(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
    MD_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld,b_CSE);
    if(b_delta) MD_delta_expression(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);

    MD_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr);
	//TSMJ_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false);

	MD_d_declare((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,!inner_zero);
	MD_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,!inner_zero,b_CSE);
    if(b_delta) MD_delta_expression((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);

    MD_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam,b_delta,b_smalld,b_fullr);
	//TSMJ_ans_expression(fp,true,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,cam+dam);
    }
	output_loop_end(fp,3);
	output_loop_end(fp,2);
	output_gather_ans(fp,aam,bam,true);
	output_loop_end(fp,1);
	output_loop_end(fp,0);

}

void TSMJ_output_K(FILE * fp,\
                     int aam, int bam, int cam, int dam,\
                     bool b_Dform,bool b_NRR, bool b_CSE,\
                     char * tail,char * ft_tail,int ipq_inner_loop){
    bool outer_zero=false;
    bool inner_zero=false;
    if(ipq_inner_loop==0){
        if(aam+bam==0) inner_zero=true;
        if(cam+dam==0) outer_zero=true;
    }
    else{
        if(aam+bam==0) outer_zero=true;
        if(cam+dam==0) inner_zero=true;
    }
    bool b_smalld=true;
    if(b_Dform){
        b_smalld=(!inner_zero)?(!outer_zero):false;
    }
    bool b_fullr=true;
    bool b_delta=true;
    bool b_two=true;
    bool b_MDQR=false;
    bool b_PmQR=false;
    bool b_aPinQR=true;
    char c_kp=(ipq_inner_loop==0)?'p':'q';
    if(b_smalld) b_aPinQR=false;
    if(get_id_k(aam,bam)!=get_id_k(cam,dam)) b_two=false;

    output_K_kernel_define_tex(fp,"TSMJ_K",c_kp,tail,aam,bam,cam,dam);//declare and define kernel function

    output_K_id_init(fp);//get CUDA thread and block id

    output_define_Pmtrx(fp,shell_len[bam]*shell_len[dam]);//define temporary variable: Pmtrx
    output_define_ans_temp(fp,64,shell_len[aam]*shell_len[cam]);//define temporary variable: ans_temp

    TSMJ_K_loop_start(fp,aam,bam,cam,dam,ipq_inner_loop,b_smalld,b_MDQR);

    output_FT(fp,aam+bam+cam+dam,bam+dam,ft_tail,b_two,ONLY_FT);

    if(!ONLY_FT){
    if(ipq_inner_loop==1) output_define_aPin(fp,1,cam,dam,b_MDQR);
    else output_define_aPin(fp,0,aam,bam,b_MDQR);
	TSMJ_R_expression(fp,aam,bam,cam,dam,(!inner_zero)?ipq_inner_loop:((ipq_inner_loop+1)%2),b_smalld,b_fullr);

	MD_d_declare(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
	if(b_NRR) TSMJ_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
	else MD_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld,b_CSE);
    if(b_delta){
        if(b_CSE){
            TSMJ_delta_declare_1(ipq_inner_loop,fp,0,0,false);
            TSMJ_delta_expression_1(ipq_inner_loop,fp,0,0,false,b_smalld);
        }
        else{
            MD_delta_expression(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
        }
    }

	if(b_CSE) TSMJ_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,false);
	else MD_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr);

	MD_d_declare((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
	if(b_NRR) TSMJ_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
    else MD_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld,b_CSE);

    if(b_delta){
        if(b_CSE){
            TSMJ_delta_declare_1((ipq_inner_loop+1)%2,fp,0,0,false);
            TSMJ_delta_expression_1((ipq_inner_loop+1)%2,fp,0,0,false,b_smalld);
        }
        else{
            MD_delta_expression((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
        }
    }

	if(b_CSE) TSMJ_ans_expression(fp,false,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,bam+dam,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,false);
	else MD_ans_expression(fp,false,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,bam+dam,b_delta,b_smalld,b_fullr);
    }
	output_loop_end(fp,2);
	output_loop_end(fp,2);
	output_gather_ans(fp,aam,cam,false);
	output_loop_end(fp,1);
	output_loop_end(fp,1);
	output_loop_end(fp,0);
}

void MD_output_K(FILE * fp,\
                     int aam, int bam, int cam, int dam,char * tail,int ipq_inner_loop){
    bool outer_zero=false;
    bool inner_zero=false;
    if(ipq_inner_loop==0){
        if(aam+bam==0) inner_zero=true;
        if(cam+dam==0) outer_zero=true;
    }
    else{
        if(aam+bam==0) outer_zero=true;
        if(cam+dam==0) inner_zero=true;
    }
    bool b_smalld=true;
    bool b_fullr=true;
    bool b_delta=true;
    bool b_two=true;
    bool b_MDQR=true;
    bool b_CSE=false;
    char c_kp=(ipq_inner_loop==0)?'p':'q';
    if(get_id_k(aam,bam)!=get_id_k(cam,dam)) b_two=false;

    output_K_kernel_define_tex(fp,"MD_K",c_kp,tail,aam,bam,cam,dam);//declare and define kernel function

    output_K_id_init(fp);//get CUDA thread and block id

    output_define_Pmtrx(fp,shell_len[bam]*shell_len[dam]);//define temporary variable: Pmtrx
    output_define_ans_temp(fp,64,shell_len[aam]*shell_len[cam]);//define temporary variable: ans_temp

    TSMJ_K_loop_start(fp,aam,bam,cam,dam,ipq_inner_loop,b_smalld,b_MDQR);

    output_FT(fp,aam+bam+cam+dam,bam+dam,tail,b_two,ONLY_FT);

    if(!ONLY_FT){
    if(ipq_inner_loop==1) output_define_aPin(fp,1,cam,dam,b_MDQR);
    else output_define_aPin(fp,0,aam,bam,b_MDQR);
	TSMJ_R_expression(fp,aam,bam,cam,dam,(!inner_zero)?ipq_inner_loop:((ipq_inner_loop+1)%2),b_smalld,b_fullr);

	MD_d_declare(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);
	MD_d_expression_0(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld,b_CSE);
    if(b_delta) MD_delta_expression(ipq_inner_loop,fp,(ipq_inner_loop==0)?aam:cam,(ipq_inner_loop==0)?bam:dam,b_smalld);

	MD_QR_expression(fp,aam,bam,cam,dam,ipq_inner_loop,outer_zero,true,false,b_delta,b_smalld,b_fullr);

	MD_d_declare((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);
	MD_d_expression_0((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld,b_CSE);
    if(b_delta) MD_delta_expression((ipq_inner_loop+1)%2,fp,(ipq_inner_loop==1)?aam:cam,(ipq_inner_loop==1)?bam:dam,b_smalld);

	MD_ans_expression(fp,false,aam,bam,cam,dam,(ipq_inner_loop+1)%2,inner_zero,bam+dam,b_delta,b_smalld,b_fullr);
    }
	output_loop_end(fp,2);
	output_loop_end(fp,2);
	output_gather_ans(fp,aam,cam,false);
	output_loop_end(fp,1);
	output_loop_end(fp,1);
	output_loop_end(fp,0);
}

void TSMJ_declare_aPin(FILE * fp, int ipq, int aam, int bam){
    for(int i=2;i<=aam+bam;i++){
        fprintf(fp,"\t\t\tdouble a%cin%d=a%cin1*a%cin%d;\n",c_PQ[ipq],i,c_PQ[ipq],c_PQ[ipq],i-1);
    }
}

void MD_d_declare(int ipq,FILE * fp,int aam, int bam,bool b_dformat){
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                if(ipq==0){
                    if(np==0 && na+nb==1) continue;
                    if(!b_dformat && np==na+nb) continue;
                    if(P_d_mark_table[0][np][na][nb]==1){
                        fprintf(fp,"\t\tdouble %c%c_%d%d%d[3];\n",c_PQ[ipq],b_dformat?'d':'D',np,na,nb);
                    }
                }
                else{
                    if(np==0 && na+nb==1) continue;
                    if(!b_dformat && np==na+nb) continue;
                    if(Q_d_mark_table[0][np][na][nb]==1){
                        fprintf(fp,"\t\tdouble %c%c_%d%d%d[3];\n",c_PQ[ipq],b_dformat?'d':'D',np,na,nb);
                    }
                }
            }
        }
    }
}

void TSMJ_d_expression_0(FILE * fp,int ipq,int np,int na,int nb,bool b_smalld,char c_d){
    char expression[SUBLEN];
    bool output_d=false;
    if(ipq==0){
    if(P_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    else{
    if(Q_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    memset(expression,0,SUBLEN*sizeof(char));
    int subid=0;
    if(np==0 && na==1 && nb==0){/*
        subid+=sprintf(expression+subid,"\t\t%cd_010[0]=%c%cX;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cd_010[1]=%c%cY;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cd_010[2]=%c%cZ;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);*/

    }
    else if(np==0 && na==0 && nb==1){/*
        subid+=sprintf(expression+subid,"\t\t%cd_001[0]=%c%cX;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cd_001[1]=%c%cY;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cd_001[2]=%c%cZ;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);*/
    }
    else{
        subid+=sprintf(expression+subid,"\t\tfor(int i=0;i<3;i++){\n");
        subid+=sprintf(expression+subid,"\t\t\t%c%c_%d%d%d[i]=",c_PQ[ipq],c_d,np,na,nb);
        if(np==0){
            if(na>=nb){
                if(np+1==na+nb-1){
                    subid+=sprintf(expression+subid,"a%cin%d",c_PQ[ipq],np+1);
                }
                else{
					subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np+1,na-1,nb);
                }
                subid+=sprintf(expression+subid,"+%c%c_010[i]*%c%c_%d%d%d[i]",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na-1,nb);
                output_d=true;
            }
            else{
                if(np+1==na+nb-1){
                    subid+=sprintf(expression+subid,"a%cin%d",c_PQ[ipq],np+1);
                }
                else{
					subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np+1,na,nb-1);
                }
                subid+=sprintf(expression+subid,"+%c%c_001[i]*%c%c_%d%d%d[i]",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na,nb-1);
                output_d=true;
            }
        }
        else{
            if(np<na+nb){
                if(b_smalld) subid+=sprintf(expression+subid,"a%cin1*",c_PQ[ipq]);
                if(na==nb){
                    if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                    subid+=sprintf(expression+subid,"(%c%c_%d%d%d[i]+%c%c_%d%d%d[i])",c_PQ[ipq],c_d,np-1,na-1,nb,c_PQ[ipq],c_d,np-1,na,nb-1);
                }
                else{
                    subid+=sprintf(expression+subid,"(");
                    bool test=false;
                    if(na!=0){
                        test=true;
                        if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                        subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na-1,nb);
                    }
                    if(nb!=0){
                        if(test){
                            subid+=sprintf(expression+subid,"+");
                        }
                        if(nb!=np) subid+=sprintf(expression+subid,"%f*",((double)nb)/((double)np));
                        subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na,nb-1);
                    }
                    subid+=sprintf(expression+subid,")");
                }
                output_d=true;
            }
            else{
                if(b_smalld){
                    subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);
                    if(np!=1){
                    subid+=sprintf(expression+subid,"*");
                    if(na==nb){
                        if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                        subid+=sprintf(expression+subid,"(%c%c_%d%d%d[i]+%c%c_%d%d%d[i])",c_PQ[ipq],c_d,np-1,na-1,nb,c_PQ[ipq],c_d,np-1,na,nb-1);
                    }
                    else{
                        subid+=sprintf(expression+subid,"(");
                        bool test=false;
                        if(na!=0){
                            test=true;
                            if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                            subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na-1,nb);
                        }
                        if(nb!=0){
                            if(test){
                                subid+=sprintf(expression+subid,"+");
                            }
                            if(nb!=np) subid+=sprintf(expression+subid,"%f*",((double)nb)/((double)np));
                            subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na,nb-1);
                        }
                        subid+=sprintf(expression+subid,")");
                    }
                    }
                    output_d=true;
                }
            }
        }
        subid+=sprintf(expression+subid,";\n\t\t\t}\n");
    }
    if(output_d) fprintf(fp,expression);
    return;
}
void TSMJ_d_expression_0(int ipq,FILE * fp,int aam, int bam,bool b_smalld){
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                TSMJ_d_expression_0(fp,ipq,np,na,nb,b_smalld,b_smalld?'d':'D');
            }
        }
    }
}

void MD_d_expression_0(FILE * fp,int ipq,int np,int na,int nb,bool b_smalld,bool b_CSE){
    char expression[SUBLEN];
    bool output_d=false;
    char c_d=(b_smalld)?'d':'D';
    if(ipq==0){
    if(P_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    else{
    if(Q_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    memset(expression,0,SUBLEN*sizeof(char));
    int subid=0;
    if(np==0 && na+nb==1){
        //for(int i=0;i<3;i++)
        //subid+=sprintf(expression+subid,\
                       "\t\t%cd_%d%d%d[%d]=%c%c%c;\n",\
                       c_PQ[ipq],np,na,nb,i,\
                       c_PQ[ipq],(na==1)?(c_AC[ipq]):(c_BD[ipq]),c_XYZ[i]);
    }
    else{
        subid+=sprintf(expression+subid,"\t\tfor(int i=0;i<3;i++){\n");
        subid+=sprintf(expression+subid,"\t\t\t%c%c_%d%d%d[i]=",c_PQ[ipq],c_d,np,na,nb);
        if(np==0){
            if(na>=nb){
                if(b_smalld){
                    if(b_CSE && (np+1)==(na+nb-1)) subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);
                    else subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np+1,(na-1),nb);

                    subid+=sprintf(expression+subid,"+%c%c_010[i]*%c%c_%d%d%d[i]",\
                                                    c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,(na-1),nb);
                }
                else{
                    if((na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"a%cin1*%c%c_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],c_d,np+1,(na-1),nb);
                    else subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);

                    if((na+nb-1)!=(np)) subid+=sprintf(expression+subid,"+%c%c_010[i]*%c%c_%d%d%d[i]",\
                                                    c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,(na-1),nb);
                    else subid+=sprintf(expression+subid,"+%c%c_010[i]",c_PQ[ipq],c_d);
                }
            }
            else{
                if(b_smalld){
                    if(b_CSE && (np+1)==(na+nb-1)) subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);
                    else subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np+1,na,(nb-1));

                    subid+=sprintf(expression+subid,"+%c%c_001[i]*%c%c_%d%d%d[i]",\
                                                    c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na,(nb-1));
                }
                else{
                    if((na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"a%cin1*%c%c_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],c_d,np+1,na,(nb-1));
                    else subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);

                    if((na+nb-1)!=(np)) subid+=sprintf(expression+subid,"+%c%c_001[i]*%c%c_%d%d%d[i]",\
                                                    c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na,(nb-1));
                    else subid+=sprintf(expression+subid,"+%c%c_001[i]",c_PQ[ipq],c_d);
                }
            }
            output_d=true;
        }
        else{
            if(b_smalld || np!=na+nb){
            output_d=true;
            if(na>=nb){
                if(np+1<=na+nb-1){
                    if(b_smalld){
                        if(!b_CSE || (na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"%d*%c%c_%d%d%d[i]+",np+1,c_PQ[ipq],c_d,np+1,na-1,nb);
                        else subid+=sprintf(expression+subid,"%d*a%cin%d+",np+1,c_PQ[ipq],np+1);
                    }
                    else{
                        if((na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"%d*a%cin1*%c%c_%d%d%d[i]+",np+1,c_PQ[ipq],c_PQ[ipq],c_d,np+1,na-1,nb);
                        else subid+=sprintf(expression+subid,"%d*a%cin1+",np+1,c_PQ[ipq]);
                    }
                }
                if(np<=na+nb-1){
                    if(b_smalld){
                        if(!b_CSE || (na+nb-1)!=(np)) subid+=sprintf(expression+subid,"%c%c_010[i]*%c%c_%d%d%d[i]+",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na-1,nb);
                        else subid+=sprintf(expression+subid,"%c%c_010[i]*a%cin%d+",c_PQ[ipq],c_d,c_PQ[ipq],np);
                    }
                    else{
                        if((na+nb-1)!=(np)) subid+=sprintf(expression+subid,"%c%c_010[i]*%c%c_%d%d%d[i]+",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na-1,nb);
                        else subid+=sprintf(expression+subid,"%c%c_010[i]+",c_PQ[ipq],c_d);
                    }
                }
                if(np-1==0 && na-1==0 && nb==0){
                    if(b_smalld) subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);
                    else subid+=sprintf(expression+subid,"1");
                }
                else{
                    if(b_smalld) subid+=sprintf(expression+subid,"a%cin1*%c%c_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],c_d,np-1,na-1,nb);
                    else subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na-1,nb);
                }
            }
            else{
                if(np+1<=na+nb-1){
                    if(b_smalld){
                        if(!b_CSE || (na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"%d*%c%c_%d%d%d[i]+",np+1,c_PQ[ipq],c_d,np+1,na,nb-1);
                        else subid+=sprintf(expression+subid,"%d*a%cin%d+",np+1,c_PQ[ipq],np+1);
                    }
                    else{
                        if((na+nb-1)!=(np+1)) subid+=sprintf(expression+subid,"%d*a%cin1*%c%c_%d%d%d[i]+",np+1,c_PQ[ipq],c_PQ[ipq],c_d,np+1,na,nb-1);
                        else subid+=sprintf(expression+subid,"%d*a%cin1+",np+1,c_PQ[ipq]);
                    }
                }
                if(np<=na+nb-1){
                    if(b_smalld){
                        if(!b_CSE || (na+nb-1)!=(np)) subid+=sprintf(expression+subid,"%c%c_001[i]*%c%c_%d%d%d[i]+",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na,nb-1);
                        else subid+=sprintf(expression+subid,"%c%c_001[i]*a%cin%d+",c_PQ[ipq],c_d,c_PQ[ipq],np);
                    }
                    else{
                        if((na+nb-1)!=(np)) subid+=sprintf(expression+subid,"%c%c_001[i]*%c%c_%d%d%d[i]+",c_PQ[ipq],c_d,c_PQ[ipq],c_d,np,na,nb-1);
                        else subid+=sprintf(expression+subid,"%c%c_001[i]+",c_PQ[ipq],c_d);
                    }
                }
                if(np-1==0 && na==0 && nb-1==0){
                    if(b_smalld) subid+=sprintf(expression+subid,"a%cin1",c_PQ[ipq]);
                    else subid+=sprintf(expression+subid,"1");
                }
                else{
                    if(b_smalld) subid+=sprintf(expression+subid,"a%cin1*%c%c_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],c_d,np-1,na,nb-1);
                    else subid+=sprintf(expression+subid,"%c%c_%d%d%d[i]",c_PQ[ipq],c_d,np-1,na,nb-1);
                }
            }
            }
        }
        subid+=sprintf(expression+subid,";\n\t\t\t}\n");
    }
    if(output_d) fprintf(fp,expression);
    return;
}
void MD_d_expression_0(int ipq,FILE * fp,int aam, int bam,bool b_smalld,bool b_CSE){
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                MD_d_expression_0(fp,ipq,np,na,nb,b_smalld,b_CSE);
            }
        }
    }
}

void TSMJ_D_expression_1(FILE * fp,int ipq,int np,int na,int nb){
    char expression[SUBLEN];
    if(ipq==0){
    if(P_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    else{
    if(Q_d_mark_table[0][np][na][nb]==0){
        return;
    }
    }
    memset(expression,0,SUBLEN*sizeof(char));
    int subid=0;
    if(np==0 && na==1 && nb==0){
        subid+=sprintf(expression+subid,"\t\t%cD_010[0]=%c%cX;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cD_010[1]=%c%cY;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cD_010[2]=%c%cZ;\n",c_PQ[ipq],c_PQ[ipq],c_AC[ipq]);

    }
    else if(np==0 && na==0 && nb==1){
        subid+=sprintf(expression+subid,"\t\t%cD_001[0]=%c%cX;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cD_001[1]=%c%cY;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);
        subid+=sprintf(expression+subid,"\t\t%cD_001[2]=%c%cZ;\n",c_PQ[ipq],c_PQ[ipq],c_BD[ipq]);
    }
    else{
        subid+=sprintf(expression+subid,"\t\tfor(int i=0;i<3;i++){\n");
        subid+=sprintf(expression+subid,"\t\t\t%cD_%d%d%d[i]=",c_PQ[ipq],np,na,nb);
        if(np==0){
            if(na>=nb){
                if(na+nb==2){
                    subid+=sprintf(expression+subid,"a%cin%d",c_PQ[ipq],1);
                }
                else{
					subid+=sprintf(expression+subid,"a%cin%d*%cD_%d%d%d[i]",c_PQ[ipq],1,c_PQ[ipq],1,na-1,nb);
                }
                subid+=sprintf(expression+subid,"+%cD_010[i]*%cD_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],np,na-1,nb);
            }
            else{
                if(na+nb==2){
                    subid+=sprintf(expression+subid,"a%cin%d",c_PQ[ipq],1);
                }
                else{
					subid+=sprintf(expression+subid,"a%cin%d*%cD_%d%d%d[i]",c_PQ[ipq],1,c_PQ[ipq],1,na,nb-1);
                }
                subid+=sprintf(expression+subid,"+%cD_001[i]*%cD_%d%d%d[i]",c_PQ[ipq],c_PQ[ipq],np,na,nb-1);
            }
        }
        else{
            if(np==na+nb){
            }
            else{
                if(na==nb){
                    if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                    subid+=sprintf(expression+subid,"(%cD_%d%d%d[i]+%cD_%d%d%d[i])",c_PQ[ipq],np-1,na-1,nb,c_PQ[ipq],np-1,na,nb-1);
                }
                else{
                    bool test=false;
                    if(na!=0){
                        test=true;
                        if(na!=np) subid+=sprintf(expression+subid,"%f*",((double)na)/((double)np));
                        subid+=sprintf(expression+subid,"%cD_%d%d%d[i]",c_PQ[ipq],np-1,na-1,nb);
                    }
                    if(nb!=0){
                        if(test){
                            subid+=sprintf(expression+subid,"+");
                        }
                        if(nb!=np) subid+=sprintf(expression+subid,"%f*",((double)nb)/((double)np));
                        subid+=sprintf(expression+subid,"%cD_%d%d%d[i]",c_PQ[ipq],np-1,na,nb-1);
                    }
                }
            }
        }
        subid+=sprintf(expression+subid,";\n\t\t\t}\n");
    }
    fprintf(fp,expression);
    return;
}
void TSMJ_D_expression_1(int ipq,FILE * fp,int aam, int bam){
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                TSMJ_D_expression_1(fp,ipq,np,na,nb);
            }
        }
    }
}

void set_QR_table(int ia,int ib, int ic, int id, int ia2,int ib2,int ic2,int id2){
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,nc=0,lc=0,mc=0,nd=0,ld=0,md=0,np=0,lp=0,mp=0;
    get_MD_nlm(ia,ia2,&na,&la,&ma);
    get_MD_nlm(ib,ib2,&nb,&lb,&mb);
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]=1;
            }
        }
    }
}
void MD_set_QR_table(int ipq,int aam, int bam, int cam, int dam){
    int ia2;
    int ib2;
    int ic2;
    int id2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;
    for(ia2=0;ia2<sa;ia2++){
        for(ib2=0;ib2<sb;ib2++){
            for(ic2=0;ic2<sc;ic2++){
                for(id2=0;id2<sd;id2++){
                    if(ipq==1) set_QR_table(aam,bam,cam,dam,ia2,ib2,ic2,id2);
                    else set_QR_table(cam,dam,aam,bam,ic2,id2,ia2,ib2);
                }
            }
        }
    }
}
void MD_zero_QR_table(){
    for(int na=0;na<MAX_AMP;na++){
    for(int nb=0;nb<MAX_AMP;nb++){
    for(int la=0;la<MAX_AMP;la++){
    for(int lb=0;lb<MAX_AMP;lb++){
    for(int ma=0;ma<MAX_AMP;ma++){
    for(int mb=0;mb<MAX_AMP;mb++){
        for(int np=0;np<MAX_P;np++){
        for(int lp=0;lp<MAX_P;lp++){
        for(int mp=0;mp<MAX_P;mp++){
            QR_table[na][nb][la][lb][ma][mb][np][lp][mp]=0;
        }
        }
        }
    }
    }
    }
    }
    }
    }
}
/*
void MD_QR_declare(FILE* fp,int np,int lp, int mp, int ic, int id, int ic2,int id2,int delta_type){
    int nc=0,lc=0,mc=0,nd=0,ld=0,md=0;
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);

    if(QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]==0){
        return;
    }
    fprintf(fp,"\tdouble QR_%d%d%d%d%d%d%d%d%d%d%d%d=0;\n",0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
}
*/
