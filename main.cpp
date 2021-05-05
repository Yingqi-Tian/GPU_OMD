#include<cstdio>
#include"TSMJ.h"
#include"global.h"
#include"expression.h"

#define INNER_P 0
#define INNER_Q 1
#define IPQ_J_CONTR 1
delta_mark * P_delta_mark_array;
delta_mark * Q_delta_mark_array;

int P_delta_mark_sum=0;
int Q_delta_mark_sum=0;

expression_encode * P_expression_encode_list;
expression_encode * Q_expression_encode_list;

int P_expression_encode_list_sum=0;
int Q_expression_encode_list_sum=0;

int P_d_mark_table[3][MAX_P][MAX_AMP][MAX_AMP]={0};
int Q_d_mark_table[3][MAX_P][MAX_AMP][MAX_AMP]={0};
int QR_table[MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_AMP][MAX_P][MAX_P][MAX_P]={0};
int MD_R_mark_table[MAX_J][MAX_J][MAX_J][MAX_J]={0};

int am_num[MAX_P]={0};
char c_ABCD[4]={'A','B','C','D'};
char c_AC[2]={'A','C'};
char c_BD[2]={'B','D'};
char c_XYZ[3]={'X','Y','Z'};
char c_PQ[2]={'P','Q'};
char c_ZE[2]={'Z','E'};
char shell_name[9]={'s','p','d','f','g','h','i','k','l'};
char kernel_tail[10]="_taylor";
extern char s_bk[2][4];

void init_am_num(){
	for(int i=0;i<MAX_P;i++){
		am_num[i]=(i+1)*(i+2)/2;
	}
}

void TSMJ_gpu_J(int aam,int bam,int cam,int dam);

int get_id_j(int aam,int bam){
    return aam*(aam+1)/2+bam;
}

int id_k[5][5]={{0,2,5,10,17},{1,3,7,12,19},{4,6,8,14,21},{9,11,13,15,23},{16,18,20,22,24}};

int get_id_k(int aam,int bam){
	if(aam>=bam){
		return aam*aam+bam*2;
	}
	else{
		return (bam+1)*(bam+1)-2*(bam-aam);
	}
    return 0;
}
/*
void TSMJ_gpu_J(int aam,int bam, int cam,int dam,bool file_w,bool bDform,bool bNRR, bool bCSE,char * tail,char * ft_tail){
    int ipq_inner_loop=IPQ_J_CONTR;
    char filename[50];
	sprintf(filename,"TSMJ_code/TSMJ_%c%c%c%c_cuda.cu",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
	FILE * fp;
	if(file_w) fp=fopen(filename,"w");
	else fp=fopen(filename,"a");
	printf("TSMJ_J_kernel[%d][%d]=TSMJ_%c%c%c%c_J%s;\n",\
        get_id_j(aam,bam),get_id_j(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);
	CSE_init(aam,bam,cam,dam);
	CSE_generate_delta_list(0,aam,bam);
	CSE_generate_delta_list(1,cam,dam);

	set_d_mark_table();

	check_d_mark_table(aam,bam,cam,dam);

	MD_set_QR_table(ipq_inner_loop,cam,dam,aam,bam);

	MD_R_init(aam,bam,cam,dam);

	TSMJ_output_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,tail,ft_tail,ipq_inner_loop);
	fclose(fp);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();
	CSE_finish();
}
*/
void TSMJ_gpu_J(FILE * fp,int aam,int bam, int cam,int dam,bool bDform,bool bNRR, bool bCSE,bool bTex, bool bJME,char * tail,char * ft_tail){
    int ipq_inner_loop=IPQ_J_CONTR;
	printf("TSMJ_J_kernel[%d][%d]=TSMJ_%c%c%c%c_J%s;\n",\
        get_id_j(aam,bam),get_id_j(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);
    if(bCSE){
        CSE_init(aam,bam,cam,dam);
        CSE_generate_delta_list(0,aam,bam);
        CSE_generate_delta_list(1,cam,dam);

        set_d_mark_table();

        check_d_mark_table(aam,bam,cam,dam);
	}
	else{
        MD_set_d_mark_table(0,aam,bam);
        MD_set_d_mark_table(1,cam,dam);
	}
	MD_set_QR_table(ipq_inner_loop,aam,bam,cam,dam);

	MD_R_init(aam,bam,cam,dam);

	TSMJ_output_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,tail,ft_tail,ipq_inner_loop);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();

    if(bCSE){
        CSE_finish();
    }
}

void HGP_MD_gpu_J_engine(FILE * fp,int aam,int bam, int cam,int dam,\
						bool bDform,bool bNRR, bool bCSE,bool bTex,\
						bool bJME,bool bra_HGP,\
						char * tail,char * ft_tail){
    int ipq_inner_loop=IPQ_J_CONTR;
	printf("TSMJ_J_kernel[%d][%d]=TSMJ_%c%c%c%c_J%s;\n",\
        get_id_j(aam,bam),get_id_j(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);
    if(bCSE){
        CSE_init(aam,bam,cam,dam);
        if(!bra_HGP) CSE_generate_delta_list(0,aam,bam);
        CSE_generate_delta_list(1,cam,dam);

        set_d_mark_table();

        check_d_mark_table(aam,bam,cam,dam);
	}
	else{
        MD_set_d_mark_table(0,aam,bam);
        MD_set_d_mark_table(1,cam,dam);
	}
	MD_set_QR_table(ipq_inner_loop,aam,bam,cam,dam);

	MD_R_init(aam,bam,cam,dam);

	TSMJ_output_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,tail,ft_tail,ipq_inner_loop);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();

    if(bCSE){
        CSE_finish();
    }
}
void gpu_output_head(FILE * fp,char * head,int cam,int dam, bool b_Jmtrx){
	printf("TSMJ_tex_bind[%d]=texture_binding_%c%c;\n",\
        get_id_j(cam,dam),\
        shell_name[cam],shell_name[dam]);
	printf("TSMJ_tex_unbind[%d]=texture_unbind_%c%c;\n",\
        get_id_j(cam,dam),\
        shell_name[cam],shell_name[dam]);
    output_include(fp);
    output_define_texture_memory(fp,b_Jmtrx,1,cam,dam);
    output_texture_function(fp,b_Jmtrx,head,1,cam,dam);
}

void gpu_output_K_head(FILE * fp,char * head, int cam,int dam){
    bool b_Jmtrx=false;
    output_include(fp);
    output_define_texture_memory(fp,b_Jmtrx,0,1,1);
    output_define_texture_memory_id(fp,0);
    output_define_texture_memory(fp,b_Jmtrx,1,1,1);
    output_define_texture_memory_id(fp,1);
    output_texture_function_K(fp,b_Jmtrx,head,0,cam,dam);
    output_texture_function_K(fp,b_Jmtrx,head,1,cam,dam);
}

void TSMJ_h_define(FILE * hp, char * H_NAME){
    fprintf(hp,"#ifndef %s_H_INCLUDED\n\
#define %s_H_INCLUDED\n",H_NAME,H_NAME);
}

void TSMJ_h_end(FILE * hp, char * H_NAME){
    fprintf(hp,"#endif // %s_H_INCLUDED\n",H_NAME);
}

void gpu_J_h(FILE * hp,char * name_string, char * tail_string,int aam,int bam, int cam,int dam){
    output_kernel_define(hp,name_string,tail_string,aam,bam,cam,dam);
    fprintf(hp,";\n\n");
}

void output_d_mark_table(int ipq,int ixyz,int np,int na, int nb){
    int (*d_mark_table)[3][MAX_P][MAX_AMP][MAX_AMP];
    if(ipq==0){
        d_mark_table=&P_d_mark_table;
    }
    else{
        d_mark_table=&Q_d_mark_table;
    }
    printf("test : %d\n",(*d_mark_table)[ixyz][np][na][nb]);
}

void MD_gpu_J(FILE * fp,int aam,int bam, int cam,int dam,char * tail){
	printf("MD_J_kernel[%d][%d]=MD_%c%c%c%c%s;\n",\
        get_id_j(aam,bam),get_id_j(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);

    int ipq_inner_loop=1;
	MD_set_d_mark_table(0,aam,bam);
	MD_set_d_mark_table(1,cam,dam);
	//check_d_mark_table(aam,bam,cam,dam);

	MD_set_QR_table(ipq_inner_loop,aam,bam,cam,dam);

	MD_R_init(aam,bam,cam,dam);

	MD_output_J(fp,aam,bam,cam,dam,tail,ipq_inner_loop);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();
}

void TSMJ_gpu_K(FILE * fp,int aam,int bam, int cam,int dam,int ipq,bool bDform,bool bNRR, bool bCSE,char * tail,char * ft_tail){
    int ipq_inner_loop;
    if((aam+bam)>=(cam+dam)) ipq_inner_loop=1;
    else ipq_inner_loop=0;
    ipq_inner_loop=ipq;
	printf("TSMJ_K_kernel[%d][%d]=TSMJ_%c%c%c%c_K%s;\n",\
        get_id_k(aam,bam),get_id_k(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);

    if(bCSE){
        CSE_init(aam,bam,cam,dam);
        CSE_generate_delta_list(0,aam,bam);
        CSE_generate_delta_list(1,cam,dam);

        set_d_mark_table();

        check_d_mark_table(aam,bam,cam,dam);
    }
    else{
        MD_set_d_mark_table(0,aam,bam);
        MD_set_d_mark_table(1,cam,dam);
    }
	MD_set_QR_table(ipq_inner_loop,aam,bam,cam,dam);

	MD_R_init(aam,bam,cam,dam);

	TSMJ_output_K(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,tail,ft_tail,ipq_inner_loop);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();
    if(bCSE){
        CSE_finish();
    }
}

void MD_gpu_K(FILE * fp,int aam,int bam, int cam,int dam,int ipq,char * tail){
    int ipq_inner_loop;
    if((aam+bam)>=(cam+dam)) ipq_inner_loop=1;
    else ipq_inner_loop=0;
    ipq_inner_loop=ipq;
	printf("MD_K_kernel[%d][%d]=MD_%c%c%c%c_K%s;\n",\
        get_id_k(aam,bam),get_id_k(cam,dam),\
        shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam],tail);

	MD_set_d_mark_table(0,aam,bam);
	MD_set_d_mark_table(1,cam,dam);
	//check_d_mark_table(aam,bam,cam,dam);

	MD_set_QR_table(ipq_inner_loop,aam,bam,cam,dam);

	MD_R_init(aam,bam,cam,dam);

	MD_output_K(fp,aam,bam,cam,dam,tail,ipq_inner_loop);

	zero_d_mark_table();
	MD_zero_QR_table();
	zero_R_mark_table();
}

int main(int argc, char * argv[]){
	int max_am=2;
	int max_tot_am=8;
	int b_start,b_end;
	int c_start,c_end;
	int d_start,d_end;
	init_am_num();
    //output_function_ptr( max_am, max_tot_am, order);
    /*
    FILE * thp;
    FILE * mhp;
    thp=fopen("TSMJ_code/TSMJ_gpu.h","a");
    //TSMJ_h_define(thp,"TSMJ_GPU");
    mhp=fopen("MD_code/MD_gpu.h","a");*/
    FILE * fp;
    char filename[50];
    bool bDform=false;
    bool bNRR=false;
    bool bCSE=false;
    bool bTex=false;
    bool bJME=false;
/*
	c_start=0;
	c_end=max_am;
    for(int cam=c_start;cam<=c_end;cam++)
    {
	d_start=0;
	d_end=cam;
    for(int dam=d_start;dam<=d_end;dam++)
    {
        sprintf(filename,"TSMJ_code/TSMJ_%c%c_cuda.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_head(fp,"TSMJ_",cam,dam,true);
    //TSMJ_h_define(mhp,"MD_GPU");
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=aam;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
        //bDform=true;bNRR=false;bCSE=false;bTex=true;bJME=false;
		//TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,"_D","_fs");

        //bDform=false;bNRR=true;bCSE=false;bTex=true;bJME=false;
		//TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,"_NRR","_fs");

        //bDform=false;bNRR=false;bCSE=true;bTex=true;bJME=false;
		//TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,"_CSE","_fs");

        bDform=true;bNRR=true;bCSE=true;bTex=true;bJME=false;
		TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,"_taylor","_taylor");

        //bDform=true;bNRR=true;bCSE=true;bTex=false;bJME=false;
		//TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,"_NTX","_fs");

        bDform=true;bNRR=true;bCSE=true;bTex=true;bJME=false;
		TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,"_fs","_fs");

        bDform=true;bNRR=true;bCSE=true;bTex=true;bJME=true;
		TSMJ_gpu_J(fp,aam,bam,cam,dam,bDform,bNRR,bCSE,bTex,bJME,"_JME","_fs");

		//gpu_J_h(thp,"TSMJ",kernel_tail,aam,bam,cam,dam);
    }
    }
        fclose(fp);
    }
    }
*/
	c_start=0;
	c_end=max_am;
    for(int cam=c_start;cam<=c_end;cam++)
    {
	d_start=0;
	d_end=cam;
    for(int dam=d_start;dam<=d_end;dam++)
    {
        sprintf(filename,"HGP_code/HGP_%c%c_cuda.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_head(fp,"HGP_",cam,dam,true);
    //TSMJ_h_define(mhp,"MD_GPU");
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=aam;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
		HGP_gpu(fp,aam,bam,cam,dam);

		//gpu_J_h(thp,"TSMJ",kernel_tail,aam,bam,cam,dam);
    }
    }
        fclose(fp);
    }
    }

/*
	c_start=0;
	c_end=max_am;
    for(int cam=c_start;cam<=c_end;cam++)
    {
	d_start=0;
	d_end=cam;
    for(int dam=d_start;dam<=d_end;dam++)
    {
        sprintf(filename,"MD_code/MD_%c%c_cuda.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_head(fp,"MD_",cam,dam,true);
    //TSMJ_h_define(mhp,"MD_GPU");
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=aam;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
		MD_gpu_J(fp,aam,bam,cam,dam,"_fs");
		MD_gpu_J(fp,aam,bam,cam,dam,"_taylor");
		MD_gpu_J(fp,aam,bam,cam,dam,"_poly");
		//MD_gpu_J(fp,aam,bam,cam,dam,"_taylor");
		//gpu_J_h(mhp,"MD",kernel_tail,aam,bam,cam,dam);
    }
    }
        fclose(fp);
    }
    }
*/
/*
    for(int cam=0;cam<=max_am;cam++)
    {
    for(int dam=0;dam<=max_am;dam++)
    {*/
/*
    for(int cam=2;cam<=max_am;cam++)
    {
    for(int dam=2;dam<=max_am;dam++)
    {
        sprintf(filename,"TSMJ_code/TSMJ_K_%c%c_cuda_D.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_K_head(fp,"TSMJ_",cam,dam);
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=max_am;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        if(get_id_k(aam,bam) < get_id_k(cam,dam)) continue;
        printf("%d %d\n",get_id_k(aam,bam),get_id_k(cam,dam));
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
        bDform=true;bNRR=false;bCSE=false;
		TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_D","_fs");
		TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_D","_fs");
        //bDform=false;bNRR=true;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_NRR","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_NRR","_fs");
        //bDform=false;bNRR=false;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_CSE","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_CSE","_fs");
        //bDform=true;bNRR=true;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_fs","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_fs","_fs");
    }
    }
        fclose(fp);
    }
    }

    for(int cam=2;cam<=max_am;cam++)
    {
    for(int dam=2;dam<=max_am;dam++)
    {
        sprintf(filename,"TSMJ_code/TSMJ_K_%c%c_cuda_NRR.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_K_head(fp,"TSMJ_",cam,dam);
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=max_am;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        if(get_id_k(aam,bam) < get_id_k(cam,dam)) continue;
        printf("%d %d\n",get_id_k(aam,bam),get_id_k(cam,dam));
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);

        //bDform=true;bNRR=false;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_D","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_D","_fs");
        bDform=false;bNRR=true;bCSE=false;
		TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_NRR","_fs");
		TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_NRR","_fs");
        //bDform=false;bNRR=false;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_CSE","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_CSE","_fs");
        //bDform=true;bNRR=true;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_fs","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_fs","_fs");
    }
    }
        fclose(fp);
    }
    }
    for(int cam=2;cam<=max_am;cam++)
    {
    for(int dam=2;dam<=max_am;dam++)
    {
        sprintf(filename,"TSMJ_code/TSMJ_K_%c%c_cuda_CSE.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_K_head(fp,"TSMJ_",cam,dam);
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=max_am;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        if(get_id_k(aam,bam) < get_id_k(cam,dam)) continue;
        printf("%d %d\n",get_id_k(aam,bam),get_id_k(cam,dam));
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);

        //bDform=true;bNRR=false;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_D","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_D","_fs");
        //bDform=false;bNRR=true;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_NRR","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_NRR","_fs");
        bDform=false;bNRR=false;bCSE=true;
		TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_CSE","_fs");
		TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_CSE","_fs");

        //bDform=true;bNRR=true;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_fs","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_fs","_fs");
    }
    }
        fclose(fp);
    }
    }
    for(int cam=2;cam<=max_am;cam++)
    {
    for(int dam=2;dam<=max_am;dam++)
    {
        sprintf(filename,"TSMJ_code/TSMJ_K_%c%c_cuda.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_K_head(fp,"TSMJ_",cam,dam);
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=max_am;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        if(get_id_k(aam,bam) < get_id_k(cam,dam)) continue;
        printf("%d %d\n",get_id_k(aam,bam),get_id_k(cam,dam));
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);

        //bDform=true;bNRR=false;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_D","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_D","_fs");
        //bDform=false;bNRR=true;bCSE=false;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_NRR","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_NRR","_fs");
        //bDform=false;bNRR=false;bCSE=true;
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_CSE","_fs");
		//TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_CSE","_fs");
        bDform=true;bNRR=true;bCSE=true;
		TSMJ_gpu_K(fp,aam,bam,cam,dam,0,bDform,bNRR,bCSE,"_fs","_fs");
		TSMJ_gpu_K(fp,aam,bam,cam,dam,1,bDform,bNRR,bCSE,"_fs","_fs");
    }
    }
        fclose(fp);
    }
    }
*/
/*
    for(int cam=0;cam<=max_am;cam++)
    {
    for(int dam=0;dam<=max_am;dam++)
    {
        sprintf(filename,"MD_code/MD_K_%c%c_cuda.cu",shell_name[cam],shell_name[dam]);
        fp=fopen(filename,"w");
        gpu_output_K_head(fp,"MD_",cam,dam);
    for(int aam=0;aam<=max_am;aam++)
    {
    for(int bam=0;bam<=max_am;bam++)
    {
        if(aam+bam+cam+dam>max_tot_am ) continue;
        if(get_id_k(aam,bam) < get_id_k(cam,dam)) continue;
        //printf("%d %d\n",get_id_k(aam,bam),get_id_k(cam,dam));
        //printf("%c%c%c%c\n",shell_name[aam],shell_name[bam],shell_name[cam],shell_name[dam]);
		MD_gpu_K(fp,aam,bam,cam,dam,0,"_fs");
		MD_gpu_K(fp,aam,bam,cam,dam,1,"_fs");
		//MD_gpu_K(fp,aam,bam,cam,dam,"_taylor");
		//MD_gpu_K(fp,aam,bam,cam,dam,"_poly");
    }
    }
        fclose(fp);
    }
    }
*/

/*
    TSMJ_h_end(thp,"TSMJ_GPU");
    fclose(thp);
    TSMJ_h_end(mhp,"MD_GPU");
    fclose(mhp);*/
	return 0;
}
