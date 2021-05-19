#include<math.h>
#include"Boys_gpu.h"
#define PI 3.1415926535897932
#define P25 17.4934183276248620
#define NTHREAD 64

texture<int2,1,cudaReadModeElementType> tex_P;
texture<int2,1,cudaReadModeElementType> tex_Zta;
texture<int2,1,cudaReadModeElementType> tex_pp;
texture<float,1,cudaReadModeElementType> tex_K2_p;
texture<int2,1,cudaReadModeElementType> tex_PA;
texture<int2,1,cudaReadModeElementType> tex_PB;
texture<unsigned int,1,cudaReadModeElementType> tex_id_bra;
texture<int2,1,cudaReadModeElementType> tex_Q;
texture<int2,1,cudaReadModeElementType> tex_Eta;
texture<int2,1,cudaReadModeElementType> tex_pq;
texture<float,1,cudaReadModeElementType> tex_K2_q;
texture<int2,1,cudaReadModeElementType> tex_QC;
texture<int2,1,cudaReadModeElementType> tex_QD;
texture<unsigned int,1,cudaReadModeElementType> tex_id_ket;

void MD_texture_binding_bra_ds(double * P_d,double * PA_d,double * PB_d,\
        double * alphaP_d,double * pp_d,float * K2_p_d,unsigned int * id_bra_d,\
        unsigned int primit_len){
    cudaBindTexture(0, tex_P, P_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_Zta, alphaP_d, sizeof(double)*primit_len);
    cudaBindTexture(0, tex_pp, pp_d, sizeof(double)*primit_len);
    cudaBindTexture(0, tex_K2_p, K2_p_d, sizeof(float)*primit_len);
    cudaBindTexture(0, tex_PA, PA_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_PB, PB_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_id_bra, id_bra_d, sizeof(unsigned int)*primit_len);

}
void MD_texture_unbind_bra_ds(){
    cudaUnbindTexture(tex_P);
    cudaUnbindTexture(tex_Zta);
    cudaUnbindTexture(tex_pp);
    cudaUnbindTexture(tex_K2_p);
    cudaUnbindTexture(tex_PA);
    cudaUnbindTexture(tex_PB);
    cudaUnbindTexture(tex_id_bra);

}

void MD_texture_binding_ket_ds(double * Q_d,double * QC_d,double * QD_d,\
        double * alphaQ_d,double * pq_d,float * K2_q_d,unsigned int * id_ket_d,\
        unsigned int primit_len){
    cudaBindTexture(0, tex_Q, Q_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_Eta, alphaQ_d, sizeof(double)*primit_len);
    cudaBindTexture(0, tex_pq, pq_d, sizeof(double)*primit_len);
    cudaBindTexture(0, tex_K2_q, K2_q_d, sizeof(float)*primit_len);
    cudaBindTexture(0, tex_QC, QC_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_QD, QD_d, sizeof(double)*primit_len*3);
    cudaBindTexture(0, tex_id_ket, id_ket_d, sizeof(unsigned int)*primit_len);

}
void MD_texture_unbind_ket_ds(){
    cudaUnbindTexture(tex_Q);
    cudaUnbindTexture(tex_Eta);
    cudaUnbindTexture(tex_pq);
    cudaUnbindTexture(tex_K2_q);
    cudaUnbindTexture(tex_QC);
    cudaUnbindTexture(tex_QD);
    cudaUnbindTexture(tex_id_ket);

}
__global__ void MD_Kp_sdds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[ii*3+0];
				Qd_010[1]=QC[ii*3+1];
				Qd_010[2]=QC[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_p=tex1Dfetch(tex_K2_p,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Zta,jj);
            double Zta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pp,jj);
            double pp=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+0);
            double PX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+1);
            double PY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+2);
            double PZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_001[3];
            temp_int2=tex1Dfetch(tex_PB,jj*3+0);
            Pd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+1);
            Pd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+2);
            Pd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_fs_4(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[4];
	double R_200[3];
	double R_300[2];
	double R_400[1];
	double R_010[4];
	double R_110[3];
	double R_210[2];
	double R_310[1];
	double R_020[3];
	double R_120[2];
	double R_220[1];
	double R_030[2];
	double R_130[1];
	double R_040[1];
	double R_001[4];
	double R_101[3];
	double R_201[2];
	double R_301[1];
	double R_011[3];
	double R_111[2];
	double R_211[1];
	double R_021[2];
	double R_121[1];
	double R_031[1];
	double R_002[3];
	double R_102[2];
	double R_202[1];
	double R_012[2];
	double R_112[1];
	double R_022[1];
	double R_003[2];
	double R_103[1];
	double R_013[1];
	double R_004[1];
	for(int i=0;i<4;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<2;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<2;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
	double P_002000000=Pd_002[0];
	double P_102000000=Pd_102[0];
	double P_202000000=Pd_202[0];
	double P_001001000=Pd_001[0]*Pd_001[1];
	double P_001101000=Pd_001[0]*Pd_101[1];
	double P_101001000=Pd_101[0]*Pd_001[1];
	double P_101101000=Pd_101[0]*Pd_101[1];
	double P_000002000=Pd_002[1];
	double P_000102000=Pd_102[1];
	double P_000202000=Pd_202[1];
	double P_001000001=Pd_001[0]*Pd_001[2];
	double P_001000101=Pd_001[0]*Pd_101[2];
	double P_101000001=Pd_101[0]*Pd_001[2];
	double P_101000101=Pd_101[0]*Pd_101[2];
	double P_000001001=Pd_001[1]*Pd_001[2];
	double P_000001101=Pd_001[1]*Pd_101[2];
	double P_000101001=Pd_101[1]*Pd_001[2];
	double P_000101101=Pd_101[1]*Pd_101[2];
	double P_000000002=Pd_002[2];
	double P_000000102=Pd_102[2];
	double P_000000202=Pd_202[2];
				double PR_002000000000=P_002000000*R_000[0]+-1*P_102000000*R_100[0]+P_202000000*R_200[0];
				double PR_001001000000=P_001001000*R_000[0]+-1*P_001101000*R_010[0]+-1*P_101001000*R_100[0]+P_101101000*R_110[0];
				double PR_000002000000=P_000002000*R_000[0]+-1*P_000102000*R_010[0]+P_000202000*R_020[0];
				double PR_001000001000=P_001000001*R_000[0]+-1*P_001000101*R_001[0]+-1*P_101000001*R_100[0]+P_101000101*R_101[0];
				double PR_000001001000=P_000001001*R_000[0]+-1*P_000001101*R_001[0]+-1*P_000101001*R_010[0]+P_000101101*R_011[0];
				double PR_000000002000=P_000000002*R_000[0]+-1*P_000000102*R_001[0]+P_000000202*R_002[0];
				double PR_002000000001=P_002000000*R_001[0]+-1*P_102000000*R_101[0]+P_202000000*R_201[0];
				double PR_001001000001=P_001001000*R_001[0]+-1*P_001101000*R_011[0]+-1*P_101001000*R_101[0]+P_101101000*R_111[0];
				double PR_000002000001=P_000002000*R_001[0]+-1*P_000102000*R_011[0]+P_000202000*R_021[0];
				double PR_001000001001=P_001000001*R_001[0]+-1*P_001000101*R_002[0]+-1*P_101000001*R_101[0]+P_101000101*R_102[0];
				double PR_000001001001=P_000001001*R_001[0]+-1*P_000001101*R_002[0]+-1*P_000101001*R_011[0]+P_000101101*R_012[0];
				double PR_000000002001=P_000000002*R_001[0]+-1*P_000000102*R_002[0]+P_000000202*R_003[0];
				double PR_002000000010=P_002000000*R_010[0]+-1*P_102000000*R_110[0]+P_202000000*R_210[0];
				double PR_001001000010=P_001001000*R_010[0]+-1*P_001101000*R_020[0]+-1*P_101001000*R_110[0]+P_101101000*R_120[0];
				double PR_000002000010=P_000002000*R_010[0]+-1*P_000102000*R_020[0]+P_000202000*R_030[0];
				double PR_001000001010=P_001000001*R_010[0]+-1*P_001000101*R_011[0]+-1*P_101000001*R_110[0]+P_101000101*R_111[0];
				double PR_000001001010=P_000001001*R_010[0]+-1*P_000001101*R_011[0]+-1*P_000101001*R_020[0]+P_000101101*R_021[0];
				double PR_000000002010=P_000000002*R_010[0]+-1*P_000000102*R_011[0]+P_000000202*R_012[0];
				double PR_002000000100=P_002000000*R_100[0]+-1*P_102000000*R_200[0]+P_202000000*R_300[0];
				double PR_001001000100=P_001001000*R_100[0]+-1*P_001101000*R_110[0]+-1*P_101001000*R_200[0]+P_101101000*R_210[0];
				double PR_000002000100=P_000002000*R_100[0]+-1*P_000102000*R_110[0]+P_000202000*R_120[0];
				double PR_001000001100=P_001000001*R_100[0]+-1*P_001000101*R_101[0]+-1*P_101000001*R_200[0]+P_101000101*R_201[0];
				double PR_000001001100=P_000001001*R_100[0]+-1*P_000001101*R_101[0]+-1*P_000101001*R_110[0]+P_000101101*R_111[0];
				double PR_000000002100=P_000000002*R_100[0]+-1*P_000000102*R_101[0]+P_000000202*R_102[0];
				double PR_002000000002=P_002000000*R_002[0]+-1*P_102000000*R_102[0]+P_202000000*R_202[0];
				double PR_001001000002=P_001001000*R_002[0]+-1*P_001101000*R_012[0]+-1*P_101001000*R_102[0]+P_101101000*R_112[0];
				double PR_000002000002=P_000002000*R_002[0]+-1*P_000102000*R_012[0]+P_000202000*R_022[0];
				double PR_001000001002=P_001000001*R_002[0]+-1*P_001000101*R_003[0]+-1*P_101000001*R_102[0]+P_101000101*R_103[0];
				double PR_000001001002=P_000001001*R_002[0]+-1*P_000001101*R_003[0]+-1*P_000101001*R_012[0]+P_000101101*R_013[0];
				double PR_000000002002=P_000000002*R_002[0]+-1*P_000000102*R_003[0]+P_000000202*R_004[0];
				double PR_002000000011=P_002000000*R_011[0]+-1*P_102000000*R_111[0]+P_202000000*R_211[0];
				double PR_001001000011=P_001001000*R_011[0]+-1*P_001101000*R_021[0]+-1*P_101001000*R_111[0]+P_101101000*R_121[0];
				double PR_000002000011=P_000002000*R_011[0]+-1*P_000102000*R_021[0]+P_000202000*R_031[0];
				double PR_001000001011=P_001000001*R_011[0]+-1*P_001000101*R_012[0]+-1*P_101000001*R_111[0]+P_101000101*R_112[0];
				double PR_000001001011=P_000001001*R_011[0]+-1*P_000001101*R_012[0]+-1*P_000101001*R_021[0]+P_000101101*R_022[0];
				double PR_000000002011=P_000000002*R_011[0]+-1*P_000000102*R_012[0]+P_000000202*R_013[0];
				double PR_002000000020=P_002000000*R_020[0]+-1*P_102000000*R_120[0]+P_202000000*R_220[0];
				double PR_001001000020=P_001001000*R_020[0]+-1*P_001101000*R_030[0]+-1*P_101001000*R_120[0]+P_101101000*R_130[0];
				double PR_000002000020=P_000002000*R_020[0]+-1*P_000102000*R_030[0]+P_000202000*R_040[0];
				double PR_001000001020=P_001000001*R_020[0]+-1*P_001000101*R_021[0]+-1*P_101000001*R_120[0]+P_101000101*R_121[0];
				double PR_000001001020=P_000001001*R_020[0]+-1*P_000001101*R_021[0]+-1*P_000101001*R_030[0]+P_000101101*R_031[0];
				double PR_000000002020=P_000000002*R_020[0]+-1*P_000000102*R_021[0]+P_000000202*R_022[0];
				double PR_002000000101=P_002000000*R_101[0]+-1*P_102000000*R_201[0]+P_202000000*R_301[0];
				double PR_001001000101=P_001001000*R_101[0]+-1*P_001101000*R_111[0]+-1*P_101001000*R_201[0]+P_101101000*R_211[0];
				double PR_000002000101=P_000002000*R_101[0]+-1*P_000102000*R_111[0]+P_000202000*R_121[0];
				double PR_001000001101=P_001000001*R_101[0]+-1*P_001000101*R_102[0]+-1*P_101000001*R_201[0]+P_101000101*R_202[0];
				double PR_000001001101=P_000001001*R_101[0]+-1*P_000001101*R_102[0]+-1*P_000101001*R_111[0]+P_000101101*R_112[0];
				double PR_000000002101=P_000000002*R_101[0]+-1*P_000000102*R_102[0]+P_000000202*R_103[0];
				double PR_002000000110=P_002000000*R_110[0]+-1*P_102000000*R_210[0]+P_202000000*R_310[0];
				double PR_001001000110=P_001001000*R_110[0]+-1*P_001101000*R_120[0]+-1*P_101001000*R_210[0]+P_101101000*R_220[0];
				double PR_000002000110=P_000002000*R_110[0]+-1*P_000102000*R_120[0]+P_000202000*R_130[0];
				double PR_001000001110=P_001000001*R_110[0]+-1*P_001000101*R_111[0]+-1*P_101000001*R_210[0]+P_101000101*R_211[0];
				double PR_000001001110=P_000001001*R_110[0]+-1*P_000001101*R_111[0]+-1*P_000101001*R_120[0]+P_000101101*R_121[0];
				double PR_000000002110=P_000000002*R_110[0]+-1*P_000000102*R_111[0]+P_000000202*R_112[0];
				double PR_002000000200=P_002000000*R_200[0]+-1*P_102000000*R_300[0]+P_202000000*R_400[0];
				double PR_001001000200=P_001001000*R_200[0]+-1*P_001101000*R_210[0]+-1*P_101001000*R_300[0]+P_101101000*R_310[0];
				double PR_000002000200=P_000002000*R_200[0]+-1*P_000102000*R_210[0]+P_000202000*R_220[0];
				double PR_001000001200=P_001000001*R_200[0]+-1*P_001000101*R_201[0]+-1*P_101000001*R_300[0]+P_101000101*R_301[0];
				double PR_000001001200=P_000001001*R_200[0]+-1*P_000001101*R_201[0]+-1*P_000101001*R_210[0]+P_000101101*R_211[0];
				double PR_000000002200=P_000000002*R_200[0]+-1*P_000000102*R_201[0]+P_000000202*R_202[0];
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(Q_020000000*PR_002000000000+Q_120000000*PR_002000000100+Q_220000000*PR_002000000200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(Q_010010000*PR_002000000000+Q_010110000*PR_002000000010+Q_110010000*PR_002000000100+Q_110110000*PR_002000000110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(Q_000020000*PR_002000000000+Q_000120000*PR_002000000010+Q_000220000*PR_002000000020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(Q_010000010*PR_002000000000+Q_010000110*PR_002000000001+Q_110000010*PR_002000000100+Q_110000110*PR_002000000101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(Q_000010010*PR_002000000000+Q_000010110*PR_002000000001+Q_000110010*PR_002000000010+Q_000110110*PR_002000000011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(Q_000000020*PR_002000000000+Q_000000120*PR_002000000001+Q_000000220*PR_002000000002);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(Q_020000000*PR_001001000000+Q_120000000*PR_001001000100+Q_220000000*PR_001001000200);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(Q_010010000*PR_001001000000+Q_010110000*PR_001001000010+Q_110010000*PR_001001000100+Q_110110000*PR_001001000110);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(Q_000020000*PR_001001000000+Q_000120000*PR_001001000010+Q_000220000*PR_001001000020);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(Q_010000010*PR_001001000000+Q_010000110*PR_001001000001+Q_110000010*PR_001001000100+Q_110000110*PR_001001000101);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(Q_000010010*PR_001001000000+Q_000010110*PR_001001000001+Q_000110010*PR_001001000010+Q_000110110*PR_001001000011);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(Q_000000020*PR_001001000000+Q_000000120*PR_001001000001+Q_000000220*PR_001001000002);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(Q_020000000*PR_000002000000+Q_120000000*PR_000002000100+Q_220000000*PR_000002000200);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(Q_010010000*PR_000002000000+Q_010110000*PR_000002000010+Q_110010000*PR_000002000100+Q_110110000*PR_000002000110);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(Q_000020000*PR_000002000000+Q_000120000*PR_000002000010+Q_000220000*PR_000002000020);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(Q_010000010*PR_000002000000+Q_010000110*PR_000002000001+Q_110000010*PR_000002000100+Q_110000110*PR_000002000101);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(Q_000010010*PR_000002000000+Q_000010110*PR_000002000001+Q_000110010*PR_000002000010+Q_000110110*PR_000002000011);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(Q_000000020*PR_000002000000+Q_000000120*PR_000002000001+Q_000000220*PR_000002000002);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(Q_020000000*PR_001000001000+Q_120000000*PR_001000001100+Q_220000000*PR_001000001200);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(Q_010010000*PR_001000001000+Q_010110000*PR_001000001010+Q_110010000*PR_001000001100+Q_110110000*PR_001000001110);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(Q_000020000*PR_001000001000+Q_000120000*PR_001000001010+Q_000220000*PR_001000001020);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(Q_010000010*PR_001000001000+Q_010000110*PR_001000001001+Q_110000010*PR_001000001100+Q_110000110*PR_001000001101);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(Q_000010010*PR_001000001000+Q_000010110*PR_001000001001+Q_000110010*PR_001000001010+Q_000110110*PR_001000001011);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(Q_000000020*PR_001000001000+Q_000000120*PR_001000001001+Q_000000220*PR_001000001002);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(Q_020000000*PR_000001001000+Q_120000000*PR_000001001100+Q_220000000*PR_000001001200);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(Q_010010000*PR_000001001000+Q_010110000*PR_000001001010+Q_110010000*PR_000001001100+Q_110110000*PR_000001001110);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(Q_000020000*PR_000001001000+Q_000120000*PR_000001001010+Q_000220000*PR_000001001020);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(Q_010000010*PR_000001001000+Q_010000110*PR_000001001001+Q_110000010*PR_000001001100+Q_110000110*PR_000001001101);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(Q_000010010*PR_000001001000+Q_000010110*PR_000001001001+Q_000110010*PR_000001001010+Q_000110110*PR_000001001011);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(Q_000000020*PR_000001001000+Q_000000120*PR_000001001001+Q_000000220*PR_000001001002);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(Q_020000000*PR_000000002000+Q_120000000*PR_000000002100+Q_220000000*PR_000000002200);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(Q_010010000*PR_000000002000+Q_010110000*PR_000000002010+Q_110010000*PR_000000002100+Q_110110000*PR_000000002110);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(Q_000020000*PR_000000002000+Q_000120000*PR_000000002010+Q_000220000*PR_000000002020);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(Q_010000010*PR_000000002000+Q_010000110*PR_000000002001+Q_110000010*PR_000000002100+Q_110000110*PR_000000002101);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(Q_000010010*PR_000000002000+Q_000010110*PR_000000002001+Q_000110010*PR_000000002010+Q_000110110*PR_000000002011);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(Q_000000020*PR_000000002000+Q_000000120*PR_000000002001+Q_000000220*PR_000000002002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<6;ians++){
                    ans_temp[tId_x*6+ians]+=ans_temp[(tId_x+num_thread)*6+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<6;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_sdds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
            unsigned int id_bra=id_bra_in[ii];
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_001[3];
				Pd_001[0]=PB[ii*3+0];
				Pd_001[1]=PB[ii*3+1];
				Pd_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
            float K2_p=K2_p_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_end-primit_ket_start;j+=tdis){
            unsigned int jj=primit_ket_start+j;
            unsigned int id_ket=tex1Dfetch(tex_id_ket,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Eta,jj);
            double Eta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pq,jj);
            double pq=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+0);
            double QX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+1);
            double QY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+2);
            double QZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Qd_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            Qd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            Qd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            Qd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_fs_4(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[4];
	double R_200[3];
	double R_300[2];
	double R_400[1];
	double R_010[4];
	double R_110[3];
	double R_210[2];
	double R_310[1];
	double R_020[3];
	double R_120[2];
	double R_220[1];
	double R_030[2];
	double R_130[1];
	double R_040[1];
	double R_001[4];
	double R_101[3];
	double R_201[2];
	double R_301[1];
	double R_011[3];
	double R_111[2];
	double R_211[1];
	double R_021[2];
	double R_121[1];
	double R_031[1];
	double R_002[3];
	double R_102[2];
	double R_202[1];
	double R_012[2];
	double R_112[1];
	double R_022[1];
	double R_003[2];
	double R_103[1];
	double R_013[1];
	double R_004[1];
	for(int i=0;i<4;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<2;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<2;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
				double QR_020000000000=Q_020000000*R_000[0]+-1*Q_120000000*R_100[0]+Q_220000000*R_200[0];
				double QR_010010000000=Q_010010000*R_000[0]+-1*Q_010110000*R_010[0]+-1*Q_110010000*R_100[0]+Q_110110000*R_110[0];
				double QR_000020000000=Q_000020000*R_000[0]+-1*Q_000120000*R_010[0]+Q_000220000*R_020[0];
				double QR_010000010000=Q_010000010*R_000[0]+-1*Q_010000110*R_001[0]+-1*Q_110000010*R_100[0]+Q_110000110*R_101[0];
				double QR_000010010000=Q_000010010*R_000[0]+-1*Q_000010110*R_001[0]+-1*Q_000110010*R_010[0]+Q_000110110*R_011[0];
				double QR_000000020000=Q_000000020*R_000[0]+-1*Q_000000120*R_001[0]+Q_000000220*R_002[0];
				double QR_020000000001=Q_020000000*R_001[0]+-1*Q_120000000*R_101[0]+Q_220000000*R_201[0];
				double QR_010010000001=Q_010010000*R_001[0]+-1*Q_010110000*R_011[0]+-1*Q_110010000*R_101[0]+Q_110110000*R_111[0];
				double QR_000020000001=Q_000020000*R_001[0]+-1*Q_000120000*R_011[0]+Q_000220000*R_021[0];
				double QR_010000010001=Q_010000010*R_001[0]+-1*Q_010000110*R_002[0]+-1*Q_110000010*R_101[0]+Q_110000110*R_102[0];
				double QR_000010010001=Q_000010010*R_001[0]+-1*Q_000010110*R_002[0]+-1*Q_000110010*R_011[0]+Q_000110110*R_012[0];
				double QR_000000020001=Q_000000020*R_001[0]+-1*Q_000000120*R_002[0]+Q_000000220*R_003[0];
				double QR_020000000010=Q_020000000*R_010[0]+-1*Q_120000000*R_110[0]+Q_220000000*R_210[0];
				double QR_010010000010=Q_010010000*R_010[0]+-1*Q_010110000*R_020[0]+-1*Q_110010000*R_110[0]+Q_110110000*R_120[0];
				double QR_000020000010=Q_000020000*R_010[0]+-1*Q_000120000*R_020[0]+Q_000220000*R_030[0];
				double QR_010000010010=Q_010000010*R_010[0]+-1*Q_010000110*R_011[0]+-1*Q_110000010*R_110[0]+Q_110000110*R_111[0];
				double QR_000010010010=Q_000010010*R_010[0]+-1*Q_000010110*R_011[0]+-1*Q_000110010*R_020[0]+Q_000110110*R_021[0];
				double QR_000000020010=Q_000000020*R_010[0]+-1*Q_000000120*R_011[0]+Q_000000220*R_012[0];
				double QR_020000000100=Q_020000000*R_100[0]+-1*Q_120000000*R_200[0]+Q_220000000*R_300[0];
				double QR_010010000100=Q_010010000*R_100[0]+-1*Q_010110000*R_110[0]+-1*Q_110010000*R_200[0]+Q_110110000*R_210[0];
				double QR_000020000100=Q_000020000*R_100[0]+-1*Q_000120000*R_110[0]+Q_000220000*R_120[0];
				double QR_010000010100=Q_010000010*R_100[0]+-1*Q_010000110*R_101[0]+-1*Q_110000010*R_200[0]+Q_110000110*R_201[0];
				double QR_000010010100=Q_000010010*R_100[0]+-1*Q_000010110*R_101[0]+-1*Q_000110010*R_110[0]+Q_000110110*R_111[0];
				double QR_000000020100=Q_000000020*R_100[0]+-1*Q_000000120*R_101[0]+Q_000000220*R_102[0];
				double QR_020000000002=Q_020000000*R_002[0]+-1*Q_120000000*R_102[0]+Q_220000000*R_202[0];
				double QR_010010000002=Q_010010000*R_002[0]+-1*Q_010110000*R_012[0]+-1*Q_110010000*R_102[0]+Q_110110000*R_112[0];
				double QR_000020000002=Q_000020000*R_002[0]+-1*Q_000120000*R_012[0]+Q_000220000*R_022[0];
				double QR_010000010002=Q_010000010*R_002[0]+-1*Q_010000110*R_003[0]+-1*Q_110000010*R_102[0]+Q_110000110*R_103[0];
				double QR_000010010002=Q_000010010*R_002[0]+-1*Q_000010110*R_003[0]+-1*Q_000110010*R_012[0]+Q_000110110*R_013[0];
				double QR_000000020002=Q_000000020*R_002[0]+-1*Q_000000120*R_003[0]+Q_000000220*R_004[0];
				double QR_020000000011=Q_020000000*R_011[0]+-1*Q_120000000*R_111[0]+Q_220000000*R_211[0];
				double QR_010010000011=Q_010010000*R_011[0]+-1*Q_010110000*R_021[0]+-1*Q_110010000*R_111[0]+Q_110110000*R_121[0];
				double QR_000020000011=Q_000020000*R_011[0]+-1*Q_000120000*R_021[0]+Q_000220000*R_031[0];
				double QR_010000010011=Q_010000010*R_011[0]+-1*Q_010000110*R_012[0]+-1*Q_110000010*R_111[0]+Q_110000110*R_112[0];
				double QR_000010010011=Q_000010010*R_011[0]+-1*Q_000010110*R_012[0]+-1*Q_000110010*R_021[0]+Q_000110110*R_022[0];
				double QR_000000020011=Q_000000020*R_011[0]+-1*Q_000000120*R_012[0]+Q_000000220*R_013[0];
				double QR_020000000020=Q_020000000*R_020[0]+-1*Q_120000000*R_120[0]+Q_220000000*R_220[0];
				double QR_010010000020=Q_010010000*R_020[0]+-1*Q_010110000*R_030[0]+-1*Q_110010000*R_120[0]+Q_110110000*R_130[0];
				double QR_000020000020=Q_000020000*R_020[0]+-1*Q_000120000*R_030[0]+Q_000220000*R_040[0];
				double QR_010000010020=Q_010000010*R_020[0]+-1*Q_010000110*R_021[0]+-1*Q_110000010*R_120[0]+Q_110000110*R_121[0];
				double QR_000010010020=Q_000010010*R_020[0]+-1*Q_000010110*R_021[0]+-1*Q_000110010*R_030[0]+Q_000110110*R_031[0];
				double QR_000000020020=Q_000000020*R_020[0]+-1*Q_000000120*R_021[0]+Q_000000220*R_022[0];
				double QR_020000000101=Q_020000000*R_101[0]+-1*Q_120000000*R_201[0]+Q_220000000*R_301[0];
				double QR_010010000101=Q_010010000*R_101[0]+-1*Q_010110000*R_111[0]+-1*Q_110010000*R_201[0]+Q_110110000*R_211[0];
				double QR_000020000101=Q_000020000*R_101[0]+-1*Q_000120000*R_111[0]+Q_000220000*R_121[0];
				double QR_010000010101=Q_010000010*R_101[0]+-1*Q_010000110*R_102[0]+-1*Q_110000010*R_201[0]+Q_110000110*R_202[0];
				double QR_000010010101=Q_000010010*R_101[0]+-1*Q_000010110*R_102[0]+-1*Q_000110010*R_111[0]+Q_000110110*R_112[0];
				double QR_000000020101=Q_000000020*R_101[0]+-1*Q_000000120*R_102[0]+Q_000000220*R_103[0];
				double QR_020000000110=Q_020000000*R_110[0]+-1*Q_120000000*R_210[0]+Q_220000000*R_310[0];
				double QR_010010000110=Q_010010000*R_110[0]+-1*Q_010110000*R_120[0]+-1*Q_110010000*R_210[0]+Q_110110000*R_220[0];
				double QR_000020000110=Q_000020000*R_110[0]+-1*Q_000120000*R_120[0]+Q_000220000*R_130[0];
				double QR_010000010110=Q_010000010*R_110[0]+-1*Q_010000110*R_111[0]+-1*Q_110000010*R_210[0]+Q_110000110*R_211[0];
				double QR_000010010110=Q_000010010*R_110[0]+-1*Q_000010110*R_111[0]+-1*Q_000110010*R_120[0]+Q_000110110*R_121[0];
				double QR_000000020110=Q_000000020*R_110[0]+-1*Q_000000120*R_111[0]+Q_000000220*R_112[0];
				double QR_020000000200=Q_020000000*R_200[0]+-1*Q_120000000*R_300[0]+Q_220000000*R_400[0];
				double QR_010010000200=Q_010010000*R_200[0]+-1*Q_010110000*R_210[0]+-1*Q_110010000*R_300[0]+Q_110110000*R_310[0];
				double QR_000020000200=Q_000020000*R_200[0]+-1*Q_000120000*R_210[0]+Q_000220000*R_220[0];
				double QR_010000010200=Q_010000010*R_200[0]+-1*Q_010000110*R_201[0]+-1*Q_110000010*R_300[0]+Q_110000110*R_301[0];
				double QR_000010010200=Q_000010010*R_200[0]+-1*Q_000010110*R_201[0]+-1*Q_000110010*R_210[0]+Q_000110110*R_211[0];
				double QR_000000020200=Q_000000020*R_200[0]+-1*Q_000000120*R_201[0]+Q_000000220*R_202[0];
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
	double P_002000000=Pd_002[0];
	double P_102000000=Pd_102[0];
	double P_202000000=Pd_202[0];
	double P_001001000=Pd_001[0]*Pd_001[1];
	double P_001101000=Pd_001[0]*Pd_101[1];
	double P_101001000=Pd_101[0]*Pd_001[1];
	double P_101101000=Pd_101[0]*Pd_101[1];
	double P_000002000=Pd_002[1];
	double P_000102000=Pd_102[1];
	double P_000202000=Pd_202[1];
	double P_001000001=Pd_001[0]*Pd_001[2];
	double P_001000101=Pd_001[0]*Pd_101[2];
	double P_101000001=Pd_101[0]*Pd_001[2];
	double P_101000101=Pd_101[0]*Pd_101[2];
	double P_000001001=Pd_001[1]*Pd_001[2];
	double P_000001101=Pd_001[1]*Pd_101[2];
	double P_000101001=Pd_101[1]*Pd_001[2];
	double P_000101101=Pd_101[1]*Pd_101[2];
	double P_000000002=Pd_002[2];
	double P_000000102=Pd_102[2];
	double P_000000202=Pd_202[2];
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_002000000*QR_020000000000+P_102000000*QR_020000000100+P_202000000*QR_020000000200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_002000000*QR_010010000000+P_102000000*QR_010010000100+P_202000000*QR_010010000200);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_002000000*QR_000020000000+P_102000000*QR_000020000100+P_202000000*QR_000020000200);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_002000000*QR_010000010000+P_102000000*QR_010000010100+P_202000000*QR_010000010200);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_002000000*QR_000010010000+P_102000000*QR_000010010100+P_202000000*QR_000010010200);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_002000000*QR_000000020000+P_102000000*QR_000000020100+P_202000000*QR_000000020200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_001001000*QR_020000000000+P_001101000*QR_020000000010+P_101001000*QR_020000000100+P_101101000*QR_020000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_001001000*QR_010010000000+P_001101000*QR_010010000010+P_101001000*QR_010010000100+P_101101000*QR_010010000110);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_001001000*QR_000020000000+P_001101000*QR_000020000010+P_101001000*QR_000020000100+P_101101000*QR_000020000110);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_001001000*QR_010000010000+P_001101000*QR_010000010010+P_101001000*QR_010000010100+P_101101000*QR_010000010110);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_001001000*QR_000010010000+P_001101000*QR_000010010010+P_101001000*QR_000010010100+P_101101000*QR_000010010110);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_001001000*QR_000000020000+P_001101000*QR_000000020010+P_101001000*QR_000000020100+P_101101000*QR_000000020110);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_000002000*QR_020000000000+P_000102000*QR_020000000010+P_000202000*QR_020000000020);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_000002000*QR_010010000000+P_000102000*QR_010010000010+P_000202000*QR_010010000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000002000*QR_000020000000+P_000102000*QR_000020000010+P_000202000*QR_000020000020);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_000002000*QR_010000010000+P_000102000*QR_010000010010+P_000202000*QR_010000010020);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000002000*QR_000010010000+P_000102000*QR_000010010010+P_000202000*QR_000010010020);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000002000*QR_000000020000+P_000102000*QR_000000020010+P_000202000*QR_000000020020);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_001000001*QR_020000000000+P_001000101*QR_020000000001+P_101000001*QR_020000000100+P_101000101*QR_020000000101);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_001000001*QR_010010000000+P_001000101*QR_010010000001+P_101000001*QR_010010000100+P_101000101*QR_010010000101);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_001000001*QR_000020000000+P_001000101*QR_000020000001+P_101000001*QR_000020000100+P_101000101*QR_000020000101);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_001000001*QR_010000010000+P_001000101*QR_010000010001+P_101000001*QR_010000010100+P_101000101*QR_010000010101);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_001000001*QR_000010010000+P_001000101*QR_000010010001+P_101000001*QR_000010010100+P_101000101*QR_000010010101);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_001000001*QR_000000020000+P_001000101*QR_000000020001+P_101000001*QR_000000020100+P_101000101*QR_000000020101);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_000001001*QR_020000000000+P_000001101*QR_020000000001+P_000101001*QR_020000000010+P_000101101*QR_020000000011);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_000001001*QR_010010000000+P_000001101*QR_010010000001+P_000101001*QR_010010000010+P_000101101*QR_010010000011);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000001001*QR_000020000000+P_000001101*QR_000020000001+P_000101001*QR_000020000010+P_000101101*QR_000020000011);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_000001001*QR_010000010000+P_000001101*QR_010000010001+P_000101001*QR_010000010010+P_000101101*QR_010000010011);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000001001*QR_000010010000+P_000001101*QR_000010010001+P_000101001*QR_000010010010+P_000101101*QR_000010010011);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000001001*QR_000000020000+P_000001101*QR_000000020001+P_000101001*QR_000000020010+P_000101101*QR_000000020011);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_000000002*QR_020000000000+P_000000102*QR_020000000001+P_000000202*QR_020000000002);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_000000002*QR_010010000000+P_000000102*QR_010010000001+P_000000202*QR_010010000002);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000000002*QR_000020000000+P_000000102*QR_000020000001+P_000000202*QR_000020000002);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_000000002*QR_010000010000+P_000000102*QR_010000010001+P_000000202*QR_010000010002);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000000002*QR_000010010000+P_000000102*QR_000010010001+P_000000202*QR_000010010002);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000000002*QR_000000020000+P_000000102*QR_000000020001+P_000000202*QR_000000020002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<6;ians++){
                    ans_temp[tId_x*6+ians]+=ans_temp[(tId_x+num_thread)*6+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<6;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_pdds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[ii*3+0];
				Qd_010[1]=QC[ii*3+1];
				Qd_010[2]=QC[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_p=tex1Dfetch(tex_K2_p,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Zta,jj);
            double Zta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pp,jj);
            double pp=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+0);
            double PX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+1);
            double PY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+2);
            double PZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_010[3];
            temp_int2=tex1Dfetch(tex_PA,jj*3+0);
            Pd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+1);
            Pd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+2);
            Pd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_001[3];
            temp_int2=tex1Dfetch(tex_PB,jj*3+0);
            Pd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+1);
            Pd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+2);
            Pd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[6];
                Ft_fs_5(5,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[5];
	double R_200[4];
	double R_300[3];
	double R_400[2];
	double R_500[1];
	double R_010[5];
	double R_110[4];
	double R_210[3];
	double R_310[2];
	double R_410[1];
	double R_020[4];
	double R_120[3];
	double R_220[2];
	double R_320[1];
	double R_030[3];
	double R_130[2];
	double R_230[1];
	double R_040[2];
	double R_140[1];
	double R_050[1];
	double R_001[5];
	double R_101[4];
	double R_201[3];
	double R_301[2];
	double R_401[1];
	double R_011[4];
	double R_111[3];
	double R_211[2];
	double R_311[1];
	double R_021[3];
	double R_121[2];
	double R_221[1];
	double R_031[2];
	double R_131[1];
	double R_041[1];
	double R_002[4];
	double R_102[3];
	double R_202[2];
	double R_302[1];
	double R_012[3];
	double R_112[2];
	double R_212[1];
	double R_022[2];
	double R_122[1];
	double R_032[1];
	double R_003[3];
	double R_103[2];
	double R_203[1];
	double R_013[2];
	double R_113[1];
	double R_023[1];
	double R_004[2];
	double R_104[1];
	double R_014[1];
	double R_005[1];
	for(int i=0;i<5;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<3;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<3;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<1;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<1;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<1;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<1;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<1;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_312[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=2*Pd_211[i]+Pd_001[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=Pd_001[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_312[i]=aPin1*Pd_211[i];
			}
	double P_012000000=Pd_012[0];
	double P_112000000=Pd_112[0];
	double P_212000000=Pd_212[0];
	double P_312000000=Pd_312[0];
	double P_011001000=Pd_011[0]*Pd_001[1];
	double P_011101000=Pd_011[0]*Pd_101[1];
	double P_111001000=Pd_111[0]*Pd_001[1];
	double P_111101000=Pd_111[0]*Pd_101[1];
	double P_211001000=Pd_211[0]*Pd_001[1];
	double P_211101000=Pd_211[0]*Pd_101[1];
	double P_010002000=Pd_010[0]*Pd_002[1];
	double P_010102000=Pd_010[0]*Pd_102[1];
	double P_010202000=Pd_010[0]*Pd_202[1];
	double P_110002000=Pd_110[0]*Pd_002[1];
	double P_110102000=Pd_110[0]*Pd_102[1];
	double P_110202000=Pd_110[0]*Pd_202[1];
	double P_011000001=Pd_011[0]*Pd_001[2];
	double P_011000101=Pd_011[0]*Pd_101[2];
	double P_111000001=Pd_111[0]*Pd_001[2];
	double P_111000101=Pd_111[0]*Pd_101[2];
	double P_211000001=Pd_211[0]*Pd_001[2];
	double P_211000101=Pd_211[0]*Pd_101[2];
	double P_010001001=Pd_010[0]*Pd_001[1]*Pd_001[2];
	double P_010001101=Pd_010[0]*Pd_001[1]*Pd_101[2];
	double P_010101001=Pd_010[0]*Pd_101[1]*Pd_001[2];
	double P_010101101=Pd_010[0]*Pd_101[1]*Pd_101[2];
	double P_110001001=Pd_110[0]*Pd_001[1]*Pd_001[2];
	double P_110001101=Pd_110[0]*Pd_001[1]*Pd_101[2];
	double P_110101001=Pd_110[0]*Pd_101[1]*Pd_001[2];
	double P_110101101=Pd_110[0]*Pd_101[1]*Pd_101[2];
	double P_010000002=Pd_010[0]*Pd_002[2];
	double P_010000102=Pd_010[0]*Pd_102[2];
	double P_010000202=Pd_010[0]*Pd_202[2];
	double P_110000002=Pd_110[0]*Pd_002[2];
	double P_110000102=Pd_110[0]*Pd_102[2];
	double P_110000202=Pd_110[0]*Pd_202[2];
	double P_002010000=Pd_002[0]*Pd_010[1];
	double P_002110000=Pd_002[0]*Pd_110[1];
	double P_102010000=Pd_102[0]*Pd_010[1];
	double P_102110000=Pd_102[0]*Pd_110[1];
	double P_202010000=Pd_202[0]*Pd_010[1];
	double P_202110000=Pd_202[0]*Pd_110[1];
	double P_001011000=Pd_001[0]*Pd_011[1];
	double P_001111000=Pd_001[0]*Pd_111[1];
	double P_001211000=Pd_001[0]*Pd_211[1];
	double P_101011000=Pd_101[0]*Pd_011[1];
	double P_101111000=Pd_101[0]*Pd_111[1];
	double P_101211000=Pd_101[0]*Pd_211[1];
	double P_000012000=Pd_012[1];
	double P_000112000=Pd_112[1];
	double P_000212000=Pd_212[1];
	double P_000312000=Pd_312[1];
	double P_001010001=Pd_001[0]*Pd_010[1]*Pd_001[2];
	double P_001010101=Pd_001[0]*Pd_010[1]*Pd_101[2];
	double P_001110001=Pd_001[0]*Pd_110[1]*Pd_001[2];
	double P_001110101=Pd_001[0]*Pd_110[1]*Pd_101[2];
	double P_101010001=Pd_101[0]*Pd_010[1]*Pd_001[2];
	double P_101010101=Pd_101[0]*Pd_010[1]*Pd_101[2];
	double P_101110001=Pd_101[0]*Pd_110[1]*Pd_001[2];
	double P_101110101=Pd_101[0]*Pd_110[1]*Pd_101[2];
	double P_000011001=Pd_011[1]*Pd_001[2];
	double P_000011101=Pd_011[1]*Pd_101[2];
	double P_000111001=Pd_111[1]*Pd_001[2];
	double P_000111101=Pd_111[1]*Pd_101[2];
	double P_000211001=Pd_211[1]*Pd_001[2];
	double P_000211101=Pd_211[1]*Pd_101[2];
	double P_000010002=Pd_010[1]*Pd_002[2];
	double P_000010102=Pd_010[1]*Pd_102[2];
	double P_000010202=Pd_010[1]*Pd_202[2];
	double P_000110002=Pd_110[1]*Pd_002[2];
	double P_000110102=Pd_110[1]*Pd_102[2];
	double P_000110202=Pd_110[1]*Pd_202[2];
	double P_002000010=Pd_002[0]*Pd_010[2];
	double P_002000110=Pd_002[0]*Pd_110[2];
	double P_102000010=Pd_102[0]*Pd_010[2];
	double P_102000110=Pd_102[0]*Pd_110[2];
	double P_202000010=Pd_202[0]*Pd_010[2];
	double P_202000110=Pd_202[0]*Pd_110[2];
	double P_001001010=Pd_001[0]*Pd_001[1]*Pd_010[2];
	double P_001001110=Pd_001[0]*Pd_001[1]*Pd_110[2];
	double P_001101010=Pd_001[0]*Pd_101[1]*Pd_010[2];
	double P_001101110=Pd_001[0]*Pd_101[1]*Pd_110[2];
	double P_101001010=Pd_101[0]*Pd_001[1]*Pd_010[2];
	double P_101001110=Pd_101[0]*Pd_001[1]*Pd_110[2];
	double P_101101010=Pd_101[0]*Pd_101[1]*Pd_010[2];
	double P_101101110=Pd_101[0]*Pd_101[1]*Pd_110[2];
	double P_000002010=Pd_002[1]*Pd_010[2];
	double P_000002110=Pd_002[1]*Pd_110[2];
	double P_000102010=Pd_102[1]*Pd_010[2];
	double P_000102110=Pd_102[1]*Pd_110[2];
	double P_000202010=Pd_202[1]*Pd_010[2];
	double P_000202110=Pd_202[1]*Pd_110[2];
	double P_001000011=Pd_001[0]*Pd_011[2];
	double P_001000111=Pd_001[0]*Pd_111[2];
	double P_001000211=Pd_001[0]*Pd_211[2];
	double P_101000011=Pd_101[0]*Pd_011[2];
	double P_101000111=Pd_101[0]*Pd_111[2];
	double P_101000211=Pd_101[0]*Pd_211[2];
	double P_000001011=Pd_001[1]*Pd_011[2];
	double P_000001111=Pd_001[1]*Pd_111[2];
	double P_000001211=Pd_001[1]*Pd_211[2];
	double P_000101011=Pd_101[1]*Pd_011[2];
	double P_000101111=Pd_101[1]*Pd_111[2];
	double P_000101211=Pd_101[1]*Pd_211[2];
	double P_000000012=Pd_012[2];
	double P_000000112=Pd_112[2];
	double P_000000212=Pd_212[2];
	double P_000000312=Pd_312[2];
				double PR_012000000000=P_012000000*R_000[0]+-1*P_112000000*R_100[0]+P_212000000*R_200[0]+-1*P_312000000*R_300[0];
				double PR_011001000000=P_011001000*R_000[0]+-1*P_011101000*R_010[0]+-1*P_111001000*R_100[0]+P_111101000*R_110[0]+P_211001000*R_200[0]+-1*P_211101000*R_210[0];
				double PR_010002000000=P_010002000*R_000[0]+-1*P_010102000*R_010[0]+P_010202000*R_020[0]+-1*P_110002000*R_100[0]+P_110102000*R_110[0]+-1*P_110202000*R_120[0];
				double PR_011000001000=P_011000001*R_000[0]+-1*P_011000101*R_001[0]+-1*P_111000001*R_100[0]+P_111000101*R_101[0]+P_211000001*R_200[0]+-1*P_211000101*R_201[0];
				double PR_010001001000=P_010001001*R_000[0]+-1*P_010001101*R_001[0]+-1*P_010101001*R_010[0]+P_010101101*R_011[0]+-1*P_110001001*R_100[0]+P_110001101*R_101[0]+P_110101001*R_110[0]+-1*P_110101101*R_111[0];
				double PR_010000002000=P_010000002*R_000[0]+-1*P_010000102*R_001[0]+P_010000202*R_002[0]+-1*P_110000002*R_100[0]+P_110000102*R_101[0]+-1*P_110000202*R_102[0];
				double PR_002010000000=P_002010000*R_000[0]+-1*P_002110000*R_010[0]+-1*P_102010000*R_100[0]+P_102110000*R_110[0]+P_202010000*R_200[0]+-1*P_202110000*R_210[0];
				double PR_001011000000=P_001011000*R_000[0]+-1*P_001111000*R_010[0]+P_001211000*R_020[0]+-1*P_101011000*R_100[0]+P_101111000*R_110[0]+-1*P_101211000*R_120[0];
				double PR_000012000000=P_000012000*R_000[0]+-1*P_000112000*R_010[0]+P_000212000*R_020[0]+-1*P_000312000*R_030[0];
				double PR_001010001000=P_001010001*R_000[0]+-1*P_001010101*R_001[0]+-1*P_001110001*R_010[0]+P_001110101*R_011[0]+-1*P_101010001*R_100[0]+P_101010101*R_101[0]+P_101110001*R_110[0]+-1*P_101110101*R_111[0];
				double PR_000011001000=P_000011001*R_000[0]+-1*P_000011101*R_001[0]+-1*P_000111001*R_010[0]+P_000111101*R_011[0]+P_000211001*R_020[0]+-1*P_000211101*R_021[0];
				double PR_000010002000=P_000010002*R_000[0]+-1*P_000010102*R_001[0]+P_000010202*R_002[0]+-1*P_000110002*R_010[0]+P_000110102*R_011[0]+-1*P_000110202*R_012[0];
				double PR_002000010000=P_002000010*R_000[0]+-1*P_002000110*R_001[0]+-1*P_102000010*R_100[0]+P_102000110*R_101[0]+P_202000010*R_200[0]+-1*P_202000110*R_201[0];
				double PR_001001010000=P_001001010*R_000[0]+-1*P_001001110*R_001[0]+-1*P_001101010*R_010[0]+P_001101110*R_011[0]+-1*P_101001010*R_100[0]+P_101001110*R_101[0]+P_101101010*R_110[0]+-1*P_101101110*R_111[0];
				double PR_000002010000=P_000002010*R_000[0]+-1*P_000002110*R_001[0]+-1*P_000102010*R_010[0]+P_000102110*R_011[0]+P_000202010*R_020[0]+-1*P_000202110*R_021[0];
				double PR_001000011000=P_001000011*R_000[0]+-1*P_001000111*R_001[0]+P_001000211*R_002[0]+-1*P_101000011*R_100[0]+P_101000111*R_101[0]+-1*P_101000211*R_102[0];
				double PR_000001011000=P_000001011*R_000[0]+-1*P_000001111*R_001[0]+P_000001211*R_002[0]+-1*P_000101011*R_010[0]+P_000101111*R_011[0]+-1*P_000101211*R_012[0];
				double PR_000000012000=P_000000012*R_000[0]+-1*P_000000112*R_001[0]+P_000000212*R_002[0]+-1*P_000000312*R_003[0];
				double PR_012000000001=P_012000000*R_001[0]+-1*P_112000000*R_101[0]+P_212000000*R_201[0]+-1*P_312000000*R_301[0];
				double PR_011001000001=P_011001000*R_001[0]+-1*P_011101000*R_011[0]+-1*P_111001000*R_101[0]+P_111101000*R_111[0]+P_211001000*R_201[0]+-1*P_211101000*R_211[0];
				double PR_010002000001=P_010002000*R_001[0]+-1*P_010102000*R_011[0]+P_010202000*R_021[0]+-1*P_110002000*R_101[0]+P_110102000*R_111[0]+-1*P_110202000*R_121[0];
				double PR_011000001001=P_011000001*R_001[0]+-1*P_011000101*R_002[0]+-1*P_111000001*R_101[0]+P_111000101*R_102[0]+P_211000001*R_201[0]+-1*P_211000101*R_202[0];
				double PR_010001001001=P_010001001*R_001[0]+-1*P_010001101*R_002[0]+-1*P_010101001*R_011[0]+P_010101101*R_012[0]+-1*P_110001001*R_101[0]+P_110001101*R_102[0]+P_110101001*R_111[0]+-1*P_110101101*R_112[0];
				double PR_010000002001=P_010000002*R_001[0]+-1*P_010000102*R_002[0]+P_010000202*R_003[0]+-1*P_110000002*R_101[0]+P_110000102*R_102[0]+-1*P_110000202*R_103[0];
				double PR_002010000001=P_002010000*R_001[0]+-1*P_002110000*R_011[0]+-1*P_102010000*R_101[0]+P_102110000*R_111[0]+P_202010000*R_201[0]+-1*P_202110000*R_211[0];
				double PR_001011000001=P_001011000*R_001[0]+-1*P_001111000*R_011[0]+P_001211000*R_021[0]+-1*P_101011000*R_101[0]+P_101111000*R_111[0]+-1*P_101211000*R_121[0];
				double PR_000012000001=P_000012000*R_001[0]+-1*P_000112000*R_011[0]+P_000212000*R_021[0]+-1*P_000312000*R_031[0];
				double PR_001010001001=P_001010001*R_001[0]+-1*P_001010101*R_002[0]+-1*P_001110001*R_011[0]+P_001110101*R_012[0]+-1*P_101010001*R_101[0]+P_101010101*R_102[0]+P_101110001*R_111[0]+-1*P_101110101*R_112[0];
				double PR_000011001001=P_000011001*R_001[0]+-1*P_000011101*R_002[0]+-1*P_000111001*R_011[0]+P_000111101*R_012[0]+P_000211001*R_021[0]+-1*P_000211101*R_022[0];
				double PR_000010002001=P_000010002*R_001[0]+-1*P_000010102*R_002[0]+P_000010202*R_003[0]+-1*P_000110002*R_011[0]+P_000110102*R_012[0]+-1*P_000110202*R_013[0];
				double PR_002000010001=P_002000010*R_001[0]+-1*P_002000110*R_002[0]+-1*P_102000010*R_101[0]+P_102000110*R_102[0]+P_202000010*R_201[0]+-1*P_202000110*R_202[0];
				double PR_001001010001=P_001001010*R_001[0]+-1*P_001001110*R_002[0]+-1*P_001101010*R_011[0]+P_001101110*R_012[0]+-1*P_101001010*R_101[0]+P_101001110*R_102[0]+P_101101010*R_111[0]+-1*P_101101110*R_112[0];
				double PR_000002010001=P_000002010*R_001[0]+-1*P_000002110*R_002[0]+-1*P_000102010*R_011[0]+P_000102110*R_012[0]+P_000202010*R_021[0]+-1*P_000202110*R_022[0];
				double PR_001000011001=P_001000011*R_001[0]+-1*P_001000111*R_002[0]+P_001000211*R_003[0]+-1*P_101000011*R_101[0]+P_101000111*R_102[0]+-1*P_101000211*R_103[0];
				double PR_000001011001=P_000001011*R_001[0]+-1*P_000001111*R_002[0]+P_000001211*R_003[0]+-1*P_000101011*R_011[0]+P_000101111*R_012[0]+-1*P_000101211*R_013[0];
				double PR_000000012001=P_000000012*R_001[0]+-1*P_000000112*R_002[0]+P_000000212*R_003[0]+-1*P_000000312*R_004[0];
				double PR_012000000010=P_012000000*R_010[0]+-1*P_112000000*R_110[0]+P_212000000*R_210[0]+-1*P_312000000*R_310[0];
				double PR_011001000010=P_011001000*R_010[0]+-1*P_011101000*R_020[0]+-1*P_111001000*R_110[0]+P_111101000*R_120[0]+P_211001000*R_210[0]+-1*P_211101000*R_220[0];
				double PR_010002000010=P_010002000*R_010[0]+-1*P_010102000*R_020[0]+P_010202000*R_030[0]+-1*P_110002000*R_110[0]+P_110102000*R_120[0]+-1*P_110202000*R_130[0];
				double PR_011000001010=P_011000001*R_010[0]+-1*P_011000101*R_011[0]+-1*P_111000001*R_110[0]+P_111000101*R_111[0]+P_211000001*R_210[0]+-1*P_211000101*R_211[0];
				double PR_010001001010=P_010001001*R_010[0]+-1*P_010001101*R_011[0]+-1*P_010101001*R_020[0]+P_010101101*R_021[0]+-1*P_110001001*R_110[0]+P_110001101*R_111[0]+P_110101001*R_120[0]+-1*P_110101101*R_121[0];
				double PR_010000002010=P_010000002*R_010[0]+-1*P_010000102*R_011[0]+P_010000202*R_012[0]+-1*P_110000002*R_110[0]+P_110000102*R_111[0]+-1*P_110000202*R_112[0];
				double PR_002010000010=P_002010000*R_010[0]+-1*P_002110000*R_020[0]+-1*P_102010000*R_110[0]+P_102110000*R_120[0]+P_202010000*R_210[0]+-1*P_202110000*R_220[0];
				double PR_001011000010=P_001011000*R_010[0]+-1*P_001111000*R_020[0]+P_001211000*R_030[0]+-1*P_101011000*R_110[0]+P_101111000*R_120[0]+-1*P_101211000*R_130[0];
				double PR_000012000010=P_000012000*R_010[0]+-1*P_000112000*R_020[0]+P_000212000*R_030[0]+-1*P_000312000*R_040[0];
				double PR_001010001010=P_001010001*R_010[0]+-1*P_001010101*R_011[0]+-1*P_001110001*R_020[0]+P_001110101*R_021[0]+-1*P_101010001*R_110[0]+P_101010101*R_111[0]+P_101110001*R_120[0]+-1*P_101110101*R_121[0];
				double PR_000011001010=P_000011001*R_010[0]+-1*P_000011101*R_011[0]+-1*P_000111001*R_020[0]+P_000111101*R_021[0]+P_000211001*R_030[0]+-1*P_000211101*R_031[0];
				double PR_000010002010=P_000010002*R_010[0]+-1*P_000010102*R_011[0]+P_000010202*R_012[0]+-1*P_000110002*R_020[0]+P_000110102*R_021[0]+-1*P_000110202*R_022[0];
				double PR_002000010010=P_002000010*R_010[0]+-1*P_002000110*R_011[0]+-1*P_102000010*R_110[0]+P_102000110*R_111[0]+P_202000010*R_210[0]+-1*P_202000110*R_211[0];
				double PR_001001010010=P_001001010*R_010[0]+-1*P_001001110*R_011[0]+-1*P_001101010*R_020[0]+P_001101110*R_021[0]+-1*P_101001010*R_110[0]+P_101001110*R_111[0]+P_101101010*R_120[0]+-1*P_101101110*R_121[0];
				double PR_000002010010=P_000002010*R_010[0]+-1*P_000002110*R_011[0]+-1*P_000102010*R_020[0]+P_000102110*R_021[0]+P_000202010*R_030[0]+-1*P_000202110*R_031[0];
				double PR_001000011010=P_001000011*R_010[0]+-1*P_001000111*R_011[0]+P_001000211*R_012[0]+-1*P_101000011*R_110[0]+P_101000111*R_111[0]+-1*P_101000211*R_112[0];
				double PR_000001011010=P_000001011*R_010[0]+-1*P_000001111*R_011[0]+P_000001211*R_012[0]+-1*P_000101011*R_020[0]+P_000101111*R_021[0]+-1*P_000101211*R_022[0];
				double PR_000000012010=P_000000012*R_010[0]+-1*P_000000112*R_011[0]+P_000000212*R_012[0]+-1*P_000000312*R_013[0];
				double PR_012000000100=P_012000000*R_100[0]+-1*P_112000000*R_200[0]+P_212000000*R_300[0]+-1*P_312000000*R_400[0];
				double PR_011001000100=P_011001000*R_100[0]+-1*P_011101000*R_110[0]+-1*P_111001000*R_200[0]+P_111101000*R_210[0]+P_211001000*R_300[0]+-1*P_211101000*R_310[0];
				double PR_010002000100=P_010002000*R_100[0]+-1*P_010102000*R_110[0]+P_010202000*R_120[0]+-1*P_110002000*R_200[0]+P_110102000*R_210[0]+-1*P_110202000*R_220[0];
				double PR_011000001100=P_011000001*R_100[0]+-1*P_011000101*R_101[0]+-1*P_111000001*R_200[0]+P_111000101*R_201[0]+P_211000001*R_300[0]+-1*P_211000101*R_301[0];
				double PR_010001001100=P_010001001*R_100[0]+-1*P_010001101*R_101[0]+-1*P_010101001*R_110[0]+P_010101101*R_111[0]+-1*P_110001001*R_200[0]+P_110001101*R_201[0]+P_110101001*R_210[0]+-1*P_110101101*R_211[0];
				double PR_010000002100=P_010000002*R_100[0]+-1*P_010000102*R_101[0]+P_010000202*R_102[0]+-1*P_110000002*R_200[0]+P_110000102*R_201[0]+-1*P_110000202*R_202[0];
				double PR_002010000100=P_002010000*R_100[0]+-1*P_002110000*R_110[0]+-1*P_102010000*R_200[0]+P_102110000*R_210[0]+P_202010000*R_300[0]+-1*P_202110000*R_310[0];
				double PR_001011000100=P_001011000*R_100[0]+-1*P_001111000*R_110[0]+P_001211000*R_120[0]+-1*P_101011000*R_200[0]+P_101111000*R_210[0]+-1*P_101211000*R_220[0];
				double PR_000012000100=P_000012000*R_100[0]+-1*P_000112000*R_110[0]+P_000212000*R_120[0]+-1*P_000312000*R_130[0];
				double PR_001010001100=P_001010001*R_100[0]+-1*P_001010101*R_101[0]+-1*P_001110001*R_110[0]+P_001110101*R_111[0]+-1*P_101010001*R_200[0]+P_101010101*R_201[0]+P_101110001*R_210[0]+-1*P_101110101*R_211[0];
				double PR_000011001100=P_000011001*R_100[0]+-1*P_000011101*R_101[0]+-1*P_000111001*R_110[0]+P_000111101*R_111[0]+P_000211001*R_120[0]+-1*P_000211101*R_121[0];
				double PR_000010002100=P_000010002*R_100[0]+-1*P_000010102*R_101[0]+P_000010202*R_102[0]+-1*P_000110002*R_110[0]+P_000110102*R_111[0]+-1*P_000110202*R_112[0];
				double PR_002000010100=P_002000010*R_100[0]+-1*P_002000110*R_101[0]+-1*P_102000010*R_200[0]+P_102000110*R_201[0]+P_202000010*R_300[0]+-1*P_202000110*R_301[0];
				double PR_001001010100=P_001001010*R_100[0]+-1*P_001001110*R_101[0]+-1*P_001101010*R_110[0]+P_001101110*R_111[0]+-1*P_101001010*R_200[0]+P_101001110*R_201[0]+P_101101010*R_210[0]+-1*P_101101110*R_211[0];
				double PR_000002010100=P_000002010*R_100[0]+-1*P_000002110*R_101[0]+-1*P_000102010*R_110[0]+P_000102110*R_111[0]+P_000202010*R_120[0]+-1*P_000202110*R_121[0];
				double PR_001000011100=P_001000011*R_100[0]+-1*P_001000111*R_101[0]+P_001000211*R_102[0]+-1*P_101000011*R_200[0]+P_101000111*R_201[0]+-1*P_101000211*R_202[0];
				double PR_000001011100=P_000001011*R_100[0]+-1*P_000001111*R_101[0]+P_000001211*R_102[0]+-1*P_000101011*R_110[0]+P_000101111*R_111[0]+-1*P_000101211*R_112[0];
				double PR_000000012100=P_000000012*R_100[0]+-1*P_000000112*R_101[0]+P_000000212*R_102[0]+-1*P_000000312*R_103[0];
				double PR_012000000002=P_012000000*R_002[0]+-1*P_112000000*R_102[0]+P_212000000*R_202[0]+-1*P_312000000*R_302[0];
				double PR_011001000002=P_011001000*R_002[0]+-1*P_011101000*R_012[0]+-1*P_111001000*R_102[0]+P_111101000*R_112[0]+P_211001000*R_202[0]+-1*P_211101000*R_212[0];
				double PR_010002000002=P_010002000*R_002[0]+-1*P_010102000*R_012[0]+P_010202000*R_022[0]+-1*P_110002000*R_102[0]+P_110102000*R_112[0]+-1*P_110202000*R_122[0];
				double PR_011000001002=P_011000001*R_002[0]+-1*P_011000101*R_003[0]+-1*P_111000001*R_102[0]+P_111000101*R_103[0]+P_211000001*R_202[0]+-1*P_211000101*R_203[0];
				double PR_010001001002=P_010001001*R_002[0]+-1*P_010001101*R_003[0]+-1*P_010101001*R_012[0]+P_010101101*R_013[0]+-1*P_110001001*R_102[0]+P_110001101*R_103[0]+P_110101001*R_112[0]+-1*P_110101101*R_113[0];
				double PR_010000002002=P_010000002*R_002[0]+-1*P_010000102*R_003[0]+P_010000202*R_004[0]+-1*P_110000002*R_102[0]+P_110000102*R_103[0]+-1*P_110000202*R_104[0];
				double PR_002010000002=P_002010000*R_002[0]+-1*P_002110000*R_012[0]+-1*P_102010000*R_102[0]+P_102110000*R_112[0]+P_202010000*R_202[0]+-1*P_202110000*R_212[0];
				double PR_001011000002=P_001011000*R_002[0]+-1*P_001111000*R_012[0]+P_001211000*R_022[0]+-1*P_101011000*R_102[0]+P_101111000*R_112[0]+-1*P_101211000*R_122[0];
				double PR_000012000002=P_000012000*R_002[0]+-1*P_000112000*R_012[0]+P_000212000*R_022[0]+-1*P_000312000*R_032[0];
				double PR_001010001002=P_001010001*R_002[0]+-1*P_001010101*R_003[0]+-1*P_001110001*R_012[0]+P_001110101*R_013[0]+-1*P_101010001*R_102[0]+P_101010101*R_103[0]+P_101110001*R_112[0]+-1*P_101110101*R_113[0];
				double PR_000011001002=P_000011001*R_002[0]+-1*P_000011101*R_003[0]+-1*P_000111001*R_012[0]+P_000111101*R_013[0]+P_000211001*R_022[0]+-1*P_000211101*R_023[0];
				double PR_000010002002=P_000010002*R_002[0]+-1*P_000010102*R_003[0]+P_000010202*R_004[0]+-1*P_000110002*R_012[0]+P_000110102*R_013[0]+-1*P_000110202*R_014[0];
				double PR_002000010002=P_002000010*R_002[0]+-1*P_002000110*R_003[0]+-1*P_102000010*R_102[0]+P_102000110*R_103[0]+P_202000010*R_202[0]+-1*P_202000110*R_203[0];
				double PR_001001010002=P_001001010*R_002[0]+-1*P_001001110*R_003[0]+-1*P_001101010*R_012[0]+P_001101110*R_013[0]+-1*P_101001010*R_102[0]+P_101001110*R_103[0]+P_101101010*R_112[0]+-1*P_101101110*R_113[0];
				double PR_000002010002=P_000002010*R_002[0]+-1*P_000002110*R_003[0]+-1*P_000102010*R_012[0]+P_000102110*R_013[0]+P_000202010*R_022[0]+-1*P_000202110*R_023[0];
				double PR_001000011002=P_001000011*R_002[0]+-1*P_001000111*R_003[0]+P_001000211*R_004[0]+-1*P_101000011*R_102[0]+P_101000111*R_103[0]+-1*P_101000211*R_104[0];
				double PR_000001011002=P_000001011*R_002[0]+-1*P_000001111*R_003[0]+P_000001211*R_004[0]+-1*P_000101011*R_012[0]+P_000101111*R_013[0]+-1*P_000101211*R_014[0];
				double PR_000000012002=P_000000012*R_002[0]+-1*P_000000112*R_003[0]+P_000000212*R_004[0]+-1*P_000000312*R_005[0];
				double PR_012000000011=P_012000000*R_011[0]+-1*P_112000000*R_111[0]+P_212000000*R_211[0]+-1*P_312000000*R_311[0];
				double PR_011001000011=P_011001000*R_011[0]+-1*P_011101000*R_021[0]+-1*P_111001000*R_111[0]+P_111101000*R_121[0]+P_211001000*R_211[0]+-1*P_211101000*R_221[0];
				double PR_010002000011=P_010002000*R_011[0]+-1*P_010102000*R_021[0]+P_010202000*R_031[0]+-1*P_110002000*R_111[0]+P_110102000*R_121[0]+-1*P_110202000*R_131[0];
				double PR_011000001011=P_011000001*R_011[0]+-1*P_011000101*R_012[0]+-1*P_111000001*R_111[0]+P_111000101*R_112[0]+P_211000001*R_211[0]+-1*P_211000101*R_212[0];
				double PR_010001001011=P_010001001*R_011[0]+-1*P_010001101*R_012[0]+-1*P_010101001*R_021[0]+P_010101101*R_022[0]+-1*P_110001001*R_111[0]+P_110001101*R_112[0]+P_110101001*R_121[0]+-1*P_110101101*R_122[0];
				double PR_010000002011=P_010000002*R_011[0]+-1*P_010000102*R_012[0]+P_010000202*R_013[0]+-1*P_110000002*R_111[0]+P_110000102*R_112[0]+-1*P_110000202*R_113[0];
				double PR_002010000011=P_002010000*R_011[0]+-1*P_002110000*R_021[0]+-1*P_102010000*R_111[0]+P_102110000*R_121[0]+P_202010000*R_211[0]+-1*P_202110000*R_221[0];
				double PR_001011000011=P_001011000*R_011[0]+-1*P_001111000*R_021[0]+P_001211000*R_031[0]+-1*P_101011000*R_111[0]+P_101111000*R_121[0]+-1*P_101211000*R_131[0];
				double PR_000012000011=P_000012000*R_011[0]+-1*P_000112000*R_021[0]+P_000212000*R_031[0]+-1*P_000312000*R_041[0];
				double PR_001010001011=P_001010001*R_011[0]+-1*P_001010101*R_012[0]+-1*P_001110001*R_021[0]+P_001110101*R_022[0]+-1*P_101010001*R_111[0]+P_101010101*R_112[0]+P_101110001*R_121[0]+-1*P_101110101*R_122[0];
				double PR_000011001011=P_000011001*R_011[0]+-1*P_000011101*R_012[0]+-1*P_000111001*R_021[0]+P_000111101*R_022[0]+P_000211001*R_031[0]+-1*P_000211101*R_032[0];
				double PR_000010002011=P_000010002*R_011[0]+-1*P_000010102*R_012[0]+P_000010202*R_013[0]+-1*P_000110002*R_021[0]+P_000110102*R_022[0]+-1*P_000110202*R_023[0];
				double PR_002000010011=P_002000010*R_011[0]+-1*P_002000110*R_012[0]+-1*P_102000010*R_111[0]+P_102000110*R_112[0]+P_202000010*R_211[0]+-1*P_202000110*R_212[0];
				double PR_001001010011=P_001001010*R_011[0]+-1*P_001001110*R_012[0]+-1*P_001101010*R_021[0]+P_001101110*R_022[0]+-1*P_101001010*R_111[0]+P_101001110*R_112[0]+P_101101010*R_121[0]+-1*P_101101110*R_122[0];
				double PR_000002010011=P_000002010*R_011[0]+-1*P_000002110*R_012[0]+-1*P_000102010*R_021[0]+P_000102110*R_022[0]+P_000202010*R_031[0]+-1*P_000202110*R_032[0];
				double PR_001000011011=P_001000011*R_011[0]+-1*P_001000111*R_012[0]+P_001000211*R_013[0]+-1*P_101000011*R_111[0]+P_101000111*R_112[0]+-1*P_101000211*R_113[0];
				double PR_000001011011=P_000001011*R_011[0]+-1*P_000001111*R_012[0]+P_000001211*R_013[0]+-1*P_000101011*R_021[0]+P_000101111*R_022[0]+-1*P_000101211*R_023[0];
				double PR_000000012011=P_000000012*R_011[0]+-1*P_000000112*R_012[0]+P_000000212*R_013[0]+-1*P_000000312*R_014[0];
				double PR_012000000020=P_012000000*R_020[0]+-1*P_112000000*R_120[0]+P_212000000*R_220[0]+-1*P_312000000*R_320[0];
				double PR_011001000020=P_011001000*R_020[0]+-1*P_011101000*R_030[0]+-1*P_111001000*R_120[0]+P_111101000*R_130[0]+P_211001000*R_220[0]+-1*P_211101000*R_230[0];
				double PR_010002000020=P_010002000*R_020[0]+-1*P_010102000*R_030[0]+P_010202000*R_040[0]+-1*P_110002000*R_120[0]+P_110102000*R_130[0]+-1*P_110202000*R_140[0];
				double PR_011000001020=P_011000001*R_020[0]+-1*P_011000101*R_021[0]+-1*P_111000001*R_120[0]+P_111000101*R_121[0]+P_211000001*R_220[0]+-1*P_211000101*R_221[0];
				double PR_010001001020=P_010001001*R_020[0]+-1*P_010001101*R_021[0]+-1*P_010101001*R_030[0]+P_010101101*R_031[0]+-1*P_110001001*R_120[0]+P_110001101*R_121[0]+P_110101001*R_130[0]+-1*P_110101101*R_131[0];
				double PR_010000002020=P_010000002*R_020[0]+-1*P_010000102*R_021[0]+P_010000202*R_022[0]+-1*P_110000002*R_120[0]+P_110000102*R_121[0]+-1*P_110000202*R_122[0];
				double PR_002010000020=P_002010000*R_020[0]+-1*P_002110000*R_030[0]+-1*P_102010000*R_120[0]+P_102110000*R_130[0]+P_202010000*R_220[0]+-1*P_202110000*R_230[0];
				double PR_001011000020=P_001011000*R_020[0]+-1*P_001111000*R_030[0]+P_001211000*R_040[0]+-1*P_101011000*R_120[0]+P_101111000*R_130[0]+-1*P_101211000*R_140[0];
				double PR_000012000020=P_000012000*R_020[0]+-1*P_000112000*R_030[0]+P_000212000*R_040[0]+-1*P_000312000*R_050[0];
				double PR_001010001020=P_001010001*R_020[0]+-1*P_001010101*R_021[0]+-1*P_001110001*R_030[0]+P_001110101*R_031[0]+-1*P_101010001*R_120[0]+P_101010101*R_121[0]+P_101110001*R_130[0]+-1*P_101110101*R_131[0];
				double PR_000011001020=P_000011001*R_020[0]+-1*P_000011101*R_021[0]+-1*P_000111001*R_030[0]+P_000111101*R_031[0]+P_000211001*R_040[0]+-1*P_000211101*R_041[0];
				double PR_000010002020=P_000010002*R_020[0]+-1*P_000010102*R_021[0]+P_000010202*R_022[0]+-1*P_000110002*R_030[0]+P_000110102*R_031[0]+-1*P_000110202*R_032[0];
				double PR_002000010020=P_002000010*R_020[0]+-1*P_002000110*R_021[0]+-1*P_102000010*R_120[0]+P_102000110*R_121[0]+P_202000010*R_220[0]+-1*P_202000110*R_221[0];
				double PR_001001010020=P_001001010*R_020[0]+-1*P_001001110*R_021[0]+-1*P_001101010*R_030[0]+P_001101110*R_031[0]+-1*P_101001010*R_120[0]+P_101001110*R_121[0]+P_101101010*R_130[0]+-1*P_101101110*R_131[0];
				double PR_000002010020=P_000002010*R_020[0]+-1*P_000002110*R_021[0]+-1*P_000102010*R_030[0]+P_000102110*R_031[0]+P_000202010*R_040[0]+-1*P_000202110*R_041[0];
				double PR_001000011020=P_001000011*R_020[0]+-1*P_001000111*R_021[0]+P_001000211*R_022[0]+-1*P_101000011*R_120[0]+P_101000111*R_121[0]+-1*P_101000211*R_122[0];
				double PR_000001011020=P_000001011*R_020[0]+-1*P_000001111*R_021[0]+P_000001211*R_022[0]+-1*P_000101011*R_030[0]+P_000101111*R_031[0]+-1*P_000101211*R_032[0];
				double PR_000000012020=P_000000012*R_020[0]+-1*P_000000112*R_021[0]+P_000000212*R_022[0]+-1*P_000000312*R_023[0];
				double PR_012000000101=P_012000000*R_101[0]+-1*P_112000000*R_201[0]+P_212000000*R_301[0]+-1*P_312000000*R_401[0];
				double PR_011001000101=P_011001000*R_101[0]+-1*P_011101000*R_111[0]+-1*P_111001000*R_201[0]+P_111101000*R_211[0]+P_211001000*R_301[0]+-1*P_211101000*R_311[0];
				double PR_010002000101=P_010002000*R_101[0]+-1*P_010102000*R_111[0]+P_010202000*R_121[0]+-1*P_110002000*R_201[0]+P_110102000*R_211[0]+-1*P_110202000*R_221[0];
				double PR_011000001101=P_011000001*R_101[0]+-1*P_011000101*R_102[0]+-1*P_111000001*R_201[0]+P_111000101*R_202[0]+P_211000001*R_301[0]+-1*P_211000101*R_302[0];
				double PR_010001001101=P_010001001*R_101[0]+-1*P_010001101*R_102[0]+-1*P_010101001*R_111[0]+P_010101101*R_112[0]+-1*P_110001001*R_201[0]+P_110001101*R_202[0]+P_110101001*R_211[0]+-1*P_110101101*R_212[0];
				double PR_010000002101=P_010000002*R_101[0]+-1*P_010000102*R_102[0]+P_010000202*R_103[0]+-1*P_110000002*R_201[0]+P_110000102*R_202[0]+-1*P_110000202*R_203[0];
				double PR_002010000101=P_002010000*R_101[0]+-1*P_002110000*R_111[0]+-1*P_102010000*R_201[0]+P_102110000*R_211[0]+P_202010000*R_301[0]+-1*P_202110000*R_311[0];
				double PR_001011000101=P_001011000*R_101[0]+-1*P_001111000*R_111[0]+P_001211000*R_121[0]+-1*P_101011000*R_201[0]+P_101111000*R_211[0]+-1*P_101211000*R_221[0];
				double PR_000012000101=P_000012000*R_101[0]+-1*P_000112000*R_111[0]+P_000212000*R_121[0]+-1*P_000312000*R_131[0];
				double PR_001010001101=P_001010001*R_101[0]+-1*P_001010101*R_102[0]+-1*P_001110001*R_111[0]+P_001110101*R_112[0]+-1*P_101010001*R_201[0]+P_101010101*R_202[0]+P_101110001*R_211[0]+-1*P_101110101*R_212[0];
				double PR_000011001101=P_000011001*R_101[0]+-1*P_000011101*R_102[0]+-1*P_000111001*R_111[0]+P_000111101*R_112[0]+P_000211001*R_121[0]+-1*P_000211101*R_122[0];
				double PR_000010002101=P_000010002*R_101[0]+-1*P_000010102*R_102[0]+P_000010202*R_103[0]+-1*P_000110002*R_111[0]+P_000110102*R_112[0]+-1*P_000110202*R_113[0];
				double PR_002000010101=P_002000010*R_101[0]+-1*P_002000110*R_102[0]+-1*P_102000010*R_201[0]+P_102000110*R_202[0]+P_202000010*R_301[0]+-1*P_202000110*R_302[0];
				double PR_001001010101=P_001001010*R_101[0]+-1*P_001001110*R_102[0]+-1*P_001101010*R_111[0]+P_001101110*R_112[0]+-1*P_101001010*R_201[0]+P_101001110*R_202[0]+P_101101010*R_211[0]+-1*P_101101110*R_212[0];
				double PR_000002010101=P_000002010*R_101[0]+-1*P_000002110*R_102[0]+-1*P_000102010*R_111[0]+P_000102110*R_112[0]+P_000202010*R_121[0]+-1*P_000202110*R_122[0];
				double PR_001000011101=P_001000011*R_101[0]+-1*P_001000111*R_102[0]+P_001000211*R_103[0]+-1*P_101000011*R_201[0]+P_101000111*R_202[0]+-1*P_101000211*R_203[0];
				double PR_000001011101=P_000001011*R_101[0]+-1*P_000001111*R_102[0]+P_000001211*R_103[0]+-1*P_000101011*R_111[0]+P_000101111*R_112[0]+-1*P_000101211*R_113[0];
				double PR_000000012101=P_000000012*R_101[0]+-1*P_000000112*R_102[0]+P_000000212*R_103[0]+-1*P_000000312*R_104[0];
				double PR_012000000110=P_012000000*R_110[0]+-1*P_112000000*R_210[0]+P_212000000*R_310[0]+-1*P_312000000*R_410[0];
				double PR_011001000110=P_011001000*R_110[0]+-1*P_011101000*R_120[0]+-1*P_111001000*R_210[0]+P_111101000*R_220[0]+P_211001000*R_310[0]+-1*P_211101000*R_320[0];
				double PR_010002000110=P_010002000*R_110[0]+-1*P_010102000*R_120[0]+P_010202000*R_130[0]+-1*P_110002000*R_210[0]+P_110102000*R_220[0]+-1*P_110202000*R_230[0];
				double PR_011000001110=P_011000001*R_110[0]+-1*P_011000101*R_111[0]+-1*P_111000001*R_210[0]+P_111000101*R_211[0]+P_211000001*R_310[0]+-1*P_211000101*R_311[0];
				double PR_010001001110=P_010001001*R_110[0]+-1*P_010001101*R_111[0]+-1*P_010101001*R_120[0]+P_010101101*R_121[0]+-1*P_110001001*R_210[0]+P_110001101*R_211[0]+P_110101001*R_220[0]+-1*P_110101101*R_221[0];
				double PR_010000002110=P_010000002*R_110[0]+-1*P_010000102*R_111[0]+P_010000202*R_112[0]+-1*P_110000002*R_210[0]+P_110000102*R_211[0]+-1*P_110000202*R_212[0];
				double PR_002010000110=P_002010000*R_110[0]+-1*P_002110000*R_120[0]+-1*P_102010000*R_210[0]+P_102110000*R_220[0]+P_202010000*R_310[0]+-1*P_202110000*R_320[0];
				double PR_001011000110=P_001011000*R_110[0]+-1*P_001111000*R_120[0]+P_001211000*R_130[0]+-1*P_101011000*R_210[0]+P_101111000*R_220[0]+-1*P_101211000*R_230[0];
				double PR_000012000110=P_000012000*R_110[0]+-1*P_000112000*R_120[0]+P_000212000*R_130[0]+-1*P_000312000*R_140[0];
				double PR_001010001110=P_001010001*R_110[0]+-1*P_001010101*R_111[0]+-1*P_001110001*R_120[0]+P_001110101*R_121[0]+-1*P_101010001*R_210[0]+P_101010101*R_211[0]+P_101110001*R_220[0]+-1*P_101110101*R_221[0];
				double PR_000011001110=P_000011001*R_110[0]+-1*P_000011101*R_111[0]+-1*P_000111001*R_120[0]+P_000111101*R_121[0]+P_000211001*R_130[0]+-1*P_000211101*R_131[0];
				double PR_000010002110=P_000010002*R_110[0]+-1*P_000010102*R_111[0]+P_000010202*R_112[0]+-1*P_000110002*R_120[0]+P_000110102*R_121[0]+-1*P_000110202*R_122[0];
				double PR_002000010110=P_002000010*R_110[0]+-1*P_002000110*R_111[0]+-1*P_102000010*R_210[0]+P_102000110*R_211[0]+P_202000010*R_310[0]+-1*P_202000110*R_311[0];
				double PR_001001010110=P_001001010*R_110[0]+-1*P_001001110*R_111[0]+-1*P_001101010*R_120[0]+P_001101110*R_121[0]+-1*P_101001010*R_210[0]+P_101001110*R_211[0]+P_101101010*R_220[0]+-1*P_101101110*R_221[0];
				double PR_000002010110=P_000002010*R_110[0]+-1*P_000002110*R_111[0]+-1*P_000102010*R_120[0]+P_000102110*R_121[0]+P_000202010*R_130[0]+-1*P_000202110*R_131[0];
				double PR_001000011110=P_001000011*R_110[0]+-1*P_001000111*R_111[0]+P_001000211*R_112[0]+-1*P_101000011*R_210[0]+P_101000111*R_211[0]+-1*P_101000211*R_212[0];
				double PR_000001011110=P_000001011*R_110[0]+-1*P_000001111*R_111[0]+P_000001211*R_112[0]+-1*P_000101011*R_120[0]+P_000101111*R_121[0]+-1*P_000101211*R_122[0];
				double PR_000000012110=P_000000012*R_110[0]+-1*P_000000112*R_111[0]+P_000000212*R_112[0]+-1*P_000000312*R_113[0];
				double PR_012000000200=P_012000000*R_200[0]+-1*P_112000000*R_300[0]+P_212000000*R_400[0]+-1*P_312000000*R_500[0];
				double PR_011001000200=P_011001000*R_200[0]+-1*P_011101000*R_210[0]+-1*P_111001000*R_300[0]+P_111101000*R_310[0]+P_211001000*R_400[0]+-1*P_211101000*R_410[0];
				double PR_010002000200=P_010002000*R_200[0]+-1*P_010102000*R_210[0]+P_010202000*R_220[0]+-1*P_110002000*R_300[0]+P_110102000*R_310[0]+-1*P_110202000*R_320[0];
				double PR_011000001200=P_011000001*R_200[0]+-1*P_011000101*R_201[0]+-1*P_111000001*R_300[0]+P_111000101*R_301[0]+P_211000001*R_400[0]+-1*P_211000101*R_401[0];
				double PR_010001001200=P_010001001*R_200[0]+-1*P_010001101*R_201[0]+-1*P_010101001*R_210[0]+P_010101101*R_211[0]+-1*P_110001001*R_300[0]+P_110001101*R_301[0]+P_110101001*R_310[0]+-1*P_110101101*R_311[0];
				double PR_010000002200=P_010000002*R_200[0]+-1*P_010000102*R_201[0]+P_010000202*R_202[0]+-1*P_110000002*R_300[0]+P_110000102*R_301[0]+-1*P_110000202*R_302[0];
				double PR_002010000200=P_002010000*R_200[0]+-1*P_002110000*R_210[0]+-1*P_102010000*R_300[0]+P_102110000*R_310[0]+P_202010000*R_400[0]+-1*P_202110000*R_410[0];
				double PR_001011000200=P_001011000*R_200[0]+-1*P_001111000*R_210[0]+P_001211000*R_220[0]+-1*P_101011000*R_300[0]+P_101111000*R_310[0]+-1*P_101211000*R_320[0];
				double PR_000012000200=P_000012000*R_200[0]+-1*P_000112000*R_210[0]+P_000212000*R_220[0]+-1*P_000312000*R_230[0];
				double PR_001010001200=P_001010001*R_200[0]+-1*P_001010101*R_201[0]+-1*P_001110001*R_210[0]+P_001110101*R_211[0]+-1*P_101010001*R_300[0]+P_101010101*R_301[0]+P_101110001*R_310[0]+-1*P_101110101*R_311[0];
				double PR_000011001200=P_000011001*R_200[0]+-1*P_000011101*R_201[0]+-1*P_000111001*R_210[0]+P_000111101*R_211[0]+P_000211001*R_220[0]+-1*P_000211101*R_221[0];
				double PR_000010002200=P_000010002*R_200[0]+-1*P_000010102*R_201[0]+P_000010202*R_202[0]+-1*P_000110002*R_210[0]+P_000110102*R_211[0]+-1*P_000110202*R_212[0];
				double PR_002000010200=P_002000010*R_200[0]+-1*P_002000110*R_201[0]+-1*P_102000010*R_300[0]+P_102000110*R_301[0]+P_202000010*R_400[0]+-1*P_202000110*R_401[0];
				double PR_001001010200=P_001001010*R_200[0]+-1*P_001001110*R_201[0]+-1*P_001101010*R_210[0]+P_001101110*R_211[0]+-1*P_101001010*R_300[0]+P_101001110*R_301[0]+P_101101010*R_310[0]+-1*P_101101110*R_311[0];
				double PR_000002010200=P_000002010*R_200[0]+-1*P_000002110*R_201[0]+-1*P_000102010*R_210[0]+P_000102110*R_211[0]+P_000202010*R_220[0]+-1*P_000202110*R_221[0];
				double PR_001000011200=P_001000011*R_200[0]+-1*P_001000111*R_201[0]+P_001000211*R_202[0]+-1*P_101000011*R_300[0]+P_101000111*R_301[0]+-1*P_101000211*R_302[0];
				double PR_000001011200=P_000001011*R_200[0]+-1*P_000001111*R_201[0]+P_000001211*R_202[0]+-1*P_000101011*R_210[0]+P_000101111*R_211[0]+-1*P_000101211*R_212[0];
				double PR_000000012200=P_000000012*R_200[0]+-1*P_000000112*R_201[0]+P_000000212*R_202[0]+-1*P_000000312*R_203[0];
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(Q_020000000*PR_012000000000+Q_120000000*PR_012000000100+Q_220000000*PR_012000000200);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(Q_010010000*PR_012000000000+Q_010110000*PR_012000000010+Q_110010000*PR_012000000100+Q_110110000*PR_012000000110);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(Q_000020000*PR_012000000000+Q_000120000*PR_012000000010+Q_000220000*PR_012000000020);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(Q_010000010*PR_012000000000+Q_010000110*PR_012000000001+Q_110000010*PR_012000000100+Q_110000110*PR_012000000101);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(Q_000010010*PR_012000000000+Q_000010110*PR_012000000001+Q_000110010*PR_012000000010+Q_000110110*PR_012000000011);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(Q_000000020*PR_012000000000+Q_000000120*PR_012000000001+Q_000000220*PR_012000000002);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(Q_020000000*PR_011001000000+Q_120000000*PR_011001000100+Q_220000000*PR_011001000200);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(Q_010010000*PR_011001000000+Q_010110000*PR_011001000010+Q_110010000*PR_011001000100+Q_110110000*PR_011001000110);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(Q_000020000*PR_011001000000+Q_000120000*PR_011001000010+Q_000220000*PR_011001000020);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(Q_010000010*PR_011001000000+Q_010000110*PR_011001000001+Q_110000010*PR_011001000100+Q_110000110*PR_011001000101);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(Q_000010010*PR_011001000000+Q_000010110*PR_011001000001+Q_000110010*PR_011001000010+Q_000110110*PR_011001000011);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(Q_000000020*PR_011001000000+Q_000000120*PR_011001000001+Q_000000220*PR_011001000002);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(Q_020000000*PR_010002000000+Q_120000000*PR_010002000100+Q_220000000*PR_010002000200);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(Q_010010000*PR_010002000000+Q_010110000*PR_010002000010+Q_110010000*PR_010002000100+Q_110110000*PR_010002000110);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(Q_000020000*PR_010002000000+Q_000120000*PR_010002000010+Q_000220000*PR_010002000020);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(Q_010000010*PR_010002000000+Q_010000110*PR_010002000001+Q_110000010*PR_010002000100+Q_110000110*PR_010002000101);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(Q_000010010*PR_010002000000+Q_000010110*PR_010002000001+Q_000110010*PR_010002000010+Q_000110110*PR_010002000011);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(Q_000000020*PR_010002000000+Q_000000120*PR_010002000001+Q_000000220*PR_010002000002);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(Q_020000000*PR_011000001000+Q_120000000*PR_011000001100+Q_220000000*PR_011000001200);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(Q_010010000*PR_011000001000+Q_010110000*PR_011000001010+Q_110010000*PR_011000001100+Q_110110000*PR_011000001110);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(Q_000020000*PR_011000001000+Q_000120000*PR_011000001010+Q_000220000*PR_011000001020);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(Q_010000010*PR_011000001000+Q_010000110*PR_011000001001+Q_110000010*PR_011000001100+Q_110000110*PR_011000001101);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(Q_000010010*PR_011000001000+Q_000010110*PR_011000001001+Q_000110010*PR_011000001010+Q_000110110*PR_011000001011);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(Q_000000020*PR_011000001000+Q_000000120*PR_011000001001+Q_000000220*PR_011000001002);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(Q_020000000*PR_010001001000+Q_120000000*PR_010001001100+Q_220000000*PR_010001001200);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(Q_010010000*PR_010001001000+Q_010110000*PR_010001001010+Q_110010000*PR_010001001100+Q_110110000*PR_010001001110);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(Q_000020000*PR_010001001000+Q_000120000*PR_010001001010+Q_000220000*PR_010001001020);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(Q_010000010*PR_010001001000+Q_010000110*PR_010001001001+Q_110000010*PR_010001001100+Q_110000110*PR_010001001101);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(Q_000010010*PR_010001001000+Q_000010110*PR_010001001001+Q_000110010*PR_010001001010+Q_000110110*PR_010001001011);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(Q_000000020*PR_010001001000+Q_000000120*PR_010001001001+Q_000000220*PR_010001001002);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(Q_020000000*PR_010000002000+Q_120000000*PR_010000002100+Q_220000000*PR_010000002200);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(Q_010010000*PR_010000002000+Q_010110000*PR_010000002010+Q_110010000*PR_010000002100+Q_110110000*PR_010000002110);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(Q_000020000*PR_010000002000+Q_000120000*PR_010000002010+Q_000220000*PR_010000002020);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(Q_010000010*PR_010000002000+Q_010000110*PR_010000002001+Q_110000010*PR_010000002100+Q_110000110*PR_010000002101);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(Q_000010010*PR_010000002000+Q_000010110*PR_010000002001+Q_000110010*PR_010000002010+Q_000110110*PR_010000002011);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(Q_000000020*PR_010000002000+Q_000000120*PR_010000002001+Q_000000220*PR_010000002002);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(Q_020000000*PR_002010000000+Q_120000000*PR_002010000100+Q_220000000*PR_002010000200);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(Q_010010000*PR_002010000000+Q_010110000*PR_002010000010+Q_110010000*PR_002010000100+Q_110110000*PR_002010000110);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(Q_000020000*PR_002010000000+Q_000120000*PR_002010000010+Q_000220000*PR_002010000020);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(Q_010000010*PR_002010000000+Q_010000110*PR_002010000001+Q_110000010*PR_002010000100+Q_110000110*PR_002010000101);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(Q_000010010*PR_002010000000+Q_000010110*PR_002010000001+Q_000110010*PR_002010000010+Q_000110110*PR_002010000011);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(Q_000000020*PR_002010000000+Q_000000120*PR_002010000001+Q_000000220*PR_002010000002);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(Q_020000000*PR_001011000000+Q_120000000*PR_001011000100+Q_220000000*PR_001011000200);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(Q_010010000*PR_001011000000+Q_010110000*PR_001011000010+Q_110010000*PR_001011000100+Q_110110000*PR_001011000110);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(Q_000020000*PR_001011000000+Q_000120000*PR_001011000010+Q_000220000*PR_001011000020);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(Q_010000010*PR_001011000000+Q_010000110*PR_001011000001+Q_110000010*PR_001011000100+Q_110000110*PR_001011000101);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(Q_000010010*PR_001011000000+Q_000010110*PR_001011000001+Q_000110010*PR_001011000010+Q_000110110*PR_001011000011);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(Q_000000020*PR_001011000000+Q_000000120*PR_001011000001+Q_000000220*PR_001011000002);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(Q_020000000*PR_000012000000+Q_120000000*PR_000012000100+Q_220000000*PR_000012000200);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(Q_010010000*PR_000012000000+Q_010110000*PR_000012000010+Q_110010000*PR_000012000100+Q_110110000*PR_000012000110);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(Q_000020000*PR_000012000000+Q_000120000*PR_000012000010+Q_000220000*PR_000012000020);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(Q_010000010*PR_000012000000+Q_010000110*PR_000012000001+Q_110000010*PR_000012000100+Q_110000110*PR_000012000101);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(Q_000010010*PR_000012000000+Q_000010110*PR_000012000001+Q_000110010*PR_000012000010+Q_000110110*PR_000012000011);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(Q_000000020*PR_000012000000+Q_000000120*PR_000012000001+Q_000000220*PR_000012000002);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(Q_020000000*PR_001010001000+Q_120000000*PR_001010001100+Q_220000000*PR_001010001200);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(Q_010010000*PR_001010001000+Q_010110000*PR_001010001010+Q_110010000*PR_001010001100+Q_110110000*PR_001010001110);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(Q_000020000*PR_001010001000+Q_000120000*PR_001010001010+Q_000220000*PR_001010001020);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(Q_010000010*PR_001010001000+Q_010000110*PR_001010001001+Q_110000010*PR_001010001100+Q_110000110*PR_001010001101);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(Q_000010010*PR_001010001000+Q_000010110*PR_001010001001+Q_000110010*PR_001010001010+Q_000110110*PR_001010001011);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(Q_000000020*PR_001010001000+Q_000000120*PR_001010001001+Q_000000220*PR_001010001002);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(Q_020000000*PR_000011001000+Q_120000000*PR_000011001100+Q_220000000*PR_000011001200);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(Q_010010000*PR_000011001000+Q_010110000*PR_000011001010+Q_110010000*PR_000011001100+Q_110110000*PR_000011001110);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(Q_000020000*PR_000011001000+Q_000120000*PR_000011001010+Q_000220000*PR_000011001020);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(Q_010000010*PR_000011001000+Q_010000110*PR_000011001001+Q_110000010*PR_000011001100+Q_110000110*PR_000011001101);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(Q_000010010*PR_000011001000+Q_000010110*PR_000011001001+Q_000110010*PR_000011001010+Q_000110110*PR_000011001011);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(Q_000000020*PR_000011001000+Q_000000120*PR_000011001001+Q_000000220*PR_000011001002);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(Q_020000000*PR_000010002000+Q_120000000*PR_000010002100+Q_220000000*PR_000010002200);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(Q_010010000*PR_000010002000+Q_010110000*PR_000010002010+Q_110010000*PR_000010002100+Q_110110000*PR_000010002110);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(Q_000020000*PR_000010002000+Q_000120000*PR_000010002010+Q_000220000*PR_000010002020);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(Q_010000010*PR_000010002000+Q_010000110*PR_000010002001+Q_110000010*PR_000010002100+Q_110000110*PR_000010002101);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(Q_000010010*PR_000010002000+Q_000010110*PR_000010002001+Q_000110010*PR_000010002010+Q_000110110*PR_000010002011);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(Q_000000020*PR_000010002000+Q_000000120*PR_000010002001+Q_000000220*PR_000010002002);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(Q_020000000*PR_002000010000+Q_120000000*PR_002000010100+Q_220000000*PR_002000010200);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(Q_010010000*PR_002000010000+Q_010110000*PR_002000010010+Q_110010000*PR_002000010100+Q_110110000*PR_002000010110);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(Q_000020000*PR_002000010000+Q_000120000*PR_002000010010+Q_000220000*PR_002000010020);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(Q_010000010*PR_002000010000+Q_010000110*PR_002000010001+Q_110000010*PR_002000010100+Q_110000110*PR_002000010101);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(Q_000010010*PR_002000010000+Q_000010110*PR_002000010001+Q_000110010*PR_002000010010+Q_000110110*PR_002000010011);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(Q_000000020*PR_002000010000+Q_000000120*PR_002000010001+Q_000000220*PR_002000010002);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(Q_020000000*PR_001001010000+Q_120000000*PR_001001010100+Q_220000000*PR_001001010200);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(Q_010010000*PR_001001010000+Q_010110000*PR_001001010010+Q_110010000*PR_001001010100+Q_110110000*PR_001001010110);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(Q_000020000*PR_001001010000+Q_000120000*PR_001001010010+Q_000220000*PR_001001010020);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(Q_010000010*PR_001001010000+Q_010000110*PR_001001010001+Q_110000010*PR_001001010100+Q_110000110*PR_001001010101);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(Q_000010010*PR_001001010000+Q_000010110*PR_001001010001+Q_000110010*PR_001001010010+Q_000110110*PR_001001010011);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(Q_000000020*PR_001001010000+Q_000000120*PR_001001010001+Q_000000220*PR_001001010002);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(Q_020000000*PR_000002010000+Q_120000000*PR_000002010100+Q_220000000*PR_000002010200);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(Q_010010000*PR_000002010000+Q_010110000*PR_000002010010+Q_110010000*PR_000002010100+Q_110110000*PR_000002010110);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(Q_000020000*PR_000002010000+Q_000120000*PR_000002010010+Q_000220000*PR_000002010020);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(Q_010000010*PR_000002010000+Q_010000110*PR_000002010001+Q_110000010*PR_000002010100+Q_110000110*PR_000002010101);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(Q_000010010*PR_000002010000+Q_000010110*PR_000002010001+Q_000110010*PR_000002010010+Q_000110110*PR_000002010011);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(Q_000000020*PR_000002010000+Q_000000120*PR_000002010001+Q_000000220*PR_000002010002);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(Q_020000000*PR_001000011000+Q_120000000*PR_001000011100+Q_220000000*PR_001000011200);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(Q_010010000*PR_001000011000+Q_010110000*PR_001000011010+Q_110010000*PR_001000011100+Q_110110000*PR_001000011110);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(Q_000020000*PR_001000011000+Q_000120000*PR_001000011010+Q_000220000*PR_001000011020);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(Q_010000010*PR_001000011000+Q_010000110*PR_001000011001+Q_110000010*PR_001000011100+Q_110000110*PR_001000011101);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(Q_000010010*PR_001000011000+Q_000010110*PR_001000011001+Q_000110010*PR_001000011010+Q_000110110*PR_001000011011);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(Q_000000020*PR_001000011000+Q_000000120*PR_001000011001+Q_000000220*PR_001000011002);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(Q_020000000*PR_000001011000+Q_120000000*PR_000001011100+Q_220000000*PR_000001011200);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(Q_010010000*PR_000001011000+Q_010110000*PR_000001011010+Q_110010000*PR_000001011100+Q_110110000*PR_000001011110);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(Q_000020000*PR_000001011000+Q_000120000*PR_000001011010+Q_000220000*PR_000001011020);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(Q_010000010*PR_000001011000+Q_010000110*PR_000001011001+Q_110000010*PR_000001011100+Q_110000110*PR_000001011101);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(Q_000010010*PR_000001011000+Q_000010110*PR_000001011001+Q_000110010*PR_000001011010+Q_000110110*PR_000001011011);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(Q_000000020*PR_000001011000+Q_000000120*PR_000001011001+Q_000000220*PR_000001011002);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(Q_020000000*PR_000000012000+Q_120000000*PR_000000012100+Q_220000000*PR_000000012200);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(Q_010010000*PR_000000012000+Q_010110000*PR_000000012010+Q_110010000*PR_000000012100+Q_110110000*PR_000000012110);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(Q_000020000*PR_000000012000+Q_000120000*PR_000000012010+Q_000220000*PR_000000012020);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(Q_010000010*PR_000000012000+Q_010000110*PR_000000012001+Q_110000010*PR_000000012100+Q_110000110*PR_000000012101);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(Q_000010010*PR_000000012000+Q_000010110*PR_000000012001+Q_000110010*PR_000000012010+Q_000110110*PR_000000012011);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(Q_000000020*PR_000000012000+Q_000000120*PR_000000012001+Q_000000220*PR_000000012002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<18;ians++){
                    ans_temp[tId_x*18+ians]+=ans_temp[(tId_x+num_thread)*18+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<18;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_pdds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
            unsigned int id_bra=id_bra_in[ii];
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Pd_001[3];
				Pd_001[0]=PB[ii*3+0];
				Pd_001[1]=PB[ii*3+1];
				Pd_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
            float K2_p=K2_p_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_end-primit_ket_start;j+=tdis){
            unsigned int jj=primit_ket_start+j;
            unsigned int id_ket=tex1Dfetch(tex_id_ket,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Eta,jj);
            double Eta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pq,jj);
            double pq=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+0);
            double QX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+1);
            double QY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+2);
            double QZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Qd_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            Qd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            Qd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            Qd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[6];
                Ft_fs_5(5,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[5];
	double R_200[4];
	double R_300[3];
	double R_400[2];
	double R_500[1];
	double R_010[5];
	double R_110[4];
	double R_210[3];
	double R_310[2];
	double R_410[1];
	double R_020[4];
	double R_120[3];
	double R_220[2];
	double R_320[1];
	double R_030[3];
	double R_130[2];
	double R_230[1];
	double R_040[2];
	double R_140[1];
	double R_050[1];
	double R_001[5];
	double R_101[4];
	double R_201[3];
	double R_301[2];
	double R_401[1];
	double R_011[4];
	double R_111[3];
	double R_211[2];
	double R_311[1];
	double R_021[3];
	double R_121[2];
	double R_221[1];
	double R_031[2];
	double R_131[1];
	double R_041[1];
	double R_002[4];
	double R_102[3];
	double R_202[2];
	double R_302[1];
	double R_012[3];
	double R_112[2];
	double R_212[1];
	double R_022[2];
	double R_122[1];
	double R_032[1];
	double R_003[3];
	double R_103[2];
	double R_203[1];
	double R_013[2];
	double R_113[1];
	double R_023[1];
	double R_004[2];
	double R_104[1];
	double R_014[1];
	double R_005[1];
	for(int i=0;i<5;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<3;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<3;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<1;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<1;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<1;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<1;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<1;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
				double QR_020000000000=Q_020000000*R_000[0]+-1*Q_120000000*R_100[0]+Q_220000000*R_200[0];
				double QR_010010000000=Q_010010000*R_000[0]+-1*Q_010110000*R_010[0]+-1*Q_110010000*R_100[0]+Q_110110000*R_110[0];
				double QR_000020000000=Q_000020000*R_000[0]+-1*Q_000120000*R_010[0]+Q_000220000*R_020[0];
				double QR_010000010000=Q_010000010*R_000[0]+-1*Q_010000110*R_001[0]+-1*Q_110000010*R_100[0]+Q_110000110*R_101[0];
				double QR_000010010000=Q_000010010*R_000[0]+-1*Q_000010110*R_001[0]+-1*Q_000110010*R_010[0]+Q_000110110*R_011[0];
				double QR_000000020000=Q_000000020*R_000[0]+-1*Q_000000120*R_001[0]+Q_000000220*R_002[0];
				double QR_020000000001=Q_020000000*R_001[0]+-1*Q_120000000*R_101[0]+Q_220000000*R_201[0];
				double QR_010010000001=Q_010010000*R_001[0]+-1*Q_010110000*R_011[0]+-1*Q_110010000*R_101[0]+Q_110110000*R_111[0];
				double QR_000020000001=Q_000020000*R_001[0]+-1*Q_000120000*R_011[0]+Q_000220000*R_021[0];
				double QR_010000010001=Q_010000010*R_001[0]+-1*Q_010000110*R_002[0]+-1*Q_110000010*R_101[0]+Q_110000110*R_102[0];
				double QR_000010010001=Q_000010010*R_001[0]+-1*Q_000010110*R_002[0]+-1*Q_000110010*R_011[0]+Q_000110110*R_012[0];
				double QR_000000020001=Q_000000020*R_001[0]+-1*Q_000000120*R_002[0]+Q_000000220*R_003[0];
				double QR_020000000010=Q_020000000*R_010[0]+-1*Q_120000000*R_110[0]+Q_220000000*R_210[0];
				double QR_010010000010=Q_010010000*R_010[0]+-1*Q_010110000*R_020[0]+-1*Q_110010000*R_110[0]+Q_110110000*R_120[0];
				double QR_000020000010=Q_000020000*R_010[0]+-1*Q_000120000*R_020[0]+Q_000220000*R_030[0];
				double QR_010000010010=Q_010000010*R_010[0]+-1*Q_010000110*R_011[0]+-1*Q_110000010*R_110[0]+Q_110000110*R_111[0];
				double QR_000010010010=Q_000010010*R_010[0]+-1*Q_000010110*R_011[0]+-1*Q_000110010*R_020[0]+Q_000110110*R_021[0];
				double QR_000000020010=Q_000000020*R_010[0]+-1*Q_000000120*R_011[0]+Q_000000220*R_012[0];
				double QR_020000000100=Q_020000000*R_100[0]+-1*Q_120000000*R_200[0]+Q_220000000*R_300[0];
				double QR_010010000100=Q_010010000*R_100[0]+-1*Q_010110000*R_110[0]+-1*Q_110010000*R_200[0]+Q_110110000*R_210[0];
				double QR_000020000100=Q_000020000*R_100[0]+-1*Q_000120000*R_110[0]+Q_000220000*R_120[0];
				double QR_010000010100=Q_010000010*R_100[0]+-1*Q_010000110*R_101[0]+-1*Q_110000010*R_200[0]+Q_110000110*R_201[0];
				double QR_000010010100=Q_000010010*R_100[0]+-1*Q_000010110*R_101[0]+-1*Q_000110010*R_110[0]+Q_000110110*R_111[0];
				double QR_000000020100=Q_000000020*R_100[0]+-1*Q_000000120*R_101[0]+Q_000000220*R_102[0];
				double QR_020000000002=Q_020000000*R_002[0]+-1*Q_120000000*R_102[0]+Q_220000000*R_202[0];
				double QR_010010000002=Q_010010000*R_002[0]+-1*Q_010110000*R_012[0]+-1*Q_110010000*R_102[0]+Q_110110000*R_112[0];
				double QR_000020000002=Q_000020000*R_002[0]+-1*Q_000120000*R_012[0]+Q_000220000*R_022[0];
				double QR_010000010002=Q_010000010*R_002[0]+-1*Q_010000110*R_003[0]+-1*Q_110000010*R_102[0]+Q_110000110*R_103[0];
				double QR_000010010002=Q_000010010*R_002[0]+-1*Q_000010110*R_003[0]+-1*Q_000110010*R_012[0]+Q_000110110*R_013[0];
				double QR_000000020002=Q_000000020*R_002[0]+-1*Q_000000120*R_003[0]+Q_000000220*R_004[0];
				double QR_020000000011=Q_020000000*R_011[0]+-1*Q_120000000*R_111[0]+Q_220000000*R_211[0];
				double QR_010010000011=Q_010010000*R_011[0]+-1*Q_010110000*R_021[0]+-1*Q_110010000*R_111[0]+Q_110110000*R_121[0];
				double QR_000020000011=Q_000020000*R_011[0]+-1*Q_000120000*R_021[0]+Q_000220000*R_031[0];
				double QR_010000010011=Q_010000010*R_011[0]+-1*Q_010000110*R_012[0]+-1*Q_110000010*R_111[0]+Q_110000110*R_112[0];
				double QR_000010010011=Q_000010010*R_011[0]+-1*Q_000010110*R_012[0]+-1*Q_000110010*R_021[0]+Q_000110110*R_022[0];
				double QR_000000020011=Q_000000020*R_011[0]+-1*Q_000000120*R_012[0]+Q_000000220*R_013[0];
				double QR_020000000020=Q_020000000*R_020[0]+-1*Q_120000000*R_120[0]+Q_220000000*R_220[0];
				double QR_010010000020=Q_010010000*R_020[0]+-1*Q_010110000*R_030[0]+-1*Q_110010000*R_120[0]+Q_110110000*R_130[0];
				double QR_000020000020=Q_000020000*R_020[0]+-1*Q_000120000*R_030[0]+Q_000220000*R_040[0];
				double QR_010000010020=Q_010000010*R_020[0]+-1*Q_010000110*R_021[0]+-1*Q_110000010*R_120[0]+Q_110000110*R_121[0];
				double QR_000010010020=Q_000010010*R_020[0]+-1*Q_000010110*R_021[0]+-1*Q_000110010*R_030[0]+Q_000110110*R_031[0];
				double QR_000000020020=Q_000000020*R_020[0]+-1*Q_000000120*R_021[0]+Q_000000220*R_022[0];
				double QR_020000000101=Q_020000000*R_101[0]+-1*Q_120000000*R_201[0]+Q_220000000*R_301[0];
				double QR_010010000101=Q_010010000*R_101[0]+-1*Q_010110000*R_111[0]+-1*Q_110010000*R_201[0]+Q_110110000*R_211[0];
				double QR_000020000101=Q_000020000*R_101[0]+-1*Q_000120000*R_111[0]+Q_000220000*R_121[0];
				double QR_010000010101=Q_010000010*R_101[0]+-1*Q_010000110*R_102[0]+-1*Q_110000010*R_201[0]+Q_110000110*R_202[0];
				double QR_000010010101=Q_000010010*R_101[0]+-1*Q_000010110*R_102[0]+-1*Q_000110010*R_111[0]+Q_000110110*R_112[0];
				double QR_000000020101=Q_000000020*R_101[0]+-1*Q_000000120*R_102[0]+Q_000000220*R_103[0];
				double QR_020000000110=Q_020000000*R_110[0]+-1*Q_120000000*R_210[0]+Q_220000000*R_310[0];
				double QR_010010000110=Q_010010000*R_110[0]+-1*Q_010110000*R_120[0]+-1*Q_110010000*R_210[0]+Q_110110000*R_220[0];
				double QR_000020000110=Q_000020000*R_110[0]+-1*Q_000120000*R_120[0]+Q_000220000*R_130[0];
				double QR_010000010110=Q_010000010*R_110[0]+-1*Q_010000110*R_111[0]+-1*Q_110000010*R_210[0]+Q_110000110*R_211[0];
				double QR_000010010110=Q_000010010*R_110[0]+-1*Q_000010110*R_111[0]+-1*Q_000110010*R_120[0]+Q_000110110*R_121[0];
				double QR_000000020110=Q_000000020*R_110[0]+-1*Q_000000120*R_111[0]+Q_000000220*R_112[0];
				double QR_020000000200=Q_020000000*R_200[0]+-1*Q_120000000*R_300[0]+Q_220000000*R_400[0];
				double QR_010010000200=Q_010010000*R_200[0]+-1*Q_010110000*R_210[0]+-1*Q_110010000*R_300[0]+Q_110110000*R_310[0];
				double QR_000020000200=Q_000020000*R_200[0]+-1*Q_000120000*R_210[0]+Q_000220000*R_220[0];
				double QR_010000010200=Q_010000010*R_200[0]+-1*Q_010000110*R_201[0]+-1*Q_110000010*R_300[0]+Q_110000110*R_301[0];
				double QR_000010010200=Q_000010010*R_200[0]+-1*Q_000010110*R_201[0]+-1*Q_000110010*R_210[0]+Q_000110110*R_211[0];
				double QR_000000020200=Q_000000020*R_200[0]+-1*Q_000000120*R_201[0]+Q_000000220*R_202[0];
				double QR_020000000003=Q_020000000*R_003[0]+-1*Q_120000000*R_103[0]+Q_220000000*R_203[0];
				double QR_010010000003=Q_010010000*R_003[0]+-1*Q_010110000*R_013[0]+-1*Q_110010000*R_103[0]+Q_110110000*R_113[0];
				double QR_000020000003=Q_000020000*R_003[0]+-1*Q_000120000*R_013[0]+Q_000220000*R_023[0];
				double QR_010000010003=Q_010000010*R_003[0]+-1*Q_010000110*R_004[0]+-1*Q_110000010*R_103[0]+Q_110000110*R_104[0];
				double QR_000010010003=Q_000010010*R_003[0]+-1*Q_000010110*R_004[0]+-1*Q_000110010*R_013[0]+Q_000110110*R_014[0];
				double QR_000000020003=Q_000000020*R_003[0]+-1*Q_000000120*R_004[0]+Q_000000220*R_005[0];
				double QR_020000000012=Q_020000000*R_012[0]+-1*Q_120000000*R_112[0]+Q_220000000*R_212[0];
				double QR_010010000012=Q_010010000*R_012[0]+-1*Q_010110000*R_022[0]+-1*Q_110010000*R_112[0]+Q_110110000*R_122[0];
				double QR_000020000012=Q_000020000*R_012[0]+-1*Q_000120000*R_022[0]+Q_000220000*R_032[0];
				double QR_010000010012=Q_010000010*R_012[0]+-1*Q_010000110*R_013[0]+-1*Q_110000010*R_112[0]+Q_110000110*R_113[0];
				double QR_000010010012=Q_000010010*R_012[0]+-1*Q_000010110*R_013[0]+-1*Q_000110010*R_022[0]+Q_000110110*R_023[0];
				double QR_000000020012=Q_000000020*R_012[0]+-1*Q_000000120*R_013[0]+Q_000000220*R_014[0];
				double QR_020000000021=Q_020000000*R_021[0]+-1*Q_120000000*R_121[0]+Q_220000000*R_221[0];
				double QR_010010000021=Q_010010000*R_021[0]+-1*Q_010110000*R_031[0]+-1*Q_110010000*R_121[0]+Q_110110000*R_131[0];
				double QR_000020000021=Q_000020000*R_021[0]+-1*Q_000120000*R_031[0]+Q_000220000*R_041[0];
				double QR_010000010021=Q_010000010*R_021[0]+-1*Q_010000110*R_022[0]+-1*Q_110000010*R_121[0]+Q_110000110*R_122[0];
				double QR_000010010021=Q_000010010*R_021[0]+-1*Q_000010110*R_022[0]+-1*Q_000110010*R_031[0]+Q_000110110*R_032[0];
				double QR_000000020021=Q_000000020*R_021[0]+-1*Q_000000120*R_022[0]+Q_000000220*R_023[0];
				double QR_020000000030=Q_020000000*R_030[0]+-1*Q_120000000*R_130[0]+Q_220000000*R_230[0];
				double QR_010010000030=Q_010010000*R_030[0]+-1*Q_010110000*R_040[0]+-1*Q_110010000*R_130[0]+Q_110110000*R_140[0];
				double QR_000020000030=Q_000020000*R_030[0]+-1*Q_000120000*R_040[0]+Q_000220000*R_050[0];
				double QR_010000010030=Q_010000010*R_030[0]+-1*Q_010000110*R_031[0]+-1*Q_110000010*R_130[0]+Q_110000110*R_131[0];
				double QR_000010010030=Q_000010010*R_030[0]+-1*Q_000010110*R_031[0]+-1*Q_000110010*R_040[0]+Q_000110110*R_041[0];
				double QR_000000020030=Q_000000020*R_030[0]+-1*Q_000000120*R_031[0]+Q_000000220*R_032[0];
				double QR_020000000102=Q_020000000*R_102[0]+-1*Q_120000000*R_202[0]+Q_220000000*R_302[0];
				double QR_010010000102=Q_010010000*R_102[0]+-1*Q_010110000*R_112[0]+-1*Q_110010000*R_202[0]+Q_110110000*R_212[0];
				double QR_000020000102=Q_000020000*R_102[0]+-1*Q_000120000*R_112[0]+Q_000220000*R_122[0];
				double QR_010000010102=Q_010000010*R_102[0]+-1*Q_010000110*R_103[0]+-1*Q_110000010*R_202[0]+Q_110000110*R_203[0];
				double QR_000010010102=Q_000010010*R_102[0]+-1*Q_000010110*R_103[0]+-1*Q_000110010*R_112[0]+Q_000110110*R_113[0];
				double QR_000000020102=Q_000000020*R_102[0]+-1*Q_000000120*R_103[0]+Q_000000220*R_104[0];
				double QR_020000000111=Q_020000000*R_111[0]+-1*Q_120000000*R_211[0]+Q_220000000*R_311[0];
				double QR_010010000111=Q_010010000*R_111[0]+-1*Q_010110000*R_121[0]+-1*Q_110010000*R_211[0]+Q_110110000*R_221[0];
				double QR_000020000111=Q_000020000*R_111[0]+-1*Q_000120000*R_121[0]+Q_000220000*R_131[0];
				double QR_010000010111=Q_010000010*R_111[0]+-1*Q_010000110*R_112[0]+-1*Q_110000010*R_211[0]+Q_110000110*R_212[0];
				double QR_000010010111=Q_000010010*R_111[0]+-1*Q_000010110*R_112[0]+-1*Q_000110010*R_121[0]+Q_000110110*R_122[0];
				double QR_000000020111=Q_000000020*R_111[0]+-1*Q_000000120*R_112[0]+Q_000000220*R_113[0];
				double QR_020000000120=Q_020000000*R_120[0]+-1*Q_120000000*R_220[0]+Q_220000000*R_320[0];
				double QR_010010000120=Q_010010000*R_120[0]+-1*Q_010110000*R_130[0]+-1*Q_110010000*R_220[0]+Q_110110000*R_230[0];
				double QR_000020000120=Q_000020000*R_120[0]+-1*Q_000120000*R_130[0]+Q_000220000*R_140[0];
				double QR_010000010120=Q_010000010*R_120[0]+-1*Q_010000110*R_121[0]+-1*Q_110000010*R_220[0]+Q_110000110*R_221[0];
				double QR_000010010120=Q_000010010*R_120[0]+-1*Q_000010110*R_121[0]+-1*Q_000110010*R_130[0]+Q_000110110*R_131[0];
				double QR_000000020120=Q_000000020*R_120[0]+-1*Q_000000120*R_121[0]+Q_000000220*R_122[0];
				double QR_020000000201=Q_020000000*R_201[0]+-1*Q_120000000*R_301[0]+Q_220000000*R_401[0];
				double QR_010010000201=Q_010010000*R_201[0]+-1*Q_010110000*R_211[0]+-1*Q_110010000*R_301[0]+Q_110110000*R_311[0];
				double QR_000020000201=Q_000020000*R_201[0]+-1*Q_000120000*R_211[0]+Q_000220000*R_221[0];
				double QR_010000010201=Q_010000010*R_201[0]+-1*Q_010000110*R_202[0]+-1*Q_110000010*R_301[0]+Q_110000110*R_302[0];
				double QR_000010010201=Q_000010010*R_201[0]+-1*Q_000010110*R_202[0]+-1*Q_000110010*R_211[0]+Q_000110110*R_212[0];
				double QR_000000020201=Q_000000020*R_201[0]+-1*Q_000000120*R_202[0]+Q_000000220*R_203[0];
				double QR_020000000210=Q_020000000*R_210[0]+-1*Q_120000000*R_310[0]+Q_220000000*R_410[0];
				double QR_010010000210=Q_010010000*R_210[0]+-1*Q_010110000*R_220[0]+-1*Q_110010000*R_310[0]+Q_110110000*R_320[0];
				double QR_000020000210=Q_000020000*R_210[0]+-1*Q_000120000*R_220[0]+Q_000220000*R_230[0];
				double QR_010000010210=Q_010000010*R_210[0]+-1*Q_010000110*R_211[0]+-1*Q_110000010*R_310[0]+Q_110000110*R_311[0];
				double QR_000010010210=Q_000010010*R_210[0]+-1*Q_000010110*R_211[0]+-1*Q_000110010*R_220[0]+Q_000110110*R_221[0];
				double QR_000000020210=Q_000000020*R_210[0]+-1*Q_000000120*R_211[0]+Q_000000220*R_212[0];
				double QR_020000000300=Q_020000000*R_300[0]+-1*Q_120000000*R_400[0]+Q_220000000*R_500[0];
				double QR_010010000300=Q_010010000*R_300[0]+-1*Q_010110000*R_310[0]+-1*Q_110010000*R_400[0]+Q_110110000*R_410[0];
				double QR_000020000300=Q_000020000*R_300[0]+-1*Q_000120000*R_310[0]+Q_000220000*R_320[0];
				double QR_010000010300=Q_010000010*R_300[0]+-1*Q_010000110*R_301[0]+-1*Q_110000010*R_400[0]+Q_110000110*R_401[0];
				double QR_000010010300=Q_000010010*R_300[0]+-1*Q_000010110*R_301[0]+-1*Q_000110010*R_310[0]+Q_000110110*R_311[0];
				double QR_000000020300=Q_000000020*R_300[0]+-1*Q_000000120*R_301[0]+Q_000000220*R_302[0];
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_312[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=2*Pd_211[i]+Pd_001[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=Pd_001[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_312[i]=aPin1*Pd_211[i];
			}
	double P_012000000=Pd_012[0];
	double P_112000000=Pd_112[0];
	double P_212000000=Pd_212[0];
	double P_312000000=Pd_312[0];
	double P_011001000=Pd_011[0]*Pd_001[1];
	double P_011101000=Pd_011[0]*Pd_101[1];
	double P_111001000=Pd_111[0]*Pd_001[1];
	double P_111101000=Pd_111[0]*Pd_101[1];
	double P_211001000=Pd_211[0]*Pd_001[1];
	double P_211101000=Pd_211[0]*Pd_101[1];
	double P_010002000=Pd_010[0]*Pd_002[1];
	double P_010102000=Pd_010[0]*Pd_102[1];
	double P_010202000=Pd_010[0]*Pd_202[1];
	double P_110002000=Pd_110[0]*Pd_002[1];
	double P_110102000=Pd_110[0]*Pd_102[1];
	double P_110202000=Pd_110[0]*Pd_202[1];
	double P_011000001=Pd_011[0]*Pd_001[2];
	double P_011000101=Pd_011[0]*Pd_101[2];
	double P_111000001=Pd_111[0]*Pd_001[2];
	double P_111000101=Pd_111[0]*Pd_101[2];
	double P_211000001=Pd_211[0]*Pd_001[2];
	double P_211000101=Pd_211[0]*Pd_101[2];
	double P_010001001=Pd_010[0]*Pd_001[1]*Pd_001[2];
	double P_010001101=Pd_010[0]*Pd_001[1]*Pd_101[2];
	double P_010101001=Pd_010[0]*Pd_101[1]*Pd_001[2];
	double P_010101101=Pd_010[0]*Pd_101[1]*Pd_101[2];
	double P_110001001=Pd_110[0]*Pd_001[1]*Pd_001[2];
	double P_110001101=Pd_110[0]*Pd_001[1]*Pd_101[2];
	double P_110101001=Pd_110[0]*Pd_101[1]*Pd_001[2];
	double P_110101101=Pd_110[0]*Pd_101[1]*Pd_101[2];
	double P_010000002=Pd_010[0]*Pd_002[2];
	double P_010000102=Pd_010[0]*Pd_102[2];
	double P_010000202=Pd_010[0]*Pd_202[2];
	double P_110000002=Pd_110[0]*Pd_002[2];
	double P_110000102=Pd_110[0]*Pd_102[2];
	double P_110000202=Pd_110[0]*Pd_202[2];
	double P_002010000=Pd_002[0]*Pd_010[1];
	double P_002110000=Pd_002[0]*Pd_110[1];
	double P_102010000=Pd_102[0]*Pd_010[1];
	double P_102110000=Pd_102[0]*Pd_110[1];
	double P_202010000=Pd_202[0]*Pd_010[1];
	double P_202110000=Pd_202[0]*Pd_110[1];
	double P_001011000=Pd_001[0]*Pd_011[1];
	double P_001111000=Pd_001[0]*Pd_111[1];
	double P_001211000=Pd_001[0]*Pd_211[1];
	double P_101011000=Pd_101[0]*Pd_011[1];
	double P_101111000=Pd_101[0]*Pd_111[1];
	double P_101211000=Pd_101[0]*Pd_211[1];
	double P_000012000=Pd_012[1];
	double P_000112000=Pd_112[1];
	double P_000212000=Pd_212[1];
	double P_000312000=Pd_312[1];
	double P_001010001=Pd_001[0]*Pd_010[1]*Pd_001[2];
	double P_001010101=Pd_001[0]*Pd_010[1]*Pd_101[2];
	double P_001110001=Pd_001[0]*Pd_110[1]*Pd_001[2];
	double P_001110101=Pd_001[0]*Pd_110[1]*Pd_101[2];
	double P_101010001=Pd_101[0]*Pd_010[1]*Pd_001[2];
	double P_101010101=Pd_101[0]*Pd_010[1]*Pd_101[2];
	double P_101110001=Pd_101[0]*Pd_110[1]*Pd_001[2];
	double P_101110101=Pd_101[0]*Pd_110[1]*Pd_101[2];
	double P_000011001=Pd_011[1]*Pd_001[2];
	double P_000011101=Pd_011[1]*Pd_101[2];
	double P_000111001=Pd_111[1]*Pd_001[2];
	double P_000111101=Pd_111[1]*Pd_101[2];
	double P_000211001=Pd_211[1]*Pd_001[2];
	double P_000211101=Pd_211[1]*Pd_101[2];
	double P_000010002=Pd_010[1]*Pd_002[2];
	double P_000010102=Pd_010[1]*Pd_102[2];
	double P_000010202=Pd_010[1]*Pd_202[2];
	double P_000110002=Pd_110[1]*Pd_002[2];
	double P_000110102=Pd_110[1]*Pd_102[2];
	double P_000110202=Pd_110[1]*Pd_202[2];
	double P_002000010=Pd_002[0]*Pd_010[2];
	double P_002000110=Pd_002[0]*Pd_110[2];
	double P_102000010=Pd_102[0]*Pd_010[2];
	double P_102000110=Pd_102[0]*Pd_110[2];
	double P_202000010=Pd_202[0]*Pd_010[2];
	double P_202000110=Pd_202[0]*Pd_110[2];
	double P_001001010=Pd_001[0]*Pd_001[1]*Pd_010[2];
	double P_001001110=Pd_001[0]*Pd_001[1]*Pd_110[2];
	double P_001101010=Pd_001[0]*Pd_101[1]*Pd_010[2];
	double P_001101110=Pd_001[0]*Pd_101[1]*Pd_110[2];
	double P_101001010=Pd_101[0]*Pd_001[1]*Pd_010[2];
	double P_101001110=Pd_101[0]*Pd_001[1]*Pd_110[2];
	double P_101101010=Pd_101[0]*Pd_101[1]*Pd_010[2];
	double P_101101110=Pd_101[0]*Pd_101[1]*Pd_110[2];
	double P_000002010=Pd_002[1]*Pd_010[2];
	double P_000002110=Pd_002[1]*Pd_110[2];
	double P_000102010=Pd_102[1]*Pd_010[2];
	double P_000102110=Pd_102[1]*Pd_110[2];
	double P_000202010=Pd_202[1]*Pd_010[2];
	double P_000202110=Pd_202[1]*Pd_110[2];
	double P_001000011=Pd_001[0]*Pd_011[2];
	double P_001000111=Pd_001[0]*Pd_111[2];
	double P_001000211=Pd_001[0]*Pd_211[2];
	double P_101000011=Pd_101[0]*Pd_011[2];
	double P_101000111=Pd_101[0]*Pd_111[2];
	double P_101000211=Pd_101[0]*Pd_211[2];
	double P_000001011=Pd_001[1]*Pd_011[2];
	double P_000001111=Pd_001[1]*Pd_111[2];
	double P_000001211=Pd_001[1]*Pd_211[2];
	double P_000101011=Pd_101[1]*Pd_011[2];
	double P_000101111=Pd_101[1]*Pd_111[2];
	double P_000101211=Pd_101[1]*Pd_211[2];
	double P_000000012=Pd_012[2];
	double P_000000112=Pd_112[2];
	double P_000000212=Pd_212[2];
	double P_000000312=Pd_312[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(P_012000000*QR_020000000000+P_112000000*QR_020000000100+P_212000000*QR_020000000200+P_312000000*QR_020000000300);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(P_012000000*QR_010010000000+P_112000000*QR_010010000100+P_212000000*QR_010010000200+P_312000000*QR_010010000300);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(P_012000000*QR_000020000000+P_112000000*QR_000020000100+P_212000000*QR_000020000200+P_312000000*QR_000020000300);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(P_012000000*QR_010000010000+P_112000000*QR_010000010100+P_212000000*QR_010000010200+P_312000000*QR_010000010300);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(P_012000000*QR_000010010000+P_112000000*QR_000010010100+P_212000000*QR_000010010200+P_312000000*QR_000010010300);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(P_012000000*QR_000000020000+P_112000000*QR_000000020100+P_212000000*QR_000000020200+P_312000000*QR_000000020300);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(P_011001000*QR_020000000000+P_011101000*QR_020000000010+P_111001000*QR_020000000100+P_111101000*QR_020000000110+P_211001000*QR_020000000200+P_211101000*QR_020000000210);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(P_011001000*QR_010010000000+P_011101000*QR_010010000010+P_111001000*QR_010010000100+P_111101000*QR_010010000110+P_211001000*QR_010010000200+P_211101000*QR_010010000210);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(P_011001000*QR_000020000000+P_011101000*QR_000020000010+P_111001000*QR_000020000100+P_111101000*QR_000020000110+P_211001000*QR_000020000200+P_211101000*QR_000020000210);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(P_011001000*QR_010000010000+P_011101000*QR_010000010010+P_111001000*QR_010000010100+P_111101000*QR_010000010110+P_211001000*QR_010000010200+P_211101000*QR_010000010210);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(P_011001000*QR_000010010000+P_011101000*QR_000010010010+P_111001000*QR_000010010100+P_111101000*QR_000010010110+P_211001000*QR_000010010200+P_211101000*QR_000010010210);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(P_011001000*QR_000000020000+P_011101000*QR_000000020010+P_111001000*QR_000000020100+P_111101000*QR_000000020110+P_211001000*QR_000000020200+P_211101000*QR_000000020210);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(P_010002000*QR_020000000000+P_010102000*QR_020000000010+P_010202000*QR_020000000020+P_110002000*QR_020000000100+P_110102000*QR_020000000110+P_110202000*QR_020000000120);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(P_010002000*QR_010010000000+P_010102000*QR_010010000010+P_010202000*QR_010010000020+P_110002000*QR_010010000100+P_110102000*QR_010010000110+P_110202000*QR_010010000120);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(P_010002000*QR_000020000000+P_010102000*QR_000020000010+P_010202000*QR_000020000020+P_110002000*QR_000020000100+P_110102000*QR_000020000110+P_110202000*QR_000020000120);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(P_010002000*QR_010000010000+P_010102000*QR_010000010010+P_010202000*QR_010000010020+P_110002000*QR_010000010100+P_110102000*QR_010000010110+P_110202000*QR_010000010120);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(P_010002000*QR_000010010000+P_010102000*QR_000010010010+P_010202000*QR_000010010020+P_110002000*QR_000010010100+P_110102000*QR_000010010110+P_110202000*QR_000010010120);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(P_010002000*QR_000000020000+P_010102000*QR_000000020010+P_010202000*QR_000000020020+P_110002000*QR_000000020100+P_110102000*QR_000000020110+P_110202000*QR_000000020120);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(P_011000001*QR_020000000000+P_011000101*QR_020000000001+P_111000001*QR_020000000100+P_111000101*QR_020000000101+P_211000001*QR_020000000200+P_211000101*QR_020000000201);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(P_011000001*QR_010010000000+P_011000101*QR_010010000001+P_111000001*QR_010010000100+P_111000101*QR_010010000101+P_211000001*QR_010010000200+P_211000101*QR_010010000201);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(P_011000001*QR_000020000000+P_011000101*QR_000020000001+P_111000001*QR_000020000100+P_111000101*QR_000020000101+P_211000001*QR_000020000200+P_211000101*QR_000020000201);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(P_011000001*QR_010000010000+P_011000101*QR_010000010001+P_111000001*QR_010000010100+P_111000101*QR_010000010101+P_211000001*QR_010000010200+P_211000101*QR_010000010201);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(P_011000001*QR_000010010000+P_011000101*QR_000010010001+P_111000001*QR_000010010100+P_111000101*QR_000010010101+P_211000001*QR_000010010200+P_211000101*QR_000010010201);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(P_011000001*QR_000000020000+P_011000101*QR_000000020001+P_111000001*QR_000000020100+P_111000101*QR_000000020101+P_211000001*QR_000000020200+P_211000101*QR_000000020201);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(P_010001001*QR_020000000000+P_010001101*QR_020000000001+P_010101001*QR_020000000010+P_010101101*QR_020000000011+P_110001001*QR_020000000100+P_110001101*QR_020000000101+P_110101001*QR_020000000110+P_110101101*QR_020000000111);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(P_010001001*QR_010010000000+P_010001101*QR_010010000001+P_010101001*QR_010010000010+P_010101101*QR_010010000011+P_110001001*QR_010010000100+P_110001101*QR_010010000101+P_110101001*QR_010010000110+P_110101101*QR_010010000111);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(P_010001001*QR_000020000000+P_010001101*QR_000020000001+P_010101001*QR_000020000010+P_010101101*QR_000020000011+P_110001001*QR_000020000100+P_110001101*QR_000020000101+P_110101001*QR_000020000110+P_110101101*QR_000020000111);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(P_010001001*QR_010000010000+P_010001101*QR_010000010001+P_010101001*QR_010000010010+P_010101101*QR_010000010011+P_110001001*QR_010000010100+P_110001101*QR_010000010101+P_110101001*QR_010000010110+P_110101101*QR_010000010111);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(P_010001001*QR_000010010000+P_010001101*QR_000010010001+P_010101001*QR_000010010010+P_010101101*QR_000010010011+P_110001001*QR_000010010100+P_110001101*QR_000010010101+P_110101001*QR_000010010110+P_110101101*QR_000010010111);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(P_010001001*QR_000000020000+P_010001101*QR_000000020001+P_010101001*QR_000000020010+P_010101101*QR_000000020011+P_110001001*QR_000000020100+P_110001101*QR_000000020101+P_110101001*QR_000000020110+P_110101101*QR_000000020111);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(P_010000002*QR_020000000000+P_010000102*QR_020000000001+P_010000202*QR_020000000002+P_110000002*QR_020000000100+P_110000102*QR_020000000101+P_110000202*QR_020000000102);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(P_010000002*QR_010010000000+P_010000102*QR_010010000001+P_010000202*QR_010010000002+P_110000002*QR_010010000100+P_110000102*QR_010010000101+P_110000202*QR_010010000102);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(P_010000002*QR_000020000000+P_010000102*QR_000020000001+P_010000202*QR_000020000002+P_110000002*QR_000020000100+P_110000102*QR_000020000101+P_110000202*QR_000020000102);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(P_010000002*QR_010000010000+P_010000102*QR_010000010001+P_010000202*QR_010000010002+P_110000002*QR_010000010100+P_110000102*QR_010000010101+P_110000202*QR_010000010102);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(P_010000002*QR_000010010000+P_010000102*QR_000010010001+P_010000202*QR_000010010002+P_110000002*QR_000010010100+P_110000102*QR_000010010101+P_110000202*QR_000010010102);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(P_010000002*QR_000000020000+P_010000102*QR_000000020001+P_010000202*QR_000000020002+P_110000002*QR_000000020100+P_110000102*QR_000000020101+P_110000202*QR_000000020102);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(P_002010000*QR_020000000000+P_002110000*QR_020000000010+P_102010000*QR_020000000100+P_102110000*QR_020000000110+P_202010000*QR_020000000200+P_202110000*QR_020000000210);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(P_002010000*QR_010010000000+P_002110000*QR_010010000010+P_102010000*QR_010010000100+P_102110000*QR_010010000110+P_202010000*QR_010010000200+P_202110000*QR_010010000210);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(P_002010000*QR_000020000000+P_002110000*QR_000020000010+P_102010000*QR_000020000100+P_102110000*QR_000020000110+P_202010000*QR_000020000200+P_202110000*QR_000020000210);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(P_002010000*QR_010000010000+P_002110000*QR_010000010010+P_102010000*QR_010000010100+P_102110000*QR_010000010110+P_202010000*QR_010000010200+P_202110000*QR_010000010210);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(P_002010000*QR_000010010000+P_002110000*QR_000010010010+P_102010000*QR_000010010100+P_102110000*QR_000010010110+P_202010000*QR_000010010200+P_202110000*QR_000010010210);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(P_002010000*QR_000000020000+P_002110000*QR_000000020010+P_102010000*QR_000000020100+P_102110000*QR_000000020110+P_202010000*QR_000000020200+P_202110000*QR_000000020210);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(P_001011000*QR_020000000000+P_001111000*QR_020000000010+P_001211000*QR_020000000020+P_101011000*QR_020000000100+P_101111000*QR_020000000110+P_101211000*QR_020000000120);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(P_001011000*QR_010010000000+P_001111000*QR_010010000010+P_001211000*QR_010010000020+P_101011000*QR_010010000100+P_101111000*QR_010010000110+P_101211000*QR_010010000120);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(P_001011000*QR_000020000000+P_001111000*QR_000020000010+P_001211000*QR_000020000020+P_101011000*QR_000020000100+P_101111000*QR_000020000110+P_101211000*QR_000020000120);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(P_001011000*QR_010000010000+P_001111000*QR_010000010010+P_001211000*QR_010000010020+P_101011000*QR_010000010100+P_101111000*QR_010000010110+P_101211000*QR_010000010120);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(P_001011000*QR_000010010000+P_001111000*QR_000010010010+P_001211000*QR_000010010020+P_101011000*QR_000010010100+P_101111000*QR_000010010110+P_101211000*QR_000010010120);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(P_001011000*QR_000000020000+P_001111000*QR_000000020010+P_001211000*QR_000000020020+P_101011000*QR_000000020100+P_101111000*QR_000000020110+P_101211000*QR_000000020120);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(P_000012000*QR_020000000000+P_000112000*QR_020000000010+P_000212000*QR_020000000020+P_000312000*QR_020000000030);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(P_000012000*QR_010010000000+P_000112000*QR_010010000010+P_000212000*QR_010010000020+P_000312000*QR_010010000030);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(P_000012000*QR_000020000000+P_000112000*QR_000020000010+P_000212000*QR_000020000020+P_000312000*QR_000020000030);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(P_000012000*QR_010000010000+P_000112000*QR_010000010010+P_000212000*QR_010000010020+P_000312000*QR_010000010030);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(P_000012000*QR_000010010000+P_000112000*QR_000010010010+P_000212000*QR_000010010020+P_000312000*QR_000010010030);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(P_000012000*QR_000000020000+P_000112000*QR_000000020010+P_000212000*QR_000000020020+P_000312000*QR_000000020030);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(P_001010001*QR_020000000000+P_001010101*QR_020000000001+P_001110001*QR_020000000010+P_001110101*QR_020000000011+P_101010001*QR_020000000100+P_101010101*QR_020000000101+P_101110001*QR_020000000110+P_101110101*QR_020000000111);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(P_001010001*QR_010010000000+P_001010101*QR_010010000001+P_001110001*QR_010010000010+P_001110101*QR_010010000011+P_101010001*QR_010010000100+P_101010101*QR_010010000101+P_101110001*QR_010010000110+P_101110101*QR_010010000111);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(P_001010001*QR_000020000000+P_001010101*QR_000020000001+P_001110001*QR_000020000010+P_001110101*QR_000020000011+P_101010001*QR_000020000100+P_101010101*QR_000020000101+P_101110001*QR_000020000110+P_101110101*QR_000020000111);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(P_001010001*QR_010000010000+P_001010101*QR_010000010001+P_001110001*QR_010000010010+P_001110101*QR_010000010011+P_101010001*QR_010000010100+P_101010101*QR_010000010101+P_101110001*QR_010000010110+P_101110101*QR_010000010111);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(P_001010001*QR_000010010000+P_001010101*QR_000010010001+P_001110001*QR_000010010010+P_001110101*QR_000010010011+P_101010001*QR_000010010100+P_101010101*QR_000010010101+P_101110001*QR_000010010110+P_101110101*QR_000010010111);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(P_001010001*QR_000000020000+P_001010101*QR_000000020001+P_001110001*QR_000000020010+P_001110101*QR_000000020011+P_101010001*QR_000000020100+P_101010101*QR_000000020101+P_101110001*QR_000000020110+P_101110101*QR_000000020111);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(P_000011001*QR_020000000000+P_000011101*QR_020000000001+P_000111001*QR_020000000010+P_000111101*QR_020000000011+P_000211001*QR_020000000020+P_000211101*QR_020000000021);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(P_000011001*QR_010010000000+P_000011101*QR_010010000001+P_000111001*QR_010010000010+P_000111101*QR_010010000011+P_000211001*QR_010010000020+P_000211101*QR_010010000021);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(P_000011001*QR_000020000000+P_000011101*QR_000020000001+P_000111001*QR_000020000010+P_000111101*QR_000020000011+P_000211001*QR_000020000020+P_000211101*QR_000020000021);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(P_000011001*QR_010000010000+P_000011101*QR_010000010001+P_000111001*QR_010000010010+P_000111101*QR_010000010011+P_000211001*QR_010000010020+P_000211101*QR_010000010021);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(P_000011001*QR_000010010000+P_000011101*QR_000010010001+P_000111001*QR_000010010010+P_000111101*QR_000010010011+P_000211001*QR_000010010020+P_000211101*QR_000010010021);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(P_000011001*QR_000000020000+P_000011101*QR_000000020001+P_000111001*QR_000000020010+P_000111101*QR_000000020011+P_000211001*QR_000000020020+P_000211101*QR_000000020021);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(P_000010002*QR_020000000000+P_000010102*QR_020000000001+P_000010202*QR_020000000002+P_000110002*QR_020000000010+P_000110102*QR_020000000011+P_000110202*QR_020000000012);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(P_000010002*QR_010010000000+P_000010102*QR_010010000001+P_000010202*QR_010010000002+P_000110002*QR_010010000010+P_000110102*QR_010010000011+P_000110202*QR_010010000012);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(P_000010002*QR_000020000000+P_000010102*QR_000020000001+P_000010202*QR_000020000002+P_000110002*QR_000020000010+P_000110102*QR_000020000011+P_000110202*QR_000020000012);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(P_000010002*QR_010000010000+P_000010102*QR_010000010001+P_000010202*QR_010000010002+P_000110002*QR_010000010010+P_000110102*QR_010000010011+P_000110202*QR_010000010012);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(P_000010002*QR_000010010000+P_000010102*QR_000010010001+P_000010202*QR_000010010002+P_000110002*QR_000010010010+P_000110102*QR_000010010011+P_000110202*QR_000010010012);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(P_000010002*QR_000000020000+P_000010102*QR_000000020001+P_000010202*QR_000000020002+P_000110002*QR_000000020010+P_000110102*QR_000000020011+P_000110202*QR_000000020012);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(P_002000010*QR_020000000000+P_002000110*QR_020000000001+P_102000010*QR_020000000100+P_102000110*QR_020000000101+P_202000010*QR_020000000200+P_202000110*QR_020000000201);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(P_002000010*QR_010010000000+P_002000110*QR_010010000001+P_102000010*QR_010010000100+P_102000110*QR_010010000101+P_202000010*QR_010010000200+P_202000110*QR_010010000201);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(P_002000010*QR_000020000000+P_002000110*QR_000020000001+P_102000010*QR_000020000100+P_102000110*QR_000020000101+P_202000010*QR_000020000200+P_202000110*QR_000020000201);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(P_002000010*QR_010000010000+P_002000110*QR_010000010001+P_102000010*QR_010000010100+P_102000110*QR_010000010101+P_202000010*QR_010000010200+P_202000110*QR_010000010201);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(P_002000010*QR_000010010000+P_002000110*QR_000010010001+P_102000010*QR_000010010100+P_102000110*QR_000010010101+P_202000010*QR_000010010200+P_202000110*QR_000010010201);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(P_002000010*QR_000000020000+P_002000110*QR_000000020001+P_102000010*QR_000000020100+P_102000110*QR_000000020101+P_202000010*QR_000000020200+P_202000110*QR_000000020201);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(P_001001010*QR_020000000000+P_001001110*QR_020000000001+P_001101010*QR_020000000010+P_001101110*QR_020000000011+P_101001010*QR_020000000100+P_101001110*QR_020000000101+P_101101010*QR_020000000110+P_101101110*QR_020000000111);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(P_001001010*QR_010010000000+P_001001110*QR_010010000001+P_001101010*QR_010010000010+P_001101110*QR_010010000011+P_101001010*QR_010010000100+P_101001110*QR_010010000101+P_101101010*QR_010010000110+P_101101110*QR_010010000111);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(P_001001010*QR_000020000000+P_001001110*QR_000020000001+P_001101010*QR_000020000010+P_001101110*QR_000020000011+P_101001010*QR_000020000100+P_101001110*QR_000020000101+P_101101010*QR_000020000110+P_101101110*QR_000020000111);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(P_001001010*QR_010000010000+P_001001110*QR_010000010001+P_001101010*QR_010000010010+P_001101110*QR_010000010011+P_101001010*QR_010000010100+P_101001110*QR_010000010101+P_101101010*QR_010000010110+P_101101110*QR_010000010111);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(P_001001010*QR_000010010000+P_001001110*QR_000010010001+P_001101010*QR_000010010010+P_001101110*QR_000010010011+P_101001010*QR_000010010100+P_101001110*QR_000010010101+P_101101010*QR_000010010110+P_101101110*QR_000010010111);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(P_001001010*QR_000000020000+P_001001110*QR_000000020001+P_001101010*QR_000000020010+P_001101110*QR_000000020011+P_101001010*QR_000000020100+P_101001110*QR_000000020101+P_101101010*QR_000000020110+P_101101110*QR_000000020111);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(P_000002010*QR_020000000000+P_000002110*QR_020000000001+P_000102010*QR_020000000010+P_000102110*QR_020000000011+P_000202010*QR_020000000020+P_000202110*QR_020000000021);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(P_000002010*QR_010010000000+P_000002110*QR_010010000001+P_000102010*QR_010010000010+P_000102110*QR_010010000011+P_000202010*QR_010010000020+P_000202110*QR_010010000021);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(P_000002010*QR_000020000000+P_000002110*QR_000020000001+P_000102010*QR_000020000010+P_000102110*QR_000020000011+P_000202010*QR_000020000020+P_000202110*QR_000020000021);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(P_000002010*QR_010000010000+P_000002110*QR_010000010001+P_000102010*QR_010000010010+P_000102110*QR_010000010011+P_000202010*QR_010000010020+P_000202110*QR_010000010021);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(P_000002010*QR_000010010000+P_000002110*QR_000010010001+P_000102010*QR_000010010010+P_000102110*QR_000010010011+P_000202010*QR_000010010020+P_000202110*QR_000010010021);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(P_000002010*QR_000000020000+P_000002110*QR_000000020001+P_000102010*QR_000000020010+P_000102110*QR_000000020011+P_000202010*QR_000000020020+P_000202110*QR_000000020021);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(P_001000011*QR_020000000000+P_001000111*QR_020000000001+P_001000211*QR_020000000002+P_101000011*QR_020000000100+P_101000111*QR_020000000101+P_101000211*QR_020000000102);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(P_001000011*QR_010010000000+P_001000111*QR_010010000001+P_001000211*QR_010010000002+P_101000011*QR_010010000100+P_101000111*QR_010010000101+P_101000211*QR_010010000102);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(P_001000011*QR_000020000000+P_001000111*QR_000020000001+P_001000211*QR_000020000002+P_101000011*QR_000020000100+P_101000111*QR_000020000101+P_101000211*QR_000020000102);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(P_001000011*QR_010000010000+P_001000111*QR_010000010001+P_001000211*QR_010000010002+P_101000011*QR_010000010100+P_101000111*QR_010000010101+P_101000211*QR_010000010102);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(P_001000011*QR_000010010000+P_001000111*QR_000010010001+P_001000211*QR_000010010002+P_101000011*QR_000010010100+P_101000111*QR_000010010101+P_101000211*QR_000010010102);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(P_001000011*QR_000000020000+P_001000111*QR_000000020001+P_001000211*QR_000000020002+P_101000011*QR_000000020100+P_101000111*QR_000000020101+P_101000211*QR_000000020102);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(P_000001011*QR_020000000000+P_000001111*QR_020000000001+P_000001211*QR_020000000002+P_000101011*QR_020000000010+P_000101111*QR_020000000011+P_000101211*QR_020000000012);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(P_000001011*QR_010010000000+P_000001111*QR_010010000001+P_000001211*QR_010010000002+P_000101011*QR_010010000010+P_000101111*QR_010010000011+P_000101211*QR_010010000012);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(P_000001011*QR_000020000000+P_000001111*QR_000020000001+P_000001211*QR_000020000002+P_000101011*QR_000020000010+P_000101111*QR_000020000011+P_000101211*QR_000020000012);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(P_000001011*QR_010000010000+P_000001111*QR_010000010001+P_000001211*QR_010000010002+P_000101011*QR_010000010010+P_000101111*QR_010000010011+P_000101211*QR_010000010012);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(P_000001011*QR_000010010000+P_000001111*QR_000010010001+P_000001211*QR_000010010002+P_000101011*QR_000010010010+P_000101111*QR_000010010011+P_000101211*QR_000010010012);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(P_000001011*QR_000000020000+P_000001111*QR_000000020001+P_000001211*QR_000000020002+P_000101011*QR_000000020010+P_000101111*QR_000000020011+P_000101211*QR_000000020012);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(P_000000012*QR_020000000000+P_000000112*QR_020000000001+P_000000212*QR_020000000002+P_000000312*QR_020000000003);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(P_000000012*QR_010010000000+P_000000112*QR_010010000001+P_000000212*QR_010010000002+P_000000312*QR_010010000003);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(P_000000012*QR_000020000000+P_000000112*QR_000020000001+P_000000212*QR_000020000002+P_000000312*QR_000020000003);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(P_000000012*QR_010000010000+P_000000112*QR_010000010001+P_000000212*QR_010000010002+P_000000312*QR_010000010003);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(P_000000012*QR_000010010000+P_000000112*QR_000010010001+P_000000212*QR_000010010002+P_000000312*QR_000010010003);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(P_000000012*QR_000000020000+P_000000112*QR_000000020001+P_000000212*QR_000000020002+P_000000312*QR_000000020003);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<18;ians++){
                    ans_temp[tId_x*18+ians]+=ans_temp[(tId_x+num_thread)*18+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<18;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_dsds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[1]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        if(i_contrc_bra>j_contrc_ket){
            if(tId_x==0){
                for(int ians=0;ians<36;ians++){
                    ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=0.0;
                }
            }
            continue;
        }
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[ii*3+0];
				Qd_010[1]=QC[ii*3+1];
				Qd_010[2]=QC[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<1;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_p=tex1Dfetch(tex_K2_p,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Zta,jj);
            double Zta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pp,jj);
            double pp=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+0);
            double PX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+1);
            double PY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+2);
            double PZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_010[3];
            temp_int2=tex1Dfetch(tex_PA,jj*3+0);
            Pd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+1);
            Pd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+2);
            Pd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_fs_4(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[4];
	double R_200[3];
	double R_300[2];
	double R_400[1];
	double R_010[4];
	double R_110[3];
	double R_210[2];
	double R_310[1];
	double R_020[3];
	double R_120[2];
	double R_220[1];
	double R_030[2];
	double R_130[1];
	double R_040[1];
	double R_001[4];
	double R_101[3];
	double R_201[2];
	double R_301[1];
	double R_011[3];
	double R_111[2];
	double R_211[1];
	double R_021[2];
	double R_121[1];
	double R_031[1];
	double R_002[3];
	double R_102[2];
	double R_202[1];
	double R_012[2];
	double R_112[1];
	double R_022[1];
	double R_003[2];
	double R_103[1];
	double R_013[1];
	double R_004[1];
	for(int i=0;i<4;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<2;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<2;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
		double Pd_110[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
	double P_020000000=Pd_020[0];
	double P_120000000=Pd_120[0];
	double P_220000000=Pd_220[0];
	double P_010010000=Pd_010[0]*Pd_010[1];
	double P_010110000=Pd_010[0]*Pd_110[1];
	double P_110010000=Pd_110[0]*Pd_010[1];
	double P_110110000=Pd_110[0]*Pd_110[1];
	double P_000020000=Pd_020[1];
	double P_000120000=Pd_120[1];
	double P_000220000=Pd_220[1];
	double P_010000010=Pd_010[0]*Pd_010[2];
	double P_010000110=Pd_010[0]*Pd_110[2];
	double P_110000010=Pd_110[0]*Pd_010[2];
	double P_110000110=Pd_110[0]*Pd_110[2];
	double P_000010010=Pd_010[1]*Pd_010[2];
	double P_000010110=Pd_010[1]*Pd_110[2];
	double P_000110010=Pd_110[1]*Pd_010[2];
	double P_000110110=Pd_110[1]*Pd_110[2];
	double P_000000020=Pd_020[2];
	double P_000000120=Pd_120[2];
	double P_000000220=Pd_220[2];
				double PR_020000000000=P_020000000*R_000[0]+-1*P_120000000*R_100[0]+P_220000000*R_200[0];
				double PR_010010000000=P_010010000*R_000[0]+-1*P_010110000*R_010[0]+-1*P_110010000*R_100[0]+P_110110000*R_110[0];
				double PR_000020000000=P_000020000*R_000[0]+-1*P_000120000*R_010[0]+P_000220000*R_020[0];
				double PR_010000010000=P_010000010*R_000[0]+-1*P_010000110*R_001[0]+-1*P_110000010*R_100[0]+P_110000110*R_101[0];
				double PR_000010010000=P_000010010*R_000[0]+-1*P_000010110*R_001[0]+-1*P_000110010*R_010[0]+P_000110110*R_011[0];
				double PR_000000020000=P_000000020*R_000[0]+-1*P_000000120*R_001[0]+P_000000220*R_002[0];
				double PR_020000000001=P_020000000*R_001[0]+-1*P_120000000*R_101[0]+P_220000000*R_201[0];
				double PR_010010000001=P_010010000*R_001[0]+-1*P_010110000*R_011[0]+-1*P_110010000*R_101[0]+P_110110000*R_111[0];
				double PR_000020000001=P_000020000*R_001[0]+-1*P_000120000*R_011[0]+P_000220000*R_021[0];
				double PR_010000010001=P_010000010*R_001[0]+-1*P_010000110*R_002[0]+-1*P_110000010*R_101[0]+P_110000110*R_102[0];
				double PR_000010010001=P_000010010*R_001[0]+-1*P_000010110*R_002[0]+-1*P_000110010*R_011[0]+P_000110110*R_012[0];
				double PR_000000020001=P_000000020*R_001[0]+-1*P_000000120*R_002[0]+P_000000220*R_003[0];
				double PR_020000000010=P_020000000*R_010[0]+-1*P_120000000*R_110[0]+P_220000000*R_210[0];
				double PR_010010000010=P_010010000*R_010[0]+-1*P_010110000*R_020[0]+-1*P_110010000*R_110[0]+P_110110000*R_120[0];
				double PR_000020000010=P_000020000*R_010[0]+-1*P_000120000*R_020[0]+P_000220000*R_030[0];
				double PR_010000010010=P_010000010*R_010[0]+-1*P_010000110*R_011[0]+-1*P_110000010*R_110[0]+P_110000110*R_111[0];
				double PR_000010010010=P_000010010*R_010[0]+-1*P_000010110*R_011[0]+-1*P_000110010*R_020[0]+P_000110110*R_021[0];
				double PR_000000020010=P_000000020*R_010[0]+-1*P_000000120*R_011[0]+P_000000220*R_012[0];
				double PR_020000000100=P_020000000*R_100[0]+-1*P_120000000*R_200[0]+P_220000000*R_300[0];
				double PR_010010000100=P_010010000*R_100[0]+-1*P_010110000*R_110[0]+-1*P_110010000*R_200[0]+P_110110000*R_210[0];
				double PR_000020000100=P_000020000*R_100[0]+-1*P_000120000*R_110[0]+P_000220000*R_120[0];
				double PR_010000010100=P_010000010*R_100[0]+-1*P_010000110*R_101[0]+-1*P_110000010*R_200[0]+P_110000110*R_201[0];
				double PR_000010010100=P_000010010*R_100[0]+-1*P_000010110*R_101[0]+-1*P_000110010*R_110[0]+P_000110110*R_111[0];
				double PR_000000020100=P_000000020*R_100[0]+-1*P_000000120*R_101[0]+P_000000220*R_102[0];
				double PR_020000000002=P_020000000*R_002[0]+-1*P_120000000*R_102[0]+P_220000000*R_202[0];
				double PR_010010000002=P_010010000*R_002[0]+-1*P_010110000*R_012[0]+-1*P_110010000*R_102[0]+P_110110000*R_112[0];
				double PR_000020000002=P_000020000*R_002[0]+-1*P_000120000*R_012[0]+P_000220000*R_022[0];
				double PR_010000010002=P_010000010*R_002[0]+-1*P_010000110*R_003[0]+-1*P_110000010*R_102[0]+P_110000110*R_103[0];
				double PR_000010010002=P_000010010*R_002[0]+-1*P_000010110*R_003[0]+-1*P_000110010*R_012[0]+P_000110110*R_013[0];
				double PR_000000020002=P_000000020*R_002[0]+-1*P_000000120*R_003[0]+P_000000220*R_004[0];
				double PR_020000000011=P_020000000*R_011[0]+-1*P_120000000*R_111[0]+P_220000000*R_211[0];
				double PR_010010000011=P_010010000*R_011[0]+-1*P_010110000*R_021[0]+-1*P_110010000*R_111[0]+P_110110000*R_121[0];
				double PR_000020000011=P_000020000*R_011[0]+-1*P_000120000*R_021[0]+P_000220000*R_031[0];
				double PR_010000010011=P_010000010*R_011[0]+-1*P_010000110*R_012[0]+-1*P_110000010*R_111[0]+P_110000110*R_112[0];
				double PR_000010010011=P_000010010*R_011[0]+-1*P_000010110*R_012[0]+-1*P_000110010*R_021[0]+P_000110110*R_022[0];
				double PR_000000020011=P_000000020*R_011[0]+-1*P_000000120*R_012[0]+P_000000220*R_013[0];
				double PR_020000000020=P_020000000*R_020[0]+-1*P_120000000*R_120[0]+P_220000000*R_220[0];
				double PR_010010000020=P_010010000*R_020[0]+-1*P_010110000*R_030[0]+-1*P_110010000*R_120[0]+P_110110000*R_130[0];
				double PR_000020000020=P_000020000*R_020[0]+-1*P_000120000*R_030[0]+P_000220000*R_040[0];
				double PR_010000010020=P_010000010*R_020[0]+-1*P_010000110*R_021[0]+-1*P_110000010*R_120[0]+P_110000110*R_121[0];
				double PR_000010010020=P_000010010*R_020[0]+-1*P_000010110*R_021[0]+-1*P_000110010*R_030[0]+P_000110110*R_031[0];
				double PR_000000020020=P_000000020*R_020[0]+-1*P_000000120*R_021[0]+P_000000220*R_022[0];
				double PR_020000000101=P_020000000*R_101[0]+-1*P_120000000*R_201[0]+P_220000000*R_301[0];
				double PR_010010000101=P_010010000*R_101[0]+-1*P_010110000*R_111[0]+-1*P_110010000*R_201[0]+P_110110000*R_211[0];
				double PR_000020000101=P_000020000*R_101[0]+-1*P_000120000*R_111[0]+P_000220000*R_121[0];
				double PR_010000010101=P_010000010*R_101[0]+-1*P_010000110*R_102[0]+-1*P_110000010*R_201[0]+P_110000110*R_202[0];
				double PR_000010010101=P_000010010*R_101[0]+-1*P_000010110*R_102[0]+-1*P_000110010*R_111[0]+P_000110110*R_112[0];
				double PR_000000020101=P_000000020*R_101[0]+-1*P_000000120*R_102[0]+P_000000220*R_103[0];
				double PR_020000000110=P_020000000*R_110[0]+-1*P_120000000*R_210[0]+P_220000000*R_310[0];
				double PR_010010000110=P_010010000*R_110[0]+-1*P_010110000*R_120[0]+-1*P_110010000*R_210[0]+P_110110000*R_220[0];
				double PR_000020000110=P_000020000*R_110[0]+-1*P_000120000*R_120[0]+P_000220000*R_130[0];
				double PR_010000010110=P_010000010*R_110[0]+-1*P_010000110*R_111[0]+-1*P_110000010*R_210[0]+P_110000110*R_211[0];
				double PR_000010010110=P_000010010*R_110[0]+-1*P_000010110*R_111[0]+-1*P_000110010*R_120[0]+P_000110110*R_121[0];
				double PR_000000020110=P_000000020*R_110[0]+-1*P_000000120*R_111[0]+P_000000220*R_112[0];
				double PR_020000000200=P_020000000*R_200[0]+-1*P_120000000*R_300[0]+P_220000000*R_400[0];
				double PR_010010000200=P_010010000*R_200[0]+-1*P_010110000*R_210[0]+-1*P_110010000*R_300[0]+P_110110000*R_310[0];
				double PR_000020000200=P_000020000*R_200[0]+-1*P_000120000*R_210[0]+P_000220000*R_220[0];
				double PR_010000010200=P_010000010*R_200[0]+-1*P_010000110*R_201[0]+-1*P_110000010*R_300[0]+P_110000110*R_301[0];
				double PR_000010010200=P_000010010*R_200[0]+-1*P_000010110*R_201[0]+-1*P_000110010*R_210[0]+P_000110110*R_211[0];
				double PR_000000020200=P_000000020*R_200[0]+-1*P_000000120*R_201[0]+P_000000220*R_202[0];
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
			ans_temp[ans_id*36+0]+=Q_020000000*PR_020000000000+Q_120000000*PR_020000000100+Q_220000000*PR_020000000200;
			ans_temp[ans_id*36+1]+=Q_010010000*PR_020000000000+Q_010110000*PR_020000000010+Q_110010000*PR_020000000100+Q_110110000*PR_020000000110;
			ans_temp[ans_id*36+2]+=Q_000020000*PR_020000000000+Q_000120000*PR_020000000010+Q_000220000*PR_020000000020;
			ans_temp[ans_id*36+3]+=Q_010000010*PR_020000000000+Q_010000110*PR_020000000001+Q_110000010*PR_020000000100+Q_110000110*PR_020000000101;
			ans_temp[ans_id*36+4]+=Q_000010010*PR_020000000000+Q_000010110*PR_020000000001+Q_000110010*PR_020000000010+Q_000110110*PR_020000000011;
			ans_temp[ans_id*36+5]+=Q_000000020*PR_020000000000+Q_000000120*PR_020000000001+Q_000000220*PR_020000000002;
			ans_temp[ans_id*36+6]+=Q_020000000*PR_010010000000+Q_120000000*PR_010010000100+Q_220000000*PR_010010000200;
			ans_temp[ans_id*36+7]+=Q_010010000*PR_010010000000+Q_010110000*PR_010010000010+Q_110010000*PR_010010000100+Q_110110000*PR_010010000110;
			ans_temp[ans_id*36+8]+=Q_000020000*PR_010010000000+Q_000120000*PR_010010000010+Q_000220000*PR_010010000020;
			ans_temp[ans_id*36+9]+=Q_010000010*PR_010010000000+Q_010000110*PR_010010000001+Q_110000010*PR_010010000100+Q_110000110*PR_010010000101;
			ans_temp[ans_id*36+10]+=Q_000010010*PR_010010000000+Q_000010110*PR_010010000001+Q_000110010*PR_010010000010+Q_000110110*PR_010010000011;
			ans_temp[ans_id*36+11]+=Q_000000020*PR_010010000000+Q_000000120*PR_010010000001+Q_000000220*PR_010010000002;
			ans_temp[ans_id*36+12]+=Q_020000000*PR_000020000000+Q_120000000*PR_000020000100+Q_220000000*PR_000020000200;
			ans_temp[ans_id*36+13]+=Q_010010000*PR_000020000000+Q_010110000*PR_000020000010+Q_110010000*PR_000020000100+Q_110110000*PR_000020000110;
			ans_temp[ans_id*36+14]+=Q_000020000*PR_000020000000+Q_000120000*PR_000020000010+Q_000220000*PR_000020000020;
			ans_temp[ans_id*36+15]+=Q_010000010*PR_000020000000+Q_010000110*PR_000020000001+Q_110000010*PR_000020000100+Q_110000110*PR_000020000101;
			ans_temp[ans_id*36+16]+=Q_000010010*PR_000020000000+Q_000010110*PR_000020000001+Q_000110010*PR_000020000010+Q_000110110*PR_000020000011;
			ans_temp[ans_id*36+17]+=Q_000000020*PR_000020000000+Q_000000120*PR_000020000001+Q_000000220*PR_000020000002;
			ans_temp[ans_id*36+18]+=Q_020000000*PR_010000010000+Q_120000000*PR_010000010100+Q_220000000*PR_010000010200;
			ans_temp[ans_id*36+19]+=Q_010010000*PR_010000010000+Q_010110000*PR_010000010010+Q_110010000*PR_010000010100+Q_110110000*PR_010000010110;
			ans_temp[ans_id*36+20]+=Q_000020000*PR_010000010000+Q_000120000*PR_010000010010+Q_000220000*PR_010000010020;
			ans_temp[ans_id*36+21]+=Q_010000010*PR_010000010000+Q_010000110*PR_010000010001+Q_110000010*PR_010000010100+Q_110000110*PR_010000010101;
			ans_temp[ans_id*36+22]+=Q_000010010*PR_010000010000+Q_000010110*PR_010000010001+Q_000110010*PR_010000010010+Q_000110110*PR_010000010011;
			ans_temp[ans_id*36+23]+=Q_000000020*PR_010000010000+Q_000000120*PR_010000010001+Q_000000220*PR_010000010002;
			ans_temp[ans_id*36+24]+=Q_020000000*PR_000010010000+Q_120000000*PR_000010010100+Q_220000000*PR_000010010200;
			ans_temp[ans_id*36+25]+=Q_010010000*PR_000010010000+Q_010110000*PR_000010010010+Q_110010000*PR_000010010100+Q_110110000*PR_000010010110;
			ans_temp[ans_id*36+26]+=Q_000020000*PR_000010010000+Q_000120000*PR_000010010010+Q_000220000*PR_000010010020;
			ans_temp[ans_id*36+27]+=Q_010000010*PR_000010010000+Q_010000110*PR_000010010001+Q_110000010*PR_000010010100+Q_110000110*PR_000010010101;
			ans_temp[ans_id*36+28]+=Q_000010010*PR_000010010000+Q_000010110*PR_000010010001+Q_000110010*PR_000010010010+Q_000110110*PR_000010010011;
			ans_temp[ans_id*36+29]+=Q_000000020*PR_000010010000+Q_000000120*PR_000010010001+Q_000000220*PR_000010010002;
			ans_temp[ans_id*36+30]+=Q_020000000*PR_000000020000+Q_120000000*PR_000000020100+Q_220000000*PR_000000020200;
			ans_temp[ans_id*36+31]+=Q_010010000*PR_000000020000+Q_010110000*PR_000000020010+Q_110010000*PR_000000020100+Q_110110000*PR_000000020110;
			ans_temp[ans_id*36+32]+=Q_000020000*PR_000000020000+Q_000120000*PR_000000020010+Q_000220000*PR_000000020020;
			ans_temp[ans_id*36+33]+=Q_010000010*PR_000000020000+Q_010000110*PR_000000020001+Q_110000010*PR_000000020100+Q_110000110*PR_000000020101;
			ans_temp[ans_id*36+34]+=Q_000010010*PR_000000020000+Q_000010110*PR_000000020001+Q_000110010*PR_000000020010+Q_000110110*PR_000000020011;
			ans_temp[ans_id*36+35]+=Q_000000020*PR_000000020000+Q_000000120*PR_000000020001+Q_000000220*PR_000000020002;
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_dsds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[1]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        if(i_contrc_bra>j_contrc_ket){
            if(tId_x==0){
                for(int ians=0;ians<36;ians++){
                    ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=0.0;
                }
            }
            continue;
        }
        for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
            unsigned int id_bra=id_bra_in[ii];
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
            float K2_p=K2_p_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_end-primit_ket_start;j+=tdis){
            unsigned int jj=primit_ket_start+j;
            unsigned int id_ket=tex1Dfetch(tex_id_ket,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<1;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Eta,jj);
            double Eta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pq,jj);
            double pq=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+0);
            double QX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+1);
            double QY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+2);
            double QZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Qd_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            Qd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            Qd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            Qd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_fs_4(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[4];
	double R_200[3];
	double R_300[2];
	double R_400[1];
	double R_010[4];
	double R_110[3];
	double R_210[2];
	double R_310[1];
	double R_020[3];
	double R_120[2];
	double R_220[1];
	double R_030[2];
	double R_130[1];
	double R_040[1];
	double R_001[4];
	double R_101[3];
	double R_201[2];
	double R_301[1];
	double R_011[3];
	double R_111[2];
	double R_211[1];
	double R_021[2];
	double R_121[1];
	double R_031[1];
	double R_002[3];
	double R_102[2];
	double R_202[1];
	double R_012[2];
	double R_112[1];
	double R_022[1];
	double R_003[2];
	double R_103[1];
	double R_013[1];
	double R_004[1];
	for(int i=0;i<4;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<2;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<2;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
				double QR_020000000000=Q_020000000*R_000[0]+-1*Q_120000000*R_100[0]+Q_220000000*R_200[0];
				double QR_010010000000=Q_010010000*R_000[0]+-1*Q_010110000*R_010[0]+-1*Q_110010000*R_100[0]+Q_110110000*R_110[0];
				double QR_000020000000=Q_000020000*R_000[0]+-1*Q_000120000*R_010[0]+Q_000220000*R_020[0];
				double QR_010000010000=Q_010000010*R_000[0]+-1*Q_010000110*R_001[0]+-1*Q_110000010*R_100[0]+Q_110000110*R_101[0];
				double QR_000010010000=Q_000010010*R_000[0]+-1*Q_000010110*R_001[0]+-1*Q_000110010*R_010[0]+Q_000110110*R_011[0];
				double QR_000000020000=Q_000000020*R_000[0]+-1*Q_000000120*R_001[0]+Q_000000220*R_002[0];
				double QR_020000000001=Q_020000000*R_001[0]+-1*Q_120000000*R_101[0]+Q_220000000*R_201[0];
				double QR_010010000001=Q_010010000*R_001[0]+-1*Q_010110000*R_011[0]+-1*Q_110010000*R_101[0]+Q_110110000*R_111[0];
				double QR_000020000001=Q_000020000*R_001[0]+-1*Q_000120000*R_011[0]+Q_000220000*R_021[0];
				double QR_010000010001=Q_010000010*R_001[0]+-1*Q_010000110*R_002[0]+-1*Q_110000010*R_101[0]+Q_110000110*R_102[0];
				double QR_000010010001=Q_000010010*R_001[0]+-1*Q_000010110*R_002[0]+-1*Q_000110010*R_011[0]+Q_000110110*R_012[0];
				double QR_000000020001=Q_000000020*R_001[0]+-1*Q_000000120*R_002[0]+Q_000000220*R_003[0];
				double QR_020000000010=Q_020000000*R_010[0]+-1*Q_120000000*R_110[0]+Q_220000000*R_210[0];
				double QR_010010000010=Q_010010000*R_010[0]+-1*Q_010110000*R_020[0]+-1*Q_110010000*R_110[0]+Q_110110000*R_120[0];
				double QR_000020000010=Q_000020000*R_010[0]+-1*Q_000120000*R_020[0]+Q_000220000*R_030[0];
				double QR_010000010010=Q_010000010*R_010[0]+-1*Q_010000110*R_011[0]+-1*Q_110000010*R_110[0]+Q_110000110*R_111[0];
				double QR_000010010010=Q_000010010*R_010[0]+-1*Q_000010110*R_011[0]+-1*Q_000110010*R_020[0]+Q_000110110*R_021[0];
				double QR_000000020010=Q_000000020*R_010[0]+-1*Q_000000120*R_011[0]+Q_000000220*R_012[0];
				double QR_020000000100=Q_020000000*R_100[0]+-1*Q_120000000*R_200[0]+Q_220000000*R_300[0];
				double QR_010010000100=Q_010010000*R_100[0]+-1*Q_010110000*R_110[0]+-1*Q_110010000*R_200[0]+Q_110110000*R_210[0];
				double QR_000020000100=Q_000020000*R_100[0]+-1*Q_000120000*R_110[0]+Q_000220000*R_120[0];
				double QR_010000010100=Q_010000010*R_100[0]+-1*Q_010000110*R_101[0]+-1*Q_110000010*R_200[0]+Q_110000110*R_201[0];
				double QR_000010010100=Q_000010010*R_100[0]+-1*Q_000010110*R_101[0]+-1*Q_000110010*R_110[0]+Q_000110110*R_111[0];
				double QR_000000020100=Q_000000020*R_100[0]+-1*Q_000000120*R_101[0]+Q_000000220*R_102[0];
				double QR_020000000002=Q_020000000*R_002[0]+-1*Q_120000000*R_102[0]+Q_220000000*R_202[0];
				double QR_010010000002=Q_010010000*R_002[0]+-1*Q_010110000*R_012[0]+-1*Q_110010000*R_102[0]+Q_110110000*R_112[0];
				double QR_000020000002=Q_000020000*R_002[0]+-1*Q_000120000*R_012[0]+Q_000220000*R_022[0];
				double QR_010000010002=Q_010000010*R_002[0]+-1*Q_010000110*R_003[0]+-1*Q_110000010*R_102[0]+Q_110000110*R_103[0];
				double QR_000010010002=Q_000010010*R_002[0]+-1*Q_000010110*R_003[0]+-1*Q_000110010*R_012[0]+Q_000110110*R_013[0];
				double QR_000000020002=Q_000000020*R_002[0]+-1*Q_000000120*R_003[0]+Q_000000220*R_004[0];
				double QR_020000000011=Q_020000000*R_011[0]+-1*Q_120000000*R_111[0]+Q_220000000*R_211[0];
				double QR_010010000011=Q_010010000*R_011[0]+-1*Q_010110000*R_021[0]+-1*Q_110010000*R_111[0]+Q_110110000*R_121[0];
				double QR_000020000011=Q_000020000*R_011[0]+-1*Q_000120000*R_021[0]+Q_000220000*R_031[0];
				double QR_010000010011=Q_010000010*R_011[0]+-1*Q_010000110*R_012[0]+-1*Q_110000010*R_111[0]+Q_110000110*R_112[0];
				double QR_000010010011=Q_000010010*R_011[0]+-1*Q_000010110*R_012[0]+-1*Q_000110010*R_021[0]+Q_000110110*R_022[0];
				double QR_000000020011=Q_000000020*R_011[0]+-1*Q_000000120*R_012[0]+Q_000000220*R_013[0];
				double QR_020000000020=Q_020000000*R_020[0]+-1*Q_120000000*R_120[0]+Q_220000000*R_220[0];
				double QR_010010000020=Q_010010000*R_020[0]+-1*Q_010110000*R_030[0]+-1*Q_110010000*R_120[0]+Q_110110000*R_130[0];
				double QR_000020000020=Q_000020000*R_020[0]+-1*Q_000120000*R_030[0]+Q_000220000*R_040[0];
				double QR_010000010020=Q_010000010*R_020[0]+-1*Q_010000110*R_021[0]+-1*Q_110000010*R_120[0]+Q_110000110*R_121[0];
				double QR_000010010020=Q_000010010*R_020[0]+-1*Q_000010110*R_021[0]+-1*Q_000110010*R_030[0]+Q_000110110*R_031[0];
				double QR_000000020020=Q_000000020*R_020[0]+-1*Q_000000120*R_021[0]+Q_000000220*R_022[0];
				double QR_020000000101=Q_020000000*R_101[0]+-1*Q_120000000*R_201[0]+Q_220000000*R_301[0];
				double QR_010010000101=Q_010010000*R_101[0]+-1*Q_010110000*R_111[0]+-1*Q_110010000*R_201[0]+Q_110110000*R_211[0];
				double QR_000020000101=Q_000020000*R_101[0]+-1*Q_000120000*R_111[0]+Q_000220000*R_121[0];
				double QR_010000010101=Q_010000010*R_101[0]+-1*Q_010000110*R_102[0]+-1*Q_110000010*R_201[0]+Q_110000110*R_202[0];
				double QR_000010010101=Q_000010010*R_101[0]+-1*Q_000010110*R_102[0]+-1*Q_000110010*R_111[0]+Q_000110110*R_112[0];
				double QR_000000020101=Q_000000020*R_101[0]+-1*Q_000000120*R_102[0]+Q_000000220*R_103[0];
				double QR_020000000110=Q_020000000*R_110[0]+-1*Q_120000000*R_210[0]+Q_220000000*R_310[0];
				double QR_010010000110=Q_010010000*R_110[0]+-1*Q_010110000*R_120[0]+-1*Q_110010000*R_210[0]+Q_110110000*R_220[0];
				double QR_000020000110=Q_000020000*R_110[0]+-1*Q_000120000*R_120[0]+Q_000220000*R_130[0];
				double QR_010000010110=Q_010000010*R_110[0]+-1*Q_010000110*R_111[0]+-1*Q_110000010*R_210[0]+Q_110000110*R_211[0];
				double QR_000010010110=Q_000010010*R_110[0]+-1*Q_000010110*R_111[0]+-1*Q_000110010*R_120[0]+Q_000110110*R_121[0];
				double QR_000000020110=Q_000000020*R_110[0]+-1*Q_000000120*R_111[0]+Q_000000220*R_112[0];
				double QR_020000000200=Q_020000000*R_200[0]+-1*Q_120000000*R_300[0]+Q_220000000*R_400[0];
				double QR_010010000200=Q_010010000*R_200[0]+-1*Q_010110000*R_210[0]+-1*Q_110010000*R_300[0]+Q_110110000*R_310[0];
				double QR_000020000200=Q_000020000*R_200[0]+-1*Q_000120000*R_210[0]+Q_000220000*R_220[0];
				double QR_010000010200=Q_010000010*R_200[0]+-1*Q_010000110*R_201[0]+-1*Q_110000010*R_300[0]+Q_110000110*R_301[0];
				double QR_000010010200=Q_000010010*R_200[0]+-1*Q_000010110*R_201[0]+-1*Q_000110010*R_210[0]+Q_000110110*R_211[0];
				double QR_000000020200=Q_000000020*R_200[0]+-1*Q_000000120*R_201[0]+Q_000000220*R_202[0];
		double Pd_110[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
	double P_020000000=Pd_020[0];
	double P_120000000=Pd_120[0];
	double P_220000000=Pd_220[0];
	double P_010010000=Pd_010[0]*Pd_010[1];
	double P_010110000=Pd_010[0]*Pd_110[1];
	double P_110010000=Pd_110[0]*Pd_010[1];
	double P_110110000=Pd_110[0]*Pd_110[1];
	double P_000020000=Pd_020[1];
	double P_000120000=Pd_120[1];
	double P_000220000=Pd_220[1];
	double P_010000010=Pd_010[0]*Pd_010[2];
	double P_010000110=Pd_010[0]*Pd_110[2];
	double P_110000010=Pd_110[0]*Pd_010[2];
	double P_110000110=Pd_110[0]*Pd_110[2];
	double P_000010010=Pd_010[1]*Pd_010[2];
	double P_000010110=Pd_010[1]*Pd_110[2];
	double P_000110010=Pd_110[1]*Pd_010[2];
	double P_000110110=Pd_110[1]*Pd_110[2];
	double P_000000020=Pd_020[2];
	double P_000000120=Pd_120[2];
	double P_000000220=Pd_220[2];
			ans_temp[ans_id*36+0]+=P_020000000*QR_020000000000+P_120000000*QR_020000000100+P_220000000*QR_020000000200;
			ans_temp[ans_id*36+1]+=P_020000000*QR_010010000000+P_120000000*QR_010010000100+P_220000000*QR_010010000200;
			ans_temp[ans_id*36+2]+=P_020000000*QR_000020000000+P_120000000*QR_000020000100+P_220000000*QR_000020000200;
			ans_temp[ans_id*36+3]+=P_020000000*QR_010000010000+P_120000000*QR_010000010100+P_220000000*QR_010000010200;
			ans_temp[ans_id*36+4]+=P_020000000*QR_000010010000+P_120000000*QR_000010010100+P_220000000*QR_000010010200;
			ans_temp[ans_id*36+5]+=P_020000000*QR_000000020000+P_120000000*QR_000000020100+P_220000000*QR_000000020200;
			ans_temp[ans_id*36+6]+=P_010010000*QR_020000000000+P_010110000*QR_020000000010+P_110010000*QR_020000000100+P_110110000*QR_020000000110;
			ans_temp[ans_id*36+7]+=P_010010000*QR_010010000000+P_010110000*QR_010010000010+P_110010000*QR_010010000100+P_110110000*QR_010010000110;
			ans_temp[ans_id*36+8]+=P_010010000*QR_000020000000+P_010110000*QR_000020000010+P_110010000*QR_000020000100+P_110110000*QR_000020000110;
			ans_temp[ans_id*36+9]+=P_010010000*QR_010000010000+P_010110000*QR_010000010010+P_110010000*QR_010000010100+P_110110000*QR_010000010110;
			ans_temp[ans_id*36+10]+=P_010010000*QR_000010010000+P_010110000*QR_000010010010+P_110010000*QR_000010010100+P_110110000*QR_000010010110;
			ans_temp[ans_id*36+11]+=P_010010000*QR_000000020000+P_010110000*QR_000000020010+P_110010000*QR_000000020100+P_110110000*QR_000000020110;
			ans_temp[ans_id*36+12]+=P_000020000*QR_020000000000+P_000120000*QR_020000000010+P_000220000*QR_020000000020;
			ans_temp[ans_id*36+13]+=P_000020000*QR_010010000000+P_000120000*QR_010010000010+P_000220000*QR_010010000020;
			ans_temp[ans_id*36+14]+=P_000020000*QR_000020000000+P_000120000*QR_000020000010+P_000220000*QR_000020000020;
			ans_temp[ans_id*36+15]+=P_000020000*QR_010000010000+P_000120000*QR_010000010010+P_000220000*QR_010000010020;
			ans_temp[ans_id*36+16]+=P_000020000*QR_000010010000+P_000120000*QR_000010010010+P_000220000*QR_000010010020;
			ans_temp[ans_id*36+17]+=P_000020000*QR_000000020000+P_000120000*QR_000000020010+P_000220000*QR_000000020020;
			ans_temp[ans_id*36+18]+=P_010000010*QR_020000000000+P_010000110*QR_020000000001+P_110000010*QR_020000000100+P_110000110*QR_020000000101;
			ans_temp[ans_id*36+19]+=P_010000010*QR_010010000000+P_010000110*QR_010010000001+P_110000010*QR_010010000100+P_110000110*QR_010010000101;
			ans_temp[ans_id*36+20]+=P_010000010*QR_000020000000+P_010000110*QR_000020000001+P_110000010*QR_000020000100+P_110000110*QR_000020000101;
			ans_temp[ans_id*36+21]+=P_010000010*QR_010000010000+P_010000110*QR_010000010001+P_110000010*QR_010000010100+P_110000110*QR_010000010101;
			ans_temp[ans_id*36+22]+=P_010000010*QR_000010010000+P_010000110*QR_000010010001+P_110000010*QR_000010010100+P_110000110*QR_000010010101;
			ans_temp[ans_id*36+23]+=P_010000010*QR_000000020000+P_010000110*QR_000000020001+P_110000010*QR_000000020100+P_110000110*QR_000000020101;
			ans_temp[ans_id*36+24]+=P_000010010*QR_020000000000+P_000010110*QR_020000000001+P_000110010*QR_020000000010+P_000110110*QR_020000000011;
			ans_temp[ans_id*36+25]+=P_000010010*QR_010010000000+P_000010110*QR_010010000001+P_000110010*QR_010010000010+P_000110110*QR_010010000011;
			ans_temp[ans_id*36+26]+=P_000010010*QR_000020000000+P_000010110*QR_000020000001+P_000110010*QR_000020000010+P_000110110*QR_000020000011;
			ans_temp[ans_id*36+27]+=P_000010010*QR_010000010000+P_000010110*QR_010000010001+P_000110010*QR_010000010010+P_000110110*QR_010000010011;
			ans_temp[ans_id*36+28]+=P_000010010*QR_000010010000+P_000010110*QR_000010010001+P_000110010*QR_000010010010+P_000110110*QR_000010010011;
			ans_temp[ans_id*36+29]+=P_000010010*QR_000000020000+P_000010110*QR_000000020001+P_000110010*QR_000000020010+P_000110110*QR_000000020011;
			ans_temp[ans_id*36+30]+=P_000000020*QR_020000000000+P_000000120*QR_020000000001+P_000000220*QR_020000000002;
			ans_temp[ans_id*36+31]+=P_000000020*QR_010010000000+P_000000120*QR_010010000001+P_000000220*QR_010010000002;
			ans_temp[ans_id*36+32]+=P_000000020*QR_000020000000+P_000000120*QR_000020000001+P_000000220*QR_000020000002;
			ans_temp[ans_id*36+33]+=P_000000020*QR_010000010000+P_000000120*QR_010000010001+P_000000220*QR_010000010002;
			ans_temp[ans_id*36+34]+=P_000000020*QR_000010010000+P_000000120*QR_000010010001+P_000000220*QR_000010010002;
			ans_temp[ans_id*36+35]+=P_000000020*QR_000000020000+P_000000120*QR_000000020001+P_000000220*QR_000000020002;
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_dpds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[3]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[ii*3+0];
				Qd_010[1]=QC[ii*3+1];
				Qd_010[2]=QC[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_p=tex1Dfetch(tex_K2_p,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Zta,jj);
            double Zta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pp,jj);
            double pp=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+0);
            double PX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+1);
            double PY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+2);
            double PZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_010[3];
            temp_int2=tex1Dfetch(tex_PA,jj*3+0);
            Pd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+1);
            Pd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+2);
            Pd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_001[3];
            temp_int2=tex1Dfetch(tex_PB,jj*3+0);
            Pd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+1);
            Pd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+2);
            Pd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[6];
                Ft_fs_5(5,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[5];
	double R_200[4];
	double R_300[3];
	double R_400[2];
	double R_500[1];
	double R_010[5];
	double R_110[4];
	double R_210[3];
	double R_310[2];
	double R_410[1];
	double R_020[4];
	double R_120[3];
	double R_220[2];
	double R_320[1];
	double R_030[3];
	double R_130[2];
	double R_230[1];
	double R_040[2];
	double R_140[1];
	double R_050[1];
	double R_001[5];
	double R_101[4];
	double R_201[3];
	double R_301[2];
	double R_401[1];
	double R_011[4];
	double R_111[3];
	double R_211[2];
	double R_311[1];
	double R_021[3];
	double R_121[2];
	double R_221[1];
	double R_031[2];
	double R_131[1];
	double R_041[1];
	double R_002[4];
	double R_102[3];
	double R_202[2];
	double R_302[1];
	double R_012[3];
	double R_112[2];
	double R_212[1];
	double R_022[2];
	double R_122[1];
	double R_032[1];
	double R_003[3];
	double R_103[2];
	double R_203[1];
	double R_013[2];
	double R_113[1];
	double R_023[1];
	double R_004[2];
	double R_104[1];
	double R_014[1];
	double R_005[1];
	for(int i=0;i<5;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<3;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<3;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<1;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<1;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<1;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<1;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<1;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
		double Pd_101[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_321[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=2*Pd_211[i]+Pd_010[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=Pd_010[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_321[i]=aPin1*Pd_211[i];
			}
	double P_021000000=Pd_021[0];
	double P_121000000=Pd_121[0];
	double P_221000000=Pd_221[0];
	double P_321000000=Pd_321[0];
	double P_020001000=Pd_020[0]*Pd_001[1];
	double P_020101000=Pd_020[0]*Pd_101[1];
	double P_120001000=Pd_120[0]*Pd_001[1];
	double P_120101000=Pd_120[0]*Pd_101[1];
	double P_220001000=Pd_220[0]*Pd_001[1];
	double P_220101000=Pd_220[0]*Pd_101[1];
	double P_020000001=Pd_020[0]*Pd_001[2];
	double P_020000101=Pd_020[0]*Pd_101[2];
	double P_120000001=Pd_120[0]*Pd_001[2];
	double P_120000101=Pd_120[0]*Pd_101[2];
	double P_220000001=Pd_220[0]*Pd_001[2];
	double P_220000101=Pd_220[0]*Pd_101[2];
	double P_011010000=Pd_011[0]*Pd_010[1];
	double P_011110000=Pd_011[0]*Pd_110[1];
	double P_111010000=Pd_111[0]*Pd_010[1];
	double P_111110000=Pd_111[0]*Pd_110[1];
	double P_211010000=Pd_211[0]*Pd_010[1];
	double P_211110000=Pd_211[0]*Pd_110[1];
	double P_010011000=Pd_010[0]*Pd_011[1];
	double P_010111000=Pd_010[0]*Pd_111[1];
	double P_010211000=Pd_010[0]*Pd_211[1];
	double P_110011000=Pd_110[0]*Pd_011[1];
	double P_110111000=Pd_110[0]*Pd_111[1];
	double P_110211000=Pd_110[0]*Pd_211[1];
	double P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
	double P_010010101=Pd_010[0]*Pd_010[1]*Pd_101[2];
	double P_010110001=Pd_010[0]*Pd_110[1]*Pd_001[2];
	double P_010110101=Pd_010[0]*Pd_110[1]*Pd_101[2];
	double P_110010001=Pd_110[0]*Pd_010[1]*Pd_001[2];
	double P_110010101=Pd_110[0]*Pd_010[1]*Pd_101[2];
	double P_110110001=Pd_110[0]*Pd_110[1]*Pd_001[2];
	double P_110110101=Pd_110[0]*Pd_110[1]*Pd_101[2];
	double P_001020000=Pd_001[0]*Pd_020[1];
	double P_001120000=Pd_001[0]*Pd_120[1];
	double P_001220000=Pd_001[0]*Pd_220[1];
	double P_101020000=Pd_101[0]*Pd_020[1];
	double P_101120000=Pd_101[0]*Pd_120[1];
	double P_101220000=Pd_101[0]*Pd_220[1];
	double P_000021000=Pd_021[1];
	double P_000121000=Pd_121[1];
	double P_000221000=Pd_221[1];
	double P_000321000=Pd_321[1];
	double P_000020001=Pd_020[1]*Pd_001[2];
	double P_000020101=Pd_020[1]*Pd_101[2];
	double P_000120001=Pd_120[1]*Pd_001[2];
	double P_000120101=Pd_120[1]*Pd_101[2];
	double P_000220001=Pd_220[1]*Pd_001[2];
	double P_000220101=Pd_220[1]*Pd_101[2];
	double P_011000010=Pd_011[0]*Pd_010[2];
	double P_011000110=Pd_011[0]*Pd_110[2];
	double P_111000010=Pd_111[0]*Pd_010[2];
	double P_111000110=Pd_111[0]*Pd_110[2];
	double P_211000010=Pd_211[0]*Pd_010[2];
	double P_211000110=Pd_211[0]*Pd_110[2];
	double P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
	double P_010001110=Pd_010[0]*Pd_001[1]*Pd_110[2];
	double P_010101010=Pd_010[0]*Pd_101[1]*Pd_010[2];
	double P_010101110=Pd_010[0]*Pd_101[1]*Pd_110[2];
	double P_110001010=Pd_110[0]*Pd_001[1]*Pd_010[2];
	double P_110001110=Pd_110[0]*Pd_001[1]*Pd_110[2];
	double P_110101010=Pd_110[0]*Pd_101[1]*Pd_010[2];
	double P_110101110=Pd_110[0]*Pd_101[1]*Pd_110[2];
	double P_010000011=Pd_010[0]*Pd_011[2];
	double P_010000111=Pd_010[0]*Pd_111[2];
	double P_010000211=Pd_010[0]*Pd_211[2];
	double P_110000011=Pd_110[0]*Pd_011[2];
	double P_110000111=Pd_110[0]*Pd_111[2];
	double P_110000211=Pd_110[0]*Pd_211[2];
	double P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
	double P_001010110=Pd_001[0]*Pd_010[1]*Pd_110[2];
	double P_001110010=Pd_001[0]*Pd_110[1]*Pd_010[2];
	double P_001110110=Pd_001[0]*Pd_110[1]*Pd_110[2];
	double P_101010010=Pd_101[0]*Pd_010[1]*Pd_010[2];
	double P_101010110=Pd_101[0]*Pd_010[1]*Pd_110[2];
	double P_101110010=Pd_101[0]*Pd_110[1]*Pd_010[2];
	double P_101110110=Pd_101[0]*Pd_110[1]*Pd_110[2];
	double P_000011010=Pd_011[1]*Pd_010[2];
	double P_000011110=Pd_011[1]*Pd_110[2];
	double P_000111010=Pd_111[1]*Pd_010[2];
	double P_000111110=Pd_111[1]*Pd_110[2];
	double P_000211010=Pd_211[1]*Pd_010[2];
	double P_000211110=Pd_211[1]*Pd_110[2];
	double P_000010011=Pd_010[1]*Pd_011[2];
	double P_000010111=Pd_010[1]*Pd_111[2];
	double P_000010211=Pd_010[1]*Pd_211[2];
	double P_000110011=Pd_110[1]*Pd_011[2];
	double P_000110111=Pd_110[1]*Pd_111[2];
	double P_000110211=Pd_110[1]*Pd_211[2];
	double P_001000020=Pd_001[0]*Pd_020[2];
	double P_001000120=Pd_001[0]*Pd_120[2];
	double P_001000220=Pd_001[0]*Pd_220[2];
	double P_101000020=Pd_101[0]*Pd_020[2];
	double P_101000120=Pd_101[0]*Pd_120[2];
	double P_101000220=Pd_101[0]*Pd_220[2];
	double P_000001020=Pd_001[1]*Pd_020[2];
	double P_000001120=Pd_001[1]*Pd_120[2];
	double P_000001220=Pd_001[1]*Pd_220[2];
	double P_000101020=Pd_101[1]*Pd_020[2];
	double P_000101120=Pd_101[1]*Pd_120[2];
	double P_000101220=Pd_101[1]*Pd_220[2];
	double P_000000021=Pd_021[2];
	double P_000000121=Pd_121[2];
	double P_000000221=Pd_221[2];
	double P_000000321=Pd_321[2];
				double PR_021000000000=P_021000000*R_000[0]+-1*P_121000000*R_100[0]+P_221000000*R_200[0]+-1*P_321000000*R_300[0];
				double PR_020001000000=P_020001000*R_000[0]+-1*P_020101000*R_010[0]+-1*P_120001000*R_100[0]+P_120101000*R_110[0]+P_220001000*R_200[0]+-1*P_220101000*R_210[0];
				double PR_020000001000=P_020000001*R_000[0]+-1*P_020000101*R_001[0]+-1*P_120000001*R_100[0]+P_120000101*R_101[0]+P_220000001*R_200[0]+-1*P_220000101*R_201[0];
				double PR_011010000000=P_011010000*R_000[0]+-1*P_011110000*R_010[0]+-1*P_111010000*R_100[0]+P_111110000*R_110[0]+P_211010000*R_200[0]+-1*P_211110000*R_210[0];
				double PR_010011000000=P_010011000*R_000[0]+-1*P_010111000*R_010[0]+P_010211000*R_020[0]+-1*P_110011000*R_100[0]+P_110111000*R_110[0]+-1*P_110211000*R_120[0];
				double PR_010010001000=P_010010001*R_000[0]+-1*P_010010101*R_001[0]+-1*P_010110001*R_010[0]+P_010110101*R_011[0]+-1*P_110010001*R_100[0]+P_110010101*R_101[0]+P_110110001*R_110[0]+-1*P_110110101*R_111[0];
				double PR_001020000000=P_001020000*R_000[0]+-1*P_001120000*R_010[0]+P_001220000*R_020[0]+-1*P_101020000*R_100[0]+P_101120000*R_110[0]+-1*P_101220000*R_120[0];
				double PR_000021000000=P_000021000*R_000[0]+-1*P_000121000*R_010[0]+P_000221000*R_020[0]+-1*P_000321000*R_030[0];
				double PR_000020001000=P_000020001*R_000[0]+-1*P_000020101*R_001[0]+-1*P_000120001*R_010[0]+P_000120101*R_011[0]+P_000220001*R_020[0]+-1*P_000220101*R_021[0];
				double PR_011000010000=P_011000010*R_000[0]+-1*P_011000110*R_001[0]+-1*P_111000010*R_100[0]+P_111000110*R_101[0]+P_211000010*R_200[0]+-1*P_211000110*R_201[0];
				double PR_010001010000=P_010001010*R_000[0]+-1*P_010001110*R_001[0]+-1*P_010101010*R_010[0]+P_010101110*R_011[0]+-1*P_110001010*R_100[0]+P_110001110*R_101[0]+P_110101010*R_110[0]+-1*P_110101110*R_111[0];
				double PR_010000011000=P_010000011*R_000[0]+-1*P_010000111*R_001[0]+P_010000211*R_002[0]+-1*P_110000011*R_100[0]+P_110000111*R_101[0]+-1*P_110000211*R_102[0];
				double PR_001010010000=P_001010010*R_000[0]+-1*P_001010110*R_001[0]+-1*P_001110010*R_010[0]+P_001110110*R_011[0]+-1*P_101010010*R_100[0]+P_101010110*R_101[0]+P_101110010*R_110[0]+-1*P_101110110*R_111[0];
				double PR_000011010000=P_000011010*R_000[0]+-1*P_000011110*R_001[0]+-1*P_000111010*R_010[0]+P_000111110*R_011[0]+P_000211010*R_020[0]+-1*P_000211110*R_021[0];
				double PR_000010011000=P_000010011*R_000[0]+-1*P_000010111*R_001[0]+P_000010211*R_002[0]+-1*P_000110011*R_010[0]+P_000110111*R_011[0]+-1*P_000110211*R_012[0];
				double PR_001000020000=P_001000020*R_000[0]+-1*P_001000120*R_001[0]+P_001000220*R_002[0]+-1*P_101000020*R_100[0]+P_101000120*R_101[0]+-1*P_101000220*R_102[0];
				double PR_000001020000=P_000001020*R_000[0]+-1*P_000001120*R_001[0]+P_000001220*R_002[0]+-1*P_000101020*R_010[0]+P_000101120*R_011[0]+-1*P_000101220*R_012[0];
				double PR_000000021000=P_000000021*R_000[0]+-1*P_000000121*R_001[0]+P_000000221*R_002[0]+-1*P_000000321*R_003[0];
				double PR_021000000001=P_021000000*R_001[0]+-1*P_121000000*R_101[0]+P_221000000*R_201[0]+-1*P_321000000*R_301[0];
				double PR_020001000001=P_020001000*R_001[0]+-1*P_020101000*R_011[0]+-1*P_120001000*R_101[0]+P_120101000*R_111[0]+P_220001000*R_201[0]+-1*P_220101000*R_211[0];
				double PR_020000001001=P_020000001*R_001[0]+-1*P_020000101*R_002[0]+-1*P_120000001*R_101[0]+P_120000101*R_102[0]+P_220000001*R_201[0]+-1*P_220000101*R_202[0];
				double PR_011010000001=P_011010000*R_001[0]+-1*P_011110000*R_011[0]+-1*P_111010000*R_101[0]+P_111110000*R_111[0]+P_211010000*R_201[0]+-1*P_211110000*R_211[0];
				double PR_010011000001=P_010011000*R_001[0]+-1*P_010111000*R_011[0]+P_010211000*R_021[0]+-1*P_110011000*R_101[0]+P_110111000*R_111[0]+-1*P_110211000*R_121[0];
				double PR_010010001001=P_010010001*R_001[0]+-1*P_010010101*R_002[0]+-1*P_010110001*R_011[0]+P_010110101*R_012[0]+-1*P_110010001*R_101[0]+P_110010101*R_102[0]+P_110110001*R_111[0]+-1*P_110110101*R_112[0];
				double PR_001020000001=P_001020000*R_001[0]+-1*P_001120000*R_011[0]+P_001220000*R_021[0]+-1*P_101020000*R_101[0]+P_101120000*R_111[0]+-1*P_101220000*R_121[0];
				double PR_000021000001=P_000021000*R_001[0]+-1*P_000121000*R_011[0]+P_000221000*R_021[0]+-1*P_000321000*R_031[0];
				double PR_000020001001=P_000020001*R_001[0]+-1*P_000020101*R_002[0]+-1*P_000120001*R_011[0]+P_000120101*R_012[0]+P_000220001*R_021[0]+-1*P_000220101*R_022[0];
				double PR_011000010001=P_011000010*R_001[0]+-1*P_011000110*R_002[0]+-1*P_111000010*R_101[0]+P_111000110*R_102[0]+P_211000010*R_201[0]+-1*P_211000110*R_202[0];
				double PR_010001010001=P_010001010*R_001[0]+-1*P_010001110*R_002[0]+-1*P_010101010*R_011[0]+P_010101110*R_012[0]+-1*P_110001010*R_101[0]+P_110001110*R_102[0]+P_110101010*R_111[0]+-1*P_110101110*R_112[0];
				double PR_010000011001=P_010000011*R_001[0]+-1*P_010000111*R_002[0]+P_010000211*R_003[0]+-1*P_110000011*R_101[0]+P_110000111*R_102[0]+-1*P_110000211*R_103[0];
				double PR_001010010001=P_001010010*R_001[0]+-1*P_001010110*R_002[0]+-1*P_001110010*R_011[0]+P_001110110*R_012[0]+-1*P_101010010*R_101[0]+P_101010110*R_102[0]+P_101110010*R_111[0]+-1*P_101110110*R_112[0];
				double PR_000011010001=P_000011010*R_001[0]+-1*P_000011110*R_002[0]+-1*P_000111010*R_011[0]+P_000111110*R_012[0]+P_000211010*R_021[0]+-1*P_000211110*R_022[0];
				double PR_000010011001=P_000010011*R_001[0]+-1*P_000010111*R_002[0]+P_000010211*R_003[0]+-1*P_000110011*R_011[0]+P_000110111*R_012[0]+-1*P_000110211*R_013[0];
				double PR_001000020001=P_001000020*R_001[0]+-1*P_001000120*R_002[0]+P_001000220*R_003[0]+-1*P_101000020*R_101[0]+P_101000120*R_102[0]+-1*P_101000220*R_103[0];
				double PR_000001020001=P_000001020*R_001[0]+-1*P_000001120*R_002[0]+P_000001220*R_003[0]+-1*P_000101020*R_011[0]+P_000101120*R_012[0]+-1*P_000101220*R_013[0];
				double PR_000000021001=P_000000021*R_001[0]+-1*P_000000121*R_002[0]+P_000000221*R_003[0]+-1*P_000000321*R_004[0];
				double PR_021000000010=P_021000000*R_010[0]+-1*P_121000000*R_110[0]+P_221000000*R_210[0]+-1*P_321000000*R_310[0];
				double PR_020001000010=P_020001000*R_010[0]+-1*P_020101000*R_020[0]+-1*P_120001000*R_110[0]+P_120101000*R_120[0]+P_220001000*R_210[0]+-1*P_220101000*R_220[0];
				double PR_020000001010=P_020000001*R_010[0]+-1*P_020000101*R_011[0]+-1*P_120000001*R_110[0]+P_120000101*R_111[0]+P_220000001*R_210[0]+-1*P_220000101*R_211[0];
				double PR_011010000010=P_011010000*R_010[0]+-1*P_011110000*R_020[0]+-1*P_111010000*R_110[0]+P_111110000*R_120[0]+P_211010000*R_210[0]+-1*P_211110000*R_220[0];
				double PR_010011000010=P_010011000*R_010[0]+-1*P_010111000*R_020[0]+P_010211000*R_030[0]+-1*P_110011000*R_110[0]+P_110111000*R_120[0]+-1*P_110211000*R_130[0];
				double PR_010010001010=P_010010001*R_010[0]+-1*P_010010101*R_011[0]+-1*P_010110001*R_020[0]+P_010110101*R_021[0]+-1*P_110010001*R_110[0]+P_110010101*R_111[0]+P_110110001*R_120[0]+-1*P_110110101*R_121[0];
				double PR_001020000010=P_001020000*R_010[0]+-1*P_001120000*R_020[0]+P_001220000*R_030[0]+-1*P_101020000*R_110[0]+P_101120000*R_120[0]+-1*P_101220000*R_130[0];
				double PR_000021000010=P_000021000*R_010[0]+-1*P_000121000*R_020[0]+P_000221000*R_030[0]+-1*P_000321000*R_040[0];
				double PR_000020001010=P_000020001*R_010[0]+-1*P_000020101*R_011[0]+-1*P_000120001*R_020[0]+P_000120101*R_021[0]+P_000220001*R_030[0]+-1*P_000220101*R_031[0];
				double PR_011000010010=P_011000010*R_010[0]+-1*P_011000110*R_011[0]+-1*P_111000010*R_110[0]+P_111000110*R_111[0]+P_211000010*R_210[0]+-1*P_211000110*R_211[0];
				double PR_010001010010=P_010001010*R_010[0]+-1*P_010001110*R_011[0]+-1*P_010101010*R_020[0]+P_010101110*R_021[0]+-1*P_110001010*R_110[0]+P_110001110*R_111[0]+P_110101010*R_120[0]+-1*P_110101110*R_121[0];
				double PR_010000011010=P_010000011*R_010[0]+-1*P_010000111*R_011[0]+P_010000211*R_012[0]+-1*P_110000011*R_110[0]+P_110000111*R_111[0]+-1*P_110000211*R_112[0];
				double PR_001010010010=P_001010010*R_010[0]+-1*P_001010110*R_011[0]+-1*P_001110010*R_020[0]+P_001110110*R_021[0]+-1*P_101010010*R_110[0]+P_101010110*R_111[0]+P_101110010*R_120[0]+-1*P_101110110*R_121[0];
				double PR_000011010010=P_000011010*R_010[0]+-1*P_000011110*R_011[0]+-1*P_000111010*R_020[0]+P_000111110*R_021[0]+P_000211010*R_030[0]+-1*P_000211110*R_031[0];
				double PR_000010011010=P_000010011*R_010[0]+-1*P_000010111*R_011[0]+P_000010211*R_012[0]+-1*P_000110011*R_020[0]+P_000110111*R_021[0]+-1*P_000110211*R_022[0];
				double PR_001000020010=P_001000020*R_010[0]+-1*P_001000120*R_011[0]+P_001000220*R_012[0]+-1*P_101000020*R_110[0]+P_101000120*R_111[0]+-1*P_101000220*R_112[0];
				double PR_000001020010=P_000001020*R_010[0]+-1*P_000001120*R_011[0]+P_000001220*R_012[0]+-1*P_000101020*R_020[0]+P_000101120*R_021[0]+-1*P_000101220*R_022[0];
				double PR_000000021010=P_000000021*R_010[0]+-1*P_000000121*R_011[0]+P_000000221*R_012[0]+-1*P_000000321*R_013[0];
				double PR_021000000100=P_021000000*R_100[0]+-1*P_121000000*R_200[0]+P_221000000*R_300[0]+-1*P_321000000*R_400[0];
				double PR_020001000100=P_020001000*R_100[0]+-1*P_020101000*R_110[0]+-1*P_120001000*R_200[0]+P_120101000*R_210[0]+P_220001000*R_300[0]+-1*P_220101000*R_310[0];
				double PR_020000001100=P_020000001*R_100[0]+-1*P_020000101*R_101[0]+-1*P_120000001*R_200[0]+P_120000101*R_201[0]+P_220000001*R_300[0]+-1*P_220000101*R_301[0];
				double PR_011010000100=P_011010000*R_100[0]+-1*P_011110000*R_110[0]+-1*P_111010000*R_200[0]+P_111110000*R_210[0]+P_211010000*R_300[0]+-1*P_211110000*R_310[0];
				double PR_010011000100=P_010011000*R_100[0]+-1*P_010111000*R_110[0]+P_010211000*R_120[0]+-1*P_110011000*R_200[0]+P_110111000*R_210[0]+-1*P_110211000*R_220[0];
				double PR_010010001100=P_010010001*R_100[0]+-1*P_010010101*R_101[0]+-1*P_010110001*R_110[0]+P_010110101*R_111[0]+-1*P_110010001*R_200[0]+P_110010101*R_201[0]+P_110110001*R_210[0]+-1*P_110110101*R_211[0];
				double PR_001020000100=P_001020000*R_100[0]+-1*P_001120000*R_110[0]+P_001220000*R_120[0]+-1*P_101020000*R_200[0]+P_101120000*R_210[0]+-1*P_101220000*R_220[0];
				double PR_000021000100=P_000021000*R_100[0]+-1*P_000121000*R_110[0]+P_000221000*R_120[0]+-1*P_000321000*R_130[0];
				double PR_000020001100=P_000020001*R_100[0]+-1*P_000020101*R_101[0]+-1*P_000120001*R_110[0]+P_000120101*R_111[0]+P_000220001*R_120[0]+-1*P_000220101*R_121[0];
				double PR_011000010100=P_011000010*R_100[0]+-1*P_011000110*R_101[0]+-1*P_111000010*R_200[0]+P_111000110*R_201[0]+P_211000010*R_300[0]+-1*P_211000110*R_301[0];
				double PR_010001010100=P_010001010*R_100[0]+-1*P_010001110*R_101[0]+-1*P_010101010*R_110[0]+P_010101110*R_111[0]+-1*P_110001010*R_200[0]+P_110001110*R_201[0]+P_110101010*R_210[0]+-1*P_110101110*R_211[0];
				double PR_010000011100=P_010000011*R_100[0]+-1*P_010000111*R_101[0]+P_010000211*R_102[0]+-1*P_110000011*R_200[0]+P_110000111*R_201[0]+-1*P_110000211*R_202[0];
				double PR_001010010100=P_001010010*R_100[0]+-1*P_001010110*R_101[0]+-1*P_001110010*R_110[0]+P_001110110*R_111[0]+-1*P_101010010*R_200[0]+P_101010110*R_201[0]+P_101110010*R_210[0]+-1*P_101110110*R_211[0];
				double PR_000011010100=P_000011010*R_100[0]+-1*P_000011110*R_101[0]+-1*P_000111010*R_110[0]+P_000111110*R_111[0]+P_000211010*R_120[0]+-1*P_000211110*R_121[0];
				double PR_000010011100=P_000010011*R_100[0]+-1*P_000010111*R_101[0]+P_000010211*R_102[0]+-1*P_000110011*R_110[0]+P_000110111*R_111[0]+-1*P_000110211*R_112[0];
				double PR_001000020100=P_001000020*R_100[0]+-1*P_001000120*R_101[0]+P_001000220*R_102[0]+-1*P_101000020*R_200[0]+P_101000120*R_201[0]+-1*P_101000220*R_202[0];
				double PR_000001020100=P_000001020*R_100[0]+-1*P_000001120*R_101[0]+P_000001220*R_102[0]+-1*P_000101020*R_110[0]+P_000101120*R_111[0]+-1*P_000101220*R_112[0];
				double PR_000000021100=P_000000021*R_100[0]+-1*P_000000121*R_101[0]+P_000000221*R_102[0]+-1*P_000000321*R_103[0];
				double PR_021000000002=P_021000000*R_002[0]+-1*P_121000000*R_102[0]+P_221000000*R_202[0]+-1*P_321000000*R_302[0];
				double PR_020001000002=P_020001000*R_002[0]+-1*P_020101000*R_012[0]+-1*P_120001000*R_102[0]+P_120101000*R_112[0]+P_220001000*R_202[0]+-1*P_220101000*R_212[0];
				double PR_020000001002=P_020000001*R_002[0]+-1*P_020000101*R_003[0]+-1*P_120000001*R_102[0]+P_120000101*R_103[0]+P_220000001*R_202[0]+-1*P_220000101*R_203[0];
				double PR_011010000002=P_011010000*R_002[0]+-1*P_011110000*R_012[0]+-1*P_111010000*R_102[0]+P_111110000*R_112[0]+P_211010000*R_202[0]+-1*P_211110000*R_212[0];
				double PR_010011000002=P_010011000*R_002[0]+-1*P_010111000*R_012[0]+P_010211000*R_022[0]+-1*P_110011000*R_102[0]+P_110111000*R_112[0]+-1*P_110211000*R_122[0];
				double PR_010010001002=P_010010001*R_002[0]+-1*P_010010101*R_003[0]+-1*P_010110001*R_012[0]+P_010110101*R_013[0]+-1*P_110010001*R_102[0]+P_110010101*R_103[0]+P_110110001*R_112[0]+-1*P_110110101*R_113[0];
				double PR_001020000002=P_001020000*R_002[0]+-1*P_001120000*R_012[0]+P_001220000*R_022[0]+-1*P_101020000*R_102[0]+P_101120000*R_112[0]+-1*P_101220000*R_122[0];
				double PR_000021000002=P_000021000*R_002[0]+-1*P_000121000*R_012[0]+P_000221000*R_022[0]+-1*P_000321000*R_032[0];
				double PR_000020001002=P_000020001*R_002[0]+-1*P_000020101*R_003[0]+-1*P_000120001*R_012[0]+P_000120101*R_013[0]+P_000220001*R_022[0]+-1*P_000220101*R_023[0];
				double PR_011000010002=P_011000010*R_002[0]+-1*P_011000110*R_003[0]+-1*P_111000010*R_102[0]+P_111000110*R_103[0]+P_211000010*R_202[0]+-1*P_211000110*R_203[0];
				double PR_010001010002=P_010001010*R_002[0]+-1*P_010001110*R_003[0]+-1*P_010101010*R_012[0]+P_010101110*R_013[0]+-1*P_110001010*R_102[0]+P_110001110*R_103[0]+P_110101010*R_112[0]+-1*P_110101110*R_113[0];
				double PR_010000011002=P_010000011*R_002[0]+-1*P_010000111*R_003[0]+P_010000211*R_004[0]+-1*P_110000011*R_102[0]+P_110000111*R_103[0]+-1*P_110000211*R_104[0];
				double PR_001010010002=P_001010010*R_002[0]+-1*P_001010110*R_003[0]+-1*P_001110010*R_012[0]+P_001110110*R_013[0]+-1*P_101010010*R_102[0]+P_101010110*R_103[0]+P_101110010*R_112[0]+-1*P_101110110*R_113[0];
				double PR_000011010002=P_000011010*R_002[0]+-1*P_000011110*R_003[0]+-1*P_000111010*R_012[0]+P_000111110*R_013[0]+P_000211010*R_022[0]+-1*P_000211110*R_023[0];
				double PR_000010011002=P_000010011*R_002[0]+-1*P_000010111*R_003[0]+P_000010211*R_004[0]+-1*P_000110011*R_012[0]+P_000110111*R_013[0]+-1*P_000110211*R_014[0];
				double PR_001000020002=P_001000020*R_002[0]+-1*P_001000120*R_003[0]+P_001000220*R_004[0]+-1*P_101000020*R_102[0]+P_101000120*R_103[0]+-1*P_101000220*R_104[0];
				double PR_000001020002=P_000001020*R_002[0]+-1*P_000001120*R_003[0]+P_000001220*R_004[0]+-1*P_000101020*R_012[0]+P_000101120*R_013[0]+-1*P_000101220*R_014[0];
				double PR_000000021002=P_000000021*R_002[0]+-1*P_000000121*R_003[0]+P_000000221*R_004[0]+-1*P_000000321*R_005[0];
				double PR_021000000011=P_021000000*R_011[0]+-1*P_121000000*R_111[0]+P_221000000*R_211[0]+-1*P_321000000*R_311[0];
				double PR_020001000011=P_020001000*R_011[0]+-1*P_020101000*R_021[0]+-1*P_120001000*R_111[0]+P_120101000*R_121[0]+P_220001000*R_211[0]+-1*P_220101000*R_221[0];
				double PR_020000001011=P_020000001*R_011[0]+-1*P_020000101*R_012[0]+-1*P_120000001*R_111[0]+P_120000101*R_112[0]+P_220000001*R_211[0]+-1*P_220000101*R_212[0];
				double PR_011010000011=P_011010000*R_011[0]+-1*P_011110000*R_021[0]+-1*P_111010000*R_111[0]+P_111110000*R_121[0]+P_211010000*R_211[0]+-1*P_211110000*R_221[0];
				double PR_010011000011=P_010011000*R_011[0]+-1*P_010111000*R_021[0]+P_010211000*R_031[0]+-1*P_110011000*R_111[0]+P_110111000*R_121[0]+-1*P_110211000*R_131[0];
				double PR_010010001011=P_010010001*R_011[0]+-1*P_010010101*R_012[0]+-1*P_010110001*R_021[0]+P_010110101*R_022[0]+-1*P_110010001*R_111[0]+P_110010101*R_112[0]+P_110110001*R_121[0]+-1*P_110110101*R_122[0];
				double PR_001020000011=P_001020000*R_011[0]+-1*P_001120000*R_021[0]+P_001220000*R_031[0]+-1*P_101020000*R_111[0]+P_101120000*R_121[0]+-1*P_101220000*R_131[0];
				double PR_000021000011=P_000021000*R_011[0]+-1*P_000121000*R_021[0]+P_000221000*R_031[0]+-1*P_000321000*R_041[0];
				double PR_000020001011=P_000020001*R_011[0]+-1*P_000020101*R_012[0]+-1*P_000120001*R_021[0]+P_000120101*R_022[0]+P_000220001*R_031[0]+-1*P_000220101*R_032[0];
				double PR_011000010011=P_011000010*R_011[0]+-1*P_011000110*R_012[0]+-1*P_111000010*R_111[0]+P_111000110*R_112[0]+P_211000010*R_211[0]+-1*P_211000110*R_212[0];
				double PR_010001010011=P_010001010*R_011[0]+-1*P_010001110*R_012[0]+-1*P_010101010*R_021[0]+P_010101110*R_022[0]+-1*P_110001010*R_111[0]+P_110001110*R_112[0]+P_110101010*R_121[0]+-1*P_110101110*R_122[0];
				double PR_010000011011=P_010000011*R_011[0]+-1*P_010000111*R_012[0]+P_010000211*R_013[0]+-1*P_110000011*R_111[0]+P_110000111*R_112[0]+-1*P_110000211*R_113[0];
				double PR_001010010011=P_001010010*R_011[0]+-1*P_001010110*R_012[0]+-1*P_001110010*R_021[0]+P_001110110*R_022[0]+-1*P_101010010*R_111[0]+P_101010110*R_112[0]+P_101110010*R_121[0]+-1*P_101110110*R_122[0];
				double PR_000011010011=P_000011010*R_011[0]+-1*P_000011110*R_012[0]+-1*P_000111010*R_021[0]+P_000111110*R_022[0]+P_000211010*R_031[0]+-1*P_000211110*R_032[0];
				double PR_000010011011=P_000010011*R_011[0]+-1*P_000010111*R_012[0]+P_000010211*R_013[0]+-1*P_000110011*R_021[0]+P_000110111*R_022[0]+-1*P_000110211*R_023[0];
				double PR_001000020011=P_001000020*R_011[0]+-1*P_001000120*R_012[0]+P_001000220*R_013[0]+-1*P_101000020*R_111[0]+P_101000120*R_112[0]+-1*P_101000220*R_113[0];
				double PR_000001020011=P_000001020*R_011[0]+-1*P_000001120*R_012[0]+P_000001220*R_013[0]+-1*P_000101020*R_021[0]+P_000101120*R_022[0]+-1*P_000101220*R_023[0];
				double PR_000000021011=P_000000021*R_011[0]+-1*P_000000121*R_012[0]+P_000000221*R_013[0]+-1*P_000000321*R_014[0];
				double PR_021000000020=P_021000000*R_020[0]+-1*P_121000000*R_120[0]+P_221000000*R_220[0]+-1*P_321000000*R_320[0];
				double PR_020001000020=P_020001000*R_020[0]+-1*P_020101000*R_030[0]+-1*P_120001000*R_120[0]+P_120101000*R_130[0]+P_220001000*R_220[0]+-1*P_220101000*R_230[0];
				double PR_020000001020=P_020000001*R_020[0]+-1*P_020000101*R_021[0]+-1*P_120000001*R_120[0]+P_120000101*R_121[0]+P_220000001*R_220[0]+-1*P_220000101*R_221[0];
				double PR_011010000020=P_011010000*R_020[0]+-1*P_011110000*R_030[0]+-1*P_111010000*R_120[0]+P_111110000*R_130[0]+P_211010000*R_220[0]+-1*P_211110000*R_230[0];
				double PR_010011000020=P_010011000*R_020[0]+-1*P_010111000*R_030[0]+P_010211000*R_040[0]+-1*P_110011000*R_120[0]+P_110111000*R_130[0]+-1*P_110211000*R_140[0];
				double PR_010010001020=P_010010001*R_020[0]+-1*P_010010101*R_021[0]+-1*P_010110001*R_030[0]+P_010110101*R_031[0]+-1*P_110010001*R_120[0]+P_110010101*R_121[0]+P_110110001*R_130[0]+-1*P_110110101*R_131[0];
				double PR_001020000020=P_001020000*R_020[0]+-1*P_001120000*R_030[0]+P_001220000*R_040[0]+-1*P_101020000*R_120[0]+P_101120000*R_130[0]+-1*P_101220000*R_140[0];
				double PR_000021000020=P_000021000*R_020[0]+-1*P_000121000*R_030[0]+P_000221000*R_040[0]+-1*P_000321000*R_050[0];
				double PR_000020001020=P_000020001*R_020[0]+-1*P_000020101*R_021[0]+-1*P_000120001*R_030[0]+P_000120101*R_031[0]+P_000220001*R_040[0]+-1*P_000220101*R_041[0];
				double PR_011000010020=P_011000010*R_020[0]+-1*P_011000110*R_021[0]+-1*P_111000010*R_120[0]+P_111000110*R_121[0]+P_211000010*R_220[0]+-1*P_211000110*R_221[0];
				double PR_010001010020=P_010001010*R_020[0]+-1*P_010001110*R_021[0]+-1*P_010101010*R_030[0]+P_010101110*R_031[0]+-1*P_110001010*R_120[0]+P_110001110*R_121[0]+P_110101010*R_130[0]+-1*P_110101110*R_131[0];
				double PR_010000011020=P_010000011*R_020[0]+-1*P_010000111*R_021[0]+P_010000211*R_022[0]+-1*P_110000011*R_120[0]+P_110000111*R_121[0]+-1*P_110000211*R_122[0];
				double PR_001010010020=P_001010010*R_020[0]+-1*P_001010110*R_021[0]+-1*P_001110010*R_030[0]+P_001110110*R_031[0]+-1*P_101010010*R_120[0]+P_101010110*R_121[0]+P_101110010*R_130[0]+-1*P_101110110*R_131[0];
				double PR_000011010020=P_000011010*R_020[0]+-1*P_000011110*R_021[0]+-1*P_000111010*R_030[0]+P_000111110*R_031[0]+P_000211010*R_040[0]+-1*P_000211110*R_041[0];
				double PR_000010011020=P_000010011*R_020[0]+-1*P_000010111*R_021[0]+P_000010211*R_022[0]+-1*P_000110011*R_030[0]+P_000110111*R_031[0]+-1*P_000110211*R_032[0];
				double PR_001000020020=P_001000020*R_020[0]+-1*P_001000120*R_021[0]+P_001000220*R_022[0]+-1*P_101000020*R_120[0]+P_101000120*R_121[0]+-1*P_101000220*R_122[0];
				double PR_000001020020=P_000001020*R_020[0]+-1*P_000001120*R_021[0]+P_000001220*R_022[0]+-1*P_000101020*R_030[0]+P_000101120*R_031[0]+-1*P_000101220*R_032[0];
				double PR_000000021020=P_000000021*R_020[0]+-1*P_000000121*R_021[0]+P_000000221*R_022[0]+-1*P_000000321*R_023[0];
				double PR_021000000101=P_021000000*R_101[0]+-1*P_121000000*R_201[0]+P_221000000*R_301[0]+-1*P_321000000*R_401[0];
				double PR_020001000101=P_020001000*R_101[0]+-1*P_020101000*R_111[0]+-1*P_120001000*R_201[0]+P_120101000*R_211[0]+P_220001000*R_301[0]+-1*P_220101000*R_311[0];
				double PR_020000001101=P_020000001*R_101[0]+-1*P_020000101*R_102[0]+-1*P_120000001*R_201[0]+P_120000101*R_202[0]+P_220000001*R_301[0]+-1*P_220000101*R_302[0];
				double PR_011010000101=P_011010000*R_101[0]+-1*P_011110000*R_111[0]+-1*P_111010000*R_201[0]+P_111110000*R_211[0]+P_211010000*R_301[0]+-1*P_211110000*R_311[0];
				double PR_010011000101=P_010011000*R_101[0]+-1*P_010111000*R_111[0]+P_010211000*R_121[0]+-1*P_110011000*R_201[0]+P_110111000*R_211[0]+-1*P_110211000*R_221[0];
				double PR_010010001101=P_010010001*R_101[0]+-1*P_010010101*R_102[0]+-1*P_010110001*R_111[0]+P_010110101*R_112[0]+-1*P_110010001*R_201[0]+P_110010101*R_202[0]+P_110110001*R_211[0]+-1*P_110110101*R_212[0];
				double PR_001020000101=P_001020000*R_101[0]+-1*P_001120000*R_111[0]+P_001220000*R_121[0]+-1*P_101020000*R_201[0]+P_101120000*R_211[0]+-1*P_101220000*R_221[0];
				double PR_000021000101=P_000021000*R_101[0]+-1*P_000121000*R_111[0]+P_000221000*R_121[0]+-1*P_000321000*R_131[0];
				double PR_000020001101=P_000020001*R_101[0]+-1*P_000020101*R_102[0]+-1*P_000120001*R_111[0]+P_000120101*R_112[0]+P_000220001*R_121[0]+-1*P_000220101*R_122[0];
				double PR_011000010101=P_011000010*R_101[0]+-1*P_011000110*R_102[0]+-1*P_111000010*R_201[0]+P_111000110*R_202[0]+P_211000010*R_301[0]+-1*P_211000110*R_302[0];
				double PR_010001010101=P_010001010*R_101[0]+-1*P_010001110*R_102[0]+-1*P_010101010*R_111[0]+P_010101110*R_112[0]+-1*P_110001010*R_201[0]+P_110001110*R_202[0]+P_110101010*R_211[0]+-1*P_110101110*R_212[0];
				double PR_010000011101=P_010000011*R_101[0]+-1*P_010000111*R_102[0]+P_010000211*R_103[0]+-1*P_110000011*R_201[0]+P_110000111*R_202[0]+-1*P_110000211*R_203[0];
				double PR_001010010101=P_001010010*R_101[0]+-1*P_001010110*R_102[0]+-1*P_001110010*R_111[0]+P_001110110*R_112[0]+-1*P_101010010*R_201[0]+P_101010110*R_202[0]+P_101110010*R_211[0]+-1*P_101110110*R_212[0];
				double PR_000011010101=P_000011010*R_101[0]+-1*P_000011110*R_102[0]+-1*P_000111010*R_111[0]+P_000111110*R_112[0]+P_000211010*R_121[0]+-1*P_000211110*R_122[0];
				double PR_000010011101=P_000010011*R_101[0]+-1*P_000010111*R_102[0]+P_000010211*R_103[0]+-1*P_000110011*R_111[0]+P_000110111*R_112[0]+-1*P_000110211*R_113[0];
				double PR_001000020101=P_001000020*R_101[0]+-1*P_001000120*R_102[0]+P_001000220*R_103[0]+-1*P_101000020*R_201[0]+P_101000120*R_202[0]+-1*P_101000220*R_203[0];
				double PR_000001020101=P_000001020*R_101[0]+-1*P_000001120*R_102[0]+P_000001220*R_103[0]+-1*P_000101020*R_111[0]+P_000101120*R_112[0]+-1*P_000101220*R_113[0];
				double PR_000000021101=P_000000021*R_101[0]+-1*P_000000121*R_102[0]+P_000000221*R_103[0]+-1*P_000000321*R_104[0];
				double PR_021000000110=P_021000000*R_110[0]+-1*P_121000000*R_210[0]+P_221000000*R_310[0]+-1*P_321000000*R_410[0];
				double PR_020001000110=P_020001000*R_110[0]+-1*P_020101000*R_120[0]+-1*P_120001000*R_210[0]+P_120101000*R_220[0]+P_220001000*R_310[0]+-1*P_220101000*R_320[0];
				double PR_020000001110=P_020000001*R_110[0]+-1*P_020000101*R_111[0]+-1*P_120000001*R_210[0]+P_120000101*R_211[0]+P_220000001*R_310[0]+-1*P_220000101*R_311[0];
				double PR_011010000110=P_011010000*R_110[0]+-1*P_011110000*R_120[0]+-1*P_111010000*R_210[0]+P_111110000*R_220[0]+P_211010000*R_310[0]+-1*P_211110000*R_320[0];
				double PR_010011000110=P_010011000*R_110[0]+-1*P_010111000*R_120[0]+P_010211000*R_130[0]+-1*P_110011000*R_210[0]+P_110111000*R_220[0]+-1*P_110211000*R_230[0];
				double PR_010010001110=P_010010001*R_110[0]+-1*P_010010101*R_111[0]+-1*P_010110001*R_120[0]+P_010110101*R_121[0]+-1*P_110010001*R_210[0]+P_110010101*R_211[0]+P_110110001*R_220[0]+-1*P_110110101*R_221[0];
				double PR_001020000110=P_001020000*R_110[0]+-1*P_001120000*R_120[0]+P_001220000*R_130[0]+-1*P_101020000*R_210[0]+P_101120000*R_220[0]+-1*P_101220000*R_230[0];
				double PR_000021000110=P_000021000*R_110[0]+-1*P_000121000*R_120[0]+P_000221000*R_130[0]+-1*P_000321000*R_140[0];
				double PR_000020001110=P_000020001*R_110[0]+-1*P_000020101*R_111[0]+-1*P_000120001*R_120[0]+P_000120101*R_121[0]+P_000220001*R_130[0]+-1*P_000220101*R_131[0];
				double PR_011000010110=P_011000010*R_110[0]+-1*P_011000110*R_111[0]+-1*P_111000010*R_210[0]+P_111000110*R_211[0]+P_211000010*R_310[0]+-1*P_211000110*R_311[0];
				double PR_010001010110=P_010001010*R_110[0]+-1*P_010001110*R_111[0]+-1*P_010101010*R_120[0]+P_010101110*R_121[0]+-1*P_110001010*R_210[0]+P_110001110*R_211[0]+P_110101010*R_220[0]+-1*P_110101110*R_221[0];
				double PR_010000011110=P_010000011*R_110[0]+-1*P_010000111*R_111[0]+P_010000211*R_112[0]+-1*P_110000011*R_210[0]+P_110000111*R_211[0]+-1*P_110000211*R_212[0];
				double PR_001010010110=P_001010010*R_110[0]+-1*P_001010110*R_111[0]+-1*P_001110010*R_120[0]+P_001110110*R_121[0]+-1*P_101010010*R_210[0]+P_101010110*R_211[0]+P_101110010*R_220[0]+-1*P_101110110*R_221[0];
				double PR_000011010110=P_000011010*R_110[0]+-1*P_000011110*R_111[0]+-1*P_000111010*R_120[0]+P_000111110*R_121[0]+P_000211010*R_130[0]+-1*P_000211110*R_131[0];
				double PR_000010011110=P_000010011*R_110[0]+-1*P_000010111*R_111[0]+P_000010211*R_112[0]+-1*P_000110011*R_120[0]+P_000110111*R_121[0]+-1*P_000110211*R_122[0];
				double PR_001000020110=P_001000020*R_110[0]+-1*P_001000120*R_111[0]+P_001000220*R_112[0]+-1*P_101000020*R_210[0]+P_101000120*R_211[0]+-1*P_101000220*R_212[0];
				double PR_000001020110=P_000001020*R_110[0]+-1*P_000001120*R_111[0]+P_000001220*R_112[0]+-1*P_000101020*R_120[0]+P_000101120*R_121[0]+-1*P_000101220*R_122[0];
				double PR_000000021110=P_000000021*R_110[0]+-1*P_000000121*R_111[0]+P_000000221*R_112[0]+-1*P_000000321*R_113[0];
				double PR_021000000200=P_021000000*R_200[0]+-1*P_121000000*R_300[0]+P_221000000*R_400[0]+-1*P_321000000*R_500[0];
				double PR_020001000200=P_020001000*R_200[0]+-1*P_020101000*R_210[0]+-1*P_120001000*R_300[0]+P_120101000*R_310[0]+P_220001000*R_400[0]+-1*P_220101000*R_410[0];
				double PR_020000001200=P_020000001*R_200[0]+-1*P_020000101*R_201[0]+-1*P_120000001*R_300[0]+P_120000101*R_301[0]+P_220000001*R_400[0]+-1*P_220000101*R_401[0];
				double PR_011010000200=P_011010000*R_200[0]+-1*P_011110000*R_210[0]+-1*P_111010000*R_300[0]+P_111110000*R_310[0]+P_211010000*R_400[0]+-1*P_211110000*R_410[0];
				double PR_010011000200=P_010011000*R_200[0]+-1*P_010111000*R_210[0]+P_010211000*R_220[0]+-1*P_110011000*R_300[0]+P_110111000*R_310[0]+-1*P_110211000*R_320[0];
				double PR_010010001200=P_010010001*R_200[0]+-1*P_010010101*R_201[0]+-1*P_010110001*R_210[0]+P_010110101*R_211[0]+-1*P_110010001*R_300[0]+P_110010101*R_301[0]+P_110110001*R_310[0]+-1*P_110110101*R_311[0];
				double PR_001020000200=P_001020000*R_200[0]+-1*P_001120000*R_210[0]+P_001220000*R_220[0]+-1*P_101020000*R_300[0]+P_101120000*R_310[0]+-1*P_101220000*R_320[0];
				double PR_000021000200=P_000021000*R_200[0]+-1*P_000121000*R_210[0]+P_000221000*R_220[0]+-1*P_000321000*R_230[0];
				double PR_000020001200=P_000020001*R_200[0]+-1*P_000020101*R_201[0]+-1*P_000120001*R_210[0]+P_000120101*R_211[0]+P_000220001*R_220[0]+-1*P_000220101*R_221[0];
				double PR_011000010200=P_011000010*R_200[0]+-1*P_011000110*R_201[0]+-1*P_111000010*R_300[0]+P_111000110*R_301[0]+P_211000010*R_400[0]+-1*P_211000110*R_401[0];
				double PR_010001010200=P_010001010*R_200[0]+-1*P_010001110*R_201[0]+-1*P_010101010*R_210[0]+P_010101110*R_211[0]+-1*P_110001010*R_300[0]+P_110001110*R_301[0]+P_110101010*R_310[0]+-1*P_110101110*R_311[0];
				double PR_010000011200=P_010000011*R_200[0]+-1*P_010000111*R_201[0]+P_010000211*R_202[0]+-1*P_110000011*R_300[0]+P_110000111*R_301[0]+-1*P_110000211*R_302[0];
				double PR_001010010200=P_001010010*R_200[0]+-1*P_001010110*R_201[0]+-1*P_001110010*R_210[0]+P_001110110*R_211[0]+-1*P_101010010*R_300[0]+P_101010110*R_301[0]+P_101110010*R_310[0]+-1*P_101110110*R_311[0];
				double PR_000011010200=P_000011010*R_200[0]+-1*P_000011110*R_201[0]+-1*P_000111010*R_210[0]+P_000111110*R_211[0]+P_000211010*R_220[0]+-1*P_000211110*R_221[0];
				double PR_000010011200=P_000010011*R_200[0]+-1*P_000010111*R_201[0]+P_000010211*R_202[0]+-1*P_000110011*R_210[0]+P_000110111*R_211[0]+-1*P_000110211*R_212[0];
				double PR_001000020200=P_001000020*R_200[0]+-1*P_001000120*R_201[0]+P_001000220*R_202[0]+-1*P_101000020*R_300[0]+P_101000120*R_301[0]+-1*P_101000220*R_302[0];
				double PR_000001020200=P_000001020*R_200[0]+-1*P_000001120*R_201[0]+P_000001220*R_202[0]+-1*P_000101020*R_210[0]+P_000101120*R_211[0]+-1*P_000101220*R_212[0];
				double PR_000000021200=P_000000021*R_200[0]+-1*P_000000121*R_201[0]+P_000000221*R_202[0]+-1*P_000000321*R_203[0];
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(Q_020000000*PR_021000000000+Q_120000000*PR_021000000100+Q_220000000*PR_021000000200);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(Q_010010000*PR_021000000000+Q_010110000*PR_021000000010+Q_110010000*PR_021000000100+Q_110110000*PR_021000000110);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(Q_000020000*PR_021000000000+Q_000120000*PR_021000000010+Q_000220000*PR_021000000020);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(Q_010000010*PR_021000000000+Q_010000110*PR_021000000001+Q_110000010*PR_021000000100+Q_110000110*PR_021000000101);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(Q_000010010*PR_021000000000+Q_000010110*PR_021000000001+Q_000110010*PR_021000000010+Q_000110110*PR_021000000011);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(Q_000000020*PR_021000000000+Q_000000120*PR_021000000001+Q_000000220*PR_021000000002);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(Q_020000000*PR_020001000000+Q_120000000*PR_020001000100+Q_220000000*PR_020001000200);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(Q_010010000*PR_020001000000+Q_010110000*PR_020001000010+Q_110010000*PR_020001000100+Q_110110000*PR_020001000110);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(Q_000020000*PR_020001000000+Q_000120000*PR_020001000010+Q_000220000*PR_020001000020);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(Q_010000010*PR_020001000000+Q_010000110*PR_020001000001+Q_110000010*PR_020001000100+Q_110000110*PR_020001000101);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(Q_000010010*PR_020001000000+Q_000010110*PR_020001000001+Q_000110010*PR_020001000010+Q_000110110*PR_020001000011);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(Q_000000020*PR_020001000000+Q_000000120*PR_020001000001+Q_000000220*PR_020001000002);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(Q_020000000*PR_020000001000+Q_120000000*PR_020000001100+Q_220000000*PR_020000001200);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(Q_010010000*PR_020000001000+Q_010110000*PR_020000001010+Q_110010000*PR_020000001100+Q_110110000*PR_020000001110);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(Q_000020000*PR_020000001000+Q_000120000*PR_020000001010+Q_000220000*PR_020000001020);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(Q_010000010*PR_020000001000+Q_010000110*PR_020000001001+Q_110000010*PR_020000001100+Q_110000110*PR_020000001101);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(Q_000010010*PR_020000001000+Q_000010110*PR_020000001001+Q_000110010*PR_020000001010+Q_000110110*PR_020000001011);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(Q_000000020*PR_020000001000+Q_000000120*PR_020000001001+Q_000000220*PR_020000001002);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(Q_020000000*PR_011010000000+Q_120000000*PR_011010000100+Q_220000000*PR_011010000200);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(Q_010010000*PR_011010000000+Q_010110000*PR_011010000010+Q_110010000*PR_011010000100+Q_110110000*PR_011010000110);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(Q_000020000*PR_011010000000+Q_000120000*PR_011010000010+Q_000220000*PR_011010000020);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(Q_010000010*PR_011010000000+Q_010000110*PR_011010000001+Q_110000010*PR_011010000100+Q_110000110*PR_011010000101);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(Q_000010010*PR_011010000000+Q_000010110*PR_011010000001+Q_000110010*PR_011010000010+Q_000110110*PR_011010000011);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(Q_000000020*PR_011010000000+Q_000000120*PR_011010000001+Q_000000220*PR_011010000002);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(Q_020000000*PR_010011000000+Q_120000000*PR_010011000100+Q_220000000*PR_010011000200);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(Q_010010000*PR_010011000000+Q_010110000*PR_010011000010+Q_110010000*PR_010011000100+Q_110110000*PR_010011000110);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(Q_000020000*PR_010011000000+Q_000120000*PR_010011000010+Q_000220000*PR_010011000020);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(Q_010000010*PR_010011000000+Q_010000110*PR_010011000001+Q_110000010*PR_010011000100+Q_110000110*PR_010011000101);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(Q_000010010*PR_010011000000+Q_000010110*PR_010011000001+Q_000110010*PR_010011000010+Q_000110110*PR_010011000011);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(Q_000000020*PR_010011000000+Q_000000120*PR_010011000001+Q_000000220*PR_010011000002);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(Q_020000000*PR_010010001000+Q_120000000*PR_010010001100+Q_220000000*PR_010010001200);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(Q_010010000*PR_010010001000+Q_010110000*PR_010010001010+Q_110010000*PR_010010001100+Q_110110000*PR_010010001110);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(Q_000020000*PR_010010001000+Q_000120000*PR_010010001010+Q_000220000*PR_010010001020);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(Q_010000010*PR_010010001000+Q_010000110*PR_010010001001+Q_110000010*PR_010010001100+Q_110000110*PR_010010001101);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(Q_000010010*PR_010010001000+Q_000010110*PR_010010001001+Q_000110010*PR_010010001010+Q_000110110*PR_010010001011);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(Q_000000020*PR_010010001000+Q_000000120*PR_010010001001+Q_000000220*PR_010010001002);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(Q_020000000*PR_001020000000+Q_120000000*PR_001020000100+Q_220000000*PR_001020000200);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(Q_010010000*PR_001020000000+Q_010110000*PR_001020000010+Q_110010000*PR_001020000100+Q_110110000*PR_001020000110);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(Q_000020000*PR_001020000000+Q_000120000*PR_001020000010+Q_000220000*PR_001020000020);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(Q_010000010*PR_001020000000+Q_010000110*PR_001020000001+Q_110000010*PR_001020000100+Q_110000110*PR_001020000101);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(Q_000010010*PR_001020000000+Q_000010110*PR_001020000001+Q_000110010*PR_001020000010+Q_000110110*PR_001020000011);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(Q_000000020*PR_001020000000+Q_000000120*PR_001020000001+Q_000000220*PR_001020000002);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(Q_020000000*PR_000021000000+Q_120000000*PR_000021000100+Q_220000000*PR_000021000200);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(Q_010010000*PR_000021000000+Q_010110000*PR_000021000010+Q_110010000*PR_000021000100+Q_110110000*PR_000021000110);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(Q_000020000*PR_000021000000+Q_000120000*PR_000021000010+Q_000220000*PR_000021000020);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(Q_010000010*PR_000021000000+Q_010000110*PR_000021000001+Q_110000010*PR_000021000100+Q_110000110*PR_000021000101);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(Q_000010010*PR_000021000000+Q_000010110*PR_000021000001+Q_000110010*PR_000021000010+Q_000110110*PR_000021000011);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(Q_000000020*PR_000021000000+Q_000000120*PR_000021000001+Q_000000220*PR_000021000002);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(Q_020000000*PR_000020001000+Q_120000000*PR_000020001100+Q_220000000*PR_000020001200);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(Q_010010000*PR_000020001000+Q_010110000*PR_000020001010+Q_110010000*PR_000020001100+Q_110110000*PR_000020001110);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(Q_000020000*PR_000020001000+Q_000120000*PR_000020001010+Q_000220000*PR_000020001020);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(Q_010000010*PR_000020001000+Q_010000110*PR_000020001001+Q_110000010*PR_000020001100+Q_110000110*PR_000020001101);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(Q_000010010*PR_000020001000+Q_000010110*PR_000020001001+Q_000110010*PR_000020001010+Q_000110110*PR_000020001011);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(Q_000000020*PR_000020001000+Q_000000120*PR_000020001001+Q_000000220*PR_000020001002);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(Q_020000000*PR_011000010000+Q_120000000*PR_011000010100+Q_220000000*PR_011000010200);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(Q_010010000*PR_011000010000+Q_010110000*PR_011000010010+Q_110010000*PR_011000010100+Q_110110000*PR_011000010110);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(Q_000020000*PR_011000010000+Q_000120000*PR_011000010010+Q_000220000*PR_011000010020);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(Q_010000010*PR_011000010000+Q_010000110*PR_011000010001+Q_110000010*PR_011000010100+Q_110000110*PR_011000010101);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(Q_000010010*PR_011000010000+Q_000010110*PR_011000010001+Q_000110010*PR_011000010010+Q_000110110*PR_011000010011);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(Q_000000020*PR_011000010000+Q_000000120*PR_011000010001+Q_000000220*PR_011000010002);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(Q_020000000*PR_010001010000+Q_120000000*PR_010001010100+Q_220000000*PR_010001010200);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(Q_010010000*PR_010001010000+Q_010110000*PR_010001010010+Q_110010000*PR_010001010100+Q_110110000*PR_010001010110);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(Q_000020000*PR_010001010000+Q_000120000*PR_010001010010+Q_000220000*PR_010001010020);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(Q_010000010*PR_010001010000+Q_010000110*PR_010001010001+Q_110000010*PR_010001010100+Q_110000110*PR_010001010101);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(Q_000010010*PR_010001010000+Q_000010110*PR_010001010001+Q_000110010*PR_010001010010+Q_000110110*PR_010001010011);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(Q_000000020*PR_010001010000+Q_000000120*PR_010001010001+Q_000000220*PR_010001010002);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(Q_020000000*PR_010000011000+Q_120000000*PR_010000011100+Q_220000000*PR_010000011200);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(Q_010010000*PR_010000011000+Q_010110000*PR_010000011010+Q_110010000*PR_010000011100+Q_110110000*PR_010000011110);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(Q_000020000*PR_010000011000+Q_000120000*PR_010000011010+Q_000220000*PR_010000011020);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(Q_010000010*PR_010000011000+Q_010000110*PR_010000011001+Q_110000010*PR_010000011100+Q_110000110*PR_010000011101);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(Q_000010010*PR_010000011000+Q_000010110*PR_010000011001+Q_000110010*PR_010000011010+Q_000110110*PR_010000011011);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(Q_000000020*PR_010000011000+Q_000000120*PR_010000011001+Q_000000220*PR_010000011002);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(Q_020000000*PR_001010010000+Q_120000000*PR_001010010100+Q_220000000*PR_001010010200);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(Q_010010000*PR_001010010000+Q_010110000*PR_001010010010+Q_110010000*PR_001010010100+Q_110110000*PR_001010010110);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(Q_000020000*PR_001010010000+Q_000120000*PR_001010010010+Q_000220000*PR_001010010020);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(Q_010000010*PR_001010010000+Q_010000110*PR_001010010001+Q_110000010*PR_001010010100+Q_110000110*PR_001010010101);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(Q_000010010*PR_001010010000+Q_000010110*PR_001010010001+Q_000110010*PR_001010010010+Q_000110110*PR_001010010011);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(Q_000000020*PR_001010010000+Q_000000120*PR_001010010001+Q_000000220*PR_001010010002);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(Q_020000000*PR_000011010000+Q_120000000*PR_000011010100+Q_220000000*PR_000011010200);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(Q_010010000*PR_000011010000+Q_010110000*PR_000011010010+Q_110010000*PR_000011010100+Q_110110000*PR_000011010110);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(Q_000020000*PR_000011010000+Q_000120000*PR_000011010010+Q_000220000*PR_000011010020);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(Q_010000010*PR_000011010000+Q_010000110*PR_000011010001+Q_110000010*PR_000011010100+Q_110000110*PR_000011010101);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(Q_000010010*PR_000011010000+Q_000010110*PR_000011010001+Q_000110010*PR_000011010010+Q_000110110*PR_000011010011);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(Q_000000020*PR_000011010000+Q_000000120*PR_000011010001+Q_000000220*PR_000011010002);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(Q_020000000*PR_000010011000+Q_120000000*PR_000010011100+Q_220000000*PR_000010011200);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(Q_010010000*PR_000010011000+Q_010110000*PR_000010011010+Q_110010000*PR_000010011100+Q_110110000*PR_000010011110);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(Q_000020000*PR_000010011000+Q_000120000*PR_000010011010+Q_000220000*PR_000010011020);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(Q_010000010*PR_000010011000+Q_010000110*PR_000010011001+Q_110000010*PR_000010011100+Q_110000110*PR_000010011101);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(Q_000010010*PR_000010011000+Q_000010110*PR_000010011001+Q_000110010*PR_000010011010+Q_000110110*PR_000010011011);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(Q_000000020*PR_000010011000+Q_000000120*PR_000010011001+Q_000000220*PR_000010011002);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(Q_020000000*PR_001000020000+Q_120000000*PR_001000020100+Q_220000000*PR_001000020200);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(Q_010010000*PR_001000020000+Q_010110000*PR_001000020010+Q_110010000*PR_001000020100+Q_110110000*PR_001000020110);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(Q_000020000*PR_001000020000+Q_000120000*PR_001000020010+Q_000220000*PR_001000020020);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(Q_010000010*PR_001000020000+Q_010000110*PR_001000020001+Q_110000010*PR_001000020100+Q_110000110*PR_001000020101);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(Q_000010010*PR_001000020000+Q_000010110*PR_001000020001+Q_000110010*PR_001000020010+Q_000110110*PR_001000020011);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(Q_000000020*PR_001000020000+Q_000000120*PR_001000020001+Q_000000220*PR_001000020002);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(Q_020000000*PR_000001020000+Q_120000000*PR_000001020100+Q_220000000*PR_000001020200);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(Q_010010000*PR_000001020000+Q_010110000*PR_000001020010+Q_110010000*PR_000001020100+Q_110110000*PR_000001020110);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(Q_000020000*PR_000001020000+Q_000120000*PR_000001020010+Q_000220000*PR_000001020020);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(Q_010000010*PR_000001020000+Q_010000110*PR_000001020001+Q_110000010*PR_000001020100+Q_110000110*PR_000001020101);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(Q_000010010*PR_000001020000+Q_000010110*PR_000001020001+Q_000110010*PR_000001020010+Q_000110110*PR_000001020011);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(Q_000000020*PR_000001020000+Q_000000120*PR_000001020001+Q_000000220*PR_000001020002);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(Q_020000000*PR_000000021000+Q_120000000*PR_000000021100+Q_220000000*PR_000000021200);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(Q_010010000*PR_000000021000+Q_010110000*PR_000000021010+Q_110010000*PR_000000021100+Q_110110000*PR_000000021110);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(Q_000020000*PR_000000021000+Q_000120000*PR_000000021010+Q_000220000*PR_000000021020);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(Q_010000010*PR_000000021000+Q_010000110*PR_000000021001+Q_110000010*PR_000000021100+Q_110000110*PR_000000021101);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(Q_000010010*PR_000000021000+Q_000010110*PR_000000021001+Q_000110010*PR_000000021010+Q_000110110*PR_000000021011);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(Q_000000020*PR_000000021000+Q_000000120*PR_000000021001+Q_000000220*PR_000000021002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_dpds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[3]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
            unsigned int id_bra=id_bra_in[ii];
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Pd_001[3];
				Pd_001[0]=PB[ii*3+0];
				Pd_001[1]=PB[ii*3+1];
				Pd_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
            float K2_p=K2_p_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_end-primit_ket_start;j+=tdis){
            unsigned int jj=primit_ket_start+j;
            unsigned int id_ket=tex1Dfetch(tex_id_ket,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Eta,jj);
            double Eta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pq,jj);
            double pq=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+0);
            double QX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+1);
            double QY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+2);
            double QZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Qd_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            Qd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            Qd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            Qd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[6];
                Ft_fs_5(5,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[5];
	double R_200[4];
	double R_300[3];
	double R_400[2];
	double R_500[1];
	double R_010[5];
	double R_110[4];
	double R_210[3];
	double R_310[2];
	double R_410[1];
	double R_020[4];
	double R_120[3];
	double R_220[2];
	double R_320[1];
	double R_030[3];
	double R_130[2];
	double R_230[1];
	double R_040[2];
	double R_140[1];
	double R_050[1];
	double R_001[5];
	double R_101[4];
	double R_201[3];
	double R_301[2];
	double R_401[1];
	double R_011[4];
	double R_111[3];
	double R_211[2];
	double R_311[1];
	double R_021[3];
	double R_121[2];
	double R_221[1];
	double R_031[2];
	double R_131[1];
	double R_041[1];
	double R_002[4];
	double R_102[3];
	double R_202[2];
	double R_302[1];
	double R_012[3];
	double R_112[2];
	double R_212[1];
	double R_022[2];
	double R_122[1];
	double R_032[1];
	double R_003[3];
	double R_103[2];
	double R_203[1];
	double R_013[2];
	double R_113[1];
	double R_023[1];
	double R_004[2];
	double R_104[1];
	double R_014[1];
	double R_005[1];
	for(int i=0;i<5;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<4;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<3;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<3;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<3;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<2;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<2;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<2;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<1;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<1;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<1;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<1;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<1;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<1;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<1;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<1;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<1;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<1;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<1;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
				double QR_020000000000=Q_020000000*R_000[0]+-1*Q_120000000*R_100[0]+Q_220000000*R_200[0];
				double QR_010010000000=Q_010010000*R_000[0]+-1*Q_010110000*R_010[0]+-1*Q_110010000*R_100[0]+Q_110110000*R_110[0];
				double QR_000020000000=Q_000020000*R_000[0]+-1*Q_000120000*R_010[0]+Q_000220000*R_020[0];
				double QR_010000010000=Q_010000010*R_000[0]+-1*Q_010000110*R_001[0]+-1*Q_110000010*R_100[0]+Q_110000110*R_101[0];
				double QR_000010010000=Q_000010010*R_000[0]+-1*Q_000010110*R_001[0]+-1*Q_000110010*R_010[0]+Q_000110110*R_011[0];
				double QR_000000020000=Q_000000020*R_000[0]+-1*Q_000000120*R_001[0]+Q_000000220*R_002[0];
				double QR_020000000001=Q_020000000*R_001[0]+-1*Q_120000000*R_101[0]+Q_220000000*R_201[0];
				double QR_010010000001=Q_010010000*R_001[0]+-1*Q_010110000*R_011[0]+-1*Q_110010000*R_101[0]+Q_110110000*R_111[0];
				double QR_000020000001=Q_000020000*R_001[0]+-1*Q_000120000*R_011[0]+Q_000220000*R_021[0];
				double QR_010000010001=Q_010000010*R_001[0]+-1*Q_010000110*R_002[0]+-1*Q_110000010*R_101[0]+Q_110000110*R_102[0];
				double QR_000010010001=Q_000010010*R_001[0]+-1*Q_000010110*R_002[0]+-1*Q_000110010*R_011[0]+Q_000110110*R_012[0];
				double QR_000000020001=Q_000000020*R_001[0]+-1*Q_000000120*R_002[0]+Q_000000220*R_003[0];
				double QR_020000000010=Q_020000000*R_010[0]+-1*Q_120000000*R_110[0]+Q_220000000*R_210[0];
				double QR_010010000010=Q_010010000*R_010[0]+-1*Q_010110000*R_020[0]+-1*Q_110010000*R_110[0]+Q_110110000*R_120[0];
				double QR_000020000010=Q_000020000*R_010[0]+-1*Q_000120000*R_020[0]+Q_000220000*R_030[0];
				double QR_010000010010=Q_010000010*R_010[0]+-1*Q_010000110*R_011[0]+-1*Q_110000010*R_110[0]+Q_110000110*R_111[0];
				double QR_000010010010=Q_000010010*R_010[0]+-1*Q_000010110*R_011[0]+-1*Q_000110010*R_020[0]+Q_000110110*R_021[0];
				double QR_000000020010=Q_000000020*R_010[0]+-1*Q_000000120*R_011[0]+Q_000000220*R_012[0];
				double QR_020000000100=Q_020000000*R_100[0]+-1*Q_120000000*R_200[0]+Q_220000000*R_300[0];
				double QR_010010000100=Q_010010000*R_100[0]+-1*Q_010110000*R_110[0]+-1*Q_110010000*R_200[0]+Q_110110000*R_210[0];
				double QR_000020000100=Q_000020000*R_100[0]+-1*Q_000120000*R_110[0]+Q_000220000*R_120[0];
				double QR_010000010100=Q_010000010*R_100[0]+-1*Q_010000110*R_101[0]+-1*Q_110000010*R_200[0]+Q_110000110*R_201[0];
				double QR_000010010100=Q_000010010*R_100[0]+-1*Q_000010110*R_101[0]+-1*Q_000110010*R_110[0]+Q_000110110*R_111[0];
				double QR_000000020100=Q_000000020*R_100[0]+-1*Q_000000120*R_101[0]+Q_000000220*R_102[0];
				double QR_020000000002=Q_020000000*R_002[0]+-1*Q_120000000*R_102[0]+Q_220000000*R_202[0];
				double QR_010010000002=Q_010010000*R_002[0]+-1*Q_010110000*R_012[0]+-1*Q_110010000*R_102[0]+Q_110110000*R_112[0];
				double QR_000020000002=Q_000020000*R_002[0]+-1*Q_000120000*R_012[0]+Q_000220000*R_022[0];
				double QR_010000010002=Q_010000010*R_002[0]+-1*Q_010000110*R_003[0]+-1*Q_110000010*R_102[0]+Q_110000110*R_103[0];
				double QR_000010010002=Q_000010010*R_002[0]+-1*Q_000010110*R_003[0]+-1*Q_000110010*R_012[0]+Q_000110110*R_013[0];
				double QR_000000020002=Q_000000020*R_002[0]+-1*Q_000000120*R_003[0]+Q_000000220*R_004[0];
				double QR_020000000011=Q_020000000*R_011[0]+-1*Q_120000000*R_111[0]+Q_220000000*R_211[0];
				double QR_010010000011=Q_010010000*R_011[0]+-1*Q_010110000*R_021[0]+-1*Q_110010000*R_111[0]+Q_110110000*R_121[0];
				double QR_000020000011=Q_000020000*R_011[0]+-1*Q_000120000*R_021[0]+Q_000220000*R_031[0];
				double QR_010000010011=Q_010000010*R_011[0]+-1*Q_010000110*R_012[0]+-1*Q_110000010*R_111[0]+Q_110000110*R_112[0];
				double QR_000010010011=Q_000010010*R_011[0]+-1*Q_000010110*R_012[0]+-1*Q_000110010*R_021[0]+Q_000110110*R_022[0];
				double QR_000000020011=Q_000000020*R_011[0]+-1*Q_000000120*R_012[0]+Q_000000220*R_013[0];
				double QR_020000000020=Q_020000000*R_020[0]+-1*Q_120000000*R_120[0]+Q_220000000*R_220[0];
				double QR_010010000020=Q_010010000*R_020[0]+-1*Q_010110000*R_030[0]+-1*Q_110010000*R_120[0]+Q_110110000*R_130[0];
				double QR_000020000020=Q_000020000*R_020[0]+-1*Q_000120000*R_030[0]+Q_000220000*R_040[0];
				double QR_010000010020=Q_010000010*R_020[0]+-1*Q_010000110*R_021[0]+-1*Q_110000010*R_120[0]+Q_110000110*R_121[0];
				double QR_000010010020=Q_000010010*R_020[0]+-1*Q_000010110*R_021[0]+-1*Q_000110010*R_030[0]+Q_000110110*R_031[0];
				double QR_000000020020=Q_000000020*R_020[0]+-1*Q_000000120*R_021[0]+Q_000000220*R_022[0];
				double QR_020000000101=Q_020000000*R_101[0]+-1*Q_120000000*R_201[0]+Q_220000000*R_301[0];
				double QR_010010000101=Q_010010000*R_101[0]+-1*Q_010110000*R_111[0]+-1*Q_110010000*R_201[0]+Q_110110000*R_211[0];
				double QR_000020000101=Q_000020000*R_101[0]+-1*Q_000120000*R_111[0]+Q_000220000*R_121[0];
				double QR_010000010101=Q_010000010*R_101[0]+-1*Q_010000110*R_102[0]+-1*Q_110000010*R_201[0]+Q_110000110*R_202[0];
				double QR_000010010101=Q_000010010*R_101[0]+-1*Q_000010110*R_102[0]+-1*Q_000110010*R_111[0]+Q_000110110*R_112[0];
				double QR_000000020101=Q_000000020*R_101[0]+-1*Q_000000120*R_102[0]+Q_000000220*R_103[0];
				double QR_020000000110=Q_020000000*R_110[0]+-1*Q_120000000*R_210[0]+Q_220000000*R_310[0];
				double QR_010010000110=Q_010010000*R_110[0]+-1*Q_010110000*R_120[0]+-1*Q_110010000*R_210[0]+Q_110110000*R_220[0];
				double QR_000020000110=Q_000020000*R_110[0]+-1*Q_000120000*R_120[0]+Q_000220000*R_130[0];
				double QR_010000010110=Q_010000010*R_110[0]+-1*Q_010000110*R_111[0]+-1*Q_110000010*R_210[0]+Q_110000110*R_211[0];
				double QR_000010010110=Q_000010010*R_110[0]+-1*Q_000010110*R_111[0]+-1*Q_000110010*R_120[0]+Q_000110110*R_121[0];
				double QR_000000020110=Q_000000020*R_110[0]+-1*Q_000000120*R_111[0]+Q_000000220*R_112[0];
				double QR_020000000200=Q_020000000*R_200[0]+-1*Q_120000000*R_300[0]+Q_220000000*R_400[0];
				double QR_010010000200=Q_010010000*R_200[0]+-1*Q_010110000*R_210[0]+-1*Q_110010000*R_300[0]+Q_110110000*R_310[0];
				double QR_000020000200=Q_000020000*R_200[0]+-1*Q_000120000*R_210[0]+Q_000220000*R_220[0];
				double QR_010000010200=Q_010000010*R_200[0]+-1*Q_010000110*R_201[0]+-1*Q_110000010*R_300[0]+Q_110000110*R_301[0];
				double QR_000010010200=Q_000010010*R_200[0]+-1*Q_000010110*R_201[0]+-1*Q_000110010*R_210[0]+Q_000110110*R_211[0];
				double QR_000000020200=Q_000000020*R_200[0]+-1*Q_000000120*R_201[0]+Q_000000220*R_202[0];
				double QR_020000000003=Q_020000000*R_003[0]+-1*Q_120000000*R_103[0]+Q_220000000*R_203[0];
				double QR_010010000003=Q_010010000*R_003[0]+-1*Q_010110000*R_013[0]+-1*Q_110010000*R_103[0]+Q_110110000*R_113[0];
				double QR_000020000003=Q_000020000*R_003[0]+-1*Q_000120000*R_013[0]+Q_000220000*R_023[0];
				double QR_010000010003=Q_010000010*R_003[0]+-1*Q_010000110*R_004[0]+-1*Q_110000010*R_103[0]+Q_110000110*R_104[0];
				double QR_000010010003=Q_000010010*R_003[0]+-1*Q_000010110*R_004[0]+-1*Q_000110010*R_013[0]+Q_000110110*R_014[0];
				double QR_000000020003=Q_000000020*R_003[0]+-1*Q_000000120*R_004[0]+Q_000000220*R_005[0];
				double QR_020000000012=Q_020000000*R_012[0]+-1*Q_120000000*R_112[0]+Q_220000000*R_212[0];
				double QR_010010000012=Q_010010000*R_012[0]+-1*Q_010110000*R_022[0]+-1*Q_110010000*R_112[0]+Q_110110000*R_122[0];
				double QR_000020000012=Q_000020000*R_012[0]+-1*Q_000120000*R_022[0]+Q_000220000*R_032[0];
				double QR_010000010012=Q_010000010*R_012[0]+-1*Q_010000110*R_013[0]+-1*Q_110000010*R_112[0]+Q_110000110*R_113[0];
				double QR_000010010012=Q_000010010*R_012[0]+-1*Q_000010110*R_013[0]+-1*Q_000110010*R_022[0]+Q_000110110*R_023[0];
				double QR_000000020012=Q_000000020*R_012[0]+-1*Q_000000120*R_013[0]+Q_000000220*R_014[0];
				double QR_020000000021=Q_020000000*R_021[0]+-1*Q_120000000*R_121[0]+Q_220000000*R_221[0];
				double QR_010010000021=Q_010010000*R_021[0]+-1*Q_010110000*R_031[0]+-1*Q_110010000*R_121[0]+Q_110110000*R_131[0];
				double QR_000020000021=Q_000020000*R_021[0]+-1*Q_000120000*R_031[0]+Q_000220000*R_041[0];
				double QR_010000010021=Q_010000010*R_021[0]+-1*Q_010000110*R_022[0]+-1*Q_110000010*R_121[0]+Q_110000110*R_122[0];
				double QR_000010010021=Q_000010010*R_021[0]+-1*Q_000010110*R_022[0]+-1*Q_000110010*R_031[0]+Q_000110110*R_032[0];
				double QR_000000020021=Q_000000020*R_021[0]+-1*Q_000000120*R_022[0]+Q_000000220*R_023[0];
				double QR_020000000030=Q_020000000*R_030[0]+-1*Q_120000000*R_130[0]+Q_220000000*R_230[0];
				double QR_010010000030=Q_010010000*R_030[0]+-1*Q_010110000*R_040[0]+-1*Q_110010000*R_130[0]+Q_110110000*R_140[0];
				double QR_000020000030=Q_000020000*R_030[0]+-1*Q_000120000*R_040[0]+Q_000220000*R_050[0];
				double QR_010000010030=Q_010000010*R_030[0]+-1*Q_010000110*R_031[0]+-1*Q_110000010*R_130[0]+Q_110000110*R_131[0];
				double QR_000010010030=Q_000010010*R_030[0]+-1*Q_000010110*R_031[0]+-1*Q_000110010*R_040[0]+Q_000110110*R_041[0];
				double QR_000000020030=Q_000000020*R_030[0]+-1*Q_000000120*R_031[0]+Q_000000220*R_032[0];
				double QR_020000000102=Q_020000000*R_102[0]+-1*Q_120000000*R_202[0]+Q_220000000*R_302[0];
				double QR_010010000102=Q_010010000*R_102[0]+-1*Q_010110000*R_112[0]+-1*Q_110010000*R_202[0]+Q_110110000*R_212[0];
				double QR_000020000102=Q_000020000*R_102[0]+-1*Q_000120000*R_112[0]+Q_000220000*R_122[0];
				double QR_010000010102=Q_010000010*R_102[0]+-1*Q_010000110*R_103[0]+-1*Q_110000010*R_202[0]+Q_110000110*R_203[0];
				double QR_000010010102=Q_000010010*R_102[0]+-1*Q_000010110*R_103[0]+-1*Q_000110010*R_112[0]+Q_000110110*R_113[0];
				double QR_000000020102=Q_000000020*R_102[0]+-1*Q_000000120*R_103[0]+Q_000000220*R_104[0];
				double QR_020000000111=Q_020000000*R_111[0]+-1*Q_120000000*R_211[0]+Q_220000000*R_311[0];
				double QR_010010000111=Q_010010000*R_111[0]+-1*Q_010110000*R_121[0]+-1*Q_110010000*R_211[0]+Q_110110000*R_221[0];
				double QR_000020000111=Q_000020000*R_111[0]+-1*Q_000120000*R_121[0]+Q_000220000*R_131[0];
				double QR_010000010111=Q_010000010*R_111[0]+-1*Q_010000110*R_112[0]+-1*Q_110000010*R_211[0]+Q_110000110*R_212[0];
				double QR_000010010111=Q_000010010*R_111[0]+-1*Q_000010110*R_112[0]+-1*Q_000110010*R_121[0]+Q_000110110*R_122[0];
				double QR_000000020111=Q_000000020*R_111[0]+-1*Q_000000120*R_112[0]+Q_000000220*R_113[0];
				double QR_020000000120=Q_020000000*R_120[0]+-1*Q_120000000*R_220[0]+Q_220000000*R_320[0];
				double QR_010010000120=Q_010010000*R_120[0]+-1*Q_010110000*R_130[0]+-1*Q_110010000*R_220[0]+Q_110110000*R_230[0];
				double QR_000020000120=Q_000020000*R_120[0]+-1*Q_000120000*R_130[0]+Q_000220000*R_140[0];
				double QR_010000010120=Q_010000010*R_120[0]+-1*Q_010000110*R_121[0]+-1*Q_110000010*R_220[0]+Q_110000110*R_221[0];
				double QR_000010010120=Q_000010010*R_120[0]+-1*Q_000010110*R_121[0]+-1*Q_000110010*R_130[0]+Q_000110110*R_131[0];
				double QR_000000020120=Q_000000020*R_120[0]+-1*Q_000000120*R_121[0]+Q_000000220*R_122[0];
				double QR_020000000201=Q_020000000*R_201[0]+-1*Q_120000000*R_301[0]+Q_220000000*R_401[0];
				double QR_010010000201=Q_010010000*R_201[0]+-1*Q_010110000*R_211[0]+-1*Q_110010000*R_301[0]+Q_110110000*R_311[0];
				double QR_000020000201=Q_000020000*R_201[0]+-1*Q_000120000*R_211[0]+Q_000220000*R_221[0];
				double QR_010000010201=Q_010000010*R_201[0]+-1*Q_010000110*R_202[0]+-1*Q_110000010*R_301[0]+Q_110000110*R_302[0];
				double QR_000010010201=Q_000010010*R_201[0]+-1*Q_000010110*R_202[0]+-1*Q_000110010*R_211[0]+Q_000110110*R_212[0];
				double QR_000000020201=Q_000000020*R_201[0]+-1*Q_000000120*R_202[0]+Q_000000220*R_203[0];
				double QR_020000000210=Q_020000000*R_210[0]+-1*Q_120000000*R_310[0]+Q_220000000*R_410[0];
				double QR_010010000210=Q_010010000*R_210[0]+-1*Q_010110000*R_220[0]+-1*Q_110010000*R_310[0]+Q_110110000*R_320[0];
				double QR_000020000210=Q_000020000*R_210[0]+-1*Q_000120000*R_220[0]+Q_000220000*R_230[0];
				double QR_010000010210=Q_010000010*R_210[0]+-1*Q_010000110*R_211[0]+-1*Q_110000010*R_310[0]+Q_110000110*R_311[0];
				double QR_000010010210=Q_000010010*R_210[0]+-1*Q_000010110*R_211[0]+-1*Q_000110010*R_220[0]+Q_000110110*R_221[0];
				double QR_000000020210=Q_000000020*R_210[0]+-1*Q_000000120*R_211[0]+Q_000000220*R_212[0];
				double QR_020000000300=Q_020000000*R_300[0]+-1*Q_120000000*R_400[0]+Q_220000000*R_500[0];
				double QR_010010000300=Q_010010000*R_300[0]+-1*Q_010110000*R_310[0]+-1*Q_110010000*R_400[0]+Q_110110000*R_410[0];
				double QR_000020000300=Q_000020000*R_300[0]+-1*Q_000120000*R_310[0]+Q_000220000*R_320[0];
				double QR_010000010300=Q_010000010*R_300[0]+-1*Q_010000110*R_301[0]+-1*Q_110000010*R_400[0]+Q_110000110*R_401[0];
				double QR_000010010300=Q_000010010*R_300[0]+-1*Q_000010110*R_301[0]+-1*Q_000110010*R_310[0]+Q_000110110*R_311[0];
				double QR_000000020300=Q_000000020*R_300[0]+-1*Q_000000120*R_301[0]+Q_000000220*R_302[0];
		double Pd_101[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_321[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=2*Pd_211[i]+Pd_010[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=Pd_010[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_321[i]=aPin1*Pd_211[i];
			}
	double P_021000000=Pd_021[0];
	double P_121000000=Pd_121[0];
	double P_221000000=Pd_221[0];
	double P_321000000=Pd_321[0];
	double P_020001000=Pd_020[0]*Pd_001[1];
	double P_020101000=Pd_020[0]*Pd_101[1];
	double P_120001000=Pd_120[0]*Pd_001[1];
	double P_120101000=Pd_120[0]*Pd_101[1];
	double P_220001000=Pd_220[0]*Pd_001[1];
	double P_220101000=Pd_220[0]*Pd_101[1];
	double P_020000001=Pd_020[0]*Pd_001[2];
	double P_020000101=Pd_020[0]*Pd_101[2];
	double P_120000001=Pd_120[0]*Pd_001[2];
	double P_120000101=Pd_120[0]*Pd_101[2];
	double P_220000001=Pd_220[0]*Pd_001[2];
	double P_220000101=Pd_220[0]*Pd_101[2];
	double P_011010000=Pd_011[0]*Pd_010[1];
	double P_011110000=Pd_011[0]*Pd_110[1];
	double P_111010000=Pd_111[0]*Pd_010[1];
	double P_111110000=Pd_111[0]*Pd_110[1];
	double P_211010000=Pd_211[0]*Pd_010[1];
	double P_211110000=Pd_211[0]*Pd_110[1];
	double P_010011000=Pd_010[0]*Pd_011[1];
	double P_010111000=Pd_010[0]*Pd_111[1];
	double P_010211000=Pd_010[0]*Pd_211[1];
	double P_110011000=Pd_110[0]*Pd_011[1];
	double P_110111000=Pd_110[0]*Pd_111[1];
	double P_110211000=Pd_110[0]*Pd_211[1];
	double P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
	double P_010010101=Pd_010[0]*Pd_010[1]*Pd_101[2];
	double P_010110001=Pd_010[0]*Pd_110[1]*Pd_001[2];
	double P_010110101=Pd_010[0]*Pd_110[1]*Pd_101[2];
	double P_110010001=Pd_110[0]*Pd_010[1]*Pd_001[2];
	double P_110010101=Pd_110[0]*Pd_010[1]*Pd_101[2];
	double P_110110001=Pd_110[0]*Pd_110[1]*Pd_001[2];
	double P_110110101=Pd_110[0]*Pd_110[1]*Pd_101[2];
	double P_001020000=Pd_001[0]*Pd_020[1];
	double P_001120000=Pd_001[0]*Pd_120[1];
	double P_001220000=Pd_001[0]*Pd_220[1];
	double P_101020000=Pd_101[0]*Pd_020[1];
	double P_101120000=Pd_101[0]*Pd_120[1];
	double P_101220000=Pd_101[0]*Pd_220[1];
	double P_000021000=Pd_021[1];
	double P_000121000=Pd_121[1];
	double P_000221000=Pd_221[1];
	double P_000321000=Pd_321[1];
	double P_000020001=Pd_020[1]*Pd_001[2];
	double P_000020101=Pd_020[1]*Pd_101[2];
	double P_000120001=Pd_120[1]*Pd_001[2];
	double P_000120101=Pd_120[1]*Pd_101[2];
	double P_000220001=Pd_220[1]*Pd_001[2];
	double P_000220101=Pd_220[1]*Pd_101[2];
	double P_011000010=Pd_011[0]*Pd_010[2];
	double P_011000110=Pd_011[0]*Pd_110[2];
	double P_111000010=Pd_111[0]*Pd_010[2];
	double P_111000110=Pd_111[0]*Pd_110[2];
	double P_211000010=Pd_211[0]*Pd_010[2];
	double P_211000110=Pd_211[0]*Pd_110[2];
	double P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
	double P_010001110=Pd_010[0]*Pd_001[1]*Pd_110[2];
	double P_010101010=Pd_010[0]*Pd_101[1]*Pd_010[2];
	double P_010101110=Pd_010[0]*Pd_101[1]*Pd_110[2];
	double P_110001010=Pd_110[0]*Pd_001[1]*Pd_010[2];
	double P_110001110=Pd_110[0]*Pd_001[1]*Pd_110[2];
	double P_110101010=Pd_110[0]*Pd_101[1]*Pd_010[2];
	double P_110101110=Pd_110[0]*Pd_101[1]*Pd_110[2];
	double P_010000011=Pd_010[0]*Pd_011[2];
	double P_010000111=Pd_010[0]*Pd_111[2];
	double P_010000211=Pd_010[0]*Pd_211[2];
	double P_110000011=Pd_110[0]*Pd_011[2];
	double P_110000111=Pd_110[0]*Pd_111[2];
	double P_110000211=Pd_110[0]*Pd_211[2];
	double P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
	double P_001010110=Pd_001[0]*Pd_010[1]*Pd_110[2];
	double P_001110010=Pd_001[0]*Pd_110[1]*Pd_010[2];
	double P_001110110=Pd_001[0]*Pd_110[1]*Pd_110[2];
	double P_101010010=Pd_101[0]*Pd_010[1]*Pd_010[2];
	double P_101010110=Pd_101[0]*Pd_010[1]*Pd_110[2];
	double P_101110010=Pd_101[0]*Pd_110[1]*Pd_010[2];
	double P_101110110=Pd_101[0]*Pd_110[1]*Pd_110[2];
	double P_000011010=Pd_011[1]*Pd_010[2];
	double P_000011110=Pd_011[1]*Pd_110[2];
	double P_000111010=Pd_111[1]*Pd_010[2];
	double P_000111110=Pd_111[1]*Pd_110[2];
	double P_000211010=Pd_211[1]*Pd_010[2];
	double P_000211110=Pd_211[1]*Pd_110[2];
	double P_000010011=Pd_010[1]*Pd_011[2];
	double P_000010111=Pd_010[1]*Pd_111[2];
	double P_000010211=Pd_010[1]*Pd_211[2];
	double P_000110011=Pd_110[1]*Pd_011[2];
	double P_000110111=Pd_110[1]*Pd_111[2];
	double P_000110211=Pd_110[1]*Pd_211[2];
	double P_001000020=Pd_001[0]*Pd_020[2];
	double P_001000120=Pd_001[0]*Pd_120[2];
	double P_001000220=Pd_001[0]*Pd_220[2];
	double P_101000020=Pd_101[0]*Pd_020[2];
	double P_101000120=Pd_101[0]*Pd_120[2];
	double P_101000220=Pd_101[0]*Pd_220[2];
	double P_000001020=Pd_001[1]*Pd_020[2];
	double P_000001120=Pd_001[1]*Pd_120[2];
	double P_000001220=Pd_001[1]*Pd_220[2];
	double P_000101020=Pd_101[1]*Pd_020[2];
	double P_000101120=Pd_101[1]*Pd_120[2];
	double P_000101220=Pd_101[1]*Pd_220[2];
	double P_000000021=Pd_021[2];
	double P_000000121=Pd_121[2];
	double P_000000221=Pd_221[2];
	double P_000000321=Pd_321[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_021000000*QR_020000000000+P_121000000*QR_020000000100+P_221000000*QR_020000000200+P_321000000*QR_020000000300);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_021000000*QR_010010000000+P_121000000*QR_010010000100+P_221000000*QR_010010000200+P_321000000*QR_010010000300);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_021000000*QR_000020000000+P_121000000*QR_000020000100+P_221000000*QR_000020000200+P_321000000*QR_000020000300);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_021000000*QR_010000010000+P_121000000*QR_010000010100+P_221000000*QR_010000010200+P_321000000*QR_010000010300);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_021000000*QR_000010010000+P_121000000*QR_000010010100+P_221000000*QR_000010010200+P_321000000*QR_000010010300);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_021000000*QR_000000020000+P_121000000*QR_000000020100+P_221000000*QR_000000020200+P_321000000*QR_000000020300);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_020001000*QR_020000000000+P_020101000*QR_020000000010+P_120001000*QR_020000000100+P_120101000*QR_020000000110+P_220001000*QR_020000000200+P_220101000*QR_020000000210);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_020001000*QR_010010000000+P_020101000*QR_010010000010+P_120001000*QR_010010000100+P_120101000*QR_010010000110+P_220001000*QR_010010000200+P_220101000*QR_010010000210);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_020001000*QR_000020000000+P_020101000*QR_000020000010+P_120001000*QR_000020000100+P_120101000*QR_000020000110+P_220001000*QR_000020000200+P_220101000*QR_000020000210);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_020001000*QR_010000010000+P_020101000*QR_010000010010+P_120001000*QR_010000010100+P_120101000*QR_010000010110+P_220001000*QR_010000010200+P_220101000*QR_010000010210);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_020001000*QR_000010010000+P_020101000*QR_000010010010+P_120001000*QR_000010010100+P_120101000*QR_000010010110+P_220001000*QR_000010010200+P_220101000*QR_000010010210);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_020001000*QR_000000020000+P_020101000*QR_000000020010+P_120001000*QR_000000020100+P_120101000*QR_000000020110+P_220001000*QR_000000020200+P_220101000*QR_000000020210);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_020000001*QR_020000000000+P_020000101*QR_020000000001+P_120000001*QR_020000000100+P_120000101*QR_020000000101+P_220000001*QR_020000000200+P_220000101*QR_020000000201);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_020000001*QR_010010000000+P_020000101*QR_010010000001+P_120000001*QR_010010000100+P_120000101*QR_010010000101+P_220000001*QR_010010000200+P_220000101*QR_010010000201);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020000001*QR_000020000000+P_020000101*QR_000020000001+P_120000001*QR_000020000100+P_120000101*QR_000020000101+P_220000001*QR_000020000200+P_220000101*QR_000020000201);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_020000001*QR_010000010000+P_020000101*QR_010000010001+P_120000001*QR_010000010100+P_120000101*QR_010000010101+P_220000001*QR_010000010200+P_220000101*QR_010000010201);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020000001*QR_000010010000+P_020000101*QR_000010010001+P_120000001*QR_000010010100+P_120000101*QR_000010010101+P_220000001*QR_000010010200+P_220000101*QR_000010010201);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020000001*QR_000000020000+P_020000101*QR_000000020001+P_120000001*QR_000000020100+P_120000101*QR_000000020101+P_220000001*QR_000000020200+P_220000101*QR_000000020201);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_011010000*QR_020000000000+P_011110000*QR_020000000010+P_111010000*QR_020000000100+P_111110000*QR_020000000110+P_211010000*QR_020000000200+P_211110000*QR_020000000210);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_011010000*QR_010010000000+P_011110000*QR_010010000010+P_111010000*QR_010010000100+P_111110000*QR_010010000110+P_211010000*QR_010010000200+P_211110000*QR_010010000210);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_011010000*QR_000020000000+P_011110000*QR_000020000010+P_111010000*QR_000020000100+P_111110000*QR_000020000110+P_211010000*QR_000020000200+P_211110000*QR_000020000210);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_011010000*QR_010000010000+P_011110000*QR_010000010010+P_111010000*QR_010000010100+P_111110000*QR_010000010110+P_211010000*QR_010000010200+P_211110000*QR_010000010210);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_011010000*QR_000010010000+P_011110000*QR_000010010010+P_111010000*QR_000010010100+P_111110000*QR_000010010110+P_211010000*QR_000010010200+P_211110000*QR_000010010210);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_011010000*QR_000000020000+P_011110000*QR_000000020010+P_111010000*QR_000000020100+P_111110000*QR_000000020110+P_211010000*QR_000000020200+P_211110000*QR_000000020210);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_010011000*QR_020000000000+P_010111000*QR_020000000010+P_010211000*QR_020000000020+P_110011000*QR_020000000100+P_110111000*QR_020000000110+P_110211000*QR_020000000120);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_010011000*QR_010010000000+P_010111000*QR_010010000010+P_010211000*QR_010010000020+P_110011000*QR_010010000100+P_110111000*QR_010010000110+P_110211000*QR_010010000120);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_010011000*QR_000020000000+P_010111000*QR_000020000010+P_010211000*QR_000020000020+P_110011000*QR_000020000100+P_110111000*QR_000020000110+P_110211000*QR_000020000120);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_010011000*QR_010000010000+P_010111000*QR_010000010010+P_010211000*QR_010000010020+P_110011000*QR_010000010100+P_110111000*QR_010000010110+P_110211000*QR_010000010120);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_010011000*QR_000010010000+P_010111000*QR_000010010010+P_010211000*QR_000010010020+P_110011000*QR_000010010100+P_110111000*QR_000010010110+P_110211000*QR_000010010120);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_010011000*QR_000000020000+P_010111000*QR_000000020010+P_010211000*QR_000000020020+P_110011000*QR_000000020100+P_110111000*QR_000000020110+P_110211000*QR_000000020120);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_010010001*QR_020000000000+P_010010101*QR_020000000001+P_010110001*QR_020000000010+P_010110101*QR_020000000011+P_110010001*QR_020000000100+P_110010101*QR_020000000101+P_110110001*QR_020000000110+P_110110101*QR_020000000111);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_010010001*QR_010010000000+P_010010101*QR_010010000001+P_010110001*QR_010010000010+P_010110101*QR_010010000011+P_110010001*QR_010010000100+P_110010101*QR_010010000101+P_110110001*QR_010010000110+P_110110101*QR_010010000111);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010010001*QR_000020000000+P_010010101*QR_000020000001+P_010110001*QR_000020000010+P_010110101*QR_000020000011+P_110010001*QR_000020000100+P_110010101*QR_000020000101+P_110110001*QR_000020000110+P_110110101*QR_000020000111);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_010010001*QR_010000010000+P_010010101*QR_010000010001+P_010110001*QR_010000010010+P_010110101*QR_010000010011+P_110010001*QR_010000010100+P_110010101*QR_010000010101+P_110110001*QR_010000010110+P_110110101*QR_010000010111);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010010001*QR_000010010000+P_010010101*QR_000010010001+P_010110001*QR_000010010010+P_010110101*QR_000010010011+P_110010001*QR_000010010100+P_110010101*QR_000010010101+P_110110001*QR_000010010110+P_110110101*QR_000010010111);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010010001*QR_000000020000+P_010010101*QR_000000020001+P_010110001*QR_000000020010+P_010110101*QR_000000020011+P_110010001*QR_000000020100+P_110010101*QR_000000020101+P_110110001*QR_000000020110+P_110110101*QR_000000020111);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_001020000*QR_020000000000+P_001120000*QR_020000000010+P_001220000*QR_020000000020+P_101020000*QR_020000000100+P_101120000*QR_020000000110+P_101220000*QR_020000000120);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_001020000*QR_010010000000+P_001120000*QR_010010000010+P_001220000*QR_010010000020+P_101020000*QR_010010000100+P_101120000*QR_010010000110+P_101220000*QR_010010000120);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_001020000*QR_000020000000+P_001120000*QR_000020000010+P_001220000*QR_000020000020+P_101020000*QR_000020000100+P_101120000*QR_000020000110+P_101220000*QR_000020000120);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_001020000*QR_010000010000+P_001120000*QR_010000010010+P_001220000*QR_010000010020+P_101020000*QR_010000010100+P_101120000*QR_010000010110+P_101220000*QR_010000010120);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_001020000*QR_000010010000+P_001120000*QR_000010010010+P_001220000*QR_000010010020+P_101020000*QR_000010010100+P_101120000*QR_000010010110+P_101220000*QR_000010010120);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_001020000*QR_000000020000+P_001120000*QR_000000020010+P_001220000*QR_000000020020+P_101020000*QR_000000020100+P_101120000*QR_000000020110+P_101220000*QR_000000020120);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_000021000*QR_020000000000+P_000121000*QR_020000000010+P_000221000*QR_020000000020+P_000321000*QR_020000000030);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_000021000*QR_010010000000+P_000121000*QR_010010000010+P_000221000*QR_010010000020+P_000321000*QR_010010000030);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_000021000*QR_000020000000+P_000121000*QR_000020000010+P_000221000*QR_000020000020+P_000321000*QR_000020000030);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_000021000*QR_010000010000+P_000121000*QR_010000010010+P_000221000*QR_010000010020+P_000321000*QR_010000010030);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_000021000*QR_000010010000+P_000121000*QR_000010010010+P_000221000*QR_000010010020+P_000321000*QR_000010010030);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_000021000*QR_000000020000+P_000121000*QR_000000020010+P_000221000*QR_000000020020+P_000321000*QR_000000020030);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_000020001*QR_020000000000+P_000020101*QR_020000000001+P_000120001*QR_020000000010+P_000120101*QR_020000000011+P_000220001*QR_020000000020+P_000220101*QR_020000000021);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_000020001*QR_010010000000+P_000020101*QR_010010000001+P_000120001*QR_010010000010+P_000120101*QR_010010000011+P_000220001*QR_010010000020+P_000220101*QR_010010000021);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000020001*QR_000020000000+P_000020101*QR_000020000001+P_000120001*QR_000020000010+P_000120101*QR_000020000011+P_000220001*QR_000020000020+P_000220101*QR_000020000021);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_000020001*QR_010000010000+P_000020101*QR_010000010001+P_000120001*QR_010000010010+P_000120101*QR_010000010011+P_000220001*QR_010000010020+P_000220101*QR_010000010021);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000020001*QR_000010010000+P_000020101*QR_000010010001+P_000120001*QR_000010010010+P_000120101*QR_000010010011+P_000220001*QR_000010010020+P_000220101*QR_000010010021);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000020001*QR_000000020000+P_000020101*QR_000000020001+P_000120001*QR_000000020010+P_000120101*QR_000000020011+P_000220001*QR_000000020020+P_000220101*QR_000000020021);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_011000010*QR_020000000000+P_011000110*QR_020000000001+P_111000010*QR_020000000100+P_111000110*QR_020000000101+P_211000010*QR_020000000200+P_211000110*QR_020000000201);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_011000010*QR_010010000000+P_011000110*QR_010010000001+P_111000010*QR_010010000100+P_111000110*QR_010010000101+P_211000010*QR_010010000200+P_211000110*QR_010010000201);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_011000010*QR_000020000000+P_011000110*QR_000020000001+P_111000010*QR_000020000100+P_111000110*QR_000020000101+P_211000010*QR_000020000200+P_211000110*QR_000020000201);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_011000010*QR_010000010000+P_011000110*QR_010000010001+P_111000010*QR_010000010100+P_111000110*QR_010000010101+P_211000010*QR_010000010200+P_211000110*QR_010000010201);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_011000010*QR_000010010000+P_011000110*QR_000010010001+P_111000010*QR_000010010100+P_111000110*QR_000010010101+P_211000010*QR_000010010200+P_211000110*QR_000010010201);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_011000010*QR_000000020000+P_011000110*QR_000000020001+P_111000010*QR_000000020100+P_111000110*QR_000000020101+P_211000010*QR_000000020200+P_211000110*QR_000000020201);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_010001010*QR_020000000000+P_010001110*QR_020000000001+P_010101010*QR_020000000010+P_010101110*QR_020000000011+P_110001010*QR_020000000100+P_110001110*QR_020000000101+P_110101010*QR_020000000110+P_110101110*QR_020000000111);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_010001010*QR_010010000000+P_010001110*QR_010010000001+P_010101010*QR_010010000010+P_010101110*QR_010010000011+P_110001010*QR_010010000100+P_110001110*QR_010010000101+P_110101010*QR_010010000110+P_110101110*QR_010010000111);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_010001010*QR_000020000000+P_010001110*QR_000020000001+P_010101010*QR_000020000010+P_010101110*QR_000020000011+P_110001010*QR_000020000100+P_110001110*QR_000020000101+P_110101010*QR_000020000110+P_110101110*QR_000020000111);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_010001010*QR_010000010000+P_010001110*QR_010000010001+P_010101010*QR_010000010010+P_010101110*QR_010000010011+P_110001010*QR_010000010100+P_110001110*QR_010000010101+P_110101010*QR_010000010110+P_110101110*QR_010000010111);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_010001010*QR_000010010000+P_010001110*QR_000010010001+P_010101010*QR_000010010010+P_010101110*QR_000010010011+P_110001010*QR_000010010100+P_110001110*QR_000010010101+P_110101010*QR_000010010110+P_110101110*QR_000010010111);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_010001010*QR_000000020000+P_010001110*QR_000000020001+P_010101010*QR_000000020010+P_010101110*QR_000000020011+P_110001010*QR_000000020100+P_110001110*QR_000000020101+P_110101010*QR_000000020110+P_110101110*QR_000000020111);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_010000011*QR_020000000000+P_010000111*QR_020000000001+P_010000211*QR_020000000002+P_110000011*QR_020000000100+P_110000111*QR_020000000101+P_110000211*QR_020000000102);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_010000011*QR_010010000000+P_010000111*QR_010010000001+P_010000211*QR_010010000002+P_110000011*QR_010010000100+P_110000111*QR_010010000101+P_110000211*QR_010010000102);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010000011*QR_000020000000+P_010000111*QR_000020000001+P_010000211*QR_000020000002+P_110000011*QR_000020000100+P_110000111*QR_000020000101+P_110000211*QR_000020000102);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_010000011*QR_010000010000+P_010000111*QR_010000010001+P_010000211*QR_010000010002+P_110000011*QR_010000010100+P_110000111*QR_010000010101+P_110000211*QR_010000010102);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010000011*QR_000010010000+P_010000111*QR_000010010001+P_010000211*QR_000010010002+P_110000011*QR_000010010100+P_110000111*QR_000010010101+P_110000211*QR_000010010102);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010000011*QR_000000020000+P_010000111*QR_000000020001+P_010000211*QR_000000020002+P_110000011*QR_000000020100+P_110000111*QR_000000020101+P_110000211*QR_000000020102);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_001010010*QR_020000000000+P_001010110*QR_020000000001+P_001110010*QR_020000000010+P_001110110*QR_020000000011+P_101010010*QR_020000000100+P_101010110*QR_020000000101+P_101110010*QR_020000000110+P_101110110*QR_020000000111);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_001010010*QR_010010000000+P_001010110*QR_010010000001+P_001110010*QR_010010000010+P_001110110*QR_010010000011+P_101010010*QR_010010000100+P_101010110*QR_010010000101+P_101110010*QR_010010000110+P_101110110*QR_010010000111);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_001010010*QR_000020000000+P_001010110*QR_000020000001+P_001110010*QR_000020000010+P_001110110*QR_000020000011+P_101010010*QR_000020000100+P_101010110*QR_000020000101+P_101110010*QR_000020000110+P_101110110*QR_000020000111);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_001010010*QR_010000010000+P_001010110*QR_010000010001+P_001110010*QR_010000010010+P_001110110*QR_010000010011+P_101010010*QR_010000010100+P_101010110*QR_010000010101+P_101110010*QR_010000010110+P_101110110*QR_010000010111);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_001010010*QR_000010010000+P_001010110*QR_000010010001+P_001110010*QR_000010010010+P_001110110*QR_000010010011+P_101010010*QR_000010010100+P_101010110*QR_000010010101+P_101110010*QR_000010010110+P_101110110*QR_000010010111);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_001010010*QR_000000020000+P_001010110*QR_000000020001+P_001110010*QR_000000020010+P_001110110*QR_000000020011+P_101010010*QR_000000020100+P_101010110*QR_000000020101+P_101110010*QR_000000020110+P_101110110*QR_000000020111);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_000011010*QR_020000000000+P_000011110*QR_020000000001+P_000111010*QR_020000000010+P_000111110*QR_020000000011+P_000211010*QR_020000000020+P_000211110*QR_020000000021);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_000011010*QR_010010000000+P_000011110*QR_010010000001+P_000111010*QR_010010000010+P_000111110*QR_010010000011+P_000211010*QR_010010000020+P_000211110*QR_010010000021);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_000011010*QR_000020000000+P_000011110*QR_000020000001+P_000111010*QR_000020000010+P_000111110*QR_000020000011+P_000211010*QR_000020000020+P_000211110*QR_000020000021);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_000011010*QR_010000010000+P_000011110*QR_010000010001+P_000111010*QR_010000010010+P_000111110*QR_010000010011+P_000211010*QR_010000010020+P_000211110*QR_010000010021);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_000011010*QR_000010010000+P_000011110*QR_000010010001+P_000111010*QR_000010010010+P_000111110*QR_000010010011+P_000211010*QR_000010010020+P_000211110*QR_000010010021);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_000011010*QR_000000020000+P_000011110*QR_000000020001+P_000111010*QR_000000020010+P_000111110*QR_000000020011+P_000211010*QR_000000020020+P_000211110*QR_000000020021);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_000010011*QR_020000000000+P_000010111*QR_020000000001+P_000010211*QR_020000000002+P_000110011*QR_020000000010+P_000110111*QR_020000000011+P_000110211*QR_020000000012);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_000010011*QR_010010000000+P_000010111*QR_010010000001+P_000010211*QR_010010000002+P_000110011*QR_010010000010+P_000110111*QR_010010000011+P_000110211*QR_010010000012);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000010011*QR_000020000000+P_000010111*QR_000020000001+P_000010211*QR_000020000002+P_000110011*QR_000020000010+P_000110111*QR_000020000011+P_000110211*QR_000020000012);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_000010011*QR_010000010000+P_000010111*QR_010000010001+P_000010211*QR_010000010002+P_000110011*QR_010000010010+P_000110111*QR_010000010011+P_000110211*QR_010000010012);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000010011*QR_000010010000+P_000010111*QR_000010010001+P_000010211*QR_000010010002+P_000110011*QR_000010010010+P_000110111*QR_000010010011+P_000110211*QR_000010010012);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000010011*QR_000000020000+P_000010111*QR_000000020001+P_000010211*QR_000000020002+P_000110011*QR_000000020010+P_000110111*QR_000000020011+P_000110211*QR_000000020012);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_001000020*QR_020000000000+P_001000120*QR_020000000001+P_001000220*QR_020000000002+P_101000020*QR_020000000100+P_101000120*QR_020000000101+P_101000220*QR_020000000102);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_001000020*QR_010010000000+P_001000120*QR_010010000001+P_001000220*QR_010010000002+P_101000020*QR_010010000100+P_101000120*QR_010010000101+P_101000220*QR_010010000102);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_001000020*QR_000020000000+P_001000120*QR_000020000001+P_001000220*QR_000020000002+P_101000020*QR_000020000100+P_101000120*QR_000020000101+P_101000220*QR_000020000102);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_001000020*QR_010000010000+P_001000120*QR_010000010001+P_001000220*QR_010000010002+P_101000020*QR_010000010100+P_101000120*QR_010000010101+P_101000220*QR_010000010102);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_001000020*QR_000010010000+P_001000120*QR_000010010001+P_001000220*QR_000010010002+P_101000020*QR_000010010100+P_101000120*QR_000010010101+P_101000220*QR_000010010102);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_001000020*QR_000000020000+P_001000120*QR_000000020001+P_001000220*QR_000000020002+P_101000020*QR_000000020100+P_101000120*QR_000000020101+P_101000220*QR_000000020102);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_000001020*QR_020000000000+P_000001120*QR_020000000001+P_000001220*QR_020000000002+P_000101020*QR_020000000010+P_000101120*QR_020000000011+P_000101220*QR_020000000012);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_000001020*QR_010010000000+P_000001120*QR_010010000001+P_000001220*QR_010010000002+P_000101020*QR_010010000010+P_000101120*QR_010010000011+P_000101220*QR_010010000012);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_000001020*QR_000020000000+P_000001120*QR_000020000001+P_000001220*QR_000020000002+P_000101020*QR_000020000010+P_000101120*QR_000020000011+P_000101220*QR_000020000012);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_000001020*QR_010000010000+P_000001120*QR_010000010001+P_000001220*QR_010000010002+P_000101020*QR_010000010010+P_000101120*QR_010000010011+P_000101220*QR_010000010012);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_000001020*QR_000010010000+P_000001120*QR_000010010001+P_000001220*QR_000010010002+P_000101020*QR_000010010010+P_000101120*QR_000010010011+P_000101220*QR_000010010012);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_000001020*QR_000000020000+P_000001120*QR_000000020001+P_000001220*QR_000000020002+P_000101020*QR_000000020010+P_000101120*QR_000000020011+P_000101220*QR_000000020012);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_000000021*QR_020000000000+P_000000121*QR_020000000001+P_000000221*QR_020000000002+P_000000321*QR_020000000003);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_000000021*QR_010010000000+P_000000121*QR_010010000001+P_000000221*QR_010010000002+P_000000321*QR_010010000003);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000000021*QR_000020000000+P_000000121*QR_000020000001+P_000000221*QR_000020000002+P_000000321*QR_000020000003);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_000000021*QR_010000010000+P_000000121*QR_010000010001+P_000000221*QR_010000010002+P_000000321*QR_010000010003);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000000021*QR_000010010000+P_000000121*QR_000010010001+P_000000221*QR_000010010002+P_000000321*QR_000010010003);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000000021*QR_000000020000+P_000000121*QR_000000020001+P_000000221*QR_000000020002+P_000000321*QR_000000020003);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_ddds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[ii*3+0];
				Qd_010[1]=QC[ii*3+1];
				Qd_010[2]=QC[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_p=tex1Dfetch(tex_K2_p,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Zta,jj);
            double Zta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pp,jj);
            double pp=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+0);
            double PX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+1);
            double PY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_P,jj*3+2);
            double PZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_010[3];
            temp_int2=tex1Dfetch(tex_PA,jj*3+0);
            Pd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+1);
            Pd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PA,jj*3+2);
            Pd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
				double Pd_001[3];
            temp_int2=tex1Dfetch(tex_PB,jj*3+0);
            Pd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+1);
            Pd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_PB,jj*3+2);
            Pd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[7];
                Ft_fs_6(6,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[6]*=64*alphaT*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[6];
	double R_200[5];
	double R_300[4];
	double R_400[3];
	double R_500[2];
	double R_600[1];
	double R_010[6];
	double R_110[5];
	double R_210[4];
	double R_310[3];
	double R_410[2];
	double R_510[1];
	double R_020[5];
	double R_120[4];
	double R_220[3];
	double R_320[2];
	double R_420[1];
	double R_030[4];
	double R_130[3];
	double R_230[2];
	double R_330[1];
	double R_040[3];
	double R_140[2];
	double R_240[1];
	double R_050[2];
	double R_150[1];
	double R_060[1];
	double R_001[6];
	double R_101[5];
	double R_201[4];
	double R_301[3];
	double R_401[2];
	double R_501[1];
	double R_011[5];
	double R_111[4];
	double R_211[3];
	double R_311[2];
	double R_411[1];
	double R_021[4];
	double R_121[3];
	double R_221[2];
	double R_321[1];
	double R_031[3];
	double R_131[2];
	double R_231[1];
	double R_041[2];
	double R_141[1];
	double R_051[1];
	double R_002[5];
	double R_102[4];
	double R_202[3];
	double R_302[2];
	double R_402[1];
	double R_012[4];
	double R_112[3];
	double R_212[2];
	double R_312[1];
	double R_022[3];
	double R_122[2];
	double R_222[1];
	double R_032[2];
	double R_132[1];
	double R_042[1];
	double R_003[4];
	double R_103[3];
	double R_203[2];
	double R_303[1];
	double R_013[3];
	double R_113[2];
	double R_213[1];
	double R_023[2];
	double R_123[1];
	double R_033[1];
	double R_004[3];
	double R_104[2];
	double R_204[1];
	double R_014[2];
	double R_114[1];
	double R_024[1];
	double R_005[2];
	double R_105[1];
	double R_015[1];
	double R_006[1];
	for(int i=0;i<6;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<6;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<6;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<5;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<5;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<5;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<4;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<4;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<4;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<4;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<4;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<4;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<4;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<4;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<3;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<3;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<3;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<3;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<3;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<3;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<3;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<3;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<3;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<2;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<2;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<2;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<2;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<2;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<2;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<2;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<2;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<2;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<2;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<2;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<2;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<2;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<2;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<2;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_600[i]=TX*R_500[i+1]+5*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_510[i]=TY*R_500[i+1];
	}
	for(int i=0;i<1;i++){
		R_420[i]=TX*R_320[i+1]+3*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_330[i]=TX*R_230[i+1]+2*R_130[i+1];
	}
	for(int i=0;i<1;i++){
		R_240[i]=TY*R_230[i+1]+3*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_150[i]=TX*R_050[i+1];
	}
	for(int i=0;i<1;i++){
		R_060[i]=TY*R_050[i+1]+5*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_501[i]=TZ*R_500[i+1];
	}
	for(int i=0;i<1;i++){
		R_411[i]=TY*R_401[i+1];
	}
	for(int i=0;i<1;i++){
		R_321[i]=TZ*R_320[i+1];
	}
	for(int i=0;i<1;i++){
		R_231[i]=TZ*R_230[i+1];
	}
	for(int i=0;i<1;i++){
		R_141[i]=TX*R_041[i+1];
	}
	for(int i=0;i<1;i++){
		R_051[i]=TZ*R_050[i+1];
	}
	for(int i=0;i<1;i++){
		R_402[i]=TX*R_302[i+1]+3*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_312[i]=TY*R_302[i+1];
	}
	for(int i=0;i<1;i++){
		R_222[i]=TX*R_122[i+1]+R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_132[i]=TX*R_032[i+1];
	}
	for(int i=0;i<1;i++){
		R_042[i]=TY*R_032[i+1]+3*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_303[i]=TX*R_203[i+1]+2*R_103[i+1];
	}
	for(int i=0;i<1;i++){
		R_213[i]=TY*R_203[i+1];
	}
	for(int i=0;i<1;i++){
		R_123[i]=TX*R_023[i+1];
	}
	for(int i=0;i<1;i++){
		R_033[i]=TY*R_023[i+1]+2*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_204[i]=TZ*R_203[i+1]+3*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_114[i]=TX*R_014[i+1];
	}
	for(int i=0;i<1;i++){
		R_024[i]=TZ*R_023[i+1]+3*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_105[i]=TX*R_005[i+1];
	}
	for(int i=0;i<1;i++){
		R_015[i]=TY*R_005[i+1];
	}
	for(int i=0;i<1;i++){
		R_006[i]=TZ*R_005[i+1]+5*R_004[i+1];
	}
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_312[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_321[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		double Pd_322[3];
		double Pd_422[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=2*Pd_211[i]+Pd_001[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=Pd_001[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_312[i]=aPin1*Pd_211[i];
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=2*Pd_211[i]+Pd_010[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=Pd_010[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_321[i]=aPin1*Pd_211[i];
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=2*Pd_212[i]+Pd_010[i]*Pd_112[i]+aPin1*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=3*Pd_312[i]+Pd_010[i]*Pd_212[i]+aPin1*Pd_112[i];
			}
		for(int i=0;i<3;i++){
			Pd_322[i]=Pd_010[i]*Pd_312[i]+aPin1*Pd_212[i];
			}
		for(int i=0;i<3;i++){
			Pd_422[i]=aPin1*Pd_312[i];
			}
	double P_022000000=Pd_022[0];
	double P_122000000=Pd_122[0];
	double P_222000000=Pd_222[0];
	double P_322000000=Pd_322[0];
	double P_422000000=Pd_422[0];
	double P_021001000=Pd_021[0]*Pd_001[1];
	double P_021101000=Pd_021[0]*Pd_101[1];
	double P_121001000=Pd_121[0]*Pd_001[1];
	double P_121101000=Pd_121[0]*Pd_101[1];
	double P_221001000=Pd_221[0]*Pd_001[1];
	double P_221101000=Pd_221[0]*Pd_101[1];
	double P_321001000=Pd_321[0]*Pd_001[1];
	double P_321101000=Pd_321[0]*Pd_101[1];
	double P_020002000=Pd_020[0]*Pd_002[1];
	double P_020102000=Pd_020[0]*Pd_102[1];
	double P_020202000=Pd_020[0]*Pd_202[1];
	double P_120002000=Pd_120[0]*Pd_002[1];
	double P_120102000=Pd_120[0]*Pd_102[1];
	double P_120202000=Pd_120[0]*Pd_202[1];
	double P_220002000=Pd_220[0]*Pd_002[1];
	double P_220102000=Pd_220[0]*Pd_102[1];
	double P_220202000=Pd_220[0]*Pd_202[1];
	double P_021000001=Pd_021[0]*Pd_001[2];
	double P_021000101=Pd_021[0]*Pd_101[2];
	double P_121000001=Pd_121[0]*Pd_001[2];
	double P_121000101=Pd_121[0]*Pd_101[2];
	double P_221000001=Pd_221[0]*Pd_001[2];
	double P_221000101=Pd_221[0]*Pd_101[2];
	double P_321000001=Pd_321[0]*Pd_001[2];
	double P_321000101=Pd_321[0]*Pd_101[2];
	double P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
	double P_020001101=Pd_020[0]*Pd_001[1]*Pd_101[2];
	double P_020101001=Pd_020[0]*Pd_101[1]*Pd_001[2];
	double P_020101101=Pd_020[0]*Pd_101[1]*Pd_101[2];
	double P_120001001=Pd_120[0]*Pd_001[1]*Pd_001[2];
	double P_120001101=Pd_120[0]*Pd_001[1]*Pd_101[2];
	double P_120101001=Pd_120[0]*Pd_101[1]*Pd_001[2];
	double P_120101101=Pd_120[0]*Pd_101[1]*Pd_101[2];
	double P_220001001=Pd_220[0]*Pd_001[1]*Pd_001[2];
	double P_220001101=Pd_220[0]*Pd_001[1]*Pd_101[2];
	double P_220101001=Pd_220[0]*Pd_101[1]*Pd_001[2];
	double P_220101101=Pd_220[0]*Pd_101[1]*Pd_101[2];
	double P_020000002=Pd_020[0]*Pd_002[2];
	double P_020000102=Pd_020[0]*Pd_102[2];
	double P_020000202=Pd_020[0]*Pd_202[2];
	double P_120000002=Pd_120[0]*Pd_002[2];
	double P_120000102=Pd_120[0]*Pd_102[2];
	double P_120000202=Pd_120[0]*Pd_202[2];
	double P_220000002=Pd_220[0]*Pd_002[2];
	double P_220000102=Pd_220[0]*Pd_102[2];
	double P_220000202=Pd_220[0]*Pd_202[2];
	double P_012010000=Pd_012[0]*Pd_010[1];
	double P_012110000=Pd_012[0]*Pd_110[1];
	double P_112010000=Pd_112[0]*Pd_010[1];
	double P_112110000=Pd_112[0]*Pd_110[1];
	double P_212010000=Pd_212[0]*Pd_010[1];
	double P_212110000=Pd_212[0]*Pd_110[1];
	double P_312010000=Pd_312[0]*Pd_010[1];
	double P_312110000=Pd_312[0]*Pd_110[1];
	double P_011011000=Pd_011[0]*Pd_011[1];
	double P_011111000=Pd_011[0]*Pd_111[1];
	double P_011211000=Pd_011[0]*Pd_211[1];
	double P_111011000=Pd_111[0]*Pd_011[1];
	double P_111111000=Pd_111[0]*Pd_111[1];
	double P_111211000=Pd_111[0]*Pd_211[1];
	double P_211011000=Pd_211[0]*Pd_011[1];
	double P_211111000=Pd_211[0]*Pd_111[1];
	double P_211211000=Pd_211[0]*Pd_211[1];
	double P_010012000=Pd_010[0]*Pd_012[1];
	double P_010112000=Pd_010[0]*Pd_112[1];
	double P_010212000=Pd_010[0]*Pd_212[1];
	double P_010312000=Pd_010[0]*Pd_312[1];
	double P_110012000=Pd_110[0]*Pd_012[1];
	double P_110112000=Pd_110[0]*Pd_112[1];
	double P_110212000=Pd_110[0]*Pd_212[1];
	double P_110312000=Pd_110[0]*Pd_312[1];
	double P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
	double P_011010101=Pd_011[0]*Pd_010[1]*Pd_101[2];
	double P_011110001=Pd_011[0]*Pd_110[1]*Pd_001[2];
	double P_011110101=Pd_011[0]*Pd_110[1]*Pd_101[2];
	double P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
	double P_111010101=Pd_111[0]*Pd_010[1]*Pd_101[2];
	double P_111110001=Pd_111[0]*Pd_110[1]*Pd_001[2];
	double P_111110101=Pd_111[0]*Pd_110[1]*Pd_101[2];
	double P_211010001=Pd_211[0]*Pd_010[1]*Pd_001[2];
	double P_211010101=Pd_211[0]*Pd_010[1]*Pd_101[2];
	double P_211110001=Pd_211[0]*Pd_110[1]*Pd_001[2];
	double P_211110101=Pd_211[0]*Pd_110[1]*Pd_101[2];
	double P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
	double P_010011101=Pd_010[0]*Pd_011[1]*Pd_101[2];
	double P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
	double P_010111101=Pd_010[0]*Pd_111[1]*Pd_101[2];
	double P_010211001=Pd_010[0]*Pd_211[1]*Pd_001[2];
	double P_010211101=Pd_010[0]*Pd_211[1]*Pd_101[2];
	double P_110011001=Pd_110[0]*Pd_011[1]*Pd_001[2];
	double P_110011101=Pd_110[0]*Pd_011[1]*Pd_101[2];
	double P_110111001=Pd_110[0]*Pd_111[1]*Pd_001[2];
	double P_110111101=Pd_110[0]*Pd_111[1]*Pd_101[2];
	double P_110211001=Pd_110[0]*Pd_211[1]*Pd_001[2];
	double P_110211101=Pd_110[0]*Pd_211[1]*Pd_101[2];
	double P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
	double P_010010102=Pd_010[0]*Pd_010[1]*Pd_102[2];
	double P_010010202=Pd_010[0]*Pd_010[1]*Pd_202[2];
	double P_010110002=Pd_010[0]*Pd_110[1]*Pd_002[2];
	double P_010110102=Pd_010[0]*Pd_110[1]*Pd_102[2];
	double P_010110202=Pd_010[0]*Pd_110[1]*Pd_202[2];
	double P_110010002=Pd_110[0]*Pd_010[1]*Pd_002[2];
	double P_110010102=Pd_110[0]*Pd_010[1]*Pd_102[2];
	double P_110010202=Pd_110[0]*Pd_010[1]*Pd_202[2];
	double P_110110002=Pd_110[0]*Pd_110[1]*Pd_002[2];
	double P_110110102=Pd_110[0]*Pd_110[1]*Pd_102[2];
	double P_110110202=Pd_110[0]*Pd_110[1]*Pd_202[2];
	double P_002020000=Pd_002[0]*Pd_020[1];
	double P_002120000=Pd_002[0]*Pd_120[1];
	double P_002220000=Pd_002[0]*Pd_220[1];
	double P_102020000=Pd_102[0]*Pd_020[1];
	double P_102120000=Pd_102[0]*Pd_120[1];
	double P_102220000=Pd_102[0]*Pd_220[1];
	double P_202020000=Pd_202[0]*Pd_020[1];
	double P_202120000=Pd_202[0]*Pd_120[1];
	double P_202220000=Pd_202[0]*Pd_220[1];
	double P_001021000=Pd_001[0]*Pd_021[1];
	double P_001121000=Pd_001[0]*Pd_121[1];
	double P_001221000=Pd_001[0]*Pd_221[1];
	double P_001321000=Pd_001[0]*Pd_321[1];
	double P_101021000=Pd_101[0]*Pd_021[1];
	double P_101121000=Pd_101[0]*Pd_121[1];
	double P_101221000=Pd_101[0]*Pd_221[1];
	double P_101321000=Pd_101[0]*Pd_321[1];
	double P_000022000=Pd_022[1];
	double P_000122000=Pd_122[1];
	double P_000222000=Pd_222[1];
	double P_000322000=Pd_322[1];
	double P_000422000=Pd_422[1];
	double P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
	double P_001020101=Pd_001[0]*Pd_020[1]*Pd_101[2];
	double P_001120001=Pd_001[0]*Pd_120[1]*Pd_001[2];
	double P_001120101=Pd_001[0]*Pd_120[1]*Pd_101[2];
	double P_001220001=Pd_001[0]*Pd_220[1]*Pd_001[2];
	double P_001220101=Pd_001[0]*Pd_220[1]*Pd_101[2];
	double P_101020001=Pd_101[0]*Pd_020[1]*Pd_001[2];
	double P_101020101=Pd_101[0]*Pd_020[1]*Pd_101[2];
	double P_101120001=Pd_101[0]*Pd_120[1]*Pd_001[2];
	double P_101120101=Pd_101[0]*Pd_120[1]*Pd_101[2];
	double P_101220001=Pd_101[0]*Pd_220[1]*Pd_001[2];
	double P_101220101=Pd_101[0]*Pd_220[1]*Pd_101[2];
	double P_000021001=Pd_021[1]*Pd_001[2];
	double P_000021101=Pd_021[1]*Pd_101[2];
	double P_000121001=Pd_121[1]*Pd_001[2];
	double P_000121101=Pd_121[1]*Pd_101[2];
	double P_000221001=Pd_221[1]*Pd_001[2];
	double P_000221101=Pd_221[1]*Pd_101[2];
	double P_000321001=Pd_321[1]*Pd_001[2];
	double P_000321101=Pd_321[1]*Pd_101[2];
	double P_000020002=Pd_020[1]*Pd_002[2];
	double P_000020102=Pd_020[1]*Pd_102[2];
	double P_000020202=Pd_020[1]*Pd_202[2];
	double P_000120002=Pd_120[1]*Pd_002[2];
	double P_000120102=Pd_120[1]*Pd_102[2];
	double P_000120202=Pd_120[1]*Pd_202[2];
	double P_000220002=Pd_220[1]*Pd_002[2];
	double P_000220102=Pd_220[1]*Pd_102[2];
	double P_000220202=Pd_220[1]*Pd_202[2];
	double P_012000010=Pd_012[0]*Pd_010[2];
	double P_012000110=Pd_012[0]*Pd_110[2];
	double P_112000010=Pd_112[0]*Pd_010[2];
	double P_112000110=Pd_112[0]*Pd_110[2];
	double P_212000010=Pd_212[0]*Pd_010[2];
	double P_212000110=Pd_212[0]*Pd_110[2];
	double P_312000010=Pd_312[0]*Pd_010[2];
	double P_312000110=Pd_312[0]*Pd_110[2];
	double P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
	double P_011001110=Pd_011[0]*Pd_001[1]*Pd_110[2];
	double P_011101010=Pd_011[0]*Pd_101[1]*Pd_010[2];
	double P_011101110=Pd_011[0]*Pd_101[1]*Pd_110[2];
	double P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
	double P_111001110=Pd_111[0]*Pd_001[1]*Pd_110[2];
	double P_111101010=Pd_111[0]*Pd_101[1]*Pd_010[2];
	double P_111101110=Pd_111[0]*Pd_101[1]*Pd_110[2];
	double P_211001010=Pd_211[0]*Pd_001[1]*Pd_010[2];
	double P_211001110=Pd_211[0]*Pd_001[1]*Pd_110[2];
	double P_211101010=Pd_211[0]*Pd_101[1]*Pd_010[2];
	double P_211101110=Pd_211[0]*Pd_101[1]*Pd_110[2];
	double P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
	double P_010002110=Pd_010[0]*Pd_002[1]*Pd_110[2];
	double P_010102010=Pd_010[0]*Pd_102[1]*Pd_010[2];
	double P_010102110=Pd_010[0]*Pd_102[1]*Pd_110[2];
	double P_010202010=Pd_010[0]*Pd_202[1]*Pd_010[2];
	double P_010202110=Pd_010[0]*Pd_202[1]*Pd_110[2];
	double P_110002010=Pd_110[0]*Pd_002[1]*Pd_010[2];
	double P_110002110=Pd_110[0]*Pd_002[1]*Pd_110[2];
	double P_110102010=Pd_110[0]*Pd_102[1]*Pd_010[2];
	double P_110102110=Pd_110[0]*Pd_102[1]*Pd_110[2];
	double P_110202010=Pd_110[0]*Pd_202[1]*Pd_010[2];
	double P_110202110=Pd_110[0]*Pd_202[1]*Pd_110[2];
	double P_011000011=Pd_011[0]*Pd_011[2];
	double P_011000111=Pd_011[0]*Pd_111[2];
	double P_011000211=Pd_011[0]*Pd_211[2];
	double P_111000011=Pd_111[0]*Pd_011[2];
	double P_111000111=Pd_111[0]*Pd_111[2];
	double P_111000211=Pd_111[0]*Pd_211[2];
	double P_211000011=Pd_211[0]*Pd_011[2];
	double P_211000111=Pd_211[0]*Pd_111[2];
	double P_211000211=Pd_211[0]*Pd_211[2];
	double P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
	double P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
	double P_010001211=Pd_010[0]*Pd_001[1]*Pd_211[2];
	double P_010101011=Pd_010[0]*Pd_101[1]*Pd_011[2];
	double P_010101111=Pd_010[0]*Pd_101[1]*Pd_111[2];
	double P_010101211=Pd_010[0]*Pd_101[1]*Pd_211[2];
	double P_110001011=Pd_110[0]*Pd_001[1]*Pd_011[2];
	double P_110001111=Pd_110[0]*Pd_001[1]*Pd_111[2];
	double P_110001211=Pd_110[0]*Pd_001[1]*Pd_211[2];
	double P_110101011=Pd_110[0]*Pd_101[1]*Pd_011[2];
	double P_110101111=Pd_110[0]*Pd_101[1]*Pd_111[2];
	double P_110101211=Pd_110[0]*Pd_101[1]*Pd_211[2];
	double P_010000012=Pd_010[0]*Pd_012[2];
	double P_010000112=Pd_010[0]*Pd_112[2];
	double P_010000212=Pd_010[0]*Pd_212[2];
	double P_010000312=Pd_010[0]*Pd_312[2];
	double P_110000012=Pd_110[0]*Pd_012[2];
	double P_110000112=Pd_110[0]*Pd_112[2];
	double P_110000212=Pd_110[0]*Pd_212[2];
	double P_110000312=Pd_110[0]*Pd_312[2];
	double P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
	double P_002010110=Pd_002[0]*Pd_010[1]*Pd_110[2];
	double P_002110010=Pd_002[0]*Pd_110[1]*Pd_010[2];
	double P_002110110=Pd_002[0]*Pd_110[1]*Pd_110[2];
	double P_102010010=Pd_102[0]*Pd_010[1]*Pd_010[2];
	double P_102010110=Pd_102[0]*Pd_010[1]*Pd_110[2];
	double P_102110010=Pd_102[0]*Pd_110[1]*Pd_010[2];
	double P_102110110=Pd_102[0]*Pd_110[1]*Pd_110[2];
	double P_202010010=Pd_202[0]*Pd_010[1]*Pd_010[2];
	double P_202010110=Pd_202[0]*Pd_010[1]*Pd_110[2];
	double P_202110010=Pd_202[0]*Pd_110[1]*Pd_010[2];
	double P_202110110=Pd_202[0]*Pd_110[1]*Pd_110[2];
	double P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
	double P_001011110=Pd_001[0]*Pd_011[1]*Pd_110[2];
	double P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
	double P_001111110=Pd_001[0]*Pd_111[1]*Pd_110[2];
	double P_001211010=Pd_001[0]*Pd_211[1]*Pd_010[2];
	double P_001211110=Pd_001[0]*Pd_211[1]*Pd_110[2];
	double P_101011010=Pd_101[0]*Pd_011[1]*Pd_010[2];
	double P_101011110=Pd_101[0]*Pd_011[1]*Pd_110[2];
	double P_101111010=Pd_101[0]*Pd_111[1]*Pd_010[2];
	double P_101111110=Pd_101[0]*Pd_111[1]*Pd_110[2];
	double P_101211010=Pd_101[0]*Pd_211[1]*Pd_010[2];
	double P_101211110=Pd_101[0]*Pd_211[1]*Pd_110[2];
	double P_000012010=Pd_012[1]*Pd_010[2];
	double P_000012110=Pd_012[1]*Pd_110[2];
	double P_000112010=Pd_112[1]*Pd_010[2];
	double P_000112110=Pd_112[1]*Pd_110[2];
	double P_000212010=Pd_212[1]*Pd_010[2];
	double P_000212110=Pd_212[1]*Pd_110[2];
	double P_000312010=Pd_312[1]*Pd_010[2];
	double P_000312110=Pd_312[1]*Pd_110[2];
	double P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
	double P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
	double P_001010211=Pd_001[0]*Pd_010[1]*Pd_211[2];
	double P_001110011=Pd_001[0]*Pd_110[1]*Pd_011[2];
	double P_001110111=Pd_001[0]*Pd_110[1]*Pd_111[2];
	double P_001110211=Pd_001[0]*Pd_110[1]*Pd_211[2];
	double P_101010011=Pd_101[0]*Pd_010[1]*Pd_011[2];
	double P_101010111=Pd_101[0]*Pd_010[1]*Pd_111[2];
	double P_101010211=Pd_101[0]*Pd_010[1]*Pd_211[2];
	double P_101110011=Pd_101[0]*Pd_110[1]*Pd_011[2];
	double P_101110111=Pd_101[0]*Pd_110[1]*Pd_111[2];
	double P_101110211=Pd_101[0]*Pd_110[1]*Pd_211[2];
	double P_000011011=Pd_011[1]*Pd_011[2];
	double P_000011111=Pd_011[1]*Pd_111[2];
	double P_000011211=Pd_011[1]*Pd_211[2];
	double P_000111011=Pd_111[1]*Pd_011[2];
	double P_000111111=Pd_111[1]*Pd_111[2];
	double P_000111211=Pd_111[1]*Pd_211[2];
	double P_000211011=Pd_211[1]*Pd_011[2];
	double P_000211111=Pd_211[1]*Pd_111[2];
	double P_000211211=Pd_211[1]*Pd_211[2];
	double P_000010012=Pd_010[1]*Pd_012[2];
	double P_000010112=Pd_010[1]*Pd_112[2];
	double P_000010212=Pd_010[1]*Pd_212[2];
	double P_000010312=Pd_010[1]*Pd_312[2];
	double P_000110012=Pd_110[1]*Pd_012[2];
	double P_000110112=Pd_110[1]*Pd_112[2];
	double P_000110212=Pd_110[1]*Pd_212[2];
	double P_000110312=Pd_110[1]*Pd_312[2];
	double P_002000020=Pd_002[0]*Pd_020[2];
	double P_002000120=Pd_002[0]*Pd_120[2];
	double P_002000220=Pd_002[0]*Pd_220[2];
	double P_102000020=Pd_102[0]*Pd_020[2];
	double P_102000120=Pd_102[0]*Pd_120[2];
	double P_102000220=Pd_102[0]*Pd_220[2];
	double P_202000020=Pd_202[0]*Pd_020[2];
	double P_202000120=Pd_202[0]*Pd_120[2];
	double P_202000220=Pd_202[0]*Pd_220[2];
	double P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
	double P_001001120=Pd_001[0]*Pd_001[1]*Pd_120[2];
	double P_001001220=Pd_001[0]*Pd_001[1]*Pd_220[2];
	double P_001101020=Pd_001[0]*Pd_101[1]*Pd_020[2];
	double P_001101120=Pd_001[0]*Pd_101[1]*Pd_120[2];
	double P_001101220=Pd_001[0]*Pd_101[1]*Pd_220[2];
	double P_101001020=Pd_101[0]*Pd_001[1]*Pd_020[2];
	double P_101001120=Pd_101[0]*Pd_001[1]*Pd_120[2];
	double P_101001220=Pd_101[0]*Pd_001[1]*Pd_220[2];
	double P_101101020=Pd_101[0]*Pd_101[1]*Pd_020[2];
	double P_101101120=Pd_101[0]*Pd_101[1]*Pd_120[2];
	double P_101101220=Pd_101[0]*Pd_101[1]*Pd_220[2];
	double P_000002020=Pd_002[1]*Pd_020[2];
	double P_000002120=Pd_002[1]*Pd_120[2];
	double P_000002220=Pd_002[1]*Pd_220[2];
	double P_000102020=Pd_102[1]*Pd_020[2];
	double P_000102120=Pd_102[1]*Pd_120[2];
	double P_000102220=Pd_102[1]*Pd_220[2];
	double P_000202020=Pd_202[1]*Pd_020[2];
	double P_000202120=Pd_202[1]*Pd_120[2];
	double P_000202220=Pd_202[1]*Pd_220[2];
	double P_001000021=Pd_001[0]*Pd_021[2];
	double P_001000121=Pd_001[0]*Pd_121[2];
	double P_001000221=Pd_001[0]*Pd_221[2];
	double P_001000321=Pd_001[0]*Pd_321[2];
	double P_101000021=Pd_101[0]*Pd_021[2];
	double P_101000121=Pd_101[0]*Pd_121[2];
	double P_101000221=Pd_101[0]*Pd_221[2];
	double P_101000321=Pd_101[0]*Pd_321[2];
	double P_000001021=Pd_001[1]*Pd_021[2];
	double P_000001121=Pd_001[1]*Pd_121[2];
	double P_000001221=Pd_001[1]*Pd_221[2];
	double P_000001321=Pd_001[1]*Pd_321[2];
	double P_000101021=Pd_101[1]*Pd_021[2];
	double P_000101121=Pd_101[1]*Pd_121[2];
	double P_000101221=Pd_101[1]*Pd_221[2];
	double P_000101321=Pd_101[1]*Pd_321[2];
	double P_000000022=Pd_022[2];
	double P_000000122=Pd_122[2];
	double P_000000222=Pd_222[2];
	double P_000000322=Pd_322[2];
	double P_000000422=Pd_422[2];
				double PR_022000000000=P_022000000*R_000[0]+-1*P_122000000*R_100[0]+P_222000000*R_200[0]+-1*P_322000000*R_300[0]+P_422000000*R_400[0];
				double PR_021001000000=P_021001000*R_000[0]+-1*P_021101000*R_010[0]+-1*P_121001000*R_100[0]+P_121101000*R_110[0]+P_221001000*R_200[0]+-1*P_221101000*R_210[0]+-1*P_321001000*R_300[0]+P_321101000*R_310[0];
				double PR_020002000000=P_020002000*R_000[0]+-1*P_020102000*R_010[0]+P_020202000*R_020[0]+-1*P_120002000*R_100[0]+P_120102000*R_110[0]+-1*P_120202000*R_120[0]+P_220002000*R_200[0]+-1*P_220102000*R_210[0]+P_220202000*R_220[0];
				double PR_021000001000=P_021000001*R_000[0]+-1*P_021000101*R_001[0]+-1*P_121000001*R_100[0]+P_121000101*R_101[0]+P_221000001*R_200[0]+-1*P_221000101*R_201[0]+-1*P_321000001*R_300[0]+P_321000101*R_301[0];
				double PR_020001001000=P_020001001*R_000[0]+-1*P_020001101*R_001[0]+-1*P_020101001*R_010[0]+P_020101101*R_011[0]+-1*P_120001001*R_100[0]+P_120001101*R_101[0]+P_120101001*R_110[0]+-1*P_120101101*R_111[0]+P_220001001*R_200[0]+-1*P_220001101*R_201[0]+-1*P_220101001*R_210[0]+P_220101101*R_211[0];
				double PR_020000002000=P_020000002*R_000[0]+-1*P_020000102*R_001[0]+P_020000202*R_002[0]+-1*P_120000002*R_100[0]+P_120000102*R_101[0]+-1*P_120000202*R_102[0]+P_220000002*R_200[0]+-1*P_220000102*R_201[0]+P_220000202*R_202[0];
				double PR_012010000000=P_012010000*R_000[0]+-1*P_012110000*R_010[0]+-1*P_112010000*R_100[0]+P_112110000*R_110[0]+P_212010000*R_200[0]+-1*P_212110000*R_210[0]+-1*P_312010000*R_300[0]+P_312110000*R_310[0];
				double PR_011011000000=P_011011000*R_000[0]+-1*P_011111000*R_010[0]+P_011211000*R_020[0]+-1*P_111011000*R_100[0]+P_111111000*R_110[0]+-1*P_111211000*R_120[0]+P_211011000*R_200[0]+-1*P_211111000*R_210[0]+P_211211000*R_220[0];
				double PR_010012000000=P_010012000*R_000[0]+-1*P_010112000*R_010[0]+P_010212000*R_020[0]+-1*P_010312000*R_030[0]+-1*P_110012000*R_100[0]+P_110112000*R_110[0]+-1*P_110212000*R_120[0]+P_110312000*R_130[0];
				double PR_011010001000=P_011010001*R_000[0]+-1*P_011010101*R_001[0]+-1*P_011110001*R_010[0]+P_011110101*R_011[0]+-1*P_111010001*R_100[0]+P_111010101*R_101[0]+P_111110001*R_110[0]+-1*P_111110101*R_111[0]+P_211010001*R_200[0]+-1*P_211010101*R_201[0]+-1*P_211110001*R_210[0]+P_211110101*R_211[0];
				double PR_010011001000=P_010011001*R_000[0]+-1*P_010011101*R_001[0]+-1*P_010111001*R_010[0]+P_010111101*R_011[0]+P_010211001*R_020[0]+-1*P_010211101*R_021[0]+-1*P_110011001*R_100[0]+P_110011101*R_101[0]+P_110111001*R_110[0]+-1*P_110111101*R_111[0]+-1*P_110211001*R_120[0]+P_110211101*R_121[0];
				double PR_010010002000=P_010010002*R_000[0]+-1*P_010010102*R_001[0]+P_010010202*R_002[0]+-1*P_010110002*R_010[0]+P_010110102*R_011[0]+-1*P_010110202*R_012[0]+-1*P_110010002*R_100[0]+P_110010102*R_101[0]+-1*P_110010202*R_102[0]+P_110110002*R_110[0]+-1*P_110110102*R_111[0]+P_110110202*R_112[0];
				double PR_002020000000=P_002020000*R_000[0]+-1*P_002120000*R_010[0]+P_002220000*R_020[0]+-1*P_102020000*R_100[0]+P_102120000*R_110[0]+-1*P_102220000*R_120[0]+P_202020000*R_200[0]+-1*P_202120000*R_210[0]+P_202220000*R_220[0];
				double PR_001021000000=P_001021000*R_000[0]+-1*P_001121000*R_010[0]+P_001221000*R_020[0]+-1*P_001321000*R_030[0]+-1*P_101021000*R_100[0]+P_101121000*R_110[0]+-1*P_101221000*R_120[0]+P_101321000*R_130[0];
				double PR_000022000000=P_000022000*R_000[0]+-1*P_000122000*R_010[0]+P_000222000*R_020[0]+-1*P_000322000*R_030[0]+P_000422000*R_040[0];
				double PR_001020001000=P_001020001*R_000[0]+-1*P_001020101*R_001[0]+-1*P_001120001*R_010[0]+P_001120101*R_011[0]+P_001220001*R_020[0]+-1*P_001220101*R_021[0]+-1*P_101020001*R_100[0]+P_101020101*R_101[0]+P_101120001*R_110[0]+-1*P_101120101*R_111[0]+-1*P_101220001*R_120[0]+P_101220101*R_121[0];
				double PR_000021001000=P_000021001*R_000[0]+-1*P_000021101*R_001[0]+-1*P_000121001*R_010[0]+P_000121101*R_011[0]+P_000221001*R_020[0]+-1*P_000221101*R_021[0]+-1*P_000321001*R_030[0]+P_000321101*R_031[0];
				double PR_000020002000=P_000020002*R_000[0]+-1*P_000020102*R_001[0]+P_000020202*R_002[0]+-1*P_000120002*R_010[0]+P_000120102*R_011[0]+-1*P_000120202*R_012[0]+P_000220002*R_020[0]+-1*P_000220102*R_021[0]+P_000220202*R_022[0];
				double PR_012000010000=P_012000010*R_000[0]+-1*P_012000110*R_001[0]+-1*P_112000010*R_100[0]+P_112000110*R_101[0]+P_212000010*R_200[0]+-1*P_212000110*R_201[0]+-1*P_312000010*R_300[0]+P_312000110*R_301[0];
				double PR_011001010000=P_011001010*R_000[0]+-1*P_011001110*R_001[0]+-1*P_011101010*R_010[0]+P_011101110*R_011[0]+-1*P_111001010*R_100[0]+P_111001110*R_101[0]+P_111101010*R_110[0]+-1*P_111101110*R_111[0]+P_211001010*R_200[0]+-1*P_211001110*R_201[0]+-1*P_211101010*R_210[0]+P_211101110*R_211[0];
				double PR_010002010000=P_010002010*R_000[0]+-1*P_010002110*R_001[0]+-1*P_010102010*R_010[0]+P_010102110*R_011[0]+P_010202010*R_020[0]+-1*P_010202110*R_021[0]+-1*P_110002010*R_100[0]+P_110002110*R_101[0]+P_110102010*R_110[0]+-1*P_110102110*R_111[0]+-1*P_110202010*R_120[0]+P_110202110*R_121[0];
				double PR_011000011000=P_011000011*R_000[0]+-1*P_011000111*R_001[0]+P_011000211*R_002[0]+-1*P_111000011*R_100[0]+P_111000111*R_101[0]+-1*P_111000211*R_102[0]+P_211000011*R_200[0]+-1*P_211000111*R_201[0]+P_211000211*R_202[0];
				double PR_010001011000=P_010001011*R_000[0]+-1*P_010001111*R_001[0]+P_010001211*R_002[0]+-1*P_010101011*R_010[0]+P_010101111*R_011[0]+-1*P_010101211*R_012[0]+-1*P_110001011*R_100[0]+P_110001111*R_101[0]+-1*P_110001211*R_102[0]+P_110101011*R_110[0]+-1*P_110101111*R_111[0]+P_110101211*R_112[0];
				double PR_010000012000=P_010000012*R_000[0]+-1*P_010000112*R_001[0]+P_010000212*R_002[0]+-1*P_010000312*R_003[0]+-1*P_110000012*R_100[0]+P_110000112*R_101[0]+-1*P_110000212*R_102[0]+P_110000312*R_103[0];
				double PR_002010010000=P_002010010*R_000[0]+-1*P_002010110*R_001[0]+-1*P_002110010*R_010[0]+P_002110110*R_011[0]+-1*P_102010010*R_100[0]+P_102010110*R_101[0]+P_102110010*R_110[0]+-1*P_102110110*R_111[0]+P_202010010*R_200[0]+-1*P_202010110*R_201[0]+-1*P_202110010*R_210[0]+P_202110110*R_211[0];
				double PR_001011010000=P_001011010*R_000[0]+-1*P_001011110*R_001[0]+-1*P_001111010*R_010[0]+P_001111110*R_011[0]+P_001211010*R_020[0]+-1*P_001211110*R_021[0]+-1*P_101011010*R_100[0]+P_101011110*R_101[0]+P_101111010*R_110[0]+-1*P_101111110*R_111[0]+-1*P_101211010*R_120[0]+P_101211110*R_121[0];
				double PR_000012010000=P_000012010*R_000[0]+-1*P_000012110*R_001[0]+-1*P_000112010*R_010[0]+P_000112110*R_011[0]+P_000212010*R_020[0]+-1*P_000212110*R_021[0]+-1*P_000312010*R_030[0]+P_000312110*R_031[0];
				double PR_001010011000=P_001010011*R_000[0]+-1*P_001010111*R_001[0]+P_001010211*R_002[0]+-1*P_001110011*R_010[0]+P_001110111*R_011[0]+-1*P_001110211*R_012[0]+-1*P_101010011*R_100[0]+P_101010111*R_101[0]+-1*P_101010211*R_102[0]+P_101110011*R_110[0]+-1*P_101110111*R_111[0]+P_101110211*R_112[0];
				double PR_000011011000=P_000011011*R_000[0]+-1*P_000011111*R_001[0]+P_000011211*R_002[0]+-1*P_000111011*R_010[0]+P_000111111*R_011[0]+-1*P_000111211*R_012[0]+P_000211011*R_020[0]+-1*P_000211111*R_021[0]+P_000211211*R_022[0];
				double PR_000010012000=P_000010012*R_000[0]+-1*P_000010112*R_001[0]+P_000010212*R_002[0]+-1*P_000010312*R_003[0]+-1*P_000110012*R_010[0]+P_000110112*R_011[0]+-1*P_000110212*R_012[0]+P_000110312*R_013[0];
				double PR_002000020000=P_002000020*R_000[0]+-1*P_002000120*R_001[0]+P_002000220*R_002[0]+-1*P_102000020*R_100[0]+P_102000120*R_101[0]+-1*P_102000220*R_102[0]+P_202000020*R_200[0]+-1*P_202000120*R_201[0]+P_202000220*R_202[0];
				double PR_001001020000=P_001001020*R_000[0]+-1*P_001001120*R_001[0]+P_001001220*R_002[0]+-1*P_001101020*R_010[0]+P_001101120*R_011[0]+-1*P_001101220*R_012[0]+-1*P_101001020*R_100[0]+P_101001120*R_101[0]+-1*P_101001220*R_102[0]+P_101101020*R_110[0]+-1*P_101101120*R_111[0]+P_101101220*R_112[0];
				double PR_000002020000=P_000002020*R_000[0]+-1*P_000002120*R_001[0]+P_000002220*R_002[0]+-1*P_000102020*R_010[0]+P_000102120*R_011[0]+-1*P_000102220*R_012[0]+P_000202020*R_020[0]+-1*P_000202120*R_021[0]+P_000202220*R_022[0];
				double PR_001000021000=P_001000021*R_000[0]+-1*P_001000121*R_001[0]+P_001000221*R_002[0]+-1*P_001000321*R_003[0]+-1*P_101000021*R_100[0]+P_101000121*R_101[0]+-1*P_101000221*R_102[0]+P_101000321*R_103[0];
				double PR_000001021000=P_000001021*R_000[0]+-1*P_000001121*R_001[0]+P_000001221*R_002[0]+-1*P_000001321*R_003[0]+-1*P_000101021*R_010[0]+P_000101121*R_011[0]+-1*P_000101221*R_012[0]+P_000101321*R_013[0];
				double PR_000000022000=P_000000022*R_000[0]+-1*P_000000122*R_001[0]+P_000000222*R_002[0]+-1*P_000000322*R_003[0]+P_000000422*R_004[0];
				double PR_022000000001=P_022000000*R_001[0]+-1*P_122000000*R_101[0]+P_222000000*R_201[0]+-1*P_322000000*R_301[0]+P_422000000*R_401[0];
				double PR_021001000001=P_021001000*R_001[0]+-1*P_021101000*R_011[0]+-1*P_121001000*R_101[0]+P_121101000*R_111[0]+P_221001000*R_201[0]+-1*P_221101000*R_211[0]+-1*P_321001000*R_301[0]+P_321101000*R_311[0];
				double PR_020002000001=P_020002000*R_001[0]+-1*P_020102000*R_011[0]+P_020202000*R_021[0]+-1*P_120002000*R_101[0]+P_120102000*R_111[0]+-1*P_120202000*R_121[0]+P_220002000*R_201[0]+-1*P_220102000*R_211[0]+P_220202000*R_221[0];
				double PR_021000001001=P_021000001*R_001[0]+-1*P_021000101*R_002[0]+-1*P_121000001*R_101[0]+P_121000101*R_102[0]+P_221000001*R_201[0]+-1*P_221000101*R_202[0]+-1*P_321000001*R_301[0]+P_321000101*R_302[0];
				double PR_020001001001=P_020001001*R_001[0]+-1*P_020001101*R_002[0]+-1*P_020101001*R_011[0]+P_020101101*R_012[0]+-1*P_120001001*R_101[0]+P_120001101*R_102[0]+P_120101001*R_111[0]+-1*P_120101101*R_112[0]+P_220001001*R_201[0]+-1*P_220001101*R_202[0]+-1*P_220101001*R_211[0]+P_220101101*R_212[0];
				double PR_020000002001=P_020000002*R_001[0]+-1*P_020000102*R_002[0]+P_020000202*R_003[0]+-1*P_120000002*R_101[0]+P_120000102*R_102[0]+-1*P_120000202*R_103[0]+P_220000002*R_201[0]+-1*P_220000102*R_202[0]+P_220000202*R_203[0];
				double PR_012010000001=P_012010000*R_001[0]+-1*P_012110000*R_011[0]+-1*P_112010000*R_101[0]+P_112110000*R_111[0]+P_212010000*R_201[0]+-1*P_212110000*R_211[0]+-1*P_312010000*R_301[0]+P_312110000*R_311[0];
				double PR_011011000001=P_011011000*R_001[0]+-1*P_011111000*R_011[0]+P_011211000*R_021[0]+-1*P_111011000*R_101[0]+P_111111000*R_111[0]+-1*P_111211000*R_121[0]+P_211011000*R_201[0]+-1*P_211111000*R_211[0]+P_211211000*R_221[0];
				double PR_010012000001=P_010012000*R_001[0]+-1*P_010112000*R_011[0]+P_010212000*R_021[0]+-1*P_010312000*R_031[0]+-1*P_110012000*R_101[0]+P_110112000*R_111[0]+-1*P_110212000*R_121[0]+P_110312000*R_131[0];
				double PR_011010001001=P_011010001*R_001[0]+-1*P_011010101*R_002[0]+-1*P_011110001*R_011[0]+P_011110101*R_012[0]+-1*P_111010001*R_101[0]+P_111010101*R_102[0]+P_111110001*R_111[0]+-1*P_111110101*R_112[0]+P_211010001*R_201[0]+-1*P_211010101*R_202[0]+-1*P_211110001*R_211[0]+P_211110101*R_212[0];
				double PR_010011001001=P_010011001*R_001[0]+-1*P_010011101*R_002[0]+-1*P_010111001*R_011[0]+P_010111101*R_012[0]+P_010211001*R_021[0]+-1*P_010211101*R_022[0]+-1*P_110011001*R_101[0]+P_110011101*R_102[0]+P_110111001*R_111[0]+-1*P_110111101*R_112[0]+-1*P_110211001*R_121[0]+P_110211101*R_122[0];
				double PR_010010002001=P_010010002*R_001[0]+-1*P_010010102*R_002[0]+P_010010202*R_003[0]+-1*P_010110002*R_011[0]+P_010110102*R_012[0]+-1*P_010110202*R_013[0]+-1*P_110010002*R_101[0]+P_110010102*R_102[0]+-1*P_110010202*R_103[0]+P_110110002*R_111[0]+-1*P_110110102*R_112[0]+P_110110202*R_113[0];
				double PR_002020000001=P_002020000*R_001[0]+-1*P_002120000*R_011[0]+P_002220000*R_021[0]+-1*P_102020000*R_101[0]+P_102120000*R_111[0]+-1*P_102220000*R_121[0]+P_202020000*R_201[0]+-1*P_202120000*R_211[0]+P_202220000*R_221[0];
				double PR_001021000001=P_001021000*R_001[0]+-1*P_001121000*R_011[0]+P_001221000*R_021[0]+-1*P_001321000*R_031[0]+-1*P_101021000*R_101[0]+P_101121000*R_111[0]+-1*P_101221000*R_121[0]+P_101321000*R_131[0];
				double PR_000022000001=P_000022000*R_001[0]+-1*P_000122000*R_011[0]+P_000222000*R_021[0]+-1*P_000322000*R_031[0]+P_000422000*R_041[0];
				double PR_001020001001=P_001020001*R_001[0]+-1*P_001020101*R_002[0]+-1*P_001120001*R_011[0]+P_001120101*R_012[0]+P_001220001*R_021[0]+-1*P_001220101*R_022[0]+-1*P_101020001*R_101[0]+P_101020101*R_102[0]+P_101120001*R_111[0]+-1*P_101120101*R_112[0]+-1*P_101220001*R_121[0]+P_101220101*R_122[0];
				double PR_000021001001=P_000021001*R_001[0]+-1*P_000021101*R_002[0]+-1*P_000121001*R_011[0]+P_000121101*R_012[0]+P_000221001*R_021[0]+-1*P_000221101*R_022[0]+-1*P_000321001*R_031[0]+P_000321101*R_032[0];
				double PR_000020002001=P_000020002*R_001[0]+-1*P_000020102*R_002[0]+P_000020202*R_003[0]+-1*P_000120002*R_011[0]+P_000120102*R_012[0]+-1*P_000120202*R_013[0]+P_000220002*R_021[0]+-1*P_000220102*R_022[0]+P_000220202*R_023[0];
				double PR_012000010001=P_012000010*R_001[0]+-1*P_012000110*R_002[0]+-1*P_112000010*R_101[0]+P_112000110*R_102[0]+P_212000010*R_201[0]+-1*P_212000110*R_202[0]+-1*P_312000010*R_301[0]+P_312000110*R_302[0];
				double PR_011001010001=P_011001010*R_001[0]+-1*P_011001110*R_002[0]+-1*P_011101010*R_011[0]+P_011101110*R_012[0]+-1*P_111001010*R_101[0]+P_111001110*R_102[0]+P_111101010*R_111[0]+-1*P_111101110*R_112[0]+P_211001010*R_201[0]+-1*P_211001110*R_202[0]+-1*P_211101010*R_211[0]+P_211101110*R_212[0];
				double PR_010002010001=P_010002010*R_001[0]+-1*P_010002110*R_002[0]+-1*P_010102010*R_011[0]+P_010102110*R_012[0]+P_010202010*R_021[0]+-1*P_010202110*R_022[0]+-1*P_110002010*R_101[0]+P_110002110*R_102[0]+P_110102010*R_111[0]+-1*P_110102110*R_112[0]+-1*P_110202010*R_121[0]+P_110202110*R_122[0];
				double PR_011000011001=P_011000011*R_001[0]+-1*P_011000111*R_002[0]+P_011000211*R_003[0]+-1*P_111000011*R_101[0]+P_111000111*R_102[0]+-1*P_111000211*R_103[0]+P_211000011*R_201[0]+-1*P_211000111*R_202[0]+P_211000211*R_203[0];
				double PR_010001011001=P_010001011*R_001[0]+-1*P_010001111*R_002[0]+P_010001211*R_003[0]+-1*P_010101011*R_011[0]+P_010101111*R_012[0]+-1*P_010101211*R_013[0]+-1*P_110001011*R_101[0]+P_110001111*R_102[0]+-1*P_110001211*R_103[0]+P_110101011*R_111[0]+-1*P_110101111*R_112[0]+P_110101211*R_113[0];
				double PR_010000012001=P_010000012*R_001[0]+-1*P_010000112*R_002[0]+P_010000212*R_003[0]+-1*P_010000312*R_004[0]+-1*P_110000012*R_101[0]+P_110000112*R_102[0]+-1*P_110000212*R_103[0]+P_110000312*R_104[0];
				double PR_002010010001=P_002010010*R_001[0]+-1*P_002010110*R_002[0]+-1*P_002110010*R_011[0]+P_002110110*R_012[0]+-1*P_102010010*R_101[0]+P_102010110*R_102[0]+P_102110010*R_111[0]+-1*P_102110110*R_112[0]+P_202010010*R_201[0]+-1*P_202010110*R_202[0]+-1*P_202110010*R_211[0]+P_202110110*R_212[0];
				double PR_001011010001=P_001011010*R_001[0]+-1*P_001011110*R_002[0]+-1*P_001111010*R_011[0]+P_001111110*R_012[0]+P_001211010*R_021[0]+-1*P_001211110*R_022[0]+-1*P_101011010*R_101[0]+P_101011110*R_102[0]+P_101111010*R_111[0]+-1*P_101111110*R_112[0]+-1*P_101211010*R_121[0]+P_101211110*R_122[0];
				double PR_000012010001=P_000012010*R_001[0]+-1*P_000012110*R_002[0]+-1*P_000112010*R_011[0]+P_000112110*R_012[0]+P_000212010*R_021[0]+-1*P_000212110*R_022[0]+-1*P_000312010*R_031[0]+P_000312110*R_032[0];
				double PR_001010011001=P_001010011*R_001[0]+-1*P_001010111*R_002[0]+P_001010211*R_003[0]+-1*P_001110011*R_011[0]+P_001110111*R_012[0]+-1*P_001110211*R_013[0]+-1*P_101010011*R_101[0]+P_101010111*R_102[0]+-1*P_101010211*R_103[0]+P_101110011*R_111[0]+-1*P_101110111*R_112[0]+P_101110211*R_113[0];
				double PR_000011011001=P_000011011*R_001[0]+-1*P_000011111*R_002[0]+P_000011211*R_003[0]+-1*P_000111011*R_011[0]+P_000111111*R_012[0]+-1*P_000111211*R_013[0]+P_000211011*R_021[0]+-1*P_000211111*R_022[0]+P_000211211*R_023[0];
				double PR_000010012001=P_000010012*R_001[0]+-1*P_000010112*R_002[0]+P_000010212*R_003[0]+-1*P_000010312*R_004[0]+-1*P_000110012*R_011[0]+P_000110112*R_012[0]+-1*P_000110212*R_013[0]+P_000110312*R_014[0];
				double PR_002000020001=P_002000020*R_001[0]+-1*P_002000120*R_002[0]+P_002000220*R_003[0]+-1*P_102000020*R_101[0]+P_102000120*R_102[0]+-1*P_102000220*R_103[0]+P_202000020*R_201[0]+-1*P_202000120*R_202[0]+P_202000220*R_203[0];
				double PR_001001020001=P_001001020*R_001[0]+-1*P_001001120*R_002[0]+P_001001220*R_003[0]+-1*P_001101020*R_011[0]+P_001101120*R_012[0]+-1*P_001101220*R_013[0]+-1*P_101001020*R_101[0]+P_101001120*R_102[0]+-1*P_101001220*R_103[0]+P_101101020*R_111[0]+-1*P_101101120*R_112[0]+P_101101220*R_113[0];
				double PR_000002020001=P_000002020*R_001[0]+-1*P_000002120*R_002[0]+P_000002220*R_003[0]+-1*P_000102020*R_011[0]+P_000102120*R_012[0]+-1*P_000102220*R_013[0]+P_000202020*R_021[0]+-1*P_000202120*R_022[0]+P_000202220*R_023[0];
				double PR_001000021001=P_001000021*R_001[0]+-1*P_001000121*R_002[0]+P_001000221*R_003[0]+-1*P_001000321*R_004[0]+-1*P_101000021*R_101[0]+P_101000121*R_102[0]+-1*P_101000221*R_103[0]+P_101000321*R_104[0];
				double PR_000001021001=P_000001021*R_001[0]+-1*P_000001121*R_002[0]+P_000001221*R_003[0]+-1*P_000001321*R_004[0]+-1*P_000101021*R_011[0]+P_000101121*R_012[0]+-1*P_000101221*R_013[0]+P_000101321*R_014[0];
				double PR_000000022001=P_000000022*R_001[0]+-1*P_000000122*R_002[0]+P_000000222*R_003[0]+-1*P_000000322*R_004[0]+P_000000422*R_005[0];
				double PR_022000000010=P_022000000*R_010[0]+-1*P_122000000*R_110[0]+P_222000000*R_210[0]+-1*P_322000000*R_310[0]+P_422000000*R_410[0];
				double PR_021001000010=P_021001000*R_010[0]+-1*P_021101000*R_020[0]+-1*P_121001000*R_110[0]+P_121101000*R_120[0]+P_221001000*R_210[0]+-1*P_221101000*R_220[0]+-1*P_321001000*R_310[0]+P_321101000*R_320[0];
				double PR_020002000010=P_020002000*R_010[0]+-1*P_020102000*R_020[0]+P_020202000*R_030[0]+-1*P_120002000*R_110[0]+P_120102000*R_120[0]+-1*P_120202000*R_130[0]+P_220002000*R_210[0]+-1*P_220102000*R_220[0]+P_220202000*R_230[0];
				double PR_021000001010=P_021000001*R_010[0]+-1*P_021000101*R_011[0]+-1*P_121000001*R_110[0]+P_121000101*R_111[0]+P_221000001*R_210[0]+-1*P_221000101*R_211[0]+-1*P_321000001*R_310[0]+P_321000101*R_311[0];
				double PR_020001001010=P_020001001*R_010[0]+-1*P_020001101*R_011[0]+-1*P_020101001*R_020[0]+P_020101101*R_021[0]+-1*P_120001001*R_110[0]+P_120001101*R_111[0]+P_120101001*R_120[0]+-1*P_120101101*R_121[0]+P_220001001*R_210[0]+-1*P_220001101*R_211[0]+-1*P_220101001*R_220[0]+P_220101101*R_221[0];
				double PR_020000002010=P_020000002*R_010[0]+-1*P_020000102*R_011[0]+P_020000202*R_012[0]+-1*P_120000002*R_110[0]+P_120000102*R_111[0]+-1*P_120000202*R_112[0]+P_220000002*R_210[0]+-1*P_220000102*R_211[0]+P_220000202*R_212[0];
				double PR_012010000010=P_012010000*R_010[0]+-1*P_012110000*R_020[0]+-1*P_112010000*R_110[0]+P_112110000*R_120[0]+P_212010000*R_210[0]+-1*P_212110000*R_220[0]+-1*P_312010000*R_310[0]+P_312110000*R_320[0];
				double PR_011011000010=P_011011000*R_010[0]+-1*P_011111000*R_020[0]+P_011211000*R_030[0]+-1*P_111011000*R_110[0]+P_111111000*R_120[0]+-1*P_111211000*R_130[0]+P_211011000*R_210[0]+-1*P_211111000*R_220[0]+P_211211000*R_230[0];
				double PR_010012000010=P_010012000*R_010[0]+-1*P_010112000*R_020[0]+P_010212000*R_030[0]+-1*P_010312000*R_040[0]+-1*P_110012000*R_110[0]+P_110112000*R_120[0]+-1*P_110212000*R_130[0]+P_110312000*R_140[0];
				double PR_011010001010=P_011010001*R_010[0]+-1*P_011010101*R_011[0]+-1*P_011110001*R_020[0]+P_011110101*R_021[0]+-1*P_111010001*R_110[0]+P_111010101*R_111[0]+P_111110001*R_120[0]+-1*P_111110101*R_121[0]+P_211010001*R_210[0]+-1*P_211010101*R_211[0]+-1*P_211110001*R_220[0]+P_211110101*R_221[0];
				double PR_010011001010=P_010011001*R_010[0]+-1*P_010011101*R_011[0]+-1*P_010111001*R_020[0]+P_010111101*R_021[0]+P_010211001*R_030[0]+-1*P_010211101*R_031[0]+-1*P_110011001*R_110[0]+P_110011101*R_111[0]+P_110111001*R_120[0]+-1*P_110111101*R_121[0]+-1*P_110211001*R_130[0]+P_110211101*R_131[0];
				double PR_010010002010=P_010010002*R_010[0]+-1*P_010010102*R_011[0]+P_010010202*R_012[0]+-1*P_010110002*R_020[0]+P_010110102*R_021[0]+-1*P_010110202*R_022[0]+-1*P_110010002*R_110[0]+P_110010102*R_111[0]+-1*P_110010202*R_112[0]+P_110110002*R_120[0]+-1*P_110110102*R_121[0]+P_110110202*R_122[0];
				double PR_002020000010=P_002020000*R_010[0]+-1*P_002120000*R_020[0]+P_002220000*R_030[0]+-1*P_102020000*R_110[0]+P_102120000*R_120[0]+-1*P_102220000*R_130[0]+P_202020000*R_210[0]+-1*P_202120000*R_220[0]+P_202220000*R_230[0];
				double PR_001021000010=P_001021000*R_010[0]+-1*P_001121000*R_020[0]+P_001221000*R_030[0]+-1*P_001321000*R_040[0]+-1*P_101021000*R_110[0]+P_101121000*R_120[0]+-1*P_101221000*R_130[0]+P_101321000*R_140[0];
				double PR_000022000010=P_000022000*R_010[0]+-1*P_000122000*R_020[0]+P_000222000*R_030[0]+-1*P_000322000*R_040[0]+P_000422000*R_050[0];
				double PR_001020001010=P_001020001*R_010[0]+-1*P_001020101*R_011[0]+-1*P_001120001*R_020[0]+P_001120101*R_021[0]+P_001220001*R_030[0]+-1*P_001220101*R_031[0]+-1*P_101020001*R_110[0]+P_101020101*R_111[0]+P_101120001*R_120[0]+-1*P_101120101*R_121[0]+-1*P_101220001*R_130[0]+P_101220101*R_131[0];
				double PR_000021001010=P_000021001*R_010[0]+-1*P_000021101*R_011[0]+-1*P_000121001*R_020[0]+P_000121101*R_021[0]+P_000221001*R_030[0]+-1*P_000221101*R_031[0]+-1*P_000321001*R_040[0]+P_000321101*R_041[0];
				double PR_000020002010=P_000020002*R_010[0]+-1*P_000020102*R_011[0]+P_000020202*R_012[0]+-1*P_000120002*R_020[0]+P_000120102*R_021[0]+-1*P_000120202*R_022[0]+P_000220002*R_030[0]+-1*P_000220102*R_031[0]+P_000220202*R_032[0];
				double PR_012000010010=P_012000010*R_010[0]+-1*P_012000110*R_011[0]+-1*P_112000010*R_110[0]+P_112000110*R_111[0]+P_212000010*R_210[0]+-1*P_212000110*R_211[0]+-1*P_312000010*R_310[0]+P_312000110*R_311[0];
				double PR_011001010010=P_011001010*R_010[0]+-1*P_011001110*R_011[0]+-1*P_011101010*R_020[0]+P_011101110*R_021[0]+-1*P_111001010*R_110[0]+P_111001110*R_111[0]+P_111101010*R_120[0]+-1*P_111101110*R_121[0]+P_211001010*R_210[0]+-1*P_211001110*R_211[0]+-1*P_211101010*R_220[0]+P_211101110*R_221[0];
				double PR_010002010010=P_010002010*R_010[0]+-1*P_010002110*R_011[0]+-1*P_010102010*R_020[0]+P_010102110*R_021[0]+P_010202010*R_030[0]+-1*P_010202110*R_031[0]+-1*P_110002010*R_110[0]+P_110002110*R_111[0]+P_110102010*R_120[0]+-1*P_110102110*R_121[0]+-1*P_110202010*R_130[0]+P_110202110*R_131[0];
				double PR_011000011010=P_011000011*R_010[0]+-1*P_011000111*R_011[0]+P_011000211*R_012[0]+-1*P_111000011*R_110[0]+P_111000111*R_111[0]+-1*P_111000211*R_112[0]+P_211000011*R_210[0]+-1*P_211000111*R_211[0]+P_211000211*R_212[0];
				double PR_010001011010=P_010001011*R_010[0]+-1*P_010001111*R_011[0]+P_010001211*R_012[0]+-1*P_010101011*R_020[0]+P_010101111*R_021[0]+-1*P_010101211*R_022[0]+-1*P_110001011*R_110[0]+P_110001111*R_111[0]+-1*P_110001211*R_112[0]+P_110101011*R_120[0]+-1*P_110101111*R_121[0]+P_110101211*R_122[0];
				double PR_010000012010=P_010000012*R_010[0]+-1*P_010000112*R_011[0]+P_010000212*R_012[0]+-1*P_010000312*R_013[0]+-1*P_110000012*R_110[0]+P_110000112*R_111[0]+-1*P_110000212*R_112[0]+P_110000312*R_113[0];
				double PR_002010010010=P_002010010*R_010[0]+-1*P_002010110*R_011[0]+-1*P_002110010*R_020[0]+P_002110110*R_021[0]+-1*P_102010010*R_110[0]+P_102010110*R_111[0]+P_102110010*R_120[0]+-1*P_102110110*R_121[0]+P_202010010*R_210[0]+-1*P_202010110*R_211[0]+-1*P_202110010*R_220[0]+P_202110110*R_221[0];
				double PR_001011010010=P_001011010*R_010[0]+-1*P_001011110*R_011[0]+-1*P_001111010*R_020[0]+P_001111110*R_021[0]+P_001211010*R_030[0]+-1*P_001211110*R_031[0]+-1*P_101011010*R_110[0]+P_101011110*R_111[0]+P_101111010*R_120[0]+-1*P_101111110*R_121[0]+-1*P_101211010*R_130[0]+P_101211110*R_131[0];
				double PR_000012010010=P_000012010*R_010[0]+-1*P_000012110*R_011[0]+-1*P_000112010*R_020[0]+P_000112110*R_021[0]+P_000212010*R_030[0]+-1*P_000212110*R_031[0]+-1*P_000312010*R_040[0]+P_000312110*R_041[0];
				double PR_001010011010=P_001010011*R_010[0]+-1*P_001010111*R_011[0]+P_001010211*R_012[0]+-1*P_001110011*R_020[0]+P_001110111*R_021[0]+-1*P_001110211*R_022[0]+-1*P_101010011*R_110[0]+P_101010111*R_111[0]+-1*P_101010211*R_112[0]+P_101110011*R_120[0]+-1*P_101110111*R_121[0]+P_101110211*R_122[0];
				double PR_000011011010=P_000011011*R_010[0]+-1*P_000011111*R_011[0]+P_000011211*R_012[0]+-1*P_000111011*R_020[0]+P_000111111*R_021[0]+-1*P_000111211*R_022[0]+P_000211011*R_030[0]+-1*P_000211111*R_031[0]+P_000211211*R_032[0];
				double PR_000010012010=P_000010012*R_010[0]+-1*P_000010112*R_011[0]+P_000010212*R_012[0]+-1*P_000010312*R_013[0]+-1*P_000110012*R_020[0]+P_000110112*R_021[0]+-1*P_000110212*R_022[0]+P_000110312*R_023[0];
				double PR_002000020010=P_002000020*R_010[0]+-1*P_002000120*R_011[0]+P_002000220*R_012[0]+-1*P_102000020*R_110[0]+P_102000120*R_111[0]+-1*P_102000220*R_112[0]+P_202000020*R_210[0]+-1*P_202000120*R_211[0]+P_202000220*R_212[0];
				double PR_001001020010=P_001001020*R_010[0]+-1*P_001001120*R_011[0]+P_001001220*R_012[0]+-1*P_001101020*R_020[0]+P_001101120*R_021[0]+-1*P_001101220*R_022[0]+-1*P_101001020*R_110[0]+P_101001120*R_111[0]+-1*P_101001220*R_112[0]+P_101101020*R_120[0]+-1*P_101101120*R_121[0]+P_101101220*R_122[0];
				double PR_000002020010=P_000002020*R_010[0]+-1*P_000002120*R_011[0]+P_000002220*R_012[0]+-1*P_000102020*R_020[0]+P_000102120*R_021[0]+-1*P_000102220*R_022[0]+P_000202020*R_030[0]+-1*P_000202120*R_031[0]+P_000202220*R_032[0];
				double PR_001000021010=P_001000021*R_010[0]+-1*P_001000121*R_011[0]+P_001000221*R_012[0]+-1*P_001000321*R_013[0]+-1*P_101000021*R_110[0]+P_101000121*R_111[0]+-1*P_101000221*R_112[0]+P_101000321*R_113[0];
				double PR_000001021010=P_000001021*R_010[0]+-1*P_000001121*R_011[0]+P_000001221*R_012[0]+-1*P_000001321*R_013[0]+-1*P_000101021*R_020[0]+P_000101121*R_021[0]+-1*P_000101221*R_022[0]+P_000101321*R_023[0];
				double PR_000000022010=P_000000022*R_010[0]+-1*P_000000122*R_011[0]+P_000000222*R_012[0]+-1*P_000000322*R_013[0]+P_000000422*R_014[0];
				double PR_022000000100=P_022000000*R_100[0]+-1*P_122000000*R_200[0]+P_222000000*R_300[0]+-1*P_322000000*R_400[0]+P_422000000*R_500[0];
				double PR_021001000100=P_021001000*R_100[0]+-1*P_021101000*R_110[0]+-1*P_121001000*R_200[0]+P_121101000*R_210[0]+P_221001000*R_300[0]+-1*P_221101000*R_310[0]+-1*P_321001000*R_400[0]+P_321101000*R_410[0];
				double PR_020002000100=P_020002000*R_100[0]+-1*P_020102000*R_110[0]+P_020202000*R_120[0]+-1*P_120002000*R_200[0]+P_120102000*R_210[0]+-1*P_120202000*R_220[0]+P_220002000*R_300[0]+-1*P_220102000*R_310[0]+P_220202000*R_320[0];
				double PR_021000001100=P_021000001*R_100[0]+-1*P_021000101*R_101[0]+-1*P_121000001*R_200[0]+P_121000101*R_201[0]+P_221000001*R_300[0]+-1*P_221000101*R_301[0]+-1*P_321000001*R_400[0]+P_321000101*R_401[0];
				double PR_020001001100=P_020001001*R_100[0]+-1*P_020001101*R_101[0]+-1*P_020101001*R_110[0]+P_020101101*R_111[0]+-1*P_120001001*R_200[0]+P_120001101*R_201[0]+P_120101001*R_210[0]+-1*P_120101101*R_211[0]+P_220001001*R_300[0]+-1*P_220001101*R_301[0]+-1*P_220101001*R_310[0]+P_220101101*R_311[0];
				double PR_020000002100=P_020000002*R_100[0]+-1*P_020000102*R_101[0]+P_020000202*R_102[0]+-1*P_120000002*R_200[0]+P_120000102*R_201[0]+-1*P_120000202*R_202[0]+P_220000002*R_300[0]+-1*P_220000102*R_301[0]+P_220000202*R_302[0];
				double PR_012010000100=P_012010000*R_100[0]+-1*P_012110000*R_110[0]+-1*P_112010000*R_200[0]+P_112110000*R_210[0]+P_212010000*R_300[0]+-1*P_212110000*R_310[0]+-1*P_312010000*R_400[0]+P_312110000*R_410[0];
				double PR_011011000100=P_011011000*R_100[0]+-1*P_011111000*R_110[0]+P_011211000*R_120[0]+-1*P_111011000*R_200[0]+P_111111000*R_210[0]+-1*P_111211000*R_220[0]+P_211011000*R_300[0]+-1*P_211111000*R_310[0]+P_211211000*R_320[0];
				double PR_010012000100=P_010012000*R_100[0]+-1*P_010112000*R_110[0]+P_010212000*R_120[0]+-1*P_010312000*R_130[0]+-1*P_110012000*R_200[0]+P_110112000*R_210[0]+-1*P_110212000*R_220[0]+P_110312000*R_230[0];
				double PR_011010001100=P_011010001*R_100[0]+-1*P_011010101*R_101[0]+-1*P_011110001*R_110[0]+P_011110101*R_111[0]+-1*P_111010001*R_200[0]+P_111010101*R_201[0]+P_111110001*R_210[0]+-1*P_111110101*R_211[0]+P_211010001*R_300[0]+-1*P_211010101*R_301[0]+-1*P_211110001*R_310[0]+P_211110101*R_311[0];
				double PR_010011001100=P_010011001*R_100[0]+-1*P_010011101*R_101[0]+-1*P_010111001*R_110[0]+P_010111101*R_111[0]+P_010211001*R_120[0]+-1*P_010211101*R_121[0]+-1*P_110011001*R_200[0]+P_110011101*R_201[0]+P_110111001*R_210[0]+-1*P_110111101*R_211[0]+-1*P_110211001*R_220[0]+P_110211101*R_221[0];
				double PR_010010002100=P_010010002*R_100[0]+-1*P_010010102*R_101[0]+P_010010202*R_102[0]+-1*P_010110002*R_110[0]+P_010110102*R_111[0]+-1*P_010110202*R_112[0]+-1*P_110010002*R_200[0]+P_110010102*R_201[0]+-1*P_110010202*R_202[0]+P_110110002*R_210[0]+-1*P_110110102*R_211[0]+P_110110202*R_212[0];
				double PR_002020000100=P_002020000*R_100[0]+-1*P_002120000*R_110[0]+P_002220000*R_120[0]+-1*P_102020000*R_200[0]+P_102120000*R_210[0]+-1*P_102220000*R_220[0]+P_202020000*R_300[0]+-1*P_202120000*R_310[0]+P_202220000*R_320[0];
				double PR_001021000100=P_001021000*R_100[0]+-1*P_001121000*R_110[0]+P_001221000*R_120[0]+-1*P_001321000*R_130[0]+-1*P_101021000*R_200[0]+P_101121000*R_210[0]+-1*P_101221000*R_220[0]+P_101321000*R_230[0];
				double PR_000022000100=P_000022000*R_100[0]+-1*P_000122000*R_110[0]+P_000222000*R_120[0]+-1*P_000322000*R_130[0]+P_000422000*R_140[0];
				double PR_001020001100=P_001020001*R_100[0]+-1*P_001020101*R_101[0]+-1*P_001120001*R_110[0]+P_001120101*R_111[0]+P_001220001*R_120[0]+-1*P_001220101*R_121[0]+-1*P_101020001*R_200[0]+P_101020101*R_201[0]+P_101120001*R_210[0]+-1*P_101120101*R_211[0]+-1*P_101220001*R_220[0]+P_101220101*R_221[0];
				double PR_000021001100=P_000021001*R_100[0]+-1*P_000021101*R_101[0]+-1*P_000121001*R_110[0]+P_000121101*R_111[0]+P_000221001*R_120[0]+-1*P_000221101*R_121[0]+-1*P_000321001*R_130[0]+P_000321101*R_131[0];
				double PR_000020002100=P_000020002*R_100[0]+-1*P_000020102*R_101[0]+P_000020202*R_102[0]+-1*P_000120002*R_110[0]+P_000120102*R_111[0]+-1*P_000120202*R_112[0]+P_000220002*R_120[0]+-1*P_000220102*R_121[0]+P_000220202*R_122[0];
				double PR_012000010100=P_012000010*R_100[0]+-1*P_012000110*R_101[0]+-1*P_112000010*R_200[0]+P_112000110*R_201[0]+P_212000010*R_300[0]+-1*P_212000110*R_301[0]+-1*P_312000010*R_400[0]+P_312000110*R_401[0];
				double PR_011001010100=P_011001010*R_100[0]+-1*P_011001110*R_101[0]+-1*P_011101010*R_110[0]+P_011101110*R_111[0]+-1*P_111001010*R_200[0]+P_111001110*R_201[0]+P_111101010*R_210[0]+-1*P_111101110*R_211[0]+P_211001010*R_300[0]+-1*P_211001110*R_301[0]+-1*P_211101010*R_310[0]+P_211101110*R_311[0];
				double PR_010002010100=P_010002010*R_100[0]+-1*P_010002110*R_101[0]+-1*P_010102010*R_110[0]+P_010102110*R_111[0]+P_010202010*R_120[0]+-1*P_010202110*R_121[0]+-1*P_110002010*R_200[0]+P_110002110*R_201[0]+P_110102010*R_210[0]+-1*P_110102110*R_211[0]+-1*P_110202010*R_220[0]+P_110202110*R_221[0];
				double PR_011000011100=P_011000011*R_100[0]+-1*P_011000111*R_101[0]+P_011000211*R_102[0]+-1*P_111000011*R_200[0]+P_111000111*R_201[0]+-1*P_111000211*R_202[0]+P_211000011*R_300[0]+-1*P_211000111*R_301[0]+P_211000211*R_302[0];
				double PR_010001011100=P_010001011*R_100[0]+-1*P_010001111*R_101[0]+P_010001211*R_102[0]+-1*P_010101011*R_110[0]+P_010101111*R_111[0]+-1*P_010101211*R_112[0]+-1*P_110001011*R_200[0]+P_110001111*R_201[0]+-1*P_110001211*R_202[0]+P_110101011*R_210[0]+-1*P_110101111*R_211[0]+P_110101211*R_212[0];
				double PR_010000012100=P_010000012*R_100[0]+-1*P_010000112*R_101[0]+P_010000212*R_102[0]+-1*P_010000312*R_103[0]+-1*P_110000012*R_200[0]+P_110000112*R_201[0]+-1*P_110000212*R_202[0]+P_110000312*R_203[0];
				double PR_002010010100=P_002010010*R_100[0]+-1*P_002010110*R_101[0]+-1*P_002110010*R_110[0]+P_002110110*R_111[0]+-1*P_102010010*R_200[0]+P_102010110*R_201[0]+P_102110010*R_210[0]+-1*P_102110110*R_211[0]+P_202010010*R_300[0]+-1*P_202010110*R_301[0]+-1*P_202110010*R_310[0]+P_202110110*R_311[0];
				double PR_001011010100=P_001011010*R_100[0]+-1*P_001011110*R_101[0]+-1*P_001111010*R_110[0]+P_001111110*R_111[0]+P_001211010*R_120[0]+-1*P_001211110*R_121[0]+-1*P_101011010*R_200[0]+P_101011110*R_201[0]+P_101111010*R_210[0]+-1*P_101111110*R_211[0]+-1*P_101211010*R_220[0]+P_101211110*R_221[0];
				double PR_000012010100=P_000012010*R_100[0]+-1*P_000012110*R_101[0]+-1*P_000112010*R_110[0]+P_000112110*R_111[0]+P_000212010*R_120[0]+-1*P_000212110*R_121[0]+-1*P_000312010*R_130[0]+P_000312110*R_131[0];
				double PR_001010011100=P_001010011*R_100[0]+-1*P_001010111*R_101[0]+P_001010211*R_102[0]+-1*P_001110011*R_110[0]+P_001110111*R_111[0]+-1*P_001110211*R_112[0]+-1*P_101010011*R_200[0]+P_101010111*R_201[0]+-1*P_101010211*R_202[0]+P_101110011*R_210[0]+-1*P_101110111*R_211[0]+P_101110211*R_212[0];
				double PR_000011011100=P_000011011*R_100[0]+-1*P_000011111*R_101[0]+P_000011211*R_102[0]+-1*P_000111011*R_110[0]+P_000111111*R_111[0]+-1*P_000111211*R_112[0]+P_000211011*R_120[0]+-1*P_000211111*R_121[0]+P_000211211*R_122[0];
				double PR_000010012100=P_000010012*R_100[0]+-1*P_000010112*R_101[0]+P_000010212*R_102[0]+-1*P_000010312*R_103[0]+-1*P_000110012*R_110[0]+P_000110112*R_111[0]+-1*P_000110212*R_112[0]+P_000110312*R_113[0];
				double PR_002000020100=P_002000020*R_100[0]+-1*P_002000120*R_101[0]+P_002000220*R_102[0]+-1*P_102000020*R_200[0]+P_102000120*R_201[0]+-1*P_102000220*R_202[0]+P_202000020*R_300[0]+-1*P_202000120*R_301[0]+P_202000220*R_302[0];
				double PR_001001020100=P_001001020*R_100[0]+-1*P_001001120*R_101[0]+P_001001220*R_102[0]+-1*P_001101020*R_110[0]+P_001101120*R_111[0]+-1*P_001101220*R_112[0]+-1*P_101001020*R_200[0]+P_101001120*R_201[0]+-1*P_101001220*R_202[0]+P_101101020*R_210[0]+-1*P_101101120*R_211[0]+P_101101220*R_212[0];
				double PR_000002020100=P_000002020*R_100[0]+-1*P_000002120*R_101[0]+P_000002220*R_102[0]+-1*P_000102020*R_110[0]+P_000102120*R_111[0]+-1*P_000102220*R_112[0]+P_000202020*R_120[0]+-1*P_000202120*R_121[0]+P_000202220*R_122[0];
				double PR_001000021100=P_001000021*R_100[0]+-1*P_001000121*R_101[0]+P_001000221*R_102[0]+-1*P_001000321*R_103[0]+-1*P_101000021*R_200[0]+P_101000121*R_201[0]+-1*P_101000221*R_202[0]+P_101000321*R_203[0];
				double PR_000001021100=P_000001021*R_100[0]+-1*P_000001121*R_101[0]+P_000001221*R_102[0]+-1*P_000001321*R_103[0]+-1*P_000101021*R_110[0]+P_000101121*R_111[0]+-1*P_000101221*R_112[0]+P_000101321*R_113[0];
				double PR_000000022100=P_000000022*R_100[0]+-1*P_000000122*R_101[0]+P_000000222*R_102[0]+-1*P_000000322*R_103[0]+P_000000422*R_104[0];
				double PR_022000000002=P_022000000*R_002[0]+-1*P_122000000*R_102[0]+P_222000000*R_202[0]+-1*P_322000000*R_302[0]+P_422000000*R_402[0];
				double PR_021001000002=P_021001000*R_002[0]+-1*P_021101000*R_012[0]+-1*P_121001000*R_102[0]+P_121101000*R_112[0]+P_221001000*R_202[0]+-1*P_221101000*R_212[0]+-1*P_321001000*R_302[0]+P_321101000*R_312[0];
				double PR_020002000002=P_020002000*R_002[0]+-1*P_020102000*R_012[0]+P_020202000*R_022[0]+-1*P_120002000*R_102[0]+P_120102000*R_112[0]+-1*P_120202000*R_122[0]+P_220002000*R_202[0]+-1*P_220102000*R_212[0]+P_220202000*R_222[0];
				double PR_021000001002=P_021000001*R_002[0]+-1*P_021000101*R_003[0]+-1*P_121000001*R_102[0]+P_121000101*R_103[0]+P_221000001*R_202[0]+-1*P_221000101*R_203[0]+-1*P_321000001*R_302[0]+P_321000101*R_303[0];
				double PR_020001001002=P_020001001*R_002[0]+-1*P_020001101*R_003[0]+-1*P_020101001*R_012[0]+P_020101101*R_013[0]+-1*P_120001001*R_102[0]+P_120001101*R_103[0]+P_120101001*R_112[0]+-1*P_120101101*R_113[0]+P_220001001*R_202[0]+-1*P_220001101*R_203[0]+-1*P_220101001*R_212[0]+P_220101101*R_213[0];
				double PR_020000002002=P_020000002*R_002[0]+-1*P_020000102*R_003[0]+P_020000202*R_004[0]+-1*P_120000002*R_102[0]+P_120000102*R_103[0]+-1*P_120000202*R_104[0]+P_220000002*R_202[0]+-1*P_220000102*R_203[0]+P_220000202*R_204[0];
				double PR_012010000002=P_012010000*R_002[0]+-1*P_012110000*R_012[0]+-1*P_112010000*R_102[0]+P_112110000*R_112[0]+P_212010000*R_202[0]+-1*P_212110000*R_212[0]+-1*P_312010000*R_302[0]+P_312110000*R_312[0];
				double PR_011011000002=P_011011000*R_002[0]+-1*P_011111000*R_012[0]+P_011211000*R_022[0]+-1*P_111011000*R_102[0]+P_111111000*R_112[0]+-1*P_111211000*R_122[0]+P_211011000*R_202[0]+-1*P_211111000*R_212[0]+P_211211000*R_222[0];
				double PR_010012000002=P_010012000*R_002[0]+-1*P_010112000*R_012[0]+P_010212000*R_022[0]+-1*P_010312000*R_032[0]+-1*P_110012000*R_102[0]+P_110112000*R_112[0]+-1*P_110212000*R_122[0]+P_110312000*R_132[0];
				double PR_011010001002=P_011010001*R_002[0]+-1*P_011010101*R_003[0]+-1*P_011110001*R_012[0]+P_011110101*R_013[0]+-1*P_111010001*R_102[0]+P_111010101*R_103[0]+P_111110001*R_112[0]+-1*P_111110101*R_113[0]+P_211010001*R_202[0]+-1*P_211010101*R_203[0]+-1*P_211110001*R_212[0]+P_211110101*R_213[0];
				double PR_010011001002=P_010011001*R_002[0]+-1*P_010011101*R_003[0]+-1*P_010111001*R_012[0]+P_010111101*R_013[0]+P_010211001*R_022[0]+-1*P_010211101*R_023[0]+-1*P_110011001*R_102[0]+P_110011101*R_103[0]+P_110111001*R_112[0]+-1*P_110111101*R_113[0]+-1*P_110211001*R_122[0]+P_110211101*R_123[0];
				double PR_010010002002=P_010010002*R_002[0]+-1*P_010010102*R_003[0]+P_010010202*R_004[0]+-1*P_010110002*R_012[0]+P_010110102*R_013[0]+-1*P_010110202*R_014[0]+-1*P_110010002*R_102[0]+P_110010102*R_103[0]+-1*P_110010202*R_104[0]+P_110110002*R_112[0]+-1*P_110110102*R_113[0]+P_110110202*R_114[0];
				double PR_002020000002=P_002020000*R_002[0]+-1*P_002120000*R_012[0]+P_002220000*R_022[0]+-1*P_102020000*R_102[0]+P_102120000*R_112[0]+-1*P_102220000*R_122[0]+P_202020000*R_202[0]+-1*P_202120000*R_212[0]+P_202220000*R_222[0];
				double PR_001021000002=P_001021000*R_002[0]+-1*P_001121000*R_012[0]+P_001221000*R_022[0]+-1*P_001321000*R_032[0]+-1*P_101021000*R_102[0]+P_101121000*R_112[0]+-1*P_101221000*R_122[0]+P_101321000*R_132[0];
				double PR_000022000002=P_000022000*R_002[0]+-1*P_000122000*R_012[0]+P_000222000*R_022[0]+-1*P_000322000*R_032[0]+P_000422000*R_042[0];
				double PR_001020001002=P_001020001*R_002[0]+-1*P_001020101*R_003[0]+-1*P_001120001*R_012[0]+P_001120101*R_013[0]+P_001220001*R_022[0]+-1*P_001220101*R_023[0]+-1*P_101020001*R_102[0]+P_101020101*R_103[0]+P_101120001*R_112[0]+-1*P_101120101*R_113[0]+-1*P_101220001*R_122[0]+P_101220101*R_123[0];
				double PR_000021001002=P_000021001*R_002[0]+-1*P_000021101*R_003[0]+-1*P_000121001*R_012[0]+P_000121101*R_013[0]+P_000221001*R_022[0]+-1*P_000221101*R_023[0]+-1*P_000321001*R_032[0]+P_000321101*R_033[0];
				double PR_000020002002=P_000020002*R_002[0]+-1*P_000020102*R_003[0]+P_000020202*R_004[0]+-1*P_000120002*R_012[0]+P_000120102*R_013[0]+-1*P_000120202*R_014[0]+P_000220002*R_022[0]+-1*P_000220102*R_023[0]+P_000220202*R_024[0];
				double PR_012000010002=P_012000010*R_002[0]+-1*P_012000110*R_003[0]+-1*P_112000010*R_102[0]+P_112000110*R_103[0]+P_212000010*R_202[0]+-1*P_212000110*R_203[0]+-1*P_312000010*R_302[0]+P_312000110*R_303[0];
				double PR_011001010002=P_011001010*R_002[0]+-1*P_011001110*R_003[0]+-1*P_011101010*R_012[0]+P_011101110*R_013[0]+-1*P_111001010*R_102[0]+P_111001110*R_103[0]+P_111101010*R_112[0]+-1*P_111101110*R_113[0]+P_211001010*R_202[0]+-1*P_211001110*R_203[0]+-1*P_211101010*R_212[0]+P_211101110*R_213[0];
				double PR_010002010002=P_010002010*R_002[0]+-1*P_010002110*R_003[0]+-1*P_010102010*R_012[0]+P_010102110*R_013[0]+P_010202010*R_022[0]+-1*P_010202110*R_023[0]+-1*P_110002010*R_102[0]+P_110002110*R_103[0]+P_110102010*R_112[0]+-1*P_110102110*R_113[0]+-1*P_110202010*R_122[0]+P_110202110*R_123[0];
				double PR_011000011002=P_011000011*R_002[0]+-1*P_011000111*R_003[0]+P_011000211*R_004[0]+-1*P_111000011*R_102[0]+P_111000111*R_103[0]+-1*P_111000211*R_104[0]+P_211000011*R_202[0]+-1*P_211000111*R_203[0]+P_211000211*R_204[0];
				double PR_010001011002=P_010001011*R_002[0]+-1*P_010001111*R_003[0]+P_010001211*R_004[0]+-1*P_010101011*R_012[0]+P_010101111*R_013[0]+-1*P_010101211*R_014[0]+-1*P_110001011*R_102[0]+P_110001111*R_103[0]+-1*P_110001211*R_104[0]+P_110101011*R_112[0]+-1*P_110101111*R_113[0]+P_110101211*R_114[0];
				double PR_010000012002=P_010000012*R_002[0]+-1*P_010000112*R_003[0]+P_010000212*R_004[0]+-1*P_010000312*R_005[0]+-1*P_110000012*R_102[0]+P_110000112*R_103[0]+-1*P_110000212*R_104[0]+P_110000312*R_105[0];
				double PR_002010010002=P_002010010*R_002[0]+-1*P_002010110*R_003[0]+-1*P_002110010*R_012[0]+P_002110110*R_013[0]+-1*P_102010010*R_102[0]+P_102010110*R_103[0]+P_102110010*R_112[0]+-1*P_102110110*R_113[0]+P_202010010*R_202[0]+-1*P_202010110*R_203[0]+-1*P_202110010*R_212[0]+P_202110110*R_213[0];
				double PR_001011010002=P_001011010*R_002[0]+-1*P_001011110*R_003[0]+-1*P_001111010*R_012[0]+P_001111110*R_013[0]+P_001211010*R_022[0]+-1*P_001211110*R_023[0]+-1*P_101011010*R_102[0]+P_101011110*R_103[0]+P_101111010*R_112[0]+-1*P_101111110*R_113[0]+-1*P_101211010*R_122[0]+P_101211110*R_123[0];
				double PR_000012010002=P_000012010*R_002[0]+-1*P_000012110*R_003[0]+-1*P_000112010*R_012[0]+P_000112110*R_013[0]+P_000212010*R_022[0]+-1*P_000212110*R_023[0]+-1*P_000312010*R_032[0]+P_000312110*R_033[0];
				double PR_001010011002=P_001010011*R_002[0]+-1*P_001010111*R_003[0]+P_001010211*R_004[0]+-1*P_001110011*R_012[0]+P_001110111*R_013[0]+-1*P_001110211*R_014[0]+-1*P_101010011*R_102[0]+P_101010111*R_103[0]+-1*P_101010211*R_104[0]+P_101110011*R_112[0]+-1*P_101110111*R_113[0]+P_101110211*R_114[0];
				double PR_000011011002=P_000011011*R_002[0]+-1*P_000011111*R_003[0]+P_000011211*R_004[0]+-1*P_000111011*R_012[0]+P_000111111*R_013[0]+-1*P_000111211*R_014[0]+P_000211011*R_022[0]+-1*P_000211111*R_023[0]+P_000211211*R_024[0];
				double PR_000010012002=P_000010012*R_002[0]+-1*P_000010112*R_003[0]+P_000010212*R_004[0]+-1*P_000010312*R_005[0]+-1*P_000110012*R_012[0]+P_000110112*R_013[0]+-1*P_000110212*R_014[0]+P_000110312*R_015[0];
				double PR_002000020002=P_002000020*R_002[0]+-1*P_002000120*R_003[0]+P_002000220*R_004[0]+-1*P_102000020*R_102[0]+P_102000120*R_103[0]+-1*P_102000220*R_104[0]+P_202000020*R_202[0]+-1*P_202000120*R_203[0]+P_202000220*R_204[0];
				double PR_001001020002=P_001001020*R_002[0]+-1*P_001001120*R_003[0]+P_001001220*R_004[0]+-1*P_001101020*R_012[0]+P_001101120*R_013[0]+-1*P_001101220*R_014[0]+-1*P_101001020*R_102[0]+P_101001120*R_103[0]+-1*P_101001220*R_104[0]+P_101101020*R_112[0]+-1*P_101101120*R_113[0]+P_101101220*R_114[0];
				double PR_000002020002=P_000002020*R_002[0]+-1*P_000002120*R_003[0]+P_000002220*R_004[0]+-1*P_000102020*R_012[0]+P_000102120*R_013[0]+-1*P_000102220*R_014[0]+P_000202020*R_022[0]+-1*P_000202120*R_023[0]+P_000202220*R_024[0];
				double PR_001000021002=P_001000021*R_002[0]+-1*P_001000121*R_003[0]+P_001000221*R_004[0]+-1*P_001000321*R_005[0]+-1*P_101000021*R_102[0]+P_101000121*R_103[0]+-1*P_101000221*R_104[0]+P_101000321*R_105[0];
				double PR_000001021002=P_000001021*R_002[0]+-1*P_000001121*R_003[0]+P_000001221*R_004[0]+-1*P_000001321*R_005[0]+-1*P_000101021*R_012[0]+P_000101121*R_013[0]+-1*P_000101221*R_014[0]+P_000101321*R_015[0];
				double PR_000000022002=P_000000022*R_002[0]+-1*P_000000122*R_003[0]+P_000000222*R_004[0]+-1*P_000000322*R_005[0]+P_000000422*R_006[0];
				double PR_022000000011=P_022000000*R_011[0]+-1*P_122000000*R_111[0]+P_222000000*R_211[0]+-1*P_322000000*R_311[0]+P_422000000*R_411[0];
				double PR_021001000011=P_021001000*R_011[0]+-1*P_021101000*R_021[0]+-1*P_121001000*R_111[0]+P_121101000*R_121[0]+P_221001000*R_211[0]+-1*P_221101000*R_221[0]+-1*P_321001000*R_311[0]+P_321101000*R_321[0];
				double PR_020002000011=P_020002000*R_011[0]+-1*P_020102000*R_021[0]+P_020202000*R_031[0]+-1*P_120002000*R_111[0]+P_120102000*R_121[0]+-1*P_120202000*R_131[0]+P_220002000*R_211[0]+-1*P_220102000*R_221[0]+P_220202000*R_231[0];
				double PR_021000001011=P_021000001*R_011[0]+-1*P_021000101*R_012[0]+-1*P_121000001*R_111[0]+P_121000101*R_112[0]+P_221000001*R_211[0]+-1*P_221000101*R_212[0]+-1*P_321000001*R_311[0]+P_321000101*R_312[0];
				double PR_020001001011=P_020001001*R_011[0]+-1*P_020001101*R_012[0]+-1*P_020101001*R_021[0]+P_020101101*R_022[0]+-1*P_120001001*R_111[0]+P_120001101*R_112[0]+P_120101001*R_121[0]+-1*P_120101101*R_122[0]+P_220001001*R_211[0]+-1*P_220001101*R_212[0]+-1*P_220101001*R_221[0]+P_220101101*R_222[0];
				double PR_020000002011=P_020000002*R_011[0]+-1*P_020000102*R_012[0]+P_020000202*R_013[0]+-1*P_120000002*R_111[0]+P_120000102*R_112[0]+-1*P_120000202*R_113[0]+P_220000002*R_211[0]+-1*P_220000102*R_212[0]+P_220000202*R_213[0];
				double PR_012010000011=P_012010000*R_011[0]+-1*P_012110000*R_021[0]+-1*P_112010000*R_111[0]+P_112110000*R_121[0]+P_212010000*R_211[0]+-1*P_212110000*R_221[0]+-1*P_312010000*R_311[0]+P_312110000*R_321[0];
				double PR_011011000011=P_011011000*R_011[0]+-1*P_011111000*R_021[0]+P_011211000*R_031[0]+-1*P_111011000*R_111[0]+P_111111000*R_121[0]+-1*P_111211000*R_131[0]+P_211011000*R_211[0]+-1*P_211111000*R_221[0]+P_211211000*R_231[0];
				double PR_010012000011=P_010012000*R_011[0]+-1*P_010112000*R_021[0]+P_010212000*R_031[0]+-1*P_010312000*R_041[0]+-1*P_110012000*R_111[0]+P_110112000*R_121[0]+-1*P_110212000*R_131[0]+P_110312000*R_141[0];
				double PR_011010001011=P_011010001*R_011[0]+-1*P_011010101*R_012[0]+-1*P_011110001*R_021[0]+P_011110101*R_022[0]+-1*P_111010001*R_111[0]+P_111010101*R_112[0]+P_111110001*R_121[0]+-1*P_111110101*R_122[0]+P_211010001*R_211[0]+-1*P_211010101*R_212[0]+-1*P_211110001*R_221[0]+P_211110101*R_222[0];
				double PR_010011001011=P_010011001*R_011[0]+-1*P_010011101*R_012[0]+-1*P_010111001*R_021[0]+P_010111101*R_022[0]+P_010211001*R_031[0]+-1*P_010211101*R_032[0]+-1*P_110011001*R_111[0]+P_110011101*R_112[0]+P_110111001*R_121[0]+-1*P_110111101*R_122[0]+-1*P_110211001*R_131[0]+P_110211101*R_132[0];
				double PR_010010002011=P_010010002*R_011[0]+-1*P_010010102*R_012[0]+P_010010202*R_013[0]+-1*P_010110002*R_021[0]+P_010110102*R_022[0]+-1*P_010110202*R_023[0]+-1*P_110010002*R_111[0]+P_110010102*R_112[0]+-1*P_110010202*R_113[0]+P_110110002*R_121[0]+-1*P_110110102*R_122[0]+P_110110202*R_123[0];
				double PR_002020000011=P_002020000*R_011[0]+-1*P_002120000*R_021[0]+P_002220000*R_031[0]+-1*P_102020000*R_111[0]+P_102120000*R_121[0]+-1*P_102220000*R_131[0]+P_202020000*R_211[0]+-1*P_202120000*R_221[0]+P_202220000*R_231[0];
				double PR_001021000011=P_001021000*R_011[0]+-1*P_001121000*R_021[0]+P_001221000*R_031[0]+-1*P_001321000*R_041[0]+-1*P_101021000*R_111[0]+P_101121000*R_121[0]+-1*P_101221000*R_131[0]+P_101321000*R_141[0];
				double PR_000022000011=P_000022000*R_011[0]+-1*P_000122000*R_021[0]+P_000222000*R_031[0]+-1*P_000322000*R_041[0]+P_000422000*R_051[0];
				double PR_001020001011=P_001020001*R_011[0]+-1*P_001020101*R_012[0]+-1*P_001120001*R_021[0]+P_001120101*R_022[0]+P_001220001*R_031[0]+-1*P_001220101*R_032[0]+-1*P_101020001*R_111[0]+P_101020101*R_112[0]+P_101120001*R_121[0]+-1*P_101120101*R_122[0]+-1*P_101220001*R_131[0]+P_101220101*R_132[0];
				double PR_000021001011=P_000021001*R_011[0]+-1*P_000021101*R_012[0]+-1*P_000121001*R_021[0]+P_000121101*R_022[0]+P_000221001*R_031[0]+-1*P_000221101*R_032[0]+-1*P_000321001*R_041[0]+P_000321101*R_042[0];
				double PR_000020002011=P_000020002*R_011[0]+-1*P_000020102*R_012[0]+P_000020202*R_013[0]+-1*P_000120002*R_021[0]+P_000120102*R_022[0]+-1*P_000120202*R_023[0]+P_000220002*R_031[0]+-1*P_000220102*R_032[0]+P_000220202*R_033[0];
				double PR_012000010011=P_012000010*R_011[0]+-1*P_012000110*R_012[0]+-1*P_112000010*R_111[0]+P_112000110*R_112[0]+P_212000010*R_211[0]+-1*P_212000110*R_212[0]+-1*P_312000010*R_311[0]+P_312000110*R_312[0];
				double PR_011001010011=P_011001010*R_011[0]+-1*P_011001110*R_012[0]+-1*P_011101010*R_021[0]+P_011101110*R_022[0]+-1*P_111001010*R_111[0]+P_111001110*R_112[0]+P_111101010*R_121[0]+-1*P_111101110*R_122[0]+P_211001010*R_211[0]+-1*P_211001110*R_212[0]+-1*P_211101010*R_221[0]+P_211101110*R_222[0];
				double PR_010002010011=P_010002010*R_011[0]+-1*P_010002110*R_012[0]+-1*P_010102010*R_021[0]+P_010102110*R_022[0]+P_010202010*R_031[0]+-1*P_010202110*R_032[0]+-1*P_110002010*R_111[0]+P_110002110*R_112[0]+P_110102010*R_121[0]+-1*P_110102110*R_122[0]+-1*P_110202010*R_131[0]+P_110202110*R_132[0];
				double PR_011000011011=P_011000011*R_011[0]+-1*P_011000111*R_012[0]+P_011000211*R_013[0]+-1*P_111000011*R_111[0]+P_111000111*R_112[0]+-1*P_111000211*R_113[0]+P_211000011*R_211[0]+-1*P_211000111*R_212[0]+P_211000211*R_213[0];
				double PR_010001011011=P_010001011*R_011[0]+-1*P_010001111*R_012[0]+P_010001211*R_013[0]+-1*P_010101011*R_021[0]+P_010101111*R_022[0]+-1*P_010101211*R_023[0]+-1*P_110001011*R_111[0]+P_110001111*R_112[0]+-1*P_110001211*R_113[0]+P_110101011*R_121[0]+-1*P_110101111*R_122[0]+P_110101211*R_123[0];
				double PR_010000012011=P_010000012*R_011[0]+-1*P_010000112*R_012[0]+P_010000212*R_013[0]+-1*P_010000312*R_014[0]+-1*P_110000012*R_111[0]+P_110000112*R_112[0]+-1*P_110000212*R_113[0]+P_110000312*R_114[0];
				double PR_002010010011=P_002010010*R_011[0]+-1*P_002010110*R_012[0]+-1*P_002110010*R_021[0]+P_002110110*R_022[0]+-1*P_102010010*R_111[0]+P_102010110*R_112[0]+P_102110010*R_121[0]+-1*P_102110110*R_122[0]+P_202010010*R_211[0]+-1*P_202010110*R_212[0]+-1*P_202110010*R_221[0]+P_202110110*R_222[0];
				double PR_001011010011=P_001011010*R_011[0]+-1*P_001011110*R_012[0]+-1*P_001111010*R_021[0]+P_001111110*R_022[0]+P_001211010*R_031[0]+-1*P_001211110*R_032[0]+-1*P_101011010*R_111[0]+P_101011110*R_112[0]+P_101111010*R_121[0]+-1*P_101111110*R_122[0]+-1*P_101211010*R_131[0]+P_101211110*R_132[0];
				double PR_000012010011=P_000012010*R_011[0]+-1*P_000012110*R_012[0]+-1*P_000112010*R_021[0]+P_000112110*R_022[0]+P_000212010*R_031[0]+-1*P_000212110*R_032[0]+-1*P_000312010*R_041[0]+P_000312110*R_042[0];
				double PR_001010011011=P_001010011*R_011[0]+-1*P_001010111*R_012[0]+P_001010211*R_013[0]+-1*P_001110011*R_021[0]+P_001110111*R_022[0]+-1*P_001110211*R_023[0]+-1*P_101010011*R_111[0]+P_101010111*R_112[0]+-1*P_101010211*R_113[0]+P_101110011*R_121[0]+-1*P_101110111*R_122[0]+P_101110211*R_123[0];
				double PR_000011011011=P_000011011*R_011[0]+-1*P_000011111*R_012[0]+P_000011211*R_013[0]+-1*P_000111011*R_021[0]+P_000111111*R_022[0]+-1*P_000111211*R_023[0]+P_000211011*R_031[0]+-1*P_000211111*R_032[0]+P_000211211*R_033[0];
				double PR_000010012011=P_000010012*R_011[0]+-1*P_000010112*R_012[0]+P_000010212*R_013[0]+-1*P_000010312*R_014[0]+-1*P_000110012*R_021[0]+P_000110112*R_022[0]+-1*P_000110212*R_023[0]+P_000110312*R_024[0];
				double PR_002000020011=P_002000020*R_011[0]+-1*P_002000120*R_012[0]+P_002000220*R_013[0]+-1*P_102000020*R_111[0]+P_102000120*R_112[0]+-1*P_102000220*R_113[0]+P_202000020*R_211[0]+-1*P_202000120*R_212[0]+P_202000220*R_213[0];
				double PR_001001020011=P_001001020*R_011[0]+-1*P_001001120*R_012[0]+P_001001220*R_013[0]+-1*P_001101020*R_021[0]+P_001101120*R_022[0]+-1*P_001101220*R_023[0]+-1*P_101001020*R_111[0]+P_101001120*R_112[0]+-1*P_101001220*R_113[0]+P_101101020*R_121[0]+-1*P_101101120*R_122[0]+P_101101220*R_123[0];
				double PR_000002020011=P_000002020*R_011[0]+-1*P_000002120*R_012[0]+P_000002220*R_013[0]+-1*P_000102020*R_021[0]+P_000102120*R_022[0]+-1*P_000102220*R_023[0]+P_000202020*R_031[0]+-1*P_000202120*R_032[0]+P_000202220*R_033[0];
				double PR_001000021011=P_001000021*R_011[0]+-1*P_001000121*R_012[0]+P_001000221*R_013[0]+-1*P_001000321*R_014[0]+-1*P_101000021*R_111[0]+P_101000121*R_112[0]+-1*P_101000221*R_113[0]+P_101000321*R_114[0];
				double PR_000001021011=P_000001021*R_011[0]+-1*P_000001121*R_012[0]+P_000001221*R_013[0]+-1*P_000001321*R_014[0]+-1*P_000101021*R_021[0]+P_000101121*R_022[0]+-1*P_000101221*R_023[0]+P_000101321*R_024[0];
				double PR_000000022011=P_000000022*R_011[0]+-1*P_000000122*R_012[0]+P_000000222*R_013[0]+-1*P_000000322*R_014[0]+P_000000422*R_015[0];
				double PR_022000000020=P_022000000*R_020[0]+-1*P_122000000*R_120[0]+P_222000000*R_220[0]+-1*P_322000000*R_320[0]+P_422000000*R_420[0];
				double PR_021001000020=P_021001000*R_020[0]+-1*P_021101000*R_030[0]+-1*P_121001000*R_120[0]+P_121101000*R_130[0]+P_221001000*R_220[0]+-1*P_221101000*R_230[0]+-1*P_321001000*R_320[0]+P_321101000*R_330[0];
				double PR_020002000020=P_020002000*R_020[0]+-1*P_020102000*R_030[0]+P_020202000*R_040[0]+-1*P_120002000*R_120[0]+P_120102000*R_130[0]+-1*P_120202000*R_140[0]+P_220002000*R_220[0]+-1*P_220102000*R_230[0]+P_220202000*R_240[0];
				double PR_021000001020=P_021000001*R_020[0]+-1*P_021000101*R_021[0]+-1*P_121000001*R_120[0]+P_121000101*R_121[0]+P_221000001*R_220[0]+-1*P_221000101*R_221[0]+-1*P_321000001*R_320[0]+P_321000101*R_321[0];
				double PR_020001001020=P_020001001*R_020[0]+-1*P_020001101*R_021[0]+-1*P_020101001*R_030[0]+P_020101101*R_031[0]+-1*P_120001001*R_120[0]+P_120001101*R_121[0]+P_120101001*R_130[0]+-1*P_120101101*R_131[0]+P_220001001*R_220[0]+-1*P_220001101*R_221[0]+-1*P_220101001*R_230[0]+P_220101101*R_231[0];
				double PR_020000002020=P_020000002*R_020[0]+-1*P_020000102*R_021[0]+P_020000202*R_022[0]+-1*P_120000002*R_120[0]+P_120000102*R_121[0]+-1*P_120000202*R_122[0]+P_220000002*R_220[0]+-1*P_220000102*R_221[0]+P_220000202*R_222[0];
				double PR_012010000020=P_012010000*R_020[0]+-1*P_012110000*R_030[0]+-1*P_112010000*R_120[0]+P_112110000*R_130[0]+P_212010000*R_220[0]+-1*P_212110000*R_230[0]+-1*P_312010000*R_320[0]+P_312110000*R_330[0];
				double PR_011011000020=P_011011000*R_020[0]+-1*P_011111000*R_030[0]+P_011211000*R_040[0]+-1*P_111011000*R_120[0]+P_111111000*R_130[0]+-1*P_111211000*R_140[0]+P_211011000*R_220[0]+-1*P_211111000*R_230[0]+P_211211000*R_240[0];
				double PR_010012000020=P_010012000*R_020[0]+-1*P_010112000*R_030[0]+P_010212000*R_040[0]+-1*P_010312000*R_050[0]+-1*P_110012000*R_120[0]+P_110112000*R_130[0]+-1*P_110212000*R_140[0]+P_110312000*R_150[0];
				double PR_011010001020=P_011010001*R_020[0]+-1*P_011010101*R_021[0]+-1*P_011110001*R_030[0]+P_011110101*R_031[0]+-1*P_111010001*R_120[0]+P_111010101*R_121[0]+P_111110001*R_130[0]+-1*P_111110101*R_131[0]+P_211010001*R_220[0]+-1*P_211010101*R_221[0]+-1*P_211110001*R_230[0]+P_211110101*R_231[0];
				double PR_010011001020=P_010011001*R_020[0]+-1*P_010011101*R_021[0]+-1*P_010111001*R_030[0]+P_010111101*R_031[0]+P_010211001*R_040[0]+-1*P_010211101*R_041[0]+-1*P_110011001*R_120[0]+P_110011101*R_121[0]+P_110111001*R_130[0]+-1*P_110111101*R_131[0]+-1*P_110211001*R_140[0]+P_110211101*R_141[0];
				double PR_010010002020=P_010010002*R_020[0]+-1*P_010010102*R_021[0]+P_010010202*R_022[0]+-1*P_010110002*R_030[0]+P_010110102*R_031[0]+-1*P_010110202*R_032[0]+-1*P_110010002*R_120[0]+P_110010102*R_121[0]+-1*P_110010202*R_122[0]+P_110110002*R_130[0]+-1*P_110110102*R_131[0]+P_110110202*R_132[0];
				double PR_002020000020=P_002020000*R_020[0]+-1*P_002120000*R_030[0]+P_002220000*R_040[0]+-1*P_102020000*R_120[0]+P_102120000*R_130[0]+-1*P_102220000*R_140[0]+P_202020000*R_220[0]+-1*P_202120000*R_230[0]+P_202220000*R_240[0];
				double PR_001021000020=P_001021000*R_020[0]+-1*P_001121000*R_030[0]+P_001221000*R_040[0]+-1*P_001321000*R_050[0]+-1*P_101021000*R_120[0]+P_101121000*R_130[0]+-1*P_101221000*R_140[0]+P_101321000*R_150[0];
				double PR_000022000020=P_000022000*R_020[0]+-1*P_000122000*R_030[0]+P_000222000*R_040[0]+-1*P_000322000*R_050[0]+P_000422000*R_060[0];
				double PR_001020001020=P_001020001*R_020[0]+-1*P_001020101*R_021[0]+-1*P_001120001*R_030[0]+P_001120101*R_031[0]+P_001220001*R_040[0]+-1*P_001220101*R_041[0]+-1*P_101020001*R_120[0]+P_101020101*R_121[0]+P_101120001*R_130[0]+-1*P_101120101*R_131[0]+-1*P_101220001*R_140[0]+P_101220101*R_141[0];
				double PR_000021001020=P_000021001*R_020[0]+-1*P_000021101*R_021[0]+-1*P_000121001*R_030[0]+P_000121101*R_031[0]+P_000221001*R_040[0]+-1*P_000221101*R_041[0]+-1*P_000321001*R_050[0]+P_000321101*R_051[0];
				double PR_000020002020=P_000020002*R_020[0]+-1*P_000020102*R_021[0]+P_000020202*R_022[0]+-1*P_000120002*R_030[0]+P_000120102*R_031[0]+-1*P_000120202*R_032[0]+P_000220002*R_040[0]+-1*P_000220102*R_041[0]+P_000220202*R_042[0];
				double PR_012000010020=P_012000010*R_020[0]+-1*P_012000110*R_021[0]+-1*P_112000010*R_120[0]+P_112000110*R_121[0]+P_212000010*R_220[0]+-1*P_212000110*R_221[0]+-1*P_312000010*R_320[0]+P_312000110*R_321[0];
				double PR_011001010020=P_011001010*R_020[0]+-1*P_011001110*R_021[0]+-1*P_011101010*R_030[0]+P_011101110*R_031[0]+-1*P_111001010*R_120[0]+P_111001110*R_121[0]+P_111101010*R_130[0]+-1*P_111101110*R_131[0]+P_211001010*R_220[0]+-1*P_211001110*R_221[0]+-1*P_211101010*R_230[0]+P_211101110*R_231[0];
				double PR_010002010020=P_010002010*R_020[0]+-1*P_010002110*R_021[0]+-1*P_010102010*R_030[0]+P_010102110*R_031[0]+P_010202010*R_040[0]+-1*P_010202110*R_041[0]+-1*P_110002010*R_120[0]+P_110002110*R_121[0]+P_110102010*R_130[0]+-1*P_110102110*R_131[0]+-1*P_110202010*R_140[0]+P_110202110*R_141[0];
				double PR_011000011020=P_011000011*R_020[0]+-1*P_011000111*R_021[0]+P_011000211*R_022[0]+-1*P_111000011*R_120[0]+P_111000111*R_121[0]+-1*P_111000211*R_122[0]+P_211000011*R_220[0]+-1*P_211000111*R_221[0]+P_211000211*R_222[0];
				double PR_010001011020=P_010001011*R_020[0]+-1*P_010001111*R_021[0]+P_010001211*R_022[0]+-1*P_010101011*R_030[0]+P_010101111*R_031[0]+-1*P_010101211*R_032[0]+-1*P_110001011*R_120[0]+P_110001111*R_121[0]+-1*P_110001211*R_122[0]+P_110101011*R_130[0]+-1*P_110101111*R_131[0]+P_110101211*R_132[0];
				double PR_010000012020=P_010000012*R_020[0]+-1*P_010000112*R_021[0]+P_010000212*R_022[0]+-1*P_010000312*R_023[0]+-1*P_110000012*R_120[0]+P_110000112*R_121[0]+-1*P_110000212*R_122[0]+P_110000312*R_123[0];
				double PR_002010010020=P_002010010*R_020[0]+-1*P_002010110*R_021[0]+-1*P_002110010*R_030[0]+P_002110110*R_031[0]+-1*P_102010010*R_120[0]+P_102010110*R_121[0]+P_102110010*R_130[0]+-1*P_102110110*R_131[0]+P_202010010*R_220[0]+-1*P_202010110*R_221[0]+-1*P_202110010*R_230[0]+P_202110110*R_231[0];
				double PR_001011010020=P_001011010*R_020[0]+-1*P_001011110*R_021[0]+-1*P_001111010*R_030[0]+P_001111110*R_031[0]+P_001211010*R_040[0]+-1*P_001211110*R_041[0]+-1*P_101011010*R_120[0]+P_101011110*R_121[0]+P_101111010*R_130[0]+-1*P_101111110*R_131[0]+-1*P_101211010*R_140[0]+P_101211110*R_141[0];
				double PR_000012010020=P_000012010*R_020[0]+-1*P_000012110*R_021[0]+-1*P_000112010*R_030[0]+P_000112110*R_031[0]+P_000212010*R_040[0]+-1*P_000212110*R_041[0]+-1*P_000312010*R_050[0]+P_000312110*R_051[0];
				double PR_001010011020=P_001010011*R_020[0]+-1*P_001010111*R_021[0]+P_001010211*R_022[0]+-1*P_001110011*R_030[0]+P_001110111*R_031[0]+-1*P_001110211*R_032[0]+-1*P_101010011*R_120[0]+P_101010111*R_121[0]+-1*P_101010211*R_122[0]+P_101110011*R_130[0]+-1*P_101110111*R_131[0]+P_101110211*R_132[0];
				double PR_000011011020=P_000011011*R_020[0]+-1*P_000011111*R_021[0]+P_000011211*R_022[0]+-1*P_000111011*R_030[0]+P_000111111*R_031[0]+-1*P_000111211*R_032[0]+P_000211011*R_040[0]+-1*P_000211111*R_041[0]+P_000211211*R_042[0];
				double PR_000010012020=P_000010012*R_020[0]+-1*P_000010112*R_021[0]+P_000010212*R_022[0]+-1*P_000010312*R_023[0]+-1*P_000110012*R_030[0]+P_000110112*R_031[0]+-1*P_000110212*R_032[0]+P_000110312*R_033[0];
				double PR_002000020020=P_002000020*R_020[0]+-1*P_002000120*R_021[0]+P_002000220*R_022[0]+-1*P_102000020*R_120[0]+P_102000120*R_121[0]+-1*P_102000220*R_122[0]+P_202000020*R_220[0]+-1*P_202000120*R_221[0]+P_202000220*R_222[0];
				double PR_001001020020=P_001001020*R_020[0]+-1*P_001001120*R_021[0]+P_001001220*R_022[0]+-1*P_001101020*R_030[0]+P_001101120*R_031[0]+-1*P_001101220*R_032[0]+-1*P_101001020*R_120[0]+P_101001120*R_121[0]+-1*P_101001220*R_122[0]+P_101101020*R_130[0]+-1*P_101101120*R_131[0]+P_101101220*R_132[0];
				double PR_000002020020=P_000002020*R_020[0]+-1*P_000002120*R_021[0]+P_000002220*R_022[0]+-1*P_000102020*R_030[0]+P_000102120*R_031[0]+-1*P_000102220*R_032[0]+P_000202020*R_040[0]+-1*P_000202120*R_041[0]+P_000202220*R_042[0];
				double PR_001000021020=P_001000021*R_020[0]+-1*P_001000121*R_021[0]+P_001000221*R_022[0]+-1*P_001000321*R_023[0]+-1*P_101000021*R_120[0]+P_101000121*R_121[0]+-1*P_101000221*R_122[0]+P_101000321*R_123[0];
				double PR_000001021020=P_000001021*R_020[0]+-1*P_000001121*R_021[0]+P_000001221*R_022[0]+-1*P_000001321*R_023[0]+-1*P_000101021*R_030[0]+P_000101121*R_031[0]+-1*P_000101221*R_032[0]+P_000101321*R_033[0];
				double PR_000000022020=P_000000022*R_020[0]+-1*P_000000122*R_021[0]+P_000000222*R_022[0]+-1*P_000000322*R_023[0]+P_000000422*R_024[0];
				double PR_022000000101=P_022000000*R_101[0]+-1*P_122000000*R_201[0]+P_222000000*R_301[0]+-1*P_322000000*R_401[0]+P_422000000*R_501[0];
				double PR_021001000101=P_021001000*R_101[0]+-1*P_021101000*R_111[0]+-1*P_121001000*R_201[0]+P_121101000*R_211[0]+P_221001000*R_301[0]+-1*P_221101000*R_311[0]+-1*P_321001000*R_401[0]+P_321101000*R_411[0];
				double PR_020002000101=P_020002000*R_101[0]+-1*P_020102000*R_111[0]+P_020202000*R_121[0]+-1*P_120002000*R_201[0]+P_120102000*R_211[0]+-1*P_120202000*R_221[0]+P_220002000*R_301[0]+-1*P_220102000*R_311[0]+P_220202000*R_321[0];
				double PR_021000001101=P_021000001*R_101[0]+-1*P_021000101*R_102[0]+-1*P_121000001*R_201[0]+P_121000101*R_202[0]+P_221000001*R_301[0]+-1*P_221000101*R_302[0]+-1*P_321000001*R_401[0]+P_321000101*R_402[0];
				double PR_020001001101=P_020001001*R_101[0]+-1*P_020001101*R_102[0]+-1*P_020101001*R_111[0]+P_020101101*R_112[0]+-1*P_120001001*R_201[0]+P_120001101*R_202[0]+P_120101001*R_211[0]+-1*P_120101101*R_212[0]+P_220001001*R_301[0]+-1*P_220001101*R_302[0]+-1*P_220101001*R_311[0]+P_220101101*R_312[0];
				double PR_020000002101=P_020000002*R_101[0]+-1*P_020000102*R_102[0]+P_020000202*R_103[0]+-1*P_120000002*R_201[0]+P_120000102*R_202[0]+-1*P_120000202*R_203[0]+P_220000002*R_301[0]+-1*P_220000102*R_302[0]+P_220000202*R_303[0];
				double PR_012010000101=P_012010000*R_101[0]+-1*P_012110000*R_111[0]+-1*P_112010000*R_201[0]+P_112110000*R_211[0]+P_212010000*R_301[0]+-1*P_212110000*R_311[0]+-1*P_312010000*R_401[0]+P_312110000*R_411[0];
				double PR_011011000101=P_011011000*R_101[0]+-1*P_011111000*R_111[0]+P_011211000*R_121[0]+-1*P_111011000*R_201[0]+P_111111000*R_211[0]+-1*P_111211000*R_221[0]+P_211011000*R_301[0]+-1*P_211111000*R_311[0]+P_211211000*R_321[0];
				double PR_010012000101=P_010012000*R_101[0]+-1*P_010112000*R_111[0]+P_010212000*R_121[0]+-1*P_010312000*R_131[0]+-1*P_110012000*R_201[0]+P_110112000*R_211[0]+-1*P_110212000*R_221[0]+P_110312000*R_231[0];
				double PR_011010001101=P_011010001*R_101[0]+-1*P_011010101*R_102[0]+-1*P_011110001*R_111[0]+P_011110101*R_112[0]+-1*P_111010001*R_201[0]+P_111010101*R_202[0]+P_111110001*R_211[0]+-1*P_111110101*R_212[0]+P_211010001*R_301[0]+-1*P_211010101*R_302[0]+-1*P_211110001*R_311[0]+P_211110101*R_312[0];
				double PR_010011001101=P_010011001*R_101[0]+-1*P_010011101*R_102[0]+-1*P_010111001*R_111[0]+P_010111101*R_112[0]+P_010211001*R_121[0]+-1*P_010211101*R_122[0]+-1*P_110011001*R_201[0]+P_110011101*R_202[0]+P_110111001*R_211[0]+-1*P_110111101*R_212[0]+-1*P_110211001*R_221[0]+P_110211101*R_222[0];
				double PR_010010002101=P_010010002*R_101[0]+-1*P_010010102*R_102[0]+P_010010202*R_103[0]+-1*P_010110002*R_111[0]+P_010110102*R_112[0]+-1*P_010110202*R_113[0]+-1*P_110010002*R_201[0]+P_110010102*R_202[0]+-1*P_110010202*R_203[0]+P_110110002*R_211[0]+-1*P_110110102*R_212[0]+P_110110202*R_213[0];
				double PR_002020000101=P_002020000*R_101[0]+-1*P_002120000*R_111[0]+P_002220000*R_121[0]+-1*P_102020000*R_201[0]+P_102120000*R_211[0]+-1*P_102220000*R_221[0]+P_202020000*R_301[0]+-1*P_202120000*R_311[0]+P_202220000*R_321[0];
				double PR_001021000101=P_001021000*R_101[0]+-1*P_001121000*R_111[0]+P_001221000*R_121[0]+-1*P_001321000*R_131[0]+-1*P_101021000*R_201[0]+P_101121000*R_211[0]+-1*P_101221000*R_221[0]+P_101321000*R_231[0];
				double PR_000022000101=P_000022000*R_101[0]+-1*P_000122000*R_111[0]+P_000222000*R_121[0]+-1*P_000322000*R_131[0]+P_000422000*R_141[0];
				double PR_001020001101=P_001020001*R_101[0]+-1*P_001020101*R_102[0]+-1*P_001120001*R_111[0]+P_001120101*R_112[0]+P_001220001*R_121[0]+-1*P_001220101*R_122[0]+-1*P_101020001*R_201[0]+P_101020101*R_202[0]+P_101120001*R_211[0]+-1*P_101120101*R_212[0]+-1*P_101220001*R_221[0]+P_101220101*R_222[0];
				double PR_000021001101=P_000021001*R_101[0]+-1*P_000021101*R_102[0]+-1*P_000121001*R_111[0]+P_000121101*R_112[0]+P_000221001*R_121[0]+-1*P_000221101*R_122[0]+-1*P_000321001*R_131[0]+P_000321101*R_132[0];
				double PR_000020002101=P_000020002*R_101[0]+-1*P_000020102*R_102[0]+P_000020202*R_103[0]+-1*P_000120002*R_111[0]+P_000120102*R_112[0]+-1*P_000120202*R_113[0]+P_000220002*R_121[0]+-1*P_000220102*R_122[0]+P_000220202*R_123[0];
				double PR_012000010101=P_012000010*R_101[0]+-1*P_012000110*R_102[0]+-1*P_112000010*R_201[0]+P_112000110*R_202[0]+P_212000010*R_301[0]+-1*P_212000110*R_302[0]+-1*P_312000010*R_401[0]+P_312000110*R_402[0];
				double PR_011001010101=P_011001010*R_101[0]+-1*P_011001110*R_102[0]+-1*P_011101010*R_111[0]+P_011101110*R_112[0]+-1*P_111001010*R_201[0]+P_111001110*R_202[0]+P_111101010*R_211[0]+-1*P_111101110*R_212[0]+P_211001010*R_301[0]+-1*P_211001110*R_302[0]+-1*P_211101010*R_311[0]+P_211101110*R_312[0];
				double PR_010002010101=P_010002010*R_101[0]+-1*P_010002110*R_102[0]+-1*P_010102010*R_111[0]+P_010102110*R_112[0]+P_010202010*R_121[0]+-1*P_010202110*R_122[0]+-1*P_110002010*R_201[0]+P_110002110*R_202[0]+P_110102010*R_211[0]+-1*P_110102110*R_212[0]+-1*P_110202010*R_221[0]+P_110202110*R_222[0];
				double PR_011000011101=P_011000011*R_101[0]+-1*P_011000111*R_102[0]+P_011000211*R_103[0]+-1*P_111000011*R_201[0]+P_111000111*R_202[0]+-1*P_111000211*R_203[0]+P_211000011*R_301[0]+-1*P_211000111*R_302[0]+P_211000211*R_303[0];
				double PR_010001011101=P_010001011*R_101[0]+-1*P_010001111*R_102[0]+P_010001211*R_103[0]+-1*P_010101011*R_111[0]+P_010101111*R_112[0]+-1*P_010101211*R_113[0]+-1*P_110001011*R_201[0]+P_110001111*R_202[0]+-1*P_110001211*R_203[0]+P_110101011*R_211[0]+-1*P_110101111*R_212[0]+P_110101211*R_213[0];
				double PR_010000012101=P_010000012*R_101[0]+-1*P_010000112*R_102[0]+P_010000212*R_103[0]+-1*P_010000312*R_104[0]+-1*P_110000012*R_201[0]+P_110000112*R_202[0]+-1*P_110000212*R_203[0]+P_110000312*R_204[0];
				double PR_002010010101=P_002010010*R_101[0]+-1*P_002010110*R_102[0]+-1*P_002110010*R_111[0]+P_002110110*R_112[0]+-1*P_102010010*R_201[0]+P_102010110*R_202[0]+P_102110010*R_211[0]+-1*P_102110110*R_212[0]+P_202010010*R_301[0]+-1*P_202010110*R_302[0]+-1*P_202110010*R_311[0]+P_202110110*R_312[0];
				double PR_001011010101=P_001011010*R_101[0]+-1*P_001011110*R_102[0]+-1*P_001111010*R_111[0]+P_001111110*R_112[0]+P_001211010*R_121[0]+-1*P_001211110*R_122[0]+-1*P_101011010*R_201[0]+P_101011110*R_202[0]+P_101111010*R_211[0]+-1*P_101111110*R_212[0]+-1*P_101211010*R_221[0]+P_101211110*R_222[0];
				double PR_000012010101=P_000012010*R_101[0]+-1*P_000012110*R_102[0]+-1*P_000112010*R_111[0]+P_000112110*R_112[0]+P_000212010*R_121[0]+-1*P_000212110*R_122[0]+-1*P_000312010*R_131[0]+P_000312110*R_132[0];
				double PR_001010011101=P_001010011*R_101[0]+-1*P_001010111*R_102[0]+P_001010211*R_103[0]+-1*P_001110011*R_111[0]+P_001110111*R_112[0]+-1*P_001110211*R_113[0]+-1*P_101010011*R_201[0]+P_101010111*R_202[0]+-1*P_101010211*R_203[0]+P_101110011*R_211[0]+-1*P_101110111*R_212[0]+P_101110211*R_213[0];
				double PR_000011011101=P_000011011*R_101[0]+-1*P_000011111*R_102[0]+P_000011211*R_103[0]+-1*P_000111011*R_111[0]+P_000111111*R_112[0]+-1*P_000111211*R_113[0]+P_000211011*R_121[0]+-1*P_000211111*R_122[0]+P_000211211*R_123[0];
				double PR_000010012101=P_000010012*R_101[0]+-1*P_000010112*R_102[0]+P_000010212*R_103[0]+-1*P_000010312*R_104[0]+-1*P_000110012*R_111[0]+P_000110112*R_112[0]+-1*P_000110212*R_113[0]+P_000110312*R_114[0];
				double PR_002000020101=P_002000020*R_101[0]+-1*P_002000120*R_102[0]+P_002000220*R_103[0]+-1*P_102000020*R_201[0]+P_102000120*R_202[0]+-1*P_102000220*R_203[0]+P_202000020*R_301[0]+-1*P_202000120*R_302[0]+P_202000220*R_303[0];
				double PR_001001020101=P_001001020*R_101[0]+-1*P_001001120*R_102[0]+P_001001220*R_103[0]+-1*P_001101020*R_111[0]+P_001101120*R_112[0]+-1*P_001101220*R_113[0]+-1*P_101001020*R_201[0]+P_101001120*R_202[0]+-1*P_101001220*R_203[0]+P_101101020*R_211[0]+-1*P_101101120*R_212[0]+P_101101220*R_213[0];
				double PR_000002020101=P_000002020*R_101[0]+-1*P_000002120*R_102[0]+P_000002220*R_103[0]+-1*P_000102020*R_111[0]+P_000102120*R_112[0]+-1*P_000102220*R_113[0]+P_000202020*R_121[0]+-1*P_000202120*R_122[0]+P_000202220*R_123[0];
				double PR_001000021101=P_001000021*R_101[0]+-1*P_001000121*R_102[0]+P_001000221*R_103[0]+-1*P_001000321*R_104[0]+-1*P_101000021*R_201[0]+P_101000121*R_202[0]+-1*P_101000221*R_203[0]+P_101000321*R_204[0];
				double PR_000001021101=P_000001021*R_101[0]+-1*P_000001121*R_102[0]+P_000001221*R_103[0]+-1*P_000001321*R_104[0]+-1*P_000101021*R_111[0]+P_000101121*R_112[0]+-1*P_000101221*R_113[0]+P_000101321*R_114[0];
				double PR_000000022101=P_000000022*R_101[0]+-1*P_000000122*R_102[0]+P_000000222*R_103[0]+-1*P_000000322*R_104[0]+P_000000422*R_105[0];
				double PR_022000000110=P_022000000*R_110[0]+-1*P_122000000*R_210[0]+P_222000000*R_310[0]+-1*P_322000000*R_410[0]+P_422000000*R_510[0];
				double PR_021001000110=P_021001000*R_110[0]+-1*P_021101000*R_120[0]+-1*P_121001000*R_210[0]+P_121101000*R_220[0]+P_221001000*R_310[0]+-1*P_221101000*R_320[0]+-1*P_321001000*R_410[0]+P_321101000*R_420[0];
				double PR_020002000110=P_020002000*R_110[0]+-1*P_020102000*R_120[0]+P_020202000*R_130[0]+-1*P_120002000*R_210[0]+P_120102000*R_220[0]+-1*P_120202000*R_230[0]+P_220002000*R_310[0]+-1*P_220102000*R_320[0]+P_220202000*R_330[0];
				double PR_021000001110=P_021000001*R_110[0]+-1*P_021000101*R_111[0]+-1*P_121000001*R_210[0]+P_121000101*R_211[0]+P_221000001*R_310[0]+-1*P_221000101*R_311[0]+-1*P_321000001*R_410[0]+P_321000101*R_411[0];
				double PR_020001001110=P_020001001*R_110[0]+-1*P_020001101*R_111[0]+-1*P_020101001*R_120[0]+P_020101101*R_121[0]+-1*P_120001001*R_210[0]+P_120001101*R_211[0]+P_120101001*R_220[0]+-1*P_120101101*R_221[0]+P_220001001*R_310[0]+-1*P_220001101*R_311[0]+-1*P_220101001*R_320[0]+P_220101101*R_321[0];
				double PR_020000002110=P_020000002*R_110[0]+-1*P_020000102*R_111[0]+P_020000202*R_112[0]+-1*P_120000002*R_210[0]+P_120000102*R_211[0]+-1*P_120000202*R_212[0]+P_220000002*R_310[0]+-1*P_220000102*R_311[0]+P_220000202*R_312[0];
				double PR_012010000110=P_012010000*R_110[0]+-1*P_012110000*R_120[0]+-1*P_112010000*R_210[0]+P_112110000*R_220[0]+P_212010000*R_310[0]+-1*P_212110000*R_320[0]+-1*P_312010000*R_410[0]+P_312110000*R_420[0];
				double PR_011011000110=P_011011000*R_110[0]+-1*P_011111000*R_120[0]+P_011211000*R_130[0]+-1*P_111011000*R_210[0]+P_111111000*R_220[0]+-1*P_111211000*R_230[0]+P_211011000*R_310[0]+-1*P_211111000*R_320[0]+P_211211000*R_330[0];
				double PR_010012000110=P_010012000*R_110[0]+-1*P_010112000*R_120[0]+P_010212000*R_130[0]+-1*P_010312000*R_140[0]+-1*P_110012000*R_210[0]+P_110112000*R_220[0]+-1*P_110212000*R_230[0]+P_110312000*R_240[0];
				double PR_011010001110=P_011010001*R_110[0]+-1*P_011010101*R_111[0]+-1*P_011110001*R_120[0]+P_011110101*R_121[0]+-1*P_111010001*R_210[0]+P_111010101*R_211[0]+P_111110001*R_220[0]+-1*P_111110101*R_221[0]+P_211010001*R_310[0]+-1*P_211010101*R_311[0]+-1*P_211110001*R_320[0]+P_211110101*R_321[0];
				double PR_010011001110=P_010011001*R_110[0]+-1*P_010011101*R_111[0]+-1*P_010111001*R_120[0]+P_010111101*R_121[0]+P_010211001*R_130[0]+-1*P_010211101*R_131[0]+-1*P_110011001*R_210[0]+P_110011101*R_211[0]+P_110111001*R_220[0]+-1*P_110111101*R_221[0]+-1*P_110211001*R_230[0]+P_110211101*R_231[0];
				double PR_010010002110=P_010010002*R_110[0]+-1*P_010010102*R_111[0]+P_010010202*R_112[0]+-1*P_010110002*R_120[0]+P_010110102*R_121[0]+-1*P_010110202*R_122[0]+-1*P_110010002*R_210[0]+P_110010102*R_211[0]+-1*P_110010202*R_212[0]+P_110110002*R_220[0]+-1*P_110110102*R_221[0]+P_110110202*R_222[0];
				double PR_002020000110=P_002020000*R_110[0]+-1*P_002120000*R_120[0]+P_002220000*R_130[0]+-1*P_102020000*R_210[0]+P_102120000*R_220[0]+-1*P_102220000*R_230[0]+P_202020000*R_310[0]+-1*P_202120000*R_320[0]+P_202220000*R_330[0];
				double PR_001021000110=P_001021000*R_110[0]+-1*P_001121000*R_120[0]+P_001221000*R_130[0]+-1*P_001321000*R_140[0]+-1*P_101021000*R_210[0]+P_101121000*R_220[0]+-1*P_101221000*R_230[0]+P_101321000*R_240[0];
				double PR_000022000110=P_000022000*R_110[0]+-1*P_000122000*R_120[0]+P_000222000*R_130[0]+-1*P_000322000*R_140[0]+P_000422000*R_150[0];
				double PR_001020001110=P_001020001*R_110[0]+-1*P_001020101*R_111[0]+-1*P_001120001*R_120[0]+P_001120101*R_121[0]+P_001220001*R_130[0]+-1*P_001220101*R_131[0]+-1*P_101020001*R_210[0]+P_101020101*R_211[0]+P_101120001*R_220[0]+-1*P_101120101*R_221[0]+-1*P_101220001*R_230[0]+P_101220101*R_231[0];
				double PR_000021001110=P_000021001*R_110[0]+-1*P_000021101*R_111[0]+-1*P_000121001*R_120[0]+P_000121101*R_121[0]+P_000221001*R_130[0]+-1*P_000221101*R_131[0]+-1*P_000321001*R_140[0]+P_000321101*R_141[0];
				double PR_000020002110=P_000020002*R_110[0]+-1*P_000020102*R_111[0]+P_000020202*R_112[0]+-1*P_000120002*R_120[0]+P_000120102*R_121[0]+-1*P_000120202*R_122[0]+P_000220002*R_130[0]+-1*P_000220102*R_131[0]+P_000220202*R_132[0];
				double PR_012000010110=P_012000010*R_110[0]+-1*P_012000110*R_111[0]+-1*P_112000010*R_210[0]+P_112000110*R_211[0]+P_212000010*R_310[0]+-1*P_212000110*R_311[0]+-1*P_312000010*R_410[0]+P_312000110*R_411[0];
				double PR_011001010110=P_011001010*R_110[0]+-1*P_011001110*R_111[0]+-1*P_011101010*R_120[0]+P_011101110*R_121[0]+-1*P_111001010*R_210[0]+P_111001110*R_211[0]+P_111101010*R_220[0]+-1*P_111101110*R_221[0]+P_211001010*R_310[0]+-1*P_211001110*R_311[0]+-1*P_211101010*R_320[0]+P_211101110*R_321[0];
				double PR_010002010110=P_010002010*R_110[0]+-1*P_010002110*R_111[0]+-1*P_010102010*R_120[0]+P_010102110*R_121[0]+P_010202010*R_130[0]+-1*P_010202110*R_131[0]+-1*P_110002010*R_210[0]+P_110002110*R_211[0]+P_110102010*R_220[0]+-1*P_110102110*R_221[0]+-1*P_110202010*R_230[0]+P_110202110*R_231[0];
				double PR_011000011110=P_011000011*R_110[0]+-1*P_011000111*R_111[0]+P_011000211*R_112[0]+-1*P_111000011*R_210[0]+P_111000111*R_211[0]+-1*P_111000211*R_212[0]+P_211000011*R_310[0]+-1*P_211000111*R_311[0]+P_211000211*R_312[0];
				double PR_010001011110=P_010001011*R_110[0]+-1*P_010001111*R_111[0]+P_010001211*R_112[0]+-1*P_010101011*R_120[0]+P_010101111*R_121[0]+-1*P_010101211*R_122[0]+-1*P_110001011*R_210[0]+P_110001111*R_211[0]+-1*P_110001211*R_212[0]+P_110101011*R_220[0]+-1*P_110101111*R_221[0]+P_110101211*R_222[0];
				double PR_010000012110=P_010000012*R_110[0]+-1*P_010000112*R_111[0]+P_010000212*R_112[0]+-1*P_010000312*R_113[0]+-1*P_110000012*R_210[0]+P_110000112*R_211[0]+-1*P_110000212*R_212[0]+P_110000312*R_213[0];
				double PR_002010010110=P_002010010*R_110[0]+-1*P_002010110*R_111[0]+-1*P_002110010*R_120[0]+P_002110110*R_121[0]+-1*P_102010010*R_210[0]+P_102010110*R_211[0]+P_102110010*R_220[0]+-1*P_102110110*R_221[0]+P_202010010*R_310[0]+-1*P_202010110*R_311[0]+-1*P_202110010*R_320[0]+P_202110110*R_321[0];
				double PR_001011010110=P_001011010*R_110[0]+-1*P_001011110*R_111[0]+-1*P_001111010*R_120[0]+P_001111110*R_121[0]+P_001211010*R_130[0]+-1*P_001211110*R_131[0]+-1*P_101011010*R_210[0]+P_101011110*R_211[0]+P_101111010*R_220[0]+-1*P_101111110*R_221[0]+-1*P_101211010*R_230[0]+P_101211110*R_231[0];
				double PR_000012010110=P_000012010*R_110[0]+-1*P_000012110*R_111[0]+-1*P_000112010*R_120[0]+P_000112110*R_121[0]+P_000212010*R_130[0]+-1*P_000212110*R_131[0]+-1*P_000312010*R_140[0]+P_000312110*R_141[0];
				double PR_001010011110=P_001010011*R_110[0]+-1*P_001010111*R_111[0]+P_001010211*R_112[0]+-1*P_001110011*R_120[0]+P_001110111*R_121[0]+-1*P_001110211*R_122[0]+-1*P_101010011*R_210[0]+P_101010111*R_211[0]+-1*P_101010211*R_212[0]+P_101110011*R_220[0]+-1*P_101110111*R_221[0]+P_101110211*R_222[0];
				double PR_000011011110=P_000011011*R_110[0]+-1*P_000011111*R_111[0]+P_000011211*R_112[0]+-1*P_000111011*R_120[0]+P_000111111*R_121[0]+-1*P_000111211*R_122[0]+P_000211011*R_130[0]+-1*P_000211111*R_131[0]+P_000211211*R_132[0];
				double PR_000010012110=P_000010012*R_110[0]+-1*P_000010112*R_111[0]+P_000010212*R_112[0]+-1*P_000010312*R_113[0]+-1*P_000110012*R_120[0]+P_000110112*R_121[0]+-1*P_000110212*R_122[0]+P_000110312*R_123[0];
				double PR_002000020110=P_002000020*R_110[0]+-1*P_002000120*R_111[0]+P_002000220*R_112[0]+-1*P_102000020*R_210[0]+P_102000120*R_211[0]+-1*P_102000220*R_212[0]+P_202000020*R_310[0]+-1*P_202000120*R_311[0]+P_202000220*R_312[0];
				double PR_001001020110=P_001001020*R_110[0]+-1*P_001001120*R_111[0]+P_001001220*R_112[0]+-1*P_001101020*R_120[0]+P_001101120*R_121[0]+-1*P_001101220*R_122[0]+-1*P_101001020*R_210[0]+P_101001120*R_211[0]+-1*P_101001220*R_212[0]+P_101101020*R_220[0]+-1*P_101101120*R_221[0]+P_101101220*R_222[0];
				double PR_000002020110=P_000002020*R_110[0]+-1*P_000002120*R_111[0]+P_000002220*R_112[0]+-1*P_000102020*R_120[0]+P_000102120*R_121[0]+-1*P_000102220*R_122[0]+P_000202020*R_130[0]+-1*P_000202120*R_131[0]+P_000202220*R_132[0];
				double PR_001000021110=P_001000021*R_110[0]+-1*P_001000121*R_111[0]+P_001000221*R_112[0]+-1*P_001000321*R_113[0]+-1*P_101000021*R_210[0]+P_101000121*R_211[0]+-1*P_101000221*R_212[0]+P_101000321*R_213[0];
				double PR_000001021110=P_000001021*R_110[0]+-1*P_000001121*R_111[0]+P_000001221*R_112[0]+-1*P_000001321*R_113[0]+-1*P_000101021*R_120[0]+P_000101121*R_121[0]+-1*P_000101221*R_122[0]+P_000101321*R_123[0];
				double PR_000000022110=P_000000022*R_110[0]+-1*P_000000122*R_111[0]+P_000000222*R_112[0]+-1*P_000000322*R_113[0]+P_000000422*R_114[0];
				double PR_022000000200=P_022000000*R_200[0]+-1*P_122000000*R_300[0]+P_222000000*R_400[0]+-1*P_322000000*R_500[0]+P_422000000*R_600[0];
				double PR_021001000200=P_021001000*R_200[0]+-1*P_021101000*R_210[0]+-1*P_121001000*R_300[0]+P_121101000*R_310[0]+P_221001000*R_400[0]+-1*P_221101000*R_410[0]+-1*P_321001000*R_500[0]+P_321101000*R_510[0];
				double PR_020002000200=P_020002000*R_200[0]+-1*P_020102000*R_210[0]+P_020202000*R_220[0]+-1*P_120002000*R_300[0]+P_120102000*R_310[0]+-1*P_120202000*R_320[0]+P_220002000*R_400[0]+-1*P_220102000*R_410[0]+P_220202000*R_420[0];
				double PR_021000001200=P_021000001*R_200[0]+-1*P_021000101*R_201[0]+-1*P_121000001*R_300[0]+P_121000101*R_301[0]+P_221000001*R_400[0]+-1*P_221000101*R_401[0]+-1*P_321000001*R_500[0]+P_321000101*R_501[0];
				double PR_020001001200=P_020001001*R_200[0]+-1*P_020001101*R_201[0]+-1*P_020101001*R_210[0]+P_020101101*R_211[0]+-1*P_120001001*R_300[0]+P_120001101*R_301[0]+P_120101001*R_310[0]+-1*P_120101101*R_311[0]+P_220001001*R_400[0]+-1*P_220001101*R_401[0]+-1*P_220101001*R_410[0]+P_220101101*R_411[0];
				double PR_020000002200=P_020000002*R_200[0]+-1*P_020000102*R_201[0]+P_020000202*R_202[0]+-1*P_120000002*R_300[0]+P_120000102*R_301[0]+-1*P_120000202*R_302[0]+P_220000002*R_400[0]+-1*P_220000102*R_401[0]+P_220000202*R_402[0];
				double PR_012010000200=P_012010000*R_200[0]+-1*P_012110000*R_210[0]+-1*P_112010000*R_300[0]+P_112110000*R_310[0]+P_212010000*R_400[0]+-1*P_212110000*R_410[0]+-1*P_312010000*R_500[0]+P_312110000*R_510[0];
				double PR_011011000200=P_011011000*R_200[0]+-1*P_011111000*R_210[0]+P_011211000*R_220[0]+-1*P_111011000*R_300[0]+P_111111000*R_310[0]+-1*P_111211000*R_320[0]+P_211011000*R_400[0]+-1*P_211111000*R_410[0]+P_211211000*R_420[0];
				double PR_010012000200=P_010012000*R_200[0]+-1*P_010112000*R_210[0]+P_010212000*R_220[0]+-1*P_010312000*R_230[0]+-1*P_110012000*R_300[0]+P_110112000*R_310[0]+-1*P_110212000*R_320[0]+P_110312000*R_330[0];
				double PR_011010001200=P_011010001*R_200[0]+-1*P_011010101*R_201[0]+-1*P_011110001*R_210[0]+P_011110101*R_211[0]+-1*P_111010001*R_300[0]+P_111010101*R_301[0]+P_111110001*R_310[0]+-1*P_111110101*R_311[0]+P_211010001*R_400[0]+-1*P_211010101*R_401[0]+-1*P_211110001*R_410[0]+P_211110101*R_411[0];
				double PR_010011001200=P_010011001*R_200[0]+-1*P_010011101*R_201[0]+-1*P_010111001*R_210[0]+P_010111101*R_211[0]+P_010211001*R_220[0]+-1*P_010211101*R_221[0]+-1*P_110011001*R_300[0]+P_110011101*R_301[0]+P_110111001*R_310[0]+-1*P_110111101*R_311[0]+-1*P_110211001*R_320[0]+P_110211101*R_321[0];
				double PR_010010002200=P_010010002*R_200[0]+-1*P_010010102*R_201[0]+P_010010202*R_202[0]+-1*P_010110002*R_210[0]+P_010110102*R_211[0]+-1*P_010110202*R_212[0]+-1*P_110010002*R_300[0]+P_110010102*R_301[0]+-1*P_110010202*R_302[0]+P_110110002*R_310[0]+-1*P_110110102*R_311[0]+P_110110202*R_312[0];
				double PR_002020000200=P_002020000*R_200[0]+-1*P_002120000*R_210[0]+P_002220000*R_220[0]+-1*P_102020000*R_300[0]+P_102120000*R_310[0]+-1*P_102220000*R_320[0]+P_202020000*R_400[0]+-1*P_202120000*R_410[0]+P_202220000*R_420[0];
				double PR_001021000200=P_001021000*R_200[0]+-1*P_001121000*R_210[0]+P_001221000*R_220[0]+-1*P_001321000*R_230[0]+-1*P_101021000*R_300[0]+P_101121000*R_310[0]+-1*P_101221000*R_320[0]+P_101321000*R_330[0];
				double PR_000022000200=P_000022000*R_200[0]+-1*P_000122000*R_210[0]+P_000222000*R_220[0]+-1*P_000322000*R_230[0]+P_000422000*R_240[0];
				double PR_001020001200=P_001020001*R_200[0]+-1*P_001020101*R_201[0]+-1*P_001120001*R_210[0]+P_001120101*R_211[0]+P_001220001*R_220[0]+-1*P_001220101*R_221[0]+-1*P_101020001*R_300[0]+P_101020101*R_301[0]+P_101120001*R_310[0]+-1*P_101120101*R_311[0]+-1*P_101220001*R_320[0]+P_101220101*R_321[0];
				double PR_000021001200=P_000021001*R_200[0]+-1*P_000021101*R_201[0]+-1*P_000121001*R_210[0]+P_000121101*R_211[0]+P_000221001*R_220[0]+-1*P_000221101*R_221[0]+-1*P_000321001*R_230[0]+P_000321101*R_231[0];
				double PR_000020002200=P_000020002*R_200[0]+-1*P_000020102*R_201[0]+P_000020202*R_202[0]+-1*P_000120002*R_210[0]+P_000120102*R_211[0]+-1*P_000120202*R_212[0]+P_000220002*R_220[0]+-1*P_000220102*R_221[0]+P_000220202*R_222[0];
				double PR_012000010200=P_012000010*R_200[0]+-1*P_012000110*R_201[0]+-1*P_112000010*R_300[0]+P_112000110*R_301[0]+P_212000010*R_400[0]+-1*P_212000110*R_401[0]+-1*P_312000010*R_500[0]+P_312000110*R_501[0];
				double PR_011001010200=P_011001010*R_200[0]+-1*P_011001110*R_201[0]+-1*P_011101010*R_210[0]+P_011101110*R_211[0]+-1*P_111001010*R_300[0]+P_111001110*R_301[0]+P_111101010*R_310[0]+-1*P_111101110*R_311[0]+P_211001010*R_400[0]+-1*P_211001110*R_401[0]+-1*P_211101010*R_410[0]+P_211101110*R_411[0];
				double PR_010002010200=P_010002010*R_200[0]+-1*P_010002110*R_201[0]+-1*P_010102010*R_210[0]+P_010102110*R_211[0]+P_010202010*R_220[0]+-1*P_010202110*R_221[0]+-1*P_110002010*R_300[0]+P_110002110*R_301[0]+P_110102010*R_310[0]+-1*P_110102110*R_311[0]+-1*P_110202010*R_320[0]+P_110202110*R_321[0];
				double PR_011000011200=P_011000011*R_200[0]+-1*P_011000111*R_201[0]+P_011000211*R_202[0]+-1*P_111000011*R_300[0]+P_111000111*R_301[0]+-1*P_111000211*R_302[0]+P_211000011*R_400[0]+-1*P_211000111*R_401[0]+P_211000211*R_402[0];
				double PR_010001011200=P_010001011*R_200[0]+-1*P_010001111*R_201[0]+P_010001211*R_202[0]+-1*P_010101011*R_210[0]+P_010101111*R_211[0]+-1*P_010101211*R_212[0]+-1*P_110001011*R_300[0]+P_110001111*R_301[0]+-1*P_110001211*R_302[0]+P_110101011*R_310[0]+-1*P_110101111*R_311[0]+P_110101211*R_312[0];
				double PR_010000012200=P_010000012*R_200[0]+-1*P_010000112*R_201[0]+P_010000212*R_202[0]+-1*P_010000312*R_203[0]+-1*P_110000012*R_300[0]+P_110000112*R_301[0]+-1*P_110000212*R_302[0]+P_110000312*R_303[0];
				double PR_002010010200=P_002010010*R_200[0]+-1*P_002010110*R_201[0]+-1*P_002110010*R_210[0]+P_002110110*R_211[0]+-1*P_102010010*R_300[0]+P_102010110*R_301[0]+P_102110010*R_310[0]+-1*P_102110110*R_311[0]+P_202010010*R_400[0]+-1*P_202010110*R_401[0]+-1*P_202110010*R_410[0]+P_202110110*R_411[0];
				double PR_001011010200=P_001011010*R_200[0]+-1*P_001011110*R_201[0]+-1*P_001111010*R_210[0]+P_001111110*R_211[0]+P_001211010*R_220[0]+-1*P_001211110*R_221[0]+-1*P_101011010*R_300[0]+P_101011110*R_301[0]+P_101111010*R_310[0]+-1*P_101111110*R_311[0]+-1*P_101211010*R_320[0]+P_101211110*R_321[0];
				double PR_000012010200=P_000012010*R_200[0]+-1*P_000012110*R_201[0]+-1*P_000112010*R_210[0]+P_000112110*R_211[0]+P_000212010*R_220[0]+-1*P_000212110*R_221[0]+-1*P_000312010*R_230[0]+P_000312110*R_231[0];
				double PR_001010011200=P_001010011*R_200[0]+-1*P_001010111*R_201[0]+P_001010211*R_202[0]+-1*P_001110011*R_210[0]+P_001110111*R_211[0]+-1*P_001110211*R_212[0]+-1*P_101010011*R_300[0]+P_101010111*R_301[0]+-1*P_101010211*R_302[0]+P_101110011*R_310[0]+-1*P_101110111*R_311[0]+P_101110211*R_312[0];
				double PR_000011011200=P_000011011*R_200[0]+-1*P_000011111*R_201[0]+P_000011211*R_202[0]+-1*P_000111011*R_210[0]+P_000111111*R_211[0]+-1*P_000111211*R_212[0]+P_000211011*R_220[0]+-1*P_000211111*R_221[0]+P_000211211*R_222[0];
				double PR_000010012200=P_000010012*R_200[0]+-1*P_000010112*R_201[0]+P_000010212*R_202[0]+-1*P_000010312*R_203[0]+-1*P_000110012*R_210[0]+P_000110112*R_211[0]+-1*P_000110212*R_212[0]+P_000110312*R_213[0];
				double PR_002000020200=P_002000020*R_200[0]+-1*P_002000120*R_201[0]+P_002000220*R_202[0]+-1*P_102000020*R_300[0]+P_102000120*R_301[0]+-1*P_102000220*R_302[0]+P_202000020*R_400[0]+-1*P_202000120*R_401[0]+P_202000220*R_402[0];
				double PR_001001020200=P_001001020*R_200[0]+-1*P_001001120*R_201[0]+P_001001220*R_202[0]+-1*P_001101020*R_210[0]+P_001101120*R_211[0]+-1*P_001101220*R_212[0]+-1*P_101001020*R_300[0]+P_101001120*R_301[0]+-1*P_101001220*R_302[0]+P_101101020*R_310[0]+-1*P_101101120*R_311[0]+P_101101220*R_312[0];
				double PR_000002020200=P_000002020*R_200[0]+-1*P_000002120*R_201[0]+P_000002220*R_202[0]+-1*P_000102020*R_210[0]+P_000102120*R_211[0]+-1*P_000102220*R_212[0]+P_000202020*R_220[0]+-1*P_000202120*R_221[0]+P_000202220*R_222[0];
				double PR_001000021200=P_001000021*R_200[0]+-1*P_001000121*R_201[0]+P_001000221*R_202[0]+-1*P_001000321*R_203[0]+-1*P_101000021*R_300[0]+P_101000121*R_301[0]+-1*P_101000221*R_302[0]+P_101000321*R_303[0];
				double PR_000001021200=P_000001021*R_200[0]+-1*P_000001121*R_201[0]+P_000001221*R_202[0]+-1*P_000001321*R_203[0]+-1*P_000101021*R_210[0]+P_000101121*R_211[0]+-1*P_000101221*R_212[0]+P_000101321*R_213[0];
				double PR_000000022200=P_000000022*R_200[0]+-1*P_000000122*R_201[0]+P_000000222*R_202[0]+-1*P_000000322*R_203[0]+P_000000422*R_204[0];
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(Q_020000000*PR_022000000000+Q_120000000*PR_022000000100+Q_220000000*PR_022000000200);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(Q_010010000*PR_022000000000+Q_010110000*PR_022000000010+Q_110010000*PR_022000000100+Q_110110000*PR_022000000110);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(Q_000020000*PR_022000000000+Q_000120000*PR_022000000010+Q_000220000*PR_022000000020);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(Q_010000010*PR_022000000000+Q_010000110*PR_022000000001+Q_110000010*PR_022000000100+Q_110000110*PR_022000000101);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(Q_000010010*PR_022000000000+Q_000010110*PR_022000000001+Q_000110010*PR_022000000010+Q_000110110*PR_022000000011);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(Q_000000020*PR_022000000000+Q_000000120*PR_022000000001+Q_000000220*PR_022000000002);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(Q_020000000*PR_021001000000+Q_120000000*PR_021001000100+Q_220000000*PR_021001000200);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(Q_010010000*PR_021001000000+Q_010110000*PR_021001000010+Q_110010000*PR_021001000100+Q_110110000*PR_021001000110);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(Q_000020000*PR_021001000000+Q_000120000*PR_021001000010+Q_000220000*PR_021001000020);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(Q_010000010*PR_021001000000+Q_010000110*PR_021001000001+Q_110000010*PR_021001000100+Q_110000110*PR_021001000101);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(Q_000010010*PR_021001000000+Q_000010110*PR_021001000001+Q_000110010*PR_021001000010+Q_000110110*PR_021001000011);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(Q_000000020*PR_021001000000+Q_000000120*PR_021001000001+Q_000000220*PR_021001000002);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(Q_020000000*PR_020002000000+Q_120000000*PR_020002000100+Q_220000000*PR_020002000200);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(Q_010010000*PR_020002000000+Q_010110000*PR_020002000010+Q_110010000*PR_020002000100+Q_110110000*PR_020002000110);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(Q_000020000*PR_020002000000+Q_000120000*PR_020002000010+Q_000220000*PR_020002000020);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(Q_010000010*PR_020002000000+Q_010000110*PR_020002000001+Q_110000010*PR_020002000100+Q_110000110*PR_020002000101);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(Q_000010010*PR_020002000000+Q_000010110*PR_020002000001+Q_000110010*PR_020002000010+Q_000110110*PR_020002000011);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(Q_000000020*PR_020002000000+Q_000000120*PR_020002000001+Q_000000220*PR_020002000002);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(Q_020000000*PR_021000001000+Q_120000000*PR_021000001100+Q_220000000*PR_021000001200);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(Q_010010000*PR_021000001000+Q_010110000*PR_021000001010+Q_110010000*PR_021000001100+Q_110110000*PR_021000001110);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(Q_000020000*PR_021000001000+Q_000120000*PR_021000001010+Q_000220000*PR_021000001020);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(Q_010000010*PR_021000001000+Q_010000110*PR_021000001001+Q_110000010*PR_021000001100+Q_110000110*PR_021000001101);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(Q_000010010*PR_021000001000+Q_000010110*PR_021000001001+Q_000110010*PR_021000001010+Q_000110110*PR_021000001011);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(Q_000000020*PR_021000001000+Q_000000120*PR_021000001001+Q_000000220*PR_021000001002);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(Q_020000000*PR_020001001000+Q_120000000*PR_020001001100+Q_220000000*PR_020001001200);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(Q_010010000*PR_020001001000+Q_010110000*PR_020001001010+Q_110010000*PR_020001001100+Q_110110000*PR_020001001110);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(Q_000020000*PR_020001001000+Q_000120000*PR_020001001010+Q_000220000*PR_020001001020);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(Q_010000010*PR_020001001000+Q_010000110*PR_020001001001+Q_110000010*PR_020001001100+Q_110000110*PR_020001001101);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(Q_000010010*PR_020001001000+Q_000010110*PR_020001001001+Q_000110010*PR_020001001010+Q_000110110*PR_020001001011);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(Q_000000020*PR_020001001000+Q_000000120*PR_020001001001+Q_000000220*PR_020001001002);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(Q_020000000*PR_020000002000+Q_120000000*PR_020000002100+Q_220000000*PR_020000002200);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(Q_010010000*PR_020000002000+Q_010110000*PR_020000002010+Q_110010000*PR_020000002100+Q_110110000*PR_020000002110);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(Q_000020000*PR_020000002000+Q_000120000*PR_020000002010+Q_000220000*PR_020000002020);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(Q_010000010*PR_020000002000+Q_010000110*PR_020000002001+Q_110000010*PR_020000002100+Q_110000110*PR_020000002101);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(Q_000010010*PR_020000002000+Q_000010110*PR_020000002001+Q_000110010*PR_020000002010+Q_000110110*PR_020000002011);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(Q_000000020*PR_020000002000+Q_000000120*PR_020000002001+Q_000000220*PR_020000002002);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(Q_020000000*PR_012010000000+Q_120000000*PR_012010000100+Q_220000000*PR_012010000200);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(Q_010010000*PR_012010000000+Q_010110000*PR_012010000010+Q_110010000*PR_012010000100+Q_110110000*PR_012010000110);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(Q_000020000*PR_012010000000+Q_000120000*PR_012010000010+Q_000220000*PR_012010000020);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(Q_010000010*PR_012010000000+Q_010000110*PR_012010000001+Q_110000010*PR_012010000100+Q_110000110*PR_012010000101);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(Q_000010010*PR_012010000000+Q_000010110*PR_012010000001+Q_000110010*PR_012010000010+Q_000110110*PR_012010000011);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(Q_000000020*PR_012010000000+Q_000000120*PR_012010000001+Q_000000220*PR_012010000002);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(Q_020000000*PR_011011000000+Q_120000000*PR_011011000100+Q_220000000*PR_011011000200);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(Q_010010000*PR_011011000000+Q_010110000*PR_011011000010+Q_110010000*PR_011011000100+Q_110110000*PR_011011000110);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(Q_000020000*PR_011011000000+Q_000120000*PR_011011000010+Q_000220000*PR_011011000020);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(Q_010000010*PR_011011000000+Q_010000110*PR_011011000001+Q_110000010*PR_011011000100+Q_110000110*PR_011011000101);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(Q_000010010*PR_011011000000+Q_000010110*PR_011011000001+Q_000110010*PR_011011000010+Q_000110110*PR_011011000011);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(Q_000000020*PR_011011000000+Q_000000120*PR_011011000001+Q_000000220*PR_011011000002);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(Q_020000000*PR_010012000000+Q_120000000*PR_010012000100+Q_220000000*PR_010012000200);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(Q_010010000*PR_010012000000+Q_010110000*PR_010012000010+Q_110010000*PR_010012000100+Q_110110000*PR_010012000110);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(Q_000020000*PR_010012000000+Q_000120000*PR_010012000010+Q_000220000*PR_010012000020);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(Q_010000010*PR_010012000000+Q_010000110*PR_010012000001+Q_110000010*PR_010012000100+Q_110000110*PR_010012000101);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(Q_000010010*PR_010012000000+Q_000010110*PR_010012000001+Q_000110010*PR_010012000010+Q_000110110*PR_010012000011);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(Q_000000020*PR_010012000000+Q_000000120*PR_010012000001+Q_000000220*PR_010012000002);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(Q_020000000*PR_011010001000+Q_120000000*PR_011010001100+Q_220000000*PR_011010001200);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(Q_010010000*PR_011010001000+Q_010110000*PR_011010001010+Q_110010000*PR_011010001100+Q_110110000*PR_011010001110);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(Q_000020000*PR_011010001000+Q_000120000*PR_011010001010+Q_000220000*PR_011010001020);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(Q_010000010*PR_011010001000+Q_010000110*PR_011010001001+Q_110000010*PR_011010001100+Q_110000110*PR_011010001101);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(Q_000010010*PR_011010001000+Q_000010110*PR_011010001001+Q_000110010*PR_011010001010+Q_000110110*PR_011010001011);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(Q_000000020*PR_011010001000+Q_000000120*PR_011010001001+Q_000000220*PR_011010001002);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(Q_020000000*PR_010011001000+Q_120000000*PR_010011001100+Q_220000000*PR_010011001200);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(Q_010010000*PR_010011001000+Q_010110000*PR_010011001010+Q_110010000*PR_010011001100+Q_110110000*PR_010011001110);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(Q_000020000*PR_010011001000+Q_000120000*PR_010011001010+Q_000220000*PR_010011001020);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(Q_010000010*PR_010011001000+Q_010000110*PR_010011001001+Q_110000010*PR_010011001100+Q_110000110*PR_010011001101);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(Q_000010010*PR_010011001000+Q_000010110*PR_010011001001+Q_000110010*PR_010011001010+Q_000110110*PR_010011001011);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(Q_000000020*PR_010011001000+Q_000000120*PR_010011001001+Q_000000220*PR_010011001002);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(Q_020000000*PR_010010002000+Q_120000000*PR_010010002100+Q_220000000*PR_010010002200);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(Q_010010000*PR_010010002000+Q_010110000*PR_010010002010+Q_110010000*PR_010010002100+Q_110110000*PR_010010002110);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(Q_000020000*PR_010010002000+Q_000120000*PR_010010002010+Q_000220000*PR_010010002020);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(Q_010000010*PR_010010002000+Q_010000110*PR_010010002001+Q_110000010*PR_010010002100+Q_110000110*PR_010010002101);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(Q_000010010*PR_010010002000+Q_000010110*PR_010010002001+Q_000110010*PR_010010002010+Q_000110110*PR_010010002011);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(Q_000000020*PR_010010002000+Q_000000120*PR_010010002001+Q_000000220*PR_010010002002);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(Q_020000000*PR_002020000000+Q_120000000*PR_002020000100+Q_220000000*PR_002020000200);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(Q_010010000*PR_002020000000+Q_010110000*PR_002020000010+Q_110010000*PR_002020000100+Q_110110000*PR_002020000110);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(Q_000020000*PR_002020000000+Q_000120000*PR_002020000010+Q_000220000*PR_002020000020);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(Q_010000010*PR_002020000000+Q_010000110*PR_002020000001+Q_110000010*PR_002020000100+Q_110000110*PR_002020000101);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(Q_000010010*PR_002020000000+Q_000010110*PR_002020000001+Q_000110010*PR_002020000010+Q_000110110*PR_002020000011);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(Q_000000020*PR_002020000000+Q_000000120*PR_002020000001+Q_000000220*PR_002020000002);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(Q_020000000*PR_001021000000+Q_120000000*PR_001021000100+Q_220000000*PR_001021000200);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(Q_010010000*PR_001021000000+Q_010110000*PR_001021000010+Q_110010000*PR_001021000100+Q_110110000*PR_001021000110);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(Q_000020000*PR_001021000000+Q_000120000*PR_001021000010+Q_000220000*PR_001021000020);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(Q_010000010*PR_001021000000+Q_010000110*PR_001021000001+Q_110000010*PR_001021000100+Q_110000110*PR_001021000101);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(Q_000010010*PR_001021000000+Q_000010110*PR_001021000001+Q_000110010*PR_001021000010+Q_000110110*PR_001021000011);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(Q_000000020*PR_001021000000+Q_000000120*PR_001021000001+Q_000000220*PR_001021000002);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(Q_020000000*PR_000022000000+Q_120000000*PR_000022000100+Q_220000000*PR_000022000200);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(Q_010010000*PR_000022000000+Q_010110000*PR_000022000010+Q_110010000*PR_000022000100+Q_110110000*PR_000022000110);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(Q_000020000*PR_000022000000+Q_000120000*PR_000022000010+Q_000220000*PR_000022000020);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(Q_010000010*PR_000022000000+Q_010000110*PR_000022000001+Q_110000010*PR_000022000100+Q_110000110*PR_000022000101);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(Q_000010010*PR_000022000000+Q_000010110*PR_000022000001+Q_000110010*PR_000022000010+Q_000110110*PR_000022000011);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(Q_000000020*PR_000022000000+Q_000000120*PR_000022000001+Q_000000220*PR_000022000002);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(Q_020000000*PR_001020001000+Q_120000000*PR_001020001100+Q_220000000*PR_001020001200);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(Q_010010000*PR_001020001000+Q_010110000*PR_001020001010+Q_110010000*PR_001020001100+Q_110110000*PR_001020001110);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(Q_000020000*PR_001020001000+Q_000120000*PR_001020001010+Q_000220000*PR_001020001020);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(Q_010000010*PR_001020001000+Q_010000110*PR_001020001001+Q_110000010*PR_001020001100+Q_110000110*PR_001020001101);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(Q_000010010*PR_001020001000+Q_000010110*PR_001020001001+Q_000110010*PR_001020001010+Q_000110110*PR_001020001011);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(Q_000000020*PR_001020001000+Q_000000120*PR_001020001001+Q_000000220*PR_001020001002);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(Q_020000000*PR_000021001000+Q_120000000*PR_000021001100+Q_220000000*PR_000021001200);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(Q_010010000*PR_000021001000+Q_010110000*PR_000021001010+Q_110010000*PR_000021001100+Q_110110000*PR_000021001110);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(Q_000020000*PR_000021001000+Q_000120000*PR_000021001010+Q_000220000*PR_000021001020);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(Q_010000010*PR_000021001000+Q_010000110*PR_000021001001+Q_110000010*PR_000021001100+Q_110000110*PR_000021001101);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(Q_000010010*PR_000021001000+Q_000010110*PR_000021001001+Q_000110010*PR_000021001010+Q_000110110*PR_000021001011);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(Q_000000020*PR_000021001000+Q_000000120*PR_000021001001+Q_000000220*PR_000021001002);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(Q_020000000*PR_000020002000+Q_120000000*PR_000020002100+Q_220000000*PR_000020002200);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(Q_010010000*PR_000020002000+Q_010110000*PR_000020002010+Q_110010000*PR_000020002100+Q_110110000*PR_000020002110);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(Q_000020000*PR_000020002000+Q_000120000*PR_000020002010+Q_000220000*PR_000020002020);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(Q_010000010*PR_000020002000+Q_010000110*PR_000020002001+Q_110000010*PR_000020002100+Q_110000110*PR_000020002101);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(Q_000010010*PR_000020002000+Q_000010110*PR_000020002001+Q_000110010*PR_000020002010+Q_000110110*PR_000020002011);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(Q_000000020*PR_000020002000+Q_000000120*PR_000020002001+Q_000000220*PR_000020002002);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(Q_020000000*PR_012000010000+Q_120000000*PR_012000010100+Q_220000000*PR_012000010200);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(Q_010010000*PR_012000010000+Q_010110000*PR_012000010010+Q_110010000*PR_012000010100+Q_110110000*PR_012000010110);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(Q_000020000*PR_012000010000+Q_000120000*PR_012000010010+Q_000220000*PR_012000010020);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(Q_010000010*PR_012000010000+Q_010000110*PR_012000010001+Q_110000010*PR_012000010100+Q_110000110*PR_012000010101);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(Q_000010010*PR_012000010000+Q_000010110*PR_012000010001+Q_000110010*PR_012000010010+Q_000110110*PR_012000010011);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(Q_000000020*PR_012000010000+Q_000000120*PR_012000010001+Q_000000220*PR_012000010002);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(Q_020000000*PR_011001010000+Q_120000000*PR_011001010100+Q_220000000*PR_011001010200);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(Q_010010000*PR_011001010000+Q_010110000*PR_011001010010+Q_110010000*PR_011001010100+Q_110110000*PR_011001010110);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(Q_000020000*PR_011001010000+Q_000120000*PR_011001010010+Q_000220000*PR_011001010020);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(Q_010000010*PR_011001010000+Q_010000110*PR_011001010001+Q_110000010*PR_011001010100+Q_110000110*PR_011001010101);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(Q_000010010*PR_011001010000+Q_000010110*PR_011001010001+Q_000110010*PR_011001010010+Q_000110110*PR_011001010011);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(Q_000000020*PR_011001010000+Q_000000120*PR_011001010001+Q_000000220*PR_011001010002);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(Q_020000000*PR_010002010000+Q_120000000*PR_010002010100+Q_220000000*PR_010002010200);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(Q_010010000*PR_010002010000+Q_010110000*PR_010002010010+Q_110010000*PR_010002010100+Q_110110000*PR_010002010110);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(Q_000020000*PR_010002010000+Q_000120000*PR_010002010010+Q_000220000*PR_010002010020);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(Q_010000010*PR_010002010000+Q_010000110*PR_010002010001+Q_110000010*PR_010002010100+Q_110000110*PR_010002010101);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(Q_000010010*PR_010002010000+Q_000010110*PR_010002010001+Q_000110010*PR_010002010010+Q_000110110*PR_010002010011);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(Q_000000020*PR_010002010000+Q_000000120*PR_010002010001+Q_000000220*PR_010002010002);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(Q_020000000*PR_011000011000+Q_120000000*PR_011000011100+Q_220000000*PR_011000011200);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(Q_010010000*PR_011000011000+Q_010110000*PR_011000011010+Q_110010000*PR_011000011100+Q_110110000*PR_011000011110);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(Q_000020000*PR_011000011000+Q_000120000*PR_011000011010+Q_000220000*PR_011000011020);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(Q_010000010*PR_011000011000+Q_010000110*PR_011000011001+Q_110000010*PR_011000011100+Q_110000110*PR_011000011101);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(Q_000010010*PR_011000011000+Q_000010110*PR_011000011001+Q_000110010*PR_011000011010+Q_000110110*PR_011000011011);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(Q_000000020*PR_011000011000+Q_000000120*PR_011000011001+Q_000000220*PR_011000011002);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(Q_020000000*PR_010001011000+Q_120000000*PR_010001011100+Q_220000000*PR_010001011200);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(Q_010010000*PR_010001011000+Q_010110000*PR_010001011010+Q_110010000*PR_010001011100+Q_110110000*PR_010001011110);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(Q_000020000*PR_010001011000+Q_000120000*PR_010001011010+Q_000220000*PR_010001011020);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(Q_010000010*PR_010001011000+Q_010000110*PR_010001011001+Q_110000010*PR_010001011100+Q_110000110*PR_010001011101);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(Q_000010010*PR_010001011000+Q_000010110*PR_010001011001+Q_000110010*PR_010001011010+Q_000110110*PR_010001011011);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(Q_000000020*PR_010001011000+Q_000000120*PR_010001011001+Q_000000220*PR_010001011002);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(Q_020000000*PR_010000012000+Q_120000000*PR_010000012100+Q_220000000*PR_010000012200);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(Q_010010000*PR_010000012000+Q_010110000*PR_010000012010+Q_110010000*PR_010000012100+Q_110110000*PR_010000012110);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(Q_000020000*PR_010000012000+Q_000120000*PR_010000012010+Q_000220000*PR_010000012020);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(Q_010000010*PR_010000012000+Q_010000110*PR_010000012001+Q_110000010*PR_010000012100+Q_110000110*PR_010000012101);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(Q_000010010*PR_010000012000+Q_000010110*PR_010000012001+Q_000110010*PR_010000012010+Q_000110110*PR_010000012011);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(Q_000000020*PR_010000012000+Q_000000120*PR_010000012001+Q_000000220*PR_010000012002);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(Q_020000000*PR_002010010000+Q_120000000*PR_002010010100+Q_220000000*PR_002010010200);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(Q_010010000*PR_002010010000+Q_010110000*PR_002010010010+Q_110010000*PR_002010010100+Q_110110000*PR_002010010110);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(Q_000020000*PR_002010010000+Q_000120000*PR_002010010010+Q_000220000*PR_002010010020);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(Q_010000010*PR_002010010000+Q_010000110*PR_002010010001+Q_110000010*PR_002010010100+Q_110000110*PR_002010010101);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(Q_000010010*PR_002010010000+Q_000010110*PR_002010010001+Q_000110010*PR_002010010010+Q_000110110*PR_002010010011);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(Q_000000020*PR_002010010000+Q_000000120*PR_002010010001+Q_000000220*PR_002010010002);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(Q_020000000*PR_001011010000+Q_120000000*PR_001011010100+Q_220000000*PR_001011010200);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(Q_010010000*PR_001011010000+Q_010110000*PR_001011010010+Q_110010000*PR_001011010100+Q_110110000*PR_001011010110);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(Q_000020000*PR_001011010000+Q_000120000*PR_001011010010+Q_000220000*PR_001011010020);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(Q_010000010*PR_001011010000+Q_010000110*PR_001011010001+Q_110000010*PR_001011010100+Q_110000110*PR_001011010101);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(Q_000010010*PR_001011010000+Q_000010110*PR_001011010001+Q_000110010*PR_001011010010+Q_000110110*PR_001011010011);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(Q_000000020*PR_001011010000+Q_000000120*PR_001011010001+Q_000000220*PR_001011010002);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(Q_020000000*PR_000012010000+Q_120000000*PR_000012010100+Q_220000000*PR_000012010200);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(Q_010010000*PR_000012010000+Q_010110000*PR_000012010010+Q_110010000*PR_000012010100+Q_110110000*PR_000012010110);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(Q_000020000*PR_000012010000+Q_000120000*PR_000012010010+Q_000220000*PR_000012010020);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(Q_010000010*PR_000012010000+Q_010000110*PR_000012010001+Q_110000010*PR_000012010100+Q_110000110*PR_000012010101);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(Q_000010010*PR_000012010000+Q_000010110*PR_000012010001+Q_000110010*PR_000012010010+Q_000110110*PR_000012010011);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(Q_000000020*PR_000012010000+Q_000000120*PR_000012010001+Q_000000220*PR_000012010002);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(Q_020000000*PR_001010011000+Q_120000000*PR_001010011100+Q_220000000*PR_001010011200);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(Q_010010000*PR_001010011000+Q_010110000*PR_001010011010+Q_110010000*PR_001010011100+Q_110110000*PR_001010011110);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(Q_000020000*PR_001010011000+Q_000120000*PR_001010011010+Q_000220000*PR_001010011020);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(Q_010000010*PR_001010011000+Q_010000110*PR_001010011001+Q_110000010*PR_001010011100+Q_110000110*PR_001010011101);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(Q_000010010*PR_001010011000+Q_000010110*PR_001010011001+Q_000110010*PR_001010011010+Q_000110110*PR_001010011011);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(Q_000000020*PR_001010011000+Q_000000120*PR_001010011001+Q_000000220*PR_001010011002);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(Q_020000000*PR_000011011000+Q_120000000*PR_000011011100+Q_220000000*PR_000011011200);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(Q_010010000*PR_000011011000+Q_010110000*PR_000011011010+Q_110010000*PR_000011011100+Q_110110000*PR_000011011110);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(Q_000020000*PR_000011011000+Q_000120000*PR_000011011010+Q_000220000*PR_000011011020);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(Q_010000010*PR_000011011000+Q_010000110*PR_000011011001+Q_110000010*PR_000011011100+Q_110000110*PR_000011011101);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(Q_000010010*PR_000011011000+Q_000010110*PR_000011011001+Q_000110010*PR_000011011010+Q_000110110*PR_000011011011);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(Q_000000020*PR_000011011000+Q_000000120*PR_000011011001+Q_000000220*PR_000011011002);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(Q_020000000*PR_000010012000+Q_120000000*PR_000010012100+Q_220000000*PR_000010012200);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(Q_010010000*PR_000010012000+Q_010110000*PR_000010012010+Q_110010000*PR_000010012100+Q_110110000*PR_000010012110);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(Q_000020000*PR_000010012000+Q_000120000*PR_000010012010+Q_000220000*PR_000010012020);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(Q_010000010*PR_000010012000+Q_010000110*PR_000010012001+Q_110000010*PR_000010012100+Q_110000110*PR_000010012101);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(Q_000010010*PR_000010012000+Q_000010110*PR_000010012001+Q_000110010*PR_000010012010+Q_000110110*PR_000010012011);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(Q_000000020*PR_000010012000+Q_000000120*PR_000010012001+Q_000000220*PR_000010012002);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(Q_020000000*PR_002000020000+Q_120000000*PR_002000020100+Q_220000000*PR_002000020200);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(Q_010010000*PR_002000020000+Q_010110000*PR_002000020010+Q_110010000*PR_002000020100+Q_110110000*PR_002000020110);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(Q_000020000*PR_002000020000+Q_000120000*PR_002000020010+Q_000220000*PR_002000020020);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(Q_010000010*PR_002000020000+Q_010000110*PR_002000020001+Q_110000010*PR_002000020100+Q_110000110*PR_002000020101);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(Q_000010010*PR_002000020000+Q_000010110*PR_002000020001+Q_000110010*PR_002000020010+Q_000110110*PR_002000020011);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(Q_000000020*PR_002000020000+Q_000000120*PR_002000020001+Q_000000220*PR_002000020002);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(Q_020000000*PR_001001020000+Q_120000000*PR_001001020100+Q_220000000*PR_001001020200);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(Q_010010000*PR_001001020000+Q_010110000*PR_001001020010+Q_110010000*PR_001001020100+Q_110110000*PR_001001020110);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(Q_000020000*PR_001001020000+Q_000120000*PR_001001020010+Q_000220000*PR_001001020020);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(Q_010000010*PR_001001020000+Q_010000110*PR_001001020001+Q_110000010*PR_001001020100+Q_110000110*PR_001001020101);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(Q_000010010*PR_001001020000+Q_000010110*PR_001001020001+Q_000110010*PR_001001020010+Q_000110110*PR_001001020011);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(Q_000000020*PR_001001020000+Q_000000120*PR_001001020001+Q_000000220*PR_001001020002);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(Q_020000000*PR_000002020000+Q_120000000*PR_000002020100+Q_220000000*PR_000002020200);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(Q_010010000*PR_000002020000+Q_010110000*PR_000002020010+Q_110010000*PR_000002020100+Q_110110000*PR_000002020110);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(Q_000020000*PR_000002020000+Q_000120000*PR_000002020010+Q_000220000*PR_000002020020);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(Q_010000010*PR_000002020000+Q_010000110*PR_000002020001+Q_110000010*PR_000002020100+Q_110000110*PR_000002020101);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(Q_000010010*PR_000002020000+Q_000010110*PR_000002020001+Q_000110010*PR_000002020010+Q_000110110*PR_000002020011);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(Q_000000020*PR_000002020000+Q_000000120*PR_000002020001+Q_000000220*PR_000002020002);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(Q_020000000*PR_001000021000+Q_120000000*PR_001000021100+Q_220000000*PR_001000021200);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(Q_010010000*PR_001000021000+Q_010110000*PR_001000021010+Q_110010000*PR_001000021100+Q_110110000*PR_001000021110);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(Q_000020000*PR_001000021000+Q_000120000*PR_001000021010+Q_000220000*PR_001000021020);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(Q_010000010*PR_001000021000+Q_010000110*PR_001000021001+Q_110000010*PR_001000021100+Q_110000110*PR_001000021101);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(Q_000010010*PR_001000021000+Q_000010110*PR_001000021001+Q_000110010*PR_001000021010+Q_000110110*PR_001000021011);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(Q_000000020*PR_001000021000+Q_000000120*PR_001000021001+Q_000000220*PR_001000021002);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(Q_020000000*PR_000001021000+Q_120000000*PR_000001021100+Q_220000000*PR_000001021200);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(Q_010010000*PR_000001021000+Q_010110000*PR_000001021010+Q_110010000*PR_000001021100+Q_110110000*PR_000001021110);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(Q_000020000*PR_000001021000+Q_000120000*PR_000001021010+Q_000220000*PR_000001021020);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(Q_010000010*PR_000001021000+Q_010000110*PR_000001021001+Q_110000010*PR_000001021100+Q_110000110*PR_000001021101);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(Q_000010010*PR_000001021000+Q_000010110*PR_000001021001+Q_000110010*PR_000001021010+Q_000110110*PR_000001021011);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(Q_000000020*PR_000001021000+Q_000000120*PR_000001021001+Q_000000220*PR_000001021002);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(Q_020000000*PR_000000022000+Q_120000000*PR_000000022100+Q_220000000*PR_000000022200);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(Q_010010000*PR_000000022000+Q_010110000*PR_000000022010+Q_110010000*PR_000000022100+Q_110110000*PR_000000022110);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(Q_000020000*PR_000000022000+Q_000120000*PR_000000022010+Q_000220000*PR_000000022020);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(Q_010000010*PR_000000022000+Q_010000110*PR_000000022001+Q_110000010*PR_000000022100+Q_110000110*PR_000000022101);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(Q_000010010*PR_000000022000+Q_000010110*PR_000000022001+Q_000110010*PR_000000022010+Q_000110110*PR_000000022011);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(Q_000000020*PR_000000022000+Q_000000120*PR_000000022001+Q_000000220*PR_000000022002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_ddds_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
                unsigned int * contrc_bra_id,\
                unsigned int * contrc_ket_id,\
                unsigned int mtrx_len,\
                double * Pmtrx_in,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p_in,\
                unsigned int * id_bra_in,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                unsigned int * id_ket_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int bId_y = blockIdx.y;
    unsigned int tdis = blockDim.x;
    unsigned int bdis_x = gridDim.x;
    unsigned int bdis_y = gridDim.y;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }

    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis_x){
    for(unsigned int j_contrc_ket=bId_y;j_contrc_ket<contrc_ket_num;j_contrc_ket+=bdis_y){
    unsigned int primit_bra_start = contrc_bra_id[i_contrc_bra  ];
    unsigned int primit_bra_end   = contrc_bra_id[i_contrc_bra+1];
    unsigned int primit_ket_start = contrc_ket_id[j_contrc_ket  ];
    unsigned int primit_ket_end   = contrc_ket_id[j_contrc_ket+1];
        for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
            unsigned int id_bra=id_bra_in[ii];
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Pd_001[3];
				Pd_001[0]=PB[ii*3+0];
				Pd_001[1]=PB[ii*3+1];
				Pd_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
            float K2_p=K2_p_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_end-primit_ket_start;j+=tdis){
            unsigned int jj=primit_ket_start+j;
            unsigned int id_ket=tex1Dfetch(tex_id_ket,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<1;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*1+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*1+p_j]);
                if(temp_P>P_max) P_max=temp_P;
            }
            }
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p*K2_q)<1.0E-14){
                    break;
                }
            if(fabsf(P_max*K2_p*K2_q)<1.0E-14) continue;
            int2 temp_int2;
            temp_int2=tex1Dfetch(tex_Eta,jj);
            double Eta=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_pq,jj);
            double pq=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+0);
            double QX=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+1);
            double QY=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_Q,jj*3+2);
            double QZ=__hiloint2double(temp_int2.y,temp_int2.x);
				double Qd_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            Qd_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            Qd_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            Qd_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[7];
                Ft_fs_6(6,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[6]*=64*alphaT*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[6];
	double R_200[5];
	double R_300[4];
	double R_400[3];
	double R_500[2];
	double R_600[1];
	double R_010[6];
	double R_110[5];
	double R_210[4];
	double R_310[3];
	double R_410[2];
	double R_510[1];
	double R_020[5];
	double R_120[4];
	double R_220[3];
	double R_320[2];
	double R_420[1];
	double R_030[4];
	double R_130[3];
	double R_230[2];
	double R_330[1];
	double R_040[3];
	double R_140[2];
	double R_240[1];
	double R_050[2];
	double R_150[1];
	double R_060[1];
	double R_001[6];
	double R_101[5];
	double R_201[4];
	double R_301[3];
	double R_401[2];
	double R_501[1];
	double R_011[5];
	double R_111[4];
	double R_211[3];
	double R_311[2];
	double R_411[1];
	double R_021[4];
	double R_121[3];
	double R_221[2];
	double R_321[1];
	double R_031[3];
	double R_131[2];
	double R_231[1];
	double R_041[2];
	double R_141[1];
	double R_051[1];
	double R_002[5];
	double R_102[4];
	double R_202[3];
	double R_302[2];
	double R_402[1];
	double R_012[4];
	double R_112[3];
	double R_212[2];
	double R_312[1];
	double R_022[3];
	double R_122[2];
	double R_222[1];
	double R_032[2];
	double R_132[1];
	double R_042[1];
	double R_003[4];
	double R_103[3];
	double R_203[2];
	double R_303[1];
	double R_013[3];
	double R_113[2];
	double R_213[1];
	double R_023[2];
	double R_123[1];
	double R_033[1];
	double R_004[3];
	double R_104[2];
	double R_204[1];
	double R_014[2];
	double R_114[1];
	double R_024[1];
	double R_005[2];
	double R_105[1];
	double R_015[1];
	double R_006[1];
	for(int i=0;i<6;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<6;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<6;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<5;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<5;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<5;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<5;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<4;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<4;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<4;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<4;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<4;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<4;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<4;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<4;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<4;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<4;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
	for(int i=0;i<3;i++){
		R_400[i]=TX*R_300[i+1]+3*R_200[i+1];
	}
	for(int i=0;i<3;i++){
		R_310[i]=TY*R_300[i+1];
	}
	for(int i=0;i<3;i++){
		R_220[i]=TX*R_120[i+1]+R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_130[i]=TX*R_030[i+1];
	}
	for(int i=0;i<3;i++){
		R_040[i]=TY*R_030[i+1]+3*R_020[i+1];
	}
	for(int i=0;i<3;i++){
		R_301[i]=TZ*R_300[i+1];
	}
	for(int i=0;i<3;i++){
		R_211[i]=TY*R_201[i+1];
	}
	for(int i=0;i<3;i++){
		R_121[i]=TX*R_021[i+1];
	}
	for(int i=0;i<3;i++){
		R_031[i]=TZ*R_030[i+1];
	}
	for(int i=0;i<3;i++){
		R_202[i]=TX*R_102[i+1]+R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_112[i]=TX*R_012[i+1];
	}
	for(int i=0;i<3;i++){
		R_022[i]=TY*R_012[i+1]+R_002[i+1];
	}
	for(int i=0;i<3;i++){
		R_103[i]=TX*R_003[i+1];
	}
	for(int i=0;i<3;i++){
		R_013[i]=TY*R_003[i+1];
	}
	for(int i=0;i<3;i++){
		R_004[i]=TZ*R_003[i+1]+3*R_002[i+1];
	}
	for(int i=0;i<2;i++){
		R_500[i]=TX*R_400[i+1]+4*R_300[i+1];
	}
	for(int i=0;i<2;i++){
		R_410[i]=TY*R_400[i+1];
	}
	for(int i=0;i<2;i++){
		R_320[i]=TX*R_220[i+1]+2*R_120[i+1];
	}
	for(int i=0;i<2;i++){
		R_230[i]=TY*R_220[i+1]+2*R_210[i+1];
	}
	for(int i=0;i<2;i++){
		R_140[i]=TX*R_040[i+1];
	}
	for(int i=0;i<2;i++){
		R_050[i]=TY*R_040[i+1]+4*R_030[i+1];
	}
	for(int i=0;i<2;i++){
		R_401[i]=TZ*R_400[i+1];
	}
	for(int i=0;i<2;i++){
		R_311[i]=TY*R_301[i+1];
	}
	for(int i=0;i<2;i++){
		R_221[i]=TZ*R_220[i+1];
	}
	for(int i=0;i<2;i++){
		R_131[i]=TX*R_031[i+1];
	}
	for(int i=0;i<2;i++){
		R_041[i]=TZ*R_040[i+1];
	}
	for(int i=0;i<2;i++){
		R_302[i]=TX*R_202[i+1]+2*R_102[i+1];
	}
	for(int i=0;i<2;i++){
		R_212[i]=TY*R_202[i+1];
	}
	for(int i=0;i<2;i++){
		R_122[i]=TX*R_022[i+1];
	}
	for(int i=0;i<2;i++){
		R_032[i]=TY*R_022[i+1]+2*R_012[i+1];
	}
	for(int i=0;i<2;i++){
		R_203[i]=TZ*R_202[i+1]+2*R_201[i+1];
	}
	for(int i=0;i<2;i++){
		R_113[i]=TX*R_013[i+1];
	}
	for(int i=0;i<2;i++){
		R_023[i]=TZ*R_022[i+1]+2*R_021[i+1];
	}
	for(int i=0;i<2;i++){
		R_104[i]=TX*R_004[i+1];
	}
	for(int i=0;i<2;i++){
		R_014[i]=TY*R_004[i+1];
	}
	for(int i=0;i<2;i++){
		R_005[i]=TZ*R_004[i+1]+4*R_003[i+1];
	}
	for(int i=0;i<1;i++){
		R_600[i]=TX*R_500[i+1]+5*R_400[i+1];
	}
	for(int i=0;i<1;i++){
		R_510[i]=TY*R_500[i+1];
	}
	for(int i=0;i<1;i++){
		R_420[i]=TX*R_320[i+1]+3*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_330[i]=TX*R_230[i+1]+2*R_130[i+1];
	}
	for(int i=0;i<1;i++){
		R_240[i]=TY*R_230[i+1]+3*R_220[i+1];
	}
	for(int i=0;i<1;i++){
		R_150[i]=TX*R_050[i+1];
	}
	for(int i=0;i<1;i++){
		R_060[i]=TY*R_050[i+1]+5*R_040[i+1];
	}
	for(int i=0;i<1;i++){
		R_501[i]=TZ*R_500[i+1];
	}
	for(int i=0;i<1;i++){
		R_411[i]=TY*R_401[i+1];
	}
	for(int i=0;i<1;i++){
		R_321[i]=TZ*R_320[i+1];
	}
	for(int i=0;i<1;i++){
		R_231[i]=TZ*R_230[i+1];
	}
	for(int i=0;i<1;i++){
		R_141[i]=TX*R_041[i+1];
	}
	for(int i=0;i<1;i++){
		R_051[i]=TZ*R_050[i+1];
	}
	for(int i=0;i<1;i++){
		R_402[i]=TX*R_302[i+1]+3*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_312[i]=TY*R_302[i+1];
	}
	for(int i=0;i<1;i++){
		R_222[i]=TX*R_122[i+1]+R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_132[i]=TX*R_032[i+1];
	}
	for(int i=0;i<1;i++){
		R_042[i]=TY*R_032[i+1]+3*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_303[i]=TX*R_203[i+1]+2*R_103[i+1];
	}
	for(int i=0;i<1;i++){
		R_213[i]=TY*R_203[i+1];
	}
	for(int i=0;i<1;i++){
		R_123[i]=TX*R_023[i+1];
	}
	for(int i=0;i<1;i++){
		R_033[i]=TY*R_023[i+1]+2*R_013[i+1];
	}
	for(int i=0;i<1;i++){
		R_204[i]=TZ*R_203[i+1]+3*R_202[i+1];
	}
	for(int i=0;i<1;i++){
		R_114[i]=TX*R_014[i+1];
	}
	for(int i=0;i<1;i++){
		R_024[i]=TZ*R_023[i+1]+3*R_022[i+1];
	}
	for(int i=0;i<1;i++){
		R_105[i]=TX*R_005[i+1];
	}
	for(int i=0;i<1;i++){
		R_015[i]=TY*R_005[i+1];
	}
	for(int i=0;i<1;i++){
		R_006[i]=TZ*R_005[i+1]+5*R_004[i+1];
	}
		double Qd_110[3];
		double Qd_020[3];
		double Qd_120[3];
		double Qd_220[3];
		for(int i=0;i<3;i++){
			Qd_110[i]=aQin1;
			}
		for(int i=0;i<3;i++){
			Qd_020[i]=Qd_110[i]+Qd_010[i]*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_120[i]=Qd_010[i]*Qd_110[i]+aQin1*Qd_010[i];
			}
		for(int i=0;i<3;i++){
			Qd_220[i]=aQin1*Qd_110[i];
			}
	double Q_020000000=Qd_020[0];
	double Q_120000000=Qd_120[0];
	double Q_220000000=Qd_220[0];
	double Q_010010000=Qd_010[0]*Qd_010[1];
	double Q_010110000=Qd_010[0]*Qd_110[1];
	double Q_110010000=Qd_110[0]*Qd_010[1];
	double Q_110110000=Qd_110[0]*Qd_110[1];
	double Q_000020000=Qd_020[1];
	double Q_000120000=Qd_120[1];
	double Q_000220000=Qd_220[1];
	double Q_010000010=Qd_010[0]*Qd_010[2];
	double Q_010000110=Qd_010[0]*Qd_110[2];
	double Q_110000010=Qd_110[0]*Qd_010[2];
	double Q_110000110=Qd_110[0]*Qd_110[2];
	double Q_000010010=Qd_010[1]*Qd_010[2];
	double Q_000010110=Qd_010[1]*Qd_110[2];
	double Q_000110010=Qd_110[1]*Qd_010[2];
	double Q_000110110=Qd_110[1]*Qd_110[2];
	double Q_000000020=Qd_020[2];
	double Q_000000120=Qd_120[2];
	double Q_000000220=Qd_220[2];
				double QR_020000000000=Q_020000000*R_000[0]+-1*Q_120000000*R_100[0]+Q_220000000*R_200[0];
				double QR_010010000000=Q_010010000*R_000[0]+-1*Q_010110000*R_010[0]+-1*Q_110010000*R_100[0]+Q_110110000*R_110[0];
				double QR_000020000000=Q_000020000*R_000[0]+-1*Q_000120000*R_010[0]+Q_000220000*R_020[0];
				double QR_010000010000=Q_010000010*R_000[0]+-1*Q_010000110*R_001[0]+-1*Q_110000010*R_100[0]+Q_110000110*R_101[0];
				double QR_000010010000=Q_000010010*R_000[0]+-1*Q_000010110*R_001[0]+-1*Q_000110010*R_010[0]+Q_000110110*R_011[0];
				double QR_000000020000=Q_000000020*R_000[0]+-1*Q_000000120*R_001[0]+Q_000000220*R_002[0];
				double QR_020000000001=Q_020000000*R_001[0]+-1*Q_120000000*R_101[0]+Q_220000000*R_201[0];
				double QR_010010000001=Q_010010000*R_001[0]+-1*Q_010110000*R_011[0]+-1*Q_110010000*R_101[0]+Q_110110000*R_111[0];
				double QR_000020000001=Q_000020000*R_001[0]+-1*Q_000120000*R_011[0]+Q_000220000*R_021[0];
				double QR_010000010001=Q_010000010*R_001[0]+-1*Q_010000110*R_002[0]+-1*Q_110000010*R_101[0]+Q_110000110*R_102[0];
				double QR_000010010001=Q_000010010*R_001[0]+-1*Q_000010110*R_002[0]+-1*Q_000110010*R_011[0]+Q_000110110*R_012[0];
				double QR_000000020001=Q_000000020*R_001[0]+-1*Q_000000120*R_002[0]+Q_000000220*R_003[0];
				double QR_020000000010=Q_020000000*R_010[0]+-1*Q_120000000*R_110[0]+Q_220000000*R_210[0];
				double QR_010010000010=Q_010010000*R_010[0]+-1*Q_010110000*R_020[0]+-1*Q_110010000*R_110[0]+Q_110110000*R_120[0];
				double QR_000020000010=Q_000020000*R_010[0]+-1*Q_000120000*R_020[0]+Q_000220000*R_030[0];
				double QR_010000010010=Q_010000010*R_010[0]+-1*Q_010000110*R_011[0]+-1*Q_110000010*R_110[0]+Q_110000110*R_111[0];
				double QR_000010010010=Q_000010010*R_010[0]+-1*Q_000010110*R_011[0]+-1*Q_000110010*R_020[0]+Q_000110110*R_021[0];
				double QR_000000020010=Q_000000020*R_010[0]+-1*Q_000000120*R_011[0]+Q_000000220*R_012[0];
				double QR_020000000100=Q_020000000*R_100[0]+-1*Q_120000000*R_200[0]+Q_220000000*R_300[0];
				double QR_010010000100=Q_010010000*R_100[0]+-1*Q_010110000*R_110[0]+-1*Q_110010000*R_200[0]+Q_110110000*R_210[0];
				double QR_000020000100=Q_000020000*R_100[0]+-1*Q_000120000*R_110[0]+Q_000220000*R_120[0];
				double QR_010000010100=Q_010000010*R_100[0]+-1*Q_010000110*R_101[0]+-1*Q_110000010*R_200[0]+Q_110000110*R_201[0];
				double QR_000010010100=Q_000010010*R_100[0]+-1*Q_000010110*R_101[0]+-1*Q_000110010*R_110[0]+Q_000110110*R_111[0];
				double QR_000000020100=Q_000000020*R_100[0]+-1*Q_000000120*R_101[0]+Q_000000220*R_102[0];
				double QR_020000000002=Q_020000000*R_002[0]+-1*Q_120000000*R_102[0]+Q_220000000*R_202[0];
				double QR_010010000002=Q_010010000*R_002[0]+-1*Q_010110000*R_012[0]+-1*Q_110010000*R_102[0]+Q_110110000*R_112[0];
				double QR_000020000002=Q_000020000*R_002[0]+-1*Q_000120000*R_012[0]+Q_000220000*R_022[0];
				double QR_010000010002=Q_010000010*R_002[0]+-1*Q_010000110*R_003[0]+-1*Q_110000010*R_102[0]+Q_110000110*R_103[0];
				double QR_000010010002=Q_000010010*R_002[0]+-1*Q_000010110*R_003[0]+-1*Q_000110010*R_012[0]+Q_000110110*R_013[0];
				double QR_000000020002=Q_000000020*R_002[0]+-1*Q_000000120*R_003[0]+Q_000000220*R_004[0];
				double QR_020000000011=Q_020000000*R_011[0]+-1*Q_120000000*R_111[0]+Q_220000000*R_211[0];
				double QR_010010000011=Q_010010000*R_011[0]+-1*Q_010110000*R_021[0]+-1*Q_110010000*R_111[0]+Q_110110000*R_121[0];
				double QR_000020000011=Q_000020000*R_011[0]+-1*Q_000120000*R_021[0]+Q_000220000*R_031[0];
				double QR_010000010011=Q_010000010*R_011[0]+-1*Q_010000110*R_012[0]+-1*Q_110000010*R_111[0]+Q_110000110*R_112[0];
				double QR_000010010011=Q_000010010*R_011[0]+-1*Q_000010110*R_012[0]+-1*Q_000110010*R_021[0]+Q_000110110*R_022[0];
				double QR_000000020011=Q_000000020*R_011[0]+-1*Q_000000120*R_012[0]+Q_000000220*R_013[0];
				double QR_020000000020=Q_020000000*R_020[0]+-1*Q_120000000*R_120[0]+Q_220000000*R_220[0];
				double QR_010010000020=Q_010010000*R_020[0]+-1*Q_010110000*R_030[0]+-1*Q_110010000*R_120[0]+Q_110110000*R_130[0];
				double QR_000020000020=Q_000020000*R_020[0]+-1*Q_000120000*R_030[0]+Q_000220000*R_040[0];
				double QR_010000010020=Q_010000010*R_020[0]+-1*Q_010000110*R_021[0]+-1*Q_110000010*R_120[0]+Q_110000110*R_121[0];
				double QR_000010010020=Q_000010010*R_020[0]+-1*Q_000010110*R_021[0]+-1*Q_000110010*R_030[0]+Q_000110110*R_031[0];
				double QR_000000020020=Q_000000020*R_020[0]+-1*Q_000000120*R_021[0]+Q_000000220*R_022[0];
				double QR_020000000101=Q_020000000*R_101[0]+-1*Q_120000000*R_201[0]+Q_220000000*R_301[0];
				double QR_010010000101=Q_010010000*R_101[0]+-1*Q_010110000*R_111[0]+-1*Q_110010000*R_201[0]+Q_110110000*R_211[0];
				double QR_000020000101=Q_000020000*R_101[0]+-1*Q_000120000*R_111[0]+Q_000220000*R_121[0];
				double QR_010000010101=Q_010000010*R_101[0]+-1*Q_010000110*R_102[0]+-1*Q_110000010*R_201[0]+Q_110000110*R_202[0];
				double QR_000010010101=Q_000010010*R_101[0]+-1*Q_000010110*R_102[0]+-1*Q_000110010*R_111[0]+Q_000110110*R_112[0];
				double QR_000000020101=Q_000000020*R_101[0]+-1*Q_000000120*R_102[0]+Q_000000220*R_103[0];
				double QR_020000000110=Q_020000000*R_110[0]+-1*Q_120000000*R_210[0]+Q_220000000*R_310[0];
				double QR_010010000110=Q_010010000*R_110[0]+-1*Q_010110000*R_120[0]+-1*Q_110010000*R_210[0]+Q_110110000*R_220[0];
				double QR_000020000110=Q_000020000*R_110[0]+-1*Q_000120000*R_120[0]+Q_000220000*R_130[0];
				double QR_010000010110=Q_010000010*R_110[0]+-1*Q_010000110*R_111[0]+-1*Q_110000010*R_210[0]+Q_110000110*R_211[0];
				double QR_000010010110=Q_000010010*R_110[0]+-1*Q_000010110*R_111[0]+-1*Q_000110010*R_120[0]+Q_000110110*R_121[0];
				double QR_000000020110=Q_000000020*R_110[0]+-1*Q_000000120*R_111[0]+Q_000000220*R_112[0];
				double QR_020000000200=Q_020000000*R_200[0]+-1*Q_120000000*R_300[0]+Q_220000000*R_400[0];
				double QR_010010000200=Q_010010000*R_200[0]+-1*Q_010110000*R_210[0]+-1*Q_110010000*R_300[0]+Q_110110000*R_310[0];
				double QR_000020000200=Q_000020000*R_200[0]+-1*Q_000120000*R_210[0]+Q_000220000*R_220[0];
				double QR_010000010200=Q_010000010*R_200[0]+-1*Q_010000110*R_201[0]+-1*Q_110000010*R_300[0]+Q_110000110*R_301[0];
				double QR_000010010200=Q_000010010*R_200[0]+-1*Q_000010110*R_201[0]+-1*Q_000110010*R_210[0]+Q_000110110*R_211[0];
				double QR_000000020200=Q_000000020*R_200[0]+-1*Q_000000120*R_201[0]+Q_000000220*R_202[0];
				double QR_020000000003=Q_020000000*R_003[0]+-1*Q_120000000*R_103[0]+Q_220000000*R_203[0];
				double QR_010010000003=Q_010010000*R_003[0]+-1*Q_010110000*R_013[0]+-1*Q_110010000*R_103[0]+Q_110110000*R_113[0];
				double QR_000020000003=Q_000020000*R_003[0]+-1*Q_000120000*R_013[0]+Q_000220000*R_023[0];
				double QR_010000010003=Q_010000010*R_003[0]+-1*Q_010000110*R_004[0]+-1*Q_110000010*R_103[0]+Q_110000110*R_104[0];
				double QR_000010010003=Q_000010010*R_003[0]+-1*Q_000010110*R_004[0]+-1*Q_000110010*R_013[0]+Q_000110110*R_014[0];
				double QR_000000020003=Q_000000020*R_003[0]+-1*Q_000000120*R_004[0]+Q_000000220*R_005[0];
				double QR_020000000012=Q_020000000*R_012[0]+-1*Q_120000000*R_112[0]+Q_220000000*R_212[0];
				double QR_010010000012=Q_010010000*R_012[0]+-1*Q_010110000*R_022[0]+-1*Q_110010000*R_112[0]+Q_110110000*R_122[0];
				double QR_000020000012=Q_000020000*R_012[0]+-1*Q_000120000*R_022[0]+Q_000220000*R_032[0];
				double QR_010000010012=Q_010000010*R_012[0]+-1*Q_010000110*R_013[0]+-1*Q_110000010*R_112[0]+Q_110000110*R_113[0];
				double QR_000010010012=Q_000010010*R_012[0]+-1*Q_000010110*R_013[0]+-1*Q_000110010*R_022[0]+Q_000110110*R_023[0];
				double QR_000000020012=Q_000000020*R_012[0]+-1*Q_000000120*R_013[0]+Q_000000220*R_014[0];
				double QR_020000000021=Q_020000000*R_021[0]+-1*Q_120000000*R_121[0]+Q_220000000*R_221[0];
				double QR_010010000021=Q_010010000*R_021[0]+-1*Q_010110000*R_031[0]+-1*Q_110010000*R_121[0]+Q_110110000*R_131[0];
				double QR_000020000021=Q_000020000*R_021[0]+-1*Q_000120000*R_031[0]+Q_000220000*R_041[0];
				double QR_010000010021=Q_010000010*R_021[0]+-1*Q_010000110*R_022[0]+-1*Q_110000010*R_121[0]+Q_110000110*R_122[0];
				double QR_000010010021=Q_000010010*R_021[0]+-1*Q_000010110*R_022[0]+-1*Q_000110010*R_031[0]+Q_000110110*R_032[0];
				double QR_000000020021=Q_000000020*R_021[0]+-1*Q_000000120*R_022[0]+Q_000000220*R_023[0];
				double QR_020000000030=Q_020000000*R_030[0]+-1*Q_120000000*R_130[0]+Q_220000000*R_230[0];
				double QR_010010000030=Q_010010000*R_030[0]+-1*Q_010110000*R_040[0]+-1*Q_110010000*R_130[0]+Q_110110000*R_140[0];
				double QR_000020000030=Q_000020000*R_030[0]+-1*Q_000120000*R_040[0]+Q_000220000*R_050[0];
				double QR_010000010030=Q_010000010*R_030[0]+-1*Q_010000110*R_031[0]+-1*Q_110000010*R_130[0]+Q_110000110*R_131[0];
				double QR_000010010030=Q_000010010*R_030[0]+-1*Q_000010110*R_031[0]+-1*Q_000110010*R_040[0]+Q_000110110*R_041[0];
				double QR_000000020030=Q_000000020*R_030[0]+-1*Q_000000120*R_031[0]+Q_000000220*R_032[0];
				double QR_020000000102=Q_020000000*R_102[0]+-1*Q_120000000*R_202[0]+Q_220000000*R_302[0];
				double QR_010010000102=Q_010010000*R_102[0]+-1*Q_010110000*R_112[0]+-1*Q_110010000*R_202[0]+Q_110110000*R_212[0];
				double QR_000020000102=Q_000020000*R_102[0]+-1*Q_000120000*R_112[0]+Q_000220000*R_122[0];
				double QR_010000010102=Q_010000010*R_102[0]+-1*Q_010000110*R_103[0]+-1*Q_110000010*R_202[0]+Q_110000110*R_203[0];
				double QR_000010010102=Q_000010010*R_102[0]+-1*Q_000010110*R_103[0]+-1*Q_000110010*R_112[0]+Q_000110110*R_113[0];
				double QR_000000020102=Q_000000020*R_102[0]+-1*Q_000000120*R_103[0]+Q_000000220*R_104[0];
				double QR_020000000111=Q_020000000*R_111[0]+-1*Q_120000000*R_211[0]+Q_220000000*R_311[0];
				double QR_010010000111=Q_010010000*R_111[0]+-1*Q_010110000*R_121[0]+-1*Q_110010000*R_211[0]+Q_110110000*R_221[0];
				double QR_000020000111=Q_000020000*R_111[0]+-1*Q_000120000*R_121[0]+Q_000220000*R_131[0];
				double QR_010000010111=Q_010000010*R_111[0]+-1*Q_010000110*R_112[0]+-1*Q_110000010*R_211[0]+Q_110000110*R_212[0];
				double QR_000010010111=Q_000010010*R_111[0]+-1*Q_000010110*R_112[0]+-1*Q_000110010*R_121[0]+Q_000110110*R_122[0];
				double QR_000000020111=Q_000000020*R_111[0]+-1*Q_000000120*R_112[0]+Q_000000220*R_113[0];
				double QR_020000000120=Q_020000000*R_120[0]+-1*Q_120000000*R_220[0]+Q_220000000*R_320[0];
				double QR_010010000120=Q_010010000*R_120[0]+-1*Q_010110000*R_130[0]+-1*Q_110010000*R_220[0]+Q_110110000*R_230[0];
				double QR_000020000120=Q_000020000*R_120[0]+-1*Q_000120000*R_130[0]+Q_000220000*R_140[0];
				double QR_010000010120=Q_010000010*R_120[0]+-1*Q_010000110*R_121[0]+-1*Q_110000010*R_220[0]+Q_110000110*R_221[0];
				double QR_000010010120=Q_000010010*R_120[0]+-1*Q_000010110*R_121[0]+-1*Q_000110010*R_130[0]+Q_000110110*R_131[0];
				double QR_000000020120=Q_000000020*R_120[0]+-1*Q_000000120*R_121[0]+Q_000000220*R_122[0];
				double QR_020000000201=Q_020000000*R_201[0]+-1*Q_120000000*R_301[0]+Q_220000000*R_401[0];
				double QR_010010000201=Q_010010000*R_201[0]+-1*Q_010110000*R_211[0]+-1*Q_110010000*R_301[0]+Q_110110000*R_311[0];
				double QR_000020000201=Q_000020000*R_201[0]+-1*Q_000120000*R_211[0]+Q_000220000*R_221[0];
				double QR_010000010201=Q_010000010*R_201[0]+-1*Q_010000110*R_202[0]+-1*Q_110000010*R_301[0]+Q_110000110*R_302[0];
				double QR_000010010201=Q_000010010*R_201[0]+-1*Q_000010110*R_202[0]+-1*Q_000110010*R_211[0]+Q_000110110*R_212[0];
				double QR_000000020201=Q_000000020*R_201[0]+-1*Q_000000120*R_202[0]+Q_000000220*R_203[0];
				double QR_020000000210=Q_020000000*R_210[0]+-1*Q_120000000*R_310[0]+Q_220000000*R_410[0];
				double QR_010010000210=Q_010010000*R_210[0]+-1*Q_010110000*R_220[0]+-1*Q_110010000*R_310[0]+Q_110110000*R_320[0];
				double QR_000020000210=Q_000020000*R_210[0]+-1*Q_000120000*R_220[0]+Q_000220000*R_230[0];
				double QR_010000010210=Q_010000010*R_210[0]+-1*Q_010000110*R_211[0]+-1*Q_110000010*R_310[0]+Q_110000110*R_311[0];
				double QR_000010010210=Q_000010010*R_210[0]+-1*Q_000010110*R_211[0]+-1*Q_000110010*R_220[0]+Q_000110110*R_221[0];
				double QR_000000020210=Q_000000020*R_210[0]+-1*Q_000000120*R_211[0]+Q_000000220*R_212[0];
				double QR_020000000300=Q_020000000*R_300[0]+-1*Q_120000000*R_400[0]+Q_220000000*R_500[0];
				double QR_010010000300=Q_010010000*R_300[0]+-1*Q_010110000*R_310[0]+-1*Q_110010000*R_400[0]+Q_110110000*R_410[0];
				double QR_000020000300=Q_000020000*R_300[0]+-1*Q_000120000*R_310[0]+Q_000220000*R_320[0];
				double QR_010000010300=Q_010000010*R_300[0]+-1*Q_010000110*R_301[0]+-1*Q_110000010*R_400[0]+Q_110000110*R_401[0];
				double QR_000010010300=Q_000010010*R_300[0]+-1*Q_000010110*R_301[0]+-1*Q_000110010*R_310[0]+Q_000110110*R_311[0];
				double QR_000000020300=Q_000000020*R_300[0]+-1*Q_000000120*R_301[0]+Q_000000220*R_302[0];
				double QR_020000000004=Q_020000000*R_004[0]+-1*Q_120000000*R_104[0]+Q_220000000*R_204[0];
				double QR_010010000004=Q_010010000*R_004[0]+-1*Q_010110000*R_014[0]+-1*Q_110010000*R_104[0]+Q_110110000*R_114[0];
				double QR_000020000004=Q_000020000*R_004[0]+-1*Q_000120000*R_014[0]+Q_000220000*R_024[0];
				double QR_010000010004=Q_010000010*R_004[0]+-1*Q_010000110*R_005[0]+-1*Q_110000010*R_104[0]+Q_110000110*R_105[0];
				double QR_000010010004=Q_000010010*R_004[0]+-1*Q_000010110*R_005[0]+-1*Q_000110010*R_014[0]+Q_000110110*R_015[0];
				double QR_000000020004=Q_000000020*R_004[0]+-1*Q_000000120*R_005[0]+Q_000000220*R_006[0];
				double QR_020000000013=Q_020000000*R_013[0]+-1*Q_120000000*R_113[0]+Q_220000000*R_213[0];
				double QR_010010000013=Q_010010000*R_013[0]+-1*Q_010110000*R_023[0]+-1*Q_110010000*R_113[0]+Q_110110000*R_123[0];
				double QR_000020000013=Q_000020000*R_013[0]+-1*Q_000120000*R_023[0]+Q_000220000*R_033[0];
				double QR_010000010013=Q_010000010*R_013[0]+-1*Q_010000110*R_014[0]+-1*Q_110000010*R_113[0]+Q_110000110*R_114[0];
				double QR_000010010013=Q_000010010*R_013[0]+-1*Q_000010110*R_014[0]+-1*Q_000110010*R_023[0]+Q_000110110*R_024[0];
				double QR_000000020013=Q_000000020*R_013[0]+-1*Q_000000120*R_014[0]+Q_000000220*R_015[0];
				double QR_020000000022=Q_020000000*R_022[0]+-1*Q_120000000*R_122[0]+Q_220000000*R_222[0];
				double QR_010010000022=Q_010010000*R_022[0]+-1*Q_010110000*R_032[0]+-1*Q_110010000*R_122[0]+Q_110110000*R_132[0];
				double QR_000020000022=Q_000020000*R_022[0]+-1*Q_000120000*R_032[0]+Q_000220000*R_042[0];
				double QR_010000010022=Q_010000010*R_022[0]+-1*Q_010000110*R_023[0]+-1*Q_110000010*R_122[0]+Q_110000110*R_123[0];
				double QR_000010010022=Q_000010010*R_022[0]+-1*Q_000010110*R_023[0]+-1*Q_000110010*R_032[0]+Q_000110110*R_033[0];
				double QR_000000020022=Q_000000020*R_022[0]+-1*Q_000000120*R_023[0]+Q_000000220*R_024[0];
				double QR_020000000031=Q_020000000*R_031[0]+-1*Q_120000000*R_131[0]+Q_220000000*R_231[0];
				double QR_010010000031=Q_010010000*R_031[0]+-1*Q_010110000*R_041[0]+-1*Q_110010000*R_131[0]+Q_110110000*R_141[0];
				double QR_000020000031=Q_000020000*R_031[0]+-1*Q_000120000*R_041[0]+Q_000220000*R_051[0];
				double QR_010000010031=Q_010000010*R_031[0]+-1*Q_010000110*R_032[0]+-1*Q_110000010*R_131[0]+Q_110000110*R_132[0];
				double QR_000010010031=Q_000010010*R_031[0]+-1*Q_000010110*R_032[0]+-1*Q_000110010*R_041[0]+Q_000110110*R_042[0];
				double QR_000000020031=Q_000000020*R_031[0]+-1*Q_000000120*R_032[0]+Q_000000220*R_033[0];
				double QR_020000000040=Q_020000000*R_040[0]+-1*Q_120000000*R_140[0]+Q_220000000*R_240[0];
				double QR_010010000040=Q_010010000*R_040[0]+-1*Q_010110000*R_050[0]+-1*Q_110010000*R_140[0]+Q_110110000*R_150[0];
				double QR_000020000040=Q_000020000*R_040[0]+-1*Q_000120000*R_050[0]+Q_000220000*R_060[0];
				double QR_010000010040=Q_010000010*R_040[0]+-1*Q_010000110*R_041[0]+-1*Q_110000010*R_140[0]+Q_110000110*R_141[0];
				double QR_000010010040=Q_000010010*R_040[0]+-1*Q_000010110*R_041[0]+-1*Q_000110010*R_050[0]+Q_000110110*R_051[0];
				double QR_000000020040=Q_000000020*R_040[0]+-1*Q_000000120*R_041[0]+Q_000000220*R_042[0];
				double QR_020000000103=Q_020000000*R_103[0]+-1*Q_120000000*R_203[0]+Q_220000000*R_303[0];
				double QR_010010000103=Q_010010000*R_103[0]+-1*Q_010110000*R_113[0]+-1*Q_110010000*R_203[0]+Q_110110000*R_213[0];
				double QR_000020000103=Q_000020000*R_103[0]+-1*Q_000120000*R_113[0]+Q_000220000*R_123[0];
				double QR_010000010103=Q_010000010*R_103[0]+-1*Q_010000110*R_104[0]+-1*Q_110000010*R_203[0]+Q_110000110*R_204[0];
				double QR_000010010103=Q_000010010*R_103[0]+-1*Q_000010110*R_104[0]+-1*Q_000110010*R_113[0]+Q_000110110*R_114[0];
				double QR_000000020103=Q_000000020*R_103[0]+-1*Q_000000120*R_104[0]+Q_000000220*R_105[0];
				double QR_020000000112=Q_020000000*R_112[0]+-1*Q_120000000*R_212[0]+Q_220000000*R_312[0];
				double QR_010010000112=Q_010010000*R_112[0]+-1*Q_010110000*R_122[0]+-1*Q_110010000*R_212[0]+Q_110110000*R_222[0];
				double QR_000020000112=Q_000020000*R_112[0]+-1*Q_000120000*R_122[0]+Q_000220000*R_132[0];
				double QR_010000010112=Q_010000010*R_112[0]+-1*Q_010000110*R_113[0]+-1*Q_110000010*R_212[0]+Q_110000110*R_213[0];
				double QR_000010010112=Q_000010010*R_112[0]+-1*Q_000010110*R_113[0]+-1*Q_000110010*R_122[0]+Q_000110110*R_123[0];
				double QR_000000020112=Q_000000020*R_112[0]+-1*Q_000000120*R_113[0]+Q_000000220*R_114[0];
				double QR_020000000121=Q_020000000*R_121[0]+-1*Q_120000000*R_221[0]+Q_220000000*R_321[0];
				double QR_010010000121=Q_010010000*R_121[0]+-1*Q_010110000*R_131[0]+-1*Q_110010000*R_221[0]+Q_110110000*R_231[0];
				double QR_000020000121=Q_000020000*R_121[0]+-1*Q_000120000*R_131[0]+Q_000220000*R_141[0];
				double QR_010000010121=Q_010000010*R_121[0]+-1*Q_010000110*R_122[0]+-1*Q_110000010*R_221[0]+Q_110000110*R_222[0];
				double QR_000010010121=Q_000010010*R_121[0]+-1*Q_000010110*R_122[0]+-1*Q_000110010*R_131[0]+Q_000110110*R_132[0];
				double QR_000000020121=Q_000000020*R_121[0]+-1*Q_000000120*R_122[0]+Q_000000220*R_123[0];
				double QR_020000000130=Q_020000000*R_130[0]+-1*Q_120000000*R_230[0]+Q_220000000*R_330[0];
				double QR_010010000130=Q_010010000*R_130[0]+-1*Q_010110000*R_140[0]+-1*Q_110010000*R_230[0]+Q_110110000*R_240[0];
				double QR_000020000130=Q_000020000*R_130[0]+-1*Q_000120000*R_140[0]+Q_000220000*R_150[0];
				double QR_010000010130=Q_010000010*R_130[0]+-1*Q_010000110*R_131[0]+-1*Q_110000010*R_230[0]+Q_110000110*R_231[0];
				double QR_000010010130=Q_000010010*R_130[0]+-1*Q_000010110*R_131[0]+-1*Q_000110010*R_140[0]+Q_000110110*R_141[0];
				double QR_000000020130=Q_000000020*R_130[0]+-1*Q_000000120*R_131[0]+Q_000000220*R_132[0];
				double QR_020000000202=Q_020000000*R_202[0]+-1*Q_120000000*R_302[0]+Q_220000000*R_402[0];
				double QR_010010000202=Q_010010000*R_202[0]+-1*Q_010110000*R_212[0]+-1*Q_110010000*R_302[0]+Q_110110000*R_312[0];
				double QR_000020000202=Q_000020000*R_202[0]+-1*Q_000120000*R_212[0]+Q_000220000*R_222[0];
				double QR_010000010202=Q_010000010*R_202[0]+-1*Q_010000110*R_203[0]+-1*Q_110000010*R_302[0]+Q_110000110*R_303[0];
				double QR_000010010202=Q_000010010*R_202[0]+-1*Q_000010110*R_203[0]+-1*Q_000110010*R_212[0]+Q_000110110*R_213[0];
				double QR_000000020202=Q_000000020*R_202[0]+-1*Q_000000120*R_203[0]+Q_000000220*R_204[0];
				double QR_020000000211=Q_020000000*R_211[0]+-1*Q_120000000*R_311[0]+Q_220000000*R_411[0];
				double QR_010010000211=Q_010010000*R_211[0]+-1*Q_010110000*R_221[0]+-1*Q_110010000*R_311[0]+Q_110110000*R_321[0];
				double QR_000020000211=Q_000020000*R_211[0]+-1*Q_000120000*R_221[0]+Q_000220000*R_231[0];
				double QR_010000010211=Q_010000010*R_211[0]+-1*Q_010000110*R_212[0]+-1*Q_110000010*R_311[0]+Q_110000110*R_312[0];
				double QR_000010010211=Q_000010010*R_211[0]+-1*Q_000010110*R_212[0]+-1*Q_000110010*R_221[0]+Q_000110110*R_222[0];
				double QR_000000020211=Q_000000020*R_211[0]+-1*Q_000000120*R_212[0]+Q_000000220*R_213[0];
				double QR_020000000220=Q_020000000*R_220[0]+-1*Q_120000000*R_320[0]+Q_220000000*R_420[0];
				double QR_010010000220=Q_010010000*R_220[0]+-1*Q_010110000*R_230[0]+-1*Q_110010000*R_320[0]+Q_110110000*R_330[0];
				double QR_000020000220=Q_000020000*R_220[0]+-1*Q_000120000*R_230[0]+Q_000220000*R_240[0];
				double QR_010000010220=Q_010000010*R_220[0]+-1*Q_010000110*R_221[0]+-1*Q_110000010*R_320[0]+Q_110000110*R_321[0];
				double QR_000010010220=Q_000010010*R_220[0]+-1*Q_000010110*R_221[0]+-1*Q_000110010*R_230[0]+Q_000110110*R_231[0];
				double QR_000000020220=Q_000000020*R_220[0]+-1*Q_000000120*R_221[0]+Q_000000220*R_222[0];
				double QR_020000000301=Q_020000000*R_301[0]+-1*Q_120000000*R_401[0]+Q_220000000*R_501[0];
				double QR_010010000301=Q_010010000*R_301[0]+-1*Q_010110000*R_311[0]+-1*Q_110010000*R_401[0]+Q_110110000*R_411[0];
				double QR_000020000301=Q_000020000*R_301[0]+-1*Q_000120000*R_311[0]+Q_000220000*R_321[0];
				double QR_010000010301=Q_010000010*R_301[0]+-1*Q_010000110*R_302[0]+-1*Q_110000010*R_401[0]+Q_110000110*R_402[0];
				double QR_000010010301=Q_000010010*R_301[0]+-1*Q_000010110*R_302[0]+-1*Q_000110010*R_311[0]+Q_000110110*R_312[0];
				double QR_000000020301=Q_000000020*R_301[0]+-1*Q_000000120*R_302[0]+Q_000000220*R_303[0];
				double QR_020000000310=Q_020000000*R_310[0]+-1*Q_120000000*R_410[0]+Q_220000000*R_510[0];
				double QR_010010000310=Q_010010000*R_310[0]+-1*Q_010110000*R_320[0]+-1*Q_110010000*R_410[0]+Q_110110000*R_420[0];
				double QR_000020000310=Q_000020000*R_310[0]+-1*Q_000120000*R_320[0]+Q_000220000*R_330[0];
				double QR_010000010310=Q_010000010*R_310[0]+-1*Q_010000110*R_311[0]+-1*Q_110000010*R_410[0]+Q_110000110*R_411[0];
				double QR_000010010310=Q_000010010*R_310[0]+-1*Q_000010110*R_311[0]+-1*Q_000110010*R_320[0]+Q_000110110*R_321[0];
				double QR_000000020310=Q_000000020*R_310[0]+-1*Q_000000120*R_311[0]+Q_000000220*R_312[0];
				double QR_020000000400=Q_020000000*R_400[0]+-1*Q_120000000*R_500[0]+Q_220000000*R_600[0];
				double QR_010010000400=Q_010010000*R_400[0]+-1*Q_010110000*R_410[0]+-1*Q_110010000*R_500[0]+Q_110110000*R_510[0];
				double QR_000020000400=Q_000020000*R_400[0]+-1*Q_000120000*R_410[0]+Q_000220000*R_420[0];
				double QR_010000010400=Q_010000010*R_400[0]+-1*Q_010000110*R_401[0]+-1*Q_110000010*R_500[0]+Q_110000110*R_501[0];
				double QR_000010010400=Q_000010010*R_400[0]+-1*Q_000010110*R_401[0]+-1*Q_000110010*R_410[0]+Q_000110110*R_411[0];
				double QR_000000020400=Q_000000020*R_400[0]+-1*Q_000000120*R_401[0]+Q_000000220*R_402[0];
		double Pd_101[3];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_202[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_312[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_220[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_321[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		double Pd_322[3];
		double Pd_422[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_002[i]=Pd_101[i]+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=Pd_001[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_202[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_110[i]=aPin1;
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=Pd_101[i]+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=Pd_010[i]*Pd_101[i]+aPin1*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_211[i]=aPin1*Pd_101[i];
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=2*Pd_211[i]+Pd_001[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=Pd_001[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_312[i]=aPin1*Pd_211[i];
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=Pd_110[i]+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=Pd_010[i]*Pd_110[i]+aPin1*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_220[i]=aPin1*Pd_110[i];
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=2*Pd_211[i]+Pd_010[i]*Pd_111[i]+aPin1*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=Pd_010[i]*Pd_211[i]+aPin1*Pd_111[i];
			}
		for(int i=0;i<3;i++){
			Pd_321[i]=aPin1*Pd_211[i];
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=2*Pd_212[i]+Pd_010[i]*Pd_112[i]+aPin1*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=3*Pd_312[i]+Pd_010[i]*Pd_212[i]+aPin1*Pd_112[i];
			}
		for(int i=0;i<3;i++){
			Pd_322[i]=Pd_010[i]*Pd_312[i]+aPin1*Pd_212[i];
			}
		for(int i=0;i<3;i++){
			Pd_422[i]=aPin1*Pd_312[i];
			}
	double P_022000000=Pd_022[0];
	double P_122000000=Pd_122[0];
	double P_222000000=Pd_222[0];
	double P_322000000=Pd_322[0];
	double P_422000000=Pd_422[0];
	double P_021001000=Pd_021[0]*Pd_001[1];
	double P_021101000=Pd_021[0]*Pd_101[1];
	double P_121001000=Pd_121[0]*Pd_001[1];
	double P_121101000=Pd_121[0]*Pd_101[1];
	double P_221001000=Pd_221[0]*Pd_001[1];
	double P_221101000=Pd_221[0]*Pd_101[1];
	double P_321001000=Pd_321[0]*Pd_001[1];
	double P_321101000=Pd_321[0]*Pd_101[1];
	double P_020002000=Pd_020[0]*Pd_002[1];
	double P_020102000=Pd_020[0]*Pd_102[1];
	double P_020202000=Pd_020[0]*Pd_202[1];
	double P_120002000=Pd_120[0]*Pd_002[1];
	double P_120102000=Pd_120[0]*Pd_102[1];
	double P_120202000=Pd_120[0]*Pd_202[1];
	double P_220002000=Pd_220[0]*Pd_002[1];
	double P_220102000=Pd_220[0]*Pd_102[1];
	double P_220202000=Pd_220[0]*Pd_202[1];
	double P_021000001=Pd_021[0]*Pd_001[2];
	double P_021000101=Pd_021[0]*Pd_101[2];
	double P_121000001=Pd_121[0]*Pd_001[2];
	double P_121000101=Pd_121[0]*Pd_101[2];
	double P_221000001=Pd_221[0]*Pd_001[2];
	double P_221000101=Pd_221[0]*Pd_101[2];
	double P_321000001=Pd_321[0]*Pd_001[2];
	double P_321000101=Pd_321[0]*Pd_101[2];
	double P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
	double P_020001101=Pd_020[0]*Pd_001[1]*Pd_101[2];
	double P_020101001=Pd_020[0]*Pd_101[1]*Pd_001[2];
	double P_020101101=Pd_020[0]*Pd_101[1]*Pd_101[2];
	double P_120001001=Pd_120[0]*Pd_001[1]*Pd_001[2];
	double P_120001101=Pd_120[0]*Pd_001[1]*Pd_101[2];
	double P_120101001=Pd_120[0]*Pd_101[1]*Pd_001[2];
	double P_120101101=Pd_120[0]*Pd_101[1]*Pd_101[2];
	double P_220001001=Pd_220[0]*Pd_001[1]*Pd_001[2];
	double P_220001101=Pd_220[0]*Pd_001[1]*Pd_101[2];
	double P_220101001=Pd_220[0]*Pd_101[1]*Pd_001[2];
	double P_220101101=Pd_220[0]*Pd_101[1]*Pd_101[2];
	double P_020000002=Pd_020[0]*Pd_002[2];
	double P_020000102=Pd_020[0]*Pd_102[2];
	double P_020000202=Pd_020[0]*Pd_202[2];
	double P_120000002=Pd_120[0]*Pd_002[2];
	double P_120000102=Pd_120[0]*Pd_102[2];
	double P_120000202=Pd_120[0]*Pd_202[2];
	double P_220000002=Pd_220[0]*Pd_002[2];
	double P_220000102=Pd_220[0]*Pd_102[2];
	double P_220000202=Pd_220[0]*Pd_202[2];
	double P_012010000=Pd_012[0]*Pd_010[1];
	double P_012110000=Pd_012[0]*Pd_110[1];
	double P_112010000=Pd_112[0]*Pd_010[1];
	double P_112110000=Pd_112[0]*Pd_110[1];
	double P_212010000=Pd_212[0]*Pd_010[1];
	double P_212110000=Pd_212[0]*Pd_110[1];
	double P_312010000=Pd_312[0]*Pd_010[1];
	double P_312110000=Pd_312[0]*Pd_110[1];
	double P_011011000=Pd_011[0]*Pd_011[1];
	double P_011111000=Pd_011[0]*Pd_111[1];
	double P_011211000=Pd_011[0]*Pd_211[1];
	double P_111011000=Pd_111[0]*Pd_011[1];
	double P_111111000=Pd_111[0]*Pd_111[1];
	double P_111211000=Pd_111[0]*Pd_211[1];
	double P_211011000=Pd_211[0]*Pd_011[1];
	double P_211111000=Pd_211[0]*Pd_111[1];
	double P_211211000=Pd_211[0]*Pd_211[1];
	double P_010012000=Pd_010[0]*Pd_012[1];
	double P_010112000=Pd_010[0]*Pd_112[1];
	double P_010212000=Pd_010[0]*Pd_212[1];
	double P_010312000=Pd_010[0]*Pd_312[1];
	double P_110012000=Pd_110[0]*Pd_012[1];
	double P_110112000=Pd_110[0]*Pd_112[1];
	double P_110212000=Pd_110[0]*Pd_212[1];
	double P_110312000=Pd_110[0]*Pd_312[1];
	double P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
	double P_011010101=Pd_011[0]*Pd_010[1]*Pd_101[2];
	double P_011110001=Pd_011[0]*Pd_110[1]*Pd_001[2];
	double P_011110101=Pd_011[0]*Pd_110[1]*Pd_101[2];
	double P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
	double P_111010101=Pd_111[0]*Pd_010[1]*Pd_101[2];
	double P_111110001=Pd_111[0]*Pd_110[1]*Pd_001[2];
	double P_111110101=Pd_111[0]*Pd_110[1]*Pd_101[2];
	double P_211010001=Pd_211[0]*Pd_010[1]*Pd_001[2];
	double P_211010101=Pd_211[0]*Pd_010[1]*Pd_101[2];
	double P_211110001=Pd_211[0]*Pd_110[1]*Pd_001[2];
	double P_211110101=Pd_211[0]*Pd_110[1]*Pd_101[2];
	double P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
	double P_010011101=Pd_010[0]*Pd_011[1]*Pd_101[2];
	double P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
	double P_010111101=Pd_010[0]*Pd_111[1]*Pd_101[2];
	double P_010211001=Pd_010[0]*Pd_211[1]*Pd_001[2];
	double P_010211101=Pd_010[0]*Pd_211[1]*Pd_101[2];
	double P_110011001=Pd_110[0]*Pd_011[1]*Pd_001[2];
	double P_110011101=Pd_110[0]*Pd_011[1]*Pd_101[2];
	double P_110111001=Pd_110[0]*Pd_111[1]*Pd_001[2];
	double P_110111101=Pd_110[0]*Pd_111[1]*Pd_101[2];
	double P_110211001=Pd_110[0]*Pd_211[1]*Pd_001[2];
	double P_110211101=Pd_110[0]*Pd_211[1]*Pd_101[2];
	double P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
	double P_010010102=Pd_010[0]*Pd_010[1]*Pd_102[2];
	double P_010010202=Pd_010[0]*Pd_010[1]*Pd_202[2];
	double P_010110002=Pd_010[0]*Pd_110[1]*Pd_002[2];
	double P_010110102=Pd_010[0]*Pd_110[1]*Pd_102[2];
	double P_010110202=Pd_010[0]*Pd_110[1]*Pd_202[2];
	double P_110010002=Pd_110[0]*Pd_010[1]*Pd_002[2];
	double P_110010102=Pd_110[0]*Pd_010[1]*Pd_102[2];
	double P_110010202=Pd_110[0]*Pd_010[1]*Pd_202[2];
	double P_110110002=Pd_110[0]*Pd_110[1]*Pd_002[2];
	double P_110110102=Pd_110[0]*Pd_110[1]*Pd_102[2];
	double P_110110202=Pd_110[0]*Pd_110[1]*Pd_202[2];
	double P_002020000=Pd_002[0]*Pd_020[1];
	double P_002120000=Pd_002[0]*Pd_120[1];
	double P_002220000=Pd_002[0]*Pd_220[1];
	double P_102020000=Pd_102[0]*Pd_020[1];
	double P_102120000=Pd_102[0]*Pd_120[1];
	double P_102220000=Pd_102[0]*Pd_220[1];
	double P_202020000=Pd_202[0]*Pd_020[1];
	double P_202120000=Pd_202[0]*Pd_120[1];
	double P_202220000=Pd_202[0]*Pd_220[1];
	double P_001021000=Pd_001[0]*Pd_021[1];
	double P_001121000=Pd_001[0]*Pd_121[1];
	double P_001221000=Pd_001[0]*Pd_221[1];
	double P_001321000=Pd_001[0]*Pd_321[1];
	double P_101021000=Pd_101[0]*Pd_021[1];
	double P_101121000=Pd_101[0]*Pd_121[1];
	double P_101221000=Pd_101[0]*Pd_221[1];
	double P_101321000=Pd_101[0]*Pd_321[1];
	double P_000022000=Pd_022[1];
	double P_000122000=Pd_122[1];
	double P_000222000=Pd_222[1];
	double P_000322000=Pd_322[1];
	double P_000422000=Pd_422[1];
	double P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
	double P_001020101=Pd_001[0]*Pd_020[1]*Pd_101[2];
	double P_001120001=Pd_001[0]*Pd_120[1]*Pd_001[2];
	double P_001120101=Pd_001[0]*Pd_120[1]*Pd_101[2];
	double P_001220001=Pd_001[0]*Pd_220[1]*Pd_001[2];
	double P_001220101=Pd_001[0]*Pd_220[1]*Pd_101[2];
	double P_101020001=Pd_101[0]*Pd_020[1]*Pd_001[2];
	double P_101020101=Pd_101[0]*Pd_020[1]*Pd_101[2];
	double P_101120001=Pd_101[0]*Pd_120[1]*Pd_001[2];
	double P_101120101=Pd_101[0]*Pd_120[1]*Pd_101[2];
	double P_101220001=Pd_101[0]*Pd_220[1]*Pd_001[2];
	double P_101220101=Pd_101[0]*Pd_220[1]*Pd_101[2];
	double P_000021001=Pd_021[1]*Pd_001[2];
	double P_000021101=Pd_021[1]*Pd_101[2];
	double P_000121001=Pd_121[1]*Pd_001[2];
	double P_000121101=Pd_121[1]*Pd_101[2];
	double P_000221001=Pd_221[1]*Pd_001[2];
	double P_000221101=Pd_221[1]*Pd_101[2];
	double P_000321001=Pd_321[1]*Pd_001[2];
	double P_000321101=Pd_321[1]*Pd_101[2];
	double P_000020002=Pd_020[1]*Pd_002[2];
	double P_000020102=Pd_020[1]*Pd_102[2];
	double P_000020202=Pd_020[1]*Pd_202[2];
	double P_000120002=Pd_120[1]*Pd_002[2];
	double P_000120102=Pd_120[1]*Pd_102[2];
	double P_000120202=Pd_120[1]*Pd_202[2];
	double P_000220002=Pd_220[1]*Pd_002[2];
	double P_000220102=Pd_220[1]*Pd_102[2];
	double P_000220202=Pd_220[1]*Pd_202[2];
	double P_012000010=Pd_012[0]*Pd_010[2];
	double P_012000110=Pd_012[0]*Pd_110[2];
	double P_112000010=Pd_112[0]*Pd_010[2];
	double P_112000110=Pd_112[0]*Pd_110[2];
	double P_212000010=Pd_212[0]*Pd_010[2];
	double P_212000110=Pd_212[0]*Pd_110[2];
	double P_312000010=Pd_312[0]*Pd_010[2];
	double P_312000110=Pd_312[0]*Pd_110[2];
	double P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
	double P_011001110=Pd_011[0]*Pd_001[1]*Pd_110[2];
	double P_011101010=Pd_011[0]*Pd_101[1]*Pd_010[2];
	double P_011101110=Pd_011[0]*Pd_101[1]*Pd_110[2];
	double P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
	double P_111001110=Pd_111[0]*Pd_001[1]*Pd_110[2];
	double P_111101010=Pd_111[0]*Pd_101[1]*Pd_010[2];
	double P_111101110=Pd_111[0]*Pd_101[1]*Pd_110[2];
	double P_211001010=Pd_211[0]*Pd_001[1]*Pd_010[2];
	double P_211001110=Pd_211[0]*Pd_001[1]*Pd_110[2];
	double P_211101010=Pd_211[0]*Pd_101[1]*Pd_010[2];
	double P_211101110=Pd_211[0]*Pd_101[1]*Pd_110[2];
	double P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
	double P_010002110=Pd_010[0]*Pd_002[1]*Pd_110[2];
	double P_010102010=Pd_010[0]*Pd_102[1]*Pd_010[2];
	double P_010102110=Pd_010[0]*Pd_102[1]*Pd_110[2];
	double P_010202010=Pd_010[0]*Pd_202[1]*Pd_010[2];
	double P_010202110=Pd_010[0]*Pd_202[1]*Pd_110[2];
	double P_110002010=Pd_110[0]*Pd_002[1]*Pd_010[2];
	double P_110002110=Pd_110[0]*Pd_002[1]*Pd_110[2];
	double P_110102010=Pd_110[0]*Pd_102[1]*Pd_010[2];
	double P_110102110=Pd_110[0]*Pd_102[1]*Pd_110[2];
	double P_110202010=Pd_110[0]*Pd_202[1]*Pd_010[2];
	double P_110202110=Pd_110[0]*Pd_202[1]*Pd_110[2];
	double P_011000011=Pd_011[0]*Pd_011[2];
	double P_011000111=Pd_011[0]*Pd_111[2];
	double P_011000211=Pd_011[0]*Pd_211[2];
	double P_111000011=Pd_111[0]*Pd_011[2];
	double P_111000111=Pd_111[0]*Pd_111[2];
	double P_111000211=Pd_111[0]*Pd_211[2];
	double P_211000011=Pd_211[0]*Pd_011[2];
	double P_211000111=Pd_211[0]*Pd_111[2];
	double P_211000211=Pd_211[0]*Pd_211[2];
	double P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
	double P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
	double P_010001211=Pd_010[0]*Pd_001[1]*Pd_211[2];
	double P_010101011=Pd_010[0]*Pd_101[1]*Pd_011[2];
	double P_010101111=Pd_010[0]*Pd_101[1]*Pd_111[2];
	double P_010101211=Pd_010[0]*Pd_101[1]*Pd_211[2];
	double P_110001011=Pd_110[0]*Pd_001[1]*Pd_011[2];
	double P_110001111=Pd_110[0]*Pd_001[1]*Pd_111[2];
	double P_110001211=Pd_110[0]*Pd_001[1]*Pd_211[2];
	double P_110101011=Pd_110[0]*Pd_101[1]*Pd_011[2];
	double P_110101111=Pd_110[0]*Pd_101[1]*Pd_111[2];
	double P_110101211=Pd_110[0]*Pd_101[1]*Pd_211[2];
	double P_010000012=Pd_010[0]*Pd_012[2];
	double P_010000112=Pd_010[0]*Pd_112[2];
	double P_010000212=Pd_010[0]*Pd_212[2];
	double P_010000312=Pd_010[0]*Pd_312[2];
	double P_110000012=Pd_110[0]*Pd_012[2];
	double P_110000112=Pd_110[0]*Pd_112[2];
	double P_110000212=Pd_110[0]*Pd_212[2];
	double P_110000312=Pd_110[0]*Pd_312[2];
	double P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
	double P_002010110=Pd_002[0]*Pd_010[1]*Pd_110[2];
	double P_002110010=Pd_002[0]*Pd_110[1]*Pd_010[2];
	double P_002110110=Pd_002[0]*Pd_110[1]*Pd_110[2];
	double P_102010010=Pd_102[0]*Pd_010[1]*Pd_010[2];
	double P_102010110=Pd_102[0]*Pd_010[1]*Pd_110[2];
	double P_102110010=Pd_102[0]*Pd_110[1]*Pd_010[2];
	double P_102110110=Pd_102[0]*Pd_110[1]*Pd_110[2];
	double P_202010010=Pd_202[0]*Pd_010[1]*Pd_010[2];
	double P_202010110=Pd_202[0]*Pd_010[1]*Pd_110[2];
	double P_202110010=Pd_202[0]*Pd_110[1]*Pd_010[2];
	double P_202110110=Pd_202[0]*Pd_110[1]*Pd_110[2];
	double P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
	double P_001011110=Pd_001[0]*Pd_011[1]*Pd_110[2];
	double P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
	double P_001111110=Pd_001[0]*Pd_111[1]*Pd_110[2];
	double P_001211010=Pd_001[0]*Pd_211[1]*Pd_010[2];
	double P_001211110=Pd_001[0]*Pd_211[1]*Pd_110[2];
	double P_101011010=Pd_101[0]*Pd_011[1]*Pd_010[2];
	double P_101011110=Pd_101[0]*Pd_011[1]*Pd_110[2];
	double P_101111010=Pd_101[0]*Pd_111[1]*Pd_010[2];
	double P_101111110=Pd_101[0]*Pd_111[1]*Pd_110[2];
	double P_101211010=Pd_101[0]*Pd_211[1]*Pd_010[2];
	double P_101211110=Pd_101[0]*Pd_211[1]*Pd_110[2];
	double P_000012010=Pd_012[1]*Pd_010[2];
	double P_000012110=Pd_012[1]*Pd_110[2];
	double P_000112010=Pd_112[1]*Pd_010[2];
	double P_000112110=Pd_112[1]*Pd_110[2];
	double P_000212010=Pd_212[1]*Pd_010[2];
	double P_000212110=Pd_212[1]*Pd_110[2];
	double P_000312010=Pd_312[1]*Pd_010[2];
	double P_000312110=Pd_312[1]*Pd_110[2];
	double P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
	double P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
	double P_001010211=Pd_001[0]*Pd_010[1]*Pd_211[2];
	double P_001110011=Pd_001[0]*Pd_110[1]*Pd_011[2];
	double P_001110111=Pd_001[0]*Pd_110[1]*Pd_111[2];
	double P_001110211=Pd_001[0]*Pd_110[1]*Pd_211[2];
	double P_101010011=Pd_101[0]*Pd_010[1]*Pd_011[2];
	double P_101010111=Pd_101[0]*Pd_010[1]*Pd_111[2];
	double P_101010211=Pd_101[0]*Pd_010[1]*Pd_211[2];
	double P_101110011=Pd_101[0]*Pd_110[1]*Pd_011[2];
	double P_101110111=Pd_101[0]*Pd_110[1]*Pd_111[2];
	double P_101110211=Pd_101[0]*Pd_110[1]*Pd_211[2];
	double P_000011011=Pd_011[1]*Pd_011[2];
	double P_000011111=Pd_011[1]*Pd_111[2];
	double P_000011211=Pd_011[1]*Pd_211[2];
	double P_000111011=Pd_111[1]*Pd_011[2];
	double P_000111111=Pd_111[1]*Pd_111[2];
	double P_000111211=Pd_111[1]*Pd_211[2];
	double P_000211011=Pd_211[1]*Pd_011[2];
	double P_000211111=Pd_211[1]*Pd_111[2];
	double P_000211211=Pd_211[1]*Pd_211[2];
	double P_000010012=Pd_010[1]*Pd_012[2];
	double P_000010112=Pd_010[1]*Pd_112[2];
	double P_000010212=Pd_010[1]*Pd_212[2];
	double P_000010312=Pd_010[1]*Pd_312[2];
	double P_000110012=Pd_110[1]*Pd_012[2];
	double P_000110112=Pd_110[1]*Pd_112[2];
	double P_000110212=Pd_110[1]*Pd_212[2];
	double P_000110312=Pd_110[1]*Pd_312[2];
	double P_002000020=Pd_002[0]*Pd_020[2];
	double P_002000120=Pd_002[0]*Pd_120[2];
	double P_002000220=Pd_002[0]*Pd_220[2];
	double P_102000020=Pd_102[0]*Pd_020[2];
	double P_102000120=Pd_102[0]*Pd_120[2];
	double P_102000220=Pd_102[0]*Pd_220[2];
	double P_202000020=Pd_202[0]*Pd_020[2];
	double P_202000120=Pd_202[0]*Pd_120[2];
	double P_202000220=Pd_202[0]*Pd_220[2];
	double P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
	double P_001001120=Pd_001[0]*Pd_001[1]*Pd_120[2];
	double P_001001220=Pd_001[0]*Pd_001[1]*Pd_220[2];
	double P_001101020=Pd_001[0]*Pd_101[1]*Pd_020[2];
	double P_001101120=Pd_001[0]*Pd_101[1]*Pd_120[2];
	double P_001101220=Pd_001[0]*Pd_101[1]*Pd_220[2];
	double P_101001020=Pd_101[0]*Pd_001[1]*Pd_020[2];
	double P_101001120=Pd_101[0]*Pd_001[1]*Pd_120[2];
	double P_101001220=Pd_101[0]*Pd_001[1]*Pd_220[2];
	double P_101101020=Pd_101[0]*Pd_101[1]*Pd_020[2];
	double P_101101120=Pd_101[0]*Pd_101[1]*Pd_120[2];
	double P_101101220=Pd_101[0]*Pd_101[1]*Pd_220[2];
	double P_000002020=Pd_002[1]*Pd_020[2];
	double P_000002120=Pd_002[1]*Pd_120[2];
	double P_000002220=Pd_002[1]*Pd_220[2];
	double P_000102020=Pd_102[1]*Pd_020[2];
	double P_000102120=Pd_102[1]*Pd_120[2];
	double P_000102220=Pd_102[1]*Pd_220[2];
	double P_000202020=Pd_202[1]*Pd_020[2];
	double P_000202120=Pd_202[1]*Pd_120[2];
	double P_000202220=Pd_202[1]*Pd_220[2];
	double P_001000021=Pd_001[0]*Pd_021[2];
	double P_001000121=Pd_001[0]*Pd_121[2];
	double P_001000221=Pd_001[0]*Pd_221[2];
	double P_001000321=Pd_001[0]*Pd_321[2];
	double P_101000021=Pd_101[0]*Pd_021[2];
	double P_101000121=Pd_101[0]*Pd_121[2];
	double P_101000221=Pd_101[0]*Pd_221[2];
	double P_101000321=Pd_101[0]*Pd_321[2];
	double P_000001021=Pd_001[1]*Pd_021[2];
	double P_000001121=Pd_001[1]*Pd_121[2];
	double P_000001221=Pd_001[1]*Pd_221[2];
	double P_000001321=Pd_001[1]*Pd_321[2];
	double P_000101021=Pd_101[1]*Pd_021[2];
	double P_000101121=Pd_101[1]*Pd_121[2];
	double P_000101221=Pd_101[1]*Pd_221[2];
	double P_000101321=Pd_101[1]*Pd_321[2];
	double P_000000022=Pd_022[2];
	double P_000000122=Pd_122[2];
	double P_000000222=Pd_222[2];
	double P_000000322=Pd_322[2];
	double P_000000422=Pd_422[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_022000000*QR_020000000000+P_122000000*QR_020000000100+P_222000000*QR_020000000200+P_322000000*QR_020000000300+P_422000000*QR_020000000400);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_022000000*QR_010010000000+P_122000000*QR_010010000100+P_222000000*QR_010010000200+P_322000000*QR_010010000300+P_422000000*QR_010010000400);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_022000000*QR_000020000000+P_122000000*QR_000020000100+P_222000000*QR_000020000200+P_322000000*QR_000020000300+P_422000000*QR_000020000400);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_022000000*QR_010000010000+P_122000000*QR_010000010100+P_222000000*QR_010000010200+P_322000000*QR_010000010300+P_422000000*QR_010000010400);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_022000000*QR_000010010000+P_122000000*QR_000010010100+P_222000000*QR_000010010200+P_322000000*QR_000010010300+P_422000000*QR_000010010400);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_022000000*QR_000000020000+P_122000000*QR_000000020100+P_222000000*QR_000000020200+P_322000000*QR_000000020300+P_422000000*QR_000000020400);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_021001000*QR_020000000000+P_021101000*QR_020000000010+P_121001000*QR_020000000100+P_121101000*QR_020000000110+P_221001000*QR_020000000200+P_221101000*QR_020000000210+P_321001000*QR_020000000300+P_321101000*QR_020000000310);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_021001000*QR_010010000000+P_021101000*QR_010010000010+P_121001000*QR_010010000100+P_121101000*QR_010010000110+P_221001000*QR_010010000200+P_221101000*QR_010010000210+P_321001000*QR_010010000300+P_321101000*QR_010010000310);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_021001000*QR_000020000000+P_021101000*QR_000020000010+P_121001000*QR_000020000100+P_121101000*QR_000020000110+P_221001000*QR_000020000200+P_221101000*QR_000020000210+P_321001000*QR_000020000300+P_321101000*QR_000020000310);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_021001000*QR_010000010000+P_021101000*QR_010000010010+P_121001000*QR_010000010100+P_121101000*QR_010000010110+P_221001000*QR_010000010200+P_221101000*QR_010000010210+P_321001000*QR_010000010300+P_321101000*QR_010000010310);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_021001000*QR_000010010000+P_021101000*QR_000010010010+P_121001000*QR_000010010100+P_121101000*QR_000010010110+P_221001000*QR_000010010200+P_221101000*QR_000010010210+P_321001000*QR_000010010300+P_321101000*QR_000010010310);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_021001000*QR_000000020000+P_021101000*QR_000000020010+P_121001000*QR_000000020100+P_121101000*QR_000000020110+P_221001000*QR_000000020200+P_221101000*QR_000000020210+P_321001000*QR_000000020300+P_321101000*QR_000000020310);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_020002000*QR_020000000000+P_020102000*QR_020000000010+P_020202000*QR_020000000020+P_120002000*QR_020000000100+P_120102000*QR_020000000110+P_120202000*QR_020000000120+P_220002000*QR_020000000200+P_220102000*QR_020000000210+P_220202000*QR_020000000220);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_020002000*QR_010010000000+P_020102000*QR_010010000010+P_020202000*QR_010010000020+P_120002000*QR_010010000100+P_120102000*QR_010010000110+P_120202000*QR_010010000120+P_220002000*QR_010010000200+P_220102000*QR_010010000210+P_220202000*QR_010010000220);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020002000*QR_000020000000+P_020102000*QR_000020000010+P_020202000*QR_000020000020+P_120002000*QR_000020000100+P_120102000*QR_000020000110+P_120202000*QR_000020000120+P_220002000*QR_000020000200+P_220102000*QR_000020000210+P_220202000*QR_000020000220);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_020002000*QR_010000010000+P_020102000*QR_010000010010+P_020202000*QR_010000010020+P_120002000*QR_010000010100+P_120102000*QR_010000010110+P_120202000*QR_010000010120+P_220002000*QR_010000010200+P_220102000*QR_010000010210+P_220202000*QR_010000010220);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020002000*QR_000010010000+P_020102000*QR_000010010010+P_020202000*QR_000010010020+P_120002000*QR_000010010100+P_120102000*QR_000010010110+P_120202000*QR_000010010120+P_220002000*QR_000010010200+P_220102000*QR_000010010210+P_220202000*QR_000010010220);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020002000*QR_000000020000+P_020102000*QR_000000020010+P_020202000*QR_000000020020+P_120002000*QR_000000020100+P_120102000*QR_000000020110+P_120202000*QR_000000020120+P_220002000*QR_000000020200+P_220102000*QR_000000020210+P_220202000*QR_000000020220);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(P_021000001*QR_020000000000+P_021000101*QR_020000000001+P_121000001*QR_020000000100+P_121000101*QR_020000000101+P_221000001*QR_020000000200+P_221000101*QR_020000000201+P_321000001*QR_020000000300+P_321000101*QR_020000000301);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(P_021000001*QR_010010000000+P_021000101*QR_010010000001+P_121000001*QR_010010000100+P_121000101*QR_010010000101+P_221000001*QR_010010000200+P_221000101*QR_010010000201+P_321000001*QR_010010000300+P_321000101*QR_010010000301);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(P_021000001*QR_000020000000+P_021000101*QR_000020000001+P_121000001*QR_000020000100+P_121000101*QR_000020000101+P_221000001*QR_000020000200+P_221000101*QR_000020000201+P_321000001*QR_000020000300+P_321000101*QR_000020000301);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(P_021000001*QR_010000010000+P_021000101*QR_010000010001+P_121000001*QR_010000010100+P_121000101*QR_010000010101+P_221000001*QR_010000010200+P_221000101*QR_010000010201+P_321000001*QR_010000010300+P_321000101*QR_010000010301);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(P_021000001*QR_000010010000+P_021000101*QR_000010010001+P_121000001*QR_000010010100+P_121000101*QR_000010010101+P_221000001*QR_000010010200+P_221000101*QR_000010010201+P_321000001*QR_000010010300+P_321000101*QR_000010010301);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(P_021000001*QR_000000020000+P_021000101*QR_000000020001+P_121000001*QR_000000020100+P_121000101*QR_000000020101+P_221000001*QR_000000020200+P_221000101*QR_000000020201+P_321000001*QR_000000020300+P_321000101*QR_000000020301);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(P_020001001*QR_020000000000+P_020001101*QR_020000000001+P_020101001*QR_020000000010+P_020101101*QR_020000000011+P_120001001*QR_020000000100+P_120001101*QR_020000000101+P_120101001*QR_020000000110+P_120101101*QR_020000000111+P_220001001*QR_020000000200+P_220001101*QR_020000000201+P_220101001*QR_020000000210+P_220101101*QR_020000000211);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(P_020001001*QR_010010000000+P_020001101*QR_010010000001+P_020101001*QR_010010000010+P_020101101*QR_010010000011+P_120001001*QR_010010000100+P_120001101*QR_010010000101+P_120101001*QR_010010000110+P_120101101*QR_010010000111+P_220001001*QR_010010000200+P_220001101*QR_010010000201+P_220101001*QR_010010000210+P_220101101*QR_010010000211);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(P_020001001*QR_000020000000+P_020001101*QR_000020000001+P_020101001*QR_000020000010+P_020101101*QR_000020000011+P_120001001*QR_000020000100+P_120001101*QR_000020000101+P_120101001*QR_000020000110+P_120101101*QR_000020000111+P_220001001*QR_000020000200+P_220001101*QR_000020000201+P_220101001*QR_000020000210+P_220101101*QR_000020000211);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(P_020001001*QR_010000010000+P_020001101*QR_010000010001+P_020101001*QR_010000010010+P_020101101*QR_010000010011+P_120001001*QR_010000010100+P_120001101*QR_010000010101+P_120101001*QR_010000010110+P_120101101*QR_010000010111+P_220001001*QR_010000010200+P_220001101*QR_010000010201+P_220101001*QR_010000010210+P_220101101*QR_010000010211);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(P_020001001*QR_000010010000+P_020001101*QR_000010010001+P_020101001*QR_000010010010+P_020101101*QR_000010010011+P_120001001*QR_000010010100+P_120001101*QR_000010010101+P_120101001*QR_000010010110+P_120101101*QR_000010010111+P_220001001*QR_000010010200+P_220001101*QR_000010010201+P_220101001*QR_000010010210+P_220101101*QR_000010010211);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(P_020001001*QR_000000020000+P_020001101*QR_000000020001+P_020101001*QR_000000020010+P_020101101*QR_000000020011+P_120001001*QR_000000020100+P_120001101*QR_000000020101+P_120101001*QR_000000020110+P_120101101*QR_000000020111+P_220001001*QR_000000020200+P_220001101*QR_000000020201+P_220101001*QR_000000020210+P_220101101*QR_000000020211);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(P_020000002*QR_020000000000+P_020000102*QR_020000000001+P_020000202*QR_020000000002+P_120000002*QR_020000000100+P_120000102*QR_020000000101+P_120000202*QR_020000000102+P_220000002*QR_020000000200+P_220000102*QR_020000000201+P_220000202*QR_020000000202);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(P_020000002*QR_010010000000+P_020000102*QR_010010000001+P_020000202*QR_010010000002+P_120000002*QR_010010000100+P_120000102*QR_010010000101+P_120000202*QR_010010000102+P_220000002*QR_010010000200+P_220000102*QR_010010000201+P_220000202*QR_010010000202);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(P_020000002*QR_000020000000+P_020000102*QR_000020000001+P_020000202*QR_000020000002+P_120000002*QR_000020000100+P_120000102*QR_000020000101+P_120000202*QR_000020000102+P_220000002*QR_000020000200+P_220000102*QR_000020000201+P_220000202*QR_000020000202);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(P_020000002*QR_010000010000+P_020000102*QR_010000010001+P_020000202*QR_010000010002+P_120000002*QR_010000010100+P_120000102*QR_010000010101+P_120000202*QR_010000010102+P_220000002*QR_010000010200+P_220000102*QR_010000010201+P_220000202*QR_010000010202);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(P_020000002*QR_000010010000+P_020000102*QR_000010010001+P_020000202*QR_000010010002+P_120000002*QR_000010010100+P_120000102*QR_000010010101+P_120000202*QR_000010010102+P_220000002*QR_000010010200+P_220000102*QR_000010010201+P_220000202*QR_000010010202);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(P_020000002*QR_000000020000+P_020000102*QR_000000020001+P_020000202*QR_000000020002+P_120000002*QR_000000020100+P_120000102*QR_000000020101+P_120000202*QR_000000020102+P_220000002*QR_000000020200+P_220000102*QR_000000020201+P_220000202*QR_000000020202);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_012010000*QR_020000000000+P_012110000*QR_020000000010+P_112010000*QR_020000000100+P_112110000*QR_020000000110+P_212010000*QR_020000000200+P_212110000*QR_020000000210+P_312010000*QR_020000000300+P_312110000*QR_020000000310);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_012010000*QR_010010000000+P_012110000*QR_010010000010+P_112010000*QR_010010000100+P_112110000*QR_010010000110+P_212010000*QR_010010000200+P_212110000*QR_010010000210+P_312010000*QR_010010000300+P_312110000*QR_010010000310);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_012010000*QR_000020000000+P_012110000*QR_000020000010+P_112010000*QR_000020000100+P_112110000*QR_000020000110+P_212010000*QR_000020000200+P_212110000*QR_000020000210+P_312010000*QR_000020000300+P_312110000*QR_000020000310);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_012010000*QR_010000010000+P_012110000*QR_010000010010+P_112010000*QR_010000010100+P_112110000*QR_010000010110+P_212010000*QR_010000010200+P_212110000*QR_010000010210+P_312010000*QR_010000010300+P_312110000*QR_010000010310);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_012010000*QR_000010010000+P_012110000*QR_000010010010+P_112010000*QR_000010010100+P_112110000*QR_000010010110+P_212010000*QR_000010010200+P_212110000*QR_000010010210+P_312010000*QR_000010010300+P_312110000*QR_000010010310);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_012010000*QR_000000020000+P_012110000*QR_000000020010+P_112010000*QR_000000020100+P_112110000*QR_000000020110+P_212010000*QR_000000020200+P_212110000*QR_000000020210+P_312010000*QR_000000020300+P_312110000*QR_000000020310);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_011011000*QR_020000000000+P_011111000*QR_020000000010+P_011211000*QR_020000000020+P_111011000*QR_020000000100+P_111111000*QR_020000000110+P_111211000*QR_020000000120+P_211011000*QR_020000000200+P_211111000*QR_020000000210+P_211211000*QR_020000000220);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_011011000*QR_010010000000+P_011111000*QR_010010000010+P_011211000*QR_010010000020+P_111011000*QR_010010000100+P_111111000*QR_010010000110+P_111211000*QR_010010000120+P_211011000*QR_010010000200+P_211111000*QR_010010000210+P_211211000*QR_010010000220);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_011011000*QR_000020000000+P_011111000*QR_000020000010+P_011211000*QR_000020000020+P_111011000*QR_000020000100+P_111111000*QR_000020000110+P_111211000*QR_000020000120+P_211011000*QR_000020000200+P_211111000*QR_000020000210+P_211211000*QR_000020000220);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_011011000*QR_010000010000+P_011111000*QR_010000010010+P_011211000*QR_010000010020+P_111011000*QR_010000010100+P_111111000*QR_010000010110+P_111211000*QR_010000010120+P_211011000*QR_010000010200+P_211111000*QR_010000010210+P_211211000*QR_010000010220);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_011011000*QR_000010010000+P_011111000*QR_000010010010+P_011211000*QR_000010010020+P_111011000*QR_000010010100+P_111111000*QR_000010010110+P_111211000*QR_000010010120+P_211011000*QR_000010010200+P_211111000*QR_000010010210+P_211211000*QR_000010010220);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_011011000*QR_000000020000+P_011111000*QR_000000020010+P_011211000*QR_000000020020+P_111011000*QR_000000020100+P_111111000*QR_000000020110+P_111211000*QR_000000020120+P_211011000*QR_000000020200+P_211111000*QR_000000020210+P_211211000*QR_000000020220);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_010012000*QR_020000000000+P_010112000*QR_020000000010+P_010212000*QR_020000000020+P_010312000*QR_020000000030+P_110012000*QR_020000000100+P_110112000*QR_020000000110+P_110212000*QR_020000000120+P_110312000*QR_020000000130);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_010012000*QR_010010000000+P_010112000*QR_010010000010+P_010212000*QR_010010000020+P_010312000*QR_010010000030+P_110012000*QR_010010000100+P_110112000*QR_010010000110+P_110212000*QR_010010000120+P_110312000*QR_010010000130);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010012000*QR_000020000000+P_010112000*QR_000020000010+P_010212000*QR_000020000020+P_010312000*QR_000020000030+P_110012000*QR_000020000100+P_110112000*QR_000020000110+P_110212000*QR_000020000120+P_110312000*QR_000020000130);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_010012000*QR_010000010000+P_010112000*QR_010000010010+P_010212000*QR_010000010020+P_010312000*QR_010000010030+P_110012000*QR_010000010100+P_110112000*QR_010000010110+P_110212000*QR_010000010120+P_110312000*QR_010000010130);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010012000*QR_000010010000+P_010112000*QR_000010010010+P_010212000*QR_000010010020+P_010312000*QR_000010010030+P_110012000*QR_000010010100+P_110112000*QR_000010010110+P_110212000*QR_000010010120+P_110312000*QR_000010010130);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010012000*QR_000000020000+P_010112000*QR_000000020010+P_010212000*QR_000000020020+P_010312000*QR_000000020030+P_110012000*QR_000000020100+P_110112000*QR_000000020110+P_110212000*QR_000000020120+P_110312000*QR_000000020130);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(P_011010001*QR_020000000000+P_011010101*QR_020000000001+P_011110001*QR_020000000010+P_011110101*QR_020000000011+P_111010001*QR_020000000100+P_111010101*QR_020000000101+P_111110001*QR_020000000110+P_111110101*QR_020000000111+P_211010001*QR_020000000200+P_211010101*QR_020000000201+P_211110001*QR_020000000210+P_211110101*QR_020000000211);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(P_011010001*QR_010010000000+P_011010101*QR_010010000001+P_011110001*QR_010010000010+P_011110101*QR_010010000011+P_111010001*QR_010010000100+P_111010101*QR_010010000101+P_111110001*QR_010010000110+P_111110101*QR_010010000111+P_211010001*QR_010010000200+P_211010101*QR_010010000201+P_211110001*QR_010010000210+P_211110101*QR_010010000211);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(P_011010001*QR_000020000000+P_011010101*QR_000020000001+P_011110001*QR_000020000010+P_011110101*QR_000020000011+P_111010001*QR_000020000100+P_111010101*QR_000020000101+P_111110001*QR_000020000110+P_111110101*QR_000020000111+P_211010001*QR_000020000200+P_211010101*QR_000020000201+P_211110001*QR_000020000210+P_211110101*QR_000020000211);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(P_011010001*QR_010000010000+P_011010101*QR_010000010001+P_011110001*QR_010000010010+P_011110101*QR_010000010011+P_111010001*QR_010000010100+P_111010101*QR_010000010101+P_111110001*QR_010000010110+P_111110101*QR_010000010111+P_211010001*QR_010000010200+P_211010101*QR_010000010201+P_211110001*QR_010000010210+P_211110101*QR_010000010211);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(P_011010001*QR_000010010000+P_011010101*QR_000010010001+P_011110001*QR_000010010010+P_011110101*QR_000010010011+P_111010001*QR_000010010100+P_111010101*QR_000010010101+P_111110001*QR_000010010110+P_111110101*QR_000010010111+P_211010001*QR_000010010200+P_211010101*QR_000010010201+P_211110001*QR_000010010210+P_211110101*QR_000010010211);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(P_011010001*QR_000000020000+P_011010101*QR_000000020001+P_011110001*QR_000000020010+P_011110101*QR_000000020011+P_111010001*QR_000000020100+P_111010101*QR_000000020101+P_111110001*QR_000000020110+P_111110101*QR_000000020111+P_211010001*QR_000000020200+P_211010101*QR_000000020201+P_211110001*QR_000000020210+P_211110101*QR_000000020211);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(P_010011001*QR_020000000000+P_010011101*QR_020000000001+P_010111001*QR_020000000010+P_010111101*QR_020000000011+P_010211001*QR_020000000020+P_010211101*QR_020000000021+P_110011001*QR_020000000100+P_110011101*QR_020000000101+P_110111001*QR_020000000110+P_110111101*QR_020000000111+P_110211001*QR_020000000120+P_110211101*QR_020000000121);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(P_010011001*QR_010010000000+P_010011101*QR_010010000001+P_010111001*QR_010010000010+P_010111101*QR_010010000011+P_010211001*QR_010010000020+P_010211101*QR_010010000021+P_110011001*QR_010010000100+P_110011101*QR_010010000101+P_110111001*QR_010010000110+P_110111101*QR_010010000111+P_110211001*QR_010010000120+P_110211101*QR_010010000121);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(P_010011001*QR_000020000000+P_010011101*QR_000020000001+P_010111001*QR_000020000010+P_010111101*QR_000020000011+P_010211001*QR_000020000020+P_010211101*QR_000020000021+P_110011001*QR_000020000100+P_110011101*QR_000020000101+P_110111001*QR_000020000110+P_110111101*QR_000020000111+P_110211001*QR_000020000120+P_110211101*QR_000020000121);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(P_010011001*QR_010000010000+P_010011101*QR_010000010001+P_010111001*QR_010000010010+P_010111101*QR_010000010011+P_010211001*QR_010000010020+P_010211101*QR_010000010021+P_110011001*QR_010000010100+P_110011101*QR_010000010101+P_110111001*QR_010000010110+P_110111101*QR_010000010111+P_110211001*QR_010000010120+P_110211101*QR_010000010121);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(P_010011001*QR_000010010000+P_010011101*QR_000010010001+P_010111001*QR_000010010010+P_010111101*QR_000010010011+P_010211001*QR_000010010020+P_010211101*QR_000010010021+P_110011001*QR_000010010100+P_110011101*QR_000010010101+P_110111001*QR_000010010110+P_110111101*QR_000010010111+P_110211001*QR_000010010120+P_110211101*QR_000010010121);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(P_010011001*QR_000000020000+P_010011101*QR_000000020001+P_010111001*QR_000000020010+P_010111101*QR_000000020011+P_010211001*QR_000000020020+P_010211101*QR_000000020021+P_110011001*QR_000000020100+P_110011101*QR_000000020101+P_110111001*QR_000000020110+P_110111101*QR_000000020111+P_110211001*QR_000000020120+P_110211101*QR_000000020121);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(P_010010002*QR_020000000000+P_010010102*QR_020000000001+P_010010202*QR_020000000002+P_010110002*QR_020000000010+P_010110102*QR_020000000011+P_010110202*QR_020000000012+P_110010002*QR_020000000100+P_110010102*QR_020000000101+P_110010202*QR_020000000102+P_110110002*QR_020000000110+P_110110102*QR_020000000111+P_110110202*QR_020000000112);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(P_010010002*QR_010010000000+P_010010102*QR_010010000001+P_010010202*QR_010010000002+P_010110002*QR_010010000010+P_010110102*QR_010010000011+P_010110202*QR_010010000012+P_110010002*QR_010010000100+P_110010102*QR_010010000101+P_110010202*QR_010010000102+P_110110002*QR_010010000110+P_110110102*QR_010010000111+P_110110202*QR_010010000112);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(P_010010002*QR_000020000000+P_010010102*QR_000020000001+P_010010202*QR_000020000002+P_010110002*QR_000020000010+P_010110102*QR_000020000011+P_010110202*QR_000020000012+P_110010002*QR_000020000100+P_110010102*QR_000020000101+P_110010202*QR_000020000102+P_110110002*QR_000020000110+P_110110102*QR_000020000111+P_110110202*QR_000020000112);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(P_010010002*QR_010000010000+P_010010102*QR_010000010001+P_010010202*QR_010000010002+P_010110002*QR_010000010010+P_010110102*QR_010000010011+P_010110202*QR_010000010012+P_110010002*QR_010000010100+P_110010102*QR_010000010101+P_110010202*QR_010000010102+P_110110002*QR_010000010110+P_110110102*QR_010000010111+P_110110202*QR_010000010112);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(P_010010002*QR_000010010000+P_010010102*QR_000010010001+P_010010202*QR_000010010002+P_010110002*QR_000010010010+P_010110102*QR_000010010011+P_010110202*QR_000010010012+P_110010002*QR_000010010100+P_110010102*QR_000010010101+P_110010202*QR_000010010102+P_110110002*QR_000010010110+P_110110102*QR_000010010111+P_110110202*QR_000010010112);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(P_010010002*QR_000000020000+P_010010102*QR_000000020001+P_010010202*QR_000000020002+P_010110002*QR_000000020010+P_010110102*QR_000000020011+P_010110202*QR_000000020012+P_110010002*QR_000000020100+P_110010102*QR_000000020101+P_110010202*QR_000000020102+P_110110002*QR_000000020110+P_110110102*QR_000000020111+P_110110202*QR_000000020112);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_002020000*QR_020000000000+P_002120000*QR_020000000010+P_002220000*QR_020000000020+P_102020000*QR_020000000100+P_102120000*QR_020000000110+P_102220000*QR_020000000120+P_202020000*QR_020000000200+P_202120000*QR_020000000210+P_202220000*QR_020000000220);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_002020000*QR_010010000000+P_002120000*QR_010010000010+P_002220000*QR_010010000020+P_102020000*QR_010010000100+P_102120000*QR_010010000110+P_102220000*QR_010010000120+P_202020000*QR_010010000200+P_202120000*QR_010010000210+P_202220000*QR_010010000220);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_002020000*QR_000020000000+P_002120000*QR_000020000010+P_002220000*QR_000020000020+P_102020000*QR_000020000100+P_102120000*QR_000020000110+P_102220000*QR_000020000120+P_202020000*QR_000020000200+P_202120000*QR_000020000210+P_202220000*QR_000020000220);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_002020000*QR_010000010000+P_002120000*QR_010000010010+P_002220000*QR_010000010020+P_102020000*QR_010000010100+P_102120000*QR_010000010110+P_102220000*QR_010000010120+P_202020000*QR_010000010200+P_202120000*QR_010000010210+P_202220000*QR_010000010220);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_002020000*QR_000010010000+P_002120000*QR_000010010010+P_002220000*QR_000010010020+P_102020000*QR_000010010100+P_102120000*QR_000010010110+P_102220000*QR_000010010120+P_202020000*QR_000010010200+P_202120000*QR_000010010210+P_202220000*QR_000010010220);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_002020000*QR_000000020000+P_002120000*QR_000000020010+P_002220000*QR_000000020020+P_102020000*QR_000000020100+P_102120000*QR_000000020110+P_102220000*QR_000000020120+P_202020000*QR_000000020200+P_202120000*QR_000000020210+P_202220000*QR_000000020220);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_001021000*QR_020000000000+P_001121000*QR_020000000010+P_001221000*QR_020000000020+P_001321000*QR_020000000030+P_101021000*QR_020000000100+P_101121000*QR_020000000110+P_101221000*QR_020000000120+P_101321000*QR_020000000130);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_001021000*QR_010010000000+P_001121000*QR_010010000010+P_001221000*QR_010010000020+P_001321000*QR_010010000030+P_101021000*QR_010010000100+P_101121000*QR_010010000110+P_101221000*QR_010010000120+P_101321000*QR_010010000130);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_001021000*QR_000020000000+P_001121000*QR_000020000010+P_001221000*QR_000020000020+P_001321000*QR_000020000030+P_101021000*QR_000020000100+P_101121000*QR_000020000110+P_101221000*QR_000020000120+P_101321000*QR_000020000130);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_001021000*QR_010000010000+P_001121000*QR_010000010010+P_001221000*QR_010000010020+P_001321000*QR_010000010030+P_101021000*QR_010000010100+P_101121000*QR_010000010110+P_101221000*QR_010000010120+P_101321000*QR_010000010130);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_001021000*QR_000010010000+P_001121000*QR_000010010010+P_001221000*QR_000010010020+P_001321000*QR_000010010030+P_101021000*QR_000010010100+P_101121000*QR_000010010110+P_101221000*QR_000010010120+P_101321000*QR_000010010130);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_001021000*QR_000000020000+P_001121000*QR_000000020010+P_001221000*QR_000000020020+P_001321000*QR_000000020030+P_101021000*QR_000000020100+P_101121000*QR_000000020110+P_101221000*QR_000000020120+P_101321000*QR_000000020130);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_000022000*QR_020000000000+P_000122000*QR_020000000010+P_000222000*QR_020000000020+P_000322000*QR_020000000030+P_000422000*QR_020000000040);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_000022000*QR_010010000000+P_000122000*QR_010010000010+P_000222000*QR_010010000020+P_000322000*QR_010010000030+P_000422000*QR_010010000040);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000022000*QR_000020000000+P_000122000*QR_000020000010+P_000222000*QR_000020000020+P_000322000*QR_000020000030+P_000422000*QR_000020000040);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_000022000*QR_010000010000+P_000122000*QR_010000010010+P_000222000*QR_010000010020+P_000322000*QR_010000010030+P_000422000*QR_010000010040);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000022000*QR_000010010000+P_000122000*QR_000010010010+P_000222000*QR_000010010020+P_000322000*QR_000010010030+P_000422000*QR_000010010040);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000022000*QR_000000020000+P_000122000*QR_000000020010+P_000222000*QR_000000020020+P_000322000*QR_000000020030+P_000422000*QR_000000020040);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(P_001020001*QR_020000000000+P_001020101*QR_020000000001+P_001120001*QR_020000000010+P_001120101*QR_020000000011+P_001220001*QR_020000000020+P_001220101*QR_020000000021+P_101020001*QR_020000000100+P_101020101*QR_020000000101+P_101120001*QR_020000000110+P_101120101*QR_020000000111+P_101220001*QR_020000000120+P_101220101*QR_020000000121);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(P_001020001*QR_010010000000+P_001020101*QR_010010000001+P_001120001*QR_010010000010+P_001120101*QR_010010000011+P_001220001*QR_010010000020+P_001220101*QR_010010000021+P_101020001*QR_010010000100+P_101020101*QR_010010000101+P_101120001*QR_010010000110+P_101120101*QR_010010000111+P_101220001*QR_010010000120+P_101220101*QR_010010000121);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(P_001020001*QR_000020000000+P_001020101*QR_000020000001+P_001120001*QR_000020000010+P_001120101*QR_000020000011+P_001220001*QR_000020000020+P_001220101*QR_000020000021+P_101020001*QR_000020000100+P_101020101*QR_000020000101+P_101120001*QR_000020000110+P_101120101*QR_000020000111+P_101220001*QR_000020000120+P_101220101*QR_000020000121);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(P_001020001*QR_010000010000+P_001020101*QR_010000010001+P_001120001*QR_010000010010+P_001120101*QR_010000010011+P_001220001*QR_010000010020+P_001220101*QR_010000010021+P_101020001*QR_010000010100+P_101020101*QR_010000010101+P_101120001*QR_010000010110+P_101120101*QR_010000010111+P_101220001*QR_010000010120+P_101220101*QR_010000010121);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(P_001020001*QR_000010010000+P_001020101*QR_000010010001+P_001120001*QR_000010010010+P_001120101*QR_000010010011+P_001220001*QR_000010010020+P_001220101*QR_000010010021+P_101020001*QR_000010010100+P_101020101*QR_000010010101+P_101120001*QR_000010010110+P_101120101*QR_000010010111+P_101220001*QR_000010010120+P_101220101*QR_000010010121);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(P_001020001*QR_000000020000+P_001020101*QR_000000020001+P_001120001*QR_000000020010+P_001120101*QR_000000020011+P_001220001*QR_000000020020+P_001220101*QR_000000020021+P_101020001*QR_000000020100+P_101020101*QR_000000020101+P_101120001*QR_000000020110+P_101120101*QR_000000020111+P_101220001*QR_000000020120+P_101220101*QR_000000020121);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(P_000021001*QR_020000000000+P_000021101*QR_020000000001+P_000121001*QR_020000000010+P_000121101*QR_020000000011+P_000221001*QR_020000000020+P_000221101*QR_020000000021+P_000321001*QR_020000000030+P_000321101*QR_020000000031);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(P_000021001*QR_010010000000+P_000021101*QR_010010000001+P_000121001*QR_010010000010+P_000121101*QR_010010000011+P_000221001*QR_010010000020+P_000221101*QR_010010000021+P_000321001*QR_010010000030+P_000321101*QR_010010000031);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(P_000021001*QR_000020000000+P_000021101*QR_000020000001+P_000121001*QR_000020000010+P_000121101*QR_000020000011+P_000221001*QR_000020000020+P_000221101*QR_000020000021+P_000321001*QR_000020000030+P_000321101*QR_000020000031);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(P_000021001*QR_010000010000+P_000021101*QR_010000010001+P_000121001*QR_010000010010+P_000121101*QR_010000010011+P_000221001*QR_010000010020+P_000221101*QR_010000010021+P_000321001*QR_010000010030+P_000321101*QR_010000010031);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(P_000021001*QR_000010010000+P_000021101*QR_000010010001+P_000121001*QR_000010010010+P_000121101*QR_000010010011+P_000221001*QR_000010010020+P_000221101*QR_000010010021+P_000321001*QR_000010010030+P_000321101*QR_000010010031);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(P_000021001*QR_000000020000+P_000021101*QR_000000020001+P_000121001*QR_000000020010+P_000121101*QR_000000020011+P_000221001*QR_000000020020+P_000221101*QR_000000020021+P_000321001*QR_000000020030+P_000321101*QR_000000020031);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(P_000020002*QR_020000000000+P_000020102*QR_020000000001+P_000020202*QR_020000000002+P_000120002*QR_020000000010+P_000120102*QR_020000000011+P_000120202*QR_020000000012+P_000220002*QR_020000000020+P_000220102*QR_020000000021+P_000220202*QR_020000000022);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(P_000020002*QR_010010000000+P_000020102*QR_010010000001+P_000020202*QR_010010000002+P_000120002*QR_010010000010+P_000120102*QR_010010000011+P_000120202*QR_010010000012+P_000220002*QR_010010000020+P_000220102*QR_010010000021+P_000220202*QR_010010000022);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(P_000020002*QR_000020000000+P_000020102*QR_000020000001+P_000020202*QR_000020000002+P_000120002*QR_000020000010+P_000120102*QR_000020000011+P_000120202*QR_000020000012+P_000220002*QR_000020000020+P_000220102*QR_000020000021+P_000220202*QR_000020000022);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(P_000020002*QR_010000010000+P_000020102*QR_010000010001+P_000020202*QR_010000010002+P_000120002*QR_010000010010+P_000120102*QR_010000010011+P_000120202*QR_010000010012+P_000220002*QR_010000010020+P_000220102*QR_010000010021+P_000220202*QR_010000010022);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(P_000020002*QR_000010010000+P_000020102*QR_000010010001+P_000020202*QR_000010010002+P_000120002*QR_000010010010+P_000120102*QR_000010010011+P_000120202*QR_000010010012+P_000220002*QR_000010010020+P_000220102*QR_000010010021+P_000220202*QR_000010010022);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(P_000020002*QR_000000020000+P_000020102*QR_000000020001+P_000020202*QR_000000020002+P_000120002*QR_000000020010+P_000120102*QR_000000020011+P_000120202*QR_000000020012+P_000220002*QR_000000020020+P_000220102*QR_000000020021+P_000220202*QR_000000020022);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_012000010*QR_020000000000+P_012000110*QR_020000000001+P_112000010*QR_020000000100+P_112000110*QR_020000000101+P_212000010*QR_020000000200+P_212000110*QR_020000000201+P_312000010*QR_020000000300+P_312000110*QR_020000000301);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_012000010*QR_010010000000+P_012000110*QR_010010000001+P_112000010*QR_010010000100+P_112000110*QR_010010000101+P_212000010*QR_010010000200+P_212000110*QR_010010000201+P_312000010*QR_010010000300+P_312000110*QR_010010000301);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_012000010*QR_000020000000+P_012000110*QR_000020000001+P_112000010*QR_000020000100+P_112000110*QR_000020000101+P_212000010*QR_000020000200+P_212000110*QR_000020000201+P_312000010*QR_000020000300+P_312000110*QR_000020000301);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_012000010*QR_010000010000+P_012000110*QR_010000010001+P_112000010*QR_010000010100+P_112000110*QR_010000010101+P_212000010*QR_010000010200+P_212000110*QR_010000010201+P_312000010*QR_010000010300+P_312000110*QR_010000010301);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_012000010*QR_000010010000+P_012000110*QR_000010010001+P_112000010*QR_000010010100+P_112000110*QR_000010010101+P_212000010*QR_000010010200+P_212000110*QR_000010010201+P_312000010*QR_000010010300+P_312000110*QR_000010010301);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_012000010*QR_000000020000+P_012000110*QR_000000020001+P_112000010*QR_000000020100+P_112000110*QR_000000020101+P_212000010*QR_000000020200+P_212000110*QR_000000020201+P_312000010*QR_000000020300+P_312000110*QR_000000020301);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_011001010*QR_020000000000+P_011001110*QR_020000000001+P_011101010*QR_020000000010+P_011101110*QR_020000000011+P_111001010*QR_020000000100+P_111001110*QR_020000000101+P_111101010*QR_020000000110+P_111101110*QR_020000000111+P_211001010*QR_020000000200+P_211001110*QR_020000000201+P_211101010*QR_020000000210+P_211101110*QR_020000000211);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_011001010*QR_010010000000+P_011001110*QR_010010000001+P_011101010*QR_010010000010+P_011101110*QR_010010000011+P_111001010*QR_010010000100+P_111001110*QR_010010000101+P_111101010*QR_010010000110+P_111101110*QR_010010000111+P_211001010*QR_010010000200+P_211001110*QR_010010000201+P_211101010*QR_010010000210+P_211101110*QR_010010000211);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_011001010*QR_000020000000+P_011001110*QR_000020000001+P_011101010*QR_000020000010+P_011101110*QR_000020000011+P_111001010*QR_000020000100+P_111001110*QR_000020000101+P_111101010*QR_000020000110+P_111101110*QR_000020000111+P_211001010*QR_000020000200+P_211001110*QR_000020000201+P_211101010*QR_000020000210+P_211101110*QR_000020000211);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_011001010*QR_010000010000+P_011001110*QR_010000010001+P_011101010*QR_010000010010+P_011101110*QR_010000010011+P_111001010*QR_010000010100+P_111001110*QR_010000010101+P_111101010*QR_010000010110+P_111101110*QR_010000010111+P_211001010*QR_010000010200+P_211001110*QR_010000010201+P_211101010*QR_010000010210+P_211101110*QR_010000010211);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_011001010*QR_000010010000+P_011001110*QR_000010010001+P_011101010*QR_000010010010+P_011101110*QR_000010010011+P_111001010*QR_000010010100+P_111001110*QR_000010010101+P_111101010*QR_000010010110+P_111101110*QR_000010010111+P_211001010*QR_000010010200+P_211001110*QR_000010010201+P_211101010*QR_000010010210+P_211101110*QR_000010010211);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_011001010*QR_000000020000+P_011001110*QR_000000020001+P_011101010*QR_000000020010+P_011101110*QR_000000020011+P_111001010*QR_000000020100+P_111001110*QR_000000020101+P_111101010*QR_000000020110+P_111101110*QR_000000020111+P_211001010*QR_000000020200+P_211001110*QR_000000020201+P_211101010*QR_000000020210+P_211101110*QR_000000020211);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_010002010*QR_020000000000+P_010002110*QR_020000000001+P_010102010*QR_020000000010+P_010102110*QR_020000000011+P_010202010*QR_020000000020+P_010202110*QR_020000000021+P_110002010*QR_020000000100+P_110002110*QR_020000000101+P_110102010*QR_020000000110+P_110102110*QR_020000000111+P_110202010*QR_020000000120+P_110202110*QR_020000000121);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_010002010*QR_010010000000+P_010002110*QR_010010000001+P_010102010*QR_010010000010+P_010102110*QR_010010000011+P_010202010*QR_010010000020+P_010202110*QR_010010000021+P_110002010*QR_010010000100+P_110002110*QR_010010000101+P_110102010*QR_010010000110+P_110102110*QR_010010000111+P_110202010*QR_010010000120+P_110202110*QR_010010000121);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010002010*QR_000020000000+P_010002110*QR_000020000001+P_010102010*QR_000020000010+P_010102110*QR_000020000011+P_010202010*QR_000020000020+P_010202110*QR_000020000021+P_110002010*QR_000020000100+P_110002110*QR_000020000101+P_110102010*QR_000020000110+P_110102110*QR_000020000111+P_110202010*QR_000020000120+P_110202110*QR_000020000121);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_010002010*QR_010000010000+P_010002110*QR_010000010001+P_010102010*QR_010000010010+P_010102110*QR_010000010011+P_010202010*QR_010000010020+P_010202110*QR_010000010021+P_110002010*QR_010000010100+P_110002110*QR_010000010101+P_110102010*QR_010000010110+P_110102110*QR_010000010111+P_110202010*QR_010000010120+P_110202110*QR_010000010121);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010002010*QR_000010010000+P_010002110*QR_000010010001+P_010102010*QR_000010010010+P_010102110*QR_000010010011+P_010202010*QR_000010010020+P_010202110*QR_000010010021+P_110002010*QR_000010010100+P_110002110*QR_000010010101+P_110102010*QR_000010010110+P_110102110*QR_000010010111+P_110202010*QR_000010010120+P_110202110*QR_000010010121);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010002010*QR_000000020000+P_010002110*QR_000000020001+P_010102010*QR_000000020010+P_010102110*QR_000000020011+P_010202010*QR_000000020020+P_010202110*QR_000000020021+P_110002010*QR_000000020100+P_110002110*QR_000000020101+P_110102010*QR_000000020110+P_110102110*QR_000000020111+P_110202010*QR_000000020120+P_110202110*QR_000000020121);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(P_011000011*QR_020000000000+P_011000111*QR_020000000001+P_011000211*QR_020000000002+P_111000011*QR_020000000100+P_111000111*QR_020000000101+P_111000211*QR_020000000102+P_211000011*QR_020000000200+P_211000111*QR_020000000201+P_211000211*QR_020000000202);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(P_011000011*QR_010010000000+P_011000111*QR_010010000001+P_011000211*QR_010010000002+P_111000011*QR_010010000100+P_111000111*QR_010010000101+P_111000211*QR_010010000102+P_211000011*QR_010010000200+P_211000111*QR_010010000201+P_211000211*QR_010010000202);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(P_011000011*QR_000020000000+P_011000111*QR_000020000001+P_011000211*QR_000020000002+P_111000011*QR_000020000100+P_111000111*QR_000020000101+P_111000211*QR_000020000102+P_211000011*QR_000020000200+P_211000111*QR_000020000201+P_211000211*QR_000020000202);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(P_011000011*QR_010000010000+P_011000111*QR_010000010001+P_011000211*QR_010000010002+P_111000011*QR_010000010100+P_111000111*QR_010000010101+P_111000211*QR_010000010102+P_211000011*QR_010000010200+P_211000111*QR_010000010201+P_211000211*QR_010000010202);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(P_011000011*QR_000010010000+P_011000111*QR_000010010001+P_011000211*QR_000010010002+P_111000011*QR_000010010100+P_111000111*QR_000010010101+P_111000211*QR_000010010102+P_211000011*QR_000010010200+P_211000111*QR_000010010201+P_211000211*QR_000010010202);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(P_011000011*QR_000000020000+P_011000111*QR_000000020001+P_011000211*QR_000000020002+P_111000011*QR_000000020100+P_111000111*QR_000000020101+P_111000211*QR_000000020102+P_211000011*QR_000000020200+P_211000111*QR_000000020201+P_211000211*QR_000000020202);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(P_010001011*QR_020000000000+P_010001111*QR_020000000001+P_010001211*QR_020000000002+P_010101011*QR_020000000010+P_010101111*QR_020000000011+P_010101211*QR_020000000012+P_110001011*QR_020000000100+P_110001111*QR_020000000101+P_110001211*QR_020000000102+P_110101011*QR_020000000110+P_110101111*QR_020000000111+P_110101211*QR_020000000112);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(P_010001011*QR_010010000000+P_010001111*QR_010010000001+P_010001211*QR_010010000002+P_010101011*QR_010010000010+P_010101111*QR_010010000011+P_010101211*QR_010010000012+P_110001011*QR_010010000100+P_110001111*QR_010010000101+P_110001211*QR_010010000102+P_110101011*QR_010010000110+P_110101111*QR_010010000111+P_110101211*QR_010010000112);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(P_010001011*QR_000020000000+P_010001111*QR_000020000001+P_010001211*QR_000020000002+P_010101011*QR_000020000010+P_010101111*QR_000020000011+P_010101211*QR_000020000012+P_110001011*QR_000020000100+P_110001111*QR_000020000101+P_110001211*QR_000020000102+P_110101011*QR_000020000110+P_110101111*QR_000020000111+P_110101211*QR_000020000112);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(P_010001011*QR_010000010000+P_010001111*QR_010000010001+P_010001211*QR_010000010002+P_010101011*QR_010000010010+P_010101111*QR_010000010011+P_010101211*QR_010000010012+P_110001011*QR_010000010100+P_110001111*QR_010000010101+P_110001211*QR_010000010102+P_110101011*QR_010000010110+P_110101111*QR_010000010111+P_110101211*QR_010000010112);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(P_010001011*QR_000010010000+P_010001111*QR_000010010001+P_010001211*QR_000010010002+P_010101011*QR_000010010010+P_010101111*QR_000010010011+P_010101211*QR_000010010012+P_110001011*QR_000010010100+P_110001111*QR_000010010101+P_110001211*QR_000010010102+P_110101011*QR_000010010110+P_110101111*QR_000010010111+P_110101211*QR_000010010112);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(P_010001011*QR_000000020000+P_010001111*QR_000000020001+P_010001211*QR_000000020002+P_010101011*QR_000000020010+P_010101111*QR_000000020011+P_010101211*QR_000000020012+P_110001011*QR_000000020100+P_110001111*QR_000000020101+P_110001211*QR_000000020102+P_110101011*QR_000000020110+P_110101111*QR_000000020111+P_110101211*QR_000000020112);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(P_010000012*QR_020000000000+P_010000112*QR_020000000001+P_010000212*QR_020000000002+P_010000312*QR_020000000003+P_110000012*QR_020000000100+P_110000112*QR_020000000101+P_110000212*QR_020000000102+P_110000312*QR_020000000103);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(P_010000012*QR_010010000000+P_010000112*QR_010010000001+P_010000212*QR_010010000002+P_010000312*QR_010010000003+P_110000012*QR_010010000100+P_110000112*QR_010010000101+P_110000212*QR_010010000102+P_110000312*QR_010010000103);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(P_010000012*QR_000020000000+P_010000112*QR_000020000001+P_010000212*QR_000020000002+P_010000312*QR_000020000003+P_110000012*QR_000020000100+P_110000112*QR_000020000101+P_110000212*QR_000020000102+P_110000312*QR_000020000103);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(P_010000012*QR_010000010000+P_010000112*QR_010000010001+P_010000212*QR_010000010002+P_010000312*QR_010000010003+P_110000012*QR_010000010100+P_110000112*QR_010000010101+P_110000212*QR_010000010102+P_110000312*QR_010000010103);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(P_010000012*QR_000010010000+P_010000112*QR_000010010001+P_010000212*QR_000010010002+P_010000312*QR_000010010003+P_110000012*QR_000010010100+P_110000112*QR_000010010101+P_110000212*QR_000010010102+P_110000312*QR_000010010103);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(P_010000012*QR_000000020000+P_010000112*QR_000000020001+P_010000212*QR_000000020002+P_010000312*QR_000000020003+P_110000012*QR_000000020100+P_110000112*QR_000000020101+P_110000212*QR_000000020102+P_110000312*QR_000000020103);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_002010010*QR_020000000000+P_002010110*QR_020000000001+P_002110010*QR_020000000010+P_002110110*QR_020000000011+P_102010010*QR_020000000100+P_102010110*QR_020000000101+P_102110010*QR_020000000110+P_102110110*QR_020000000111+P_202010010*QR_020000000200+P_202010110*QR_020000000201+P_202110010*QR_020000000210+P_202110110*QR_020000000211);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_002010010*QR_010010000000+P_002010110*QR_010010000001+P_002110010*QR_010010000010+P_002110110*QR_010010000011+P_102010010*QR_010010000100+P_102010110*QR_010010000101+P_102110010*QR_010010000110+P_102110110*QR_010010000111+P_202010010*QR_010010000200+P_202010110*QR_010010000201+P_202110010*QR_010010000210+P_202110110*QR_010010000211);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_002010010*QR_000020000000+P_002010110*QR_000020000001+P_002110010*QR_000020000010+P_002110110*QR_000020000011+P_102010010*QR_000020000100+P_102010110*QR_000020000101+P_102110010*QR_000020000110+P_102110110*QR_000020000111+P_202010010*QR_000020000200+P_202010110*QR_000020000201+P_202110010*QR_000020000210+P_202110110*QR_000020000211);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_002010010*QR_010000010000+P_002010110*QR_010000010001+P_002110010*QR_010000010010+P_002110110*QR_010000010011+P_102010010*QR_010000010100+P_102010110*QR_010000010101+P_102110010*QR_010000010110+P_102110110*QR_010000010111+P_202010010*QR_010000010200+P_202010110*QR_010000010201+P_202110010*QR_010000010210+P_202110110*QR_010000010211);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_002010010*QR_000010010000+P_002010110*QR_000010010001+P_002110010*QR_000010010010+P_002110110*QR_000010010011+P_102010010*QR_000010010100+P_102010110*QR_000010010101+P_102110010*QR_000010010110+P_102110110*QR_000010010111+P_202010010*QR_000010010200+P_202010110*QR_000010010201+P_202110010*QR_000010010210+P_202110110*QR_000010010211);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_002010010*QR_000000020000+P_002010110*QR_000000020001+P_002110010*QR_000000020010+P_002110110*QR_000000020011+P_102010010*QR_000000020100+P_102010110*QR_000000020101+P_102110010*QR_000000020110+P_102110110*QR_000000020111+P_202010010*QR_000000020200+P_202010110*QR_000000020201+P_202110010*QR_000000020210+P_202110110*QR_000000020211);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_001011010*QR_020000000000+P_001011110*QR_020000000001+P_001111010*QR_020000000010+P_001111110*QR_020000000011+P_001211010*QR_020000000020+P_001211110*QR_020000000021+P_101011010*QR_020000000100+P_101011110*QR_020000000101+P_101111010*QR_020000000110+P_101111110*QR_020000000111+P_101211010*QR_020000000120+P_101211110*QR_020000000121);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_001011010*QR_010010000000+P_001011110*QR_010010000001+P_001111010*QR_010010000010+P_001111110*QR_010010000011+P_001211010*QR_010010000020+P_001211110*QR_010010000021+P_101011010*QR_010010000100+P_101011110*QR_010010000101+P_101111010*QR_010010000110+P_101111110*QR_010010000111+P_101211010*QR_010010000120+P_101211110*QR_010010000121);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_001011010*QR_000020000000+P_001011110*QR_000020000001+P_001111010*QR_000020000010+P_001111110*QR_000020000011+P_001211010*QR_000020000020+P_001211110*QR_000020000021+P_101011010*QR_000020000100+P_101011110*QR_000020000101+P_101111010*QR_000020000110+P_101111110*QR_000020000111+P_101211010*QR_000020000120+P_101211110*QR_000020000121);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_001011010*QR_010000010000+P_001011110*QR_010000010001+P_001111010*QR_010000010010+P_001111110*QR_010000010011+P_001211010*QR_010000010020+P_001211110*QR_010000010021+P_101011010*QR_010000010100+P_101011110*QR_010000010101+P_101111010*QR_010000010110+P_101111110*QR_010000010111+P_101211010*QR_010000010120+P_101211110*QR_010000010121);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_001011010*QR_000010010000+P_001011110*QR_000010010001+P_001111010*QR_000010010010+P_001111110*QR_000010010011+P_001211010*QR_000010010020+P_001211110*QR_000010010021+P_101011010*QR_000010010100+P_101011110*QR_000010010101+P_101111010*QR_000010010110+P_101111110*QR_000010010111+P_101211010*QR_000010010120+P_101211110*QR_000010010121);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_001011010*QR_000000020000+P_001011110*QR_000000020001+P_001111010*QR_000000020010+P_001111110*QR_000000020011+P_001211010*QR_000000020020+P_001211110*QR_000000020021+P_101011010*QR_000000020100+P_101011110*QR_000000020101+P_101111010*QR_000000020110+P_101111110*QR_000000020111+P_101211010*QR_000000020120+P_101211110*QR_000000020121);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_000012010*QR_020000000000+P_000012110*QR_020000000001+P_000112010*QR_020000000010+P_000112110*QR_020000000011+P_000212010*QR_020000000020+P_000212110*QR_020000000021+P_000312010*QR_020000000030+P_000312110*QR_020000000031);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_000012010*QR_010010000000+P_000012110*QR_010010000001+P_000112010*QR_010010000010+P_000112110*QR_010010000011+P_000212010*QR_010010000020+P_000212110*QR_010010000021+P_000312010*QR_010010000030+P_000312110*QR_010010000031);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000012010*QR_000020000000+P_000012110*QR_000020000001+P_000112010*QR_000020000010+P_000112110*QR_000020000011+P_000212010*QR_000020000020+P_000212110*QR_000020000021+P_000312010*QR_000020000030+P_000312110*QR_000020000031);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_000012010*QR_010000010000+P_000012110*QR_010000010001+P_000112010*QR_010000010010+P_000112110*QR_010000010011+P_000212010*QR_010000010020+P_000212110*QR_010000010021+P_000312010*QR_010000010030+P_000312110*QR_010000010031);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000012010*QR_000010010000+P_000012110*QR_000010010001+P_000112010*QR_000010010010+P_000112110*QR_000010010011+P_000212010*QR_000010010020+P_000212110*QR_000010010021+P_000312010*QR_000010010030+P_000312110*QR_000010010031);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000012010*QR_000000020000+P_000012110*QR_000000020001+P_000112010*QR_000000020010+P_000112110*QR_000000020011+P_000212010*QR_000000020020+P_000212110*QR_000000020021+P_000312010*QR_000000020030+P_000312110*QR_000000020031);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(P_001010011*QR_020000000000+P_001010111*QR_020000000001+P_001010211*QR_020000000002+P_001110011*QR_020000000010+P_001110111*QR_020000000011+P_001110211*QR_020000000012+P_101010011*QR_020000000100+P_101010111*QR_020000000101+P_101010211*QR_020000000102+P_101110011*QR_020000000110+P_101110111*QR_020000000111+P_101110211*QR_020000000112);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(P_001010011*QR_010010000000+P_001010111*QR_010010000001+P_001010211*QR_010010000002+P_001110011*QR_010010000010+P_001110111*QR_010010000011+P_001110211*QR_010010000012+P_101010011*QR_010010000100+P_101010111*QR_010010000101+P_101010211*QR_010010000102+P_101110011*QR_010010000110+P_101110111*QR_010010000111+P_101110211*QR_010010000112);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(P_001010011*QR_000020000000+P_001010111*QR_000020000001+P_001010211*QR_000020000002+P_001110011*QR_000020000010+P_001110111*QR_000020000011+P_001110211*QR_000020000012+P_101010011*QR_000020000100+P_101010111*QR_000020000101+P_101010211*QR_000020000102+P_101110011*QR_000020000110+P_101110111*QR_000020000111+P_101110211*QR_000020000112);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(P_001010011*QR_010000010000+P_001010111*QR_010000010001+P_001010211*QR_010000010002+P_001110011*QR_010000010010+P_001110111*QR_010000010011+P_001110211*QR_010000010012+P_101010011*QR_010000010100+P_101010111*QR_010000010101+P_101010211*QR_010000010102+P_101110011*QR_010000010110+P_101110111*QR_010000010111+P_101110211*QR_010000010112);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(P_001010011*QR_000010010000+P_001010111*QR_000010010001+P_001010211*QR_000010010002+P_001110011*QR_000010010010+P_001110111*QR_000010010011+P_001110211*QR_000010010012+P_101010011*QR_000010010100+P_101010111*QR_000010010101+P_101010211*QR_000010010102+P_101110011*QR_000010010110+P_101110111*QR_000010010111+P_101110211*QR_000010010112);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(P_001010011*QR_000000020000+P_001010111*QR_000000020001+P_001010211*QR_000000020002+P_001110011*QR_000000020010+P_001110111*QR_000000020011+P_001110211*QR_000000020012+P_101010011*QR_000000020100+P_101010111*QR_000000020101+P_101010211*QR_000000020102+P_101110011*QR_000000020110+P_101110111*QR_000000020111+P_101110211*QR_000000020112);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(P_000011011*QR_020000000000+P_000011111*QR_020000000001+P_000011211*QR_020000000002+P_000111011*QR_020000000010+P_000111111*QR_020000000011+P_000111211*QR_020000000012+P_000211011*QR_020000000020+P_000211111*QR_020000000021+P_000211211*QR_020000000022);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(P_000011011*QR_010010000000+P_000011111*QR_010010000001+P_000011211*QR_010010000002+P_000111011*QR_010010000010+P_000111111*QR_010010000011+P_000111211*QR_010010000012+P_000211011*QR_010010000020+P_000211111*QR_010010000021+P_000211211*QR_010010000022);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(P_000011011*QR_000020000000+P_000011111*QR_000020000001+P_000011211*QR_000020000002+P_000111011*QR_000020000010+P_000111111*QR_000020000011+P_000111211*QR_000020000012+P_000211011*QR_000020000020+P_000211111*QR_000020000021+P_000211211*QR_000020000022);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(P_000011011*QR_010000010000+P_000011111*QR_010000010001+P_000011211*QR_010000010002+P_000111011*QR_010000010010+P_000111111*QR_010000010011+P_000111211*QR_010000010012+P_000211011*QR_010000010020+P_000211111*QR_010000010021+P_000211211*QR_010000010022);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(P_000011011*QR_000010010000+P_000011111*QR_000010010001+P_000011211*QR_000010010002+P_000111011*QR_000010010010+P_000111111*QR_000010010011+P_000111211*QR_000010010012+P_000211011*QR_000010010020+P_000211111*QR_000010010021+P_000211211*QR_000010010022);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(P_000011011*QR_000000020000+P_000011111*QR_000000020001+P_000011211*QR_000000020002+P_000111011*QR_000000020010+P_000111111*QR_000000020011+P_000111211*QR_000000020012+P_000211011*QR_000000020020+P_000211111*QR_000000020021+P_000211211*QR_000000020022);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(P_000010012*QR_020000000000+P_000010112*QR_020000000001+P_000010212*QR_020000000002+P_000010312*QR_020000000003+P_000110012*QR_020000000010+P_000110112*QR_020000000011+P_000110212*QR_020000000012+P_000110312*QR_020000000013);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(P_000010012*QR_010010000000+P_000010112*QR_010010000001+P_000010212*QR_010010000002+P_000010312*QR_010010000003+P_000110012*QR_010010000010+P_000110112*QR_010010000011+P_000110212*QR_010010000012+P_000110312*QR_010010000013);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(P_000010012*QR_000020000000+P_000010112*QR_000020000001+P_000010212*QR_000020000002+P_000010312*QR_000020000003+P_000110012*QR_000020000010+P_000110112*QR_000020000011+P_000110212*QR_000020000012+P_000110312*QR_000020000013);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(P_000010012*QR_010000010000+P_000010112*QR_010000010001+P_000010212*QR_010000010002+P_000010312*QR_010000010003+P_000110012*QR_010000010010+P_000110112*QR_010000010011+P_000110212*QR_010000010012+P_000110312*QR_010000010013);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(P_000010012*QR_000010010000+P_000010112*QR_000010010001+P_000010212*QR_000010010002+P_000010312*QR_000010010003+P_000110012*QR_000010010010+P_000110112*QR_000010010011+P_000110212*QR_000010010012+P_000110312*QR_000010010013);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(P_000010012*QR_000000020000+P_000010112*QR_000000020001+P_000010212*QR_000000020002+P_000010312*QR_000000020003+P_000110012*QR_000000020010+P_000110112*QR_000000020011+P_000110212*QR_000000020012+P_000110312*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_002000020*QR_020000000000+P_002000120*QR_020000000001+P_002000220*QR_020000000002+P_102000020*QR_020000000100+P_102000120*QR_020000000101+P_102000220*QR_020000000102+P_202000020*QR_020000000200+P_202000120*QR_020000000201+P_202000220*QR_020000000202);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_002000020*QR_010010000000+P_002000120*QR_010010000001+P_002000220*QR_010010000002+P_102000020*QR_010010000100+P_102000120*QR_010010000101+P_102000220*QR_010010000102+P_202000020*QR_010010000200+P_202000120*QR_010010000201+P_202000220*QR_010010000202);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_002000020*QR_000020000000+P_002000120*QR_000020000001+P_002000220*QR_000020000002+P_102000020*QR_000020000100+P_102000120*QR_000020000101+P_102000220*QR_000020000102+P_202000020*QR_000020000200+P_202000120*QR_000020000201+P_202000220*QR_000020000202);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_002000020*QR_010000010000+P_002000120*QR_010000010001+P_002000220*QR_010000010002+P_102000020*QR_010000010100+P_102000120*QR_010000010101+P_102000220*QR_010000010102+P_202000020*QR_010000010200+P_202000120*QR_010000010201+P_202000220*QR_010000010202);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_002000020*QR_000010010000+P_002000120*QR_000010010001+P_002000220*QR_000010010002+P_102000020*QR_000010010100+P_102000120*QR_000010010101+P_102000220*QR_000010010102+P_202000020*QR_000010010200+P_202000120*QR_000010010201+P_202000220*QR_000010010202);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_002000020*QR_000000020000+P_002000120*QR_000000020001+P_002000220*QR_000000020002+P_102000020*QR_000000020100+P_102000120*QR_000000020101+P_102000220*QR_000000020102+P_202000020*QR_000000020200+P_202000120*QR_000000020201+P_202000220*QR_000000020202);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_001001020*QR_020000000000+P_001001120*QR_020000000001+P_001001220*QR_020000000002+P_001101020*QR_020000000010+P_001101120*QR_020000000011+P_001101220*QR_020000000012+P_101001020*QR_020000000100+P_101001120*QR_020000000101+P_101001220*QR_020000000102+P_101101020*QR_020000000110+P_101101120*QR_020000000111+P_101101220*QR_020000000112);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_001001020*QR_010010000000+P_001001120*QR_010010000001+P_001001220*QR_010010000002+P_001101020*QR_010010000010+P_001101120*QR_010010000011+P_001101220*QR_010010000012+P_101001020*QR_010010000100+P_101001120*QR_010010000101+P_101001220*QR_010010000102+P_101101020*QR_010010000110+P_101101120*QR_010010000111+P_101101220*QR_010010000112);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_001001020*QR_000020000000+P_001001120*QR_000020000001+P_001001220*QR_000020000002+P_001101020*QR_000020000010+P_001101120*QR_000020000011+P_001101220*QR_000020000012+P_101001020*QR_000020000100+P_101001120*QR_000020000101+P_101001220*QR_000020000102+P_101101020*QR_000020000110+P_101101120*QR_000020000111+P_101101220*QR_000020000112);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_001001020*QR_010000010000+P_001001120*QR_010000010001+P_001001220*QR_010000010002+P_001101020*QR_010000010010+P_001101120*QR_010000010011+P_001101220*QR_010000010012+P_101001020*QR_010000010100+P_101001120*QR_010000010101+P_101001220*QR_010000010102+P_101101020*QR_010000010110+P_101101120*QR_010000010111+P_101101220*QR_010000010112);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_001001020*QR_000010010000+P_001001120*QR_000010010001+P_001001220*QR_000010010002+P_001101020*QR_000010010010+P_001101120*QR_000010010011+P_001101220*QR_000010010012+P_101001020*QR_000010010100+P_101001120*QR_000010010101+P_101001220*QR_000010010102+P_101101020*QR_000010010110+P_101101120*QR_000010010111+P_101101220*QR_000010010112);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_001001020*QR_000000020000+P_001001120*QR_000000020001+P_001001220*QR_000000020002+P_001101020*QR_000000020010+P_001101120*QR_000000020011+P_001101220*QR_000000020012+P_101001020*QR_000000020100+P_101001120*QR_000000020101+P_101001220*QR_000000020102+P_101101020*QR_000000020110+P_101101120*QR_000000020111+P_101101220*QR_000000020112);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_000002020*QR_020000000000+P_000002120*QR_020000000001+P_000002220*QR_020000000002+P_000102020*QR_020000000010+P_000102120*QR_020000000011+P_000102220*QR_020000000012+P_000202020*QR_020000000020+P_000202120*QR_020000000021+P_000202220*QR_020000000022);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_000002020*QR_010010000000+P_000002120*QR_010010000001+P_000002220*QR_010010000002+P_000102020*QR_010010000010+P_000102120*QR_010010000011+P_000102220*QR_010010000012+P_000202020*QR_010010000020+P_000202120*QR_010010000021+P_000202220*QR_010010000022);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000002020*QR_000020000000+P_000002120*QR_000020000001+P_000002220*QR_000020000002+P_000102020*QR_000020000010+P_000102120*QR_000020000011+P_000102220*QR_000020000012+P_000202020*QR_000020000020+P_000202120*QR_000020000021+P_000202220*QR_000020000022);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_000002020*QR_010000010000+P_000002120*QR_010000010001+P_000002220*QR_010000010002+P_000102020*QR_010000010010+P_000102120*QR_010000010011+P_000102220*QR_010000010012+P_000202020*QR_010000010020+P_000202120*QR_010000010021+P_000202220*QR_010000010022);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000002020*QR_000010010000+P_000002120*QR_000010010001+P_000002220*QR_000010010002+P_000102020*QR_000010010010+P_000102120*QR_000010010011+P_000102220*QR_000010010012+P_000202020*QR_000010010020+P_000202120*QR_000010010021+P_000202220*QR_000010010022);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000002020*QR_000000020000+P_000002120*QR_000000020001+P_000002220*QR_000000020002+P_000102020*QR_000000020010+P_000102120*QR_000000020011+P_000102220*QR_000000020012+P_000202020*QR_000000020020+P_000202120*QR_000000020021+P_000202220*QR_000000020022);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(P_001000021*QR_020000000000+P_001000121*QR_020000000001+P_001000221*QR_020000000002+P_001000321*QR_020000000003+P_101000021*QR_020000000100+P_101000121*QR_020000000101+P_101000221*QR_020000000102+P_101000321*QR_020000000103);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(P_001000021*QR_010010000000+P_001000121*QR_010010000001+P_001000221*QR_010010000002+P_001000321*QR_010010000003+P_101000021*QR_010010000100+P_101000121*QR_010010000101+P_101000221*QR_010010000102+P_101000321*QR_010010000103);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(P_001000021*QR_000020000000+P_001000121*QR_000020000001+P_001000221*QR_000020000002+P_001000321*QR_000020000003+P_101000021*QR_000020000100+P_101000121*QR_000020000101+P_101000221*QR_000020000102+P_101000321*QR_000020000103);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(P_001000021*QR_010000010000+P_001000121*QR_010000010001+P_001000221*QR_010000010002+P_001000321*QR_010000010003+P_101000021*QR_010000010100+P_101000121*QR_010000010101+P_101000221*QR_010000010102+P_101000321*QR_010000010103);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(P_001000021*QR_000010010000+P_001000121*QR_000010010001+P_001000221*QR_000010010002+P_001000321*QR_000010010003+P_101000021*QR_000010010100+P_101000121*QR_000010010101+P_101000221*QR_000010010102+P_101000321*QR_000010010103);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(P_001000021*QR_000000020000+P_001000121*QR_000000020001+P_001000221*QR_000000020002+P_001000321*QR_000000020003+P_101000021*QR_000000020100+P_101000121*QR_000000020101+P_101000221*QR_000000020102+P_101000321*QR_000000020103);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(P_000001021*QR_020000000000+P_000001121*QR_020000000001+P_000001221*QR_020000000002+P_000001321*QR_020000000003+P_000101021*QR_020000000010+P_000101121*QR_020000000011+P_000101221*QR_020000000012+P_000101321*QR_020000000013);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(P_000001021*QR_010010000000+P_000001121*QR_010010000001+P_000001221*QR_010010000002+P_000001321*QR_010010000003+P_000101021*QR_010010000010+P_000101121*QR_010010000011+P_000101221*QR_010010000012+P_000101321*QR_010010000013);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(P_000001021*QR_000020000000+P_000001121*QR_000020000001+P_000001221*QR_000020000002+P_000001321*QR_000020000003+P_000101021*QR_000020000010+P_000101121*QR_000020000011+P_000101221*QR_000020000012+P_000101321*QR_000020000013);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(P_000001021*QR_010000010000+P_000001121*QR_010000010001+P_000001221*QR_010000010002+P_000001321*QR_010000010003+P_000101021*QR_010000010010+P_000101121*QR_010000010011+P_000101221*QR_010000010012+P_000101321*QR_010000010013);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(P_000001021*QR_000010010000+P_000001121*QR_000010010001+P_000001221*QR_000010010002+P_000001321*QR_000010010003+P_000101021*QR_000010010010+P_000101121*QR_000010010011+P_000101221*QR_000010010012+P_000101321*QR_000010010013);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(P_000001021*QR_000000020000+P_000001121*QR_000000020001+P_000001221*QR_000000020002+P_000001321*QR_000000020003+P_000101021*QR_000000020010+P_000101121*QR_000000020011+P_000101221*QR_000000020012+P_000101321*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(P_000000022*QR_020000000000+P_000000122*QR_020000000001+P_000000222*QR_020000000002+P_000000322*QR_020000000003+P_000000422*QR_020000000004);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(P_000000022*QR_010010000000+P_000000122*QR_010010000001+P_000000222*QR_010010000002+P_000000322*QR_010010000003+P_000000422*QR_010010000004);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(P_000000022*QR_000020000000+P_000000122*QR_000020000001+P_000000222*QR_000020000002+P_000000322*QR_000020000003+P_000000422*QR_000020000004);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(P_000000022*QR_010000010000+P_000000122*QR_010000010001+P_000000222*QR_010000010002+P_000000322*QR_010000010003+P_000000422*QR_010000010004);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(P_000000022*QR_000010010000+P_000000122*QR_000010010001+P_000000222*QR_000010010002+P_000000322*QR_000010010003+P_000000422*QR_000010010004);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(P_000000022*QR_000000020000+P_000000122*QR_000000020001+P_000000222*QR_000000020002+P_000000322*QR_000000020003+P_000000422*QR_000000020004);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<36;ians++){
                    ans_temp[tId_x*36+ians]+=ans_temp[(tId_x+num_thread)*36+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<36;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
	}
}
