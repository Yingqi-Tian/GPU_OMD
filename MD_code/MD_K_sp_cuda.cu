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

void MD_texture_binding_bra_sp(double * P_d,double * PA_d,double * PB_d,\
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
void MD_texture_unbind_bra_sp(){
    cudaUnbindTexture(tex_P);
    cudaUnbindTexture(tex_Zta);
    cudaUnbindTexture(tex_pp);
    cudaUnbindTexture(tex_K2_p);
    cudaUnbindTexture(tex_PA);
    cudaUnbindTexture(tex_PB);
    cudaUnbindTexture(tex_id_bra);

}

void MD_texture_binding_ket_sp(double * Q_d,double * QC_d,double * QD_d,\
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
void MD_texture_unbind_ket_sp(){
    cudaUnbindTexture(tex_Q);
    cudaUnbindTexture(tex_Eta);
    cudaUnbindTexture(tex_pq);
    cudaUnbindTexture(tex_K2_q);
    cudaUnbindTexture(tex_QC);
    cudaUnbindTexture(tex_QD);
    cudaUnbindTexture(tex_id_ket);

}
__global__ void MD_Kp_spsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

    __shared__ double ans_temp[NTHREAD*1];
    for(int i=0;i<1;i++){
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
                for(int ians=0;ians<1;ians++){
                    ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=0.0;
                }
            }
            continue;
        }
        for(unsigned int ii=primit_ket_start;ii<primit_ket_end;ii++){
            unsigned int id_ket=id_ket_in[ii];
				double QX=Q[ii*3+0];
				double QY=Q[ii*3+1];
				double QZ=Q[ii*3+2];
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[2];
	double R_200[1];
	double R_010[2];
	double R_110[1];
	double R_020[1];
	double R_001[2];
	double R_101[1];
	double R_011[1];
	double R_002[1];
	for(int i=0;i<2;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
		double Pd_101[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
	double P_001000000=Pd_001[0];
	double P_101000000=Pd_101[0];
	double P_000001000=Pd_001[1];
	double P_000101000=Pd_101[1];
	double P_000000001=Pd_001[2];
	double P_000000101=Pd_101[2];
				double PR_001000000000=P_001000000*R_000[0]+-1*P_101000000*R_100[0];
				double PR_000001000000=P_000001000*R_000[0]+-1*P_000101000*R_010[0];
				double PR_000000001000=P_000000001*R_000[0]+-1*P_000000101*R_001[0];
				double PR_001000000001=P_001000000*R_001[0]+-1*P_101000000*R_101[0];
				double PR_000001000001=P_000001000*R_001[0]+-1*P_000101000*R_011[0];
				double PR_000000001001=P_000000001*R_001[0]+-1*P_000000101*R_002[0];
				double PR_001000000010=P_001000000*R_010[0]+-1*P_101000000*R_110[0];
				double PR_000001000010=P_000001000*R_010[0]+-1*P_000101000*R_020[0];
				double PR_000000001010=P_000000001*R_010[0]+-1*P_000000101*R_011[0];
				double PR_001000000100=P_001000000*R_100[0]+-1*P_101000000*R_200[0];
				double PR_000001000100=P_000001000*R_100[0]+-1*P_000101000*R_110[0];
				double PR_000000001100=P_000000001*R_100[0]+-1*P_000000101*R_101[0];
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(Q_001000000*PR_001000000000+Q_101000000*PR_001000000100);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(Q_000001000*PR_001000000000+Q_000101000*PR_001000000010);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(Q_000000001*PR_001000000000+Q_000000101*PR_001000000001);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(Q_001000000*PR_000001000000+Q_101000000*PR_000001000100);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(Q_000001000*PR_000001000000+Q_000101000*PR_000001000010);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(Q_000000001*PR_000001000000+Q_000000101*PR_000001000001);
			ans_temp[ans_id*1+0]+=Pmtrx[6]*(Q_001000000*PR_000000001000+Q_101000000*PR_000000001100);
			ans_temp[ans_id*1+0]+=Pmtrx[7]*(Q_000001000*PR_000000001000+Q_000101000*PR_000000001010);
			ans_temp[ans_id*1+0]+=Pmtrx[8]*(Q_000000001*PR_000000001000+Q_000000101*PR_000000001001);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<1;ians++){
                    ans_temp[tId_x*1+ians]+=ans_temp[(tId_x+num_thread)*1+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<1;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_spsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

    __shared__ double ans_temp[NTHREAD*1];
    for(int i=0;i<1;i++){
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
                for(int ians=0;ians<1;ians++){
                    ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=0.0;
                }
            }
            continue;
        }
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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[2];
	double R_200[1];
	double R_010[2];
	double R_110[1];
	double R_020[1];
	double R_001[2];
	double R_101[1];
	double R_011[1];
	double R_002[1];
	for(int i=0;i<2;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<1;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
		double Pd_101[3];
		for(int i=0;i<3;i++){
			Pd_101[i]=aPin1;
			}
	double P_001000000=Pd_001[0];
	double P_101000000=Pd_101[0];
	double P_000001000=Pd_001[1];
	double P_000101000=Pd_101[1];
	double P_000000001=Pd_001[2];
	double P_000000101=Pd_101[2];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(P_001000000*QR_001000000000+P_101000000*QR_001000000100);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(P_001000000*QR_000001000000+P_101000000*QR_000001000100);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(P_001000000*QR_000000001000+P_101000000*QR_000000001100);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(P_000001000*QR_001000000000+P_000101000*QR_001000000010);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(P_000001000*QR_000001000000+P_000101000*QR_000001000010);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(P_000001000*QR_000000001000+P_000101000*QR_000000001010);
			ans_temp[ans_id*1+0]+=Pmtrx[6]*(P_000000001*QR_001000000000+P_000000101*QR_001000000001);
			ans_temp[ans_id*1+0]+=Pmtrx[7]*(P_000000001*QR_000001000000+P_000000101*QR_000001000001);
			ans_temp[ans_id*1+0]+=Pmtrx[8]*(P_000000001*QR_000000001000+P_000000101*QR_000000001001);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<1;ians++){
                    ans_temp[tId_x*1+ians]+=ans_temp[(tId_x+num_thread)*1+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<1;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_sdsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

    __shared__ double ans_temp[NTHREAD*1];
    for(int i=0;i<1;i++){
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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(Q_001000000*PR_002000000000+Q_101000000*PR_002000000100);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(Q_000001000*PR_002000000000+Q_000101000*PR_002000000010);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(Q_000000001*PR_002000000000+Q_000000101*PR_002000000001);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(Q_001000000*PR_001001000000+Q_101000000*PR_001001000100);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(Q_000001000*PR_001001000000+Q_000101000*PR_001001000010);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(Q_000000001*PR_001001000000+Q_000000101*PR_001001000001);
			ans_temp[ans_id*1+0]+=Pmtrx[6]*(Q_001000000*PR_000002000000+Q_101000000*PR_000002000100);
			ans_temp[ans_id*1+0]+=Pmtrx[7]*(Q_000001000*PR_000002000000+Q_000101000*PR_000002000010);
			ans_temp[ans_id*1+0]+=Pmtrx[8]*(Q_000000001*PR_000002000000+Q_000000101*PR_000002000001);
			ans_temp[ans_id*1+0]+=Pmtrx[9]*(Q_001000000*PR_001000001000+Q_101000000*PR_001000001100);
			ans_temp[ans_id*1+0]+=Pmtrx[10]*(Q_000001000*PR_001000001000+Q_000101000*PR_001000001010);
			ans_temp[ans_id*1+0]+=Pmtrx[11]*(Q_000000001*PR_001000001000+Q_000000101*PR_001000001001);
			ans_temp[ans_id*1+0]+=Pmtrx[12]*(Q_001000000*PR_000001001000+Q_101000000*PR_000001001100);
			ans_temp[ans_id*1+0]+=Pmtrx[13]*(Q_000001000*PR_000001001000+Q_000101000*PR_000001001010);
			ans_temp[ans_id*1+0]+=Pmtrx[14]*(Q_000000001*PR_000001001000+Q_000000101*PR_000001001001);
			ans_temp[ans_id*1+0]+=Pmtrx[15]*(Q_001000000*PR_000000002000+Q_101000000*PR_000000002100);
			ans_temp[ans_id*1+0]+=Pmtrx[16]*(Q_000001000*PR_000000002000+Q_000101000*PR_000000002010);
			ans_temp[ans_id*1+0]+=Pmtrx[17]*(Q_000000001*PR_000000002000+Q_000000101*PR_000000002001);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<1;ians++){
                    ans_temp[tId_x*1+ians]+=ans_temp[(tId_x+num_thread)*1+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<1;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_sdsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

    __shared__ double ans_temp[NTHREAD*1];
    for(int i=0;i<1;i++){
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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
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
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(P_002000000*QR_001000000000+P_102000000*QR_001000000100+P_202000000*QR_001000000200);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(P_002000000*QR_000001000000+P_102000000*QR_000001000100+P_202000000*QR_000001000200);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(P_002000000*QR_000000001000+P_102000000*QR_000000001100+P_202000000*QR_000000001200);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(P_001001000*QR_001000000000+P_001101000*QR_001000000010+P_101001000*QR_001000000100+P_101101000*QR_001000000110);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(P_001001000*QR_000001000000+P_001101000*QR_000001000010+P_101001000*QR_000001000100+P_101101000*QR_000001000110);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(P_001001000*QR_000000001000+P_001101000*QR_000000001010+P_101001000*QR_000000001100+P_101101000*QR_000000001110);
			ans_temp[ans_id*1+0]+=Pmtrx[6]*(P_000002000*QR_001000000000+P_000102000*QR_001000000010+P_000202000*QR_001000000020);
			ans_temp[ans_id*1+0]+=Pmtrx[7]*(P_000002000*QR_000001000000+P_000102000*QR_000001000010+P_000202000*QR_000001000020);
			ans_temp[ans_id*1+0]+=Pmtrx[8]*(P_000002000*QR_000000001000+P_000102000*QR_000000001010+P_000202000*QR_000000001020);
			ans_temp[ans_id*1+0]+=Pmtrx[9]*(P_001000001*QR_001000000000+P_001000101*QR_001000000001+P_101000001*QR_001000000100+P_101000101*QR_001000000101);
			ans_temp[ans_id*1+0]+=Pmtrx[10]*(P_001000001*QR_000001000000+P_001000101*QR_000001000001+P_101000001*QR_000001000100+P_101000101*QR_000001000101);
			ans_temp[ans_id*1+0]+=Pmtrx[11]*(P_001000001*QR_000000001000+P_001000101*QR_000000001001+P_101000001*QR_000000001100+P_101000101*QR_000000001101);
			ans_temp[ans_id*1+0]+=Pmtrx[12]*(P_000001001*QR_001000000000+P_000001101*QR_001000000001+P_000101001*QR_001000000010+P_000101101*QR_001000000011);
			ans_temp[ans_id*1+0]+=Pmtrx[13]*(P_000001001*QR_000001000000+P_000001101*QR_000001000001+P_000101001*QR_000001000010+P_000101101*QR_000001000011);
			ans_temp[ans_id*1+0]+=Pmtrx[14]*(P_000001001*QR_000000001000+P_000001101*QR_000000001001+P_000101001*QR_000000001010+P_000101101*QR_000000001011);
			ans_temp[ans_id*1+0]+=Pmtrx[15]*(P_000000002*QR_001000000000+P_000000102*QR_001000000001+P_000000202*QR_001000000002);
			ans_temp[ans_id*1+0]+=Pmtrx[16]*(P_000000002*QR_000001000000+P_000000102*QR_000001000001+P_000000202*QR_000001000002);
			ans_temp[ans_id*1+0]+=Pmtrx[17]*(P_000000002*QR_000000001000+P_000000102*QR_000000001001+P_000000202*QR_000000001002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<1;ians++){
                    ans_temp[tId_x*1+ians]+=ans_temp[(tId_x+num_thread)*1+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<1;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_ppsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

    __shared__ double ans_temp[NTHREAD*3];
    for(int i=0;i<3;i++){
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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
		double Pd_101[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
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
	double P_011000000=Pd_011[0];
	double P_111000000=Pd_111[0];
	double P_211000000=Pd_211[0];
	double P_010001000=Pd_010[0]*Pd_001[1];
	double P_010101000=Pd_010[0]*Pd_101[1];
	double P_110001000=Pd_110[0]*Pd_001[1];
	double P_110101000=Pd_110[0]*Pd_101[1];
	double P_010000001=Pd_010[0]*Pd_001[2];
	double P_010000101=Pd_010[0]*Pd_101[2];
	double P_110000001=Pd_110[0]*Pd_001[2];
	double P_110000101=Pd_110[0]*Pd_101[2];
	double P_001010000=Pd_001[0]*Pd_010[1];
	double P_001110000=Pd_001[0]*Pd_110[1];
	double P_101010000=Pd_101[0]*Pd_010[1];
	double P_101110000=Pd_101[0]*Pd_110[1];
	double P_000011000=Pd_011[1];
	double P_000111000=Pd_111[1];
	double P_000211000=Pd_211[1];
	double P_000010001=Pd_010[1]*Pd_001[2];
	double P_000010101=Pd_010[1]*Pd_101[2];
	double P_000110001=Pd_110[1]*Pd_001[2];
	double P_000110101=Pd_110[1]*Pd_101[2];
	double P_001000010=Pd_001[0]*Pd_010[2];
	double P_001000110=Pd_001[0]*Pd_110[2];
	double P_101000010=Pd_101[0]*Pd_010[2];
	double P_101000110=Pd_101[0]*Pd_110[2];
	double P_000001010=Pd_001[1]*Pd_010[2];
	double P_000001110=Pd_001[1]*Pd_110[2];
	double P_000101010=Pd_101[1]*Pd_010[2];
	double P_000101110=Pd_101[1]*Pd_110[2];
	double P_000000011=Pd_011[2];
	double P_000000111=Pd_111[2];
	double P_000000211=Pd_211[2];
				double PR_011000000000=P_011000000*R_000[0]+-1*P_111000000*R_100[0]+P_211000000*R_200[0];
				double PR_010001000000=P_010001000*R_000[0]+-1*P_010101000*R_010[0]+-1*P_110001000*R_100[0]+P_110101000*R_110[0];
				double PR_010000001000=P_010000001*R_000[0]+-1*P_010000101*R_001[0]+-1*P_110000001*R_100[0]+P_110000101*R_101[0];
				double PR_001010000000=P_001010000*R_000[0]+-1*P_001110000*R_010[0]+-1*P_101010000*R_100[0]+P_101110000*R_110[0];
				double PR_000011000000=P_000011000*R_000[0]+-1*P_000111000*R_010[0]+P_000211000*R_020[0];
				double PR_000010001000=P_000010001*R_000[0]+-1*P_000010101*R_001[0]+-1*P_000110001*R_010[0]+P_000110101*R_011[0];
				double PR_001000010000=P_001000010*R_000[0]+-1*P_001000110*R_001[0]+-1*P_101000010*R_100[0]+P_101000110*R_101[0];
				double PR_000001010000=P_000001010*R_000[0]+-1*P_000001110*R_001[0]+-1*P_000101010*R_010[0]+P_000101110*R_011[0];
				double PR_000000011000=P_000000011*R_000[0]+-1*P_000000111*R_001[0]+P_000000211*R_002[0];
				double PR_011000000001=P_011000000*R_001[0]+-1*P_111000000*R_101[0]+P_211000000*R_201[0];
				double PR_010001000001=P_010001000*R_001[0]+-1*P_010101000*R_011[0]+-1*P_110001000*R_101[0]+P_110101000*R_111[0];
				double PR_010000001001=P_010000001*R_001[0]+-1*P_010000101*R_002[0]+-1*P_110000001*R_101[0]+P_110000101*R_102[0];
				double PR_001010000001=P_001010000*R_001[0]+-1*P_001110000*R_011[0]+-1*P_101010000*R_101[0]+P_101110000*R_111[0];
				double PR_000011000001=P_000011000*R_001[0]+-1*P_000111000*R_011[0]+P_000211000*R_021[0];
				double PR_000010001001=P_000010001*R_001[0]+-1*P_000010101*R_002[0]+-1*P_000110001*R_011[0]+P_000110101*R_012[0];
				double PR_001000010001=P_001000010*R_001[0]+-1*P_001000110*R_002[0]+-1*P_101000010*R_101[0]+P_101000110*R_102[0];
				double PR_000001010001=P_000001010*R_001[0]+-1*P_000001110*R_002[0]+-1*P_000101010*R_011[0]+P_000101110*R_012[0];
				double PR_000000011001=P_000000011*R_001[0]+-1*P_000000111*R_002[0]+P_000000211*R_003[0];
				double PR_011000000010=P_011000000*R_010[0]+-1*P_111000000*R_110[0]+P_211000000*R_210[0];
				double PR_010001000010=P_010001000*R_010[0]+-1*P_010101000*R_020[0]+-1*P_110001000*R_110[0]+P_110101000*R_120[0];
				double PR_010000001010=P_010000001*R_010[0]+-1*P_010000101*R_011[0]+-1*P_110000001*R_110[0]+P_110000101*R_111[0];
				double PR_001010000010=P_001010000*R_010[0]+-1*P_001110000*R_020[0]+-1*P_101010000*R_110[0]+P_101110000*R_120[0];
				double PR_000011000010=P_000011000*R_010[0]+-1*P_000111000*R_020[0]+P_000211000*R_030[0];
				double PR_000010001010=P_000010001*R_010[0]+-1*P_000010101*R_011[0]+-1*P_000110001*R_020[0]+P_000110101*R_021[0];
				double PR_001000010010=P_001000010*R_010[0]+-1*P_001000110*R_011[0]+-1*P_101000010*R_110[0]+P_101000110*R_111[0];
				double PR_000001010010=P_000001010*R_010[0]+-1*P_000001110*R_011[0]+-1*P_000101010*R_020[0]+P_000101110*R_021[0];
				double PR_000000011010=P_000000011*R_010[0]+-1*P_000000111*R_011[0]+P_000000211*R_012[0];
				double PR_011000000100=P_011000000*R_100[0]+-1*P_111000000*R_200[0]+P_211000000*R_300[0];
				double PR_010001000100=P_010001000*R_100[0]+-1*P_010101000*R_110[0]+-1*P_110001000*R_200[0]+P_110101000*R_210[0];
				double PR_010000001100=P_010000001*R_100[0]+-1*P_010000101*R_101[0]+-1*P_110000001*R_200[0]+P_110000101*R_201[0];
				double PR_001010000100=P_001010000*R_100[0]+-1*P_001110000*R_110[0]+-1*P_101010000*R_200[0]+P_101110000*R_210[0];
				double PR_000011000100=P_000011000*R_100[0]+-1*P_000111000*R_110[0]+P_000211000*R_120[0];
				double PR_000010001100=P_000010001*R_100[0]+-1*P_000010101*R_101[0]+-1*P_000110001*R_110[0]+P_000110101*R_111[0];
				double PR_001000010100=P_001000010*R_100[0]+-1*P_001000110*R_101[0]+-1*P_101000010*R_200[0]+P_101000110*R_201[0];
				double PR_000001010100=P_000001010*R_100[0]+-1*P_000001110*R_101[0]+-1*P_000101010*R_110[0]+P_000101110*R_111[0];
				double PR_000000011100=P_000000011*R_100[0]+-1*P_000000111*R_101[0]+P_000000211*R_102[0];
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(Q_001000000*PR_011000000000+Q_101000000*PR_011000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(Q_000001000*PR_011000000000+Q_000101000*PR_011000000010);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(Q_000000001*PR_011000000000+Q_000000101*PR_011000000001);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(Q_001000000*PR_010001000000+Q_101000000*PR_010001000100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(Q_000001000*PR_010001000000+Q_000101000*PR_010001000010);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(Q_000000001*PR_010001000000+Q_000000101*PR_010001000001);
			ans_temp[ans_id*3+0]+=Pmtrx[6]*(Q_001000000*PR_010000001000+Q_101000000*PR_010000001100);
			ans_temp[ans_id*3+0]+=Pmtrx[7]*(Q_000001000*PR_010000001000+Q_000101000*PR_010000001010);
			ans_temp[ans_id*3+0]+=Pmtrx[8]*(Q_000000001*PR_010000001000+Q_000000101*PR_010000001001);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(Q_001000000*PR_001010000000+Q_101000000*PR_001010000100);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(Q_000001000*PR_001010000000+Q_000101000*PR_001010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(Q_000000001*PR_001010000000+Q_000000101*PR_001010000001);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(Q_001000000*PR_000011000000+Q_101000000*PR_000011000100);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(Q_000001000*PR_000011000000+Q_000101000*PR_000011000010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(Q_000000001*PR_000011000000+Q_000000101*PR_000011000001);
			ans_temp[ans_id*3+1]+=Pmtrx[6]*(Q_001000000*PR_000010001000+Q_101000000*PR_000010001100);
			ans_temp[ans_id*3+1]+=Pmtrx[7]*(Q_000001000*PR_000010001000+Q_000101000*PR_000010001010);
			ans_temp[ans_id*3+1]+=Pmtrx[8]*(Q_000000001*PR_000010001000+Q_000000101*PR_000010001001);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(Q_001000000*PR_001000010000+Q_101000000*PR_001000010100);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(Q_000001000*PR_001000010000+Q_000101000*PR_001000010010);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(Q_000000001*PR_001000010000+Q_000000101*PR_001000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(Q_001000000*PR_000001010000+Q_101000000*PR_000001010100);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(Q_000001000*PR_000001010000+Q_000101000*PR_000001010010);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(Q_000000001*PR_000001010000+Q_000000101*PR_000001010001);
			ans_temp[ans_id*3+2]+=Pmtrx[6]*(Q_001000000*PR_000000011000+Q_101000000*PR_000000011100);
			ans_temp[ans_id*3+2]+=Pmtrx[7]*(Q_000001000*PR_000000011000+Q_000101000*PR_000000011010);
			ans_temp[ans_id*3+2]+=Pmtrx[8]*(Q_000000001*PR_000000011000+Q_000000101*PR_000000011001);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<3;ians++){
                    ans_temp[tId_x*3+ians]+=ans_temp[(tId_x+num_thread)*3+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<3;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_ppsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

    __shared__ double ans_temp[NTHREAD*3];
    for(int i=0;i<3;i++){
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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
		double Pd_101[3];
		double Pd_110[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_211[3];
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
	double P_011000000=Pd_011[0];
	double P_111000000=Pd_111[0];
	double P_211000000=Pd_211[0];
	double P_010001000=Pd_010[0]*Pd_001[1];
	double P_010101000=Pd_010[0]*Pd_101[1];
	double P_110001000=Pd_110[0]*Pd_001[1];
	double P_110101000=Pd_110[0]*Pd_101[1];
	double P_010000001=Pd_010[0]*Pd_001[2];
	double P_010000101=Pd_010[0]*Pd_101[2];
	double P_110000001=Pd_110[0]*Pd_001[2];
	double P_110000101=Pd_110[0]*Pd_101[2];
	double P_001010000=Pd_001[0]*Pd_010[1];
	double P_001110000=Pd_001[0]*Pd_110[1];
	double P_101010000=Pd_101[0]*Pd_010[1];
	double P_101110000=Pd_101[0]*Pd_110[1];
	double P_000011000=Pd_011[1];
	double P_000111000=Pd_111[1];
	double P_000211000=Pd_211[1];
	double P_000010001=Pd_010[1]*Pd_001[2];
	double P_000010101=Pd_010[1]*Pd_101[2];
	double P_000110001=Pd_110[1]*Pd_001[2];
	double P_000110101=Pd_110[1]*Pd_101[2];
	double P_001000010=Pd_001[0]*Pd_010[2];
	double P_001000110=Pd_001[0]*Pd_110[2];
	double P_101000010=Pd_101[0]*Pd_010[2];
	double P_101000110=Pd_101[0]*Pd_110[2];
	double P_000001010=Pd_001[1]*Pd_010[2];
	double P_000001110=Pd_001[1]*Pd_110[2];
	double P_000101010=Pd_101[1]*Pd_010[2];
	double P_000101110=Pd_101[1]*Pd_110[2];
	double P_000000011=Pd_011[2];
	double P_000000111=Pd_111[2];
	double P_000000211=Pd_211[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_011000000*QR_001000000000+P_111000000*QR_001000000100+P_211000000*QR_001000000200);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_011000000*QR_000001000000+P_111000000*QR_000001000100+P_211000000*QR_000001000200);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_011000000*QR_000000001000+P_111000000*QR_000000001100+P_211000000*QR_000000001200);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_010001000*QR_001000000000+P_010101000*QR_001000000010+P_110001000*QR_001000000100+P_110101000*QR_001000000110);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_010001000*QR_000001000000+P_010101000*QR_000001000010+P_110001000*QR_000001000100+P_110101000*QR_000001000110);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_010001000*QR_000000001000+P_010101000*QR_000000001010+P_110001000*QR_000000001100+P_110101000*QR_000000001110);
			ans_temp[ans_id*3+0]+=Pmtrx[6]*(P_010000001*QR_001000000000+P_010000101*QR_001000000001+P_110000001*QR_001000000100+P_110000101*QR_001000000101);
			ans_temp[ans_id*3+0]+=Pmtrx[7]*(P_010000001*QR_000001000000+P_010000101*QR_000001000001+P_110000001*QR_000001000100+P_110000101*QR_000001000101);
			ans_temp[ans_id*3+0]+=Pmtrx[8]*(P_010000001*QR_000000001000+P_010000101*QR_000000001001+P_110000001*QR_000000001100+P_110000101*QR_000000001101);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_001010000*QR_001000000000+P_001110000*QR_001000000010+P_101010000*QR_001000000100+P_101110000*QR_001000000110);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_001010000*QR_000001000000+P_001110000*QR_000001000010+P_101010000*QR_000001000100+P_101110000*QR_000001000110);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_001010000*QR_000000001000+P_001110000*QR_000000001010+P_101010000*QR_000000001100+P_101110000*QR_000000001110);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_000011000*QR_001000000000+P_000111000*QR_001000000010+P_000211000*QR_001000000020);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_000011000*QR_000001000000+P_000111000*QR_000001000010+P_000211000*QR_000001000020);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_000011000*QR_000000001000+P_000111000*QR_000000001010+P_000211000*QR_000000001020);
			ans_temp[ans_id*3+1]+=Pmtrx[6]*(P_000010001*QR_001000000000+P_000010101*QR_001000000001+P_000110001*QR_001000000010+P_000110101*QR_001000000011);
			ans_temp[ans_id*3+1]+=Pmtrx[7]*(P_000010001*QR_000001000000+P_000010101*QR_000001000001+P_000110001*QR_000001000010+P_000110101*QR_000001000011);
			ans_temp[ans_id*3+1]+=Pmtrx[8]*(P_000010001*QR_000000001000+P_000010101*QR_000000001001+P_000110001*QR_000000001010+P_000110101*QR_000000001011);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_001000010*QR_001000000000+P_001000110*QR_001000000001+P_101000010*QR_001000000100+P_101000110*QR_001000000101);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_001000010*QR_000001000000+P_001000110*QR_000001000001+P_101000010*QR_000001000100+P_101000110*QR_000001000101);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_001000010*QR_000000001000+P_001000110*QR_000000001001+P_101000010*QR_000000001100+P_101000110*QR_000000001101);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_000001010*QR_001000000000+P_000001110*QR_001000000001+P_000101010*QR_001000000010+P_000101110*QR_001000000011);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_000001010*QR_000001000000+P_000001110*QR_000001000001+P_000101010*QR_000001000010+P_000101110*QR_000001000011);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_000001010*QR_000000001000+P_000001110*QR_000000001001+P_000101010*QR_000000001010+P_000101110*QR_000000001011);
			ans_temp[ans_id*3+2]+=Pmtrx[6]*(P_000000011*QR_001000000000+P_000000111*QR_001000000001+P_000000211*QR_001000000002);
			ans_temp[ans_id*3+2]+=Pmtrx[7]*(P_000000011*QR_000001000000+P_000000111*QR_000001000001+P_000000211*QR_000001000002);
			ans_temp[ans_id*3+2]+=Pmtrx[8]*(P_000000011*QR_000000001000+P_000000111*QR_000000001001+P_000000211*QR_000000001002);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<3;ians++){
                    ans_temp[tId_x*3+ians]+=ans_temp[(tId_x+num_thread)*3+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<3;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_pdsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

    __shared__ double ans_temp[NTHREAD*3];
    for(int i=0;i<3;i++){
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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(Q_001000000*PR_012000000000+Q_101000000*PR_012000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(Q_000001000*PR_012000000000+Q_000101000*PR_012000000010);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(Q_000000001*PR_012000000000+Q_000000101*PR_012000000001);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(Q_001000000*PR_011001000000+Q_101000000*PR_011001000100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(Q_000001000*PR_011001000000+Q_000101000*PR_011001000010);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(Q_000000001*PR_011001000000+Q_000000101*PR_011001000001);
			ans_temp[ans_id*3+0]+=Pmtrx[6]*(Q_001000000*PR_010002000000+Q_101000000*PR_010002000100);
			ans_temp[ans_id*3+0]+=Pmtrx[7]*(Q_000001000*PR_010002000000+Q_000101000*PR_010002000010);
			ans_temp[ans_id*3+0]+=Pmtrx[8]*(Q_000000001*PR_010002000000+Q_000000101*PR_010002000001);
			ans_temp[ans_id*3+0]+=Pmtrx[9]*(Q_001000000*PR_011000001000+Q_101000000*PR_011000001100);
			ans_temp[ans_id*3+0]+=Pmtrx[10]*(Q_000001000*PR_011000001000+Q_000101000*PR_011000001010);
			ans_temp[ans_id*3+0]+=Pmtrx[11]*(Q_000000001*PR_011000001000+Q_000000101*PR_011000001001);
			ans_temp[ans_id*3+0]+=Pmtrx[12]*(Q_001000000*PR_010001001000+Q_101000000*PR_010001001100);
			ans_temp[ans_id*3+0]+=Pmtrx[13]*(Q_000001000*PR_010001001000+Q_000101000*PR_010001001010);
			ans_temp[ans_id*3+0]+=Pmtrx[14]*(Q_000000001*PR_010001001000+Q_000000101*PR_010001001001);
			ans_temp[ans_id*3+0]+=Pmtrx[15]*(Q_001000000*PR_010000002000+Q_101000000*PR_010000002100);
			ans_temp[ans_id*3+0]+=Pmtrx[16]*(Q_000001000*PR_010000002000+Q_000101000*PR_010000002010);
			ans_temp[ans_id*3+0]+=Pmtrx[17]*(Q_000000001*PR_010000002000+Q_000000101*PR_010000002001);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(Q_001000000*PR_002010000000+Q_101000000*PR_002010000100);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(Q_000001000*PR_002010000000+Q_000101000*PR_002010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(Q_000000001*PR_002010000000+Q_000000101*PR_002010000001);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(Q_001000000*PR_001011000000+Q_101000000*PR_001011000100);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(Q_000001000*PR_001011000000+Q_000101000*PR_001011000010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(Q_000000001*PR_001011000000+Q_000000101*PR_001011000001);
			ans_temp[ans_id*3+1]+=Pmtrx[6]*(Q_001000000*PR_000012000000+Q_101000000*PR_000012000100);
			ans_temp[ans_id*3+1]+=Pmtrx[7]*(Q_000001000*PR_000012000000+Q_000101000*PR_000012000010);
			ans_temp[ans_id*3+1]+=Pmtrx[8]*(Q_000000001*PR_000012000000+Q_000000101*PR_000012000001);
			ans_temp[ans_id*3+1]+=Pmtrx[9]*(Q_001000000*PR_001010001000+Q_101000000*PR_001010001100);
			ans_temp[ans_id*3+1]+=Pmtrx[10]*(Q_000001000*PR_001010001000+Q_000101000*PR_001010001010);
			ans_temp[ans_id*3+1]+=Pmtrx[11]*(Q_000000001*PR_001010001000+Q_000000101*PR_001010001001);
			ans_temp[ans_id*3+1]+=Pmtrx[12]*(Q_001000000*PR_000011001000+Q_101000000*PR_000011001100);
			ans_temp[ans_id*3+1]+=Pmtrx[13]*(Q_000001000*PR_000011001000+Q_000101000*PR_000011001010);
			ans_temp[ans_id*3+1]+=Pmtrx[14]*(Q_000000001*PR_000011001000+Q_000000101*PR_000011001001);
			ans_temp[ans_id*3+1]+=Pmtrx[15]*(Q_001000000*PR_000010002000+Q_101000000*PR_000010002100);
			ans_temp[ans_id*3+1]+=Pmtrx[16]*(Q_000001000*PR_000010002000+Q_000101000*PR_000010002010);
			ans_temp[ans_id*3+1]+=Pmtrx[17]*(Q_000000001*PR_000010002000+Q_000000101*PR_000010002001);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(Q_001000000*PR_002000010000+Q_101000000*PR_002000010100);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(Q_000001000*PR_002000010000+Q_000101000*PR_002000010010);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(Q_000000001*PR_002000010000+Q_000000101*PR_002000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(Q_001000000*PR_001001010000+Q_101000000*PR_001001010100);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(Q_000001000*PR_001001010000+Q_000101000*PR_001001010010);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(Q_000000001*PR_001001010000+Q_000000101*PR_001001010001);
			ans_temp[ans_id*3+2]+=Pmtrx[6]*(Q_001000000*PR_000002010000+Q_101000000*PR_000002010100);
			ans_temp[ans_id*3+2]+=Pmtrx[7]*(Q_000001000*PR_000002010000+Q_000101000*PR_000002010010);
			ans_temp[ans_id*3+2]+=Pmtrx[8]*(Q_000000001*PR_000002010000+Q_000000101*PR_000002010001);
			ans_temp[ans_id*3+2]+=Pmtrx[9]*(Q_001000000*PR_001000011000+Q_101000000*PR_001000011100);
			ans_temp[ans_id*3+2]+=Pmtrx[10]*(Q_000001000*PR_001000011000+Q_000101000*PR_001000011010);
			ans_temp[ans_id*3+2]+=Pmtrx[11]*(Q_000000001*PR_001000011000+Q_000000101*PR_001000011001);
			ans_temp[ans_id*3+2]+=Pmtrx[12]*(Q_001000000*PR_000001011000+Q_101000000*PR_000001011100);
			ans_temp[ans_id*3+2]+=Pmtrx[13]*(Q_000001000*PR_000001011000+Q_000101000*PR_000001011010);
			ans_temp[ans_id*3+2]+=Pmtrx[14]*(Q_000000001*PR_000001011000+Q_000000101*PR_000001011001);
			ans_temp[ans_id*3+2]+=Pmtrx[15]*(Q_001000000*PR_000000012000+Q_101000000*PR_000000012100);
			ans_temp[ans_id*3+2]+=Pmtrx[16]*(Q_000001000*PR_000000012000+Q_000101000*PR_000000012010);
			ans_temp[ans_id*3+2]+=Pmtrx[17]*(Q_000000001*PR_000000012000+Q_000000101*PR_000000012001);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<3;ians++){
                    ans_temp[tId_x*3+ians]+=ans_temp[(tId_x+num_thread)*3+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<3;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
	}
}
__global__ void MD_Kq_pdsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

    __shared__ double ans_temp[NTHREAD*3];
    for(int i=0;i<3;i++){
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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
				double QR_001000000003=Q_001000000*R_003[0]+-1*Q_101000000*R_103[0];
				double QR_000001000003=Q_000001000*R_003[0]+-1*Q_000101000*R_013[0];
				double QR_000000001003=Q_000000001*R_003[0]+-1*Q_000000101*R_004[0];
				double QR_001000000012=Q_001000000*R_012[0]+-1*Q_101000000*R_112[0];
				double QR_000001000012=Q_000001000*R_012[0]+-1*Q_000101000*R_022[0];
				double QR_000000001012=Q_000000001*R_012[0]+-1*Q_000000101*R_013[0];
				double QR_001000000021=Q_001000000*R_021[0]+-1*Q_101000000*R_121[0];
				double QR_000001000021=Q_000001000*R_021[0]+-1*Q_000101000*R_031[0];
				double QR_000000001021=Q_000000001*R_021[0]+-1*Q_000000101*R_022[0];
				double QR_001000000030=Q_001000000*R_030[0]+-1*Q_101000000*R_130[0];
				double QR_000001000030=Q_000001000*R_030[0]+-1*Q_000101000*R_040[0];
				double QR_000000001030=Q_000000001*R_030[0]+-1*Q_000000101*R_031[0];
				double QR_001000000102=Q_001000000*R_102[0]+-1*Q_101000000*R_202[0];
				double QR_000001000102=Q_000001000*R_102[0]+-1*Q_000101000*R_112[0];
				double QR_000000001102=Q_000000001*R_102[0]+-1*Q_000000101*R_103[0];
				double QR_001000000111=Q_001000000*R_111[0]+-1*Q_101000000*R_211[0];
				double QR_000001000111=Q_000001000*R_111[0]+-1*Q_000101000*R_121[0];
				double QR_000000001111=Q_000000001*R_111[0]+-1*Q_000000101*R_112[0];
				double QR_001000000120=Q_001000000*R_120[0]+-1*Q_101000000*R_220[0];
				double QR_000001000120=Q_000001000*R_120[0]+-1*Q_000101000*R_130[0];
				double QR_000000001120=Q_000000001*R_120[0]+-1*Q_000000101*R_121[0];
				double QR_001000000201=Q_001000000*R_201[0]+-1*Q_101000000*R_301[0];
				double QR_000001000201=Q_000001000*R_201[0]+-1*Q_000101000*R_211[0];
				double QR_000000001201=Q_000000001*R_201[0]+-1*Q_000000101*R_202[0];
				double QR_001000000210=Q_001000000*R_210[0]+-1*Q_101000000*R_310[0];
				double QR_000001000210=Q_000001000*R_210[0]+-1*Q_000101000*R_220[0];
				double QR_000000001210=Q_000000001*R_210[0]+-1*Q_000000101*R_211[0];
				double QR_001000000300=Q_001000000*R_300[0]+-1*Q_101000000*R_400[0];
				double QR_000001000300=Q_000001000*R_300[0]+-1*Q_000101000*R_310[0];
				double QR_000000001300=Q_000000001*R_300[0]+-1*Q_000000101*R_301[0];
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
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_012000000*QR_001000000000+P_112000000*QR_001000000100+P_212000000*QR_001000000200+P_312000000*QR_001000000300);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_012000000*QR_000001000000+P_112000000*QR_000001000100+P_212000000*QR_000001000200+P_312000000*QR_000001000300);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_012000000*QR_000000001000+P_112000000*QR_000000001100+P_212000000*QR_000000001200+P_312000000*QR_000000001300);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_011001000*QR_001000000000+P_011101000*QR_001000000010+P_111001000*QR_001000000100+P_111101000*QR_001000000110+P_211001000*QR_001000000200+P_211101000*QR_001000000210);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_011001000*QR_000001000000+P_011101000*QR_000001000010+P_111001000*QR_000001000100+P_111101000*QR_000001000110+P_211001000*QR_000001000200+P_211101000*QR_000001000210);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_011001000*QR_000000001000+P_011101000*QR_000000001010+P_111001000*QR_000000001100+P_111101000*QR_000000001110+P_211001000*QR_000000001200+P_211101000*QR_000000001210);
			ans_temp[ans_id*3+0]+=Pmtrx[6]*(P_010002000*QR_001000000000+P_010102000*QR_001000000010+P_010202000*QR_001000000020+P_110002000*QR_001000000100+P_110102000*QR_001000000110+P_110202000*QR_001000000120);
			ans_temp[ans_id*3+0]+=Pmtrx[7]*(P_010002000*QR_000001000000+P_010102000*QR_000001000010+P_010202000*QR_000001000020+P_110002000*QR_000001000100+P_110102000*QR_000001000110+P_110202000*QR_000001000120);
			ans_temp[ans_id*3+0]+=Pmtrx[8]*(P_010002000*QR_000000001000+P_010102000*QR_000000001010+P_010202000*QR_000000001020+P_110002000*QR_000000001100+P_110102000*QR_000000001110+P_110202000*QR_000000001120);
			ans_temp[ans_id*3+0]+=Pmtrx[9]*(P_011000001*QR_001000000000+P_011000101*QR_001000000001+P_111000001*QR_001000000100+P_111000101*QR_001000000101+P_211000001*QR_001000000200+P_211000101*QR_001000000201);
			ans_temp[ans_id*3+0]+=Pmtrx[10]*(P_011000001*QR_000001000000+P_011000101*QR_000001000001+P_111000001*QR_000001000100+P_111000101*QR_000001000101+P_211000001*QR_000001000200+P_211000101*QR_000001000201);
			ans_temp[ans_id*3+0]+=Pmtrx[11]*(P_011000001*QR_000000001000+P_011000101*QR_000000001001+P_111000001*QR_000000001100+P_111000101*QR_000000001101+P_211000001*QR_000000001200+P_211000101*QR_000000001201);
			ans_temp[ans_id*3+0]+=Pmtrx[12]*(P_010001001*QR_001000000000+P_010001101*QR_001000000001+P_010101001*QR_001000000010+P_010101101*QR_001000000011+P_110001001*QR_001000000100+P_110001101*QR_001000000101+P_110101001*QR_001000000110+P_110101101*QR_001000000111);
			ans_temp[ans_id*3+0]+=Pmtrx[13]*(P_010001001*QR_000001000000+P_010001101*QR_000001000001+P_010101001*QR_000001000010+P_010101101*QR_000001000011+P_110001001*QR_000001000100+P_110001101*QR_000001000101+P_110101001*QR_000001000110+P_110101101*QR_000001000111);
			ans_temp[ans_id*3+0]+=Pmtrx[14]*(P_010001001*QR_000000001000+P_010001101*QR_000000001001+P_010101001*QR_000000001010+P_010101101*QR_000000001011+P_110001001*QR_000000001100+P_110001101*QR_000000001101+P_110101001*QR_000000001110+P_110101101*QR_000000001111);
			ans_temp[ans_id*3+0]+=Pmtrx[15]*(P_010000002*QR_001000000000+P_010000102*QR_001000000001+P_010000202*QR_001000000002+P_110000002*QR_001000000100+P_110000102*QR_001000000101+P_110000202*QR_001000000102);
			ans_temp[ans_id*3+0]+=Pmtrx[16]*(P_010000002*QR_000001000000+P_010000102*QR_000001000001+P_010000202*QR_000001000002+P_110000002*QR_000001000100+P_110000102*QR_000001000101+P_110000202*QR_000001000102);
			ans_temp[ans_id*3+0]+=Pmtrx[17]*(P_010000002*QR_000000001000+P_010000102*QR_000000001001+P_010000202*QR_000000001002+P_110000002*QR_000000001100+P_110000102*QR_000000001101+P_110000202*QR_000000001102);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_002010000*QR_001000000000+P_002110000*QR_001000000010+P_102010000*QR_001000000100+P_102110000*QR_001000000110+P_202010000*QR_001000000200+P_202110000*QR_001000000210);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_002010000*QR_000001000000+P_002110000*QR_000001000010+P_102010000*QR_000001000100+P_102110000*QR_000001000110+P_202010000*QR_000001000200+P_202110000*QR_000001000210);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_002010000*QR_000000001000+P_002110000*QR_000000001010+P_102010000*QR_000000001100+P_102110000*QR_000000001110+P_202010000*QR_000000001200+P_202110000*QR_000000001210);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_001011000*QR_001000000000+P_001111000*QR_001000000010+P_001211000*QR_001000000020+P_101011000*QR_001000000100+P_101111000*QR_001000000110+P_101211000*QR_001000000120);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_001011000*QR_000001000000+P_001111000*QR_000001000010+P_001211000*QR_000001000020+P_101011000*QR_000001000100+P_101111000*QR_000001000110+P_101211000*QR_000001000120);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_001011000*QR_000000001000+P_001111000*QR_000000001010+P_001211000*QR_000000001020+P_101011000*QR_000000001100+P_101111000*QR_000000001110+P_101211000*QR_000000001120);
			ans_temp[ans_id*3+1]+=Pmtrx[6]*(P_000012000*QR_001000000000+P_000112000*QR_001000000010+P_000212000*QR_001000000020+P_000312000*QR_001000000030);
			ans_temp[ans_id*3+1]+=Pmtrx[7]*(P_000012000*QR_000001000000+P_000112000*QR_000001000010+P_000212000*QR_000001000020+P_000312000*QR_000001000030);
			ans_temp[ans_id*3+1]+=Pmtrx[8]*(P_000012000*QR_000000001000+P_000112000*QR_000000001010+P_000212000*QR_000000001020+P_000312000*QR_000000001030);
			ans_temp[ans_id*3+1]+=Pmtrx[9]*(P_001010001*QR_001000000000+P_001010101*QR_001000000001+P_001110001*QR_001000000010+P_001110101*QR_001000000011+P_101010001*QR_001000000100+P_101010101*QR_001000000101+P_101110001*QR_001000000110+P_101110101*QR_001000000111);
			ans_temp[ans_id*3+1]+=Pmtrx[10]*(P_001010001*QR_000001000000+P_001010101*QR_000001000001+P_001110001*QR_000001000010+P_001110101*QR_000001000011+P_101010001*QR_000001000100+P_101010101*QR_000001000101+P_101110001*QR_000001000110+P_101110101*QR_000001000111);
			ans_temp[ans_id*3+1]+=Pmtrx[11]*(P_001010001*QR_000000001000+P_001010101*QR_000000001001+P_001110001*QR_000000001010+P_001110101*QR_000000001011+P_101010001*QR_000000001100+P_101010101*QR_000000001101+P_101110001*QR_000000001110+P_101110101*QR_000000001111);
			ans_temp[ans_id*3+1]+=Pmtrx[12]*(P_000011001*QR_001000000000+P_000011101*QR_001000000001+P_000111001*QR_001000000010+P_000111101*QR_001000000011+P_000211001*QR_001000000020+P_000211101*QR_001000000021);
			ans_temp[ans_id*3+1]+=Pmtrx[13]*(P_000011001*QR_000001000000+P_000011101*QR_000001000001+P_000111001*QR_000001000010+P_000111101*QR_000001000011+P_000211001*QR_000001000020+P_000211101*QR_000001000021);
			ans_temp[ans_id*3+1]+=Pmtrx[14]*(P_000011001*QR_000000001000+P_000011101*QR_000000001001+P_000111001*QR_000000001010+P_000111101*QR_000000001011+P_000211001*QR_000000001020+P_000211101*QR_000000001021);
			ans_temp[ans_id*3+1]+=Pmtrx[15]*(P_000010002*QR_001000000000+P_000010102*QR_001000000001+P_000010202*QR_001000000002+P_000110002*QR_001000000010+P_000110102*QR_001000000011+P_000110202*QR_001000000012);
			ans_temp[ans_id*3+1]+=Pmtrx[16]*(P_000010002*QR_000001000000+P_000010102*QR_000001000001+P_000010202*QR_000001000002+P_000110002*QR_000001000010+P_000110102*QR_000001000011+P_000110202*QR_000001000012);
			ans_temp[ans_id*3+1]+=Pmtrx[17]*(P_000010002*QR_000000001000+P_000010102*QR_000000001001+P_000010202*QR_000000001002+P_000110002*QR_000000001010+P_000110102*QR_000000001011+P_000110202*QR_000000001012);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_002000010*QR_001000000000+P_002000110*QR_001000000001+P_102000010*QR_001000000100+P_102000110*QR_001000000101+P_202000010*QR_001000000200+P_202000110*QR_001000000201);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_002000010*QR_000001000000+P_002000110*QR_000001000001+P_102000010*QR_000001000100+P_102000110*QR_000001000101+P_202000010*QR_000001000200+P_202000110*QR_000001000201);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_002000010*QR_000000001000+P_002000110*QR_000000001001+P_102000010*QR_000000001100+P_102000110*QR_000000001101+P_202000010*QR_000000001200+P_202000110*QR_000000001201);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_001001010*QR_001000000000+P_001001110*QR_001000000001+P_001101010*QR_001000000010+P_001101110*QR_001000000011+P_101001010*QR_001000000100+P_101001110*QR_001000000101+P_101101010*QR_001000000110+P_101101110*QR_001000000111);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_001001010*QR_000001000000+P_001001110*QR_000001000001+P_001101010*QR_000001000010+P_001101110*QR_000001000011+P_101001010*QR_000001000100+P_101001110*QR_000001000101+P_101101010*QR_000001000110+P_101101110*QR_000001000111);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_001001010*QR_000000001000+P_001001110*QR_000000001001+P_001101010*QR_000000001010+P_001101110*QR_000000001011+P_101001010*QR_000000001100+P_101001110*QR_000000001101+P_101101010*QR_000000001110+P_101101110*QR_000000001111);
			ans_temp[ans_id*3+2]+=Pmtrx[6]*(P_000002010*QR_001000000000+P_000002110*QR_001000000001+P_000102010*QR_001000000010+P_000102110*QR_001000000011+P_000202010*QR_001000000020+P_000202110*QR_001000000021);
			ans_temp[ans_id*3+2]+=Pmtrx[7]*(P_000002010*QR_000001000000+P_000002110*QR_000001000001+P_000102010*QR_000001000010+P_000102110*QR_000001000011+P_000202010*QR_000001000020+P_000202110*QR_000001000021);
			ans_temp[ans_id*3+2]+=Pmtrx[8]*(P_000002010*QR_000000001000+P_000002110*QR_000000001001+P_000102010*QR_000000001010+P_000102110*QR_000000001011+P_000202010*QR_000000001020+P_000202110*QR_000000001021);
			ans_temp[ans_id*3+2]+=Pmtrx[9]*(P_001000011*QR_001000000000+P_001000111*QR_001000000001+P_001000211*QR_001000000002+P_101000011*QR_001000000100+P_101000111*QR_001000000101+P_101000211*QR_001000000102);
			ans_temp[ans_id*3+2]+=Pmtrx[10]*(P_001000011*QR_000001000000+P_001000111*QR_000001000001+P_001000211*QR_000001000002+P_101000011*QR_000001000100+P_101000111*QR_000001000101+P_101000211*QR_000001000102);
			ans_temp[ans_id*3+2]+=Pmtrx[11]*(P_001000011*QR_000000001000+P_001000111*QR_000000001001+P_001000211*QR_000000001002+P_101000011*QR_000000001100+P_101000111*QR_000000001101+P_101000211*QR_000000001102);
			ans_temp[ans_id*3+2]+=Pmtrx[12]*(P_000001011*QR_001000000000+P_000001111*QR_001000000001+P_000001211*QR_001000000002+P_000101011*QR_001000000010+P_000101111*QR_001000000011+P_000101211*QR_001000000012);
			ans_temp[ans_id*3+2]+=Pmtrx[13]*(P_000001011*QR_000001000000+P_000001111*QR_000001000001+P_000001211*QR_000001000002+P_000101011*QR_000001000010+P_000101111*QR_000001000011+P_000101211*QR_000001000012);
			ans_temp[ans_id*3+2]+=Pmtrx[14]*(P_000001011*QR_000000001000+P_000001111*QR_000000001001+P_000001211*QR_000000001002+P_000101011*QR_000000001010+P_000101111*QR_000000001011+P_000101211*QR_000000001012);
			ans_temp[ans_id*3+2]+=Pmtrx[15]*(P_000000012*QR_001000000000+P_000000112*QR_001000000001+P_000000212*QR_001000000002+P_000000312*QR_001000000003);
			ans_temp[ans_id*3+2]+=Pmtrx[16]*(P_000000012*QR_000001000000+P_000000112*QR_000001000001+P_000000212*QR_000001000002+P_000000312*QR_000001000003);
			ans_temp[ans_id*3+2]+=Pmtrx[17]*(P_000000012*QR_000000001000+P_000000112*QR_000000001001+P_000000212*QR_000000001002+P_000000312*QR_000000001003);
		}
		}
        __syncthreads();
        int num_thread=NTHREAD/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<3;ians++){
                    ans_temp[tId_x*3+ians]+=ans_temp[(tId_x+num_thread)*3+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<3;ians++){
                ans[(i_contrc_bra*contrc_ket_num+j_contrc_ket)*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
	}
}
__global__ void MD_Kp_dssp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<1;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aPin1=1/(2*Zta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(Q_001000000*PR_020000000000+Q_101000000*PR_020000000100);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(Q_000001000*PR_020000000000+Q_000101000*PR_020000000010);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(Q_000000001*PR_020000000000+Q_000000101*PR_020000000001);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(Q_001000000*PR_010010000000+Q_101000000*PR_010010000100);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(Q_000001000*PR_010010000000+Q_000101000*PR_010010000010);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(Q_000000001*PR_010010000000+Q_000000101*PR_010010000001);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(Q_001000000*PR_000020000000+Q_101000000*PR_000020000100);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(Q_000001000*PR_000020000000+Q_000101000*PR_000020000010);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(Q_000000001*PR_000020000000+Q_000000101*PR_000020000001);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(Q_001000000*PR_010000010000+Q_101000000*PR_010000010100);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(Q_000001000*PR_010000010000+Q_000101000*PR_010000010010);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(Q_000000001*PR_010000010000+Q_000000101*PR_010000010001);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(Q_001000000*PR_000010010000+Q_101000000*PR_000010010100);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(Q_000001000*PR_000010010000+Q_000101000*PR_000010010010);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(Q_000000001*PR_000010010000+Q_000000101*PR_000010010001);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(Q_001000000*PR_000000020000+Q_101000000*PR_000000020100);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(Q_000001000*PR_000000020000+Q_000101000*PR_000000020010);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(Q_000000001*PR_000000020000+Q_000000101*PR_000000020001);
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
__global__ void MD_Kq_dssp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<1;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
                double alphaT=rsqrt(Eta+Zta);
                double lmd=4*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[4];
                Ft_fs_3(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
	double R_100[3];
	double R_200[2];
	double R_300[1];
	double R_010[3];
	double R_110[2];
	double R_210[1];
	double R_020[2];
	double R_120[1];
	double R_030[1];
	double R_001[3];
	double R_101[2];
	double R_201[1];
	double R_011[2];
	double R_111[1];
	double R_021[1];
	double R_002[2];
	double R_102[1];
	double R_012[1];
	double R_003[1];
	for(int i=0;i<3;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<3;i++){
		R_001[i]=TZ*R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_200[i]=TX*R_100[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_110[i]=TX*R_010[i+1];
	}
	for(int i=0;i<2;i++){
		R_020[i]=TY*R_010[i+1]+R_000[i+1];
	}
	for(int i=0;i<2;i++){
		R_101[i]=TX*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_011[i]=TY*R_001[i+1];
	}
	for(int i=0;i<2;i++){
		R_002[i]=TZ*R_001[i+1]+R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_300[i]=TX*R_200[i+1]+2*R_100[i+1];
	}
	for(int i=0;i<1;i++){
		R_210[i]=TY*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_120[i]=TX*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_030[i]=TY*R_020[i+1]+2*R_010[i+1];
	}
	for(int i=0;i<1;i++){
		R_201[i]=TZ*R_200[i+1];
	}
	for(int i=0;i<1;i++){
		R_111[i]=TX*R_011[i+1];
	}
	for(int i=0;i<1;i++){
		R_021[i]=TZ*R_020[i+1];
	}
	for(int i=0;i<1;i++){
		R_102[i]=TX*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_012[i]=TY*R_002[i+1];
	}
	for(int i=0;i<1;i++){
		R_003[i]=TZ*R_002[i+1]+2*R_001[i+1];
	}
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
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
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_020000000*QR_001000000000+P_120000000*QR_001000000100+P_220000000*QR_001000000200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_020000000*QR_000001000000+P_120000000*QR_000001000100+P_220000000*QR_000001000200);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_020000000*QR_000000001000+P_120000000*QR_000000001100+P_220000000*QR_000000001200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_010010000*QR_001000000000+P_010110000*QR_001000000010+P_110010000*QR_001000000100+P_110110000*QR_001000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_010010000*QR_000001000000+P_010110000*QR_000001000010+P_110010000*QR_000001000100+P_110110000*QR_000001000110);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_010010000*QR_000000001000+P_010110000*QR_000000001010+P_110010000*QR_000000001100+P_110110000*QR_000000001110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_000020000*QR_001000000000+P_000120000*QR_001000000010+P_000220000*QR_001000000020);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_000020000*QR_000001000000+P_000120000*QR_000001000010+P_000220000*QR_000001000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000020000*QR_000000001000+P_000120000*QR_000000001010+P_000220000*QR_000000001020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_010000010*QR_001000000000+P_010000110*QR_001000000001+P_110000010*QR_001000000100+P_110000110*QR_001000000101);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_010000010*QR_000001000000+P_010000110*QR_000001000001+P_110000010*QR_000001000100+P_110000110*QR_000001000101);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_010000010*QR_000000001000+P_010000110*QR_000000001001+P_110000010*QR_000000001100+P_110000110*QR_000000001101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_000010010*QR_001000000000+P_000010110*QR_001000000001+P_000110010*QR_001000000010+P_000110110*QR_001000000011);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_000010010*QR_000001000000+P_000010110*QR_000001000001+P_000110010*QR_000001000010+P_000110110*QR_000001000011);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000010010*QR_000000001000+P_000010110*QR_000000001001+P_000110010*QR_000000001010+P_000110110*QR_000000001011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_000000020*QR_001000000000+P_000000120*QR_001000000001+P_000000220*QR_001000000002);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_000000020*QR_000001000000+P_000000120*QR_000001000001+P_000000220*QR_000001000002);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000000020*QR_000000001000+P_000000120*QR_000000001001+P_000000220*QR_000000001002);
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
__global__ void MD_Kp_dpsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(Q_001000000*PR_021000000000+Q_101000000*PR_021000000100);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(Q_000001000*PR_021000000000+Q_000101000*PR_021000000010);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(Q_000000001*PR_021000000000+Q_000000101*PR_021000000001);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(Q_001000000*PR_020001000000+Q_101000000*PR_020001000100);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(Q_000001000*PR_020001000000+Q_000101000*PR_020001000010);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(Q_000000001*PR_020001000000+Q_000000101*PR_020001000001);
			ans_temp[ans_id*6+0]+=Pmtrx[6]*(Q_001000000*PR_020000001000+Q_101000000*PR_020000001100);
			ans_temp[ans_id*6+0]+=Pmtrx[7]*(Q_000001000*PR_020000001000+Q_000101000*PR_020000001010);
			ans_temp[ans_id*6+0]+=Pmtrx[8]*(Q_000000001*PR_020000001000+Q_000000101*PR_020000001001);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(Q_001000000*PR_011010000000+Q_101000000*PR_011010000100);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(Q_000001000*PR_011010000000+Q_000101000*PR_011010000010);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(Q_000000001*PR_011010000000+Q_000000101*PR_011010000001);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(Q_001000000*PR_010011000000+Q_101000000*PR_010011000100);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(Q_000001000*PR_010011000000+Q_000101000*PR_010011000010);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(Q_000000001*PR_010011000000+Q_000000101*PR_010011000001);
			ans_temp[ans_id*6+1]+=Pmtrx[6]*(Q_001000000*PR_010010001000+Q_101000000*PR_010010001100);
			ans_temp[ans_id*6+1]+=Pmtrx[7]*(Q_000001000*PR_010010001000+Q_000101000*PR_010010001010);
			ans_temp[ans_id*6+1]+=Pmtrx[8]*(Q_000000001*PR_010010001000+Q_000000101*PR_010010001001);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(Q_001000000*PR_001020000000+Q_101000000*PR_001020000100);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(Q_000001000*PR_001020000000+Q_000101000*PR_001020000010);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(Q_000000001*PR_001020000000+Q_000000101*PR_001020000001);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(Q_001000000*PR_000021000000+Q_101000000*PR_000021000100);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(Q_000001000*PR_000021000000+Q_000101000*PR_000021000010);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(Q_000000001*PR_000021000000+Q_000000101*PR_000021000001);
			ans_temp[ans_id*6+2]+=Pmtrx[6]*(Q_001000000*PR_000020001000+Q_101000000*PR_000020001100);
			ans_temp[ans_id*6+2]+=Pmtrx[7]*(Q_000001000*PR_000020001000+Q_000101000*PR_000020001010);
			ans_temp[ans_id*6+2]+=Pmtrx[8]*(Q_000000001*PR_000020001000+Q_000000101*PR_000020001001);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(Q_001000000*PR_011000010000+Q_101000000*PR_011000010100);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(Q_000001000*PR_011000010000+Q_000101000*PR_011000010010);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(Q_000000001*PR_011000010000+Q_000000101*PR_011000010001);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(Q_001000000*PR_010001010000+Q_101000000*PR_010001010100);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(Q_000001000*PR_010001010000+Q_000101000*PR_010001010010);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(Q_000000001*PR_010001010000+Q_000000101*PR_010001010001);
			ans_temp[ans_id*6+3]+=Pmtrx[6]*(Q_001000000*PR_010000011000+Q_101000000*PR_010000011100);
			ans_temp[ans_id*6+3]+=Pmtrx[7]*(Q_000001000*PR_010000011000+Q_000101000*PR_010000011010);
			ans_temp[ans_id*6+3]+=Pmtrx[8]*(Q_000000001*PR_010000011000+Q_000000101*PR_010000011001);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(Q_001000000*PR_001010010000+Q_101000000*PR_001010010100);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(Q_000001000*PR_001010010000+Q_000101000*PR_001010010010);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(Q_000000001*PR_001010010000+Q_000000101*PR_001010010001);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(Q_001000000*PR_000011010000+Q_101000000*PR_000011010100);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(Q_000001000*PR_000011010000+Q_000101000*PR_000011010010);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(Q_000000001*PR_000011010000+Q_000000101*PR_000011010001);
			ans_temp[ans_id*6+4]+=Pmtrx[6]*(Q_001000000*PR_000010011000+Q_101000000*PR_000010011100);
			ans_temp[ans_id*6+4]+=Pmtrx[7]*(Q_000001000*PR_000010011000+Q_000101000*PR_000010011010);
			ans_temp[ans_id*6+4]+=Pmtrx[8]*(Q_000000001*PR_000010011000+Q_000000101*PR_000010011001);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(Q_001000000*PR_001000020000+Q_101000000*PR_001000020100);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(Q_000001000*PR_001000020000+Q_000101000*PR_001000020010);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(Q_000000001*PR_001000020000+Q_000000101*PR_001000020001);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(Q_001000000*PR_000001020000+Q_101000000*PR_000001020100);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(Q_000001000*PR_000001020000+Q_000101000*PR_000001020010);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(Q_000000001*PR_000001020000+Q_000000101*PR_000001020001);
			ans_temp[ans_id*6+5]+=Pmtrx[6]*(Q_001000000*PR_000000021000+Q_101000000*PR_000000021100);
			ans_temp[ans_id*6+5]+=Pmtrx[7]*(Q_000001000*PR_000000021000+Q_000101000*PR_000000021010);
			ans_temp[ans_id*6+5]+=Pmtrx[8]*(Q_000000001*PR_000000021000+Q_000000101*PR_000000021001);
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
__global__ void MD_Kq_dpsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[9]={0.0};

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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<3;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
				double QR_001000000003=Q_001000000*R_003[0]+-1*Q_101000000*R_103[0];
				double QR_000001000003=Q_000001000*R_003[0]+-1*Q_000101000*R_013[0];
				double QR_000000001003=Q_000000001*R_003[0]+-1*Q_000000101*R_004[0];
				double QR_001000000012=Q_001000000*R_012[0]+-1*Q_101000000*R_112[0];
				double QR_000001000012=Q_000001000*R_012[0]+-1*Q_000101000*R_022[0];
				double QR_000000001012=Q_000000001*R_012[0]+-1*Q_000000101*R_013[0];
				double QR_001000000021=Q_001000000*R_021[0]+-1*Q_101000000*R_121[0];
				double QR_000001000021=Q_000001000*R_021[0]+-1*Q_000101000*R_031[0];
				double QR_000000001021=Q_000000001*R_021[0]+-1*Q_000000101*R_022[0];
				double QR_001000000030=Q_001000000*R_030[0]+-1*Q_101000000*R_130[0];
				double QR_000001000030=Q_000001000*R_030[0]+-1*Q_000101000*R_040[0];
				double QR_000000001030=Q_000000001*R_030[0]+-1*Q_000000101*R_031[0];
				double QR_001000000102=Q_001000000*R_102[0]+-1*Q_101000000*R_202[0];
				double QR_000001000102=Q_000001000*R_102[0]+-1*Q_000101000*R_112[0];
				double QR_000000001102=Q_000000001*R_102[0]+-1*Q_000000101*R_103[0];
				double QR_001000000111=Q_001000000*R_111[0]+-1*Q_101000000*R_211[0];
				double QR_000001000111=Q_000001000*R_111[0]+-1*Q_000101000*R_121[0];
				double QR_000000001111=Q_000000001*R_111[0]+-1*Q_000000101*R_112[0];
				double QR_001000000120=Q_001000000*R_120[0]+-1*Q_101000000*R_220[0];
				double QR_000001000120=Q_000001000*R_120[0]+-1*Q_000101000*R_130[0];
				double QR_000000001120=Q_000000001*R_120[0]+-1*Q_000000101*R_121[0];
				double QR_001000000201=Q_001000000*R_201[0]+-1*Q_101000000*R_301[0];
				double QR_000001000201=Q_000001000*R_201[0]+-1*Q_000101000*R_211[0];
				double QR_000000001201=Q_000000001*R_201[0]+-1*Q_000000101*R_202[0];
				double QR_001000000210=Q_001000000*R_210[0]+-1*Q_101000000*R_310[0];
				double QR_000001000210=Q_000001000*R_210[0]+-1*Q_000101000*R_220[0];
				double QR_000000001210=Q_000000001*R_210[0]+-1*Q_000000101*R_211[0];
				double QR_001000000300=Q_001000000*R_300[0]+-1*Q_101000000*R_400[0];
				double QR_000001000300=Q_000001000*R_300[0]+-1*Q_000101000*R_310[0];
				double QR_000000001300=Q_000000001*R_300[0]+-1*Q_000000101*R_301[0];
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
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_021000000*QR_001000000000+P_121000000*QR_001000000100+P_221000000*QR_001000000200+P_321000000*QR_001000000300);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_021000000*QR_000001000000+P_121000000*QR_000001000100+P_221000000*QR_000001000200+P_321000000*QR_000001000300);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_021000000*QR_000000001000+P_121000000*QR_000000001100+P_221000000*QR_000000001200+P_321000000*QR_000000001300);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_020001000*QR_001000000000+P_020101000*QR_001000000010+P_120001000*QR_001000000100+P_120101000*QR_001000000110+P_220001000*QR_001000000200+P_220101000*QR_001000000210);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_020001000*QR_000001000000+P_020101000*QR_000001000010+P_120001000*QR_000001000100+P_120101000*QR_000001000110+P_220001000*QR_000001000200+P_220101000*QR_000001000210);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_020001000*QR_000000001000+P_020101000*QR_000000001010+P_120001000*QR_000000001100+P_120101000*QR_000000001110+P_220001000*QR_000000001200+P_220101000*QR_000000001210);
			ans_temp[ans_id*6+0]+=Pmtrx[6]*(P_020000001*QR_001000000000+P_020000101*QR_001000000001+P_120000001*QR_001000000100+P_120000101*QR_001000000101+P_220000001*QR_001000000200+P_220000101*QR_001000000201);
			ans_temp[ans_id*6+0]+=Pmtrx[7]*(P_020000001*QR_000001000000+P_020000101*QR_000001000001+P_120000001*QR_000001000100+P_120000101*QR_000001000101+P_220000001*QR_000001000200+P_220000101*QR_000001000201);
			ans_temp[ans_id*6+0]+=Pmtrx[8]*(P_020000001*QR_000000001000+P_020000101*QR_000000001001+P_120000001*QR_000000001100+P_120000101*QR_000000001101+P_220000001*QR_000000001200+P_220000101*QR_000000001201);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_011010000*QR_001000000000+P_011110000*QR_001000000010+P_111010000*QR_001000000100+P_111110000*QR_001000000110+P_211010000*QR_001000000200+P_211110000*QR_001000000210);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_011010000*QR_000001000000+P_011110000*QR_000001000010+P_111010000*QR_000001000100+P_111110000*QR_000001000110+P_211010000*QR_000001000200+P_211110000*QR_000001000210);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_011010000*QR_000000001000+P_011110000*QR_000000001010+P_111010000*QR_000000001100+P_111110000*QR_000000001110+P_211010000*QR_000000001200+P_211110000*QR_000000001210);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_010011000*QR_001000000000+P_010111000*QR_001000000010+P_010211000*QR_001000000020+P_110011000*QR_001000000100+P_110111000*QR_001000000110+P_110211000*QR_001000000120);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_010011000*QR_000001000000+P_010111000*QR_000001000010+P_010211000*QR_000001000020+P_110011000*QR_000001000100+P_110111000*QR_000001000110+P_110211000*QR_000001000120);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_010011000*QR_000000001000+P_010111000*QR_000000001010+P_010211000*QR_000000001020+P_110011000*QR_000000001100+P_110111000*QR_000000001110+P_110211000*QR_000000001120);
			ans_temp[ans_id*6+1]+=Pmtrx[6]*(P_010010001*QR_001000000000+P_010010101*QR_001000000001+P_010110001*QR_001000000010+P_010110101*QR_001000000011+P_110010001*QR_001000000100+P_110010101*QR_001000000101+P_110110001*QR_001000000110+P_110110101*QR_001000000111);
			ans_temp[ans_id*6+1]+=Pmtrx[7]*(P_010010001*QR_000001000000+P_010010101*QR_000001000001+P_010110001*QR_000001000010+P_010110101*QR_000001000011+P_110010001*QR_000001000100+P_110010101*QR_000001000101+P_110110001*QR_000001000110+P_110110101*QR_000001000111);
			ans_temp[ans_id*6+1]+=Pmtrx[8]*(P_010010001*QR_000000001000+P_010010101*QR_000000001001+P_010110001*QR_000000001010+P_010110101*QR_000000001011+P_110010001*QR_000000001100+P_110010101*QR_000000001101+P_110110001*QR_000000001110+P_110110101*QR_000000001111);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_001020000*QR_001000000000+P_001120000*QR_001000000010+P_001220000*QR_001000000020+P_101020000*QR_001000000100+P_101120000*QR_001000000110+P_101220000*QR_001000000120);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_001020000*QR_000001000000+P_001120000*QR_000001000010+P_001220000*QR_000001000020+P_101020000*QR_000001000100+P_101120000*QR_000001000110+P_101220000*QR_000001000120);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_001020000*QR_000000001000+P_001120000*QR_000000001010+P_001220000*QR_000000001020+P_101020000*QR_000000001100+P_101120000*QR_000000001110+P_101220000*QR_000000001120);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_000021000*QR_001000000000+P_000121000*QR_001000000010+P_000221000*QR_001000000020+P_000321000*QR_001000000030);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000021000*QR_000001000000+P_000121000*QR_000001000010+P_000221000*QR_000001000020+P_000321000*QR_000001000030);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000021000*QR_000000001000+P_000121000*QR_000000001010+P_000221000*QR_000000001020+P_000321000*QR_000000001030);
			ans_temp[ans_id*6+2]+=Pmtrx[6]*(P_000020001*QR_001000000000+P_000020101*QR_001000000001+P_000120001*QR_001000000010+P_000120101*QR_001000000011+P_000220001*QR_001000000020+P_000220101*QR_001000000021);
			ans_temp[ans_id*6+2]+=Pmtrx[7]*(P_000020001*QR_000001000000+P_000020101*QR_000001000001+P_000120001*QR_000001000010+P_000120101*QR_000001000011+P_000220001*QR_000001000020+P_000220101*QR_000001000021);
			ans_temp[ans_id*6+2]+=Pmtrx[8]*(P_000020001*QR_000000001000+P_000020101*QR_000000001001+P_000120001*QR_000000001010+P_000120101*QR_000000001011+P_000220001*QR_000000001020+P_000220101*QR_000000001021);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_011000010*QR_001000000000+P_011000110*QR_001000000001+P_111000010*QR_001000000100+P_111000110*QR_001000000101+P_211000010*QR_001000000200+P_211000110*QR_001000000201);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_011000010*QR_000001000000+P_011000110*QR_000001000001+P_111000010*QR_000001000100+P_111000110*QR_000001000101+P_211000010*QR_000001000200+P_211000110*QR_000001000201);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_011000010*QR_000000001000+P_011000110*QR_000000001001+P_111000010*QR_000000001100+P_111000110*QR_000000001101+P_211000010*QR_000000001200+P_211000110*QR_000000001201);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_010001010*QR_001000000000+P_010001110*QR_001000000001+P_010101010*QR_001000000010+P_010101110*QR_001000000011+P_110001010*QR_001000000100+P_110001110*QR_001000000101+P_110101010*QR_001000000110+P_110101110*QR_001000000111);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_010001010*QR_000001000000+P_010001110*QR_000001000001+P_010101010*QR_000001000010+P_010101110*QR_000001000011+P_110001010*QR_000001000100+P_110001110*QR_000001000101+P_110101010*QR_000001000110+P_110101110*QR_000001000111);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_010001010*QR_000000001000+P_010001110*QR_000000001001+P_010101010*QR_000000001010+P_010101110*QR_000000001011+P_110001010*QR_000000001100+P_110001110*QR_000000001101+P_110101010*QR_000000001110+P_110101110*QR_000000001111);
			ans_temp[ans_id*6+3]+=Pmtrx[6]*(P_010000011*QR_001000000000+P_010000111*QR_001000000001+P_010000211*QR_001000000002+P_110000011*QR_001000000100+P_110000111*QR_001000000101+P_110000211*QR_001000000102);
			ans_temp[ans_id*6+3]+=Pmtrx[7]*(P_010000011*QR_000001000000+P_010000111*QR_000001000001+P_010000211*QR_000001000002+P_110000011*QR_000001000100+P_110000111*QR_000001000101+P_110000211*QR_000001000102);
			ans_temp[ans_id*6+3]+=Pmtrx[8]*(P_010000011*QR_000000001000+P_010000111*QR_000000001001+P_010000211*QR_000000001002+P_110000011*QR_000000001100+P_110000111*QR_000000001101+P_110000211*QR_000000001102);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_001010010*QR_001000000000+P_001010110*QR_001000000001+P_001110010*QR_001000000010+P_001110110*QR_001000000011+P_101010010*QR_001000000100+P_101010110*QR_001000000101+P_101110010*QR_001000000110+P_101110110*QR_001000000111);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_001010010*QR_000001000000+P_001010110*QR_000001000001+P_001110010*QR_000001000010+P_001110110*QR_000001000011+P_101010010*QR_000001000100+P_101010110*QR_000001000101+P_101110010*QR_000001000110+P_101110110*QR_000001000111);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_001010010*QR_000000001000+P_001010110*QR_000000001001+P_001110010*QR_000000001010+P_001110110*QR_000000001011+P_101010010*QR_000000001100+P_101010110*QR_000000001101+P_101110010*QR_000000001110+P_101110110*QR_000000001111);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_000011010*QR_001000000000+P_000011110*QR_001000000001+P_000111010*QR_001000000010+P_000111110*QR_001000000011+P_000211010*QR_001000000020+P_000211110*QR_001000000021);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000011010*QR_000001000000+P_000011110*QR_000001000001+P_000111010*QR_000001000010+P_000111110*QR_000001000011+P_000211010*QR_000001000020+P_000211110*QR_000001000021);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000011010*QR_000000001000+P_000011110*QR_000000001001+P_000111010*QR_000000001010+P_000111110*QR_000000001011+P_000211010*QR_000000001020+P_000211110*QR_000000001021);
			ans_temp[ans_id*6+4]+=Pmtrx[6]*(P_000010011*QR_001000000000+P_000010111*QR_001000000001+P_000010211*QR_001000000002+P_000110011*QR_001000000010+P_000110111*QR_001000000011+P_000110211*QR_001000000012);
			ans_temp[ans_id*6+4]+=Pmtrx[7]*(P_000010011*QR_000001000000+P_000010111*QR_000001000001+P_000010211*QR_000001000002+P_000110011*QR_000001000010+P_000110111*QR_000001000011+P_000110211*QR_000001000012);
			ans_temp[ans_id*6+4]+=Pmtrx[8]*(P_000010011*QR_000000001000+P_000010111*QR_000000001001+P_000010211*QR_000000001002+P_000110011*QR_000000001010+P_000110111*QR_000000001011+P_000110211*QR_000000001012);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_001000020*QR_001000000000+P_001000120*QR_001000000001+P_001000220*QR_001000000002+P_101000020*QR_001000000100+P_101000120*QR_001000000101+P_101000220*QR_001000000102);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_001000020*QR_000001000000+P_001000120*QR_000001000001+P_001000220*QR_000001000002+P_101000020*QR_000001000100+P_101000120*QR_000001000101+P_101000220*QR_000001000102);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_001000020*QR_000000001000+P_001000120*QR_000000001001+P_001000220*QR_000000001002+P_101000020*QR_000000001100+P_101000120*QR_000000001101+P_101000220*QR_000000001102);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_000001020*QR_001000000000+P_000001120*QR_001000000001+P_000001220*QR_001000000002+P_000101020*QR_001000000010+P_000101120*QR_001000000011+P_000101220*QR_001000000012);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000001020*QR_000001000000+P_000001120*QR_000001000001+P_000001220*QR_000001000002+P_000101020*QR_000001000010+P_000101120*QR_000001000011+P_000101220*QR_000001000012);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000001020*QR_000000001000+P_000001120*QR_000000001001+P_000001220*QR_000000001002+P_000101020*QR_000000001010+P_000101120*QR_000000001011+P_000101220*QR_000000001012);
			ans_temp[ans_id*6+5]+=Pmtrx[6]*(P_000000021*QR_001000000000+P_000000121*QR_001000000001+P_000000221*QR_001000000002+P_000000321*QR_001000000003);
			ans_temp[ans_id*6+5]+=Pmtrx[7]*(P_000000021*QR_000001000000+P_000000121*QR_000001000001+P_000000221*QR_000001000002+P_000000321*QR_000001000003);
			ans_temp[ans_id*6+5]+=Pmtrx[8]*(P_000000021*QR_000000001000+P_000000121*QR_000000001001+P_000000221*QR_000000001002+P_000000321*QR_000000001003);
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
__global__ void MD_Kp_ddsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

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
				double Qd_001[3];
				Qd_001[0]=QD[ii*3+0];
				Qd_001[1]=QD[ii*3+1];
				Qd_001[2]=QD[ii*3+2];
				double Eta=Eta_in[ii];
				double pq=pq_in[ii];
            float K2_q=K2_q_in[ii];
				double aQin1=1/(2*Eta);
        for(unsigned int j=tId_x;j<primit_bra_end-primit_bra_start;j+=tdis){
            unsigned int jj=primit_bra_start+j;
            unsigned int id_bra=tex1Dfetch(tex_id_bra,jj);
            double P_max=0.0;
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(Q_001000000*PR_022000000000+Q_101000000*PR_022000000100);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(Q_000001000*PR_022000000000+Q_000101000*PR_022000000010);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(Q_000000001*PR_022000000000+Q_000000101*PR_022000000001);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(Q_001000000*PR_021001000000+Q_101000000*PR_021001000100);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(Q_000001000*PR_021001000000+Q_000101000*PR_021001000010);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(Q_000000001*PR_021001000000+Q_000000101*PR_021001000001);
			ans_temp[ans_id*6+0]+=Pmtrx[6]*(Q_001000000*PR_020002000000+Q_101000000*PR_020002000100);
			ans_temp[ans_id*6+0]+=Pmtrx[7]*(Q_000001000*PR_020002000000+Q_000101000*PR_020002000010);
			ans_temp[ans_id*6+0]+=Pmtrx[8]*(Q_000000001*PR_020002000000+Q_000000101*PR_020002000001);
			ans_temp[ans_id*6+0]+=Pmtrx[9]*(Q_001000000*PR_021000001000+Q_101000000*PR_021000001100);
			ans_temp[ans_id*6+0]+=Pmtrx[10]*(Q_000001000*PR_021000001000+Q_000101000*PR_021000001010);
			ans_temp[ans_id*6+0]+=Pmtrx[11]*(Q_000000001*PR_021000001000+Q_000000101*PR_021000001001);
			ans_temp[ans_id*6+0]+=Pmtrx[12]*(Q_001000000*PR_020001001000+Q_101000000*PR_020001001100);
			ans_temp[ans_id*6+0]+=Pmtrx[13]*(Q_000001000*PR_020001001000+Q_000101000*PR_020001001010);
			ans_temp[ans_id*6+0]+=Pmtrx[14]*(Q_000000001*PR_020001001000+Q_000000101*PR_020001001001);
			ans_temp[ans_id*6+0]+=Pmtrx[15]*(Q_001000000*PR_020000002000+Q_101000000*PR_020000002100);
			ans_temp[ans_id*6+0]+=Pmtrx[16]*(Q_000001000*PR_020000002000+Q_000101000*PR_020000002010);
			ans_temp[ans_id*6+0]+=Pmtrx[17]*(Q_000000001*PR_020000002000+Q_000000101*PR_020000002001);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(Q_001000000*PR_012010000000+Q_101000000*PR_012010000100);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(Q_000001000*PR_012010000000+Q_000101000*PR_012010000010);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(Q_000000001*PR_012010000000+Q_000000101*PR_012010000001);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(Q_001000000*PR_011011000000+Q_101000000*PR_011011000100);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(Q_000001000*PR_011011000000+Q_000101000*PR_011011000010);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(Q_000000001*PR_011011000000+Q_000000101*PR_011011000001);
			ans_temp[ans_id*6+1]+=Pmtrx[6]*(Q_001000000*PR_010012000000+Q_101000000*PR_010012000100);
			ans_temp[ans_id*6+1]+=Pmtrx[7]*(Q_000001000*PR_010012000000+Q_000101000*PR_010012000010);
			ans_temp[ans_id*6+1]+=Pmtrx[8]*(Q_000000001*PR_010012000000+Q_000000101*PR_010012000001);
			ans_temp[ans_id*6+1]+=Pmtrx[9]*(Q_001000000*PR_011010001000+Q_101000000*PR_011010001100);
			ans_temp[ans_id*6+1]+=Pmtrx[10]*(Q_000001000*PR_011010001000+Q_000101000*PR_011010001010);
			ans_temp[ans_id*6+1]+=Pmtrx[11]*(Q_000000001*PR_011010001000+Q_000000101*PR_011010001001);
			ans_temp[ans_id*6+1]+=Pmtrx[12]*(Q_001000000*PR_010011001000+Q_101000000*PR_010011001100);
			ans_temp[ans_id*6+1]+=Pmtrx[13]*(Q_000001000*PR_010011001000+Q_000101000*PR_010011001010);
			ans_temp[ans_id*6+1]+=Pmtrx[14]*(Q_000000001*PR_010011001000+Q_000000101*PR_010011001001);
			ans_temp[ans_id*6+1]+=Pmtrx[15]*(Q_001000000*PR_010010002000+Q_101000000*PR_010010002100);
			ans_temp[ans_id*6+1]+=Pmtrx[16]*(Q_000001000*PR_010010002000+Q_000101000*PR_010010002010);
			ans_temp[ans_id*6+1]+=Pmtrx[17]*(Q_000000001*PR_010010002000+Q_000000101*PR_010010002001);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(Q_001000000*PR_002020000000+Q_101000000*PR_002020000100);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(Q_000001000*PR_002020000000+Q_000101000*PR_002020000010);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(Q_000000001*PR_002020000000+Q_000000101*PR_002020000001);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(Q_001000000*PR_001021000000+Q_101000000*PR_001021000100);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(Q_000001000*PR_001021000000+Q_000101000*PR_001021000010);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(Q_000000001*PR_001021000000+Q_000000101*PR_001021000001);
			ans_temp[ans_id*6+2]+=Pmtrx[6]*(Q_001000000*PR_000022000000+Q_101000000*PR_000022000100);
			ans_temp[ans_id*6+2]+=Pmtrx[7]*(Q_000001000*PR_000022000000+Q_000101000*PR_000022000010);
			ans_temp[ans_id*6+2]+=Pmtrx[8]*(Q_000000001*PR_000022000000+Q_000000101*PR_000022000001);
			ans_temp[ans_id*6+2]+=Pmtrx[9]*(Q_001000000*PR_001020001000+Q_101000000*PR_001020001100);
			ans_temp[ans_id*6+2]+=Pmtrx[10]*(Q_000001000*PR_001020001000+Q_000101000*PR_001020001010);
			ans_temp[ans_id*6+2]+=Pmtrx[11]*(Q_000000001*PR_001020001000+Q_000000101*PR_001020001001);
			ans_temp[ans_id*6+2]+=Pmtrx[12]*(Q_001000000*PR_000021001000+Q_101000000*PR_000021001100);
			ans_temp[ans_id*6+2]+=Pmtrx[13]*(Q_000001000*PR_000021001000+Q_000101000*PR_000021001010);
			ans_temp[ans_id*6+2]+=Pmtrx[14]*(Q_000000001*PR_000021001000+Q_000000101*PR_000021001001);
			ans_temp[ans_id*6+2]+=Pmtrx[15]*(Q_001000000*PR_000020002000+Q_101000000*PR_000020002100);
			ans_temp[ans_id*6+2]+=Pmtrx[16]*(Q_000001000*PR_000020002000+Q_000101000*PR_000020002010);
			ans_temp[ans_id*6+2]+=Pmtrx[17]*(Q_000000001*PR_000020002000+Q_000000101*PR_000020002001);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(Q_001000000*PR_012000010000+Q_101000000*PR_012000010100);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(Q_000001000*PR_012000010000+Q_000101000*PR_012000010010);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(Q_000000001*PR_012000010000+Q_000000101*PR_012000010001);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(Q_001000000*PR_011001010000+Q_101000000*PR_011001010100);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(Q_000001000*PR_011001010000+Q_000101000*PR_011001010010);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(Q_000000001*PR_011001010000+Q_000000101*PR_011001010001);
			ans_temp[ans_id*6+3]+=Pmtrx[6]*(Q_001000000*PR_010002010000+Q_101000000*PR_010002010100);
			ans_temp[ans_id*6+3]+=Pmtrx[7]*(Q_000001000*PR_010002010000+Q_000101000*PR_010002010010);
			ans_temp[ans_id*6+3]+=Pmtrx[8]*(Q_000000001*PR_010002010000+Q_000000101*PR_010002010001);
			ans_temp[ans_id*6+3]+=Pmtrx[9]*(Q_001000000*PR_011000011000+Q_101000000*PR_011000011100);
			ans_temp[ans_id*6+3]+=Pmtrx[10]*(Q_000001000*PR_011000011000+Q_000101000*PR_011000011010);
			ans_temp[ans_id*6+3]+=Pmtrx[11]*(Q_000000001*PR_011000011000+Q_000000101*PR_011000011001);
			ans_temp[ans_id*6+3]+=Pmtrx[12]*(Q_001000000*PR_010001011000+Q_101000000*PR_010001011100);
			ans_temp[ans_id*6+3]+=Pmtrx[13]*(Q_000001000*PR_010001011000+Q_000101000*PR_010001011010);
			ans_temp[ans_id*6+3]+=Pmtrx[14]*(Q_000000001*PR_010001011000+Q_000000101*PR_010001011001);
			ans_temp[ans_id*6+3]+=Pmtrx[15]*(Q_001000000*PR_010000012000+Q_101000000*PR_010000012100);
			ans_temp[ans_id*6+3]+=Pmtrx[16]*(Q_000001000*PR_010000012000+Q_000101000*PR_010000012010);
			ans_temp[ans_id*6+3]+=Pmtrx[17]*(Q_000000001*PR_010000012000+Q_000000101*PR_010000012001);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(Q_001000000*PR_002010010000+Q_101000000*PR_002010010100);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(Q_000001000*PR_002010010000+Q_000101000*PR_002010010010);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(Q_000000001*PR_002010010000+Q_000000101*PR_002010010001);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(Q_001000000*PR_001011010000+Q_101000000*PR_001011010100);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(Q_000001000*PR_001011010000+Q_000101000*PR_001011010010);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(Q_000000001*PR_001011010000+Q_000000101*PR_001011010001);
			ans_temp[ans_id*6+4]+=Pmtrx[6]*(Q_001000000*PR_000012010000+Q_101000000*PR_000012010100);
			ans_temp[ans_id*6+4]+=Pmtrx[7]*(Q_000001000*PR_000012010000+Q_000101000*PR_000012010010);
			ans_temp[ans_id*6+4]+=Pmtrx[8]*(Q_000000001*PR_000012010000+Q_000000101*PR_000012010001);
			ans_temp[ans_id*6+4]+=Pmtrx[9]*(Q_001000000*PR_001010011000+Q_101000000*PR_001010011100);
			ans_temp[ans_id*6+4]+=Pmtrx[10]*(Q_000001000*PR_001010011000+Q_000101000*PR_001010011010);
			ans_temp[ans_id*6+4]+=Pmtrx[11]*(Q_000000001*PR_001010011000+Q_000000101*PR_001010011001);
			ans_temp[ans_id*6+4]+=Pmtrx[12]*(Q_001000000*PR_000011011000+Q_101000000*PR_000011011100);
			ans_temp[ans_id*6+4]+=Pmtrx[13]*(Q_000001000*PR_000011011000+Q_000101000*PR_000011011010);
			ans_temp[ans_id*6+4]+=Pmtrx[14]*(Q_000000001*PR_000011011000+Q_000000101*PR_000011011001);
			ans_temp[ans_id*6+4]+=Pmtrx[15]*(Q_001000000*PR_000010012000+Q_101000000*PR_000010012100);
			ans_temp[ans_id*6+4]+=Pmtrx[16]*(Q_000001000*PR_000010012000+Q_000101000*PR_000010012010);
			ans_temp[ans_id*6+4]+=Pmtrx[17]*(Q_000000001*PR_000010012000+Q_000000101*PR_000010012001);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(Q_001000000*PR_002000020000+Q_101000000*PR_002000020100);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(Q_000001000*PR_002000020000+Q_000101000*PR_002000020010);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(Q_000000001*PR_002000020000+Q_000000101*PR_002000020001);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(Q_001000000*PR_001001020000+Q_101000000*PR_001001020100);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(Q_000001000*PR_001001020000+Q_000101000*PR_001001020010);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(Q_000000001*PR_001001020000+Q_000000101*PR_001001020001);
			ans_temp[ans_id*6+5]+=Pmtrx[6]*(Q_001000000*PR_000002020000+Q_101000000*PR_000002020100);
			ans_temp[ans_id*6+5]+=Pmtrx[7]*(Q_000001000*PR_000002020000+Q_000101000*PR_000002020010);
			ans_temp[ans_id*6+5]+=Pmtrx[8]*(Q_000000001*PR_000002020000+Q_000000101*PR_000002020001);
			ans_temp[ans_id*6+5]+=Pmtrx[9]*(Q_001000000*PR_001000021000+Q_101000000*PR_001000021100);
			ans_temp[ans_id*6+5]+=Pmtrx[10]*(Q_000001000*PR_001000021000+Q_000101000*PR_001000021010);
			ans_temp[ans_id*6+5]+=Pmtrx[11]*(Q_000000001*PR_001000021000+Q_000000101*PR_001000021001);
			ans_temp[ans_id*6+5]+=Pmtrx[12]*(Q_001000000*PR_000001021000+Q_101000000*PR_000001021100);
			ans_temp[ans_id*6+5]+=Pmtrx[13]*(Q_000001000*PR_000001021000+Q_000101000*PR_000001021010);
			ans_temp[ans_id*6+5]+=Pmtrx[14]*(Q_000000001*PR_000001021000+Q_000000101*PR_000001021001);
			ans_temp[ans_id*6+5]+=Pmtrx[15]*(Q_001000000*PR_000000022000+Q_101000000*PR_000000022100);
			ans_temp[ans_id*6+5]+=Pmtrx[16]*(Q_000001000*PR_000000022000+Q_000101000*PR_000000022010);
			ans_temp[ans_id*6+5]+=Pmtrx[17]*(Q_000000001*PR_000000022000+Q_000000101*PR_000000022001);
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
__global__ void MD_Kq_ddsp_fs(unsigned int contrc_bra_num,unsigned int contrc_ket_num,\
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
    double Pmtrx[18]={0.0};

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
            for(int p_j=0;p_j<3;p_j++){
            for(int p_i=0;p_i<6;p_i++){
                Pmtrx[p_i*3+p_j]=Pmtrx_in[(id_ket+p_j)*mtrx_len+(id_bra+p_i)];
                double temp_P=fabsf(Pmtrx[p_i*3+p_j]);
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
				double Qd_001[3];
            temp_int2=tex1Dfetch(tex_QD,jj*3+0);
            Qd_001[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+1);
            Qd_001[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QD,jj*3+2);
            Qd_001[2]=__hiloint2double(temp_int2.y,temp_int2.x);
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
		double Qd_101[3];
		for(int i=0;i<3;i++){
			Qd_101[i]=aQin1;
			}
	double Q_001000000=Qd_001[0];
	double Q_101000000=Qd_101[0];
	double Q_000001000=Qd_001[1];
	double Q_000101000=Qd_101[1];
	double Q_000000001=Qd_001[2];
	double Q_000000101=Qd_101[2];
				double QR_001000000000=Q_001000000*R_000[0]+-1*Q_101000000*R_100[0];
				double QR_000001000000=Q_000001000*R_000[0]+-1*Q_000101000*R_010[0];
				double QR_000000001000=Q_000000001*R_000[0]+-1*Q_000000101*R_001[0];
				double QR_001000000001=Q_001000000*R_001[0]+-1*Q_101000000*R_101[0];
				double QR_000001000001=Q_000001000*R_001[0]+-1*Q_000101000*R_011[0];
				double QR_000000001001=Q_000000001*R_001[0]+-1*Q_000000101*R_002[0];
				double QR_001000000010=Q_001000000*R_010[0]+-1*Q_101000000*R_110[0];
				double QR_000001000010=Q_000001000*R_010[0]+-1*Q_000101000*R_020[0];
				double QR_000000001010=Q_000000001*R_010[0]+-1*Q_000000101*R_011[0];
				double QR_001000000100=Q_001000000*R_100[0]+-1*Q_101000000*R_200[0];
				double QR_000001000100=Q_000001000*R_100[0]+-1*Q_000101000*R_110[0];
				double QR_000000001100=Q_000000001*R_100[0]+-1*Q_000000101*R_101[0];
				double QR_001000000002=Q_001000000*R_002[0]+-1*Q_101000000*R_102[0];
				double QR_000001000002=Q_000001000*R_002[0]+-1*Q_000101000*R_012[0];
				double QR_000000001002=Q_000000001*R_002[0]+-1*Q_000000101*R_003[0];
				double QR_001000000011=Q_001000000*R_011[0]+-1*Q_101000000*R_111[0];
				double QR_000001000011=Q_000001000*R_011[0]+-1*Q_000101000*R_021[0];
				double QR_000000001011=Q_000000001*R_011[0]+-1*Q_000000101*R_012[0];
				double QR_001000000020=Q_001000000*R_020[0]+-1*Q_101000000*R_120[0];
				double QR_000001000020=Q_000001000*R_020[0]+-1*Q_000101000*R_030[0];
				double QR_000000001020=Q_000000001*R_020[0]+-1*Q_000000101*R_021[0];
				double QR_001000000101=Q_001000000*R_101[0]+-1*Q_101000000*R_201[0];
				double QR_000001000101=Q_000001000*R_101[0]+-1*Q_000101000*R_111[0];
				double QR_000000001101=Q_000000001*R_101[0]+-1*Q_000000101*R_102[0];
				double QR_001000000110=Q_001000000*R_110[0]+-1*Q_101000000*R_210[0];
				double QR_000001000110=Q_000001000*R_110[0]+-1*Q_000101000*R_120[0];
				double QR_000000001110=Q_000000001*R_110[0]+-1*Q_000000101*R_111[0];
				double QR_001000000200=Q_001000000*R_200[0]+-1*Q_101000000*R_300[0];
				double QR_000001000200=Q_000001000*R_200[0]+-1*Q_000101000*R_210[0];
				double QR_000000001200=Q_000000001*R_200[0]+-1*Q_000000101*R_201[0];
				double QR_001000000003=Q_001000000*R_003[0]+-1*Q_101000000*R_103[0];
				double QR_000001000003=Q_000001000*R_003[0]+-1*Q_000101000*R_013[0];
				double QR_000000001003=Q_000000001*R_003[0]+-1*Q_000000101*R_004[0];
				double QR_001000000012=Q_001000000*R_012[0]+-1*Q_101000000*R_112[0];
				double QR_000001000012=Q_000001000*R_012[0]+-1*Q_000101000*R_022[0];
				double QR_000000001012=Q_000000001*R_012[0]+-1*Q_000000101*R_013[0];
				double QR_001000000021=Q_001000000*R_021[0]+-1*Q_101000000*R_121[0];
				double QR_000001000021=Q_000001000*R_021[0]+-1*Q_000101000*R_031[0];
				double QR_000000001021=Q_000000001*R_021[0]+-1*Q_000000101*R_022[0];
				double QR_001000000030=Q_001000000*R_030[0]+-1*Q_101000000*R_130[0];
				double QR_000001000030=Q_000001000*R_030[0]+-1*Q_000101000*R_040[0];
				double QR_000000001030=Q_000000001*R_030[0]+-1*Q_000000101*R_031[0];
				double QR_001000000102=Q_001000000*R_102[0]+-1*Q_101000000*R_202[0];
				double QR_000001000102=Q_000001000*R_102[0]+-1*Q_000101000*R_112[0];
				double QR_000000001102=Q_000000001*R_102[0]+-1*Q_000000101*R_103[0];
				double QR_001000000111=Q_001000000*R_111[0]+-1*Q_101000000*R_211[0];
				double QR_000001000111=Q_000001000*R_111[0]+-1*Q_000101000*R_121[0];
				double QR_000000001111=Q_000000001*R_111[0]+-1*Q_000000101*R_112[0];
				double QR_001000000120=Q_001000000*R_120[0]+-1*Q_101000000*R_220[0];
				double QR_000001000120=Q_000001000*R_120[0]+-1*Q_000101000*R_130[0];
				double QR_000000001120=Q_000000001*R_120[0]+-1*Q_000000101*R_121[0];
				double QR_001000000201=Q_001000000*R_201[0]+-1*Q_101000000*R_301[0];
				double QR_000001000201=Q_000001000*R_201[0]+-1*Q_000101000*R_211[0];
				double QR_000000001201=Q_000000001*R_201[0]+-1*Q_000000101*R_202[0];
				double QR_001000000210=Q_001000000*R_210[0]+-1*Q_101000000*R_310[0];
				double QR_000001000210=Q_000001000*R_210[0]+-1*Q_000101000*R_220[0];
				double QR_000000001210=Q_000000001*R_210[0]+-1*Q_000000101*R_211[0];
				double QR_001000000300=Q_001000000*R_300[0]+-1*Q_101000000*R_400[0];
				double QR_000001000300=Q_000001000*R_300[0]+-1*Q_000101000*R_310[0];
				double QR_000000001300=Q_000000001*R_300[0]+-1*Q_000000101*R_301[0];
				double QR_001000000004=Q_001000000*R_004[0]+-1*Q_101000000*R_104[0];
				double QR_000001000004=Q_000001000*R_004[0]+-1*Q_000101000*R_014[0];
				double QR_000000001004=Q_000000001*R_004[0]+-1*Q_000000101*R_005[0];
				double QR_001000000013=Q_001000000*R_013[0]+-1*Q_101000000*R_113[0];
				double QR_000001000013=Q_000001000*R_013[0]+-1*Q_000101000*R_023[0];
				double QR_000000001013=Q_000000001*R_013[0]+-1*Q_000000101*R_014[0];
				double QR_001000000022=Q_001000000*R_022[0]+-1*Q_101000000*R_122[0];
				double QR_000001000022=Q_000001000*R_022[0]+-1*Q_000101000*R_032[0];
				double QR_000000001022=Q_000000001*R_022[0]+-1*Q_000000101*R_023[0];
				double QR_001000000031=Q_001000000*R_031[0]+-1*Q_101000000*R_131[0];
				double QR_000001000031=Q_000001000*R_031[0]+-1*Q_000101000*R_041[0];
				double QR_000000001031=Q_000000001*R_031[0]+-1*Q_000000101*R_032[0];
				double QR_001000000040=Q_001000000*R_040[0]+-1*Q_101000000*R_140[0];
				double QR_000001000040=Q_000001000*R_040[0]+-1*Q_000101000*R_050[0];
				double QR_000000001040=Q_000000001*R_040[0]+-1*Q_000000101*R_041[0];
				double QR_001000000103=Q_001000000*R_103[0]+-1*Q_101000000*R_203[0];
				double QR_000001000103=Q_000001000*R_103[0]+-1*Q_000101000*R_113[0];
				double QR_000000001103=Q_000000001*R_103[0]+-1*Q_000000101*R_104[0];
				double QR_001000000112=Q_001000000*R_112[0]+-1*Q_101000000*R_212[0];
				double QR_000001000112=Q_000001000*R_112[0]+-1*Q_000101000*R_122[0];
				double QR_000000001112=Q_000000001*R_112[0]+-1*Q_000000101*R_113[0];
				double QR_001000000121=Q_001000000*R_121[0]+-1*Q_101000000*R_221[0];
				double QR_000001000121=Q_000001000*R_121[0]+-1*Q_000101000*R_131[0];
				double QR_000000001121=Q_000000001*R_121[0]+-1*Q_000000101*R_122[0];
				double QR_001000000130=Q_001000000*R_130[0]+-1*Q_101000000*R_230[0];
				double QR_000001000130=Q_000001000*R_130[0]+-1*Q_000101000*R_140[0];
				double QR_000000001130=Q_000000001*R_130[0]+-1*Q_000000101*R_131[0];
				double QR_001000000202=Q_001000000*R_202[0]+-1*Q_101000000*R_302[0];
				double QR_000001000202=Q_000001000*R_202[0]+-1*Q_000101000*R_212[0];
				double QR_000000001202=Q_000000001*R_202[0]+-1*Q_000000101*R_203[0];
				double QR_001000000211=Q_001000000*R_211[0]+-1*Q_101000000*R_311[0];
				double QR_000001000211=Q_000001000*R_211[0]+-1*Q_000101000*R_221[0];
				double QR_000000001211=Q_000000001*R_211[0]+-1*Q_000000101*R_212[0];
				double QR_001000000220=Q_001000000*R_220[0]+-1*Q_101000000*R_320[0];
				double QR_000001000220=Q_000001000*R_220[0]+-1*Q_000101000*R_230[0];
				double QR_000000001220=Q_000000001*R_220[0]+-1*Q_000000101*R_221[0];
				double QR_001000000301=Q_001000000*R_301[0]+-1*Q_101000000*R_401[0];
				double QR_000001000301=Q_000001000*R_301[0]+-1*Q_000101000*R_311[0];
				double QR_000000001301=Q_000000001*R_301[0]+-1*Q_000000101*R_302[0];
				double QR_001000000310=Q_001000000*R_310[0]+-1*Q_101000000*R_410[0];
				double QR_000001000310=Q_000001000*R_310[0]+-1*Q_000101000*R_320[0];
				double QR_000000001310=Q_000000001*R_310[0]+-1*Q_000000101*R_311[0];
				double QR_001000000400=Q_001000000*R_400[0]+-1*Q_101000000*R_500[0];
				double QR_000001000400=Q_000001000*R_400[0]+-1*Q_000101000*R_410[0];
				double QR_000000001400=Q_000000001*R_400[0]+-1*Q_000000101*R_401[0];
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
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_022000000*QR_001000000000+P_122000000*QR_001000000100+P_222000000*QR_001000000200+P_322000000*QR_001000000300+P_422000000*QR_001000000400);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_022000000*QR_000001000000+P_122000000*QR_000001000100+P_222000000*QR_000001000200+P_322000000*QR_000001000300+P_422000000*QR_000001000400);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_022000000*QR_000000001000+P_122000000*QR_000000001100+P_222000000*QR_000000001200+P_322000000*QR_000000001300+P_422000000*QR_000000001400);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_021001000*QR_001000000000+P_021101000*QR_001000000010+P_121001000*QR_001000000100+P_121101000*QR_001000000110+P_221001000*QR_001000000200+P_221101000*QR_001000000210+P_321001000*QR_001000000300+P_321101000*QR_001000000310);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_021001000*QR_000001000000+P_021101000*QR_000001000010+P_121001000*QR_000001000100+P_121101000*QR_000001000110+P_221001000*QR_000001000200+P_221101000*QR_000001000210+P_321001000*QR_000001000300+P_321101000*QR_000001000310);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_021001000*QR_000000001000+P_021101000*QR_000000001010+P_121001000*QR_000000001100+P_121101000*QR_000000001110+P_221001000*QR_000000001200+P_221101000*QR_000000001210+P_321001000*QR_000000001300+P_321101000*QR_000000001310);
			ans_temp[ans_id*6+0]+=Pmtrx[6]*(P_020002000*QR_001000000000+P_020102000*QR_001000000010+P_020202000*QR_001000000020+P_120002000*QR_001000000100+P_120102000*QR_001000000110+P_120202000*QR_001000000120+P_220002000*QR_001000000200+P_220102000*QR_001000000210+P_220202000*QR_001000000220);
			ans_temp[ans_id*6+0]+=Pmtrx[7]*(P_020002000*QR_000001000000+P_020102000*QR_000001000010+P_020202000*QR_000001000020+P_120002000*QR_000001000100+P_120102000*QR_000001000110+P_120202000*QR_000001000120+P_220002000*QR_000001000200+P_220102000*QR_000001000210+P_220202000*QR_000001000220);
			ans_temp[ans_id*6+0]+=Pmtrx[8]*(P_020002000*QR_000000001000+P_020102000*QR_000000001010+P_020202000*QR_000000001020+P_120002000*QR_000000001100+P_120102000*QR_000000001110+P_120202000*QR_000000001120+P_220002000*QR_000000001200+P_220102000*QR_000000001210+P_220202000*QR_000000001220);
			ans_temp[ans_id*6+0]+=Pmtrx[9]*(P_021000001*QR_001000000000+P_021000101*QR_001000000001+P_121000001*QR_001000000100+P_121000101*QR_001000000101+P_221000001*QR_001000000200+P_221000101*QR_001000000201+P_321000001*QR_001000000300+P_321000101*QR_001000000301);
			ans_temp[ans_id*6+0]+=Pmtrx[10]*(P_021000001*QR_000001000000+P_021000101*QR_000001000001+P_121000001*QR_000001000100+P_121000101*QR_000001000101+P_221000001*QR_000001000200+P_221000101*QR_000001000201+P_321000001*QR_000001000300+P_321000101*QR_000001000301);
			ans_temp[ans_id*6+0]+=Pmtrx[11]*(P_021000001*QR_000000001000+P_021000101*QR_000000001001+P_121000001*QR_000000001100+P_121000101*QR_000000001101+P_221000001*QR_000000001200+P_221000101*QR_000000001201+P_321000001*QR_000000001300+P_321000101*QR_000000001301);
			ans_temp[ans_id*6+0]+=Pmtrx[12]*(P_020001001*QR_001000000000+P_020001101*QR_001000000001+P_020101001*QR_001000000010+P_020101101*QR_001000000011+P_120001001*QR_001000000100+P_120001101*QR_001000000101+P_120101001*QR_001000000110+P_120101101*QR_001000000111+P_220001001*QR_001000000200+P_220001101*QR_001000000201+P_220101001*QR_001000000210+P_220101101*QR_001000000211);
			ans_temp[ans_id*6+0]+=Pmtrx[13]*(P_020001001*QR_000001000000+P_020001101*QR_000001000001+P_020101001*QR_000001000010+P_020101101*QR_000001000011+P_120001001*QR_000001000100+P_120001101*QR_000001000101+P_120101001*QR_000001000110+P_120101101*QR_000001000111+P_220001001*QR_000001000200+P_220001101*QR_000001000201+P_220101001*QR_000001000210+P_220101101*QR_000001000211);
			ans_temp[ans_id*6+0]+=Pmtrx[14]*(P_020001001*QR_000000001000+P_020001101*QR_000000001001+P_020101001*QR_000000001010+P_020101101*QR_000000001011+P_120001001*QR_000000001100+P_120001101*QR_000000001101+P_120101001*QR_000000001110+P_120101101*QR_000000001111+P_220001001*QR_000000001200+P_220001101*QR_000000001201+P_220101001*QR_000000001210+P_220101101*QR_000000001211);
			ans_temp[ans_id*6+0]+=Pmtrx[15]*(P_020000002*QR_001000000000+P_020000102*QR_001000000001+P_020000202*QR_001000000002+P_120000002*QR_001000000100+P_120000102*QR_001000000101+P_120000202*QR_001000000102+P_220000002*QR_001000000200+P_220000102*QR_001000000201+P_220000202*QR_001000000202);
			ans_temp[ans_id*6+0]+=Pmtrx[16]*(P_020000002*QR_000001000000+P_020000102*QR_000001000001+P_020000202*QR_000001000002+P_120000002*QR_000001000100+P_120000102*QR_000001000101+P_120000202*QR_000001000102+P_220000002*QR_000001000200+P_220000102*QR_000001000201+P_220000202*QR_000001000202);
			ans_temp[ans_id*6+0]+=Pmtrx[17]*(P_020000002*QR_000000001000+P_020000102*QR_000000001001+P_020000202*QR_000000001002+P_120000002*QR_000000001100+P_120000102*QR_000000001101+P_120000202*QR_000000001102+P_220000002*QR_000000001200+P_220000102*QR_000000001201+P_220000202*QR_000000001202);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_012010000*QR_001000000000+P_012110000*QR_001000000010+P_112010000*QR_001000000100+P_112110000*QR_001000000110+P_212010000*QR_001000000200+P_212110000*QR_001000000210+P_312010000*QR_001000000300+P_312110000*QR_001000000310);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_012010000*QR_000001000000+P_012110000*QR_000001000010+P_112010000*QR_000001000100+P_112110000*QR_000001000110+P_212010000*QR_000001000200+P_212110000*QR_000001000210+P_312010000*QR_000001000300+P_312110000*QR_000001000310);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_012010000*QR_000000001000+P_012110000*QR_000000001010+P_112010000*QR_000000001100+P_112110000*QR_000000001110+P_212010000*QR_000000001200+P_212110000*QR_000000001210+P_312010000*QR_000000001300+P_312110000*QR_000000001310);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_011011000*QR_001000000000+P_011111000*QR_001000000010+P_011211000*QR_001000000020+P_111011000*QR_001000000100+P_111111000*QR_001000000110+P_111211000*QR_001000000120+P_211011000*QR_001000000200+P_211111000*QR_001000000210+P_211211000*QR_001000000220);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_011011000*QR_000001000000+P_011111000*QR_000001000010+P_011211000*QR_000001000020+P_111011000*QR_000001000100+P_111111000*QR_000001000110+P_111211000*QR_000001000120+P_211011000*QR_000001000200+P_211111000*QR_000001000210+P_211211000*QR_000001000220);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_011011000*QR_000000001000+P_011111000*QR_000000001010+P_011211000*QR_000000001020+P_111011000*QR_000000001100+P_111111000*QR_000000001110+P_111211000*QR_000000001120+P_211011000*QR_000000001200+P_211111000*QR_000000001210+P_211211000*QR_000000001220);
			ans_temp[ans_id*6+1]+=Pmtrx[6]*(P_010012000*QR_001000000000+P_010112000*QR_001000000010+P_010212000*QR_001000000020+P_010312000*QR_001000000030+P_110012000*QR_001000000100+P_110112000*QR_001000000110+P_110212000*QR_001000000120+P_110312000*QR_001000000130);
			ans_temp[ans_id*6+1]+=Pmtrx[7]*(P_010012000*QR_000001000000+P_010112000*QR_000001000010+P_010212000*QR_000001000020+P_010312000*QR_000001000030+P_110012000*QR_000001000100+P_110112000*QR_000001000110+P_110212000*QR_000001000120+P_110312000*QR_000001000130);
			ans_temp[ans_id*6+1]+=Pmtrx[8]*(P_010012000*QR_000000001000+P_010112000*QR_000000001010+P_010212000*QR_000000001020+P_010312000*QR_000000001030+P_110012000*QR_000000001100+P_110112000*QR_000000001110+P_110212000*QR_000000001120+P_110312000*QR_000000001130);
			ans_temp[ans_id*6+1]+=Pmtrx[9]*(P_011010001*QR_001000000000+P_011010101*QR_001000000001+P_011110001*QR_001000000010+P_011110101*QR_001000000011+P_111010001*QR_001000000100+P_111010101*QR_001000000101+P_111110001*QR_001000000110+P_111110101*QR_001000000111+P_211010001*QR_001000000200+P_211010101*QR_001000000201+P_211110001*QR_001000000210+P_211110101*QR_001000000211);
			ans_temp[ans_id*6+1]+=Pmtrx[10]*(P_011010001*QR_000001000000+P_011010101*QR_000001000001+P_011110001*QR_000001000010+P_011110101*QR_000001000011+P_111010001*QR_000001000100+P_111010101*QR_000001000101+P_111110001*QR_000001000110+P_111110101*QR_000001000111+P_211010001*QR_000001000200+P_211010101*QR_000001000201+P_211110001*QR_000001000210+P_211110101*QR_000001000211);
			ans_temp[ans_id*6+1]+=Pmtrx[11]*(P_011010001*QR_000000001000+P_011010101*QR_000000001001+P_011110001*QR_000000001010+P_011110101*QR_000000001011+P_111010001*QR_000000001100+P_111010101*QR_000000001101+P_111110001*QR_000000001110+P_111110101*QR_000000001111+P_211010001*QR_000000001200+P_211010101*QR_000000001201+P_211110001*QR_000000001210+P_211110101*QR_000000001211);
			ans_temp[ans_id*6+1]+=Pmtrx[12]*(P_010011001*QR_001000000000+P_010011101*QR_001000000001+P_010111001*QR_001000000010+P_010111101*QR_001000000011+P_010211001*QR_001000000020+P_010211101*QR_001000000021+P_110011001*QR_001000000100+P_110011101*QR_001000000101+P_110111001*QR_001000000110+P_110111101*QR_001000000111+P_110211001*QR_001000000120+P_110211101*QR_001000000121);
			ans_temp[ans_id*6+1]+=Pmtrx[13]*(P_010011001*QR_000001000000+P_010011101*QR_000001000001+P_010111001*QR_000001000010+P_010111101*QR_000001000011+P_010211001*QR_000001000020+P_010211101*QR_000001000021+P_110011001*QR_000001000100+P_110011101*QR_000001000101+P_110111001*QR_000001000110+P_110111101*QR_000001000111+P_110211001*QR_000001000120+P_110211101*QR_000001000121);
			ans_temp[ans_id*6+1]+=Pmtrx[14]*(P_010011001*QR_000000001000+P_010011101*QR_000000001001+P_010111001*QR_000000001010+P_010111101*QR_000000001011+P_010211001*QR_000000001020+P_010211101*QR_000000001021+P_110011001*QR_000000001100+P_110011101*QR_000000001101+P_110111001*QR_000000001110+P_110111101*QR_000000001111+P_110211001*QR_000000001120+P_110211101*QR_000000001121);
			ans_temp[ans_id*6+1]+=Pmtrx[15]*(P_010010002*QR_001000000000+P_010010102*QR_001000000001+P_010010202*QR_001000000002+P_010110002*QR_001000000010+P_010110102*QR_001000000011+P_010110202*QR_001000000012+P_110010002*QR_001000000100+P_110010102*QR_001000000101+P_110010202*QR_001000000102+P_110110002*QR_001000000110+P_110110102*QR_001000000111+P_110110202*QR_001000000112);
			ans_temp[ans_id*6+1]+=Pmtrx[16]*(P_010010002*QR_000001000000+P_010010102*QR_000001000001+P_010010202*QR_000001000002+P_010110002*QR_000001000010+P_010110102*QR_000001000011+P_010110202*QR_000001000012+P_110010002*QR_000001000100+P_110010102*QR_000001000101+P_110010202*QR_000001000102+P_110110002*QR_000001000110+P_110110102*QR_000001000111+P_110110202*QR_000001000112);
			ans_temp[ans_id*6+1]+=Pmtrx[17]*(P_010010002*QR_000000001000+P_010010102*QR_000000001001+P_010010202*QR_000000001002+P_010110002*QR_000000001010+P_010110102*QR_000000001011+P_010110202*QR_000000001012+P_110010002*QR_000000001100+P_110010102*QR_000000001101+P_110010202*QR_000000001102+P_110110002*QR_000000001110+P_110110102*QR_000000001111+P_110110202*QR_000000001112);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_002020000*QR_001000000000+P_002120000*QR_001000000010+P_002220000*QR_001000000020+P_102020000*QR_001000000100+P_102120000*QR_001000000110+P_102220000*QR_001000000120+P_202020000*QR_001000000200+P_202120000*QR_001000000210+P_202220000*QR_001000000220);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_002020000*QR_000001000000+P_002120000*QR_000001000010+P_002220000*QR_000001000020+P_102020000*QR_000001000100+P_102120000*QR_000001000110+P_102220000*QR_000001000120+P_202020000*QR_000001000200+P_202120000*QR_000001000210+P_202220000*QR_000001000220);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_002020000*QR_000000001000+P_002120000*QR_000000001010+P_002220000*QR_000000001020+P_102020000*QR_000000001100+P_102120000*QR_000000001110+P_102220000*QR_000000001120+P_202020000*QR_000000001200+P_202120000*QR_000000001210+P_202220000*QR_000000001220);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_001021000*QR_001000000000+P_001121000*QR_001000000010+P_001221000*QR_001000000020+P_001321000*QR_001000000030+P_101021000*QR_001000000100+P_101121000*QR_001000000110+P_101221000*QR_001000000120+P_101321000*QR_001000000130);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_001021000*QR_000001000000+P_001121000*QR_000001000010+P_001221000*QR_000001000020+P_001321000*QR_000001000030+P_101021000*QR_000001000100+P_101121000*QR_000001000110+P_101221000*QR_000001000120+P_101321000*QR_000001000130);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_001021000*QR_000000001000+P_001121000*QR_000000001010+P_001221000*QR_000000001020+P_001321000*QR_000000001030+P_101021000*QR_000000001100+P_101121000*QR_000000001110+P_101221000*QR_000000001120+P_101321000*QR_000000001130);
			ans_temp[ans_id*6+2]+=Pmtrx[6]*(P_000022000*QR_001000000000+P_000122000*QR_001000000010+P_000222000*QR_001000000020+P_000322000*QR_001000000030+P_000422000*QR_001000000040);
			ans_temp[ans_id*6+2]+=Pmtrx[7]*(P_000022000*QR_000001000000+P_000122000*QR_000001000010+P_000222000*QR_000001000020+P_000322000*QR_000001000030+P_000422000*QR_000001000040);
			ans_temp[ans_id*6+2]+=Pmtrx[8]*(P_000022000*QR_000000001000+P_000122000*QR_000000001010+P_000222000*QR_000000001020+P_000322000*QR_000000001030+P_000422000*QR_000000001040);
			ans_temp[ans_id*6+2]+=Pmtrx[9]*(P_001020001*QR_001000000000+P_001020101*QR_001000000001+P_001120001*QR_001000000010+P_001120101*QR_001000000011+P_001220001*QR_001000000020+P_001220101*QR_001000000021+P_101020001*QR_001000000100+P_101020101*QR_001000000101+P_101120001*QR_001000000110+P_101120101*QR_001000000111+P_101220001*QR_001000000120+P_101220101*QR_001000000121);
			ans_temp[ans_id*6+2]+=Pmtrx[10]*(P_001020001*QR_000001000000+P_001020101*QR_000001000001+P_001120001*QR_000001000010+P_001120101*QR_000001000011+P_001220001*QR_000001000020+P_001220101*QR_000001000021+P_101020001*QR_000001000100+P_101020101*QR_000001000101+P_101120001*QR_000001000110+P_101120101*QR_000001000111+P_101220001*QR_000001000120+P_101220101*QR_000001000121);
			ans_temp[ans_id*6+2]+=Pmtrx[11]*(P_001020001*QR_000000001000+P_001020101*QR_000000001001+P_001120001*QR_000000001010+P_001120101*QR_000000001011+P_001220001*QR_000000001020+P_001220101*QR_000000001021+P_101020001*QR_000000001100+P_101020101*QR_000000001101+P_101120001*QR_000000001110+P_101120101*QR_000000001111+P_101220001*QR_000000001120+P_101220101*QR_000000001121);
			ans_temp[ans_id*6+2]+=Pmtrx[12]*(P_000021001*QR_001000000000+P_000021101*QR_001000000001+P_000121001*QR_001000000010+P_000121101*QR_001000000011+P_000221001*QR_001000000020+P_000221101*QR_001000000021+P_000321001*QR_001000000030+P_000321101*QR_001000000031);
			ans_temp[ans_id*6+2]+=Pmtrx[13]*(P_000021001*QR_000001000000+P_000021101*QR_000001000001+P_000121001*QR_000001000010+P_000121101*QR_000001000011+P_000221001*QR_000001000020+P_000221101*QR_000001000021+P_000321001*QR_000001000030+P_000321101*QR_000001000031);
			ans_temp[ans_id*6+2]+=Pmtrx[14]*(P_000021001*QR_000000001000+P_000021101*QR_000000001001+P_000121001*QR_000000001010+P_000121101*QR_000000001011+P_000221001*QR_000000001020+P_000221101*QR_000000001021+P_000321001*QR_000000001030+P_000321101*QR_000000001031);
			ans_temp[ans_id*6+2]+=Pmtrx[15]*(P_000020002*QR_001000000000+P_000020102*QR_001000000001+P_000020202*QR_001000000002+P_000120002*QR_001000000010+P_000120102*QR_001000000011+P_000120202*QR_001000000012+P_000220002*QR_001000000020+P_000220102*QR_001000000021+P_000220202*QR_001000000022);
			ans_temp[ans_id*6+2]+=Pmtrx[16]*(P_000020002*QR_000001000000+P_000020102*QR_000001000001+P_000020202*QR_000001000002+P_000120002*QR_000001000010+P_000120102*QR_000001000011+P_000120202*QR_000001000012+P_000220002*QR_000001000020+P_000220102*QR_000001000021+P_000220202*QR_000001000022);
			ans_temp[ans_id*6+2]+=Pmtrx[17]*(P_000020002*QR_000000001000+P_000020102*QR_000000001001+P_000020202*QR_000000001002+P_000120002*QR_000000001010+P_000120102*QR_000000001011+P_000120202*QR_000000001012+P_000220002*QR_000000001020+P_000220102*QR_000000001021+P_000220202*QR_000000001022);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_012000010*QR_001000000000+P_012000110*QR_001000000001+P_112000010*QR_001000000100+P_112000110*QR_001000000101+P_212000010*QR_001000000200+P_212000110*QR_001000000201+P_312000010*QR_001000000300+P_312000110*QR_001000000301);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_012000010*QR_000001000000+P_012000110*QR_000001000001+P_112000010*QR_000001000100+P_112000110*QR_000001000101+P_212000010*QR_000001000200+P_212000110*QR_000001000201+P_312000010*QR_000001000300+P_312000110*QR_000001000301);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_012000010*QR_000000001000+P_012000110*QR_000000001001+P_112000010*QR_000000001100+P_112000110*QR_000000001101+P_212000010*QR_000000001200+P_212000110*QR_000000001201+P_312000010*QR_000000001300+P_312000110*QR_000000001301);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_011001010*QR_001000000000+P_011001110*QR_001000000001+P_011101010*QR_001000000010+P_011101110*QR_001000000011+P_111001010*QR_001000000100+P_111001110*QR_001000000101+P_111101010*QR_001000000110+P_111101110*QR_001000000111+P_211001010*QR_001000000200+P_211001110*QR_001000000201+P_211101010*QR_001000000210+P_211101110*QR_001000000211);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_011001010*QR_000001000000+P_011001110*QR_000001000001+P_011101010*QR_000001000010+P_011101110*QR_000001000011+P_111001010*QR_000001000100+P_111001110*QR_000001000101+P_111101010*QR_000001000110+P_111101110*QR_000001000111+P_211001010*QR_000001000200+P_211001110*QR_000001000201+P_211101010*QR_000001000210+P_211101110*QR_000001000211);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_011001010*QR_000000001000+P_011001110*QR_000000001001+P_011101010*QR_000000001010+P_011101110*QR_000000001011+P_111001010*QR_000000001100+P_111001110*QR_000000001101+P_111101010*QR_000000001110+P_111101110*QR_000000001111+P_211001010*QR_000000001200+P_211001110*QR_000000001201+P_211101010*QR_000000001210+P_211101110*QR_000000001211);
			ans_temp[ans_id*6+3]+=Pmtrx[6]*(P_010002010*QR_001000000000+P_010002110*QR_001000000001+P_010102010*QR_001000000010+P_010102110*QR_001000000011+P_010202010*QR_001000000020+P_010202110*QR_001000000021+P_110002010*QR_001000000100+P_110002110*QR_001000000101+P_110102010*QR_001000000110+P_110102110*QR_001000000111+P_110202010*QR_001000000120+P_110202110*QR_001000000121);
			ans_temp[ans_id*6+3]+=Pmtrx[7]*(P_010002010*QR_000001000000+P_010002110*QR_000001000001+P_010102010*QR_000001000010+P_010102110*QR_000001000011+P_010202010*QR_000001000020+P_010202110*QR_000001000021+P_110002010*QR_000001000100+P_110002110*QR_000001000101+P_110102010*QR_000001000110+P_110102110*QR_000001000111+P_110202010*QR_000001000120+P_110202110*QR_000001000121);
			ans_temp[ans_id*6+3]+=Pmtrx[8]*(P_010002010*QR_000000001000+P_010002110*QR_000000001001+P_010102010*QR_000000001010+P_010102110*QR_000000001011+P_010202010*QR_000000001020+P_010202110*QR_000000001021+P_110002010*QR_000000001100+P_110002110*QR_000000001101+P_110102010*QR_000000001110+P_110102110*QR_000000001111+P_110202010*QR_000000001120+P_110202110*QR_000000001121);
			ans_temp[ans_id*6+3]+=Pmtrx[9]*(P_011000011*QR_001000000000+P_011000111*QR_001000000001+P_011000211*QR_001000000002+P_111000011*QR_001000000100+P_111000111*QR_001000000101+P_111000211*QR_001000000102+P_211000011*QR_001000000200+P_211000111*QR_001000000201+P_211000211*QR_001000000202);
			ans_temp[ans_id*6+3]+=Pmtrx[10]*(P_011000011*QR_000001000000+P_011000111*QR_000001000001+P_011000211*QR_000001000002+P_111000011*QR_000001000100+P_111000111*QR_000001000101+P_111000211*QR_000001000102+P_211000011*QR_000001000200+P_211000111*QR_000001000201+P_211000211*QR_000001000202);
			ans_temp[ans_id*6+3]+=Pmtrx[11]*(P_011000011*QR_000000001000+P_011000111*QR_000000001001+P_011000211*QR_000000001002+P_111000011*QR_000000001100+P_111000111*QR_000000001101+P_111000211*QR_000000001102+P_211000011*QR_000000001200+P_211000111*QR_000000001201+P_211000211*QR_000000001202);
			ans_temp[ans_id*6+3]+=Pmtrx[12]*(P_010001011*QR_001000000000+P_010001111*QR_001000000001+P_010001211*QR_001000000002+P_010101011*QR_001000000010+P_010101111*QR_001000000011+P_010101211*QR_001000000012+P_110001011*QR_001000000100+P_110001111*QR_001000000101+P_110001211*QR_001000000102+P_110101011*QR_001000000110+P_110101111*QR_001000000111+P_110101211*QR_001000000112);
			ans_temp[ans_id*6+3]+=Pmtrx[13]*(P_010001011*QR_000001000000+P_010001111*QR_000001000001+P_010001211*QR_000001000002+P_010101011*QR_000001000010+P_010101111*QR_000001000011+P_010101211*QR_000001000012+P_110001011*QR_000001000100+P_110001111*QR_000001000101+P_110001211*QR_000001000102+P_110101011*QR_000001000110+P_110101111*QR_000001000111+P_110101211*QR_000001000112);
			ans_temp[ans_id*6+3]+=Pmtrx[14]*(P_010001011*QR_000000001000+P_010001111*QR_000000001001+P_010001211*QR_000000001002+P_010101011*QR_000000001010+P_010101111*QR_000000001011+P_010101211*QR_000000001012+P_110001011*QR_000000001100+P_110001111*QR_000000001101+P_110001211*QR_000000001102+P_110101011*QR_000000001110+P_110101111*QR_000000001111+P_110101211*QR_000000001112);
			ans_temp[ans_id*6+3]+=Pmtrx[15]*(P_010000012*QR_001000000000+P_010000112*QR_001000000001+P_010000212*QR_001000000002+P_010000312*QR_001000000003+P_110000012*QR_001000000100+P_110000112*QR_001000000101+P_110000212*QR_001000000102+P_110000312*QR_001000000103);
			ans_temp[ans_id*6+3]+=Pmtrx[16]*(P_010000012*QR_000001000000+P_010000112*QR_000001000001+P_010000212*QR_000001000002+P_010000312*QR_000001000003+P_110000012*QR_000001000100+P_110000112*QR_000001000101+P_110000212*QR_000001000102+P_110000312*QR_000001000103);
			ans_temp[ans_id*6+3]+=Pmtrx[17]*(P_010000012*QR_000000001000+P_010000112*QR_000000001001+P_010000212*QR_000000001002+P_010000312*QR_000000001003+P_110000012*QR_000000001100+P_110000112*QR_000000001101+P_110000212*QR_000000001102+P_110000312*QR_000000001103);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_002010010*QR_001000000000+P_002010110*QR_001000000001+P_002110010*QR_001000000010+P_002110110*QR_001000000011+P_102010010*QR_001000000100+P_102010110*QR_001000000101+P_102110010*QR_001000000110+P_102110110*QR_001000000111+P_202010010*QR_001000000200+P_202010110*QR_001000000201+P_202110010*QR_001000000210+P_202110110*QR_001000000211);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_002010010*QR_000001000000+P_002010110*QR_000001000001+P_002110010*QR_000001000010+P_002110110*QR_000001000011+P_102010010*QR_000001000100+P_102010110*QR_000001000101+P_102110010*QR_000001000110+P_102110110*QR_000001000111+P_202010010*QR_000001000200+P_202010110*QR_000001000201+P_202110010*QR_000001000210+P_202110110*QR_000001000211);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_002010010*QR_000000001000+P_002010110*QR_000000001001+P_002110010*QR_000000001010+P_002110110*QR_000000001011+P_102010010*QR_000000001100+P_102010110*QR_000000001101+P_102110010*QR_000000001110+P_102110110*QR_000000001111+P_202010010*QR_000000001200+P_202010110*QR_000000001201+P_202110010*QR_000000001210+P_202110110*QR_000000001211);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_001011010*QR_001000000000+P_001011110*QR_001000000001+P_001111010*QR_001000000010+P_001111110*QR_001000000011+P_001211010*QR_001000000020+P_001211110*QR_001000000021+P_101011010*QR_001000000100+P_101011110*QR_001000000101+P_101111010*QR_001000000110+P_101111110*QR_001000000111+P_101211010*QR_001000000120+P_101211110*QR_001000000121);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_001011010*QR_000001000000+P_001011110*QR_000001000001+P_001111010*QR_000001000010+P_001111110*QR_000001000011+P_001211010*QR_000001000020+P_001211110*QR_000001000021+P_101011010*QR_000001000100+P_101011110*QR_000001000101+P_101111010*QR_000001000110+P_101111110*QR_000001000111+P_101211010*QR_000001000120+P_101211110*QR_000001000121);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_001011010*QR_000000001000+P_001011110*QR_000000001001+P_001111010*QR_000000001010+P_001111110*QR_000000001011+P_001211010*QR_000000001020+P_001211110*QR_000000001021+P_101011010*QR_000000001100+P_101011110*QR_000000001101+P_101111010*QR_000000001110+P_101111110*QR_000000001111+P_101211010*QR_000000001120+P_101211110*QR_000000001121);
			ans_temp[ans_id*6+4]+=Pmtrx[6]*(P_000012010*QR_001000000000+P_000012110*QR_001000000001+P_000112010*QR_001000000010+P_000112110*QR_001000000011+P_000212010*QR_001000000020+P_000212110*QR_001000000021+P_000312010*QR_001000000030+P_000312110*QR_001000000031);
			ans_temp[ans_id*6+4]+=Pmtrx[7]*(P_000012010*QR_000001000000+P_000012110*QR_000001000001+P_000112010*QR_000001000010+P_000112110*QR_000001000011+P_000212010*QR_000001000020+P_000212110*QR_000001000021+P_000312010*QR_000001000030+P_000312110*QR_000001000031);
			ans_temp[ans_id*6+4]+=Pmtrx[8]*(P_000012010*QR_000000001000+P_000012110*QR_000000001001+P_000112010*QR_000000001010+P_000112110*QR_000000001011+P_000212010*QR_000000001020+P_000212110*QR_000000001021+P_000312010*QR_000000001030+P_000312110*QR_000000001031);
			ans_temp[ans_id*6+4]+=Pmtrx[9]*(P_001010011*QR_001000000000+P_001010111*QR_001000000001+P_001010211*QR_001000000002+P_001110011*QR_001000000010+P_001110111*QR_001000000011+P_001110211*QR_001000000012+P_101010011*QR_001000000100+P_101010111*QR_001000000101+P_101010211*QR_001000000102+P_101110011*QR_001000000110+P_101110111*QR_001000000111+P_101110211*QR_001000000112);
			ans_temp[ans_id*6+4]+=Pmtrx[10]*(P_001010011*QR_000001000000+P_001010111*QR_000001000001+P_001010211*QR_000001000002+P_001110011*QR_000001000010+P_001110111*QR_000001000011+P_001110211*QR_000001000012+P_101010011*QR_000001000100+P_101010111*QR_000001000101+P_101010211*QR_000001000102+P_101110011*QR_000001000110+P_101110111*QR_000001000111+P_101110211*QR_000001000112);
			ans_temp[ans_id*6+4]+=Pmtrx[11]*(P_001010011*QR_000000001000+P_001010111*QR_000000001001+P_001010211*QR_000000001002+P_001110011*QR_000000001010+P_001110111*QR_000000001011+P_001110211*QR_000000001012+P_101010011*QR_000000001100+P_101010111*QR_000000001101+P_101010211*QR_000000001102+P_101110011*QR_000000001110+P_101110111*QR_000000001111+P_101110211*QR_000000001112);
			ans_temp[ans_id*6+4]+=Pmtrx[12]*(P_000011011*QR_001000000000+P_000011111*QR_001000000001+P_000011211*QR_001000000002+P_000111011*QR_001000000010+P_000111111*QR_001000000011+P_000111211*QR_001000000012+P_000211011*QR_001000000020+P_000211111*QR_001000000021+P_000211211*QR_001000000022);
			ans_temp[ans_id*6+4]+=Pmtrx[13]*(P_000011011*QR_000001000000+P_000011111*QR_000001000001+P_000011211*QR_000001000002+P_000111011*QR_000001000010+P_000111111*QR_000001000011+P_000111211*QR_000001000012+P_000211011*QR_000001000020+P_000211111*QR_000001000021+P_000211211*QR_000001000022);
			ans_temp[ans_id*6+4]+=Pmtrx[14]*(P_000011011*QR_000000001000+P_000011111*QR_000000001001+P_000011211*QR_000000001002+P_000111011*QR_000000001010+P_000111111*QR_000000001011+P_000111211*QR_000000001012+P_000211011*QR_000000001020+P_000211111*QR_000000001021+P_000211211*QR_000000001022);
			ans_temp[ans_id*6+4]+=Pmtrx[15]*(P_000010012*QR_001000000000+P_000010112*QR_001000000001+P_000010212*QR_001000000002+P_000010312*QR_001000000003+P_000110012*QR_001000000010+P_000110112*QR_001000000011+P_000110212*QR_001000000012+P_000110312*QR_001000000013);
			ans_temp[ans_id*6+4]+=Pmtrx[16]*(P_000010012*QR_000001000000+P_000010112*QR_000001000001+P_000010212*QR_000001000002+P_000010312*QR_000001000003+P_000110012*QR_000001000010+P_000110112*QR_000001000011+P_000110212*QR_000001000012+P_000110312*QR_000001000013);
			ans_temp[ans_id*6+4]+=Pmtrx[17]*(P_000010012*QR_000000001000+P_000010112*QR_000000001001+P_000010212*QR_000000001002+P_000010312*QR_000000001003+P_000110012*QR_000000001010+P_000110112*QR_000000001011+P_000110212*QR_000000001012+P_000110312*QR_000000001013);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_002000020*QR_001000000000+P_002000120*QR_001000000001+P_002000220*QR_001000000002+P_102000020*QR_001000000100+P_102000120*QR_001000000101+P_102000220*QR_001000000102+P_202000020*QR_001000000200+P_202000120*QR_001000000201+P_202000220*QR_001000000202);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_002000020*QR_000001000000+P_002000120*QR_000001000001+P_002000220*QR_000001000002+P_102000020*QR_000001000100+P_102000120*QR_000001000101+P_102000220*QR_000001000102+P_202000020*QR_000001000200+P_202000120*QR_000001000201+P_202000220*QR_000001000202);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_002000020*QR_000000001000+P_002000120*QR_000000001001+P_002000220*QR_000000001002+P_102000020*QR_000000001100+P_102000120*QR_000000001101+P_102000220*QR_000000001102+P_202000020*QR_000000001200+P_202000120*QR_000000001201+P_202000220*QR_000000001202);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_001001020*QR_001000000000+P_001001120*QR_001000000001+P_001001220*QR_001000000002+P_001101020*QR_001000000010+P_001101120*QR_001000000011+P_001101220*QR_001000000012+P_101001020*QR_001000000100+P_101001120*QR_001000000101+P_101001220*QR_001000000102+P_101101020*QR_001000000110+P_101101120*QR_001000000111+P_101101220*QR_001000000112);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_001001020*QR_000001000000+P_001001120*QR_000001000001+P_001001220*QR_000001000002+P_001101020*QR_000001000010+P_001101120*QR_000001000011+P_001101220*QR_000001000012+P_101001020*QR_000001000100+P_101001120*QR_000001000101+P_101001220*QR_000001000102+P_101101020*QR_000001000110+P_101101120*QR_000001000111+P_101101220*QR_000001000112);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_001001020*QR_000000001000+P_001001120*QR_000000001001+P_001001220*QR_000000001002+P_001101020*QR_000000001010+P_001101120*QR_000000001011+P_001101220*QR_000000001012+P_101001020*QR_000000001100+P_101001120*QR_000000001101+P_101001220*QR_000000001102+P_101101020*QR_000000001110+P_101101120*QR_000000001111+P_101101220*QR_000000001112);
			ans_temp[ans_id*6+5]+=Pmtrx[6]*(P_000002020*QR_001000000000+P_000002120*QR_001000000001+P_000002220*QR_001000000002+P_000102020*QR_001000000010+P_000102120*QR_001000000011+P_000102220*QR_001000000012+P_000202020*QR_001000000020+P_000202120*QR_001000000021+P_000202220*QR_001000000022);
			ans_temp[ans_id*6+5]+=Pmtrx[7]*(P_000002020*QR_000001000000+P_000002120*QR_000001000001+P_000002220*QR_000001000002+P_000102020*QR_000001000010+P_000102120*QR_000001000011+P_000102220*QR_000001000012+P_000202020*QR_000001000020+P_000202120*QR_000001000021+P_000202220*QR_000001000022);
			ans_temp[ans_id*6+5]+=Pmtrx[8]*(P_000002020*QR_000000001000+P_000002120*QR_000000001001+P_000002220*QR_000000001002+P_000102020*QR_000000001010+P_000102120*QR_000000001011+P_000102220*QR_000000001012+P_000202020*QR_000000001020+P_000202120*QR_000000001021+P_000202220*QR_000000001022);
			ans_temp[ans_id*6+5]+=Pmtrx[9]*(P_001000021*QR_001000000000+P_001000121*QR_001000000001+P_001000221*QR_001000000002+P_001000321*QR_001000000003+P_101000021*QR_001000000100+P_101000121*QR_001000000101+P_101000221*QR_001000000102+P_101000321*QR_001000000103);
			ans_temp[ans_id*6+5]+=Pmtrx[10]*(P_001000021*QR_000001000000+P_001000121*QR_000001000001+P_001000221*QR_000001000002+P_001000321*QR_000001000003+P_101000021*QR_000001000100+P_101000121*QR_000001000101+P_101000221*QR_000001000102+P_101000321*QR_000001000103);
			ans_temp[ans_id*6+5]+=Pmtrx[11]*(P_001000021*QR_000000001000+P_001000121*QR_000000001001+P_001000221*QR_000000001002+P_001000321*QR_000000001003+P_101000021*QR_000000001100+P_101000121*QR_000000001101+P_101000221*QR_000000001102+P_101000321*QR_000000001103);
			ans_temp[ans_id*6+5]+=Pmtrx[12]*(P_000001021*QR_001000000000+P_000001121*QR_001000000001+P_000001221*QR_001000000002+P_000001321*QR_001000000003+P_000101021*QR_001000000010+P_000101121*QR_001000000011+P_000101221*QR_001000000012+P_000101321*QR_001000000013);
			ans_temp[ans_id*6+5]+=Pmtrx[13]*(P_000001021*QR_000001000000+P_000001121*QR_000001000001+P_000001221*QR_000001000002+P_000001321*QR_000001000003+P_000101021*QR_000001000010+P_000101121*QR_000001000011+P_000101221*QR_000001000012+P_000101321*QR_000001000013);
			ans_temp[ans_id*6+5]+=Pmtrx[14]*(P_000001021*QR_000000001000+P_000001121*QR_000000001001+P_000001221*QR_000000001002+P_000001321*QR_000000001003+P_000101021*QR_000000001010+P_000101121*QR_000000001011+P_000101221*QR_000000001012+P_000101321*QR_000000001013);
			ans_temp[ans_id*6+5]+=Pmtrx[15]*(P_000000022*QR_001000000000+P_000000122*QR_001000000001+P_000000222*QR_001000000002+P_000000322*QR_001000000003+P_000000422*QR_001000000004);
			ans_temp[ans_id*6+5]+=Pmtrx[16]*(P_000000022*QR_000001000000+P_000000122*QR_000001000001+P_000000222*QR_000001000002+P_000000322*QR_000001000003+P_000000422*QR_000001000004);
			ans_temp[ans_id*6+5]+=Pmtrx[17]*(P_000000022*QR_000000001000+P_000000122*QR_000000001001+P_000000222*QR_000000001002+P_000000322*QR_000000001003+P_000000422*QR_000000001004);
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
