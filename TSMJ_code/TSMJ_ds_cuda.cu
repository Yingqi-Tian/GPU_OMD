#include<math.h>
#include"Boys_gpu.h"
#define PI 3.1415926535897932
#define P25 17.4934183276248620
#define NTHREAD_32 32
#define NTHREAD_64 64
#define NTHREAD_128 128
texture<int2,1,cudaReadModeElementType> tex_Q;
texture<int2,1,cudaReadModeElementType> tex_Eta;
texture<int2,1,cudaReadModeElementType> tex_pq;
texture<float,1,cudaReadModeElementType> tex_K2_q;
texture<int2,1,cudaReadModeElementType> tex_QC;
texture<int2,1,cudaReadModeElementType> tex_Pmtrx;

void TSMJ_texture_binding_ds(double * Q_d,double * QC_d,double * QD_d,\
        double * alphaQ_d,double * pq_d,float * K2_q_d,double * Pmtrx_d,\
        unsigned int contrc_ket_start_pr,unsigned int primit_ket_len,unsigned int contrc_Pmtrx_start_pr){
    cudaBindTexture(0, tex_Q, Q_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);
    cudaBindTexture(0, tex_Eta, alphaQ_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_pq, pq_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_K2_q, K2_q_d, sizeof(float)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_QC, QC_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);
    cudaBindTexture(0, tex_Pmtrx, Pmtrx_d, sizeof(double)*(contrc_Pmtrx_start_pr+primit_ket_len)*6);
}

void TSMJ_texture_unbind_ds(){
    cudaUnbindTexture(tex_Q);
    cudaUnbindTexture(tex_Eta);
    cudaUnbindTexture(tex_pq);
    cudaUnbindTexture(tex_K2_q);
    cudaUnbindTexture(tex_QC);
    cudaUnbindTexture(tex_Pmtrx);

}
__global__ void TSMJ_ssds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*1];
    for(int i=0;i<1;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
				double QD_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            QD_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            QD_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            QD_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_taylor(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
	R_000[1]*=aQin1;
	R_000[2]*=aQin2;
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
	for(int i=1;i<2;i++){
		R_000[i]*=aQin1;
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
		double QD_020[3];
		for(int i=0;i<3;i++){
			QD_020[i]=aQin1+QD_010[i]*QD_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=QD_020[0];
			Q_010010000=QD_010[0]*QD_010[1];
			Q_000020000=QD_020[1];
			Q_010000010=QD_010[0]*QD_010[2];
			Q_000010010=QD_010[1]*QD_010[2];
			Q_000000020=QD_020[2];
			a1Q_010000000_1=QD_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=QD_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=QD_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+R_002[0];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(QR_020000000000);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(QR_010010000000);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(QR_000020000000);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(QR_010000010000);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(QR_000010010000);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(QR_000000020000);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
}
__global__ void TSMJ_ssds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*1];
    for(int i=0;i<1;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double QD_010[3];
				QD_010[0]=QC[jj*3+0];
				QD_010[1]=QC[jj*3+1];
				QD_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
			double aQin2=aQin1*aQin1;
	R_000[1]*=aQin1;
	R_000[2]*=aQin2;
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
	for(int i=1;i<2;i++){
		R_000[i]*=aQin1;
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
		double QD_020[3];
		for(int i=0;i<3;i++){
			QD_020[i]=aQin1+QD_010[i]*QD_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=QD_020[0];
			Q_010010000=QD_010[0]*QD_010[1];
			Q_000020000=QD_020[1];
			Q_010000010=QD_010[0]*QD_010[2];
			Q_000010010=QD_010[1]*QD_010[2];
			Q_000000020=QD_020[2];
			a1Q_010000000_1=QD_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=QD_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=QD_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+R_002[0];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(QR_020000000000);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(QR_010010000000);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(QR_000020000000);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(QR_010000010000);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(QR_000010010000);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(QR_000000020000);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
}
__global__ void TSMJ_ssds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*1];
    for(int i=0;i<1;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
				double QD_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            QD_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            QD_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            QD_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
			double aQin2=aQin1*aQin1;
	R_000[1]*=aQin1;
	R_000[2]*=aQin2;
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
	for(int i=1;i<2;i++){
		R_000[i]*=aQin1;
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
		double QD_020[3];
		for(int i=0;i<3;i++){
			QD_020[i]=aQin1+QD_010[i]*QD_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=QD_020[0];
			Q_010010000=QD_010[0]*QD_010[1];
			Q_000020000=QD_020[1];
			Q_010000010=QD_010[0]*QD_010[2];
			Q_000010010=QD_010[1]*QD_010[2];
			Q_000000020=QD_020[2];
			a1Q_010000000_1=QD_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=QD_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=QD_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+R_002[0];
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(QR_020000000000);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(QR_010010000000);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(QR_000020000000);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(QR_010000010000);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(QR_000010010000);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(QR_000000020000);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
}
__global__ void TSMJ_ssds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*1];
    for(int i=0;i<1;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
				double QD_010[3];
            temp_int2=tex1Dfetch(tex_QC,jj*3+0);
            QD_010[0]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+1);
            QD_010[1]=__hiloint2double(temp_int2.y,temp_int2.x);
            temp_int2=tex1Dfetch(tex_QC,jj*3+2);
            QD_010[2]=__hiloint2double(temp_int2.y,temp_int2.x);
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
			double aQin2=aQin1*aQin1;
	R_000[1]*=aQin1;
	R_000[2]*=aQin2;
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
	for(int i=1;i<2;i++){
		R_000[i]*=aQin1;
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
		double QD_020[3];
		for(int i=0;i<3;i++){
			QD_020[i]=aQin1+QD_010[i]*QD_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=QD_020[0];
			Q_010010000=QD_010[0]*QD_010[1];
			Q_000020000=QD_020[1];
			Q_010000010=QD_010[0]*QD_010[2];
			Q_000010010=QD_010[1]*QD_010[2];
			Q_000000020=QD_020[2];
			a1Q_010000000_1=QD_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=QD_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=QD_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+R_002[0];
			}
			ans_temp[ans_id*1+0]+=Pmtrx[0]*(QR_020000000000);
			ans_temp[ans_id*1+0]+=Pmtrx[1]*(QR_010010000000);
			ans_temp[ans_id*1+0]+=Pmtrx[2]*(QR_000020000000);
			ans_temp[ans_id*1+0]+=Pmtrx[3]*(QR_010000010000);
			ans_temp[ans_id*1+0]+=Pmtrx[4]*(QR_000010010000);
			ans_temp[ans_id*1+0]+=Pmtrx[5]*(QR_000000020000);
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*1+ians]=ans_temp[(tId_x)*1+ians];
            }
        }
	}
}
__global__ void TSMJ_psds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*3];
    for(int i=0;i<3;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[4];
                Ft_taylor(3,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=Pd_010[0];
			P_000010000=Pd_010[1];
			P_000000010=Pd_010[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_010000000*QR_020000000000+aPin1*QR_020000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_010000000*QR_010010000000+aPin1*QR_010010000100);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_010000000*QR_000020000000+aPin1*QR_000020000100);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_010000000*QR_010000010000+aPin1*QR_010000010100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_010000000*QR_000010010000+aPin1*QR_000010010100);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_010000000*QR_000000020000+aPin1*QR_000000020100);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_000010000*QR_020000000000+aPin1*QR_020000000010);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_000010000*QR_010010000000+aPin1*QR_010010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_000010000*QR_000020000000+aPin1*QR_000020000010);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_000010000*QR_010000010000+aPin1*QR_010000010010);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_000010000*QR_000010010000+aPin1*QR_000010010010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_000010000*QR_000000020000+aPin1*QR_000000020010);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_000000010*QR_020000000000+aPin1*QR_020000000001);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_000000010*QR_010010000000+aPin1*QR_010010000001);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_000000010*QR_000020000000+aPin1*QR_000020000001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_000000010*QR_010000010000+aPin1*QR_010000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_000000010*QR_000010010000+aPin1*QR_000010010001);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_000000010*QR_000000020000+aPin1*QR_000000020001);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
}
__global__ void TSMJ_psds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*3];
    for(int i=0;i<3;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[jj*3+0];
				Qd_010[1]=QC[jj*3+1];
				Qd_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=Pd_010[0];
			P_000010000=Pd_010[1];
			P_000000010=Pd_010[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_010000000*QR_020000000000+aPin1*QR_020000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_010000000*QR_010010000000+aPin1*QR_010010000100);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_010000000*QR_000020000000+aPin1*QR_000020000100);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_010000000*QR_010000010000+aPin1*QR_010000010100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_010000000*QR_000010010000+aPin1*QR_000010010100);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_010000000*QR_000000020000+aPin1*QR_000000020100);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_000010000*QR_020000000000+aPin1*QR_020000000010);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_000010000*QR_010010000000+aPin1*QR_010010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_000010000*QR_000020000000+aPin1*QR_000020000010);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_000010000*QR_010000010000+aPin1*QR_010000010010);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_000010000*QR_000010010000+aPin1*QR_000010010010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_000010000*QR_000000020000+aPin1*QR_000000020010);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_000000010*QR_020000000000+aPin1*QR_020000000001);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_000000010*QR_010010000000+aPin1*QR_010010000001);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_000000010*QR_000020000000+aPin1*QR_000020000001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_000000010*QR_010000010000+aPin1*QR_010000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_000000010*QR_000010010000+aPin1*QR_000010010001);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_000000010*QR_000000020000+aPin1*QR_000000020001);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
}
__global__ void TSMJ_psds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*3];
    for(int i=0;i<3;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=Pd_010[0];
			P_000010000=Pd_010[1];
			P_000000010=Pd_010[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_010000000*QR_020000000000+aPin1*QR_020000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_010000000*QR_010010000000+aPin1*QR_010010000100);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_010000000*QR_000020000000+aPin1*QR_000020000100);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_010000000*QR_010000010000+aPin1*QR_010000010100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_010000000*QR_000010010000+aPin1*QR_000010010100);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_010000000*QR_000000020000+aPin1*QR_000000020100);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_000010000*QR_020000000000+aPin1*QR_020000000010);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_000010000*QR_010010000000+aPin1*QR_010010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_000010000*QR_000020000000+aPin1*QR_000020000010);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_000010000*QR_010000010000+aPin1*QR_010000010010);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_000010000*QR_000010010000+aPin1*QR_000010010010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_000010000*QR_000000020000+aPin1*QR_000000020010);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_000000010*QR_020000000000+aPin1*QR_020000000001);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_000000010*QR_010010000000+aPin1*QR_010010000001);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_000000010*QR_000020000000+aPin1*QR_000020000001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_000000010*QR_010000010000+aPin1*QR_010000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_000000010*QR_000010010000+aPin1*QR_000010010001);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_000000010*QR_000000020000+aPin1*QR_000000020001);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
}
__global__ void TSMJ_psds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*3];
    for(int i=0;i<3;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
			double QR_020000000001=0;
			double QR_010010000001=0;
			double QR_000020000001=0;
			double QR_010000010001=0;
			double QR_000010010001=0;
			double QR_000000020001=0;
			double QR_020000000010=0;
			double QR_010010000010=0;
			double QR_000020000010=0;
			double QR_010000010010=0;
			double QR_000010010010=0;
			double QR_000000020010=0;
			double QR_020000000100=0;
			double QR_010010000100=0;
			double QR_000020000100=0;
			double QR_010000010100=0;
			double QR_000010010100=0;
			double QR_000000020100=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			QR_020000000001+=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			QR_010010000001+=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			QR_000020000001+=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			QR_010000010001+=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			QR_000010010001+=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			QR_000000020001+=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			QR_020000000010+=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			QR_010010000010+=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			QR_000020000010+=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			QR_010000010010+=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000010010010+=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			QR_000000020010+=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			QR_020000000100+=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			QR_010010000100+=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			QR_000020000100+=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			QR_010000010100+=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			QR_000010010100+=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000000020100+=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			}
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=Pd_010[0];
			P_000010000=Pd_010[1];
			P_000000010=Pd_010[2];
			ans_temp[ans_id*3+0]+=Pmtrx[0]*(P_010000000*QR_020000000000+aPin1*QR_020000000100);
			ans_temp[ans_id*3+0]+=Pmtrx[1]*(P_010000000*QR_010010000000+aPin1*QR_010010000100);
			ans_temp[ans_id*3+0]+=Pmtrx[2]*(P_010000000*QR_000020000000+aPin1*QR_000020000100);
			ans_temp[ans_id*3+0]+=Pmtrx[3]*(P_010000000*QR_010000010000+aPin1*QR_010000010100);
			ans_temp[ans_id*3+0]+=Pmtrx[4]*(P_010000000*QR_000010010000+aPin1*QR_000010010100);
			ans_temp[ans_id*3+0]+=Pmtrx[5]*(P_010000000*QR_000000020000+aPin1*QR_000000020100);
			ans_temp[ans_id*3+1]+=Pmtrx[0]*(P_000010000*QR_020000000000+aPin1*QR_020000000010);
			ans_temp[ans_id*3+1]+=Pmtrx[1]*(P_000010000*QR_010010000000+aPin1*QR_010010000010);
			ans_temp[ans_id*3+1]+=Pmtrx[2]*(P_000010000*QR_000020000000+aPin1*QR_000020000010);
			ans_temp[ans_id*3+1]+=Pmtrx[3]*(P_000010000*QR_010000010000+aPin1*QR_010000010010);
			ans_temp[ans_id*3+1]+=Pmtrx[4]*(P_000010000*QR_000010010000+aPin1*QR_000010010010);
			ans_temp[ans_id*3+1]+=Pmtrx[5]*(P_000010000*QR_000000020000+aPin1*QR_000000020010);
			ans_temp[ans_id*3+2]+=Pmtrx[0]*(P_000000010*QR_020000000000+aPin1*QR_020000000001);
			ans_temp[ans_id*3+2]+=Pmtrx[1]*(P_000000010*QR_010010000000+aPin1*QR_010010000001);
			ans_temp[ans_id*3+2]+=Pmtrx[2]*(P_000000010*QR_000020000000+aPin1*QR_000020000001);
			ans_temp[ans_id*3+2]+=Pmtrx[3]*(P_000000010*QR_010000010000+aPin1*QR_010000010001);
			ans_temp[ans_id*3+2]+=Pmtrx[4]*(P_000000010*QR_000010010000+aPin1*QR_000010010001);
			ans_temp[ans_id*3+2]+=Pmtrx[5]*(P_000000010*QR_000000020000+aPin1*QR_000000020001);
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*3+ians]=ans_temp[(tId_x)*3+ians];
            }
        }
	}
}
__global__ void TSMJ_ppds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*9];
    for(int i=0;i<9;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_taylor(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_011[3];
		double Pd_111[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
			double P_011000000;
			double P_111000000;
			double P_010001000;
			double P_010000001;
			double P_001010000;
			double P_000011000;
			double P_000111000;
			double P_000010001;
			double P_001000010;
			double P_000001010;
			double P_000000011;
			double P_000000111;
			double a1P_010000000_1;
			double a1P_000001000_1;
			double a1P_000000001_1;
			double a1P_001000000_1;
			double a1P_000010000_1;
			double a1P_000000010_1;
			P_011000000=Pd_011[0];
			P_111000000=Pd_111[0];
			P_010001000=Pd_010[0]*Pd_001[1];
			P_010000001=Pd_010[0]*Pd_001[2];
			P_001010000=Pd_001[0]*Pd_010[1];
			P_000011000=Pd_011[1];
			P_000111000=Pd_111[1];
			P_000010001=Pd_010[1]*Pd_001[2];
			P_001000010=Pd_001[0]*Pd_010[2];
			P_000001010=Pd_001[1]*Pd_010[2];
			P_000000011=Pd_011[2];
			P_000000111=Pd_111[2];
			a1P_010000000_1=Pd_010[0];
			a1P_000001000_1=Pd_001[1];
			a1P_000000001_1=Pd_001[2];
			a1P_001000000_1=Pd_001[0];
			a1P_000010000_1=Pd_010[1];
			a1P_000000010_1=Pd_010[2];
			ans_temp[ans_id*9+0]+=Pmtrx[0]*(P_011000000*QR_020000000000+P_111000000*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*9+0]+=Pmtrx[1]*(P_011000000*QR_010010000000+P_111000000*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*9+0]+=Pmtrx[2]*(P_011000000*QR_000020000000+P_111000000*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*9+0]+=Pmtrx[3]*(P_011000000*QR_010000010000+P_111000000*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*9+0]+=Pmtrx[4]*(P_011000000*QR_000010010000+P_111000000*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*9+0]+=Pmtrx[5]*(P_011000000*QR_000000020000+P_111000000*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*9+1]+=Pmtrx[0]*(P_010001000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000001000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+1]+=Pmtrx[1]*(P_010001000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000001000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+1]+=Pmtrx[2]*(P_010001000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000001000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+1]+=Pmtrx[3]*(P_010001000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000001000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+1]+=Pmtrx[4]*(P_010001000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000001000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+1]+=Pmtrx[5]*(P_010001000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000001000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+2]+=Pmtrx[0]*(P_010000001*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000001_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+2]+=Pmtrx[1]*(P_010000001*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000001_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+2]+=Pmtrx[2]*(P_010000001*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000001_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+2]+=Pmtrx[3]*(P_010000001*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000001_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+2]+=Pmtrx[4]*(P_010000001*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000001_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+2]+=Pmtrx[5]*(P_010000001*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000001_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+3]+=Pmtrx[0]*(P_001010000*QR_020000000000+a1P_001000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+3]+=Pmtrx[1]*(P_001010000*QR_010010000000+a1P_001000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+3]+=Pmtrx[2]*(P_001010000*QR_000020000000+a1P_001000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+3]+=Pmtrx[3]*(P_001010000*QR_010000010000+a1P_001000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+3]+=Pmtrx[4]*(P_001010000*QR_000010010000+a1P_001000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+3]+=Pmtrx[5]*(P_001010000*QR_000000020000+a1P_001000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+4]+=Pmtrx[0]*(P_000011000*QR_020000000000+P_000111000*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*9+4]+=Pmtrx[1]*(P_000011000*QR_010010000000+P_000111000*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*9+4]+=Pmtrx[2]*(P_000011000*QR_000020000000+P_000111000*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*9+4]+=Pmtrx[3]*(P_000011000*QR_010000010000+P_000111000*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*9+4]+=Pmtrx[4]*(P_000011000*QR_000010010000+P_000111000*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*9+4]+=Pmtrx[5]*(P_000011000*QR_000000020000+P_000111000*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*9+5]+=Pmtrx[0]*(P_000010001*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000001_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+5]+=Pmtrx[1]*(P_000010001*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000001_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+5]+=Pmtrx[2]*(P_000010001*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000001_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+5]+=Pmtrx[3]*(P_000010001*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000001_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+5]+=Pmtrx[4]*(P_000010001*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000001_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+5]+=Pmtrx[5]*(P_000010001*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000001_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+6]+=Pmtrx[0]*(P_001000010*QR_020000000000+a1P_001000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+6]+=Pmtrx[1]*(P_001000010*QR_010010000000+a1P_001000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+6]+=Pmtrx[2]*(P_001000010*QR_000020000000+a1P_001000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+6]+=Pmtrx[3]*(P_001000010*QR_010000010000+a1P_001000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+6]+=Pmtrx[4]*(P_001000010*QR_000010010000+a1P_001000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+6]+=Pmtrx[5]*(P_001000010*QR_000000020000+a1P_001000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+7]+=Pmtrx[0]*(P_000001010*QR_020000000000+a1P_000001000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+7]+=Pmtrx[1]*(P_000001010*QR_010010000000+a1P_000001000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+7]+=Pmtrx[2]*(P_000001010*QR_000020000000+a1P_000001000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+7]+=Pmtrx[3]*(P_000001010*QR_010000010000+a1P_000001000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+7]+=Pmtrx[4]*(P_000001010*QR_000010010000+a1P_000001000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+7]+=Pmtrx[5]*(P_000001010*QR_000000020000+a1P_000001000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+8]+=Pmtrx[0]*(P_000000011*QR_020000000000+P_000000111*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*9+8]+=Pmtrx[1]*(P_000000011*QR_010010000000+P_000000111*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*9+8]+=Pmtrx[2]*(P_000000011*QR_000020000000+P_000000111*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*9+8]+=Pmtrx[3]*(P_000000011*QR_010000010000+P_000000111*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*9+8]+=Pmtrx[4]*(P_000000011*QR_000010010000+P_000000111*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*9+8]+=Pmtrx[5]*(P_000000011*QR_000000020000+P_000000111*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<9;ians++){
                    ans_temp[tId_x*9+ians]+=ans_temp[(tId_x+num_thread)*9+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<9;ians++){
                ans[i_contrc_bra*9+ians]=ans_temp[(tId_x)*9+ians];
            }
        }
	}
}
__global__ void TSMJ_ppds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*9];
    for(int i=0;i<9;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[jj*3+0];
				Qd_010[1]=QC[jj*3+1];
				Qd_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_011[3];
		double Pd_111[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
			double P_011000000;
			double P_111000000;
			double P_010001000;
			double P_010000001;
			double P_001010000;
			double P_000011000;
			double P_000111000;
			double P_000010001;
			double P_001000010;
			double P_000001010;
			double P_000000011;
			double P_000000111;
			double a1P_010000000_1;
			double a1P_000001000_1;
			double a1P_000000001_1;
			double a1P_001000000_1;
			double a1P_000010000_1;
			double a1P_000000010_1;
			P_011000000=Pd_011[0];
			P_111000000=Pd_111[0];
			P_010001000=Pd_010[0]*Pd_001[1];
			P_010000001=Pd_010[0]*Pd_001[2];
			P_001010000=Pd_001[0]*Pd_010[1];
			P_000011000=Pd_011[1];
			P_000111000=Pd_111[1];
			P_000010001=Pd_010[1]*Pd_001[2];
			P_001000010=Pd_001[0]*Pd_010[2];
			P_000001010=Pd_001[1]*Pd_010[2];
			P_000000011=Pd_011[2];
			P_000000111=Pd_111[2];
			a1P_010000000_1=Pd_010[0];
			a1P_000001000_1=Pd_001[1];
			a1P_000000001_1=Pd_001[2];
			a1P_001000000_1=Pd_001[0];
			a1P_000010000_1=Pd_010[1];
			a1P_000000010_1=Pd_010[2];
			ans_temp[ans_id*9+0]+=Pmtrx[0]*(P_011000000*QR_020000000000+P_111000000*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*9+0]+=Pmtrx[1]*(P_011000000*QR_010010000000+P_111000000*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*9+0]+=Pmtrx[2]*(P_011000000*QR_000020000000+P_111000000*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*9+0]+=Pmtrx[3]*(P_011000000*QR_010000010000+P_111000000*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*9+0]+=Pmtrx[4]*(P_011000000*QR_000010010000+P_111000000*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*9+0]+=Pmtrx[5]*(P_011000000*QR_000000020000+P_111000000*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*9+1]+=Pmtrx[0]*(P_010001000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000001000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+1]+=Pmtrx[1]*(P_010001000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000001000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+1]+=Pmtrx[2]*(P_010001000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000001000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+1]+=Pmtrx[3]*(P_010001000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000001000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+1]+=Pmtrx[4]*(P_010001000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000001000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+1]+=Pmtrx[5]*(P_010001000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000001000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+2]+=Pmtrx[0]*(P_010000001*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000001_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+2]+=Pmtrx[1]*(P_010000001*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000001_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+2]+=Pmtrx[2]*(P_010000001*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000001_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+2]+=Pmtrx[3]*(P_010000001*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000001_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+2]+=Pmtrx[4]*(P_010000001*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000001_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+2]+=Pmtrx[5]*(P_010000001*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000001_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+3]+=Pmtrx[0]*(P_001010000*QR_020000000000+a1P_001000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+3]+=Pmtrx[1]*(P_001010000*QR_010010000000+a1P_001000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+3]+=Pmtrx[2]*(P_001010000*QR_000020000000+a1P_001000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+3]+=Pmtrx[3]*(P_001010000*QR_010000010000+a1P_001000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+3]+=Pmtrx[4]*(P_001010000*QR_000010010000+a1P_001000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+3]+=Pmtrx[5]*(P_001010000*QR_000000020000+a1P_001000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+4]+=Pmtrx[0]*(P_000011000*QR_020000000000+P_000111000*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*9+4]+=Pmtrx[1]*(P_000011000*QR_010010000000+P_000111000*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*9+4]+=Pmtrx[2]*(P_000011000*QR_000020000000+P_000111000*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*9+4]+=Pmtrx[3]*(P_000011000*QR_010000010000+P_000111000*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*9+4]+=Pmtrx[4]*(P_000011000*QR_000010010000+P_000111000*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*9+4]+=Pmtrx[5]*(P_000011000*QR_000000020000+P_000111000*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*9+5]+=Pmtrx[0]*(P_000010001*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000001_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+5]+=Pmtrx[1]*(P_000010001*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000001_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+5]+=Pmtrx[2]*(P_000010001*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000001_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+5]+=Pmtrx[3]*(P_000010001*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000001_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+5]+=Pmtrx[4]*(P_000010001*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000001_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+5]+=Pmtrx[5]*(P_000010001*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000001_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+6]+=Pmtrx[0]*(P_001000010*QR_020000000000+a1P_001000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+6]+=Pmtrx[1]*(P_001000010*QR_010010000000+a1P_001000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+6]+=Pmtrx[2]*(P_001000010*QR_000020000000+a1P_001000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+6]+=Pmtrx[3]*(P_001000010*QR_010000010000+a1P_001000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+6]+=Pmtrx[4]*(P_001000010*QR_000010010000+a1P_001000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+6]+=Pmtrx[5]*(P_001000010*QR_000000020000+a1P_001000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+7]+=Pmtrx[0]*(P_000001010*QR_020000000000+a1P_000001000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+7]+=Pmtrx[1]*(P_000001010*QR_010010000000+a1P_000001000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+7]+=Pmtrx[2]*(P_000001010*QR_000020000000+a1P_000001000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+7]+=Pmtrx[3]*(P_000001010*QR_010000010000+a1P_000001000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+7]+=Pmtrx[4]*(P_000001010*QR_000010010000+a1P_000001000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+7]+=Pmtrx[5]*(P_000001010*QR_000000020000+a1P_000001000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+8]+=Pmtrx[0]*(P_000000011*QR_020000000000+P_000000111*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*9+8]+=Pmtrx[1]*(P_000000011*QR_010010000000+P_000000111*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*9+8]+=Pmtrx[2]*(P_000000011*QR_000020000000+P_000000111*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*9+8]+=Pmtrx[3]*(P_000000011*QR_010000010000+P_000000111*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*9+8]+=Pmtrx[4]*(P_000000011*QR_000010010000+P_000000111*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*9+8]+=Pmtrx[5]*(P_000000011*QR_000000020000+P_000000111*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<9;ians++){
                    ans_temp[tId_x*9+ians]+=ans_temp[(tId_x+num_thread)*9+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<9;ians++){
                ans[i_contrc_bra*9+ians]=ans_temp[(tId_x)*9+ians];
            }
        }
	}
}
__global__ void TSMJ_ppds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*9];
    for(int i=0;i<9;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_011[3];
		double Pd_111[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
			double P_011000000;
			double P_111000000;
			double P_010001000;
			double P_010000001;
			double P_001010000;
			double P_000011000;
			double P_000111000;
			double P_000010001;
			double P_001000010;
			double P_000001010;
			double P_000000011;
			double P_000000111;
			double a1P_010000000_1;
			double a1P_000001000_1;
			double a1P_000000001_1;
			double a1P_001000000_1;
			double a1P_000010000_1;
			double a1P_000000010_1;
			P_011000000=Pd_011[0];
			P_111000000=Pd_111[0];
			P_010001000=Pd_010[0]*Pd_001[1];
			P_010000001=Pd_010[0]*Pd_001[2];
			P_001010000=Pd_001[0]*Pd_010[1];
			P_000011000=Pd_011[1];
			P_000111000=Pd_111[1];
			P_000010001=Pd_010[1]*Pd_001[2];
			P_001000010=Pd_001[0]*Pd_010[2];
			P_000001010=Pd_001[1]*Pd_010[2];
			P_000000011=Pd_011[2];
			P_000000111=Pd_111[2];
			a1P_010000000_1=Pd_010[0];
			a1P_000001000_1=Pd_001[1];
			a1P_000000001_1=Pd_001[2];
			a1P_001000000_1=Pd_001[0];
			a1P_000010000_1=Pd_010[1];
			a1P_000000010_1=Pd_010[2];
			ans_temp[ans_id*9+0]+=Pmtrx[0]*(P_011000000*QR_020000000000+P_111000000*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*9+0]+=Pmtrx[1]*(P_011000000*QR_010010000000+P_111000000*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*9+0]+=Pmtrx[2]*(P_011000000*QR_000020000000+P_111000000*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*9+0]+=Pmtrx[3]*(P_011000000*QR_010000010000+P_111000000*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*9+0]+=Pmtrx[4]*(P_011000000*QR_000010010000+P_111000000*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*9+0]+=Pmtrx[5]*(P_011000000*QR_000000020000+P_111000000*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*9+1]+=Pmtrx[0]*(P_010001000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000001000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+1]+=Pmtrx[1]*(P_010001000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000001000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+1]+=Pmtrx[2]*(P_010001000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000001000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+1]+=Pmtrx[3]*(P_010001000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000001000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+1]+=Pmtrx[4]*(P_010001000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000001000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+1]+=Pmtrx[5]*(P_010001000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000001000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+2]+=Pmtrx[0]*(P_010000001*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000001_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+2]+=Pmtrx[1]*(P_010000001*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000001_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+2]+=Pmtrx[2]*(P_010000001*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000001_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+2]+=Pmtrx[3]*(P_010000001*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000001_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+2]+=Pmtrx[4]*(P_010000001*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000001_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+2]+=Pmtrx[5]*(P_010000001*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000001_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+3]+=Pmtrx[0]*(P_001010000*QR_020000000000+a1P_001000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+3]+=Pmtrx[1]*(P_001010000*QR_010010000000+a1P_001000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+3]+=Pmtrx[2]*(P_001010000*QR_000020000000+a1P_001000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+3]+=Pmtrx[3]*(P_001010000*QR_010000010000+a1P_001000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+3]+=Pmtrx[4]*(P_001010000*QR_000010010000+a1P_001000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+3]+=Pmtrx[5]*(P_001010000*QR_000000020000+a1P_001000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+4]+=Pmtrx[0]*(P_000011000*QR_020000000000+P_000111000*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*9+4]+=Pmtrx[1]*(P_000011000*QR_010010000000+P_000111000*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*9+4]+=Pmtrx[2]*(P_000011000*QR_000020000000+P_000111000*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*9+4]+=Pmtrx[3]*(P_000011000*QR_010000010000+P_000111000*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*9+4]+=Pmtrx[4]*(P_000011000*QR_000010010000+P_000111000*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*9+4]+=Pmtrx[5]*(P_000011000*QR_000000020000+P_000111000*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*9+5]+=Pmtrx[0]*(P_000010001*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000001_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+5]+=Pmtrx[1]*(P_000010001*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000001_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+5]+=Pmtrx[2]*(P_000010001*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000001_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+5]+=Pmtrx[3]*(P_000010001*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000001_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+5]+=Pmtrx[4]*(P_000010001*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000001_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+5]+=Pmtrx[5]*(P_000010001*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000001_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+6]+=Pmtrx[0]*(P_001000010*QR_020000000000+a1P_001000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+6]+=Pmtrx[1]*(P_001000010*QR_010010000000+a1P_001000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+6]+=Pmtrx[2]*(P_001000010*QR_000020000000+a1P_001000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+6]+=Pmtrx[3]*(P_001000010*QR_010000010000+a1P_001000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+6]+=Pmtrx[4]*(P_001000010*QR_000010010000+a1P_001000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+6]+=Pmtrx[5]*(P_001000010*QR_000000020000+a1P_001000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+7]+=Pmtrx[0]*(P_000001010*QR_020000000000+a1P_000001000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+7]+=Pmtrx[1]*(P_000001010*QR_010010000000+a1P_000001000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+7]+=Pmtrx[2]*(P_000001010*QR_000020000000+a1P_000001000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+7]+=Pmtrx[3]*(P_000001010*QR_010000010000+a1P_000001000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+7]+=Pmtrx[4]*(P_000001010*QR_000010010000+a1P_000001000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+7]+=Pmtrx[5]*(P_000001010*QR_000000020000+a1P_000001000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+8]+=Pmtrx[0]*(P_000000011*QR_020000000000+P_000000111*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*9+8]+=Pmtrx[1]*(P_000000011*QR_010010000000+P_000000111*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*9+8]+=Pmtrx[2]*(P_000000011*QR_000020000000+P_000000111*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*9+8]+=Pmtrx[3]*(P_000000011*QR_010000010000+P_000000111*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*9+8]+=Pmtrx[4]*(P_000000011*QR_000010010000+P_000000111*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*9+8]+=Pmtrx[5]*(P_000000011*QR_000000020000+P_000000111*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<9;ians++){
                    ans_temp[tId_x*9+ians]+=ans_temp[(tId_x+num_thread)*9+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<9;ians++){
                ans[i_contrc_bra*9+ians]=ans_temp[(tId_x)*9+ians];
            }
        }
	}
}
__global__ void TSMJ_ppds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*9];
    for(int i=0;i<9;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
			double QR_020000000001=0;
			double QR_010010000001=0;
			double QR_000020000001=0;
			double QR_010000010001=0;
			double QR_000010010001=0;
			double QR_000000020001=0;
			double QR_020000000010=0;
			double QR_010010000010=0;
			double QR_000020000010=0;
			double QR_010000010010=0;
			double QR_000010010010=0;
			double QR_000000020010=0;
			double QR_020000000100=0;
			double QR_010010000100=0;
			double QR_000020000100=0;
			double QR_010000010100=0;
			double QR_000010010100=0;
			double QR_000000020100=0;
			double QR_020000000002=0;
			double QR_010010000002=0;
			double QR_000020000002=0;
			double QR_010000010002=0;
			double QR_000010010002=0;
			double QR_000000020002=0;
			double QR_020000000011=0;
			double QR_010010000011=0;
			double QR_000020000011=0;
			double QR_010000010011=0;
			double QR_000010010011=0;
			double QR_000000020011=0;
			double QR_020000000020=0;
			double QR_010010000020=0;
			double QR_000020000020=0;
			double QR_010000010020=0;
			double QR_000010010020=0;
			double QR_000000020020=0;
			double QR_020000000101=0;
			double QR_010010000101=0;
			double QR_000020000101=0;
			double QR_010000010101=0;
			double QR_000010010101=0;
			double QR_000000020101=0;
			double QR_020000000110=0;
			double QR_010010000110=0;
			double QR_000020000110=0;
			double QR_010000010110=0;
			double QR_000010010110=0;
			double QR_000000020110=0;
			double QR_020000000200=0;
			double QR_010010000200=0;
			double QR_000020000200=0;
			double QR_010000010200=0;
			double QR_000010010200=0;
			double QR_000000020200=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			QR_020000000001+=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			QR_010010000001+=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			QR_000020000001+=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			QR_010000010001+=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			QR_000010010001+=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			QR_000000020001+=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			QR_020000000010+=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			QR_010010000010+=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			QR_000020000010+=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			QR_010000010010+=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000010010010+=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			QR_000000020010+=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			QR_020000000100+=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			QR_010010000100+=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			QR_000020000100+=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			QR_010000010100+=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			QR_000010010100+=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000000020100+=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			QR_020000000002+=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			QR_010010000002+=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			QR_000020000002+=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			QR_010000010002+=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			QR_000010010002+=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			QR_000000020002+=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			QR_020000000011+=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			QR_010010000011+=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			QR_000020000011+=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			QR_010000010011+=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000010010011+=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			QR_000000020011+=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			QR_020000000020+=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			QR_010010000020+=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			QR_000020000020+=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			QR_010000010020+=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000010010020+=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			QR_000000020020+=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			QR_020000000101+=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			QR_010010000101+=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			QR_000020000101+=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			QR_010000010101+=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			QR_000010010101+=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000000020101+=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			QR_020000000110+=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			QR_010010000110+=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			QR_000020000110+=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			QR_010000010110+=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000010010110+=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000000020110+=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			QR_020000000200+=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			QR_010010000200+=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			QR_000020000200+=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			QR_010000010200+=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			QR_000010010200+=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000000020200+=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			}
		double Pd_011[3];
		double Pd_111[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
			double P_011000000;
			double P_111000000;
			double P_010001000;
			double P_010000001;
			double P_001010000;
			double P_000011000;
			double P_000111000;
			double P_000010001;
			double P_001000010;
			double P_000001010;
			double P_000000011;
			double P_000000111;
			double a1P_010000000_1;
			double a1P_000001000_1;
			double a1P_000000001_1;
			double a1P_001000000_1;
			double a1P_000010000_1;
			double a1P_000000010_1;
			P_011000000=Pd_011[0];
			P_111000000=Pd_111[0];
			P_010001000=Pd_010[0]*Pd_001[1];
			P_010000001=Pd_010[0]*Pd_001[2];
			P_001010000=Pd_001[0]*Pd_010[1];
			P_000011000=Pd_011[1];
			P_000111000=Pd_111[1];
			P_000010001=Pd_010[1]*Pd_001[2];
			P_001000010=Pd_001[0]*Pd_010[2];
			P_000001010=Pd_001[1]*Pd_010[2];
			P_000000011=Pd_011[2];
			P_000000111=Pd_111[2];
			a1P_010000000_1=Pd_010[0];
			a1P_000001000_1=Pd_001[1];
			a1P_000000001_1=Pd_001[2];
			a1P_001000000_1=Pd_001[0];
			a1P_000010000_1=Pd_010[1];
			a1P_000000010_1=Pd_010[2];
			ans_temp[ans_id*9+0]+=Pmtrx[0]*(P_011000000*QR_020000000000+P_111000000*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*9+0]+=Pmtrx[1]*(P_011000000*QR_010010000000+P_111000000*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*9+0]+=Pmtrx[2]*(P_011000000*QR_000020000000+P_111000000*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*9+0]+=Pmtrx[3]*(P_011000000*QR_010000010000+P_111000000*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*9+0]+=Pmtrx[4]*(P_011000000*QR_000010010000+P_111000000*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*9+0]+=Pmtrx[5]*(P_011000000*QR_000000020000+P_111000000*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*9+1]+=Pmtrx[0]*(P_010001000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000001000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+1]+=Pmtrx[1]*(P_010001000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000001000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+1]+=Pmtrx[2]*(P_010001000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000001000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+1]+=Pmtrx[3]*(P_010001000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000001000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+1]+=Pmtrx[4]*(P_010001000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000001000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+1]+=Pmtrx[5]*(P_010001000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000001000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+2]+=Pmtrx[0]*(P_010000001*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000001_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+2]+=Pmtrx[1]*(P_010000001*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000001_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+2]+=Pmtrx[2]*(P_010000001*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000001_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+2]+=Pmtrx[3]*(P_010000001*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000001_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+2]+=Pmtrx[4]*(P_010000001*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000001_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+2]+=Pmtrx[5]*(P_010000001*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000001_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+3]+=Pmtrx[0]*(P_001010000*QR_020000000000+a1P_001000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*9+3]+=Pmtrx[1]*(P_001010000*QR_010010000000+a1P_001000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*9+3]+=Pmtrx[2]*(P_001010000*QR_000020000000+a1P_001000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*9+3]+=Pmtrx[3]*(P_001010000*QR_010000010000+a1P_001000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*9+3]+=Pmtrx[4]*(P_001010000*QR_000010010000+a1P_001000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*9+3]+=Pmtrx[5]*(P_001010000*QR_000000020000+a1P_001000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*9+4]+=Pmtrx[0]*(P_000011000*QR_020000000000+P_000111000*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*9+4]+=Pmtrx[1]*(P_000011000*QR_010010000000+P_000111000*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*9+4]+=Pmtrx[2]*(P_000011000*QR_000020000000+P_000111000*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*9+4]+=Pmtrx[3]*(P_000011000*QR_010000010000+P_000111000*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*9+4]+=Pmtrx[4]*(P_000011000*QR_000010010000+P_000111000*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*9+4]+=Pmtrx[5]*(P_000011000*QR_000000020000+P_000111000*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*9+5]+=Pmtrx[0]*(P_000010001*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000001_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+5]+=Pmtrx[1]*(P_000010001*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000001_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+5]+=Pmtrx[2]*(P_000010001*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000001_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+5]+=Pmtrx[3]*(P_000010001*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000001_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+5]+=Pmtrx[4]*(P_000010001*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000001_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+5]+=Pmtrx[5]*(P_000010001*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000001_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+6]+=Pmtrx[0]*(P_001000010*QR_020000000000+a1P_001000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*9+6]+=Pmtrx[1]*(P_001000010*QR_010010000000+a1P_001000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*9+6]+=Pmtrx[2]*(P_001000010*QR_000020000000+a1P_001000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*9+6]+=Pmtrx[3]*(P_001000010*QR_010000010000+a1P_001000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*9+6]+=Pmtrx[4]*(P_001000010*QR_000010010000+a1P_001000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*9+6]+=Pmtrx[5]*(P_001000010*QR_000000020000+a1P_001000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*9+7]+=Pmtrx[0]*(P_000001010*QR_020000000000+a1P_000001000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*9+7]+=Pmtrx[1]*(P_000001010*QR_010010000000+a1P_000001000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*9+7]+=Pmtrx[2]*(P_000001010*QR_000020000000+a1P_000001000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*9+7]+=Pmtrx[3]*(P_000001010*QR_010000010000+a1P_000001000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*9+7]+=Pmtrx[4]*(P_000001010*QR_000010010000+a1P_000001000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*9+7]+=Pmtrx[5]*(P_000001010*QR_000000020000+a1P_000001000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*9+8]+=Pmtrx[0]*(P_000000011*QR_020000000000+P_000000111*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*9+8]+=Pmtrx[1]*(P_000000011*QR_010010000000+P_000000111*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*9+8]+=Pmtrx[2]*(P_000000011*QR_000020000000+P_000000111*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*9+8]+=Pmtrx[3]*(P_000000011*QR_010000010000+P_000000111*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*9+8]+=Pmtrx[4]*(P_000000011*QR_000010010000+P_000000111*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*9+8]+=Pmtrx[5]*(P_000000011*QR_000000020000+P_000000111*QR_000000020001+aPin2*QR_000000020002);
		}
        __syncthreads();
        int num_thread=tdis/2;
        while (num_thread!=0){
            __syncthreads();
            if(tId_x<num_thread){
                for(int ians=0;ians<9;ians++){
                    ans_temp[tId_x*9+ians]+=ans_temp[(tId_x+num_thread)*9+ians];
                }
            }
            num_thread/=2;
        }
        if(tId_x==0){
            for(int ians=0;ians<9;ians++){
                ans[i_contrc_bra*9+ians]=ans_temp[(tId_x)*9+ians];
            }
        }
	}
}
__global__ void TSMJ_dsds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[5];
                Ft_taylor(4,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_020[3];
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
			double P_020000000;
			double P_010010000;
			double P_000020000;
			double P_010000010;
			double P_000010010;
			double P_000000020;
			double a1P_010000000_1;
			double a1P_010000000_2;
			double a1P_000010000_1;
			double a1P_000010000_2;
			double a1P_000000010_1;
			double a1P_000000010_2;
			P_020000000=Pd_020[0];
			P_010010000=Pd_010[0]*Pd_010[1];
			P_000020000=Pd_020[1];
			P_010000010=Pd_010[0]*Pd_010[2];
			P_000010010=Pd_010[1]*Pd_010[2];
			P_000000020=Pd_020[2];
			a1P_010000000_1=Pd_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=Pd_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=Pd_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_020000000*QR_020000000000+a1P_010000000_2*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_020000000*QR_010010000000+a1P_010000000_2*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_020000000*QR_000020000000+a1P_010000000_2*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_020000000*QR_010000010000+a1P_010000000_2*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_020000000*QR_000010010000+a1P_010000000_2*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_020000000*QR_000000020000+a1P_010000000_2*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_010010000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_010010000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_010010000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_010010000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_010010000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_010010000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_000020000*QR_020000000000+a1P_000010000_2*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_000020000*QR_010010000000+a1P_000010000_2*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000020000*QR_000020000000+a1P_000010000_2*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_000020000*QR_010000010000+a1P_000010000_2*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000020000*QR_000010010000+a1P_000010000_2*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000020000*QR_000000020000+a1P_000010000_2*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_010000010*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_010000010*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_010000010*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_010000010*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_010000010*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_010000010*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_000010010*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_000010010*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000010010*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_000010010*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000010010*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000010010*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_000000020*QR_020000000000+a1P_000000010_2*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_000000020*QR_010010000000+a1P_000000010_2*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000000020*QR_000020000000+a1P_000000010_2*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_000000020*QR_010000010000+a1P_000000010_2*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000000020*QR_000010010000+a1P_000000010_2*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000000020*QR_000000020000+a1P_000000010_2*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
}
__global__ void TSMJ_dsds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[jj*3+0];
				Qd_010[1]=QC[jj*3+1];
				Qd_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_020[3];
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
			double P_020000000;
			double P_010010000;
			double P_000020000;
			double P_010000010;
			double P_000010010;
			double P_000000020;
			double a1P_010000000_1;
			double a1P_010000000_2;
			double a1P_000010000_1;
			double a1P_000010000_2;
			double a1P_000000010_1;
			double a1P_000000010_2;
			P_020000000=Pd_020[0];
			P_010010000=Pd_010[0]*Pd_010[1];
			P_000020000=Pd_020[1];
			P_010000010=Pd_010[0]*Pd_010[2];
			P_000010010=Pd_010[1]*Pd_010[2];
			P_000000020=Pd_020[2];
			a1P_010000000_1=Pd_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=Pd_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=Pd_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_020000000*QR_020000000000+a1P_010000000_2*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_020000000*QR_010010000000+a1P_010000000_2*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_020000000*QR_000020000000+a1P_010000000_2*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_020000000*QR_010000010000+a1P_010000000_2*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_020000000*QR_000010010000+a1P_010000000_2*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_020000000*QR_000000020000+a1P_010000000_2*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_010010000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_010010000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_010010000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_010010000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_010010000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_010010000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_000020000*QR_020000000000+a1P_000010000_2*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_000020000*QR_010010000000+a1P_000010000_2*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000020000*QR_000020000000+a1P_000010000_2*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_000020000*QR_010000010000+a1P_000010000_2*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000020000*QR_000010010000+a1P_000010000_2*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000020000*QR_000000020000+a1P_000010000_2*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_010000010*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_010000010*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_010000010*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_010000010*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_010000010*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_010000010*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_000010010*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_000010010*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000010010*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_000010010*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000010010*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000010010*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_000000020*QR_020000000000+a1P_000000010_2*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_000000020*QR_010010000000+a1P_000000010_2*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000000020*QR_000020000000+a1P_000000010_2*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_000000020*QR_010000010000+a1P_000000010_2*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000000020*QR_000010010000+a1P_000000010_2*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000000020*QR_000000020000+a1P_000000010_2*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
}
__global__ void TSMJ_dsds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
		double Pd_020[3];
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
			double P_020000000;
			double P_010010000;
			double P_000020000;
			double P_010000010;
			double P_000010010;
			double P_000000020;
			double a1P_010000000_1;
			double a1P_010000000_2;
			double a1P_000010000_1;
			double a1P_000010000_2;
			double a1P_000000010_1;
			double a1P_000000010_2;
			P_020000000=Pd_020[0];
			P_010010000=Pd_010[0]*Pd_010[1];
			P_000020000=Pd_020[1];
			P_010000010=Pd_010[0]*Pd_010[2];
			P_000010010=Pd_010[1]*Pd_010[2];
			P_000000020=Pd_020[2];
			a1P_010000000_1=Pd_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=Pd_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=Pd_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_020000000*QR_020000000000+a1P_010000000_2*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_020000000*QR_010010000000+a1P_010000000_2*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_020000000*QR_000020000000+a1P_010000000_2*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_020000000*QR_010000010000+a1P_010000000_2*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_020000000*QR_000010010000+a1P_010000000_2*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_020000000*QR_000000020000+a1P_010000000_2*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_010010000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_010010000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_010010000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_010010000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_010010000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_010010000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_000020000*QR_020000000000+a1P_000010000_2*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_000020000*QR_010010000000+a1P_000010000_2*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000020000*QR_000020000000+a1P_000010000_2*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_000020000*QR_010000010000+a1P_000010000_2*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000020000*QR_000010010000+a1P_000010000_2*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000020000*QR_000000020000+a1P_000010000_2*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_010000010*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_010000010*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_010000010*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_010000010*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_010000010*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_010000010*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_000010010*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_000010010*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000010010*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_000010010*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000010010*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000010010*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_000000020*QR_020000000000+a1P_000000010_2*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_000000020*QR_010010000000+a1P_000000010_2*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000000020*QR_000020000000+a1P_000000010_2*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_000000020*QR_010000010000+a1P_000000010_2*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000000020*QR_000010010000+a1P_000000010_2*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000000020*QR_000000020000+a1P_000000010_2*QR_000000020001+aPin2*QR_000000020002);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
}
__global__ void TSMJ_dsds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*6];
    for(int i=0;i<6;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
				double PX=P[ii*3+0];
				double PY=P[ii*3+1];
				double PZ=P[ii*3+2];
				double Pd_010[3];
				Pd_010[0]=PA[ii*3+0];
				Pd_010[1]=PA[ii*3+1];
				Pd_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
			double QR_020000000001=0;
			double QR_010010000001=0;
			double QR_000020000001=0;
			double QR_010000010001=0;
			double QR_000010010001=0;
			double QR_000000020001=0;
			double QR_020000000010=0;
			double QR_010010000010=0;
			double QR_000020000010=0;
			double QR_010000010010=0;
			double QR_000010010010=0;
			double QR_000000020010=0;
			double QR_020000000100=0;
			double QR_010010000100=0;
			double QR_000020000100=0;
			double QR_010000010100=0;
			double QR_000010010100=0;
			double QR_000000020100=0;
			double QR_020000000002=0;
			double QR_010010000002=0;
			double QR_000020000002=0;
			double QR_010000010002=0;
			double QR_000010010002=0;
			double QR_000000020002=0;
			double QR_020000000011=0;
			double QR_010010000011=0;
			double QR_000020000011=0;
			double QR_010000010011=0;
			double QR_000010010011=0;
			double QR_000000020011=0;
			double QR_020000000020=0;
			double QR_010010000020=0;
			double QR_000020000020=0;
			double QR_010000010020=0;
			double QR_000010010020=0;
			double QR_000000020020=0;
			double QR_020000000101=0;
			double QR_010010000101=0;
			double QR_000020000101=0;
			double QR_010000010101=0;
			double QR_000010010101=0;
			double QR_000000020101=0;
			double QR_020000000110=0;
			double QR_010010000110=0;
			double QR_000020000110=0;
			double QR_010000010110=0;
			double QR_000010010110=0;
			double QR_000000020110=0;
			double QR_020000000200=0;
			double QR_010010000200=0;
			double QR_000020000200=0;
			double QR_010000010200=0;
			double QR_000010010200=0;
			double QR_000000020200=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			QR_020000000001+=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			QR_010010000001+=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			QR_000020000001+=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			QR_010000010001+=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			QR_000010010001+=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			QR_000000020001+=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			QR_020000000010+=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			QR_010010000010+=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			QR_000020000010+=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			QR_010000010010+=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000010010010+=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			QR_000000020010+=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			QR_020000000100+=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			QR_010010000100+=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			QR_000020000100+=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			QR_010000010100+=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			QR_000010010100+=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000000020100+=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			QR_020000000002+=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			QR_010010000002+=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			QR_000020000002+=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			QR_010000010002+=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			QR_000010010002+=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			QR_000000020002+=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			QR_020000000011+=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			QR_010010000011+=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			QR_000020000011+=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			QR_010000010011+=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000010010011+=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			QR_000000020011+=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			QR_020000000020+=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			QR_010010000020+=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			QR_000020000020+=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			QR_010000010020+=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000010010020+=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			QR_000000020020+=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			QR_020000000101+=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			QR_010010000101+=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			QR_000020000101+=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			QR_010000010101+=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			QR_000010010101+=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000000020101+=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			QR_020000000110+=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			QR_010010000110+=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			QR_000020000110+=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			QR_010000010110+=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000010010110+=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000000020110+=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			QR_020000000200+=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			QR_010010000200+=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			QR_000020000200+=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			QR_010000010200+=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			QR_000010010200+=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000000020200+=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			}
		double Pd_020[3];
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
			double P_020000000;
			double P_010010000;
			double P_000020000;
			double P_010000010;
			double P_000010010;
			double P_000000020;
			double a1P_010000000_1;
			double a1P_010000000_2;
			double a1P_000010000_1;
			double a1P_000010000_2;
			double a1P_000000010_1;
			double a1P_000000010_2;
			P_020000000=Pd_020[0];
			P_010010000=Pd_010[0]*Pd_010[1];
			P_000020000=Pd_020[1];
			P_010000010=Pd_010[0]*Pd_010[2];
			P_000010010=Pd_010[1]*Pd_010[2];
			P_000000020=Pd_020[2];
			a1P_010000000_1=Pd_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=Pd_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=Pd_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=Pmtrx[0]*(P_020000000*QR_020000000000+a1P_010000000_2*QR_020000000100+aPin2*QR_020000000200);
			ans_temp[ans_id*6+0]+=Pmtrx[1]*(P_020000000*QR_010010000000+a1P_010000000_2*QR_010010000100+aPin2*QR_010010000200);
			ans_temp[ans_id*6+0]+=Pmtrx[2]*(P_020000000*QR_000020000000+a1P_010000000_2*QR_000020000100+aPin2*QR_000020000200);
			ans_temp[ans_id*6+0]+=Pmtrx[3]*(P_020000000*QR_010000010000+a1P_010000000_2*QR_010000010100+aPin2*QR_010000010200);
			ans_temp[ans_id*6+0]+=Pmtrx[4]*(P_020000000*QR_000010010000+a1P_010000000_2*QR_000010010100+aPin2*QR_000010010200);
			ans_temp[ans_id*6+0]+=Pmtrx[5]*(P_020000000*QR_000000020000+a1P_010000000_2*QR_000000020100+aPin2*QR_000000020200);
			ans_temp[ans_id*6+1]+=Pmtrx[0]*(P_010010000*QR_020000000000+a1P_010000000_1*QR_020000000010+a1P_000010000_1*QR_020000000100+aPin2*QR_020000000110);
			ans_temp[ans_id*6+1]+=Pmtrx[1]*(P_010010000*QR_010010000000+a1P_010000000_1*QR_010010000010+a1P_000010000_1*QR_010010000100+aPin2*QR_010010000110);
			ans_temp[ans_id*6+1]+=Pmtrx[2]*(P_010010000*QR_000020000000+a1P_010000000_1*QR_000020000010+a1P_000010000_1*QR_000020000100+aPin2*QR_000020000110);
			ans_temp[ans_id*6+1]+=Pmtrx[3]*(P_010010000*QR_010000010000+a1P_010000000_1*QR_010000010010+a1P_000010000_1*QR_010000010100+aPin2*QR_010000010110);
			ans_temp[ans_id*6+1]+=Pmtrx[4]*(P_010010000*QR_000010010000+a1P_010000000_1*QR_000010010010+a1P_000010000_1*QR_000010010100+aPin2*QR_000010010110);
			ans_temp[ans_id*6+1]+=Pmtrx[5]*(P_010010000*QR_000000020000+a1P_010000000_1*QR_000000020010+a1P_000010000_1*QR_000000020100+aPin2*QR_000000020110);
			ans_temp[ans_id*6+2]+=Pmtrx[0]*(P_000020000*QR_020000000000+a1P_000010000_2*QR_020000000010+aPin2*QR_020000000020);
			ans_temp[ans_id*6+2]+=Pmtrx[1]*(P_000020000*QR_010010000000+a1P_000010000_2*QR_010010000010+aPin2*QR_010010000020);
			ans_temp[ans_id*6+2]+=Pmtrx[2]*(P_000020000*QR_000020000000+a1P_000010000_2*QR_000020000010+aPin2*QR_000020000020);
			ans_temp[ans_id*6+2]+=Pmtrx[3]*(P_000020000*QR_010000010000+a1P_000010000_2*QR_010000010010+aPin2*QR_010000010020);
			ans_temp[ans_id*6+2]+=Pmtrx[4]*(P_000020000*QR_000010010000+a1P_000010000_2*QR_000010010010+aPin2*QR_000010010020);
			ans_temp[ans_id*6+2]+=Pmtrx[5]*(P_000020000*QR_000000020000+a1P_000010000_2*QR_000000020010+aPin2*QR_000000020020);
			ans_temp[ans_id*6+3]+=Pmtrx[0]*(P_010000010*QR_020000000000+a1P_010000000_1*QR_020000000001+a1P_000000010_1*QR_020000000100+aPin2*QR_020000000101);
			ans_temp[ans_id*6+3]+=Pmtrx[1]*(P_010000010*QR_010010000000+a1P_010000000_1*QR_010010000001+a1P_000000010_1*QR_010010000100+aPin2*QR_010010000101);
			ans_temp[ans_id*6+3]+=Pmtrx[2]*(P_010000010*QR_000020000000+a1P_010000000_1*QR_000020000001+a1P_000000010_1*QR_000020000100+aPin2*QR_000020000101);
			ans_temp[ans_id*6+3]+=Pmtrx[3]*(P_010000010*QR_010000010000+a1P_010000000_1*QR_010000010001+a1P_000000010_1*QR_010000010100+aPin2*QR_010000010101);
			ans_temp[ans_id*6+3]+=Pmtrx[4]*(P_010000010*QR_000010010000+a1P_010000000_1*QR_000010010001+a1P_000000010_1*QR_000010010100+aPin2*QR_000010010101);
			ans_temp[ans_id*6+3]+=Pmtrx[5]*(P_010000010*QR_000000020000+a1P_010000000_1*QR_000000020001+a1P_000000010_1*QR_000000020100+aPin2*QR_000000020101);
			ans_temp[ans_id*6+4]+=Pmtrx[0]*(P_000010010*QR_020000000000+a1P_000010000_1*QR_020000000001+a1P_000000010_1*QR_020000000010+aPin2*QR_020000000011);
			ans_temp[ans_id*6+4]+=Pmtrx[1]*(P_000010010*QR_010010000000+a1P_000010000_1*QR_010010000001+a1P_000000010_1*QR_010010000010+aPin2*QR_010010000011);
			ans_temp[ans_id*6+4]+=Pmtrx[2]*(P_000010010*QR_000020000000+a1P_000010000_1*QR_000020000001+a1P_000000010_1*QR_000020000010+aPin2*QR_000020000011);
			ans_temp[ans_id*6+4]+=Pmtrx[3]*(P_000010010*QR_010000010000+a1P_000010000_1*QR_010000010001+a1P_000000010_1*QR_010000010010+aPin2*QR_010000010011);
			ans_temp[ans_id*6+4]+=Pmtrx[4]*(P_000010010*QR_000010010000+a1P_000010000_1*QR_000010010001+a1P_000000010_1*QR_000010010010+aPin2*QR_000010010011);
			ans_temp[ans_id*6+4]+=Pmtrx[5]*(P_000010010*QR_000000020000+a1P_000010000_1*QR_000000020001+a1P_000000010_1*QR_000000020010+aPin2*QR_000000020011);
			ans_temp[ans_id*6+5]+=Pmtrx[0]*(P_000000020*QR_020000000000+a1P_000000010_2*QR_020000000001+aPin2*QR_020000000002);
			ans_temp[ans_id*6+5]+=Pmtrx[1]*(P_000000020*QR_010010000000+a1P_000000010_2*QR_010010000001+aPin2*QR_010010000002);
			ans_temp[ans_id*6+5]+=Pmtrx[2]*(P_000000020*QR_000020000000+a1P_000000010_2*QR_000020000001+aPin2*QR_000020000002);
			ans_temp[ans_id*6+5]+=Pmtrx[3]*(P_000000020*QR_010000010000+a1P_000000010_2*QR_010000010001+aPin2*QR_010000010002);
			ans_temp[ans_id*6+5]+=Pmtrx[4]*(P_000000020*QR_000010010000+a1P_000000010_2*QR_000010010001+aPin2*QR_000010010002);
			ans_temp[ans_id*6+5]+=Pmtrx[5]*(P_000000020*QR_000000020000+a1P_000000010_2*QR_000000020001+aPin2*QR_000000020002);
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*6+ians]=ans_temp[(tId_x)*6+ians];
            }
        }
	}
}
__global__ void TSMJ_dpds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[6];
                Ft_taylor(5,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
			double P_021000000;
			double P_121000000;
			double P_221000000;
			double P_020001000;
			double P_020000001;
			double P_011010000;
			double P_111010000;
			double P_010011000;
			double P_010111000;
			double P_010010001;
			double P_001020000;
			double P_000021000;
			double P_000121000;
			double P_000221000;
			double P_000020001;
			double P_011000010;
			double P_111000010;
			double P_010001010;
			double P_010000011;
			double P_010000111;
			double P_001010010;
			double P_000011010;
			double P_000111010;
			double P_000010011;
			double P_000010111;
			double P_001000020;
			double P_000001020;
			double P_000000021;
			double P_000000121;
			double P_000000221;
			double a1P_020000000_1;
			double a1P_010001000_1;
			double a1P_010001000_2;
			double a2P_010000000_1;
			double a2P_010000000_2;
			double a2P_000001000_1;
			double a1P_010000001_1;
			double a1P_010000001_2;
			double a2P_000000001_1;
			double a1P_011000000_1;
			double a1P_111000000_1;
			double a2P_000010000_1;
			double a2P_000010000_2;
			double a1P_000011000_1;
			double a1P_000111000_1;
			double a1P_010010000_1;
			double a1P_000010001_1;
			double a1P_000010001_2;
			double a1P_001010000_1;
			double a1P_001010000_2;
			double a2P_001000000_1;
			double a1P_000020000_1;
			double a2P_000000010_1;
			double a2P_000000010_2;
			double a1P_010000010_1;
			double a1P_000001010_1;
			double a1P_000001010_2;
			double a1P_000000011_1;
			double a1P_000000111_1;
			double a1P_001000010_1;
			double a1P_001000010_2;
			double a1P_000010010_1;
			double a1P_000000020_1;
			P_021000000=Pd_021[0];
			P_121000000=Pd_121[0];
			P_221000000=Pd_221[0];
			P_020001000=Pd_020[0]*Pd_001[1];
			P_020000001=Pd_020[0]*Pd_001[2];
			P_011010000=Pd_011[0]*Pd_010[1];
			P_111010000=Pd_111[0]*Pd_010[1];
			P_010011000=Pd_010[0]*Pd_011[1];
			P_010111000=Pd_010[0]*Pd_111[1];
			P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
			P_001020000=Pd_001[0]*Pd_020[1];
			P_000021000=Pd_021[1];
			P_000121000=Pd_121[1];
			P_000221000=Pd_221[1];
			P_000020001=Pd_020[1]*Pd_001[2];
			P_011000010=Pd_011[0]*Pd_010[2];
			P_111000010=Pd_111[0]*Pd_010[2];
			P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
			P_010000011=Pd_010[0]*Pd_011[2];
			P_010000111=Pd_010[0]*Pd_111[2];
			P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
			P_000011010=Pd_011[1]*Pd_010[2];
			P_000111010=Pd_111[1]*Pd_010[2];
			P_000010011=Pd_010[1]*Pd_011[2];
			P_000010111=Pd_010[1]*Pd_111[2];
			P_001000020=Pd_001[0]*Pd_020[2];
			P_000001020=Pd_001[1]*Pd_020[2];
			P_000000021=Pd_021[2];
			P_000000121=Pd_121[2];
			P_000000221=Pd_221[2];
			a1P_020000000_1=Pd_020[0];
			a1P_010001000_1=Pd_010[0]*Pd_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=Pd_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=Pd_001[1];
			a1P_010000001_1=Pd_010[0]*Pd_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=Pd_001[2];
			a1P_011000000_1=Pd_011[0];
			a1P_111000000_1=Pd_111[0];
			a2P_000010000_1=Pd_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=Pd_011[1];
			a1P_000111000_1=Pd_111[1];
			a1P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010001_1=Pd_010[1]*Pd_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=Pd_001[0]*Pd_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=Pd_001[0];
			a1P_000020000_1=Pd_020[1];
			a2P_000000010_1=Pd_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000001010_1=Pd_001[1]*Pd_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=Pd_011[2];
			a1P_000000111_1=Pd_111[2];
			a1P_001000010_1=Pd_001[0]*Pd_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_000000020_1=Pd_020[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(P_021000000*QR_020000000000+P_121000000*QR_020000000100+P_221000000*QR_020000000200+aPin3*QR_020000000300);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(P_021000000*QR_010010000000+P_121000000*QR_010010000100+P_221000000*QR_010010000200+aPin3*QR_010010000300);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(P_021000000*QR_000020000000+P_121000000*QR_000020000100+P_221000000*QR_000020000200+aPin3*QR_000020000300);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(P_021000000*QR_010000010000+P_121000000*QR_010000010100+P_221000000*QR_010000010200+aPin3*QR_010000010300);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(P_021000000*QR_000010010000+P_121000000*QR_000010010100+P_221000000*QR_000010010200+aPin3*QR_000010010300);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(P_021000000*QR_000000020000+P_121000000*QR_000000020100+P_221000000*QR_000000020200+aPin3*QR_000000020300);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(P_020001000*QR_020000000000+a1P_020000000_1*QR_020000000010+a1P_010001000_2*QR_020000000100+a2P_010000000_2*QR_020000000110+a2P_000001000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(P_020001000*QR_010010000000+a1P_020000000_1*QR_010010000010+a1P_010001000_2*QR_010010000100+a2P_010000000_2*QR_010010000110+a2P_000001000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(P_020001000*QR_000020000000+a1P_020000000_1*QR_000020000010+a1P_010001000_2*QR_000020000100+a2P_010000000_2*QR_000020000110+a2P_000001000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(P_020001000*QR_010000010000+a1P_020000000_1*QR_010000010010+a1P_010001000_2*QR_010000010100+a2P_010000000_2*QR_010000010110+a2P_000001000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(P_020001000*QR_000010010000+a1P_020000000_1*QR_000010010010+a1P_010001000_2*QR_000010010100+a2P_010000000_2*QR_000010010110+a2P_000001000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(P_020001000*QR_000000020000+a1P_020000000_1*QR_000000020010+a1P_010001000_2*QR_000000020100+a2P_010000000_2*QR_000000020110+a2P_000001000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(P_020000001*QR_020000000000+a1P_020000000_1*QR_020000000001+a1P_010000001_2*QR_020000000100+a2P_010000000_2*QR_020000000101+a2P_000000001_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(P_020000001*QR_010010000000+a1P_020000000_1*QR_010010000001+a1P_010000001_2*QR_010010000100+a2P_010000000_2*QR_010010000101+a2P_000000001_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(P_020000001*QR_000020000000+a1P_020000000_1*QR_000020000001+a1P_010000001_2*QR_000020000100+a2P_010000000_2*QR_000020000101+a2P_000000001_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(P_020000001*QR_010000010000+a1P_020000000_1*QR_010000010001+a1P_010000001_2*QR_010000010100+a2P_010000000_2*QR_010000010101+a2P_000000001_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(P_020000001*QR_000010010000+a1P_020000000_1*QR_000010010001+a1P_010000001_2*QR_000010010100+a2P_010000000_2*QR_000010010101+a2P_000000001_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(P_020000001*QR_000000020000+a1P_020000000_1*QR_000000020001+a1P_010000001_2*QR_000000020100+a2P_010000000_2*QR_000000020101+a2P_000000001_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(P_011010000*QR_020000000000+a1P_011000000_1*QR_020000000010+P_111010000*QR_020000000100+a1P_111000000_1*QR_020000000110+a2P_000010000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(P_011010000*QR_010010000000+a1P_011000000_1*QR_010010000010+P_111010000*QR_010010000100+a1P_111000000_1*QR_010010000110+a2P_000010000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(P_011010000*QR_000020000000+a1P_011000000_1*QR_000020000010+P_111010000*QR_000020000100+a1P_111000000_1*QR_000020000110+a2P_000010000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(P_011010000*QR_010000010000+a1P_011000000_1*QR_010000010010+P_111010000*QR_010000010100+a1P_111000000_1*QR_010000010110+a2P_000010000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(P_011010000*QR_000010010000+a1P_011000000_1*QR_000010010010+P_111010000*QR_000010010100+a1P_111000000_1*QR_000010010110+a2P_000010000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(P_011010000*QR_000000020000+a1P_011000000_1*QR_000000020010+P_111010000*QR_000000020100+a1P_111000000_1*QR_000000020110+a2P_000010000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(P_010011000*QR_020000000000+P_010111000*QR_020000000010+a2P_010000000_1*QR_020000000020+a1P_000011000_1*QR_020000000100+a1P_000111000_1*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(P_010011000*QR_010010000000+P_010111000*QR_010010000010+a2P_010000000_1*QR_010010000020+a1P_000011000_1*QR_010010000100+a1P_000111000_1*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(P_010011000*QR_000020000000+P_010111000*QR_000020000010+a2P_010000000_1*QR_000020000020+a1P_000011000_1*QR_000020000100+a1P_000111000_1*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(P_010011000*QR_010000010000+P_010111000*QR_010000010010+a2P_010000000_1*QR_010000010020+a1P_000011000_1*QR_010000010100+a1P_000111000_1*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(P_010011000*QR_000010010000+P_010111000*QR_000010010010+a2P_010000000_1*QR_000010010020+a1P_000011000_1*QR_000010010100+a1P_000111000_1*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(P_010011000*QR_000000020000+P_010111000*QR_000000020010+a2P_010000000_1*QR_000000020020+a1P_000011000_1*QR_000000020100+a1P_000111000_1*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(P_010010001*QR_020000000000+a1P_010010000_1*QR_020000000001+a1P_010000001_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000010001_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000001_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(P_010010001*QR_010010000000+a1P_010010000_1*QR_010010000001+a1P_010000001_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000010001_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000001_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(P_010010001*QR_000020000000+a1P_010010000_1*QR_000020000001+a1P_010000001_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000010001_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000001_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(P_010010001*QR_010000010000+a1P_010010000_1*QR_010000010001+a1P_010000001_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000010001_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000001_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(P_010010001*QR_000010010000+a1P_010010000_1*QR_000010010001+a1P_010000001_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000010001_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000001_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(P_010010001*QR_000000020000+a1P_010010000_1*QR_000000020001+a1P_010000001_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000010001_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000001_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(P_001020000*QR_020000000000+a1P_001010000_2*QR_020000000010+a2P_001000000_1*QR_020000000020+a1P_000020000_1*QR_020000000100+a2P_000010000_2*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(P_001020000*QR_010010000000+a1P_001010000_2*QR_010010000010+a2P_001000000_1*QR_010010000020+a1P_000020000_1*QR_010010000100+a2P_000010000_2*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(P_001020000*QR_000020000000+a1P_001010000_2*QR_000020000010+a2P_001000000_1*QR_000020000020+a1P_000020000_1*QR_000020000100+a2P_000010000_2*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(P_001020000*QR_010000010000+a1P_001010000_2*QR_010000010010+a2P_001000000_1*QR_010000010020+a1P_000020000_1*QR_010000010100+a2P_000010000_2*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(P_001020000*QR_000010010000+a1P_001010000_2*QR_000010010010+a2P_001000000_1*QR_000010010020+a1P_000020000_1*QR_000010010100+a2P_000010000_2*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(P_001020000*QR_000000020000+a1P_001010000_2*QR_000000020010+a2P_001000000_1*QR_000000020020+a1P_000020000_1*QR_000000020100+a2P_000010000_2*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(P_000021000*QR_020000000000+P_000121000*QR_020000000010+P_000221000*QR_020000000020+aPin3*QR_020000000030);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(P_000021000*QR_010010000000+P_000121000*QR_010010000010+P_000221000*QR_010010000020+aPin3*QR_010010000030);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(P_000021000*QR_000020000000+P_000121000*QR_000020000010+P_000221000*QR_000020000020+aPin3*QR_000020000030);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(P_000021000*QR_010000010000+P_000121000*QR_010000010010+P_000221000*QR_010000010020+aPin3*QR_010000010030);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(P_000021000*QR_000010010000+P_000121000*QR_000010010010+P_000221000*QR_000010010020+aPin3*QR_000010010030);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(P_000021000*QR_000000020000+P_000121000*QR_000000020010+P_000221000*QR_000000020020+aPin3*QR_000000020030);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(P_000020001*QR_020000000000+a1P_000020000_1*QR_020000000001+a1P_000010001_2*QR_020000000010+a2P_000010000_2*QR_020000000011+a2P_000000001_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(P_000020001*QR_010010000000+a1P_000020000_1*QR_010010000001+a1P_000010001_2*QR_010010000010+a2P_000010000_2*QR_010010000011+a2P_000000001_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(P_000020001*QR_000020000000+a1P_000020000_1*QR_000020000001+a1P_000010001_2*QR_000020000010+a2P_000010000_2*QR_000020000011+a2P_000000001_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(P_000020001*QR_010000010000+a1P_000020000_1*QR_010000010001+a1P_000010001_2*QR_010000010010+a2P_000010000_2*QR_010000010011+a2P_000000001_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(P_000020001*QR_000010010000+a1P_000020000_1*QR_000010010001+a1P_000010001_2*QR_000010010010+a2P_000010000_2*QR_000010010011+a2P_000000001_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(P_000020001*QR_000000020000+a1P_000020000_1*QR_000000020001+a1P_000010001_2*QR_000000020010+a2P_000010000_2*QR_000000020011+a2P_000000001_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(P_011000010*QR_020000000000+a1P_011000000_1*QR_020000000001+P_111000010*QR_020000000100+a1P_111000000_1*QR_020000000101+a2P_000000010_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(P_011000010*QR_010010000000+a1P_011000000_1*QR_010010000001+P_111000010*QR_010010000100+a1P_111000000_1*QR_010010000101+a2P_000000010_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(P_011000010*QR_000020000000+a1P_011000000_1*QR_000020000001+P_111000010*QR_000020000100+a1P_111000000_1*QR_000020000101+a2P_000000010_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(P_011000010*QR_010000010000+a1P_011000000_1*QR_010000010001+P_111000010*QR_010000010100+a1P_111000000_1*QR_010000010101+a2P_000000010_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(P_011000010*QR_000010010000+a1P_011000000_1*QR_000010010001+P_111000010*QR_000010010100+a1P_111000000_1*QR_000010010101+a2P_000000010_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(P_011000010*QR_000000020000+a1P_011000000_1*QR_000000020001+P_111000010*QR_000000020100+a1P_111000000_1*QR_000000020101+a2P_000000010_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(P_010001010*QR_020000000000+a1P_010001000_1*QR_020000000001+a1P_010000010_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000001010_1*QR_020000000100+a2P_000001000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(P_010001010*QR_010010000000+a1P_010001000_1*QR_010010000001+a1P_010000010_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000001010_1*QR_010010000100+a2P_000001000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(P_010001010*QR_000020000000+a1P_010001000_1*QR_000020000001+a1P_010000010_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000001010_1*QR_000020000100+a2P_000001000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(P_010001010*QR_010000010000+a1P_010001000_1*QR_010000010001+a1P_010000010_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000001010_1*QR_010000010100+a2P_000001000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(P_010001010*QR_000010010000+a1P_010001000_1*QR_000010010001+a1P_010000010_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000001010_1*QR_000010010100+a2P_000001000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(P_010001010*QR_000000020000+a1P_010001000_1*QR_000000020001+a1P_010000010_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000001010_1*QR_000000020100+a2P_000001000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(P_010000011*QR_020000000000+P_010000111*QR_020000000001+a2P_010000000_1*QR_020000000002+a1P_000000011_1*QR_020000000100+a1P_000000111_1*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(P_010000011*QR_010010000000+P_010000111*QR_010010000001+a2P_010000000_1*QR_010010000002+a1P_000000011_1*QR_010010000100+a1P_000000111_1*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(P_010000011*QR_000020000000+P_010000111*QR_000020000001+a2P_010000000_1*QR_000020000002+a1P_000000011_1*QR_000020000100+a1P_000000111_1*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(P_010000011*QR_010000010000+P_010000111*QR_010000010001+a2P_010000000_1*QR_010000010002+a1P_000000011_1*QR_010000010100+a1P_000000111_1*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(P_010000011*QR_000010010000+P_010000111*QR_000010010001+a2P_010000000_1*QR_000010010002+a1P_000000011_1*QR_000010010100+a1P_000000111_1*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(P_010000011*QR_000000020000+P_010000111*QR_000000020001+a2P_010000000_1*QR_000000020002+a1P_000000011_1*QR_000000020100+a1P_000000111_1*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(P_001010010*QR_020000000000+a1P_001010000_1*QR_020000000001+a1P_001000010_1*QR_020000000010+a2P_001000000_1*QR_020000000011+a1P_000010010_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(P_001010010*QR_010010000000+a1P_001010000_1*QR_010010000001+a1P_001000010_1*QR_010010000010+a2P_001000000_1*QR_010010000011+a1P_000010010_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(P_001010010*QR_000020000000+a1P_001010000_1*QR_000020000001+a1P_001000010_1*QR_000020000010+a2P_001000000_1*QR_000020000011+a1P_000010010_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(P_001010010*QR_010000010000+a1P_001010000_1*QR_010000010001+a1P_001000010_1*QR_010000010010+a2P_001000000_1*QR_010000010011+a1P_000010010_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(P_001010010*QR_000010010000+a1P_001010000_1*QR_000010010001+a1P_001000010_1*QR_000010010010+a2P_001000000_1*QR_000010010011+a1P_000010010_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(P_001010010*QR_000000020000+a1P_001010000_1*QR_000000020001+a1P_001000010_1*QR_000000020010+a2P_001000000_1*QR_000000020011+a1P_000010010_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(P_000011010*QR_020000000000+a1P_000011000_1*QR_020000000001+P_000111010*QR_020000000010+a1P_000111000_1*QR_020000000011+a2P_000000010_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(P_000011010*QR_010010000000+a1P_000011000_1*QR_010010000001+P_000111010*QR_010010000010+a1P_000111000_1*QR_010010000011+a2P_000000010_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(P_000011010*QR_000020000000+a1P_000011000_1*QR_000020000001+P_000111010*QR_000020000010+a1P_000111000_1*QR_000020000011+a2P_000000010_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(P_000011010*QR_010000010000+a1P_000011000_1*QR_010000010001+P_000111010*QR_010000010010+a1P_000111000_1*QR_010000010011+a2P_000000010_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(P_000011010*QR_000010010000+a1P_000011000_1*QR_000010010001+P_000111010*QR_000010010010+a1P_000111000_1*QR_000010010011+a2P_000000010_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(P_000011010*QR_000000020000+a1P_000011000_1*QR_000000020001+P_000111010*QR_000000020010+a1P_000111000_1*QR_000000020011+a2P_000000010_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(P_000010011*QR_020000000000+P_000010111*QR_020000000001+a2P_000010000_1*QR_020000000002+a1P_000000011_1*QR_020000000010+a1P_000000111_1*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(P_000010011*QR_010010000000+P_000010111*QR_010010000001+a2P_000010000_1*QR_010010000002+a1P_000000011_1*QR_010010000010+a1P_000000111_1*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(P_000010011*QR_000020000000+P_000010111*QR_000020000001+a2P_000010000_1*QR_000020000002+a1P_000000011_1*QR_000020000010+a1P_000000111_1*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(P_000010011*QR_010000010000+P_000010111*QR_010000010001+a2P_000010000_1*QR_010000010002+a1P_000000011_1*QR_010000010010+a1P_000000111_1*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(P_000010011*QR_000010010000+P_000010111*QR_000010010001+a2P_000010000_1*QR_000010010002+a1P_000000011_1*QR_000010010010+a1P_000000111_1*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(P_000010011*QR_000000020000+P_000010111*QR_000000020001+a2P_000010000_1*QR_000000020002+a1P_000000011_1*QR_000000020010+a1P_000000111_1*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(P_001000020*QR_020000000000+a1P_001000010_2*QR_020000000001+a2P_001000000_1*QR_020000000002+a1P_000000020_1*QR_020000000100+a2P_000000010_2*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(P_001000020*QR_010010000000+a1P_001000010_2*QR_010010000001+a2P_001000000_1*QR_010010000002+a1P_000000020_1*QR_010010000100+a2P_000000010_2*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(P_001000020*QR_000020000000+a1P_001000010_2*QR_000020000001+a2P_001000000_1*QR_000020000002+a1P_000000020_1*QR_000020000100+a2P_000000010_2*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(P_001000020*QR_010000010000+a1P_001000010_2*QR_010000010001+a2P_001000000_1*QR_010000010002+a1P_000000020_1*QR_010000010100+a2P_000000010_2*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(P_001000020*QR_000010010000+a1P_001000010_2*QR_000010010001+a2P_001000000_1*QR_000010010002+a1P_000000020_1*QR_000010010100+a2P_000000010_2*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(P_001000020*QR_000000020000+a1P_001000010_2*QR_000000020001+a2P_001000000_1*QR_000000020002+a1P_000000020_1*QR_000000020100+a2P_000000010_2*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(P_000001020*QR_020000000000+a1P_000001010_2*QR_020000000001+a2P_000001000_1*QR_020000000002+a1P_000000020_1*QR_020000000010+a2P_000000010_2*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(P_000001020*QR_010010000000+a1P_000001010_2*QR_010010000001+a2P_000001000_1*QR_010010000002+a1P_000000020_1*QR_010010000010+a2P_000000010_2*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(P_000001020*QR_000020000000+a1P_000001010_2*QR_000020000001+a2P_000001000_1*QR_000020000002+a1P_000000020_1*QR_000020000010+a2P_000000010_2*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(P_000001020*QR_010000010000+a1P_000001010_2*QR_010000010001+a2P_000001000_1*QR_010000010002+a1P_000000020_1*QR_010000010010+a2P_000000010_2*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(P_000001020*QR_000010010000+a1P_000001010_2*QR_000010010001+a2P_000001000_1*QR_000010010002+a1P_000000020_1*QR_000010010010+a2P_000000010_2*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(P_000001020*QR_000000020000+a1P_000001010_2*QR_000000020001+a2P_000001000_1*QR_000000020002+a1P_000000020_1*QR_000000020010+a2P_000000010_2*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(P_000000021*QR_020000000000+P_000000121*QR_020000000001+P_000000221*QR_020000000002+aPin3*QR_020000000003);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(P_000000021*QR_010010000000+P_000000121*QR_010010000001+P_000000221*QR_010010000002+aPin3*QR_010010000003);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(P_000000021*QR_000020000000+P_000000121*QR_000020000001+P_000000221*QR_000020000002+aPin3*QR_000020000003);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(P_000000021*QR_010000010000+P_000000121*QR_010000010001+P_000000221*QR_010000010002+aPin3*QR_010000010003);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(P_000000021*QR_000010010000+P_000000121*QR_000010010001+P_000000221*QR_000010010002+aPin3*QR_000010010003);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(P_000000021*QR_000000020000+P_000000121*QR_000000020001+P_000000221*QR_000000020002+aPin3*QR_000000020003);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
}
__global__ void TSMJ_dpds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[jj*3+0];
				Qd_010[1]=QC[jj*3+1];
				Qd_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
			double P_021000000;
			double P_121000000;
			double P_221000000;
			double P_020001000;
			double P_020000001;
			double P_011010000;
			double P_111010000;
			double P_010011000;
			double P_010111000;
			double P_010010001;
			double P_001020000;
			double P_000021000;
			double P_000121000;
			double P_000221000;
			double P_000020001;
			double P_011000010;
			double P_111000010;
			double P_010001010;
			double P_010000011;
			double P_010000111;
			double P_001010010;
			double P_000011010;
			double P_000111010;
			double P_000010011;
			double P_000010111;
			double P_001000020;
			double P_000001020;
			double P_000000021;
			double P_000000121;
			double P_000000221;
			double a1P_020000000_1;
			double a1P_010001000_1;
			double a1P_010001000_2;
			double a2P_010000000_1;
			double a2P_010000000_2;
			double a2P_000001000_1;
			double a1P_010000001_1;
			double a1P_010000001_2;
			double a2P_000000001_1;
			double a1P_011000000_1;
			double a1P_111000000_1;
			double a2P_000010000_1;
			double a2P_000010000_2;
			double a1P_000011000_1;
			double a1P_000111000_1;
			double a1P_010010000_1;
			double a1P_000010001_1;
			double a1P_000010001_2;
			double a1P_001010000_1;
			double a1P_001010000_2;
			double a2P_001000000_1;
			double a1P_000020000_1;
			double a2P_000000010_1;
			double a2P_000000010_2;
			double a1P_010000010_1;
			double a1P_000001010_1;
			double a1P_000001010_2;
			double a1P_000000011_1;
			double a1P_000000111_1;
			double a1P_001000010_1;
			double a1P_001000010_2;
			double a1P_000010010_1;
			double a1P_000000020_1;
			P_021000000=Pd_021[0];
			P_121000000=Pd_121[0];
			P_221000000=Pd_221[0];
			P_020001000=Pd_020[0]*Pd_001[1];
			P_020000001=Pd_020[0]*Pd_001[2];
			P_011010000=Pd_011[0]*Pd_010[1];
			P_111010000=Pd_111[0]*Pd_010[1];
			P_010011000=Pd_010[0]*Pd_011[1];
			P_010111000=Pd_010[0]*Pd_111[1];
			P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
			P_001020000=Pd_001[0]*Pd_020[1];
			P_000021000=Pd_021[1];
			P_000121000=Pd_121[1];
			P_000221000=Pd_221[1];
			P_000020001=Pd_020[1]*Pd_001[2];
			P_011000010=Pd_011[0]*Pd_010[2];
			P_111000010=Pd_111[0]*Pd_010[2];
			P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
			P_010000011=Pd_010[0]*Pd_011[2];
			P_010000111=Pd_010[0]*Pd_111[2];
			P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
			P_000011010=Pd_011[1]*Pd_010[2];
			P_000111010=Pd_111[1]*Pd_010[2];
			P_000010011=Pd_010[1]*Pd_011[2];
			P_000010111=Pd_010[1]*Pd_111[2];
			P_001000020=Pd_001[0]*Pd_020[2];
			P_000001020=Pd_001[1]*Pd_020[2];
			P_000000021=Pd_021[2];
			P_000000121=Pd_121[2];
			P_000000221=Pd_221[2];
			a1P_020000000_1=Pd_020[0];
			a1P_010001000_1=Pd_010[0]*Pd_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=Pd_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=Pd_001[1];
			a1P_010000001_1=Pd_010[0]*Pd_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=Pd_001[2];
			a1P_011000000_1=Pd_011[0];
			a1P_111000000_1=Pd_111[0];
			a2P_000010000_1=Pd_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=Pd_011[1];
			a1P_000111000_1=Pd_111[1];
			a1P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010001_1=Pd_010[1]*Pd_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=Pd_001[0]*Pd_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=Pd_001[0];
			a1P_000020000_1=Pd_020[1];
			a2P_000000010_1=Pd_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000001010_1=Pd_001[1]*Pd_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=Pd_011[2];
			a1P_000000111_1=Pd_111[2];
			a1P_001000010_1=Pd_001[0]*Pd_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_000000020_1=Pd_020[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(P_021000000*QR_020000000000+P_121000000*QR_020000000100+P_221000000*QR_020000000200+aPin3*QR_020000000300);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(P_021000000*QR_010010000000+P_121000000*QR_010010000100+P_221000000*QR_010010000200+aPin3*QR_010010000300);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(P_021000000*QR_000020000000+P_121000000*QR_000020000100+P_221000000*QR_000020000200+aPin3*QR_000020000300);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(P_021000000*QR_010000010000+P_121000000*QR_010000010100+P_221000000*QR_010000010200+aPin3*QR_010000010300);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(P_021000000*QR_000010010000+P_121000000*QR_000010010100+P_221000000*QR_000010010200+aPin3*QR_000010010300);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(P_021000000*QR_000000020000+P_121000000*QR_000000020100+P_221000000*QR_000000020200+aPin3*QR_000000020300);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(P_020001000*QR_020000000000+a1P_020000000_1*QR_020000000010+a1P_010001000_2*QR_020000000100+a2P_010000000_2*QR_020000000110+a2P_000001000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(P_020001000*QR_010010000000+a1P_020000000_1*QR_010010000010+a1P_010001000_2*QR_010010000100+a2P_010000000_2*QR_010010000110+a2P_000001000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(P_020001000*QR_000020000000+a1P_020000000_1*QR_000020000010+a1P_010001000_2*QR_000020000100+a2P_010000000_2*QR_000020000110+a2P_000001000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(P_020001000*QR_010000010000+a1P_020000000_1*QR_010000010010+a1P_010001000_2*QR_010000010100+a2P_010000000_2*QR_010000010110+a2P_000001000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(P_020001000*QR_000010010000+a1P_020000000_1*QR_000010010010+a1P_010001000_2*QR_000010010100+a2P_010000000_2*QR_000010010110+a2P_000001000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(P_020001000*QR_000000020000+a1P_020000000_1*QR_000000020010+a1P_010001000_2*QR_000000020100+a2P_010000000_2*QR_000000020110+a2P_000001000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(P_020000001*QR_020000000000+a1P_020000000_1*QR_020000000001+a1P_010000001_2*QR_020000000100+a2P_010000000_2*QR_020000000101+a2P_000000001_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(P_020000001*QR_010010000000+a1P_020000000_1*QR_010010000001+a1P_010000001_2*QR_010010000100+a2P_010000000_2*QR_010010000101+a2P_000000001_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(P_020000001*QR_000020000000+a1P_020000000_1*QR_000020000001+a1P_010000001_2*QR_000020000100+a2P_010000000_2*QR_000020000101+a2P_000000001_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(P_020000001*QR_010000010000+a1P_020000000_1*QR_010000010001+a1P_010000001_2*QR_010000010100+a2P_010000000_2*QR_010000010101+a2P_000000001_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(P_020000001*QR_000010010000+a1P_020000000_1*QR_000010010001+a1P_010000001_2*QR_000010010100+a2P_010000000_2*QR_000010010101+a2P_000000001_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(P_020000001*QR_000000020000+a1P_020000000_1*QR_000000020001+a1P_010000001_2*QR_000000020100+a2P_010000000_2*QR_000000020101+a2P_000000001_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(P_011010000*QR_020000000000+a1P_011000000_1*QR_020000000010+P_111010000*QR_020000000100+a1P_111000000_1*QR_020000000110+a2P_000010000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(P_011010000*QR_010010000000+a1P_011000000_1*QR_010010000010+P_111010000*QR_010010000100+a1P_111000000_1*QR_010010000110+a2P_000010000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(P_011010000*QR_000020000000+a1P_011000000_1*QR_000020000010+P_111010000*QR_000020000100+a1P_111000000_1*QR_000020000110+a2P_000010000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(P_011010000*QR_010000010000+a1P_011000000_1*QR_010000010010+P_111010000*QR_010000010100+a1P_111000000_1*QR_010000010110+a2P_000010000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(P_011010000*QR_000010010000+a1P_011000000_1*QR_000010010010+P_111010000*QR_000010010100+a1P_111000000_1*QR_000010010110+a2P_000010000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(P_011010000*QR_000000020000+a1P_011000000_1*QR_000000020010+P_111010000*QR_000000020100+a1P_111000000_1*QR_000000020110+a2P_000010000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(P_010011000*QR_020000000000+P_010111000*QR_020000000010+a2P_010000000_1*QR_020000000020+a1P_000011000_1*QR_020000000100+a1P_000111000_1*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(P_010011000*QR_010010000000+P_010111000*QR_010010000010+a2P_010000000_1*QR_010010000020+a1P_000011000_1*QR_010010000100+a1P_000111000_1*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(P_010011000*QR_000020000000+P_010111000*QR_000020000010+a2P_010000000_1*QR_000020000020+a1P_000011000_1*QR_000020000100+a1P_000111000_1*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(P_010011000*QR_010000010000+P_010111000*QR_010000010010+a2P_010000000_1*QR_010000010020+a1P_000011000_1*QR_010000010100+a1P_000111000_1*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(P_010011000*QR_000010010000+P_010111000*QR_000010010010+a2P_010000000_1*QR_000010010020+a1P_000011000_1*QR_000010010100+a1P_000111000_1*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(P_010011000*QR_000000020000+P_010111000*QR_000000020010+a2P_010000000_1*QR_000000020020+a1P_000011000_1*QR_000000020100+a1P_000111000_1*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(P_010010001*QR_020000000000+a1P_010010000_1*QR_020000000001+a1P_010000001_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000010001_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000001_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(P_010010001*QR_010010000000+a1P_010010000_1*QR_010010000001+a1P_010000001_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000010001_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000001_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(P_010010001*QR_000020000000+a1P_010010000_1*QR_000020000001+a1P_010000001_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000010001_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000001_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(P_010010001*QR_010000010000+a1P_010010000_1*QR_010000010001+a1P_010000001_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000010001_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000001_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(P_010010001*QR_000010010000+a1P_010010000_1*QR_000010010001+a1P_010000001_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000010001_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000001_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(P_010010001*QR_000000020000+a1P_010010000_1*QR_000000020001+a1P_010000001_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000010001_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000001_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(P_001020000*QR_020000000000+a1P_001010000_2*QR_020000000010+a2P_001000000_1*QR_020000000020+a1P_000020000_1*QR_020000000100+a2P_000010000_2*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(P_001020000*QR_010010000000+a1P_001010000_2*QR_010010000010+a2P_001000000_1*QR_010010000020+a1P_000020000_1*QR_010010000100+a2P_000010000_2*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(P_001020000*QR_000020000000+a1P_001010000_2*QR_000020000010+a2P_001000000_1*QR_000020000020+a1P_000020000_1*QR_000020000100+a2P_000010000_2*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(P_001020000*QR_010000010000+a1P_001010000_2*QR_010000010010+a2P_001000000_1*QR_010000010020+a1P_000020000_1*QR_010000010100+a2P_000010000_2*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(P_001020000*QR_000010010000+a1P_001010000_2*QR_000010010010+a2P_001000000_1*QR_000010010020+a1P_000020000_1*QR_000010010100+a2P_000010000_2*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(P_001020000*QR_000000020000+a1P_001010000_2*QR_000000020010+a2P_001000000_1*QR_000000020020+a1P_000020000_1*QR_000000020100+a2P_000010000_2*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(P_000021000*QR_020000000000+P_000121000*QR_020000000010+P_000221000*QR_020000000020+aPin3*QR_020000000030);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(P_000021000*QR_010010000000+P_000121000*QR_010010000010+P_000221000*QR_010010000020+aPin3*QR_010010000030);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(P_000021000*QR_000020000000+P_000121000*QR_000020000010+P_000221000*QR_000020000020+aPin3*QR_000020000030);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(P_000021000*QR_010000010000+P_000121000*QR_010000010010+P_000221000*QR_010000010020+aPin3*QR_010000010030);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(P_000021000*QR_000010010000+P_000121000*QR_000010010010+P_000221000*QR_000010010020+aPin3*QR_000010010030);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(P_000021000*QR_000000020000+P_000121000*QR_000000020010+P_000221000*QR_000000020020+aPin3*QR_000000020030);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(P_000020001*QR_020000000000+a1P_000020000_1*QR_020000000001+a1P_000010001_2*QR_020000000010+a2P_000010000_2*QR_020000000011+a2P_000000001_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(P_000020001*QR_010010000000+a1P_000020000_1*QR_010010000001+a1P_000010001_2*QR_010010000010+a2P_000010000_2*QR_010010000011+a2P_000000001_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(P_000020001*QR_000020000000+a1P_000020000_1*QR_000020000001+a1P_000010001_2*QR_000020000010+a2P_000010000_2*QR_000020000011+a2P_000000001_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(P_000020001*QR_010000010000+a1P_000020000_1*QR_010000010001+a1P_000010001_2*QR_010000010010+a2P_000010000_2*QR_010000010011+a2P_000000001_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(P_000020001*QR_000010010000+a1P_000020000_1*QR_000010010001+a1P_000010001_2*QR_000010010010+a2P_000010000_2*QR_000010010011+a2P_000000001_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(P_000020001*QR_000000020000+a1P_000020000_1*QR_000000020001+a1P_000010001_2*QR_000000020010+a2P_000010000_2*QR_000000020011+a2P_000000001_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(P_011000010*QR_020000000000+a1P_011000000_1*QR_020000000001+P_111000010*QR_020000000100+a1P_111000000_1*QR_020000000101+a2P_000000010_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(P_011000010*QR_010010000000+a1P_011000000_1*QR_010010000001+P_111000010*QR_010010000100+a1P_111000000_1*QR_010010000101+a2P_000000010_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(P_011000010*QR_000020000000+a1P_011000000_1*QR_000020000001+P_111000010*QR_000020000100+a1P_111000000_1*QR_000020000101+a2P_000000010_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(P_011000010*QR_010000010000+a1P_011000000_1*QR_010000010001+P_111000010*QR_010000010100+a1P_111000000_1*QR_010000010101+a2P_000000010_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(P_011000010*QR_000010010000+a1P_011000000_1*QR_000010010001+P_111000010*QR_000010010100+a1P_111000000_1*QR_000010010101+a2P_000000010_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(P_011000010*QR_000000020000+a1P_011000000_1*QR_000000020001+P_111000010*QR_000000020100+a1P_111000000_1*QR_000000020101+a2P_000000010_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(P_010001010*QR_020000000000+a1P_010001000_1*QR_020000000001+a1P_010000010_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000001010_1*QR_020000000100+a2P_000001000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(P_010001010*QR_010010000000+a1P_010001000_1*QR_010010000001+a1P_010000010_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000001010_1*QR_010010000100+a2P_000001000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(P_010001010*QR_000020000000+a1P_010001000_1*QR_000020000001+a1P_010000010_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000001010_1*QR_000020000100+a2P_000001000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(P_010001010*QR_010000010000+a1P_010001000_1*QR_010000010001+a1P_010000010_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000001010_1*QR_010000010100+a2P_000001000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(P_010001010*QR_000010010000+a1P_010001000_1*QR_000010010001+a1P_010000010_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000001010_1*QR_000010010100+a2P_000001000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(P_010001010*QR_000000020000+a1P_010001000_1*QR_000000020001+a1P_010000010_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000001010_1*QR_000000020100+a2P_000001000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(P_010000011*QR_020000000000+P_010000111*QR_020000000001+a2P_010000000_1*QR_020000000002+a1P_000000011_1*QR_020000000100+a1P_000000111_1*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(P_010000011*QR_010010000000+P_010000111*QR_010010000001+a2P_010000000_1*QR_010010000002+a1P_000000011_1*QR_010010000100+a1P_000000111_1*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(P_010000011*QR_000020000000+P_010000111*QR_000020000001+a2P_010000000_1*QR_000020000002+a1P_000000011_1*QR_000020000100+a1P_000000111_1*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(P_010000011*QR_010000010000+P_010000111*QR_010000010001+a2P_010000000_1*QR_010000010002+a1P_000000011_1*QR_010000010100+a1P_000000111_1*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(P_010000011*QR_000010010000+P_010000111*QR_000010010001+a2P_010000000_1*QR_000010010002+a1P_000000011_1*QR_000010010100+a1P_000000111_1*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(P_010000011*QR_000000020000+P_010000111*QR_000000020001+a2P_010000000_1*QR_000000020002+a1P_000000011_1*QR_000000020100+a1P_000000111_1*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(P_001010010*QR_020000000000+a1P_001010000_1*QR_020000000001+a1P_001000010_1*QR_020000000010+a2P_001000000_1*QR_020000000011+a1P_000010010_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(P_001010010*QR_010010000000+a1P_001010000_1*QR_010010000001+a1P_001000010_1*QR_010010000010+a2P_001000000_1*QR_010010000011+a1P_000010010_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(P_001010010*QR_000020000000+a1P_001010000_1*QR_000020000001+a1P_001000010_1*QR_000020000010+a2P_001000000_1*QR_000020000011+a1P_000010010_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(P_001010010*QR_010000010000+a1P_001010000_1*QR_010000010001+a1P_001000010_1*QR_010000010010+a2P_001000000_1*QR_010000010011+a1P_000010010_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(P_001010010*QR_000010010000+a1P_001010000_1*QR_000010010001+a1P_001000010_1*QR_000010010010+a2P_001000000_1*QR_000010010011+a1P_000010010_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(P_001010010*QR_000000020000+a1P_001010000_1*QR_000000020001+a1P_001000010_1*QR_000000020010+a2P_001000000_1*QR_000000020011+a1P_000010010_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(P_000011010*QR_020000000000+a1P_000011000_1*QR_020000000001+P_000111010*QR_020000000010+a1P_000111000_1*QR_020000000011+a2P_000000010_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(P_000011010*QR_010010000000+a1P_000011000_1*QR_010010000001+P_000111010*QR_010010000010+a1P_000111000_1*QR_010010000011+a2P_000000010_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(P_000011010*QR_000020000000+a1P_000011000_1*QR_000020000001+P_000111010*QR_000020000010+a1P_000111000_1*QR_000020000011+a2P_000000010_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(P_000011010*QR_010000010000+a1P_000011000_1*QR_010000010001+P_000111010*QR_010000010010+a1P_000111000_1*QR_010000010011+a2P_000000010_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(P_000011010*QR_000010010000+a1P_000011000_1*QR_000010010001+P_000111010*QR_000010010010+a1P_000111000_1*QR_000010010011+a2P_000000010_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(P_000011010*QR_000000020000+a1P_000011000_1*QR_000000020001+P_000111010*QR_000000020010+a1P_000111000_1*QR_000000020011+a2P_000000010_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(P_000010011*QR_020000000000+P_000010111*QR_020000000001+a2P_000010000_1*QR_020000000002+a1P_000000011_1*QR_020000000010+a1P_000000111_1*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(P_000010011*QR_010010000000+P_000010111*QR_010010000001+a2P_000010000_1*QR_010010000002+a1P_000000011_1*QR_010010000010+a1P_000000111_1*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(P_000010011*QR_000020000000+P_000010111*QR_000020000001+a2P_000010000_1*QR_000020000002+a1P_000000011_1*QR_000020000010+a1P_000000111_1*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(P_000010011*QR_010000010000+P_000010111*QR_010000010001+a2P_000010000_1*QR_010000010002+a1P_000000011_1*QR_010000010010+a1P_000000111_1*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(P_000010011*QR_000010010000+P_000010111*QR_000010010001+a2P_000010000_1*QR_000010010002+a1P_000000011_1*QR_000010010010+a1P_000000111_1*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(P_000010011*QR_000000020000+P_000010111*QR_000000020001+a2P_000010000_1*QR_000000020002+a1P_000000011_1*QR_000000020010+a1P_000000111_1*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(P_001000020*QR_020000000000+a1P_001000010_2*QR_020000000001+a2P_001000000_1*QR_020000000002+a1P_000000020_1*QR_020000000100+a2P_000000010_2*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(P_001000020*QR_010010000000+a1P_001000010_2*QR_010010000001+a2P_001000000_1*QR_010010000002+a1P_000000020_1*QR_010010000100+a2P_000000010_2*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(P_001000020*QR_000020000000+a1P_001000010_2*QR_000020000001+a2P_001000000_1*QR_000020000002+a1P_000000020_1*QR_000020000100+a2P_000000010_2*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(P_001000020*QR_010000010000+a1P_001000010_2*QR_010000010001+a2P_001000000_1*QR_010000010002+a1P_000000020_1*QR_010000010100+a2P_000000010_2*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(P_001000020*QR_000010010000+a1P_001000010_2*QR_000010010001+a2P_001000000_1*QR_000010010002+a1P_000000020_1*QR_000010010100+a2P_000000010_2*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(P_001000020*QR_000000020000+a1P_001000010_2*QR_000000020001+a2P_001000000_1*QR_000000020002+a1P_000000020_1*QR_000000020100+a2P_000000010_2*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(P_000001020*QR_020000000000+a1P_000001010_2*QR_020000000001+a2P_000001000_1*QR_020000000002+a1P_000000020_1*QR_020000000010+a2P_000000010_2*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(P_000001020*QR_010010000000+a1P_000001010_2*QR_010010000001+a2P_000001000_1*QR_010010000002+a1P_000000020_1*QR_010010000010+a2P_000000010_2*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(P_000001020*QR_000020000000+a1P_000001010_2*QR_000020000001+a2P_000001000_1*QR_000020000002+a1P_000000020_1*QR_000020000010+a2P_000000010_2*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(P_000001020*QR_010000010000+a1P_000001010_2*QR_010000010001+a2P_000001000_1*QR_010000010002+a1P_000000020_1*QR_010000010010+a2P_000000010_2*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(P_000001020*QR_000010010000+a1P_000001010_2*QR_000010010001+a2P_000001000_1*QR_000010010002+a1P_000000020_1*QR_000010010010+a2P_000000010_2*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(P_000001020*QR_000000020000+a1P_000001010_2*QR_000000020001+a2P_000001000_1*QR_000000020002+a1P_000000020_1*QR_000000020010+a2P_000000010_2*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(P_000000021*QR_020000000000+P_000000121*QR_020000000001+P_000000221*QR_020000000002+aPin3*QR_020000000003);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(P_000000021*QR_010010000000+P_000000121*QR_010010000001+P_000000221*QR_010010000002+aPin3*QR_010010000003);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(P_000000021*QR_000020000000+P_000000121*QR_000020000001+P_000000221*QR_000020000002+aPin3*QR_000020000003);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(P_000000021*QR_010000010000+P_000000121*QR_010000010001+P_000000221*QR_010000010002+aPin3*QR_010000010003);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(P_000000021*QR_000010010000+P_000000121*QR_000010010001+P_000000221*QR_000010010002+aPin3*QR_000010010003);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(P_000000021*QR_000000020000+P_000000121*QR_000000020001+P_000000221*QR_000000020002+aPin3*QR_000000020003);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
}
__global__ void TSMJ_dpds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
			double P_021000000;
			double P_121000000;
			double P_221000000;
			double P_020001000;
			double P_020000001;
			double P_011010000;
			double P_111010000;
			double P_010011000;
			double P_010111000;
			double P_010010001;
			double P_001020000;
			double P_000021000;
			double P_000121000;
			double P_000221000;
			double P_000020001;
			double P_011000010;
			double P_111000010;
			double P_010001010;
			double P_010000011;
			double P_010000111;
			double P_001010010;
			double P_000011010;
			double P_000111010;
			double P_000010011;
			double P_000010111;
			double P_001000020;
			double P_000001020;
			double P_000000021;
			double P_000000121;
			double P_000000221;
			double a1P_020000000_1;
			double a1P_010001000_1;
			double a1P_010001000_2;
			double a2P_010000000_1;
			double a2P_010000000_2;
			double a2P_000001000_1;
			double a1P_010000001_1;
			double a1P_010000001_2;
			double a2P_000000001_1;
			double a1P_011000000_1;
			double a1P_111000000_1;
			double a2P_000010000_1;
			double a2P_000010000_2;
			double a1P_000011000_1;
			double a1P_000111000_1;
			double a1P_010010000_1;
			double a1P_000010001_1;
			double a1P_000010001_2;
			double a1P_001010000_1;
			double a1P_001010000_2;
			double a2P_001000000_1;
			double a1P_000020000_1;
			double a2P_000000010_1;
			double a2P_000000010_2;
			double a1P_010000010_1;
			double a1P_000001010_1;
			double a1P_000001010_2;
			double a1P_000000011_1;
			double a1P_000000111_1;
			double a1P_001000010_1;
			double a1P_001000010_2;
			double a1P_000010010_1;
			double a1P_000000020_1;
			P_021000000=Pd_021[0];
			P_121000000=Pd_121[0];
			P_221000000=Pd_221[0];
			P_020001000=Pd_020[0]*Pd_001[1];
			P_020000001=Pd_020[0]*Pd_001[2];
			P_011010000=Pd_011[0]*Pd_010[1];
			P_111010000=Pd_111[0]*Pd_010[1];
			P_010011000=Pd_010[0]*Pd_011[1];
			P_010111000=Pd_010[0]*Pd_111[1];
			P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
			P_001020000=Pd_001[0]*Pd_020[1];
			P_000021000=Pd_021[1];
			P_000121000=Pd_121[1];
			P_000221000=Pd_221[1];
			P_000020001=Pd_020[1]*Pd_001[2];
			P_011000010=Pd_011[0]*Pd_010[2];
			P_111000010=Pd_111[0]*Pd_010[2];
			P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
			P_010000011=Pd_010[0]*Pd_011[2];
			P_010000111=Pd_010[0]*Pd_111[2];
			P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
			P_000011010=Pd_011[1]*Pd_010[2];
			P_000111010=Pd_111[1]*Pd_010[2];
			P_000010011=Pd_010[1]*Pd_011[2];
			P_000010111=Pd_010[1]*Pd_111[2];
			P_001000020=Pd_001[0]*Pd_020[2];
			P_000001020=Pd_001[1]*Pd_020[2];
			P_000000021=Pd_021[2];
			P_000000121=Pd_121[2];
			P_000000221=Pd_221[2];
			a1P_020000000_1=Pd_020[0];
			a1P_010001000_1=Pd_010[0]*Pd_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=Pd_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=Pd_001[1];
			a1P_010000001_1=Pd_010[0]*Pd_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=Pd_001[2];
			a1P_011000000_1=Pd_011[0];
			a1P_111000000_1=Pd_111[0];
			a2P_000010000_1=Pd_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=Pd_011[1];
			a1P_000111000_1=Pd_111[1];
			a1P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010001_1=Pd_010[1]*Pd_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=Pd_001[0]*Pd_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=Pd_001[0];
			a1P_000020000_1=Pd_020[1];
			a2P_000000010_1=Pd_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000001010_1=Pd_001[1]*Pd_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=Pd_011[2];
			a1P_000000111_1=Pd_111[2];
			a1P_001000010_1=Pd_001[0]*Pd_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_000000020_1=Pd_020[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(P_021000000*QR_020000000000+P_121000000*QR_020000000100+P_221000000*QR_020000000200+aPin3*QR_020000000300);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(P_021000000*QR_010010000000+P_121000000*QR_010010000100+P_221000000*QR_010010000200+aPin3*QR_010010000300);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(P_021000000*QR_000020000000+P_121000000*QR_000020000100+P_221000000*QR_000020000200+aPin3*QR_000020000300);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(P_021000000*QR_010000010000+P_121000000*QR_010000010100+P_221000000*QR_010000010200+aPin3*QR_010000010300);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(P_021000000*QR_000010010000+P_121000000*QR_000010010100+P_221000000*QR_000010010200+aPin3*QR_000010010300);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(P_021000000*QR_000000020000+P_121000000*QR_000000020100+P_221000000*QR_000000020200+aPin3*QR_000000020300);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(P_020001000*QR_020000000000+a1P_020000000_1*QR_020000000010+a1P_010001000_2*QR_020000000100+a2P_010000000_2*QR_020000000110+a2P_000001000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(P_020001000*QR_010010000000+a1P_020000000_1*QR_010010000010+a1P_010001000_2*QR_010010000100+a2P_010000000_2*QR_010010000110+a2P_000001000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(P_020001000*QR_000020000000+a1P_020000000_1*QR_000020000010+a1P_010001000_2*QR_000020000100+a2P_010000000_2*QR_000020000110+a2P_000001000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(P_020001000*QR_010000010000+a1P_020000000_1*QR_010000010010+a1P_010001000_2*QR_010000010100+a2P_010000000_2*QR_010000010110+a2P_000001000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(P_020001000*QR_000010010000+a1P_020000000_1*QR_000010010010+a1P_010001000_2*QR_000010010100+a2P_010000000_2*QR_000010010110+a2P_000001000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(P_020001000*QR_000000020000+a1P_020000000_1*QR_000000020010+a1P_010001000_2*QR_000000020100+a2P_010000000_2*QR_000000020110+a2P_000001000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(P_020000001*QR_020000000000+a1P_020000000_1*QR_020000000001+a1P_010000001_2*QR_020000000100+a2P_010000000_2*QR_020000000101+a2P_000000001_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(P_020000001*QR_010010000000+a1P_020000000_1*QR_010010000001+a1P_010000001_2*QR_010010000100+a2P_010000000_2*QR_010010000101+a2P_000000001_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(P_020000001*QR_000020000000+a1P_020000000_1*QR_000020000001+a1P_010000001_2*QR_000020000100+a2P_010000000_2*QR_000020000101+a2P_000000001_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(P_020000001*QR_010000010000+a1P_020000000_1*QR_010000010001+a1P_010000001_2*QR_010000010100+a2P_010000000_2*QR_010000010101+a2P_000000001_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(P_020000001*QR_000010010000+a1P_020000000_1*QR_000010010001+a1P_010000001_2*QR_000010010100+a2P_010000000_2*QR_000010010101+a2P_000000001_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(P_020000001*QR_000000020000+a1P_020000000_1*QR_000000020001+a1P_010000001_2*QR_000000020100+a2P_010000000_2*QR_000000020101+a2P_000000001_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(P_011010000*QR_020000000000+a1P_011000000_1*QR_020000000010+P_111010000*QR_020000000100+a1P_111000000_1*QR_020000000110+a2P_000010000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(P_011010000*QR_010010000000+a1P_011000000_1*QR_010010000010+P_111010000*QR_010010000100+a1P_111000000_1*QR_010010000110+a2P_000010000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(P_011010000*QR_000020000000+a1P_011000000_1*QR_000020000010+P_111010000*QR_000020000100+a1P_111000000_1*QR_000020000110+a2P_000010000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(P_011010000*QR_010000010000+a1P_011000000_1*QR_010000010010+P_111010000*QR_010000010100+a1P_111000000_1*QR_010000010110+a2P_000010000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(P_011010000*QR_000010010000+a1P_011000000_1*QR_000010010010+P_111010000*QR_000010010100+a1P_111000000_1*QR_000010010110+a2P_000010000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(P_011010000*QR_000000020000+a1P_011000000_1*QR_000000020010+P_111010000*QR_000000020100+a1P_111000000_1*QR_000000020110+a2P_000010000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(P_010011000*QR_020000000000+P_010111000*QR_020000000010+a2P_010000000_1*QR_020000000020+a1P_000011000_1*QR_020000000100+a1P_000111000_1*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(P_010011000*QR_010010000000+P_010111000*QR_010010000010+a2P_010000000_1*QR_010010000020+a1P_000011000_1*QR_010010000100+a1P_000111000_1*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(P_010011000*QR_000020000000+P_010111000*QR_000020000010+a2P_010000000_1*QR_000020000020+a1P_000011000_1*QR_000020000100+a1P_000111000_1*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(P_010011000*QR_010000010000+P_010111000*QR_010000010010+a2P_010000000_1*QR_010000010020+a1P_000011000_1*QR_010000010100+a1P_000111000_1*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(P_010011000*QR_000010010000+P_010111000*QR_000010010010+a2P_010000000_1*QR_000010010020+a1P_000011000_1*QR_000010010100+a1P_000111000_1*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(P_010011000*QR_000000020000+P_010111000*QR_000000020010+a2P_010000000_1*QR_000000020020+a1P_000011000_1*QR_000000020100+a1P_000111000_1*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(P_010010001*QR_020000000000+a1P_010010000_1*QR_020000000001+a1P_010000001_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000010001_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000001_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(P_010010001*QR_010010000000+a1P_010010000_1*QR_010010000001+a1P_010000001_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000010001_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000001_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(P_010010001*QR_000020000000+a1P_010010000_1*QR_000020000001+a1P_010000001_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000010001_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000001_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(P_010010001*QR_010000010000+a1P_010010000_1*QR_010000010001+a1P_010000001_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000010001_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000001_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(P_010010001*QR_000010010000+a1P_010010000_1*QR_000010010001+a1P_010000001_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000010001_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000001_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(P_010010001*QR_000000020000+a1P_010010000_1*QR_000000020001+a1P_010000001_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000010001_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000001_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(P_001020000*QR_020000000000+a1P_001010000_2*QR_020000000010+a2P_001000000_1*QR_020000000020+a1P_000020000_1*QR_020000000100+a2P_000010000_2*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(P_001020000*QR_010010000000+a1P_001010000_2*QR_010010000010+a2P_001000000_1*QR_010010000020+a1P_000020000_1*QR_010010000100+a2P_000010000_2*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(P_001020000*QR_000020000000+a1P_001010000_2*QR_000020000010+a2P_001000000_1*QR_000020000020+a1P_000020000_1*QR_000020000100+a2P_000010000_2*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(P_001020000*QR_010000010000+a1P_001010000_2*QR_010000010010+a2P_001000000_1*QR_010000010020+a1P_000020000_1*QR_010000010100+a2P_000010000_2*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(P_001020000*QR_000010010000+a1P_001010000_2*QR_000010010010+a2P_001000000_1*QR_000010010020+a1P_000020000_1*QR_000010010100+a2P_000010000_2*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(P_001020000*QR_000000020000+a1P_001010000_2*QR_000000020010+a2P_001000000_1*QR_000000020020+a1P_000020000_1*QR_000000020100+a2P_000010000_2*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(P_000021000*QR_020000000000+P_000121000*QR_020000000010+P_000221000*QR_020000000020+aPin3*QR_020000000030);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(P_000021000*QR_010010000000+P_000121000*QR_010010000010+P_000221000*QR_010010000020+aPin3*QR_010010000030);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(P_000021000*QR_000020000000+P_000121000*QR_000020000010+P_000221000*QR_000020000020+aPin3*QR_000020000030);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(P_000021000*QR_010000010000+P_000121000*QR_010000010010+P_000221000*QR_010000010020+aPin3*QR_010000010030);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(P_000021000*QR_000010010000+P_000121000*QR_000010010010+P_000221000*QR_000010010020+aPin3*QR_000010010030);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(P_000021000*QR_000000020000+P_000121000*QR_000000020010+P_000221000*QR_000000020020+aPin3*QR_000000020030);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(P_000020001*QR_020000000000+a1P_000020000_1*QR_020000000001+a1P_000010001_2*QR_020000000010+a2P_000010000_2*QR_020000000011+a2P_000000001_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(P_000020001*QR_010010000000+a1P_000020000_1*QR_010010000001+a1P_000010001_2*QR_010010000010+a2P_000010000_2*QR_010010000011+a2P_000000001_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(P_000020001*QR_000020000000+a1P_000020000_1*QR_000020000001+a1P_000010001_2*QR_000020000010+a2P_000010000_2*QR_000020000011+a2P_000000001_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(P_000020001*QR_010000010000+a1P_000020000_1*QR_010000010001+a1P_000010001_2*QR_010000010010+a2P_000010000_2*QR_010000010011+a2P_000000001_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(P_000020001*QR_000010010000+a1P_000020000_1*QR_000010010001+a1P_000010001_2*QR_000010010010+a2P_000010000_2*QR_000010010011+a2P_000000001_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(P_000020001*QR_000000020000+a1P_000020000_1*QR_000000020001+a1P_000010001_2*QR_000000020010+a2P_000010000_2*QR_000000020011+a2P_000000001_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(P_011000010*QR_020000000000+a1P_011000000_1*QR_020000000001+P_111000010*QR_020000000100+a1P_111000000_1*QR_020000000101+a2P_000000010_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(P_011000010*QR_010010000000+a1P_011000000_1*QR_010010000001+P_111000010*QR_010010000100+a1P_111000000_1*QR_010010000101+a2P_000000010_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(P_011000010*QR_000020000000+a1P_011000000_1*QR_000020000001+P_111000010*QR_000020000100+a1P_111000000_1*QR_000020000101+a2P_000000010_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(P_011000010*QR_010000010000+a1P_011000000_1*QR_010000010001+P_111000010*QR_010000010100+a1P_111000000_1*QR_010000010101+a2P_000000010_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(P_011000010*QR_000010010000+a1P_011000000_1*QR_000010010001+P_111000010*QR_000010010100+a1P_111000000_1*QR_000010010101+a2P_000000010_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(P_011000010*QR_000000020000+a1P_011000000_1*QR_000000020001+P_111000010*QR_000000020100+a1P_111000000_1*QR_000000020101+a2P_000000010_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(P_010001010*QR_020000000000+a1P_010001000_1*QR_020000000001+a1P_010000010_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000001010_1*QR_020000000100+a2P_000001000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(P_010001010*QR_010010000000+a1P_010001000_1*QR_010010000001+a1P_010000010_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000001010_1*QR_010010000100+a2P_000001000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(P_010001010*QR_000020000000+a1P_010001000_1*QR_000020000001+a1P_010000010_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000001010_1*QR_000020000100+a2P_000001000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(P_010001010*QR_010000010000+a1P_010001000_1*QR_010000010001+a1P_010000010_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000001010_1*QR_010000010100+a2P_000001000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(P_010001010*QR_000010010000+a1P_010001000_1*QR_000010010001+a1P_010000010_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000001010_1*QR_000010010100+a2P_000001000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(P_010001010*QR_000000020000+a1P_010001000_1*QR_000000020001+a1P_010000010_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000001010_1*QR_000000020100+a2P_000001000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(P_010000011*QR_020000000000+P_010000111*QR_020000000001+a2P_010000000_1*QR_020000000002+a1P_000000011_1*QR_020000000100+a1P_000000111_1*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(P_010000011*QR_010010000000+P_010000111*QR_010010000001+a2P_010000000_1*QR_010010000002+a1P_000000011_1*QR_010010000100+a1P_000000111_1*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(P_010000011*QR_000020000000+P_010000111*QR_000020000001+a2P_010000000_1*QR_000020000002+a1P_000000011_1*QR_000020000100+a1P_000000111_1*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(P_010000011*QR_010000010000+P_010000111*QR_010000010001+a2P_010000000_1*QR_010000010002+a1P_000000011_1*QR_010000010100+a1P_000000111_1*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(P_010000011*QR_000010010000+P_010000111*QR_000010010001+a2P_010000000_1*QR_000010010002+a1P_000000011_1*QR_000010010100+a1P_000000111_1*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(P_010000011*QR_000000020000+P_010000111*QR_000000020001+a2P_010000000_1*QR_000000020002+a1P_000000011_1*QR_000000020100+a1P_000000111_1*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(P_001010010*QR_020000000000+a1P_001010000_1*QR_020000000001+a1P_001000010_1*QR_020000000010+a2P_001000000_1*QR_020000000011+a1P_000010010_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(P_001010010*QR_010010000000+a1P_001010000_1*QR_010010000001+a1P_001000010_1*QR_010010000010+a2P_001000000_1*QR_010010000011+a1P_000010010_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(P_001010010*QR_000020000000+a1P_001010000_1*QR_000020000001+a1P_001000010_1*QR_000020000010+a2P_001000000_1*QR_000020000011+a1P_000010010_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(P_001010010*QR_010000010000+a1P_001010000_1*QR_010000010001+a1P_001000010_1*QR_010000010010+a2P_001000000_1*QR_010000010011+a1P_000010010_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(P_001010010*QR_000010010000+a1P_001010000_1*QR_000010010001+a1P_001000010_1*QR_000010010010+a2P_001000000_1*QR_000010010011+a1P_000010010_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(P_001010010*QR_000000020000+a1P_001010000_1*QR_000000020001+a1P_001000010_1*QR_000000020010+a2P_001000000_1*QR_000000020011+a1P_000010010_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(P_000011010*QR_020000000000+a1P_000011000_1*QR_020000000001+P_000111010*QR_020000000010+a1P_000111000_1*QR_020000000011+a2P_000000010_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(P_000011010*QR_010010000000+a1P_000011000_1*QR_010010000001+P_000111010*QR_010010000010+a1P_000111000_1*QR_010010000011+a2P_000000010_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(P_000011010*QR_000020000000+a1P_000011000_1*QR_000020000001+P_000111010*QR_000020000010+a1P_000111000_1*QR_000020000011+a2P_000000010_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(P_000011010*QR_010000010000+a1P_000011000_1*QR_010000010001+P_000111010*QR_010000010010+a1P_000111000_1*QR_010000010011+a2P_000000010_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(P_000011010*QR_000010010000+a1P_000011000_1*QR_000010010001+P_000111010*QR_000010010010+a1P_000111000_1*QR_000010010011+a2P_000000010_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(P_000011010*QR_000000020000+a1P_000011000_1*QR_000000020001+P_000111010*QR_000000020010+a1P_000111000_1*QR_000000020011+a2P_000000010_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(P_000010011*QR_020000000000+P_000010111*QR_020000000001+a2P_000010000_1*QR_020000000002+a1P_000000011_1*QR_020000000010+a1P_000000111_1*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(P_000010011*QR_010010000000+P_000010111*QR_010010000001+a2P_000010000_1*QR_010010000002+a1P_000000011_1*QR_010010000010+a1P_000000111_1*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(P_000010011*QR_000020000000+P_000010111*QR_000020000001+a2P_000010000_1*QR_000020000002+a1P_000000011_1*QR_000020000010+a1P_000000111_1*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(P_000010011*QR_010000010000+P_000010111*QR_010000010001+a2P_000010000_1*QR_010000010002+a1P_000000011_1*QR_010000010010+a1P_000000111_1*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(P_000010011*QR_000010010000+P_000010111*QR_000010010001+a2P_000010000_1*QR_000010010002+a1P_000000011_1*QR_000010010010+a1P_000000111_1*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(P_000010011*QR_000000020000+P_000010111*QR_000000020001+a2P_000010000_1*QR_000000020002+a1P_000000011_1*QR_000000020010+a1P_000000111_1*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(P_001000020*QR_020000000000+a1P_001000010_2*QR_020000000001+a2P_001000000_1*QR_020000000002+a1P_000000020_1*QR_020000000100+a2P_000000010_2*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(P_001000020*QR_010010000000+a1P_001000010_2*QR_010010000001+a2P_001000000_1*QR_010010000002+a1P_000000020_1*QR_010010000100+a2P_000000010_2*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(P_001000020*QR_000020000000+a1P_001000010_2*QR_000020000001+a2P_001000000_1*QR_000020000002+a1P_000000020_1*QR_000020000100+a2P_000000010_2*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(P_001000020*QR_010000010000+a1P_001000010_2*QR_010000010001+a2P_001000000_1*QR_010000010002+a1P_000000020_1*QR_010000010100+a2P_000000010_2*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(P_001000020*QR_000010010000+a1P_001000010_2*QR_000010010001+a2P_001000000_1*QR_000010010002+a1P_000000020_1*QR_000010010100+a2P_000000010_2*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(P_001000020*QR_000000020000+a1P_001000010_2*QR_000000020001+a2P_001000000_1*QR_000000020002+a1P_000000020_1*QR_000000020100+a2P_000000010_2*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(P_000001020*QR_020000000000+a1P_000001010_2*QR_020000000001+a2P_000001000_1*QR_020000000002+a1P_000000020_1*QR_020000000010+a2P_000000010_2*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(P_000001020*QR_010010000000+a1P_000001010_2*QR_010010000001+a2P_000001000_1*QR_010010000002+a1P_000000020_1*QR_010010000010+a2P_000000010_2*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(P_000001020*QR_000020000000+a1P_000001010_2*QR_000020000001+a2P_000001000_1*QR_000020000002+a1P_000000020_1*QR_000020000010+a2P_000000010_2*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(P_000001020*QR_010000010000+a1P_000001010_2*QR_010000010001+a2P_000001000_1*QR_010000010002+a1P_000000020_1*QR_010000010010+a2P_000000010_2*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(P_000001020*QR_000010010000+a1P_000001010_2*QR_000010010001+a2P_000001000_1*QR_000010010002+a1P_000000020_1*QR_000010010010+a2P_000000010_2*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(P_000001020*QR_000000020000+a1P_000001010_2*QR_000000020001+a2P_000001000_1*QR_000000020002+a1P_000000020_1*QR_000000020010+a2P_000000010_2*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(P_000000021*QR_020000000000+P_000000121*QR_020000000001+P_000000221*QR_020000000002+aPin3*QR_020000000003);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(P_000000021*QR_010010000000+P_000000121*QR_010010000001+P_000000221*QR_010010000002+aPin3*QR_010010000003);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(P_000000021*QR_000020000000+P_000000121*QR_000020000001+P_000000221*QR_000020000002+aPin3*QR_000020000003);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(P_000000021*QR_010000010000+P_000000121*QR_010000010001+P_000000221*QR_010000010002+aPin3*QR_010000010003);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(P_000000021*QR_000010010000+P_000000121*QR_000010010001+P_000000221*QR_000010010002+aPin3*QR_000010010003);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(P_000000021*QR_000000020000+P_000000121*QR_000000020001+P_000000221*QR_000000020002+aPin3*QR_000000020003);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
}
__global__ void TSMJ_dpds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*18];
    for(int i=0;i<18;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
			double QR_020000000001=0;
			double QR_010010000001=0;
			double QR_000020000001=0;
			double QR_010000010001=0;
			double QR_000010010001=0;
			double QR_000000020001=0;
			double QR_020000000010=0;
			double QR_010010000010=0;
			double QR_000020000010=0;
			double QR_010000010010=0;
			double QR_000010010010=0;
			double QR_000000020010=0;
			double QR_020000000100=0;
			double QR_010010000100=0;
			double QR_000020000100=0;
			double QR_010000010100=0;
			double QR_000010010100=0;
			double QR_000000020100=0;
			double QR_020000000002=0;
			double QR_010010000002=0;
			double QR_000020000002=0;
			double QR_010000010002=0;
			double QR_000010010002=0;
			double QR_000000020002=0;
			double QR_020000000011=0;
			double QR_010010000011=0;
			double QR_000020000011=0;
			double QR_010000010011=0;
			double QR_000010010011=0;
			double QR_000000020011=0;
			double QR_020000000020=0;
			double QR_010010000020=0;
			double QR_000020000020=0;
			double QR_010000010020=0;
			double QR_000010010020=0;
			double QR_000000020020=0;
			double QR_020000000101=0;
			double QR_010010000101=0;
			double QR_000020000101=0;
			double QR_010000010101=0;
			double QR_000010010101=0;
			double QR_000000020101=0;
			double QR_020000000110=0;
			double QR_010010000110=0;
			double QR_000020000110=0;
			double QR_010000010110=0;
			double QR_000010010110=0;
			double QR_000000020110=0;
			double QR_020000000200=0;
			double QR_010010000200=0;
			double QR_000020000200=0;
			double QR_010000010200=0;
			double QR_000010010200=0;
			double QR_000000020200=0;
			double QR_020000000003=0;
			double QR_010010000003=0;
			double QR_000020000003=0;
			double QR_010000010003=0;
			double QR_000010010003=0;
			double QR_000000020003=0;
			double QR_020000000012=0;
			double QR_010010000012=0;
			double QR_000020000012=0;
			double QR_010000010012=0;
			double QR_000010010012=0;
			double QR_000000020012=0;
			double QR_020000000021=0;
			double QR_010010000021=0;
			double QR_000020000021=0;
			double QR_010000010021=0;
			double QR_000010010021=0;
			double QR_000000020021=0;
			double QR_020000000030=0;
			double QR_010010000030=0;
			double QR_000020000030=0;
			double QR_010000010030=0;
			double QR_000010010030=0;
			double QR_000000020030=0;
			double QR_020000000102=0;
			double QR_010010000102=0;
			double QR_000020000102=0;
			double QR_010000010102=0;
			double QR_000010010102=0;
			double QR_000000020102=0;
			double QR_020000000111=0;
			double QR_010010000111=0;
			double QR_000020000111=0;
			double QR_010000010111=0;
			double QR_000010010111=0;
			double QR_000000020111=0;
			double QR_020000000120=0;
			double QR_010010000120=0;
			double QR_000020000120=0;
			double QR_010000010120=0;
			double QR_000010010120=0;
			double QR_000000020120=0;
			double QR_020000000201=0;
			double QR_010010000201=0;
			double QR_000020000201=0;
			double QR_010000010201=0;
			double QR_000010010201=0;
			double QR_000000020201=0;
			double QR_020000000210=0;
			double QR_010010000210=0;
			double QR_000020000210=0;
			double QR_010000010210=0;
			double QR_000010010210=0;
			double QR_000000020210=0;
			double QR_020000000300=0;
			double QR_010010000300=0;
			double QR_000020000300=0;
			double QR_010000010300=0;
			double QR_000010010300=0;
			double QR_000000020300=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			QR_020000000001+=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			QR_010010000001+=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			QR_000020000001+=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			QR_010000010001+=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			QR_000010010001+=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			QR_000000020001+=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			QR_020000000010+=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			QR_010010000010+=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			QR_000020000010+=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			QR_010000010010+=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000010010010+=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			QR_000000020010+=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			QR_020000000100+=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			QR_010010000100+=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			QR_000020000100+=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			QR_010000010100+=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			QR_000010010100+=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000000020100+=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			QR_020000000002+=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			QR_010010000002+=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			QR_000020000002+=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			QR_010000010002+=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			QR_000010010002+=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			QR_000000020002+=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			QR_020000000011+=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			QR_010010000011+=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			QR_000020000011+=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			QR_010000010011+=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000010010011+=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			QR_000000020011+=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			QR_020000000020+=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			QR_010010000020+=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			QR_000020000020+=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			QR_010000010020+=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000010010020+=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			QR_000000020020+=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			QR_020000000101+=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			QR_010010000101+=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			QR_000020000101+=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			QR_010000010101+=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			QR_000010010101+=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000000020101+=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			QR_020000000110+=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			QR_010010000110+=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			QR_000020000110+=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			QR_010000010110+=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000010010110+=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000000020110+=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			QR_020000000200+=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			QR_010010000200+=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			QR_000020000200+=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			QR_010000010200+=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			QR_000010010200+=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000000020200+=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			QR_020000000003+=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			QR_010010000003+=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			QR_000020000003+=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			QR_010000010003+=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			QR_000010010003+=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			QR_000000020003+=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			QR_020000000012+=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			QR_010010000012+=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			QR_000020000012+=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			QR_010000010012+=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			QR_000010010012+=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			QR_000000020012+=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			QR_020000000021+=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			QR_010010000021+=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			QR_000020000021+=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			QR_010000010021+=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			QR_000010010021+=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			QR_000000020021+=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			QR_020000000030+=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			QR_010010000030+=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			QR_000020000030+=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			QR_010000010030+=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			QR_000010010030+=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			QR_000000020030+=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			QR_020000000102+=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			QR_010010000102+=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			QR_000020000102+=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			QR_010000010102+=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			QR_000010010102+=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			QR_000000020102+=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			QR_020000000111+=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			QR_010010000111+=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			QR_000020000111+=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			QR_010000010111+=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			QR_000010010111+=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			QR_000000020111+=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			QR_020000000120+=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			QR_010010000120+=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			QR_000020000120+=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			QR_010000010120+=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			QR_000010010120+=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			QR_000000020120+=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			QR_020000000201+=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			QR_010010000201+=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			QR_000020000201+=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			QR_010000010201+=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			QR_000010010201+=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			QR_000000020201+=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			QR_020000000210+=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			QR_010010000210+=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			QR_000020000210+=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			QR_010000010210+=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			QR_000010010210+=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			QR_000000020210+=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			QR_020000000300+=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			QR_010010000300+=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			QR_000020000300+=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			QR_010000010300+=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			QR_000010010300+=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			QR_000000020300+=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
			}
		double Pd_011[3];
		double Pd_111[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
			double P_021000000;
			double P_121000000;
			double P_221000000;
			double P_020001000;
			double P_020000001;
			double P_011010000;
			double P_111010000;
			double P_010011000;
			double P_010111000;
			double P_010010001;
			double P_001020000;
			double P_000021000;
			double P_000121000;
			double P_000221000;
			double P_000020001;
			double P_011000010;
			double P_111000010;
			double P_010001010;
			double P_010000011;
			double P_010000111;
			double P_001010010;
			double P_000011010;
			double P_000111010;
			double P_000010011;
			double P_000010111;
			double P_001000020;
			double P_000001020;
			double P_000000021;
			double P_000000121;
			double P_000000221;
			double a1P_020000000_1;
			double a1P_010001000_1;
			double a1P_010001000_2;
			double a2P_010000000_1;
			double a2P_010000000_2;
			double a2P_000001000_1;
			double a1P_010000001_1;
			double a1P_010000001_2;
			double a2P_000000001_1;
			double a1P_011000000_1;
			double a1P_111000000_1;
			double a2P_000010000_1;
			double a2P_000010000_2;
			double a1P_000011000_1;
			double a1P_000111000_1;
			double a1P_010010000_1;
			double a1P_000010001_1;
			double a1P_000010001_2;
			double a1P_001010000_1;
			double a1P_001010000_2;
			double a2P_001000000_1;
			double a1P_000020000_1;
			double a2P_000000010_1;
			double a2P_000000010_2;
			double a1P_010000010_1;
			double a1P_000001010_1;
			double a1P_000001010_2;
			double a1P_000000011_1;
			double a1P_000000111_1;
			double a1P_001000010_1;
			double a1P_001000010_2;
			double a1P_000010010_1;
			double a1P_000000020_1;
			P_021000000=Pd_021[0];
			P_121000000=Pd_121[0];
			P_221000000=Pd_221[0];
			P_020001000=Pd_020[0]*Pd_001[1];
			P_020000001=Pd_020[0]*Pd_001[2];
			P_011010000=Pd_011[0]*Pd_010[1];
			P_111010000=Pd_111[0]*Pd_010[1];
			P_010011000=Pd_010[0]*Pd_011[1];
			P_010111000=Pd_010[0]*Pd_111[1];
			P_010010001=Pd_010[0]*Pd_010[1]*Pd_001[2];
			P_001020000=Pd_001[0]*Pd_020[1];
			P_000021000=Pd_021[1];
			P_000121000=Pd_121[1];
			P_000221000=Pd_221[1];
			P_000020001=Pd_020[1]*Pd_001[2];
			P_011000010=Pd_011[0]*Pd_010[2];
			P_111000010=Pd_111[0]*Pd_010[2];
			P_010001010=Pd_010[0]*Pd_001[1]*Pd_010[2];
			P_010000011=Pd_010[0]*Pd_011[2];
			P_010000111=Pd_010[0]*Pd_111[2];
			P_001010010=Pd_001[0]*Pd_010[1]*Pd_010[2];
			P_000011010=Pd_011[1]*Pd_010[2];
			P_000111010=Pd_111[1]*Pd_010[2];
			P_000010011=Pd_010[1]*Pd_011[2];
			P_000010111=Pd_010[1]*Pd_111[2];
			P_001000020=Pd_001[0]*Pd_020[2];
			P_000001020=Pd_001[1]*Pd_020[2];
			P_000000021=Pd_021[2];
			P_000000121=Pd_121[2];
			P_000000221=Pd_221[2];
			a1P_020000000_1=Pd_020[0];
			a1P_010001000_1=Pd_010[0]*Pd_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=Pd_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=Pd_001[1];
			a1P_010000001_1=Pd_010[0]*Pd_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=Pd_001[2];
			a1P_011000000_1=Pd_011[0];
			a1P_111000000_1=Pd_111[0];
			a2P_000010000_1=Pd_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=Pd_011[1];
			a1P_000111000_1=Pd_111[1];
			a1P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010001_1=Pd_010[1]*Pd_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=Pd_001[0]*Pd_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=Pd_001[0];
			a1P_000020000_1=Pd_020[1];
			a2P_000000010_1=Pd_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000001010_1=Pd_001[1]*Pd_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=Pd_011[2];
			a1P_000000111_1=Pd_111[2];
			a1P_001000010_1=Pd_001[0]*Pd_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_000000020_1=Pd_020[2];
			ans_temp[ans_id*18+0]+=Pmtrx[0]*(P_021000000*QR_020000000000+P_121000000*QR_020000000100+P_221000000*QR_020000000200+aPin3*QR_020000000300);
			ans_temp[ans_id*18+0]+=Pmtrx[1]*(P_021000000*QR_010010000000+P_121000000*QR_010010000100+P_221000000*QR_010010000200+aPin3*QR_010010000300);
			ans_temp[ans_id*18+0]+=Pmtrx[2]*(P_021000000*QR_000020000000+P_121000000*QR_000020000100+P_221000000*QR_000020000200+aPin3*QR_000020000300);
			ans_temp[ans_id*18+0]+=Pmtrx[3]*(P_021000000*QR_010000010000+P_121000000*QR_010000010100+P_221000000*QR_010000010200+aPin3*QR_010000010300);
			ans_temp[ans_id*18+0]+=Pmtrx[4]*(P_021000000*QR_000010010000+P_121000000*QR_000010010100+P_221000000*QR_000010010200+aPin3*QR_000010010300);
			ans_temp[ans_id*18+0]+=Pmtrx[5]*(P_021000000*QR_000000020000+P_121000000*QR_000000020100+P_221000000*QR_000000020200+aPin3*QR_000000020300);
			ans_temp[ans_id*18+1]+=Pmtrx[0]*(P_020001000*QR_020000000000+a1P_020000000_1*QR_020000000010+a1P_010001000_2*QR_020000000100+a2P_010000000_2*QR_020000000110+a2P_000001000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+1]+=Pmtrx[1]*(P_020001000*QR_010010000000+a1P_020000000_1*QR_010010000010+a1P_010001000_2*QR_010010000100+a2P_010000000_2*QR_010010000110+a2P_000001000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+1]+=Pmtrx[2]*(P_020001000*QR_000020000000+a1P_020000000_1*QR_000020000010+a1P_010001000_2*QR_000020000100+a2P_010000000_2*QR_000020000110+a2P_000001000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+1]+=Pmtrx[3]*(P_020001000*QR_010000010000+a1P_020000000_1*QR_010000010010+a1P_010001000_2*QR_010000010100+a2P_010000000_2*QR_010000010110+a2P_000001000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+1]+=Pmtrx[4]*(P_020001000*QR_000010010000+a1P_020000000_1*QR_000010010010+a1P_010001000_2*QR_000010010100+a2P_010000000_2*QR_000010010110+a2P_000001000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+1]+=Pmtrx[5]*(P_020001000*QR_000000020000+a1P_020000000_1*QR_000000020010+a1P_010001000_2*QR_000000020100+a2P_010000000_2*QR_000000020110+a2P_000001000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+2]+=Pmtrx[0]*(P_020000001*QR_020000000000+a1P_020000000_1*QR_020000000001+a1P_010000001_2*QR_020000000100+a2P_010000000_2*QR_020000000101+a2P_000000001_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+2]+=Pmtrx[1]*(P_020000001*QR_010010000000+a1P_020000000_1*QR_010010000001+a1P_010000001_2*QR_010010000100+a2P_010000000_2*QR_010010000101+a2P_000000001_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+2]+=Pmtrx[2]*(P_020000001*QR_000020000000+a1P_020000000_1*QR_000020000001+a1P_010000001_2*QR_000020000100+a2P_010000000_2*QR_000020000101+a2P_000000001_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+2]+=Pmtrx[3]*(P_020000001*QR_010000010000+a1P_020000000_1*QR_010000010001+a1P_010000001_2*QR_010000010100+a2P_010000000_2*QR_010000010101+a2P_000000001_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+2]+=Pmtrx[4]*(P_020000001*QR_000010010000+a1P_020000000_1*QR_000010010001+a1P_010000001_2*QR_000010010100+a2P_010000000_2*QR_000010010101+a2P_000000001_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+2]+=Pmtrx[5]*(P_020000001*QR_000000020000+a1P_020000000_1*QR_000000020001+a1P_010000001_2*QR_000000020100+a2P_010000000_2*QR_000000020101+a2P_000000001_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+3]+=Pmtrx[0]*(P_011010000*QR_020000000000+a1P_011000000_1*QR_020000000010+P_111010000*QR_020000000100+a1P_111000000_1*QR_020000000110+a2P_000010000_1*QR_020000000200+aPin3*QR_020000000210);
			ans_temp[ans_id*18+3]+=Pmtrx[1]*(P_011010000*QR_010010000000+a1P_011000000_1*QR_010010000010+P_111010000*QR_010010000100+a1P_111000000_1*QR_010010000110+a2P_000010000_1*QR_010010000200+aPin3*QR_010010000210);
			ans_temp[ans_id*18+3]+=Pmtrx[2]*(P_011010000*QR_000020000000+a1P_011000000_1*QR_000020000010+P_111010000*QR_000020000100+a1P_111000000_1*QR_000020000110+a2P_000010000_1*QR_000020000200+aPin3*QR_000020000210);
			ans_temp[ans_id*18+3]+=Pmtrx[3]*(P_011010000*QR_010000010000+a1P_011000000_1*QR_010000010010+P_111010000*QR_010000010100+a1P_111000000_1*QR_010000010110+a2P_000010000_1*QR_010000010200+aPin3*QR_010000010210);
			ans_temp[ans_id*18+3]+=Pmtrx[4]*(P_011010000*QR_000010010000+a1P_011000000_1*QR_000010010010+P_111010000*QR_000010010100+a1P_111000000_1*QR_000010010110+a2P_000010000_1*QR_000010010200+aPin3*QR_000010010210);
			ans_temp[ans_id*18+3]+=Pmtrx[5]*(P_011010000*QR_000000020000+a1P_011000000_1*QR_000000020010+P_111010000*QR_000000020100+a1P_111000000_1*QR_000000020110+a2P_000010000_1*QR_000000020200+aPin3*QR_000000020210);
			ans_temp[ans_id*18+4]+=Pmtrx[0]*(P_010011000*QR_020000000000+P_010111000*QR_020000000010+a2P_010000000_1*QR_020000000020+a1P_000011000_1*QR_020000000100+a1P_000111000_1*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+4]+=Pmtrx[1]*(P_010011000*QR_010010000000+P_010111000*QR_010010000010+a2P_010000000_1*QR_010010000020+a1P_000011000_1*QR_010010000100+a1P_000111000_1*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+4]+=Pmtrx[2]*(P_010011000*QR_000020000000+P_010111000*QR_000020000010+a2P_010000000_1*QR_000020000020+a1P_000011000_1*QR_000020000100+a1P_000111000_1*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+4]+=Pmtrx[3]*(P_010011000*QR_010000010000+P_010111000*QR_010000010010+a2P_010000000_1*QR_010000010020+a1P_000011000_1*QR_010000010100+a1P_000111000_1*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+4]+=Pmtrx[4]*(P_010011000*QR_000010010000+P_010111000*QR_000010010010+a2P_010000000_1*QR_000010010020+a1P_000011000_1*QR_000010010100+a1P_000111000_1*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+4]+=Pmtrx[5]*(P_010011000*QR_000000020000+P_010111000*QR_000000020010+a2P_010000000_1*QR_000000020020+a1P_000011000_1*QR_000000020100+a1P_000111000_1*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+5]+=Pmtrx[0]*(P_010010001*QR_020000000000+a1P_010010000_1*QR_020000000001+a1P_010000001_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000010001_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000001_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+5]+=Pmtrx[1]*(P_010010001*QR_010010000000+a1P_010010000_1*QR_010010000001+a1P_010000001_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000010001_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000001_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+5]+=Pmtrx[2]*(P_010010001*QR_000020000000+a1P_010010000_1*QR_000020000001+a1P_010000001_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000010001_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000001_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+5]+=Pmtrx[3]*(P_010010001*QR_010000010000+a1P_010010000_1*QR_010000010001+a1P_010000001_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000010001_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000001_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+5]+=Pmtrx[4]*(P_010010001*QR_000010010000+a1P_010010000_1*QR_000010010001+a1P_010000001_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000010001_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000001_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+5]+=Pmtrx[5]*(P_010010001*QR_000000020000+a1P_010010000_1*QR_000000020001+a1P_010000001_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000010001_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000001_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+6]+=Pmtrx[0]*(P_001020000*QR_020000000000+a1P_001010000_2*QR_020000000010+a2P_001000000_1*QR_020000000020+a1P_000020000_1*QR_020000000100+a2P_000010000_2*QR_020000000110+aPin3*QR_020000000120);
			ans_temp[ans_id*18+6]+=Pmtrx[1]*(P_001020000*QR_010010000000+a1P_001010000_2*QR_010010000010+a2P_001000000_1*QR_010010000020+a1P_000020000_1*QR_010010000100+a2P_000010000_2*QR_010010000110+aPin3*QR_010010000120);
			ans_temp[ans_id*18+6]+=Pmtrx[2]*(P_001020000*QR_000020000000+a1P_001010000_2*QR_000020000010+a2P_001000000_1*QR_000020000020+a1P_000020000_1*QR_000020000100+a2P_000010000_2*QR_000020000110+aPin3*QR_000020000120);
			ans_temp[ans_id*18+6]+=Pmtrx[3]*(P_001020000*QR_010000010000+a1P_001010000_2*QR_010000010010+a2P_001000000_1*QR_010000010020+a1P_000020000_1*QR_010000010100+a2P_000010000_2*QR_010000010110+aPin3*QR_010000010120);
			ans_temp[ans_id*18+6]+=Pmtrx[4]*(P_001020000*QR_000010010000+a1P_001010000_2*QR_000010010010+a2P_001000000_1*QR_000010010020+a1P_000020000_1*QR_000010010100+a2P_000010000_2*QR_000010010110+aPin3*QR_000010010120);
			ans_temp[ans_id*18+6]+=Pmtrx[5]*(P_001020000*QR_000000020000+a1P_001010000_2*QR_000000020010+a2P_001000000_1*QR_000000020020+a1P_000020000_1*QR_000000020100+a2P_000010000_2*QR_000000020110+aPin3*QR_000000020120);
			ans_temp[ans_id*18+7]+=Pmtrx[0]*(P_000021000*QR_020000000000+P_000121000*QR_020000000010+P_000221000*QR_020000000020+aPin3*QR_020000000030);
			ans_temp[ans_id*18+7]+=Pmtrx[1]*(P_000021000*QR_010010000000+P_000121000*QR_010010000010+P_000221000*QR_010010000020+aPin3*QR_010010000030);
			ans_temp[ans_id*18+7]+=Pmtrx[2]*(P_000021000*QR_000020000000+P_000121000*QR_000020000010+P_000221000*QR_000020000020+aPin3*QR_000020000030);
			ans_temp[ans_id*18+7]+=Pmtrx[3]*(P_000021000*QR_010000010000+P_000121000*QR_010000010010+P_000221000*QR_010000010020+aPin3*QR_010000010030);
			ans_temp[ans_id*18+7]+=Pmtrx[4]*(P_000021000*QR_000010010000+P_000121000*QR_000010010010+P_000221000*QR_000010010020+aPin3*QR_000010010030);
			ans_temp[ans_id*18+7]+=Pmtrx[5]*(P_000021000*QR_000000020000+P_000121000*QR_000000020010+P_000221000*QR_000000020020+aPin3*QR_000000020030);
			ans_temp[ans_id*18+8]+=Pmtrx[0]*(P_000020001*QR_020000000000+a1P_000020000_1*QR_020000000001+a1P_000010001_2*QR_020000000010+a2P_000010000_2*QR_020000000011+a2P_000000001_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+8]+=Pmtrx[1]*(P_000020001*QR_010010000000+a1P_000020000_1*QR_010010000001+a1P_000010001_2*QR_010010000010+a2P_000010000_2*QR_010010000011+a2P_000000001_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+8]+=Pmtrx[2]*(P_000020001*QR_000020000000+a1P_000020000_1*QR_000020000001+a1P_000010001_2*QR_000020000010+a2P_000010000_2*QR_000020000011+a2P_000000001_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+8]+=Pmtrx[3]*(P_000020001*QR_010000010000+a1P_000020000_1*QR_010000010001+a1P_000010001_2*QR_010000010010+a2P_000010000_2*QR_010000010011+a2P_000000001_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+8]+=Pmtrx[4]*(P_000020001*QR_000010010000+a1P_000020000_1*QR_000010010001+a1P_000010001_2*QR_000010010010+a2P_000010000_2*QR_000010010011+a2P_000000001_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+8]+=Pmtrx[5]*(P_000020001*QR_000000020000+a1P_000020000_1*QR_000000020001+a1P_000010001_2*QR_000000020010+a2P_000010000_2*QR_000000020011+a2P_000000001_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+9]+=Pmtrx[0]*(P_011000010*QR_020000000000+a1P_011000000_1*QR_020000000001+P_111000010*QR_020000000100+a1P_111000000_1*QR_020000000101+a2P_000000010_1*QR_020000000200+aPin3*QR_020000000201);
			ans_temp[ans_id*18+9]+=Pmtrx[1]*(P_011000010*QR_010010000000+a1P_011000000_1*QR_010010000001+P_111000010*QR_010010000100+a1P_111000000_1*QR_010010000101+a2P_000000010_1*QR_010010000200+aPin3*QR_010010000201);
			ans_temp[ans_id*18+9]+=Pmtrx[2]*(P_011000010*QR_000020000000+a1P_011000000_1*QR_000020000001+P_111000010*QR_000020000100+a1P_111000000_1*QR_000020000101+a2P_000000010_1*QR_000020000200+aPin3*QR_000020000201);
			ans_temp[ans_id*18+9]+=Pmtrx[3]*(P_011000010*QR_010000010000+a1P_011000000_1*QR_010000010001+P_111000010*QR_010000010100+a1P_111000000_1*QR_010000010101+a2P_000000010_1*QR_010000010200+aPin3*QR_010000010201);
			ans_temp[ans_id*18+9]+=Pmtrx[4]*(P_011000010*QR_000010010000+a1P_011000000_1*QR_000010010001+P_111000010*QR_000010010100+a1P_111000000_1*QR_000010010101+a2P_000000010_1*QR_000010010200+aPin3*QR_000010010201);
			ans_temp[ans_id*18+9]+=Pmtrx[5]*(P_011000010*QR_000000020000+a1P_011000000_1*QR_000000020001+P_111000010*QR_000000020100+a1P_111000000_1*QR_000000020101+a2P_000000010_1*QR_000000020200+aPin3*QR_000000020201);
			ans_temp[ans_id*18+10]+=Pmtrx[0]*(P_010001010*QR_020000000000+a1P_010001000_1*QR_020000000001+a1P_010000010_1*QR_020000000010+a2P_010000000_1*QR_020000000011+a1P_000001010_1*QR_020000000100+a2P_000001000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+10]+=Pmtrx[1]*(P_010001010*QR_010010000000+a1P_010001000_1*QR_010010000001+a1P_010000010_1*QR_010010000010+a2P_010000000_1*QR_010010000011+a1P_000001010_1*QR_010010000100+a2P_000001000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+10]+=Pmtrx[2]*(P_010001010*QR_000020000000+a1P_010001000_1*QR_000020000001+a1P_010000010_1*QR_000020000010+a2P_010000000_1*QR_000020000011+a1P_000001010_1*QR_000020000100+a2P_000001000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+10]+=Pmtrx[3]*(P_010001010*QR_010000010000+a1P_010001000_1*QR_010000010001+a1P_010000010_1*QR_010000010010+a2P_010000000_1*QR_010000010011+a1P_000001010_1*QR_010000010100+a2P_000001000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+10]+=Pmtrx[4]*(P_010001010*QR_000010010000+a1P_010001000_1*QR_000010010001+a1P_010000010_1*QR_000010010010+a2P_010000000_1*QR_000010010011+a1P_000001010_1*QR_000010010100+a2P_000001000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+10]+=Pmtrx[5]*(P_010001010*QR_000000020000+a1P_010001000_1*QR_000000020001+a1P_010000010_1*QR_000000020010+a2P_010000000_1*QR_000000020011+a1P_000001010_1*QR_000000020100+a2P_000001000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+11]+=Pmtrx[0]*(P_010000011*QR_020000000000+P_010000111*QR_020000000001+a2P_010000000_1*QR_020000000002+a1P_000000011_1*QR_020000000100+a1P_000000111_1*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+11]+=Pmtrx[1]*(P_010000011*QR_010010000000+P_010000111*QR_010010000001+a2P_010000000_1*QR_010010000002+a1P_000000011_1*QR_010010000100+a1P_000000111_1*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+11]+=Pmtrx[2]*(P_010000011*QR_000020000000+P_010000111*QR_000020000001+a2P_010000000_1*QR_000020000002+a1P_000000011_1*QR_000020000100+a1P_000000111_1*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+11]+=Pmtrx[3]*(P_010000011*QR_010000010000+P_010000111*QR_010000010001+a2P_010000000_1*QR_010000010002+a1P_000000011_1*QR_010000010100+a1P_000000111_1*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+11]+=Pmtrx[4]*(P_010000011*QR_000010010000+P_010000111*QR_000010010001+a2P_010000000_1*QR_000010010002+a1P_000000011_1*QR_000010010100+a1P_000000111_1*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+11]+=Pmtrx[5]*(P_010000011*QR_000000020000+P_010000111*QR_000000020001+a2P_010000000_1*QR_000000020002+a1P_000000011_1*QR_000000020100+a1P_000000111_1*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+12]+=Pmtrx[0]*(P_001010010*QR_020000000000+a1P_001010000_1*QR_020000000001+a1P_001000010_1*QR_020000000010+a2P_001000000_1*QR_020000000011+a1P_000010010_1*QR_020000000100+a2P_000010000_1*QR_020000000101+a2P_000000010_1*QR_020000000110+aPin3*QR_020000000111);
			ans_temp[ans_id*18+12]+=Pmtrx[1]*(P_001010010*QR_010010000000+a1P_001010000_1*QR_010010000001+a1P_001000010_1*QR_010010000010+a2P_001000000_1*QR_010010000011+a1P_000010010_1*QR_010010000100+a2P_000010000_1*QR_010010000101+a2P_000000010_1*QR_010010000110+aPin3*QR_010010000111);
			ans_temp[ans_id*18+12]+=Pmtrx[2]*(P_001010010*QR_000020000000+a1P_001010000_1*QR_000020000001+a1P_001000010_1*QR_000020000010+a2P_001000000_1*QR_000020000011+a1P_000010010_1*QR_000020000100+a2P_000010000_1*QR_000020000101+a2P_000000010_1*QR_000020000110+aPin3*QR_000020000111);
			ans_temp[ans_id*18+12]+=Pmtrx[3]*(P_001010010*QR_010000010000+a1P_001010000_1*QR_010000010001+a1P_001000010_1*QR_010000010010+a2P_001000000_1*QR_010000010011+a1P_000010010_1*QR_010000010100+a2P_000010000_1*QR_010000010101+a2P_000000010_1*QR_010000010110+aPin3*QR_010000010111);
			ans_temp[ans_id*18+12]+=Pmtrx[4]*(P_001010010*QR_000010010000+a1P_001010000_1*QR_000010010001+a1P_001000010_1*QR_000010010010+a2P_001000000_1*QR_000010010011+a1P_000010010_1*QR_000010010100+a2P_000010000_1*QR_000010010101+a2P_000000010_1*QR_000010010110+aPin3*QR_000010010111);
			ans_temp[ans_id*18+12]+=Pmtrx[5]*(P_001010010*QR_000000020000+a1P_001010000_1*QR_000000020001+a1P_001000010_1*QR_000000020010+a2P_001000000_1*QR_000000020011+a1P_000010010_1*QR_000000020100+a2P_000010000_1*QR_000000020101+a2P_000000010_1*QR_000000020110+aPin3*QR_000000020111);
			ans_temp[ans_id*18+13]+=Pmtrx[0]*(P_000011010*QR_020000000000+a1P_000011000_1*QR_020000000001+P_000111010*QR_020000000010+a1P_000111000_1*QR_020000000011+a2P_000000010_1*QR_020000000020+aPin3*QR_020000000021);
			ans_temp[ans_id*18+13]+=Pmtrx[1]*(P_000011010*QR_010010000000+a1P_000011000_1*QR_010010000001+P_000111010*QR_010010000010+a1P_000111000_1*QR_010010000011+a2P_000000010_1*QR_010010000020+aPin3*QR_010010000021);
			ans_temp[ans_id*18+13]+=Pmtrx[2]*(P_000011010*QR_000020000000+a1P_000011000_1*QR_000020000001+P_000111010*QR_000020000010+a1P_000111000_1*QR_000020000011+a2P_000000010_1*QR_000020000020+aPin3*QR_000020000021);
			ans_temp[ans_id*18+13]+=Pmtrx[3]*(P_000011010*QR_010000010000+a1P_000011000_1*QR_010000010001+P_000111010*QR_010000010010+a1P_000111000_1*QR_010000010011+a2P_000000010_1*QR_010000010020+aPin3*QR_010000010021);
			ans_temp[ans_id*18+13]+=Pmtrx[4]*(P_000011010*QR_000010010000+a1P_000011000_1*QR_000010010001+P_000111010*QR_000010010010+a1P_000111000_1*QR_000010010011+a2P_000000010_1*QR_000010010020+aPin3*QR_000010010021);
			ans_temp[ans_id*18+13]+=Pmtrx[5]*(P_000011010*QR_000000020000+a1P_000011000_1*QR_000000020001+P_000111010*QR_000000020010+a1P_000111000_1*QR_000000020011+a2P_000000010_1*QR_000000020020+aPin3*QR_000000020021);
			ans_temp[ans_id*18+14]+=Pmtrx[0]*(P_000010011*QR_020000000000+P_000010111*QR_020000000001+a2P_000010000_1*QR_020000000002+a1P_000000011_1*QR_020000000010+a1P_000000111_1*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+14]+=Pmtrx[1]*(P_000010011*QR_010010000000+P_000010111*QR_010010000001+a2P_000010000_1*QR_010010000002+a1P_000000011_1*QR_010010000010+a1P_000000111_1*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+14]+=Pmtrx[2]*(P_000010011*QR_000020000000+P_000010111*QR_000020000001+a2P_000010000_1*QR_000020000002+a1P_000000011_1*QR_000020000010+a1P_000000111_1*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+14]+=Pmtrx[3]*(P_000010011*QR_010000010000+P_000010111*QR_010000010001+a2P_000010000_1*QR_010000010002+a1P_000000011_1*QR_010000010010+a1P_000000111_1*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+14]+=Pmtrx[4]*(P_000010011*QR_000010010000+P_000010111*QR_000010010001+a2P_000010000_1*QR_000010010002+a1P_000000011_1*QR_000010010010+a1P_000000111_1*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+14]+=Pmtrx[5]*(P_000010011*QR_000000020000+P_000010111*QR_000000020001+a2P_000010000_1*QR_000000020002+a1P_000000011_1*QR_000000020010+a1P_000000111_1*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+15]+=Pmtrx[0]*(P_001000020*QR_020000000000+a1P_001000010_2*QR_020000000001+a2P_001000000_1*QR_020000000002+a1P_000000020_1*QR_020000000100+a2P_000000010_2*QR_020000000101+aPin3*QR_020000000102);
			ans_temp[ans_id*18+15]+=Pmtrx[1]*(P_001000020*QR_010010000000+a1P_001000010_2*QR_010010000001+a2P_001000000_1*QR_010010000002+a1P_000000020_1*QR_010010000100+a2P_000000010_2*QR_010010000101+aPin3*QR_010010000102);
			ans_temp[ans_id*18+15]+=Pmtrx[2]*(P_001000020*QR_000020000000+a1P_001000010_2*QR_000020000001+a2P_001000000_1*QR_000020000002+a1P_000000020_1*QR_000020000100+a2P_000000010_2*QR_000020000101+aPin3*QR_000020000102);
			ans_temp[ans_id*18+15]+=Pmtrx[3]*(P_001000020*QR_010000010000+a1P_001000010_2*QR_010000010001+a2P_001000000_1*QR_010000010002+a1P_000000020_1*QR_010000010100+a2P_000000010_2*QR_010000010101+aPin3*QR_010000010102);
			ans_temp[ans_id*18+15]+=Pmtrx[4]*(P_001000020*QR_000010010000+a1P_001000010_2*QR_000010010001+a2P_001000000_1*QR_000010010002+a1P_000000020_1*QR_000010010100+a2P_000000010_2*QR_000010010101+aPin3*QR_000010010102);
			ans_temp[ans_id*18+15]+=Pmtrx[5]*(P_001000020*QR_000000020000+a1P_001000010_2*QR_000000020001+a2P_001000000_1*QR_000000020002+a1P_000000020_1*QR_000000020100+a2P_000000010_2*QR_000000020101+aPin3*QR_000000020102);
			ans_temp[ans_id*18+16]+=Pmtrx[0]*(P_000001020*QR_020000000000+a1P_000001010_2*QR_020000000001+a2P_000001000_1*QR_020000000002+a1P_000000020_1*QR_020000000010+a2P_000000010_2*QR_020000000011+aPin3*QR_020000000012);
			ans_temp[ans_id*18+16]+=Pmtrx[1]*(P_000001020*QR_010010000000+a1P_000001010_2*QR_010010000001+a2P_000001000_1*QR_010010000002+a1P_000000020_1*QR_010010000010+a2P_000000010_2*QR_010010000011+aPin3*QR_010010000012);
			ans_temp[ans_id*18+16]+=Pmtrx[2]*(P_000001020*QR_000020000000+a1P_000001010_2*QR_000020000001+a2P_000001000_1*QR_000020000002+a1P_000000020_1*QR_000020000010+a2P_000000010_2*QR_000020000011+aPin3*QR_000020000012);
			ans_temp[ans_id*18+16]+=Pmtrx[3]*(P_000001020*QR_010000010000+a1P_000001010_2*QR_010000010001+a2P_000001000_1*QR_010000010002+a1P_000000020_1*QR_010000010010+a2P_000000010_2*QR_010000010011+aPin3*QR_010000010012);
			ans_temp[ans_id*18+16]+=Pmtrx[4]*(P_000001020*QR_000010010000+a1P_000001010_2*QR_000010010001+a2P_000001000_1*QR_000010010002+a1P_000000020_1*QR_000010010010+a2P_000000010_2*QR_000010010011+aPin3*QR_000010010012);
			ans_temp[ans_id*18+16]+=Pmtrx[5]*(P_000001020*QR_000000020000+a1P_000001010_2*QR_000000020001+a2P_000001000_1*QR_000000020002+a1P_000000020_1*QR_000000020010+a2P_000000010_2*QR_000000020011+aPin3*QR_000000020012);
			ans_temp[ans_id*18+17]+=Pmtrx[0]*(P_000000021*QR_020000000000+P_000000121*QR_020000000001+P_000000221*QR_020000000002+aPin3*QR_020000000003);
			ans_temp[ans_id*18+17]+=Pmtrx[1]*(P_000000021*QR_010010000000+P_000000121*QR_010010000001+P_000000221*QR_010010000002+aPin3*QR_010010000003);
			ans_temp[ans_id*18+17]+=Pmtrx[2]*(P_000000021*QR_000020000000+P_000000121*QR_000020000001+P_000000221*QR_000020000002+aPin3*QR_000020000003);
			ans_temp[ans_id*18+17]+=Pmtrx[3]*(P_000000021*QR_010000010000+P_000000121*QR_010000010001+P_000000221*QR_010000010002+aPin3*QR_010000010003);
			ans_temp[ans_id*18+17]+=Pmtrx[4]*(P_000000021*QR_000010010000+P_000000121*QR_000010010001+P_000000221*QR_000010010002+aPin3*QR_000010010003);
			ans_temp[ans_id*18+17]+=Pmtrx[5]*(P_000000021*QR_000000020000+P_000000121*QR_000000020001+P_000000221*QR_000000020002+aPin3*QR_000000020003);
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*18+ians]=ans_temp[(tId_x)*18+ians];
            }
        }
	}
}
__global__ void TSMJ_ddds_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double aPin4=aPin1*aPin3;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[7];
                Ft_taylor(6,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
                R_000[3]*=-8*alphaT*alphaT*alphaT*lmd;
                R_000[4]*=16*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[5]*=-32*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
                R_000[6]*=64*alphaT*alphaT*alphaT*alphaT*alphaT*alphaT*lmd;
				double aQin1=1/(2*Eta);
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
			double QR_020000000004=Q_020000000*R_004[0]-a1Q_010000000_2*R_104[0]+aQin2*R_204[0];
			double QR_010010000004=Q_010010000*R_004[0]-a1Q_010000000_1*R_014[0]-a1Q_000010000_1*R_104[0]+aQin2*R_114[0];
			double QR_000020000004=Q_000020000*R_004[0]-a1Q_000010000_2*R_014[0]+aQin2*R_024[0];
			double QR_010000010004=Q_010000010*R_004[0]-a1Q_010000000_1*R_005[0]-a1Q_000000010_1*R_104[0]+aQin2*R_105[0];
			double QR_000010010004=Q_000010010*R_004[0]-a1Q_000010000_1*R_005[0]-a1Q_000000010_1*R_014[0]+aQin2*R_015[0];
			double QR_000000020004=Q_000000020*R_004[0]-a1Q_000000010_2*R_005[0]+aQin2*R_006[0];
			double QR_020000000013=Q_020000000*R_013[0]-a1Q_010000000_2*R_113[0]+aQin2*R_213[0];
			double QR_010010000013=Q_010010000*R_013[0]-a1Q_010000000_1*R_023[0]-a1Q_000010000_1*R_113[0]+aQin2*R_123[0];
			double QR_000020000013=Q_000020000*R_013[0]-a1Q_000010000_2*R_023[0]+aQin2*R_033[0];
			double QR_010000010013=Q_010000010*R_013[0]-a1Q_010000000_1*R_014[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000010010013=Q_000010010*R_013[0]-a1Q_000010000_1*R_014[0]-a1Q_000000010_1*R_023[0]+aQin2*R_024[0];
			double QR_000000020013=Q_000000020*R_013[0]-a1Q_000000010_2*R_014[0]+aQin2*R_015[0];
			double QR_020000000022=Q_020000000*R_022[0]-a1Q_010000000_2*R_122[0]+aQin2*R_222[0];
			double QR_010010000022=Q_010010000*R_022[0]-a1Q_010000000_1*R_032[0]-a1Q_000010000_1*R_122[0]+aQin2*R_132[0];
			double QR_000020000022=Q_000020000*R_022[0]-a1Q_000010000_2*R_032[0]+aQin2*R_042[0];
			double QR_010000010022=Q_010000010*R_022[0]-a1Q_010000000_1*R_023[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000010010022=Q_000010010*R_022[0]-a1Q_000010000_1*R_023[0]-a1Q_000000010_1*R_032[0]+aQin2*R_033[0];
			double QR_000000020022=Q_000000020*R_022[0]-a1Q_000000010_2*R_023[0]+aQin2*R_024[0];
			double QR_020000000031=Q_020000000*R_031[0]-a1Q_010000000_2*R_131[0]+aQin2*R_231[0];
			double QR_010010000031=Q_010010000*R_031[0]-a1Q_010000000_1*R_041[0]-a1Q_000010000_1*R_131[0]+aQin2*R_141[0];
			double QR_000020000031=Q_000020000*R_031[0]-a1Q_000010000_2*R_041[0]+aQin2*R_051[0];
			double QR_010000010031=Q_010000010*R_031[0]-a1Q_010000000_1*R_032[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000010010031=Q_000010010*R_031[0]-a1Q_000010000_1*R_032[0]-a1Q_000000010_1*R_041[0]+aQin2*R_042[0];
			double QR_000000020031=Q_000000020*R_031[0]-a1Q_000000010_2*R_032[0]+aQin2*R_033[0];
			double QR_020000000040=Q_020000000*R_040[0]-a1Q_010000000_2*R_140[0]+aQin2*R_240[0];
			double QR_010010000040=Q_010010000*R_040[0]-a1Q_010000000_1*R_050[0]-a1Q_000010000_1*R_140[0]+aQin2*R_150[0];
			double QR_000020000040=Q_000020000*R_040[0]-a1Q_000010000_2*R_050[0]+aQin2*R_060[0];
			double QR_010000010040=Q_010000010*R_040[0]-a1Q_010000000_1*R_041[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000010010040=Q_000010010*R_040[0]-a1Q_000010000_1*R_041[0]-a1Q_000000010_1*R_050[0]+aQin2*R_051[0];
			double QR_000000020040=Q_000000020*R_040[0]-a1Q_000000010_2*R_041[0]+aQin2*R_042[0];
			double QR_020000000103=Q_020000000*R_103[0]-a1Q_010000000_2*R_203[0]+aQin2*R_303[0];
			double QR_010010000103=Q_010010000*R_103[0]-a1Q_010000000_1*R_113[0]-a1Q_000010000_1*R_203[0]+aQin2*R_213[0];
			double QR_000020000103=Q_000020000*R_103[0]-a1Q_000010000_2*R_113[0]+aQin2*R_123[0];
			double QR_010000010103=Q_010000010*R_103[0]-a1Q_010000000_1*R_104[0]-a1Q_000000010_1*R_203[0]+aQin2*R_204[0];
			double QR_000010010103=Q_000010010*R_103[0]-a1Q_000010000_1*R_104[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000000020103=Q_000000020*R_103[0]-a1Q_000000010_2*R_104[0]+aQin2*R_105[0];
			double QR_020000000112=Q_020000000*R_112[0]-a1Q_010000000_2*R_212[0]+aQin2*R_312[0];
			double QR_010010000112=Q_010010000*R_112[0]-a1Q_010000000_1*R_122[0]-a1Q_000010000_1*R_212[0]+aQin2*R_222[0];
			double QR_000020000112=Q_000020000*R_112[0]-a1Q_000010000_2*R_122[0]+aQin2*R_132[0];
			double QR_010000010112=Q_010000010*R_112[0]-a1Q_010000000_1*R_113[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000010010112=Q_000010010*R_112[0]-a1Q_000010000_1*R_113[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000000020112=Q_000000020*R_112[0]-a1Q_000000010_2*R_113[0]+aQin2*R_114[0];
			double QR_020000000121=Q_020000000*R_121[0]-a1Q_010000000_2*R_221[0]+aQin2*R_321[0];
			double QR_010010000121=Q_010010000*R_121[0]-a1Q_010000000_1*R_131[0]-a1Q_000010000_1*R_221[0]+aQin2*R_231[0];
			double QR_000020000121=Q_000020000*R_121[0]-a1Q_000010000_2*R_131[0]+aQin2*R_141[0];
			double QR_010000010121=Q_010000010*R_121[0]-a1Q_010000000_1*R_122[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000010010121=Q_000010010*R_121[0]-a1Q_000010000_1*R_122[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000000020121=Q_000000020*R_121[0]-a1Q_000000010_2*R_122[0]+aQin2*R_123[0];
			double QR_020000000130=Q_020000000*R_130[0]-a1Q_010000000_2*R_230[0]+aQin2*R_330[0];
			double QR_010010000130=Q_010010000*R_130[0]-a1Q_010000000_1*R_140[0]-a1Q_000010000_1*R_230[0]+aQin2*R_240[0];
			double QR_000020000130=Q_000020000*R_130[0]-a1Q_000010000_2*R_140[0]+aQin2*R_150[0];
			double QR_010000010130=Q_010000010*R_130[0]-a1Q_010000000_1*R_131[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000010010130=Q_000010010*R_130[0]-a1Q_000010000_1*R_131[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000000020130=Q_000000020*R_130[0]-a1Q_000000010_2*R_131[0]+aQin2*R_132[0];
			double QR_020000000202=Q_020000000*R_202[0]-a1Q_010000000_2*R_302[0]+aQin2*R_402[0];
			double QR_010010000202=Q_010010000*R_202[0]-a1Q_010000000_1*R_212[0]-a1Q_000010000_1*R_302[0]+aQin2*R_312[0];
			double QR_000020000202=Q_000020000*R_202[0]-a1Q_000010000_2*R_212[0]+aQin2*R_222[0];
			double QR_010000010202=Q_010000010*R_202[0]-a1Q_010000000_1*R_203[0]-a1Q_000000010_1*R_302[0]+aQin2*R_303[0];
			double QR_000010010202=Q_000010010*R_202[0]-a1Q_000010000_1*R_203[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000000020202=Q_000000020*R_202[0]-a1Q_000000010_2*R_203[0]+aQin2*R_204[0];
			double QR_020000000211=Q_020000000*R_211[0]-a1Q_010000000_2*R_311[0]+aQin2*R_411[0];
			double QR_010010000211=Q_010010000*R_211[0]-a1Q_010000000_1*R_221[0]-a1Q_000010000_1*R_311[0]+aQin2*R_321[0];
			double QR_000020000211=Q_000020000*R_211[0]-a1Q_000010000_2*R_221[0]+aQin2*R_231[0];
			double QR_010000010211=Q_010000010*R_211[0]-a1Q_010000000_1*R_212[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000010010211=Q_000010010*R_211[0]-a1Q_000010000_1*R_212[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000000020211=Q_000000020*R_211[0]-a1Q_000000010_2*R_212[0]+aQin2*R_213[0];
			double QR_020000000220=Q_020000000*R_220[0]-a1Q_010000000_2*R_320[0]+aQin2*R_420[0];
			double QR_010010000220=Q_010010000*R_220[0]-a1Q_010000000_1*R_230[0]-a1Q_000010000_1*R_320[0]+aQin2*R_330[0];
			double QR_000020000220=Q_000020000*R_220[0]-a1Q_000010000_2*R_230[0]+aQin2*R_240[0];
			double QR_010000010220=Q_010000010*R_220[0]-a1Q_010000000_1*R_221[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000010010220=Q_000010010*R_220[0]-a1Q_000010000_1*R_221[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000000020220=Q_000000020*R_220[0]-a1Q_000000010_2*R_221[0]+aQin2*R_222[0];
			double QR_020000000301=Q_020000000*R_301[0]-a1Q_010000000_2*R_401[0]+aQin2*R_501[0];
			double QR_010010000301=Q_010010000*R_301[0]-a1Q_010000000_1*R_311[0]-a1Q_000010000_1*R_401[0]+aQin2*R_411[0];
			double QR_000020000301=Q_000020000*R_301[0]-a1Q_000010000_2*R_311[0]+aQin2*R_321[0];
			double QR_010000010301=Q_010000010*R_301[0]-a1Q_010000000_1*R_302[0]-a1Q_000000010_1*R_401[0]+aQin2*R_402[0];
			double QR_000010010301=Q_000010010*R_301[0]-a1Q_000010000_1*R_302[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000000020301=Q_000000020*R_301[0]-a1Q_000000010_2*R_302[0]+aQin2*R_303[0];
			double QR_020000000310=Q_020000000*R_310[0]-a1Q_010000000_2*R_410[0]+aQin2*R_510[0];
			double QR_010010000310=Q_010010000*R_310[0]-a1Q_010000000_1*R_320[0]-a1Q_000010000_1*R_410[0]+aQin2*R_420[0];
			double QR_000020000310=Q_000020000*R_310[0]-a1Q_000010000_2*R_320[0]+aQin2*R_330[0];
			double QR_010000010310=Q_010000010*R_310[0]-a1Q_010000000_1*R_311[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000010010310=Q_000010010*R_310[0]-a1Q_000010000_1*R_311[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000000020310=Q_000000020*R_310[0]-a1Q_000000010_2*R_311[0]+aQin2*R_312[0];
			double QR_020000000400=Q_020000000*R_400[0]-a1Q_010000000_2*R_500[0]+aQin2*R_600[0];
			double QR_010010000400=Q_010010000*R_400[0]-a1Q_010000000_1*R_410[0]-a1Q_000010000_1*R_500[0]+aQin2*R_510[0];
			double QR_000020000400=Q_000020000*R_400[0]-a1Q_000010000_2*R_410[0]+aQin2*R_420[0];
			double QR_010000010400=Q_010000010*R_400[0]-a1Q_010000000_1*R_401[0]-a1Q_000000010_1*R_500[0]+aQin2*R_501[0];
			double QR_000010010400=Q_000010010*R_400[0]-a1Q_000010000_1*R_401[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000000020400=Q_000000020*R_400[0]-a1Q_000000010_2*R_401[0]+aQin2*R_402[0];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		for(int i=0;i<3;i++){
			Pd_002[i]=aPin1+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=aPin1*(2.000000*Pd_001[i]);
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=aPin1*(Pd_002[i]+2.000000*Pd_011[i]);
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=aPin1*(0.500000*Pd_102[i]+Pd_111[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=aPin1*2.000000*(Pd_012[i]+Pd_021[i]);
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=aPin1*(Pd_112[i]+Pd_121[i]);
			}
			double P_022000000;
			double P_122000000;
			double P_222000000;
			double P_021001000;
			double P_121001000;
			double P_221001000;
			double P_020002000;
			double P_021000001;
			double P_121000001;
			double P_221000001;
			double P_020001001;
			double P_020000002;
			double P_012010000;
			double P_112010000;
			double P_212010000;
			double P_011011000;
			double P_011111000;
			double P_111011000;
			double P_111111000;
			double P_010012000;
			double P_010112000;
			double P_010212000;
			double P_011010001;
			double P_111010001;
			double P_010011001;
			double P_010111001;
			double P_010010002;
			double P_002020000;
			double P_001021000;
			double P_001121000;
			double P_001221000;
			double P_000022000;
			double P_000122000;
			double P_000222000;
			double P_001020001;
			double P_000021001;
			double P_000121001;
			double P_000221001;
			double P_000020002;
			double P_012000010;
			double P_112000010;
			double P_212000010;
			double P_011001010;
			double P_111001010;
			double P_010002010;
			double P_011000011;
			double P_011000111;
			double P_111000011;
			double P_111000111;
			double P_010001011;
			double P_010001111;
			double P_010000012;
			double P_010000112;
			double P_010000212;
			double P_002010010;
			double P_001011010;
			double P_001111010;
			double P_000012010;
			double P_000112010;
			double P_000212010;
			double P_001010011;
			double P_001010111;
			double P_000011011;
			double P_000011111;
			double P_000111011;
			double P_000111111;
			double P_000010012;
			double P_000010112;
			double P_000010212;
			double P_002000020;
			double P_001001020;
			double P_000002020;
			double P_001000021;
			double P_001000121;
			double P_001000221;
			double P_000001021;
			double P_000001121;
			double P_000001221;
			double P_000000022;
			double P_000000122;
			double P_000000222;
			double a2P_111000000_1;
			double a2P_111000000_2;
			double a1P_021000000_1;
			double a1P_121000000_1;
			double a1P_221000000_1;
			double a3P_000001000_1;
			double a3P_000001000_2;
			double a1P_020001000_1;
			double a1P_020001000_2;
			double a2P_020000000_1;
			double a1P_010002000_1;
			double a1P_010002000_2;
			double a2P_010001000_1;
			double a2P_010001000_4;
			double a2P_010001000_2;
			double a3P_010000000_1;
			double a3P_010000000_2;
			double a2P_000002000_1;
			double a3P_000000001_1;
			double a3P_000000001_2;
			double a1P_020000001_1;
			double a1P_020000001_2;
			double a1P_010001001_1;
			double a1P_010001001_2;
			double a2P_010000001_1;
			double a2P_010000001_2;
			double a2P_010000001_4;
			double a2P_000001001_1;
			double a1P_010000002_1;
			double a1P_010000002_2;
			double a2P_000000002_1;
			double a1P_012000000_1;
			double a1P_112000000_1;
			double a1P_212000000_1;
			double a3P_000010000_1;
			double a3P_000010000_2;
			double a2P_011000000_1;
			double a2P_000011000_1;
			double a2P_000111000_1;
			double a2P_000111000_2;
			double a1P_000012000_1;
			double a1P_000112000_1;
			double a1P_000212000_1;
			double a1P_011010000_1;
			double a1P_011000001_1;
			double a1P_111010000_1;
			double a1P_111000001_1;
			double a2P_000010001_1;
			double a2P_000010001_2;
			double a2P_000010001_4;
			double a1P_010011000_1;
			double a1P_010111000_1;
			double a1P_000011001_1;
			double a1P_000111001_1;
			double a1P_010010001_1;
			double a1P_010010001_2;
			double a2P_010010000_1;
			double a1P_000010002_1;
			double a1P_000010002_2;
			double a1P_002010000_1;
			double a1P_002010000_2;
			double a2P_002000000_1;
			double a1P_001020000_1;
			double a1P_001020000_2;
			double a2P_001010000_1;
			double a2P_001010000_4;
			double a2P_001010000_2;
			double a3P_001000000_1;
			double a3P_001000000_2;
			double a2P_000020000_1;
			double a1P_000021000_1;
			double a1P_000121000_1;
			double a1P_000221000_1;
			double a1P_001010001_1;
			double a1P_001010001_2;
			double a2P_001000001_1;
			double a1P_000020001_1;
			double a1P_000020001_2;
			double a3P_000000010_1;
			double a3P_000000010_2;
			double a1P_011001000_1;
			double a1P_011000010_1;
			double a1P_111001000_1;
			double a1P_111000010_1;
			double a2P_000001010_1;
			double a2P_000001010_2;
			double a2P_000001010_4;
			double a1P_010001010_1;
			double a1P_010001010_2;
			double a2P_010000010_1;
			double a1P_000002010_1;
			double a1P_000002010_2;
			double a2P_000000011_1;
			double a2P_000000111_1;
			double a2P_000000111_2;
			double a1P_010000011_1;
			double a1P_010000111_1;
			double a1P_000001011_1;
			double a1P_000001111_1;
			double a1P_000000012_1;
			double a1P_000000112_1;
			double a1P_000000212_1;
			double a1P_002000010_1;
			double a1P_002000010_2;
			double a1P_001010010_1;
			double a1P_001010010_2;
			double a2P_001000010_1;
			double a2P_001000010_2;
			double a2P_001000010_4;
			double a2P_000010010_1;
			double a1P_001011000_1;
			double a1P_001111000_1;
			double a1P_000011010_1;
			double a1P_000111010_1;
			double a1P_001000011_1;
			double a1P_001000111_1;
			double a1P_000010011_1;
			double a1P_000010111_1;
			double a1P_001000020_1;
			double a1P_001000020_2;
			double a2P_000000020_1;
			double a1P_001001010_1;
			double a1P_001001010_2;
			double a2P_001001000_1;
			double a1P_000001020_1;
			double a1P_000001020_2;
			double a1P_000000021_1;
			double a1P_000000121_1;
			double a1P_000000221_1;
			P_022000000=Pd_022[0];
			P_122000000=Pd_122[0];
			P_222000000=Pd_222[0];
			P_021001000=Pd_021[0]*Pd_001[1];
			P_121001000=Pd_121[0]*Pd_001[1];
			P_221001000=Pd_221[0]*Pd_001[1];
			P_020002000=Pd_020[0]*Pd_002[1];
			P_021000001=Pd_021[0]*Pd_001[2];
			P_121000001=Pd_121[0]*Pd_001[2];
			P_221000001=Pd_221[0]*Pd_001[2];
			P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
			P_020000002=Pd_020[0]*Pd_002[2];
			P_012010000=Pd_012[0]*Pd_010[1];
			P_112010000=Pd_112[0]*Pd_010[1];
			P_212010000=Pd_212[0]*Pd_010[1];
			P_011011000=Pd_011[0]*Pd_011[1];
			P_011111000=Pd_011[0]*Pd_111[1];
			P_111011000=Pd_111[0]*Pd_011[1];
			P_111111000=Pd_111[0]*Pd_111[1];
			P_010012000=Pd_010[0]*Pd_012[1];
			P_010112000=Pd_010[0]*Pd_112[1];
			P_010212000=Pd_010[0]*Pd_212[1];
			P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
			P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
			P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
			P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
			P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
			P_002020000=Pd_002[0]*Pd_020[1];
			P_001021000=Pd_001[0]*Pd_021[1];
			P_001121000=Pd_001[0]*Pd_121[1];
			P_001221000=Pd_001[0]*Pd_221[1];
			P_000022000=Pd_022[1];
			P_000122000=Pd_122[1];
			P_000222000=Pd_222[1];
			P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
			P_000021001=Pd_021[1]*Pd_001[2];
			P_000121001=Pd_121[1]*Pd_001[2];
			P_000221001=Pd_221[1]*Pd_001[2];
			P_000020002=Pd_020[1]*Pd_002[2];
			P_012000010=Pd_012[0]*Pd_010[2];
			P_112000010=Pd_112[0]*Pd_010[2];
			P_212000010=Pd_212[0]*Pd_010[2];
			P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
			P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
			P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
			P_011000011=Pd_011[0]*Pd_011[2];
			P_011000111=Pd_011[0]*Pd_111[2];
			P_111000011=Pd_111[0]*Pd_011[2];
			P_111000111=Pd_111[0]*Pd_111[2];
			P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
			P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
			P_010000012=Pd_010[0]*Pd_012[2];
			P_010000112=Pd_010[0]*Pd_112[2];
			P_010000212=Pd_010[0]*Pd_212[2];
			P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
			P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
			P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
			P_000012010=Pd_012[1]*Pd_010[2];
			P_000112010=Pd_112[1]*Pd_010[2];
			P_000212010=Pd_212[1]*Pd_010[2];
			P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
			P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
			P_000011011=Pd_011[1]*Pd_011[2];
			P_000011111=Pd_011[1]*Pd_111[2];
			P_000111011=Pd_111[1]*Pd_011[2];
			P_000111111=Pd_111[1]*Pd_111[2];
			P_000010012=Pd_010[1]*Pd_012[2];
			P_000010112=Pd_010[1]*Pd_112[2];
			P_000010212=Pd_010[1]*Pd_212[2];
			P_002000020=Pd_002[0]*Pd_020[2];
			P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
			P_000002020=Pd_002[1]*Pd_020[2];
			P_001000021=Pd_001[0]*Pd_021[2];
			P_001000121=Pd_001[0]*Pd_121[2];
			P_001000221=Pd_001[0]*Pd_221[2];
			P_000001021=Pd_001[1]*Pd_021[2];
			P_000001121=Pd_001[1]*Pd_121[2];
			P_000001221=Pd_001[1]*Pd_221[2];
			P_000000022=Pd_022[2];
			P_000000122=Pd_122[2];
			P_000000222=Pd_222[2];
			a2P_111000000_1=Pd_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=Pd_021[0];
			a1P_121000000_1=Pd_121[0];
			a1P_221000000_1=Pd_221[0];
			a3P_000001000_1=Pd_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=Pd_020[0]*Pd_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=Pd_020[0];
			a1P_010002000_1=Pd_010[0]*Pd_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=Pd_010[0]*Pd_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=Pd_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=Pd_002[1];
			a3P_000000001_1=Pd_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=Pd_020[0]*Pd_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=Pd_010[0]*Pd_001[1]*Pd_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=Pd_010[0]*Pd_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=Pd_001[1]*Pd_001[2];
			a1P_010000002_1=Pd_010[0]*Pd_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=Pd_002[2];
			a1P_012000000_1=Pd_012[0];
			a1P_112000000_1=Pd_112[0];
			a1P_212000000_1=Pd_212[0];
			a3P_000010000_1=Pd_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=Pd_011[0];
			a2P_000011000_1=Pd_011[1];
			a2P_000111000_1=Pd_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=Pd_012[1];
			a1P_000112000_1=Pd_112[1];
			a1P_000212000_1=Pd_212[1];
			a1P_011010000_1=Pd_011[0]*Pd_010[1];
			a1P_011000001_1=Pd_011[0]*Pd_001[2];
			a1P_111010000_1=Pd_111[0]*Pd_010[1];
			a1P_111000001_1=Pd_111[0]*Pd_001[2];
			a2P_000010001_1=Pd_010[1]*Pd_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=Pd_010[0]*Pd_011[1];
			a1P_010111000_1=Pd_010[0]*Pd_111[1];
			a1P_000011001_1=Pd_011[1]*Pd_001[2];
			a1P_000111001_1=Pd_111[1]*Pd_001[2];
			a1P_010010001_1=Pd_010[0]*Pd_010[1]*Pd_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010002_1=Pd_010[1]*Pd_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=Pd_002[0]*Pd_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=Pd_002[0];
			a1P_001020000_1=Pd_001[0]*Pd_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=Pd_001[0]*Pd_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=Pd_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=Pd_020[1];
			a1P_000021000_1=Pd_021[1];
			a1P_000121000_1=Pd_121[1];
			a1P_000221000_1=Pd_221[1];
			a1P_001010001_1=Pd_001[0]*Pd_010[1]*Pd_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=Pd_001[0]*Pd_001[2];
			a1P_000020001_1=Pd_020[1]*Pd_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=Pd_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=Pd_011[0]*Pd_001[1];
			a1P_011000010_1=Pd_011[0]*Pd_010[2];
			a1P_111001000_1=Pd_111[0]*Pd_001[1];
			a1P_111000010_1=Pd_111[0]*Pd_010[2];
			a2P_000001010_1=Pd_001[1]*Pd_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=Pd_010[0]*Pd_001[1]*Pd_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000002010_1=Pd_002[1]*Pd_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=Pd_011[2];
			a2P_000000111_1=Pd_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=Pd_010[0]*Pd_011[2];
			a1P_010000111_1=Pd_010[0]*Pd_111[2];
			a1P_000001011_1=Pd_001[1]*Pd_011[2];
			a1P_000001111_1=Pd_001[1]*Pd_111[2];
			a1P_000000012_1=Pd_012[2];
			a1P_000000112_1=Pd_112[2];
			a1P_000000212_1=Pd_212[2];
			a1P_002000010_1=Pd_002[0]*Pd_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=Pd_001[0]*Pd_010[1]*Pd_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=Pd_001[0]*Pd_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_001011000_1=Pd_001[0]*Pd_011[1];
			a1P_001111000_1=Pd_001[0]*Pd_111[1];
			a1P_000011010_1=Pd_011[1]*Pd_010[2];
			a1P_000111010_1=Pd_111[1]*Pd_010[2];
			a1P_001000011_1=Pd_001[0]*Pd_011[2];
			a1P_001000111_1=Pd_001[0]*Pd_111[2];
			a1P_000010011_1=Pd_010[1]*Pd_011[2];
			a1P_000010111_1=Pd_010[1]*Pd_111[2];
			a1P_001000020_1=Pd_001[0]*Pd_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=Pd_020[2];
			a1P_001001010_1=Pd_001[0]*Pd_001[1]*Pd_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=Pd_001[0]*Pd_001[1];
			a1P_000001020_1=Pd_001[1]*Pd_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=Pd_021[2];
			a1P_000000121_1=Pd_121[2];
			a1P_000000221_1=Pd_221[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_022000000*QR_020000000000+P_122000000*QR_020000000100+P_222000000*QR_020000000200+a2P_111000000_2*QR_020000000300+aPin4*QR_020000000400);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_022000000*QR_010010000000+P_122000000*QR_010010000100+P_222000000*QR_010010000200+a2P_111000000_2*QR_010010000300+aPin4*QR_010010000400);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_022000000*QR_000020000000+P_122000000*QR_000020000100+P_222000000*QR_000020000200+a2P_111000000_2*QR_000020000300+aPin4*QR_000020000400);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(P_022000000*QR_010000010000+P_122000000*QR_010000010100+P_222000000*QR_010000010200+a2P_111000000_2*QR_010000010300+aPin4*QR_010000010400);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(P_022000000*QR_000010010000+P_122000000*QR_000010010100+P_222000000*QR_000010010200+a2P_111000000_2*QR_000010010300+aPin4*QR_000010010400);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(P_022000000*QR_000000020000+P_122000000*QR_000000020100+P_222000000*QR_000000020200+a2P_111000000_2*QR_000000020300+aPin4*QR_000000020400);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_021001000*QR_020000000000+a1P_021000000_1*QR_020000000010+P_121001000*QR_020000000100+a1P_121000000_1*QR_020000000110+P_221001000*QR_020000000200+a1P_221000000_1*QR_020000000210+a3P_000001000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_021001000*QR_010010000000+a1P_021000000_1*QR_010010000010+P_121001000*QR_010010000100+a1P_121000000_1*QR_010010000110+P_221001000*QR_010010000200+a1P_221000000_1*QR_010010000210+a3P_000001000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_021001000*QR_000020000000+a1P_021000000_1*QR_000020000010+P_121001000*QR_000020000100+a1P_121000000_1*QR_000020000110+P_221001000*QR_000020000200+a1P_221000000_1*QR_000020000210+a3P_000001000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(P_021001000*QR_010000010000+a1P_021000000_1*QR_010000010010+P_121001000*QR_010000010100+a1P_121000000_1*QR_010000010110+P_221001000*QR_010000010200+a1P_221000000_1*QR_010000010210+a3P_000001000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(P_021001000*QR_000010010000+a1P_021000000_1*QR_000010010010+P_121001000*QR_000010010100+a1P_121000000_1*QR_000010010110+P_221001000*QR_000010010200+a1P_221000000_1*QR_000010010210+a3P_000001000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(P_021001000*QR_000000020000+a1P_021000000_1*QR_000000020010+P_121001000*QR_000000020100+a1P_121000000_1*QR_000000020110+P_221001000*QR_000000020200+a1P_221000000_1*QR_000000020210+a3P_000001000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_020002000*QR_020000000000+a1P_020001000_2*QR_020000000010+a2P_020000000_1*QR_020000000020+a1P_010002000_2*QR_020000000100+a2P_010001000_4*QR_020000000110+a3P_010000000_2*QR_020000000120+a2P_000002000_1*QR_020000000200+a3P_000001000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_020002000*QR_010010000000+a1P_020001000_2*QR_010010000010+a2P_020000000_1*QR_010010000020+a1P_010002000_2*QR_010010000100+a2P_010001000_4*QR_010010000110+a3P_010000000_2*QR_010010000120+a2P_000002000_1*QR_010010000200+a3P_000001000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020002000*QR_000020000000+a1P_020001000_2*QR_000020000010+a2P_020000000_1*QR_000020000020+a1P_010002000_2*QR_000020000100+a2P_010001000_4*QR_000020000110+a3P_010000000_2*QR_000020000120+a2P_000002000_1*QR_000020000200+a3P_000001000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(P_020002000*QR_010000010000+a1P_020001000_2*QR_010000010010+a2P_020000000_1*QR_010000010020+a1P_010002000_2*QR_010000010100+a2P_010001000_4*QR_010000010110+a3P_010000000_2*QR_010000010120+a2P_000002000_1*QR_010000010200+a3P_000001000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(P_020002000*QR_000010010000+a1P_020001000_2*QR_000010010010+a2P_020000000_1*QR_000010010020+a1P_010002000_2*QR_000010010100+a2P_010001000_4*QR_000010010110+a3P_010000000_2*QR_000010010120+a2P_000002000_1*QR_000010010200+a3P_000001000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(P_020002000*QR_000000020000+a1P_020001000_2*QR_000000020010+a2P_020000000_1*QR_000000020020+a1P_010002000_2*QR_000000020100+a2P_010001000_4*QR_000000020110+a3P_010000000_2*QR_000000020120+a2P_000002000_1*QR_000000020200+a3P_000001000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_021000001*QR_020000000000+a1P_021000000_1*QR_020000000001+P_121000001*QR_020000000100+a1P_121000000_1*QR_020000000101+P_221000001*QR_020000000200+a1P_221000000_1*QR_020000000201+a3P_000000001_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_021000001*QR_010010000000+a1P_021000000_1*QR_010010000001+P_121000001*QR_010010000100+a1P_121000000_1*QR_010010000101+P_221000001*QR_010010000200+a1P_221000000_1*QR_010010000201+a3P_000000001_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_021000001*QR_000020000000+a1P_021000000_1*QR_000020000001+P_121000001*QR_000020000100+a1P_121000000_1*QR_000020000101+P_221000001*QR_000020000200+a1P_221000000_1*QR_000020000201+a3P_000000001_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(P_021000001*QR_010000010000+a1P_021000000_1*QR_010000010001+P_121000001*QR_010000010100+a1P_121000000_1*QR_010000010101+P_221000001*QR_010000010200+a1P_221000000_1*QR_010000010201+a3P_000000001_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(P_021000001*QR_000010010000+a1P_021000000_1*QR_000010010001+P_121000001*QR_000010010100+a1P_121000000_1*QR_000010010101+P_221000001*QR_000010010200+a1P_221000000_1*QR_000010010201+a3P_000000001_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(P_021000001*QR_000000020000+a1P_021000000_1*QR_000000020001+P_121000001*QR_000000020100+a1P_121000000_1*QR_000000020101+P_221000001*QR_000000020200+a1P_221000000_1*QR_000000020201+a3P_000000001_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_020001001*QR_020000000000+a1P_020001000_1*QR_020000000001+a1P_020000001_1*QR_020000000010+a2P_020000000_1*QR_020000000011+a1P_010001001_2*QR_020000000100+a2P_010001000_2*QR_020000000101+a2P_010000001_2*QR_020000000110+a3P_010000000_2*QR_020000000111+a2P_000001001_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_020001001*QR_010010000000+a1P_020001000_1*QR_010010000001+a1P_020000001_1*QR_010010000010+a2P_020000000_1*QR_010010000011+a1P_010001001_2*QR_010010000100+a2P_010001000_2*QR_010010000101+a2P_010000001_2*QR_010010000110+a3P_010000000_2*QR_010010000111+a2P_000001001_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020001001*QR_000020000000+a1P_020001000_1*QR_000020000001+a1P_020000001_1*QR_000020000010+a2P_020000000_1*QR_000020000011+a1P_010001001_2*QR_000020000100+a2P_010001000_2*QR_000020000101+a2P_010000001_2*QR_000020000110+a3P_010000000_2*QR_000020000111+a2P_000001001_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(P_020001001*QR_010000010000+a1P_020001000_1*QR_010000010001+a1P_020000001_1*QR_010000010010+a2P_020000000_1*QR_010000010011+a1P_010001001_2*QR_010000010100+a2P_010001000_2*QR_010000010101+a2P_010000001_2*QR_010000010110+a3P_010000000_2*QR_010000010111+a2P_000001001_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(P_020001001*QR_000010010000+a1P_020001000_1*QR_000010010001+a1P_020000001_1*QR_000010010010+a2P_020000000_1*QR_000010010011+a1P_010001001_2*QR_000010010100+a2P_010001000_2*QR_000010010101+a2P_010000001_2*QR_000010010110+a3P_010000000_2*QR_000010010111+a2P_000001001_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(P_020001001*QR_000000020000+a1P_020001000_1*QR_000000020001+a1P_020000001_1*QR_000000020010+a2P_020000000_1*QR_000000020011+a1P_010001001_2*QR_000000020100+a2P_010001000_2*QR_000000020101+a2P_010000001_2*QR_000000020110+a3P_010000000_2*QR_000000020111+a2P_000001001_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_020000002*QR_020000000000+a1P_020000001_2*QR_020000000001+a2P_020000000_1*QR_020000000002+a1P_010000002_2*QR_020000000100+a2P_010000001_4*QR_020000000101+a3P_010000000_2*QR_020000000102+a2P_000000002_1*QR_020000000200+a3P_000000001_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_020000002*QR_010010000000+a1P_020000001_2*QR_010010000001+a2P_020000000_1*QR_010010000002+a1P_010000002_2*QR_010010000100+a2P_010000001_4*QR_010010000101+a3P_010000000_2*QR_010010000102+a2P_000000002_1*QR_010010000200+a3P_000000001_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020000002*QR_000020000000+a1P_020000001_2*QR_000020000001+a2P_020000000_1*QR_000020000002+a1P_010000002_2*QR_000020000100+a2P_010000001_4*QR_000020000101+a3P_010000000_2*QR_000020000102+a2P_000000002_1*QR_000020000200+a3P_000000001_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(P_020000002*QR_010000010000+a1P_020000001_2*QR_010000010001+a2P_020000000_1*QR_010000010002+a1P_010000002_2*QR_010000010100+a2P_010000001_4*QR_010000010101+a3P_010000000_2*QR_010000010102+a2P_000000002_1*QR_010000010200+a3P_000000001_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(P_020000002*QR_000010010000+a1P_020000001_2*QR_000010010001+a2P_020000000_1*QR_000010010002+a1P_010000002_2*QR_000010010100+a2P_010000001_4*QR_000010010101+a3P_010000000_2*QR_000010010102+a2P_000000002_1*QR_000010010200+a3P_000000001_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(P_020000002*QR_000000020000+a1P_020000001_2*QR_000000020001+a2P_020000000_1*QR_000000020002+a1P_010000002_2*QR_000000020100+a2P_010000001_4*QR_000000020101+a3P_010000000_2*QR_000000020102+a2P_000000002_1*QR_000000020200+a3P_000000001_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_012010000*QR_020000000000+a1P_012000000_1*QR_020000000010+P_112010000*QR_020000000100+a1P_112000000_1*QR_020000000110+P_212010000*QR_020000000200+a1P_212000000_1*QR_020000000210+a3P_000010000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_012010000*QR_010010000000+a1P_012000000_1*QR_010010000010+P_112010000*QR_010010000100+a1P_112000000_1*QR_010010000110+P_212010000*QR_010010000200+a1P_212000000_1*QR_010010000210+a3P_000010000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_012010000*QR_000020000000+a1P_012000000_1*QR_000020000010+P_112010000*QR_000020000100+a1P_112000000_1*QR_000020000110+P_212010000*QR_000020000200+a1P_212000000_1*QR_000020000210+a3P_000010000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(P_012010000*QR_010000010000+a1P_012000000_1*QR_010000010010+P_112010000*QR_010000010100+a1P_112000000_1*QR_010000010110+P_212010000*QR_010000010200+a1P_212000000_1*QR_010000010210+a3P_000010000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(P_012010000*QR_000010010000+a1P_012000000_1*QR_000010010010+P_112010000*QR_000010010100+a1P_112000000_1*QR_000010010110+P_212010000*QR_000010010200+a1P_212000000_1*QR_000010010210+a3P_000010000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(P_012010000*QR_000000020000+a1P_012000000_1*QR_000000020010+P_112010000*QR_000000020100+a1P_112000000_1*QR_000000020110+P_212010000*QR_000000020200+a1P_212000000_1*QR_000000020210+a3P_000010000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_011011000*QR_020000000000+P_011111000*QR_020000000010+a2P_011000000_1*QR_020000000020+P_111011000*QR_020000000100+P_111111000*QR_020000000110+a2P_111000000_1*QR_020000000120+a2P_000011000_1*QR_020000000200+a2P_000111000_1*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_011011000*QR_010010000000+P_011111000*QR_010010000010+a2P_011000000_1*QR_010010000020+P_111011000*QR_010010000100+P_111111000*QR_010010000110+a2P_111000000_1*QR_010010000120+a2P_000011000_1*QR_010010000200+a2P_000111000_1*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_011011000*QR_000020000000+P_011111000*QR_000020000010+a2P_011000000_1*QR_000020000020+P_111011000*QR_000020000100+P_111111000*QR_000020000110+a2P_111000000_1*QR_000020000120+a2P_000011000_1*QR_000020000200+a2P_000111000_1*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(P_011011000*QR_010000010000+P_011111000*QR_010000010010+a2P_011000000_1*QR_010000010020+P_111011000*QR_010000010100+P_111111000*QR_010000010110+a2P_111000000_1*QR_010000010120+a2P_000011000_1*QR_010000010200+a2P_000111000_1*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(P_011011000*QR_000010010000+P_011111000*QR_000010010010+a2P_011000000_1*QR_000010010020+P_111011000*QR_000010010100+P_111111000*QR_000010010110+a2P_111000000_1*QR_000010010120+a2P_000011000_1*QR_000010010200+a2P_000111000_1*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(P_011011000*QR_000000020000+P_011111000*QR_000000020010+a2P_011000000_1*QR_000000020020+P_111011000*QR_000000020100+P_111111000*QR_000000020110+a2P_111000000_1*QR_000000020120+a2P_000011000_1*QR_000000020200+a2P_000111000_1*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_010012000*QR_020000000000+P_010112000*QR_020000000010+P_010212000*QR_020000000020+a3P_010000000_1*QR_020000000030+a1P_000012000_1*QR_020000000100+a1P_000112000_1*QR_020000000110+a1P_000212000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_010012000*QR_010010000000+P_010112000*QR_010010000010+P_010212000*QR_010010000020+a3P_010000000_1*QR_010010000030+a1P_000012000_1*QR_010010000100+a1P_000112000_1*QR_010010000110+a1P_000212000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010012000*QR_000020000000+P_010112000*QR_000020000010+P_010212000*QR_000020000020+a3P_010000000_1*QR_000020000030+a1P_000012000_1*QR_000020000100+a1P_000112000_1*QR_000020000110+a1P_000212000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(P_010012000*QR_010000010000+P_010112000*QR_010000010010+P_010212000*QR_010000010020+a3P_010000000_1*QR_010000010030+a1P_000012000_1*QR_010000010100+a1P_000112000_1*QR_010000010110+a1P_000212000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(P_010012000*QR_000010010000+P_010112000*QR_000010010010+P_010212000*QR_000010010020+a3P_010000000_1*QR_000010010030+a1P_000012000_1*QR_000010010100+a1P_000112000_1*QR_000010010110+a1P_000212000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(P_010012000*QR_000000020000+P_010112000*QR_000000020010+P_010212000*QR_000000020020+a3P_010000000_1*QR_000000020030+a1P_000012000_1*QR_000000020100+a1P_000112000_1*QR_000000020110+a1P_000212000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_011010001*QR_020000000000+a1P_011010000_1*QR_020000000001+a1P_011000001_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111010001*QR_020000000100+a1P_111010000_1*QR_020000000101+a1P_111000001_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000010001_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_011010001*QR_010010000000+a1P_011010000_1*QR_010010000001+a1P_011000001_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111010001*QR_010010000100+a1P_111010000_1*QR_010010000101+a1P_111000001_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000010001_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_011010001*QR_000020000000+a1P_011010000_1*QR_000020000001+a1P_011000001_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111010001*QR_000020000100+a1P_111010000_1*QR_000020000101+a1P_111000001_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000010001_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(P_011010001*QR_010000010000+a1P_011010000_1*QR_010000010001+a1P_011000001_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111010001*QR_010000010100+a1P_111010000_1*QR_010000010101+a1P_111000001_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000010001_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(P_011010001*QR_000010010000+a1P_011010000_1*QR_000010010001+a1P_011000001_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111010001*QR_000010010100+a1P_111010000_1*QR_000010010101+a1P_111000001_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000010001_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(P_011010001*QR_000000020000+a1P_011010000_1*QR_000000020001+a1P_011000001_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111010001*QR_000000020100+a1P_111010000_1*QR_000000020101+a1P_111000001_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000010001_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_010011001*QR_020000000000+a1P_010011000_1*QR_020000000001+P_010111001*QR_020000000010+a1P_010111000_1*QR_020000000011+a2P_010000001_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000011001_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111001_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_010011001*QR_010010000000+a1P_010011000_1*QR_010010000001+P_010111001*QR_010010000010+a1P_010111000_1*QR_010010000011+a2P_010000001_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000011001_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111001_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010011001*QR_000020000000+a1P_010011000_1*QR_000020000001+P_010111001*QR_000020000010+a1P_010111000_1*QR_000020000011+a2P_010000001_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000011001_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111001_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(P_010011001*QR_010000010000+a1P_010011000_1*QR_010000010001+P_010111001*QR_010000010010+a1P_010111000_1*QR_010000010011+a2P_010000001_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000011001_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111001_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(P_010011001*QR_000010010000+a1P_010011000_1*QR_000010010001+P_010111001*QR_000010010010+a1P_010111000_1*QR_000010010011+a2P_010000001_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000011001_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111001_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(P_010011001*QR_000000020000+a1P_010011000_1*QR_000000020001+P_010111001*QR_000000020010+a1P_010111000_1*QR_000000020011+a2P_010000001_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000011001_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111001_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_010010002*QR_020000000000+a1P_010010001_2*QR_020000000001+a2P_010010000_1*QR_020000000002+a1P_010000002_1*QR_020000000010+a2P_010000001_2*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000010002_1*QR_020000000100+a2P_000010001_2*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000002_1*QR_020000000110+a3P_000000001_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_010010002*QR_010010000000+a1P_010010001_2*QR_010010000001+a2P_010010000_1*QR_010010000002+a1P_010000002_1*QR_010010000010+a2P_010000001_2*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000010002_1*QR_010010000100+a2P_000010001_2*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000002_1*QR_010010000110+a3P_000000001_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010010002*QR_000020000000+a1P_010010001_2*QR_000020000001+a2P_010010000_1*QR_000020000002+a1P_010000002_1*QR_000020000010+a2P_010000001_2*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000010002_1*QR_000020000100+a2P_000010001_2*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000002_1*QR_000020000110+a3P_000000001_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(P_010010002*QR_010000010000+a1P_010010001_2*QR_010000010001+a2P_010010000_1*QR_010000010002+a1P_010000002_1*QR_010000010010+a2P_010000001_2*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000010002_1*QR_010000010100+a2P_000010001_2*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000002_1*QR_010000010110+a3P_000000001_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(P_010010002*QR_000010010000+a1P_010010001_2*QR_000010010001+a2P_010010000_1*QR_000010010002+a1P_010000002_1*QR_000010010010+a2P_010000001_2*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000010002_1*QR_000010010100+a2P_000010001_2*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000002_1*QR_000010010110+a3P_000000001_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(P_010010002*QR_000000020000+a1P_010010001_2*QR_000000020001+a2P_010010000_1*QR_000000020002+a1P_010000002_1*QR_000000020010+a2P_010000001_2*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000010002_1*QR_000000020100+a2P_000010001_2*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000002_1*QR_000000020110+a3P_000000001_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_002020000*QR_020000000000+a1P_002010000_2*QR_020000000010+a2P_002000000_1*QR_020000000020+a1P_001020000_2*QR_020000000100+a2P_001010000_4*QR_020000000110+a3P_001000000_2*QR_020000000120+a2P_000020000_1*QR_020000000200+a3P_000010000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_002020000*QR_010010000000+a1P_002010000_2*QR_010010000010+a2P_002000000_1*QR_010010000020+a1P_001020000_2*QR_010010000100+a2P_001010000_4*QR_010010000110+a3P_001000000_2*QR_010010000120+a2P_000020000_1*QR_010010000200+a3P_000010000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_002020000*QR_000020000000+a1P_002010000_2*QR_000020000010+a2P_002000000_1*QR_000020000020+a1P_001020000_2*QR_000020000100+a2P_001010000_4*QR_000020000110+a3P_001000000_2*QR_000020000120+a2P_000020000_1*QR_000020000200+a3P_000010000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(P_002020000*QR_010000010000+a1P_002010000_2*QR_010000010010+a2P_002000000_1*QR_010000010020+a1P_001020000_2*QR_010000010100+a2P_001010000_4*QR_010000010110+a3P_001000000_2*QR_010000010120+a2P_000020000_1*QR_010000010200+a3P_000010000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(P_002020000*QR_000010010000+a1P_002010000_2*QR_000010010010+a2P_002000000_1*QR_000010010020+a1P_001020000_2*QR_000010010100+a2P_001010000_4*QR_000010010110+a3P_001000000_2*QR_000010010120+a2P_000020000_1*QR_000010010200+a3P_000010000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(P_002020000*QR_000000020000+a1P_002010000_2*QR_000000020010+a2P_002000000_1*QR_000000020020+a1P_001020000_2*QR_000000020100+a2P_001010000_4*QR_000000020110+a3P_001000000_2*QR_000000020120+a2P_000020000_1*QR_000000020200+a3P_000010000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_001021000*QR_020000000000+P_001121000*QR_020000000010+P_001221000*QR_020000000020+a3P_001000000_1*QR_020000000030+a1P_000021000_1*QR_020000000100+a1P_000121000_1*QR_020000000110+a1P_000221000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_001021000*QR_010010000000+P_001121000*QR_010010000010+P_001221000*QR_010010000020+a3P_001000000_1*QR_010010000030+a1P_000021000_1*QR_010010000100+a1P_000121000_1*QR_010010000110+a1P_000221000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_001021000*QR_000020000000+P_001121000*QR_000020000010+P_001221000*QR_000020000020+a3P_001000000_1*QR_000020000030+a1P_000021000_1*QR_000020000100+a1P_000121000_1*QR_000020000110+a1P_000221000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(P_001021000*QR_010000010000+P_001121000*QR_010000010010+P_001221000*QR_010000010020+a3P_001000000_1*QR_010000010030+a1P_000021000_1*QR_010000010100+a1P_000121000_1*QR_010000010110+a1P_000221000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(P_001021000*QR_000010010000+P_001121000*QR_000010010010+P_001221000*QR_000010010020+a3P_001000000_1*QR_000010010030+a1P_000021000_1*QR_000010010100+a1P_000121000_1*QR_000010010110+a1P_000221000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(P_001021000*QR_000000020000+P_001121000*QR_000000020010+P_001221000*QR_000000020020+a3P_001000000_1*QR_000000020030+a1P_000021000_1*QR_000000020100+a1P_000121000_1*QR_000000020110+a1P_000221000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_000022000*QR_020000000000+P_000122000*QR_020000000010+P_000222000*QR_020000000020+a2P_000111000_2*QR_020000000030+aPin4*QR_020000000040);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_000022000*QR_010010000000+P_000122000*QR_010010000010+P_000222000*QR_010010000020+a2P_000111000_2*QR_010010000030+aPin4*QR_010010000040);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000022000*QR_000020000000+P_000122000*QR_000020000010+P_000222000*QR_000020000020+a2P_000111000_2*QR_000020000030+aPin4*QR_000020000040);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(P_000022000*QR_010000010000+P_000122000*QR_010000010010+P_000222000*QR_010000010020+a2P_000111000_2*QR_010000010030+aPin4*QR_010000010040);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(P_000022000*QR_000010010000+P_000122000*QR_000010010010+P_000222000*QR_000010010020+a2P_000111000_2*QR_000010010030+aPin4*QR_000010010040);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(P_000022000*QR_000000020000+P_000122000*QR_000000020010+P_000222000*QR_000000020020+a2P_000111000_2*QR_000000020030+aPin4*QR_000000020040);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_001020001*QR_020000000000+a1P_001020000_1*QR_020000000001+a1P_001010001_2*QR_020000000010+a2P_001010000_2*QR_020000000011+a2P_001000001_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000020001_1*QR_020000000100+a2P_000020000_1*QR_020000000101+a2P_000010001_2*QR_020000000110+a3P_000010000_2*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_001020001*QR_010010000000+a1P_001020000_1*QR_010010000001+a1P_001010001_2*QR_010010000010+a2P_001010000_2*QR_010010000011+a2P_001000001_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000020001_1*QR_010010000100+a2P_000020000_1*QR_010010000101+a2P_000010001_2*QR_010010000110+a3P_000010000_2*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_001020001*QR_000020000000+a1P_001020000_1*QR_000020000001+a1P_001010001_2*QR_000020000010+a2P_001010000_2*QR_000020000011+a2P_001000001_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000020001_1*QR_000020000100+a2P_000020000_1*QR_000020000101+a2P_000010001_2*QR_000020000110+a3P_000010000_2*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(P_001020001*QR_010000010000+a1P_001020000_1*QR_010000010001+a1P_001010001_2*QR_010000010010+a2P_001010000_2*QR_010000010011+a2P_001000001_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000020001_1*QR_010000010100+a2P_000020000_1*QR_010000010101+a2P_000010001_2*QR_010000010110+a3P_000010000_2*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(P_001020001*QR_000010010000+a1P_001020000_1*QR_000010010001+a1P_001010001_2*QR_000010010010+a2P_001010000_2*QR_000010010011+a2P_001000001_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000020001_1*QR_000010010100+a2P_000020000_1*QR_000010010101+a2P_000010001_2*QR_000010010110+a3P_000010000_2*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(P_001020001*QR_000000020000+a1P_001020000_1*QR_000000020001+a1P_001010001_2*QR_000000020010+a2P_001010000_2*QR_000000020011+a2P_001000001_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000020001_1*QR_000000020100+a2P_000020000_1*QR_000000020101+a2P_000010001_2*QR_000000020110+a3P_000010000_2*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_000021001*QR_020000000000+a1P_000021000_1*QR_020000000001+P_000121001*QR_020000000010+a1P_000121000_1*QR_020000000011+P_000221001*QR_020000000020+a1P_000221000_1*QR_020000000021+a3P_000000001_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_000021001*QR_010010000000+a1P_000021000_1*QR_010010000001+P_000121001*QR_010010000010+a1P_000121000_1*QR_010010000011+P_000221001*QR_010010000020+a1P_000221000_1*QR_010010000021+a3P_000000001_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000021001*QR_000020000000+a1P_000021000_1*QR_000020000001+P_000121001*QR_000020000010+a1P_000121000_1*QR_000020000011+P_000221001*QR_000020000020+a1P_000221000_1*QR_000020000021+a3P_000000001_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(P_000021001*QR_010000010000+a1P_000021000_1*QR_010000010001+P_000121001*QR_010000010010+a1P_000121000_1*QR_010000010011+P_000221001*QR_010000010020+a1P_000221000_1*QR_010000010021+a3P_000000001_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(P_000021001*QR_000010010000+a1P_000021000_1*QR_000010010001+P_000121001*QR_000010010010+a1P_000121000_1*QR_000010010011+P_000221001*QR_000010010020+a1P_000221000_1*QR_000010010021+a3P_000000001_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(P_000021001*QR_000000020000+a1P_000021000_1*QR_000000020001+P_000121001*QR_000000020010+a1P_000121000_1*QR_000000020011+P_000221001*QR_000000020020+a1P_000221000_1*QR_000000020021+a3P_000000001_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_000020002*QR_020000000000+a1P_000020001_2*QR_020000000001+a2P_000020000_1*QR_020000000002+a1P_000010002_2*QR_020000000010+a2P_000010001_4*QR_020000000011+a3P_000010000_2*QR_020000000012+a2P_000000002_1*QR_020000000020+a3P_000000001_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_000020002*QR_010010000000+a1P_000020001_2*QR_010010000001+a2P_000020000_1*QR_010010000002+a1P_000010002_2*QR_010010000010+a2P_000010001_4*QR_010010000011+a3P_000010000_2*QR_010010000012+a2P_000000002_1*QR_010010000020+a3P_000000001_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000020002*QR_000020000000+a1P_000020001_2*QR_000020000001+a2P_000020000_1*QR_000020000002+a1P_000010002_2*QR_000020000010+a2P_000010001_4*QR_000020000011+a3P_000010000_2*QR_000020000012+a2P_000000002_1*QR_000020000020+a3P_000000001_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(P_000020002*QR_010000010000+a1P_000020001_2*QR_010000010001+a2P_000020000_1*QR_010000010002+a1P_000010002_2*QR_010000010010+a2P_000010001_4*QR_010000010011+a3P_000010000_2*QR_010000010012+a2P_000000002_1*QR_010000010020+a3P_000000001_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(P_000020002*QR_000010010000+a1P_000020001_2*QR_000010010001+a2P_000020000_1*QR_000010010002+a1P_000010002_2*QR_000010010010+a2P_000010001_4*QR_000010010011+a3P_000010000_2*QR_000010010012+a2P_000000002_1*QR_000010010020+a3P_000000001_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(P_000020002*QR_000000020000+a1P_000020001_2*QR_000000020001+a2P_000020000_1*QR_000000020002+a1P_000010002_2*QR_000000020010+a2P_000010001_4*QR_000000020011+a3P_000010000_2*QR_000000020012+a2P_000000002_1*QR_000000020020+a3P_000000001_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_012000010*QR_020000000000+a1P_012000000_1*QR_020000000001+P_112000010*QR_020000000100+a1P_112000000_1*QR_020000000101+P_212000010*QR_020000000200+a1P_212000000_1*QR_020000000201+a3P_000000010_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_012000010*QR_010010000000+a1P_012000000_1*QR_010010000001+P_112000010*QR_010010000100+a1P_112000000_1*QR_010010000101+P_212000010*QR_010010000200+a1P_212000000_1*QR_010010000201+a3P_000000010_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_012000010*QR_000020000000+a1P_012000000_1*QR_000020000001+P_112000010*QR_000020000100+a1P_112000000_1*QR_000020000101+P_212000010*QR_000020000200+a1P_212000000_1*QR_000020000201+a3P_000000010_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(P_012000010*QR_010000010000+a1P_012000000_1*QR_010000010001+P_112000010*QR_010000010100+a1P_112000000_1*QR_010000010101+P_212000010*QR_010000010200+a1P_212000000_1*QR_010000010201+a3P_000000010_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(P_012000010*QR_000010010000+a1P_012000000_1*QR_000010010001+P_112000010*QR_000010010100+a1P_112000000_1*QR_000010010101+P_212000010*QR_000010010200+a1P_212000000_1*QR_000010010201+a3P_000000010_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(P_012000010*QR_000000020000+a1P_012000000_1*QR_000000020001+P_112000010*QR_000000020100+a1P_112000000_1*QR_000000020101+P_212000010*QR_000000020200+a1P_212000000_1*QR_000000020201+a3P_000000010_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_011001010*QR_020000000000+a1P_011001000_1*QR_020000000001+a1P_011000010_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111001010*QR_020000000100+a1P_111001000_1*QR_020000000101+a1P_111000010_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000001010_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_011001010*QR_010010000000+a1P_011001000_1*QR_010010000001+a1P_011000010_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111001010*QR_010010000100+a1P_111001000_1*QR_010010000101+a1P_111000010_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000001010_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_011001010*QR_000020000000+a1P_011001000_1*QR_000020000001+a1P_011000010_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111001010*QR_000020000100+a1P_111001000_1*QR_000020000101+a1P_111000010_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000001010_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(P_011001010*QR_010000010000+a1P_011001000_1*QR_010000010001+a1P_011000010_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111001010*QR_010000010100+a1P_111001000_1*QR_010000010101+a1P_111000010_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000001010_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(P_011001010*QR_000010010000+a1P_011001000_1*QR_000010010001+a1P_011000010_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111001010*QR_000010010100+a1P_111001000_1*QR_000010010101+a1P_111000010_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000001010_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(P_011001010*QR_000000020000+a1P_011001000_1*QR_000000020001+a1P_011000010_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111001010*QR_000000020100+a1P_111001000_1*QR_000000020101+a1P_111000010_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000001010_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_010002010*QR_020000000000+a1P_010002000_1*QR_020000000001+a1P_010001010_2*QR_020000000010+a2P_010001000_2*QR_020000000011+a2P_010000010_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000002010_1*QR_020000000100+a2P_000002000_1*QR_020000000101+a2P_000001010_2*QR_020000000110+a3P_000001000_2*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_010002010*QR_010010000000+a1P_010002000_1*QR_010010000001+a1P_010001010_2*QR_010010000010+a2P_010001000_2*QR_010010000011+a2P_010000010_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000002010_1*QR_010010000100+a2P_000002000_1*QR_010010000101+a2P_000001010_2*QR_010010000110+a3P_000001000_2*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010002010*QR_000020000000+a1P_010002000_1*QR_000020000001+a1P_010001010_2*QR_000020000010+a2P_010001000_2*QR_000020000011+a2P_010000010_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000002010_1*QR_000020000100+a2P_000002000_1*QR_000020000101+a2P_000001010_2*QR_000020000110+a3P_000001000_2*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(P_010002010*QR_010000010000+a1P_010002000_1*QR_010000010001+a1P_010001010_2*QR_010000010010+a2P_010001000_2*QR_010000010011+a2P_010000010_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000002010_1*QR_010000010100+a2P_000002000_1*QR_010000010101+a2P_000001010_2*QR_010000010110+a3P_000001000_2*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(P_010002010*QR_000010010000+a1P_010002000_1*QR_000010010001+a1P_010001010_2*QR_000010010010+a2P_010001000_2*QR_000010010011+a2P_010000010_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000002010_1*QR_000010010100+a2P_000002000_1*QR_000010010101+a2P_000001010_2*QR_000010010110+a3P_000001000_2*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(P_010002010*QR_000000020000+a1P_010002000_1*QR_000000020001+a1P_010001010_2*QR_000000020010+a2P_010001000_2*QR_000000020011+a2P_010000010_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000002010_1*QR_000000020100+a2P_000002000_1*QR_000000020101+a2P_000001010_2*QR_000000020110+a3P_000001000_2*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_011000011*QR_020000000000+P_011000111*QR_020000000001+a2P_011000000_1*QR_020000000002+P_111000011*QR_020000000100+P_111000111*QR_020000000101+a2P_111000000_1*QR_020000000102+a2P_000000011_1*QR_020000000200+a2P_000000111_1*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_011000011*QR_010010000000+P_011000111*QR_010010000001+a2P_011000000_1*QR_010010000002+P_111000011*QR_010010000100+P_111000111*QR_010010000101+a2P_111000000_1*QR_010010000102+a2P_000000011_1*QR_010010000200+a2P_000000111_1*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_011000011*QR_000020000000+P_011000111*QR_000020000001+a2P_011000000_1*QR_000020000002+P_111000011*QR_000020000100+P_111000111*QR_000020000101+a2P_111000000_1*QR_000020000102+a2P_000000011_1*QR_000020000200+a2P_000000111_1*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(P_011000011*QR_010000010000+P_011000111*QR_010000010001+a2P_011000000_1*QR_010000010002+P_111000011*QR_010000010100+P_111000111*QR_010000010101+a2P_111000000_1*QR_010000010102+a2P_000000011_1*QR_010000010200+a2P_000000111_1*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(P_011000011*QR_000010010000+P_011000111*QR_000010010001+a2P_011000000_1*QR_000010010002+P_111000011*QR_000010010100+P_111000111*QR_000010010101+a2P_111000000_1*QR_000010010102+a2P_000000011_1*QR_000010010200+a2P_000000111_1*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(P_011000011*QR_000000020000+P_011000111*QR_000000020001+a2P_011000000_1*QR_000000020002+P_111000011*QR_000000020100+P_111000111*QR_000000020101+a2P_111000000_1*QR_000000020102+a2P_000000011_1*QR_000000020200+a2P_000000111_1*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_010001011*QR_020000000000+P_010001111*QR_020000000001+a2P_010001000_1*QR_020000000002+a1P_010000011_1*QR_020000000010+a1P_010000111_1*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000001011_1*QR_020000000100+a1P_000001111_1*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_010001011*QR_010010000000+P_010001111*QR_010010000001+a2P_010001000_1*QR_010010000002+a1P_010000011_1*QR_010010000010+a1P_010000111_1*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000001011_1*QR_010010000100+a1P_000001111_1*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010001011*QR_000020000000+P_010001111*QR_000020000001+a2P_010001000_1*QR_000020000002+a1P_010000011_1*QR_000020000010+a1P_010000111_1*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000001011_1*QR_000020000100+a1P_000001111_1*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(P_010001011*QR_010000010000+P_010001111*QR_010000010001+a2P_010001000_1*QR_010000010002+a1P_010000011_1*QR_010000010010+a1P_010000111_1*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000001011_1*QR_010000010100+a1P_000001111_1*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(P_010001011*QR_000010010000+P_010001111*QR_000010010001+a2P_010001000_1*QR_000010010002+a1P_010000011_1*QR_000010010010+a1P_010000111_1*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000001011_1*QR_000010010100+a1P_000001111_1*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(P_010001011*QR_000000020000+P_010001111*QR_000000020001+a2P_010001000_1*QR_000000020002+a1P_010000011_1*QR_000000020010+a1P_010000111_1*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000001011_1*QR_000000020100+a1P_000001111_1*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_010000012*QR_020000000000+P_010000112*QR_020000000001+P_010000212*QR_020000000002+a3P_010000000_1*QR_020000000003+a1P_000000012_1*QR_020000000100+a1P_000000112_1*QR_020000000101+a1P_000000212_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_010000012*QR_010010000000+P_010000112*QR_010010000001+P_010000212*QR_010010000002+a3P_010000000_1*QR_010010000003+a1P_000000012_1*QR_010010000100+a1P_000000112_1*QR_010010000101+a1P_000000212_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010000012*QR_000020000000+P_010000112*QR_000020000001+P_010000212*QR_000020000002+a3P_010000000_1*QR_000020000003+a1P_000000012_1*QR_000020000100+a1P_000000112_1*QR_000020000101+a1P_000000212_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(P_010000012*QR_010000010000+P_010000112*QR_010000010001+P_010000212*QR_010000010002+a3P_010000000_1*QR_010000010003+a1P_000000012_1*QR_010000010100+a1P_000000112_1*QR_010000010101+a1P_000000212_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(P_010000012*QR_000010010000+P_010000112*QR_000010010001+P_010000212*QR_000010010002+a3P_010000000_1*QR_000010010003+a1P_000000012_1*QR_000010010100+a1P_000000112_1*QR_000010010101+a1P_000000212_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(P_010000012*QR_000000020000+P_010000112*QR_000000020001+P_010000212*QR_000000020002+a3P_010000000_1*QR_000000020003+a1P_000000012_1*QR_000000020100+a1P_000000112_1*QR_000000020101+a1P_000000212_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_002010010*QR_020000000000+a1P_002010000_1*QR_020000000001+a1P_002000010_1*QR_020000000010+a2P_002000000_1*QR_020000000011+a1P_001010010_2*QR_020000000100+a2P_001010000_2*QR_020000000101+a2P_001000010_2*QR_020000000110+a3P_001000000_2*QR_020000000111+a2P_000010010_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_002010010*QR_010010000000+a1P_002010000_1*QR_010010000001+a1P_002000010_1*QR_010010000010+a2P_002000000_1*QR_010010000011+a1P_001010010_2*QR_010010000100+a2P_001010000_2*QR_010010000101+a2P_001000010_2*QR_010010000110+a3P_001000000_2*QR_010010000111+a2P_000010010_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_002010010*QR_000020000000+a1P_002010000_1*QR_000020000001+a1P_002000010_1*QR_000020000010+a2P_002000000_1*QR_000020000011+a1P_001010010_2*QR_000020000100+a2P_001010000_2*QR_000020000101+a2P_001000010_2*QR_000020000110+a3P_001000000_2*QR_000020000111+a2P_000010010_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(P_002010010*QR_010000010000+a1P_002010000_1*QR_010000010001+a1P_002000010_1*QR_010000010010+a2P_002000000_1*QR_010000010011+a1P_001010010_2*QR_010000010100+a2P_001010000_2*QR_010000010101+a2P_001000010_2*QR_010000010110+a3P_001000000_2*QR_010000010111+a2P_000010010_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(P_002010010*QR_000010010000+a1P_002010000_1*QR_000010010001+a1P_002000010_1*QR_000010010010+a2P_002000000_1*QR_000010010011+a1P_001010010_2*QR_000010010100+a2P_001010000_2*QR_000010010101+a2P_001000010_2*QR_000010010110+a3P_001000000_2*QR_000010010111+a2P_000010010_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(P_002010010*QR_000000020000+a1P_002010000_1*QR_000000020001+a1P_002000010_1*QR_000000020010+a2P_002000000_1*QR_000000020011+a1P_001010010_2*QR_000000020100+a2P_001010000_2*QR_000000020101+a2P_001000010_2*QR_000000020110+a3P_001000000_2*QR_000000020111+a2P_000010010_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_001011010*QR_020000000000+a1P_001011000_1*QR_020000000001+P_001111010*QR_020000000010+a1P_001111000_1*QR_020000000011+a2P_001000010_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000011010_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111010_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_001011010*QR_010010000000+a1P_001011000_1*QR_010010000001+P_001111010*QR_010010000010+a1P_001111000_1*QR_010010000011+a2P_001000010_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000011010_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111010_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_001011010*QR_000020000000+a1P_001011000_1*QR_000020000001+P_001111010*QR_000020000010+a1P_001111000_1*QR_000020000011+a2P_001000010_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000011010_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111010_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(P_001011010*QR_010000010000+a1P_001011000_1*QR_010000010001+P_001111010*QR_010000010010+a1P_001111000_1*QR_010000010011+a2P_001000010_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000011010_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111010_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(P_001011010*QR_000010010000+a1P_001011000_1*QR_000010010001+P_001111010*QR_000010010010+a1P_001111000_1*QR_000010010011+a2P_001000010_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000011010_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111010_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(P_001011010*QR_000000020000+a1P_001011000_1*QR_000000020001+P_001111010*QR_000000020010+a1P_001111000_1*QR_000000020011+a2P_001000010_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000011010_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111010_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_000012010*QR_020000000000+a1P_000012000_1*QR_020000000001+P_000112010*QR_020000000010+a1P_000112000_1*QR_020000000011+P_000212010*QR_020000000020+a1P_000212000_1*QR_020000000021+a3P_000000010_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_000012010*QR_010010000000+a1P_000012000_1*QR_010010000001+P_000112010*QR_010010000010+a1P_000112000_1*QR_010010000011+P_000212010*QR_010010000020+a1P_000212000_1*QR_010010000021+a3P_000000010_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000012010*QR_000020000000+a1P_000012000_1*QR_000020000001+P_000112010*QR_000020000010+a1P_000112000_1*QR_000020000011+P_000212010*QR_000020000020+a1P_000212000_1*QR_000020000021+a3P_000000010_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(P_000012010*QR_010000010000+a1P_000012000_1*QR_010000010001+P_000112010*QR_010000010010+a1P_000112000_1*QR_010000010011+P_000212010*QR_010000010020+a1P_000212000_1*QR_010000010021+a3P_000000010_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(P_000012010*QR_000010010000+a1P_000012000_1*QR_000010010001+P_000112010*QR_000010010010+a1P_000112000_1*QR_000010010011+P_000212010*QR_000010010020+a1P_000212000_1*QR_000010010021+a3P_000000010_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(P_000012010*QR_000000020000+a1P_000012000_1*QR_000000020001+P_000112010*QR_000000020010+a1P_000112000_1*QR_000000020011+P_000212010*QR_000000020020+a1P_000212000_1*QR_000000020021+a3P_000000010_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_001010011*QR_020000000000+P_001010111*QR_020000000001+a2P_001010000_1*QR_020000000002+a1P_001000011_1*QR_020000000010+a1P_001000111_1*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000010011_1*QR_020000000100+a1P_000010111_1*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_001010011*QR_010010000000+P_001010111*QR_010010000001+a2P_001010000_1*QR_010010000002+a1P_001000011_1*QR_010010000010+a1P_001000111_1*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000010011_1*QR_010010000100+a1P_000010111_1*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_001010011*QR_000020000000+P_001010111*QR_000020000001+a2P_001010000_1*QR_000020000002+a1P_001000011_1*QR_000020000010+a1P_001000111_1*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000010011_1*QR_000020000100+a1P_000010111_1*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(P_001010011*QR_010000010000+P_001010111*QR_010000010001+a2P_001010000_1*QR_010000010002+a1P_001000011_1*QR_010000010010+a1P_001000111_1*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000010011_1*QR_010000010100+a1P_000010111_1*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(P_001010011*QR_000010010000+P_001010111*QR_000010010001+a2P_001010000_1*QR_000010010002+a1P_001000011_1*QR_000010010010+a1P_001000111_1*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000010011_1*QR_000010010100+a1P_000010111_1*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(P_001010011*QR_000000020000+P_001010111*QR_000000020001+a2P_001010000_1*QR_000000020002+a1P_001000011_1*QR_000000020010+a1P_001000111_1*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000010011_1*QR_000000020100+a1P_000010111_1*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_000011011*QR_020000000000+P_000011111*QR_020000000001+a2P_000011000_1*QR_020000000002+P_000111011*QR_020000000010+P_000111111*QR_020000000011+a2P_000111000_1*QR_020000000012+a2P_000000011_1*QR_020000000020+a2P_000000111_1*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_000011011*QR_010010000000+P_000011111*QR_010010000001+a2P_000011000_1*QR_010010000002+P_000111011*QR_010010000010+P_000111111*QR_010010000011+a2P_000111000_1*QR_010010000012+a2P_000000011_1*QR_010010000020+a2P_000000111_1*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000011011*QR_000020000000+P_000011111*QR_000020000001+a2P_000011000_1*QR_000020000002+P_000111011*QR_000020000010+P_000111111*QR_000020000011+a2P_000111000_1*QR_000020000012+a2P_000000011_1*QR_000020000020+a2P_000000111_1*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(P_000011011*QR_010000010000+P_000011111*QR_010000010001+a2P_000011000_1*QR_010000010002+P_000111011*QR_010000010010+P_000111111*QR_010000010011+a2P_000111000_1*QR_010000010012+a2P_000000011_1*QR_010000010020+a2P_000000111_1*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(P_000011011*QR_000010010000+P_000011111*QR_000010010001+a2P_000011000_1*QR_000010010002+P_000111011*QR_000010010010+P_000111111*QR_000010010011+a2P_000111000_1*QR_000010010012+a2P_000000011_1*QR_000010010020+a2P_000000111_1*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(P_000011011*QR_000000020000+P_000011111*QR_000000020001+a2P_000011000_1*QR_000000020002+P_000111011*QR_000000020010+P_000111111*QR_000000020011+a2P_000111000_1*QR_000000020012+a2P_000000011_1*QR_000000020020+a2P_000000111_1*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_000010012*QR_020000000000+P_000010112*QR_020000000001+P_000010212*QR_020000000002+a3P_000010000_1*QR_020000000003+a1P_000000012_1*QR_020000000010+a1P_000000112_1*QR_020000000011+a1P_000000212_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_000010012*QR_010010000000+P_000010112*QR_010010000001+P_000010212*QR_010010000002+a3P_000010000_1*QR_010010000003+a1P_000000012_1*QR_010010000010+a1P_000000112_1*QR_010010000011+a1P_000000212_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000010012*QR_000020000000+P_000010112*QR_000020000001+P_000010212*QR_000020000002+a3P_000010000_1*QR_000020000003+a1P_000000012_1*QR_000020000010+a1P_000000112_1*QR_000020000011+a1P_000000212_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(P_000010012*QR_010000010000+P_000010112*QR_010000010001+P_000010212*QR_010000010002+a3P_000010000_1*QR_010000010003+a1P_000000012_1*QR_010000010010+a1P_000000112_1*QR_010000010011+a1P_000000212_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(P_000010012*QR_000010010000+P_000010112*QR_000010010001+P_000010212*QR_000010010002+a3P_000010000_1*QR_000010010003+a1P_000000012_1*QR_000010010010+a1P_000000112_1*QR_000010010011+a1P_000000212_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(P_000010012*QR_000000020000+P_000010112*QR_000000020001+P_000010212*QR_000000020002+a3P_000010000_1*QR_000000020003+a1P_000000012_1*QR_000000020010+a1P_000000112_1*QR_000000020011+a1P_000000212_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_002000020*QR_020000000000+a1P_002000010_2*QR_020000000001+a2P_002000000_1*QR_020000000002+a1P_001000020_2*QR_020000000100+a2P_001000010_4*QR_020000000101+a3P_001000000_2*QR_020000000102+a2P_000000020_1*QR_020000000200+a3P_000000010_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_002000020*QR_010010000000+a1P_002000010_2*QR_010010000001+a2P_002000000_1*QR_010010000002+a1P_001000020_2*QR_010010000100+a2P_001000010_4*QR_010010000101+a3P_001000000_2*QR_010010000102+a2P_000000020_1*QR_010010000200+a3P_000000010_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_002000020*QR_000020000000+a1P_002000010_2*QR_000020000001+a2P_002000000_1*QR_000020000002+a1P_001000020_2*QR_000020000100+a2P_001000010_4*QR_000020000101+a3P_001000000_2*QR_000020000102+a2P_000000020_1*QR_000020000200+a3P_000000010_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(P_002000020*QR_010000010000+a1P_002000010_2*QR_010000010001+a2P_002000000_1*QR_010000010002+a1P_001000020_2*QR_010000010100+a2P_001000010_4*QR_010000010101+a3P_001000000_2*QR_010000010102+a2P_000000020_1*QR_010000010200+a3P_000000010_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(P_002000020*QR_000010010000+a1P_002000010_2*QR_000010010001+a2P_002000000_1*QR_000010010002+a1P_001000020_2*QR_000010010100+a2P_001000010_4*QR_000010010101+a3P_001000000_2*QR_000010010102+a2P_000000020_1*QR_000010010200+a3P_000000010_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(P_002000020*QR_000000020000+a1P_002000010_2*QR_000000020001+a2P_002000000_1*QR_000000020002+a1P_001000020_2*QR_000000020100+a2P_001000010_4*QR_000000020101+a3P_001000000_2*QR_000000020102+a2P_000000020_1*QR_000000020200+a3P_000000010_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_001001020*QR_020000000000+a1P_001001010_2*QR_020000000001+a2P_001001000_1*QR_020000000002+a1P_001000020_1*QR_020000000010+a2P_001000010_2*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000001020_1*QR_020000000100+a2P_000001010_2*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000020_1*QR_020000000110+a3P_000000010_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_001001020*QR_010010000000+a1P_001001010_2*QR_010010000001+a2P_001001000_1*QR_010010000002+a1P_001000020_1*QR_010010000010+a2P_001000010_2*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000001020_1*QR_010010000100+a2P_000001010_2*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000020_1*QR_010010000110+a3P_000000010_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_001001020*QR_000020000000+a1P_001001010_2*QR_000020000001+a2P_001001000_1*QR_000020000002+a1P_001000020_1*QR_000020000010+a2P_001000010_2*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000001020_1*QR_000020000100+a2P_000001010_2*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000020_1*QR_000020000110+a3P_000000010_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(P_001001020*QR_010000010000+a1P_001001010_2*QR_010000010001+a2P_001001000_1*QR_010000010002+a1P_001000020_1*QR_010000010010+a2P_001000010_2*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000001020_1*QR_010000010100+a2P_000001010_2*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000020_1*QR_010000010110+a3P_000000010_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(P_001001020*QR_000010010000+a1P_001001010_2*QR_000010010001+a2P_001001000_1*QR_000010010002+a1P_001000020_1*QR_000010010010+a2P_001000010_2*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000001020_1*QR_000010010100+a2P_000001010_2*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000020_1*QR_000010010110+a3P_000000010_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(P_001001020*QR_000000020000+a1P_001001010_2*QR_000000020001+a2P_001001000_1*QR_000000020002+a1P_001000020_1*QR_000000020010+a2P_001000010_2*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000001020_1*QR_000000020100+a2P_000001010_2*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000020_1*QR_000000020110+a3P_000000010_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_000002020*QR_020000000000+a1P_000002010_2*QR_020000000001+a2P_000002000_1*QR_020000000002+a1P_000001020_2*QR_020000000010+a2P_000001010_4*QR_020000000011+a3P_000001000_2*QR_020000000012+a2P_000000020_1*QR_020000000020+a3P_000000010_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_000002020*QR_010010000000+a1P_000002010_2*QR_010010000001+a2P_000002000_1*QR_010010000002+a1P_000001020_2*QR_010010000010+a2P_000001010_4*QR_010010000011+a3P_000001000_2*QR_010010000012+a2P_000000020_1*QR_010010000020+a3P_000000010_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000002020*QR_000020000000+a1P_000002010_2*QR_000020000001+a2P_000002000_1*QR_000020000002+a1P_000001020_2*QR_000020000010+a2P_000001010_4*QR_000020000011+a3P_000001000_2*QR_000020000012+a2P_000000020_1*QR_000020000020+a3P_000000010_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(P_000002020*QR_010000010000+a1P_000002010_2*QR_010000010001+a2P_000002000_1*QR_010000010002+a1P_000001020_2*QR_010000010010+a2P_000001010_4*QR_010000010011+a3P_000001000_2*QR_010000010012+a2P_000000020_1*QR_010000010020+a3P_000000010_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(P_000002020*QR_000010010000+a1P_000002010_2*QR_000010010001+a2P_000002000_1*QR_000010010002+a1P_000001020_2*QR_000010010010+a2P_000001010_4*QR_000010010011+a3P_000001000_2*QR_000010010012+a2P_000000020_1*QR_000010010020+a3P_000000010_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(P_000002020*QR_000000020000+a1P_000002010_2*QR_000000020001+a2P_000002000_1*QR_000000020002+a1P_000001020_2*QR_000000020010+a2P_000001010_4*QR_000000020011+a3P_000001000_2*QR_000000020012+a2P_000000020_1*QR_000000020020+a3P_000000010_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_001000021*QR_020000000000+P_001000121*QR_020000000001+P_001000221*QR_020000000002+a3P_001000000_1*QR_020000000003+a1P_000000021_1*QR_020000000100+a1P_000000121_1*QR_020000000101+a1P_000000221_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_001000021*QR_010010000000+P_001000121*QR_010010000001+P_001000221*QR_010010000002+a3P_001000000_1*QR_010010000003+a1P_000000021_1*QR_010010000100+a1P_000000121_1*QR_010010000101+a1P_000000221_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_001000021*QR_000020000000+P_001000121*QR_000020000001+P_001000221*QR_000020000002+a3P_001000000_1*QR_000020000003+a1P_000000021_1*QR_000020000100+a1P_000000121_1*QR_000020000101+a1P_000000221_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(P_001000021*QR_010000010000+P_001000121*QR_010000010001+P_001000221*QR_010000010002+a3P_001000000_1*QR_010000010003+a1P_000000021_1*QR_010000010100+a1P_000000121_1*QR_010000010101+a1P_000000221_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(P_001000021*QR_000010010000+P_001000121*QR_000010010001+P_001000221*QR_000010010002+a3P_001000000_1*QR_000010010003+a1P_000000021_1*QR_000010010100+a1P_000000121_1*QR_000010010101+a1P_000000221_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(P_001000021*QR_000000020000+P_001000121*QR_000000020001+P_001000221*QR_000000020002+a3P_001000000_1*QR_000000020003+a1P_000000021_1*QR_000000020100+a1P_000000121_1*QR_000000020101+a1P_000000221_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_000001021*QR_020000000000+P_000001121*QR_020000000001+P_000001221*QR_020000000002+a3P_000001000_1*QR_020000000003+a1P_000000021_1*QR_020000000010+a1P_000000121_1*QR_020000000011+a1P_000000221_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_000001021*QR_010010000000+P_000001121*QR_010010000001+P_000001221*QR_010010000002+a3P_000001000_1*QR_010010000003+a1P_000000021_1*QR_010010000010+a1P_000000121_1*QR_010010000011+a1P_000000221_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000001021*QR_000020000000+P_000001121*QR_000020000001+P_000001221*QR_000020000002+a3P_000001000_1*QR_000020000003+a1P_000000021_1*QR_000020000010+a1P_000000121_1*QR_000020000011+a1P_000000221_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(P_000001021*QR_010000010000+P_000001121*QR_010000010001+P_000001221*QR_010000010002+a3P_000001000_1*QR_010000010003+a1P_000000021_1*QR_010000010010+a1P_000000121_1*QR_010000010011+a1P_000000221_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(P_000001021*QR_000010010000+P_000001121*QR_000010010001+P_000001221*QR_000010010002+a3P_000001000_1*QR_000010010003+a1P_000000021_1*QR_000010010010+a1P_000000121_1*QR_000010010011+a1P_000000221_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(P_000001021*QR_000000020000+P_000001121*QR_000000020001+P_000001221*QR_000000020002+a3P_000001000_1*QR_000000020003+a1P_000000021_1*QR_000000020010+a1P_000000121_1*QR_000000020011+a1P_000000221_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_000000022*QR_020000000000+P_000000122*QR_020000000001+P_000000222*QR_020000000002+a2P_000000111_2*QR_020000000003+aPin4*QR_020000000004);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_000000022*QR_010010000000+P_000000122*QR_010010000001+P_000000222*QR_010010000002+a2P_000000111_2*QR_010010000003+aPin4*QR_010010000004);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000000022*QR_000020000000+P_000000122*QR_000020000001+P_000000222*QR_000020000002+a2P_000000111_2*QR_000020000003+aPin4*QR_000020000004);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(P_000000022*QR_010000010000+P_000000122*QR_010000010001+P_000000222*QR_010000010002+a2P_000000111_2*QR_010000010003+aPin4*QR_010000010004);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(P_000000022*QR_000010010000+P_000000122*QR_000010010001+P_000000222*QR_000010010002+a2P_000000111_2*QR_000010010003+aPin4*QR_000010010004);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(P_000000022*QR_000000020000+P_000000122*QR_000000020001+P_000000222*QR_000000020002+a2P_000000111_2*QR_000000020003+aPin4*QR_000000020004);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
}
__global__ void TSMJ_ddds_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * Q,\
                double * QC,\
                double * QD,\
                double * Eta_in,\
                double * pq_in,\
                float * K2_q_in,\
                double * Pmtrx_in,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double aPin4=aPin1*aPin3;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=K2_q_in[jj];
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
				double QX=Q[jj*3+0];
				double QY=Q[jj*3+1];
				double QZ=Q[jj*3+2];
				double Qd_010[3];
				Qd_010[0]=QC[jj*3+0];
				Qd_010[1]=QC[jj*3+1];
				Qd_010[2]=QC[jj*3+2];
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
			double QR_020000000004=Q_020000000*R_004[0]-a1Q_010000000_2*R_104[0]+aQin2*R_204[0];
			double QR_010010000004=Q_010010000*R_004[0]-a1Q_010000000_1*R_014[0]-a1Q_000010000_1*R_104[0]+aQin2*R_114[0];
			double QR_000020000004=Q_000020000*R_004[0]-a1Q_000010000_2*R_014[0]+aQin2*R_024[0];
			double QR_010000010004=Q_010000010*R_004[0]-a1Q_010000000_1*R_005[0]-a1Q_000000010_1*R_104[0]+aQin2*R_105[0];
			double QR_000010010004=Q_000010010*R_004[0]-a1Q_000010000_1*R_005[0]-a1Q_000000010_1*R_014[0]+aQin2*R_015[0];
			double QR_000000020004=Q_000000020*R_004[0]-a1Q_000000010_2*R_005[0]+aQin2*R_006[0];
			double QR_020000000013=Q_020000000*R_013[0]-a1Q_010000000_2*R_113[0]+aQin2*R_213[0];
			double QR_010010000013=Q_010010000*R_013[0]-a1Q_010000000_1*R_023[0]-a1Q_000010000_1*R_113[0]+aQin2*R_123[0];
			double QR_000020000013=Q_000020000*R_013[0]-a1Q_000010000_2*R_023[0]+aQin2*R_033[0];
			double QR_010000010013=Q_010000010*R_013[0]-a1Q_010000000_1*R_014[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000010010013=Q_000010010*R_013[0]-a1Q_000010000_1*R_014[0]-a1Q_000000010_1*R_023[0]+aQin2*R_024[0];
			double QR_000000020013=Q_000000020*R_013[0]-a1Q_000000010_2*R_014[0]+aQin2*R_015[0];
			double QR_020000000022=Q_020000000*R_022[0]-a1Q_010000000_2*R_122[0]+aQin2*R_222[0];
			double QR_010010000022=Q_010010000*R_022[0]-a1Q_010000000_1*R_032[0]-a1Q_000010000_1*R_122[0]+aQin2*R_132[0];
			double QR_000020000022=Q_000020000*R_022[0]-a1Q_000010000_2*R_032[0]+aQin2*R_042[0];
			double QR_010000010022=Q_010000010*R_022[0]-a1Q_010000000_1*R_023[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000010010022=Q_000010010*R_022[0]-a1Q_000010000_1*R_023[0]-a1Q_000000010_1*R_032[0]+aQin2*R_033[0];
			double QR_000000020022=Q_000000020*R_022[0]-a1Q_000000010_2*R_023[0]+aQin2*R_024[0];
			double QR_020000000031=Q_020000000*R_031[0]-a1Q_010000000_2*R_131[0]+aQin2*R_231[0];
			double QR_010010000031=Q_010010000*R_031[0]-a1Q_010000000_1*R_041[0]-a1Q_000010000_1*R_131[0]+aQin2*R_141[0];
			double QR_000020000031=Q_000020000*R_031[0]-a1Q_000010000_2*R_041[0]+aQin2*R_051[0];
			double QR_010000010031=Q_010000010*R_031[0]-a1Q_010000000_1*R_032[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000010010031=Q_000010010*R_031[0]-a1Q_000010000_1*R_032[0]-a1Q_000000010_1*R_041[0]+aQin2*R_042[0];
			double QR_000000020031=Q_000000020*R_031[0]-a1Q_000000010_2*R_032[0]+aQin2*R_033[0];
			double QR_020000000040=Q_020000000*R_040[0]-a1Q_010000000_2*R_140[0]+aQin2*R_240[0];
			double QR_010010000040=Q_010010000*R_040[0]-a1Q_010000000_1*R_050[0]-a1Q_000010000_1*R_140[0]+aQin2*R_150[0];
			double QR_000020000040=Q_000020000*R_040[0]-a1Q_000010000_2*R_050[0]+aQin2*R_060[0];
			double QR_010000010040=Q_010000010*R_040[0]-a1Q_010000000_1*R_041[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000010010040=Q_000010010*R_040[0]-a1Q_000010000_1*R_041[0]-a1Q_000000010_1*R_050[0]+aQin2*R_051[0];
			double QR_000000020040=Q_000000020*R_040[0]-a1Q_000000010_2*R_041[0]+aQin2*R_042[0];
			double QR_020000000103=Q_020000000*R_103[0]-a1Q_010000000_2*R_203[0]+aQin2*R_303[0];
			double QR_010010000103=Q_010010000*R_103[0]-a1Q_010000000_1*R_113[0]-a1Q_000010000_1*R_203[0]+aQin2*R_213[0];
			double QR_000020000103=Q_000020000*R_103[0]-a1Q_000010000_2*R_113[0]+aQin2*R_123[0];
			double QR_010000010103=Q_010000010*R_103[0]-a1Q_010000000_1*R_104[0]-a1Q_000000010_1*R_203[0]+aQin2*R_204[0];
			double QR_000010010103=Q_000010010*R_103[0]-a1Q_000010000_1*R_104[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000000020103=Q_000000020*R_103[0]-a1Q_000000010_2*R_104[0]+aQin2*R_105[0];
			double QR_020000000112=Q_020000000*R_112[0]-a1Q_010000000_2*R_212[0]+aQin2*R_312[0];
			double QR_010010000112=Q_010010000*R_112[0]-a1Q_010000000_1*R_122[0]-a1Q_000010000_1*R_212[0]+aQin2*R_222[0];
			double QR_000020000112=Q_000020000*R_112[0]-a1Q_000010000_2*R_122[0]+aQin2*R_132[0];
			double QR_010000010112=Q_010000010*R_112[0]-a1Q_010000000_1*R_113[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000010010112=Q_000010010*R_112[0]-a1Q_000010000_1*R_113[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000000020112=Q_000000020*R_112[0]-a1Q_000000010_2*R_113[0]+aQin2*R_114[0];
			double QR_020000000121=Q_020000000*R_121[0]-a1Q_010000000_2*R_221[0]+aQin2*R_321[0];
			double QR_010010000121=Q_010010000*R_121[0]-a1Q_010000000_1*R_131[0]-a1Q_000010000_1*R_221[0]+aQin2*R_231[0];
			double QR_000020000121=Q_000020000*R_121[0]-a1Q_000010000_2*R_131[0]+aQin2*R_141[0];
			double QR_010000010121=Q_010000010*R_121[0]-a1Q_010000000_1*R_122[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000010010121=Q_000010010*R_121[0]-a1Q_000010000_1*R_122[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000000020121=Q_000000020*R_121[0]-a1Q_000000010_2*R_122[0]+aQin2*R_123[0];
			double QR_020000000130=Q_020000000*R_130[0]-a1Q_010000000_2*R_230[0]+aQin2*R_330[0];
			double QR_010010000130=Q_010010000*R_130[0]-a1Q_010000000_1*R_140[0]-a1Q_000010000_1*R_230[0]+aQin2*R_240[0];
			double QR_000020000130=Q_000020000*R_130[0]-a1Q_000010000_2*R_140[0]+aQin2*R_150[0];
			double QR_010000010130=Q_010000010*R_130[0]-a1Q_010000000_1*R_131[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000010010130=Q_000010010*R_130[0]-a1Q_000010000_1*R_131[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000000020130=Q_000000020*R_130[0]-a1Q_000000010_2*R_131[0]+aQin2*R_132[0];
			double QR_020000000202=Q_020000000*R_202[0]-a1Q_010000000_2*R_302[0]+aQin2*R_402[0];
			double QR_010010000202=Q_010010000*R_202[0]-a1Q_010000000_1*R_212[0]-a1Q_000010000_1*R_302[0]+aQin2*R_312[0];
			double QR_000020000202=Q_000020000*R_202[0]-a1Q_000010000_2*R_212[0]+aQin2*R_222[0];
			double QR_010000010202=Q_010000010*R_202[0]-a1Q_010000000_1*R_203[0]-a1Q_000000010_1*R_302[0]+aQin2*R_303[0];
			double QR_000010010202=Q_000010010*R_202[0]-a1Q_000010000_1*R_203[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000000020202=Q_000000020*R_202[0]-a1Q_000000010_2*R_203[0]+aQin2*R_204[0];
			double QR_020000000211=Q_020000000*R_211[0]-a1Q_010000000_2*R_311[0]+aQin2*R_411[0];
			double QR_010010000211=Q_010010000*R_211[0]-a1Q_010000000_1*R_221[0]-a1Q_000010000_1*R_311[0]+aQin2*R_321[0];
			double QR_000020000211=Q_000020000*R_211[0]-a1Q_000010000_2*R_221[0]+aQin2*R_231[0];
			double QR_010000010211=Q_010000010*R_211[0]-a1Q_010000000_1*R_212[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000010010211=Q_000010010*R_211[0]-a1Q_000010000_1*R_212[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000000020211=Q_000000020*R_211[0]-a1Q_000000010_2*R_212[0]+aQin2*R_213[0];
			double QR_020000000220=Q_020000000*R_220[0]-a1Q_010000000_2*R_320[0]+aQin2*R_420[0];
			double QR_010010000220=Q_010010000*R_220[0]-a1Q_010000000_1*R_230[0]-a1Q_000010000_1*R_320[0]+aQin2*R_330[0];
			double QR_000020000220=Q_000020000*R_220[0]-a1Q_000010000_2*R_230[0]+aQin2*R_240[0];
			double QR_010000010220=Q_010000010*R_220[0]-a1Q_010000000_1*R_221[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000010010220=Q_000010010*R_220[0]-a1Q_000010000_1*R_221[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000000020220=Q_000000020*R_220[0]-a1Q_000000010_2*R_221[0]+aQin2*R_222[0];
			double QR_020000000301=Q_020000000*R_301[0]-a1Q_010000000_2*R_401[0]+aQin2*R_501[0];
			double QR_010010000301=Q_010010000*R_301[0]-a1Q_010000000_1*R_311[0]-a1Q_000010000_1*R_401[0]+aQin2*R_411[0];
			double QR_000020000301=Q_000020000*R_301[0]-a1Q_000010000_2*R_311[0]+aQin2*R_321[0];
			double QR_010000010301=Q_010000010*R_301[0]-a1Q_010000000_1*R_302[0]-a1Q_000000010_1*R_401[0]+aQin2*R_402[0];
			double QR_000010010301=Q_000010010*R_301[0]-a1Q_000010000_1*R_302[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000000020301=Q_000000020*R_301[0]-a1Q_000000010_2*R_302[0]+aQin2*R_303[0];
			double QR_020000000310=Q_020000000*R_310[0]-a1Q_010000000_2*R_410[0]+aQin2*R_510[0];
			double QR_010010000310=Q_010010000*R_310[0]-a1Q_010000000_1*R_320[0]-a1Q_000010000_1*R_410[0]+aQin2*R_420[0];
			double QR_000020000310=Q_000020000*R_310[0]-a1Q_000010000_2*R_320[0]+aQin2*R_330[0];
			double QR_010000010310=Q_010000010*R_310[0]-a1Q_010000000_1*R_311[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000010010310=Q_000010010*R_310[0]-a1Q_000010000_1*R_311[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000000020310=Q_000000020*R_310[0]-a1Q_000000010_2*R_311[0]+aQin2*R_312[0];
			double QR_020000000400=Q_020000000*R_400[0]-a1Q_010000000_2*R_500[0]+aQin2*R_600[0];
			double QR_010010000400=Q_010010000*R_400[0]-a1Q_010000000_1*R_410[0]-a1Q_000010000_1*R_500[0]+aQin2*R_510[0];
			double QR_000020000400=Q_000020000*R_400[0]-a1Q_000010000_2*R_410[0]+aQin2*R_420[0];
			double QR_010000010400=Q_010000010*R_400[0]-a1Q_010000000_1*R_401[0]-a1Q_000000010_1*R_500[0]+aQin2*R_501[0];
			double QR_000010010400=Q_000010010*R_400[0]-a1Q_000010000_1*R_401[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000000020400=Q_000000020*R_400[0]-a1Q_000000010_2*R_401[0]+aQin2*R_402[0];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		for(int i=0;i<3;i++){
			Pd_002[i]=aPin1+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=aPin1*(2.000000*Pd_001[i]);
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=aPin1*(Pd_002[i]+2.000000*Pd_011[i]);
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=aPin1*(0.500000*Pd_102[i]+Pd_111[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=aPin1*2.000000*(Pd_012[i]+Pd_021[i]);
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=aPin1*(Pd_112[i]+Pd_121[i]);
			}
			double P_022000000;
			double P_122000000;
			double P_222000000;
			double P_021001000;
			double P_121001000;
			double P_221001000;
			double P_020002000;
			double P_021000001;
			double P_121000001;
			double P_221000001;
			double P_020001001;
			double P_020000002;
			double P_012010000;
			double P_112010000;
			double P_212010000;
			double P_011011000;
			double P_011111000;
			double P_111011000;
			double P_111111000;
			double P_010012000;
			double P_010112000;
			double P_010212000;
			double P_011010001;
			double P_111010001;
			double P_010011001;
			double P_010111001;
			double P_010010002;
			double P_002020000;
			double P_001021000;
			double P_001121000;
			double P_001221000;
			double P_000022000;
			double P_000122000;
			double P_000222000;
			double P_001020001;
			double P_000021001;
			double P_000121001;
			double P_000221001;
			double P_000020002;
			double P_012000010;
			double P_112000010;
			double P_212000010;
			double P_011001010;
			double P_111001010;
			double P_010002010;
			double P_011000011;
			double P_011000111;
			double P_111000011;
			double P_111000111;
			double P_010001011;
			double P_010001111;
			double P_010000012;
			double P_010000112;
			double P_010000212;
			double P_002010010;
			double P_001011010;
			double P_001111010;
			double P_000012010;
			double P_000112010;
			double P_000212010;
			double P_001010011;
			double P_001010111;
			double P_000011011;
			double P_000011111;
			double P_000111011;
			double P_000111111;
			double P_000010012;
			double P_000010112;
			double P_000010212;
			double P_002000020;
			double P_001001020;
			double P_000002020;
			double P_001000021;
			double P_001000121;
			double P_001000221;
			double P_000001021;
			double P_000001121;
			double P_000001221;
			double P_000000022;
			double P_000000122;
			double P_000000222;
			double a2P_111000000_1;
			double a2P_111000000_2;
			double a1P_021000000_1;
			double a1P_121000000_1;
			double a1P_221000000_1;
			double a3P_000001000_1;
			double a3P_000001000_2;
			double a1P_020001000_1;
			double a1P_020001000_2;
			double a2P_020000000_1;
			double a1P_010002000_1;
			double a1P_010002000_2;
			double a2P_010001000_1;
			double a2P_010001000_4;
			double a2P_010001000_2;
			double a3P_010000000_1;
			double a3P_010000000_2;
			double a2P_000002000_1;
			double a3P_000000001_1;
			double a3P_000000001_2;
			double a1P_020000001_1;
			double a1P_020000001_2;
			double a1P_010001001_1;
			double a1P_010001001_2;
			double a2P_010000001_1;
			double a2P_010000001_2;
			double a2P_010000001_4;
			double a2P_000001001_1;
			double a1P_010000002_1;
			double a1P_010000002_2;
			double a2P_000000002_1;
			double a1P_012000000_1;
			double a1P_112000000_1;
			double a1P_212000000_1;
			double a3P_000010000_1;
			double a3P_000010000_2;
			double a2P_011000000_1;
			double a2P_000011000_1;
			double a2P_000111000_1;
			double a2P_000111000_2;
			double a1P_000012000_1;
			double a1P_000112000_1;
			double a1P_000212000_1;
			double a1P_011010000_1;
			double a1P_011000001_1;
			double a1P_111010000_1;
			double a1P_111000001_1;
			double a2P_000010001_1;
			double a2P_000010001_2;
			double a2P_000010001_4;
			double a1P_010011000_1;
			double a1P_010111000_1;
			double a1P_000011001_1;
			double a1P_000111001_1;
			double a1P_010010001_1;
			double a1P_010010001_2;
			double a2P_010010000_1;
			double a1P_000010002_1;
			double a1P_000010002_2;
			double a1P_002010000_1;
			double a1P_002010000_2;
			double a2P_002000000_1;
			double a1P_001020000_1;
			double a1P_001020000_2;
			double a2P_001010000_1;
			double a2P_001010000_4;
			double a2P_001010000_2;
			double a3P_001000000_1;
			double a3P_001000000_2;
			double a2P_000020000_1;
			double a1P_000021000_1;
			double a1P_000121000_1;
			double a1P_000221000_1;
			double a1P_001010001_1;
			double a1P_001010001_2;
			double a2P_001000001_1;
			double a1P_000020001_1;
			double a1P_000020001_2;
			double a3P_000000010_1;
			double a3P_000000010_2;
			double a1P_011001000_1;
			double a1P_011000010_1;
			double a1P_111001000_1;
			double a1P_111000010_1;
			double a2P_000001010_1;
			double a2P_000001010_2;
			double a2P_000001010_4;
			double a1P_010001010_1;
			double a1P_010001010_2;
			double a2P_010000010_1;
			double a1P_000002010_1;
			double a1P_000002010_2;
			double a2P_000000011_1;
			double a2P_000000111_1;
			double a2P_000000111_2;
			double a1P_010000011_1;
			double a1P_010000111_1;
			double a1P_000001011_1;
			double a1P_000001111_1;
			double a1P_000000012_1;
			double a1P_000000112_1;
			double a1P_000000212_1;
			double a1P_002000010_1;
			double a1P_002000010_2;
			double a1P_001010010_1;
			double a1P_001010010_2;
			double a2P_001000010_1;
			double a2P_001000010_2;
			double a2P_001000010_4;
			double a2P_000010010_1;
			double a1P_001011000_1;
			double a1P_001111000_1;
			double a1P_000011010_1;
			double a1P_000111010_1;
			double a1P_001000011_1;
			double a1P_001000111_1;
			double a1P_000010011_1;
			double a1P_000010111_1;
			double a1P_001000020_1;
			double a1P_001000020_2;
			double a2P_000000020_1;
			double a1P_001001010_1;
			double a1P_001001010_2;
			double a2P_001001000_1;
			double a1P_000001020_1;
			double a1P_000001020_2;
			double a1P_000000021_1;
			double a1P_000000121_1;
			double a1P_000000221_1;
			P_022000000=Pd_022[0];
			P_122000000=Pd_122[0];
			P_222000000=Pd_222[0];
			P_021001000=Pd_021[0]*Pd_001[1];
			P_121001000=Pd_121[0]*Pd_001[1];
			P_221001000=Pd_221[0]*Pd_001[1];
			P_020002000=Pd_020[0]*Pd_002[1];
			P_021000001=Pd_021[0]*Pd_001[2];
			P_121000001=Pd_121[0]*Pd_001[2];
			P_221000001=Pd_221[0]*Pd_001[2];
			P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
			P_020000002=Pd_020[0]*Pd_002[2];
			P_012010000=Pd_012[0]*Pd_010[1];
			P_112010000=Pd_112[0]*Pd_010[1];
			P_212010000=Pd_212[0]*Pd_010[1];
			P_011011000=Pd_011[0]*Pd_011[1];
			P_011111000=Pd_011[0]*Pd_111[1];
			P_111011000=Pd_111[0]*Pd_011[1];
			P_111111000=Pd_111[0]*Pd_111[1];
			P_010012000=Pd_010[0]*Pd_012[1];
			P_010112000=Pd_010[0]*Pd_112[1];
			P_010212000=Pd_010[0]*Pd_212[1];
			P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
			P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
			P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
			P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
			P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
			P_002020000=Pd_002[0]*Pd_020[1];
			P_001021000=Pd_001[0]*Pd_021[1];
			P_001121000=Pd_001[0]*Pd_121[1];
			P_001221000=Pd_001[0]*Pd_221[1];
			P_000022000=Pd_022[1];
			P_000122000=Pd_122[1];
			P_000222000=Pd_222[1];
			P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
			P_000021001=Pd_021[1]*Pd_001[2];
			P_000121001=Pd_121[1]*Pd_001[2];
			P_000221001=Pd_221[1]*Pd_001[2];
			P_000020002=Pd_020[1]*Pd_002[2];
			P_012000010=Pd_012[0]*Pd_010[2];
			P_112000010=Pd_112[0]*Pd_010[2];
			P_212000010=Pd_212[0]*Pd_010[2];
			P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
			P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
			P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
			P_011000011=Pd_011[0]*Pd_011[2];
			P_011000111=Pd_011[0]*Pd_111[2];
			P_111000011=Pd_111[0]*Pd_011[2];
			P_111000111=Pd_111[0]*Pd_111[2];
			P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
			P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
			P_010000012=Pd_010[0]*Pd_012[2];
			P_010000112=Pd_010[0]*Pd_112[2];
			P_010000212=Pd_010[0]*Pd_212[2];
			P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
			P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
			P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
			P_000012010=Pd_012[1]*Pd_010[2];
			P_000112010=Pd_112[1]*Pd_010[2];
			P_000212010=Pd_212[1]*Pd_010[2];
			P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
			P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
			P_000011011=Pd_011[1]*Pd_011[2];
			P_000011111=Pd_011[1]*Pd_111[2];
			P_000111011=Pd_111[1]*Pd_011[2];
			P_000111111=Pd_111[1]*Pd_111[2];
			P_000010012=Pd_010[1]*Pd_012[2];
			P_000010112=Pd_010[1]*Pd_112[2];
			P_000010212=Pd_010[1]*Pd_212[2];
			P_002000020=Pd_002[0]*Pd_020[2];
			P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
			P_000002020=Pd_002[1]*Pd_020[2];
			P_001000021=Pd_001[0]*Pd_021[2];
			P_001000121=Pd_001[0]*Pd_121[2];
			P_001000221=Pd_001[0]*Pd_221[2];
			P_000001021=Pd_001[1]*Pd_021[2];
			P_000001121=Pd_001[1]*Pd_121[2];
			P_000001221=Pd_001[1]*Pd_221[2];
			P_000000022=Pd_022[2];
			P_000000122=Pd_122[2];
			P_000000222=Pd_222[2];
			a2P_111000000_1=Pd_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=Pd_021[0];
			a1P_121000000_1=Pd_121[0];
			a1P_221000000_1=Pd_221[0];
			a3P_000001000_1=Pd_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=Pd_020[0]*Pd_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=Pd_020[0];
			a1P_010002000_1=Pd_010[0]*Pd_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=Pd_010[0]*Pd_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=Pd_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=Pd_002[1];
			a3P_000000001_1=Pd_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=Pd_020[0]*Pd_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=Pd_010[0]*Pd_001[1]*Pd_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=Pd_010[0]*Pd_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=Pd_001[1]*Pd_001[2];
			a1P_010000002_1=Pd_010[0]*Pd_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=Pd_002[2];
			a1P_012000000_1=Pd_012[0];
			a1P_112000000_1=Pd_112[0];
			a1P_212000000_1=Pd_212[0];
			a3P_000010000_1=Pd_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=Pd_011[0];
			a2P_000011000_1=Pd_011[1];
			a2P_000111000_1=Pd_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=Pd_012[1];
			a1P_000112000_1=Pd_112[1];
			a1P_000212000_1=Pd_212[1];
			a1P_011010000_1=Pd_011[0]*Pd_010[1];
			a1P_011000001_1=Pd_011[0]*Pd_001[2];
			a1P_111010000_1=Pd_111[0]*Pd_010[1];
			a1P_111000001_1=Pd_111[0]*Pd_001[2];
			a2P_000010001_1=Pd_010[1]*Pd_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=Pd_010[0]*Pd_011[1];
			a1P_010111000_1=Pd_010[0]*Pd_111[1];
			a1P_000011001_1=Pd_011[1]*Pd_001[2];
			a1P_000111001_1=Pd_111[1]*Pd_001[2];
			a1P_010010001_1=Pd_010[0]*Pd_010[1]*Pd_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010002_1=Pd_010[1]*Pd_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=Pd_002[0]*Pd_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=Pd_002[0];
			a1P_001020000_1=Pd_001[0]*Pd_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=Pd_001[0]*Pd_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=Pd_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=Pd_020[1];
			a1P_000021000_1=Pd_021[1];
			a1P_000121000_1=Pd_121[1];
			a1P_000221000_1=Pd_221[1];
			a1P_001010001_1=Pd_001[0]*Pd_010[1]*Pd_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=Pd_001[0]*Pd_001[2];
			a1P_000020001_1=Pd_020[1]*Pd_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=Pd_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=Pd_011[0]*Pd_001[1];
			a1P_011000010_1=Pd_011[0]*Pd_010[2];
			a1P_111001000_1=Pd_111[0]*Pd_001[1];
			a1P_111000010_1=Pd_111[0]*Pd_010[2];
			a2P_000001010_1=Pd_001[1]*Pd_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=Pd_010[0]*Pd_001[1]*Pd_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000002010_1=Pd_002[1]*Pd_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=Pd_011[2];
			a2P_000000111_1=Pd_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=Pd_010[0]*Pd_011[2];
			a1P_010000111_1=Pd_010[0]*Pd_111[2];
			a1P_000001011_1=Pd_001[1]*Pd_011[2];
			a1P_000001111_1=Pd_001[1]*Pd_111[2];
			a1P_000000012_1=Pd_012[2];
			a1P_000000112_1=Pd_112[2];
			a1P_000000212_1=Pd_212[2];
			a1P_002000010_1=Pd_002[0]*Pd_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=Pd_001[0]*Pd_010[1]*Pd_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=Pd_001[0]*Pd_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_001011000_1=Pd_001[0]*Pd_011[1];
			a1P_001111000_1=Pd_001[0]*Pd_111[1];
			a1P_000011010_1=Pd_011[1]*Pd_010[2];
			a1P_000111010_1=Pd_111[1]*Pd_010[2];
			a1P_001000011_1=Pd_001[0]*Pd_011[2];
			a1P_001000111_1=Pd_001[0]*Pd_111[2];
			a1P_000010011_1=Pd_010[1]*Pd_011[2];
			a1P_000010111_1=Pd_010[1]*Pd_111[2];
			a1P_001000020_1=Pd_001[0]*Pd_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=Pd_020[2];
			a1P_001001010_1=Pd_001[0]*Pd_001[1]*Pd_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=Pd_001[0]*Pd_001[1];
			a1P_000001020_1=Pd_001[1]*Pd_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=Pd_021[2];
			a1P_000000121_1=Pd_121[2];
			a1P_000000221_1=Pd_221[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_022000000*QR_020000000000+P_122000000*QR_020000000100+P_222000000*QR_020000000200+a2P_111000000_2*QR_020000000300+aPin4*QR_020000000400);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_022000000*QR_010010000000+P_122000000*QR_010010000100+P_222000000*QR_010010000200+a2P_111000000_2*QR_010010000300+aPin4*QR_010010000400);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_022000000*QR_000020000000+P_122000000*QR_000020000100+P_222000000*QR_000020000200+a2P_111000000_2*QR_000020000300+aPin4*QR_000020000400);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(P_022000000*QR_010000010000+P_122000000*QR_010000010100+P_222000000*QR_010000010200+a2P_111000000_2*QR_010000010300+aPin4*QR_010000010400);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(P_022000000*QR_000010010000+P_122000000*QR_000010010100+P_222000000*QR_000010010200+a2P_111000000_2*QR_000010010300+aPin4*QR_000010010400);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(P_022000000*QR_000000020000+P_122000000*QR_000000020100+P_222000000*QR_000000020200+a2P_111000000_2*QR_000000020300+aPin4*QR_000000020400);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_021001000*QR_020000000000+a1P_021000000_1*QR_020000000010+P_121001000*QR_020000000100+a1P_121000000_1*QR_020000000110+P_221001000*QR_020000000200+a1P_221000000_1*QR_020000000210+a3P_000001000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_021001000*QR_010010000000+a1P_021000000_1*QR_010010000010+P_121001000*QR_010010000100+a1P_121000000_1*QR_010010000110+P_221001000*QR_010010000200+a1P_221000000_1*QR_010010000210+a3P_000001000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_021001000*QR_000020000000+a1P_021000000_1*QR_000020000010+P_121001000*QR_000020000100+a1P_121000000_1*QR_000020000110+P_221001000*QR_000020000200+a1P_221000000_1*QR_000020000210+a3P_000001000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(P_021001000*QR_010000010000+a1P_021000000_1*QR_010000010010+P_121001000*QR_010000010100+a1P_121000000_1*QR_010000010110+P_221001000*QR_010000010200+a1P_221000000_1*QR_010000010210+a3P_000001000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(P_021001000*QR_000010010000+a1P_021000000_1*QR_000010010010+P_121001000*QR_000010010100+a1P_121000000_1*QR_000010010110+P_221001000*QR_000010010200+a1P_221000000_1*QR_000010010210+a3P_000001000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(P_021001000*QR_000000020000+a1P_021000000_1*QR_000000020010+P_121001000*QR_000000020100+a1P_121000000_1*QR_000000020110+P_221001000*QR_000000020200+a1P_221000000_1*QR_000000020210+a3P_000001000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_020002000*QR_020000000000+a1P_020001000_2*QR_020000000010+a2P_020000000_1*QR_020000000020+a1P_010002000_2*QR_020000000100+a2P_010001000_4*QR_020000000110+a3P_010000000_2*QR_020000000120+a2P_000002000_1*QR_020000000200+a3P_000001000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_020002000*QR_010010000000+a1P_020001000_2*QR_010010000010+a2P_020000000_1*QR_010010000020+a1P_010002000_2*QR_010010000100+a2P_010001000_4*QR_010010000110+a3P_010000000_2*QR_010010000120+a2P_000002000_1*QR_010010000200+a3P_000001000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020002000*QR_000020000000+a1P_020001000_2*QR_000020000010+a2P_020000000_1*QR_000020000020+a1P_010002000_2*QR_000020000100+a2P_010001000_4*QR_000020000110+a3P_010000000_2*QR_000020000120+a2P_000002000_1*QR_000020000200+a3P_000001000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(P_020002000*QR_010000010000+a1P_020001000_2*QR_010000010010+a2P_020000000_1*QR_010000010020+a1P_010002000_2*QR_010000010100+a2P_010001000_4*QR_010000010110+a3P_010000000_2*QR_010000010120+a2P_000002000_1*QR_010000010200+a3P_000001000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(P_020002000*QR_000010010000+a1P_020001000_2*QR_000010010010+a2P_020000000_1*QR_000010010020+a1P_010002000_2*QR_000010010100+a2P_010001000_4*QR_000010010110+a3P_010000000_2*QR_000010010120+a2P_000002000_1*QR_000010010200+a3P_000001000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(P_020002000*QR_000000020000+a1P_020001000_2*QR_000000020010+a2P_020000000_1*QR_000000020020+a1P_010002000_2*QR_000000020100+a2P_010001000_4*QR_000000020110+a3P_010000000_2*QR_000000020120+a2P_000002000_1*QR_000000020200+a3P_000001000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_021000001*QR_020000000000+a1P_021000000_1*QR_020000000001+P_121000001*QR_020000000100+a1P_121000000_1*QR_020000000101+P_221000001*QR_020000000200+a1P_221000000_1*QR_020000000201+a3P_000000001_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_021000001*QR_010010000000+a1P_021000000_1*QR_010010000001+P_121000001*QR_010010000100+a1P_121000000_1*QR_010010000101+P_221000001*QR_010010000200+a1P_221000000_1*QR_010010000201+a3P_000000001_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_021000001*QR_000020000000+a1P_021000000_1*QR_000020000001+P_121000001*QR_000020000100+a1P_121000000_1*QR_000020000101+P_221000001*QR_000020000200+a1P_221000000_1*QR_000020000201+a3P_000000001_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(P_021000001*QR_010000010000+a1P_021000000_1*QR_010000010001+P_121000001*QR_010000010100+a1P_121000000_1*QR_010000010101+P_221000001*QR_010000010200+a1P_221000000_1*QR_010000010201+a3P_000000001_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(P_021000001*QR_000010010000+a1P_021000000_1*QR_000010010001+P_121000001*QR_000010010100+a1P_121000000_1*QR_000010010101+P_221000001*QR_000010010200+a1P_221000000_1*QR_000010010201+a3P_000000001_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(P_021000001*QR_000000020000+a1P_021000000_1*QR_000000020001+P_121000001*QR_000000020100+a1P_121000000_1*QR_000000020101+P_221000001*QR_000000020200+a1P_221000000_1*QR_000000020201+a3P_000000001_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_020001001*QR_020000000000+a1P_020001000_1*QR_020000000001+a1P_020000001_1*QR_020000000010+a2P_020000000_1*QR_020000000011+a1P_010001001_2*QR_020000000100+a2P_010001000_2*QR_020000000101+a2P_010000001_2*QR_020000000110+a3P_010000000_2*QR_020000000111+a2P_000001001_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_020001001*QR_010010000000+a1P_020001000_1*QR_010010000001+a1P_020000001_1*QR_010010000010+a2P_020000000_1*QR_010010000011+a1P_010001001_2*QR_010010000100+a2P_010001000_2*QR_010010000101+a2P_010000001_2*QR_010010000110+a3P_010000000_2*QR_010010000111+a2P_000001001_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020001001*QR_000020000000+a1P_020001000_1*QR_000020000001+a1P_020000001_1*QR_000020000010+a2P_020000000_1*QR_000020000011+a1P_010001001_2*QR_000020000100+a2P_010001000_2*QR_000020000101+a2P_010000001_2*QR_000020000110+a3P_010000000_2*QR_000020000111+a2P_000001001_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(P_020001001*QR_010000010000+a1P_020001000_1*QR_010000010001+a1P_020000001_1*QR_010000010010+a2P_020000000_1*QR_010000010011+a1P_010001001_2*QR_010000010100+a2P_010001000_2*QR_010000010101+a2P_010000001_2*QR_010000010110+a3P_010000000_2*QR_010000010111+a2P_000001001_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(P_020001001*QR_000010010000+a1P_020001000_1*QR_000010010001+a1P_020000001_1*QR_000010010010+a2P_020000000_1*QR_000010010011+a1P_010001001_2*QR_000010010100+a2P_010001000_2*QR_000010010101+a2P_010000001_2*QR_000010010110+a3P_010000000_2*QR_000010010111+a2P_000001001_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(P_020001001*QR_000000020000+a1P_020001000_1*QR_000000020001+a1P_020000001_1*QR_000000020010+a2P_020000000_1*QR_000000020011+a1P_010001001_2*QR_000000020100+a2P_010001000_2*QR_000000020101+a2P_010000001_2*QR_000000020110+a3P_010000000_2*QR_000000020111+a2P_000001001_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_020000002*QR_020000000000+a1P_020000001_2*QR_020000000001+a2P_020000000_1*QR_020000000002+a1P_010000002_2*QR_020000000100+a2P_010000001_4*QR_020000000101+a3P_010000000_2*QR_020000000102+a2P_000000002_1*QR_020000000200+a3P_000000001_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_020000002*QR_010010000000+a1P_020000001_2*QR_010010000001+a2P_020000000_1*QR_010010000002+a1P_010000002_2*QR_010010000100+a2P_010000001_4*QR_010010000101+a3P_010000000_2*QR_010010000102+a2P_000000002_1*QR_010010000200+a3P_000000001_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020000002*QR_000020000000+a1P_020000001_2*QR_000020000001+a2P_020000000_1*QR_000020000002+a1P_010000002_2*QR_000020000100+a2P_010000001_4*QR_000020000101+a3P_010000000_2*QR_000020000102+a2P_000000002_1*QR_000020000200+a3P_000000001_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(P_020000002*QR_010000010000+a1P_020000001_2*QR_010000010001+a2P_020000000_1*QR_010000010002+a1P_010000002_2*QR_010000010100+a2P_010000001_4*QR_010000010101+a3P_010000000_2*QR_010000010102+a2P_000000002_1*QR_010000010200+a3P_000000001_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(P_020000002*QR_000010010000+a1P_020000001_2*QR_000010010001+a2P_020000000_1*QR_000010010002+a1P_010000002_2*QR_000010010100+a2P_010000001_4*QR_000010010101+a3P_010000000_2*QR_000010010102+a2P_000000002_1*QR_000010010200+a3P_000000001_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(P_020000002*QR_000000020000+a1P_020000001_2*QR_000000020001+a2P_020000000_1*QR_000000020002+a1P_010000002_2*QR_000000020100+a2P_010000001_4*QR_000000020101+a3P_010000000_2*QR_000000020102+a2P_000000002_1*QR_000000020200+a3P_000000001_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_012010000*QR_020000000000+a1P_012000000_1*QR_020000000010+P_112010000*QR_020000000100+a1P_112000000_1*QR_020000000110+P_212010000*QR_020000000200+a1P_212000000_1*QR_020000000210+a3P_000010000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_012010000*QR_010010000000+a1P_012000000_1*QR_010010000010+P_112010000*QR_010010000100+a1P_112000000_1*QR_010010000110+P_212010000*QR_010010000200+a1P_212000000_1*QR_010010000210+a3P_000010000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_012010000*QR_000020000000+a1P_012000000_1*QR_000020000010+P_112010000*QR_000020000100+a1P_112000000_1*QR_000020000110+P_212010000*QR_000020000200+a1P_212000000_1*QR_000020000210+a3P_000010000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(P_012010000*QR_010000010000+a1P_012000000_1*QR_010000010010+P_112010000*QR_010000010100+a1P_112000000_1*QR_010000010110+P_212010000*QR_010000010200+a1P_212000000_1*QR_010000010210+a3P_000010000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(P_012010000*QR_000010010000+a1P_012000000_1*QR_000010010010+P_112010000*QR_000010010100+a1P_112000000_1*QR_000010010110+P_212010000*QR_000010010200+a1P_212000000_1*QR_000010010210+a3P_000010000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(P_012010000*QR_000000020000+a1P_012000000_1*QR_000000020010+P_112010000*QR_000000020100+a1P_112000000_1*QR_000000020110+P_212010000*QR_000000020200+a1P_212000000_1*QR_000000020210+a3P_000010000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_011011000*QR_020000000000+P_011111000*QR_020000000010+a2P_011000000_1*QR_020000000020+P_111011000*QR_020000000100+P_111111000*QR_020000000110+a2P_111000000_1*QR_020000000120+a2P_000011000_1*QR_020000000200+a2P_000111000_1*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_011011000*QR_010010000000+P_011111000*QR_010010000010+a2P_011000000_1*QR_010010000020+P_111011000*QR_010010000100+P_111111000*QR_010010000110+a2P_111000000_1*QR_010010000120+a2P_000011000_1*QR_010010000200+a2P_000111000_1*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_011011000*QR_000020000000+P_011111000*QR_000020000010+a2P_011000000_1*QR_000020000020+P_111011000*QR_000020000100+P_111111000*QR_000020000110+a2P_111000000_1*QR_000020000120+a2P_000011000_1*QR_000020000200+a2P_000111000_1*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(P_011011000*QR_010000010000+P_011111000*QR_010000010010+a2P_011000000_1*QR_010000010020+P_111011000*QR_010000010100+P_111111000*QR_010000010110+a2P_111000000_1*QR_010000010120+a2P_000011000_1*QR_010000010200+a2P_000111000_1*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(P_011011000*QR_000010010000+P_011111000*QR_000010010010+a2P_011000000_1*QR_000010010020+P_111011000*QR_000010010100+P_111111000*QR_000010010110+a2P_111000000_1*QR_000010010120+a2P_000011000_1*QR_000010010200+a2P_000111000_1*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(P_011011000*QR_000000020000+P_011111000*QR_000000020010+a2P_011000000_1*QR_000000020020+P_111011000*QR_000000020100+P_111111000*QR_000000020110+a2P_111000000_1*QR_000000020120+a2P_000011000_1*QR_000000020200+a2P_000111000_1*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_010012000*QR_020000000000+P_010112000*QR_020000000010+P_010212000*QR_020000000020+a3P_010000000_1*QR_020000000030+a1P_000012000_1*QR_020000000100+a1P_000112000_1*QR_020000000110+a1P_000212000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_010012000*QR_010010000000+P_010112000*QR_010010000010+P_010212000*QR_010010000020+a3P_010000000_1*QR_010010000030+a1P_000012000_1*QR_010010000100+a1P_000112000_1*QR_010010000110+a1P_000212000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010012000*QR_000020000000+P_010112000*QR_000020000010+P_010212000*QR_000020000020+a3P_010000000_1*QR_000020000030+a1P_000012000_1*QR_000020000100+a1P_000112000_1*QR_000020000110+a1P_000212000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(P_010012000*QR_010000010000+P_010112000*QR_010000010010+P_010212000*QR_010000010020+a3P_010000000_1*QR_010000010030+a1P_000012000_1*QR_010000010100+a1P_000112000_1*QR_010000010110+a1P_000212000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(P_010012000*QR_000010010000+P_010112000*QR_000010010010+P_010212000*QR_000010010020+a3P_010000000_1*QR_000010010030+a1P_000012000_1*QR_000010010100+a1P_000112000_1*QR_000010010110+a1P_000212000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(P_010012000*QR_000000020000+P_010112000*QR_000000020010+P_010212000*QR_000000020020+a3P_010000000_1*QR_000000020030+a1P_000012000_1*QR_000000020100+a1P_000112000_1*QR_000000020110+a1P_000212000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_011010001*QR_020000000000+a1P_011010000_1*QR_020000000001+a1P_011000001_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111010001*QR_020000000100+a1P_111010000_1*QR_020000000101+a1P_111000001_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000010001_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_011010001*QR_010010000000+a1P_011010000_1*QR_010010000001+a1P_011000001_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111010001*QR_010010000100+a1P_111010000_1*QR_010010000101+a1P_111000001_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000010001_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_011010001*QR_000020000000+a1P_011010000_1*QR_000020000001+a1P_011000001_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111010001*QR_000020000100+a1P_111010000_1*QR_000020000101+a1P_111000001_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000010001_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(P_011010001*QR_010000010000+a1P_011010000_1*QR_010000010001+a1P_011000001_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111010001*QR_010000010100+a1P_111010000_1*QR_010000010101+a1P_111000001_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000010001_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(P_011010001*QR_000010010000+a1P_011010000_1*QR_000010010001+a1P_011000001_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111010001*QR_000010010100+a1P_111010000_1*QR_000010010101+a1P_111000001_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000010001_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(P_011010001*QR_000000020000+a1P_011010000_1*QR_000000020001+a1P_011000001_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111010001*QR_000000020100+a1P_111010000_1*QR_000000020101+a1P_111000001_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000010001_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_010011001*QR_020000000000+a1P_010011000_1*QR_020000000001+P_010111001*QR_020000000010+a1P_010111000_1*QR_020000000011+a2P_010000001_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000011001_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111001_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_010011001*QR_010010000000+a1P_010011000_1*QR_010010000001+P_010111001*QR_010010000010+a1P_010111000_1*QR_010010000011+a2P_010000001_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000011001_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111001_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010011001*QR_000020000000+a1P_010011000_1*QR_000020000001+P_010111001*QR_000020000010+a1P_010111000_1*QR_000020000011+a2P_010000001_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000011001_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111001_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(P_010011001*QR_010000010000+a1P_010011000_1*QR_010000010001+P_010111001*QR_010000010010+a1P_010111000_1*QR_010000010011+a2P_010000001_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000011001_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111001_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(P_010011001*QR_000010010000+a1P_010011000_1*QR_000010010001+P_010111001*QR_000010010010+a1P_010111000_1*QR_000010010011+a2P_010000001_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000011001_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111001_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(P_010011001*QR_000000020000+a1P_010011000_1*QR_000000020001+P_010111001*QR_000000020010+a1P_010111000_1*QR_000000020011+a2P_010000001_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000011001_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111001_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_010010002*QR_020000000000+a1P_010010001_2*QR_020000000001+a2P_010010000_1*QR_020000000002+a1P_010000002_1*QR_020000000010+a2P_010000001_2*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000010002_1*QR_020000000100+a2P_000010001_2*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000002_1*QR_020000000110+a3P_000000001_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_010010002*QR_010010000000+a1P_010010001_2*QR_010010000001+a2P_010010000_1*QR_010010000002+a1P_010000002_1*QR_010010000010+a2P_010000001_2*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000010002_1*QR_010010000100+a2P_000010001_2*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000002_1*QR_010010000110+a3P_000000001_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010010002*QR_000020000000+a1P_010010001_2*QR_000020000001+a2P_010010000_1*QR_000020000002+a1P_010000002_1*QR_000020000010+a2P_010000001_2*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000010002_1*QR_000020000100+a2P_000010001_2*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000002_1*QR_000020000110+a3P_000000001_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(P_010010002*QR_010000010000+a1P_010010001_2*QR_010000010001+a2P_010010000_1*QR_010000010002+a1P_010000002_1*QR_010000010010+a2P_010000001_2*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000010002_1*QR_010000010100+a2P_000010001_2*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000002_1*QR_010000010110+a3P_000000001_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(P_010010002*QR_000010010000+a1P_010010001_2*QR_000010010001+a2P_010010000_1*QR_000010010002+a1P_010000002_1*QR_000010010010+a2P_010000001_2*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000010002_1*QR_000010010100+a2P_000010001_2*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000002_1*QR_000010010110+a3P_000000001_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(P_010010002*QR_000000020000+a1P_010010001_2*QR_000000020001+a2P_010010000_1*QR_000000020002+a1P_010000002_1*QR_000000020010+a2P_010000001_2*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000010002_1*QR_000000020100+a2P_000010001_2*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000002_1*QR_000000020110+a3P_000000001_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_002020000*QR_020000000000+a1P_002010000_2*QR_020000000010+a2P_002000000_1*QR_020000000020+a1P_001020000_2*QR_020000000100+a2P_001010000_4*QR_020000000110+a3P_001000000_2*QR_020000000120+a2P_000020000_1*QR_020000000200+a3P_000010000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_002020000*QR_010010000000+a1P_002010000_2*QR_010010000010+a2P_002000000_1*QR_010010000020+a1P_001020000_2*QR_010010000100+a2P_001010000_4*QR_010010000110+a3P_001000000_2*QR_010010000120+a2P_000020000_1*QR_010010000200+a3P_000010000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_002020000*QR_000020000000+a1P_002010000_2*QR_000020000010+a2P_002000000_1*QR_000020000020+a1P_001020000_2*QR_000020000100+a2P_001010000_4*QR_000020000110+a3P_001000000_2*QR_000020000120+a2P_000020000_1*QR_000020000200+a3P_000010000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(P_002020000*QR_010000010000+a1P_002010000_2*QR_010000010010+a2P_002000000_1*QR_010000010020+a1P_001020000_2*QR_010000010100+a2P_001010000_4*QR_010000010110+a3P_001000000_2*QR_010000010120+a2P_000020000_1*QR_010000010200+a3P_000010000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(P_002020000*QR_000010010000+a1P_002010000_2*QR_000010010010+a2P_002000000_1*QR_000010010020+a1P_001020000_2*QR_000010010100+a2P_001010000_4*QR_000010010110+a3P_001000000_2*QR_000010010120+a2P_000020000_1*QR_000010010200+a3P_000010000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(P_002020000*QR_000000020000+a1P_002010000_2*QR_000000020010+a2P_002000000_1*QR_000000020020+a1P_001020000_2*QR_000000020100+a2P_001010000_4*QR_000000020110+a3P_001000000_2*QR_000000020120+a2P_000020000_1*QR_000000020200+a3P_000010000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_001021000*QR_020000000000+P_001121000*QR_020000000010+P_001221000*QR_020000000020+a3P_001000000_1*QR_020000000030+a1P_000021000_1*QR_020000000100+a1P_000121000_1*QR_020000000110+a1P_000221000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_001021000*QR_010010000000+P_001121000*QR_010010000010+P_001221000*QR_010010000020+a3P_001000000_1*QR_010010000030+a1P_000021000_1*QR_010010000100+a1P_000121000_1*QR_010010000110+a1P_000221000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_001021000*QR_000020000000+P_001121000*QR_000020000010+P_001221000*QR_000020000020+a3P_001000000_1*QR_000020000030+a1P_000021000_1*QR_000020000100+a1P_000121000_1*QR_000020000110+a1P_000221000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(P_001021000*QR_010000010000+P_001121000*QR_010000010010+P_001221000*QR_010000010020+a3P_001000000_1*QR_010000010030+a1P_000021000_1*QR_010000010100+a1P_000121000_1*QR_010000010110+a1P_000221000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(P_001021000*QR_000010010000+P_001121000*QR_000010010010+P_001221000*QR_000010010020+a3P_001000000_1*QR_000010010030+a1P_000021000_1*QR_000010010100+a1P_000121000_1*QR_000010010110+a1P_000221000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(P_001021000*QR_000000020000+P_001121000*QR_000000020010+P_001221000*QR_000000020020+a3P_001000000_1*QR_000000020030+a1P_000021000_1*QR_000000020100+a1P_000121000_1*QR_000000020110+a1P_000221000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_000022000*QR_020000000000+P_000122000*QR_020000000010+P_000222000*QR_020000000020+a2P_000111000_2*QR_020000000030+aPin4*QR_020000000040);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_000022000*QR_010010000000+P_000122000*QR_010010000010+P_000222000*QR_010010000020+a2P_000111000_2*QR_010010000030+aPin4*QR_010010000040);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000022000*QR_000020000000+P_000122000*QR_000020000010+P_000222000*QR_000020000020+a2P_000111000_2*QR_000020000030+aPin4*QR_000020000040);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(P_000022000*QR_010000010000+P_000122000*QR_010000010010+P_000222000*QR_010000010020+a2P_000111000_2*QR_010000010030+aPin4*QR_010000010040);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(P_000022000*QR_000010010000+P_000122000*QR_000010010010+P_000222000*QR_000010010020+a2P_000111000_2*QR_000010010030+aPin4*QR_000010010040);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(P_000022000*QR_000000020000+P_000122000*QR_000000020010+P_000222000*QR_000000020020+a2P_000111000_2*QR_000000020030+aPin4*QR_000000020040);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_001020001*QR_020000000000+a1P_001020000_1*QR_020000000001+a1P_001010001_2*QR_020000000010+a2P_001010000_2*QR_020000000011+a2P_001000001_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000020001_1*QR_020000000100+a2P_000020000_1*QR_020000000101+a2P_000010001_2*QR_020000000110+a3P_000010000_2*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_001020001*QR_010010000000+a1P_001020000_1*QR_010010000001+a1P_001010001_2*QR_010010000010+a2P_001010000_2*QR_010010000011+a2P_001000001_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000020001_1*QR_010010000100+a2P_000020000_1*QR_010010000101+a2P_000010001_2*QR_010010000110+a3P_000010000_2*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_001020001*QR_000020000000+a1P_001020000_1*QR_000020000001+a1P_001010001_2*QR_000020000010+a2P_001010000_2*QR_000020000011+a2P_001000001_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000020001_1*QR_000020000100+a2P_000020000_1*QR_000020000101+a2P_000010001_2*QR_000020000110+a3P_000010000_2*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(P_001020001*QR_010000010000+a1P_001020000_1*QR_010000010001+a1P_001010001_2*QR_010000010010+a2P_001010000_2*QR_010000010011+a2P_001000001_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000020001_1*QR_010000010100+a2P_000020000_1*QR_010000010101+a2P_000010001_2*QR_010000010110+a3P_000010000_2*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(P_001020001*QR_000010010000+a1P_001020000_1*QR_000010010001+a1P_001010001_2*QR_000010010010+a2P_001010000_2*QR_000010010011+a2P_001000001_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000020001_1*QR_000010010100+a2P_000020000_1*QR_000010010101+a2P_000010001_2*QR_000010010110+a3P_000010000_2*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(P_001020001*QR_000000020000+a1P_001020000_1*QR_000000020001+a1P_001010001_2*QR_000000020010+a2P_001010000_2*QR_000000020011+a2P_001000001_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000020001_1*QR_000000020100+a2P_000020000_1*QR_000000020101+a2P_000010001_2*QR_000000020110+a3P_000010000_2*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_000021001*QR_020000000000+a1P_000021000_1*QR_020000000001+P_000121001*QR_020000000010+a1P_000121000_1*QR_020000000011+P_000221001*QR_020000000020+a1P_000221000_1*QR_020000000021+a3P_000000001_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_000021001*QR_010010000000+a1P_000021000_1*QR_010010000001+P_000121001*QR_010010000010+a1P_000121000_1*QR_010010000011+P_000221001*QR_010010000020+a1P_000221000_1*QR_010010000021+a3P_000000001_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000021001*QR_000020000000+a1P_000021000_1*QR_000020000001+P_000121001*QR_000020000010+a1P_000121000_1*QR_000020000011+P_000221001*QR_000020000020+a1P_000221000_1*QR_000020000021+a3P_000000001_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(P_000021001*QR_010000010000+a1P_000021000_1*QR_010000010001+P_000121001*QR_010000010010+a1P_000121000_1*QR_010000010011+P_000221001*QR_010000010020+a1P_000221000_1*QR_010000010021+a3P_000000001_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(P_000021001*QR_000010010000+a1P_000021000_1*QR_000010010001+P_000121001*QR_000010010010+a1P_000121000_1*QR_000010010011+P_000221001*QR_000010010020+a1P_000221000_1*QR_000010010021+a3P_000000001_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(P_000021001*QR_000000020000+a1P_000021000_1*QR_000000020001+P_000121001*QR_000000020010+a1P_000121000_1*QR_000000020011+P_000221001*QR_000000020020+a1P_000221000_1*QR_000000020021+a3P_000000001_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_000020002*QR_020000000000+a1P_000020001_2*QR_020000000001+a2P_000020000_1*QR_020000000002+a1P_000010002_2*QR_020000000010+a2P_000010001_4*QR_020000000011+a3P_000010000_2*QR_020000000012+a2P_000000002_1*QR_020000000020+a3P_000000001_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_000020002*QR_010010000000+a1P_000020001_2*QR_010010000001+a2P_000020000_1*QR_010010000002+a1P_000010002_2*QR_010010000010+a2P_000010001_4*QR_010010000011+a3P_000010000_2*QR_010010000012+a2P_000000002_1*QR_010010000020+a3P_000000001_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000020002*QR_000020000000+a1P_000020001_2*QR_000020000001+a2P_000020000_1*QR_000020000002+a1P_000010002_2*QR_000020000010+a2P_000010001_4*QR_000020000011+a3P_000010000_2*QR_000020000012+a2P_000000002_1*QR_000020000020+a3P_000000001_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(P_000020002*QR_010000010000+a1P_000020001_2*QR_010000010001+a2P_000020000_1*QR_010000010002+a1P_000010002_2*QR_010000010010+a2P_000010001_4*QR_010000010011+a3P_000010000_2*QR_010000010012+a2P_000000002_1*QR_010000010020+a3P_000000001_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(P_000020002*QR_000010010000+a1P_000020001_2*QR_000010010001+a2P_000020000_1*QR_000010010002+a1P_000010002_2*QR_000010010010+a2P_000010001_4*QR_000010010011+a3P_000010000_2*QR_000010010012+a2P_000000002_1*QR_000010010020+a3P_000000001_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(P_000020002*QR_000000020000+a1P_000020001_2*QR_000000020001+a2P_000020000_1*QR_000000020002+a1P_000010002_2*QR_000000020010+a2P_000010001_4*QR_000000020011+a3P_000010000_2*QR_000000020012+a2P_000000002_1*QR_000000020020+a3P_000000001_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_012000010*QR_020000000000+a1P_012000000_1*QR_020000000001+P_112000010*QR_020000000100+a1P_112000000_1*QR_020000000101+P_212000010*QR_020000000200+a1P_212000000_1*QR_020000000201+a3P_000000010_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_012000010*QR_010010000000+a1P_012000000_1*QR_010010000001+P_112000010*QR_010010000100+a1P_112000000_1*QR_010010000101+P_212000010*QR_010010000200+a1P_212000000_1*QR_010010000201+a3P_000000010_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_012000010*QR_000020000000+a1P_012000000_1*QR_000020000001+P_112000010*QR_000020000100+a1P_112000000_1*QR_000020000101+P_212000010*QR_000020000200+a1P_212000000_1*QR_000020000201+a3P_000000010_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(P_012000010*QR_010000010000+a1P_012000000_1*QR_010000010001+P_112000010*QR_010000010100+a1P_112000000_1*QR_010000010101+P_212000010*QR_010000010200+a1P_212000000_1*QR_010000010201+a3P_000000010_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(P_012000010*QR_000010010000+a1P_012000000_1*QR_000010010001+P_112000010*QR_000010010100+a1P_112000000_1*QR_000010010101+P_212000010*QR_000010010200+a1P_212000000_1*QR_000010010201+a3P_000000010_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(P_012000010*QR_000000020000+a1P_012000000_1*QR_000000020001+P_112000010*QR_000000020100+a1P_112000000_1*QR_000000020101+P_212000010*QR_000000020200+a1P_212000000_1*QR_000000020201+a3P_000000010_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_011001010*QR_020000000000+a1P_011001000_1*QR_020000000001+a1P_011000010_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111001010*QR_020000000100+a1P_111001000_1*QR_020000000101+a1P_111000010_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000001010_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_011001010*QR_010010000000+a1P_011001000_1*QR_010010000001+a1P_011000010_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111001010*QR_010010000100+a1P_111001000_1*QR_010010000101+a1P_111000010_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000001010_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_011001010*QR_000020000000+a1P_011001000_1*QR_000020000001+a1P_011000010_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111001010*QR_000020000100+a1P_111001000_1*QR_000020000101+a1P_111000010_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000001010_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(P_011001010*QR_010000010000+a1P_011001000_1*QR_010000010001+a1P_011000010_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111001010*QR_010000010100+a1P_111001000_1*QR_010000010101+a1P_111000010_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000001010_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(P_011001010*QR_000010010000+a1P_011001000_1*QR_000010010001+a1P_011000010_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111001010*QR_000010010100+a1P_111001000_1*QR_000010010101+a1P_111000010_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000001010_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(P_011001010*QR_000000020000+a1P_011001000_1*QR_000000020001+a1P_011000010_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111001010*QR_000000020100+a1P_111001000_1*QR_000000020101+a1P_111000010_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000001010_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_010002010*QR_020000000000+a1P_010002000_1*QR_020000000001+a1P_010001010_2*QR_020000000010+a2P_010001000_2*QR_020000000011+a2P_010000010_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000002010_1*QR_020000000100+a2P_000002000_1*QR_020000000101+a2P_000001010_2*QR_020000000110+a3P_000001000_2*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_010002010*QR_010010000000+a1P_010002000_1*QR_010010000001+a1P_010001010_2*QR_010010000010+a2P_010001000_2*QR_010010000011+a2P_010000010_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000002010_1*QR_010010000100+a2P_000002000_1*QR_010010000101+a2P_000001010_2*QR_010010000110+a3P_000001000_2*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010002010*QR_000020000000+a1P_010002000_1*QR_000020000001+a1P_010001010_2*QR_000020000010+a2P_010001000_2*QR_000020000011+a2P_010000010_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000002010_1*QR_000020000100+a2P_000002000_1*QR_000020000101+a2P_000001010_2*QR_000020000110+a3P_000001000_2*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(P_010002010*QR_010000010000+a1P_010002000_1*QR_010000010001+a1P_010001010_2*QR_010000010010+a2P_010001000_2*QR_010000010011+a2P_010000010_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000002010_1*QR_010000010100+a2P_000002000_1*QR_010000010101+a2P_000001010_2*QR_010000010110+a3P_000001000_2*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(P_010002010*QR_000010010000+a1P_010002000_1*QR_000010010001+a1P_010001010_2*QR_000010010010+a2P_010001000_2*QR_000010010011+a2P_010000010_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000002010_1*QR_000010010100+a2P_000002000_1*QR_000010010101+a2P_000001010_2*QR_000010010110+a3P_000001000_2*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(P_010002010*QR_000000020000+a1P_010002000_1*QR_000000020001+a1P_010001010_2*QR_000000020010+a2P_010001000_2*QR_000000020011+a2P_010000010_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000002010_1*QR_000000020100+a2P_000002000_1*QR_000000020101+a2P_000001010_2*QR_000000020110+a3P_000001000_2*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_011000011*QR_020000000000+P_011000111*QR_020000000001+a2P_011000000_1*QR_020000000002+P_111000011*QR_020000000100+P_111000111*QR_020000000101+a2P_111000000_1*QR_020000000102+a2P_000000011_1*QR_020000000200+a2P_000000111_1*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_011000011*QR_010010000000+P_011000111*QR_010010000001+a2P_011000000_1*QR_010010000002+P_111000011*QR_010010000100+P_111000111*QR_010010000101+a2P_111000000_1*QR_010010000102+a2P_000000011_1*QR_010010000200+a2P_000000111_1*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_011000011*QR_000020000000+P_011000111*QR_000020000001+a2P_011000000_1*QR_000020000002+P_111000011*QR_000020000100+P_111000111*QR_000020000101+a2P_111000000_1*QR_000020000102+a2P_000000011_1*QR_000020000200+a2P_000000111_1*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(P_011000011*QR_010000010000+P_011000111*QR_010000010001+a2P_011000000_1*QR_010000010002+P_111000011*QR_010000010100+P_111000111*QR_010000010101+a2P_111000000_1*QR_010000010102+a2P_000000011_1*QR_010000010200+a2P_000000111_1*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(P_011000011*QR_000010010000+P_011000111*QR_000010010001+a2P_011000000_1*QR_000010010002+P_111000011*QR_000010010100+P_111000111*QR_000010010101+a2P_111000000_1*QR_000010010102+a2P_000000011_1*QR_000010010200+a2P_000000111_1*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(P_011000011*QR_000000020000+P_011000111*QR_000000020001+a2P_011000000_1*QR_000000020002+P_111000011*QR_000000020100+P_111000111*QR_000000020101+a2P_111000000_1*QR_000000020102+a2P_000000011_1*QR_000000020200+a2P_000000111_1*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_010001011*QR_020000000000+P_010001111*QR_020000000001+a2P_010001000_1*QR_020000000002+a1P_010000011_1*QR_020000000010+a1P_010000111_1*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000001011_1*QR_020000000100+a1P_000001111_1*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_010001011*QR_010010000000+P_010001111*QR_010010000001+a2P_010001000_1*QR_010010000002+a1P_010000011_1*QR_010010000010+a1P_010000111_1*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000001011_1*QR_010010000100+a1P_000001111_1*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010001011*QR_000020000000+P_010001111*QR_000020000001+a2P_010001000_1*QR_000020000002+a1P_010000011_1*QR_000020000010+a1P_010000111_1*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000001011_1*QR_000020000100+a1P_000001111_1*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(P_010001011*QR_010000010000+P_010001111*QR_010000010001+a2P_010001000_1*QR_010000010002+a1P_010000011_1*QR_010000010010+a1P_010000111_1*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000001011_1*QR_010000010100+a1P_000001111_1*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(P_010001011*QR_000010010000+P_010001111*QR_000010010001+a2P_010001000_1*QR_000010010002+a1P_010000011_1*QR_000010010010+a1P_010000111_1*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000001011_1*QR_000010010100+a1P_000001111_1*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(P_010001011*QR_000000020000+P_010001111*QR_000000020001+a2P_010001000_1*QR_000000020002+a1P_010000011_1*QR_000000020010+a1P_010000111_1*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000001011_1*QR_000000020100+a1P_000001111_1*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_010000012*QR_020000000000+P_010000112*QR_020000000001+P_010000212*QR_020000000002+a3P_010000000_1*QR_020000000003+a1P_000000012_1*QR_020000000100+a1P_000000112_1*QR_020000000101+a1P_000000212_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_010000012*QR_010010000000+P_010000112*QR_010010000001+P_010000212*QR_010010000002+a3P_010000000_1*QR_010010000003+a1P_000000012_1*QR_010010000100+a1P_000000112_1*QR_010010000101+a1P_000000212_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010000012*QR_000020000000+P_010000112*QR_000020000001+P_010000212*QR_000020000002+a3P_010000000_1*QR_000020000003+a1P_000000012_1*QR_000020000100+a1P_000000112_1*QR_000020000101+a1P_000000212_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(P_010000012*QR_010000010000+P_010000112*QR_010000010001+P_010000212*QR_010000010002+a3P_010000000_1*QR_010000010003+a1P_000000012_1*QR_010000010100+a1P_000000112_1*QR_010000010101+a1P_000000212_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(P_010000012*QR_000010010000+P_010000112*QR_000010010001+P_010000212*QR_000010010002+a3P_010000000_1*QR_000010010003+a1P_000000012_1*QR_000010010100+a1P_000000112_1*QR_000010010101+a1P_000000212_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(P_010000012*QR_000000020000+P_010000112*QR_000000020001+P_010000212*QR_000000020002+a3P_010000000_1*QR_000000020003+a1P_000000012_1*QR_000000020100+a1P_000000112_1*QR_000000020101+a1P_000000212_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_002010010*QR_020000000000+a1P_002010000_1*QR_020000000001+a1P_002000010_1*QR_020000000010+a2P_002000000_1*QR_020000000011+a1P_001010010_2*QR_020000000100+a2P_001010000_2*QR_020000000101+a2P_001000010_2*QR_020000000110+a3P_001000000_2*QR_020000000111+a2P_000010010_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_002010010*QR_010010000000+a1P_002010000_1*QR_010010000001+a1P_002000010_1*QR_010010000010+a2P_002000000_1*QR_010010000011+a1P_001010010_2*QR_010010000100+a2P_001010000_2*QR_010010000101+a2P_001000010_2*QR_010010000110+a3P_001000000_2*QR_010010000111+a2P_000010010_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_002010010*QR_000020000000+a1P_002010000_1*QR_000020000001+a1P_002000010_1*QR_000020000010+a2P_002000000_1*QR_000020000011+a1P_001010010_2*QR_000020000100+a2P_001010000_2*QR_000020000101+a2P_001000010_2*QR_000020000110+a3P_001000000_2*QR_000020000111+a2P_000010010_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(P_002010010*QR_010000010000+a1P_002010000_1*QR_010000010001+a1P_002000010_1*QR_010000010010+a2P_002000000_1*QR_010000010011+a1P_001010010_2*QR_010000010100+a2P_001010000_2*QR_010000010101+a2P_001000010_2*QR_010000010110+a3P_001000000_2*QR_010000010111+a2P_000010010_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(P_002010010*QR_000010010000+a1P_002010000_1*QR_000010010001+a1P_002000010_1*QR_000010010010+a2P_002000000_1*QR_000010010011+a1P_001010010_2*QR_000010010100+a2P_001010000_2*QR_000010010101+a2P_001000010_2*QR_000010010110+a3P_001000000_2*QR_000010010111+a2P_000010010_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(P_002010010*QR_000000020000+a1P_002010000_1*QR_000000020001+a1P_002000010_1*QR_000000020010+a2P_002000000_1*QR_000000020011+a1P_001010010_2*QR_000000020100+a2P_001010000_2*QR_000000020101+a2P_001000010_2*QR_000000020110+a3P_001000000_2*QR_000000020111+a2P_000010010_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_001011010*QR_020000000000+a1P_001011000_1*QR_020000000001+P_001111010*QR_020000000010+a1P_001111000_1*QR_020000000011+a2P_001000010_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000011010_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111010_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_001011010*QR_010010000000+a1P_001011000_1*QR_010010000001+P_001111010*QR_010010000010+a1P_001111000_1*QR_010010000011+a2P_001000010_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000011010_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111010_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_001011010*QR_000020000000+a1P_001011000_1*QR_000020000001+P_001111010*QR_000020000010+a1P_001111000_1*QR_000020000011+a2P_001000010_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000011010_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111010_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(P_001011010*QR_010000010000+a1P_001011000_1*QR_010000010001+P_001111010*QR_010000010010+a1P_001111000_1*QR_010000010011+a2P_001000010_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000011010_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111010_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(P_001011010*QR_000010010000+a1P_001011000_1*QR_000010010001+P_001111010*QR_000010010010+a1P_001111000_1*QR_000010010011+a2P_001000010_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000011010_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111010_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(P_001011010*QR_000000020000+a1P_001011000_1*QR_000000020001+P_001111010*QR_000000020010+a1P_001111000_1*QR_000000020011+a2P_001000010_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000011010_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111010_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_000012010*QR_020000000000+a1P_000012000_1*QR_020000000001+P_000112010*QR_020000000010+a1P_000112000_1*QR_020000000011+P_000212010*QR_020000000020+a1P_000212000_1*QR_020000000021+a3P_000000010_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_000012010*QR_010010000000+a1P_000012000_1*QR_010010000001+P_000112010*QR_010010000010+a1P_000112000_1*QR_010010000011+P_000212010*QR_010010000020+a1P_000212000_1*QR_010010000021+a3P_000000010_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000012010*QR_000020000000+a1P_000012000_1*QR_000020000001+P_000112010*QR_000020000010+a1P_000112000_1*QR_000020000011+P_000212010*QR_000020000020+a1P_000212000_1*QR_000020000021+a3P_000000010_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(P_000012010*QR_010000010000+a1P_000012000_1*QR_010000010001+P_000112010*QR_010000010010+a1P_000112000_1*QR_010000010011+P_000212010*QR_010000010020+a1P_000212000_1*QR_010000010021+a3P_000000010_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(P_000012010*QR_000010010000+a1P_000012000_1*QR_000010010001+P_000112010*QR_000010010010+a1P_000112000_1*QR_000010010011+P_000212010*QR_000010010020+a1P_000212000_1*QR_000010010021+a3P_000000010_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(P_000012010*QR_000000020000+a1P_000012000_1*QR_000000020001+P_000112010*QR_000000020010+a1P_000112000_1*QR_000000020011+P_000212010*QR_000000020020+a1P_000212000_1*QR_000000020021+a3P_000000010_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_001010011*QR_020000000000+P_001010111*QR_020000000001+a2P_001010000_1*QR_020000000002+a1P_001000011_1*QR_020000000010+a1P_001000111_1*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000010011_1*QR_020000000100+a1P_000010111_1*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_001010011*QR_010010000000+P_001010111*QR_010010000001+a2P_001010000_1*QR_010010000002+a1P_001000011_1*QR_010010000010+a1P_001000111_1*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000010011_1*QR_010010000100+a1P_000010111_1*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_001010011*QR_000020000000+P_001010111*QR_000020000001+a2P_001010000_1*QR_000020000002+a1P_001000011_1*QR_000020000010+a1P_001000111_1*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000010011_1*QR_000020000100+a1P_000010111_1*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(P_001010011*QR_010000010000+P_001010111*QR_010000010001+a2P_001010000_1*QR_010000010002+a1P_001000011_1*QR_010000010010+a1P_001000111_1*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000010011_1*QR_010000010100+a1P_000010111_1*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(P_001010011*QR_000010010000+P_001010111*QR_000010010001+a2P_001010000_1*QR_000010010002+a1P_001000011_1*QR_000010010010+a1P_001000111_1*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000010011_1*QR_000010010100+a1P_000010111_1*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(P_001010011*QR_000000020000+P_001010111*QR_000000020001+a2P_001010000_1*QR_000000020002+a1P_001000011_1*QR_000000020010+a1P_001000111_1*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000010011_1*QR_000000020100+a1P_000010111_1*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_000011011*QR_020000000000+P_000011111*QR_020000000001+a2P_000011000_1*QR_020000000002+P_000111011*QR_020000000010+P_000111111*QR_020000000011+a2P_000111000_1*QR_020000000012+a2P_000000011_1*QR_020000000020+a2P_000000111_1*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_000011011*QR_010010000000+P_000011111*QR_010010000001+a2P_000011000_1*QR_010010000002+P_000111011*QR_010010000010+P_000111111*QR_010010000011+a2P_000111000_1*QR_010010000012+a2P_000000011_1*QR_010010000020+a2P_000000111_1*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000011011*QR_000020000000+P_000011111*QR_000020000001+a2P_000011000_1*QR_000020000002+P_000111011*QR_000020000010+P_000111111*QR_000020000011+a2P_000111000_1*QR_000020000012+a2P_000000011_1*QR_000020000020+a2P_000000111_1*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(P_000011011*QR_010000010000+P_000011111*QR_010000010001+a2P_000011000_1*QR_010000010002+P_000111011*QR_010000010010+P_000111111*QR_010000010011+a2P_000111000_1*QR_010000010012+a2P_000000011_1*QR_010000010020+a2P_000000111_1*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(P_000011011*QR_000010010000+P_000011111*QR_000010010001+a2P_000011000_1*QR_000010010002+P_000111011*QR_000010010010+P_000111111*QR_000010010011+a2P_000111000_1*QR_000010010012+a2P_000000011_1*QR_000010010020+a2P_000000111_1*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(P_000011011*QR_000000020000+P_000011111*QR_000000020001+a2P_000011000_1*QR_000000020002+P_000111011*QR_000000020010+P_000111111*QR_000000020011+a2P_000111000_1*QR_000000020012+a2P_000000011_1*QR_000000020020+a2P_000000111_1*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_000010012*QR_020000000000+P_000010112*QR_020000000001+P_000010212*QR_020000000002+a3P_000010000_1*QR_020000000003+a1P_000000012_1*QR_020000000010+a1P_000000112_1*QR_020000000011+a1P_000000212_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_000010012*QR_010010000000+P_000010112*QR_010010000001+P_000010212*QR_010010000002+a3P_000010000_1*QR_010010000003+a1P_000000012_1*QR_010010000010+a1P_000000112_1*QR_010010000011+a1P_000000212_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000010012*QR_000020000000+P_000010112*QR_000020000001+P_000010212*QR_000020000002+a3P_000010000_1*QR_000020000003+a1P_000000012_1*QR_000020000010+a1P_000000112_1*QR_000020000011+a1P_000000212_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(P_000010012*QR_010000010000+P_000010112*QR_010000010001+P_000010212*QR_010000010002+a3P_000010000_1*QR_010000010003+a1P_000000012_1*QR_010000010010+a1P_000000112_1*QR_010000010011+a1P_000000212_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(P_000010012*QR_000010010000+P_000010112*QR_000010010001+P_000010212*QR_000010010002+a3P_000010000_1*QR_000010010003+a1P_000000012_1*QR_000010010010+a1P_000000112_1*QR_000010010011+a1P_000000212_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(P_000010012*QR_000000020000+P_000010112*QR_000000020001+P_000010212*QR_000000020002+a3P_000010000_1*QR_000000020003+a1P_000000012_1*QR_000000020010+a1P_000000112_1*QR_000000020011+a1P_000000212_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_002000020*QR_020000000000+a1P_002000010_2*QR_020000000001+a2P_002000000_1*QR_020000000002+a1P_001000020_2*QR_020000000100+a2P_001000010_4*QR_020000000101+a3P_001000000_2*QR_020000000102+a2P_000000020_1*QR_020000000200+a3P_000000010_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_002000020*QR_010010000000+a1P_002000010_2*QR_010010000001+a2P_002000000_1*QR_010010000002+a1P_001000020_2*QR_010010000100+a2P_001000010_4*QR_010010000101+a3P_001000000_2*QR_010010000102+a2P_000000020_1*QR_010010000200+a3P_000000010_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_002000020*QR_000020000000+a1P_002000010_2*QR_000020000001+a2P_002000000_1*QR_000020000002+a1P_001000020_2*QR_000020000100+a2P_001000010_4*QR_000020000101+a3P_001000000_2*QR_000020000102+a2P_000000020_1*QR_000020000200+a3P_000000010_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(P_002000020*QR_010000010000+a1P_002000010_2*QR_010000010001+a2P_002000000_1*QR_010000010002+a1P_001000020_2*QR_010000010100+a2P_001000010_4*QR_010000010101+a3P_001000000_2*QR_010000010102+a2P_000000020_1*QR_010000010200+a3P_000000010_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(P_002000020*QR_000010010000+a1P_002000010_2*QR_000010010001+a2P_002000000_1*QR_000010010002+a1P_001000020_2*QR_000010010100+a2P_001000010_4*QR_000010010101+a3P_001000000_2*QR_000010010102+a2P_000000020_1*QR_000010010200+a3P_000000010_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(P_002000020*QR_000000020000+a1P_002000010_2*QR_000000020001+a2P_002000000_1*QR_000000020002+a1P_001000020_2*QR_000000020100+a2P_001000010_4*QR_000000020101+a3P_001000000_2*QR_000000020102+a2P_000000020_1*QR_000000020200+a3P_000000010_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_001001020*QR_020000000000+a1P_001001010_2*QR_020000000001+a2P_001001000_1*QR_020000000002+a1P_001000020_1*QR_020000000010+a2P_001000010_2*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000001020_1*QR_020000000100+a2P_000001010_2*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000020_1*QR_020000000110+a3P_000000010_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_001001020*QR_010010000000+a1P_001001010_2*QR_010010000001+a2P_001001000_1*QR_010010000002+a1P_001000020_1*QR_010010000010+a2P_001000010_2*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000001020_1*QR_010010000100+a2P_000001010_2*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000020_1*QR_010010000110+a3P_000000010_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_001001020*QR_000020000000+a1P_001001010_2*QR_000020000001+a2P_001001000_1*QR_000020000002+a1P_001000020_1*QR_000020000010+a2P_001000010_2*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000001020_1*QR_000020000100+a2P_000001010_2*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000020_1*QR_000020000110+a3P_000000010_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(P_001001020*QR_010000010000+a1P_001001010_2*QR_010000010001+a2P_001001000_1*QR_010000010002+a1P_001000020_1*QR_010000010010+a2P_001000010_2*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000001020_1*QR_010000010100+a2P_000001010_2*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000020_1*QR_010000010110+a3P_000000010_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(P_001001020*QR_000010010000+a1P_001001010_2*QR_000010010001+a2P_001001000_1*QR_000010010002+a1P_001000020_1*QR_000010010010+a2P_001000010_2*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000001020_1*QR_000010010100+a2P_000001010_2*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000020_1*QR_000010010110+a3P_000000010_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(P_001001020*QR_000000020000+a1P_001001010_2*QR_000000020001+a2P_001001000_1*QR_000000020002+a1P_001000020_1*QR_000000020010+a2P_001000010_2*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000001020_1*QR_000000020100+a2P_000001010_2*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000020_1*QR_000000020110+a3P_000000010_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_000002020*QR_020000000000+a1P_000002010_2*QR_020000000001+a2P_000002000_1*QR_020000000002+a1P_000001020_2*QR_020000000010+a2P_000001010_4*QR_020000000011+a3P_000001000_2*QR_020000000012+a2P_000000020_1*QR_020000000020+a3P_000000010_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_000002020*QR_010010000000+a1P_000002010_2*QR_010010000001+a2P_000002000_1*QR_010010000002+a1P_000001020_2*QR_010010000010+a2P_000001010_4*QR_010010000011+a3P_000001000_2*QR_010010000012+a2P_000000020_1*QR_010010000020+a3P_000000010_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000002020*QR_000020000000+a1P_000002010_2*QR_000020000001+a2P_000002000_1*QR_000020000002+a1P_000001020_2*QR_000020000010+a2P_000001010_4*QR_000020000011+a3P_000001000_2*QR_000020000012+a2P_000000020_1*QR_000020000020+a3P_000000010_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(P_000002020*QR_010000010000+a1P_000002010_2*QR_010000010001+a2P_000002000_1*QR_010000010002+a1P_000001020_2*QR_010000010010+a2P_000001010_4*QR_010000010011+a3P_000001000_2*QR_010000010012+a2P_000000020_1*QR_010000010020+a3P_000000010_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(P_000002020*QR_000010010000+a1P_000002010_2*QR_000010010001+a2P_000002000_1*QR_000010010002+a1P_000001020_2*QR_000010010010+a2P_000001010_4*QR_000010010011+a3P_000001000_2*QR_000010010012+a2P_000000020_1*QR_000010010020+a3P_000000010_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(P_000002020*QR_000000020000+a1P_000002010_2*QR_000000020001+a2P_000002000_1*QR_000000020002+a1P_000001020_2*QR_000000020010+a2P_000001010_4*QR_000000020011+a3P_000001000_2*QR_000000020012+a2P_000000020_1*QR_000000020020+a3P_000000010_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_001000021*QR_020000000000+P_001000121*QR_020000000001+P_001000221*QR_020000000002+a3P_001000000_1*QR_020000000003+a1P_000000021_1*QR_020000000100+a1P_000000121_1*QR_020000000101+a1P_000000221_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_001000021*QR_010010000000+P_001000121*QR_010010000001+P_001000221*QR_010010000002+a3P_001000000_1*QR_010010000003+a1P_000000021_1*QR_010010000100+a1P_000000121_1*QR_010010000101+a1P_000000221_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_001000021*QR_000020000000+P_001000121*QR_000020000001+P_001000221*QR_000020000002+a3P_001000000_1*QR_000020000003+a1P_000000021_1*QR_000020000100+a1P_000000121_1*QR_000020000101+a1P_000000221_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(P_001000021*QR_010000010000+P_001000121*QR_010000010001+P_001000221*QR_010000010002+a3P_001000000_1*QR_010000010003+a1P_000000021_1*QR_010000010100+a1P_000000121_1*QR_010000010101+a1P_000000221_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(P_001000021*QR_000010010000+P_001000121*QR_000010010001+P_001000221*QR_000010010002+a3P_001000000_1*QR_000010010003+a1P_000000021_1*QR_000010010100+a1P_000000121_1*QR_000010010101+a1P_000000221_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(P_001000021*QR_000000020000+P_001000121*QR_000000020001+P_001000221*QR_000000020002+a3P_001000000_1*QR_000000020003+a1P_000000021_1*QR_000000020100+a1P_000000121_1*QR_000000020101+a1P_000000221_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_000001021*QR_020000000000+P_000001121*QR_020000000001+P_000001221*QR_020000000002+a3P_000001000_1*QR_020000000003+a1P_000000021_1*QR_020000000010+a1P_000000121_1*QR_020000000011+a1P_000000221_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_000001021*QR_010010000000+P_000001121*QR_010010000001+P_000001221*QR_010010000002+a3P_000001000_1*QR_010010000003+a1P_000000021_1*QR_010010000010+a1P_000000121_1*QR_010010000011+a1P_000000221_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000001021*QR_000020000000+P_000001121*QR_000020000001+P_000001221*QR_000020000002+a3P_000001000_1*QR_000020000003+a1P_000000021_1*QR_000020000010+a1P_000000121_1*QR_000020000011+a1P_000000221_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(P_000001021*QR_010000010000+P_000001121*QR_010000010001+P_000001221*QR_010000010002+a3P_000001000_1*QR_010000010003+a1P_000000021_1*QR_010000010010+a1P_000000121_1*QR_010000010011+a1P_000000221_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(P_000001021*QR_000010010000+P_000001121*QR_000010010001+P_000001221*QR_000010010002+a3P_000001000_1*QR_000010010003+a1P_000000021_1*QR_000010010010+a1P_000000121_1*QR_000010010011+a1P_000000221_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(P_000001021*QR_000000020000+P_000001121*QR_000000020001+P_000001221*QR_000000020002+a3P_000001000_1*QR_000000020003+a1P_000000021_1*QR_000000020010+a1P_000000121_1*QR_000000020011+a1P_000000221_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_000000022*QR_020000000000+P_000000122*QR_020000000001+P_000000222*QR_020000000002+a2P_000000111_2*QR_020000000003+aPin4*QR_020000000004);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_000000022*QR_010010000000+P_000000122*QR_010010000001+P_000000222*QR_010010000002+a2P_000000111_2*QR_010010000003+aPin4*QR_010010000004);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000000022*QR_000020000000+P_000000122*QR_000020000001+P_000000222*QR_000020000002+a2P_000000111_2*QR_000020000003+aPin4*QR_000020000004);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(P_000000022*QR_010000010000+P_000000122*QR_010000010001+P_000000222*QR_010000010002+a2P_000000111_2*QR_010000010003+aPin4*QR_010000010004);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(P_000000022*QR_000010010000+P_000000122*QR_000010010001+P_000000222*QR_000010010002+a2P_000000111_2*QR_000010010003+aPin4*QR_000010010004);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(P_000000022*QR_000000020000+P_000000122*QR_000000020001+P_000000222*QR_000000020002+a2P_000000111_2*QR_000000020003+aPin4*QR_000000020004);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
}
__global__ void TSMJ_ddds_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double aPin4=aPin1*aPin3;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			double QR_020000000000=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			double QR_010010000000=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			double QR_000020000000=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			double QR_010000010000=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			double QR_000010010000=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			double QR_000000020000=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			double QR_020000000001=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			double QR_010010000001=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			double QR_000020000001=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			double QR_010000010001=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			double QR_000010010001=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			double QR_000000020001=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			double QR_020000000010=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			double QR_010010000010=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			double QR_000020000010=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			double QR_010000010010=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000010010010=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			double QR_000000020010=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			double QR_020000000100=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			double QR_010010000100=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			double QR_000020000100=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			double QR_010000010100=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			double QR_000010010100=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			double QR_000000020100=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			double QR_020000000002=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			double QR_010010000002=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			double QR_000020000002=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			double QR_010000010002=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			double QR_000010010002=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			double QR_000000020002=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			double QR_020000000011=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			double QR_010010000011=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			double QR_000020000011=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			double QR_010000010011=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000010010011=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			double QR_000000020011=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			double QR_020000000020=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			double QR_010010000020=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			double QR_000020000020=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			double QR_010000010020=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000010010020=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			double QR_000000020020=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			double QR_020000000101=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			double QR_010010000101=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			double QR_000020000101=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			double QR_010000010101=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			double QR_000010010101=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			double QR_000000020101=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			double QR_020000000110=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			double QR_010010000110=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			double QR_000020000110=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			double QR_010000010110=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000010010110=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			double QR_000000020110=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			double QR_020000000200=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			double QR_010010000200=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			double QR_000020000200=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			double QR_010000010200=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			double QR_000010010200=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			double QR_000000020200=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			double QR_020000000003=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			double QR_010010000003=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			double QR_000020000003=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			double QR_010000010003=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			double QR_000010010003=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			double QR_000000020003=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			double QR_020000000012=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			double QR_010010000012=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			double QR_000020000012=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			double QR_010000010012=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000010010012=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			double QR_000000020012=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			double QR_020000000021=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			double QR_010010000021=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			double QR_000020000021=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			double QR_010000010021=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000010010021=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			double QR_000000020021=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			double QR_020000000030=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			double QR_010010000030=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			double QR_000020000030=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			double QR_010000010030=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000010010030=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			double QR_000000020030=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			double QR_020000000102=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			double QR_010010000102=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			double QR_000020000102=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			double QR_010000010102=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			double QR_000010010102=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			double QR_000000020102=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			double QR_020000000111=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			double QR_010010000111=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			double QR_000020000111=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			double QR_010000010111=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000010010111=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			double QR_000000020111=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			double QR_020000000120=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			double QR_010010000120=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			double QR_000020000120=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			double QR_010000010120=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000010010120=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			double QR_000000020120=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			double QR_020000000201=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			double QR_010010000201=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			double QR_000020000201=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			double QR_010000010201=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			double QR_000010010201=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			double QR_000000020201=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			double QR_020000000210=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			double QR_010010000210=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			double QR_000020000210=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			double QR_010000010210=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000010010210=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			double QR_000000020210=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			double QR_020000000300=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			double QR_010010000300=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			double QR_000020000300=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			double QR_010000010300=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			double QR_000010010300=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			double QR_000000020300=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
			double QR_020000000004=Q_020000000*R_004[0]-a1Q_010000000_2*R_104[0]+aQin2*R_204[0];
			double QR_010010000004=Q_010010000*R_004[0]-a1Q_010000000_1*R_014[0]-a1Q_000010000_1*R_104[0]+aQin2*R_114[0];
			double QR_000020000004=Q_000020000*R_004[0]-a1Q_000010000_2*R_014[0]+aQin2*R_024[0];
			double QR_010000010004=Q_010000010*R_004[0]-a1Q_010000000_1*R_005[0]-a1Q_000000010_1*R_104[0]+aQin2*R_105[0];
			double QR_000010010004=Q_000010010*R_004[0]-a1Q_000010000_1*R_005[0]-a1Q_000000010_1*R_014[0]+aQin2*R_015[0];
			double QR_000000020004=Q_000000020*R_004[0]-a1Q_000000010_2*R_005[0]+aQin2*R_006[0];
			double QR_020000000013=Q_020000000*R_013[0]-a1Q_010000000_2*R_113[0]+aQin2*R_213[0];
			double QR_010010000013=Q_010010000*R_013[0]-a1Q_010000000_1*R_023[0]-a1Q_000010000_1*R_113[0]+aQin2*R_123[0];
			double QR_000020000013=Q_000020000*R_013[0]-a1Q_000010000_2*R_023[0]+aQin2*R_033[0];
			double QR_010000010013=Q_010000010*R_013[0]-a1Q_010000000_1*R_014[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000010010013=Q_000010010*R_013[0]-a1Q_000010000_1*R_014[0]-a1Q_000000010_1*R_023[0]+aQin2*R_024[0];
			double QR_000000020013=Q_000000020*R_013[0]-a1Q_000000010_2*R_014[0]+aQin2*R_015[0];
			double QR_020000000022=Q_020000000*R_022[0]-a1Q_010000000_2*R_122[0]+aQin2*R_222[0];
			double QR_010010000022=Q_010010000*R_022[0]-a1Q_010000000_1*R_032[0]-a1Q_000010000_1*R_122[0]+aQin2*R_132[0];
			double QR_000020000022=Q_000020000*R_022[0]-a1Q_000010000_2*R_032[0]+aQin2*R_042[0];
			double QR_010000010022=Q_010000010*R_022[0]-a1Q_010000000_1*R_023[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000010010022=Q_000010010*R_022[0]-a1Q_000010000_1*R_023[0]-a1Q_000000010_1*R_032[0]+aQin2*R_033[0];
			double QR_000000020022=Q_000000020*R_022[0]-a1Q_000000010_2*R_023[0]+aQin2*R_024[0];
			double QR_020000000031=Q_020000000*R_031[0]-a1Q_010000000_2*R_131[0]+aQin2*R_231[0];
			double QR_010010000031=Q_010010000*R_031[0]-a1Q_010000000_1*R_041[0]-a1Q_000010000_1*R_131[0]+aQin2*R_141[0];
			double QR_000020000031=Q_000020000*R_031[0]-a1Q_000010000_2*R_041[0]+aQin2*R_051[0];
			double QR_010000010031=Q_010000010*R_031[0]-a1Q_010000000_1*R_032[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000010010031=Q_000010010*R_031[0]-a1Q_000010000_1*R_032[0]-a1Q_000000010_1*R_041[0]+aQin2*R_042[0];
			double QR_000000020031=Q_000000020*R_031[0]-a1Q_000000010_2*R_032[0]+aQin2*R_033[0];
			double QR_020000000040=Q_020000000*R_040[0]-a1Q_010000000_2*R_140[0]+aQin2*R_240[0];
			double QR_010010000040=Q_010010000*R_040[0]-a1Q_010000000_1*R_050[0]-a1Q_000010000_1*R_140[0]+aQin2*R_150[0];
			double QR_000020000040=Q_000020000*R_040[0]-a1Q_000010000_2*R_050[0]+aQin2*R_060[0];
			double QR_010000010040=Q_010000010*R_040[0]-a1Q_010000000_1*R_041[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000010010040=Q_000010010*R_040[0]-a1Q_000010000_1*R_041[0]-a1Q_000000010_1*R_050[0]+aQin2*R_051[0];
			double QR_000000020040=Q_000000020*R_040[0]-a1Q_000000010_2*R_041[0]+aQin2*R_042[0];
			double QR_020000000103=Q_020000000*R_103[0]-a1Q_010000000_2*R_203[0]+aQin2*R_303[0];
			double QR_010010000103=Q_010010000*R_103[0]-a1Q_010000000_1*R_113[0]-a1Q_000010000_1*R_203[0]+aQin2*R_213[0];
			double QR_000020000103=Q_000020000*R_103[0]-a1Q_000010000_2*R_113[0]+aQin2*R_123[0];
			double QR_010000010103=Q_010000010*R_103[0]-a1Q_010000000_1*R_104[0]-a1Q_000000010_1*R_203[0]+aQin2*R_204[0];
			double QR_000010010103=Q_000010010*R_103[0]-a1Q_000010000_1*R_104[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			double QR_000000020103=Q_000000020*R_103[0]-a1Q_000000010_2*R_104[0]+aQin2*R_105[0];
			double QR_020000000112=Q_020000000*R_112[0]-a1Q_010000000_2*R_212[0]+aQin2*R_312[0];
			double QR_010010000112=Q_010010000*R_112[0]-a1Q_010000000_1*R_122[0]-a1Q_000010000_1*R_212[0]+aQin2*R_222[0];
			double QR_000020000112=Q_000020000*R_112[0]-a1Q_000010000_2*R_122[0]+aQin2*R_132[0];
			double QR_010000010112=Q_010000010*R_112[0]-a1Q_010000000_1*R_113[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000010010112=Q_000010010*R_112[0]-a1Q_000010000_1*R_113[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			double QR_000000020112=Q_000000020*R_112[0]-a1Q_000000010_2*R_113[0]+aQin2*R_114[0];
			double QR_020000000121=Q_020000000*R_121[0]-a1Q_010000000_2*R_221[0]+aQin2*R_321[0];
			double QR_010010000121=Q_010010000*R_121[0]-a1Q_010000000_1*R_131[0]-a1Q_000010000_1*R_221[0]+aQin2*R_231[0];
			double QR_000020000121=Q_000020000*R_121[0]-a1Q_000010000_2*R_131[0]+aQin2*R_141[0];
			double QR_010000010121=Q_010000010*R_121[0]-a1Q_010000000_1*R_122[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000010010121=Q_000010010*R_121[0]-a1Q_000010000_1*R_122[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			double QR_000000020121=Q_000000020*R_121[0]-a1Q_000000010_2*R_122[0]+aQin2*R_123[0];
			double QR_020000000130=Q_020000000*R_130[0]-a1Q_010000000_2*R_230[0]+aQin2*R_330[0];
			double QR_010010000130=Q_010010000*R_130[0]-a1Q_010000000_1*R_140[0]-a1Q_000010000_1*R_230[0]+aQin2*R_240[0];
			double QR_000020000130=Q_000020000*R_130[0]-a1Q_000010000_2*R_140[0]+aQin2*R_150[0];
			double QR_010000010130=Q_010000010*R_130[0]-a1Q_010000000_1*R_131[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000010010130=Q_000010010*R_130[0]-a1Q_000010000_1*R_131[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			double QR_000000020130=Q_000000020*R_130[0]-a1Q_000000010_2*R_131[0]+aQin2*R_132[0];
			double QR_020000000202=Q_020000000*R_202[0]-a1Q_010000000_2*R_302[0]+aQin2*R_402[0];
			double QR_010010000202=Q_010010000*R_202[0]-a1Q_010000000_1*R_212[0]-a1Q_000010000_1*R_302[0]+aQin2*R_312[0];
			double QR_000020000202=Q_000020000*R_202[0]-a1Q_000010000_2*R_212[0]+aQin2*R_222[0];
			double QR_010000010202=Q_010000010*R_202[0]-a1Q_010000000_1*R_203[0]-a1Q_000000010_1*R_302[0]+aQin2*R_303[0];
			double QR_000010010202=Q_000010010*R_202[0]-a1Q_000010000_1*R_203[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			double QR_000000020202=Q_000000020*R_202[0]-a1Q_000000010_2*R_203[0]+aQin2*R_204[0];
			double QR_020000000211=Q_020000000*R_211[0]-a1Q_010000000_2*R_311[0]+aQin2*R_411[0];
			double QR_010010000211=Q_010010000*R_211[0]-a1Q_010000000_1*R_221[0]-a1Q_000010000_1*R_311[0]+aQin2*R_321[0];
			double QR_000020000211=Q_000020000*R_211[0]-a1Q_000010000_2*R_221[0]+aQin2*R_231[0];
			double QR_010000010211=Q_010000010*R_211[0]-a1Q_010000000_1*R_212[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000010010211=Q_000010010*R_211[0]-a1Q_000010000_1*R_212[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			double QR_000000020211=Q_000000020*R_211[0]-a1Q_000000010_2*R_212[0]+aQin2*R_213[0];
			double QR_020000000220=Q_020000000*R_220[0]-a1Q_010000000_2*R_320[0]+aQin2*R_420[0];
			double QR_010010000220=Q_010010000*R_220[0]-a1Q_010000000_1*R_230[0]-a1Q_000010000_1*R_320[0]+aQin2*R_330[0];
			double QR_000020000220=Q_000020000*R_220[0]-a1Q_000010000_2*R_230[0]+aQin2*R_240[0];
			double QR_010000010220=Q_010000010*R_220[0]-a1Q_010000000_1*R_221[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000010010220=Q_000010010*R_220[0]-a1Q_000010000_1*R_221[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			double QR_000000020220=Q_000000020*R_220[0]-a1Q_000000010_2*R_221[0]+aQin2*R_222[0];
			double QR_020000000301=Q_020000000*R_301[0]-a1Q_010000000_2*R_401[0]+aQin2*R_501[0];
			double QR_010010000301=Q_010010000*R_301[0]-a1Q_010000000_1*R_311[0]-a1Q_000010000_1*R_401[0]+aQin2*R_411[0];
			double QR_000020000301=Q_000020000*R_301[0]-a1Q_000010000_2*R_311[0]+aQin2*R_321[0];
			double QR_010000010301=Q_010000010*R_301[0]-a1Q_010000000_1*R_302[0]-a1Q_000000010_1*R_401[0]+aQin2*R_402[0];
			double QR_000010010301=Q_000010010*R_301[0]-a1Q_000010000_1*R_302[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			double QR_000000020301=Q_000000020*R_301[0]-a1Q_000000010_2*R_302[0]+aQin2*R_303[0];
			double QR_020000000310=Q_020000000*R_310[0]-a1Q_010000000_2*R_410[0]+aQin2*R_510[0];
			double QR_010010000310=Q_010010000*R_310[0]-a1Q_010000000_1*R_320[0]-a1Q_000010000_1*R_410[0]+aQin2*R_420[0];
			double QR_000020000310=Q_000020000*R_310[0]-a1Q_000010000_2*R_320[0]+aQin2*R_330[0];
			double QR_010000010310=Q_010000010*R_310[0]-a1Q_010000000_1*R_311[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000010010310=Q_000010010*R_310[0]-a1Q_000010000_1*R_311[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			double QR_000000020310=Q_000000020*R_310[0]-a1Q_000000010_2*R_311[0]+aQin2*R_312[0];
			double QR_020000000400=Q_020000000*R_400[0]-a1Q_010000000_2*R_500[0]+aQin2*R_600[0];
			double QR_010010000400=Q_010010000*R_400[0]-a1Q_010000000_1*R_410[0]-a1Q_000010000_1*R_500[0]+aQin2*R_510[0];
			double QR_000020000400=Q_000020000*R_400[0]-a1Q_000010000_2*R_410[0]+aQin2*R_420[0];
			double QR_010000010400=Q_010000010*R_400[0]-a1Q_010000000_1*R_401[0]-a1Q_000000010_1*R_500[0]+aQin2*R_501[0];
			double QR_000010010400=Q_000010010*R_400[0]-a1Q_000010000_1*R_401[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			double QR_000000020400=Q_000000020*R_400[0]-a1Q_000000010_2*R_401[0]+aQin2*R_402[0];
		double Pd_002[3];
		double Pd_102[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		for(int i=0;i<3;i++){
			Pd_002[i]=aPin1+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=aPin1*(2.000000*Pd_001[i]);
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=aPin1*(Pd_002[i]+2.000000*Pd_011[i]);
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=aPin1*(0.500000*Pd_102[i]+Pd_111[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=aPin1*2.000000*(Pd_012[i]+Pd_021[i]);
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=aPin1*(Pd_112[i]+Pd_121[i]);
			}
			double P_022000000;
			double P_122000000;
			double P_222000000;
			double P_021001000;
			double P_121001000;
			double P_221001000;
			double P_020002000;
			double P_021000001;
			double P_121000001;
			double P_221000001;
			double P_020001001;
			double P_020000002;
			double P_012010000;
			double P_112010000;
			double P_212010000;
			double P_011011000;
			double P_011111000;
			double P_111011000;
			double P_111111000;
			double P_010012000;
			double P_010112000;
			double P_010212000;
			double P_011010001;
			double P_111010001;
			double P_010011001;
			double P_010111001;
			double P_010010002;
			double P_002020000;
			double P_001021000;
			double P_001121000;
			double P_001221000;
			double P_000022000;
			double P_000122000;
			double P_000222000;
			double P_001020001;
			double P_000021001;
			double P_000121001;
			double P_000221001;
			double P_000020002;
			double P_012000010;
			double P_112000010;
			double P_212000010;
			double P_011001010;
			double P_111001010;
			double P_010002010;
			double P_011000011;
			double P_011000111;
			double P_111000011;
			double P_111000111;
			double P_010001011;
			double P_010001111;
			double P_010000012;
			double P_010000112;
			double P_010000212;
			double P_002010010;
			double P_001011010;
			double P_001111010;
			double P_000012010;
			double P_000112010;
			double P_000212010;
			double P_001010011;
			double P_001010111;
			double P_000011011;
			double P_000011111;
			double P_000111011;
			double P_000111111;
			double P_000010012;
			double P_000010112;
			double P_000010212;
			double P_002000020;
			double P_001001020;
			double P_000002020;
			double P_001000021;
			double P_001000121;
			double P_001000221;
			double P_000001021;
			double P_000001121;
			double P_000001221;
			double P_000000022;
			double P_000000122;
			double P_000000222;
			double a2P_111000000_1;
			double a2P_111000000_2;
			double a1P_021000000_1;
			double a1P_121000000_1;
			double a1P_221000000_1;
			double a3P_000001000_1;
			double a3P_000001000_2;
			double a1P_020001000_1;
			double a1P_020001000_2;
			double a2P_020000000_1;
			double a1P_010002000_1;
			double a1P_010002000_2;
			double a2P_010001000_1;
			double a2P_010001000_4;
			double a2P_010001000_2;
			double a3P_010000000_1;
			double a3P_010000000_2;
			double a2P_000002000_1;
			double a3P_000000001_1;
			double a3P_000000001_2;
			double a1P_020000001_1;
			double a1P_020000001_2;
			double a1P_010001001_1;
			double a1P_010001001_2;
			double a2P_010000001_1;
			double a2P_010000001_2;
			double a2P_010000001_4;
			double a2P_000001001_1;
			double a1P_010000002_1;
			double a1P_010000002_2;
			double a2P_000000002_1;
			double a1P_012000000_1;
			double a1P_112000000_1;
			double a1P_212000000_1;
			double a3P_000010000_1;
			double a3P_000010000_2;
			double a2P_011000000_1;
			double a2P_000011000_1;
			double a2P_000111000_1;
			double a2P_000111000_2;
			double a1P_000012000_1;
			double a1P_000112000_1;
			double a1P_000212000_1;
			double a1P_011010000_1;
			double a1P_011000001_1;
			double a1P_111010000_1;
			double a1P_111000001_1;
			double a2P_000010001_1;
			double a2P_000010001_2;
			double a2P_000010001_4;
			double a1P_010011000_1;
			double a1P_010111000_1;
			double a1P_000011001_1;
			double a1P_000111001_1;
			double a1P_010010001_1;
			double a1P_010010001_2;
			double a2P_010010000_1;
			double a1P_000010002_1;
			double a1P_000010002_2;
			double a1P_002010000_1;
			double a1P_002010000_2;
			double a2P_002000000_1;
			double a1P_001020000_1;
			double a1P_001020000_2;
			double a2P_001010000_1;
			double a2P_001010000_4;
			double a2P_001010000_2;
			double a3P_001000000_1;
			double a3P_001000000_2;
			double a2P_000020000_1;
			double a1P_000021000_1;
			double a1P_000121000_1;
			double a1P_000221000_1;
			double a1P_001010001_1;
			double a1P_001010001_2;
			double a2P_001000001_1;
			double a1P_000020001_1;
			double a1P_000020001_2;
			double a3P_000000010_1;
			double a3P_000000010_2;
			double a1P_011001000_1;
			double a1P_011000010_1;
			double a1P_111001000_1;
			double a1P_111000010_1;
			double a2P_000001010_1;
			double a2P_000001010_2;
			double a2P_000001010_4;
			double a1P_010001010_1;
			double a1P_010001010_2;
			double a2P_010000010_1;
			double a1P_000002010_1;
			double a1P_000002010_2;
			double a2P_000000011_1;
			double a2P_000000111_1;
			double a2P_000000111_2;
			double a1P_010000011_1;
			double a1P_010000111_1;
			double a1P_000001011_1;
			double a1P_000001111_1;
			double a1P_000000012_1;
			double a1P_000000112_1;
			double a1P_000000212_1;
			double a1P_002000010_1;
			double a1P_002000010_2;
			double a1P_001010010_1;
			double a1P_001010010_2;
			double a2P_001000010_1;
			double a2P_001000010_2;
			double a2P_001000010_4;
			double a2P_000010010_1;
			double a1P_001011000_1;
			double a1P_001111000_1;
			double a1P_000011010_1;
			double a1P_000111010_1;
			double a1P_001000011_1;
			double a1P_001000111_1;
			double a1P_000010011_1;
			double a1P_000010111_1;
			double a1P_001000020_1;
			double a1P_001000020_2;
			double a2P_000000020_1;
			double a1P_001001010_1;
			double a1P_001001010_2;
			double a2P_001001000_1;
			double a1P_000001020_1;
			double a1P_000001020_2;
			double a1P_000000021_1;
			double a1P_000000121_1;
			double a1P_000000221_1;
			P_022000000=Pd_022[0];
			P_122000000=Pd_122[0];
			P_222000000=Pd_222[0];
			P_021001000=Pd_021[0]*Pd_001[1];
			P_121001000=Pd_121[0]*Pd_001[1];
			P_221001000=Pd_221[0]*Pd_001[1];
			P_020002000=Pd_020[0]*Pd_002[1];
			P_021000001=Pd_021[0]*Pd_001[2];
			P_121000001=Pd_121[0]*Pd_001[2];
			P_221000001=Pd_221[0]*Pd_001[2];
			P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
			P_020000002=Pd_020[0]*Pd_002[2];
			P_012010000=Pd_012[0]*Pd_010[1];
			P_112010000=Pd_112[0]*Pd_010[1];
			P_212010000=Pd_212[0]*Pd_010[1];
			P_011011000=Pd_011[0]*Pd_011[1];
			P_011111000=Pd_011[0]*Pd_111[1];
			P_111011000=Pd_111[0]*Pd_011[1];
			P_111111000=Pd_111[0]*Pd_111[1];
			P_010012000=Pd_010[0]*Pd_012[1];
			P_010112000=Pd_010[0]*Pd_112[1];
			P_010212000=Pd_010[0]*Pd_212[1];
			P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
			P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
			P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
			P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
			P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
			P_002020000=Pd_002[0]*Pd_020[1];
			P_001021000=Pd_001[0]*Pd_021[1];
			P_001121000=Pd_001[0]*Pd_121[1];
			P_001221000=Pd_001[0]*Pd_221[1];
			P_000022000=Pd_022[1];
			P_000122000=Pd_122[1];
			P_000222000=Pd_222[1];
			P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
			P_000021001=Pd_021[1]*Pd_001[2];
			P_000121001=Pd_121[1]*Pd_001[2];
			P_000221001=Pd_221[1]*Pd_001[2];
			P_000020002=Pd_020[1]*Pd_002[2];
			P_012000010=Pd_012[0]*Pd_010[2];
			P_112000010=Pd_112[0]*Pd_010[2];
			P_212000010=Pd_212[0]*Pd_010[2];
			P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
			P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
			P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
			P_011000011=Pd_011[0]*Pd_011[2];
			P_011000111=Pd_011[0]*Pd_111[2];
			P_111000011=Pd_111[0]*Pd_011[2];
			P_111000111=Pd_111[0]*Pd_111[2];
			P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
			P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
			P_010000012=Pd_010[0]*Pd_012[2];
			P_010000112=Pd_010[0]*Pd_112[2];
			P_010000212=Pd_010[0]*Pd_212[2];
			P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
			P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
			P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
			P_000012010=Pd_012[1]*Pd_010[2];
			P_000112010=Pd_112[1]*Pd_010[2];
			P_000212010=Pd_212[1]*Pd_010[2];
			P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
			P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
			P_000011011=Pd_011[1]*Pd_011[2];
			P_000011111=Pd_011[1]*Pd_111[2];
			P_000111011=Pd_111[1]*Pd_011[2];
			P_000111111=Pd_111[1]*Pd_111[2];
			P_000010012=Pd_010[1]*Pd_012[2];
			P_000010112=Pd_010[1]*Pd_112[2];
			P_000010212=Pd_010[1]*Pd_212[2];
			P_002000020=Pd_002[0]*Pd_020[2];
			P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
			P_000002020=Pd_002[1]*Pd_020[2];
			P_001000021=Pd_001[0]*Pd_021[2];
			P_001000121=Pd_001[0]*Pd_121[2];
			P_001000221=Pd_001[0]*Pd_221[2];
			P_000001021=Pd_001[1]*Pd_021[2];
			P_000001121=Pd_001[1]*Pd_121[2];
			P_000001221=Pd_001[1]*Pd_221[2];
			P_000000022=Pd_022[2];
			P_000000122=Pd_122[2];
			P_000000222=Pd_222[2];
			a2P_111000000_1=Pd_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=Pd_021[0];
			a1P_121000000_1=Pd_121[0];
			a1P_221000000_1=Pd_221[0];
			a3P_000001000_1=Pd_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=Pd_020[0]*Pd_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=Pd_020[0];
			a1P_010002000_1=Pd_010[0]*Pd_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=Pd_010[0]*Pd_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=Pd_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=Pd_002[1];
			a3P_000000001_1=Pd_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=Pd_020[0]*Pd_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=Pd_010[0]*Pd_001[1]*Pd_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=Pd_010[0]*Pd_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=Pd_001[1]*Pd_001[2];
			a1P_010000002_1=Pd_010[0]*Pd_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=Pd_002[2];
			a1P_012000000_1=Pd_012[0];
			a1P_112000000_1=Pd_112[0];
			a1P_212000000_1=Pd_212[0];
			a3P_000010000_1=Pd_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=Pd_011[0];
			a2P_000011000_1=Pd_011[1];
			a2P_000111000_1=Pd_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=Pd_012[1];
			a1P_000112000_1=Pd_112[1];
			a1P_000212000_1=Pd_212[1];
			a1P_011010000_1=Pd_011[0]*Pd_010[1];
			a1P_011000001_1=Pd_011[0]*Pd_001[2];
			a1P_111010000_1=Pd_111[0]*Pd_010[1];
			a1P_111000001_1=Pd_111[0]*Pd_001[2];
			a2P_000010001_1=Pd_010[1]*Pd_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=Pd_010[0]*Pd_011[1];
			a1P_010111000_1=Pd_010[0]*Pd_111[1];
			a1P_000011001_1=Pd_011[1]*Pd_001[2];
			a1P_000111001_1=Pd_111[1]*Pd_001[2];
			a1P_010010001_1=Pd_010[0]*Pd_010[1]*Pd_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010002_1=Pd_010[1]*Pd_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=Pd_002[0]*Pd_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=Pd_002[0];
			a1P_001020000_1=Pd_001[0]*Pd_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=Pd_001[0]*Pd_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=Pd_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=Pd_020[1];
			a1P_000021000_1=Pd_021[1];
			a1P_000121000_1=Pd_121[1];
			a1P_000221000_1=Pd_221[1];
			a1P_001010001_1=Pd_001[0]*Pd_010[1]*Pd_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=Pd_001[0]*Pd_001[2];
			a1P_000020001_1=Pd_020[1]*Pd_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=Pd_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=Pd_011[0]*Pd_001[1];
			a1P_011000010_1=Pd_011[0]*Pd_010[2];
			a1P_111001000_1=Pd_111[0]*Pd_001[1];
			a1P_111000010_1=Pd_111[0]*Pd_010[2];
			a2P_000001010_1=Pd_001[1]*Pd_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=Pd_010[0]*Pd_001[1]*Pd_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000002010_1=Pd_002[1]*Pd_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=Pd_011[2];
			a2P_000000111_1=Pd_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=Pd_010[0]*Pd_011[2];
			a1P_010000111_1=Pd_010[0]*Pd_111[2];
			a1P_000001011_1=Pd_001[1]*Pd_011[2];
			a1P_000001111_1=Pd_001[1]*Pd_111[2];
			a1P_000000012_1=Pd_012[2];
			a1P_000000112_1=Pd_112[2];
			a1P_000000212_1=Pd_212[2];
			a1P_002000010_1=Pd_002[0]*Pd_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=Pd_001[0]*Pd_010[1]*Pd_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=Pd_001[0]*Pd_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_001011000_1=Pd_001[0]*Pd_011[1];
			a1P_001111000_1=Pd_001[0]*Pd_111[1];
			a1P_000011010_1=Pd_011[1]*Pd_010[2];
			a1P_000111010_1=Pd_111[1]*Pd_010[2];
			a1P_001000011_1=Pd_001[0]*Pd_011[2];
			a1P_001000111_1=Pd_001[0]*Pd_111[2];
			a1P_000010011_1=Pd_010[1]*Pd_011[2];
			a1P_000010111_1=Pd_010[1]*Pd_111[2];
			a1P_001000020_1=Pd_001[0]*Pd_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=Pd_020[2];
			a1P_001001010_1=Pd_001[0]*Pd_001[1]*Pd_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=Pd_001[0]*Pd_001[1];
			a1P_000001020_1=Pd_001[1]*Pd_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=Pd_021[2];
			a1P_000000121_1=Pd_121[2];
			a1P_000000221_1=Pd_221[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_022000000*QR_020000000000+P_122000000*QR_020000000100+P_222000000*QR_020000000200+a2P_111000000_2*QR_020000000300+aPin4*QR_020000000400);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_022000000*QR_010010000000+P_122000000*QR_010010000100+P_222000000*QR_010010000200+a2P_111000000_2*QR_010010000300+aPin4*QR_010010000400);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_022000000*QR_000020000000+P_122000000*QR_000020000100+P_222000000*QR_000020000200+a2P_111000000_2*QR_000020000300+aPin4*QR_000020000400);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(P_022000000*QR_010000010000+P_122000000*QR_010000010100+P_222000000*QR_010000010200+a2P_111000000_2*QR_010000010300+aPin4*QR_010000010400);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(P_022000000*QR_000010010000+P_122000000*QR_000010010100+P_222000000*QR_000010010200+a2P_111000000_2*QR_000010010300+aPin4*QR_000010010400);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(P_022000000*QR_000000020000+P_122000000*QR_000000020100+P_222000000*QR_000000020200+a2P_111000000_2*QR_000000020300+aPin4*QR_000000020400);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_021001000*QR_020000000000+a1P_021000000_1*QR_020000000010+P_121001000*QR_020000000100+a1P_121000000_1*QR_020000000110+P_221001000*QR_020000000200+a1P_221000000_1*QR_020000000210+a3P_000001000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_021001000*QR_010010000000+a1P_021000000_1*QR_010010000010+P_121001000*QR_010010000100+a1P_121000000_1*QR_010010000110+P_221001000*QR_010010000200+a1P_221000000_1*QR_010010000210+a3P_000001000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_021001000*QR_000020000000+a1P_021000000_1*QR_000020000010+P_121001000*QR_000020000100+a1P_121000000_1*QR_000020000110+P_221001000*QR_000020000200+a1P_221000000_1*QR_000020000210+a3P_000001000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(P_021001000*QR_010000010000+a1P_021000000_1*QR_010000010010+P_121001000*QR_010000010100+a1P_121000000_1*QR_010000010110+P_221001000*QR_010000010200+a1P_221000000_1*QR_010000010210+a3P_000001000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(P_021001000*QR_000010010000+a1P_021000000_1*QR_000010010010+P_121001000*QR_000010010100+a1P_121000000_1*QR_000010010110+P_221001000*QR_000010010200+a1P_221000000_1*QR_000010010210+a3P_000001000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(P_021001000*QR_000000020000+a1P_021000000_1*QR_000000020010+P_121001000*QR_000000020100+a1P_121000000_1*QR_000000020110+P_221001000*QR_000000020200+a1P_221000000_1*QR_000000020210+a3P_000001000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_020002000*QR_020000000000+a1P_020001000_2*QR_020000000010+a2P_020000000_1*QR_020000000020+a1P_010002000_2*QR_020000000100+a2P_010001000_4*QR_020000000110+a3P_010000000_2*QR_020000000120+a2P_000002000_1*QR_020000000200+a3P_000001000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_020002000*QR_010010000000+a1P_020001000_2*QR_010010000010+a2P_020000000_1*QR_010010000020+a1P_010002000_2*QR_010010000100+a2P_010001000_4*QR_010010000110+a3P_010000000_2*QR_010010000120+a2P_000002000_1*QR_010010000200+a3P_000001000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020002000*QR_000020000000+a1P_020001000_2*QR_000020000010+a2P_020000000_1*QR_000020000020+a1P_010002000_2*QR_000020000100+a2P_010001000_4*QR_000020000110+a3P_010000000_2*QR_000020000120+a2P_000002000_1*QR_000020000200+a3P_000001000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(P_020002000*QR_010000010000+a1P_020001000_2*QR_010000010010+a2P_020000000_1*QR_010000010020+a1P_010002000_2*QR_010000010100+a2P_010001000_4*QR_010000010110+a3P_010000000_2*QR_010000010120+a2P_000002000_1*QR_010000010200+a3P_000001000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(P_020002000*QR_000010010000+a1P_020001000_2*QR_000010010010+a2P_020000000_1*QR_000010010020+a1P_010002000_2*QR_000010010100+a2P_010001000_4*QR_000010010110+a3P_010000000_2*QR_000010010120+a2P_000002000_1*QR_000010010200+a3P_000001000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(P_020002000*QR_000000020000+a1P_020001000_2*QR_000000020010+a2P_020000000_1*QR_000000020020+a1P_010002000_2*QR_000000020100+a2P_010001000_4*QR_000000020110+a3P_010000000_2*QR_000000020120+a2P_000002000_1*QR_000000020200+a3P_000001000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_021000001*QR_020000000000+a1P_021000000_1*QR_020000000001+P_121000001*QR_020000000100+a1P_121000000_1*QR_020000000101+P_221000001*QR_020000000200+a1P_221000000_1*QR_020000000201+a3P_000000001_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_021000001*QR_010010000000+a1P_021000000_1*QR_010010000001+P_121000001*QR_010010000100+a1P_121000000_1*QR_010010000101+P_221000001*QR_010010000200+a1P_221000000_1*QR_010010000201+a3P_000000001_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_021000001*QR_000020000000+a1P_021000000_1*QR_000020000001+P_121000001*QR_000020000100+a1P_121000000_1*QR_000020000101+P_221000001*QR_000020000200+a1P_221000000_1*QR_000020000201+a3P_000000001_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(P_021000001*QR_010000010000+a1P_021000000_1*QR_010000010001+P_121000001*QR_010000010100+a1P_121000000_1*QR_010000010101+P_221000001*QR_010000010200+a1P_221000000_1*QR_010000010201+a3P_000000001_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(P_021000001*QR_000010010000+a1P_021000000_1*QR_000010010001+P_121000001*QR_000010010100+a1P_121000000_1*QR_000010010101+P_221000001*QR_000010010200+a1P_221000000_1*QR_000010010201+a3P_000000001_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(P_021000001*QR_000000020000+a1P_021000000_1*QR_000000020001+P_121000001*QR_000000020100+a1P_121000000_1*QR_000000020101+P_221000001*QR_000000020200+a1P_221000000_1*QR_000000020201+a3P_000000001_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_020001001*QR_020000000000+a1P_020001000_1*QR_020000000001+a1P_020000001_1*QR_020000000010+a2P_020000000_1*QR_020000000011+a1P_010001001_2*QR_020000000100+a2P_010001000_2*QR_020000000101+a2P_010000001_2*QR_020000000110+a3P_010000000_2*QR_020000000111+a2P_000001001_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_020001001*QR_010010000000+a1P_020001000_1*QR_010010000001+a1P_020000001_1*QR_010010000010+a2P_020000000_1*QR_010010000011+a1P_010001001_2*QR_010010000100+a2P_010001000_2*QR_010010000101+a2P_010000001_2*QR_010010000110+a3P_010000000_2*QR_010010000111+a2P_000001001_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020001001*QR_000020000000+a1P_020001000_1*QR_000020000001+a1P_020000001_1*QR_000020000010+a2P_020000000_1*QR_000020000011+a1P_010001001_2*QR_000020000100+a2P_010001000_2*QR_000020000101+a2P_010000001_2*QR_000020000110+a3P_010000000_2*QR_000020000111+a2P_000001001_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(P_020001001*QR_010000010000+a1P_020001000_1*QR_010000010001+a1P_020000001_1*QR_010000010010+a2P_020000000_1*QR_010000010011+a1P_010001001_2*QR_010000010100+a2P_010001000_2*QR_010000010101+a2P_010000001_2*QR_010000010110+a3P_010000000_2*QR_010000010111+a2P_000001001_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(P_020001001*QR_000010010000+a1P_020001000_1*QR_000010010001+a1P_020000001_1*QR_000010010010+a2P_020000000_1*QR_000010010011+a1P_010001001_2*QR_000010010100+a2P_010001000_2*QR_000010010101+a2P_010000001_2*QR_000010010110+a3P_010000000_2*QR_000010010111+a2P_000001001_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(P_020001001*QR_000000020000+a1P_020001000_1*QR_000000020001+a1P_020000001_1*QR_000000020010+a2P_020000000_1*QR_000000020011+a1P_010001001_2*QR_000000020100+a2P_010001000_2*QR_000000020101+a2P_010000001_2*QR_000000020110+a3P_010000000_2*QR_000000020111+a2P_000001001_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_020000002*QR_020000000000+a1P_020000001_2*QR_020000000001+a2P_020000000_1*QR_020000000002+a1P_010000002_2*QR_020000000100+a2P_010000001_4*QR_020000000101+a3P_010000000_2*QR_020000000102+a2P_000000002_1*QR_020000000200+a3P_000000001_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_020000002*QR_010010000000+a1P_020000001_2*QR_010010000001+a2P_020000000_1*QR_010010000002+a1P_010000002_2*QR_010010000100+a2P_010000001_4*QR_010010000101+a3P_010000000_2*QR_010010000102+a2P_000000002_1*QR_010010000200+a3P_000000001_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020000002*QR_000020000000+a1P_020000001_2*QR_000020000001+a2P_020000000_1*QR_000020000002+a1P_010000002_2*QR_000020000100+a2P_010000001_4*QR_000020000101+a3P_010000000_2*QR_000020000102+a2P_000000002_1*QR_000020000200+a3P_000000001_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(P_020000002*QR_010000010000+a1P_020000001_2*QR_010000010001+a2P_020000000_1*QR_010000010002+a1P_010000002_2*QR_010000010100+a2P_010000001_4*QR_010000010101+a3P_010000000_2*QR_010000010102+a2P_000000002_1*QR_010000010200+a3P_000000001_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(P_020000002*QR_000010010000+a1P_020000001_2*QR_000010010001+a2P_020000000_1*QR_000010010002+a1P_010000002_2*QR_000010010100+a2P_010000001_4*QR_000010010101+a3P_010000000_2*QR_000010010102+a2P_000000002_1*QR_000010010200+a3P_000000001_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(P_020000002*QR_000000020000+a1P_020000001_2*QR_000000020001+a2P_020000000_1*QR_000000020002+a1P_010000002_2*QR_000000020100+a2P_010000001_4*QR_000000020101+a3P_010000000_2*QR_000000020102+a2P_000000002_1*QR_000000020200+a3P_000000001_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_012010000*QR_020000000000+a1P_012000000_1*QR_020000000010+P_112010000*QR_020000000100+a1P_112000000_1*QR_020000000110+P_212010000*QR_020000000200+a1P_212000000_1*QR_020000000210+a3P_000010000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_012010000*QR_010010000000+a1P_012000000_1*QR_010010000010+P_112010000*QR_010010000100+a1P_112000000_1*QR_010010000110+P_212010000*QR_010010000200+a1P_212000000_1*QR_010010000210+a3P_000010000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_012010000*QR_000020000000+a1P_012000000_1*QR_000020000010+P_112010000*QR_000020000100+a1P_112000000_1*QR_000020000110+P_212010000*QR_000020000200+a1P_212000000_1*QR_000020000210+a3P_000010000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(P_012010000*QR_010000010000+a1P_012000000_1*QR_010000010010+P_112010000*QR_010000010100+a1P_112000000_1*QR_010000010110+P_212010000*QR_010000010200+a1P_212000000_1*QR_010000010210+a3P_000010000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(P_012010000*QR_000010010000+a1P_012000000_1*QR_000010010010+P_112010000*QR_000010010100+a1P_112000000_1*QR_000010010110+P_212010000*QR_000010010200+a1P_212000000_1*QR_000010010210+a3P_000010000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(P_012010000*QR_000000020000+a1P_012000000_1*QR_000000020010+P_112010000*QR_000000020100+a1P_112000000_1*QR_000000020110+P_212010000*QR_000000020200+a1P_212000000_1*QR_000000020210+a3P_000010000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_011011000*QR_020000000000+P_011111000*QR_020000000010+a2P_011000000_1*QR_020000000020+P_111011000*QR_020000000100+P_111111000*QR_020000000110+a2P_111000000_1*QR_020000000120+a2P_000011000_1*QR_020000000200+a2P_000111000_1*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_011011000*QR_010010000000+P_011111000*QR_010010000010+a2P_011000000_1*QR_010010000020+P_111011000*QR_010010000100+P_111111000*QR_010010000110+a2P_111000000_1*QR_010010000120+a2P_000011000_1*QR_010010000200+a2P_000111000_1*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_011011000*QR_000020000000+P_011111000*QR_000020000010+a2P_011000000_1*QR_000020000020+P_111011000*QR_000020000100+P_111111000*QR_000020000110+a2P_111000000_1*QR_000020000120+a2P_000011000_1*QR_000020000200+a2P_000111000_1*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(P_011011000*QR_010000010000+P_011111000*QR_010000010010+a2P_011000000_1*QR_010000010020+P_111011000*QR_010000010100+P_111111000*QR_010000010110+a2P_111000000_1*QR_010000010120+a2P_000011000_1*QR_010000010200+a2P_000111000_1*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(P_011011000*QR_000010010000+P_011111000*QR_000010010010+a2P_011000000_1*QR_000010010020+P_111011000*QR_000010010100+P_111111000*QR_000010010110+a2P_111000000_1*QR_000010010120+a2P_000011000_1*QR_000010010200+a2P_000111000_1*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(P_011011000*QR_000000020000+P_011111000*QR_000000020010+a2P_011000000_1*QR_000000020020+P_111011000*QR_000000020100+P_111111000*QR_000000020110+a2P_111000000_1*QR_000000020120+a2P_000011000_1*QR_000000020200+a2P_000111000_1*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_010012000*QR_020000000000+P_010112000*QR_020000000010+P_010212000*QR_020000000020+a3P_010000000_1*QR_020000000030+a1P_000012000_1*QR_020000000100+a1P_000112000_1*QR_020000000110+a1P_000212000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_010012000*QR_010010000000+P_010112000*QR_010010000010+P_010212000*QR_010010000020+a3P_010000000_1*QR_010010000030+a1P_000012000_1*QR_010010000100+a1P_000112000_1*QR_010010000110+a1P_000212000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010012000*QR_000020000000+P_010112000*QR_000020000010+P_010212000*QR_000020000020+a3P_010000000_1*QR_000020000030+a1P_000012000_1*QR_000020000100+a1P_000112000_1*QR_000020000110+a1P_000212000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(P_010012000*QR_010000010000+P_010112000*QR_010000010010+P_010212000*QR_010000010020+a3P_010000000_1*QR_010000010030+a1P_000012000_1*QR_010000010100+a1P_000112000_1*QR_010000010110+a1P_000212000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(P_010012000*QR_000010010000+P_010112000*QR_000010010010+P_010212000*QR_000010010020+a3P_010000000_1*QR_000010010030+a1P_000012000_1*QR_000010010100+a1P_000112000_1*QR_000010010110+a1P_000212000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(P_010012000*QR_000000020000+P_010112000*QR_000000020010+P_010212000*QR_000000020020+a3P_010000000_1*QR_000000020030+a1P_000012000_1*QR_000000020100+a1P_000112000_1*QR_000000020110+a1P_000212000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_011010001*QR_020000000000+a1P_011010000_1*QR_020000000001+a1P_011000001_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111010001*QR_020000000100+a1P_111010000_1*QR_020000000101+a1P_111000001_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000010001_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_011010001*QR_010010000000+a1P_011010000_1*QR_010010000001+a1P_011000001_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111010001*QR_010010000100+a1P_111010000_1*QR_010010000101+a1P_111000001_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000010001_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_011010001*QR_000020000000+a1P_011010000_1*QR_000020000001+a1P_011000001_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111010001*QR_000020000100+a1P_111010000_1*QR_000020000101+a1P_111000001_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000010001_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(P_011010001*QR_010000010000+a1P_011010000_1*QR_010000010001+a1P_011000001_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111010001*QR_010000010100+a1P_111010000_1*QR_010000010101+a1P_111000001_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000010001_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(P_011010001*QR_000010010000+a1P_011010000_1*QR_000010010001+a1P_011000001_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111010001*QR_000010010100+a1P_111010000_1*QR_000010010101+a1P_111000001_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000010001_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(P_011010001*QR_000000020000+a1P_011010000_1*QR_000000020001+a1P_011000001_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111010001*QR_000000020100+a1P_111010000_1*QR_000000020101+a1P_111000001_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000010001_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_010011001*QR_020000000000+a1P_010011000_1*QR_020000000001+P_010111001*QR_020000000010+a1P_010111000_1*QR_020000000011+a2P_010000001_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000011001_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111001_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_010011001*QR_010010000000+a1P_010011000_1*QR_010010000001+P_010111001*QR_010010000010+a1P_010111000_1*QR_010010000011+a2P_010000001_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000011001_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111001_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010011001*QR_000020000000+a1P_010011000_1*QR_000020000001+P_010111001*QR_000020000010+a1P_010111000_1*QR_000020000011+a2P_010000001_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000011001_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111001_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(P_010011001*QR_010000010000+a1P_010011000_1*QR_010000010001+P_010111001*QR_010000010010+a1P_010111000_1*QR_010000010011+a2P_010000001_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000011001_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111001_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(P_010011001*QR_000010010000+a1P_010011000_1*QR_000010010001+P_010111001*QR_000010010010+a1P_010111000_1*QR_000010010011+a2P_010000001_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000011001_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111001_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(P_010011001*QR_000000020000+a1P_010011000_1*QR_000000020001+P_010111001*QR_000000020010+a1P_010111000_1*QR_000000020011+a2P_010000001_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000011001_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111001_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_010010002*QR_020000000000+a1P_010010001_2*QR_020000000001+a2P_010010000_1*QR_020000000002+a1P_010000002_1*QR_020000000010+a2P_010000001_2*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000010002_1*QR_020000000100+a2P_000010001_2*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000002_1*QR_020000000110+a3P_000000001_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_010010002*QR_010010000000+a1P_010010001_2*QR_010010000001+a2P_010010000_1*QR_010010000002+a1P_010000002_1*QR_010010000010+a2P_010000001_2*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000010002_1*QR_010010000100+a2P_000010001_2*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000002_1*QR_010010000110+a3P_000000001_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010010002*QR_000020000000+a1P_010010001_2*QR_000020000001+a2P_010010000_1*QR_000020000002+a1P_010000002_1*QR_000020000010+a2P_010000001_2*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000010002_1*QR_000020000100+a2P_000010001_2*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000002_1*QR_000020000110+a3P_000000001_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(P_010010002*QR_010000010000+a1P_010010001_2*QR_010000010001+a2P_010010000_1*QR_010000010002+a1P_010000002_1*QR_010000010010+a2P_010000001_2*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000010002_1*QR_010000010100+a2P_000010001_2*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000002_1*QR_010000010110+a3P_000000001_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(P_010010002*QR_000010010000+a1P_010010001_2*QR_000010010001+a2P_010010000_1*QR_000010010002+a1P_010000002_1*QR_000010010010+a2P_010000001_2*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000010002_1*QR_000010010100+a2P_000010001_2*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000002_1*QR_000010010110+a3P_000000001_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(P_010010002*QR_000000020000+a1P_010010001_2*QR_000000020001+a2P_010010000_1*QR_000000020002+a1P_010000002_1*QR_000000020010+a2P_010000001_2*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000010002_1*QR_000000020100+a2P_000010001_2*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000002_1*QR_000000020110+a3P_000000001_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_002020000*QR_020000000000+a1P_002010000_2*QR_020000000010+a2P_002000000_1*QR_020000000020+a1P_001020000_2*QR_020000000100+a2P_001010000_4*QR_020000000110+a3P_001000000_2*QR_020000000120+a2P_000020000_1*QR_020000000200+a3P_000010000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_002020000*QR_010010000000+a1P_002010000_2*QR_010010000010+a2P_002000000_1*QR_010010000020+a1P_001020000_2*QR_010010000100+a2P_001010000_4*QR_010010000110+a3P_001000000_2*QR_010010000120+a2P_000020000_1*QR_010010000200+a3P_000010000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_002020000*QR_000020000000+a1P_002010000_2*QR_000020000010+a2P_002000000_1*QR_000020000020+a1P_001020000_2*QR_000020000100+a2P_001010000_4*QR_000020000110+a3P_001000000_2*QR_000020000120+a2P_000020000_1*QR_000020000200+a3P_000010000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(P_002020000*QR_010000010000+a1P_002010000_2*QR_010000010010+a2P_002000000_1*QR_010000010020+a1P_001020000_2*QR_010000010100+a2P_001010000_4*QR_010000010110+a3P_001000000_2*QR_010000010120+a2P_000020000_1*QR_010000010200+a3P_000010000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(P_002020000*QR_000010010000+a1P_002010000_2*QR_000010010010+a2P_002000000_1*QR_000010010020+a1P_001020000_2*QR_000010010100+a2P_001010000_4*QR_000010010110+a3P_001000000_2*QR_000010010120+a2P_000020000_1*QR_000010010200+a3P_000010000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(P_002020000*QR_000000020000+a1P_002010000_2*QR_000000020010+a2P_002000000_1*QR_000000020020+a1P_001020000_2*QR_000000020100+a2P_001010000_4*QR_000000020110+a3P_001000000_2*QR_000000020120+a2P_000020000_1*QR_000000020200+a3P_000010000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_001021000*QR_020000000000+P_001121000*QR_020000000010+P_001221000*QR_020000000020+a3P_001000000_1*QR_020000000030+a1P_000021000_1*QR_020000000100+a1P_000121000_1*QR_020000000110+a1P_000221000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_001021000*QR_010010000000+P_001121000*QR_010010000010+P_001221000*QR_010010000020+a3P_001000000_1*QR_010010000030+a1P_000021000_1*QR_010010000100+a1P_000121000_1*QR_010010000110+a1P_000221000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_001021000*QR_000020000000+P_001121000*QR_000020000010+P_001221000*QR_000020000020+a3P_001000000_1*QR_000020000030+a1P_000021000_1*QR_000020000100+a1P_000121000_1*QR_000020000110+a1P_000221000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(P_001021000*QR_010000010000+P_001121000*QR_010000010010+P_001221000*QR_010000010020+a3P_001000000_1*QR_010000010030+a1P_000021000_1*QR_010000010100+a1P_000121000_1*QR_010000010110+a1P_000221000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(P_001021000*QR_000010010000+P_001121000*QR_000010010010+P_001221000*QR_000010010020+a3P_001000000_1*QR_000010010030+a1P_000021000_1*QR_000010010100+a1P_000121000_1*QR_000010010110+a1P_000221000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(P_001021000*QR_000000020000+P_001121000*QR_000000020010+P_001221000*QR_000000020020+a3P_001000000_1*QR_000000020030+a1P_000021000_1*QR_000000020100+a1P_000121000_1*QR_000000020110+a1P_000221000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_000022000*QR_020000000000+P_000122000*QR_020000000010+P_000222000*QR_020000000020+a2P_000111000_2*QR_020000000030+aPin4*QR_020000000040);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_000022000*QR_010010000000+P_000122000*QR_010010000010+P_000222000*QR_010010000020+a2P_000111000_2*QR_010010000030+aPin4*QR_010010000040);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000022000*QR_000020000000+P_000122000*QR_000020000010+P_000222000*QR_000020000020+a2P_000111000_2*QR_000020000030+aPin4*QR_000020000040);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(P_000022000*QR_010000010000+P_000122000*QR_010000010010+P_000222000*QR_010000010020+a2P_000111000_2*QR_010000010030+aPin4*QR_010000010040);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(P_000022000*QR_000010010000+P_000122000*QR_000010010010+P_000222000*QR_000010010020+a2P_000111000_2*QR_000010010030+aPin4*QR_000010010040);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(P_000022000*QR_000000020000+P_000122000*QR_000000020010+P_000222000*QR_000000020020+a2P_000111000_2*QR_000000020030+aPin4*QR_000000020040);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_001020001*QR_020000000000+a1P_001020000_1*QR_020000000001+a1P_001010001_2*QR_020000000010+a2P_001010000_2*QR_020000000011+a2P_001000001_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000020001_1*QR_020000000100+a2P_000020000_1*QR_020000000101+a2P_000010001_2*QR_020000000110+a3P_000010000_2*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_001020001*QR_010010000000+a1P_001020000_1*QR_010010000001+a1P_001010001_2*QR_010010000010+a2P_001010000_2*QR_010010000011+a2P_001000001_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000020001_1*QR_010010000100+a2P_000020000_1*QR_010010000101+a2P_000010001_2*QR_010010000110+a3P_000010000_2*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_001020001*QR_000020000000+a1P_001020000_1*QR_000020000001+a1P_001010001_2*QR_000020000010+a2P_001010000_2*QR_000020000011+a2P_001000001_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000020001_1*QR_000020000100+a2P_000020000_1*QR_000020000101+a2P_000010001_2*QR_000020000110+a3P_000010000_2*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(P_001020001*QR_010000010000+a1P_001020000_1*QR_010000010001+a1P_001010001_2*QR_010000010010+a2P_001010000_2*QR_010000010011+a2P_001000001_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000020001_1*QR_010000010100+a2P_000020000_1*QR_010000010101+a2P_000010001_2*QR_010000010110+a3P_000010000_2*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(P_001020001*QR_000010010000+a1P_001020000_1*QR_000010010001+a1P_001010001_2*QR_000010010010+a2P_001010000_2*QR_000010010011+a2P_001000001_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000020001_1*QR_000010010100+a2P_000020000_1*QR_000010010101+a2P_000010001_2*QR_000010010110+a3P_000010000_2*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(P_001020001*QR_000000020000+a1P_001020000_1*QR_000000020001+a1P_001010001_2*QR_000000020010+a2P_001010000_2*QR_000000020011+a2P_001000001_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000020001_1*QR_000000020100+a2P_000020000_1*QR_000000020101+a2P_000010001_2*QR_000000020110+a3P_000010000_2*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_000021001*QR_020000000000+a1P_000021000_1*QR_020000000001+P_000121001*QR_020000000010+a1P_000121000_1*QR_020000000011+P_000221001*QR_020000000020+a1P_000221000_1*QR_020000000021+a3P_000000001_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_000021001*QR_010010000000+a1P_000021000_1*QR_010010000001+P_000121001*QR_010010000010+a1P_000121000_1*QR_010010000011+P_000221001*QR_010010000020+a1P_000221000_1*QR_010010000021+a3P_000000001_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000021001*QR_000020000000+a1P_000021000_1*QR_000020000001+P_000121001*QR_000020000010+a1P_000121000_1*QR_000020000011+P_000221001*QR_000020000020+a1P_000221000_1*QR_000020000021+a3P_000000001_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(P_000021001*QR_010000010000+a1P_000021000_1*QR_010000010001+P_000121001*QR_010000010010+a1P_000121000_1*QR_010000010011+P_000221001*QR_010000010020+a1P_000221000_1*QR_010000010021+a3P_000000001_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(P_000021001*QR_000010010000+a1P_000021000_1*QR_000010010001+P_000121001*QR_000010010010+a1P_000121000_1*QR_000010010011+P_000221001*QR_000010010020+a1P_000221000_1*QR_000010010021+a3P_000000001_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(P_000021001*QR_000000020000+a1P_000021000_1*QR_000000020001+P_000121001*QR_000000020010+a1P_000121000_1*QR_000000020011+P_000221001*QR_000000020020+a1P_000221000_1*QR_000000020021+a3P_000000001_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_000020002*QR_020000000000+a1P_000020001_2*QR_020000000001+a2P_000020000_1*QR_020000000002+a1P_000010002_2*QR_020000000010+a2P_000010001_4*QR_020000000011+a3P_000010000_2*QR_020000000012+a2P_000000002_1*QR_020000000020+a3P_000000001_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_000020002*QR_010010000000+a1P_000020001_2*QR_010010000001+a2P_000020000_1*QR_010010000002+a1P_000010002_2*QR_010010000010+a2P_000010001_4*QR_010010000011+a3P_000010000_2*QR_010010000012+a2P_000000002_1*QR_010010000020+a3P_000000001_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000020002*QR_000020000000+a1P_000020001_2*QR_000020000001+a2P_000020000_1*QR_000020000002+a1P_000010002_2*QR_000020000010+a2P_000010001_4*QR_000020000011+a3P_000010000_2*QR_000020000012+a2P_000000002_1*QR_000020000020+a3P_000000001_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(P_000020002*QR_010000010000+a1P_000020001_2*QR_010000010001+a2P_000020000_1*QR_010000010002+a1P_000010002_2*QR_010000010010+a2P_000010001_4*QR_010000010011+a3P_000010000_2*QR_010000010012+a2P_000000002_1*QR_010000010020+a3P_000000001_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(P_000020002*QR_000010010000+a1P_000020001_2*QR_000010010001+a2P_000020000_1*QR_000010010002+a1P_000010002_2*QR_000010010010+a2P_000010001_4*QR_000010010011+a3P_000010000_2*QR_000010010012+a2P_000000002_1*QR_000010010020+a3P_000000001_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(P_000020002*QR_000000020000+a1P_000020001_2*QR_000000020001+a2P_000020000_1*QR_000000020002+a1P_000010002_2*QR_000000020010+a2P_000010001_4*QR_000000020011+a3P_000010000_2*QR_000000020012+a2P_000000002_1*QR_000000020020+a3P_000000001_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_012000010*QR_020000000000+a1P_012000000_1*QR_020000000001+P_112000010*QR_020000000100+a1P_112000000_1*QR_020000000101+P_212000010*QR_020000000200+a1P_212000000_1*QR_020000000201+a3P_000000010_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_012000010*QR_010010000000+a1P_012000000_1*QR_010010000001+P_112000010*QR_010010000100+a1P_112000000_1*QR_010010000101+P_212000010*QR_010010000200+a1P_212000000_1*QR_010010000201+a3P_000000010_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_012000010*QR_000020000000+a1P_012000000_1*QR_000020000001+P_112000010*QR_000020000100+a1P_112000000_1*QR_000020000101+P_212000010*QR_000020000200+a1P_212000000_1*QR_000020000201+a3P_000000010_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(P_012000010*QR_010000010000+a1P_012000000_1*QR_010000010001+P_112000010*QR_010000010100+a1P_112000000_1*QR_010000010101+P_212000010*QR_010000010200+a1P_212000000_1*QR_010000010201+a3P_000000010_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(P_012000010*QR_000010010000+a1P_012000000_1*QR_000010010001+P_112000010*QR_000010010100+a1P_112000000_1*QR_000010010101+P_212000010*QR_000010010200+a1P_212000000_1*QR_000010010201+a3P_000000010_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(P_012000010*QR_000000020000+a1P_012000000_1*QR_000000020001+P_112000010*QR_000000020100+a1P_112000000_1*QR_000000020101+P_212000010*QR_000000020200+a1P_212000000_1*QR_000000020201+a3P_000000010_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_011001010*QR_020000000000+a1P_011001000_1*QR_020000000001+a1P_011000010_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111001010*QR_020000000100+a1P_111001000_1*QR_020000000101+a1P_111000010_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000001010_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_011001010*QR_010010000000+a1P_011001000_1*QR_010010000001+a1P_011000010_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111001010*QR_010010000100+a1P_111001000_1*QR_010010000101+a1P_111000010_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000001010_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_011001010*QR_000020000000+a1P_011001000_1*QR_000020000001+a1P_011000010_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111001010*QR_000020000100+a1P_111001000_1*QR_000020000101+a1P_111000010_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000001010_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(P_011001010*QR_010000010000+a1P_011001000_1*QR_010000010001+a1P_011000010_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111001010*QR_010000010100+a1P_111001000_1*QR_010000010101+a1P_111000010_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000001010_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(P_011001010*QR_000010010000+a1P_011001000_1*QR_000010010001+a1P_011000010_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111001010*QR_000010010100+a1P_111001000_1*QR_000010010101+a1P_111000010_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000001010_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(P_011001010*QR_000000020000+a1P_011001000_1*QR_000000020001+a1P_011000010_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111001010*QR_000000020100+a1P_111001000_1*QR_000000020101+a1P_111000010_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000001010_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_010002010*QR_020000000000+a1P_010002000_1*QR_020000000001+a1P_010001010_2*QR_020000000010+a2P_010001000_2*QR_020000000011+a2P_010000010_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000002010_1*QR_020000000100+a2P_000002000_1*QR_020000000101+a2P_000001010_2*QR_020000000110+a3P_000001000_2*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_010002010*QR_010010000000+a1P_010002000_1*QR_010010000001+a1P_010001010_2*QR_010010000010+a2P_010001000_2*QR_010010000011+a2P_010000010_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000002010_1*QR_010010000100+a2P_000002000_1*QR_010010000101+a2P_000001010_2*QR_010010000110+a3P_000001000_2*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010002010*QR_000020000000+a1P_010002000_1*QR_000020000001+a1P_010001010_2*QR_000020000010+a2P_010001000_2*QR_000020000011+a2P_010000010_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000002010_1*QR_000020000100+a2P_000002000_1*QR_000020000101+a2P_000001010_2*QR_000020000110+a3P_000001000_2*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(P_010002010*QR_010000010000+a1P_010002000_1*QR_010000010001+a1P_010001010_2*QR_010000010010+a2P_010001000_2*QR_010000010011+a2P_010000010_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000002010_1*QR_010000010100+a2P_000002000_1*QR_010000010101+a2P_000001010_2*QR_010000010110+a3P_000001000_2*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(P_010002010*QR_000010010000+a1P_010002000_1*QR_000010010001+a1P_010001010_2*QR_000010010010+a2P_010001000_2*QR_000010010011+a2P_010000010_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000002010_1*QR_000010010100+a2P_000002000_1*QR_000010010101+a2P_000001010_2*QR_000010010110+a3P_000001000_2*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(P_010002010*QR_000000020000+a1P_010002000_1*QR_000000020001+a1P_010001010_2*QR_000000020010+a2P_010001000_2*QR_000000020011+a2P_010000010_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000002010_1*QR_000000020100+a2P_000002000_1*QR_000000020101+a2P_000001010_2*QR_000000020110+a3P_000001000_2*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_011000011*QR_020000000000+P_011000111*QR_020000000001+a2P_011000000_1*QR_020000000002+P_111000011*QR_020000000100+P_111000111*QR_020000000101+a2P_111000000_1*QR_020000000102+a2P_000000011_1*QR_020000000200+a2P_000000111_1*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_011000011*QR_010010000000+P_011000111*QR_010010000001+a2P_011000000_1*QR_010010000002+P_111000011*QR_010010000100+P_111000111*QR_010010000101+a2P_111000000_1*QR_010010000102+a2P_000000011_1*QR_010010000200+a2P_000000111_1*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_011000011*QR_000020000000+P_011000111*QR_000020000001+a2P_011000000_1*QR_000020000002+P_111000011*QR_000020000100+P_111000111*QR_000020000101+a2P_111000000_1*QR_000020000102+a2P_000000011_1*QR_000020000200+a2P_000000111_1*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(P_011000011*QR_010000010000+P_011000111*QR_010000010001+a2P_011000000_1*QR_010000010002+P_111000011*QR_010000010100+P_111000111*QR_010000010101+a2P_111000000_1*QR_010000010102+a2P_000000011_1*QR_010000010200+a2P_000000111_1*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(P_011000011*QR_000010010000+P_011000111*QR_000010010001+a2P_011000000_1*QR_000010010002+P_111000011*QR_000010010100+P_111000111*QR_000010010101+a2P_111000000_1*QR_000010010102+a2P_000000011_1*QR_000010010200+a2P_000000111_1*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(P_011000011*QR_000000020000+P_011000111*QR_000000020001+a2P_011000000_1*QR_000000020002+P_111000011*QR_000000020100+P_111000111*QR_000000020101+a2P_111000000_1*QR_000000020102+a2P_000000011_1*QR_000000020200+a2P_000000111_1*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_010001011*QR_020000000000+P_010001111*QR_020000000001+a2P_010001000_1*QR_020000000002+a1P_010000011_1*QR_020000000010+a1P_010000111_1*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000001011_1*QR_020000000100+a1P_000001111_1*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_010001011*QR_010010000000+P_010001111*QR_010010000001+a2P_010001000_1*QR_010010000002+a1P_010000011_1*QR_010010000010+a1P_010000111_1*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000001011_1*QR_010010000100+a1P_000001111_1*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010001011*QR_000020000000+P_010001111*QR_000020000001+a2P_010001000_1*QR_000020000002+a1P_010000011_1*QR_000020000010+a1P_010000111_1*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000001011_1*QR_000020000100+a1P_000001111_1*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(P_010001011*QR_010000010000+P_010001111*QR_010000010001+a2P_010001000_1*QR_010000010002+a1P_010000011_1*QR_010000010010+a1P_010000111_1*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000001011_1*QR_010000010100+a1P_000001111_1*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(P_010001011*QR_000010010000+P_010001111*QR_000010010001+a2P_010001000_1*QR_000010010002+a1P_010000011_1*QR_000010010010+a1P_010000111_1*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000001011_1*QR_000010010100+a1P_000001111_1*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(P_010001011*QR_000000020000+P_010001111*QR_000000020001+a2P_010001000_1*QR_000000020002+a1P_010000011_1*QR_000000020010+a1P_010000111_1*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000001011_1*QR_000000020100+a1P_000001111_1*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_010000012*QR_020000000000+P_010000112*QR_020000000001+P_010000212*QR_020000000002+a3P_010000000_1*QR_020000000003+a1P_000000012_1*QR_020000000100+a1P_000000112_1*QR_020000000101+a1P_000000212_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_010000012*QR_010010000000+P_010000112*QR_010010000001+P_010000212*QR_010010000002+a3P_010000000_1*QR_010010000003+a1P_000000012_1*QR_010010000100+a1P_000000112_1*QR_010010000101+a1P_000000212_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010000012*QR_000020000000+P_010000112*QR_000020000001+P_010000212*QR_000020000002+a3P_010000000_1*QR_000020000003+a1P_000000012_1*QR_000020000100+a1P_000000112_1*QR_000020000101+a1P_000000212_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(P_010000012*QR_010000010000+P_010000112*QR_010000010001+P_010000212*QR_010000010002+a3P_010000000_1*QR_010000010003+a1P_000000012_1*QR_010000010100+a1P_000000112_1*QR_010000010101+a1P_000000212_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(P_010000012*QR_000010010000+P_010000112*QR_000010010001+P_010000212*QR_000010010002+a3P_010000000_1*QR_000010010003+a1P_000000012_1*QR_000010010100+a1P_000000112_1*QR_000010010101+a1P_000000212_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(P_010000012*QR_000000020000+P_010000112*QR_000000020001+P_010000212*QR_000000020002+a3P_010000000_1*QR_000000020003+a1P_000000012_1*QR_000000020100+a1P_000000112_1*QR_000000020101+a1P_000000212_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_002010010*QR_020000000000+a1P_002010000_1*QR_020000000001+a1P_002000010_1*QR_020000000010+a2P_002000000_1*QR_020000000011+a1P_001010010_2*QR_020000000100+a2P_001010000_2*QR_020000000101+a2P_001000010_2*QR_020000000110+a3P_001000000_2*QR_020000000111+a2P_000010010_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_002010010*QR_010010000000+a1P_002010000_1*QR_010010000001+a1P_002000010_1*QR_010010000010+a2P_002000000_1*QR_010010000011+a1P_001010010_2*QR_010010000100+a2P_001010000_2*QR_010010000101+a2P_001000010_2*QR_010010000110+a3P_001000000_2*QR_010010000111+a2P_000010010_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_002010010*QR_000020000000+a1P_002010000_1*QR_000020000001+a1P_002000010_1*QR_000020000010+a2P_002000000_1*QR_000020000011+a1P_001010010_2*QR_000020000100+a2P_001010000_2*QR_000020000101+a2P_001000010_2*QR_000020000110+a3P_001000000_2*QR_000020000111+a2P_000010010_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(P_002010010*QR_010000010000+a1P_002010000_1*QR_010000010001+a1P_002000010_1*QR_010000010010+a2P_002000000_1*QR_010000010011+a1P_001010010_2*QR_010000010100+a2P_001010000_2*QR_010000010101+a2P_001000010_2*QR_010000010110+a3P_001000000_2*QR_010000010111+a2P_000010010_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(P_002010010*QR_000010010000+a1P_002010000_1*QR_000010010001+a1P_002000010_1*QR_000010010010+a2P_002000000_1*QR_000010010011+a1P_001010010_2*QR_000010010100+a2P_001010000_2*QR_000010010101+a2P_001000010_2*QR_000010010110+a3P_001000000_2*QR_000010010111+a2P_000010010_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(P_002010010*QR_000000020000+a1P_002010000_1*QR_000000020001+a1P_002000010_1*QR_000000020010+a2P_002000000_1*QR_000000020011+a1P_001010010_2*QR_000000020100+a2P_001010000_2*QR_000000020101+a2P_001000010_2*QR_000000020110+a3P_001000000_2*QR_000000020111+a2P_000010010_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_001011010*QR_020000000000+a1P_001011000_1*QR_020000000001+P_001111010*QR_020000000010+a1P_001111000_1*QR_020000000011+a2P_001000010_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000011010_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111010_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_001011010*QR_010010000000+a1P_001011000_1*QR_010010000001+P_001111010*QR_010010000010+a1P_001111000_1*QR_010010000011+a2P_001000010_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000011010_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111010_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_001011010*QR_000020000000+a1P_001011000_1*QR_000020000001+P_001111010*QR_000020000010+a1P_001111000_1*QR_000020000011+a2P_001000010_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000011010_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111010_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(P_001011010*QR_010000010000+a1P_001011000_1*QR_010000010001+P_001111010*QR_010000010010+a1P_001111000_1*QR_010000010011+a2P_001000010_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000011010_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111010_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(P_001011010*QR_000010010000+a1P_001011000_1*QR_000010010001+P_001111010*QR_000010010010+a1P_001111000_1*QR_000010010011+a2P_001000010_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000011010_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111010_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(P_001011010*QR_000000020000+a1P_001011000_1*QR_000000020001+P_001111010*QR_000000020010+a1P_001111000_1*QR_000000020011+a2P_001000010_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000011010_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111010_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_000012010*QR_020000000000+a1P_000012000_1*QR_020000000001+P_000112010*QR_020000000010+a1P_000112000_1*QR_020000000011+P_000212010*QR_020000000020+a1P_000212000_1*QR_020000000021+a3P_000000010_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_000012010*QR_010010000000+a1P_000012000_1*QR_010010000001+P_000112010*QR_010010000010+a1P_000112000_1*QR_010010000011+P_000212010*QR_010010000020+a1P_000212000_1*QR_010010000021+a3P_000000010_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000012010*QR_000020000000+a1P_000012000_1*QR_000020000001+P_000112010*QR_000020000010+a1P_000112000_1*QR_000020000011+P_000212010*QR_000020000020+a1P_000212000_1*QR_000020000021+a3P_000000010_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(P_000012010*QR_010000010000+a1P_000012000_1*QR_010000010001+P_000112010*QR_010000010010+a1P_000112000_1*QR_010000010011+P_000212010*QR_010000010020+a1P_000212000_1*QR_010000010021+a3P_000000010_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(P_000012010*QR_000010010000+a1P_000012000_1*QR_000010010001+P_000112010*QR_000010010010+a1P_000112000_1*QR_000010010011+P_000212010*QR_000010010020+a1P_000212000_1*QR_000010010021+a3P_000000010_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(P_000012010*QR_000000020000+a1P_000012000_1*QR_000000020001+P_000112010*QR_000000020010+a1P_000112000_1*QR_000000020011+P_000212010*QR_000000020020+a1P_000212000_1*QR_000000020021+a3P_000000010_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_001010011*QR_020000000000+P_001010111*QR_020000000001+a2P_001010000_1*QR_020000000002+a1P_001000011_1*QR_020000000010+a1P_001000111_1*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000010011_1*QR_020000000100+a1P_000010111_1*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_001010011*QR_010010000000+P_001010111*QR_010010000001+a2P_001010000_1*QR_010010000002+a1P_001000011_1*QR_010010000010+a1P_001000111_1*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000010011_1*QR_010010000100+a1P_000010111_1*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_001010011*QR_000020000000+P_001010111*QR_000020000001+a2P_001010000_1*QR_000020000002+a1P_001000011_1*QR_000020000010+a1P_001000111_1*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000010011_1*QR_000020000100+a1P_000010111_1*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(P_001010011*QR_010000010000+P_001010111*QR_010000010001+a2P_001010000_1*QR_010000010002+a1P_001000011_1*QR_010000010010+a1P_001000111_1*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000010011_1*QR_010000010100+a1P_000010111_1*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(P_001010011*QR_000010010000+P_001010111*QR_000010010001+a2P_001010000_1*QR_000010010002+a1P_001000011_1*QR_000010010010+a1P_001000111_1*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000010011_1*QR_000010010100+a1P_000010111_1*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(P_001010011*QR_000000020000+P_001010111*QR_000000020001+a2P_001010000_1*QR_000000020002+a1P_001000011_1*QR_000000020010+a1P_001000111_1*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000010011_1*QR_000000020100+a1P_000010111_1*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_000011011*QR_020000000000+P_000011111*QR_020000000001+a2P_000011000_1*QR_020000000002+P_000111011*QR_020000000010+P_000111111*QR_020000000011+a2P_000111000_1*QR_020000000012+a2P_000000011_1*QR_020000000020+a2P_000000111_1*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_000011011*QR_010010000000+P_000011111*QR_010010000001+a2P_000011000_1*QR_010010000002+P_000111011*QR_010010000010+P_000111111*QR_010010000011+a2P_000111000_1*QR_010010000012+a2P_000000011_1*QR_010010000020+a2P_000000111_1*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000011011*QR_000020000000+P_000011111*QR_000020000001+a2P_000011000_1*QR_000020000002+P_000111011*QR_000020000010+P_000111111*QR_000020000011+a2P_000111000_1*QR_000020000012+a2P_000000011_1*QR_000020000020+a2P_000000111_1*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(P_000011011*QR_010000010000+P_000011111*QR_010000010001+a2P_000011000_1*QR_010000010002+P_000111011*QR_010000010010+P_000111111*QR_010000010011+a2P_000111000_1*QR_010000010012+a2P_000000011_1*QR_010000010020+a2P_000000111_1*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(P_000011011*QR_000010010000+P_000011111*QR_000010010001+a2P_000011000_1*QR_000010010002+P_000111011*QR_000010010010+P_000111111*QR_000010010011+a2P_000111000_1*QR_000010010012+a2P_000000011_1*QR_000010010020+a2P_000000111_1*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(P_000011011*QR_000000020000+P_000011111*QR_000000020001+a2P_000011000_1*QR_000000020002+P_000111011*QR_000000020010+P_000111111*QR_000000020011+a2P_000111000_1*QR_000000020012+a2P_000000011_1*QR_000000020020+a2P_000000111_1*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_000010012*QR_020000000000+P_000010112*QR_020000000001+P_000010212*QR_020000000002+a3P_000010000_1*QR_020000000003+a1P_000000012_1*QR_020000000010+a1P_000000112_1*QR_020000000011+a1P_000000212_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_000010012*QR_010010000000+P_000010112*QR_010010000001+P_000010212*QR_010010000002+a3P_000010000_1*QR_010010000003+a1P_000000012_1*QR_010010000010+a1P_000000112_1*QR_010010000011+a1P_000000212_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000010012*QR_000020000000+P_000010112*QR_000020000001+P_000010212*QR_000020000002+a3P_000010000_1*QR_000020000003+a1P_000000012_1*QR_000020000010+a1P_000000112_1*QR_000020000011+a1P_000000212_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(P_000010012*QR_010000010000+P_000010112*QR_010000010001+P_000010212*QR_010000010002+a3P_000010000_1*QR_010000010003+a1P_000000012_1*QR_010000010010+a1P_000000112_1*QR_010000010011+a1P_000000212_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(P_000010012*QR_000010010000+P_000010112*QR_000010010001+P_000010212*QR_000010010002+a3P_000010000_1*QR_000010010003+a1P_000000012_1*QR_000010010010+a1P_000000112_1*QR_000010010011+a1P_000000212_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(P_000010012*QR_000000020000+P_000010112*QR_000000020001+P_000010212*QR_000000020002+a3P_000010000_1*QR_000000020003+a1P_000000012_1*QR_000000020010+a1P_000000112_1*QR_000000020011+a1P_000000212_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_002000020*QR_020000000000+a1P_002000010_2*QR_020000000001+a2P_002000000_1*QR_020000000002+a1P_001000020_2*QR_020000000100+a2P_001000010_4*QR_020000000101+a3P_001000000_2*QR_020000000102+a2P_000000020_1*QR_020000000200+a3P_000000010_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_002000020*QR_010010000000+a1P_002000010_2*QR_010010000001+a2P_002000000_1*QR_010010000002+a1P_001000020_2*QR_010010000100+a2P_001000010_4*QR_010010000101+a3P_001000000_2*QR_010010000102+a2P_000000020_1*QR_010010000200+a3P_000000010_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_002000020*QR_000020000000+a1P_002000010_2*QR_000020000001+a2P_002000000_1*QR_000020000002+a1P_001000020_2*QR_000020000100+a2P_001000010_4*QR_000020000101+a3P_001000000_2*QR_000020000102+a2P_000000020_1*QR_000020000200+a3P_000000010_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(P_002000020*QR_010000010000+a1P_002000010_2*QR_010000010001+a2P_002000000_1*QR_010000010002+a1P_001000020_2*QR_010000010100+a2P_001000010_4*QR_010000010101+a3P_001000000_2*QR_010000010102+a2P_000000020_1*QR_010000010200+a3P_000000010_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(P_002000020*QR_000010010000+a1P_002000010_2*QR_000010010001+a2P_002000000_1*QR_000010010002+a1P_001000020_2*QR_000010010100+a2P_001000010_4*QR_000010010101+a3P_001000000_2*QR_000010010102+a2P_000000020_1*QR_000010010200+a3P_000000010_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(P_002000020*QR_000000020000+a1P_002000010_2*QR_000000020001+a2P_002000000_1*QR_000000020002+a1P_001000020_2*QR_000000020100+a2P_001000010_4*QR_000000020101+a3P_001000000_2*QR_000000020102+a2P_000000020_1*QR_000000020200+a3P_000000010_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_001001020*QR_020000000000+a1P_001001010_2*QR_020000000001+a2P_001001000_1*QR_020000000002+a1P_001000020_1*QR_020000000010+a2P_001000010_2*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000001020_1*QR_020000000100+a2P_000001010_2*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000020_1*QR_020000000110+a3P_000000010_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_001001020*QR_010010000000+a1P_001001010_2*QR_010010000001+a2P_001001000_1*QR_010010000002+a1P_001000020_1*QR_010010000010+a2P_001000010_2*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000001020_1*QR_010010000100+a2P_000001010_2*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000020_1*QR_010010000110+a3P_000000010_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_001001020*QR_000020000000+a1P_001001010_2*QR_000020000001+a2P_001001000_1*QR_000020000002+a1P_001000020_1*QR_000020000010+a2P_001000010_2*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000001020_1*QR_000020000100+a2P_000001010_2*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000020_1*QR_000020000110+a3P_000000010_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(P_001001020*QR_010000010000+a1P_001001010_2*QR_010000010001+a2P_001001000_1*QR_010000010002+a1P_001000020_1*QR_010000010010+a2P_001000010_2*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000001020_1*QR_010000010100+a2P_000001010_2*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000020_1*QR_010000010110+a3P_000000010_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(P_001001020*QR_000010010000+a1P_001001010_2*QR_000010010001+a2P_001001000_1*QR_000010010002+a1P_001000020_1*QR_000010010010+a2P_001000010_2*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000001020_1*QR_000010010100+a2P_000001010_2*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000020_1*QR_000010010110+a3P_000000010_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(P_001001020*QR_000000020000+a1P_001001010_2*QR_000000020001+a2P_001001000_1*QR_000000020002+a1P_001000020_1*QR_000000020010+a2P_001000010_2*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000001020_1*QR_000000020100+a2P_000001010_2*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000020_1*QR_000000020110+a3P_000000010_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_000002020*QR_020000000000+a1P_000002010_2*QR_020000000001+a2P_000002000_1*QR_020000000002+a1P_000001020_2*QR_020000000010+a2P_000001010_4*QR_020000000011+a3P_000001000_2*QR_020000000012+a2P_000000020_1*QR_020000000020+a3P_000000010_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_000002020*QR_010010000000+a1P_000002010_2*QR_010010000001+a2P_000002000_1*QR_010010000002+a1P_000001020_2*QR_010010000010+a2P_000001010_4*QR_010010000011+a3P_000001000_2*QR_010010000012+a2P_000000020_1*QR_010010000020+a3P_000000010_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000002020*QR_000020000000+a1P_000002010_2*QR_000020000001+a2P_000002000_1*QR_000020000002+a1P_000001020_2*QR_000020000010+a2P_000001010_4*QR_000020000011+a3P_000001000_2*QR_000020000012+a2P_000000020_1*QR_000020000020+a3P_000000010_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(P_000002020*QR_010000010000+a1P_000002010_2*QR_010000010001+a2P_000002000_1*QR_010000010002+a1P_000001020_2*QR_010000010010+a2P_000001010_4*QR_010000010011+a3P_000001000_2*QR_010000010012+a2P_000000020_1*QR_010000010020+a3P_000000010_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(P_000002020*QR_000010010000+a1P_000002010_2*QR_000010010001+a2P_000002000_1*QR_000010010002+a1P_000001020_2*QR_000010010010+a2P_000001010_4*QR_000010010011+a3P_000001000_2*QR_000010010012+a2P_000000020_1*QR_000010010020+a3P_000000010_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(P_000002020*QR_000000020000+a1P_000002010_2*QR_000000020001+a2P_000002000_1*QR_000000020002+a1P_000001020_2*QR_000000020010+a2P_000001010_4*QR_000000020011+a3P_000001000_2*QR_000000020012+a2P_000000020_1*QR_000000020020+a3P_000000010_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_001000021*QR_020000000000+P_001000121*QR_020000000001+P_001000221*QR_020000000002+a3P_001000000_1*QR_020000000003+a1P_000000021_1*QR_020000000100+a1P_000000121_1*QR_020000000101+a1P_000000221_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_001000021*QR_010010000000+P_001000121*QR_010010000001+P_001000221*QR_010010000002+a3P_001000000_1*QR_010010000003+a1P_000000021_1*QR_010010000100+a1P_000000121_1*QR_010010000101+a1P_000000221_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_001000021*QR_000020000000+P_001000121*QR_000020000001+P_001000221*QR_000020000002+a3P_001000000_1*QR_000020000003+a1P_000000021_1*QR_000020000100+a1P_000000121_1*QR_000020000101+a1P_000000221_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(P_001000021*QR_010000010000+P_001000121*QR_010000010001+P_001000221*QR_010000010002+a3P_001000000_1*QR_010000010003+a1P_000000021_1*QR_010000010100+a1P_000000121_1*QR_010000010101+a1P_000000221_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(P_001000021*QR_000010010000+P_001000121*QR_000010010001+P_001000221*QR_000010010002+a3P_001000000_1*QR_000010010003+a1P_000000021_1*QR_000010010100+a1P_000000121_1*QR_000010010101+a1P_000000221_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(P_001000021*QR_000000020000+P_001000121*QR_000000020001+P_001000221*QR_000000020002+a3P_001000000_1*QR_000000020003+a1P_000000021_1*QR_000000020100+a1P_000000121_1*QR_000000020101+a1P_000000221_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_000001021*QR_020000000000+P_000001121*QR_020000000001+P_000001221*QR_020000000002+a3P_000001000_1*QR_020000000003+a1P_000000021_1*QR_020000000010+a1P_000000121_1*QR_020000000011+a1P_000000221_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_000001021*QR_010010000000+P_000001121*QR_010010000001+P_000001221*QR_010010000002+a3P_000001000_1*QR_010010000003+a1P_000000021_1*QR_010010000010+a1P_000000121_1*QR_010010000011+a1P_000000221_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000001021*QR_000020000000+P_000001121*QR_000020000001+P_000001221*QR_000020000002+a3P_000001000_1*QR_000020000003+a1P_000000021_1*QR_000020000010+a1P_000000121_1*QR_000020000011+a1P_000000221_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(P_000001021*QR_010000010000+P_000001121*QR_010000010001+P_000001221*QR_010000010002+a3P_000001000_1*QR_010000010003+a1P_000000021_1*QR_010000010010+a1P_000000121_1*QR_010000010011+a1P_000000221_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(P_000001021*QR_000010010000+P_000001121*QR_000010010001+P_000001221*QR_000010010002+a3P_000001000_1*QR_000010010003+a1P_000000021_1*QR_000010010010+a1P_000000121_1*QR_000010010011+a1P_000000221_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(P_000001021*QR_000000020000+P_000001121*QR_000000020001+P_000001221*QR_000000020002+a3P_000001000_1*QR_000000020003+a1P_000000021_1*QR_000000020010+a1P_000000121_1*QR_000000020011+a1P_000000221_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_000000022*QR_020000000000+P_000000122*QR_020000000001+P_000000222*QR_020000000002+a2P_000000111_2*QR_020000000003+aPin4*QR_020000000004);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_000000022*QR_010010000000+P_000000122*QR_010010000001+P_000000222*QR_010010000002+a2P_000000111_2*QR_010010000003+aPin4*QR_010010000004);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000000022*QR_000020000000+P_000000122*QR_000020000001+P_000000222*QR_000020000002+a2P_000000111_2*QR_000020000003+aPin4*QR_000020000004);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(P_000000022*QR_010000010000+P_000000122*QR_010000010001+P_000000222*QR_010000010002+a2P_000000111_2*QR_010000010003+aPin4*QR_010000010004);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(P_000000022*QR_000010010000+P_000000122*QR_000010010001+P_000000222*QR_000010010002+a2P_000000111_2*QR_000010010003+aPin4*QR_000010010004);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(P_000000022*QR_000000020000+P_000000122*QR_000000020001+P_000000222*QR_000000020002+a2P_000000111_2*QR_000000020003+aPin4*QR_000000020004);
			}
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
}
__global__ void TSMJ_ddds_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
                unsigned int contrc_bra_start_pr,\
                unsigned int contrc_ket_start_pr,\
                unsigned int contrc_Pmtrx_start_pr,\
                unsigned int * contrc_bra_id,\
                double * P,\
                double * PA,\
                double * PB,\
                double * Zta_in,\
                double * pp_in,\
                float * K2_p,\
                double * ans){

    unsigned int tId_x = threadIdx.x;
    unsigned int bId_x = blockIdx.x;
    unsigned int tdis = blockDim.x;
    unsigned int bdis = gridDim.x;
    unsigned int ans_id=tId_x;
    double Pmtrx[6]={0.0};

    __shared__ double ans_temp[NTHREAD_128*36];
    for(int i=0;i<36;i++){
        ans_temp[i*tdis+tId_x]=0.0;
    }


    for(unsigned int i_contrc_bra=bId_x;i_contrc_bra<contrc_bra_num;i_contrc_bra+=bdis){
        unsigned int primit_bra_start = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra  ];
        unsigned int primit_bra_end   = contrc_bra_start_pr+contrc_bra_id[i_contrc_bra+1];
            for(unsigned int ii=primit_bra_start;ii<primit_bra_end;ii++){
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
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double aPin4=aPin1*aPin3;
			double QR_020000000000=0;
			double QR_010010000000=0;
			double QR_000020000000=0;
			double QR_010000010000=0;
			double QR_000010010000=0;
			double QR_000000020000=0;
			double QR_020000000001=0;
			double QR_010010000001=0;
			double QR_000020000001=0;
			double QR_010000010001=0;
			double QR_000010010001=0;
			double QR_000000020001=0;
			double QR_020000000010=0;
			double QR_010010000010=0;
			double QR_000020000010=0;
			double QR_010000010010=0;
			double QR_000010010010=0;
			double QR_000000020010=0;
			double QR_020000000100=0;
			double QR_010010000100=0;
			double QR_000020000100=0;
			double QR_010000010100=0;
			double QR_000010010100=0;
			double QR_000000020100=0;
			double QR_020000000002=0;
			double QR_010010000002=0;
			double QR_000020000002=0;
			double QR_010000010002=0;
			double QR_000010010002=0;
			double QR_000000020002=0;
			double QR_020000000011=0;
			double QR_010010000011=0;
			double QR_000020000011=0;
			double QR_010000010011=0;
			double QR_000010010011=0;
			double QR_000000020011=0;
			double QR_020000000020=0;
			double QR_010010000020=0;
			double QR_000020000020=0;
			double QR_010000010020=0;
			double QR_000010010020=0;
			double QR_000000020020=0;
			double QR_020000000101=0;
			double QR_010010000101=0;
			double QR_000020000101=0;
			double QR_010000010101=0;
			double QR_000010010101=0;
			double QR_000000020101=0;
			double QR_020000000110=0;
			double QR_010010000110=0;
			double QR_000020000110=0;
			double QR_010000010110=0;
			double QR_000010010110=0;
			double QR_000000020110=0;
			double QR_020000000200=0;
			double QR_010010000200=0;
			double QR_000020000200=0;
			double QR_010000010200=0;
			double QR_000010010200=0;
			double QR_000000020200=0;
			double QR_020000000003=0;
			double QR_010010000003=0;
			double QR_000020000003=0;
			double QR_010000010003=0;
			double QR_000010010003=0;
			double QR_000000020003=0;
			double QR_020000000012=0;
			double QR_010010000012=0;
			double QR_000020000012=0;
			double QR_010000010012=0;
			double QR_000010010012=0;
			double QR_000000020012=0;
			double QR_020000000021=0;
			double QR_010010000021=0;
			double QR_000020000021=0;
			double QR_010000010021=0;
			double QR_000010010021=0;
			double QR_000000020021=0;
			double QR_020000000030=0;
			double QR_010010000030=0;
			double QR_000020000030=0;
			double QR_010000010030=0;
			double QR_000010010030=0;
			double QR_000000020030=0;
			double QR_020000000102=0;
			double QR_010010000102=0;
			double QR_000020000102=0;
			double QR_010000010102=0;
			double QR_000010010102=0;
			double QR_000000020102=0;
			double QR_020000000111=0;
			double QR_010010000111=0;
			double QR_000020000111=0;
			double QR_010000010111=0;
			double QR_000010010111=0;
			double QR_000000020111=0;
			double QR_020000000120=0;
			double QR_010010000120=0;
			double QR_000020000120=0;
			double QR_010000010120=0;
			double QR_000010010120=0;
			double QR_000000020120=0;
			double QR_020000000201=0;
			double QR_010010000201=0;
			double QR_000020000201=0;
			double QR_010000010201=0;
			double QR_000010010201=0;
			double QR_000000020201=0;
			double QR_020000000210=0;
			double QR_010010000210=0;
			double QR_000020000210=0;
			double QR_010000010210=0;
			double QR_000010010210=0;
			double QR_000000020210=0;
			double QR_020000000300=0;
			double QR_010010000300=0;
			double QR_000020000300=0;
			double QR_010000010300=0;
			double QR_000010010300=0;
			double QR_000000020300=0;
			double QR_020000000004=0;
			double QR_010010000004=0;
			double QR_000020000004=0;
			double QR_010000010004=0;
			double QR_000010010004=0;
			double QR_000000020004=0;
			double QR_020000000013=0;
			double QR_010010000013=0;
			double QR_000020000013=0;
			double QR_010000010013=0;
			double QR_000010010013=0;
			double QR_000000020013=0;
			double QR_020000000022=0;
			double QR_010010000022=0;
			double QR_000020000022=0;
			double QR_010000010022=0;
			double QR_000010010022=0;
			double QR_000000020022=0;
			double QR_020000000031=0;
			double QR_010010000031=0;
			double QR_000020000031=0;
			double QR_010000010031=0;
			double QR_000010010031=0;
			double QR_000000020031=0;
			double QR_020000000040=0;
			double QR_010010000040=0;
			double QR_000020000040=0;
			double QR_010000010040=0;
			double QR_000010010040=0;
			double QR_000000020040=0;
			double QR_020000000103=0;
			double QR_010010000103=0;
			double QR_000020000103=0;
			double QR_010000010103=0;
			double QR_000010010103=0;
			double QR_000000020103=0;
			double QR_020000000112=0;
			double QR_010010000112=0;
			double QR_000020000112=0;
			double QR_010000010112=0;
			double QR_000010010112=0;
			double QR_000000020112=0;
			double QR_020000000121=0;
			double QR_010010000121=0;
			double QR_000020000121=0;
			double QR_010000010121=0;
			double QR_000010010121=0;
			double QR_000000020121=0;
			double QR_020000000130=0;
			double QR_010010000130=0;
			double QR_000020000130=0;
			double QR_010000010130=0;
			double QR_000010010130=0;
			double QR_000000020130=0;
			double QR_020000000202=0;
			double QR_010010000202=0;
			double QR_000020000202=0;
			double QR_010000010202=0;
			double QR_000010010202=0;
			double QR_000000020202=0;
			double QR_020000000211=0;
			double QR_010010000211=0;
			double QR_000020000211=0;
			double QR_010000010211=0;
			double QR_000010010211=0;
			double QR_000000020211=0;
			double QR_020000000220=0;
			double QR_010010000220=0;
			double QR_000020000220=0;
			double QR_010000010220=0;
			double QR_000010010220=0;
			double QR_000000020220=0;
			double QR_020000000301=0;
			double QR_010010000301=0;
			double QR_000020000301=0;
			double QR_010000010301=0;
			double QR_000010010301=0;
			double QR_000000020301=0;
			double QR_020000000310=0;
			double QR_010010000310=0;
			double QR_000020000310=0;
			double QR_010000010310=0;
			double QR_000010010310=0;
			double QR_000000020310=0;
			double QR_020000000400=0;
			double QR_010010000400=0;
			double QR_000020000400=0;
			double QR_010000010400=0;
			double QR_000010010400=0;
			double QR_000000020400=0;
        for(unsigned int j=tId_x;j<primit_ket_len;j+=tdis){
            unsigned int jj=contrc_ket_start_pr+j;
            float K2_q=tex1Dfetch(tex_K2_q,jj);
                if(fabsf(K2_p[ii]*K2_q)<1.0E-14){
                    break;
                }
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*6;
        double P_max=0.0;
        for(int p_i=0;p_i<6;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
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
			double aQin2=aQin1*aQin1;
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
		double Qd_020[3];
		for(int i=0;i<3;i++){
			Qd_020[i]=aQin1+Qd_010[i]*Qd_010[i];
			}
			double Q_020000000;
			double Q_010010000;
			double Q_000020000;
			double Q_010000010;
			double Q_000010010;
			double Q_000000020;
			double a1Q_010000000_1;
			double a1Q_010000000_2;
			double a1Q_000010000_1;
			double a1Q_000010000_2;
			double a1Q_000000010_1;
			double a1Q_000000010_2;
			Q_020000000=Qd_020[0];
			Q_010010000=Qd_010[0]*Qd_010[1];
			Q_000020000=Qd_020[1];
			Q_010000010=Qd_010[0]*Qd_010[2];
			Q_000010010=Qd_010[1]*Qd_010[2];
			Q_000000020=Qd_020[2];
			a1Q_010000000_1=Qd_010[0];
			a1Q_010000000_2=2*a1Q_010000000_1;
			a1Q_000010000_1=Qd_010[1];
			a1Q_000010000_2=2*a1Q_000010000_1;
			a1Q_000000010_1=Qd_010[2];
			a1Q_000000010_2=2*a1Q_000000010_1;
			QR_020000000000+=Q_020000000*R_000[0]-a1Q_010000000_2*R_100[0]+aQin2*R_200[0];
			QR_010010000000+=Q_010010000*R_000[0]-a1Q_010000000_1*R_010[0]-a1Q_000010000_1*R_100[0]+aQin2*R_110[0];
			QR_000020000000+=Q_000020000*R_000[0]-a1Q_000010000_2*R_010[0]+aQin2*R_020[0];
			QR_010000010000+=Q_010000010*R_000[0]-a1Q_010000000_1*R_001[0]-a1Q_000000010_1*R_100[0]+aQin2*R_101[0];
			QR_000010010000+=Q_000010010*R_000[0]-a1Q_000010000_1*R_001[0]-a1Q_000000010_1*R_010[0]+aQin2*R_011[0];
			QR_000000020000+=Q_000000020*R_000[0]-a1Q_000000010_2*R_001[0]+aQin2*R_002[0];
			QR_020000000001+=Q_020000000*R_001[0]-a1Q_010000000_2*R_101[0]+aQin2*R_201[0];
			QR_010010000001+=Q_010010000*R_001[0]-a1Q_010000000_1*R_011[0]-a1Q_000010000_1*R_101[0]+aQin2*R_111[0];
			QR_000020000001+=Q_000020000*R_001[0]-a1Q_000010000_2*R_011[0]+aQin2*R_021[0];
			QR_010000010001+=Q_010000010*R_001[0]-a1Q_010000000_1*R_002[0]-a1Q_000000010_1*R_101[0]+aQin2*R_102[0];
			QR_000010010001+=Q_000010010*R_001[0]-a1Q_000010000_1*R_002[0]-a1Q_000000010_1*R_011[0]+aQin2*R_012[0];
			QR_000000020001+=Q_000000020*R_001[0]-a1Q_000000010_2*R_002[0]+aQin2*R_003[0];
			QR_020000000010+=Q_020000000*R_010[0]-a1Q_010000000_2*R_110[0]+aQin2*R_210[0];
			QR_010010000010+=Q_010010000*R_010[0]-a1Q_010000000_1*R_020[0]-a1Q_000010000_1*R_110[0]+aQin2*R_120[0];
			QR_000020000010+=Q_000020000*R_010[0]-a1Q_000010000_2*R_020[0]+aQin2*R_030[0];
			QR_010000010010+=Q_010000010*R_010[0]-a1Q_010000000_1*R_011[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000010010010+=Q_000010010*R_010[0]-a1Q_000010000_1*R_011[0]-a1Q_000000010_1*R_020[0]+aQin2*R_021[0];
			QR_000000020010+=Q_000000020*R_010[0]-a1Q_000000010_2*R_011[0]+aQin2*R_012[0];
			QR_020000000100+=Q_020000000*R_100[0]-a1Q_010000000_2*R_200[0]+aQin2*R_300[0];
			QR_010010000100+=Q_010010000*R_100[0]-a1Q_010000000_1*R_110[0]-a1Q_000010000_1*R_200[0]+aQin2*R_210[0];
			QR_000020000100+=Q_000020000*R_100[0]-a1Q_000010000_2*R_110[0]+aQin2*R_120[0];
			QR_010000010100+=Q_010000010*R_100[0]-a1Q_010000000_1*R_101[0]-a1Q_000000010_1*R_200[0]+aQin2*R_201[0];
			QR_000010010100+=Q_000010010*R_100[0]-a1Q_000010000_1*R_101[0]-a1Q_000000010_1*R_110[0]+aQin2*R_111[0];
			QR_000000020100+=Q_000000020*R_100[0]-a1Q_000000010_2*R_101[0]+aQin2*R_102[0];
			QR_020000000002+=Q_020000000*R_002[0]-a1Q_010000000_2*R_102[0]+aQin2*R_202[0];
			QR_010010000002+=Q_010010000*R_002[0]-a1Q_010000000_1*R_012[0]-a1Q_000010000_1*R_102[0]+aQin2*R_112[0];
			QR_000020000002+=Q_000020000*R_002[0]-a1Q_000010000_2*R_012[0]+aQin2*R_022[0];
			QR_010000010002+=Q_010000010*R_002[0]-a1Q_010000000_1*R_003[0]-a1Q_000000010_1*R_102[0]+aQin2*R_103[0];
			QR_000010010002+=Q_000010010*R_002[0]-a1Q_000010000_1*R_003[0]-a1Q_000000010_1*R_012[0]+aQin2*R_013[0];
			QR_000000020002+=Q_000000020*R_002[0]-a1Q_000000010_2*R_003[0]+aQin2*R_004[0];
			QR_020000000011+=Q_020000000*R_011[0]-a1Q_010000000_2*R_111[0]+aQin2*R_211[0];
			QR_010010000011+=Q_010010000*R_011[0]-a1Q_010000000_1*R_021[0]-a1Q_000010000_1*R_111[0]+aQin2*R_121[0];
			QR_000020000011+=Q_000020000*R_011[0]-a1Q_000010000_2*R_021[0]+aQin2*R_031[0];
			QR_010000010011+=Q_010000010*R_011[0]-a1Q_010000000_1*R_012[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000010010011+=Q_000010010*R_011[0]-a1Q_000010000_1*R_012[0]-a1Q_000000010_1*R_021[0]+aQin2*R_022[0];
			QR_000000020011+=Q_000000020*R_011[0]-a1Q_000000010_2*R_012[0]+aQin2*R_013[0];
			QR_020000000020+=Q_020000000*R_020[0]-a1Q_010000000_2*R_120[0]+aQin2*R_220[0];
			QR_010010000020+=Q_010010000*R_020[0]-a1Q_010000000_1*R_030[0]-a1Q_000010000_1*R_120[0]+aQin2*R_130[0];
			QR_000020000020+=Q_000020000*R_020[0]-a1Q_000010000_2*R_030[0]+aQin2*R_040[0];
			QR_010000010020+=Q_010000010*R_020[0]-a1Q_010000000_1*R_021[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000010010020+=Q_000010010*R_020[0]-a1Q_000010000_1*R_021[0]-a1Q_000000010_1*R_030[0]+aQin2*R_031[0];
			QR_000000020020+=Q_000000020*R_020[0]-a1Q_000000010_2*R_021[0]+aQin2*R_022[0];
			QR_020000000101+=Q_020000000*R_101[0]-a1Q_010000000_2*R_201[0]+aQin2*R_301[0];
			QR_010010000101+=Q_010010000*R_101[0]-a1Q_010000000_1*R_111[0]-a1Q_000010000_1*R_201[0]+aQin2*R_211[0];
			QR_000020000101+=Q_000020000*R_101[0]-a1Q_000010000_2*R_111[0]+aQin2*R_121[0];
			QR_010000010101+=Q_010000010*R_101[0]-a1Q_010000000_1*R_102[0]-a1Q_000000010_1*R_201[0]+aQin2*R_202[0];
			QR_000010010101+=Q_000010010*R_101[0]-a1Q_000010000_1*R_102[0]-a1Q_000000010_1*R_111[0]+aQin2*R_112[0];
			QR_000000020101+=Q_000000020*R_101[0]-a1Q_000000010_2*R_102[0]+aQin2*R_103[0];
			QR_020000000110+=Q_020000000*R_110[0]-a1Q_010000000_2*R_210[0]+aQin2*R_310[0];
			QR_010010000110+=Q_010010000*R_110[0]-a1Q_010000000_1*R_120[0]-a1Q_000010000_1*R_210[0]+aQin2*R_220[0];
			QR_000020000110+=Q_000020000*R_110[0]-a1Q_000010000_2*R_120[0]+aQin2*R_130[0];
			QR_010000010110+=Q_010000010*R_110[0]-a1Q_010000000_1*R_111[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000010010110+=Q_000010010*R_110[0]-a1Q_000010000_1*R_111[0]-a1Q_000000010_1*R_120[0]+aQin2*R_121[0];
			QR_000000020110+=Q_000000020*R_110[0]-a1Q_000000010_2*R_111[0]+aQin2*R_112[0];
			QR_020000000200+=Q_020000000*R_200[0]-a1Q_010000000_2*R_300[0]+aQin2*R_400[0];
			QR_010010000200+=Q_010010000*R_200[0]-a1Q_010000000_1*R_210[0]-a1Q_000010000_1*R_300[0]+aQin2*R_310[0];
			QR_000020000200+=Q_000020000*R_200[0]-a1Q_000010000_2*R_210[0]+aQin2*R_220[0];
			QR_010000010200+=Q_010000010*R_200[0]-a1Q_010000000_1*R_201[0]-a1Q_000000010_1*R_300[0]+aQin2*R_301[0];
			QR_000010010200+=Q_000010010*R_200[0]-a1Q_000010000_1*R_201[0]-a1Q_000000010_1*R_210[0]+aQin2*R_211[0];
			QR_000000020200+=Q_000000020*R_200[0]-a1Q_000000010_2*R_201[0]+aQin2*R_202[0];
			QR_020000000003+=Q_020000000*R_003[0]-a1Q_010000000_2*R_103[0]+aQin2*R_203[0];
			QR_010010000003+=Q_010010000*R_003[0]-a1Q_010000000_1*R_013[0]-a1Q_000010000_1*R_103[0]+aQin2*R_113[0];
			QR_000020000003+=Q_000020000*R_003[0]-a1Q_000010000_2*R_013[0]+aQin2*R_023[0];
			QR_010000010003+=Q_010000010*R_003[0]-a1Q_010000000_1*R_004[0]-a1Q_000000010_1*R_103[0]+aQin2*R_104[0];
			QR_000010010003+=Q_000010010*R_003[0]-a1Q_000010000_1*R_004[0]-a1Q_000000010_1*R_013[0]+aQin2*R_014[0];
			QR_000000020003+=Q_000000020*R_003[0]-a1Q_000000010_2*R_004[0]+aQin2*R_005[0];
			QR_020000000012+=Q_020000000*R_012[0]-a1Q_010000000_2*R_112[0]+aQin2*R_212[0];
			QR_010010000012+=Q_010010000*R_012[0]-a1Q_010000000_1*R_022[0]-a1Q_000010000_1*R_112[0]+aQin2*R_122[0];
			QR_000020000012+=Q_000020000*R_012[0]-a1Q_000010000_2*R_022[0]+aQin2*R_032[0];
			QR_010000010012+=Q_010000010*R_012[0]-a1Q_010000000_1*R_013[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			QR_000010010012+=Q_000010010*R_012[0]-a1Q_000010000_1*R_013[0]-a1Q_000000010_1*R_022[0]+aQin2*R_023[0];
			QR_000000020012+=Q_000000020*R_012[0]-a1Q_000000010_2*R_013[0]+aQin2*R_014[0];
			QR_020000000021+=Q_020000000*R_021[0]-a1Q_010000000_2*R_121[0]+aQin2*R_221[0];
			QR_010010000021+=Q_010010000*R_021[0]-a1Q_010000000_1*R_031[0]-a1Q_000010000_1*R_121[0]+aQin2*R_131[0];
			QR_000020000021+=Q_000020000*R_021[0]-a1Q_000010000_2*R_031[0]+aQin2*R_041[0];
			QR_010000010021+=Q_010000010*R_021[0]-a1Q_010000000_1*R_022[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			QR_000010010021+=Q_000010010*R_021[0]-a1Q_000010000_1*R_022[0]-a1Q_000000010_1*R_031[0]+aQin2*R_032[0];
			QR_000000020021+=Q_000000020*R_021[0]-a1Q_000000010_2*R_022[0]+aQin2*R_023[0];
			QR_020000000030+=Q_020000000*R_030[0]-a1Q_010000000_2*R_130[0]+aQin2*R_230[0];
			QR_010010000030+=Q_010010000*R_030[0]-a1Q_010000000_1*R_040[0]-a1Q_000010000_1*R_130[0]+aQin2*R_140[0];
			QR_000020000030+=Q_000020000*R_030[0]-a1Q_000010000_2*R_040[0]+aQin2*R_050[0];
			QR_010000010030+=Q_010000010*R_030[0]-a1Q_010000000_1*R_031[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			QR_000010010030+=Q_000010010*R_030[0]-a1Q_000010000_1*R_031[0]-a1Q_000000010_1*R_040[0]+aQin2*R_041[0];
			QR_000000020030+=Q_000000020*R_030[0]-a1Q_000000010_2*R_031[0]+aQin2*R_032[0];
			QR_020000000102+=Q_020000000*R_102[0]-a1Q_010000000_2*R_202[0]+aQin2*R_302[0];
			QR_010010000102+=Q_010010000*R_102[0]-a1Q_010000000_1*R_112[0]-a1Q_000010000_1*R_202[0]+aQin2*R_212[0];
			QR_000020000102+=Q_000020000*R_102[0]-a1Q_000010000_2*R_112[0]+aQin2*R_122[0];
			QR_010000010102+=Q_010000010*R_102[0]-a1Q_010000000_1*R_103[0]-a1Q_000000010_1*R_202[0]+aQin2*R_203[0];
			QR_000010010102+=Q_000010010*R_102[0]-a1Q_000010000_1*R_103[0]-a1Q_000000010_1*R_112[0]+aQin2*R_113[0];
			QR_000000020102+=Q_000000020*R_102[0]-a1Q_000000010_2*R_103[0]+aQin2*R_104[0];
			QR_020000000111+=Q_020000000*R_111[0]-a1Q_010000000_2*R_211[0]+aQin2*R_311[0];
			QR_010010000111+=Q_010010000*R_111[0]-a1Q_010000000_1*R_121[0]-a1Q_000010000_1*R_211[0]+aQin2*R_221[0];
			QR_000020000111+=Q_000020000*R_111[0]-a1Q_000010000_2*R_121[0]+aQin2*R_131[0];
			QR_010000010111+=Q_010000010*R_111[0]-a1Q_010000000_1*R_112[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			QR_000010010111+=Q_000010010*R_111[0]-a1Q_000010000_1*R_112[0]-a1Q_000000010_1*R_121[0]+aQin2*R_122[0];
			QR_000000020111+=Q_000000020*R_111[0]-a1Q_000000010_2*R_112[0]+aQin2*R_113[0];
			QR_020000000120+=Q_020000000*R_120[0]-a1Q_010000000_2*R_220[0]+aQin2*R_320[0];
			QR_010010000120+=Q_010010000*R_120[0]-a1Q_010000000_1*R_130[0]-a1Q_000010000_1*R_220[0]+aQin2*R_230[0];
			QR_000020000120+=Q_000020000*R_120[0]-a1Q_000010000_2*R_130[0]+aQin2*R_140[0];
			QR_010000010120+=Q_010000010*R_120[0]-a1Q_010000000_1*R_121[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			QR_000010010120+=Q_000010010*R_120[0]-a1Q_000010000_1*R_121[0]-a1Q_000000010_1*R_130[0]+aQin2*R_131[0];
			QR_000000020120+=Q_000000020*R_120[0]-a1Q_000000010_2*R_121[0]+aQin2*R_122[0];
			QR_020000000201+=Q_020000000*R_201[0]-a1Q_010000000_2*R_301[0]+aQin2*R_401[0];
			QR_010010000201+=Q_010010000*R_201[0]-a1Q_010000000_1*R_211[0]-a1Q_000010000_1*R_301[0]+aQin2*R_311[0];
			QR_000020000201+=Q_000020000*R_201[0]-a1Q_000010000_2*R_211[0]+aQin2*R_221[0];
			QR_010000010201+=Q_010000010*R_201[0]-a1Q_010000000_1*R_202[0]-a1Q_000000010_1*R_301[0]+aQin2*R_302[0];
			QR_000010010201+=Q_000010010*R_201[0]-a1Q_000010000_1*R_202[0]-a1Q_000000010_1*R_211[0]+aQin2*R_212[0];
			QR_000000020201+=Q_000000020*R_201[0]-a1Q_000000010_2*R_202[0]+aQin2*R_203[0];
			QR_020000000210+=Q_020000000*R_210[0]-a1Q_010000000_2*R_310[0]+aQin2*R_410[0];
			QR_010010000210+=Q_010010000*R_210[0]-a1Q_010000000_1*R_220[0]-a1Q_000010000_1*R_310[0]+aQin2*R_320[0];
			QR_000020000210+=Q_000020000*R_210[0]-a1Q_000010000_2*R_220[0]+aQin2*R_230[0];
			QR_010000010210+=Q_010000010*R_210[0]-a1Q_010000000_1*R_211[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			QR_000010010210+=Q_000010010*R_210[0]-a1Q_000010000_1*R_211[0]-a1Q_000000010_1*R_220[0]+aQin2*R_221[0];
			QR_000000020210+=Q_000000020*R_210[0]-a1Q_000000010_2*R_211[0]+aQin2*R_212[0];
			QR_020000000300+=Q_020000000*R_300[0]-a1Q_010000000_2*R_400[0]+aQin2*R_500[0];
			QR_010010000300+=Q_010010000*R_300[0]-a1Q_010000000_1*R_310[0]-a1Q_000010000_1*R_400[0]+aQin2*R_410[0];
			QR_000020000300+=Q_000020000*R_300[0]-a1Q_000010000_2*R_310[0]+aQin2*R_320[0];
			QR_010000010300+=Q_010000010*R_300[0]-a1Q_010000000_1*R_301[0]-a1Q_000000010_1*R_400[0]+aQin2*R_401[0];
			QR_000010010300+=Q_000010010*R_300[0]-a1Q_000010000_1*R_301[0]-a1Q_000000010_1*R_310[0]+aQin2*R_311[0];
			QR_000000020300+=Q_000000020*R_300[0]-a1Q_000000010_2*R_301[0]+aQin2*R_302[0];
			QR_020000000004+=Q_020000000*R_004[0]-a1Q_010000000_2*R_104[0]+aQin2*R_204[0];
			QR_010010000004+=Q_010010000*R_004[0]-a1Q_010000000_1*R_014[0]-a1Q_000010000_1*R_104[0]+aQin2*R_114[0];
			QR_000020000004+=Q_000020000*R_004[0]-a1Q_000010000_2*R_014[0]+aQin2*R_024[0];
			QR_010000010004+=Q_010000010*R_004[0]-a1Q_010000000_1*R_005[0]-a1Q_000000010_1*R_104[0]+aQin2*R_105[0];
			QR_000010010004+=Q_000010010*R_004[0]-a1Q_000010000_1*R_005[0]-a1Q_000000010_1*R_014[0]+aQin2*R_015[0];
			QR_000000020004+=Q_000000020*R_004[0]-a1Q_000000010_2*R_005[0]+aQin2*R_006[0];
			QR_020000000013+=Q_020000000*R_013[0]-a1Q_010000000_2*R_113[0]+aQin2*R_213[0];
			QR_010010000013+=Q_010010000*R_013[0]-a1Q_010000000_1*R_023[0]-a1Q_000010000_1*R_113[0]+aQin2*R_123[0];
			QR_000020000013+=Q_000020000*R_013[0]-a1Q_000010000_2*R_023[0]+aQin2*R_033[0];
			QR_010000010013+=Q_010000010*R_013[0]-a1Q_010000000_1*R_014[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			QR_000010010013+=Q_000010010*R_013[0]-a1Q_000010000_1*R_014[0]-a1Q_000000010_1*R_023[0]+aQin2*R_024[0];
			QR_000000020013+=Q_000000020*R_013[0]-a1Q_000000010_2*R_014[0]+aQin2*R_015[0];
			QR_020000000022+=Q_020000000*R_022[0]-a1Q_010000000_2*R_122[0]+aQin2*R_222[0];
			QR_010010000022+=Q_010010000*R_022[0]-a1Q_010000000_1*R_032[0]-a1Q_000010000_1*R_122[0]+aQin2*R_132[0];
			QR_000020000022+=Q_000020000*R_022[0]-a1Q_000010000_2*R_032[0]+aQin2*R_042[0];
			QR_010000010022+=Q_010000010*R_022[0]-a1Q_010000000_1*R_023[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			QR_000010010022+=Q_000010010*R_022[0]-a1Q_000010000_1*R_023[0]-a1Q_000000010_1*R_032[0]+aQin2*R_033[0];
			QR_000000020022+=Q_000000020*R_022[0]-a1Q_000000010_2*R_023[0]+aQin2*R_024[0];
			QR_020000000031+=Q_020000000*R_031[0]-a1Q_010000000_2*R_131[0]+aQin2*R_231[0];
			QR_010010000031+=Q_010010000*R_031[0]-a1Q_010000000_1*R_041[0]-a1Q_000010000_1*R_131[0]+aQin2*R_141[0];
			QR_000020000031+=Q_000020000*R_031[0]-a1Q_000010000_2*R_041[0]+aQin2*R_051[0];
			QR_010000010031+=Q_010000010*R_031[0]-a1Q_010000000_1*R_032[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			QR_000010010031+=Q_000010010*R_031[0]-a1Q_000010000_1*R_032[0]-a1Q_000000010_1*R_041[0]+aQin2*R_042[0];
			QR_000000020031+=Q_000000020*R_031[0]-a1Q_000000010_2*R_032[0]+aQin2*R_033[0];
			QR_020000000040+=Q_020000000*R_040[0]-a1Q_010000000_2*R_140[0]+aQin2*R_240[0];
			QR_010010000040+=Q_010010000*R_040[0]-a1Q_010000000_1*R_050[0]-a1Q_000010000_1*R_140[0]+aQin2*R_150[0];
			QR_000020000040+=Q_000020000*R_040[0]-a1Q_000010000_2*R_050[0]+aQin2*R_060[0];
			QR_010000010040+=Q_010000010*R_040[0]-a1Q_010000000_1*R_041[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			QR_000010010040+=Q_000010010*R_040[0]-a1Q_000010000_1*R_041[0]-a1Q_000000010_1*R_050[0]+aQin2*R_051[0];
			QR_000000020040+=Q_000000020*R_040[0]-a1Q_000000010_2*R_041[0]+aQin2*R_042[0];
			QR_020000000103+=Q_020000000*R_103[0]-a1Q_010000000_2*R_203[0]+aQin2*R_303[0];
			QR_010010000103+=Q_010010000*R_103[0]-a1Q_010000000_1*R_113[0]-a1Q_000010000_1*R_203[0]+aQin2*R_213[0];
			QR_000020000103+=Q_000020000*R_103[0]-a1Q_000010000_2*R_113[0]+aQin2*R_123[0];
			QR_010000010103+=Q_010000010*R_103[0]-a1Q_010000000_1*R_104[0]-a1Q_000000010_1*R_203[0]+aQin2*R_204[0];
			QR_000010010103+=Q_000010010*R_103[0]-a1Q_000010000_1*R_104[0]-a1Q_000000010_1*R_113[0]+aQin2*R_114[0];
			QR_000000020103+=Q_000000020*R_103[0]-a1Q_000000010_2*R_104[0]+aQin2*R_105[0];
			QR_020000000112+=Q_020000000*R_112[0]-a1Q_010000000_2*R_212[0]+aQin2*R_312[0];
			QR_010010000112+=Q_010010000*R_112[0]-a1Q_010000000_1*R_122[0]-a1Q_000010000_1*R_212[0]+aQin2*R_222[0];
			QR_000020000112+=Q_000020000*R_112[0]-a1Q_000010000_2*R_122[0]+aQin2*R_132[0];
			QR_010000010112+=Q_010000010*R_112[0]-a1Q_010000000_1*R_113[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			QR_000010010112+=Q_000010010*R_112[0]-a1Q_000010000_1*R_113[0]-a1Q_000000010_1*R_122[0]+aQin2*R_123[0];
			QR_000000020112+=Q_000000020*R_112[0]-a1Q_000000010_2*R_113[0]+aQin2*R_114[0];
			QR_020000000121+=Q_020000000*R_121[0]-a1Q_010000000_2*R_221[0]+aQin2*R_321[0];
			QR_010010000121+=Q_010010000*R_121[0]-a1Q_010000000_1*R_131[0]-a1Q_000010000_1*R_221[0]+aQin2*R_231[0];
			QR_000020000121+=Q_000020000*R_121[0]-a1Q_000010000_2*R_131[0]+aQin2*R_141[0];
			QR_010000010121+=Q_010000010*R_121[0]-a1Q_010000000_1*R_122[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			QR_000010010121+=Q_000010010*R_121[0]-a1Q_000010000_1*R_122[0]-a1Q_000000010_1*R_131[0]+aQin2*R_132[0];
			QR_000000020121+=Q_000000020*R_121[0]-a1Q_000000010_2*R_122[0]+aQin2*R_123[0];
			QR_020000000130+=Q_020000000*R_130[0]-a1Q_010000000_2*R_230[0]+aQin2*R_330[0];
			QR_010010000130+=Q_010010000*R_130[0]-a1Q_010000000_1*R_140[0]-a1Q_000010000_1*R_230[0]+aQin2*R_240[0];
			QR_000020000130+=Q_000020000*R_130[0]-a1Q_000010000_2*R_140[0]+aQin2*R_150[0];
			QR_010000010130+=Q_010000010*R_130[0]-a1Q_010000000_1*R_131[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			QR_000010010130+=Q_000010010*R_130[0]-a1Q_000010000_1*R_131[0]-a1Q_000000010_1*R_140[0]+aQin2*R_141[0];
			QR_000000020130+=Q_000000020*R_130[0]-a1Q_000000010_2*R_131[0]+aQin2*R_132[0];
			QR_020000000202+=Q_020000000*R_202[0]-a1Q_010000000_2*R_302[0]+aQin2*R_402[0];
			QR_010010000202+=Q_010010000*R_202[0]-a1Q_010000000_1*R_212[0]-a1Q_000010000_1*R_302[0]+aQin2*R_312[0];
			QR_000020000202+=Q_000020000*R_202[0]-a1Q_000010000_2*R_212[0]+aQin2*R_222[0];
			QR_010000010202+=Q_010000010*R_202[0]-a1Q_010000000_1*R_203[0]-a1Q_000000010_1*R_302[0]+aQin2*R_303[0];
			QR_000010010202+=Q_000010010*R_202[0]-a1Q_000010000_1*R_203[0]-a1Q_000000010_1*R_212[0]+aQin2*R_213[0];
			QR_000000020202+=Q_000000020*R_202[0]-a1Q_000000010_2*R_203[0]+aQin2*R_204[0];
			QR_020000000211+=Q_020000000*R_211[0]-a1Q_010000000_2*R_311[0]+aQin2*R_411[0];
			QR_010010000211+=Q_010010000*R_211[0]-a1Q_010000000_1*R_221[0]-a1Q_000010000_1*R_311[0]+aQin2*R_321[0];
			QR_000020000211+=Q_000020000*R_211[0]-a1Q_000010000_2*R_221[0]+aQin2*R_231[0];
			QR_010000010211+=Q_010000010*R_211[0]-a1Q_010000000_1*R_212[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			QR_000010010211+=Q_000010010*R_211[0]-a1Q_000010000_1*R_212[0]-a1Q_000000010_1*R_221[0]+aQin2*R_222[0];
			QR_000000020211+=Q_000000020*R_211[0]-a1Q_000000010_2*R_212[0]+aQin2*R_213[0];
			QR_020000000220+=Q_020000000*R_220[0]-a1Q_010000000_2*R_320[0]+aQin2*R_420[0];
			QR_010010000220+=Q_010010000*R_220[0]-a1Q_010000000_1*R_230[0]-a1Q_000010000_1*R_320[0]+aQin2*R_330[0];
			QR_000020000220+=Q_000020000*R_220[0]-a1Q_000010000_2*R_230[0]+aQin2*R_240[0];
			QR_010000010220+=Q_010000010*R_220[0]-a1Q_010000000_1*R_221[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			QR_000010010220+=Q_000010010*R_220[0]-a1Q_000010000_1*R_221[0]-a1Q_000000010_1*R_230[0]+aQin2*R_231[0];
			QR_000000020220+=Q_000000020*R_220[0]-a1Q_000000010_2*R_221[0]+aQin2*R_222[0];
			QR_020000000301+=Q_020000000*R_301[0]-a1Q_010000000_2*R_401[0]+aQin2*R_501[0];
			QR_010010000301+=Q_010010000*R_301[0]-a1Q_010000000_1*R_311[0]-a1Q_000010000_1*R_401[0]+aQin2*R_411[0];
			QR_000020000301+=Q_000020000*R_301[0]-a1Q_000010000_2*R_311[0]+aQin2*R_321[0];
			QR_010000010301+=Q_010000010*R_301[0]-a1Q_010000000_1*R_302[0]-a1Q_000000010_1*R_401[0]+aQin2*R_402[0];
			QR_000010010301+=Q_000010010*R_301[0]-a1Q_000010000_1*R_302[0]-a1Q_000000010_1*R_311[0]+aQin2*R_312[0];
			QR_000000020301+=Q_000000020*R_301[0]-a1Q_000000010_2*R_302[0]+aQin2*R_303[0];
			QR_020000000310+=Q_020000000*R_310[0]-a1Q_010000000_2*R_410[0]+aQin2*R_510[0];
			QR_010010000310+=Q_010010000*R_310[0]-a1Q_010000000_1*R_320[0]-a1Q_000010000_1*R_410[0]+aQin2*R_420[0];
			QR_000020000310+=Q_000020000*R_310[0]-a1Q_000010000_2*R_320[0]+aQin2*R_330[0];
			QR_010000010310+=Q_010000010*R_310[0]-a1Q_010000000_1*R_311[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			QR_000010010310+=Q_000010010*R_310[0]-a1Q_000010000_1*R_311[0]-a1Q_000000010_1*R_320[0]+aQin2*R_321[0];
			QR_000000020310+=Q_000000020*R_310[0]-a1Q_000000010_2*R_311[0]+aQin2*R_312[0];
			QR_020000000400+=Q_020000000*R_400[0]-a1Q_010000000_2*R_500[0]+aQin2*R_600[0];
			QR_010010000400+=Q_010010000*R_400[0]-a1Q_010000000_1*R_410[0]-a1Q_000010000_1*R_500[0]+aQin2*R_510[0];
			QR_000020000400+=Q_000020000*R_400[0]-a1Q_000010000_2*R_410[0]+aQin2*R_420[0];
			QR_010000010400+=Q_010000010*R_400[0]-a1Q_010000000_1*R_401[0]-a1Q_000000010_1*R_500[0]+aQin2*R_501[0];
			QR_000010010400+=Q_000010010*R_400[0]-a1Q_000010000_1*R_401[0]-a1Q_000000010_1*R_410[0]+aQin2*R_411[0];
			QR_000000020400+=Q_000000020*R_400[0]-a1Q_000000010_2*R_401[0]+aQin2*R_402[0];
			}
		double Pd_002[3];
		double Pd_102[3];
		double Pd_011[3];
		double Pd_111[3];
		double Pd_012[3];
		double Pd_112[3];
		double Pd_212[3];
		double Pd_020[3];
		double Pd_120[3];
		double Pd_021[3];
		double Pd_121[3];
		double Pd_221[3];
		double Pd_022[3];
		double Pd_122[3];
		double Pd_222[3];
		for(int i=0;i<3;i++){
			Pd_002[i]=aPin1+Pd_001[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_102[i]=aPin1*(2.000000*Pd_001[i]);
			}
		for(int i=0;i<3;i++){
			Pd_011[i]=aPin1+Pd_010[i]*Pd_001[i];
			}
		for(int i=0;i<3;i++){
			Pd_111[i]=aPin1*(Pd_001[i]+Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_012[i]=Pd_111[i]+Pd_001[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_112[i]=aPin1*(Pd_002[i]+2.000000*Pd_011[i]);
			}
		for(int i=0;i<3;i++){
			Pd_212[i]=aPin1*(0.500000*Pd_102[i]+Pd_111[i]);
			}
		for(int i=0;i<3;i++){
			Pd_020[i]=aPin1+Pd_010[i]*Pd_010[i];
			}
		for(int i=0;i<3;i++){
			Pd_120[i]=aPin1*(2.000000*Pd_010[i]);
			}
		for(int i=0;i<3;i++){
			Pd_021[i]=Pd_111[i]+Pd_010[i]*Pd_011[i];
			}
		for(int i=0;i<3;i++){
			Pd_121[i]=aPin1*(2.000000*Pd_011[i]+Pd_020[i]);
			}
		for(int i=0;i<3;i++){
			Pd_221[i]=aPin1*(Pd_111[i]+0.500000*Pd_120[i]);
			}
		for(int i=0;i<3;i++){
			Pd_022[i]=Pd_112[i]+Pd_010[i]*Pd_012[i];
			}
		for(int i=0;i<3;i++){
			Pd_122[i]=aPin1*2.000000*(Pd_012[i]+Pd_021[i]);
			}
		for(int i=0;i<3;i++){
			Pd_222[i]=aPin1*(Pd_112[i]+Pd_121[i]);
			}
			double P_022000000;
			double P_122000000;
			double P_222000000;
			double P_021001000;
			double P_121001000;
			double P_221001000;
			double P_020002000;
			double P_021000001;
			double P_121000001;
			double P_221000001;
			double P_020001001;
			double P_020000002;
			double P_012010000;
			double P_112010000;
			double P_212010000;
			double P_011011000;
			double P_011111000;
			double P_111011000;
			double P_111111000;
			double P_010012000;
			double P_010112000;
			double P_010212000;
			double P_011010001;
			double P_111010001;
			double P_010011001;
			double P_010111001;
			double P_010010002;
			double P_002020000;
			double P_001021000;
			double P_001121000;
			double P_001221000;
			double P_000022000;
			double P_000122000;
			double P_000222000;
			double P_001020001;
			double P_000021001;
			double P_000121001;
			double P_000221001;
			double P_000020002;
			double P_012000010;
			double P_112000010;
			double P_212000010;
			double P_011001010;
			double P_111001010;
			double P_010002010;
			double P_011000011;
			double P_011000111;
			double P_111000011;
			double P_111000111;
			double P_010001011;
			double P_010001111;
			double P_010000012;
			double P_010000112;
			double P_010000212;
			double P_002010010;
			double P_001011010;
			double P_001111010;
			double P_000012010;
			double P_000112010;
			double P_000212010;
			double P_001010011;
			double P_001010111;
			double P_000011011;
			double P_000011111;
			double P_000111011;
			double P_000111111;
			double P_000010012;
			double P_000010112;
			double P_000010212;
			double P_002000020;
			double P_001001020;
			double P_000002020;
			double P_001000021;
			double P_001000121;
			double P_001000221;
			double P_000001021;
			double P_000001121;
			double P_000001221;
			double P_000000022;
			double P_000000122;
			double P_000000222;
			double a2P_111000000_1;
			double a2P_111000000_2;
			double a1P_021000000_1;
			double a1P_121000000_1;
			double a1P_221000000_1;
			double a3P_000001000_1;
			double a3P_000001000_2;
			double a1P_020001000_1;
			double a1P_020001000_2;
			double a2P_020000000_1;
			double a1P_010002000_1;
			double a1P_010002000_2;
			double a2P_010001000_1;
			double a2P_010001000_4;
			double a2P_010001000_2;
			double a3P_010000000_1;
			double a3P_010000000_2;
			double a2P_000002000_1;
			double a3P_000000001_1;
			double a3P_000000001_2;
			double a1P_020000001_1;
			double a1P_020000001_2;
			double a1P_010001001_1;
			double a1P_010001001_2;
			double a2P_010000001_1;
			double a2P_010000001_2;
			double a2P_010000001_4;
			double a2P_000001001_1;
			double a1P_010000002_1;
			double a1P_010000002_2;
			double a2P_000000002_1;
			double a1P_012000000_1;
			double a1P_112000000_1;
			double a1P_212000000_1;
			double a3P_000010000_1;
			double a3P_000010000_2;
			double a2P_011000000_1;
			double a2P_000011000_1;
			double a2P_000111000_1;
			double a2P_000111000_2;
			double a1P_000012000_1;
			double a1P_000112000_1;
			double a1P_000212000_1;
			double a1P_011010000_1;
			double a1P_011000001_1;
			double a1P_111010000_1;
			double a1P_111000001_1;
			double a2P_000010001_1;
			double a2P_000010001_2;
			double a2P_000010001_4;
			double a1P_010011000_1;
			double a1P_010111000_1;
			double a1P_000011001_1;
			double a1P_000111001_1;
			double a1P_010010001_1;
			double a1P_010010001_2;
			double a2P_010010000_1;
			double a1P_000010002_1;
			double a1P_000010002_2;
			double a1P_002010000_1;
			double a1P_002010000_2;
			double a2P_002000000_1;
			double a1P_001020000_1;
			double a1P_001020000_2;
			double a2P_001010000_1;
			double a2P_001010000_4;
			double a2P_001010000_2;
			double a3P_001000000_1;
			double a3P_001000000_2;
			double a2P_000020000_1;
			double a1P_000021000_1;
			double a1P_000121000_1;
			double a1P_000221000_1;
			double a1P_001010001_1;
			double a1P_001010001_2;
			double a2P_001000001_1;
			double a1P_000020001_1;
			double a1P_000020001_2;
			double a3P_000000010_1;
			double a3P_000000010_2;
			double a1P_011001000_1;
			double a1P_011000010_1;
			double a1P_111001000_1;
			double a1P_111000010_1;
			double a2P_000001010_1;
			double a2P_000001010_2;
			double a2P_000001010_4;
			double a1P_010001010_1;
			double a1P_010001010_2;
			double a2P_010000010_1;
			double a1P_000002010_1;
			double a1P_000002010_2;
			double a2P_000000011_1;
			double a2P_000000111_1;
			double a2P_000000111_2;
			double a1P_010000011_1;
			double a1P_010000111_1;
			double a1P_000001011_1;
			double a1P_000001111_1;
			double a1P_000000012_1;
			double a1P_000000112_1;
			double a1P_000000212_1;
			double a1P_002000010_1;
			double a1P_002000010_2;
			double a1P_001010010_1;
			double a1P_001010010_2;
			double a2P_001000010_1;
			double a2P_001000010_2;
			double a2P_001000010_4;
			double a2P_000010010_1;
			double a1P_001011000_1;
			double a1P_001111000_1;
			double a1P_000011010_1;
			double a1P_000111010_1;
			double a1P_001000011_1;
			double a1P_001000111_1;
			double a1P_000010011_1;
			double a1P_000010111_1;
			double a1P_001000020_1;
			double a1P_001000020_2;
			double a2P_000000020_1;
			double a1P_001001010_1;
			double a1P_001001010_2;
			double a2P_001001000_1;
			double a1P_000001020_1;
			double a1P_000001020_2;
			double a1P_000000021_1;
			double a1P_000000121_1;
			double a1P_000000221_1;
			P_022000000=Pd_022[0];
			P_122000000=Pd_122[0];
			P_222000000=Pd_222[0];
			P_021001000=Pd_021[0]*Pd_001[1];
			P_121001000=Pd_121[0]*Pd_001[1];
			P_221001000=Pd_221[0]*Pd_001[1];
			P_020002000=Pd_020[0]*Pd_002[1];
			P_021000001=Pd_021[0]*Pd_001[2];
			P_121000001=Pd_121[0]*Pd_001[2];
			P_221000001=Pd_221[0]*Pd_001[2];
			P_020001001=Pd_020[0]*Pd_001[1]*Pd_001[2];
			P_020000002=Pd_020[0]*Pd_002[2];
			P_012010000=Pd_012[0]*Pd_010[1];
			P_112010000=Pd_112[0]*Pd_010[1];
			P_212010000=Pd_212[0]*Pd_010[1];
			P_011011000=Pd_011[0]*Pd_011[1];
			P_011111000=Pd_011[0]*Pd_111[1];
			P_111011000=Pd_111[0]*Pd_011[1];
			P_111111000=Pd_111[0]*Pd_111[1];
			P_010012000=Pd_010[0]*Pd_012[1];
			P_010112000=Pd_010[0]*Pd_112[1];
			P_010212000=Pd_010[0]*Pd_212[1];
			P_011010001=Pd_011[0]*Pd_010[1]*Pd_001[2];
			P_111010001=Pd_111[0]*Pd_010[1]*Pd_001[2];
			P_010011001=Pd_010[0]*Pd_011[1]*Pd_001[2];
			P_010111001=Pd_010[0]*Pd_111[1]*Pd_001[2];
			P_010010002=Pd_010[0]*Pd_010[1]*Pd_002[2];
			P_002020000=Pd_002[0]*Pd_020[1];
			P_001021000=Pd_001[0]*Pd_021[1];
			P_001121000=Pd_001[0]*Pd_121[1];
			P_001221000=Pd_001[0]*Pd_221[1];
			P_000022000=Pd_022[1];
			P_000122000=Pd_122[1];
			P_000222000=Pd_222[1];
			P_001020001=Pd_001[0]*Pd_020[1]*Pd_001[2];
			P_000021001=Pd_021[1]*Pd_001[2];
			P_000121001=Pd_121[1]*Pd_001[2];
			P_000221001=Pd_221[1]*Pd_001[2];
			P_000020002=Pd_020[1]*Pd_002[2];
			P_012000010=Pd_012[0]*Pd_010[2];
			P_112000010=Pd_112[0]*Pd_010[2];
			P_212000010=Pd_212[0]*Pd_010[2];
			P_011001010=Pd_011[0]*Pd_001[1]*Pd_010[2];
			P_111001010=Pd_111[0]*Pd_001[1]*Pd_010[2];
			P_010002010=Pd_010[0]*Pd_002[1]*Pd_010[2];
			P_011000011=Pd_011[0]*Pd_011[2];
			P_011000111=Pd_011[0]*Pd_111[2];
			P_111000011=Pd_111[0]*Pd_011[2];
			P_111000111=Pd_111[0]*Pd_111[2];
			P_010001011=Pd_010[0]*Pd_001[1]*Pd_011[2];
			P_010001111=Pd_010[0]*Pd_001[1]*Pd_111[2];
			P_010000012=Pd_010[0]*Pd_012[2];
			P_010000112=Pd_010[0]*Pd_112[2];
			P_010000212=Pd_010[0]*Pd_212[2];
			P_002010010=Pd_002[0]*Pd_010[1]*Pd_010[2];
			P_001011010=Pd_001[0]*Pd_011[1]*Pd_010[2];
			P_001111010=Pd_001[0]*Pd_111[1]*Pd_010[2];
			P_000012010=Pd_012[1]*Pd_010[2];
			P_000112010=Pd_112[1]*Pd_010[2];
			P_000212010=Pd_212[1]*Pd_010[2];
			P_001010011=Pd_001[0]*Pd_010[1]*Pd_011[2];
			P_001010111=Pd_001[0]*Pd_010[1]*Pd_111[2];
			P_000011011=Pd_011[1]*Pd_011[2];
			P_000011111=Pd_011[1]*Pd_111[2];
			P_000111011=Pd_111[1]*Pd_011[2];
			P_000111111=Pd_111[1]*Pd_111[2];
			P_000010012=Pd_010[1]*Pd_012[2];
			P_000010112=Pd_010[1]*Pd_112[2];
			P_000010212=Pd_010[1]*Pd_212[2];
			P_002000020=Pd_002[0]*Pd_020[2];
			P_001001020=Pd_001[0]*Pd_001[1]*Pd_020[2];
			P_000002020=Pd_002[1]*Pd_020[2];
			P_001000021=Pd_001[0]*Pd_021[2];
			P_001000121=Pd_001[0]*Pd_121[2];
			P_001000221=Pd_001[0]*Pd_221[2];
			P_000001021=Pd_001[1]*Pd_021[2];
			P_000001121=Pd_001[1]*Pd_121[2];
			P_000001221=Pd_001[1]*Pd_221[2];
			P_000000022=Pd_022[2];
			P_000000122=Pd_122[2];
			P_000000222=Pd_222[2];
			a2P_111000000_1=Pd_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=Pd_021[0];
			a1P_121000000_1=Pd_121[0];
			a1P_221000000_1=Pd_221[0];
			a3P_000001000_1=Pd_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=Pd_020[0]*Pd_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=Pd_020[0];
			a1P_010002000_1=Pd_010[0]*Pd_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=Pd_010[0]*Pd_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=Pd_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=Pd_002[1];
			a3P_000000001_1=Pd_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=Pd_020[0]*Pd_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=Pd_010[0]*Pd_001[1]*Pd_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=Pd_010[0]*Pd_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=Pd_001[1]*Pd_001[2];
			a1P_010000002_1=Pd_010[0]*Pd_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=Pd_002[2];
			a1P_012000000_1=Pd_012[0];
			a1P_112000000_1=Pd_112[0];
			a1P_212000000_1=Pd_212[0];
			a3P_000010000_1=Pd_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=Pd_011[0];
			a2P_000011000_1=Pd_011[1];
			a2P_000111000_1=Pd_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=Pd_012[1];
			a1P_000112000_1=Pd_112[1];
			a1P_000212000_1=Pd_212[1];
			a1P_011010000_1=Pd_011[0]*Pd_010[1];
			a1P_011000001_1=Pd_011[0]*Pd_001[2];
			a1P_111010000_1=Pd_111[0]*Pd_010[1];
			a1P_111000001_1=Pd_111[0]*Pd_001[2];
			a2P_000010001_1=Pd_010[1]*Pd_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=Pd_010[0]*Pd_011[1];
			a1P_010111000_1=Pd_010[0]*Pd_111[1];
			a1P_000011001_1=Pd_011[1]*Pd_001[2];
			a1P_000111001_1=Pd_111[1]*Pd_001[2];
			a1P_010010001_1=Pd_010[0]*Pd_010[1]*Pd_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=Pd_010[0]*Pd_010[1];
			a1P_000010002_1=Pd_010[1]*Pd_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=Pd_002[0]*Pd_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=Pd_002[0];
			a1P_001020000_1=Pd_001[0]*Pd_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=Pd_001[0]*Pd_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=Pd_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=Pd_020[1];
			a1P_000021000_1=Pd_021[1];
			a1P_000121000_1=Pd_121[1];
			a1P_000221000_1=Pd_221[1];
			a1P_001010001_1=Pd_001[0]*Pd_010[1]*Pd_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=Pd_001[0]*Pd_001[2];
			a1P_000020001_1=Pd_020[1]*Pd_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=Pd_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=Pd_011[0]*Pd_001[1];
			a1P_011000010_1=Pd_011[0]*Pd_010[2];
			a1P_111001000_1=Pd_111[0]*Pd_001[1];
			a1P_111000010_1=Pd_111[0]*Pd_010[2];
			a2P_000001010_1=Pd_001[1]*Pd_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=Pd_010[0]*Pd_001[1]*Pd_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=Pd_010[0]*Pd_010[2];
			a1P_000002010_1=Pd_002[1]*Pd_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=Pd_011[2];
			a2P_000000111_1=Pd_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=Pd_010[0]*Pd_011[2];
			a1P_010000111_1=Pd_010[0]*Pd_111[2];
			a1P_000001011_1=Pd_001[1]*Pd_011[2];
			a1P_000001111_1=Pd_001[1]*Pd_111[2];
			a1P_000000012_1=Pd_012[2];
			a1P_000000112_1=Pd_112[2];
			a1P_000000212_1=Pd_212[2];
			a1P_002000010_1=Pd_002[0]*Pd_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=Pd_001[0]*Pd_010[1]*Pd_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=Pd_001[0]*Pd_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=Pd_010[1]*Pd_010[2];
			a1P_001011000_1=Pd_001[0]*Pd_011[1];
			a1P_001111000_1=Pd_001[0]*Pd_111[1];
			a1P_000011010_1=Pd_011[1]*Pd_010[2];
			a1P_000111010_1=Pd_111[1]*Pd_010[2];
			a1P_001000011_1=Pd_001[0]*Pd_011[2];
			a1P_001000111_1=Pd_001[0]*Pd_111[2];
			a1P_000010011_1=Pd_010[1]*Pd_011[2];
			a1P_000010111_1=Pd_010[1]*Pd_111[2];
			a1P_001000020_1=Pd_001[0]*Pd_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=Pd_020[2];
			a1P_001001010_1=Pd_001[0]*Pd_001[1]*Pd_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=Pd_001[0]*Pd_001[1];
			a1P_000001020_1=Pd_001[1]*Pd_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=Pd_021[2];
			a1P_000000121_1=Pd_121[2];
			a1P_000000221_1=Pd_221[2];
			ans_temp[ans_id*36+0]+=Pmtrx[0]*(P_022000000*QR_020000000000+P_122000000*QR_020000000100+P_222000000*QR_020000000200+a2P_111000000_2*QR_020000000300+aPin4*QR_020000000400);
			ans_temp[ans_id*36+0]+=Pmtrx[1]*(P_022000000*QR_010010000000+P_122000000*QR_010010000100+P_222000000*QR_010010000200+a2P_111000000_2*QR_010010000300+aPin4*QR_010010000400);
			ans_temp[ans_id*36+0]+=Pmtrx[2]*(P_022000000*QR_000020000000+P_122000000*QR_000020000100+P_222000000*QR_000020000200+a2P_111000000_2*QR_000020000300+aPin4*QR_000020000400);
			ans_temp[ans_id*36+0]+=Pmtrx[3]*(P_022000000*QR_010000010000+P_122000000*QR_010000010100+P_222000000*QR_010000010200+a2P_111000000_2*QR_010000010300+aPin4*QR_010000010400);
			ans_temp[ans_id*36+0]+=Pmtrx[4]*(P_022000000*QR_000010010000+P_122000000*QR_000010010100+P_222000000*QR_000010010200+a2P_111000000_2*QR_000010010300+aPin4*QR_000010010400);
			ans_temp[ans_id*36+0]+=Pmtrx[5]*(P_022000000*QR_000000020000+P_122000000*QR_000000020100+P_222000000*QR_000000020200+a2P_111000000_2*QR_000000020300+aPin4*QR_000000020400);
			ans_temp[ans_id*36+1]+=Pmtrx[0]*(P_021001000*QR_020000000000+a1P_021000000_1*QR_020000000010+P_121001000*QR_020000000100+a1P_121000000_1*QR_020000000110+P_221001000*QR_020000000200+a1P_221000000_1*QR_020000000210+a3P_000001000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+1]+=Pmtrx[1]*(P_021001000*QR_010010000000+a1P_021000000_1*QR_010010000010+P_121001000*QR_010010000100+a1P_121000000_1*QR_010010000110+P_221001000*QR_010010000200+a1P_221000000_1*QR_010010000210+a3P_000001000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+1]+=Pmtrx[2]*(P_021001000*QR_000020000000+a1P_021000000_1*QR_000020000010+P_121001000*QR_000020000100+a1P_121000000_1*QR_000020000110+P_221001000*QR_000020000200+a1P_221000000_1*QR_000020000210+a3P_000001000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+1]+=Pmtrx[3]*(P_021001000*QR_010000010000+a1P_021000000_1*QR_010000010010+P_121001000*QR_010000010100+a1P_121000000_1*QR_010000010110+P_221001000*QR_010000010200+a1P_221000000_1*QR_010000010210+a3P_000001000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+1]+=Pmtrx[4]*(P_021001000*QR_000010010000+a1P_021000000_1*QR_000010010010+P_121001000*QR_000010010100+a1P_121000000_1*QR_000010010110+P_221001000*QR_000010010200+a1P_221000000_1*QR_000010010210+a3P_000001000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+1]+=Pmtrx[5]*(P_021001000*QR_000000020000+a1P_021000000_1*QR_000000020010+P_121001000*QR_000000020100+a1P_121000000_1*QR_000000020110+P_221001000*QR_000000020200+a1P_221000000_1*QR_000000020210+a3P_000001000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+2]+=Pmtrx[0]*(P_020002000*QR_020000000000+a1P_020001000_2*QR_020000000010+a2P_020000000_1*QR_020000000020+a1P_010002000_2*QR_020000000100+a2P_010001000_4*QR_020000000110+a3P_010000000_2*QR_020000000120+a2P_000002000_1*QR_020000000200+a3P_000001000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+2]+=Pmtrx[1]*(P_020002000*QR_010010000000+a1P_020001000_2*QR_010010000010+a2P_020000000_1*QR_010010000020+a1P_010002000_2*QR_010010000100+a2P_010001000_4*QR_010010000110+a3P_010000000_2*QR_010010000120+a2P_000002000_1*QR_010010000200+a3P_000001000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+2]+=Pmtrx[2]*(P_020002000*QR_000020000000+a1P_020001000_2*QR_000020000010+a2P_020000000_1*QR_000020000020+a1P_010002000_2*QR_000020000100+a2P_010001000_4*QR_000020000110+a3P_010000000_2*QR_000020000120+a2P_000002000_1*QR_000020000200+a3P_000001000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+2]+=Pmtrx[3]*(P_020002000*QR_010000010000+a1P_020001000_2*QR_010000010010+a2P_020000000_1*QR_010000010020+a1P_010002000_2*QR_010000010100+a2P_010001000_4*QR_010000010110+a3P_010000000_2*QR_010000010120+a2P_000002000_1*QR_010000010200+a3P_000001000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+2]+=Pmtrx[4]*(P_020002000*QR_000010010000+a1P_020001000_2*QR_000010010010+a2P_020000000_1*QR_000010010020+a1P_010002000_2*QR_000010010100+a2P_010001000_4*QR_000010010110+a3P_010000000_2*QR_000010010120+a2P_000002000_1*QR_000010010200+a3P_000001000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+2]+=Pmtrx[5]*(P_020002000*QR_000000020000+a1P_020001000_2*QR_000000020010+a2P_020000000_1*QR_000000020020+a1P_010002000_2*QR_000000020100+a2P_010001000_4*QR_000000020110+a3P_010000000_2*QR_000000020120+a2P_000002000_1*QR_000000020200+a3P_000001000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+3]+=Pmtrx[0]*(P_021000001*QR_020000000000+a1P_021000000_1*QR_020000000001+P_121000001*QR_020000000100+a1P_121000000_1*QR_020000000101+P_221000001*QR_020000000200+a1P_221000000_1*QR_020000000201+a3P_000000001_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+3]+=Pmtrx[1]*(P_021000001*QR_010010000000+a1P_021000000_1*QR_010010000001+P_121000001*QR_010010000100+a1P_121000000_1*QR_010010000101+P_221000001*QR_010010000200+a1P_221000000_1*QR_010010000201+a3P_000000001_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+3]+=Pmtrx[2]*(P_021000001*QR_000020000000+a1P_021000000_1*QR_000020000001+P_121000001*QR_000020000100+a1P_121000000_1*QR_000020000101+P_221000001*QR_000020000200+a1P_221000000_1*QR_000020000201+a3P_000000001_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+3]+=Pmtrx[3]*(P_021000001*QR_010000010000+a1P_021000000_1*QR_010000010001+P_121000001*QR_010000010100+a1P_121000000_1*QR_010000010101+P_221000001*QR_010000010200+a1P_221000000_1*QR_010000010201+a3P_000000001_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+3]+=Pmtrx[4]*(P_021000001*QR_000010010000+a1P_021000000_1*QR_000010010001+P_121000001*QR_000010010100+a1P_121000000_1*QR_000010010101+P_221000001*QR_000010010200+a1P_221000000_1*QR_000010010201+a3P_000000001_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+3]+=Pmtrx[5]*(P_021000001*QR_000000020000+a1P_021000000_1*QR_000000020001+P_121000001*QR_000000020100+a1P_121000000_1*QR_000000020101+P_221000001*QR_000000020200+a1P_221000000_1*QR_000000020201+a3P_000000001_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+4]+=Pmtrx[0]*(P_020001001*QR_020000000000+a1P_020001000_1*QR_020000000001+a1P_020000001_1*QR_020000000010+a2P_020000000_1*QR_020000000011+a1P_010001001_2*QR_020000000100+a2P_010001000_2*QR_020000000101+a2P_010000001_2*QR_020000000110+a3P_010000000_2*QR_020000000111+a2P_000001001_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+4]+=Pmtrx[1]*(P_020001001*QR_010010000000+a1P_020001000_1*QR_010010000001+a1P_020000001_1*QR_010010000010+a2P_020000000_1*QR_010010000011+a1P_010001001_2*QR_010010000100+a2P_010001000_2*QR_010010000101+a2P_010000001_2*QR_010010000110+a3P_010000000_2*QR_010010000111+a2P_000001001_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+4]+=Pmtrx[2]*(P_020001001*QR_000020000000+a1P_020001000_1*QR_000020000001+a1P_020000001_1*QR_000020000010+a2P_020000000_1*QR_000020000011+a1P_010001001_2*QR_000020000100+a2P_010001000_2*QR_000020000101+a2P_010000001_2*QR_000020000110+a3P_010000000_2*QR_000020000111+a2P_000001001_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+4]+=Pmtrx[3]*(P_020001001*QR_010000010000+a1P_020001000_1*QR_010000010001+a1P_020000001_1*QR_010000010010+a2P_020000000_1*QR_010000010011+a1P_010001001_2*QR_010000010100+a2P_010001000_2*QR_010000010101+a2P_010000001_2*QR_010000010110+a3P_010000000_2*QR_010000010111+a2P_000001001_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+4]+=Pmtrx[4]*(P_020001001*QR_000010010000+a1P_020001000_1*QR_000010010001+a1P_020000001_1*QR_000010010010+a2P_020000000_1*QR_000010010011+a1P_010001001_2*QR_000010010100+a2P_010001000_2*QR_000010010101+a2P_010000001_2*QR_000010010110+a3P_010000000_2*QR_000010010111+a2P_000001001_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+4]+=Pmtrx[5]*(P_020001001*QR_000000020000+a1P_020001000_1*QR_000000020001+a1P_020000001_1*QR_000000020010+a2P_020000000_1*QR_000000020011+a1P_010001001_2*QR_000000020100+a2P_010001000_2*QR_000000020101+a2P_010000001_2*QR_000000020110+a3P_010000000_2*QR_000000020111+a2P_000001001_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+5]+=Pmtrx[0]*(P_020000002*QR_020000000000+a1P_020000001_2*QR_020000000001+a2P_020000000_1*QR_020000000002+a1P_010000002_2*QR_020000000100+a2P_010000001_4*QR_020000000101+a3P_010000000_2*QR_020000000102+a2P_000000002_1*QR_020000000200+a3P_000000001_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+5]+=Pmtrx[1]*(P_020000002*QR_010010000000+a1P_020000001_2*QR_010010000001+a2P_020000000_1*QR_010010000002+a1P_010000002_2*QR_010010000100+a2P_010000001_4*QR_010010000101+a3P_010000000_2*QR_010010000102+a2P_000000002_1*QR_010010000200+a3P_000000001_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+5]+=Pmtrx[2]*(P_020000002*QR_000020000000+a1P_020000001_2*QR_000020000001+a2P_020000000_1*QR_000020000002+a1P_010000002_2*QR_000020000100+a2P_010000001_4*QR_000020000101+a3P_010000000_2*QR_000020000102+a2P_000000002_1*QR_000020000200+a3P_000000001_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+5]+=Pmtrx[3]*(P_020000002*QR_010000010000+a1P_020000001_2*QR_010000010001+a2P_020000000_1*QR_010000010002+a1P_010000002_2*QR_010000010100+a2P_010000001_4*QR_010000010101+a3P_010000000_2*QR_010000010102+a2P_000000002_1*QR_010000010200+a3P_000000001_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+5]+=Pmtrx[4]*(P_020000002*QR_000010010000+a1P_020000001_2*QR_000010010001+a2P_020000000_1*QR_000010010002+a1P_010000002_2*QR_000010010100+a2P_010000001_4*QR_000010010101+a3P_010000000_2*QR_000010010102+a2P_000000002_1*QR_000010010200+a3P_000000001_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+5]+=Pmtrx[5]*(P_020000002*QR_000000020000+a1P_020000001_2*QR_000000020001+a2P_020000000_1*QR_000000020002+a1P_010000002_2*QR_000000020100+a2P_010000001_4*QR_000000020101+a3P_010000000_2*QR_000000020102+a2P_000000002_1*QR_000000020200+a3P_000000001_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+6]+=Pmtrx[0]*(P_012010000*QR_020000000000+a1P_012000000_1*QR_020000000010+P_112010000*QR_020000000100+a1P_112000000_1*QR_020000000110+P_212010000*QR_020000000200+a1P_212000000_1*QR_020000000210+a3P_000010000_1*QR_020000000300+aPin4*QR_020000000310);
			ans_temp[ans_id*36+6]+=Pmtrx[1]*(P_012010000*QR_010010000000+a1P_012000000_1*QR_010010000010+P_112010000*QR_010010000100+a1P_112000000_1*QR_010010000110+P_212010000*QR_010010000200+a1P_212000000_1*QR_010010000210+a3P_000010000_1*QR_010010000300+aPin4*QR_010010000310);
			ans_temp[ans_id*36+6]+=Pmtrx[2]*(P_012010000*QR_000020000000+a1P_012000000_1*QR_000020000010+P_112010000*QR_000020000100+a1P_112000000_1*QR_000020000110+P_212010000*QR_000020000200+a1P_212000000_1*QR_000020000210+a3P_000010000_1*QR_000020000300+aPin4*QR_000020000310);
			ans_temp[ans_id*36+6]+=Pmtrx[3]*(P_012010000*QR_010000010000+a1P_012000000_1*QR_010000010010+P_112010000*QR_010000010100+a1P_112000000_1*QR_010000010110+P_212010000*QR_010000010200+a1P_212000000_1*QR_010000010210+a3P_000010000_1*QR_010000010300+aPin4*QR_010000010310);
			ans_temp[ans_id*36+6]+=Pmtrx[4]*(P_012010000*QR_000010010000+a1P_012000000_1*QR_000010010010+P_112010000*QR_000010010100+a1P_112000000_1*QR_000010010110+P_212010000*QR_000010010200+a1P_212000000_1*QR_000010010210+a3P_000010000_1*QR_000010010300+aPin4*QR_000010010310);
			ans_temp[ans_id*36+6]+=Pmtrx[5]*(P_012010000*QR_000000020000+a1P_012000000_1*QR_000000020010+P_112010000*QR_000000020100+a1P_112000000_1*QR_000000020110+P_212010000*QR_000000020200+a1P_212000000_1*QR_000000020210+a3P_000010000_1*QR_000000020300+aPin4*QR_000000020310);
			ans_temp[ans_id*36+7]+=Pmtrx[0]*(P_011011000*QR_020000000000+P_011111000*QR_020000000010+a2P_011000000_1*QR_020000000020+P_111011000*QR_020000000100+P_111111000*QR_020000000110+a2P_111000000_1*QR_020000000120+a2P_000011000_1*QR_020000000200+a2P_000111000_1*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+7]+=Pmtrx[1]*(P_011011000*QR_010010000000+P_011111000*QR_010010000010+a2P_011000000_1*QR_010010000020+P_111011000*QR_010010000100+P_111111000*QR_010010000110+a2P_111000000_1*QR_010010000120+a2P_000011000_1*QR_010010000200+a2P_000111000_1*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+7]+=Pmtrx[2]*(P_011011000*QR_000020000000+P_011111000*QR_000020000010+a2P_011000000_1*QR_000020000020+P_111011000*QR_000020000100+P_111111000*QR_000020000110+a2P_111000000_1*QR_000020000120+a2P_000011000_1*QR_000020000200+a2P_000111000_1*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+7]+=Pmtrx[3]*(P_011011000*QR_010000010000+P_011111000*QR_010000010010+a2P_011000000_1*QR_010000010020+P_111011000*QR_010000010100+P_111111000*QR_010000010110+a2P_111000000_1*QR_010000010120+a2P_000011000_1*QR_010000010200+a2P_000111000_1*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+7]+=Pmtrx[4]*(P_011011000*QR_000010010000+P_011111000*QR_000010010010+a2P_011000000_1*QR_000010010020+P_111011000*QR_000010010100+P_111111000*QR_000010010110+a2P_111000000_1*QR_000010010120+a2P_000011000_1*QR_000010010200+a2P_000111000_1*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+7]+=Pmtrx[5]*(P_011011000*QR_000000020000+P_011111000*QR_000000020010+a2P_011000000_1*QR_000000020020+P_111011000*QR_000000020100+P_111111000*QR_000000020110+a2P_111000000_1*QR_000000020120+a2P_000011000_1*QR_000000020200+a2P_000111000_1*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+8]+=Pmtrx[0]*(P_010012000*QR_020000000000+P_010112000*QR_020000000010+P_010212000*QR_020000000020+a3P_010000000_1*QR_020000000030+a1P_000012000_1*QR_020000000100+a1P_000112000_1*QR_020000000110+a1P_000212000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+8]+=Pmtrx[1]*(P_010012000*QR_010010000000+P_010112000*QR_010010000010+P_010212000*QR_010010000020+a3P_010000000_1*QR_010010000030+a1P_000012000_1*QR_010010000100+a1P_000112000_1*QR_010010000110+a1P_000212000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+8]+=Pmtrx[2]*(P_010012000*QR_000020000000+P_010112000*QR_000020000010+P_010212000*QR_000020000020+a3P_010000000_1*QR_000020000030+a1P_000012000_1*QR_000020000100+a1P_000112000_1*QR_000020000110+a1P_000212000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+8]+=Pmtrx[3]*(P_010012000*QR_010000010000+P_010112000*QR_010000010010+P_010212000*QR_010000010020+a3P_010000000_1*QR_010000010030+a1P_000012000_1*QR_010000010100+a1P_000112000_1*QR_010000010110+a1P_000212000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+8]+=Pmtrx[4]*(P_010012000*QR_000010010000+P_010112000*QR_000010010010+P_010212000*QR_000010010020+a3P_010000000_1*QR_000010010030+a1P_000012000_1*QR_000010010100+a1P_000112000_1*QR_000010010110+a1P_000212000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+8]+=Pmtrx[5]*(P_010012000*QR_000000020000+P_010112000*QR_000000020010+P_010212000*QR_000000020020+a3P_010000000_1*QR_000000020030+a1P_000012000_1*QR_000000020100+a1P_000112000_1*QR_000000020110+a1P_000212000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+9]+=Pmtrx[0]*(P_011010001*QR_020000000000+a1P_011010000_1*QR_020000000001+a1P_011000001_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111010001*QR_020000000100+a1P_111010000_1*QR_020000000101+a1P_111000001_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000010001_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000001_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+9]+=Pmtrx[1]*(P_011010001*QR_010010000000+a1P_011010000_1*QR_010010000001+a1P_011000001_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111010001*QR_010010000100+a1P_111010000_1*QR_010010000101+a1P_111000001_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000010001_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000001_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+9]+=Pmtrx[2]*(P_011010001*QR_000020000000+a1P_011010000_1*QR_000020000001+a1P_011000001_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111010001*QR_000020000100+a1P_111010000_1*QR_000020000101+a1P_111000001_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000010001_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000001_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+9]+=Pmtrx[3]*(P_011010001*QR_010000010000+a1P_011010000_1*QR_010000010001+a1P_011000001_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111010001*QR_010000010100+a1P_111010000_1*QR_010000010101+a1P_111000001_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000010001_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000001_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+9]+=Pmtrx[4]*(P_011010001*QR_000010010000+a1P_011010000_1*QR_000010010001+a1P_011000001_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111010001*QR_000010010100+a1P_111010000_1*QR_000010010101+a1P_111000001_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000010001_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000001_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+9]+=Pmtrx[5]*(P_011010001*QR_000000020000+a1P_011010000_1*QR_000000020001+a1P_011000001_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111010001*QR_000000020100+a1P_111010000_1*QR_000000020101+a1P_111000001_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000010001_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000001_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+10]+=Pmtrx[0]*(P_010011001*QR_020000000000+a1P_010011000_1*QR_020000000001+P_010111001*QR_020000000010+a1P_010111000_1*QR_020000000011+a2P_010000001_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000011001_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111001_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+10]+=Pmtrx[1]*(P_010011001*QR_010010000000+a1P_010011000_1*QR_010010000001+P_010111001*QR_010010000010+a1P_010111000_1*QR_010010000011+a2P_010000001_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000011001_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111001_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+10]+=Pmtrx[2]*(P_010011001*QR_000020000000+a1P_010011000_1*QR_000020000001+P_010111001*QR_000020000010+a1P_010111000_1*QR_000020000011+a2P_010000001_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000011001_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111001_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+10]+=Pmtrx[3]*(P_010011001*QR_010000010000+a1P_010011000_1*QR_010000010001+P_010111001*QR_010000010010+a1P_010111000_1*QR_010000010011+a2P_010000001_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000011001_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111001_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+10]+=Pmtrx[4]*(P_010011001*QR_000010010000+a1P_010011000_1*QR_000010010001+P_010111001*QR_000010010010+a1P_010111000_1*QR_000010010011+a2P_010000001_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000011001_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111001_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+10]+=Pmtrx[5]*(P_010011001*QR_000000020000+a1P_010011000_1*QR_000000020001+P_010111001*QR_000000020010+a1P_010111000_1*QR_000000020011+a2P_010000001_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000011001_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111001_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+11]+=Pmtrx[0]*(P_010010002*QR_020000000000+a1P_010010001_2*QR_020000000001+a2P_010010000_1*QR_020000000002+a1P_010000002_1*QR_020000000010+a2P_010000001_2*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000010002_1*QR_020000000100+a2P_000010001_2*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000002_1*QR_020000000110+a3P_000000001_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+11]+=Pmtrx[1]*(P_010010002*QR_010010000000+a1P_010010001_2*QR_010010000001+a2P_010010000_1*QR_010010000002+a1P_010000002_1*QR_010010000010+a2P_010000001_2*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000010002_1*QR_010010000100+a2P_000010001_2*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000002_1*QR_010010000110+a3P_000000001_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+11]+=Pmtrx[2]*(P_010010002*QR_000020000000+a1P_010010001_2*QR_000020000001+a2P_010010000_1*QR_000020000002+a1P_010000002_1*QR_000020000010+a2P_010000001_2*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000010002_1*QR_000020000100+a2P_000010001_2*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000002_1*QR_000020000110+a3P_000000001_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+11]+=Pmtrx[3]*(P_010010002*QR_010000010000+a1P_010010001_2*QR_010000010001+a2P_010010000_1*QR_010000010002+a1P_010000002_1*QR_010000010010+a2P_010000001_2*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000010002_1*QR_010000010100+a2P_000010001_2*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000002_1*QR_010000010110+a3P_000000001_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+11]+=Pmtrx[4]*(P_010010002*QR_000010010000+a1P_010010001_2*QR_000010010001+a2P_010010000_1*QR_000010010002+a1P_010000002_1*QR_000010010010+a2P_010000001_2*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000010002_1*QR_000010010100+a2P_000010001_2*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000002_1*QR_000010010110+a3P_000000001_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+11]+=Pmtrx[5]*(P_010010002*QR_000000020000+a1P_010010001_2*QR_000000020001+a2P_010010000_1*QR_000000020002+a1P_010000002_1*QR_000000020010+a2P_010000001_2*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000010002_1*QR_000000020100+a2P_000010001_2*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000002_1*QR_000000020110+a3P_000000001_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+12]+=Pmtrx[0]*(P_002020000*QR_020000000000+a1P_002010000_2*QR_020000000010+a2P_002000000_1*QR_020000000020+a1P_001020000_2*QR_020000000100+a2P_001010000_4*QR_020000000110+a3P_001000000_2*QR_020000000120+a2P_000020000_1*QR_020000000200+a3P_000010000_2*QR_020000000210+aPin4*QR_020000000220);
			ans_temp[ans_id*36+12]+=Pmtrx[1]*(P_002020000*QR_010010000000+a1P_002010000_2*QR_010010000010+a2P_002000000_1*QR_010010000020+a1P_001020000_2*QR_010010000100+a2P_001010000_4*QR_010010000110+a3P_001000000_2*QR_010010000120+a2P_000020000_1*QR_010010000200+a3P_000010000_2*QR_010010000210+aPin4*QR_010010000220);
			ans_temp[ans_id*36+12]+=Pmtrx[2]*(P_002020000*QR_000020000000+a1P_002010000_2*QR_000020000010+a2P_002000000_1*QR_000020000020+a1P_001020000_2*QR_000020000100+a2P_001010000_4*QR_000020000110+a3P_001000000_2*QR_000020000120+a2P_000020000_1*QR_000020000200+a3P_000010000_2*QR_000020000210+aPin4*QR_000020000220);
			ans_temp[ans_id*36+12]+=Pmtrx[3]*(P_002020000*QR_010000010000+a1P_002010000_2*QR_010000010010+a2P_002000000_1*QR_010000010020+a1P_001020000_2*QR_010000010100+a2P_001010000_4*QR_010000010110+a3P_001000000_2*QR_010000010120+a2P_000020000_1*QR_010000010200+a3P_000010000_2*QR_010000010210+aPin4*QR_010000010220);
			ans_temp[ans_id*36+12]+=Pmtrx[4]*(P_002020000*QR_000010010000+a1P_002010000_2*QR_000010010010+a2P_002000000_1*QR_000010010020+a1P_001020000_2*QR_000010010100+a2P_001010000_4*QR_000010010110+a3P_001000000_2*QR_000010010120+a2P_000020000_1*QR_000010010200+a3P_000010000_2*QR_000010010210+aPin4*QR_000010010220);
			ans_temp[ans_id*36+12]+=Pmtrx[5]*(P_002020000*QR_000000020000+a1P_002010000_2*QR_000000020010+a2P_002000000_1*QR_000000020020+a1P_001020000_2*QR_000000020100+a2P_001010000_4*QR_000000020110+a3P_001000000_2*QR_000000020120+a2P_000020000_1*QR_000000020200+a3P_000010000_2*QR_000000020210+aPin4*QR_000000020220);
			ans_temp[ans_id*36+13]+=Pmtrx[0]*(P_001021000*QR_020000000000+P_001121000*QR_020000000010+P_001221000*QR_020000000020+a3P_001000000_1*QR_020000000030+a1P_000021000_1*QR_020000000100+a1P_000121000_1*QR_020000000110+a1P_000221000_1*QR_020000000120+aPin4*QR_020000000130);
			ans_temp[ans_id*36+13]+=Pmtrx[1]*(P_001021000*QR_010010000000+P_001121000*QR_010010000010+P_001221000*QR_010010000020+a3P_001000000_1*QR_010010000030+a1P_000021000_1*QR_010010000100+a1P_000121000_1*QR_010010000110+a1P_000221000_1*QR_010010000120+aPin4*QR_010010000130);
			ans_temp[ans_id*36+13]+=Pmtrx[2]*(P_001021000*QR_000020000000+P_001121000*QR_000020000010+P_001221000*QR_000020000020+a3P_001000000_1*QR_000020000030+a1P_000021000_1*QR_000020000100+a1P_000121000_1*QR_000020000110+a1P_000221000_1*QR_000020000120+aPin4*QR_000020000130);
			ans_temp[ans_id*36+13]+=Pmtrx[3]*(P_001021000*QR_010000010000+P_001121000*QR_010000010010+P_001221000*QR_010000010020+a3P_001000000_1*QR_010000010030+a1P_000021000_1*QR_010000010100+a1P_000121000_1*QR_010000010110+a1P_000221000_1*QR_010000010120+aPin4*QR_010000010130);
			ans_temp[ans_id*36+13]+=Pmtrx[4]*(P_001021000*QR_000010010000+P_001121000*QR_000010010010+P_001221000*QR_000010010020+a3P_001000000_1*QR_000010010030+a1P_000021000_1*QR_000010010100+a1P_000121000_1*QR_000010010110+a1P_000221000_1*QR_000010010120+aPin4*QR_000010010130);
			ans_temp[ans_id*36+13]+=Pmtrx[5]*(P_001021000*QR_000000020000+P_001121000*QR_000000020010+P_001221000*QR_000000020020+a3P_001000000_1*QR_000000020030+a1P_000021000_1*QR_000000020100+a1P_000121000_1*QR_000000020110+a1P_000221000_1*QR_000000020120+aPin4*QR_000000020130);
			ans_temp[ans_id*36+14]+=Pmtrx[0]*(P_000022000*QR_020000000000+P_000122000*QR_020000000010+P_000222000*QR_020000000020+a2P_000111000_2*QR_020000000030+aPin4*QR_020000000040);
			ans_temp[ans_id*36+14]+=Pmtrx[1]*(P_000022000*QR_010010000000+P_000122000*QR_010010000010+P_000222000*QR_010010000020+a2P_000111000_2*QR_010010000030+aPin4*QR_010010000040);
			ans_temp[ans_id*36+14]+=Pmtrx[2]*(P_000022000*QR_000020000000+P_000122000*QR_000020000010+P_000222000*QR_000020000020+a2P_000111000_2*QR_000020000030+aPin4*QR_000020000040);
			ans_temp[ans_id*36+14]+=Pmtrx[3]*(P_000022000*QR_010000010000+P_000122000*QR_010000010010+P_000222000*QR_010000010020+a2P_000111000_2*QR_010000010030+aPin4*QR_010000010040);
			ans_temp[ans_id*36+14]+=Pmtrx[4]*(P_000022000*QR_000010010000+P_000122000*QR_000010010010+P_000222000*QR_000010010020+a2P_000111000_2*QR_000010010030+aPin4*QR_000010010040);
			ans_temp[ans_id*36+14]+=Pmtrx[5]*(P_000022000*QR_000000020000+P_000122000*QR_000000020010+P_000222000*QR_000000020020+a2P_000111000_2*QR_000000020030+aPin4*QR_000000020040);
			ans_temp[ans_id*36+15]+=Pmtrx[0]*(P_001020001*QR_020000000000+a1P_001020000_1*QR_020000000001+a1P_001010001_2*QR_020000000010+a2P_001010000_2*QR_020000000011+a2P_001000001_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000020001_1*QR_020000000100+a2P_000020000_1*QR_020000000101+a2P_000010001_2*QR_020000000110+a3P_000010000_2*QR_020000000111+a3P_000000001_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+15]+=Pmtrx[1]*(P_001020001*QR_010010000000+a1P_001020000_1*QR_010010000001+a1P_001010001_2*QR_010010000010+a2P_001010000_2*QR_010010000011+a2P_001000001_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000020001_1*QR_010010000100+a2P_000020000_1*QR_010010000101+a2P_000010001_2*QR_010010000110+a3P_000010000_2*QR_010010000111+a3P_000000001_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+15]+=Pmtrx[2]*(P_001020001*QR_000020000000+a1P_001020000_1*QR_000020000001+a1P_001010001_2*QR_000020000010+a2P_001010000_2*QR_000020000011+a2P_001000001_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000020001_1*QR_000020000100+a2P_000020000_1*QR_000020000101+a2P_000010001_2*QR_000020000110+a3P_000010000_2*QR_000020000111+a3P_000000001_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+15]+=Pmtrx[3]*(P_001020001*QR_010000010000+a1P_001020000_1*QR_010000010001+a1P_001010001_2*QR_010000010010+a2P_001010000_2*QR_010000010011+a2P_001000001_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000020001_1*QR_010000010100+a2P_000020000_1*QR_010000010101+a2P_000010001_2*QR_010000010110+a3P_000010000_2*QR_010000010111+a3P_000000001_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+15]+=Pmtrx[4]*(P_001020001*QR_000010010000+a1P_001020000_1*QR_000010010001+a1P_001010001_2*QR_000010010010+a2P_001010000_2*QR_000010010011+a2P_001000001_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000020001_1*QR_000010010100+a2P_000020000_1*QR_000010010101+a2P_000010001_2*QR_000010010110+a3P_000010000_2*QR_000010010111+a3P_000000001_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+15]+=Pmtrx[5]*(P_001020001*QR_000000020000+a1P_001020000_1*QR_000000020001+a1P_001010001_2*QR_000000020010+a2P_001010000_2*QR_000000020011+a2P_001000001_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000020001_1*QR_000000020100+a2P_000020000_1*QR_000000020101+a2P_000010001_2*QR_000000020110+a3P_000010000_2*QR_000000020111+a3P_000000001_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+16]+=Pmtrx[0]*(P_000021001*QR_020000000000+a1P_000021000_1*QR_020000000001+P_000121001*QR_020000000010+a1P_000121000_1*QR_020000000011+P_000221001*QR_020000000020+a1P_000221000_1*QR_020000000021+a3P_000000001_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+16]+=Pmtrx[1]*(P_000021001*QR_010010000000+a1P_000021000_1*QR_010010000001+P_000121001*QR_010010000010+a1P_000121000_1*QR_010010000011+P_000221001*QR_010010000020+a1P_000221000_1*QR_010010000021+a3P_000000001_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+16]+=Pmtrx[2]*(P_000021001*QR_000020000000+a1P_000021000_1*QR_000020000001+P_000121001*QR_000020000010+a1P_000121000_1*QR_000020000011+P_000221001*QR_000020000020+a1P_000221000_1*QR_000020000021+a3P_000000001_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+16]+=Pmtrx[3]*(P_000021001*QR_010000010000+a1P_000021000_1*QR_010000010001+P_000121001*QR_010000010010+a1P_000121000_1*QR_010000010011+P_000221001*QR_010000010020+a1P_000221000_1*QR_010000010021+a3P_000000001_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+16]+=Pmtrx[4]*(P_000021001*QR_000010010000+a1P_000021000_1*QR_000010010001+P_000121001*QR_000010010010+a1P_000121000_1*QR_000010010011+P_000221001*QR_000010010020+a1P_000221000_1*QR_000010010021+a3P_000000001_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+16]+=Pmtrx[5]*(P_000021001*QR_000000020000+a1P_000021000_1*QR_000000020001+P_000121001*QR_000000020010+a1P_000121000_1*QR_000000020011+P_000221001*QR_000000020020+a1P_000221000_1*QR_000000020021+a3P_000000001_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+17]+=Pmtrx[0]*(P_000020002*QR_020000000000+a1P_000020001_2*QR_020000000001+a2P_000020000_1*QR_020000000002+a1P_000010002_2*QR_020000000010+a2P_000010001_4*QR_020000000011+a3P_000010000_2*QR_020000000012+a2P_000000002_1*QR_020000000020+a3P_000000001_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+17]+=Pmtrx[1]*(P_000020002*QR_010010000000+a1P_000020001_2*QR_010010000001+a2P_000020000_1*QR_010010000002+a1P_000010002_2*QR_010010000010+a2P_000010001_4*QR_010010000011+a3P_000010000_2*QR_010010000012+a2P_000000002_1*QR_010010000020+a3P_000000001_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+17]+=Pmtrx[2]*(P_000020002*QR_000020000000+a1P_000020001_2*QR_000020000001+a2P_000020000_1*QR_000020000002+a1P_000010002_2*QR_000020000010+a2P_000010001_4*QR_000020000011+a3P_000010000_2*QR_000020000012+a2P_000000002_1*QR_000020000020+a3P_000000001_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+17]+=Pmtrx[3]*(P_000020002*QR_010000010000+a1P_000020001_2*QR_010000010001+a2P_000020000_1*QR_010000010002+a1P_000010002_2*QR_010000010010+a2P_000010001_4*QR_010000010011+a3P_000010000_2*QR_010000010012+a2P_000000002_1*QR_010000010020+a3P_000000001_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+17]+=Pmtrx[4]*(P_000020002*QR_000010010000+a1P_000020001_2*QR_000010010001+a2P_000020000_1*QR_000010010002+a1P_000010002_2*QR_000010010010+a2P_000010001_4*QR_000010010011+a3P_000010000_2*QR_000010010012+a2P_000000002_1*QR_000010010020+a3P_000000001_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+17]+=Pmtrx[5]*(P_000020002*QR_000000020000+a1P_000020001_2*QR_000000020001+a2P_000020000_1*QR_000000020002+a1P_000010002_2*QR_000000020010+a2P_000010001_4*QR_000000020011+a3P_000010000_2*QR_000000020012+a2P_000000002_1*QR_000000020020+a3P_000000001_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+18]+=Pmtrx[0]*(P_012000010*QR_020000000000+a1P_012000000_1*QR_020000000001+P_112000010*QR_020000000100+a1P_112000000_1*QR_020000000101+P_212000010*QR_020000000200+a1P_212000000_1*QR_020000000201+a3P_000000010_1*QR_020000000300+aPin4*QR_020000000301);
			ans_temp[ans_id*36+18]+=Pmtrx[1]*(P_012000010*QR_010010000000+a1P_012000000_1*QR_010010000001+P_112000010*QR_010010000100+a1P_112000000_1*QR_010010000101+P_212000010*QR_010010000200+a1P_212000000_1*QR_010010000201+a3P_000000010_1*QR_010010000300+aPin4*QR_010010000301);
			ans_temp[ans_id*36+18]+=Pmtrx[2]*(P_012000010*QR_000020000000+a1P_012000000_1*QR_000020000001+P_112000010*QR_000020000100+a1P_112000000_1*QR_000020000101+P_212000010*QR_000020000200+a1P_212000000_1*QR_000020000201+a3P_000000010_1*QR_000020000300+aPin4*QR_000020000301);
			ans_temp[ans_id*36+18]+=Pmtrx[3]*(P_012000010*QR_010000010000+a1P_012000000_1*QR_010000010001+P_112000010*QR_010000010100+a1P_112000000_1*QR_010000010101+P_212000010*QR_010000010200+a1P_212000000_1*QR_010000010201+a3P_000000010_1*QR_010000010300+aPin4*QR_010000010301);
			ans_temp[ans_id*36+18]+=Pmtrx[4]*(P_012000010*QR_000010010000+a1P_012000000_1*QR_000010010001+P_112000010*QR_000010010100+a1P_112000000_1*QR_000010010101+P_212000010*QR_000010010200+a1P_212000000_1*QR_000010010201+a3P_000000010_1*QR_000010010300+aPin4*QR_000010010301);
			ans_temp[ans_id*36+18]+=Pmtrx[5]*(P_012000010*QR_000000020000+a1P_012000000_1*QR_000000020001+P_112000010*QR_000000020100+a1P_112000000_1*QR_000000020101+P_212000010*QR_000000020200+a1P_212000000_1*QR_000000020201+a3P_000000010_1*QR_000000020300+aPin4*QR_000000020301);
			ans_temp[ans_id*36+19]+=Pmtrx[0]*(P_011001010*QR_020000000000+a1P_011001000_1*QR_020000000001+a1P_011000010_1*QR_020000000010+a2P_011000000_1*QR_020000000011+P_111001010*QR_020000000100+a1P_111001000_1*QR_020000000101+a1P_111000010_1*QR_020000000110+a2P_111000000_1*QR_020000000111+a2P_000001010_1*QR_020000000200+a3P_000001000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+19]+=Pmtrx[1]*(P_011001010*QR_010010000000+a1P_011001000_1*QR_010010000001+a1P_011000010_1*QR_010010000010+a2P_011000000_1*QR_010010000011+P_111001010*QR_010010000100+a1P_111001000_1*QR_010010000101+a1P_111000010_1*QR_010010000110+a2P_111000000_1*QR_010010000111+a2P_000001010_1*QR_010010000200+a3P_000001000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+19]+=Pmtrx[2]*(P_011001010*QR_000020000000+a1P_011001000_1*QR_000020000001+a1P_011000010_1*QR_000020000010+a2P_011000000_1*QR_000020000011+P_111001010*QR_000020000100+a1P_111001000_1*QR_000020000101+a1P_111000010_1*QR_000020000110+a2P_111000000_1*QR_000020000111+a2P_000001010_1*QR_000020000200+a3P_000001000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+19]+=Pmtrx[3]*(P_011001010*QR_010000010000+a1P_011001000_1*QR_010000010001+a1P_011000010_1*QR_010000010010+a2P_011000000_1*QR_010000010011+P_111001010*QR_010000010100+a1P_111001000_1*QR_010000010101+a1P_111000010_1*QR_010000010110+a2P_111000000_1*QR_010000010111+a2P_000001010_1*QR_010000010200+a3P_000001000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+19]+=Pmtrx[4]*(P_011001010*QR_000010010000+a1P_011001000_1*QR_000010010001+a1P_011000010_1*QR_000010010010+a2P_011000000_1*QR_000010010011+P_111001010*QR_000010010100+a1P_111001000_1*QR_000010010101+a1P_111000010_1*QR_000010010110+a2P_111000000_1*QR_000010010111+a2P_000001010_1*QR_000010010200+a3P_000001000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+19]+=Pmtrx[5]*(P_011001010*QR_000000020000+a1P_011001000_1*QR_000000020001+a1P_011000010_1*QR_000000020010+a2P_011000000_1*QR_000000020011+P_111001010*QR_000000020100+a1P_111001000_1*QR_000000020101+a1P_111000010_1*QR_000000020110+a2P_111000000_1*QR_000000020111+a2P_000001010_1*QR_000000020200+a3P_000001000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+20]+=Pmtrx[0]*(P_010002010*QR_020000000000+a1P_010002000_1*QR_020000000001+a1P_010001010_2*QR_020000000010+a2P_010001000_2*QR_020000000011+a2P_010000010_1*QR_020000000020+a3P_010000000_1*QR_020000000021+a1P_000002010_1*QR_020000000100+a2P_000002000_1*QR_020000000101+a2P_000001010_2*QR_020000000110+a3P_000001000_2*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+20]+=Pmtrx[1]*(P_010002010*QR_010010000000+a1P_010002000_1*QR_010010000001+a1P_010001010_2*QR_010010000010+a2P_010001000_2*QR_010010000011+a2P_010000010_1*QR_010010000020+a3P_010000000_1*QR_010010000021+a1P_000002010_1*QR_010010000100+a2P_000002000_1*QR_010010000101+a2P_000001010_2*QR_010010000110+a3P_000001000_2*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+20]+=Pmtrx[2]*(P_010002010*QR_000020000000+a1P_010002000_1*QR_000020000001+a1P_010001010_2*QR_000020000010+a2P_010001000_2*QR_000020000011+a2P_010000010_1*QR_000020000020+a3P_010000000_1*QR_000020000021+a1P_000002010_1*QR_000020000100+a2P_000002000_1*QR_000020000101+a2P_000001010_2*QR_000020000110+a3P_000001000_2*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+20]+=Pmtrx[3]*(P_010002010*QR_010000010000+a1P_010002000_1*QR_010000010001+a1P_010001010_2*QR_010000010010+a2P_010001000_2*QR_010000010011+a2P_010000010_1*QR_010000010020+a3P_010000000_1*QR_010000010021+a1P_000002010_1*QR_010000010100+a2P_000002000_1*QR_010000010101+a2P_000001010_2*QR_010000010110+a3P_000001000_2*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+20]+=Pmtrx[4]*(P_010002010*QR_000010010000+a1P_010002000_1*QR_000010010001+a1P_010001010_2*QR_000010010010+a2P_010001000_2*QR_000010010011+a2P_010000010_1*QR_000010010020+a3P_010000000_1*QR_000010010021+a1P_000002010_1*QR_000010010100+a2P_000002000_1*QR_000010010101+a2P_000001010_2*QR_000010010110+a3P_000001000_2*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+20]+=Pmtrx[5]*(P_010002010*QR_000000020000+a1P_010002000_1*QR_000000020001+a1P_010001010_2*QR_000000020010+a2P_010001000_2*QR_000000020011+a2P_010000010_1*QR_000000020020+a3P_010000000_1*QR_000000020021+a1P_000002010_1*QR_000000020100+a2P_000002000_1*QR_000000020101+a2P_000001010_2*QR_000000020110+a3P_000001000_2*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+21]+=Pmtrx[0]*(P_011000011*QR_020000000000+P_011000111*QR_020000000001+a2P_011000000_1*QR_020000000002+P_111000011*QR_020000000100+P_111000111*QR_020000000101+a2P_111000000_1*QR_020000000102+a2P_000000011_1*QR_020000000200+a2P_000000111_1*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+21]+=Pmtrx[1]*(P_011000011*QR_010010000000+P_011000111*QR_010010000001+a2P_011000000_1*QR_010010000002+P_111000011*QR_010010000100+P_111000111*QR_010010000101+a2P_111000000_1*QR_010010000102+a2P_000000011_1*QR_010010000200+a2P_000000111_1*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+21]+=Pmtrx[2]*(P_011000011*QR_000020000000+P_011000111*QR_000020000001+a2P_011000000_1*QR_000020000002+P_111000011*QR_000020000100+P_111000111*QR_000020000101+a2P_111000000_1*QR_000020000102+a2P_000000011_1*QR_000020000200+a2P_000000111_1*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+21]+=Pmtrx[3]*(P_011000011*QR_010000010000+P_011000111*QR_010000010001+a2P_011000000_1*QR_010000010002+P_111000011*QR_010000010100+P_111000111*QR_010000010101+a2P_111000000_1*QR_010000010102+a2P_000000011_1*QR_010000010200+a2P_000000111_1*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+21]+=Pmtrx[4]*(P_011000011*QR_000010010000+P_011000111*QR_000010010001+a2P_011000000_1*QR_000010010002+P_111000011*QR_000010010100+P_111000111*QR_000010010101+a2P_111000000_1*QR_000010010102+a2P_000000011_1*QR_000010010200+a2P_000000111_1*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+21]+=Pmtrx[5]*(P_011000011*QR_000000020000+P_011000111*QR_000000020001+a2P_011000000_1*QR_000000020002+P_111000011*QR_000000020100+P_111000111*QR_000000020101+a2P_111000000_1*QR_000000020102+a2P_000000011_1*QR_000000020200+a2P_000000111_1*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+22]+=Pmtrx[0]*(P_010001011*QR_020000000000+P_010001111*QR_020000000001+a2P_010001000_1*QR_020000000002+a1P_010000011_1*QR_020000000010+a1P_010000111_1*QR_020000000011+a3P_010000000_1*QR_020000000012+a1P_000001011_1*QR_020000000100+a1P_000001111_1*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+22]+=Pmtrx[1]*(P_010001011*QR_010010000000+P_010001111*QR_010010000001+a2P_010001000_1*QR_010010000002+a1P_010000011_1*QR_010010000010+a1P_010000111_1*QR_010010000011+a3P_010000000_1*QR_010010000012+a1P_000001011_1*QR_010010000100+a1P_000001111_1*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+22]+=Pmtrx[2]*(P_010001011*QR_000020000000+P_010001111*QR_000020000001+a2P_010001000_1*QR_000020000002+a1P_010000011_1*QR_000020000010+a1P_010000111_1*QR_000020000011+a3P_010000000_1*QR_000020000012+a1P_000001011_1*QR_000020000100+a1P_000001111_1*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+22]+=Pmtrx[3]*(P_010001011*QR_010000010000+P_010001111*QR_010000010001+a2P_010001000_1*QR_010000010002+a1P_010000011_1*QR_010000010010+a1P_010000111_1*QR_010000010011+a3P_010000000_1*QR_010000010012+a1P_000001011_1*QR_010000010100+a1P_000001111_1*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+22]+=Pmtrx[4]*(P_010001011*QR_000010010000+P_010001111*QR_000010010001+a2P_010001000_1*QR_000010010002+a1P_010000011_1*QR_000010010010+a1P_010000111_1*QR_000010010011+a3P_010000000_1*QR_000010010012+a1P_000001011_1*QR_000010010100+a1P_000001111_1*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+22]+=Pmtrx[5]*(P_010001011*QR_000000020000+P_010001111*QR_000000020001+a2P_010001000_1*QR_000000020002+a1P_010000011_1*QR_000000020010+a1P_010000111_1*QR_000000020011+a3P_010000000_1*QR_000000020012+a1P_000001011_1*QR_000000020100+a1P_000001111_1*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+23]+=Pmtrx[0]*(P_010000012*QR_020000000000+P_010000112*QR_020000000001+P_010000212*QR_020000000002+a3P_010000000_1*QR_020000000003+a1P_000000012_1*QR_020000000100+a1P_000000112_1*QR_020000000101+a1P_000000212_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+23]+=Pmtrx[1]*(P_010000012*QR_010010000000+P_010000112*QR_010010000001+P_010000212*QR_010010000002+a3P_010000000_1*QR_010010000003+a1P_000000012_1*QR_010010000100+a1P_000000112_1*QR_010010000101+a1P_000000212_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+23]+=Pmtrx[2]*(P_010000012*QR_000020000000+P_010000112*QR_000020000001+P_010000212*QR_000020000002+a3P_010000000_1*QR_000020000003+a1P_000000012_1*QR_000020000100+a1P_000000112_1*QR_000020000101+a1P_000000212_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+23]+=Pmtrx[3]*(P_010000012*QR_010000010000+P_010000112*QR_010000010001+P_010000212*QR_010000010002+a3P_010000000_1*QR_010000010003+a1P_000000012_1*QR_010000010100+a1P_000000112_1*QR_010000010101+a1P_000000212_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+23]+=Pmtrx[4]*(P_010000012*QR_000010010000+P_010000112*QR_000010010001+P_010000212*QR_000010010002+a3P_010000000_1*QR_000010010003+a1P_000000012_1*QR_000010010100+a1P_000000112_1*QR_000010010101+a1P_000000212_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+23]+=Pmtrx[5]*(P_010000012*QR_000000020000+P_010000112*QR_000000020001+P_010000212*QR_000000020002+a3P_010000000_1*QR_000000020003+a1P_000000012_1*QR_000000020100+a1P_000000112_1*QR_000000020101+a1P_000000212_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+24]+=Pmtrx[0]*(P_002010010*QR_020000000000+a1P_002010000_1*QR_020000000001+a1P_002000010_1*QR_020000000010+a2P_002000000_1*QR_020000000011+a1P_001010010_2*QR_020000000100+a2P_001010000_2*QR_020000000101+a2P_001000010_2*QR_020000000110+a3P_001000000_2*QR_020000000111+a2P_000010010_1*QR_020000000200+a3P_000010000_1*QR_020000000201+a3P_000000010_1*QR_020000000210+aPin4*QR_020000000211);
			ans_temp[ans_id*36+24]+=Pmtrx[1]*(P_002010010*QR_010010000000+a1P_002010000_1*QR_010010000001+a1P_002000010_1*QR_010010000010+a2P_002000000_1*QR_010010000011+a1P_001010010_2*QR_010010000100+a2P_001010000_2*QR_010010000101+a2P_001000010_2*QR_010010000110+a3P_001000000_2*QR_010010000111+a2P_000010010_1*QR_010010000200+a3P_000010000_1*QR_010010000201+a3P_000000010_1*QR_010010000210+aPin4*QR_010010000211);
			ans_temp[ans_id*36+24]+=Pmtrx[2]*(P_002010010*QR_000020000000+a1P_002010000_1*QR_000020000001+a1P_002000010_1*QR_000020000010+a2P_002000000_1*QR_000020000011+a1P_001010010_2*QR_000020000100+a2P_001010000_2*QR_000020000101+a2P_001000010_2*QR_000020000110+a3P_001000000_2*QR_000020000111+a2P_000010010_1*QR_000020000200+a3P_000010000_1*QR_000020000201+a3P_000000010_1*QR_000020000210+aPin4*QR_000020000211);
			ans_temp[ans_id*36+24]+=Pmtrx[3]*(P_002010010*QR_010000010000+a1P_002010000_1*QR_010000010001+a1P_002000010_1*QR_010000010010+a2P_002000000_1*QR_010000010011+a1P_001010010_2*QR_010000010100+a2P_001010000_2*QR_010000010101+a2P_001000010_2*QR_010000010110+a3P_001000000_2*QR_010000010111+a2P_000010010_1*QR_010000010200+a3P_000010000_1*QR_010000010201+a3P_000000010_1*QR_010000010210+aPin4*QR_010000010211);
			ans_temp[ans_id*36+24]+=Pmtrx[4]*(P_002010010*QR_000010010000+a1P_002010000_1*QR_000010010001+a1P_002000010_1*QR_000010010010+a2P_002000000_1*QR_000010010011+a1P_001010010_2*QR_000010010100+a2P_001010000_2*QR_000010010101+a2P_001000010_2*QR_000010010110+a3P_001000000_2*QR_000010010111+a2P_000010010_1*QR_000010010200+a3P_000010000_1*QR_000010010201+a3P_000000010_1*QR_000010010210+aPin4*QR_000010010211);
			ans_temp[ans_id*36+24]+=Pmtrx[5]*(P_002010010*QR_000000020000+a1P_002010000_1*QR_000000020001+a1P_002000010_1*QR_000000020010+a2P_002000000_1*QR_000000020011+a1P_001010010_2*QR_000000020100+a2P_001010000_2*QR_000000020101+a2P_001000010_2*QR_000000020110+a3P_001000000_2*QR_000000020111+a2P_000010010_1*QR_000000020200+a3P_000010000_1*QR_000000020201+a3P_000000010_1*QR_000000020210+aPin4*QR_000000020211);
			ans_temp[ans_id*36+25]+=Pmtrx[0]*(P_001011010*QR_020000000000+a1P_001011000_1*QR_020000000001+P_001111010*QR_020000000010+a1P_001111000_1*QR_020000000011+a2P_001000010_1*QR_020000000020+a3P_001000000_1*QR_020000000021+a1P_000011010_1*QR_020000000100+a2P_000011000_1*QR_020000000101+a1P_000111010_1*QR_020000000110+a2P_000111000_1*QR_020000000111+a3P_000000010_1*QR_020000000120+aPin4*QR_020000000121);
			ans_temp[ans_id*36+25]+=Pmtrx[1]*(P_001011010*QR_010010000000+a1P_001011000_1*QR_010010000001+P_001111010*QR_010010000010+a1P_001111000_1*QR_010010000011+a2P_001000010_1*QR_010010000020+a3P_001000000_1*QR_010010000021+a1P_000011010_1*QR_010010000100+a2P_000011000_1*QR_010010000101+a1P_000111010_1*QR_010010000110+a2P_000111000_1*QR_010010000111+a3P_000000010_1*QR_010010000120+aPin4*QR_010010000121);
			ans_temp[ans_id*36+25]+=Pmtrx[2]*(P_001011010*QR_000020000000+a1P_001011000_1*QR_000020000001+P_001111010*QR_000020000010+a1P_001111000_1*QR_000020000011+a2P_001000010_1*QR_000020000020+a3P_001000000_1*QR_000020000021+a1P_000011010_1*QR_000020000100+a2P_000011000_1*QR_000020000101+a1P_000111010_1*QR_000020000110+a2P_000111000_1*QR_000020000111+a3P_000000010_1*QR_000020000120+aPin4*QR_000020000121);
			ans_temp[ans_id*36+25]+=Pmtrx[3]*(P_001011010*QR_010000010000+a1P_001011000_1*QR_010000010001+P_001111010*QR_010000010010+a1P_001111000_1*QR_010000010011+a2P_001000010_1*QR_010000010020+a3P_001000000_1*QR_010000010021+a1P_000011010_1*QR_010000010100+a2P_000011000_1*QR_010000010101+a1P_000111010_1*QR_010000010110+a2P_000111000_1*QR_010000010111+a3P_000000010_1*QR_010000010120+aPin4*QR_010000010121);
			ans_temp[ans_id*36+25]+=Pmtrx[4]*(P_001011010*QR_000010010000+a1P_001011000_1*QR_000010010001+P_001111010*QR_000010010010+a1P_001111000_1*QR_000010010011+a2P_001000010_1*QR_000010010020+a3P_001000000_1*QR_000010010021+a1P_000011010_1*QR_000010010100+a2P_000011000_1*QR_000010010101+a1P_000111010_1*QR_000010010110+a2P_000111000_1*QR_000010010111+a3P_000000010_1*QR_000010010120+aPin4*QR_000010010121);
			ans_temp[ans_id*36+25]+=Pmtrx[5]*(P_001011010*QR_000000020000+a1P_001011000_1*QR_000000020001+P_001111010*QR_000000020010+a1P_001111000_1*QR_000000020011+a2P_001000010_1*QR_000000020020+a3P_001000000_1*QR_000000020021+a1P_000011010_1*QR_000000020100+a2P_000011000_1*QR_000000020101+a1P_000111010_1*QR_000000020110+a2P_000111000_1*QR_000000020111+a3P_000000010_1*QR_000000020120+aPin4*QR_000000020121);
			ans_temp[ans_id*36+26]+=Pmtrx[0]*(P_000012010*QR_020000000000+a1P_000012000_1*QR_020000000001+P_000112010*QR_020000000010+a1P_000112000_1*QR_020000000011+P_000212010*QR_020000000020+a1P_000212000_1*QR_020000000021+a3P_000000010_1*QR_020000000030+aPin4*QR_020000000031);
			ans_temp[ans_id*36+26]+=Pmtrx[1]*(P_000012010*QR_010010000000+a1P_000012000_1*QR_010010000001+P_000112010*QR_010010000010+a1P_000112000_1*QR_010010000011+P_000212010*QR_010010000020+a1P_000212000_1*QR_010010000021+a3P_000000010_1*QR_010010000030+aPin4*QR_010010000031);
			ans_temp[ans_id*36+26]+=Pmtrx[2]*(P_000012010*QR_000020000000+a1P_000012000_1*QR_000020000001+P_000112010*QR_000020000010+a1P_000112000_1*QR_000020000011+P_000212010*QR_000020000020+a1P_000212000_1*QR_000020000021+a3P_000000010_1*QR_000020000030+aPin4*QR_000020000031);
			ans_temp[ans_id*36+26]+=Pmtrx[3]*(P_000012010*QR_010000010000+a1P_000012000_1*QR_010000010001+P_000112010*QR_010000010010+a1P_000112000_1*QR_010000010011+P_000212010*QR_010000010020+a1P_000212000_1*QR_010000010021+a3P_000000010_1*QR_010000010030+aPin4*QR_010000010031);
			ans_temp[ans_id*36+26]+=Pmtrx[4]*(P_000012010*QR_000010010000+a1P_000012000_1*QR_000010010001+P_000112010*QR_000010010010+a1P_000112000_1*QR_000010010011+P_000212010*QR_000010010020+a1P_000212000_1*QR_000010010021+a3P_000000010_1*QR_000010010030+aPin4*QR_000010010031);
			ans_temp[ans_id*36+26]+=Pmtrx[5]*(P_000012010*QR_000000020000+a1P_000012000_1*QR_000000020001+P_000112010*QR_000000020010+a1P_000112000_1*QR_000000020011+P_000212010*QR_000000020020+a1P_000212000_1*QR_000000020021+a3P_000000010_1*QR_000000020030+aPin4*QR_000000020031);
			ans_temp[ans_id*36+27]+=Pmtrx[0]*(P_001010011*QR_020000000000+P_001010111*QR_020000000001+a2P_001010000_1*QR_020000000002+a1P_001000011_1*QR_020000000010+a1P_001000111_1*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000010011_1*QR_020000000100+a1P_000010111_1*QR_020000000101+a3P_000010000_1*QR_020000000102+a2P_000000011_1*QR_020000000110+a2P_000000111_1*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+27]+=Pmtrx[1]*(P_001010011*QR_010010000000+P_001010111*QR_010010000001+a2P_001010000_1*QR_010010000002+a1P_001000011_1*QR_010010000010+a1P_001000111_1*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000010011_1*QR_010010000100+a1P_000010111_1*QR_010010000101+a3P_000010000_1*QR_010010000102+a2P_000000011_1*QR_010010000110+a2P_000000111_1*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+27]+=Pmtrx[2]*(P_001010011*QR_000020000000+P_001010111*QR_000020000001+a2P_001010000_1*QR_000020000002+a1P_001000011_1*QR_000020000010+a1P_001000111_1*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000010011_1*QR_000020000100+a1P_000010111_1*QR_000020000101+a3P_000010000_1*QR_000020000102+a2P_000000011_1*QR_000020000110+a2P_000000111_1*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+27]+=Pmtrx[3]*(P_001010011*QR_010000010000+P_001010111*QR_010000010001+a2P_001010000_1*QR_010000010002+a1P_001000011_1*QR_010000010010+a1P_001000111_1*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000010011_1*QR_010000010100+a1P_000010111_1*QR_010000010101+a3P_000010000_1*QR_010000010102+a2P_000000011_1*QR_010000010110+a2P_000000111_1*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+27]+=Pmtrx[4]*(P_001010011*QR_000010010000+P_001010111*QR_000010010001+a2P_001010000_1*QR_000010010002+a1P_001000011_1*QR_000010010010+a1P_001000111_1*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000010011_1*QR_000010010100+a1P_000010111_1*QR_000010010101+a3P_000010000_1*QR_000010010102+a2P_000000011_1*QR_000010010110+a2P_000000111_1*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+27]+=Pmtrx[5]*(P_001010011*QR_000000020000+P_001010111*QR_000000020001+a2P_001010000_1*QR_000000020002+a1P_001000011_1*QR_000000020010+a1P_001000111_1*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000010011_1*QR_000000020100+a1P_000010111_1*QR_000000020101+a3P_000010000_1*QR_000000020102+a2P_000000011_1*QR_000000020110+a2P_000000111_1*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+28]+=Pmtrx[0]*(P_000011011*QR_020000000000+P_000011111*QR_020000000001+a2P_000011000_1*QR_020000000002+P_000111011*QR_020000000010+P_000111111*QR_020000000011+a2P_000111000_1*QR_020000000012+a2P_000000011_1*QR_020000000020+a2P_000000111_1*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+28]+=Pmtrx[1]*(P_000011011*QR_010010000000+P_000011111*QR_010010000001+a2P_000011000_1*QR_010010000002+P_000111011*QR_010010000010+P_000111111*QR_010010000011+a2P_000111000_1*QR_010010000012+a2P_000000011_1*QR_010010000020+a2P_000000111_1*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+28]+=Pmtrx[2]*(P_000011011*QR_000020000000+P_000011111*QR_000020000001+a2P_000011000_1*QR_000020000002+P_000111011*QR_000020000010+P_000111111*QR_000020000011+a2P_000111000_1*QR_000020000012+a2P_000000011_1*QR_000020000020+a2P_000000111_1*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+28]+=Pmtrx[3]*(P_000011011*QR_010000010000+P_000011111*QR_010000010001+a2P_000011000_1*QR_010000010002+P_000111011*QR_010000010010+P_000111111*QR_010000010011+a2P_000111000_1*QR_010000010012+a2P_000000011_1*QR_010000010020+a2P_000000111_1*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+28]+=Pmtrx[4]*(P_000011011*QR_000010010000+P_000011111*QR_000010010001+a2P_000011000_1*QR_000010010002+P_000111011*QR_000010010010+P_000111111*QR_000010010011+a2P_000111000_1*QR_000010010012+a2P_000000011_1*QR_000010010020+a2P_000000111_1*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+28]+=Pmtrx[5]*(P_000011011*QR_000000020000+P_000011111*QR_000000020001+a2P_000011000_1*QR_000000020002+P_000111011*QR_000000020010+P_000111111*QR_000000020011+a2P_000111000_1*QR_000000020012+a2P_000000011_1*QR_000000020020+a2P_000000111_1*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+29]+=Pmtrx[0]*(P_000010012*QR_020000000000+P_000010112*QR_020000000001+P_000010212*QR_020000000002+a3P_000010000_1*QR_020000000003+a1P_000000012_1*QR_020000000010+a1P_000000112_1*QR_020000000011+a1P_000000212_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+29]+=Pmtrx[1]*(P_000010012*QR_010010000000+P_000010112*QR_010010000001+P_000010212*QR_010010000002+a3P_000010000_1*QR_010010000003+a1P_000000012_1*QR_010010000010+a1P_000000112_1*QR_010010000011+a1P_000000212_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+29]+=Pmtrx[2]*(P_000010012*QR_000020000000+P_000010112*QR_000020000001+P_000010212*QR_000020000002+a3P_000010000_1*QR_000020000003+a1P_000000012_1*QR_000020000010+a1P_000000112_1*QR_000020000011+a1P_000000212_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+29]+=Pmtrx[3]*(P_000010012*QR_010000010000+P_000010112*QR_010000010001+P_000010212*QR_010000010002+a3P_000010000_1*QR_010000010003+a1P_000000012_1*QR_010000010010+a1P_000000112_1*QR_010000010011+a1P_000000212_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+29]+=Pmtrx[4]*(P_000010012*QR_000010010000+P_000010112*QR_000010010001+P_000010212*QR_000010010002+a3P_000010000_1*QR_000010010003+a1P_000000012_1*QR_000010010010+a1P_000000112_1*QR_000010010011+a1P_000000212_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+29]+=Pmtrx[5]*(P_000010012*QR_000000020000+P_000010112*QR_000000020001+P_000010212*QR_000000020002+a3P_000010000_1*QR_000000020003+a1P_000000012_1*QR_000000020010+a1P_000000112_1*QR_000000020011+a1P_000000212_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+30]+=Pmtrx[0]*(P_002000020*QR_020000000000+a1P_002000010_2*QR_020000000001+a2P_002000000_1*QR_020000000002+a1P_001000020_2*QR_020000000100+a2P_001000010_4*QR_020000000101+a3P_001000000_2*QR_020000000102+a2P_000000020_1*QR_020000000200+a3P_000000010_2*QR_020000000201+aPin4*QR_020000000202);
			ans_temp[ans_id*36+30]+=Pmtrx[1]*(P_002000020*QR_010010000000+a1P_002000010_2*QR_010010000001+a2P_002000000_1*QR_010010000002+a1P_001000020_2*QR_010010000100+a2P_001000010_4*QR_010010000101+a3P_001000000_2*QR_010010000102+a2P_000000020_1*QR_010010000200+a3P_000000010_2*QR_010010000201+aPin4*QR_010010000202);
			ans_temp[ans_id*36+30]+=Pmtrx[2]*(P_002000020*QR_000020000000+a1P_002000010_2*QR_000020000001+a2P_002000000_1*QR_000020000002+a1P_001000020_2*QR_000020000100+a2P_001000010_4*QR_000020000101+a3P_001000000_2*QR_000020000102+a2P_000000020_1*QR_000020000200+a3P_000000010_2*QR_000020000201+aPin4*QR_000020000202);
			ans_temp[ans_id*36+30]+=Pmtrx[3]*(P_002000020*QR_010000010000+a1P_002000010_2*QR_010000010001+a2P_002000000_1*QR_010000010002+a1P_001000020_2*QR_010000010100+a2P_001000010_4*QR_010000010101+a3P_001000000_2*QR_010000010102+a2P_000000020_1*QR_010000010200+a3P_000000010_2*QR_010000010201+aPin4*QR_010000010202);
			ans_temp[ans_id*36+30]+=Pmtrx[4]*(P_002000020*QR_000010010000+a1P_002000010_2*QR_000010010001+a2P_002000000_1*QR_000010010002+a1P_001000020_2*QR_000010010100+a2P_001000010_4*QR_000010010101+a3P_001000000_2*QR_000010010102+a2P_000000020_1*QR_000010010200+a3P_000000010_2*QR_000010010201+aPin4*QR_000010010202);
			ans_temp[ans_id*36+30]+=Pmtrx[5]*(P_002000020*QR_000000020000+a1P_002000010_2*QR_000000020001+a2P_002000000_1*QR_000000020002+a1P_001000020_2*QR_000000020100+a2P_001000010_4*QR_000000020101+a3P_001000000_2*QR_000000020102+a2P_000000020_1*QR_000000020200+a3P_000000010_2*QR_000000020201+aPin4*QR_000000020202);
			ans_temp[ans_id*36+31]+=Pmtrx[0]*(P_001001020*QR_020000000000+a1P_001001010_2*QR_020000000001+a2P_001001000_1*QR_020000000002+a1P_001000020_1*QR_020000000010+a2P_001000010_2*QR_020000000011+a3P_001000000_1*QR_020000000012+a1P_000001020_1*QR_020000000100+a2P_000001010_2*QR_020000000101+a3P_000001000_1*QR_020000000102+a2P_000000020_1*QR_020000000110+a3P_000000010_2*QR_020000000111+aPin4*QR_020000000112);
			ans_temp[ans_id*36+31]+=Pmtrx[1]*(P_001001020*QR_010010000000+a1P_001001010_2*QR_010010000001+a2P_001001000_1*QR_010010000002+a1P_001000020_1*QR_010010000010+a2P_001000010_2*QR_010010000011+a3P_001000000_1*QR_010010000012+a1P_000001020_1*QR_010010000100+a2P_000001010_2*QR_010010000101+a3P_000001000_1*QR_010010000102+a2P_000000020_1*QR_010010000110+a3P_000000010_2*QR_010010000111+aPin4*QR_010010000112);
			ans_temp[ans_id*36+31]+=Pmtrx[2]*(P_001001020*QR_000020000000+a1P_001001010_2*QR_000020000001+a2P_001001000_1*QR_000020000002+a1P_001000020_1*QR_000020000010+a2P_001000010_2*QR_000020000011+a3P_001000000_1*QR_000020000012+a1P_000001020_1*QR_000020000100+a2P_000001010_2*QR_000020000101+a3P_000001000_1*QR_000020000102+a2P_000000020_1*QR_000020000110+a3P_000000010_2*QR_000020000111+aPin4*QR_000020000112);
			ans_temp[ans_id*36+31]+=Pmtrx[3]*(P_001001020*QR_010000010000+a1P_001001010_2*QR_010000010001+a2P_001001000_1*QR_010000010002+a1P_001000020_1*QR_010000010010+a2P_001000010_2*QR_010000010011+a3P_001000000_1*QR_010000010012+a1P_000001020_1*QR_010000010100+a2P_000001010_2*QR_010000010101+a3P_000001000_1*QR_010000010102+a2P_000000020_1*QR_010000010110+a3P_000000010_2*QR_010000010111+aPin4*QR_010000010112);
			ans_temp[ans_id*36+31]+=Pmtrx[4]*(P_001001020*QR_000010010000+a1P_001001010_2*QR_000010010001+a2P_001001000_1*QR_000010010002+a1P_001000020_1*QR_000010010010+a2P_001000010_2*QR_000010010011+a3P_001000000_1*QR_000010010012+a1P_000001020_1*QR_000010010100+a2P_000001010_2*QR_000010010101+a3P_000001000_1*QR_000010010102+a2P_000000020_1*QR_000010010110+a3P_000000010_2*QR_000010010111+aPin4*QR_000010010112);
			ans_temp[ans_id*36+31]+=Pmtrx[5]*(P_001001020*QR_000000020000+a1P_001001010_2*QR_000000020001+a2P_001001000_1*QR_000000020002+a1P_001000020_1*QR_000000020010+a2P_001000010_2*QR_000000020011+a3P_001000000_1*QR_000000020012+a1P_000001020_1*QR_000000020100+a2P_000001010_2*QR_000000020101+a3P_000001000_1*QR_000000020102+a2P_000000020_1*QR_000000020110+a3P_000000010_2*QR_000000020111+aPin4*QR_000000020112);
			ans_temp[ans_id*36+32]+=Pmtrx[0]*(P_000002020*QR_020000000000+a1P_000002010_2*QR_020000000001+a2P_000002000_1*QR_020000000002+a1P_000001020_2*QR_020000000010+a2P_000001010_4*QR_020000000011+a3P_000001000_2*QR_020000000012+a2P_000000020_1*QR_020000000020+a3P_000000010_2*QR_020000000021+aPin4*QR_020000000022);
			ans_temp[ans_id*36+32]+=Pmtrx[1]*(P_000002020*QR_010010000000+a1P_000002010_2*QR_010010000001+a2P_000002000_1*QR_010010000002+a1P_000001020_2*QR_010010000010+a2P_000001010_4*QR_010010000011+a3P_000001000_2*QR_010010000012+a2P_000000020_1*QR_010010000020+a3P_000000010_2*QR_010010000021+aPin4*QR_010010000022);
			ans_temp[ans_id*36+32]+=Pmtrx[2]*(P_000002020*QR_000020000000+a1P_000002010_2*QR_000020000001+a2P_000002000_1*QR_000020000002+a1P_000001020_2*QR_000020000010+a2P_000001010_4*QR_000020000011+a3P_000001000_2*QR_000020000012+a2P_000000020_1*QR_000020000020+a3P_000000010_2*QR_000020000021+aPin4*QR_000020000022);
			ans_temp[ans_id*36+32]+=Pmtrx[3]*(P_000002020*QR_010000010000+a1P_000002010_2*QR_010000010001+a2P_000002000_1*QR_010000010002+a1P_000001020_2*QR_010000010010+a2P_000001010_4*QR_010000010011+a3P_000001000_2*QR_010000010012+a2P_000000020_1*QR_010000010020+a3P_000000010_2*QR_010000010021+aPin4*QR_010000010022);
			ans_temp[ans_id*36+32]+=Pmtrx[4]*(P_000002020*QR_000010010000+a1P_000002010_2*QR_000010010001+a2P_000002000_1*QR_000010010002+a1P_000001020_2*QR_000010010010+a2P_000001010_4*QR_000010010011+a3P_000001000_2*QR_000010010012+a2P_000000020_1*QR_000010010020+a3P_000000010_2*QR_000010010021+aPin4*QR_000010010022);
			ans_temp[ans_id*36+32]+=Pmtrx[5]*(P_000002020*QR_000000020000+a1P_000002010_2*QR_000000020001+a2P_000002000_1*QR_000000020002+a1P_000001020_2*QR_000000020010+a2P_000001010_4*QR_000000020011+a3P_000001000_2*QR_000000020012+a2P_000000020_1*QR_000000020020+a3P_000000010_2*QR_000000020021+aPin4*QR_000000020022);
			ans_temp[ans_id*36+33]+=Pmtrx[0]*(P_001000021*QR_020000000000+P_001000121*QR_020000000001+P_001000221*QR_020000000002+a3P_001000000_1*QR_020000000003+a1P_000000021_1*QR_020000000100+a1P_000000121_1*QR_020000000101+a1P_000000221_1*QR_020000000102+aPin4*QR_020000000103);
			ans_temp[ans_id*36+33]+=Pmtrx[1]*(P_001000021*QR_010010000000+P_001000121*QR_010010000001+P_001000221*QR_010010000002+a3P_001000000_1*QR_010010000003+a1P_000000021_1*QR_010010000100+a1P_000000121_1*QR_010010000101+a1P_000000221_1*QR_010010000102+aPin4*QR_010010000103);
			ans_temp[ans_id*36+33]+=Pmtrx[2]*(P_001000021*QR_000020000000+P_001000121*QR_000020000001+P_001000221*QR_000020000002+a3P_001000000_1*QR_000020000003+a1P_000000021_1*QR_000020000100+a1P_000000121_1*QR_000020000101+a1P_000000221_1*QR_000020000102+aPin4*QR_000020000103);
			ans_temp[ans_id*36+33]+=Pmtrx[3]*(P_001000021*QR_010000010000+P_001000121*QR_010000010001+P_001000221*QR_010000010002+a3P_001000000_1*QR_010000010003+a1P_000000021_1*QR_010000010100+a1P_000000121_1*QR_010000010101+a1P_000000221_1*QR_010000010102+aPin4*QR_010000010103);
			ans_temp[ans_id*36+33]+=Pmtrx[4]*(P_001000021*QR_000010010000+P_001000121*QR_000010010001+P_001000221*QR_000010010002+a3P_001000000_1*QR_000010010003+a1P_000000021_1*QR_000010010100+a1P_000000121_1*QR_000010010101+a1P_000000221_1*QR_000010010102+aPin4*QR_000010010103);
			ans_temp[ans_id*36+33]+=Pmtrx[5]*(P_001000021*QR_000000020000+P_001000121*QR_000000020001+P_001000221*QR_000000020002+a3P_001000000_1*QR_000000020003+a1P_000000021_1*QR_000000020100+a1P_000000121_1*QR_000000020101+a1P_000000221_1*QR_000000020102+aPin4*QR_000000020103);
			ans_temp[ans_id*36+34]+=Pmtrx[0]*(P_000001021*QR_020000000000+P_000001121*QR_020000000001+P_000001221*QR_020000000002+a3P_000001000_1*QR_020000000003+a1P_000000021_1*QR_020000000010+a1P_000000121_1*QR_020000000011+a1P_000000221_1*QR_020000000012+aPin4*QR_020000000013);
			ans_temp[ans_id*36+34]+=Pmtrx[1]*(P_000001021*QR_010010000000+P_000001121*QR_010010000001+P_000001221*QR_010010000002+a3P_000001000_1*QR_010010000003+a1P_000000021_1*QR_010010000010+a1P_000000121_1*QR_010010000011+a1P_000000221_1*QR_010010000012+aPin4*QR_010010000013);
			ans_temp[ans_id*36+34]+=Pmtrx[2]*(P_000001021*QR_000020000000+P_000001121*QR_000020000001+P_000001221*QR_000020000002+a3P_000001000_1*QR_000020000003+a1P_000000021_1*QR_000020000010+a1P_000000121_1*QR_000020000011+a1P_000000221_1*QR_000020000012+aPin4*QR_000020000013);
			ans_temp[ans_id*36+34]+=Pmtrx[3]*(P_000001021*QR_010000010000+P_000001121*QR_010000010001+P_000001221*QR_010000010002+a3P_000001000_1*QR_010000010003+a1P_000000021_1*QR_010000010010+a1P_000000121_1*QR_010000010011+a1P_000000221_1*QR_010000010012+aPin4*QR_010000010013);
			ans_temp[ans_id*36+34]+=Pmtrx[4]*(P_000001021*QR_000010010000+P_000001121*QR_000010010001+P_000001221*QR_000010010002+a3P_000001000_1*QR_000010010003+a1P_000000021_1*QR_000010010010+a1P_000000121_1*QR_000010010011+a1P_000000221_1*QR_000010010012+aPin4*QR_000010010013);
			ans_temp[ans_id*36+34]+=Pmtrx[5]*(P_000001021*QR_000000020000+P_000001121*QR_000000020001+P_000001221*QR_000000020002+a3P_000001000_1*QR_000000020003+a1P_000000021_1*QR_000000020010+a1P_000000121_1*QR_000000020011+a1P_000000221_1*QR_000000020012+aPin4*QR_000000020013);
			ans_temp[ans_id*36+35]+=Pmtrx[0]*(P_000000022*QR_020000000000+P_000000122*QR_020000000001+P_000000222*QR_020000000002+a2P_000000111_2*QR_020000000003+aPin4*QR_020000000004);
			ans_temp[ans_id*36+35]+=Pmtrx[1]*(P_000000022*QR_010010000000+P_000000122*QR_010010000001+P_000000222*QR_010010000002+a2P_000000111_2*QR_010010000003+aPin4*QR_010010000004);
			ans_temp[ans_id*36+35]+=Pmtrx[2]*(P_000000022*QR_000020000000+P_000000122*QR_000020000001+P_000000222*QR_000020000002+a2P_000000111_2*QR_000020000003+aPin4*QR_000020000004);
			ans_temp[ans_id*36+35]+=Pmtrx[3]*(P_000000022*QR_010000010000+P_000000122*QR_010000010001+P_000000222*QR_010000010002+a2P_000000111_2*QR_010000010003+aPin4*QR_010000010004);
			ans_temp[ans_id*36+35]+=Pmtrx[4]*(P_000000022*QR_000010010000+P_000000122*QR_000010010001+P_000000222*QR_000010010002+a2P_000000111_2*QR_000010010003+aPin4*QR_000010010004);
			ans_temp[ans_id*36+35]+=Pmtrx[5]*(P_000000022*QR_000000020000+P_000000122*QR_000000020001+P_000000222*QR_000000020002+a2P_000000111_2*QR_000000020003+aPin4*QR_000000020004);
		}
        __syncthreads();
        int num_thread=tdis/2;
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
                ans[i_contrc_bra*36+ians]=ans_temp[(tId_x)*36+ians];
            }
        }
	}
}
