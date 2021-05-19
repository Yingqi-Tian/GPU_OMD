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
texture<int2,1,cudaReadModeElementType> tex_Pmtrx;

void TSMJ_texture_binding_ss(double * Q_d,double * QC_d,double * QD_d,\
        double * alphaQ_d,double * pq_d,float * K2_q_d,double * Pmtrx_d,\
        unsigned int contrc_ket_start_pr,unsigned int primit_ket_len,unsigned int contrc_Pmtrx_start_pr){
    cudaBindTexture(0, tex_Q, Q_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len)*3);
    cudaBindTexture(0, tex_Eta, alphaQ_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_pq, pq_d, sizeof(double)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_K2_q, K2_q_d, sizeof(float)*(contrc_ket_start_pr+primit_ket_len));
    cudaBindTexture(0, tex_Pmtrx, Pmtrx_d, sizeof(double)*(contrc_Pmtrx_start_pr+primit_ket_len)*1);
}

void TSMJ_texture_unbind_ss(){
    cudaUnbindTexture(tex_Q);
    cudaUnbindTexture(tex_Eta);
    cudaUnbindTexture(tex_pq);
    cudaUnbindTexture(tex_K2_q);
    cudaUnbindTexture(tex_Pmtrx);

}
__global__ void TSMJ_ssss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double T=alphaT*((PX-QX)*(PX-QX)+(PY-QY)*(PY-QY)+(PZ-QZ)*(PZ-QZ));
                double R_000[1];
                Ft_taylor(0,T,R_000);
                R_000[0]*=lmd;
			ans_temp[ans_id*1+0]+=R_000[0];
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
__global__ void TSMJ_ssss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double T=alphaT*((PX-QX)*(PX-QX)+(PY-QY)*(PY-QY)+(PZ-QZ)*(PZ-QZ));
                double R_000[1];
                Ft_fs_0(0,T,R_000);
                R_000[0]*=lmd;
			ans_temp[ans_id*1+0]+=R_000[0];
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
__global__ void TSMJ_ssss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double T=alphaT*((PX-QX)*(PX-QX)+(PY-QY)*(PY-QY)+(PZ-QZ)*(PZ-QZ));
                double R_000[1];
                Ft_fs_0(0,T,R_000);
                R_000[0]*=lmd;
			ans_temp[ans_id*1+0]+=R_000[0];
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
__global__ void TSMJ_ssss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
			double QR_000000000000=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double T=alphaT*((PX-QX)*(PX-QX)+(PY-QY)*(PY-QY)+(PZ-QZ)*(PZ-QZ));
                double R_000[1];
                Ft_fs_0(0,T,R_000);
                R_000[0]*=lmd;
			QR_000000000000+=R_000[0];
			}
			ans_temp[ans_id*1+0]+=QR_000000000000;
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
__global__ void TSMJ_psss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[2];
                Ft_taylor(1,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
	R_000[1]*=aPin1;
	double R_100[1];
	double R_010[1];
	double R_001[1];
	for(int i=0;i<1;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_001[i]=TZ*R_000[i+1];
	}
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=PD_010[0];
			P_000010000=PD_010[1];
			P_000000010=PD_010[2];
			ans_temp[ans_id*3+0]+=P_010000000*R_000[0]+R_100[0];
			ans_temp[ans_id*3+1]+=P_000010000*R_000[0]+R_010[0];
			ans_temp[ans_id*3+2]+=P_000000010*R_000[0]+R_001[0];
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
__global__ void TSMJ_psss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[2];
                Ft_fs_1(1,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
	R_000[1]*=aPin1;
	double R_100[1];
	double R_010[1];
	double R_001[1];
	for(int i=0;i<1;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_001[i]=TZ*R_000[i+1];
	}
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=PD_010[0];
			P_000010000=PD_010[1];
			P_000000010=PD_010[2];
			ans_temp[ans_id*3+0]+=P_010000000*R_000[0]+R_100[0];
			ans_temp[ans_id*3+1]+=P_000010000*R_000[0]+R_010[0];
			ans_temp[ans_id*3+2]+=P_000000010*R_000[0]+R_001[0];
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
__global__ void TSMJ_psss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[2];
                Ft_fs_1(1,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
	R_000[1]*=aPin1;
	double R_100[1];
	double R_010[1];
	double R_001[1];
	for(int i=0;i<1;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_001[i]=TZ*R_000[i+1];
	}
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=PD_010[0];
			P_000010000=PD_010[1];
			P_000000010=PD_010[2];
			ans_temp[ans_id*3+0]+=P_010000000*R_000[0]+R_100[0];
			ans_temp[ans_id*3+1]+=P_000010000*R_000[0]+R_010[0];
			ans_temp[ans_id*3+2]+=P_000000010*R_000[0]+R_001[0];
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
__global__ void TSMJ_psss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double QR_000000000000=0;
			double QR_000000000001=0;
			double QR_000000000010=0;
			double QR_000000000100=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[2];
                Ft_fs_1(1,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
	R_000[1]*=aPin1;
	double R_100[1];
	double R_010[1];
	double R_001[1];
	for(int i=0;i<1;i++){
		R_100[i]=TX*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_010[i]=TY*R_000[i+1];
	}
	for(int i=0;i<1;i++){
		R_001[i]=TZ*R_000[i+1];
	}
			QR_000000000000+=R_000[0];
			QR_000000000001+=aPin1*(R_001[0]);
			QR_000000000010+=aPin1*(R_010[0]);
			QR_000000000100+=aPin1*(R_100[0]);
			}
			double P_010000000;
			double P_000010000;
			double P_000000010;
			P_010000000=PD_010[0];
			P_000010000=PD_010[1];
			P_000000010=PD_010[2];
			ans_temp[ans_id*3+0]+=P_010000000*QR_000000000000+QR_000000000100;
			ans_temp[ans_id*3+1]+=P_000010000*QR_000000000000+QR_000000000010;
			ans_temp[ans_id*3+2]+=P_000000010*QR_000000000000+QR_000000000001;
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
__global__ void TSMJ_ppss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_taylor(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
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
			P_011000000=PD_011[0];
			P_111000000=PD_111[0];
			P_010001000=PD_010[0]*PD_001[1];
			P_010000001=PD_010[0]*PD_001[2];
			P_001010000=PD_001[0]*PD_010[1];
			P_000011000=PD_011[1];
			P_000111000=PD_111[1];
			P_000010001=PD_010[1]*PD_001[2];
			P_001000010=PD_001[0]*PD_010[2];
			P_000001010=PD_001[1]*PD_010[2];
			P_000000011=PD_011[2];
			P_000000111=PD_111[2];
			a1P_010000000_1=PD_010[0];
			a1P_000001000_1=PD_001[1];
			a1P_000000001_1=PD_001[2];
			a1P_001000000_1=PD_001[0];
			a1P_000010000_1=PD_010[1];
			a1P_000000010_1=PD_010[2];
			ans_temp[ans_id*9+0]+=P_011000000*R_000[0]+P_111000000*R_100[0]+R_200[0];
			ans_temp[ans_id*9+1]+=P_010001000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000001000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+2]+=P_010000001*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000001_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+3]+=P_001010000*R_000[0]+a1P_001000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+4]+=P_000011000*R_000[0]+P_000111000*R_010[0]+R_020[0];
			ans_temp[ans_id*9+5]+=P_000010001*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000001_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+6]+=P_001000010*R_000[0]+a1P_001000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+7]+=P_000001010*R_000[0]+a1P_000001000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+8]+=P_000000011*R_000[0]+P_000000111*R_001[0]+R_002[0];
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
__global__ void TSMJ_ppss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
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
			P_011000000=PD_011[0];
			P_111000000=PD_111[0];
			P_010001000=PD_010[0]*PD_001[1];
			P_010000001=PD_010[0]*PD_001[2];
			P_001010000=PD_001[0]*PD_010[1];
			P_000011000=PD_011[1];
			P_000111000=PD_111[1];
			P_000010001=PD_010[1]*PD_001[2];
			P_001000010=PD_001[0]*PD_010[2];
			P_000001010=PD_001[1]*PD_010[2];
			P_000000011=PD_011[2];
			P_000000111=PD_111[2];
			a1P_010000000_1=PD_010[0];
			a1P_000001000_1=PD_001[1];
			a1P_000000001_1=PD_001[2];
			a1P_001000000_1=PD_001[0];
			a1P_000010000_1=PD_010[1];
			a1P_000000010_1=PD_010[2];
			ans_temp[ans_id*9+0]+=P_011000000*R_000[0]+P_111000000*R_100[0]+R_200[0];
			ans_temp[ans_id*9+1]+=P_010001000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000001000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+2]+=P_010000001*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000001_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+3]+=P_001010000*R_000[0]+a1P_001000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+4]+=P_000011000*R_000[0]+P_000111000*R_010[0]+R_020[0];
			ans_temp[ans_id*9+5]+=P_000010001*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000001_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+6]+=P_001000010*R_000[0]+a1P_001000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+7]+=P_000001010*R_000[0]+a1P_000001000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+8]+=P_000000011*R_000[0]+P_000000111*R_001[0]+R_002[0];
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
__global__ void TSMJ_ppss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
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
			P_011000000=PD_011[0];
			P_111000000=PD_111[0];
			P_010001000=PD_010[0]*PD_001[1];
			P_010000001=PD_010[0]*PD_001[2];
			P_001010000=PD_001[0]*PD_010[1];
			P_000011000=PD_011[1];
			P_000111000=PD_111[1];
			P_000010001=PD_010[1]*PD_001[2];
			P_001000010=PD_001[0]*PD_010[2];
			P_000001010=PD_001[1]*PD_010[2];
			P_000000011=PD_011[2];
			P_000000111=PD_111[2];
			a1P_010000000_1=PD_010[0];
			a1P_000001000_1=PD_001[1];
			a1P_000000001_1=PD_001[2];
			a1P_001000000_1=PD_001[0];
			a1P_000010000_1=PD_010[1];
			a1P_000000010_1=PD_010[2];
			ans_temp[ans_id*9+0]+=P_011000000*R_000[0]+P_111000000*R_100[0]+R_200[0];
			ans_temp[ans_id*9+1]+=P_010001000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000001000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+2]+=P_010000001*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000001_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+3]+=P_001010000*R_000[0]+a1P_001000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*9+4]+=P_000011000*R_000[0]+P_000111000*R_010[0]+R_020[0];
			ans_temp[ans_id*9+5]+=P_000010001*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000001_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+6]+=P_001000010*R_000[0]+a1P_001000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*9+7]+=P_000001010*R_000[0]+a1P_000001000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*9+8]+=P_000000011*R_000[0]+P_000000111*R_001[0]+R_002[0];
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
__global__ void TSMJ_ppss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double QR_000000000000=0;
			double QR_000000000001=0;
			double QR_000000000010=0;
			double QR_000000000100=0;
			double QR_000000000002=0;
			double QR_000000000011=0;
			double QR_000000000020=0;
			double QR_000000000101=0;
			double QR_000000000110=0;
			double QR_000000000200=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
			QR_000000000000+=R_000[0];
			QR_000000000001+=aPin1*(R_001[0]);
			QR_000000000010+=aPin1*(R_010[0]);
			QR_000000000100+=aPin1*(R_100[0]);
			QR_000000000002+=aPin2*(R_002[0]);
			QR_000000000011+=aPin2*(R_011[0]);
			QR_000000000020+=aPin2*(R_020[0]);
			QR_000000000101+=aPin2*(R_101[0]);
			QR_000000000110+=aPin2*(R_110[0]);
			QR_000000000200+=aPin2*(R_200[0]);
			}
		double PD_011[3];
		double PD_111[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
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
			P_011000000=PD_011[0];
			P_111000000=PD_111[0];
			P_010001000=PD_010[0]*PD_001[1];
			P_010000001=PD_010[0]*PD_001[2];
			P_001010000=PD_001[0]*PD_010[1];
			P_000011000=PD_011[1];
			P_000111000=PD_111[1];
			P_000010001=PD_010[1]*PD_001[2];
			P_001000010=PD_001[0]*PD_010[2];
			P_000001010=PD_001[1]*PD_010[2];
			P_000000011=PD_011[2];
			P_000000111=PD_111[2];
			a1P_010000000_1=PD_010[0];
			a1P_000001000_1=PD_001[1];
			a1P_000000001_1=PD_001[2];
			a1P_001000000_1=PD_001[0];
			a1P_000010000_1=PD_010[1];
			a1P_000000010_1=PD_010[2];
			ans_temp[ans_id*9+0]+=P_011000000*QR_000000000000+P_111000000*QR_000000000100+QR_000000000200;
			ans_temp[ans_id*9+1]+=P_010001000*QR_000000000000+a1P_010000000_1*QR_000000000010+a1P_000001000_1*QR_000000000100+QR_000000000110;
			ans_temp[ans_id*9+2]+=P_010000001*QR_000000000000+a1P_010000000_1*QR_000000000001+a1P_000000001_1*QR_000000000100+QR_000000000101;
			ans_temp[ans_id*9+3]+=P_001010000*QR_000000000000+a1P_001000000_1*QR_000000000010+a1P_000010000_1*QR_000000000100+QR_000000000110;
			ans_temp[ans_id*9+4]+=P_000011000*QR_000000000000+P_000111000*QR_000000000010+QR_000000000020;
			ans_temp[ans_id*9+5]+=P_000010001*QR_000000000000+a1P_000010000_1*QR_000000000001+a1P_000000001_1*QR_000000000010+QR_000000000011;
			ans_temp[ans_id*9+6]+=P_001000010*QR_000000000000+a1P_001000000_1*QR_000000000001+a1P_000000010_1*QR_000000000100+QR_000000000101;
			ans_temp[ans_id*9+7]+=P_000001010*QR_000000000000+a1P_000001000_1*QR_000000000001+a1P_000000010_1*QR_000000000010+QR_000000000011;
			ans_temp[ans_id*9+8]+=P_000000011*QR_000000000000+P_000000111*QR_000000000001+QR_000000000002;
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
__global__ void TSMJ_dsss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_taylor(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_020[3];
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
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
			P_020000000=PD_020[0];
			P_010010000=PD_010[0]*PD_010[1];
			P_000020000=PD_020[1];
			P_010000010=PD_010[0]*PD_010[2];
			P_000010010=PD_010[1]*PD_010[2];
			P_000000020=PD_020[2];
			a1P_010000000_1=PD_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=PD_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=PD_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=P_020000000*R_000[0]+a1P_010000000_2*R_100[0]+R_200[0];
			ans_temp[ans_id*6+1]+=P_010010000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*6+2]+=P_000020000*R_000[0]+a1P_000010000_2*R_010[0]+R_020[0];
			ans_temp[ans_id*6+3]+=P_010000010*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*6+4]+=P_000010010*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*6+5]+=P_000000020*R_000[0]+a1P_000000010_2*R_001[0]+R_002[0];
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
__global__ void TSMJ_dsss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_020[3];
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
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
			P_020000000=PD_020[0];
			P_010010000=PD_010[0]*PD_010[1];
			P_000020000=PD_020[1];
			P_010000010=PD_010[0]*PD_010[2];
			P_000010010=PD_010[1]*PD_010[2];
			P_000000020=PD_020[2];
			a1P_010000000_1=PD_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=PD_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=PD_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=P_020000000*R_000[0]+a1P_010000000_2*R_100[0]+R_200[0];
			ans_temp[ans_id*6+1]+=P_010010000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*6+2]+=P_000020000*R_000[0]+a1P_000010000_2*R_010[0]+R_020[0];
			ans_temp[ans_id*6+3]+=P_010000010*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*6+4]+=P_000010010*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*6+5]+=P_000000020*R_000[0]+a1P_000000010_2*R_001[0]+R_002[0];
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
__global__ void TSMJ_dsss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
		double PD_020[3];
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
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
			P_020000000=PD_020[0];
			P_010010000=PD_010[0]*PD_010[1];
			P_000020000=PD_020[1];
			P_010000010=PD_010[0]*PD_010[2];
			P_000010010=PD_010[1]*PD_010[2];
			P_000000020=PD_020[2];
			a1P_010000000_1=PD_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=PD_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=PD_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=P_020000000*R_000[0]+a1P_010000000_2*R_100[0]+R_200[0];
			ans_temp[ans_id*6+1]+=P_010010000*R_000[0]+a1P_010000000_1*R_010[0]+a1P_000010000_1*R_100[0]+R_110[0];
			ans_temp[ans_id*6+2]+=P_000020000*R_000[0]+a1P_000010000_2*R_010[0]+R_020[0];
			ans_temp[ans_id*6+3]+=P_010000010*R_000[0]+a1P_010000000_1*R_001[0]+a1P_000000010_1*R_100[0]+R_101[0];
			ans_temp[ans_id*6+4]+=P_000010010*R_000[0]+a1P_000010000_1*R_001[0]+a1P_000000010_1*R_010[0]+R_011[0];
			ans_temp[ans_id*6+5]+=P_000000020*R_000[0]+a1P_000000010_2*R_001[0]+R_002[0];
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
__global__ void TSMJ_dsss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double QR_000000000000=0;
			double QR_000000000001=0;
			double QR_000000000010=0;
			double QR_000000000100=0;
			double QR_000000000002=0;
			double QR_000000000011=0;
			double QR_000000000020=0;
			double QR_000000000101=0;
			double QR_000000000110=0;
			double QR_000000000200=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
                double TX=PX-QX;
                double TY=PY-QY;
                double TZ=PZ-QZ;
                double T=alphaT*(TX*TX+TY*TY+TZ*TZ);
                double R_000[3];
                Ft_fs_1(2,T,R_000);
                R_000[0]*=lmd;
                R_000[1]*=-2*alphaT*lmd;
                R_000[2]*=4*alphaT*alphaT*lmd;
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
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
		R_000[i]*=aPin1;
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
			QR_000000000000+=R_000[0];
			QR_000000000001+=aPin1*(R_001[0]);
			QR_000000000010+=aPin1*(R_010[0]);
			QR_000000000100+=aPin1*(R_100[0]);
			QR_000000000002+=aPin2*(R_002[0]);
			QR_000000000011+=aPin2*(R_011[0]);
			QR_000000000020+=aPin2*(R_020[0]);
			QR_000000000101+=aPin2*(R_101[0]);
			QR_000000000110+=aPin2*(R_110[0]);
			QR_000000000200+=aPin2*(R_200[0]);
			}
		double PD_020[3];
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
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
			P_020000000=PD_020[0];
			P_010010000=PD_010[0]*PD_010[1];
			P_000020000=PD_020[1];
			P_010000010=PD_010[0]*PD_010[2];
			P_000010010=PD_010[1]*PD_010[2];
			P_000000020=PD_020[2];
			a1P_010000000_1=PD_010[0];
			a1P_010000000_2=2*a1P_010000000_1;
			a1P_000010000_1=PD_010[1];
			a1P_000010000_2=2*a1P_000010000_1;
			a1P_000000010_1=PD_010[2];
			a1P_000000010_2=2*a1P_000000010_1;
			ans_temp[ans_id*6+0]+=P_020000000*QR_000000000000+a1P_010000000_2*QR_000000000100+QR_000000000200;
			ans_temp[ans_id*6+1]+=P_010010000*QR_000000000000+a1P_010000000_1*QR_000000000010+a1P_000010000_1*QR_000000000100+QR_000000000110;
			ans_temp[ans_id*6+2]+=P_000020000*QR_000000000000+a1P_000010000_2*QR_000000000010+QR_000000000020;
			ans_temp[ans_id*6+3]+=P_010000010*QR_000000000000+a1P_010000000_1*QR_000000000001+a1P_000000010_1*QR_000000000100+QR_000000000101;
			ans_temp[ans_id*6+4]+=P_000010010*QR_000000000000+a1P_000010000_1*QR_000000000001+a1P_000000010_1*QR_000000000010+QR_000000000011;
			ans_temp[ans_id*6+5]+=P_000000020*QR_000000000000+a1P_000000010_2*QR_000000000001+QR_000000000002;
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
__global__ void TSMJ_dpss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
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
	for(int i=1;i<3;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_001[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
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
			P_021000000=PD_021[0];
			P_121000000=PD_121[0];
			P_221000000=PD_221[0];
			P_020001000=PD_020[0]*PD_001[1];
			P_020000001=PD_020[0]*PD_001[2];
			P_011010000=PD_011[0]*PD_010[1];
			P_111010000=PD_111[0]*PD_010[1];
			P_010011000=PD_010[0]*PD_011[1];
			P_010111000=PD_010[0]*PD_111[1];
			P_010010001=PD_010[0]*PD_010[1]*PD_001[2];
			P_001020000=PD_001[0]*PD_020[1];
			P_000021000=PD_021[1];
			P_000121000=PD_121[1];
			P_000221000=PD_221[1];
			P_000020001=PD_020[1]*PD_001[2];
			P_011000010=PD_011[0]*PD_010[2];
			P_111000010=PD_111[0]*PD_010[2];
			P_010001010=PD_010[0]*PD_001[1]*PD_010[2];
			P_010000011=PD_010[0]*PD_011[2];
			P_010000111=PD_010[0]*PD_111[2];
			P_001010010=PD_001[0]*PD_010[1]*PD_010[2];
			P_000011010=PD_011[1]*PD_010[2];
			P_000111010=PD_111[1]*PD_010[2];
			P_000010011=PD_010[1]*PD_011[2];
			P_000010111=PD_010[1]*PD_111[2];
			P_001000020=PD_001[0]*PD_020[2];
			P_000001020=PD_001[1]*PD_020[2];
			P_000000021=PD_021[2];
			P_000000121=PD_121[2];
			P_000000221=PD_221[2];
			a1P_020000000_1=PD_020[0];
			a1P_010001000_1=PD_010[0]*PD_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=PD_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=PD_001[1];
			a1P_010000001_1=PD_010[0]*PD_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=PD_001[2];
			a1P_011000000_1=PD_011[0];
			a1P_111000000_1=PD_111[0];
			a2P_000010000_1=PD_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=PD_011[1];
			a1P_000111000_1=PD_111[1];
			a1P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010001_1=PD_010[1]*PD_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=PD_001[0]*PD_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=PD_001[0];
			a1P_000020000_1=PD_020[1];
			a2P_000000010_1=PD_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000001010_1=PD_001[1]*PD_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=PD_011[2];
			a1P_000000111_1=PD_111[2];
			a1P_001000010_1=PD_001[0]*PD_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=PD_010[1]*PD_010[2];
			a1P_000000020_1=PD_020[2];
			ans_temp[ans_id*18+0]+=P_021000000*R_000[0]+P_121000000*R_100[0]+P_221000000*R_200[0]+R_300[0];
			ans_temp[ans_id*18+1]+=P_020001000*R_000[0]+a1P_020000000_1*R_010[0]+a1P_010001000_2*R_100[0]+a2P_010000000_2*R_110[0]+a2P_000001000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+2]+=P_020000001*R_000[0]+a1P_020000000_1*R_001[0]+a1P_010000001_2*R_100[0]+a2P_010000000_2*R_101[0]+a2P_000000001_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+3]+=P_011010000*R_000[0]+a1P_011000000_1*R_010[0]+P_111010000*R_100[0]+a1P_111000000_1*R_110[0]+a2P_000010000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+4]+=P_010011000*R_000[0]+P_010111000*R_010[0]+a2P_010000000_1*R_020[0]+a1P_000011000_1*R_100[0]+a1P_000111000_1*R_110[0]+R_120[0];
			ans_temp[ans_id*18+5]+=P_010010001*R_000[0]+a1P_010010000_1*R_001[0]+a1P_010000001_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000010001_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000001_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+6]+=P_001020000*R_000[0]+a1P_001010000_2*R_010[0]+a2P_001000000_1*R_020[0]+a1P_000020000_1*R_100[0]+a2P_000010000_2*R_110[0]+R_120[0];
			ans_temp[ans_id*18+7]+=P_000021000*R_000[0]+P_000121000*R_010[0]+P_000221000*R_020[0]+R_030[0];
			ans_temp[ans_id*18+8]+=P_000020001*R_000[0]+a1P_000020000_1*R_001[0]+a1P_000010001_2*R_010[0]+a2P_000010000_2*R_011[0]+a2P_000000001_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+9]+=P_011000010*R_000[0]+a1P_011000000_1*R_001[0]+P_111000010*R_100[0]+a1P_111000000_1*R_101[0]+a2P_000000010_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+10]+=P_010001010*R_000[0]+a1P_010001000_1*R_001[0]+a1P_010000010_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000001010_1*R_100[0]+a2P_000001000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+11]+=P_010000011*R_000[0]+P_010000111*R_001[0]+a2P_010000000_1*R_002[0]+a1P_000000011_1*R_100[0]+a1P_000000111_1*R_101[0]+R_102[0];
			ans_temp[ans_id*18+12]+=P_001010010*R_000[0]+a1P_001010000_1*R_001[0]+a1P_001000010_1*R_010[0]+a2P_001000000_1*R_011[0]+a1P_000010010_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+13]+=P_000011010*R_000[0]+a1P_000011000_1*R_001[0]+P_000111010*R_010[0]+a1P_000111000_1*R_011[0]+a2P_000000010_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+14]+=P_000010011*R_000[0]+P_000010111*R_001[0]+a2P_000010000_1*R_002[0]+a1P_000000011_1*R_010[0]+a1P_000000111_1*R_011[0]+R_012[0];
			ans_temp[ans_id*18+15]+=P_001000020*R_000[0]+a1P_001000010_2*R_001[0]+a2P_001000000_1*R_002[0]+a1P_000000020_1*R_100[0]+a2P_000000010_2*R_101[0]+R_102[0];
			ans_temp[ans_id*18+16]+=P_000001020*R_000[0]+a1P_000001010_2*R_001[0]+a2P_000001000_1*R_002[0]+a1P_000000020_1*R_010[0]+a2P_000000010_2*R_011[0]+R_012[0];
			ans_temp[ans_id*18+17]+=P_000000021*R_000[0]+P_000000121*R_001[0]+P_000000221*R_002[0]+R_003[0];
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
__global__ void TSMJ_dpss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
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
	for(int i=1;i<3;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_001[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
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
			P_021000000=PD_021[0];
			P_121000000=PD_121[0];
			P_221000000=PD_221[0];
			P_020001000=PD_020[0]*PD_001[1];
			P_020000001=PD_020[0]*PD_001[2];
			P_011010000=PD_011[0]*PD_010[1];
			P_111010000=PD_111[0]*PD_010[1];
			P_010011000=PD_010[0]*PD_011[1];
			P_010111000=PD_010[0]*PD_111[1];
			P_010010001=PD_010[0]*PD_010[1]*PD_001[2];
			P_001020000=PD_001[0]*PD_020[1];
			P_000021000=PD_021[1];
			P_000121000=PD_121[1];
			P_000221000=PD_221[1];
			P_000020001=PD_020[1]*PD_001[2];
			P_011000010=PD_011[0]*PD_010[2];
			P_111000010=PD_111[0]*PD_010[2];
			P_010001010=PD_010[0]*PD_001[1]*PD_010[2];
			P_010000011=PD_010[0]*PD_011[2];
			P_010000111=PD_010[0]*PD_111[2];
			P_001010010=PD_001[0]*PD_010[1]*PD_010[2];
			P_000011010=PD_011[1]*PD_010[2];
			P_000111010=PD_111[1]*PD_010[2];
			P_000010011=PD_010[1]*PD_011[2];
			P_000010111=PD_010[1]*PD_111[2];
			P_001000020=PD_001[0]*PD_020[2];
			P_000001020=PD_001[1]*PD_020[2];
			P_000000021=PD_021[2];
			P_000000121=PD_121[2];
			P_000000221=PD_221[2];
			a1P_020000000_1=PD_020[0];
			a1P_010001000_1=PD_010[0]*PD_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=PD_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=PD_001[1];
			a1P_010000001_1=PD_010[0]*PD_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=PD_001[2];
			a1P_011000000_1=PD_011[0];
			a1P_111000000_1=PD_111[0];
			a2P_000010000_1=PD_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=PD_011[1];
			a1P_000111000_1=PD_111[1];
			a1P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010001_1=PD_010[1]*PD_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=PD_001[0]*PD_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=PD_001[0];
			a1P_000020000_1=PD_020[1];
			a2P_000000010_1=PD_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000001010_1=PD_001[1]*PD_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=PD_011[2];
			a1P_000000111_1=PD_111[2];
			a1P_001000010_1=PD_001[0]*PD_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=PD_010[1]*PD_010[2];
			a1P_000000020_1=PD_020[2];
			ans_temp[ans_id*18+0]+=P_021000000*R_000[0]+P_121000000*R_100[0]+P_221000000*R_200[0]+R_300[0];
			ans_temp[ans_id*18+1]+=P_020001000*R_000[0]+a1P_020000000_1*R_010[0]+a1P_010001000_2*R_100[0]+a2P_010000000_2*R_110[0]+a2P_000001000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+2]+=P_020000001*R_000[0]+a1P_020000000_1*R_001[0]+a1P_010000001_2*R_100[0]+a2P_010000000_2*R_101[0]+a2P_000000001_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+3]+=P_011010000*R_000[0]+a1P_011000000_1*R_010[0]+P_111010000*R_100[0]+a1P_111000000_1*R_110[0]+a2P_000010000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+4]+=P_010011000*R_000[0]+P_010111000*R_010[0]+a2P_010000000_1*R_020[0]+a1P_000011000_1*R_100[0]+a1P_000111000_1*R_110[0]+R_120[0];
			ans_temp[ans_id*18+5]+=P_010010001*R_000[0]+a1P_010010000_1*R_001[0]+a1P_010000001_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000010001_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000001_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+6]+=P_001020000*R_000[0]+a1P_001010000_2*R_010[0]+a2P_001000000_1*R_020[0]+a1P_000020000_1*R_100[0]+a2P_000010000_2*R_110[0]+R_120[0];
			ans_temp[ans_id*18+7]+=P_000021000*R_000[0]+P_000121000*R_010[0]+P_000221000*R_020[0]+R_030[0];
			ans_temp[ans_id*18+8]+=P_000020001*R_000[0]+a1P_000020000_1*R_001[0]+a1P_000010001_2*R_010[0]+a2P_000010000_2*R_011[0]+a2P_000000001_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+9]+=P_011000010*R_000[0]+a1P_011000000_1*R_001[0]+P_111000010*R_100[0]+a1P_111000000_1*R_101[0]+a2P_000000010_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+10]+=P_010001010*R_000[0]+a1P_010001000_1*R_001[0]+a1P_010000010_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000001010_1*R_100[0]+a2P_000001000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+11]+=P_010000011*R_000[0]+P_010000111*R_001[0]+a2P_010000000_1*R_002[0]+a1P_000000011_1*R_100[0]+a1P_000000111_1*R_101[0]+R_102[0];
			ans_temp[ans_id*18+12]+=P_001010010*R_000[0]+a1P_001010000_1*R_001[0]+a1P_001000010_1*R_010[0]+a2P_001000000_1*R_011[0]+a1P_000010010_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+13]+=P_000011010*R_000[0]+a1P_000011000_1*R_001[0]+P_000111010*R_010[0]+a1P_000111000_1*R_011[0]+a2P_000000010_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+14]+=P_000010011*R_000[0]+P_000010111*R_001[0]+a2P_000010000_1*R_002[0]+a1P_000000011_1*R_010[0]+a1P_000000111_1*R_011[0]+R_012[0];
			ans_temp[ans_id*18+15]+=P_001000020*R_000[0]+a1P_001000010_2*R_001[0]+a2P_001000000_1*R_002[0]+a1P_000000020_1*R_100[0]+a2P_000000010_2*R_101[0]+R_102[0];
			ans_temp[ans_id*18+16]+=P_000001020*R_000[0]+a1P_000001010_2*R_001[0]+a2P_000001000_1*R_002[0]+a1P_000000020_1*R_010[0]+a2P_000000010_2*R_011[0]+R_012[0];
			ans_temp[ans_id*18+17]+=P_000000021*R_000[0]+P_000000121*R_001[0]+P_000000221*R_002[0]+R_003[0];
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
__global__ void TSMJ_dpss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
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
	for(int i=1;i<3;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_001[i]*=aPin1;
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
		double PD_011[3];
		double PD_111[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
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
			P_021000000=PD_021[0];
			P_121000000=PD_121[0];
			P_221000000=PD_221[0];
			P_020001000=PD_020[0]*PD_001[1];
			P_020000001=PD_020[0]*PD_001[2];
			P_011010000=PD_011[0]*PD_010[1];
			P_111010000=PD_111[0]*PD_010[1];
			P_010011000=PD_010[0]*PD_011[1];
			P_010111000=PD_010[0]*PD_111[1];
			P_010010001=PD_010[0]*PD_010[1]*PD_001[2];
			P_001020000=PD_001[0]*PD_020[1];
			P_000021000=PD_021[1];
			P_000121000=PD_121[1];
			P_000221000=PD_221[1];
			P_000020001=PD_020[1]*PD_001[2];
			P_011000010=PD_011[0]*PD_010[2];
			P_111000010=PD_111[0]*PD_010[2];
			P_010001010=PD_010[0]*PD_001[1]*PD_010[2];
			P_010000011=PD_010[0]*PD_011[2];
			P_010000111=PD_010[0]*PD_111[2];
			P_001010010=PD_001[0]*PD_010[1]*PD_010[2];
			P_000011010=PD_011[1]*PD_010[2];
			P_000111010=PD_111[1]*PD_010[2];
			P_000010011=PD_010[1]*PD_011[2];
			P_000010111=PD_010[1]*PD_111[2];
			P_001000020=PD_001[0]*PD_020[2];
			P_000001020=PD_001[1]*PD_020[2];
			P_000000021=PD_021[2];
			P_000000121=PD_121[2];
			P_000000221=PD_221[2];
			a1P_020000000_1=PD_020[0];
			a1P_010001000_1=PD_010[0]*PD_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=PD_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=PD_001[1];
			a1P_010000001_1=PD_010[0]*PD_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=PD_001[2];
			a1P_011000000_1=PD_011[0];
			a1P_111000000_1=PD_111[0];
			a2P_000010000_1=PD_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=PD_011[1];
			a1P_000111000_1=PD_111[1];
			a1P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010001_1=PD_010[1]*PD_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=PD_001[0]*PD_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=PD_001[0];
			a1P_000020000_1=PD_020[1];
			a2P_000000010_1=PD_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000001010_1=PD_001[1]*PD_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=PD_011[2];
			a1P_000000111_1=PD_111[2];
			a1P_001000010_1=PD_001[0]*PD_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=PD_010[1]*PD_010[2];
			a1P_000000020_1=PD_020[2];
			ans_temp[ans_id*18+0]+=P_021000000*R_000[0]+P_121000000*R_100[0]+P_221000000*R_200[0]+R_300[0];
			ans_temp[ans_id*18+1]+=P_020001000*R_000[0]+a1P_020000000_1*R_010[0]+a1P_010001000_2*R_100[0]+a2P_010000000_2*R_110[0]+a2P_000001000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+2]+=P_020000001*R_000[0]+a1P_020000000_1*R_001[0]+a1P_010000001_2*R_100[0]+a2P_010000000_2*R_101[0]+a2P_000000001_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+3]+=P_011010000*R_000[0]+a1P_011000000_1*R_010[0]+P_111010000*R_100[0]+a1P_111000000_1*R_110[0]+a2P_000010000_1*R_200[0]+R_210[0];
			ans_temp[ans_id*18+4]+=P_010011000*R_000[0]+P_010111000*R_010[0]+a2P_010000000_1*R_020[0]+a1P_000011000_1*R_100[0]+a1P_000111000_1*R_110[0]+R_120[0];
			ans_temp[ans_id*18+5]+=P_010010001*R_000[0]+a1P_010010000_1*R_001[0]+a1P_010000001_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000010001_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000001_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+6]+=P_001020000*R_000[0]+a1P_001010000_2*R_010[0]+a2P_001000000_1*R_020[0]+a1P_000020000_1*R_100[0]+a2P_000010000_2*R_110[0]+R_120[0];
			ans_temp[ans_id*18+7]+=P_000021000*R_000[0]+P_000121000*R_010[0]+P_000221000*R_020[0]+R_030[0];
			ans_temp[ans_id*18+8]+=P_000020001*R_000[0]+a1P_000020000_1*R_001[0]+a1P_000010001_2*R_010[0]+a2P_000010000_2*R_011[0]+a2P_000000001_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+9]+=P_011000010*R_000[0]+a1P_011000000_1*R_001[0]+P_111000010*R_100[0]+a1P_111000000_1*R_101[0]+a2P_000000010_1*R_200[0]+R_201[0];
			ans_temp[ans_id*18+10]+=P_010001010*R_000[0]+a1P_010001000_1*R_001[0]+a1P_010000010_1*R_010[0]+a2P_010000000_1*R_011[0]+a1P_000001010_1*R_100[0]+a2P_000001000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+11]+=P_010000011*R_000[0]+P_010000111*R_001[0]+a2P_010000000_1*R_002[0]+a1P_000000011_1*R_100[0]+a1P_000000111_1*R_101[0]+R_102[0];
			ans_temp[ans_id*18+12]+=P_001010010*R_000[0]+a1P_001010000_1*R_001[0]+a1P_001000010_1*R_010[0]+a2P_001000000_1*R_011[0]+a1P_000010010_1*R_100[0]+a2P_000010000_1*R_101[0]+a2P_000000010_1*R_110[0]+R_111[0];
			ans_temp[ans_id*18+13]+=P_000011010*R_000[0]+a1P_000011000_1*R_001[0]+P_000111010*R_010[0]+a1P_000111000_1*R_011[0]+a2P_000000010_1*R_020[0]+R_021[0];
			ans_temp[ans_id*18+14]+=P_000010011*R_000[0]+P_000010111*R_001[0]+a2P_000010000_1*R_002[0]+a1P_000000011_1*R_010[0]+a1P_000000111_1*R_011[0]+R_012[0];
			ans_temp[ans_id*18+15]+=P_001000020*R_000[0]+a1P_001000010_2*R_001[0]+a2P_001000000_1*R_002[0]+a1P_000000020_1*R_100[0]+a2P_000000010_2*R_101[0]+R_102[0];
			ans_temp[ans_id*18+16]+=P_000001020*R_000[0]+a1P_000001010_2*R_001[0]+a2P_000001000_1*R_002[0]+a1P_000000020_1*R_010[0]+a2P_000000010_2*R_011[0]+R_012[0];
			ans_temp[ans_id*18+17]+=P_000000021*R_000[0]+P_000000121*R_001[0]+P_000000221*R_002[0]+R_003[0];
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
__global__ void TSMJ_dpss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double QR_000000000000=0;
			double QR_000000000001=0;
			double QR_000000000010=0;
			double QR_000000000100=0;
			double QR_000000000002=0;
			double QR_000000000011=0;
			double QR_000000000020=0;
			double QR_000000000101=0;
			double QR_000000000110=0;
			double QR_000000000200=0;
			double QR_000000000003=0;
			double QR_000000000012=0;
			double QR_000000000021=0;
			double QR_000000000030=0;
			double QR_000000000102=0;
			double QR_000000000111=0;
			double QR_000000000120=0;
			double QR_000000000201=0;
			double QR_000000000210=0;
			double QR_000000000300=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
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
	for(int i=1;i<3;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_001[i]*=aPin1;
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
			QR_000000000000+=R_000[0];
			QR_000000000001+=aPin1*(R_001[0]);
			QR_000000000010+=aPin1*(R_010[0]);
			QR_000000000100+=aPin1*(R_100[0]);
			QR_000000000002+=aPin2*(R_002[0]);
			QR_000000000011+=aPin2*(R_011[0]);
			QR_000000000020+=aPin2*(R_020[0]);
			QR_000000000101+=aPin2*(R_101[0]);
			QR_000000000110+=aPin2*(R_110[0]);
			QR_000000000200+=aPin2*(R_200[0]);
			QR_000000000003+=aPin3*(R_003[0]);
			QR_000000000012+=aPin3*(R_012[0]);
			QR_000000000021+=aPin3*(R_021[0]);
			QR_000000000030+=aPin3*(R_030[0]);
			QR_000000000102+=aPin3*(R_102[0]);
			QR_000000000111+=aPin3*(R_111[0]);
			QR_000000000120+=aPin3*(R_120[0]);
			QR_000000000201+=aPin3*(R_201[0]);
			QR_000000000210+=aPin3*(R_210[0]);
			QR_000000000300+=aPin3*(R_300[0]);
			}
		double PD_011[3];
		double PD_111[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
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
			P_021000000=PD_021[0];
			P_121000000=PD_121[0];
			P_221000000=PD_221[0];
			P_020001000=PD_020[0]*PD_001[1];
			P_020000001=PD_020[0]*PD_001[2];
			P_011010000=PD_011[0]*PD_010[1];
			P_111010000=PD_111[0]*PD_010[1];
			P_010011000=PD_010[0]*PD_011[1];
			P_010111000=PD_010[0]*PD_111[1];
			P_010010001=PD_010[0]*PD_010[1]*PD_001[2];
			P_001020000=PD_001[0]*PD_020[1];
			P_000021000=PD_021[1];
			P_000121000=PD_121[1];
			P_000221000=PD_221[1];
			P_000020001=PD_020[1]*PD_001[2];
			P_011000010=PD_011[0]*PD_010[2];
			P_111000010=PD_111[0]*PD_010[2];
			P_010001010=PD_010[0]*PD_001[1]*PD_010[2];
			P_010000011=PD_010[0]*PD_011[2];
			P_010000111=PD_010[0]*PD_111[2];
			P_001010010=PD_001[0]*PD_010[1]*PD_010[2];
			P_000011010=PD_011[1]*PD_010[2];
			P_000111010=PD_111[1]*PD_010[2];
			P_000010011=PD_010[1]*PD_011[2];
			P_000010111=PD_010[1]*PD_111[2];
			P_001000020=PD_001[0]*PD_020[2];
			P_000001020=PD_001[1]*PD_020[2];
			P_000000021=PD_021[2];
			P_000000121=PD_121[2];
			P_000000221=PD_221[2];
			a1P_020000000_1=PD_020[0];
			a1P_010001000_1=PD_010[0]*PD_001[1];
			a1P_010001000_2=2*a1P_010001000_1;
			a2P_010000000_1=PD_010[0];
			a2P_010000000_2=2*a2P_010000000_1;
			a2P_000001000_1=PD_001[1];
			a1P_010000001_1=PD_010[0]*PD_001[2];
			a1P_010000001_2=2*a1P_010000001_1;
			a2P_000000001_1=PD_001[2];
			a1P_011000000_1=PD_011[0];
			a1P_111000000_1=PD_111[0];
			a2P_000010000_1=PD_010[1];
			a2P_000010000_2=2*a2P_000010000_1;
			a1P_000011000_1=PD_011[1];
			a1P_000111000_1=PD_111[1];
			a1P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010001_1=PD_010[1]*PD_001[2];
			a1P_000010001_2=2*a1P_000010001_1;
			a1P_001010000_1=PD_001[0]*PD_010[1];
			a1P_001010000_2=2*a1P_001010000_1;
			a2P_001000000_1=PD_001[0];
			a1P_000020000_1=PD_020[1];
			a2P_000000010_1=PD_010[2];
			a2P_000000010_2=2*a2P_000000010_1;
			a1P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000001010_1=PD_001[1]*PD_010[2];
			a1P_000001010_2=2*a1P_000001010_1;
			a1P_000000011_1=PD_011[2];
			a1P_000000111_1=PD_111[2];
			a1P_001000010_1=PD_001[0]*PD_010[2];
			a1P_001000010_2=2*a1P_001000010_1;
			a1P_000010010_1=PD_010[1]*PD_010[2];
			a1P_000000020_1=PD_020[2];
			ans_temp[ans_id*18+0]+=P_021000000*QR_000000000000+P_121000000*QR_000000000100+P_221000000*QR_000000000200+QR_000000000300;
			ans_temp[ans_id*18+1]+=P_020001000*QR_000000000000+a1P_020000000_1*QR_000000000010+a1P_010001000_2*QR_000000000100+a2P_010000000_2*QR_000000000110+a2P_000001000_1*QR_000000000200+QR_000000000210;
			ans_temp[ans_id*18+2]+=P_020000001*QR_000000000000+a1P_020000000_1*QR_000000000001+a1P_010000001_2*QR_000000000100+a2P_010000000_2*QR_000000000101+a2P_000000001_1*QR_000000000200+QR_000000000201;
			ans_temp[ans_id*18+3]+=P_011010000*QR_000000000000+a1P_011000000_1*QR_000000000010+P_111010000*QR_000000000100+a1P_111000000_1*QR_000000000110+a2P_000010000_1*QR_000000000200+QR_000000000210;
			ans_temp[ans_id*18+4]+=P_010011000*QR_000000000000+P_010111000*QR_000000000010+a2P_010000000_1*QR_000000000020+a1P_000011000_1*QR_000000000100+a1P_000111000_1*QR_000000000110+QR_000000000120;
			ans_temp[ans_id*18+5]+=P_010010001*QR_000000000000+a1P_010010000_1*QR_000000000001+a1P_010000001_1*QR_000000000010+a2P_010000000_1*QR_000000000011+a1P_000010001_1*QR_000000000100+a2P_000010000_1*QR_000000000101+a2P_000000001_1*QR_000000000110+QR_000000000111;
			ans_temp[ans_id*18+6]+=P_001020000*QR_000000000000+a1P_001010000_2*QR_000000000010+a2P_001000000_1*QR_000000000020+a1P_000020000_1*QR_000000000100+a2P_000010000_2*QR_000000000110+QR_000000000120;
			ans_temp[ans_id*18+7]+=P_000021000*QR_000000000000+P_000121000*QR_000000000010+P_000221000*QR_000000000020+QR_000000000030;
			ans_temp[ans_id*18+8]+=P_000020001*QR_000000000000+a1P_000020000_1*QR_000000000001+a1P_000010001_2*QR_000000000010+a2P_000010000_2*QR_000000000011+a2P_000000001_1*QR_000000000020+QR_000000000021;
			ans_temp[ans_id*18+9]+=P_011000010*QR_000000000000+a1P_011000000_1*QR_000000000001+P_111000010*QR_000000000100+a1P_111000000_1*QR_000000000101+a2P_000000010_1*QR_000000000200+QR_000000000201;
			ans_temp[ans_id*18+10]+=P_010001010*QR_000000000000+a1P_010001000_1*QR_000000000001+a1P_010000010_1*QR_000000000010+a2P_010000000_1*QR_000000000011+a1P_000001010_1*QR_000000000100+a2P_000001000_1*QR_000000000101+a2P_000000010_1*QR_000000000110+QR_000000000111;
			ans_temp[ans_id*18+11]+=P_010000011*QR_000000000000+P_010000111*QR_000000000001+a2P_010000000_1*QR_000000000002+a1P_000000011_1*QR_000000000100+a1P_000000111_1*QR_000000000101+QR_000000000102;
			ans_temp[ans_id*18+12]+=P_001010010*QR_000000000000+a1P_001010000_1*QR_000000000001+a1P_001000010_1*QR_000000000010+a2P_001000000_1*QR_000000000011+a1P_000010010_1*QR_000000000100+a2P_000010000_1*QR_000000000101+a2P_000000010_1*QR_000000000110+QR_000000000111;
			ans_temp[ans_id*18+13]+=P_000011010*QR_000000000000+a1P_000011000_1*QR_000000000001+P_000111010*QR_000000000010+a1P_000111000_1*QR_000000000011+a2P_000000010_1*QR_000000000020+QR_000000000021;
			ans_temp[ans_id*18+14]+=P_000010011*QR_000000000000+P_000010111*QR_000000000001+a2P_000010000_1*QR_000000000002+a1P_000000011_1*QR_000000000010+a1P_000000111_1*QR_000000000011+QR_000000000012;
			ans_temp[ans_id*18+15]+=P_001000020*QR_000000000000+a1P_001000010_2*QR_000000000001+a2P_001000000_1*QR_000000000002+a1P_000000020_1*QR_000000000100+a2P_000000010_2*QR_000000000101+QR_000000000102;
			ans_temp[ans_id*18+16]+=P_000001020*QR_000000000000+a1P_000001010_2*QR_000000000001+a2P_000001000_1*QR_000000000002+a1P_000000020_1*QR_000000000010+a2P_000000010_2*QR_000000000011+QR_000000000012;
			ans_temp[ans_id*18+17]+=P_000000021*QR_000000000000+P_000000121*QR_000000000001+P_000000221*QR_000000000002+QR_000000000003;
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
__global__ void TSMJ_ddss_taylor(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
                double alphaT=rsqrt(Eta+Zta);
                double lmd=2*P25*pp*pq*alphaT;
                alphaT=Eta*Zta*alphaT*alphaT;
                lmd*=Pmtrx[0];
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
	R_000[4]*=aPin4;
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
	for(int i=1;i<4;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<3;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_001[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_200[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_020[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_002[i]*=aPin1;
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
		double PD_002[3];
		double PD_102[3];
		double PD_011[3];
		double PD_111[3];
		double PD_012[3];
		double PD_112[3];
		double PD_212[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		double PD_022[3];
		double PD_122[3];
		double PD_222[3];
		for(int i=0;i<3;i++){
			PD_002[i]=aPin1+PD_001[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_102[i]=(2.000000*PD_001[i]);
			}
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_012[i]=PD_111[i]+PD_001[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_112[i]=(PD_002[i]+2.000000*PD_011[i]);
			}
		for(int i=0;i<3;i++){
			PD_212[i]=(0.500000*PD_102[i]+PD_111[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
			}
		for(int i=0;i<3;i++){
			PD_022[i]=PD_112[i]+PD_010[i]*PD_012[i];
			}
		for(int i=0;i<3;i++){
			PD_122[i]=2.000000*(PD_012[i]+PD_021[i]);
			}
		for(int i=0;i<3;i++){
			PD_222[i]=(PD_112[i]+PD_121[i]);
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
			P_022000000=PD_022[0];
			P_122000000=PD_122[0];
			P_222000000=PD_222[0];
			P_021001000=PD_021[0]*PD_001[1];
			P_121001000=PD_121[0]*PD_001[1];
			P_221001000=PD_221[0]*PD_001[1];
			P_020002000=PD_020[0]*PD_002[1];
			P_021000001=PD_021[0]*PD_001[2];
			P_121000001=PD_121[0]*PD_001[2];
			P_221000001=PD_221[0]*PD_001[2];
			P_020001001=PD_020[0]*PD_001[1]*PD_001[2];
			P_020000002=PD_020[0]*PD_002[2];
			P_012010000=PD_012[0]*PD_010[1];
			P_112010000=PD_112[0]*PD_010[1];
			P_212010000=PD_212[0]*PD_010[1];
			P_011011000=PD_011[0]*PD_011[1];
			P_011111000=PD_011[0]*PD_111[1];
			P_111011000=PD_111[0]*PD_011[1];
			P_111111000=PD_111[0]*PD_111[1];
			P_010012000=PD_010[0]*PD_012[1];
			P_010112000=PD_010[0]*PD_112[1];
			P_010212000=PD_010[0]*PD_212[1];
			P_011010001=PD_011[0]*PD_010[1]*PD_001[2];
			P_111010001=PD_111[0]*PD_010[1]*PD_001[2];
			P_010011001=PD_010[0]*PD_011[1]*PD_001[2];
			P_010111001=PD_010[0]*PD_111[1]*PD_001[2];
			P_010010002=PD_010[0]*PD_010[1]*PD_002[2];
			P_002020000=PD_002[0]*PD_020[1];
			P_001021000=PD_001[0]*PD_021[1];
			P_001121000=PD_001[0]*PD_121[1];
			P_001221000=PD_001[0]*PD_221[1];
			P_000022000=PD_022[1];
			P_000122000=PD_122[1];
			P_000222000=PD_222[1];
			P_001020001=PD_001[0]*PD_020[1]*PD_001[2];
			P_000021001=PD_021[1]*PD_001[2];
			P_000121001=PD_121[1]*PD_001[2];
			P_000221001=PD_221[1]*PD_001[2];
			P_000020002=PD_020[1]*PD_002[2];
			P_012000010=PD_012[0]*PD_010[2];
			P_112000010=PD_112[0]*PD_010[2];
			P_212000010=PD_212[0]*PD_010[2];
			P_011001010=PD_011[0]*PD_001[1]*PD_010[2];
			P_111001010=PD_111[0]*PD_001[1]*PD_010[2];
			P_010002010=PD_010[0]*PD_002[1]*PD_010[2];
			P_011000011=PD_011[0]*PD_011[2];
			P_011000111=PD_011[0]*PD_111[2];
			P_111000011=PD_111[0]*PD_011[2];
			P_111000111=PD_111[0]*PD_111[2];
			P_010001011=PD_010[0]*PD_001[1]*PD_011[2];
			P_010001111=PD_010[0]*PD_001[1]*PD_111[2];
			P_010000012=PD_010[0]*PD_012[2];
			P_010000112=PD_010[0]*PD_112[2];
			P_010000212=PD_010[0]*PD_212[2];
			P_002010010=PD_002[0]*PD_010[1]*PD_010[2];
			P_001011010=PD_001[0]*PD_011[1]*PD_010[2];
			P_001111010=PD_001[0]*PD_111[1]*PD_010[2];
			P_000012010=PD_012[1]*PD_010[2];
			P_000112010=PD_112[1]*PD_010[2];
			P_000212010=PD_212[1]*PD_010[2];
			P_001010011=PD_001[0]*PD_010[1]*PD_011[2];
			P_001010111=PD_001[0]*PD_010[1]*PD_111[2];
			P_000011011=PD_011[1]*PD_011[2];
			P_000011111=PD_011[1]*PD_111[2];
			P_000111011=PD_111[1]*PD_011[2];
			P_000111111=PD_111[1]*PD_111[2];
			P_000010012=PD_010[1]*PD_012[2];
			P_000010112=PD_010[1]*PD_112[2];
			P_000010212=PD_010[1]*PD_212[2];
			P_002000020=PD_002[0]*PD_020[2];
			P_001001020=PD_001[0]*PD_001[1]*PD_020[2];
			P_000002020=PD_002[1]*PD_020[2];
			P_001000021=PD_001[0]*PD_021[2];
			P_001000121=PD_001[0]*PD_121[2];
			P_001000221=PD_001[0]*PD_221[2];
			P_000001021=PD_001[1]*PD_021[2];
			P_000001121=PD_001[1]*PD_121[2];
			P_000001221=PD_001[1]*PD_221[2];
			P_000000022=PD_022[2];
			P_000000122=PD_122[2];
			P_000000222=PD_222[2];
			a2P_111000000_1=PD_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=PD_021[0];
			a1P_121000000_1=PD_121[0];
			a1P_221000000_1=PD_221[0];
			a3P_000001000_1=PD_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=PD_020[0]*PD_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=PD_020[0];
			a1P_010002000_1=PD_010[0]*PD_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=PD_010[0]*PD_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=PD_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=PD_002[1];
			a3P_000000001_1=PD_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=PD_020[0]*PD_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=PD_010[0]*PD_001[1]*PD_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=PD_010[0]*PD_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=PD_001[1]*PD_001[2];
			a1P_010000002_1=PD_010[0]*PD_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=PD_002[2];
			a1P_012000000_1=PD_012[0];
			a1P_112000000_1=PD_112[0];
			a1P_212000000_1=PD_212[0];
			a3P_000010000_1=PD_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=PD_011[0];
			a2P_000011000_1=PD_011[1];
			a2P_000111000_1=PD_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=PD_012[1];
			a1P_000112000_1=PD_112[1];
			a1P_000212000_1=PD_212[1];
			a1P_011010000_1=PD_011[0]*PD_010[1];
			a1P_011000001_1=PD_011[0]*PD_001[2];
			a1P_111010000_1=PD_111[0]*PD_010[1];
			a1P_111000001_1=PD_111[0]*PD_001[2];
			a2P_000010001_1=PD_010[1]*PD_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=PD_010[0]*PD_011[1];
			a1P_010111000_1=PD_010[0]*PD_111[1];
			a1P_000011001_1=PD_011[1]*PD_001[2];
			a1P_000111001_1=PD_111[1]*PD_001[2];
			a1P_010010001_1=PD_010[0]*PD_010[1]*PD_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010002_1=PD_010[1]*PD_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=PD_002[0]*PD_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=PD_002[0];
			a1P_001020000_1=PD_001[0]*PD_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=PD_001[0]*PD_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=PD_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=PD_020[1];
			a1P_000021000_1=PD_021[1];
			a1P_000121000_1=PD_121[1];
			a1P_000221000_1=PD_221[1];
			a1P_001010001_1=PD_001[0]*PD_010[1]*PD_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=PD_001[0]*PD_001[2];
			a1P_000020001_1=PD_020[1]*PD_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=PD_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=PD_011[0]*PD_001[1];
			a1P_011000010_1=PD_011[0]*PD_010[2];
			a1P_111001000_1=PD_111[0]*PD_001[1];
			a1P_111000010_1=PD_111[0]*PD_010[2];
			a2P_000001010_1=PD_001[1]*PD_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=PD_010[0]*PD_001[1]*PD_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000002010_1=PD_002[1]*PD_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=PD_011[2];
			a2P_000000111_1=PD_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=PD_010[0]*PD_011[2];
			a1P_010000111_1=PD_010[0]*PD_111[2];
			a1P_000001011_1=PD_001[1]*PD_011[2];
			a1P_000001111_1=PD_001[1]*PD_111[2];
			a1P_000000012_1=PD_012[2];
			a1P_000000112_1=PD_112[2];
			a1P_000000212_1=PD_212[2];
			a1P_002000010_1=PD_002[0]*PD_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=PD_001[0]*PD_010[1]*PD_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=PD_001[0]*PD_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=PD_010[1]*PD_010[2];
			a1P_001011000_1=PD_001[0]*PD_011[1];
			a1P_001111000_1=PD_001[0]*PD_111[1];
			a1P_000011010_1=PD_011[1]*PD_010[2];
			a1P_000111010_1=PD_111[1]*PD_010[2];
			a1P_001000011_1=PD_001[0]*PD_011[2];
			a1P_001000111_1=PD_001[0]*PD_111[2];
			a1P_000010011_1=PD_010[1]*PD_011[2];
			a1P_000010111_1=PD_010[1]*PD_111[2];
			a1P_001000020_1=PD_001[0]*PD_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=PD_020[2];
			a1P_001001010_1=PD_001[0]*PD_001[1]*PD_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=PD_001[0]*PD_001[1];
			a1P_000001020_1=PD_001[1]*PD_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=PD_021[2];
			a1P_000000121_1=PD_121[2];
			a1P_000000221_1=PD_221[2];
			ans_temp[ans_id*36+0]+=P_022000000*R_000[0]+P_122000000*R_100[0]+P_222000000*R_200[0]+a2P_111000000_2*R_300[0]+R_400[0];
			ans_temp[ans_id*36+1]+=P_021001000*R_000[0]+a1P_021000000_1*R_010[0]+P_121001000*R_100[0]+a1P_121000000_1*R_110[0]+P_221001000*R_200[0]+a1P_221000000_1*R_210[0]+a3P_000001000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+2]+=P_020002000*R_000[0]+a1P_020001000_2*R_010[0]+a2P_020000000_1*R_020[0]+a1P_010002000_2*R_100[0]+a2P_010001000_4*R_110[0]+a3P_010000000_2*R_120[0]+a2P_000002000_1*R_200[0]+a3P_000001000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+3]+=P_021000001*R_000[0]+a1P_021000000_1*R_001[0]+P_121000001*R_100[0]+a1P_121000000_1*R_101[0]+P_221000001*R_200[0]+a1P_221000000_1*R_201[0]+a3P_000000001_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+4]+=P_020001001*R_000[0]+a1P_020001000_1*R_001[0]+a1P_020000001_1*R_010[0]+a2P_020000000_1*R_011[0]+a1P_010001001_2*R_100[0]+a2P_010001000_2*R_101[0]+a2P_010000001_2*R_110[0]+a3P_010000000_2*R_111[0]+a2P_000001001_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+5]+=P_020000002*R_000[0]+a1P_020000001_2*R_001[0]+a2P_020000000_1*R_002[0]+a1P_010000002_2*R_100[0]+a2P_010000001_4*R_101[0]+a3P_010000000_2*R_102[0]+a2P_000000002_1*R_200[0]+a3P_000000001_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+6]+=P_012010000*R_000[0]+a1P_012000000_1*R_010[0]+P_112010000*R_100[0]+a1P_112000000_1*R_110[0]+P_212010000*R_200[0]+a1P_212000000_1*R_210[0]+a3P_000010000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+7]+=P_011011000*R_000[0]+P_011111000*R_010[0]+a2P_011000000_1*R_020[0]+P_111011000*R_100[0]+P_111111000*R_110[0]+a2P_111000000_1*R_120[0]+a2P_000011000_1*R_200[0]+a2P_000111000_1*R_210[0]+R_220[0];
			ans_temp[ans_id*36+8]+=P_010012000*R_000[0]+P_010112000*R_010[0]+P_010212000*R_020[0]+a3P_010000000_1*R_030[0]+a1P_000012000_1*R_100[0]+a1P_000112000_1*R_110[0]+a1P_000212000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+9]+=P_011010001*R_000[0]+a1P_011010000_1*R_001[0]+a1P_011000001_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111010001*R_100[0]+a1P_111010000_1*R_101[0]+a1P_111000001_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000010001_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+10]+=P_010011001*R_000[0]+a1P_010011000_1*R_001[0]+P_010111001*R_010[0]+a1P_010111000_1*R_011[0]+a2P_010000001_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000011001_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111001_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+11]+=P_010010002*R_000[0]+a1P_010010001_2*R_001[0]+a2P_010010000_1*R_002[0]+a1P_010000002_1*R_010[0]+a2P_010000001_2*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000010002_1*R_100[0]+a2P_000010001_2*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000002_1*R_110[0]+a3P_000000001_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+12]+=P_002020000*R_000[0]+a1P_002010000_2*R_010[0]+a2P_002000000_1*R_020[0]+a1P_001020000_2*R_100[0]+a2P_001010000_4*R_110[0]+a3P_001000000_2*R_120[0]+a2P_000020000_1*R_200[0]+a3P_000010000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+13]+=P_001021000*R_000[0]+P_001121000*R_010[0]+P_001221000*R_020[0]+a3P_001000000_1*R_030[0]+a1P_000021000_1*R_100[0]+a1P_000121000_1*R_110[0]+a1P_000221000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+14]+=P_000022000*R_000[0]+P_000122000*R_010[0]+P_000222000*R_020[0]+a2P_000111000_2*R_030[0]+R_040[0];
			ans_temp[ans_id*36+15]+=P_001020001*R_000[0]+a1P_001020000_1*R_001[0]+a1P_001010001_2*R_010[0]+a2P_001010000_2*R_011[0]+a2P_001000001_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000020001_1*R_100[0]+a2P_000020000_1*R_101[0]+a2P_000010001_2*R_110[0]+a3P_000010000_2*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+16]+=P_000021001*R_000[0]+a1P_000021000_1*R_001[0]+P_000121001*R_010[0]+a1P_000121000_1*R_011[0]+P_000221001*R_020[0]+a1P_000221000_1*R_021[0]+a3P_000000001_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+17]+=P_000020002*R_000[0]+a1P_000020001_2*R_001[0]+a2P_000020000_1*R_002[0]+a1P_000010002_2*R_010[0]+a2P_000010001_4*R_011[0]+a3P_000010000_2*R_012[0]+a2P_000000002_1*R_020[0]+a3P_000000001_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+18]+=P_012000010*R_000[0]+a1P_012000000_1*R_001[0]+P_112000010*R_100[0]+a1P_112000000_1*R_101[0]+P_212000010*R_200[0]+a1P_212000000_1*R_201[0]+a3P_000000010_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+19]+=P_011001010*R_000[0]+a1P_011001000_1*R_001[0]+a1P_011000010_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111001010*R_100[0]+a1P_111001000_1*R_101[0]+a1P_111000010_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000001010_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+20]+=P_010002010*R_000[0]+a1P_010002000_1*R_001[0]+a1P_010001010_2*R_010[0]+a2P_010001000_2*R_011[0]+a2P_010000010_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000002010_1*R_100[0]+a2P_000002000_1*R_101[0]+a2P_000001010_2*R_110[0]+a3P_000001000_2*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+21]+=P_011000011*R_000[0]+P_011000111*R_001[0]+a2P_011000000_1*R_002[0]+P_111000011*R_100[0]+P_111000111*R_101[0]+a2P_111000000_1*R_102[0]+a2P_000000011_1*R_200[0]+a2P_000000111_1*R_201[0]+R_202[0];
			ans_temp[ans_id*36+22]+=P_010001011*R_000[0]+P_010001111*R_001[0]+a2P_010001000_1*R_002[0]+a1P_010000011_1*R_010[0]+a1P_010000111_1*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000001011_1*R_100[0]+a1P_000001111_1*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+23]+=P_010000012*R_000[0]+P_010000112*R_001[0]+P_010000212*R_002[0]+a3P_010000000_1*R_003[0]+a1P_000000012_1*R_100[0]+a1P_000000112_1*R_101[0]+a1P_000000212_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+24]+=P_002010010*R_000[0]+a1P_002010000_1*R_001[0]+a1P_002000010_1*R_010[0]+a2P_002000000_1*R_011[0]+a1P_001010010_2*R_100[0]+a2P_001010000_2*R_101[0]+a2P_001000010_2*R_110[0]+a3P_001000000_2*R_111[0]+a2P_000010010_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+25]+=P_001011010*R_000[0]+a1P_001011000_1*R_001[0]+P_001111010*R_010[0]+a1P_001111000_1*R_011[0]+a2P_001000010_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000011010_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111010_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+26]+=P_000012010*R_000[0]+a1P_000012000_1*R_001[0]+P_000112010*R_010[0]+a1P_000112000_1*R_011[0]+P_000212010*R_020[0]+a1P_000212000_1*R_021[0]+a3P_000000010_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+27]+=P_001010011*R_000[0]+P_001010111*R_001[0]+a2P_001010000_1*R_002[0]+a1P_001000011_1*R_010[0]+a1P_001000111_1*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000010011_1*R_100[0]+a1P_000010111_1*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+28]+=P_000011011*R_000[0]+P_000011111*R_001[0]+a2P_000011000_1*R_002[0]+P_000111011*R_010[0]+P_000111111*R_011[0]+a2P_000111000_1*R_012[0]+a2P_000000011_1*R_020[0]+a2P_000000111_1*R_021[0]+R_022[0];
			ans_temp[ans_id*36+29]+=P_000010012*R_000[0]+P_000010112*R_001[0]+P_000010212*R_002[0]+a3P_000010000_1*R_003[0]+a1P_000000012_1*R_010[0]+a1P_000000112_1*R_011[0]+a1P_000000212_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+30]+=P_002000020*R_000[0]+a1P_002000010_2*R_001[0]+a2P_002000000_1*R_002[0]+a1P_001000020_2*R_100[0]+a2P_001000010_4*R_101[0]+a3P_001000000_2*R_102[0]+a2P_000000020_1*R_200[0]+a3P_000000010_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+31]+=P_001001020*R_000[0]+a1P_001001010_2*R_001[0]+a2P_001001000_1*R_002[0]+a1P_001000020_1*R_010[0]+a2P_001000010_2*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000001020_1*R_100[0]+a2P_000001010_2*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000020_1*R_110[0]+a3P_000000010_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+32]+=P_000002020*R_000[0]+a1P_000002010_2*R_001[0]+a2P_000002000_1*R_002[0]+a1P_000001020_2*R_010[0]+a2P_000001010_4*R_011[0]+a3P_000001000_2*R_012[0]+a2P_000000020_1*R_020[0]+a3P_000000010_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+33]+=P_001000021*R_000[0]+P_001000121*R_001[0]+P_001000221*R_002[0]+a3P_001000000_1*R_003[0]+a1P_000000021_1*R_100[0]+a1P_000000121_1*R_101[0]+a1P_000000221_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+34]+=P_000001021*R_000[0]+P_000001121*R_001[0]+P_000001221*R_002[0]+a3P_000001000_1*R_003[0]+a1P_000000021_1*R_010[0]+a1P_000000121_1*R_011[0]+a1P_000000221_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+35]+=P_000000022*R_000[0]+P_000000122*R_001[0]+P_000000222*R_002[0]+a2P_000000111_2*R_003[0]+R_004[0];
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
__global__ void TSMJ_ddss_NTX(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
				double Eta=Eta_in[jj];
				double pq=pq_in[jj];
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            Pmtrx[p_i]=Pmtrx_in[p_jj+p_i];
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
	R_000[4]*=aPin4;
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
	for(int i=1;i<4;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<3;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_001[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_200[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_020[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_002[i]*=aPin1;
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
		double PD_002[3];
		double PD_102[3];
		double PD_011[3];
		double PD_111[3];
		double PD_012[3];
		double PD_112[3];
		double PD_212[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		double PD_022[3];
		double PD_122[3];
		double PD_222[3];
		for(int i=0;i<3;i++){
			PD_002[i]=aPin1+PD_001[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_102[i]=(2.000000*PD_001[i]);
			}
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_012[i]=PD_111[i]+PD_001[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_112[i]=(PD_002[i]+2.000000*PD_011[i]);
			}
		for(int i=0;i<3;i++){
			PD_212[i]=(0.500000*PD_102[i]+PD_111[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
			}
		for(int i=0;i<3;i++){
			PD_022[i]=PD_112[i]+PD_010[i]*PD_012[i];
			}
		for(int i=0;i<3;i++){
			PD_122[i]=2.000000*(PD_012[i]+PD_021[i]);
			}
		for(int i=0;i<3;i++){
			PD_222[i]=(PD_112[i]+PD_121[i]);
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
			P_022000000=PD_022[0];
			P_122000000=PD_122[0];
			P_222000000=PD_222[0];
			P_021001000=PD_021[0]*PD_001[1];
			P_121001000=PD_121[0]*PD_001[1];
			P_221001000=PD_221[0]*PD_001[1];
			P_020002000=PD_020[0]*PD_002[1];
			P_021000001=PD_021[0]*PD_001[2];
			P_121000001=PD_121[0]*PD_001[2];
			P_221000001=PD_221[0]*PD_001[2];
			P_020001001=PD_020[0]*PD_001[1]*PD_001[2];
			P_020000002=PD_020[0]*PD_002[2];
			P_012010000=PD_012[0]*PD_010[1];
			P_112010000=PD_112[0]*PD_010[1];
			P_212010000=PD_212[0]*PD_010[1];
			P_011011000=PD_011[0]*PD_011[1];
			P_011111000=PD_011[0]*PD_111[1];
			P_111011000=PD_111[0]*PD_011[1];
			P_111111000=PD_111[0]*PD_111[1];
			P_010012000=PD_010[0]*PD_012[1];
			P_010112000=PD_010[0]*PD_112[1];
			P_010212000=PD_010[0]*PD_212[1];
			P_011010001=PD_011[0]*PD_010[1]*PD_001[2];
			P_111010001=PD_111[0]*PD_010[1]*PD_001[2];
			P_010011001=PD_010[0]*PD_011[1]*PD_001[2];
			P_010111001=PD_010[0]*PD_111[1]*PD_001[2];
			P_010010002=PD_010[0]*PD_010[1]*PD_002[2];
			P_002020000=PD_002[0]*PD_020[1];
			P_001021000=PD_001[0]*PD_021[1];
			P_001121000=PD_001[0]*PD_121[1];
			P_001221000=PD_001[0]*PD_221[1];
			P_000022000=PD_022[1];
			P_000122000=PD_122[1];
			P_000222000=PD_222[1];
			P_001020001=PD_001[0]*PD_020[1]*PD_001[2];
			P_000021001=PD_021[1]*PD_001[2];
			P_000121001=PD_121[1]*PD_001[2];
			P_000221001=PD_221[1]*PD_001[2];
			P_000020002=PD_020[1]*PD_002[2];
			P_012000010=PD_012[0]*PD_010[2];
			P_112000010=PD_112[0]*PD_010[2];
			P_212000010=PD_212[0]*PD_010[2];
			P_011001010=PD_011[0]*PD_001[1]*PD_010[2];
			P_111001010=PD_111[0]*PD_001[1]*PD_010[2];
			P_010002010=PD_010[0]*PD_002[1]*PD_010[2];
			P_011000011=PD_011[0]*PD_011[2];
			P_011000111=PD_011[0]*PD_111[2];
			P_111000011=PD_111[0]*PD_011[2];
			P_111000111=PD_111[0]*PD_111[2];
			P_010001011=PD_010[0]*PD_001[1]*PD_011[2];
			P_010001111=PD_010[0]*PD_001[1]*PD_111[2];
			P_010000012=PD_010[0]*PD_012[2];
			P_010000112=PD_010[0]*PD_112[2];
			P_010000212=PD_010[0]*PD_212[2];
			P_002010010=PD_002[0]*PD_010[1]*PD_010[2];
			P_001011010=PD_001[0]*PD_011[1]*PD_010[2];
			P_001111010=PD_001[0]*PD_111[1]*PD_010[2];
			P_000012010=PD_012[1]*PD_010[2];
			P_000112010=PD_112[1]*PD_010[2];
			P_000212010=PD_212[1]*PD_010[2];
			P_001010011=PD_001[0]*PD_010[1]*PD_011[2];
			P_001010111=PD_001[0]*PD_010[1]*PD_111[2];
			P_000011011=PD_011[1]*PD_011[2];
			P_000011111=PD_011[1]*PD_111[2];
			P_000111011=PD_111[1]*PD_011[2];
			P_000111111=PD_111[1]*PD_111[2];
			P_000010012=PD_010[1]*PD_012[2];
			P_000010112=PD_010[1]*PD_112[2];
			P_000010212=PD_010[1]*PD_212[2];
			P_002000020=PD_002[0]*PD_020[2];
			P_001001020=PD_001[0]*PD_001[1]*PD_020[2];
			P_000002020=PD_002[1]*PD_020[2];
			P_001000021=PD_001[0]*PD_021[2];
			P_001000121=PD_001[0]*PD_121[2];
			P_001000221=PD_001[0]*PD_221[2];
			P_000001021=PD_001[1]*PD_021[2];
			P_000001121=PD_001[1]*PD_121[2];
			P_000001221=PD_001[1]*PD_221[2];
			P_000000022=PD_022[2];
			P_000000122=PD_122[2];
			P_000000222=PD_222[2];
			a2P_111000000_1=PD_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=PD_021[0];
			a1P_121000000_1=PD_121[0];
			a1P_221000000_1=PD_221[0];
			a3P_000001000_1=PD_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=PD_020[0]*PD_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=PD_020[0];
			a1P_010002000_1=PD_010[0]*PD_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=PD_010[0]*PD_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=PD_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=PD_002[1];
			a3P_000000001_1=PD_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=PD_020[0]*PD_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=PD_010[0]*PD_001[1]*PD_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=PD_010[0]*PD_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=PD_001[1]*PD_001[2];
			a1P_010000002_1=PD_010[0]*PD_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=PD_002[2];
			a1P_012000000_1=PD_012[0];
			a1P_112000000_1=PD_112[0];
			a1P_212000000_1=PD_212[0];
			a3P_000010000_1=PD_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=PD_011[0];
			a2P_000011000_1=PD_011[1];
			a2P_000111000_1=PD_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=PD_012[1];
			a1P_000112000_1=PD_112[1];
			a1P_000212000_1=PD_212[1];
			a1P_011010000_1=PD_011[0]*PD_010[1];
			a1P_011000001_1=PD_011[0]*PD_001[2];
			a1P_111010000_1=PD_111[0]*PD_010[1];
			a1P_111000001_1=PD_111[0]*PD_001[2];
			a2P_000010001_1=PD_010[1]*PD_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=PD_010[0]*PD_011[1];
			a1P_010111000_1=PD_010[0]*PD_111[1];
			a1P_000011001_1=PD_011[1]*PD_001[2];
			a1P_000111001_1=PD_111[1]*PD_001[2];
			a1P_010010001_1=PD_010[0]*PD_010[1]*PD_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010002_1=PD_010[1]*PD_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=PD_002[0]*PD_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=PD_002[0];
			a1P_001020000_1=PD_001[0]*PD_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=PD_001[0]*PD_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=PD_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=PD_020[1];
			a1P_000021000_1=PD_021[1];
			a1P_000121000_1=PD_121[1];
			a1P_000221000_1=PD_221[1];
			a1P_001010001_1=PD_001[0]*PD_010[1]*PD_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=PD_001[0]*PD_001[2];
			a1P_000020001_1=PD_020[1]*PD_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=PD_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=PD_011[0]*PD_001[1];
			a1P_011000010_1=PD_011[0]*PD_010[2];
			a1P_111001000_1=PD_111[0]*PD_001[1];
			a1P_111000010_1=PD_111[0]*PD_010[2];
			a2P_000001010_1=PD_001[1]*PD_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=PD_010[0]*PD_001[1]*PD_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000002010_1=PD_002[1]*PD_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=PD_011[2];
			a2P_000000111_1=PD_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=PD_010[0]*PD_011[2];
			a1P_010000111_1=PD_010[0]*PD_111[2];
			a1P_000001011_1=PD_001[1]*PD_011[2];
			a1P_000001111_1=PD_001[1]*PD_111[2];
			a1P_000000012_1=PD_012[2];
			a1P_000000112_1=PD_112[2];
			a1P_000000212_1=PD_212[2];
			a1P_002000010_1=PD_002[0]*PD_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=PD_001[0]*PD_010[1]*PD_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=PD_001[0]*PD_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=PD_010[1]*PD_010[2];
			a1P_001011000_1=PD_001[0]*PD_011[1];
			a1P_001111000_1=PD_001[0]*PD_111[1];
			a1P_000011010_1=PD_011[1]*PD_010[2];
			a1P_000111010_1=PD_111[1]*PD_010[2];
			a1P_001000011_1=PD_001[0]*PD_011[2];
			a1P_001000111_1=PD_001[0]*PD_111[2];
			a1P_000010011_1=PD_010[1]*PD_011[2];
			a1P_000010111_1=PD_010[1]*PD_111[2];
			a1P_001000020_1=PD_001[0]*PD_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=PD_020[2];
			a1P_001001010_1=PD_001[0]*PD_001[1]*PD_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=PD_001[0]*PD_001[1];
			a1P_000001020_1=PD_001[1]*PD_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=PD_021[2];
			a1P_000000121_1=PD_121[2];
			a1P_000000221_1=PD_221[2];
			ans_temp[ans_id*36+0]+=P_022000000*R_000[0]+P_122000000*R_100[0]+P_222000000*R_200[0]+a2P_111000000_2*R_300[0]+R_400[0];
			ans_temp[ans_id*36+1]+=P_021001000*R_000[0]+a1P_021000000_1*R_010[0]+P_121001000*R_100[0]+a1P_121000000_1*R_110[0]+P_221001000*R_200[0]+a1P_221000000_1*R_210[0]+a3P_000001000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+2]+=P_020002000*R_000[0]+a1P_020001000_2*R_010[0]+a2P_020000000_1*R_020[0]+a1P_010002000_2*R_100[0]+a2P_010001000_4*R_110[0]+a3P_010000000_2*R_120[0]+a2P_000002000_1*R_200[0]+a3P_000001000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+3]+=P_021000001*R_000[0]+a1P_021000000_1*R_001[0]+P_121000001*R_100[0]+a1P_121000000_1*R_101[0]+P_221000001*R_200[0]+a1P_221000000_1*R_201[0]+a3P_000000001_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+4]+=P_020001001*R_000[0]+a1P_020001000_1*R_001[0]+a1P_020000001_1*R_010[0]+a2P_020000000_1*R_011[0]+a1P_010001001_2*R_100[0]+a2P_010001000_2*R_101[0]+a2P_010000001_2*R_110[0]+a3P_010000000_2*R_111[0]+a2P_000001001_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+5]+=P_020000002*R_000[0]+a1P_020000001_2*R_001[0]+a2P_020000000_1*R_002[0]+a1P_010000002_2*R_100[0]+a2P_010000001_4*R_101[0]+a3P_010000000_2*R_102[0]+a2P_000000002_1*R_200[0]+a3P_000000001_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+6]+=P_012010000*R_000[0]+a1P_012000000_1*R_010[0]+P_112010000*R_100[0]+a1P_112000000_1*R_110[0]+P_212010000*R_200[0]+a1P_212000000_1*R_210[0]+a3P_000010000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+7]+=P_011011000*R_000[0]+P_011111000*R_010[0]+a2P_011000000_1*R_020[0]+P_111011000*R_100[0]+P_111111000*R_110[0]+a2P_111000000_1*R_120[0]+a2P_000011000_1*R_200[0]+a2P_000111000_1*R_210[0]+R_220[0];
			ans_temp[ans_id*36+8]+=P_010012000*R_000[0]+P_010112000*R_010[0]+P_010212000*R_020[0]+a3P_010000000_1*R_030[0]+a1P_000012000_1*R_100[0]+a1P_000112000_1*R_110[0]+a1P_000212000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+9]+=P_011010001*R_000[0]+a1P_011010000_1*R_001[0]+a1P_011000001_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111010001*R_100[0]+a1P_111010000_1*R_101[0]+a1P_111000001_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000010001_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+10]+=P_010011001*R_000[0]+a1P_010011000_1*R_001[0]+P_010111001*R_010[0]+a1P_010111000_1*R_011[0]+a2P_010000001_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000011001_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111001_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+11]+=P_010010002*R_000[0]+a1P_010010001_2*R_001[0]+a2P_010010000_1*R_002[0]+a1P_010000002_1*R_010[0]+a2P_010000001_2*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000010002_1*R_100[0]+a2P_000010001_2*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000002_1*R_110[0]+a3P_000000001_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+12]+=P_002020000*R_000[0]+a1P_002010000_2*R_010[0]+a2P_002000000_1*R_020[0]+a1P_001020000_2*R_100[0]+a2P_001010000_4*R_110[0]+a3P_001000000_2*R_120[0]+a2P_000020000_1*R_200[0]+a3P_000010000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+13]+=P_001021000*R_000[0]+P_001121000*R_010[0]+P_001221000*R_020[0]+a3P_001000000_1*R_030[0]+a1P_000021000_1*R_100[0]+a1P_000121000_1*R_110[0]+a1P_000221000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+14]+=P_000022000*R_000[0]+P_000122000*R_010[0]+P_000222000*R_020[0]+a2P_000111000_2*R_030[0]+R_040[0];
			ans_temp[ans_id*36+15]+=P_001020001*R_000[0]+a1P_001020000_1*R_001[0]+a1P_001010001_2*R_010[0]+a2P_001010000_2*R_011[0]+a2P_001000001_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000020001_1*R_100[0]+a2P_000020000_1*R_101[0]+a2P_000010001_2*R_110[0]+a3P_000010000_2*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+16]+=P_000021001*R_000[0]+a1P_000021000_1*R_001[0]+P_000121001*R_010[0]+a1P_000121000_1*R_011[0]+P_000221001*R_020[0]+a1P_000221000_1*R_021[0]+a3P_000000001_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+17]+=P_000020002*R_000[0]+a1P_000020001_2*R_001[0]+a2P_000020000_1*R_002[0]+a1P_000010002_2*R_010[0]+a2P_000010001_4*R_011[0]+a3P_000010000_2*R_012[0]+a2P_000000002_1*R_020[0]+a3P_000000001_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+18]+=P_012000010*R_000[0]+a1P_012000000_1*R_001[0]+P_112000010*R_100[0]+a1P_112000000_1*R_101[0]+P_212000010*R_200[0]+a1P_212000000_1*R_201[0]+a3P_000000010_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+19]+=P_011001010*R_000[0]+a1P_011001000_1*R_001[0]+a1P_011000010_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111001010*R_100[0]+a1P_111001000_1*R_101[0]+a1P_111000010_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000001010_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+20]+=P_010002010*R_000[0]+a1P_010002000_1*R_001[0]+a1P_010001010_2*R_010[0]+a2P_010001000_2*R_011[0]+a2P_010000010_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000002010_1*R_100[0]+a2P_000002000_1*R_101[0]+a2P_000001010_2*R_110[0]+a3P_000001000_2*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+21]+=P_011000011*R_000[0]+P_011000111*R_001[0]+a2P_011000000_1*R_002[0]+P_111000011*R_100[0]+P_111000111*R_101[0]+a2P_111000000_1*R_102[0]+a2P_000000011_1*R_200[0]+a2P_000000111_1*R_201[0]+R_202[0];
			ans_temp[ans_id*36+22]+=P_010001011*R_000[0]+P_010001111*R_001[0]+a2P_010001000_1*R_002[0]+a1P_010000011_1*R_010[0]+a1P_010000111_1*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000001011_1*R_100[0]+a1P_000001111_1*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+23]+=P_010000012*R_000[0]+P_010000112*R_001[0]+P_010000212*R_002[0]+a3P_010000000_1*R_003[0]+a1P_000000012_1*R_100[0]+a1P_000000112_1*R_101[0]+a1P_000000212_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+24]+=P_002010010*R_000[0]+a1P_002010000_1*R_001[0]+a1P_002000010_1*R_010[0]+a2P_002000000_1*R_011[0]+a1P_001010010_2*R_100[0]+a2P_001010000_2*R_101[0]+a2P_001000010_2*R_110[0]+a3P_001000000_2*R_111[0]+a2P_000010010_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+25]+=P_001011010*R_000[0]+a1P_001011000_1*R_001[0]+P_001111010*R_010[0]+a1P_001111000_1*R_011[0]+a2P_001000010_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000011010_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111010_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+26]+=P_000012010*R_000[0]+a1P_000012000_1*R_001[0]+P_000112010*R_010[0]+a1P_000112000_1*R_011[0]+P_000212010*R_020[0]+a1P_000212000_1*R_021[0]+a3P_000000010_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+27]+=P_001010011*R_000[0]+P_001010111*R_001[0]+a2P_001010000_1*R_002[0]+a1P_001000011_1*R_010[0]+a1P_001000111_1*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000010011_1*R_100[0]+a1P_000010111_1*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+28]+=P_000011011*R_000[0]+P_000011111*R_001[0]+a2P_000011000_1*R_002[0]+P_000111011*R_010[0]+P_000111111*R_011[0]+a2P_000111000_1*R_012[0]+a2P_000000011_1*R_020[0]+a2P_000000111_1*R_021[0]+R_022[0];
			ans_temp[ans_id*36+29]+=P_000010012*R_000[0]+P_000010112*R_001[0]+P_000010212*R_002[0]+a3P_000010000_1*R_003[0]+a1P_000000012_1*R_010[0]+a1P_000000112_1*R_011[0]+a1P_000000212_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+30]+=P_002000020*R_000[0]+a1P_002000010_2*R_001[0]+a2P_002000000_1*R_002[0]+a1P_001000020_2*R_100[0]+a2P_001000010_4*R_101[0]+a3P_001000000_2*R_102[0]+a2P_000000020_1*R_200[0]+a3P_000000010_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+31]+=P_001001020*R_000[0]+a1P_001001010_2*R_001[0]+a2P_001001000_1*R_002[0]+a1P_001000020_1*R_010[0]+a2P_001000010_2*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000001020_1*R_100[0]+a2P_000001010_2*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000020_1*R_110[0]+a3P_000000010_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+32]+=P_000002020*R_000[0]+a1P_000002010_2*R_001[0]+a2P_000002000_1*R_002[0]+a1P_000001020_2*R_010[0]+a2P_000001010_4*R_011[0]+a3P_000001000_2*R_012[0]+a2P_000000020_1*R_020[0]+a3P_000000010_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+33]+=P_001000021*R_000[0]+P_001000121*R_001[0]+P_001000221*R_002[0]+a3P_001000000_1*R_003[0]+a1P_000000021_1*R_100[0]+a1P_000000121_1*R_101[0]+a1P_000000221_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+34]+=P_000001021*R_000[0]+P_000001121*R_001[0]+P_000001221*R_002[0]+a3P_000001000_1*R_003[0]+a1P_000000021_1*R_010[0]+a1P_000000121_1*R_011[0]+a1P_000000221_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+35]+=P_000000022*R_000[0]+P_000000122*R_001[0]+P_000000222*R_002[0]+a2P_000000111_2*R_003[0]+R_004[0];
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
__global__ void TSMJ_ddss_fs(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
	R_000[4]*=aPin4;
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
	for(int i=1;i<4;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<3;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_001[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_200[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_020[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_002[i]*=aPin1;
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
		double PD_002[3];
		double PD_102[3];
		double PD_011[3];
		double PD_111[3];
		double PD_012[3];
		double PD_112[3];
		double PD_212[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		double PD_022[3];
		double PD_122[3];
		double PD_222[3];
		for(int i=0;i<3;i++){
			PD_002[i]=aPin1+PD_001[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_102[i]=(2.000000*PD_001[i]);
			}
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_012[i]=PD_111[i]+PD_001[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_112[i]=(PD_002[i]+2.000000*PD_011[i]);
			}
		for(int i=0;i<3;i++){
			PD_212[i]=(0.500000*PD_102[i]+PD_111[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
			}
		for(int i=0;i<3;i++){
			PD_022[i]=PD_112[i]+PD_010[i]*PD_012[i];
			}
		for(int i=0;i<3;i++){
			PD_122[i]=2.000000*(PD_012[i]+PD_021[i]);
			}
		for(int i=0;i<3;i++){
			PD_222[i]=(PD_112[i]+PD_121[i]);
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
			P_022000000=PD_022[0];
			P_122000000=PD_122[0];
			P_222000000=PD_222[0];
			P_021001000=PD_021[0]*PD_001[1];
			P_121001000=PD_121[0]*PD_001[1];
			P_221001000=PD_221[0]*PD_001[1];
			P_020002000=PD_020[0]*PD_002[1];
			P_021000001=PD_021[0]*PD_001[2];
			P_121000001=PD_121[0]*PD_001[2];
			P_221000001=PD_221[0]*PD_001[2];
			P_020001001=PD_020[0]*PD_001[1]*PD_001[2];
			P_020000002=PD_020[0]*PD_002[2];
			P_012010000=PD_012[0]*PD_010[1];
			P_112010000=PD_112[0]*PD_010[1];
			P_212010000=PD_212[0]*PD_010[1];
			P_011011000=PD_011[0]*PD_011[1];
			P_011111000=PD_011[0]*PD_111[1];
			P_111011000=PD_111[0]*PD_011[1];
			P_111111000=PD_111[0]*PD_111[1];
			P_010012000=PD_010[0]*PD_012[1];
			P_010112000=PD_010[0]*PD_112[1];
			P_010212000=PD_010[0]*PD_212[1];
			P_011010001=PD_011[0]*PD_010[1]*PD_001[2];
			P_111010001=PD_111[0]*PD_010[1]*PD_001[2];
			P_010011001=PD_010[0]*PD_011[1]*PD_001[2];
			P_010111001=PD_010[0]*PD_111[1]*PD_001[2];
			P_010010002=PD_010[0]*PD_010[1]*PD_002[2];
			P_002020000=PD_002[0]*PD_020[1];
			P_001021000=PD_001[0]*PD_021[1];
			P_001121000=PD_001[0]*PD_121[1];
			P_001221000=PD_001[0]*PD_221[1];
			P_000022000=PD_022[1];
			P_000122000=PD_122[1];
			P_000222000=PD_222[1];
			P_001020001=PD_001[0]*PD_020[1]*PD_001[2];
			P_000021001=PD_021[1]*PD_001[2];
			P_000121001=PD_121[1]*PD_001[2];
			P_000221001=PD_221[1]*PD_001[2];
			P_000020002=PD_020[1]*PD_002[2];
			P_012000010=PD_012[0]*PD_010[2];
			P_112000010=PD_112[0]*PD_010[2];
			P_212000010=PD_212[0]*PD_010[2];
			P_011001010=PD_011[0]*PD_001[1]*PD_010[2];
			P_111001010=PD_111[0]*PD_001[1]*PD_010[2];
			P_010002010=PD_010[0]*PD_002[1]*PD_010[2];
			P_011000011=PD_011[0]*PD_011[2];
			P_011000111=PD_011[0]*PD_111[2];
			P_111000011=PD_111[0]*PD_011[2];
			P_111000111=PD_111[0]*PD_111[2];
			P_010001011=PD_010[0]*PD_001[1]*PD_011[2];
			P_010001111=PD_010[0]*PD_001[1]*PD_111[2];
			P_010000012=PD_010[0]*PD_012[2];
			P_010000112=PD_010[0]*PD_112[2];
			P_010000212=PD_010[0]*PD_212[2];
			P_002010010=PD_002[0]*PD_010[1]*PD_010[2];
			P_001011010=PD_001[0]*PD_011[1]*PD_010[2];
			P_001111010=PD_001[0]*PD_111[1]*PD_010[2];
			P_000012010=PD_012[1]*PD_010[2];
			P_000112010=PD_112[1]*PD_010[2];
			P_000212010=PD_212[1]*PD_010[2];
			P_001010011=PD_001[0]*PD_010[1]*PD_011[2];
			P_001010111=PD_001[0]*PD_010[1]*PD_111[2];
			P_000011011=PD_011[1]*PD_011[2];
			P_000011111=PD_011[1]*PD_111[2];
			P_000111011=PD_111[1]*PD_011[2];
			P_000111111=PD_111[1]*PD_111[2];
			P_000010012=PD_010[1]*PD_012[2];
			P_000010112=PD_010[1]*PD_112[2];
			P_000010212=PD_010[1]*PD_212[2];
			P_002000020=PD_002[0]*PD_020[2];
			P_001001020=PD_001[0]*PD_001[1]*PD_020[2];
			P_000002020=PD_002[1]*PD_020[2];
			P_001000021=PD_001[0]*PD_021[2];
			P_001000121=PD_001[0]*PD_121[2];
			P_001000221=PD_001[0]*PD_221[2];
			P_000001021=PD_001[1]*PD_021[2];
			P_000001121=PD_001[1]*PD_121[2];
			P_000001221=PD_001[1]*PD_221[2];
			P_000000022=PD_022[2];
			P_000000122=PD_122[2];
			P_000000222=PD_222[2];
			a2P_111000000_1=PD_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=PD_021[0];
			a1P_121000000_1=PD_121[0];
			a1P_221000000_1=PD_221[0];
			a3P_000001000_1=PD_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=PD_020[0]*PD_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=PD_020[0];
			a1P_010002000_1=PD_010[0]*PD_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=PD_010[0]*PD_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=PD_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=PD_002[1];
			a3P_000000001_1=PD_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=PD_020[0]*PD_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=PD_010[0]*PD_001[1]*PD_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=PD_010[0]*PD_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=PD_001[1]*PD_001[2];
			a1P_010000002_1=PD_010[0]*PD_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=PD_002[2];
			a1P_012000000_1=PD_012[0];
			a1P_112000000_1=PD_112[0];
			a1P_212000000_1=PD_212[0];
			a3P_000010000_1=PD_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=PD_011[0];
			a2P_000011000_1=PD_011[1];
			a2P_000111000_1=PD_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=PD_012[1];
			a1P_000112000_1=PD_112[1];
			a1P_000212000_1=PD_212[1];
			a1P_011010000_1=PD_011[0]*PD_010[1];
			a1P_011000001_1=PD_011[0]*PD_001[2];
			a1P_111010000_1=PD_111[0]*PD_010[1];
			a1P_111000001_1=PD_111[0]*PD_001[2];
			a2P_000010001_1=PD_010[1]*PD_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=PD_010[0]*PD_011[1];
			a1P_010111000_1=PD_010[0]*PD_111[1];
			a1P_000011001_1=PD_011[1]*PD_001[2];
			a1P_000111001_1=PD_111[1]*PD_001[2];
			a1P_010010001_1=PD_010[0]*PD_010[1]*PD_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010002_1=PD_010[1]*PD_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=PD_002[0]*PD_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=PD_002[0];
			a1P_001020000_1=PD_001[0]*PD_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=PD_001[0]*PD_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=PD_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=PD_020[1];
			a1P_000021000_1=PD_021[1];
			a1P_000121000_1=PD_121[1];
			a1P_000221000_1=PD_221[1];
			a1P_001010001_1=PD_001[0]*PD_010[1]*PD_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=PD_001[0]*PD_001[2];
			a1P_000020001_1=PD_020[1]*PD_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=PD_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=PD_011[0]*PD_001[1];
			a1P_011000010_1=PD_011[0]*PD_010[2];
			a1P_111001000_1=PD_111[0]*PD_001[1];
			a1P_111000010_1=PD_111[0]*PD_010[2];
			a2P_000001010_1=PD_001[1]*PD_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=PD_010[0]*PD_001[1]*PD_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000002010_1=PD_002[1]*PD_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=PD_011[2];
			a2P_000000111_1=PD_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=PD_010[0]*PD_011[2];
			a1P_010000111_1=PD_010[0]*PD_111[2];
			a1P_000001011_1=PD_001[1]*PD_011[2];
			a1P_000001111_1=PD_001[1]*PD_111[2];
			a1P_000000012_1=PD_012[2];
			a1P_000000112_1=PD_112[2];
			a1P_000000212_1=PD_212[2];
			a1P_002000010_1=PD_002[0]*PD_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=PD_001[0]*PD_010[1]*PD_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=PD_001[0]*PD_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=PD_010[1]*PD_010[2];
			a1P_001011000_1=PD_001[0]*PD_011[1];
			a1P_001111000_1=PD_001[0]*PD_111[1];
			a1P_000011010_1=PD_011[1]*PD_010[2];
			a1P_000111010_1=PD_111[1]*PD_010[2];
			a1P_001000011_1=PD_001[0]*PD_011[2];
			a1P_001000111_1=PD_001[0]*PD_111[2];
			a1P_000010011_1=PD_010[1]*PD_011[2];
			a1P_000010111_1=PD_010[1]*PD_111[2];
			a1P_001000020_1=PD_001[0]*PD_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=PD_020[2];
			a1P_001001010_1=PD_001[0]*PD_001[1]*PD_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=PD_001[0]*PD_001[1];
			a1P_000001020_1=PD_001[1]*PD_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=PD_021[2];
			a1P_000000121_1=PD_121[2];
			a1P_000000221_1=PD_221[2];
			ans_temp[ans_id*36+0]+=P_022000000*R_000[0]+P_122000000*R_100[0]+P_222000000*R_200[0]+a2P_111000000_2*R_300[0]+R_400[0];
			ans_temp[ans_id*36+1]+=P_021001000*R_000[0]+a1P_021000000_1*R_010[0]+P_121001000*R_100[0]+a1P_121000000_1*R_110[0]+P_221001000*R_200[0]+a1P_221000000_1*R_210[0]+a3P_000001000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+2]+=P_020002000*R_000[0]+a1P_020001000_2*R_010[0]+a2P_020000000_1*R_020[0]+a1P_010002000_2*R_100[0]+a2P_010001000_4*R_110[0]+a3P_010000000_2*R_120[0]+a2P_000002000_1*R_200[0]+a3P_000001000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+3]+=P_021000001*R_000[0]+a1P_021000000_1*R_001[0]+P_121000001*R_100[0]+a1P_121000000_1*R_101[0]+P_221000001*R_200[0]+a1P_221000000_1*R_201[0]+a3P_000000001_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+4]+=P_020001001*R_000[0]+a1P_020001000_1*R_001[0]+a1P_020000001_1*R_010[0]+a2P_020000000_1*R_011[0]+a1P_010001001_2*R_100[0]+a2P_010001000_2*R_101[0]+a2P_010000001_2*R_110[0]+a3P_010000000_2*R_111[0]+a2P_000001001_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+5]+=P_020000002*R_000[0]+a1P_020000001_2*R_001[0]+a2P_020000000_1*R_002[0]+a1P_010000002_2*R_100[0]+a2P_010000001_4*R_101[0]+a3P_010000000_2*R_102[0]+a2P_000000002_1*R_200[0]+a3P_000000001_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+6]+=P_012010000*R_000[0]+a1P_012000000_1*R_010[0]+P_112010000*R_100[0]+a1P_112000000_1*R_110[0]+P_212010000*R_200[0]+a1P_212000000_1*R_210[0]+a3P_000010000_1*R_300[0]+R_310[0];
			ans_temp[ans_id*36+7]+=P_011011000*R_000[0]+P_011111000*R_010[0]+a2P_011000000_1*R_020[0]+P_111011000*R_100[0]+P_111111000*R_110[0]+a2P_111000000_1*R_120[0]+a2P_000011000_1*R_200[0]+a2P_000111000_1*R_210[0]+R_220[0];
			ans_temp[ans_id*36+8]+=P_010012000*R_000[0]+P_010112000*R_010[0]+P_010212000*R_020[0]+a3P_010000000_1*R_030[0]+a1P_000012000_1*R_100[0]+a1P_000112000_1*R_110[0]+a1P_000212000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+9]+=P_011010001*R_000[0]+a1P_011010000_1*R_001[0]+a1P_011000001_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111010001*R_100[0]+a1P_111010000_1*R_101[0]+a1P_111000001_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000010001_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000001_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+10]+=P_010011001*R_000[0]+a1P_010011000_1*R_001[0]+P_010111001*R_010[0]+a1P_010111000_1*R_011[0]+a2P_010000001_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000011001_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111001_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+11]+=P_010010002*R_000[0]+a1P_010010001_2*R_001[0]+a2P_010010000_1*R_002[0]+a1P_010000002_1*R_010[0]+a2P_010000001_2*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000010002_1*R_100[0]+a2P_000010001_2*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000002_1*R_110[0]+a3P_000000001_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+12]+=P_002020000*R_000[0]+a1P_002010000_2*R_010[0]+a2P_002000000_1*R_020[0]+a1P_001020000_2*R_100[0]+a2P_001010000_4*R_110[0]+a3P_001000000_2*R_120[0]+a2P_000020000_1*R_200[0]+a3P_000010000_2*R_210[0]+R_220[0];
			ans_temp[ans_id*36+13]+=P_001021000*R_000[0]+P_001121000*R_010[0]+P_001221000*R_020[0]+a3P_001000000_1*R_030[0]+a1P_000021000_1*R_100[0]+a1P_000121000_1*R_110[0]+a1P_000221000_1*R_120[0]+R_130[0];
			ans_temp[ans_id*36+14]+=P_000022000*R_000[0]+P_000122000*R_010[0]+P_000222000*R_020[0]+a2P_000111000_2*R_030[0]+R_040[0];
			ans_temp[ans_id*36+15]+=P_001020001*R_000[0]+a1P_001020000_1*R_001[0]+a1P_001010001_2*R_010[0]+a2P_001010000_2*R_011[0]+a2P_001000001_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000020001_1*R_100[0]+a2P_000020000_1*R_101[0]+a2P_000010001_2*R_110[0]+a3P_000010000_2*R_111[0]+a3P_000000001_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+16]+=P_000021001*R_000[0]+a1P_000021000_1*R_001[0]+P_000121001*R_010[0]+a1P_000121000_1*R_011[0]+P_000221001*R_020[0]+a1P_000221000_1*R_021[0]+a3P_000000001_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+17]+=P_000020002*R_000[0]+a1P_000020001_2*R_001[0]+a2P_000020000_1*R_002[0]+a1P_000010002_2*R_010[0]+a2P_000010001_4*R_011[0]+a3P_000010000_2*R_012[0]+a2P_000000002_1*R_020[0]+a3P_000000001_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+18]+=P_012000010*R_000[0]+a1P_012000000_1*R_001[0]+P_112000010*R_100[0]+a1P_112000000_1*R_101[0]+P_212000010*R_200[0]+a1P_212000000_1*R_201[0]+a3P_000000010_1*R_300[0]+R_301[0];
			ans_temp[ans_id*36+19]+=P_011001010*R_000[0]+a1P_011001000_1*R_001[0]+a1P_011000010_1*R_010[0]+a2P_011000000_1*R_011[0]+P_111001010*R_100[0]+a1P_111001000_1*R_101[0]+a1P_111000010_1*R_110[0]+a2P_111000000_1*R_111[0]+a2P_000001010_1*R_200[0]+a3P_000001000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+20]+=P_010002010*R_000[0]+a1P_010002000_1*R_001[0]+a1P_010001010_2*R_010[0]+a2P_010001000_2*R_011[0]+a2P_010000010_1*R_020[0]+a3P_010000000_1*R_021[0]+a1P_000002010_1*R_100[0]+a2P_000002000_1*R_101[0]+a2P_000001010_2*R_110[0]+a3P_000001000_2*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+21]+=P_011000011*R_000[0]+P_011000111*R_001[0]+a2P_011000000_1*R_002[0]+P_111000011*R_100[0]+P_111000111*R_101[0]+a2P_111000000_1*R_102[0]+a2P_000000011_1*R_200[0]+a2P_000000111_1*R_201[0]+R_202[0];
			ans_temp[ans_id*36+22]+=P_010001011*R_000[0]+P_010001111*R_001[0]+a2P_010001000_1*R_002[0]+a1P_010000011_1*R_010[0]+a1P_010000111_1*R_011[0]+a3P_010000000_1*R_012[0]+a1P_000001011_1*R_100[0]+a1P_000001111_1*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+23]+=P_010000012*R_000[0]+P_010000112*R_001[0]+P_010000212*R_002[0]+a3P_010000000_1*R_003[0]+a1P_000000012_1*R_100[0]+a1P_000000112_1*R_101[0]+a1P_000000212_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+24]+=P_002010010*R_000[0]+a1P_002010000_1*R_001[0]+a1P_002000010_1*R_010[0]+a2P_002000000_1*R_011[0]+a1P_001010010_2*R_100[0]+a2P_001010000_2*R_101[0]+a2P_001000010_2*R_110[0]+a3P_001000000_2*R_111[0]+a2P_000010010_1*R_200[0]+a3P_000010000_1*R_201[0]+a3P_000000010_1*R_210[0]+R_211[0];
			ans_temp[ans_id*36+25]+=P_001011010*R_000[0]+a1P_001011000_1*R_001[0]+P_001111010*R_010[0]+a1P_001111000_1*R_011[0]+a2P_001000010_1*R_020[0]+a3P_001000000_1*R_021[0]+a1P_000011010_1*R_100[0]+a2P_000011000_1*R_101[0]+a1P_000111010_1*R_110[0]+a2P_000111000_1*R_111[0]+a3P_000000010_1*R_120[0]+R_121[0];
			ans_temp[ans_id*36+26]+=P_000012010*R_000[0]+a1P_000012000_1*R_001[0]+P_000112010*R_010[0]+a1P_000112000_1*R_011[0]+P_000212010*R_020[0]+a1P_000212000_1*R_021[0]+a3P_000000010_1*R_030[0]+R_031[0];
			ans_temp[ans_id*36+27]+=P_001010011*R_000[0]+P_001010111*R_001[0]+a2P_001010000_1*R_002[0]+a1P_001000011_1*R_010[0]+a1P_001000111_1*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000010011_1*R_100[0]+a1P_000010111_1*R_101[0]+a3P_000010000_1*R_102[0]+a2P_000000011_1*R_110[0]+a2P_000000111_1*R_111[0]+R_112[0];
			ans_temp[ans_id*36+28]+=P_000011011*R_000[0]+P_000011111*R_001[0]+a2P_000011000_1*R_002[0]+P_000111011*R_010[0]+P_000111111*R_011[0]+a2P_000111000_1*R_012[0]+a2P_000000011_1*R_020[0]+a2P_000000111_1*R_021[0]+R_022[0];
			ans_temp[ans_id*36+29]+=P_000010012*R_000[0]+P_000010112*R_001[0]+P_000010212*R_002[0]+a3P_000010000_1*R_003[0]+a1P_000000012_1*R_010[0]+a1P_000000112_1*R_011[0]+a1P_000000212_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+30]+=P_002000020*R_000[0]+a1P_002000010_2*R_001[0]+a2P_002000000_1*R_002[0]+a1P_001000020_2*R_100[0]+a2P_001000010_4*R_101[0]+a3P_001000000_2*R_102[0]+a2P_000000020_1*R_200[0]+a3P_000000010_2*R_201[0]+R_202[0];
			ans_temp[ans_id*36+31]+=P_001001020*R_000[0]+a1P_001001010_2*R_001[0]+a2P_001001000_1*R_002[0]+a1P_001000020_1*R_010[0]+a2P_001000010_2*R_011[0]+a3P_001000000_1*R_012[0]+a1P_000001020_1*R_100[0]+a2P_000001010_2*R_101[0]+a3P_000001000_1*R_102[0]+a2P_000000020_1*R_110[0]+a3P_000000010_2*R_111[0]+R_112[0];
			ans_temp[ans_id*36+32]+=P_000002020*R_000[0]+a1P_000002010_2*R_001[0]+a2P_000002000_1*R_002[0]+a1P_000001020_2*R_010[0]+a2P_000001010_4*R_011[0]+a3P_000001000_2*R_012[0]+a2P_000000020_1*R_020[0]+a3P_000000010_2*R_021[0]+R_022[0];
			ans_temp[ans_id*36+33]+=P_001000021*R_000[0]+P_001000121*R_001[0]+P_001000221*R_002[0]+a3P_001000000_1*R_003[0]+a1P_000000021_1*R_100[0]+a1P_000000121_1*R_101[0]+a1P_000000221_1*R_102[0]+R_103[0];
			ans_temp[ans_id*36+34]+=P_000001021*R_000[0]+P_000001121*R_001[0]+P_000001221*R_002[0]+a3P_000001000_1*R_003[0]+a1P_000000021_1*R_010[0]+a1P_000000121_1*R_011[0]+a1P_000000221_1*R_012[0]+R_013[0];
			ans_temp[ans_id*36+35]+=P_000000022*R_000[0]+P_000000122*R_001[0]+P_000000222*R_002[0]+a2P_000000111_2*R_003[0]+R_004[0];
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
__global__ void TSMJ_ddss_JME(unsigned int contrc_bra_num,unsigned int primit_ket_len,\
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
    double Pmtrx[1]={0.0};

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
				double PD_010[3];
				PD_010[0]=PA[ii*3+0];
				PD_010[1]=PA[ii*3+1];
				PD_010[2]=PA[ii*3+2];
				double PD_001[3];
				PD_001[0]=PB[ii*3+0];
				PD_001[1]=PB[ii*3+1];
				PD_001[2]=PB[ii*3+2];
				double Zta=Zta_in[ii];
				double pp=pp_in[ii];
				double aPin1=1/(2*Zta);
			double aPin2=aPin1*aPin1;
			double aPin3=aPin1*aPin2;
			double aPin4=aPin1*aPin3;
			double QR_000000000000=0;
			double QR_000000000001=0;
			double QR_000000000010=0;
			double QR_000000000100=0;
			double QR_000000000002=0;
			double QR_000000000011=0;
			double QR_000000000020=0;
			double QR_000000000101=0;
			double QR_000000000110=0;
			double QR_000000000200=0;
			double QR_000000000003=0;
			double QR_000000000012=0;
			double QR_000000000021=0;
			double QR_000000000030=0;
			double QR_000000000102=0;
			double QR_000000000111=0;
			double QR_000000000120=0;
			double QR_000000000201=0;
			double QR_000000000210=0;
			double QR_000000000300=0;
			double QR_000000000004=0;
			double QR_000000000013=0;
			double QR_000000000022=0;
			double QR_000000000031=0;
			double QR_000000000040=0;
			double QR_000000000103=0;
			double QR_000000000112=0;
			double QR_000000000121=0;
			double QR_000000000130=0;
			double QR_000000000202=0;
			double QR_000000000211=0;
			double QR_000000000220=0;
			double QR_000000000301=0;
			double QR_000000000310=0;
			double QR_000000000400=0;
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
            unsigned int p_jj=contrc_Pmtrx_start_pr+j*1;
        double P_max=0.0;
        for(int p_i=0;p_i<1;p_i++){
            int2 Pmtrx_2=tex1Dfetch(tex_Pmtrx,p_jj+p_i);
            Pmtrx[p_i]=__hiloint2double(Pmtrx_2.y,Pmtrx_2.x);
            if(P_max<Pmtrx[p_i]) P_max=Pmtrx[p_i];
            }
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
	R_000[1]*=aPin1;
	R_000[2]*=aPin2;
	R_000[3]*=aPin3;
	R_000[4]*=aPin4;
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
	for(int i=1;i<4;i++){
		R_000[i]*=aPin1;
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
	for(int i=1;i<3;i++){
		R_100[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_010[i]*=aPin1;
	}
	for(int i=1;i<3;i++){
		R_001[i]*=aPin1;
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
	for(int i=1;i<2;i++){
		R_200[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_020[i]*=aPin1;
	}
	for(int i=1;i<2;i++){
		R_002[i]*=aPin1;
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
			QR_000000000000+=R_000[0];
			QR_000000000001+=aPin1*(R_001[0]);
			QR_000000000010+=aPin1*(R_010[0]);
			QR_000000000100+=aPin1*(R_100[0]);
			QR_000000000002+=aPin2*(R_002[0]);
			QR_000000000011+=aPin2*(R_011[0]);
			QR_000000000020+=aPin2*(R_020[0]);
			QR_000000000101+=aPin2*(R_101[0]);
			QR_000000000110+=aPin2*(R_110[0]);
			QR_000000000200+=aPin2*(R_200[0]);
			QR_000000000003+=aPin3*(R_003[0]);
			QR_000000000012+=aPin3*(R_012[0]);
			QR_000000000021+=aPin3*(R_021[0]);
			QR_000000000030+=aPin3*(R_030[0]);
			QR_000000000102+=aPin3*(R_102[0]);
			QR_000000000111+=aPin3*(R_111[0]);
			QR_000000000120+=aPin3*(R_120[0]);
			QR_000000000201+=aPin3*(R_201[0]);
			QR_000000000210+=aPin3*(R_210[0]);
			QR_000000000300+=aPin3*(R_300[0]);
			QR_000000000004+=aPin4*(R_004[0]);
			QR_000000000013+=aPin4*(R_013[0]);
			QR_000000000022+=aPin4*(R_022[0]);
			QR_000000000031+=aPin4*(R_031[0]);
			QR_000000000040+=aPin4*(R_040[0]);
			QR_000000000103+=aPin4*(R_103[0]);
			QR_000000000112+=aPin4*(R_112[0]);
			QR_000000000121+=aPin4*(R_121[0]);
			QR_000000000130+=aPin4*(R_130[0]);
			QR_000000000202+=aPin4*(R_202[0]);
			QR_000000000211+=aPin4*(R_211[0]);
			QR_000000000220+=aPin4*(R_220[0]);
			QR_000000000301+=aPin4*(R_301[0]);
			QR_000000000310+=aPin4*(R_310[0]);
			QR_000000000400+=aPin4*(R_400[0]);
			}
		double PD_002[3];
		double PD_102[3];
		double PD_011[3];
		double PD_111[3];
		double PD_012[3];
		double PD_112[3];
		double PD_212[3];
		double PD_020[3];
		double PD_120[3];
		double PD_021[3];
		double PD_121[3];
		double PD_221[3];
		double PD_022[3];
		double PD_122[3];
		double PD_222[3];
		for(int i=0;i<3;i++){
			PD_002[i]=aPin1+PD_001[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_102[i]=(2.000000*PD_001[i]);
			}
		for(int i=0;i<3;i++){
			PD_011[i]=aPin1+PD_010[i]*PD_001[i];
			}
		for(int i=0;i<3;i++){
			PD_111[i]=(PD_001[i]+PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_012[i]=PD_111[i]+PD_001[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_112[i]=(PD_002[i]+2.000000*PD_011[i]);
			}
		for(int i=0;i<3;i++){
			PD_212[i]=(0.500000*PD_102[i]+PD_111[i]);
			}
		for(int i=0;i<3;i++){
			PD_020[i]=aPin1+PD_010[i]*PD_010[i];
			}
		for(int i=0;i<3;i++){
			PD_120[i]=(2.000000*PD_010[i]);
			}
		for(int i=0;i<3;i++){
			PD_021[i]=PD_111[i]+PD_010[i]*PD_011[i];
			}
		for(int i=0;i<3;i++){
			PD_121[i]=(2.000000*PD_011[i]+PD_020[i]);
			}
		for(int i=0;i<3;i++){
			PD_221[i]=(PD_111[i]+0.500000*PD_120[i]);
			}
		for(int i=0;i<3;i++){
			PD_022[i]=PD_112[i]+PD_010[i]*PD_012[i];
			}
		for(int i=0;i<3;i++){
			PD_122[i]=2.000000*(PD_012[i]+PD_021[i]);
			}
		for(int i=0;i<3;i++){
			PD_222[i]=(PD_112[i]+PD_121[i]);
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
			P_022000000=PD_022[0];
			P_122000000=PD_122[0];
			P_222000000=PD_222[0];
			P_021001000=PD_021[0]*PD_001[1];
			P_121001000=PD_121[0]*PD_001[1];
			P_221001000=PD_221[0]*PD_001[1];
			P_020002000=PD_020[0]*PD_002[1];
			P_021000001=PD_021[0]*PD_001[2];
			P_121000001=PD_121[0]*PD_001[2];
			P_221000001=PD_221[0]*PD_001[2];
			P_020001001=PD_020[0]*PD_001[1]*PD_001[2];
			P_020000002=PD_020[0]*PD_002[2];
			P_012010000=PD_012[0]*PD_010[1];
			P_112010000=PD_112[0]*PD_010[1];
			P_212010000=PD_212[0]*PD_010[1];
			P_011011000=PD_011[0]*PD_011[1];
			P_011111000=PD_011[0]*PD_111[1];
			P_111011000=PD_111[0]*PD_011[1];
			P_111111000=PD_111[0]*PD_111[1];
			P_010012000=PD_010[0]*PD_012[1];
			P_010112000=PD_010[0]*PD_112[1];
			P_010212000=PD_010[0]*PD_212[1];
			P_011010001=PD_011[0]*PD_010[1]*PD_001[2];
			P_111010001=PD_111[0]*PD_010[1]*PD_001[2];
			P_010011001=PD_010[0]*PD_011[1]*PD_001[2];
			P_010111001=PD_010[0]*PD_111[1]*PD_001[2];
			P_010010002=PD_010[0]*PD_010[1]*PD_002[2];
			P_002020000=PD_002[0]*PD_020[1];
			P_001021000=PD_001[0]*PD_021[1];
			P_001121000=PD_001[0]*PD_121[1];
			P_001221000=PD_001[0]*PD_221[1];
			P_000022000=PD_022[1];
			P_000122000=PD_122[1];
			P_000222000=PD_222[1];
			P_001020001=PD_001[0]*PD_020[1]*PD_001[2];
			P_000021001=PD_021[1]*PD_001[2];
			P_000121001=PD_121[1]*PD_001[2];
			P_000221001=PD_221[1]*PD_001[2];
			P_000020002=PD_020[1]*PD_002[2];
			P_012000010=PD_012[0]*PD_010[2];
			P_112000010=PD_112[0]*PD_010[2];
			P_212000010=PD_212[0]*PD_010[2];
			P_011001010=PD_011[0]*PD_001[1]*PD_010[2];
			P_111001010=PD_111[0]*PD_001[1]*PD_010[2];
			P_010002010=PD_010[0]*PD_002[1]*PD_010[2];
			P_011000011=PD_011[0]*PD_011[2];
			P_011000111=PD_011[0]*PD_111[2];
			P_111000011=PD_111[0]*PD_011[2];
			P_111000111=PD_111[0]*PD_111[2];
			P_010001011=PD_010[0]*PD_001[1]*PD_011[2];
			P_010001111=PD_010[0]*PD_001[1]*PD_111[2];
			P_010000012=PD_010[0]*PD_012[2];
			P_010000112=PD_010[0]*PD_112[2];
			P_010000212=PD_010[0]*PD_212[2];
			P_002010010=PD_002[0]*PD_010[1]*PD_010[2];
			P_001011010=PD_001[0]*PD_011[1]*PD_010[2];
			P_001111010=PD_001[0]*PD_111[1]*PD_010[2];
			P_000012010=PD_012[1]*PD_010[2];
			P_000112010=PD_112[1]*PD_010[2];
			P_000212010=PD_212[1]*PD_010[2];
			P_001010011=PD_001[0]*PD_010[1]*PD_011[2];
			P_001010111=PD_001[0]*PD_010[1]*PD_111[2];
			P_000011011=PD_011[1]*PD_011[2];
			P_000011111=PD_011[1]*PD_111[2];
			P_000111011=PD_111[1]*PD_011[2];
			P_000111111=PD_111[1]*PD_111[2];
			P_000010012=PD_010[1]*PD_012[2];
			P_000010112=PD_010[1]*PD_112[2];
			P_000010212=PD_010[1]*PD_212[2];
			P_002000020=PD_002[0]*PD_020[2];
			P_001001020=PD_001[0]*PD_001[1]*PD_020[2];
			P_000002020=PD_002[1]*PD_020[2];
			P_001000021=PD_001[0]*PD_021[2];
			P_001000121=PD_001[0]*PD_121[2];
			P_001000221=PD_001[0]*PD_221[2];
			P_000001021=PD_001[1]*PD_021[2];
			P_000001121=PD_001[1]*PD_121[2];
			P_000001221=PD_001[1]*PD_221[2];
			P_000000022=PD_022[2];
			P_000000122=PD_122[2];
			P_000000222=PD_222[2];
			a2P_111000000_1=PD_111[0];
			a2P_111000000_2=2*a2P_111000000_1;
			a1P_021000000_1=PD_021[0];
			a1P_121000000_1=PD_121[0];
			a1P_221000000_1=PD_221[0];
			a3P_000001000_1=PD_001[1];
			a3P_000001000_2=2*a3P_000001000_1;
			a1P_020001000_1=PD_020[0]*PD_001[1];
			a1P_020001000_2=2*a1P_020001000_1;
			a2P_020000000_1=PD_020[0];
			a1P_010002000_1=PD_010[0]*PD_002[1];
			a1P_010002000_2=2*a1P_010002000_1;
			a2P_010001000_1=PD_010[0]*PD_001[1];
			a2P_010001000_4=4*a2P_010001000_1;
			a2P_010001000_2=2*a2P_010001000_1;
			a3P_010000000_1=PD_010[0];
			a3P_010000000_2=2*a3P_010000000_1;
			a2P_000002000_1=PD_002[1];
			a3P_000000001_1=PD_001[2];
			a3P_000000001_2=2*a3P_000000001_1;
			a1P_020000001_1=PD_020[0]*PD_001[2];
			a1P_020000001_2=2*a1P_020000001_1;
			a1P_010001001_1=PD_010[0]*PD_001[1]*PD_001[2];
			a1P_010001001_2=2*a1P_010001001_1;
			a2P_010000001_1=PD_010[0]*PD_001[2];
			a2P_010000001_2=2*a2P_010000001_1;
			a2P_010000001_4=4*a2P_010000001_1;
			a2P_000001001_1=PD_001[1]*PD_001[2];
			a1P_010000002_1=PD_010[0]*PD_002[2];
			a1P_010000002_2=2*a1P_010000002_1;
			a2P_000000002_1=PD_002[2];
			a1P_012000000_1=PD_012[0];
			a1P_112000000_1=PD_112[0];
			a1P_212000000_1=PD_212[0];
			a3P_000010000_1=PD_010[1];
			a3P_000010000_2=2*a3P_000010000_1;
			a2P_011000000_1=PD_011[0];
			a2P_000011000_1=PD_011[1];
			a2P_000111000_1=PD_111[1];
			a2P_000111000_2=2*a2P_000111000_1;
			a1P_000012000_1=PD_012[1];
			a1P_000112000_1=PD_112[1];
			a1P_000212000_1=PD_212[1];
			a1P_011010000_1=PD_011[0]*PD_010[1];
			a1P_011000001_1=PD_011[0]*PD_001[2];
			a1P_111010000_1=PD_111[0]*PD_010[1];
			a1P_111000001_1=PD_111[0]*PD_001[2];
			a2P_000010001_1=PD_010[1]*PD_001[2];
			a2P_000010001_2=2*a2P_000010001_1;
			a2P_000010001_4=4*a2P_000010001_1;
			a1P_010011000_1=PD_010[0]*PD_011[1];
			a1P_010111000_1=PD_010[0]*PD_111[1];
			a1P_000011001_1=PD_011[1]*PD_001[2];
			a1P_000111001_1=PD_111[1]*PD_001[2];
			a1P_010010001_1=PD_010[0]*PD_010[1]*PD_001[2];
			a1P_010010001_2=2*a1P_010010001_1;
			a2P_010010000_1=PD_010[0]*PD_010[1];
			a1P_000010002_1=PD_010[1]*PD_002[2];
			a1P_000010002_2=2*a1P_000010002_1;
			a1P_002010000_1=PD_002[0]*PD_010[1];
			a1P_002010000_2=2*a1P_002010000_1;
			a2P_002000000_1=PD_002[0];
			a1P_001020000_1=PD_001[0]*PD_020[1];
			a1P_001020000_2=2*a1P_001020000_1;
			a2P_001010000_1=PD_001[0]*PD_010[1];
			a2P_001010000_4=4*a2P_001010000_1;
			a2P_001010000_2=2*a2P_001010000_1;
			a3P_001000000_1=PD_001[0];
			a3P_001000000_2=2*a3P_001000000_1;
			a2P_000020000_1=PD_020[1];
			a1P_000021000_1=PD_021[1];
			a1P_000121000_1=PD_121[1];
			a1P_000221000_1=PD_221[1];
			a1P_001010001_1=PD_001[0]*PD_010[1]*PD_001[2];
			a1P_001010001_2=2*a1P_001010001_1;
			a2P_001000001_1=PD_001[0]*PD_001[2];
			a1P_000020001_1=PD_020[1]*PD_001[2];
			a1P_000020001_2=2*a1P_000020001_1;
			a3P_000000010_1=PD_010[2];
			a3P_000000010_2=2*a3P_000000010_1;
			a1P_011001000_1=PD_011[0]*PD_001[1];
			a1P_011000010_1=PD_011[0]*PD_010[2];
			a1P_111001000_1=PD_111[0]*PD_001[1];
			a1P_111000010_1=PD_111[0]*PD_010[2];
			a2P_000001010_1=PD_001[1]*PD_010[2];
			a2P_000001010_2=2*a2P_000001010_1;
			a2P_000001010_4=4*a2P_000001010_1;
			a1P_010001010_1=PD_010[0]*PD_001[1]*PD_010[2];
			a1P_010001010_2=2*a1P_010001010_1;
			a2P_010000010_1=PD_010[0]*PD_010[2];
			a1P_000002010_1=PD_002[1]*PD_010[2];
			a1P_000002010_2=2*a1P_000002010_1;
			a2P_000000011_1=PD_011[2];
			a2P_000000111_1=PD_111[2];
			a2P_000000111_2=2*a2P_000000111_1;
			a1P_010000011_1=PD_010[0]*PD_011[2];
			a1P_010000111_1=PD_010[0]*PD_111[2];
			a1P_000001011_1=PD_001[1]*PD_011[2];
			a1P_000001111_1=PD_001[1]*PD_111[2];
			a1P_000000012_1=PD_012[2];
			a1P_000000112_1=PD_112[2];
			a1P_000000212_1=PD_212[2];
			a1P_002000010_1=PD_002[0]*PD_010[2];
			a1P_002000010_2=2*a1P_002000010_1;
			a1P_001010010_1=PD_001[0]*PD_010[1]*PD_010[2];
			a1P_001010010_2=2*a1P_001010010_1;
			a2P_001000010_1=PD_001[0]*PD_010[2];
			a2P_001000010_2=2*a2P_001000010_1;
			a2P_001000010_4=4*a2P_001000010_1;
			a2P_000010010_1=PD_010[1]*PD_010[2];
			a1P_001011000_1=PD_001[0]*PD_011[1];
			a1P_001111000_1=PD_001[0]*PD_111[1];
			a1P_000011010_1=PD_011[1]*PD_010[2];
			a1P_000111010_1=PD_111[1]*PD_010[2];
			a1P_001000011_1=PD_001[0]*PD_011[2];
			a1P_001000111_1=PD_001[0]*PD_111[2];
			a1P_000010011_1=PD_010[1]*PD_011[2];
			a1P_000010111_1=PD_010[1]*PD_111[2];
			a1P_001000020_1=PD_001[0]*PD_020[2];
			a1P_001000020_2=2*a1P_001000020_1;
			a2P_000000020_1=PD_020[2];
			a1P_001001010_1=PD_001[0]*PD_001[1]*PD_010[2];
			a1P_001001010_2=2*a1P_001001010_1;
			a2P_001001000_1=PD_001[0]*PD_001[1];
			a1P_000001020_1=PD_001[1]*PD_020[2];
			a1P_000001020_2=2*a1P_000001020_1;
			a1P_000000021_1=PD_021[2];
			a1P_000000121_1=PD_121[2];
			a1P_000000221_1=PD_221[2];
			ans_temp[ans_id*36+0]+=P_022000000*QR_000000000000+P_122000000*QR_000000000100+P_222000000*QR_000000000200+a2P_111000000_2*QR_000000000300+QR_000000000400;
			ans_temp[ans_id*36+1]+=P_021001000*QR_000000000000+a1P_021000000_1*QR_000000000010+P_121001000*QR_000000000100+a1P_121000000_1*QR_000000000110+P_221001000*QR_000000000200+a1P_221000000_1*QR_000000000210+a3P_000001000_1*QR_000000000300+QR_000000000310;
			ans_temp[ans_id*36+2]+=P_020002000*QR_000000000000+a1P_020001000_2*QR_000000000010+a2P_020000000_1*QR_000000000020+a1P_010002000_2*QR_000000000100+a2P_010001000_4*QR_000000000110+a3P_010000000_2*QR_000000000120+a2P_000002000_1*QR_000000000200+a3P_000001000_2*QR_000000000210+QR_000000000220;
			ans_temp[ans_id*36+3]+=P_021000001*QR_000000000000+a1P_021000000_1*QR_000000000001+P_121000001*QR_000000000100+a1P_121000000_1*QR_000000000101+P_221000001*QR_000000000200+a1P_221000000_1*QR_000000000201+a3P_000000001_1*QR_000000000300+QR_000000000301;
			ans_temp[ans_id*36+4]+=P_020001001*QR_000000000000+a1P_020001000_1*QR_000000000001+a1P_020000001_1*QR_000000000010+a2P_020000000_1*QR_000000000011+a1P_010001001_2*QR_000000000100+a2P_010001000_2*QR_000000000101+a2P_010000001_2*QR_000000000110+a3P_010000000_2*QR_000000000111+a2P_000001001_1*QR_000000000200+a3P_000001000_1*QR_000000000201+a3P_000000001_1*QR_000000000210+QR_000000000211;
			ans_temp[ans_id*36+5]+=P_020000002*QR_000000000000+a1P_020000001_2*QR_000000000001+a2P_020000000_1*QR_000000000002+a1P_010000002_2*QR_000000000100+a2P_010000001_4*QR_000000000101+a3P_010000000_2*QR_000000000102+a2P_000000002_1*QR_000000000200+a3P_000000001_2*QR_000000000201+QR_000000000202;
			ans_temp[ans_id*36+6]+=P_012010000*QR_000000000000+a1P_012000000_1*QR_000000000010+P_112010000*QR_000000000100+a1P_112000000_1*QR_000000000110+P_212010000*QR_000000000200+a1P_212000000_1*QR_000000000210+a3P_000010000_1*QR_000000000300+QR_000000000310;
			ans_temp[ans_id*36+7]+=P_011011000*QR_000000000000+P_011111000*QR_000000000010+a2P_011000000_1*QR_000000000020+P_111011000*QR_000000000100+P_111111000*QR_000000000110+a2P_111000000_1*QR_000000000120+a2P_000011000_1*QR_000000000200+a2P_000111000_1*QR_000000000210+QR_000000000220;
			ans_temp[ans_id*36+8]+=P_010012000*QR_000000000000+P_010112000*QR_000000000010+P_010212000*QR_000000000020+a3P_010000000_1*QR_000000000030+a1P_000012000_1*QR_000000000100+a1P_000112000_1*QR_000000000110+a1P_000212000_1*QR_000000000120+QR_000000000130;
			ans_temp[ans_id*36+9]+=P_011010001*QR_000000000000+a1P_011010000_1*QR_000000000001+a1P_011000001_1*QR_000000000010+a2P_011000000_1*QR_000000000011+P_111010001*QR_000000000100+a1P_111010000_1*QR_000000000101+a1P_111000001_1*QR_000000000110+a2P_111000000_1*QR_000000000111+a2P_000010001_1*QR_000000000200+a3P_000010000_1*QR_000000000201+a3P_000000001_1*QR_000000000210+QR_000000000211;
			ans_temp[ans_id*36+10]+=P_010011001*QR_000000000000+a1P_010011000_1*QR_000000000001+P_010111001*QR_000000000010+a1P_010111000_1*QR_000000000011+a2P_010000001_1*QR_000000000020+a3P_010000000_1*QR_000000000021+a1P_000011001_1*QR_000000000100+a2P_000011000_1*QR_000000000101+a1P_000111001_1*QR_000000000110+a2P_000111000_1*QR_000000000111+a3P_000000001_1*QR_000000000120+QR_000000000121;
			ans_temp[ans_id*36+11]+=P_010010002*QR_000000000000+a1P_010010001_2*QR_000000000001+a2P_010010000_1*QR_000000000002+a1P_010000002_1*QR_000000000010+a2P_010000001_2*QR_000000000011+a3P_010000000_1*QR_000000000012+a1P_000010002_1*QR_000000000100+a2P_000010001_2*QR_000000000101+a3P_000010000_1*QR_000000000102+a2P_000000002_1*QR_000000000110+a3P_000000001_2*QR_000000000111+QR_000000000112;
			ans_temp[ans_id*36+12]+=P_002020000*QR_000000000000+a1P_002010000_2*QR_000000000010+a2P_002000000_1*QR_000000000020+a1P_001020000_2*QR_000000000100+a2P_001010000_4*QR_000000000110+a3P_001000000_2*QR_000000000120+a2P_000020000_1*QR_000000000200+a3P_000010000_2*QR_000000000210+QR_000000000220;
			ans_temp[ans_id*36+13]+=P_001021000*QR_000000000000+P_001121000*QR_000000000010+P_001221000*QR_000000000020+a3P_001000000_1*QR_000000000030+a1P_000021000_1*QR_000000000100+a1P_000121000_1*QR_000000000110+a1P_000221000_1*QR_000000000120+QR_000000000130;
			ans_temp[ans_id*36+14]+=P_000022000*QR_000000000000+P_000122000*QR_000000000010+P_000222000*QR_000000000020+a2P_000111000_2*QR_000000000030+QR_000000000040;
			ans_temp[ans_id*36+15]+=P_001020001*QR_000000000000+a1P_001020000_1*QR_000000000001+a1P_001010001_2*QR_000000000010+a2P_001010000_2*QR_000000000011+a2P_001000001_1*QR_000000000020+a3P_001000000_1*QR_000000000021+a1P_000020001_1*QR_000000000100+a2P_000020000_1*QR_000000000101+a2P_000010001_2*QR_000000000110+a3P_000010000_2*QR_000000000111+a3P_000000001_1*QR_000000000120+QR_000000000121;
			ans_temp[ans_id*36+16]+=P_000021001*QR_000000000000+a1P_000021000_1*QR_000000000001+P_000121001*QR_000000000010+a1P_000121000_1*QR_000000000011+P_000221001*QR_000000000020+a1P_000221000_1*QR_000000000021+a3P_000000001_1*QR_000000000030+QR_000000000031;
			ans_temp[ans_id*36+17]+=P_000020002*QR_000000000000+a1P_000020001_2*QR_000000000001+a2P_000020000_1*QR_000000000002+a1P_000010002_2*QR_000000000010+a2P_000010001_4*QR_000000000011+a3P_000010000_2*QR_000000000012+a2P_000000002_1*QR_000000000020+a3P_000000001_2*QR_000000000021+QR_000000000022;
			ans_temp[ans_id*36+18]+=P_012000010*QR_000000000000+a1P_012000000_1*QR_000000000001+P_112000010*QR_000000000100+a1P_112000000_1*QR_000000000101+P_212000010*QR_000000000200+a1P_212000000_1*QR_000000000201+a3P_000000010_1*QR_000000000300+QR_000000000301;
			ans_temp[ans_id*36+19]+=P_011001010*QR_000000000000+a1P_011001000_1*QR_000000000001+a1P_011000010_1*QR_000000000010+a2P_011000000_1*QR_000000000011+P_111001010*QR_000000000100+a1P_111001000_1*QR_000000000101+a1P_111000010_1*QR_000000000110+a2P_111000000_1*QR_000000000111+a2P_000001010_1*QR_000000000200+a3P_000001000_1*QR_000000000201+a3P_000000010_1*QR_000000000210+QR_000000000211;
			ans_temp[ans_id*36+20]+=P_010002010*QR_000000000000+a1P_010002000_1*QR_000000000001+a1P_010001010_2*QR_000000000010+a2P_010001000_2*QR_000000000011+a2P_010000010_1*QR_000000000020+a3P_010000000_1*QR_000000000021+a1P_000002010_1*QR_000000000100+a2P_000002000_1*QR_000000000101+a2P_000001010_2*QR_000000000110+a3P_000001000_2*QR_000000000111+a3P_000000010_1*QR_000000000120+QR_000000000121;
			ans_temp[ans_id*36+21]+=P_011000011*QR_000000000000+P_011000111*QR_000000000001+a2P_011000000_1*QR_000000000002+P_111000011*QR_000000000100+P_111000111*QR_000000000101+a2P_111000000_1*QR_000000000102+a2P_000000011_1*QR_000000000200+a2P_000000111_1*QR_000000000201+QR_000000000202;
			ans_temp[ans_id*36+22]+=P_010001011*QR_000000000000+P_010001111*QR_000000000001+a2P_010001000_1*QR_000000000002+a1P_010000011_1*QR_000000000010+a1P_010000111_1*QR_000000000011+a3P_010000000_1*QR_000000000012+a1P_000001011_1*QR_000000000100+a1P_000001111_1*QR_000000000101+a3P_000001000_1*QR_000000000102+a2P_000000011_1*QR_000000000110+a2P_000000111_1*QR_000000000111+QR_000000000112;
			ans_temp[ans_id*36+23]+=P_010000012*QR_000000000000+P_010000112*QR_000000000001+P_010000212*QR_000000000002+a3P_010000000_1*QR_000000000003+a1P_000000012_1*QR_000000000100+a1P_000000112_1*QR_000000000101+a1P_000000212_1*QR_000000000102+QR_000000000103;
			ans_temp[ans_id*36+24]+=P_002010010*QR_000000000000+a1P_002010000_1*QR_000000000001+a1P_002000010_1*QR_000000000010+a2P_002000000_1*QR_000000000011+a1P_001010010_2*QR_000000000100+a2P_001010000_2*QR_000000000101+a2P_001000010_2*QR_000000000110+a3P_001000000_2*QR_000000000111+a2P_000010010_1*QR_000000000200+a3P_000010000_1*QR_000000000201+a3P_000000010_1*QR_000000000210+QR_000000000211;
			ans_temp[ans_id*36+25]+=P_001011010*QR_000000000000+a1P_001011000_1*QR_000000000001+P_001111010*QR_000000000010+a1P_001111000_1*QR_000000000011+a2P_001000010_1*QR_000000000020+a3P_001000000_1*QR_000000000021+a1P_000011010_1*QR_000000000100+a2P_000011000_1*QR_000000000101+a1P_000111010_1*QR_000000000110+a2P_000111000_1*QR_000000000111+a3P_000000010_1*QR_000000000120+QR_000000000121;
			ans_temp[ans_id*36+26]+=P_000012010*QR_000000000000+a1P_000012000_1*QR_000000000001+P_000112010*QR_000000000010+a1P_000112000_1*QR_000000000011+P_000212010*QR_000000000020+a1P_000212000_1*QR_000000000021+a3P_000000010_1*QR_000000000030+QR_000000000031;
			ans_temp[ans_id*36+27]+=P_001010011*QR_000000000000+P_001010111*QR_000000000001+a2P_001010000_1*QR_000000000002+a1P_001000011_1*QR_000000000010+a1P_001000111_1*QR_000000000011+a3P_001000000_1*QR_000000000012+a1P_000010011_1*QR_000000000100+a1P_000010111_1*QR_000000000101+a3P_000010000_1*QR_000000000102+a2P_000000011_1*QR_000000000110+a2P_000000111_1*QR_000000000111+QR_000000000112;
			ans_temp[ans_id*36+28]+=P_000011011*QR_000000000000+P_000011111*QR_000000000001+a2P_000011000_1*QR_000000000002+P_000111011*QR_000000000010+P_000111111*QR_000000000011+a2P_000111000_1*QR_000000000012+a2P_000000011_1*QR_000000000020+a2P_000000111_1*QR_000000000021+QR_000000000022;
			ans_temp[ans_id*36+29]+=P_000010012*QR_000000000000+P_000010112*QR_000000000001+P_000010212*QR_000000000002+a3P_000010000_1*QR_000000000003+a1P_000000012_1*QR_000000000010+a1P_000000112_1*QR_000000000011+a1P_000000212_1*QR_000000000012+QR_000000000013;
			ans_temp[ans_id*36+30]+=P_002000020*QR_000000000000+a1P_002000010_2*QR_000000000001+a2P_002000000_1*QR_000000000002+a1P_001000020_2*QR_000000000100+a2P_001000010_4*QR_000000000101+a3P_001000000_2*QR_000000000102+a2P_000000020_1*QR_000000000200+a3P_000000010_2*QR_000000000201+QR_000000000202;
			ans_temp[ans_id*36+31]+=P_001001020*QR_000000000000+a1P_001001010_2*QR_000000000001+a2P_001001000_1*QR_000000000002+a1P_001000020_1*QR_000000000010+a2P_001000010_2*QR_000000000011+a3P_001000000_1*QR_000000000012+a1P_000001020_1*QR_000000000100+a2P_000001010_2*QR_000000000101+a3P_000001000_1*QR_000000000102+a2P_000000020_1*QR_000000000110+a3P_000000010_2*QR_000000000111+QR_000000000112;
			ans_temp[ans_id*36+32]+=P_000002020*QR_000000000000+a1P_000002010_2*QR_000000000001+a2P_000002000_1*QR_000000000002+a1P_000001020_2*QR_000000000010+a2P_000001010_4*QR_000000000011+a3P_000001000_2*QR_000000000012+a2P_000000020_1*QR_000000000020+a3P_000000010_2*QR_000000000021+QR_000000000022;
			ans_temp[ans_id*36+33]+=P_001000021*QR_000000000000+P_001000121*QR_000000000001+P_001000221*QR_000000000002+a3P_001000000_1*QR_000000000003+a1P_000000021_1*QR_000000000100+a1P_000000121_1*QR_000000000101+a1P_000000221_1*QR_000000000102+QR_000000000103;
			ans_temp[ans_id*36+34]+=P_000001021*QR_000000000000+P_000001121*QR_000000000001+P_000001221*QR_000000000002+a3P_000001000_1*QR_000000000003+a1P_000000021_1*QR_000000000010+a1P_000000121_1*QR_000000000011+a1P_000000221_1*QR_000000000012+QR_000000000013;
			ans_temp[ans_id*36+35]+=P_000000022*QR_000000000000+P_000000122*QR_000000000001+P_000000222*QR_000000000002+a2P_000000111_2*QR_000000000003+QR_000000000004;
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
