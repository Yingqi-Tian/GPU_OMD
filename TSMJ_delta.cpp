#include<stdio.h>
#include<memory.h>
#include"global.h"
#include"expression.h"
#define SUBLEN 150

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
extern char c_AC[2];
extern char c_BD[2];
extern char c_XYZ[3];

void get_MD_nlm(int aam, int ia, int * na, int * la, int * ma);

int GCD(int a,int b);

bool check_R(int nx,int ny,int nz){
    bool b_R=true;
    if(nx==1 ||ny==1 || nz==1) b_R=false;
    return b_R;
}

int get_R_expression(char * expression,int subid,int a,int b,int c,int p){
    if(a==1){
        subid+=sprintf(expression+subid,"TX*");
        a-=1;p+=1;
    }
    if(b==1){
        subid+=sprintf(expression+subid,"TY*");
        b-=1;p+=1;
    }
    if(c==1){
        subid+=sprintf(expression+subid,"TZ*");
        c-=1;p+=1;
    }
    subid+=sprintf(expression+subid,"R_%d%d%d[%d]",a,b,c,p);
    return subid;
}

int get_R_expression(char * expression,int subid,int a,int b,int c,char * i_string,int p,bool b_fullr){
    if(b_fullr){
    subid+=sprintf(expression+subid,"R_%d%d%d[%s%d]",a,b,c,i_string,p);
    }
    else{
    if(a==1){
        subid+=sprintf(expression+subid,"TX*");
        a-=1;p+=1;
    }
    if(b==1){
        subid+=sprintf(expression+subid,"TY*");
        b-=1;p+=1;
    }
    if(c==1){
        subid+=sprintf(expression+subid,"TZ*");
        c-=1;p+=1;
    }
    subid+=sprintf(expression+subid,"R_%d%d%d[%s%d]",a,b,c,i_string,p);
    }
    return subid;
}
int combination(int a,int b){
    if(b==0) return 1;
    int temp=1;
    for(int i=1;i<=b;i++){
        temp*=(a-i+1);
    }
    for(int i=1;i<=b;i++){
        temp/=i;
    }
    return temp;
}
int get_d_expression(char * expression,int subid,int ipq,bool b_smalld,int p,int a,int b,int ixyz){
    if(p==a+b || a+b==0){return subid;}
    if(a==0){
        if(b>1){
        subid+=sprintf(expression+subid,"%d*",combination(b,b-p));
        b=b-p;p=0;
        }
    }
    else if(b==0){
        if(a>1){
        subid+=sprintf(expression+subid,"%d*",combination(a,a-p));
        a=a-p;p=0;
        }
    }
    else{
        int k=GCD(a,b);
        if(k>1){
            if(p==k*(a+b)-1){
                subid+=sprintf(expression+subid,"%d*",k);
                a=a/k;b=b/k;p=a+b-1;
            }
        }

    }/*
    if(a==0 && b==1) subid+=sprintf(expression+subid,"%c%c%c",c_PQ[ipq],c_BD[ipq],c_XYZ[ixyz]);
    else if(a==1 && b==0) subid+=sprintf(expression+subid,"%c%c%c",c_PQ[ipq],c_AC[ipq],c_XYZ[ixyz]);
    else subid+=sprintf(expression+subid,"%c%c_%d%d%d[%d]",c_PQ[ipq],(b_smalld)?'d':'D',p,a,b,ixyz);
    */
    subid+=sprintf(expression+subid,"%c%c_%d%d%d[%d]",c_PQ[ipq],(b_smalld)?'d':'D',p,a,b,ixyz);
    return subid;
}

void TSMJ_delta_declare_0(int ipq,FILE * fp,int type,int delta_type){
    int np,na,nb;
    int lp,la,lb;
    int mp,ma,mb;
    int nalpha;
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_sum=P_delta_mark_sum;
        delta_mark_array=&P_delta_mark_array;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_sum=Q_delta_mark_sum;
        delta_mark_array=&Q_delta_mark_array;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }
    for(int i=0;i<delta_mark_sum;i++){
        if((*delta_mark_array)[i].status==0){
            fprintf(fp,"\t\t\tdouble ");
            (*delta_mark_array)[i].print_expression(fp,c_PQ[ipq]);
            fprintf(fp,";\n");
        }
    }
    for(int i=0;i<expression_encode_list_sum;i++){
        nalpha=(*expression_encode_list)[i].exp_code[0];
        np=(*expression_encode_list)[i].exp_code[1];
        na=(*expression_encode_list)[i].exp_code[2];
        nb=(*expression_encode_list)[i].exp_code[3];
        lp=(*expression_encode_list)[i].exp_code[4];
        la=(*expression_encode_list)[i].exp_code[5];
        lb=(*expression_encode_list)[i].exp_code[6];
        mp=(*expression_encode_list)[i].exp_code[7];
        ma=(*expression_encode_list)[i].exp_code[8];
        mb=(*expression_encode_list)[i].exp_code[9];

        int test_nlm=
                np+na+nb+\
                lp+la+lb+\
                mp+ma+mb;
        if(test_nlm!=0){
            fprintf(fp,"\t\t\tdouble a%d%c_%d%d%d%d%d%d%d%d%d_1;\n",\
                nalpha,\
                c_PQ[ipq],\
                np,na,nb,\
                lp,la,lb,\
                mp,ma,mb);

            for(int j=0;j<(*expression_encode_list)[i].coef_sum;j++)\
                fprintf(fp,"\t\t\tdouble a%d%c_%d%d%d%d%d%d%d%d%d_%d;\n",\
                        nalpha,\
                        c_PQ[ipq],\
                        np,na,nb,\
                        lp,la,lb,\
                        mp,ma,mb,\
                        (*expression_encode_list)[i].coef[j]);
        }
    }
}

void TSMJ_delta_declare_1(int ipq,FILE * fp,int type,int delta_type,bool only_q){
    int np,na,nb;
    int lp,la,lb;
    int mp,ma,mb;
    int nalpha;
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_sum=P_delta_mark_sum;
        delta_mark_array=&P_delta_mark_array;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_sum=Q_delta_mark_sum;
        delta_mark_array=&Q_delta_mark_array;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }
    for(int i=0;i<delta_mark_sum;i++){
        if((*delta_mark_array)[i].status==0){
            fprintf(fp,"\t\t\tdouble ");
            (*delta_mark_array)[i].print_expression(fp,c_PQ[ipq]);
            fprintf(fp,";\n");
        }
    }
    for(int i=0;i<expression_encode_list_sum;i++){
        nalpha=(*expression_encode_list)[i].exp_code[0];
        np=(*expression_encode_list)[i].exp_code[1];
        na=(*expression_encode_list)[i].exp_code[2];
        nb=(*expression_encode_list)[i].exp_code[3];
        lp=(*expression_encode_list)[i].exp_code[4];
        la=(*expression_encode_list)[i].exp_code[5];
        lb=(*expression_encode_list)[i].exp_code[6];
        mp=(*expression_encode_list)[i].exp_code[7];
        ma=(*expression_encode_list)[i].exp_code[8];
        mb=(*expression_encode_list)[i].exp_code[9];
        if(np+na+nb+lp+la+lb+mp+ma+mb!=0){
        	if(only_q){
            fprintf(fp,"\t\t\tdouble %c_%d%d%d%d%d%d%d%d%d_1;\n",\
                c_PQ[ipq],\
                np,na,nb,\
                lp,la,lb,\
                mp,ma,mb);
        	}
        	else{
				fprintf(fp,"\t\t\tdouble a%d%c_%d%d%d%d%d%d%d%d%d_1;\n",\
					nalpha,\
					c_PQ[ipq],\
					np,na,nb,\
					lp,la,lb,\
					mp,ma,mb);
        	}
        }
        for(int j=0;j<(*expression_encode_list)[i].coef_sum;j++){
			if(only_q){
            fprintf(fp,"\t\t\tdouble %c_%d%d%d%d%d%d%d%d%d_%d;\n",\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb,\
                    (*expression_encode_list)[i].coef[j]);
			}
			else{
				fprintf(fp,"\t\t\tdouble a%d%c_%d%d%d%d%d%d%d%d%d_%d;\n",\
						nalpha,\
						c_PQ[ipq],\
						np,na,nb,\
						lp,la,lb,\
						mp,ma,mb,\
						(*expression_encode_list)[i].coef[j]);
			}
        }
    }
}
void TSMJ_delta_expression_0(int ipq,FILE * fp,int delta_type,int type){
    bool test_np=true;
    bool test_lp=true;
    bool test_mp=true;
    bool test=false;
    int np,na,nb;
    int lp,la,lb;
    int mp,ma,mb;
    int nalpha;
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_array=&P_delta_mark_array;
        delta_mark_sum=P_delta_mark_sum;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_array=&Q_delta_mark_array;
        delta_mark_sum=Q_delta_mark_sum;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }
    for(int i=0;i<delta_mark_sum;i++){
        if((*delta_mark_array)[i].status==0){
            np=(*delta_mark_array)[i].id[0];
            na=(*delta_mark_array)[i].id[1];
            nb=(*delta_mark_array)[i].id[2];
            lp=(*delta_mark_array)[i].id[3];
            la=(*delta_mark_array)[i].id[4];
            lb=(*delta_mark_array)[i].id[5];
            mp=(*delta_mark_array)[i].id[6];
            ma=(*delta_mark_array)[i].id[7];
            mb=(*delta_mark_array)[i].id[8];
            fprintf(fp,"\t\t\t%c_%d%d%d%d%d%d%d%d%d=",c_PQ[ipq],np,na,nb,lp,la,lb,mp,ma,mb);

            nalpha=0;
            test_np=true;
            test_lp=true;
            test_mp=true;
            test=false;


            if(np==na+nb){
                nalpha+=np;
                test_np=false;
            }
            if(lp==la+lb){
                nalpha+=lp;
                test_lp=false;
            }
            if(mp==ma+mb){
                nalpha+=mp;
                test_mp=false;
            }
            if(test_np){
                fprintf(fp,"%cd_%d%d%d[0]",c_PQ[ipq],np,na,nb);
                test=true;
            }
            if(test_lp){
				if(test) fprintf(fp,"*");
                fprintf(fp,"%cd_%d%d%d[1]",c_PQ[ipq],lp,la,lb);
                test=true;
            }
            if(test_mp){
				if(test) fprintf(fp,"*");
                fprintf(fp,"%cd_%d%d%d[2]",c_PQ[ipq],mp,ma,mb);
                test=true;
            }
            if(nalpha>0){
				if(test) fprintf(fp,"*");
				fprintf(fp,"a%cin%d",c_PQ[ipq],nalpha);
            }
            fprintf(fp,";\n");
        }
    }
    for(int i=0;i<expression_encode_list_sum;i++){
        nalpha=(*expression_encode_list)[i].exp_code[0];
        np=(*expression_encode_list)[i].exp_code[1];
        na=(*expression_encode_list)[i].exp_code[2];
        nb=(*expression_encode_list)[i].exp_code[3];
        lp=(*expression_encode_list)[i].exp_code[4];
        la=(*expression_encode_list)[i].exp_code[5];
        lb=(*expression_encode_list)[i].exp_code[6];
        mp=(*expression_encode_list)[i].exp_code[7];
        ma=(*expression_encode_list)[i].exp_code[8];
        mb=(*expression_encode_list)[i].exp_code[9];

        int test_nlm=
                np+na+nb+\
                lp+la+lb+\
                mp+ma+mb;
        if(test_nlm!=0){
        fprintf(fp,"\t\t\ta%d%c_%d%d%d%d%d%d%d%d%d_1=",\
                nalpha,\
                c_PQ[ipq],\
                np,na,nb,\
                lp,la,lb,\
                mp,ma,mb);
        fprintf(fp,"a%cin%d",c_PQ[ipq],nalpha);
        if(np+na+nb!=0){
            fprintf(fp,"*%cd_%d%d%d[0]",c_PQ[ipq],np,na,nb);
        }
        if(lp+la+lb!=0){
            fprintf(fp,"*%cd_%d%d%d[1]",c_PQ[ipq],lp,la,lb);
        }
        if(mp+ma+mb!=0){
            fprintf(fp,"*%cd_%d%d%d[2]",c_PQ[ipq],mp,ma,mb);
        }
        fprintf(fp,";\n");
        for(int j=0;j<(*expression_encode_list)[i].coef_sum;j++){
            fprintf(fp,"\t\t\ta%d%c_%d%d%d%d%d%d%d%d%d_%d=%d*a%d%c_%d%d%d%d%d%d%d%d%d_1;\n",\
                    nalpha,\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb,\
                    (*expression_encode_list)[i].coef[j],\
                    (*expression_encode_list)[i].coef[j],\
                    nalpha,\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb);
        }
        }
    }
};
void TSMJ_delta_expression_1(int ipq,FILE * fp,int delta_type,int type,bool only_q,bool b_smalld){
    bool test_np=true;
    bool test_lp=true;
    bool test_mp=true;
    bool test=false;
    int np,na,nb;
    int lp,la,lb;
    int mp,ma,mb;
    int nalpha;
    char c_d=(b_smalld)?'d':'D';
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_array=&P_delta_mark_array;
        delta_mark_sum=P_delta_mark_sum;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_array=&Q_delta_mark_array;
        delta_mark_sum=Q_delta_mark_sum;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }
    for(int i=0;i<delta_mark_sum;i++){
        if((*delta_mark_array)[i].status==0){
            np=(*delta_mark_array)[i].id[0];
            na=(*delta_mark_array)[i].id[1];
            nb=(*delta_mark_array)[i].id[2];
            lp=(*delta_mark_array)[i].id[3];
            la=(*delta_mark_array)[i].id[4];
            lb=(*delta_mark_array)[i].id[5];
            mp=(*delta_mark_array)[i].id[6];
            ma=(*delta_mark_array)[i].id[7];
            mb=(*delta_mark_array)[i].id[8];
            nalpha=0;
            test_np=true;
            test_lp=true;
            test_mp=true;
            test=false;

            fprintf(fp,"\t\t\t%c_%d%d%d%d%d%d%d%d%d=",c_PQ[ipq],np,na,nb,lp,la,lb,mp,ma,mb);

            if(np==na+nb){
                nalpha+=np;
                test_np=false;
            }
            if(lp==la+lb){
                nalpha+=lp;
                test_lp=false;
            }
            if(mp==ma+mb){
                nalpha+=mp;
                test_mp=false;
            }
            if(test_np){
                fprintf(fp,"%c%c_%d%d%d[0]",c_PQ[ipq],c_d,np,na,nb);
                test=true;
            }
            if(test_lp){
                if(test){
                    fprintf(fp,"*");
                }
                fprintf(fp,"%c%c_%d%d%d[1]",c_PQ[ipq],c_d,lp,la,lb);
                test=true;
            }
            if(test_mp){
                if(test){
                    fprintf(fp,"*");
                }
                fprintf(fp,"%c%c_%d%d%d[2]",c_PQ[ipq],c_d,mp,ma,mb);
                test=true;
            }
            if(nalpha>0){
                if(delta_type==0 || (type==1 && ipq==0)){
                    if(test){
                        fprintf(fp,"*");
                    }
                    fprintf(fp,"a%cin%d",c_PQ[ipq],nalpha);
                    test=true;
                }
            }
            fprintf(fp,";\n");
        }
    }
    for(int i=0;i<expression_encode_list_sum;i++){
        nalpha=(*expression_encode_list)[i].exp_code[0];
        np=(*expression_encode_list)[i].exp_code[1];
        na=(*expression_encode_list)[i].exp_code[2];
        nb=(*expression_encode_list)[i].exp_code[3];
        lp=(*expression_encode_list)[i].exp_code[4];
        la=(*expression_encode_list)[i].exp_code[5];
        lb=(*expression_encode_list)[i].exp_code[6];
        mp=(*expression_encode_list)[i].exp_code[7];
        ma=(*expression_encode_list)[i].exp_code[8];
        mb=(*expression_encode_list)[i].exp_code[9];
        bool test_out=false;
        if(np+na+nb+lp+la+lb+mp+ma+mb!=0){
			if(only_q){
				fprintf(fp,"\t\t\t%c_%d%d%d%d%d%d%d%d%d_1=",\
					c_PQ[ipq],\
					np,na,nb,\
					lp,la,lb,\
					mp,ma,mb);
			}
			else{
				fprintf(fp,"\t\t\ta%d%c_%d%d%d%d%d%d%d%d%d_1=",\
					nalpha,\
					c_PQ[ipq],\
					np,na,nb,\
					lp,la,lb,\
					mp,ma,mb);
			}
			if(np+na+nb!=0){
				fprintf(fp,"%c%c_%d%d%d[0]",c_PQ[ipq],c_d,np,na,nb);
				test_out=true;
			}
			if(lp+la+lb!=0){
				if(test_out) fprintf(fp,"*");
				fprintf(fp,"%c%c_%d%d%d[1]",c_PQ[ipq],c_d,lp,la,lb);
				test_out=true;
			}
			if(mp+ma+mb!=0){
				if(test_out) fprintf(fp,"*");
				fprintf(fp,"%c%c_%d%d%d[2]",c_PQ[ipq],c_d,mp,ma,mb);
			}
			fprintf(fp,";\n");
        }
        for(int j=0;j<(*expression_encode_list)[i].coef_sum;j++){
			if(only_q){
            fprintf(fp,"\t\t\t%c_%d%d%d%d%d%d%d%d%d_%d=%d*%c_%d%d%d%d%d%d%d%d%d_1;\n",\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb,\
                    (*expression_encode_list)[i].coef[j],\
                    (*expression_encode_list)[i].coef[j],\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb);
			}
			else {
            fprintf(fp,"\t\t\ta%d%c_%d%d%d%d%d%d%d%d%d_%d=%d*a%d%c_%d%d%d%d%d%d%d%d%d_1;\n",\
                    nalpha,\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb,\
                    (*expression_encode_list)[i].coef[j],\
                    (*expression_encode_list)[i].coef[j],\
                    nalpha,\
                    c_PQ[ipq],\
                    np,na,nb,\
                    lp,la,lb,\
                    mp,ma,mb);
			}
        }
    }
};
void MD_R_table_setup(int nx,int ny, int nz,int nj){
    if(nx<0||ny<0||nz<0){return;}
    if(MD_R_mark_table[nj][nx][ny][nz]!=0){return;}
    MD_R_mark_table[nj][nx][ny][nz]+=1;
    if(nx>=ny && nx>=nz){
        MD_R_table_setup(nx-1,ny,nz,nj+1);
        MD_R_table_setup(nx-2,ny,nz,nj+1);
    }
    else if(ny>=nx && ny>=nz){
        MD_R_table_setup(nx,ny-1,nz,nj+1);
        MD_R_table_setup(nx,ny-2,nz,nj+1);
    }
    else{
        MD_R_table_setup(nx,ny,nz-1,nj+1);
        MD_R_table_setup(nx,ny,nz-2,nj+1);
    }
}
void MD_R_setup(int ia,int ib, int ic, int id, int ia2,int ib2,int ic2,int id2){
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,nc=0,lc=0,mc=0,nd=0,ld=0,md=0,np=0,lp=0,mp=0,nq=0,lq=0,mq=0;
    get_MD_nlm(ia,ia2,&na,&la,&ma);
    get_MD_nlm(ib,ib2,&nb,&lb,&mb);
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                for(nq=0;nq<=nc+nd;nq++){
                    for(lq=0;lq<=lc+ld;lq++){
                        for(mq=0;mq<=mc+md;mq++){
                            MD_R_table_setup(np+nq,lp+lq,mp+mq,0);
                        }
                    }
                }
            }
        }
    }
};
void MD_R_init(int aam, int bam, int cam, int dam){
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
                    MD_R_setup(aam,bam,cam,dam,ia2,ib2,ic2,id2);
                }
            }
        }
    }
};
void zero_R_mark_table(){
    for(int j=0;j<MAX_J;j++){
        for(int x=0;x<MAX_J;x++){
        for(int y=0;y<MAX_J;y++){
        for(int z=0;z<MAX_J;z++){
            MD_R_mark_table[j][x][y][z]=0;
        }
        }
        }
    }
}
int R_subexpression(int ixyz,int nx, int ny, int nz,char * subexpression,int subid,int ipq,bool b_fullr,bool b_smalld){
    int nxyz[3];
    nxyz[0]=nx;
    nxyz[1]=ny;
    nxyz[2]=nz;
    int test=nxyz[ixyz];
    if(test==1){
        nxyz[ixyz]-=1;
        subid+=sprintf(subexpression+subid,"T%c*",c_XYZ[ixyz]);
        subid=get_R_expression(subexpression,subid,nxyz[0],nxyz[1],nxyz[2],"i+",1,b_fullr);
    }
    else{
        nxyz[ixyz]-=1;
        subid+=sprintf(subexpression+subid,"T%c*",c_XYZ[ixyz]);
        subid=get_R_expression(subexpression,subid,nxyz[0],nxyz[1],nxyz[2],"i+",1,b_fullr);
        subid+=sprintf(subexpression+subid,"+");
        if(test==2){
            nxyz[ixyz]-=1;
            if(!b_smalld && !b_fullr) subid+=sprintf(subexpression+subid,"a%cin1*",c_PQ[ipq]);
            subid=get_R_expression(subexpression,subid,nxyz[0],nxyz[1],nxyz[2],"i+",1,b_fullr);
        }
        else{
            nxyz[ixyz]-=1;
            subid+=sprintf(subexpression+subid,"%d*",test-1);
            if(!b_smalld && !b_fullr) subid+=sprintf(subexpression+subid,"a%cin1*",c_PQ[ipq]);
            subid=get_R_expression(subexpression,subid,nxyz[0],nxyz[1],nxyz[2],"i+",1,b_fullr);
        }
    }
    return subid;
}
void TSMJ_R_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,bool b_smalld,bool b_fullr){
    char subexpression[SUBLEN];
    int subid=0;
    int temp_max=aam+bam+cam+dam+1;
    int nj;
    memset(subexpression,0,SUBLEN*sizeof(char));

    subid=0;
    if(!b_smalld){
        for(int i=0;i<aam+bam+cam+dam;i++){
            fprintf(fp,"\tR_000[%d]*=a%cin%d;\n",i+1,c_PQ[ipq],i+1);
        }
    }

	for(int nz=0;nz<temp_max;nz++){
        for(int ny=0;ny<temp_max;ny++){
			for(int nx=0;nx<temp_max;nx++){
				nj=nx+ny+nz;
				if(MD_R_mark_table[0][nx][ny][nz]==0){
					continue;
				}
				if(nx==0&&ny==0&&nz==0){
					continue;
				}
				memset(subexpression,0,SUBLEN*sizeof(char));
				subid=0;
				if(b_fullr){
                    subid+=sprintf(subexpression+subid,"\tdouble R_%d%d%d[%d];\n",nx,ny,nz,temp_max-nj);
				}
				else{
                    if(check_R(nx,ny,nz)){
                        subid+=sprintf(subexpression+subid,"\tdouble R_%d%d%d[%d];\n",nx,ny,nz,temp_max-nj);
                    }
				}
				fprintf(fp,"%s",subexpression);
			}
		}
	}
    for(int nxyz=1;nxyz<temp_max;nxyz++){
        for(int nz=0;nz<=nxyz;nz++){
            for(int ny=0;ny<=nxyz-nz;ny++){
                int nx=nxyz-nz-ny;
                if(MD_R_mark_table[0][nx][ny][nz]==0){
                    continue;
                }
                memset(subexpression,0,SUBLEN*sizeof(char));
                subid=0;
                subid+=sprintf(subexpression+subid,"\tfor(int i=0;i<%d;i++){\n\t",temp_max-nxyz);
                subid+=sprintf(subexpression+subid,"\tR_%d%d%d[i]=",nx,ny,nz);
                //fprintf(fp,"%s",subexpression);

                //memset(subexpression,0,SUBLEN*sizeof(char));
                //subid=0;
                if(nx==1){
                    subid=R_subexpression(0,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                else if(ny==1){
                    subid=R_subexpression(1,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                else if(nz==1){
                    subid=R_subexpression(2,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                else if(nx>=ny && nx>=nz){
                    subid=R_subexpression(0,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                else if(ny>=nx && ny>=nz){
                    subid=R_subexpression(1,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                else{
                    subid=R_subexpression(2,nx,ny,nz,subexpression,subid,ipq,b_fullr,b_smalld);
                }
                subid+=sprintf(subexpression+subid,";\n");
                subid+=sprintf(subexpression+subid,"\t}\n");
				if(b_fullr){
                    fprintf(fp,"%s",subexpression);
				}
				else{
                    if(check_R(nx,ny,nz)){
                    fprintf(fp,"%s",subexpression);
                    }
				}
            }
        }
        if(!b_smalld && b_fullr){
			if(nxyz<temp_max-1){
            for(int nz=0;nz<=nxyz-1;nz++){
                for(int ny=0;ny<=nxyz-1-nz;ny++){
                    int nx=nxyz-1-nz-ny;
                    bool test_for_R2=false;
                    if(MD_R_mark_table[0][nx+2][ny][nz]!=0){
                        if(nx+2>=ny && nx+2>=nz && ny!=1 && nz!=1){
                            test_for_R2=true;
                        }
                    }
                    if(MD_R_mark_table[0][nx][ny+2][nz]!=0){
                        if(ny+2>=nx && ny+2>=nz && nx!=1 && nz!=1){
                            test_for_R2=true;
                        }
                    }
                    if(MD_R_mark_table[0][nx][ny][nz+2]!=0){
                        if(nz+2>=ny && nz+2>=nx && ny!=1 && nx!=1){
                            test_for_R2=true;
                        }
                    }
                    if(test_for_R2){
						fprintf(fp,"\tfor(int i=1;i<%d;i++){\n",temp_max-nxyz);
						fprintf(fp,"\t\tR_%d%d%d[i]*=a%cin1;\n\t}\n",nx,ny,nz,c_PQ[ipq]);
                    }
                }
            }
			}
        }
    }
}
void MD_QR_declare(FILE* fp,int np,int lp, int mp, int ic, int id, int ic2,int id2,int ipq){
    int nc=0,lc=0,mc=0,nd=0,ld=0,md=0;
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);

    if(QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]==0){
        return;
    }
    fprintf(fp,"\tdouble %cR_%d%d%d%d%d%d%d%d%d%d%d%d=0;\n",c_PQ[ipq],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
}
void MD_QR_declare(FILE * fp,int aam, int bam, int cam, int dam,int ipq){
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;
    for(int nlmp=0;nlmp<aam+bam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(int ic2=0;ic2<sc;ic2++){
                        for(int id2=0;id2<sd;id2++){
                            MD_QR_declare(fp,np,lp,mp,cam,dam,ic2,id2,ipq);
                        }
                    }
            }
        }
    }
}
void TSMJ_QR_expression(FILE* fp,int np,int lp, int mp, int ic, int id, int ic2,int id2,int ipq,int id_p,\
                        bool outer_zero,bool b_declare,bool summation,bool b_delta,bool b_smalld,\
                        bool b_fullr,bool b_PmQR,bool b_aPinQR,bool b_QRsum){
    char subexpression[SUBLEN];
    int subid=0;
    bool temp_test=false;
    int next_nq,next_nc,next_nd;
    int next_lq,next_lc,next_ld;
    int next_mq,next_mc,next_md;
    int nalpha;
    int temp_code=0;
    int temp_coef=0;
    int index=0;
    int nc=0,lc=0,mc=0,nd=0,ld=0,md=0,nq=0,lq=0,mq=0;
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);
    if(!b_QRsum && nc+lc+mc+nd+ld+md+nq+lq+mq==0){
        return;
    }
    if(QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]==0){
        return;
    }
    if(b_declare) fprintf(fp,"\t\t\tdouble ");
    else fprintf(fp,"\t\t\t");
    fprintf(fp,"%cR_%d%d%d%d%d%d%d%d%d%d%d%d",c_PQ[ipq],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
    if(summation){
        fprintf(fp,"+=");
    }
    else fprintf(fp,"=");
    if(b_PmQR) fprintf(fp,"Pmtrx[%d]*(",id_p);
    char c_P=c_PQ[(ipq+1)%2];
    if(b_aPinQR && np+lp+mp>0) fprintf(fp,"a%cin%d*(",c_P,np+lp+mp);

    temp_test=false;
    char pm[2]={'+','-'};
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_array=&P_delta_mark_array;
        delta_mark_sum=P_delta_mark_sum;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_array=&Q_delta_mark_array;
        delta_mark_sum=Q_delta_mark_sum;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }

    for(nq=0;nq<=nc+nd;nq++){
        for(lq=0;lq<=lc+ld;lq++){
            for(mq=0;mq<=mc+md;mq++){
                if(temp_test){
                    if(ipq==0){
                        fprintf(fp,"+");
                    }
                    else{
                        if((nq+lq+mq)%2==1){
                            fprintf(fp,"-");
                        }
                        else{
                            fprintf(fp,"+");
                        }
                    }
                }
                memset(subexpression,0,SUBLEN*sizeof(char));
                subid=0;
                if(ic+id!=0){
                    for(int i=0;i<delta_mark_sum;i++){
                        if((*delta_mark_array)[i].id[0]==nq &&\
                            (*delta_mark_array)[i].id[1]==nc &&\
                            (*delta_mark_array)[i].id[2]==nd &&\
                            (*delta_mark_array)[i].id[3]==lq &&\
                            (*delta_mark_array)[i].id[4]==lc &&\
                            (*delta_mark_array)[i].id[5]==ld &&\
                            (*delta_mark_array)[i].id[6]==mq &&\
                            (*delta_mark_array)[i].id[7]==mc &&\
                            (*delta_mark_array)[i].id[8]==md){
                                index=i;
                        }
                    }
                    int temp_status=(*delta_mark_array)[index].status;
                    if(temp_status==0){
                        if(b_delta){
                            subid+=sprintf(subexpression+subid,"%c_%d%d%d%d%d%d%d%d%d*",c_PQ[ipq],nq,nc,nd,lq,lc,ld,mq,mc,md);
                        }
                        else{
                        if(nq+nc+nd>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,nq,nc,nd,0);
                            subid+=sprintf(subexpression+subid,"*");
                        }
                        if(lq+lc+ld>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,lq,lc,ld,1);
                            subid+=sprintf(subexpression+subid,"*",c_PQ[ipq],lq,lc,ld);
                        }
                        if(mq+mc+md>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,mq,mc,md,2);
                            subid+=sprintf(subexpression+subid,"*",c_PQ[ipq],mq,mc,md);
                        }
                        }
                    }
                    else{
                        temp_code=(*delta_mark_array)[index].expression_code;
                        next_md=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_mc=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_mq=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        next_ld=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_lc=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_lq=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        next_nd=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_nc=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_nq=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        nalpha=temp_code;
                        temp_coef=(*delta_mark_array)[index].coef;
                        int test_nlm=
                                    next_nq+next_nc+next_nd+\
                                    next_lq+next_lc+next_ld+\
                                    next_mq+next_mc+next_md;\
                        if(test_nlm==0){
                            if(b_smalld || !outer_zero) subid+=sprintf(subexpression+subid,"a%cin%d*",c_PQ[ipq],nalpha);
                        }
                        else{
                        if(b_delta){
                            subid+=sprintf(subexpression+subid,"a%d%c_%d%d%d%d%d%d%d%d%d_%d*",nalpha,c_PQ[ipq],\
                                    next_nq,next_nc,next_nd,\
                                    next_lq,next_lc,next_ld,\
                                    next_mq,next_mc,next_md,\
                                    temp_coef);
                        }
                        else{
                            if(temp_coef>1) subid+=sprintf(subexpression+subid,"%d*",temp_coef);
                            if(b_smalld && nalpha>0) subid+=sprintf(subexpression+subid,"a%cin%d*",c_PQ[ipq],nalpha);
                            if(next_nq+next_nc+next_nd>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,next_nq,next_nc,next_nd,0);
                                subid+=sprintf(subexpression+subid,"*");
                            }
                            if(next_lq+next_lc+next_ld>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,next_lq,next_lc,next_ld,1);
                                subid+=sprintf(subexpression+subid,"*");
                            }
                            if(next_mq+next_mc+next_md>0){
                            subid=get_d_expression(subexpression,subid,ipq,b_smalld,next_mq,next_mc,next_md,2);
                                subid+=sprintf(subexpression+subid,"*");
                            }
                        }
                        }
                    }
                }
                if(b_fullr){
                    subid+=sprintf(subexpression+subid,"R_%d%d%d[0]",np+nq,lp+lq,mp+mq);
                }
                else{
                    subid=get_R_expression(subexpression,subid,np+nq,lp+lq,mp+mq,0);
                }
                fprintf(fp,"%s",subexpression);
                temp_test=true;
            }
        }
    }
    if(b_PmQR) fprintf(fp,")");
    if(b_aPinQR && np+lp+mp>0) fprintf(fp,")");
    fprintf(fp,";\n");
}
void TSMJ_QR_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,\
                        bool outer_zero,bool b_declare,bool summation,bool b_delta,bool b_smalld,bool b_fullr,bool b_PmQR,bool b_aPinQR,bool b_QRsum){
    int ic2;
    int id2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;
    if(ipq==0){
        sc=(aam+2)*(aam+1)/2;
        sd=(bam+2)*(bam+1)/2;
    for(int nlmp=0;nlmp<cam+dam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            int id_p=ic2*sd+id2;
                            TSMJ_QR_expression(fp,np,lp,mp,aam,bam,ic2,id2,ipq,id_p,\
                                               outer_zero,b_declare,summation,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,b_QRsum);
                        }
                    }
            }
        }
    }
    }
    else{
        sc=(cam+2)*(cam+1)/2;
        sd=(dam+2)*(dam+1)/2;
    for(int nlmp=0;nlmp<aam+bam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            int id_p=ic2*sd+id2;
                            TSMJ_QR_expression(fp,np,lp,mp,cam,dam,ic2,id2,ipq,id_p,\
                                               outer_zero,b_declare,summation,b_delta,b_smalld,b_fullr,b_PmQR,b_aPinQR,b_QRsum);
                        }
                    }
            }
        }
    }
    }
}

void TSMJ_QR_declare(FILE* fp,int np,int lp, int mp, int ic, int id, int ic2,int id2,int ipq,bool b_JME){
    char subexpression[SUBLEN];
    int subid=0;
    bool temp_test=false;
    int next_nq,next_nc,next_nd;
    int next_lq,next_lc,next_ld;
    int next_mq,next_mc,next_md;
    int nalpha;
    int temp_code=0;
    int temp_coef=0;
    int index=0;
    int nc=0,lc=0,mc=0,nd=0,ld=0,md=0,nq=0,lq=0,mq=0;
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);
    if(!b_JME && nc+lc+mc+nd+ld+md+nq+lq+mq==0){
        return;
    }
    if(QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]==0){
        return;
    }
    fprintf(fp,"\t\t\tdouble ");
    fprintf(fp,"%cR_%d%d%d%d%d%d%d%d%d%d%d%d=0;\n",c_PQ[ipq],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
}
void TSMJ_QR_declare(FILE * fp, int aam, int bam, int cam, int dam,int ipq,bool b_JME){
    int ic2;
    int id2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;
    if(ipq==0){
        sc=(aam+2)*(aam+1)/2;
        sd=(bam+2)*(bam+1)/2;
    for(int nlmp=0;nlmp<cam+dam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            TSMJ_QR_declare(fp,np,lp,mp,aam,bam,ic2,id2,ipq,b_JME);
                        }
                    }
            }
        }
    }
    }
    else{
        sc=(cam+2)*(cam+1)/2;
        sd=(dam+2)*(dam+1)/2;
    for(int nlmp=0;nlmp<aam+bam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            TSMJ_QR_declare(fp,np,lp,mp,cam,dam,ic2,id2,ipq,b_JME);
                        }
                    }
            }
        }
    }
    }
}
void TSMJ_ans_expression(FILE* fp,int ans_len,int ans_id,\
                         int ia,int ib, int ic, int id,\
                         int ia2,int ib2,int ic2,int id2,\
                         int ipq,\
                         bool inner_zero,char c_d,\
                         int output_P,int p_id,\
                         bool b_delta,bool b_smalld,bool b_fullr,\
                         bool b_PmQR,bool b_aPinQR,bool b_QRsum){
    int next_np,next_na,next_nb;
    int next_lp,next_la,next_lb;
    int next_mp,next_ma,next_mb;
    int nalpha;
    int temp_code;
    int temp_coef;
    int index;
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,nc=0,lc=0,mc=0,nd=0,ld=0,md=0,np=0,lp=0,mp=0;
    if(ipq==0){
        get_MD_nlm(ia,ia2,&na,&la,&ma);
        get_MD_nlm(ib,ib2,&nb,&lb,&mb);
        get_MD_nlm(ic,ic2,&nc,&lc,&mc);
        get_MD_nlm(id,id2,&nd,&ld,&md);
    }
    else{
        get_MD_nlm(ic,ic2,&na,&la,&ma);
        get_MD_nlm(id,id2,&nb,&lb,&mb);
        get_MD_nlm(ia,ia2,&nc,&lc,&mc);
        get_MD_nlm(ib,ib2,&nd,&ld,&md);
    }
	fprintf(fp,"\t\t\tans_temp[ans_id*%d+%d]+=",ans_len,ans_id);
    if(!b_PmQR && output_P>0) fprintf(fp,"Pmtrx[%d]*(",p_id);
    char pm[2]={'+','-'};
    delta_mark ** delta_mark_array;
    int delta_mark_sum;
    expression_encode ** expression_encode_list;
    int expression_encode_list_sum;
    if(ipq==0){
        delta_mark_array=&P_delta_mark_array;
        delta_mark_sum=P_delta_mark_sum;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=P_expression_encode_list_sum;
    }
    else{
        delta_mark_array=&Q_delta_mark_array;
        delta_mark_sum=Q_delta_mark_sum;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=Q_expression_encode_list_sum;
    }
    bool temp=false;
    bool test_iab=false;
    if(ipq==0){
        if(ia+ib!=0) test_iab=true;
    }
    else{
        if(ic+id!=0) test_iab=true;
    }
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                if(temp){
                    if(ipq==0){
                        fprintf(fp,"+");
                    }
                    else{
                        if((np+lp+mp)%2==1){
                            fprintf(fp,"-");
                        }
                        else{
                            fprintf(fp,"+");
                        }
                    }
                }
                if(test_iab){
                    for(int i=0;i<delta_mark_sum;i++){
                        if((*delta_mark_array)[i].id[0]==np &&\
                           (*delta_mark_array)[i].id[1]==na &&\
                           (*delta_mark_array)[i].id[2]==nb &&\
                           (*delta_mark_array)[i].id[3]==lp &&\
                           (*delta_mark_array)[i].id[4]==la &&\
                           (*delta_mark_array)[i].id[5]==lb &&\
                           (*delta_mark_array)[i].id[6]==mp &&\
                           (*delta_mark_array)[i].id[7]==ma &&\
                           (*delta_mark_array)[i].id[8]==mb){
                            index=i;
                            break;
                        }
                    }
                    int temp_status=(*delta_mark_array)[index].status;
                    if(temp_status==0){
                        if(b_delta){
                        fprintf(fp,"%c_%d%d%d%d%d%d%d%d%d*",c_PQ[ipq],np,na,nb,lp,la,lb,mp,ma,mb);
                        }
                        else{
                        if(np+na+nb>0) fprintf(fp,"%c%c_%d%d%d[0]*",c_PQ[ipq],c_d,np,na,nb);
                        if(lp+la+lb>0) fprintf(fp,"%c%c_%d%d%d[1]*",c_PQ[ipq],c_d,lp,la,lb);
                        if(mp+ma+mb>0) fprintf(fp,"%c%c_%d%d%d[2]*",c_PQ[ipq],c_d,mp,ma,mb);
                        }
                    }
                    else{
                        temp_code=(*delta_mark_array)[index].expression_code;
                        next_mb=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_ma=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_mp=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        next_lb=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_la=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_lp=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        next_nb=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_na=temp_code%MAX_AMP;
                        temp_code=temp_code/MAX_AMP;
                        next_np=temp_code%MAX_P;
                        temp_code=temp_code/MAX_P;
                        nalpha=temp_code;
                        temp_coef=(*delta_mark_array)[index].coef;
                        if(inner_zero){
                            {
                                if(b_delta){
                                int test_nlm=
                                    next_np+next_na+next_nb+\
                                    next_lp+next_la+next_lb+\
                                    next_mp+next_ma+next_mb;
                                if(test_nlm==0){
                                    if(b_smalld && nalpha>0 && !b_aPinQR) fprintf(fp,"a%cin%d*",c_PQ[ipq],nalpha);
                                }
                                else{
                                fprintf(fp,"a%d%c_%d%d%d%d%d%d%d%d%d_%d*",nalpha,c_PQ[ipq],\
                                    next_np,next_na,next_nb,\
                                    next_lp,next_la,next_lb,\
                                    next_mp,next_ma,next_mb,\
                                    temp_coef);
                                }
                                }
                                else{
                                if(temp_coef>1) fprintf(fp,"%d*",temp_coef);
                                if(next_np+next_na+next_nb>0) fprintf(fp,"%c%c_%d%d%d[0]*",c_PQ[ipq],c_d,next_np,next_na,next_nb);
                                if(next_lp+next_la+next_lb>0) fprintf(fp,"%c%c_%d%d%d[1]*",c_PQ[ipq],c_d,next_lp,next_la,next_lb);
                                if(next_mp+next_ma+next_mb>0) fprintf(fp,"%c%c_%d%d%d[2]*",c_PQ[ipq],c_d,next_mp,next_ma,next_mb);
                                }
                            }
                        }
                        else{
                            int test_nlm=
                                    next_np+next_na+next_nb+\
                                    next_lp+next_la+next_lb+\
                                    next_mp+next_ma+next_mb;
                            if(test_nlm==0){
                                if(!b_aPinQR) fprintf(fp,"a%cin%d*",c_PQ[ipq],nalpha);
                            }
                            else{
                                if(b_delta){
                                fprintf(fp,"a%d%c_%d%d%d%d%d%d%d%d%d_%d*",nalpha,c_PQ[ipq],\
                                    next_np,next_na,next_nb,\
                                    next_lp,next_la,next_lb,\
                                    next_mp,next_ma,next_mb,\
                                    temp_coef);
                                }
                                else{
                                if(temp_coef>1) fprintf(fp,"%d*",temp_coef);
                                if(b_smalld && nalpha>0 && !b_aPinQR) fprintf(fp,"a%cin%d*",c_PQ[ipq],nalpha);
                                if(next_np+next_na+next_nb>0) fprintf(fp,"%c%c_%d%d%d[0]*",c_PQ[ipq],c_d,next_np,next_na,next_nb);
                                if(next_lp+next_la+next_lb>0) fprintf(fp,"%c%c_%d%d%d[1]*",c_PQ[ipq],c_d,next_lp,next_la,next_lb);
                                if(next_mp+next_ma+next_mb>0) fprintf(fp,"%c%c_%d%d%d[2]*",c_PQ[ipq],c_d,next_mp,next_ma,next_mb);
                                }
                            }
                        }
                    }
                }
                if(!b_QRsum && nc+nd+lc+ld+mc+md==0){
                    char R_expression[20];
                    if(b_fullr){
                        sprintf(R_expression,"R_%d%d%d[0]",np,lp,mp);
                    }
                    else{
                        get_R_expression(R_expression,0,np,lp,mp,0);
                    }
                    fprintf(fp,"%s",R_expression);
                }
                else fprintf(fp,"%cR_%d%d%d%d%d%d%d%d%d%d%d%d",c_PQ[(ipq+1)%2],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
                temp=true;
            }
        }
    }
    if(!b_PmQR && output_P>0) fprintf(fp,")");
    fprintf(fp,";\n");
};

void TSMJ_ans_expression(FILE * fp,bool J_type, int aam, int bam, int cam, int dam,int ipq,\
                         bool inner_zero,int output_P,bool b_delta,bool b_smalld,\
                         bool b_fullr,bool b_PmQR,bool aPinQR,bool b_QRsum){
    int ia2;
    int ib2;
    int ic2;
    int id2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;

    int ans_id=0;
    int ans_len=0;
    int p_id=0;
    for(ia2=0;ia2<sa;ia2++){
        for(ib2=0;ib2<sb;ib2++){
            for(ic2=0;ic2<sc;ic2++){
                for(id2=0;id2<sd;id2++){
                    if(J_type){
                        ans_id=ia2*sb+ib2;
                        ans_len=sa*sb;
                        p_id=ic2*sd+id2;
                    }
                    else{
                        ans_id=ia2*sc+ic2;
                        ans_len=sa*sc;
                        p_id=ib2*sd+id2;
                    }
                    TSMJ_ans_expression(fp,ans_len,ans_id,aam,bam,cam,dam,ia2,ib2,ic2,id2,ipq,\
                                        inner_zero,(!inner_zero)?'d':'D',output_P,p_id,b_delta,b_smalld,b_fullr,b_PmQR,aPinQR,b_QRsum);
                }
            }
        }
    }
}

void MD_delta_expression(int ipq,int ia,int ib,int ia2,int ib2,bool b_smalld,FILE * fp){
#ifndef SUBLEN
#define SUBLEN 100
#endif // SUBLEN
    char subexpression[SUBLEN];
    char c_d=(b_smalld)?'d':'D';
    int subid=0;
    int sa=(ia+2)*(ia+1)/2;
    int sb=(ib+2)*(ib+1)/2;
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,np=0,lp=0,mp=0;
    bool test=false;
    get_MD_nlm(ia,ia2,&na,&la,&ma);
    get_MD_nlm(ib,ib2,&nb,&lb,&mb);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                if(np+na+nb+lp+la+lb+mp+ma+mb==0) continue;
                memset(subexpression,0,SUBLEN*sizeof(char));
                subid=0;
                test=false;
                subid+=sprintf(subexpression+subid,"\tdouble %c_%d%d%d%d%d%d%d%d%d=",c_PQ[ipq],np,na,nb,lp,la,lb,mp,ma,mb);
                if(np!=0 || na!=0 || nb!=0){
                    if(b_smalld || np!=(na+nb)){
                    subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[%d]",c_PQ[ipq],c_d,np,na,nb,0);
                    test=true;
                    }
                }
                if(lp!=0 || la!=0 || lb!=0){
                    if(b_smalld || lp!=(la+lb)){
                    if(test){subid+=sprintf(subexpression+subid,"*");}
                    subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[%d]",c_PQ[ipq],c_d,lp,la,lb,1);
                    test=true;
                    }
                }
                if(mp!=0 || ma!=0 || mb!=0){
                    if(b_smalld || mp!=(ma+mb)){
                    if(test){subid+=sprintf(subexpression+subid,"*");}
                        subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[%d]",c_PQ[ipq],c_d,mp,ma,mb,2);
                    }
                }
                subid+=sprintf(subexpression+subid,";\n");
                if(b_smalld || (np+lp+mp)!=(na+nb+la+lb+ma+mb)) fprintf(fp,"%s",subexpression);
            }
        }
    }
};

void MD_delta_expression(int ipq,FILE * fp,int aam,int bam,bool b_smalld){
    int ia2;
    int ib2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    for(ia2=0;ia2<sa;ia2++){
        for(ib2=0;ib2<sb;ib2++){
            MD_delta_expression(ipq,aam,bam,ia2,ib2,b_smalld,fp);
        }
    }
};

void MD_QR_expression(FILE* fp,int np,int lp, int mp, int ic, int id, int ic2,int id2,int ipq,\
                        bool outer_zero,bool b_declare,bool summation,bool b_delta,bool b_smalld,bool b_fullr){
#ifndef SUBLEN
#define SUBLEN 100
#endif // SUBLEN
    char subexpression[SUBLEN];
    int subid=0;
    char c_d=(b_smalld)?'d':'D';
    int sc=(ic+2)*(ic+1)/2;
    int sd=(id+2)*(id+1)/2;
    int nc=0,lc=0,mc=0,nd=0,ld=0,md=0,nq=0,lq=0,mq=0;
    get_MD_nlm(ic,ic2,&nc,&lc,&mc);
    get_MD_nlm(id,id2,&nd,&ld,&md);

    if(QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]==0){
        return;
    }
    QR_table[nc][nd][lc][ld][mc][md][np][lp][mp]=0;
    if(b_declare) fprintf(fp,"\t\t\t\tdouble ");
    else fprintf(fp,"\t\t\t\t");
    fprintf(fp,"%cR_%d%d%d%d%d%d%d%d%d%d%d%d",c_PQ[ipq],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
    if(summation) fprintf(fp,"+=");
    else fprintf(fp,"=");
    for(nq=0;nq<=nc+nd;nq++){
        for(lq=0;lq<=lc+ld;lq++){
            for(mq=0;mq<=mc+md;mq++){
                memset(subexpression,0,SUBLEN*sizeof(char));
                subid=0;
                if(ic+id!=0){
                    if(b_delta){
                        if(b_smalld || (nq+lq+mq)!=(nc+nd+lc+ld+mc+md))subid+=sprintf(subexpression+subid,"%c_%d%d%d%d%d%d%d%d%d*",c_PQ[ipq],nq,nc,nd,lq,lc,ld,mq,mc,md);
                    }
                    else{
                        if((b_smalld || nq!=(nc+nd)) && nq+nc+nd!=0) subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[0]*",c_PQ[ipq],c_d,nq,nc,nd);
                        if((b_smalld || lq!=(lc+ld)) && lq+lc+ld!=0) subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[1]*",c_PQ[ipq],c_d,lq,lc,ld);
                        if((b_smalld || mq!=(mc+md)) && mq+mc+md!=0) subid+=sprintf(subexpression+subid,"%c%c_%d%d%d[2]*",c_PQ[ipq],c_d,mq,mc,md);
                    }
                }
                if(b_fullr){
                    subid+=sprintf(subexpression+subid,"R_%d%d%d[0]",np+nq,lp+lq,mp+mq);
                }
                else{
                    subid=get_R_expression(subexpression,subid,np+nq,lp+lq,mp+mq,0);
                }
                if((nq+lq+mq)%2==1){
                    fprintf(fp,"-1*");
                }
                fprintf(fp,"%s",subexpression);

                if((nq!=nc+nd) || (lq!=lc+ld) || (mq!=mc+md)){
                    fprintf(fp,"+");
                }
            }
        }
    }
    fprintf(fp,";\n");
}

void MD_ans_expression(FILE* fp,int ans_len,int ans_id,\
                         int ia,int ib, int ic, int id,\
                         int ia2,int ib2,int ic2,int id2,\
                         int ipq,bool b_delta,\
                         bool inner_zero,bool b_smalld,char c_d,bool b_fullr,\
                         int output_P,int p_id){
#ifndef SUBLEN
#define SUBLEN 100
#endif // SUBLEN
    char subexpression[SUBLEN];
    int subid=0;
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,nc=0,lc=0,mc=0,nd=0,ld=0,md=0,np=0,lp=0,mp=0,nq=0,lq=0,mq=0;
    if(ipq==0){
        get_MD_nlm(ia,ia2,&na,&la,&ma);
        get_MD_nlm(ib,ib2,&nb,&lb,&mb);
        get_MD_nlm(ic,ic2,&nc,&lc,&mc);
        get_MD_nlm(id,id2,&nd,&ld,&md);
    }
    else{
        get_MD_nlm(ic,ic2,&na,&la,&ma);
        get_MD_nlm(id,id2,&nb,&lb,&mb);
        get_MD_nlm(ia,ia2,&nc,&lc,&mc);
        get_MD_nlm(ib,ib2,&nd,&ld,&md);
    }
	fprintf(fp,"\t\t\tans_temp[ans_id*%d+%d]+=",ans_len,ans_id);
    if(output_P!=0) fprintf(fp,"Pmtrx[%d]*(",p_id);

    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                if(ia+ib!=0){
                    if(b_delta){
                        if(b_smalld || (np+lp+mp)!=(na+nb+la+lb+ma+mb)){
                        if(np+na+nb+lp+la+lb+mp+ma+mb!=0) fprintf(fp,"%c_%d%d%d%d%d%d%d%d%d*",c_PQ[ipq],np,na,nb,lp,la,lb,mp,ma,mb);
                        }
                    }
                    else{
                        if(np+na+nb!=0) fprintf(fp,"%c%c_%d%d%d[0]*",c_PQ[ipq],c_d,np,na,nb);
                        if(lp+la+lb!=0) fprintf(fp,"%c%c_%d%d%d[1]*",c_PQ[ipq],c_d,lp,la,lb);
                        if(mp+ma+mb!=0) fprintf(fp,"%c%c_%d%d%d[2]*",c_PQ[ipq],c_d,mp,ma,mb);
                    }
                }
                fprintf(fp,"%cR_%d%d%d%d%d%d%d%d%d%d%d%d",c_PQ[(ipq+1)%2],0,nc,nd,0,lc,ld,0,mc,md,np+0,lp+0,mp+0);
                if((np!=na+nb) || (lp!=la+lb) || (mp!=ma+mb)){
                    fprintf(fp,"+");
                }
            }
        }
    }
    if(output_P!=0) fprintf(fp,")");
    fprintf(fp,";\n");
};

void MD_QR_expression(FILE * fp, int aam, int bam, int cam, int dam,int ipq,\
                        bool outer_zero,bool b_declare,bool summation,bool b_delta,bool b_smalld,bool b_fullr){
    int ic2;
    int id2;
    int sc;
    int sd;
    if(ipq==0){
        sc=(aam+2)*(aam+1)/2;
        sd=(bam+2)*(bam+1)/2;
    for(int nlmp=0;nlmp<cam+dam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            MD_QR_expression(fp,np,lp,mp,aam,bam,ic2,id2,ipq,\
                                               outer_zero,b_declare,summation,b_delta,b_smalld,b_fullr);
                        }
                    }
            }
        }
    }
    }
    else{
        sc=(cam+2)*(cam+1)/2;
        sd=(dam+2)*(dam+1)/2;
    for(int nlmp=0;nlmp<aam+bam+1;nlmp++){
        for(int np=0;np<=nlmp;np++){
            for(int lp=0;lp<=nlmp-np;lp++){
                int mp=nlmp-np-lp;
                    for(ic2=0;ic2<sc;ic2++){
                        for(id2=0;id2<sd;id2++){
                            MD_QR_expression(fp,np,lp,mp,cam,dam,ic2,id2,ipq,\
                                               outer_zero,b_declare,summation,b_delta,b_smalld,b_fullr);
                        }
                    }
            }
        }
    }
    }
};

void MD_ans_expression(FILE * fp,bool J_type, int aam, int bam, int cam, int dam,\
                       int ipq,bool inner_zero,int output_P,bool b_delta,bool b_smalld,bool b_fullr){
    int ia2;
    int ib2;
    int ic2;
    int id2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;

    int ans_id=0;
    int ans_len=0;
    int p_id=0;
    for(ia2=0;ia2<sa;ia2++){
        for(ib2=0;ib2<sb;ib2++){
            for(ic2=0;ic2<sc;ic2++){
                for(id2=0;id2<sd;id2++){
                    if(J_type){
                        ans_id=ia2*sb+ib2;
                        ans_len=sa*sb;
                        p_id=ic2*sd+id2;
                    }
                    else{
                        ans_id=ia2*sc+ic2;
                        ans_len=sa*sc;
                        p_id=ib2*sd+id2;
                    }
                    MD_ans_expression(fp,ans_len,ans_id,aam,bam,cam,dam,ia2,ib2,ic2,id2,ipq,\
                                      b_delta,inner_zero,b_smalld,(b_smalld)?'d':'D',b_fullr,output_P,p_id);
                }
            }
        }
    }
};
