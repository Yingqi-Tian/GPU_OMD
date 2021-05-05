#include"global.h"
#include"expression.h"

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

void MD_d_construct(int ipq,int ixyz,int np,int na, int nb){
    int (*d_mark_table)[3][MAX_P][MAX_AMP][MAX_AMP];
    if(ipq==0){
        d_mark_table=&P_d_mark_table;
    }
    else{
        d_mark_table=&Q_d_mark_table;
    }
    if(np==0 && na==0 &&nb==0){return;}
    if(np>(na+nb)||np<0||na<0||nb<0){return;}
    if((*d_mark_table)[ixyz][np][na][nb]!=0){return;}
    (*d_mark_table)[ixyz][np][na][nb]+=1;
    if(na>=nb){
        MD_d_construct(ipq, ixyz, np+1, na-1, nb);
        MD_d_construct(ipq, ixyz, np  , na-1, nb);
        MD_d_construct(ipq, ixyz, np-1, na-1, nb);
    }
    else{
        MD_d_construct(ipq, ixyz, np+1, na, nb-1);
        MD_d_construct(ipq, ixyz, np  , na, nb-1);
        MD_d_construct(ipq, ixyz, np-1, na, nb-1);
    }
}

void MD_d_setup(int ipq,int ia,int ib, int ia2,int ib2){
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,np=0,lp=0,mp=0;
    get_MD_nlm(ia,ia2,&na,&la,&ma);
    get_MD_nlm(ib,ib2,&nb,&lb,&mb);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                MD_d_construct(ipq,0,np,na,nb);
                MD_d_construct(ipq,1,lp,la,lb);
                MD_d_construct(ipq,2,mp,ma,mb);
            }
        }
    }
};

void MD_set_d_mark_table(int ipq,int aam,int bam){
    int ia2;
    int ib2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    for(ia2=0;ia2<sa;ia2++){
    for(ib2=0;ib2<sb;ib2++){
        MD_d_setup(ipq,aam,bam,ia2,ib2);
    }
    }
}

void set_d_mark_table(){
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,np=0,lp=0,mp=0;
    for(int i=0;i<P_delta_mark_sum;i++){
        if(P_delta_mark_array[i].status==0){
            np=P_delta_mark_array[i].id[0];
            na=P_delta_mark_array[i].id[1];
            nb=P_delta_mark_array[i].id[2];
            lp=P_delta_mark_array[i].id[3];
            la=P_delta_mark_array[i].id[4];
            lb=P_delta_mark_array[i].id[5];
            mp=P_delta_mark_array[i].id[6];
            ma=P_delta_mark_array[i].id[7];
            mb=P_delta_mark_array[i].id[8];
            if(P_d_mark_table[0][np][na][nb]==0) P_d_mark_table[0][np][na][nb]=1;
            if(P_d_mark_table[1][lp][la][lb]==0) P_d_mark_table[1][lp][la][lb]=1;
            if(P_d_mark_table[2][mp][ma][mb]==0) P_d_mark_table[2][mp][ma][mb]=1;
        }
    }
    for(int i=0;i<P_expression_encode_list_sum;i++){
        np=P_expression_encode_list[i].exp_code[1];
        na=P_expression_encode_list[i].exp_code[2];
        nb=P_expression_encode_list[i].exp_code[3];
        lp=P_expression_encode_list[i].exp_code[4];
        la=P_expression_encode_list[i].exp_code[5];
        lb=P_expression_encode_list[i].exp_code[6];
        mp=P_expression_encode_list[i].exp_code[7];
        ma=P_expression_encode_list[i].exp_code[8];
        mb=P_expression_encode_list[i].exp_code[9];
        if(P_d_mark_table[0][np][na][nb]==0) P_d_mark_table[0][np][na][nb]=1;
        if(P_d_mark_table[1][lp][la][lb]==0) P_d_mark_table[1][lp][la][lb]=1;
        if(P_d_mark_table[2][mp][ma][mb]==0) P_d_mark_table[2][mp][ma][mb]=1;
    }
    for(int i=0;i<Q_delta_mark_sum;i++){
        if(Q_delta_mark_array[i].status==0){
            np=Q_delta_mark_array[i].id[0];
            na=Q_delta_mark_array[i].id[1];
            nb=Q_delta_mark_array[i].id[2];
            lp=Q_delta_mark_array[i].id[3];
            la=Q_delta_mark_array[i].id[4];
            lb=Q_delta_mark_array[i].id[5];
            mp=Q_delta_mark_array[i].id[6];
            ma=Q_delta_mark_array[i].id[7];
            mb=Q_delta_mark_array[i].id[8];
            if(Q_d_mark_table[0][np][na][nb]==0) Q_d_mark_table[0][np][na][nb]=1;
            if(Q_d_mark_table[1][lp][la][lb]==0) Q_d_mark_table[1][lp][la][lb]=1;
            if(Q_d_mark_table[2][mp][ma][mb]==0) Q_d_mark_table[2][mp][ma][mb]=1;
        }
    }
    for(int i=0;i<Q_expression_encode_list_sum;i++){
        np=Q_expression_encode_list[i].exp_code[1];
        na=Q_expression_encode_list[i].exp_code[2];
        nb=Q_expression_encode_list[i].exp_code[3];
        lp=Q_expression_encode_list[i].exp_code[4];
        la=Q_expression_encode_list[i].exp_code[5];
        lb=Q_expression_encode_list[i].exp_code[6];
        mp=Q_expression_encode_list[i].exp_code[7];
        ma=Q_expression_encode_list[i].exp_code[8];
        mb=Q_expression_encode_list[i].exp_code[9];
        if(Q_d_mark_table[0][np][na][nb]==0) Q_d_mark_table[0][np][na][nb]=1;
        if(Q_d_mark_table[1][lp][la][lb]==0) Q_d_mark_table[1][lp][la][lb]=1;
        if(Q_d_mark_table[2][mp][ma][mb]==0) Q_d_mark_table[2][mp][ma][mb]=1;
    }
};

void check_d_mark_table(int ipq, int ixyz,int np, int na, int nb){
    if(ipq==0){
    if(P_d_mark_table[ixyz][np][na][nb]==1){
        if(np==na+nb){
            P_d_mark_table[ixyz][np][na][nb]=0;
            return;
        }
        if(np!=0){
            if(na!=0){
                if(P_d_mark_table[ixyz][np-1][na-1][nb]!=1 && np!=na+nb){
                    P_d_mark_table[ixyz][np-1][na-1][nb]=1;
                }
            }
            if(nb!=0){
                if(P_d_mark_table[ixyz][np-1][na][nb-1]!=1 && np!=na+nb){
                    P_d_mark_table[ixyz][np-1][na][nb-1]=1;
                }
            }
        }
        else{
            if(na>=nb){
                if(P_d_mark_table[ixyz][np+1][na-1][nb]!=1 && np<na+nb-2){
                    P_d_mark_table[ixyz][np+1][na-1][nb]=1;
                }
                if(P_d_mark_table[ixyz][np][na-1][nb]!=1 && np<na+nb-1){
                    P_d_mark_table[ixyz][np][na-1][nb]=1;
                }
            }
            else{
                if( P_d_mark_table[ixyz][np+1][na][nb-1]!=1 && np<na+nb-2){
                    P_d_mark_table[ixyz][np+1][na][nb-1]=1;
                }
                if( P_d_mark_table[ixyz][np][na][nb-1]!=1 && np<na+nb-1){
                    P_d_mark_table[ixyz][np][na][nb-1]=1;
                }
            }
        }
    }
    }
    else{
    if(Q_d_mark_table[ixyz][np][na][nb]==1){
        if(np==na+nb){
            Q_d_mark_table[ixyz][np][na][nb]=0;
            return;
        }
        if(np!=0){
            if(na!=0){
                if(Q_d_mark_table[ixyz][np-1][na-1][nb]!=1 && np!=na+nb){
                    Q_d_mark_table[ixyz][np-1][na-1][nb]=1;
                }
            }
            if(nb!=0){
                if(Q_d_mark_table[ixyz][np-1][na][nb-1]!=1 && np!=na+nb){
                    Q_d_mark_table[ixyz][np-1][na][nb-1]=1;
                }
            }
        }
        else{
            if(na>=nb){
                if(Q_d_mark_table[ixyz][np+1][na-1][nb]!=1 && np<na+nb-2){
                    Q_d_mark_table[ixyz][np+1][na-1][nb]=1;
                }
                if(Q_d_mark_table[ixyz][np][na-1][nb]!=1 && np<na+nb-1){
                    Q_d_mark_table[ixyz][np][na-1][nb]=1;
                }
            }
            else{
                if( Q_d_mark_table[ixyz][np+1][na][nb-1]!=1 && np<na+nb-2){
                    Q_d_mark_table[ixyz][np+1][na][nb-1]=1;
                }
                if( Q_d_mark_table[ixyz][np][na][nb-1]!=1 && np<na+nb-1){
                    Q_d_mark_table[ixyz][np][na][nb-1]=1;
                }
            }
        }
    }
    }
}
void check_d_mark_table(int aam,int bam,int cam,int dam){
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                for(int ixyz=0;ixyz<3;ixyz++){
                    check_d_mark_table(0,ixyz,np,na,nb);
                }
            }
        }
    }
    for(int na=0;na<=aam;na++){
        for(int nb=0;nb<=bam;nb++){
            for(int np=0;np<=aam+bam;np++){
                if(P_d_mark_table[0][np][na][nb]==0){
                    if(P_d_mark_table[1][np][na][nb]==1) P_d_mark_table[0][np][na][nb]=1;
                    if(P_d_mark_table[2][np][na][nb]==1) P_d_mark_table[0][np][na][nb]=1;
                }
            }
        }
    }

    for(int na=0;na<=cam;na++){
        for(int nb=0;nb<=dam;nb++){
            for(int np=0;np<=cam+dam;np++){
                for(int ixyz=0;ixyz<3;ixyz++){
                    check_d_mark_table(1,ixyz,np,na,nb);
                }
            }
        }
    }
    for(int na=0;na<=cam;na++){
        for(int nb=0;nb<=dam;nb++){
            for(int np=0;np<=cam+dam;np++){
                if(Q_d_mark_table[0][np][na][nb]==0){
                    if(Q_d_mark_table[1][np][na][nb]==1) Q_d_mark_table[0][np][na][nb]=1;
                    if(Q_d_mark_table[2][np][na][nb]==1) Q_d_mark_table[0][np][na][nb]=1;
                }
            }
        }
    }
    for(int i=0;i<3;i++){
        P_d_mark_table[i][0][0][0]=0;
        Q_d_mark_table[i][0][0][0]=0;
    }
}
void zero_d_mark_table(){
    for(int np=0;np<MAX_P;np++){
    for(int na=0;na<MAX_AMP;na++){
    for(int nb=0;nb<MAX_AMP;nb++){
        P_d_mark_table[0][np][na][nb]=0;
        P_d_mark_table[1][np][na][nb]=0;
        P_d_mark_table[2][np][na][nb]=0;
    }
    }
    }
    for(int np=0;np<MAX_P;np++){
    for(int na=0;na<MAX_AMP;na++){
    for(int nb=0;nb<MAX_AMP;nb++){
        Q_d_mark_table[0][np][na][nb]=0;
        Q_d_mark_table[1][np][na][nb]=0;
        Q_d_mark_table[2][np][na][nb]=0;
    }
    }
    }
};
