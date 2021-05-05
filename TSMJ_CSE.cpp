#include<stdio.h>
#include"global.h"
#include"expression.h"
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

void get_MD_nlm(int aam, int ia, int * na, int * la, int * ma){
    if(ia>=aam+1){
        (*ma)++;
        ia=ia-aam-1;
        aam--;
        get_MD_nlm(aam,ia,na,la,ma);
    }
    else{
        (*la)=ia;
        (*na)=aam-(*la);
    }
};
int GCD(int a,int b){
    if(a%b==0){
        return b;
    }
    else{
        return GCD(b,a%b);
    }
}
int array_push(expression_encode ** expression_encode_list, int expression_code_in, int coef_in, int sum){
    for(int i=0;i<sum;i++){
        if((*expression_encode_list)[i].expression_code==expression_code_in){
            if(!(*expression_encode_list)[i].add(coef_in)){
                printf("expression add error!\n");
            }
            return sum;
        }
    }
    (*expression_encode_list)[sum].init(expression_code_in,coef_in);
    return sum+1;
}
int get_delta_sum(int aam,int bam,int ia,int ib){
    int sum=0;
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,np=0,lp=0,mp=0;
    get_MD_nlm(aam,ia,&na,&la,&ma);
    get_MD_nlm(bam,ib,&nb,&lb,&mb);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                sum++;
            }
        }
    }
    return sum;
};
void CSE_init(int aam,int bam,int cam,int dam){
    int sum=0;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    int sc=(cam+2)*(cam+1)/2;
    int sd=(dam+2)*(dam+1)/2;
    sum=0;
    for(int ia=0;ia<sa;ia++){
        for(int ib=0;ib<sb;ib++){
            sum+=get_delta_sum(aam,bam,ia,ib);
        }
    }
    P_delta_mark_sum=sum;
    P_expression_encode_list=new expression_encode[sum];
    P_delta_mark_array=new delta_mark[sum];
    sum=0;
    for(int ic=0;ic<sc;ic++){
        for(int id=0;id<sd;id++){
            sum+=get_delta_sum(cam,dam,ic,id);
        }
    }
    Q_delta_mark_sum=sum;
    Q_expression_encode_list=new expression_encode[sum];
    Q_delta_mark_array=new delta_mark[sum];
};
int CSE_setup_delta_list(int ipq, int index, int ia,int ib,int ia2,int ib2){
    int na=0,la=0,ma=0,nb=0,lb=0,mb=0,np=0,lp=0,mp=0;
    int temp_expression_code=0;
    delta_mark ** delta_mark_array;
    expression_encode ** expression_encode_list;
    int * expression_encode_list_sum;
    if(ipq==0){
        delta_mark_array=&P_delta_mark_array;
        expression_encode_list=&P_expression_encode_list;
        expression_encode_list_sum=&P_expression_encode_list_sum;
    }
    else{
        delta_mark_array=&Q_delta_mark_array;
        expression_encode_list=&Q_expression_encode_list;
        expression_encode_list_sum=&Q_expression_encode_list_sum;
    }
    get_MD_nlm(ia,ia2,&na,&la,&ma);
    get_MD_nlm(ib,ib2,&nb,&lb,&mb);
    for(np=0;np<=na+nb;np++){
        for(lp=0;lp<=la+lb;lp++){
            for(mp=0;mp<=ma+mb;mp++){
                if(np+na+nb+lp+la+lb+mp+ma+mb==0) continue;
                bool test_exp_add=false;
                int temp_coef=1;
                int temp_alpha=0;
                int temp;
                int temp_p[3];
                int temp_a[3];
                int temp_b[3];
                temp_p[0]=np;
                temp_a[0]=na;
                temp_b[0]=nb;
                temp_p[1]=lp;
                temp_a[1]=la;
                temp_b[1]=lb;
                temp_p[2]=mp;
                temp_a[2]=ma;
                temp_b[2]=mb;
                for(int i=0;i<3;i++){
                    if(temp_p[i]==temp_a[i]+temp_b[i]){
                        temp_alpha+=temp_p[i];
                        temp_p[i]=0;
                        temp_a[i]=0;
                        temp_b[i]=0;
                    }
                    else if(temp_a[i]==0){
                        if(temp_p[i]!=0){
                            temp_alpha+=temp_p[i];
                            temp=temp_b[i];
                            temp_b[i]-=temp_p[i];
                            temp_p[i]=0;
                            for(int u=0;u<temp_b[i];u++){
                                temp_coef=temp_coef*(temp-u)/(u+1);
                            }
                        }
                    }
                    else if(temp_b[i]==0){
                        if(temp_p[i]!=0){
                            temp_alpha+=temp_p[i];
                            temp=temp_a[i];
                            temp_a[i]-=temp_p[i];
                            temp_p[i]=0;
                            for(int u=0;u<temp_a[i];u++){
                                temp_coef=temp_coef*(temp-u)/(u+1);
                            }
                        }
                    }
                    else if(temp_p[i]==temp_a[i]+temp_b[i]-1){
                        if(temp_a[i]>temp_b[i]){
                            temp=GCD(temp_a[i],temp_b[i]);
                        }
                        else{
                            temp=GCD(temp_b[i],temp_a[i]);
                        }
                        if(temp>1){
                            temp_coef*=temp;
                            temp_a[i]=temp_a[i]/temp;
                            temp_b[i]=temp_b[i]/temp;
                            temp_alpha=temp_alpha+(temp-1)*(temp_a[i]+temp_b[i]);
                            temp_p[i]=temp_a[i]+temp_b[i]-1;
                        }
                    }
                }
                temp_expression_code=0;
                temp_expression_code+=temp_alpha;
                for(int i=0;i<3;i++){
                    temp_expression_code=temp_expression_code*MAX_P+temp_p[i];
                    temp_expression_code=temp_expression_code*MAX_AMP+temp_a[i];
                    temp_expression_code=temp_expression_code*MAX_AMP+temp_b[i];
                }
                if(temp_alpha==0){
                    (*delta_mark_array)[index].init(0,\
                                                np,na,nb,\
                                                lp,la,lb,\
                                                mp,ma,mb,\
                                                temp_expression_code,\
                                                temp_coef);
                }
                else{
                    (*delta_mark_array)[index].init(1,\
                                                np,na,nb,\
                                                lp,la,lb,\
                                                mp,ma,mb,\
                                                temp_expression_code,\
                                                temp_coef);
                    (*expression_encode_list_sum)=array_push(expression_encode_list,temp_expression_code,temp_coef,*expression_encode_list_sum);
                }
                index++;
            }
        }
    }
    return index;
};
void CSE_generate_delta_list(int ipq,int aam,int bam){
    int index=0;
    int ia2,ib2;
    int sa=(aam+2)*(aam+1)/2;
    int sb=(bam+2)*(bam+1)/2;
    for(ia2=0;ia2<sa;ia2++){
        for(ib2=0;ib2<sb;ib2++){
            index=CSE_setup_delta_list(ipq,index,aam,bam,ia2,ib2);
        }
    }
}
void CSE_finish(){
        delete(P_expression_encode_list);
        delete(Q_expression_encode_list);
        delete(P_delta_mark_array);
        delete(Q_delta_mark_array);
        P_delta_mark_sum=0;
        Q_delta_mark_sum=0;
        P_expression_encode_list_sum=0;
        Q_expression_encode_list_sum=0;
}

