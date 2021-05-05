#ifndef EXPRESSION_H_INCLUDED
#define EXPRESSION_H_INCLUDED

#include<cstdio>

#ifndef MAX_AM
#define MAX_AM 4
#define MAX_AMP (MAX_AM+1)
#define MAX_P (MAX_AM*2+1)
#define MAX_J (MAX_AM*4+1)
#endif // MAX_AM

class delta_mark{
public:
    int status;//0 unique. 1 equal to next. 2 next*coef
    int id[9];//np,na,nb,lp,la,lb,mp,ma,mb
    int expression_code;
    int coef;
    delta_mark(){
    	for(int i=0;i<9;i++){
			id[i]=0;
    	}
    	expression_code=0;
        status=-1;
        coef=0;
    };
    void init(int status_in,\
			  int np,int na,int nb,\
			  int lp,int la,int lb,\
			  int mp,int ma,int mb,\
			  int expression_code_in,\
              int coef_in){
        status=status_in;
        id[0]=np;
        id[1]=na;
        id[2]=nb;
        id[3]=lp;
        id[4]=la;
        id[5]=lb;
        id[6]=mp;
        id[7]=ma;
        id[8]=mb;
        expression_code=expression_code_in;
        coef=coef_in;

    }
    void print(){
        int temp_expression_coed=expression_code;
    	for(int i=0;i<9;i++){
			printf("%d",id[i]);
    	}
        printf(" : %d\n\t%d,",status,coef);
        for(int i=0;i<3;i++){
			printf("%d",temp_expression_coed%MAX_AMP);
			temp_expression_coed=temp_expression_coed/MAX_AMP;
			printf("%d",temp_expression_coed%MAX_AMP);
			temp_expression_coed=temp_expression_coed/MAX_AMP;
			printf("%d",temp_expression_coed%MAX_P);
			temp_expression_coed=temp_expression_coed/MAX_P;
        }
        printf("_a%d\n",temp_expression_coed);
    };
    void print_expression(FILE * fp, char PQ){
        fprintf(fp,"%c_",PQ);
        for(int i=0;i<9;i++)
            fprintf(fp,"%d",id[i]);
    };
};

class expression_encode{
public:
    int expression_code;
    int coef_sum;
    int coef[20];
    int exp_code[10];//{alpha,np,na,nb,lp,la,lb,mp,ma,mb}
    expression_encode(){
        expression_code=0;
        coef_sum=0;
        for(int i=0;i<20;i++){
            coef[i]=0;
        }
        for(int i=0;i<10;i++){
            exp_code[i]=0;
        }
    }
    void init(int expression_code_in, int coef_in){
        expression_code=expression_code_in;
        int temp_expression_coed=expression_code;
        if(coef_in!=1){
            coef[coef_sum]=coef_in;
            coef_sum++;
        }
        for(int i=0;i<3;i++){
            exp_code[9-i*3]=temp_expression_coed%MAX_AMP;
            temp_expression_coed=temp_expression_coed/MAX_AMP;
            exp_code[8-i*3]=temp_expression_coed%MAX_AMP;
            temp_expression_coed=temp_expression_coed/MAX_AMP;
            exp_code[7-i*3]=temp_expression_coed%MAX_P;
            temp_expression_coed=temp_expression_coed/MAX_P;
        }
        exp_code[0]=temp_expression_coed;
    }
    bool add(int coef_in){
        if(coef_sum>=20){
            return false;
        }
        if(coef_in!=1){
            for(int i=0;i<coef_sum;i++){
                if(coef[i]==coef_in){
                    return true;
                }
            }
            coef[coef_sum]=coef_in;
            coef_sum++;
        }
        return true;
    }
};


#endif // EXPRESSION_H_INCLUDED
