#!/usr/bin/env python3u
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:16:40 2020

@author: jameshuang
"""

import pandas as pd   
from scipy.interpolate import CubicSpline
from scipy import stats 
import matplotlib.pyplot as plt
import numpy as np
import os 
import time
import random


def hazard(h,t):
    return 1-np.exp(-h*t)
def D(t):
    r=0.001;
    return np.exp(-r*t);

def basket_defaultswap(R,n,tau,T):
    Time=np.arange(0.5,T+0.5,0.5); #semi pmt
    R=[0.3,0.1,0.2,0.1,0.3,0.1,0.2,0.2,0.1,0.3];
    s=10;
    R=R[n-1];
    V_val=(1-R)*D(tau)*(1 if tau<=T else 0);
    temp=0;
    for i,tm in enumerate(Time):
        if tau>T:
            for tm2 in Time:
                temp+=s*D(tm2);
            break;
        if tm <=tau: #tau<Ti
            temp+=s*D(tm);
        else: #Ti<tau
            temp+=s*D(tau)*(tau-Time[i-1])/(tm-Time[i-1]);
            break;
    V_prot=temp;
    return V_val-V_prot



def CP(A,T,n):

    #A_read=pd.read_csv('A_mat');
    #A_array=np.array(A_read);
    #A=A_array;
    N=len(A);
    risk_factor=len(A[0]);
    

    
    B=[];
    for i in range(N):
        #A[i,:]=A[i,:]/sumA[i];
        B.append(np.sqrt(1-np.sum(A[i,:]**2)));
    
    Z=np.random.normal(0,1,risk_factor);
    
    h=[0.05,0.01,0.02,0.02,0.03,0.1,0.03,0.09,0.1,0.05];
    h=np.array(h);
    if N!=10:
        h=h[:N];

    p=[];p_Z=[];
    for i in range(N): 
        p.append(hazard(h[i],T));
        
        bi=B[i];
        temp_mu= np.sum( A[i,:]*Z );
        temp= stats.norm.cdf( (stats.norm.ppf(p[i]) - temp_mu) /bi);
        p_Z.append(temp);
     
    #######
    ar = [];
    for i in range(N+1):
        arr=[]; 
        for j in range(i+1):
            arr.append(-1);
        ar.append(arr);  
    
    #N+1
    for i in range(N+1): #N+1
        if i<=n-1:
            ar[N][i]=0;
        else:
            ar[N][i]=1;
    #N 
    for i in range(N-1,-1,-1): #|
        for k in range(i,-1,-1): #-
            ar[i][k]=p[i]*ar[i+1][k+1] + (1-p[i])*ar[i+1][k];

    q_ar = [];
    for i in range(N):
        q_arr=[]; 
        for j in range(i+1):
            q_arr.append(-1);
        q_ar.append(q_arr); 
    
    for i in range(N-1,-1,-1): #|
        for k in range(i,-1,-1): #-
            if ar[i][k]==0:
                q_ar[i][k]=0;
            else: 
                q_ar[i][k]=p[i]*ar[i+1][k+1]/ar[i][k];

    V=np.random.uniform(0,1,N);
    k=0;
    Y=[];U=[];L=[];L_check=[];
    q_Z=[];
    over_reached=[];
    for i in range(N):  
        q_Z.append(q_ar[i][k]);
        if V[i]<= q_Z[i]:
            Y.append(1);
            k=k+1;
            tmp= V[i]*p_Z[i]/q_Z[i];
            U.append(tmp);
            L.append(p_Z[i]/q_Z[i]);
        else:
            Y.append(0);
            tmp= p_Z[i] + (V[i]-q_Z[i])*(1-p_Z[i])/(1-q_Z[i]);
            U.append(tmp);
            L.append((1-p_Z[i])/(1-q_Z[i]));
        
        #if sum(Y)>=n and i==N-1:
            #print("{} defaults but simulate {}".format(n,sum(Y)));
            #over_reached=Y;        
        
        L_check.append(ar[i][int(np.sum(Y[:i]))]/ar[i+1][int(np.sum(Y[:i+1])) ] ); #independence case.is const.
             
    W=[];Tau=[];
    for i in range(N):
        W.append( np.sum( A[i,:]*Z )+B[i]*stats.norm.ppf(U[i]) );
    
    temp=-np.log(1-stats.norm.cdf(W))/h;  #inverse F, hazard rate is const.
    Tau=temp;
    
    #print('L<1',L);
    L=np.prod(L);


   
    
    V_return =(10*n)*L; #(100*np.sum(Y))*L;

    return V_return;
    
def JK(A,T,n):

    #A_read=pd.read_csv('A_mat');
    #A_array=np.array(A_read);
    #A=A_array;
    N=len(A);
    
    risk_factor=len(A[0]);
    B=[];
    for i in range(N):
        #A[i,:]=A[i,:]/sumA[i];
        B.append([np.sqrt(1-np.sum(A[i,:]**2))]);
        
    Sigma= np.matmul(A,np.transpose(A))+np.matmul(B,np.transpose(B));
    C=np.linalg.cholesky(Sigma);
    #pd.DataFrame(A).to_csv("A_mat",index=None);
    

    Z=np.random.normal(0,1,N);
    h=[0.05,0.01,0.02,0.02,0.03,0.1,0.03,0.09,0.1,0.05];
    h=np.array(h);
    if N!=10:
        h=h[:N];

    p=[]; #independent case
    for i in range(N): 
        p.append(hazard(h[i],T)); #independent case
        
    p_condi=[]; #dependent case
    for i in range(N):
        if i==0:
            p_condi.append(p[i]);
           
         
        else:
            temp1=0;
            temp1 = sum( C[i,:i]*Z[:i] );               
            W_i= ( stats.norm.ppf(p[i])- temp1 )/C[i,i];
            p_condi.append(stats.norm.cdf(W_i));

    V=np.random.uniform(0,1,N);
    k=0;
    Y=[];U=[];L=[];p_Z=[]; 
    p=p_condi;

    for i in range(N):
        if i==0:
            #p_Z.append(n/N);  #JK prob
            p_Z.append( max (n/N,p[i]) );  #modified JK prob
            if V[i]<=p_Z[i]:
                Y.append(1);
                k=1;
            else:
                Y.append(0);
                k=0;
                 
        else:
            #p_Z.append((n-k)/(N-(i+1)+1) if k<n else p[i]);  #JK prob
            p_Z.append( max( (n-k)/(N-(i+1)+1),p[i] ) );  #modified JK prob
            if V[i]<=p_Z[i]:
                Y.append(1);
                k=k+1;
            else:
                Y.append(0);
                

        if Y[i]==1:
            tmp= V[i]*p[i]/p_Z[i];
            U.append(tmp);
            L.append(p[i]/p_Z[i]);
        else:
            tmp= p[i] + (V[i]-p_Z[i])*(1-p[i])/(1-p_Z[i]);
            U.append(tmp);
            L.append((1-p[i])/(1-p_Z[i]));
        
        
    W=[];Tau=[];
    for i in range(N):
        if i==0:
           temp = stats.norm.cdf( C[i,i]*stats.norm.ppf(U[i]) );
           W.append(temp);
        else:
           temp1=0;
           for j in range(i): 
               temp1 += np.sum( C[i,j]*Z[j] );  #Z_bar = inverse(U)
               
           temp2 = C[i,i]*stats.norm.ppf(U[i]);
           W.append(temp1+temp2);
        
    
    temp=-np.log(1-stats.norm.cdf(W))/h;  #inverse F, hazard rate is const.
    Tau=temp;
    
    L=np.prod(L);       
    V_return=(10*n)*L;  #(100*np.sum(Y))*L;
    
    return V_return; 

def MC(A,T,n):
    
    #A_read=pd.read_csv('A_mat');
    #A_array=np.array(A_read);
    #A=A_array;
    N=len(A);
    risk_factor=len(A[0]);

    
    U=np.random.uniform(0,1,N);
    Z=stats.norm.ppf(U);
    B=[];
    for i in range(N):
        #A[i,:]=A[i,:]/sumA[i];
        B.append([np.sqrt(1-np.sum(A[i,:]**2))]);
    
    h=[0.05,0.01,0.02,0.02,0.03,0.1,0.03,0.09,0.1,0.05];
    h=np.array(h);
    if N!=10:
        h=h[:N];

        
    Sigma= np.matmul(A,np.transpose(A))+np.matmul(B,np.transpose(B));
    C=np.linalg.cholesky(Sigma);
    W=np.matmul(C,np.transpose(Z));
    Tau=-np.log(1-stats.norm.cdf(W))/h;
    I= 1 if np.sum(Tau<=T)>=n  else 0;
    #I= 1 if np.sum(Tau<=T)<n  else 0;
    #print(I,np.sum(Tau<=T));
    
    V_return = (10*n)*I;

    
    return V_return;


if __name__ == '__main__':
    n=3; #nth default
    A_read=pd.read_csv('A_mat');
    #A_read=pd.read_csv('A7_mat');
    #A_read=pd.read_csv('A10_mat.csv');
    A_read=A_read.dropna(axis=1);
    A_array=np.array(A_read);
    A=A_array;
    #A3=abs(A);
    #A=A3;
    print();
 
    Nsim=10;
    JK_Nsim=[];
    CP_Nsim=[];
    MC_Nsim=[];
    #for i in range(Nsim):
       #JK_Nsim.append( JK(A,10,n) );
       #CP_Nsim.append( CP(A,10,n) );
       #MC_Nsim.append( MC(A,10,n) );

    #print('{:>20} {:>20} {:>20}'.format('JK_mean','CP_mean','MC_mean'));
    #print('{:12} {:>12} {:>12}'.format(np.mean(JK_Nsim),np.mean(CP_Nsim),np.mean(MC_Nsim)));
    #print('{:>20} {:>20} {:>20}'.format('JK_var','CP_var','MC_var'));
    #print('{:>12} {:>12} {:>12}'.format(np.var(JK_Nsim),np.var(CP_Nsim),np.var(MC_Nsim)));
    
    #plt.hist(CP_Nsim,label='CP');
    #plt.hist(MC_Nsim,label='MC');
    #plt.hist(JK_Nsim,label='JK');
    #plt.legend();plt.grid();
    
    

    MAT=np.arange(0.01,20,1);
    for i in range(n,n+1):
        JK_Nsim=[];JK_mat=[];
        CP_Nsim=[];CP_mat=[];
        MC_Nsim=[];MC_mat=[]; 
        for mat in MAT:
            for nsim in range(Nsim):
                CP_Nsim.append( CP(A,mat,i) );
                JK_Nsim.append( JK(A,mat,i) );
                MC_Nsim.append( MC(A,mat,i) );
            JK_mat.append(np.var(JK_Nsim));
            CP_mat.append(np.var(CP_Nsim));
            MC_mat.append(np.var(MC_Nsim));


        plt.plot(MAT,JK_mat,label='JK');
        plt.plot(MAT,CP_mat,label='CP');
        plt.plot(MAT,MC_mat,label='MC');
        plt.legend();plt.grid();
        plt.xlabel('matruty T')
        plt.ylabel('variance')
        plt.title('%d Assets %dth default. Nsim %d' % (len(A),i,Nsim));
        #plt.savefig('N%dD%d_N%d.png' % (len(A),i,Nsim));
        #plt.clf();


    





