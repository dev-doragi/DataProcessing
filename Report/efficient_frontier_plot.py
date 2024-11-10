import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import inv    
import scipy.optimize as sco
from scipy.optimize import minimize


# 효율적 프론티어 그리기   

# 포트폴리오 기대수익률 계산

def get_portf_rtn(w, avg_rtns):    
    return np.sum(avg_rtns * w)
    
# 포트폴리오 변동성 계산

def get_portf_vol(w, cov_mat):  
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

# 효율적 프론티어 산출 (포트폴리오수익률, 공분산행렬, 기대수익률 범위)

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):  
    
    efficient_portfolios = []
    
    n_assets = len(avg_rtns)   # 자산갯수
    args = (cov_mat) # 함수에 들어갈 인수 정의
    
    bounds = tuple((0.0,1) for asset in range(n_assets))  # 자산별 비중 제약조건 설정
    initial_guess = n_assets * [1. / n_assets, ]          # 초기값 0.2 씩 5개 자산에 배정 
    
    for ret in rtns_range:  # 기대수익률별 최적투자비중을 산출 
        
        constraints = ({'type': 'eq', 
                        'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret}, # 포트기대수익률이 나오게 하는 X(W) 구하기
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 자산별 비중합은 1
        efficient_portfolio = minimize(get_portf_vol, initial_guess, 
                                           args=args, method='SLSQP', 
                                           constraints=constraints,
                                           bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)
    
    return efficient_portfolios

# 프론티어를 구성하는 변동성, 기대수익률, 자산별 투자비중 값 구기기

def get_efficient_frontier_value(avg_rtns, cov_mat,nums): # nums: 기대수익률 갯수
    
    # 효율적 프론티어에 표현할 기대수익률의 범위 설정
    rtns_range=  np.linspace(min(avg_rtns),max(avg_rtns), nums)  
    
    # 효율적 프론티어 함수 실행 
    efficient_portfolios = get_efficient_frontier(avg_rtns,cov_mat,rtns_range)
    vols_range = [x['fun'] for x in efficient_portfolios] # 포트폴리오 변동성 추출
    weight_range = [x['x'] for x in efficient_portfolios] # 자산별 최적비중 추출

    # 위 함수에서 계산된 값들을  데이터 프레임화
    
    pvx=pd.DataFrame(vols_range)  #  변동성 범위
    prt=pd.DataFrame(rtns_range)  #  기대수익률 범위
    pw=pd.DataFrame(weight_range) #  자산별 비중
       
    # 위 자료 합치기 
    portfolio_result_df=pd.concat([pvx,prt,pw],axis=1,join='outer')  
      
   # 리스크 최소값에 해당하는 기대수익률 찾기  (프론티어 시작점)
    exp_ret= portfolio_result_df[portfolio_result_df.iloc[:,0]==portfolio_result_df.iloc[:,0].min()].iloc[:,1].min()
    port_result = portfolio_result_df[portfolio_result_df.iloc[:,1]>= exp_ret]  # 프론티어 도출

    return round( port_result,4)
  


