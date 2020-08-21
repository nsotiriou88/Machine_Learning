# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

def eva_dfrocpr(df):
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    dfrocpr = df.sort_values('pred')\
      .groupby('pred')['label'].agg([n0,n1,len])\
      .reset_index().rename(columns={'n0':'countN','n1':'countP','len':'countpred'})\
      .assign(
        FN = lambda x: np.cumsum(x.countP), 
        TN = lambda x: np.cumsum(x.countN) 
      ).assign(
        TP = lambda x: sum(x.countP) - x.FN, 
        FP = lambda x: sum(x.countN) - x.TN
      ).assign(
        TPR = lambda x: x.TP/(x.TP+x.FN), 
        FPR = lambda x: x.FP/(x.TN+x.FP), 
        precision = lambda x: x.TP/(x.TP+x.FP), 
        recall = lambda x: x.TP/(x.TP+x.FN)
      ).assign(
        F1 = lambda x: 2*x.precision*x.recall/(x.precision+x.recall)
      )
    return dfrocpr

def perf_eva(label, pred, title=None, groupnum=None, plot_type=["ks", "roc"], show_plot=True, positive="bad|1", seed=186):
    '''
    KS, ROC, Lift, PR
    ------
    perf_eva provides performance evaluations, such as 
    kolmogorov-smirnow(ks), ROC, lift and precision-recall curves, 
    based on provided label and predicted probability values.
    
    Params
    ------
    label: Label values, such as 0s and 1s, 0 represent for good 
      and 1 for bad.
    pred: Predicted probability or score.
    title: Title of plot, default is "performance".
    groupnum: The group number when calculating KS.  Default NULL, 
      which means the number of sample size.
    plot_type: Types of performance plot, such as "ks", "lift", "roc", "pr". 
      Default c("ks", "roc").
    show_plot: Logical value, default is TRUE. It means whether to show plot.
    positive: Value of positive class, default is "bad|1".
    seed: Integer, default is 186. The specify seed is used for random sorting data.
    
    Returns
    ------
    dict
        ks, auc, gini values, and figure objects
    
    Details
    ------
    Accuracy = 
        true positive and true negative/total cases
    Error rate = 
        false positive and false negative/total cases
    TPR, True Positive Rate(Recall or Sensitivity) = 
        true positive/total actual positive
    PPV, Positive Predicted Value(Precision) = 
        true positive/total predicted positive
    TNR, True Negative Rate(Specificity) = 
        true negative/total actual negative
    NPV, Negative Predicted Value = 
        true negative/total predicted negative
        
    Examples
    ------
    import scorecardpy
    
    # load data
    dat = sc.germancredit()
    
    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")
    
    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)
    
    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']
    
    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X, y)
    
    # predicted proability
    dt_pred = lr.predict_proba(X)[:,1]
    # performace ------
    # Example I # only ks & auc values
    sc.perf_eva(y, dt_pred, show_plot=False)
    
    # Example II # ks & roc plot
    sc.perf_eva(y, dt_pred)
    
    # Example III # ks, lift, roc & pr plot
    sc.perf_eva(y, dt_pred, plot_type = ["ks","lift","roc","pr"])
    '''
    
    # inputs checking
    if len(label) != len(pred):
        warnings.warn('Incorrect inputs; label and pred should be list with the same length.')
    # if pred is score
    if np.mean(pred) < 0 or np.mean(pred) > 1:
        warnings.warn('Since the average of pred is not in [0,1], it is treated as predicted score but not probability.')
        pred = -pred
    # random sort datatable
    df = pd.DataFrame({'label':label, 'pred':pred}).sample(frac=1, random_state=seed)
    # remove NAs
    if any(np.unique(df.isna())):
        warnings.warn('The NANs in \'label\' or \'pred\' were removed.')
        df = df.dropna()
    # check label
    df = check_y(df, 'label', positive)
    # title
    title='' if title is None else str(title)+': '
    
    ### data ###
    # dfkslift ------
    if any([i in plot_type for i in ['ks', 'lift']]):
        dfkslift = eva_dfkslift(df, groupnum)
        if 'ks' in plot_type: df_ks = dfkslift
        if 'lift' in plot_type: df_lift = dfkslift
    # dfrocpr ------
    if any([i in plot_type for i in ["roc","pr",'f1']]):
        dfrocpr = eva_dfrocpr(df)
        if 'roc' in plot_type: df_roc = dfrocpr
        if 'pr' in plot_type: df_pr = dfrocpr
        if 'f1' in plot_type: df_f1 = dfrocpr
    ### return list ### 
    rt = {}
    # plot, KS ------
    if 'ks' in plot_type:
        rt['KS'] = round(dfkslift.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)
    # plot, ROC ------
    if 'roc' in plot_type:
        auc = pd.concat(
          [dfrocpr[['FPR','TPR']], pd.DataFrame({'FPR':[0,1], 'TPR':[0,1]})], 
          ignore_index=True).sort_values(['FPR','TPR'])\
          .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
          ).assign(
            auc=lambda x: (x.TPR+x.TPR_lag)*(x.FPR-x.FPR_lag)/2
          )['auc'].sum()
        ### 
        rt['AUC'] = round(auc, 4)
        rt['Gini'] = round(2*auc-1, 4)
    
    ### export plot ### 
    if show_plot:
        plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
        subplot_nrows = np.ceil(len(plist)/2)
        subplot_ncols = np.ceil(len(plist)/subplot_nrows)
        
        fig = plt.figure()
        for i in np.arange(len(plist)):
            plt.subplot(subplot_nrows,subplot_ncols,i+1)
            eval(plist[i])
        plt.show()
        rt['pic'] = fig
    # return 
    return rt