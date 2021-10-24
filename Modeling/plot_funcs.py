# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:08:38 2020

@author: Nicholas Sotiriou - github: @nsotiriou88 // nsotiriou88@gmail.com
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as plty
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def missing_data(data):
    '''
    Print in Dataframe format, missing data from datasets.
    '''
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types

    return(np.transpose(tt))


def plot_feature_scatter(df1, df2, features, grid=(5,2)):
    '''
    Scatter plots to identify correletion in features.
    '''
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(grid[0],grid[1],figsize=(14,20))

    for feature in features:
        i += 1
        plt.subplot(grid[0],grid[1],i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show()

    return


def plot_feature_categ(df1, feature, target, order=None):
    '''
    Plot frequency bars for categorical features bwtween two classes.
    '''
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,1,figsize=(14,14))
    
    datasets =[(df1.loc[df1[target] == 0], 'target 0')]
    datasets.append((df1.loc[df1[target] == 1], 'target 1'))
    
    if order==None:
        order = df1[feature].value_counts().index

    for dataset in datasets:
        i += 1
        plt.subplot(2,1,i)
        sns.countplot(x=feature, data=dataset[0], order=order)
        plt.title(dataset[1], fontsize=16)
        plt.xlabel(feature, fontsize=12)
        plt.xticks(rotation=45)
    plt.show()

    return


def plot_feature_categ_single_graph(df1, feature, target):
    '''
    Plot frequency bars for categorical features bwtween two classes in one graph.
    '''
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    plt.subplot(1,1,1)
    
    df_counts = (df1.groupby([target])[feature]
                         .value_counts(normalize=True)
                         .rename('percentage')
                         .mul(100)
                         .reset_index()
                         .sort_values(target))
    
    p = sns.barplot(x=feature, y='percentage', hue=target, data=df_counts)
    plt.title(feature+' count plots grouped by target', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

    return


def corr_heatmap_plot(df, cmap='coolwarm'):
    '''
    Heatmap to visualise the correlation between variables.
    '''
    sns.set_style('white')
    corr = df.corr()
    fig = plt.figure(figsize=(12, 12), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(corr.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    plt.show()

    return


def plot_feature_distribution(df1, df2, label1, label2, features, grid=(2, 5)):
    '''
    Distribution plots for features. Can be used to compare different
    classes in same dataframe or different dataframes.
    '''
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(grid[0],grid[1],figsize=(16,7))

    for feature in features:
        i += 1
        plt.subplot(grid[0],grid[1],i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()

    return


def plot_new_feature_distribution(df1, df2, label1, label2, features, grid=(2, 4)):
    '''
    Distribution plots for features. Can be used to compare different
    classes in same dataframe or different dataframes (different grid).
    '''
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(grid[0],grid[1],figsize=(18,8))

    for feature in features:
        i += 1
        plt.subplot(grid[0],grid[1],i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show()

    return


def confusion_mat_plot(y_test, y_pred):
    '''
    Visualise confusion matrix and other metrics.
    
    Returns tuple of (accuracy, precision, recall, specificity, f_score)
    '''
    sns.set_style('white')
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    hist, xbins, ybins, im = ax.hist2d([0,0,1,1], [0,1,0,1], bins=2,
                                       weights=[cm[0,0],cm[0,1],cm[1,0],cm[1,1]],
                                       cmin=0, cmax=np.max(cm), cmap='PuBu')
    plt.title('Confusion Matrix')
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            ax.text(xbins[i]+0.25,ybins[j]+0.25,int(hist[i,j]),color='black',
            ha='center',va='center', fontweight='bold')
    ax.set_xticks([0.25, 0.75])
    ax.set_yticks([0.25, 0.75])
    ax.set_xticklabels(['Negative','Positive'])
    ax.set_yticklabels(['Negative','Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    fig.colorbar(im)
    plt.show()
    
    total = np.sum(cm)
    # Accuracy
    accuracy = (tp+tn)/total*100
    print('accuracy:', round(accuracy, 2), '%')

    # Precision
    precision = tp/(tp+fp)*100
    print('precision:', round(precision, 2), '%')

    # Recall/Sensitivity/TPR
    recall = tp/(tp+fn)*100
    print('Recall:', round(recall, 2), '%')

    # Specificity
    specificity = tn/(tn+fp)*100
    print('specificity:', round(specificity, 2), '%')

    # F-Score
    f_score = 2*recall*precision/(recall+precision)
    print('F-score:', round(f_score, 2), '%')
    
    # fpr = fp/(fp+tn) - false-positive rate
    
    return (accuracy, precision, recall, specificity, f_score)


def plot_confusion_matrix_slider(y_test, y_proba, cmap='Blues', savefig=False):
    '''
    Interactive plot for confusion matrix on different cut-off points with
    slider. Uses Plotly library on Jupyter notebook (need Javascript on
    background).
    '''
    sns.set_style('whitegrid')
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=('Confusion Matrix', 'ML Metrics'),
                        horizontal_spacing=0.13)
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F-Score']
    pred_labels = ['Negative', 'Positive']

    for cutoff in np.arange(0.2, 0.825, 0.025):
        metrics = [0, 0, 0, 0, 0]
        y_pred = np.where(y_proba < cutoff, 0, 1)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total = np.sum(cm)
        metrics[0] = round((tp+tn)/total*100, 1) # Accuracy
        metrics[1] = round(tp/(tp+fp)*100, 1) # Precision
        metrics[2] = round(tp/(tp+fn)*100, 1) # Recall/Sensitivity/TPR
        metrics[3] = round(tn/(tn+fp)*100, 1) # Specificity
        metrics[4] = round(2*metrics[1]*metrics[2]/(metrics[1]+metrics[2]), 1) # F-Score
    
        #add bar plot for metrics
        fig.add_trace(go.Bar(x=metrics, y=metrics_labels, text=metrics, visible=False,
                            textposition='inside', showlegend=False, orientation='h',
                            marker=dict(color='rgba(158, 202, 225, 0.6)',
                            line=dict(color='rgba(8, 48, 107, 0.8)', width=1.5))),
                            row=1, col=1)

        # add heatmap/cm plots
        z_data = [[tn,fp], [fn,tp]]
        fig.add_trace(go.Heatmap(x=pred_labels, y=pred_labels,
                    z=z_data, colorscale=cmap, hoverinfo='z',
                    visible=False), row=1, col=2)
        
        #annotate heatmap
        fig.add_trace(go.Scatter(text=[str(cm[0,0]), str(cm[0,1]), str(cm[1,0]),
                    str(cm[1,1])], visible=False,
                    x=['Negative', 'Positive', 'Negative', 'Positive'],
                    y=['Negative', 'Negative', 'Positive', 'Positive'],
                    mode='text', textfont=dict(size=16, color='black'),
                    showlegend=False), row=1, col=2)
    
    # make one step visible (n models in total)
    # 12+1th element is the 50% cut-off
    plots = 3 # (Bar plots, heatmap and annotation scatterplot for heatmap)
    for j in range(plots):
        fig.data[12*plots+j].visible = True
    
    # Create and add slider
    cutoffs = []
    for i in range(int(len(fig.data)/plots)):
        cutoff = dict(method='restyle', args=['visible', [False]*len(fig.data)],
                      label=str(20+i*2.5)+'%',)
        for j in range(plots):
            cutoff['args'][1][i*plots+j] = True # Toggle (i*6+j)'th trace to "visible"
        cutoffs.append(cutoff)
    
    sliders = [dict(active=12, currentvalue={'prefix': 'Cut-off: '},
                    pad={'t': 50}, steps=cutoffs)]
    
    fig.update_layout(sliders=sliders, showlegend=True,
                      title='Model Comparison on Different Cut-offs')
    
    # updated axes labels
    fig.update_xaxes(tickvals=[0, 20, 40, 60, 80, 100], range=[0,100], row=1, col=1)
    fig.update_xaxes(title_text='Predicted Label', row=1, col=2)
    fig.update_yaxes(title_text='True Label', row=1, col=2)
    
    # Show and save figure
    fig.show()
    if savefig==True:
        plty.plot(fig, filename='confmat_slider.html', auto_open=False)

    return


def auc_plot(X_test, y_test, clf, w=None):
    '''
    AUC curve plot and Gini metric.
    
    Returns Gini.
    '''
    sns.set_style('white')
    plt.figure(figsize=(10, 8))
    y_pred_proba = clf.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, sample_weight=w)
    AUC = roc_auc_score(y_test, y_pred_proba, sample_weight=w)
    plt.plot(fpr, tpr, 'r-', label='ROC curve, area='+str(round(AUC, 2)))
    plt.plot([0,1], [0,1], 'c--')
    plt.legend(loc=4)
    plt.show()
    
    gini = (2*AUC-1)*100
    print('gini', round(gini, 3), '%')
    
    return gini


def auc_plot_all(X_test, y_test, clf, models, w=None):
    '''
    Plot all AUC from different models in one graph.
    
    Parameters
    ----------
    X_test: list of all X_test datasets for prediction.
    
    y_test: list of all y_test datasets.
    
    clf: list of trained  classifiers.
    
    models: list of strings with a description about the model.
    
    w: list of sample weights, if applicable to the model.
    For mixed models, use [w1, None, w3, ...].
    '''
    if w==None:
        w = [None for i in range(len(models))]
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(16,16))
    
    for i in range(len(models)):
        y_pred_proba = clf[i].predict_proba(X_test[i])[::,1]
        fpr, tpr, _ = roc_curve(y_test[i], y_pred_proba, sample_weight=w[i])
        AUC = roc_auc_score(y_test[i], y_pred_proba, sample_weight=w[i])
        plt.plot(fpr, tpr, linestyle='-', label=models[i]+'-AUC='+str(round(AUC,2)))
    
    plt.plot([0,1], [0,1], 'c--')
    plt.title('ROC Curve for All Models', fontsize=22)
    plt.legend(loc=4)
    plt.show()

    return


def plot_models_radar(X_list, y_list, clf_list, models, w_list=None,
    cutoff=0.5, fill=True, figsize=(12,12)):
    '''
    Plot of all model metrics in one radar graph for 50% cut-off.
    '''
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    if w_list==None:
        w_list = [None for i in range(len(models))]
    theta_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                    'F-Score']
    theta = np.linspace(0, 2*np.pi, len(theta_labels), endpoint=False).tolist()
    theta += theta[:1]
    if cutoff > 1 or cutoff < 0:
        cutoff = 0.5
    cutoff_str = str(round(cutoff*100,0))

    for i, model in enumerate(models):
        metrics = [0,0,0,0,0,0]
        #calculate gini
        y_pred_proba = clf_list[i].predict_proba(X_list[i])[::,1]
        fpr, tpr, _ = roc_curve(y_list[i], y_pred_proba, sample_weight=w_list[i])
        AUC = roc_auc_score(y_list[i], y_pred_proba, sample_weight=w_list[i])
        Gini = (2*AUC-1)*100

        #calculate other metrics based on cut-off point
        y_pred = np.where(y_pred_proba < cutoff, 0, 1)
        cm = confusion_matrix(y_list[i], y_pred)
        tn, fp, fn, tp = confusion_matrix(y_list[i], y_pred).ravel()
        total = np.sum(cm)
        metrics[0] = (tp+tn)/total*100 # Accuracy
        metrics[1] = tp/(tp+fp)*100 # Precision
        metrics[2] = tp/(tp+fn)*100 # Recall/Sensitivity/TPR
        metrics[3] = tn/(tn+fp)*100 # Specificity
        metrics[4] = 2*metrics[1]*metrics[2]/(metrics[1]+metrics[2]) # F-Score
        metrics[5] = metrics[0]

        # add radar plots
        ax.plot(theta, metrics, linewidth=2, label=model+'(Gini:'+str(round(Gini,1))+'%)')
        if fill:
            ax.fill(theta, metrics, alpha=0.2)
    
    # Fix axis to go in the right order and start from 12 o'clock
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    theta_labels.append('Accuracy')
    ax.set_thetagrids(np.degrees(theta), theta_labels)
    ax.set_ylim(0, 100)
    ax.set_rlabel_position(180/len(theta_labels))
    plt.title('Model Comparison - '+cutoff_str+'% cut-off point', fontsize=22)
    plt.legend(loc='lower right')
    plt.show()

    return


def plot_model_metrics_radar_slider(X_list, y_list, clf_list, models, w_list=None,
    savefig=False):
    '''
    Interactive plot for metrics and Gini with slider for different
    cut-off points. Works with Plotly and Javascript (in Jupyter
    Notebooks).

    Parameters
    ----------
    X_list: list of all X_test datasets for prediction.
    
    y_list: list of all y_test datasets.
    
    clf_list: list of trained  classifiers.
    
    models: list of strings with a description about the model.
    
    w_list: list of sample weights, if applicable to the model.
    For mixed models, use [w1, None, w3, ...].
    '''
    sns.set_style('whitegrid')
    fig = go.Figure()

    if w_list==None:
        w_list = [None for i in range(len(models))]
    theta_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                    'F-Score', 'Accuracy']
    color_pallete = ['peru', 'darkviolet', 'deepskyblue', 'black', 'yellow'
                     'red', 'green', 'blue']
    
    for cutoff in np.arange(0.2, 0.825, 0.025):
        for i, model in enumerate(models):
            metrics = [0, 0, 0, 0, 0, 0]
            # calculate gini
            y_pred_proba = clf_list[i].predict_proba(X_list[i])[::, 1]
            fpr, tpr, _ = roc_curve(y_list[i], y_pred_proba, sample_weight=w_list[i])
            AUC = roc_auc_score(y_list[i], y_pred_proba, sample_weight=w_list[i])
            Gini = (2*AUC-1)*100

            # calculate other metrics based on cut-off point
            y_pred = np.where(y_pred_proba < cutoff, 0, 1)
            cm = confusion_matrix(y_list[i], y_pred)
            tn, fp, fn, tp = confusion_matrix(y_list[i], y_pred).ravel()
            total = np.sum(cm)
            metrics[0] = (tp+tn)/total*100 # Accuracy
            metrics[1] = tp/(tp+fp)*100 # Precision
            metrics[2] = tp/(tp+fn)*100 # Recall/Sensitivity/TPR
            metrics[3] = tn/(tn+fp)*100 # Specificity
            metrics[4] = 2*metrics[1]*metrics[2]/(metrics[1]+metrics[2]) # F-Score
            metrics[5] = metrics[0]

            # add radar plots
            fig.add_trace(go.Scatterpolar(r=metrics, theta=theta_labels,
                                          mode='lines', visible=False,
                                          line_color=color_pallete[i],
                                          name=model+'(Gini:'+str(round(Gini, 1))+'%)'))
    
    # make one step visible (n models in total)
    for j in range(len(models)):
        fig.data[12*len(models)+j].visible = True
    
    # Create and add slider
    cutoffs = []
    for i in range(int(len(fig.data)/len(models))):
        cutoff = dict(method='restyle', args=['visible', [False]*len(fig.data)],
                      label=str(20+i*2.5)+'%',)
        for j in range(len(models)):
            cutoff['args'][1][i*len(models)+j] = True # Toggle (i*n*j)'th trace to "visible"
        cutoffs.append(cutoff)
    
    sliders = [dict(active=12, currentvalue={'prefix': 'Cut-off: '},
                    pad={'t': 50}, steps=cutoffs)]
    
    fig.update_layout(sliders=sliders, showlegend=True,
                      title='Model Comparison on Different Cut-offs',
                      polar=dict(radialaxis_angle=30, radialaxis_range=[0, 100],
                                 angularaxis=dict(direction='clockwise', period=5),
                                 angularaxis_dtick=20))
    
    fig.show()
    if savefig==True:
        plty.plot(fig, filename='radar_slider.html', auto_open=False)

    return


def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")

    return
