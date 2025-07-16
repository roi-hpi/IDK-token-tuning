
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
#seaborn:
import seaborn as sns
sns.set_theme(style="whitegrid")



metrics_general=["baseline_accuracy","gold_prediction_rate"]
metrics_rel_to_self=["precision_rel_to_self","recall_rel_to_self","f1_rel_to_self","accuracy_rel_to_self","fpr_rel_to_self"]
metrics_rel_to_baseline=["precision_rel_to_baseline","recall_rel_to_baseline","f1_rel_to_baseline","accuracy_rel_to_baseline","fpr_rel_to_baseline"]
all_metrics=metrics_general+metrics_rel_to_self+metrics_rel_to_baseline
dataset_names=["LAMA-google_re","LAMA-squad","LAMA-trex"]


def get_comparison_df_one_ds(dataset_name, results_base_dir):
    """
    reads the stats_results.json file for each model in the given dataset
    :returns a dataframe for given dataset with the model names as rows and the metrics as columns"""
    ds_path=os.path.join(results_base_dir,  dataset_name)
    model_names = os.listdir(ds_path)
    df = pd.DataFrame(index=model_names, columns=all_metrics)
    for model_name in model_names:
        path=os.path.join(ds_path,model_name, "stats_results.json")
        try:
            stats_results = json.load(open(path))
        except:
            print("-----------------------------------")
            print("ERROR! no stats_results.json in "+path)
            print("-----------------------------------")
            continue

        for metric in metrics_general:
            try:
                df.loc[model_name, metric] = stats_results[metric]
            except:
                df.loc[model_name, metric] = np.nan
        for metric in metrics_rel_to_self:
            try:
                df.loc[model_name, metric] = stats_results['rel_to_self'][metric]
            except:
                df.loc[model_name, metric] = np.nan
        for metric in metrics_rel_to_baseline:
            try:
                df.loc[model_name, metric] = stats_results['rel_to_baseline'][metric]
            except:
                df.loc[model_name, metric] = np.nan

    df["tp/all"]=np.nan
    #then add values:
    for model_name in model_names:
        df.loc[model_name, "tp/all"] = df.loc[model_name, "accuracy_rel_to_self"] - df.loc[model_name, "gold_prediction_rate"]
    #put it at first column:
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df
def get_comparison_df_all_ds():
    """ :returns a dictionary of dataframes, one for each dataset, with the model names as rows and the metrics as columns
    also saves the dataframes to csv files"""
    cur_wd = os.getcwd()
    results_base_dir = os.path.join(cur_wd, "results_new_tp")
    save_dir=os.path.join(cur_wd,"comparison_new_tp")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_dict={}

    for dataset_name in dataset_names:
        df_dataset= get_comparison_df_one_ds(dataset_name, results_base_dir)
        save_name=dataset_name+"_comparison.csv"
        save_path=os.path.join(save_dir, save_name)
        df_dataset.to_csv(save_path)
        df_dict[dataset_name]=df_dataset
    print(f"saved comparison dataframes in {save_dir}")
    return df_dict

def get_max_comparison_df(df_comparison_dict):
    """

    :param df_comparison_dict: a dictionary of dataframes, one for each dataset, with the model names as rows and the metrics as columns
    :return:  a dataframe with the best model for each dataset for each metric and the metric value
    also saves the dataframe to a csv file
    """
    dataset_names=list(df_comparison_dict.keys())
    comp_keys=np.array(all_metrics).flatten()
    #add tp/all to comp_keys:
    comp_keys=np.append(comp_keys,"tp/all")

    #create a dfwith rows comp_keys and columns dataset_names
    df_max_comp=pd.DataFrame(index=comp_keys, columns=dataset_names)
    for comp_key in comp_keys:
        #print best model for each dataset for the given metric  and the metric value
        for dataset_name in dataset_names:
            df_dataset=df_comparison_dict[dataset_name]
            best_model=df_dataset[comp_key].astype(np.float32).idxmax()
            best_value=df_dataset[comp_key].max()
            df_max_comp.loc[comp_key, dataset_name]=best_model+":  "+str(round(best_value,4))

    #save df_max_comp
    cur_wd = os.getcwd()

    save_dir=os.path.join(cur_wd,"comparison_new_tp")
    save_name="comparison_best.csv"
    save_path=os.path.join(save_dir, save_name)
    df_max_comp.to_csv(save_path)
    print(f"saved best model comparison in {save_path}")
    return df_max_comp

def plot_ROC_one_dataset(df_dataset, dataset_name,plot_dir):
    recall_name="recall_rel_to_self"
    fpr_name="fpr_rel_to_self"
    # recall_name="recall_rel_to_self"
    # fpr_name="precision_rel_to_self"
    #todo give different names to no reg and reg
    idk_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    #----for no reg----------------
    model_names = ["idk-bert-0.1","idk-bert-0.2","idk-bert-0.3","idk-bert-0.4","idk-bert-0.5"]
    fpr_no_reg = df_dataset.loc[model_names,fpr_name ].astype(np.float32)
    tpr_no_reg = df_dataset.loc[model_names, recall_name].astype(np.float32)





    plt.plot(fpr_no_reg,tpr_no_reg , label="no regularization",color="blue",marker="o")
    for i, txt in enumerate(idk_weights):
        plt.annotate(txt, (fpr_no_reg[i], tpr_no_reg[i],),fontsize=8)

    #----for reg----------------
    model_names = ["idk-bert-reg-0.1","idk-bert-reg-0.2","idk-bert-reg-0.3","idk-bert-reg-0.4","idk-bert-reg-0.5"]
    fpr_reg = df_dataset.loc[model_names, fpr_name].astype(np.float32)
    tpr_reg = df_dataset.loc[model_names, recall_name].astype(np.float32)



    plt.plot(fpr_reg,tpr_reg , label="regularization",color="orange",marker="o",linestyle="--",)
    for i, txt in enumerate(idk_weights):
        plt.annotate(txt, (fpr_reg[i], tpr_reg[i]),fontsize=8)

    # #plot x=y between min to max fpr vaplues:
    # min_fpr=min(min(fpr_no_reg),min(fpr_reg))
    # max_fpr=max(max(fpr_no_reg),max(fpr_reg))
    # x=np.linspace(0,max_fpr,100)
    # # x=np.linspace(0,0.5,100)
    #
    # y=x
    # plt.plot(x,y,color="grey",linestyle=":")

    #------------------------------adaptive----------------------------------

    #for adaptive no reg:
    model_names_adaptive = ["idk-bert-adaptive-0.2","idk-bert-adaptive-0.5"]
    adaptive_idk_weights = [0.2,0.5]
    adaptive_marker="+"


    recall_adaptive = df_dataset.loc[model_names_adaptive, recall_name].astype(np.float32)
    fpr_adaptive = df_dataset.loc[model_names_adaptive, fpr_name].astype(np.float32)

    plt.scatter(fpr_adaptive, recall_adaptive, color="red", marker=adaptive_marker)


    #for adaptive reg:
    model_names_adaptive_reg = ["idk-bert-reg-adaptive-0.2","idk-bert-reg-adaptive-0.5"]
    adaptive_reg_idk_weights = [0.2,0.5]
    adaptive_reg_marker="*"

    recall_adaptive_reg = df_dataset.loc[model_names_adaptive_reg, recall_name].astype(np.float32)
    fpr_adaptive_reg = df_dataset.loc[model_names_adaptive_reg, fpr_name].astype(np.float32)

    plt.scatter(fpr_adaptive_reg, recall_adaptive_reg, color="black", marker=adaptive_reg_marker,linestyle=":")
    #legend for adaptive:
    plt.plot([], [], label="adaptive",color="red",marker=adaptive_marker,linestyle=":")
    plt.plot([], [], label="adaptive reg",color="black",marker=adaptive_reg_marker,linestyle=":")
    #annotate adaptive:
    for i, txt in enumerate(adaptive_idk_weights):
        plt.annotate(txt, (fpr_adaptive[i], recall_adaptive[i]),fontsize=8)
    for i, txt in enumerate(adaptive_reg_idk_weights):
        plt.annotate(txt, (fpr_adaptive_reg[i], recall_adaptive_reg[i]),fontsize=8)
    #-------------------------------------------------------------------------

    # plt.ylim(0, 1)
    plt.xlabel(fpr_name)
    plt.ylabel(recall_name)
    plt.title(f"ROC curve for {dataset_name}")
    plt.legend()

    # plt.show()
    #scale the axes such that x=y is a diagonal line:
    # plt.axis('scaled')

    plt.savefig(os.path.join(plot_dir, dataset_name+"_ROC.pdf"))


    plt.close()


    #----for reg----------------
    # model_names = ["idk-bert-no-idk,idk-bert-0.1-reg","idk-bert-0.2-reg","idk-bert-0.3-reg","idk-bert-0.4-reg","idk-bert-0.5-reg"]
def plot_ROC_all_datasets(df_comparison_dict):
    #make dir plots if not exists
    cur_wd = os.getcwd()
    plots_dir = os.path.join(cur_wd, "comparison_new_tp", "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for dataset_name in dataset_names:
        df_dataset=df_comparison_dict[dataset_name]
        plot_ROC_one_dataset(df_dataset, dataset_name,plots_dir)

    print(f"saved ROC plots in {plots_dir}")
def plot_ROC_LAMAtrex_scaled(weight):
    path = f"results_scaled/LAMA-trex/idk-bert-{weight}_stats_scaled.csv"
    df = pd.read_csv(path, index_col=0)
    dataset_name="scaled_05"
    # todo give different names to no reg and reg
    idk_weights = df.index

    precisions_no_reg = df.loc[df.index, "precision"].astype(np.float32)
    recalls_reg = df.loc[df.index, "recall"].astype(np.float32)
    plt.plot(recalls_reg, precisions_no_reg, color="blue", marker="o")
    for weight in idk_weights:
        plt.annotate(weight, (recalls_reg[weight], precisions_no_reg[weight]))


    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"ROC curve for {dataset_name}")

    # plt.show()


    plt.savefig(f"comparison/plots/LAMA-trex_metrics_scaled_{weight}.pdf")
    plt.show()

def plot_metrics_LAMAtrex():
    path = "comparison/LAMA-trex_comparison.csv"
    df = pd.read_csv(path, index_col=0)
    # cols=["precision", "recall", "f1", "accuracy", "gold_prediction_rate",'tnr',
    #             "percent_current_idk_and_baseline_wrong_from_baseline_wrong",
    #       "percent_current_wrong_and_baseline_wrong_from_baseline_wrong",
    #             "percent_current_idk_and_baseline_wrong_from_current_idk"]
    cols = [
        "percent_current_idk_and_baseline_wrong_from_baseline_wrong",
        "percent_current_wrong_and_baseline_wrong_from_baseline_wrong",
        "percent_current_correct_and_baseline_wrong_from_baseline_wrong",
        "percent_current_idk_and_baseline_wrong_from_current_idk",
        "percent_current_idk_from_all_idk"]
    df = df[cols]
    # rename columns:
    df.columns = ["c_idk&b_wrong/b_wrong", "c_wrong&b_wrong/b_wrong", "c_correct&b_wrong/b_wrong",
                  "c_idk&b_wrong/c_idk", "c_idk/all"]

    no_reg_rows = ["idk-bert-no-idk", "idk-bert-0.1", "idk-bert-0.2", "idk-bert-0.3", "idk-bert-0.4",
                   "idk-bert-0.5"]
    reg_rows = ["idk-bert-no-idk", "idk-bert-reg-0.1", "idk-bert-reg-0.2", "idk-bert-reg-0.3", "idk-bert-reg-0.4",
                "idk-bert-reg-0.5"]

    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    colors = ["blue", "lightblue", "grey", "green", "pink"]

    for i, col in enumerate(df.columns):
        # the y axis is the metric value
        y_no_reg = df.loc[no_reg_rows, col].astype(np.float32) * 100
        y_reg = df.loc[reg_rows, col].astype(np.float32) * 100
        # the x axis is the idk weights

        plt.plot(x, y_no_reg, label=col, marker="o", color=colors[i])
        plt.plot(x, y_reg, label=col + "reg", marker="o", color=colors[i], linestyle="--")

    plt.xticks(x)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # #small font,legend below the graph
    plt.legend()
    # small font
    plt.rcParams.update({'font.size': 8})
    plt.xlabel("idk weight")
    plt.ylabel("percent")
    plt.title("LAMA-trex metrics")
    plt.savefig("comparison/plots/LAMA-trex_metrics.pdf")
    plt.show()
def plot_prec_recall(df_comparison_dict, dataset_name="LAMA-trex"):
    #make dir plots if not exists
    cur_wd = os.getcwd()
    plot_dir = os.path.join(cur_wd, "comparison_new_tp", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    df_dataset=df_comparison_dict[dataset_name]
    #todo give different names to no reg and reg
    idk_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    #----for no reg----------------
    model_names_no_reg = ["idk-bert-0.1","idk-bert-0.2","idk-bert-0.3","idk-bert-0.4","idk-bert-0.5"]
    precision_no_reg = df_dataset.loc[model_names_no_reg, "precision_rel_to_self"].astype(np.float32)
    recall_no_reg = df_dataset.loc[model_names_no_reg, "recall_rel_to_self"].astype(np.float32)
    f1_no_reg = df_dataset.loc[model_names_no_reg, "f1_rel_to_self"].astype(np.float32)

    plt.plot(idk_weights,recall_no_reg , label="recall",color="blue",marker="o")
    plt.plot(idk_weights,precision_no_reg , label="precision",color="green",marker="o")
    plt.plot(idk_weights,f1_no_reg , label="f1",color="red",marker="o")



    #----for reg----------------
    model_names_reg = ["idk-bert-reg-0.1","idk-bert-reg-0.2","idk-bert-reg-0.3","idk-bert-reg-0.4","idk-bert-reg-0.5"]
    precision_reg = df_dataset.loc[model_names_reg, "precision_rel_to_self"].astype(np.float32)
    recall_reg = df_dataset.loc[model_names_reg, "recall_rel_to_self"].astype(np.float32)
    f1_reg = df_dataset.loc[model_names_reg, "f1_rel_to_self"].astype(np.float32)

    plt.plot(idk_weights,recall_reg ,color="blue",marker="x",linestyle="--")
    plt.plot(idk_weights,precision_reg ,color="green",marker="x",linestyle="--")
    plt.plot(idk_weights,f1_reg ,color="red",marker="x",linestyle="--")


    #for adaptive no reg:
    model_names_adaptive = ["idk-bert-adaptive-0.2","idk-bert-adaptive-0.5"]
    adaptive_idk_weights = [0.2,0.5]
    adaptive_marker="+"

    precision_adaptive = df_dataset.loc[model_names_adaptive, "precision_rel_to_self"].astype(np.float32)
    recall_adaptive = df_dataset.loc[model_names_adaptive, "recall_rel_to_self"].astype(np.float32)
    f1_adaptive = df_dataset.loc[model_names_adaptive, "f1_rel_to_self"].astype(np.float32)
    # plt.plot(adaptive_idk_weights, gold_prediction_rate_adaptive, color="orange", marker="+")
    # plt.plot(adaptive_idk_weights, accuracy_adaptive, color="purple", marker="+")
    plt.scatter(adaptive_idk_weights, precision_adaptive, color="blue", marker=adaptive_marker)
    plt.scatter(adaptive_idk_weights, recall_adaptive, color="green", marker=adaptive_marker)
    plt.scatter(adaptive_idk_weights, f1_adaptive, color="red", marker=adaptive_marker)

    #for adaptive reg:
    model_names_adaptive = ["idk-bert-reg-adaptive-0.2","idk-bert-reg-adaptive-0.5"]
    adaptive_reg_idk_weights = [0.2,0.5]
    adaptive_reg_marker="*"

    precision_adaptive = df_dataset.loc[model_names_adaptive, "precision_rel_to_self"].astype(np.float32)
    recall_adaptive = df_dataset.loc[model_names_adaptive, "recall_rel_to_self"].astype(np.float32)
    f1_adaptive = df_dataset.loc[model_names_adaptive, "f1_rel_to_self"].astype(np.float32)

    plt.scatter(adaptive_reg_idk_weights, precision_adaptive, color="blue", marker=adaptive_reg_marker,linestyle=":")
    plt.scatter(adaptive_reg_idk_weights, recall_adaptive, color="green", marker=adaptive_reg_marker,linestyle=":")
    plt.scatter(adaptive_reg_idk_weights, f1_adaptive, color="red", marker=adaptive_reg_marker,linestyle=":")


    #add legend to the markers types:
    plt.plot([], [], label="no regularization",color="black",marker="o")
    plt.plot([], [], label="regularization",color="black",marker="x",linestyle="--")
    plt.plot([], [], label="adaptive",color="black",marker=adaptive_marker,linestyle=":")
    plt.plot([], [], label="adaptive reg",color="black",marker=adaptive_reg_marker,linestyle=":")

    # plt.ylim(0, 1)
    plt.xlabel("IDK-Weight")
    # plt.ylabel("TPR (Recall)")
    plt.title(f"Precision, Recall and F1 for {dataset_name}")
    plt.legend()

    # plt.show()

    plt.savefig(os.path.join(plot_dir, dataset_name+"_prec_recall_f1.pdf"))
    plt.close()

def plot_gold_and_accuracy(df_comparison_dict, dataset_name="LAMA-trex"):
    #make dir plots if not exists
    cur_wd = os.getcwd()
    plot_dir = os.path.join(cur_wd, "comparison_new_tp", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    df_dataset=df_comparison_dict[dataset_name]
    #todo give different names to no reg and reg
    idk_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    #----for no reg----------------
    model_names_no_reg = ["idk-bert-0.1","idk-bert-0.2","idk-bert-0.3","idk-bert-0.4","idk-bert-0.5"]

    gold_prediction_rate_no_reg = df_dataset.loc[model_names_no_reg, "gold_prediction_rate"].astype(np.float32)
    accuracy_no_reg = df_dataset.loc[model_names_no_reg, "accuracy_rel_to_self"].astype(np.float32)

    plt.plot(idk_weights,gold_prediction_rate_no_reg , label="gold_prediction_rate",color="orange",marker="o")
    plt.plot(idk_weights,accuracy_no_reg , label="accuracy",color="purple",marker="o")



    #----for reg----------------
    model_names_reg = ["idk-bert-reg-0.1","idk-bert-reg-0.2","idk-bert-reg-0.3","idk-bert-reg-0.4","idk-bert-reg-0.5"]

    gold_prediction_rate_reg = df_dataset.loc[model_names_reg, "gold_prediction_rate"].astype(np.float32)
    accuracy_reg = df_dataset.loc[model_names_reg, "accuracy_rel_to_self"].astype(np.float32)

    plt.plot(idk_weights,gold_prediction_rate_reg ,color="orange",marker="x",linestyle="--")
    plt.plot(idk_weights,accuracy_reg ,color="purple",marker="x",linestyle="--")


    accuracy_baseline = df_dataset.loc["idk-bert-0.1", "baseline_accuracy"]
    plt.plot(idk_weights, [accuracy_baseline]*len(idk_weights), color="grey",linewidth=3,linestyle=":")
    #write above the line:
    plt.annotate("baseline accuracy", (idk_weights[0], accuracy_baseline+0.01))


    #for adaptive no reg:
    model_names_adaptive = ["idk-bert-adaptive-0.2","idk-bert-adaptive-0.5"]
    adaptive_idk_weights = [0.2,0.5]
    adaptive_marker="+"

    gold_prediction_rate_adaptive = df_dataset.loc[model_names_adaptive, "gold_prediction_rate"].astype(np.float32)
    accuracy_adaptive = df_dataset.loc[model_names_adaptive, "accuracy_rel_to_self"].astype(np.float32)
    # plt.plot(adaptive_idk_weights, gold_prediction_rate_adaptive, color="orange", marker="+")
    # plt.plot(adaptive_idk_weights, accuracy_adaptive, color="purple", marker="+")
    plt.scatter(adaptive_idk_weights, gold_prediction_rate_adaptive, color="orange", marker=adaptive_marker)
    plt.scatter(adaptive_idk_weights, accuracy_adaptive, color="purple", marker=adaptive_marker)

    #for adaptive reg:
    model_names_adaptive = ["idk-bert-reg-adaptive-0.2","idk-bert-reg-adaptive-0.5"]
    adaptive_reg_idk_weights = [0.2,0.5]
    adaptive_reg_marker="*"

    gold_prediction_rate_adaptive_reg = df_dataset.loc[model_names_adaptive, "gold_prediction_rate"].astype(np.float32)
    accuracy_adaptive_reg = df_dataset.loc[model_names_adaptive, "accuracy_rel_to_self"].astype(np.float32)
    plt.scatter(adaptive_reg_idk_weights, gold_prediction_rate_adaptive_reg, color="orange", marker=adaptive_reg_marker,linestyle=":")
    plt.scatter(adaptive_reg_idk_weights, accuracy_adaptive_reg, color="purple", marker=adaptive_reg_marker,linestyle=":")


    #add legend to the markers types:
    plt.plot([], [], label="no regularization",color="black",marker="o")
    plt.plot([], [], label="regularization",color="black",marker="x",linestyle="--")
    plt.plot([], [], label="adaptive",color="black",marker=adaptive_marker,linestyle=":")
    plt.plot([], [], label="adaptive reg",color="black",marker=adaptive_reg_marker,linestyle=":")


    # plt.ylim(0, 1)
    plt.xlabel("IDK-Weight")
    # plt.ylabel("TPR (Recall)")
    plt.legend()

    # plt.show()

    plt.savefig(os.path.join(plot_dir, dataset_name+"_acc_gold.pdf"))
    plt.close()

def plot_2_metrics(df_comparison_dict, dataset_name="LAMA-trex"):
    metric1="gold_prediction_rate"
    metric_2="tp/all" #default None
    metric1_color="orange"
    metric2_color="green"
    plot_adaptive=True

    #make dir plots if not exists
    cur_wd = os.getcwd()
    plot_dir = os.path.join(cur_wd, "comparison_new_tp", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    df_dataset=df_comparison_dict[dataset_name]
    #todo give different names to no reg and reg
    idk_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    #----for no reg----------------
    model_names_no_reg = ["idk-bert-0.1","idk-bert-0.2","idk-bert-0.3","idk-bert-0.4","idk-bert-0.5"]

    gold_prediction_rate_no_reg = df_dataset.loc[model_names_no_reg, metric1].astype(np.float32)
    # accuracy_no_reg = df_dataset.loc[model_names_no_reg, "f1_rel_to_self"].astype(np.float32)
    recall_no_reg = df_dataset.loc[model_names_no_reg, metric_2].astype(np.float32)

    plt.plot(idk_weights,gold_prediction_rate_no_reg , label=metric1,color=metric1_color,marker="o")
    # plt.plot(idk_weights,accuracy_no_reg , label="f1",color="purple",marker="o")
    plt.plot(idk_weights,recall_no_reg , label=metric_2,color=metric2_color,marker="o")



    #----for reg----------------
    model_names_reg = ["idk-bert-reg-0.1","idk-bert-reg-0.2","idk-bert-reg-0.3","idk-bert-reg-0.4","idk-bert-reg-0.5"]

    gold_prediction_rate_reg = df_dataset.loc[model_names_reg, metric1].astype(np.float32)
    # accuracy_reg = df_dataset.loc[model_names_reg, "f1_rel_to_self"].astype(np.float32)
    recall_reg= df_dataset.loc[model_names_reg, metric_2].astype(np.float32)

    plt.plot(idk_weights,gold_prediction_rate_reg ,color=metric1_color,marker="x",linestyle="--")
    # plt.plot(idk_weights,accuracy_reg ,color="purple",marker="x",linestyle="--")
    plt.plot(idk_weights,recall_reg ,color=metric2_color,marker="x",linestyle="--")


    accuracy_baseline = df_dataset.loc["idk-bert-0.1", "baseline_accuracy"]
    plt.plot(idk_weights, [accuracy_baseline]*len(idk_weights), color="grey",linewidth=3,linestyle=":")
    #write above the line:
    plt.annotate("baseline accuracy", (idk_weights[0], accuracy_baseline+0.01),fontsize=8)

    if plot_adaptive:
        #for adaptive no reg:
        model_names_adaptive = ["idk-bert-adaptive-0.2","idk-bert-adaptive-0.5"]
        adaptive_idk_weights = [0.2,0.5]
        adaptive_marker="+"

        gold_prediction_rate_adaptive = df_dataset.loc[model_names_adaptive, metric1].astype(np.float32)
        # accuracy_adaptive = df_dataset.loc[model_names_adaptive, "f1_rel_to_self"]
        recall_adaptive = df_dataset.loc[model_names_adaptive, metric_2].astype(np.float32)

        plt.scatter(adaptive_idk_weights, gold_prediction_rate_adaptive, color=metric1_color, marker=adaptive_marker)
        # plt.scatter(adaptive_idk_weights, accuracy_adaptive, color="purple", marker=adaptive_marker)
        plt.scatter(adaptive_idk_weights, recall_adaptive, color=metric2_color, marker=adaptive_marker)

        #for adaptive reg:
        model_names_adaptive = ["idk-bert-reg-adaptive-0.2","idk-bert-reg-adaptive-0.5"]
        adaptive_reg_idk_weights = [0.2,0.5]
        adaptive_reg_marker="*"

        gold_prediction_rate_adaptive_reg = df_dataset.loc[model_names_adaptive, metric1].astype(np.float32)
        # accuracy_adaptive_reg = df_dataset.loc[model_names_adaptive, "f1_rel_to_self"].astype(np.float32)
        recall_adaptive_reg = df_dataset.loc[model_names_adaptive, metric_2].astype(np.float32)

        plt.scatter(adaptive_reg_idk_weights, gold_prediction_rate_adaptive_reg, color=metric1_color, marker=adaptive_reg_marker,linestyle=":")
        # plt.scatter(adaptive_reg_idk_weights, accuracy_adaptive_reg, color="purple", marker=adaptive_reg_marker,linestyle=":")
        plt.scatter(adaptive_reg_idk_weights, recall_adaptive_reg, color=metric2_color, marker=adaptive_reg_marker,linestyle=":")

        plt.plot([], [], label="adaptive",color="black",marker=adaptive_marker,linestyle=":")
        plt.plot([], [], label="adaptive reg",color="black",marker=adaptive_reg_marker,linestyle=":")
    #add legend to the markers types:
    plt.plot([], [], label="no regularization",color="black",marker="o")
    plt.plot([], [], label="regularization",color="black",marker="x",linestyle="--")



    # plt.ylim(0, 1)
    plt.xlabel("IDK-Weight")
    # plt.ylabel("TPR (Recall)")
    plt.legend()

    # plt.show()

    plt.savefig(os.path.join(plot_dir, dataset_name+"_f1_gold.pdf"))
    plt.close()
#main
if __name__ == '__main__':

    # # run all of this to get the comparison dataframes and ROC plots:
    df_comparison_dict=get_comparison_df_all_ds()
    df_max_comp=get_max_comparison_df(df_comparison_dict)
    plot_ROC_all_datasets(df_comparison_dict)
    plot_prec_recall(df_comparison_dict)
    plot_gold_and_accuracy(df_comparison_dict)
    plot_2_metrics(df_comparison_dict)





