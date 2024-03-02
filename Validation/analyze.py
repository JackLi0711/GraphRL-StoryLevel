import os
import matplotlib.pyplot as plt


def analyze_design(result_root):
    '''
    Plot histograms based on the scores of pass case and fail case
    '''
    pass_root = result_root / "pass"
    fail_root = result_root / "fail"

    # pass cases
    pass_scores = []
    for file_name in os.listdir(pass_root):
        score = float(file_name.split("_")[0])
        print("score:", score)
        pass_scores.append(score)
    
    # fail cases
    fail_scores = []
    for file_name in os.listdir(fail_root):
        score = float(file_name.split("_")[0])
        fail_scores.append(score)
    
    # plot
    fig = plt.figure(figsize=(20, 8), facecolor="w")

    ax = fig.add_subplot(1, 2, 1)
    ax.hist(pass_scores, bins=range(4, 26), edgecolor='black')
    ax.set_title(f"Pass case num: {len(pass_scores)}", fontsize=18)
    ax.set_xlabel("Scores", fontsize=15)
    ax.set_ylabel("Num of cases", fontsize=15)
    ax.set_xticks(range(4, 26))
    ax.invert_xaxis()

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(fail_scores, bins=range(4, 26), edgecolor='black')
    ax.set_title(f"Fail case num: {len(fail_scores)}", fontsize=18)
    ax.set_xlabel("Scores", fontsize=15)
    ax.set_ylabel("Num of cases", fontsize=15)
    ax.set_xticks(range(4, 26))
    ax.invert_xaxis()

    plt.savefig(result_root / "analyze.png")
    plt.close()
    



def analyze_design_rank(ai_score, result_root):
    '''
    Get the PR of GraphRL's design among the sampling designs
    '''
    pass_root = result_root / "pass"
    fail_root = result_root / "fail"

    # pass cases
    pass_scores = []
    for file_name in os.listdir(pass_root):
        score = float(file_name.split("_")[0])
        pass_scores.append(score)
    
    # fail cases
    fail_scores = []
    for file_name in os.listdir(fail_root):
        score = float(file_name.split("_")[0])
        fail_scores.append(score)
    
    # calculate the rank
    pass_scores.sort(reverse=True)
    ai_rank = 0
    while ai_rank < len(pass_scores):
        if ai_score < pass_scores[ai_rank]:
            ai_rank += 1
        else:
            break
    
    ai_rank_percentage = 100 - ai_rank / len(pass_scores + fail_scores) * 100

    return ai_rank_percentage




def analyze_ai_ranks(result_root, ranks, chances):
    '''
    Plot the PR histogram
    '''
    # ranks: key (geo_name), value (rank_percentage)

    fig = plt.figure(figsize=(8, 6), facecolor="w")

    ax = fig.add_subplot(1, 1, 1)
    ax.hist(ranks.values(), bins=range(0, 101, 10), edgecolor='black')
    # ax.set_title(f"Pass case num: {len(pass_scores)}", fontsize=18)
    ax.set_xlabel("PR Rank", fontsize=15)
    ax.set_ylabel("Num of cases", fontsize=15)
    ax.set_xticks(range(0, 101, 10))
    ax.invert_xaxis()
    ax.set_ylim(top=80)
    
    plt.savefig(result_root / f"analyze_training_space_{chances}chances.png")
    plt.close()



