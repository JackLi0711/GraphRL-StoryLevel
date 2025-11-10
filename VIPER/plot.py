import matplotlib.pyplot as plt
from pathlib import Path


def plot_policy_scores(score_list: list[float], checkpoint_dir: Path) -> None:
    """Plot the scores of all extracted policies."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_list) + 1), score_list, marker='o', linestyle='-', color='b')
    plt.title("Scores of Extracted Policies")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "policy_scores.png")
    plt.close()

def plot_fail_names(fail_names: list[str], checkpoint_dir: Path) -> None:
    name_count_pairs = {}
    for names in fail_names:
        for name in names:
            if name == None: name = "none"
            if name in name_count_pairs:
                name_count_pairs[name] += 1
            else:
                name_count_pairs[name] = 1

    plt.figure(figsize=(12, 6))
    plt.bar(name_count_pairs.keys(), name_count_pairs.values())
    plt.title("Evaluation Fail Names", fontsize=16)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "fail_names.png")
    plt.close()

def plot_fail_reasons(fail_reasons: list[str], checkpoint_dir: Path) -> None:
    reason_count_pairs = {}
    for reasons in fail_reasons:
        for reason in reasons:
            if reason == None: reason = "none"
            if reason in reason_count_pairs:
                reason_count_pairs[reason] += 1
            else:
                reason_count_pairs[reason] = 1

    plt.figure(figsize=(12, 6))
    plt.bar(reason_count_pairs.keys(), reason_count_pairs.values())
    plt.title("Evaluation Fail Reasons", fontsize=16)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "fail_reasons.png")
    plt.close()
