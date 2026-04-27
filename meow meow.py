import pandas as pd


def gini_impurity(y):
    if y.empty:
        return 0

    probs = y.value_counts(normalize=True)
    return 1 - (probs ** 2).sum()


def divide_dataset(data, column, border):
    left_part = data[data[column] < border]
    right_part = data[data[column] >= border]
    return left_part, right_part


def best_split_search(data):
    best_score = -1
    best_split = None

    initial_gini = gini_impurity(data['Accept'])
    predictors = data.columns[:-1]

    for col in predictors:
        unique_vals = sorted(data[col].unique())

        thresholds = []
        for i in range(len(unique_vals) - 1):
            thresholds.append((unique_vals[i] + unique_vals[i + 1]) / 2)

        for threshold in thresholds:
            left, right = divide_dataset(data, col, threshold)

            if left.empty or right.empty:
                continue

            left_weight = len(left) / len(data)
            right_weight = len(right) / len(data)

            combined_gini = (
                left_weight * gini_impurity(left['Accept']) +
                right_weight * gini_impurity(right['Accept'])
            )

            information_gain = initial_gini - combined_gini

            if information_gain > best_score:
                best_score = information_gain
                best_split = {
                    "column": col,
                    "threshold": threshold,
                    "gain": information_gain
                }

    return best_split


def create_tree(data, current_depth=0, max_depth=3):
    if gini_impurity(data['Accept']) == 0 or current_depth == max_depth:
        return data['Accept'].mode()[0]

    split = best_split_search(data)

    if split is None or split["gain"] <= 0:
        return data['Accept'].mode()[0]

    left_branch, right_branch = divide_dataset(
        data,
        split["column"],
        split["threshold"]
    )

    return {
        "column": split["column"],
        "threshold": split["threshold"],
        "gain": split["gain"],
        "left": create_tree(left_branch, current_depth + 1, max_depth),a
        "right": create_tree(right_branch, current_depth + 1, max_depth)
    }


def show_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + f"--> Класс: {tree}")
        return

    print(f"{indent}[{tree['column']} < {tree['threshold']}] "
          f"(Gain = {tree['gain']:.4f})")

    print(indent + "Да:")
    show_tree(tree["left"], indent + "   ")

    print(indent + "Нет:")
    show_tree(tree["right"], indent + "   ")


dataset = pd.read_csv("data/postupleni.csv")

decision_tree = create_tree(dataset)

print("Построенное дерево CART:")
show_tree(decision_tree)