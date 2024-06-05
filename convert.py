import json
import os
import pandas as pd
import sys


abs_path = os.path.dirname(os.path.realpath(__file__))
VALID_DATASETS = ["mnist", "fashion", "cifar10", "cifar100", "svhn"]


def convert(dataset_name):
    if dataset_name not in ["mnist", "fashion", "cifar10", "cifar100", "svhn"]:
        raise ValueError("dataset_name must be in " + str(VALID_DATASETS))
    for p in [0.05, 0.15, 0.25]:
        for cycle in range(5):
            for label in range(100 if dataset_name == "cifar100" else 10):
                path = os.path.join(abs_path, "log/{}/{}/{}/{}".format(dataset_name, p, cycle, label))
                with open(path + "/results.json") as json_file:
                    data = json.load(json_file)
                    df = pd.DataFrame(data=data["test_scores"], columns=["Id", "Label", "Loss"])
                    df = df.set_index("Id").sort_values("Loss")
                    filename = "-".join([str(x) for x in [dataset_name, label, p, cycle / 10, 30, "DeepSVDD", "train", "loss.csv"]])
                    filepath = os.path.join(abs_path, "convert/" + dataset_name + "/0/" + filename)
                    df.to_csv(filepath)


if __name__ == "__main__":
    convert(sys.argv[1])
