_base_ = "pubtabnet.py"


test_dataloader = dict(
    dataset=dict(
        type="TabularDatasetJson",
        ann_file="~/data/icdar-task-b/final_eval.json",
    ),
)

test_evaluator = dict(ignore=["b"])  # in all <td></td> elements
