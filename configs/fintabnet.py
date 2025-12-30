_base_ = "pubtabnet.py"


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=520, keep_ratio=True),
    dict(type="Pad", size=(520, 520)),
    dict(type="FillBbox", cell=_base_.cell_tokens),
    dict(type="FlipBbox"),
    dict(type="FormBbox"),
    dict(type="ToOTSL"),
    dict(
        type="Normalize",
        mean=[128, 128, 128],
        std=[128, 128, 128],
    ),
    dict(type="ImageToTensor", keys=["img"]),
    dict(
        type="Annotate",
        keys=["img"],
        meta=[
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "html",
            "cell",
            "bbox",
        ],
    ),
]

train_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="train"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="test"),
    ),
)

test_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="val"),
    ),
)
