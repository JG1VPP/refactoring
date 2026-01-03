from pathlib import Path

from mmengine import Config

from mutab.data.dataset import TableDataset, TableDataset2

config = Config.fromfile("configs/pubtabnet.py")

data1 = TableDataset(
    ann_file="~/data/mutab_pubtab250.pkl",
    filter_cfg=dict(split="train"),
    pipeline=config.pipeline,
    test_mode=False,
)
print(len(data1))
data2 = TableDataset2(
    ann_file="~/data/mmocr_pubtab250/train", pipeline=config.pipeline2, test_mode=False
)
print(len(data2))

for item1, item2 in zip(data1, data2):
    path1 = Path(item1["targets"].get("img_path")).stem
    path2 = Path(item2["targets"].get("img_path")).stem

    html1 = item1["targets"].get("html")
    html2 = item2["targets"].get("html")

    cell1 = list("".join(v).strip() for v in item1["targets"].get("cell"))
    cell2 = list("".join(v).strip() for v in item2["targets"].get("cell"))

    bbox1 = item1["targets"].get("bbox")
    bbox2 = item2["targets"].get("bbox")

    bbox1 = list(list(map(int, bb)) for bb in bbox1)
    bbox2 = list(list(map(int, bb)) for bb in bbox2)

    print(path1)

    assert path1 == path2
    assert html1 == html2

    if cell1 != cell2:
        for c1, c2 in zip(cell1, cell2):
            print(c1 == c2, c1, c2)

    assert cell1 == cell2
    assert bbox1 == bbox2
