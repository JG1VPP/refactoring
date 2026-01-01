import numpy as np
from pathlib import Path
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.registry import DATASETS

cell_tokens = [
    '<td></td>', '<td', '<eb></eb>', '<eb1></eb1>', '<eb2></eb2>',
    '<eb3></eb3>', '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>', '<eb7></eb7>',
    '<eb8></eb8>', '<eb9></eb9>', '<eb10></eb10>'
]


@DATASETS.register_module()
class TableDataset(BaseDataset):
    @property
    def split(self):
        return self.filter_cfg.get("split")

    def parse(self, ann: str, **info):
        list_b = []
        list_c = []

        with open(ann) as f:
            path = f.readline().strip()
            html = f.readline().strip().split(",")
            body = list(f.readlines())

        for value in body:
            bbox, cell = value.strip().split("<;>")
            bbox = tuple(map(int, bbox.split(",")))
            cell = cell.split("\t")
            list_b.append(bbox)
            if bbox != (0, 0, 0, 0):
                list_c.append(cell)

        info.update(img_path=path, html=html)
        info.update(cell=list_c, bbox=list_b)

        return self.align(**info)

    def align(self, html, bbox, **info):
        queue = iter(bbox)
        boxes = np.zeros((len(html), 4))
        for idx, cell in enumerate(html):
            if cell in cell_tokens:
                boxes[idx] = next(queue)
        return dict(html=html, gt_bboxes=boxes, **info)

    def load_data_list(self) -> List[dict]:
        # load preprocessed pickle
        path = Path(self.ann_file).expanduser()
        
        data = []

        for ann in path.rglob("*.txt"):
            data.append(self.parse(ann))
        return data
        #data = load(path, file_format="pickle")

        #return list(data[self.split])
