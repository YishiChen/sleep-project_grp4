import math
import os
import sys
from typing import Iterable
import torchvision.transforms as T
import numpy as np
import torch
import matplotlib.pyplot as plt
import util.misc as utils
from util import box_ops
from models.matcher import build_matcher
from tqdm import tqdm
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

@torch.no_grad()
def eval_score(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, data_dir):
    model.eval()
    criterion.eval()

    ma_conf_noiou = np.zeros((4,3))
    conf_ma = np.zeros((4,3))

    for s_idx, (samples, targets, records, *_) in enumerate(tqdm(data_loader)):
        if s_idx % 500 == 0:
            print(s_idx, 'samples processed')
        targets_new = []
        # NEW TARGET IS LIST(DICTIONARY(TENSOR)))
        for i, target in enumerate(targets):
            boxes = target[:, :2]
            cxs = boxes.mean(dim=1)
            cys = torch.zeros(cxs.size(dim=0)).add(0.5)
            ws = boxes[:, 1] - boxes[:, 0]
            hs = torch.ones(cxs.size(dim=0))
            boxes = torch.column_stack((cxs, cys, ws, hs))
            labels = target[:, 2].long()
            dict_t = {'boxes': boxes, 'labels': labels}
            targets_new.append(dict_t)

        #samples = samples[:, None, :, :]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets_new]
        outputs = model(samples)

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox[:, [0, 2]], tgt_bbox, p=1)
        # -- Make target y-coordinates equal to the predict to not punish model for y prediction -- #
        cost_bbox = torch.cdist(out_bbox[:, [0, 2]], tgt_bbox[:, [0, 2]], p=1)

        # Compute the giou cost betwen boxes [N, M]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = 5 * cost_bbox * cost_bbox + 1 * cost_class + 2 * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        query_idx = indices[0][0]
        tgt_idx = indices[0][1]

        giou = -cost_giou[query_idx, tgt_idx]
        labels = tgt_ids[tgt_idx]
        pred_labels = out_prob.argmax(-1)[query_idx]

        for i in range(len(giou)):
            l = labels[i]
            pl = pred_labels[i]
            if pl > 3:
                pl = 3
            if giou[i] >= 0.3:
                conf_ma[pl, l] += 1
            ma_conf_noiou[pl, l] += 1



    if data_dir =="D:/10channel":
        np.save('D:/predictions/' + args.backbone + 'n.npy', ma_conf_noiou)
        np.save('D:/predictions/' + args.backbone + 'i.npy', conf_ma)
    else:
        np.save('/scratch/s203877/' + args.backbone + 'n.npy', ma_conf_noiou)
        np.save('/scratch/s203877/' + args.backbone + 'i.npy', conf_ma)

    return 'fuck off'
