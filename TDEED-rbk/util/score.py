"""
File containing auxiliar score functions
"""

#Standard imports
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import numpy as np
BALL_KEYWORDS = {
    "PASS", "CROSS", "THROW IN", "SHOT",
    "FREE KICK", "Corner", "Kickoff", "SUCCESSFUL THROUGH BALL"
}
def _delta_array(name):
    if name == "at1":   return np.array([1])
    if name == "at2":   return np.array([2])
    if name == "at3":   return np.array([3])
    if name == "tight": return np.arange(5) * 1 + 1
    if name == "loose": return np.arange(12) * 5 + 5
    raise ValueError(name)

def compute_amAP_mix(targets_numpy,
                     detections_numpy,
                     closests_numpy,
                     classes,
                     framerate=25,
                     event_team=False):
    inv_classes = {}
    ball_ids, nonball_ids = [], []

    if event_team:
        for k, v in classes.items():
            cls_id = v - 1
            inv_classes[cls_id] = k
            base_name = k.split('-')[0]
            (ball_ids if base_name in BALL_KEYWORDS else nonball_ids).append(cls_id)
    else:
        base_class_to_id = {}
        for k, v in classes.items():
            base = k.split('-')[0]
            if base not in base_class_to_id:
                cls_id = int((v - 1) / 2)
                base_class_to_id[base] = cls_id
                inv_classes[cls_id] = base
                (ball_ids if base in BALL_KEYWORDS else nonball_ids).append(cls_id)

    ball_ids.sort()
    nonball_ids.sort()

    num_classes = targets_numpy[0].shape[1]
    ball_ids = [i for i in ball_ids if i < num_classes]
    nonball_ids = [i for i in nonball_ids if i < num_classes]

    def _select_cols(arr_list, cols):
        return [a[:, cols] for a in arr_list]

    res_ball = average_mAP(
        _select_cols(targets_numpy,    ball_ids),
        _select_cols(detections_numpy, ball_ids),
        _select_cols(closests_numpy,   ball_ids),
        framerate=framerate, deltas=_delta_array("at1")
    ) if ball_ids else (0.0, [0.0] * num_classes)

    res_nb = average_mAP(
        _select_cols(targets_numpy,    nonball_ids),
        _select_cols(detections_numpy, nonball_ids),
        _select_cols(closests_numpy,   nonball_ids),
        framerate=framerate, deltas=_delta_array("at3")
    ) if nonball_ids else (0.0, [0.0] * num_classes)

    mAP_per_class = np.zeros(num_classes)
    for i, class_id in enumerate(ball_ids):
        mAP_per_class[class_id] = res_ball[1][i]
    for i, class_id in enumerate(nonball_ids):
        mAP_per_class[class_id] = res_nb[1][i]

    if event_team:
        ntargets = np.zeros_like(mAP_per_class)
        for t in targets_numpy:
            ntargets += t.sum(axis=0)

        n_pairs = len(mAP_per_class) // 2
        mAP_per_class = mAP_per_class * ntargets
        mAP_per_class = [
            (mAP_per_class[i*2] + mAP_per_class[i*2+1]) /
            (ntargets[i*2] + ntargets[i*2+1] + 1e-9)
            for i in range(n_pairs)
        ]

    mAP_global = np.mean(mAP_per_class)
    return {
        "mAP": mAP_global,
        "mAP_per_class": mAP_per_class,
        "mAP_visible": None,
        "mAP_per_class_visible": None,
        "mAP_unshown": None,
        "mAP_per_class_unshown": None
    }


def compute_amAP(targets_numpy, detections_numpy, closests_numpy, framerate=25, metric = 'tight', event_team = False):

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])

    if event_team:
        ntargets = np.zeros(targets_numpy[0].shape[1])
        for i in range(len(targets_numpy)):
            ntargets += targets_numpy[i].sum(axis=0)

    mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate=framerate, deltas = deltas)

    if event_team:
        mAP_per_class = mAP_per_class * ntargets
        mAP_per_class = [(mAP_per_class[i*2] + mAP_per_class[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class) // 2)]
        mAP = np.mean(mAP_per_class)

        mAP_per_class_visible = mAP_per_class_visible * ntargets
        mAP_per_class_visible = [(mAP_per_class_visible[i*2] + mAP_per_class_visible[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class_visible) // 2)]
        mAP_visible = np.mean(mAP_per_class_visible)

        mAP_per_class_unshown = mAP_per_class_unshown * ntargets
        mAP_per_class_unshown = [(mAP_per_class_unshown[i*2] + mAP_per_class_unshown[(i*2)+1]) / (ntargets[i*2] + ntargets[i*2+1]) for i in range(len(mAP_per_class_unshown) // 2)]
        mAP_unshown = np.mean(mAP_per_class_unshown)

    return {"mAP": mAP, "mAP_per_class": mAP_per_class, "mAP_visible": mAP_visible, "mAP_per_class_visible": mAP_per_class_visible, 
            "mAP_unshown": mAP_unshown, "mAP_per_class_unshown": mAP_per_class_unshown}
















