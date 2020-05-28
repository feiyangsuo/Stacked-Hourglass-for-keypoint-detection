import numpy as np
import json
import math
import os


def calOKS(res_path, lab_path):
    with open(res_path, "r") as f:
        res = json.load(f)
    with open(lab_path, "r") as f:
        lab = json.load(f)

    # TODO:
    # sigma = {"eye": 1, "mouth": 1, "backfin": 1, "chestfin": 1, "analfin": 1, "tail": 1, "backfin2": 1}
    sigma = cal_sigma(res_path="data/results/predicted2/predictedres_best.json",
                      lab_path="data/Fish/Annotations/keypoint/via_export_json.json")

    OKS = 0
    for k_lab in lab.keys():
        img_name = lab[k_lab]["filename"]
        for k_res in res.keys():
            if img_name == res[k_res]["filename"]:
                break
        if img_name != res[k_res]["filename"]:
            print("fuck")
            continue
        img_lab = lab[k_lab]
        img_res = res[k_res]
        # I don't know why, but, via json's size is all wrong. 操你妈了个逼
        # assert (img_lab["filename"], img_lab["size"]) == (img_res["filename"], img_res["size"])
        regions_lab = img_lab["regions"]
        regions_res = img_res["regions"]

        numerator = 0
        S2 = img_res["size"]
        for region_lab in regions_lab:
            name = region_lab["region_attributes"]["keypoint"]
            point_lab = (region_lab["shape_attributes"]["cx"], region_lab["shape_attributes"]["cy"])
            for region_res in regions_res:
                if region_res["region_attributes"]["keypoint"] == name:
                    point_res = (region_res["shape_attributes"]["cx"], region_res["shape_attributes"]["cy"])
                    d2 = (point_lab[0] - point_res[0])**2 + (point_lab[1] - point_res[1])**2
                    numerator += math.exp(-d2 / (2 * S2 * sigma[name]**2))
        denominator = len(regions_lab)
        OKS += numerator / denominator

    OKS /= len(lab)
    return OKS


def cal_sigma(res_path, lab_path):
    with open(res_path, "r") as f:
        res = json.load(f)
    with open(lab_path, "r") as f:
        lab = json.load(f)
    d_S = {"eye": 0, "mouth": 0, "backfin": 0, "chestfin": 0, "analfin": 0, "tail": 0, "backfin2": 0}
    num = {"eye": 0, "mouth": 0, "backfin": 0, "chestfin": 0, "analfin": 0, "tail": 0, "backfin2": 0}

    for k_lab in lab.keys():
        img_name = lab[k_lab]["filename"]
        for k_res in res.keys():
            if img_name == res[k_res]["filename"]:
                break
        if img_name != res[k_res]["filename"]:
            continue
        img_lab = lab[k_lab]
        img_res = res[k_res]
        regions_lab = img_lab["regions"]
        regions_res = img_res["regions"]
        S = math.sqrt(img_res["size"])
        for region_lab in regions_lab:
            name = region_lab["region_attributes"]["keypoint"]
            point_lab = (region_lab["shape_attributes"]["cx"], region_lab["shape_attributes"]["cy"])
            for region_res in regions_res:
                if region_res["region_attributes"]["keypoint"] == name:
                    point_res = (region_res["shape_attributes"]["cx"], region_res["shape_attributes"]["cy"])
                    d = math.sqrt((point_lab[0] - point_res[0]) ** 2 + (point_lab[1] - point_res[1]) ** 2)
                    d_S[name] += d/S
                    num[name] += 1
    for k in d_S.keys():
        d_S[k] /= num[k]

    return d_S


if __name__ == '__main__':
    for n in range(3):
        print(calOKS(res_path="data/results/predicted2/predictedres_best.stage{}.json".format(n),
                     lab_path="data/Fish/Annotations/keypoint/via_export_json.json"))

