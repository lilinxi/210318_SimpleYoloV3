from collections import OrderedDict

import torch

if __name__ == "__main__":
    # weights_name = "yolov3"
    weights_name = "darknet53"

    # 读取原始权重列表
    raw_weights_file = open("./raw_" + weights_name + "_weights_state_dict.txt", "r")
    raw_weights_lines = raw_weights_file.readlines()
    print("raw_weights_lines:", len(raw_weights_lines))
    # 读取新的权重列表
    demo_weights_file = open("./demo_" + weights_name + "_weights_state_dict.txt", "r")
    demo_weights_lines = demo_weights_file.readlines()
    print("demo_weights_lines:", len(demo_weights_lines))
    # 去除无效映射
    demo_weights_lines = [line for line in demo_weights_lines if not "num_batches_tracked" in line]
    print("demo_weights_lines:", len(demo_weights_lines))
    # 去除末尾换行
    raw_weights_lines = [line[:-1] for line in raw_weights_lines]
    demo_weights_lines = [line[:-1] for line in demo_weights_lines]

    assert len(raw_weights_lines) == len(demo_weights_lines)

    # 读取原始权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_weights: dict = torch.load("./raw_" + weights_name + "_weights.pth", map_location=device)
    # 一一映射
    demo_weights: dict = OrderedDict()
    for i, raw_key in enumerate(raw_weights_lines):
        demo_key = demo_weights_lines[i]
        demo_weights[demo_key] = raw_weights[raw_key]

    # 保存新的权重
    torch.save(demo_weights, "./demo_" + weights_name + "_weights.pth")
