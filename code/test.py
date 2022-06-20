import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from networks.vnet_dual_task import VNet
from utils.test_util import test_all_case


test_save_path = "../model/pancreas/test_12labels/"
model_save_path = "../model/pancreas/train_12labels"
num_classes = 2

@torch.no_grad()
def test_calculate_metric(net, save_result = False, show_detail = False):

    net.eval()
    with open('../dataset/pancreas/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ['../dataset/pancreas/data/' + item.replace('\n', '') for item in image_list]
    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                               save_result=save_result, test_save_path=test_save_path,
                               metric_detail=show_detail)

    return avg_metric


if __name__ == '__main__':

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(model_save_path, 'best.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    metric = test_calculate_metric(net, True, True)
    print(metric)
