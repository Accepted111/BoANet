import os
import sys
from tensorboardX import SummaryWriter
import argparse
import logging
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.vnet_dual_task import VNet
from utils import ramps, losses
from test import test_calculate_metric
from data_processing.data_loader import pancreas, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--unsupervised_rampup', type=float,
                    default=40.0, help='unsupervised_rampup')

args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

num_classes, save_step = 2, 50

train_data_path = '../dataset/pancreas'
snapshot_path = "../model/pancreas/train_12labels"
patch_size = (96, 96, 96)

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

def create_model(ema=False):
    net = VNet(n_channels=1, n_classes=num_classes - 1,
               normalization='batchnorm', has_dropout=True)
    if torch.cuda.is_available():
        model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def get_current_unsupervised_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, args.unsupervised_rampup)


if __name__ == "__main__":

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

    db_train = pancreas(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = 12  # default 12
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 62))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    result = []

    for epoch in range(1, max_epoch):
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            segmentation, background = model(volume_batch)

            segmentation_soft = torch.sigmoid(segmentation)
            background_soft = torch.sigmoid(background)
            foreground = torch.sigmoid(-1500 * background)
            mask_segmentation = (segmentation_soft > 0.5).type(torch.float32)
            mask_background = (background_soft > 0.5).type(torch.float32)
            boundary = mask_segmentation * mask_background

            boundary[boundary == 1.0] = 6.0
            boundary[boundary == 0] = 1.0

            loss_seg_dice = losses.dice_loss(segmentation_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            loss_bag_wce = losses.weighted_cross_entropy_loss(-background[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs])

            unsupervised_dist = (foreground[:] - segmentation_soft[:]) ** 2
            boundary_geometry_constraint_loss = (unsupervised_dist * boundary).sum(dim=(2, 3, 4)) / (boundary.sum(dim=(2, 3, 4)) + 1e-16)
            boundary_geometry_constraint_loss = boundary_geometry_constraint_loss.mean()

            supervised_loss = loss_seg_dice + loss_bag_wce
            unsupervised_weight = get_current_unsupervised_weight(iter_num//150)

            total_loss = supervised_loss + unsupervised_weight * boundary_geometry_constraint_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            writer.add_scalar('train/total_loss', total_loss, iter_num)
            writer.add_scalar('train/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/loss_bag_wce', loss_bag_wce, iter_num)
            writer.add_scalar('train/bgc_loss', boundary_geometry_constraint_loss, iter_num)
            writer.add_scalar('train/learning_rate', lr_, iter_num)
            writer.add_scalar('train/unsupervised_weight', unsupervised_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_seg_dice: %f, loss_bag_wce: %f, bgc_loss: %f'
                %(iter_num, total_loss.item(), loss_seg_dice.item(), loss_bag_wce.item(), boundary_geometry_constraint_loss.item())
            )

            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 30:71:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = label_batch[0, :, :, 30:71:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Label', grid_image, iter_num)

                image = segmentation_soft[0, 0:1, :, :, 30:71:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Segmentation', grid_image, iter_num)

                image = background_soft[0, 0:1, :, :, 30:71:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Background', grid_image, iter_num)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

        if epoch % save_step == 0:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            avg_metric = test_calculate_metric(model)
            val_dice = avg_metric[0]
            val_jc = avg_metric[1]
            val_asd = avg_metric[2]
            val_hd = avg_metric[3]
            val_mean = (val_dice + val_hd) / 2

            temp = "iteration_test %d: Dice: %f, JC: %f, ASD: %f, HD: %f"%(iter_num, val_dice, val_jc, val_asd, val_hd)
            result.append(temp)
            logging.info(temp)

            if val_mean > best_performance:
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), save_mode_path)
                best_performance = val_mean

    for log in result:
        logging.info(log)
    writer.close()
