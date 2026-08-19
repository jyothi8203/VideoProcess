import argparse
import copy
import csv
import math
import os, json
import warnings
from wordfreq import zipf_frequency

import cv2
import numpy as np
import torch
import tqdm
from timm import utils
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

from PIL import Image
from torchvision import transforms
from text_crnn.model_crnn import CRNN
from text_crnn.utils_crnn import Converter
# from ctcdecode import CTCBeamDecoder
import torch.nn.functional as F

warnings.filterwarnings("ignore")

data_dir = '../Dataset/TotalText'


def lr(args):
    return 1E-2 * args.batch_size * args.world_size / 16


def train(args):
    # Model
    model = nn.DBNet()
    model = util.load_checkpoint(model, ckpt='./weights/imagenet.pt')
    model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(util.weight_decay(model), lr(args), momentum=0.9, nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    sampler = None
    filenames = []
    with open('../Dataset/TotalText/train.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/train/' + filename)

    dataset = Dataset(args, filenames, train=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None,
                             sampler=sampler, num_workers=8, pin_memory=True)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    num_steps = len(loader)

    criterion = util.ComputeLoss().cuda()
    amp_scale = torch.cuda.amp.GradScaler()
    scheduler = util.CosineLR(args, optimizer)
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'loss', 'Recall', 'Precision', 'F1'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            optimizer.zero_grad()
            for samples, targets in p_bar:
                samples = samples.cuda()

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                amp_scale.step(optimizer)
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)
                avg_loss.update(loss.item(), samples.size(0))
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            # Scheduler
            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)

                logger.writerow({'Precision': str(f'{last[0]:.3f}'),
                                 'Recall': str(f'{last[1]:.3f}'),
                                 'F1': str(f'{last[2]:.3f}'),
                                 'loss': str(f'{avg_loss.avg:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                log.flush()

                # Update best F1
                if best < last[2]:
                    best = last[2]

                # Save model
                save = copy.deepcopy(ema.ema)
                save = {'epoch': epoch,
                        'model': save.half()}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[2]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    filenames = []
    with open('../Dataset/TotalText/test.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/test/' + filename)

    dataset = Dataset(args, filenames, train=False)
    loader = data.DataLoader(dataset, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load(f'./weights/last.pt',weights_only = False)
        # pt_mdl = os.path.join(os.getcwd(),'weights/last.pt')
        # model = torch.load(pt_mdl,weights_only = False)
        model = model['model']
        model = model.float()
        model.cuda()

    model.eval()

    results = []

    evaluator = util.QuadMeasurer(is_polygon=True)
    for sample, target in tqdm.tqdm(loader, ('%10s' * 3) % ('precision', 'recall', 'F1')):
        # print(f'Target:{target}\n')
        output = model(sample.cuda())
        output = util.mask_to_box(target, output.cpu(), is_polygon=True)
        result = output
        # result = evaluator.validate_measure(target, output)
        results.append(result)
    # precision, recall, f1 = evaluator.gather_measure(results)
    # # Print results
    # print(('%10s' * 3) % (f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'))
    #
    # # Return results
    # model.float()  # for training
    # return precision, recall, f1
    return results

@torch.no_grad()
def dbnet_text_extract_orig(img):
    h, w = img.shape[:2]
    new_w = int(w * (32 / h))

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((32, new_w)), transforms.Grayscale(),
                                    transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
    image = transform(img)
    image = image.unsqueeze(0)
    image = image.cuda()
    return image

@torch.no_grad()
def dbnet_text_extract_gray(gray):
    # convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    new_w = int(w * (32 / h))
    new_w = max(64, min(new_w, 256))   # clamp

    # resize
    gray = cv2.resize(gray, (new_w, 32))

    # normalize
    gray = gray.astype("float32") / 255.0
    gray = (gray - 0.5) / 0.5

    # to tensor
    tensor = torch.from_numpy(gray)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,32,W)

    return tensor.cuda()

def predicted_beam_search_text(output, beam_width=5, blank=0):
    # output: [T, 1, C]
    log_probs = F.log_softmax(output, dim=2)
    log_probs = log_probs.squeeze(1)  # [T, C]

    T, C = log_probs.shape

    beams = [([], 0.0)]  # (sequence, score)

    for t in range(T):
        new_beams = []

        for seq, score in beams:
            for c in range(C):
                new_seq = seq + [c]
                new_score = score + log_probs[t, c].item()
                new_beams.append((new_seq, new_score))

        # keep top K beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    # best sequence
    best_seq = beams[0][0]

    # CTC collapse
    collapsed = []
    prev = None
    for c in best_seq:
        if c != prev and c != blank:
            collapsed.append(c)
        prev = c

    return collapsed

def english_score(text):
    vowels = "aeiou"
    v = sum(c in vowels for c in text)
    return v / (len(text) + 1e-6)

def word_score(text, conf, var):
    if len(text) < 3 or conf < 0.6 or var > 0.2:
        return -1
    lang_score = zipf_frequency(text, 'en')
    lang_penalty = -0.5 if lang_score <2 else 0
    return conf - var + 0.3 * lang_score + lang_penalty


import re
def filter_text(txt):
    if not re.match("^[a-z0-9]+$", txt):
        return ""
    if english_score(txt) < 0.2:
        return ""
    return txt


def predict_text(out_label,valid_width = None, max_width = None):
    # _, predicted_label = out_label.max(2)
    probs = torch.softmax(out_label, dim=2)
    max_probs, indices = probs.max(2)
    # Remove padded region
    T = indices.size(0)
    if valid_width is not None:
        # approximate mapping width → time steps
        valid_T = int(T * (valid_width / max_width))
        indices = indices[:valid_T]
        max_probs = max_probs[:valid_T]
    confidence = max_probs.mean().item()
    # character stability (important)
    char_var = max_probs.std().item()

    predicted_label = indices.transpose(1, 0).contiguous().view(-1)
    converter = Converter('0123456789abcdefghijklmnopqrstuvwxyz-')
    predicted_length = [predicted_label.size(0)]
    predicted_label = converter.decode(predicted_label, predicted_length, raw=False)
    return predicted_label, confidence, char_var

def dbnet_image_formated(img,mn,stand, in_sz=800):
    shp = img.shape[:2]
    width = shp[1] * in_sz / shp[0]  # 800 fixed height
    width = math.ceil(width / 32) * 32

    x = cv2.resize(img, dsize=(width, in_sz))
    # print(f'Args inpt_size: {args.input_size}, ImageShape:{shape}, Resized:{width}X{args.input_size}')

    x = x.astype('float32') / 255.0
    x = x - mn
    x = x / stand
    x = x.transpose((2, 0, 1))[::-1]
    x = np.ascontiguousarray(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.cuda()
    return x, shp

def crnn_crop_bbox(img,bbox,vert=False):
### my code
    bbox = np.array(bbox)
    x_min = bbox[:, 0].min()
    y_min = bbox[:, 1].min()
    x_max = bbox[:, 0].max()
    y_max = bbox[:, 1].max()
    out_wd =  x_max - x_min
    out_ht = y_max - y_min
    src_pts = np.float32([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]])
    dest_pts = np.float32([[0,0],[out_wd-1,0],[out_wd-1,out_ht-1],[0,out_ht-1]])
    H, mask = cv2.findHomography(src_pts,dest_pts)
    crp_img = cv2.warpPerspective(img, H, (out_wd, out_ht))

    # if out_ht > out_wd:
    #     candidates = [np.rot90(crp_img, 1), np.rot90(crp_img, 3)]
    #     out_ht = x_max - x_min
    #     out_wd = y_max - y_min
    # else:
    #     candidates = [crp_img, np.rot90(crp_img, 2)]
    aspect_ratio = out_ht / (out_wd + 1e-6)

    if aspect_ratio > 3:
        # stacked text → try ALL rotations
        candidates = [crp_img, np.rot90(crp_img, 1),np.rot90(crp_img, 2), np.rot90(crp_img, 3)]

    elif out_ht > out_wd:
        # likely rotated
        candidates = [np.rot90(crp_img, 1), np.rot90(crp_img, 3)]

    else:
        candidates = [crp_img, np.rot90(crp_img, 2)]

    # cv2.imwrite("Cropped.jpg", crp_img)
    #
    # cv2.imwrite("Cropped.jpg", crp_img)
    return candidates, out_wd, out_ht

def is_curved(points, threshold=2.0):
    pts = np.array(points)

    # fit line: y = mx + c
    x = pts[:,0]
    y = pts[:,1]

    coeffs = np.polyfit(x, y, 1)
    y_pred = coeffs[0]*x + coeffs[1]

    error = np.mean((y - y_pred)**2)

    return error > threshold

def is_curved_straight(bbox):
    points = np.array(bbox)
    poly_area = cv2.contourArea(points)
    rect = cv2.minAreaRect(points)
    w, h = rect[1]
    rect_area = w * h
    ratio = poly_area / (rect_area + 1e-6)
    is_curved = ratio < 0.7

    return is_curved


def is_curved_bbox(points):
    pts = np.array(points)
    x_range = pts[:,0].max() - pts[:,0].min()
    y_range = pts[:,1].max() - pts[:,1].min()

    return y_range / x_range > 0.3

def evaluate_metric(gt_file,result_file):
    with open(gt_file,'r') as fl:
        gt_dict = json.load(fl)

    with open(result_file,'r') as fl:
        res_data = fl.readlines()

    gt_txt = []
    summ_txt = []
    summ_vect = []
    glb_gt = []
    glb_summ = []
    fl = open('gt_usr_txt_results.txt','w')
    for line in res_data:
        line_data = line.split(',')

        if len(line_data)<3:
            continue
        k = line_data[0].split('/')[5].split('.')[0]
        k = 'poly_gt_'+k
        print(k)
        vl = gt_dict[k]
        for ent in vl["entities"]:
            if len(ent["ornt"]) >= 1 and ent["ornt"][0] != "c" and ent["ornt"][0] != "#":
                txt = ent["text"][0].lower()

                gt_txt.append(txt)
                fl.write(f'{txt},')
        fl.write('::')
        for i in range(1,len(line_data)-1,2):
            txt = line_data[i+1]
            if txt!= 'c' and txt!='lowv' and txt!='non-eng' and txt!='lowc' and txt!='lowt':
                summ_txt.append(txt)
                fl.write(f'{txt},')
        fl.write('::')
        for txt in gt_txt:
            if txt in summ_txt:
                summ_vect.append(1)
                fl.write('1,')
            else:
                summ_vect.append(0)
                fl.write('0,')
        fl.write('\n')
        glb_gt.extend(gt_txt)
        glb_summ.extend(summ_vect)
        summ_vect = []
        gt_txt = []
        summ_txt = []

        # acc = mean(summ_vect)
        # print(f"Accuracy:{acc}\n")
    fl.close()
    return glb_gt, glb_summ

def batch_pad_width(batch):
    widths = [t.shape[-1] for t in batch]
    max_w = max(widths)
    # padded_batch = []
    # for t in batch:
    #     pad_w = max_w - t.shape[-1]
    #
    #     padded = torch.nn.functional.pad(
    #         t, (0, pad_w, 0, 0), value=0
    #     )
    #
    #     padded_batch.append(padded)
    # return torch.cat(padded_batch, dim=0), widths, max_w
    B = len(batch)
    batch_tensor = torch.zeros((B, 1, 32, max_w), dtype=torch.float32).cuda()

    for i, t in enumerate(batch):
        w = t.shape[-1]
        batch_tensor[i, :, :, :w] = t
    return batch_tensor, widths, max_w

@torch.no_grad()
def demo(args, extract = False, model=None):
    filenames = []
    with open('../Dataset/TotalText/test.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/Test/' + filename)

    if model is None:
        pt_mdl = os.path.join(os.getcwd(),'weights/last.pt')
        model = torch.load(pt_mdl,weights_only = False)
        model = model['model']
        model = model.float()
        model.cuda()

    model.eval()
    model.info()
    mean = np.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
    std = np.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')

    # print('load trained model...')
    crnn = CRNN(1, 37, 256)
    crnn = crnn.cuda()
    crnn_state_dict = torch.load('./text_crnn/trained_model_crnn/crnn.pth', weights_only=False)
    # print(f'crnn.pth state dict\n{crnn_state_dict.keys()}\n')

    crnn.load_state_dict(crnn_state_dict)
    # print("Success")

    crnn.eval()
    fout = open('./results_text_extract.txt', 'w')
    for filename in tqdm.tqdm(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        x,shape = dbnet_image_formated(image,args.input_size,mean,std)
        output = model(x)
        output = util.mask_to_box(targets={'shape': [shape]}, outputs=output.cpu(), is_polygon=True)
        # output = util.mask_to_box(targets={'shape': [shape]}, outputs=output.cpu(), is_polygon=False)

        boxes, scores = output[0][0], output[1][0]
        # for box in boxes:
        #     box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
        #     cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=5)
        #
        # cv2.imwrite(f'./data_boxes/{os.path.basename(filename)}', image)
        fout.write(f'\n{filename},')
        for box in boxes:
            # bbox = box[0]
           # score = box[1]
       #  if score > 0.7:
            vertical = curved = False
            print(len(box))
            fout.write(f'{len(box)},')
            # if is_curved(box):
            # if is_curved_bbox(box):
            if is_curved_straight(box) == True:
                print("Curved")
                fout.write("c,")
                continue
                # if h > w:
                #     crop = np.rot90(crop, k=1)  # or k=3 depending on direction

            box = np.array(box)
            if box[:, 0].min() == 0 and box[:,0].max() == 0 and box[:,1].min() == 0 and box[:,1].max() == 0:
                continue

            crpd_imgs, crpd_w, crpd_h = crnn_crop_bbox(image,box,vertical)
            results = []
            for crpd_img in crpd_imgs:
                timg = dbnet_text_extract(crpd_img, crpd_w, crpd_h)
                out_data = crnn(timg)
                txt, conf, var = predict_text(out_data)

                print(f"text={txt}, conf={conf:.2f}, var={var:.2f}")
                results.append((txt, conf, var))
            best = max(results, key=lambda x: score(x[1], x[2], x[0]))
            final_text = best[0]
            print(f'{final_text}')
            # beam_text = predict_beam_search_text(out_data)
            # print(f'{txt},{beam_text}')
            fout.write(f'{final_text},')

    fout.close()
    # evaluate_metric('tot_txt_grnd_trth_file.json','results_text_extract.txt')

@torch.no_grad()
def init_model(model=None):
    if model is None:
        pt_mdl = os.path.join(os.getcwd(),'weights/last.pt')
        model = torch.load(pt_mdl,weights_only = False)
        model = model['model']
        model = model.float()
        model.cuda()

    model.eval()

    crnn = CRNN(1, 37, 256)
    crnn = crnn.cuda()
    crnn_state_dict = torch.load('./text_crnn/trained_model_crnn/crnn.pth', weights_only=False)

    crnn.load_state_dict(crnn_state_dict)

    crnn.eval()
    return model,crnn

@torch.no_grad()
def demo_process(summ_img_clr,summ_img_gry,model,crnn):
    mean = np.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
    std = np.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')
    #standard Luma transform (Y = 0.299R + 0.587G + 0.114B), which better matches human perception.
    # standard Luma transform (Y = 0.299R + 0.587G + 0.114B), which better matches human perception.
    # Grayscale Mean: Calculate as a weighted sum of the RGB means.
    # mean = 0.299 * 0.485 + 0.587 * 0.456 + 0.114 * 0.406 ≈ 0.459
    # Grayscale Standard Deviation: Use the properties of variance to calculate the new standard deviation.
    # Variance = (0.299² * std_R²) + (0.587² * std_G²) + (0.114² * std_B²)
    # std = sqrt(Variance) ≈ 0.226
    # Derived Grayscale values
    # mean = np.array([0.459]).reshape((1, 1, 1)).astype('float32')
    # std = np.array([0.226]).reshape((1, 1, 1)).astype('float32')
    #
    x,shape = dbnet_image_formated(summ_img_clr,mean,std)
    output = model(x)
    output = util.mask_to_box(targets={'shape': [shape]}, outputs=output.cpu(), is_polygon=True)

    boxes, scores = output[0][0], output[1][0]
    words_list = []
    batch = []

    for box in boxes:
        vertical = curved = False
        # print(len(box))
        if is_curved_straight(box) == True:
            # print("Curved")
            continue
        box = np.array(box)
        if box[:, 0].min() == 0 and box[:,0].max() == 0 and box[:,1].min() == 0 and box[:,1].max() == 0:
            continue
        crpd_imgs, crpd_w, crpd_h = crnn_crop_bbox(summ_img_gry,box,vertical)
        timg = dbnet_text_extract_gray(crpd_imgs[0])
        timg1 = dbnet_text_extract_gray(crpd_imgs[1])
        batch.append(timg)
        batch.append(timg1)
    if len(batch)<2:
        return []
    # batch = torch.cat(batch, dim=0) fails for different widths of batch of tensors
    padded_batch, orig_wdths, mx_wdth = batch_pad_width(batch)
    out_data = crnn(padded_batch)
    out_len = out_data.size(1)
    for i in range(0,out_len,2):
        results = []
        for j in range(2):
            data = out_data[:,i+j,:]
            data = data.unsqueeze(1)
            txt, conf, var = predict_text(data,orig_wdths[i+j],mx_wdth)
            # print(f"text={txt}, conf={conf:.2f}, var={var:.2f}")
            wrd_scr = word_score(txt, conf, var)
            if wrd_scr > 0:
                results.append((wrd_scr, txt))

        if results:
            best_score, best_text = max(results)
            final_text = filter_text(best_text)
            if final_text != "":
                words_list.append(final_text)

        # batch = []
        # for crpd_img in crpd_imgs:
        #     timg = dbnet_text_extract_gray(crpd_img)
            # batch.append(timg)
        # batch = torch.cat(batch,dim=0)
        # out_data = crnn(batch)
        #
        # for i in range(len(crpd_imgs)):
        #     txt, conf, var = predict_text(out_data[i].unsqueeze(0))
        #     # print(f"text={txt}, conf={conf:.2f}, var={var:.2f}")
        #     wrd_scr = word_score(txt, conf, var)
        #     if wrd_scr != -1:
        #         results.append((wrd_scr, txt))
        #
        # if results:
        #     best_score, best_text = max(results)
        #     final_text = filter_text(best_text)
        #     if final_text != "":
        #         words_list.append(final_text)
        # del batch
        # torch.cuda.empty_cache()
    return words_list

def profile(args):
    import thop
    model = nn.DBNet().fuse()
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(copy.copy(model), inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=800, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    # profile(args)

    if args.train:
        train(args)
    if args.test:
        res = test(args)
        # print(res)
        with open('test_results_002.txt','w') as fout:
            for res_item in res:
                fout.write(f'{res_item}\n')

    if args.demo:
        text_extract = True
        # demo(args)
        gt_summ, usr_summ = evaluate_metric('tot_txt_grnd_trth_file.json','results_text_1_extract.txt')
        acc = np.mean(usr_summ)
        # with open('gt_usr_results.txt','w') as fl:
        #     fl.write("|".join(gt_summ))
        #     fl.write("|".join(str(usr_summ)))
        print(f'Accuracy:{acc}\n')
        print("Accuracy considering only horizontal and multi oriented but not curved")


def process_text_depth_segment(frm,mdl1,crnn1):
    txt = demo_process(frm,mdl1,crnn1)
    print(txt)
# if __name__ == "__main__":
    # main()
# mdl,crnn = init_model()
# process_text_depth_segment(cur_frm_data,mdl,crnn)


# Tiny CNN Classifier (Recommended)
#
# Train a binary classifier:
#
# input: cropped text image
# output: English / Non-English
# Model:
# MobileNetV3-small
# Size:
# < 5 MB
# Speed:
# < 1 ms per crop
# Training data:
#
# English text crops
#
# Chinese text crops (easy to collect)
#
# Output:
# if prob_english < 0.7:
#     skip CRNN
# | Case    | Confidence | Variance |
# | ------- | ---------- | -------- |
# | English | ~0.8–0.95  | low      |
# | Chinese | ~0.3–0.6   | high     |
