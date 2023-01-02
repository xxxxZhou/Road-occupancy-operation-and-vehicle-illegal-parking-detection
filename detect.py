
import argparse
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import ctypes

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        # source=ROOT / 'data/cramera',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'prediction',  # save results to project/name
        name='result',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=4,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = input("请输入摄像头网口地址、视频文件等：")
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 创建文件夹返回路径
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]


    object_detection = input("请输入特定检测目标（truck，car，bird等）：")
    object_list = object_detection.split(',')
    warning_time = input("请输入触发警告时长（s）：")
    warning_time = int(warning_time)

    iter = dataset.__iter__()
    _, _, img0s, _, _ = dataset.__next__()
    im = img0s[0].copy()
    points, lt_point, rb_point, hl, wl, width, height = {}, {}, {}, {}, {}, {}, {}
    x_coord = []
    y_coord = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            x_coord.append(x)
            y_coord.append(y)
            cv2.circle(im, (int(x), int(y)), 4, (0, 0, 255), thickness=5)
            cv2.putText(im, xy, (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 255), thickness=5)
            cv2.imshow("image", im)

    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.resizeWindow("image", screensize)  # 设置窗口大小
    cv2.imshow("image", im)
    cv2.waitKey(0)

    points_count = int(len(x_coord) / 2)
    points['point_0:'] = [x_coord[0], y_coord[0], x_coord[1], y_coord[1]]
    if points_count == 2:
        points['point_1:'] = [x_coord[2], y_coord[2], x_coord[3], y_coord[3]]
    if points_count == 3:
        points['point_1:'] = [x_coord[2], y_coord[2], x_coord[3], y_coord[3]]
        points['point_2:'] = [x_coord[4], y_coord[4], x_coord[5], y_coord[5]]

    for point in range(points_count):
        lt_point[f'point_{point}:'] = points[f'point_{point}:'][:2]
        rb_point[f'point_{point}:'] = points[f'point_{point}:'][2:4]
        width[f'point_{point}:']= abs(points[f'point_{point}:'][2] - points[f'point_{point}:'][0])
        height[f'point_{point}:'] = abs(points[f'point_{point}:'][3] - points[f'point_{point}:'][1])

        hl[f'point_{point}:'] = [points[f'point_{point}:'][1] / im.shape[0], points[f'point_{point}:'][1] / im.shape[0],
                          points[f'point_{point}:'][3] / im.shape[0], points[f'point_{point}:'][3] / im.shape[0]]
        wl[f'point_{point}:'] = [points[f'point_{point}:'][0] / im.shape[1], (points[f'point_{point}:'][0] + width[f'point_{point}:']) / im.shape[1],
                                 (points[f'point_{point}:'][0] + width[f'point_{point}:']) / im.shape[1], points[f'point_{point}:'][0] / im.shape[1]]

    m = 0
    time_record, detection_result, warning_count = [], {}, []
    for path, img, im0s, vid_cap, s in dataset:
        # mask for certain region
        for b in range(0, img.shape[0]):
            # mask = np.zeros([img[b].shape[1], img[b].shape[2]], dtype=np.uint8)
            # mask = np.zeros([img.shape[1], img.shape[2]], dtype=np.uint8)
            positions = {}
            mask_region =[]
            for position in range(points_count):
                positions[f'position_{position}:'] = np.array([[int(img[b].shape[2] * wl[f'point_{position}:'][0]), int(img[b].shape[1] * hl[f'point_{position}:'][0])],  # pts1
                                [int(img[b].shape[2] * wl[f'point_{position}:'][1]), int(img[b].shape[1] * hl[f'point_{position}:'][1])],  # pts2
                                [int(img[b].shape[2] * wl[f'point_{position}:'][2]), int(img[b].shape[1] * hl[f'point_{position}:'][2])],  # pts3
                                [int(img[b].shape[2] * wl[f'point_{position}:'][3]), int(img[b].shape[1] * hl[f'point_{position}:'][3])]], np.int32)
                mask_region.append(positions[f'position_{position}:'])
            mask = cv2.fillPoly(mask, mask_region, (255, 255, 255))
            imgc = img[b].transpose((1, 2, 0))
            imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
            # cv2.imshow('1',imgc)
            img[b] = imgc.transpose((2, 0, 1))

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # 让图片增加一个维度
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #推理结果的非极大值抑制 (NMS) 以拒绝重叠的边界框
        dt[2] += time_sync() - t3

        # Process predictions
        target = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count

            posi = {}
            mask_reg = []
            for pos in range(points_count):
                posi[f'point_{pos}:'] = np.array([lt_point[f'point_{pos}:'], [points[f'point_{pos}:'][0] + width[f'point_{pos}:'], points[f'point_{pos}:'][1]],
                                                  rb_point[f'point_{pos}:'], [points[f'point_{pos}:'][0], points[f'point_{pos}:'][1] + height[f'point_{pos}:']]], np.int32)

                cv2.putText(im0, f'Detection Region {pos+1}', (int(points[f'point_{pos}:'][0] - 5), int(points[f'point_{pos}:'][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 2, cv2.LINE_AA)
                mask_reg.append(posi[f'point_{pos}:'])

            zeros = np.zeros((im0.shape), dtype=np.uint8)
            mask = cv2.fillPoly(zeros, mask_reg, color=(0, 165, 255))
            im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
            cv2.polylines(im0, mask_reg, True, (0, 0, 255), 3)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    target.append(names[int(c)])
                    detection_result[f'result{m}'] = target

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        if f'result{m}' in detection_result:     #判断是否检测到目标
            det_ob = any(True if ob in detection_result[f'result{m}'] else False for ob in object_list)
            if det_ob:     #判断检测的目标是否在设定的检测列表内
                t = time_sync()      #计时
                time_record.append(t)
                if max(time_record) - min(time_record) >= warning_time:        #判断目标在该区域逗留时间是否超过设定的时间
                    time_record.clear()     #计时列表清空
                    detection_result.clear()   #检测结果清空
                    print("Long stay is prohibited in this area! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")       #反馈警告信息

                    warning_count.append('warning')         #记录警告次数
                    if len(warning_count) == 3:        #判断警告次数是否超过设定次数
                        print("Please leave this area immediately! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")     #超过警告次数时，做出最后的警告反馈
                        warning_count.clear()          #清空警告次数列表
                        if save_img:            #保存目标违规图像
                            current_time = time.asctime()
                            cur1 = current_time.replace(" ","_")
                            cur2 = cur1.replace(":","_")
                            cv2.imwrite(str(save_dir) + '/' + str(cur2) + '.jpg',im0)
                    m = 0
            else:
                time_record.clear()
                warning_count.clear()
        else:
            time_record.clear()
            warning_count.clear()
        m += 1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default='http://116.205.185.63/live/YF022010416.flv',help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave',action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'prediction', help='save results to project/name')
    parser.add_argument('--name', default='result', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=4, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)