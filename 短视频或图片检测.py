
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
        nosave=False,  # do not save images/videos
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
    source = input("???????????????????????????????????????????????????")
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # ???????????????????????????
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

    object_detection = input("??????????????????????????????truck???car???bird?????????")
    object_list = object_detection.split(',')
    warning_time = input("??????????????????????????????s??????")
    warning_time = int(warning_time)

    iter = dataset.__iter__()
    _, _, img0s, _, _, im_width, im_height = dataset.__next__()
    im = img0s
    points, lt_point, rb_point, hl, wl, width, height = {}, {}, {}, {}, {}, {}, {}
    x_coord = []
    y_coord = []

    tpPointsChoose = []
    drawing = False
    tempFlag = False
    def draw_ROI(event, x, y, flags, param):
        global point1, tpPointsChoose, pts, drawing, tempFlag
        if event == cv2.EVENT_LBUTTONDOWN:
            tempFlag = True
            drawing = False
            point1 = (x, y)
            tpPointsChoose.append((x, y))  # ????????????
        if event == cv2.EVENT_RBUTTONDOWN:
            tempFlag = True
            drawing = True
            pts = np.array([tpPointsChoose], np.int32)
            pts1 = tpPointsChoose[1:len(tpPointsChoose)]
            print(pts1)
        if event == cv2.EVENT_MBUTTONDOWN:
            tempFlag = False
            drawing = True
            tpPointsChoose = []

    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_ROI)
    cv2.resizeWindow("image", screensize)  # ??????????????????

    if (tempFlag == True and drawing == False):  # ????????????
        cv2.circle(im, point1, 5, (0, 255, 0), 5)
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(im, tpPointsChoose[i], tpPointsChoose[i + 1], (255, 0, 0), 2)
    if (tempFlag == True and drawing == True):  # ????????????
        cv2.polylines(im, [pts], True, (0, 0, 255), thickness=2)
    if (tempFlag == False and drawing == True):  # ????????????
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(im, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)

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

    if source.split('.')[-1] == 'mp4':
        save_path = str(save_dir / 'detect_result.mp4')
        image_width = im_width
        image_height = im_height
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 30, (int(image_width), int(image_height)))

    m = 0
    time_record, detection_result, warning_count = [], {}, []
    number = 1
    frame_count = 6
    for path, img, im0s, vid_cap, s, image_width, image_height in dataset:
        # mask for certain region
        if (number % frame_count == 0):
            mask = np.zeros([img.shape[1], img.shape[2]], dtype=np.uint8)
            # mask = np.zeros([img.shape[1], img.shape[2]], dtype=np.uint8)
            positions = {}
            mask_region =[]
            for position in range(points_count):
                positions[f'position_{position}:'] = np.array([[int(img.shape[2] * wl[f'point_{position}:'][0]), int(img.shape[1] * hl[f'point_{position}:'][0])],  # pts1
                                [int(img.shape[2] * wl[f'point_{position}:'][1]), int(img.shape[1] * hl[f'point_{position}:'][1])],  # pts2
                                [int(img.shape[2] * wl[f'point_{position}:'][2]), int(img.shape[1] * hl[f'point_{position}:'][2])],  # pts3
                                [int(img.shape[2] * wl[f'point_{position}:'][3]), int(img.shape[1] * hl[f'point_{position}:'][3])]], np.int32)
                mask_region.append(positions[f'position_{position}:'])
            mask = cv2.fillPoly(mask, mask_region, (255, 255, 255))
            imgc = img.transpose((1, 2, 0))
            imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
            img = imgc.transpose((2, 0, 1))

            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # ???????????????????????????
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #????????????????????????????????? (NMS) ???????????????????????????
            dt[2] += time_sync() - t3

            # Process predictions
            target = []
            for i, det in enumerate(pred):  # per image
                seen += 1
                # p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                p, s, im0, frame = path[i], f'{i}: ', im0s, dataset.count
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

                    user32 = ctypes.windll.user32
                    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(str(p), on_EVENT_LBUTTONDOWN)
                    cv2.resizeWindow(str(p), screensize)  # ??????????????????
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

            if source.split('.')[-1] == 'png' or source.split('.')[-1] == 'jpg':
                cv2.imwrite(str(save_dir) + '/'  + 'detect_result.jpg', im0)
            else:
                vid_writer.write(im0)   #????????????????????????
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        number += 1


        if f'result{m}' in detection_result:     #???????????????????????????
            det_ob = any(True if ob in detection_result[f'result{m}'] else False for ob in object_list)
            if det_ob:     #??????????????????????????????????????????????????????
                t = time_sync()      #??????
                time_record.append(t)
                if max(time_record) - min(time_record) >= warning_time:        #???????????????????????????????????????????????????????????????
                    time_record.clear()     #??????????????????
                    detection_result.clear()   #??????????????????
                    print("Long stay is prohibited in this area! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")       #??????????????????

                    warning_count.append('warning')         #??????????????????
                    if len(warning_count) == 3:        #??????????????????????????????????????????
                        print("Please leave this area immediately! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")     #???????????????????????????????????????????????????
                        warning_count.clear()          #????????????????????????
                        if save_img:            #????????????????????????
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
    parser.add_argument('--view-img', default = True, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default = False, action='store_true', help='do not save images/videos')
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
