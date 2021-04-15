import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from flask import Flask, render_template , request , jsonify, send_file
from PIL import Image
import os , io , sys
import numpy as np
import uuid
from flask import Flask, render_template , request , jsonify, send_from_directory
from py_edamam import Edamam
from flask import Response
from flask import jsonify, make_response


app = Flask(__name__)

save_dir = '/root/Food_Calorie_Estimation/detect/images'
weights = 'weights/weights/best2.pt'  # Custom trained weights
processed_image_folder = os.path.join(app.root_path, 'runs/detect/exp')

@app.route('/processed_image/<path:filename>', methods=['GET'])
def processed_image(filename):
    return send_from_directory(processed_image_folder, filename)


def detect(source, save_img=False):
    imgsz = 640   # Image resolution
    res = {} # Response
    detected_classes = []
    view_img, save_txt = opt.view_img, opt.save_txt
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path(opt.project) / opt.name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    detected_classes.append(names[int(c)])
                    # n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        res['url'] = str("/processed_image/") + str(source.split("/")[-1])

    res['classes'] = detected_classes
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(res)
    return res


@app.route('/' , methods=['POST'])
def get_image():
    try:
        #import pdb;pdb.set_trace()
        image = request.files['image'].read() # Holds byte image
        npimg = np.fromstring(image, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        print('[INFO] Saving Image..')
        filename = str(uuid.uuid4()) + str('.jpg')
        img_path = os.path.join(save_dir,filename)
        cv2.imwrite(img_path, img)
        print('[INFO] Image Saved!')   

        #Detect function call
        print(img_path)
        
        res = detect(img_path)
        
        if len(res["classes"]) == 0:            
            response={
                "status": False,
                "message": "Classes not found.",
                "data":res

            }
            return make_response(jsonify(response),400)
            
        #fl=send_file(filename,mimetype='image/jpg')
        # return(send_file(img_path,mimetype='image/jpg'))
        response={
            "status":True,
            "data":res,
            "message":"Following classes detected."

        }
        return make_response(jsonify(response), 200)
    except Exception:
        response={
         "status":False,
         "message":"Image not loaded."   
        }
        return make_response(jsonify(response),400)
    #return response



def get_calories(class_name, volume=100):
    query = str(volume) + " gm " + str(class_name)
    api = Edamam(
        nutrition_appid='92a2c29f',
        nutrition_appkey='caffdf3cb786a101c0a8a74138b15a08',
    )

    calories = api.search_nutrient(query)
    return calories['calories']


@app.route('/cal' , methods=['POST'])
def calories():
    l=[ 'rice','eels-on-rice','pilaf','chicken-n-egg-on-rice','pork-cutlet-on-rice','beef-curry','sushi',
'chicken-rice','fried-rice','tempura-bowl','bibimbap','toast','croissant','roll-bread','raisin-bread',
'chip-butty','hamburger','pizza','sandwiches','udon-noodle','tempura-udon','soba-noodle','ramen-noodle',
'beef-noodle','tensin-noodle','fried-noodle','spaghetti','Japanese-style-pancake','takoyaki','gratin',
'sauteed-vegetables','croquette','grilled-eggplant','sauteed-spinach','vegetable-tempura','miso-soup',
'potage','sausage','oden','omelet','ganmodoki','jiaozi','stew','teriyaki-grilled-fish','fried-fish',
'grilled-salmon','salmon-meuniere','sashimi','grilled-pacific-saury-','sukiyaki','sweet-and-sour-pork',
'lightly-roasted-fish','steamed-egg-hotchpotch','tempura','fried-chicken','sirloin-cutlet','nanbanzuke',
'boiled-fish','seasoned-beef-with-potatoes','hambarg-steak','beef-steak','dried-fish','ginger-pork-saute',
'spicy-chili-flavored-tofu','yakitori','cabbage-roll','rolled-omelet','egg-sunny-side-up','fermented-soybeans',
'cold-tofu','egg-roll','chilled-noodle','stir-fried-beef-and-peppers','simmered-pork','boiled-chicken-and-vegetables',
'sashimi-bowl','sushi-bowl','fish-shaped-pancake-with-bean-jam','shrimp-with-chill-source',
'roast-chicken','steamed-meat-dumpling','omelet-with-fried-rice','cutlet-curry','spaghetti-meat-sauce',
'fried-shrimp','potato-salad','green-salad','macaroni-salad','Japanese-tofu-and-vegetable-chowder',
'pork-miso-soup','chinese-soup','beef-bowl','kinpira-style-sauteed-burdock','rice-ball',
'pizza-toast','dipping-noodles','hot-dog','french-fries','mixed-rice','goya-chanpuru',
'apple pie','baby back ribs','bread pudding',   'breakfast burrito',   'cheesecake','chicken curry',   
'chocolate cake',  'waffles','hot dog', 'spring roll', 'donuts','french fries','tacos','garlic bread',  
'pork chop','pancakes','ice cream','samosa','pizza']
    res = {}
    if request.method == 'POST':
        content = request.form
        #import pdb;pdb.set_trace()
        # print(len(content))
        #for i in content:
        try:
            if content['class_name'] not in l:
                response={
                    "status":False,
                    "message":"Dishes not found"
                }
                return make_response(jsonify(response),400)
            if len(content['volume']) ==0:
                response={
                    "status":False,
                    "message":"Volume should not be empty not found"
                }
                return make_response(jsonify(response),400)
                
            cal = get_calories(content['class_name'], content['volume'])
            response_class=content['class_name']
            print(response_class)

            
            res[response_class]=cal

            response={
                "status": True,
                "data": {'calories':cal,'class_name':response_class},
                "message": "Calorie count"
            }   
            return make_response(jsonify(response), 200)
        except Exception:
            response={
                "status":False,
                "message":"Dish not found."
            }
            return make_response(jsonify(response),400) 


        
        
        
#            return Response(response,status=200)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.20, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print('[INFO]',opt)
    check_requirements(exclude=('pycocotools', 'thop'))
    
    app.run(host='0.0.0.0', port='5002', debug=True)
