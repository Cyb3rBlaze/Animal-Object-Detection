import torchvision.transforms as transforms
import torchvision

from PIL import Image, ImageDraw

''' apply non-max suppression - pulled from 
    https://colab.research.google.com/drive/1NziO_b-SW9KmWFh-6C8to9H_QAdpmCBZ?usp=sharing#scrollTo=5fsSRjrIh_k5
    
    Also refer: https://pytorch.org/vision/main/generated/torchvision.ops.nms.html'''
def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'].cpu(), orig_prediction['scores'].cpu(), iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'].cpu()[keep]
    final_prediction['scores'] = final_prediction['scores'].cpu()[keep]
    final_prediction['labels'] = final_prediction['labels'].cpu()[keep]
    
    return final_prediction

def inverse_transform(image, annotations, fp, save=True):
    # image recovering inverse transform
    transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
        transforms.ToPILImage()])

    image = transform(image)
        
    # (x0, y0), (x1, y1)
    dims = image.size
    draw = ImageDraw.Draw(image)

    annotations = apply_nms(annotations, iou_thresh=0.01)

    # max 5 bounding boxes - can also use score thesholding + non-max suppression
    for i in range(len(annotations["boxes"])):
        pred_class = ""
        # buffalo
        if annotations["labels"][i] == 1:
            pred_class = "red"
        # elephant
        elif annotations["labels"][i] == 2:
            pred_class = "green"
        # rhino
        elif annotations["labels"][i] == 3:
            pred_class = "blue"
        # zebra
        elif annotations["labels"][i] == 4:
            pred_class = "orange"
        # background
        else:
            pred_class = "yellow"

        draw.rectangle([annotations["boxes"][i][0], annotations["boxes"][i][1], annotations["boxes"][i][2], annotations["boxes"][i][3]], outline=pred_class, width=3)
    
    if save:
        image.save(fp)