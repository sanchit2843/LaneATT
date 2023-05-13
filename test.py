import torch
from lib.config import Config
from lib.experiment import Experiment
import argparse
import cv2 
import numpy as np


GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# helper function for drawing the annotation
def draw_annotation( pred=None, img=None):

    color = PRED_HIT_COLOR
    for i, l in enumerate(pred):
        points = l.points
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        # points += pad
        xs, ys = points[:, 0], points[:, 1]
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                            tuple(curr_p),
                            tuple(next_p),
                            color=color,
                            thickness=3)
    return img

def main():
    # findin the model path 
    # Change the model name for different models, can be a parameter too, but this is easy for debugging
    model_name = "mobileone"
    model_name_path = "laneatt_"+model_name+"_tusimple"
    cfg_path = "./experiments/"+model_name_path+"/config.yaml"
    #loading the config file 
    cfg = Config(cfg_path)
    # getting the model using the config file
    model =  cfg.get_model()
    # model path created
    model_path =  "./experiments/"+model_name_path+"/models/model_0100.pt"
    model.load_state_dict(torch.load(model_path)['model'])
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model = model.to(device)
    model.eval()

    # for loading an image  
    # pipeline to get the images
    # image_path =  "./image/3.jpg"#"./image/delhi.png"
    # image = cv2.imread(image_path)
    # for loading a video
    input_file = cv2.VideoCapture('test.mp4')

    # Get the frames per second (FPS) and size of the video used for writing in the output video
    fps = int(input_file.get(cv2.CAP_PROP_FPS))
    width = int(input_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through the frames in the input video
    while True:
        # Read the next frame from the input video
        ret, image = input_file.read()
    
        # Stop the loop if we have reached the end of the video
        if not ret:
            break

        # preprocessing the image
        resize_img = cv2.resize(image, (640,360))
        tensor_image = torch.from_numpy(resize_img.transpose((2, 0, 1))).float()
        tensor_image = tensor_image / 255.0
        images = tensor_image.unsqueeze(0) 

        
        with torch.no_grad():
            images = images.to(device)
            test_parameters = cfg.get_test_parameters()
            output = model(images, **test_parameters)
            predictions = model.decode(output, as_lanes=True)
            # visualize the output
            img = (images[0].cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)

            img = draw_annotation( img=img, pred=predictions[0])
            
            cv2.imshow('pred', img)
            key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()




          
    
if __name__ == '__main__':
    main()