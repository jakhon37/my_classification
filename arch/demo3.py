import cv2
import numpy as np
import onnxruntime as ort
import onnx
import torch
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
from utils import cxy_wh_2_rect, hann1d, hann2d, img2tensor
import math 


class Tracker(object):
    """Wraps the tracker for evaluation and running purposes."""
    def __init__(self, model_path: str) -> None:
        self.onnx_model = onnx.load(model_path) 
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.template_factor = 2.0
        self.search_factor = 4.0
        self.template_size = 128
        self.search_size = 256
        self.stride = 16
        self.feat_sz = self.search_size // self.stride
        self.output_window = hann2d(np.array([self.feat_sz, self.feat_sz]), centered=True)
        self.z = None
        self.state = None
        

    def initialize(self, image, target_bb):
        # get subwindow
        z_patch_arr, resize_factor, z_amask_arr = self.sample_target(image, target_bb, self.template_factor,
                                                    output_sz=self.template_size)
        # nparry -> onnx input tensor
        self.z = img2tensor(z_patch_arr)
        # get box_mask_z
        self.box_mask_z = self.generate_mask_cond()
        # save states
        self.state = target_bb
    
    def track(self, image):
        img_H, img_W, _ = image.shape
        
        # get subwindow
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(image, self.state, self.search_factor,
                                                    output_sz=self.search_size)
        # nparry -> onnx input tensor
        x = img2tensor(x_patch_arr)
        outputs = self.ort_session.run(None, {'z': self.z.astype(np.float32), 'x': x.astype(np.float32)})
        
        out_score_map = outputs[1]
        out_size_map = outputs[2]
        out_offset_map = outputs[3]

        # add hann windows
        response = self.output_window * out_score_map
        pred_boxes = self.cal_bbox(response, out_size_map, out_offset_map)
        pred_box = (pred_boxes * self.search_size / resize_factor).tolist()
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), img_H, img_W, margin=10)

        return self.state
    

    def sample_target(self, im, target_bb, search_area_factor, output_sz):
        """Extracts a square crop centered at target_bb box, of are search_area_factor^2 times target_bb area
        args: 
            im - cv image
            target_bb - target box [x_left, y_left, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = list(target_bb)
        else:
            x, y , w, h, sc = target_bb
        x1,y1, x2,y2 = x,y, w,h
        # crop image
        # crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        # if crop_sz < 1:
        #     raise Exception("Too small bounding box.")
        
        # cx, cy = x + 0.5 * w, y + 0.5 * h
        # x1 = round(cx - crop_sz * 0.5)
        # y1 = round(cy - crop_sz * 0.5)

        # x2 = x1 + crop_sz
        # y2 = y1 + crop_sz 

        # x1_pad = max(0, -x1)
        # x2_pad = max(x2 - im.shape[1] + 1, 0)

        # y1_pad = max(0, -y1)
        # y2_pad = max(y2 - im.shape[0] + 1, 0)
        # Calculate padding amounts for each side
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]


        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        print(f'imcrop: {im_crop.shape}')
        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H,W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz))

        return im_crop_padded, resize_factor, att_mask
    
    def transform_bbox_to_crop(self, box_in: list, resize_factor, crop_type='template', normalize=True) -> list:
        """Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
        args:
            box_in: list [x1, y1, w, h], not normalized, the box for which the co-ordinates are to be transformed
            resize_factor - the ratio between the original image scale and the scale of the image crop

        returns:
            List - transformed co-ordinates of box_in
        """
        
        if crop_type == 'template':
            crop_sz = self.template_size
        elif crop_type == 'search':
            crop_sz = self.search_size
        else:
            raise NotImplementedError
        
        box_out_center_x = (crop_sz[0] - 1) / 2
        box_out_center_y = (crop_sz[1] - 1) / 2
        box_out_w = box_in[2] * resize_factor
        box_out_h = box_in[3] * resize_factor

        # normalized
        box_out_x1 = (box_out_center_x - 0.5 * box_out_w)
        box_out_y1 = (box_out_center_y - 0.5 * box_out_h)
        box_out = [box_out_x1, box_out_y1, box_out_w, box_out_h]

        if normalize:
            return [i / crop_sz for i in box_out]
        else:
            return box_out
        
    def generate_mask_cond(self):
        template_size = self.template_size
        stride = self.stride
        template_feat_size = template_size// stride # 128 // 16 = 8

        # MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT'

        box_mask_z = np.zeros([1, template_feat_size, template_feat_size])
        box_mask_z[:, slice(3, 4), slice(3, 4)] = 1
        box_mask_z = np.reshape(box_mask_z, (1, -1)).astype(np.int32)

        return box_mask_z
    
    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        score_map_ctr = torch.from_numpy(score_map_ctr)
        size_map = torch.from_numpy(size_map)
        offset_map = torch.from_numpy(offset_map)
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
        bbox = bbox.numpy()[0]

        if return_score:
            return bbox, max_score
        return bbox
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W-margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H-margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2-x1)
        h = max(margin, y2-y1)
        return [x1, y1, w, h]

def run1(tracker, video_path):
    '''
    tracker: mobilevit-track
    video_path: 0 or video path
    '''
    img_list = sorted(os.listdir(video_path))

    for frame_count, img_path in enumerate(img_list):

        frame = cv2.imread(os.path.join(video_path, img_path))
        frame_track = frame.copy()

        # 0. Init Tracker
        if frame_count == 0:
            bbox = (196,51,139,194)
            tracker.initialize(frame_track, bbox)
        else:
            bbox = tracker.track(frame_track) # bbox (x, y, w, h)
        x, y, w, h = bbox

        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        location = cxy_wh_2_rect(target_pos, target_sz)

        x1, y1, x2, y2 = int(location[0]), int(location[1]), \
            int(location[0] + location[2]), int(location[1] + location[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("img", frame)
        cv2.waitKey(1)

def detect_and_draw(image, model, font):
    # Run YOLO inference on the image
    results = model(image, device='cpu', imgsz=384, conf=0.05, iou=0.15)

    # Convert to RGB for PIL processing
    image = Image.fromarray(image)
    image_d = image.copy()
    draw = ImageDraw.Draw(image_d)

    bbox_to_track = []  # Store the bbox for the first detection

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls)
            confidence = float(box.conf)
            label = f"{r.names[cls_idx]}: {confidence:.2f}"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Draw bounding box and label on the image
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), label, fill="red", font=font)

            # Convert to (x, y, w, h) format for tracker initialization
            if len(bbox_to_track)<1 and label=='person':  # Take the first object for tracking
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:  # Ensure valid bbox
                    bbox_to_track = (x1, y1, w, h)

    if len(bbox_to_track)<1:
        print("No object detected by YOLO in the current frame.")
    else:
        print(f"YOLO detected object with bounding box: {bbox_to_track}")

    return np.array(image_d), bbox_to_track


# Object Detection using YOLOv8
def detect_and_draw1(image, model, font):
    # Run YOLO inference on the image
    # results = model(image, device=0, imgsz=384, conf=0.1, iou=0.15)
    results = model(image, device='cpu', imgsz=384, conf=0.1, iou=0.15)

    # Convert to RGB for PIL processing
    image = Image.fromarray(image)
    image_d = image.copy()
    draw = ImageDraw.Draw(image_d)

    cropped_images = []
    bboxes = []
    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls)
            confidence = float(box.conf)
            label = f"{r.names[cls_idx]}: {confidence:.2f}"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            if label=='person':
                bboxes.append((x1, y1, x2, y2))
            # Draw bounding box and label on the image
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), label, fill="red", font=font)

            # Crop the detected object
            cropped_img = np.array(image.crop((x1, y1, x2, y2)))
            cropped_images.append(cropped_img)

    return np.array(image_d), cropped_images, bboxes

def run(tracker, video_path, output_path, model, font):
    '''
    tracker: MobileViT tracker object
    video_path: Path to input video file
    output_path: Path to save the output video with tracking
    model: YOLO object detection model
    font: Font used for drawing labels
    '''
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get frame width, height, and frames per second (FPS) from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Progress bar setup
    pbar = tqdm(total=total_frames, desc="Processing Video Frames")

    frame_count = 0
    bbox = None  # Bounding box for tracking
    init_frame = True 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading video file.")
            break


        # Copy frame for tracking
        frame_track = frame.copy()
        frame, bbox = detect_and_draw(frame, model, font)
        # bbox = bboxes[0]
        print(f'box: {bbox}')
        # if bbox: print(f'box len: {len(bbox)}')
        # Initialize Tracker on the first frame with a bounding box (x, y, w, h)
        if init_frame and len(bbox)>0:
                    # Run object detection on the frame

            if len(bbox)>0:
                # bbox = (196, 51, 139, 194)  # Initial bounding box (replace with your own)
                tracker.initialize(frame_track, bbox)
                init_frame = False
            else: 
                print("No initial bounding box detected. Skipping initialization.")
                # continue   # You can decide whether to skip or break here
        elif not init_frame:
            # Track the object in subsequent frames
            if len(bbox)>0:
                bbox = tracker.track(frame_track)  # bbox (x, y, w, h)
        if len(bbox)>0:
            x, y, w, h, sc = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Write the processed frame to the output video file
        out.write(frame)

        # Update progress bar
        pbar.update(1)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    pbar.close()


if __name__ == "__main__":
    onnxpath = '/home/aivar/deep/track/job/MVT/pretrained_models/mobilevit_256_128x1_got10k_ep100_cosine_annealing/MobileViT_Track_ep0300.onnx'
    mvt_track = Tracker(model_path=onnxpath)
    
    video_path = '/home/aivar/deep/track/job/MVT/assets/test_data/1.mp4'  # Input video path
    output_path = '/home/aivar/deep/track/job/MVT/assets/output_video.mp4'  # Output video path

    # Load YOLO model for object detection
    weightPath = '/home/aivar/deep/track/job/yolov_face/weights/yolov8n_face_relu6.pt'
    weightPath='/home/aivar/deep/track/job/yolov_face/weights/yolov8n.pt'
    model = YOLO(weightPath)

    # Load the font for drawing labels
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    # Run tracker and object detection
    run(mvt_track, video_path, output_path, model, font)
    
# if __name__ == "__main__":
#     onnxpath = '/home/aivar/deep/track/job/MVT/pretrained_models/mobilevit_256_128x1_got10k_ep100_cosine_annealing/MobileViT_Track_ep0300.onnx'
#     mvt_track = Tracker(model_path=onnxpath)
    
#     video_path = '/home/aivar/deep/track/job/MVT/assets/test_data/1.mp4'  # Input video path
#     output_path = '/home/aivar/deep/track/job/MVT/assets/output_video.mp4'  # Output video path
    
#     run(mvt_track, video_path, output_path)
    
    
# if __name__ == "__main__":
    
#     onnxpath = '/home/aivar/deep/track/job/MVT/pretrained_models/mobilevit_256_128x1_got10k_ep100_cosine_annealing/MobileViT_Track_ep0300.onnx'
#     mvt_track = Tracker(model_path=onnxpath)
#     video_path = '/home/aivar/deep/track/job/MVT/assets/test_data'
#     run(mvt_track, video_path)