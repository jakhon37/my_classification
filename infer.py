import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
from models import *
from config import config
from utils import load_checkpoint

def get_transform():
    # Use the same transformations as validation set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize using the same mean and std as training
        # transforms.Normalize(mean=[...], std=[...])
    ])
    return transform

def load_model(model_name, checkpoint_path, num_classes, device):
    model_class = globals()[model_name]
    model = model_class(num_classes).to(device)
    model, _ = load_checkpoint(checkpoint_path, model)
    model.eval()
    return model

def predict_image(model, image_path, transform, device, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        # print(f'_, predicted: {_, predicted}')
        class_idx = predicted.item()
        class_name = class_names[class_idx]
    return class_name

def main():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--image', type=str, required=False, 
                        default='/home/aivar/deep/classify/datasets/dog&cat/test/cats/cat-4004_jpg.rf.48897ede3bd454536f55e8fa612bf5c7.jpg', #'/home/aivar/deep/classify/datasets/dog&cat/test/dogs/dog-4063_jpg.rf.7a4b40a0416db57ac2cb8ec4a69c74fe.jpg' , #
                        help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default=config['model_name'], help='Model name')
    parser.add_argument('--classes', type=str, default=None, help='Path to class names file')
    args = parser.parse_args()
    
    device = config['device']
    transform = get_transform()
    
    # Load class names
    if args.classes:
        with open(args.classes, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
            # class_names = ['cat', 'dog']
            
    else:
        # Provide default class names if not specified
        class_names = [str(i) for i in range(config['num_classes'])]
        class_names = ['cat', 'dog']
    
    # Load model
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config['save_dir'], f"{args.model}_best.pth")
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config['save_dir'], f"{args.model}_epoch_10.pth")
    # checkpoint_path = os.path.join(config['save_dir'], f"{config['model_name']}_epoch_10.pth")
    
    model = load_model(args.model, checkpoint_path, config['num_classes'], device)
    
    # Predict
    class_name = predict_image(model, args.image, transform, device, class_names)
    print(f"The predicted class is: {class_name}")

if __name__ == '__main__':
    main()
