import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
from PIL import Image


# is a tensor of size (3 x 224 x 224)
class Noise(nn.Module):
    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        self.noise = nn.Parameter(torch.zeros((1, c, h, w)))

    def forward(self, img):
        return self.noise + img


def process_image(pil_image):
    """
    preprocess the input image by resizing and normalizing

    Attributes:
        pil_image : image in pil format
    
    Return:
        input_batch : torch tensor as the input to the model in (n, c, h, w) shape 
    """
    # preprocess the image before feeding to model
    preprocess = transforms.Compose(
        [
            # resize the image to the model input size
            transforms.Resize((224, 224)),
            # convert to tensor format
            transforms.ToTensor(),
            # normalize data to imagenet standard
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(pil_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def reverse_process(input_batch):
    """
    reverse transform the input_batch to the pil image
    invert of process_image

    Attributes:
        input_batch : torch tensor as the input to the model in (n, c, h, w) shape 
    Return:
        pil_image : image in pil format
    """
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    input_batch = invTrans(input_batch)

    pil_image = transforms.ToPILImage()(input_batch.squeeze(0)).convert("RGB")
    return pil_image


def load_model(model_name):
    """
    load a pretrained model from torchvision.models

    Attributes:
        model_name : the name of the model we want to break
    
    Return:
        model : loaded model instance (nn.Module)
    """
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    # add more models here
    # turn model into evaluation mode (ex. disable dropout)
    model.eval()
    return model


def train_noise(
    optimizer,
    celoss,
    model,
    noise,
    target,
    input_batch,
    thres=0.02,
    thres_count=10,
    n=1000,
):
    """
    execute the training process to update noise

    Attributes:
        optimizer: the optimizer we use to update the parameters

        celoss: the loss function

        model: the pretrained model we want to break

        noise: the object of Noise class

        target: the target class for the input image

        input_batch: the input data in batch format (n, c, h, w) shape 
        
        thres: the lower bound of loss we can break

        thres_count: how many times the loss is lower than thres before break

        n: the maximum number of iterations
    """
    # (i.e to detect convergency)
    # increase this to get a more stable value
    for _ in range(n):
        # clear grad
        optimizer.zero_grad()
        # feed to the model
        output = model(noise(input_batch))
        # compute loss
        loss = celoss(output, target)
        if loss < thres:
            thres_count -= 1
            if thres_count <= 0:
                break
        # perform backprob
        loss.backward()
        # update noise
        optimizer.step()


def predict(
    input_batch, model, noise, label_file="imagenet.txt", add_noise=False, top_k=1
):
    """
    make a prediction using model with input_batch

    Attributes:
        input_batch: the input data in batch format (n, c, h, w) shape 

        model: the pretrained model we want to break

        noise: the object of Noise class

        label_file: path the the class/label file

        add_noise: True if we want to predict the noised image

        top_k: return the confidences for top k classes
    
    Return:
        classes(list): all top_k predicted class names (sorted with confidences)

        confidences(list): all top_k predicted confidences (decreasing order)
    """
    with torch.no_grad():
        if add_noise:
            output = model(noise(input_batch))
        else:
            output = model(input_batch)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open(label_file, "r") as f:  # open the class file
        categories = [s.strip() for s in f.readlines()]
    # Show top categories of prediction per image
    topk_prob, topkcatid = torch.topk(probabilities, top_k)

    classes = []
    confidences = []
    for i in range(topk_prob.size(0)):
        classes.append(categories[topkcatid[i]])
        confidences.append(topk_prob[i].item())
    return classes, confidences


def add_noise(
    pil_image,
    model_name,
    label_file,
    target=None,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
):
    """
    the main function to train and add noise to the image to break model with <model_name>

    Attributes:
        pil_image: input image in PIL format (RGB)
        
        model_name: the name of the model we want to break

        label_file: path to the label/class txt file

        target (int/None): the target class if the user defined one

        lr: the learning rate for optimizer

        momentum: the momentum for optimizer

        weight_decay: the weight_decay for optimizer

    Return:
        noised_image: noised image in PIL format

        original_image: original image without transform or noise, but resized

        pure_noise: the noise itself in PIL format

        top_1_class_no_noise(list): the top predicted class without noising the image

        top_1_confidence_no_noise(list): the top predicted confidence without noising the image

        top_1_class_noised(list): the top predicted class with noising the image

        top_1_confidence_noised(list): the top predicted confidence with noising the image
    """
    noise = Noise()
    model = load_model(model_name)
    # if the user sets a target class
    if target is not None:
        target = torch.tensor([target])
    else:
        target = torch.tensor([2])  # great white shark

    input_batch = process_image(pil_image)
    original_image = reverse_process(input_batch)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")
        noise.to("cuda")
        target = target.to("cuda")

    # loss function to train noise
    celoss = nn.CrossEntropyLoss()
    # optimizer to train noise
    optimizer = optim.SGD(
        noise.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    train_noise(
        optimizer,
        celoss,
        model,
        noise,
        target,
        input_batch,
        thres=0.02,
        thres_count=10,
        n=1000,
    )

    # we can change to top 5 by specifying top_k parameter to predict()
    top_1_class_no_noise, top_1_confidence_no_noise = predict(
        input_batch, model, noise, label_file=label_file, add_noise=False
    )

    top_1_class_noised, top_1_confidence_noised = predict(
        input_batch, model, noise, label_file=label_file, add_noise=True
    )

    noised_image = reverse_process(noise(input_batch))

    pure_noise = reverse_process(noise.noise)

    return (
        noised_image,
        original_image,
        pure_noise,
        top_1_class_no_noise,
        top_1_confidence_no_noise,
        top_1_class_noised,
        top_1_confidence_noised,
    )


if __name__ == "__main__":
    # example usage
    filename = "golden_retriever.jpg"
    input_image = Image.open(filename)
    results = add_noise(
        pil_image=input_image,
        model_name="alexnet",
        label_file="imagenet.txt",
        target=2,  # great white shark
    )
    results[0].save("noised_image.png", "PNG")
    results[1].save("original_image.png", "PNG")
    results[2].save("pure_noise.png", "PNG")
    print("top_1_class_no_noise", results[3])
    print("top_1_confidence_no_noise", results[4])
    print("top_1_class_noised", results[5])
    print("top_1_confidence_noised", results[6])
    # Example output
    # Three saved PNG image and stdout:
    """
    top_1_class_no_noise ['golden retriever']
    top_1_confidence_no_noise [0.9894764423370361]
    top_1_class_noised ['great white shark']
    top_1_confidence_noised [0.9657450318336487]
    """
