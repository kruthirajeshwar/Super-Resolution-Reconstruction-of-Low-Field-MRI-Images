import torch
from torch import nn
from torchvision.models import mobilenet_v2

class Generator_Loss(nn.Module):
    
    def __init__(self):
        super(Generator_Loss, self).__init__()

        # Initialize the mobile net v2 model to extract features for perceptual loss
        vgg = mobilenet_v2(pretrained=True)

        # Get the feature extraction layers of the network and set the network to evaluation mode
        loss_network = nn.Sequential(*list(vgg.features)).eval()

        # Freeze the weights of the loss network
        for param in loss_network.parameters():
            param.requires_grad = False

        # Save the perceptual loss estimator network    
        self.loss_network = loss_network

        # Define the MSE loss object
        self.mse_loss = nn.MSELoss()

        # Define the TV loss object
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):

        # Calculate the Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Calculate the Perception/Content Loss (Eucleadian distance b/w generated & actual image feature maps)
        with torch.no_grad():
            perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        # Calculate Naive Image Loss (mse between the generated & actual images, (pixel diff mse))
        image_loss = self.mse_loss(out_images, target_images)
        
        # Calculate the TV Loss
        tv_loss = self.tv_loss(out_images)

        total_loss = image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

        return total_loss


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        
        super(TVLoss, self).__init__()

        # Initialize the TV loss weight
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        
        # Get the batch size, height and width of the input images tensor
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        # Calculate total number of values in the height and width dimensions of the input tensor
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        # Total variation along the height dimension of the input tenso
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # Total variation along the width dimension of the input tensor
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        # Calculate the total variation loss per sample
        tv_loss = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return tv_loss

    @staticmethod
    def tensor_size(t):
        
        return t.size()[1] * t.size()[2] * t.size()[3]