import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

def is_image_file(filename):
    
    ## Mark image valid if it is any of the allowed formats
    valid_img = any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    
    return valid_img


def calculate_valid_crop_size(crop_size, upscale_factor):
    
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform():

    return transforms.Compose([
        transforms.ToTensor(),
    ])


def train_lr_transform():

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

class TrainDataset(Dataset):

    def __init__(self, dataset_dir):

        super(TrainDataset, self).__init__()

        ## Initialize the required variables

        # List of image filenames
        self.image_filenames = dataset_dir

        # Create the transforms for the High Resolution Images
        self.hr_transform = train_hr_transform()

        # Create the transforms for the Low Resolution Images
        self.lr_transform = train_lr_transform()

    def __getitem__(self, index):

        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):

        return len(self.image_filenames)


class ValDataset(Dataset):
    
    def __init__(self, dataset_dir):
       
        super(ValDataset, self).__init__()

        ## Initialize the required variables

        # List of image filenames
        self.image_filenames = dataset_dir

    def __getitem__(self, index):
        

        ## Read the actual HR image
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        
        ## Create the LR image transformer by downsampling the HR image and applying bicubic interpolation
        lr_scale = transforms.Resize((256,256), interpolation=Image.BICUBIC)

        ## Create the LR Image from the original HR Image using the LR Image transformer
        lr_image = lr_scale(hr_image)

        return to_tensor(lr_image), to_tensor(hr_image)

    def __len__(self):
        return len(self.image_filenames)