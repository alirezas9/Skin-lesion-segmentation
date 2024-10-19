import cv2 as cv
import torch
import torchvision
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt 

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint", end= '')
    torch.save(state, filename)
    print('\rsaved succesfully\n')

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint", end= '')
    model.load_state_dict(checkpoint["state_dict"])
    print("\rloaded succesfully\n")

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = data(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = data(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            y = y.reshape(preds.shape)
            loss += criterion(preds, y)
    loss = loss.item()/len(loader)
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/(3*num_pixels)*100:.2f}")
    model.train()

    return loss


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds - preds.min()) / (preds.max() - preds.min()) # to scale preds between 0 and 1
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        y = y * 255.0 / torch.max(y)
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
        if idx == 3:
            break   # save some of images not all of them
    model.train()

def load_and_show_img(path):
    x = cv.imread(path)

    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(x, cv.COLOR_BGR2RGB))
    plt.axis('off')


   
def find_bbox(mask):

    LowerTH = 1
    UpperTH = 120
    Canny = cv.Canny(mask, LowerTH, UpperTH)

    contours,_ = cv.findContours(image = (Canny), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_KCOS)
    sortedList = sorted(contours, key=cv.contourArea, reverse=True) 
    # c1 = np.zeros_like(im)
    cv.drawContours(image=mask, contours=sortedList, contourIdx=-1, color=255, thickness=1, lineType=cv.LINE_AA)
    # cv.imshow("Contour", copy)
    for (i,c) in enumerate(contours):
        M = cv.moments(c)
        if M['m00'] > 15:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.drawContours(mask, [c], -1, (255,0,0), 1)

    return cv.boundingRect(sortedList[0])

def draw_bbox(img, mask):
    x,y,w,h = find_bbox(mask)
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    return img

# Run to test bbox
if __name__ == "__main__":
    num  = '0000018'
    img = cv.imread("ISBI2016_ISIC_Part1_Training_Data\\training\ISIC_{}.jpg".format(num))
    mask = cv.imread("ISBI2016_ISIC_Part1_Training_GroundTruth\gt_training\ISIC_{}_segmentation.png".format(num), 0)
    img = draw_bbox(img, mask)
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()