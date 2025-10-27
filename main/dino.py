import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from vision_transformer import DinoVIT
from utils import Solarization, GaussianBlur, cosine_scheduler, clip_gradients
from tqdm import tqdm
import os
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import wandb


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, crop_size=32):
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=global_crops_scale, 
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=global_crops_scale, 
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=local_crops_scale, 
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        crops.extend([self.local_transform(image) for _ in range(self.local_crops_number)])
        return crops


class DINODataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        crops = self.transform(img)
        return crops

def get_dino_dataloader(batch_size=128, local_crops_number=8, num_workers=4):
    base_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    
    dino_aug = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=local_crops_number,
        crop_size=32
    )
    
    dino_dataset = DINODataset(base_dataset, dino_aug)
    
    dataloader = DataLoader(
        dino_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader

class MultiCropWrapper(nn.Module):
    def __init__(self, model):
        super(MultiCropWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        concatenated = torch.cat(x, dim=0)
        logits = self.model(concatenated)
        return logits


class DinoLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, 
                 teacher_temp, nepochs, warmup_teacher_temp_epochs, student_temp,
                 center_momentum):
        super(DinoLoss, self).__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for teacher_idx, t in enumerate(teacher_out):
            for student_idx, s in enumerate(student_out):
                if teacher_idx == student_idx:
                    continue
                loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)



@torch.no_grad()
def extract_features(model, dataloader, device):

    model.eval()
    features_list = []
    labels_list = []
    
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        features = model.model.backbone(images)
        
        features_list.append(features.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    return features, labels


def evaluate_knn(student, device, k=20, num_workers=4):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    train_features, train_labels = extract_features(student, train_loader, device)
    
    test_features, test_labels = extract_features(student, test_loader, device)
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    train_pred = knn.predict(train_features)
    test_pred = knn.predict(test_features)
    
    train_acc = accuracy_score(train_labels, train_pred) * 100
    test_acc = accuracy_score(test_labels, test_pred) * 100
    
    return train_acc, test_acc



def train_dino(args):
    
    wandb.init(
        project="dino-cifar10",
        name=f"dino-{args.embedding_dim}d-{args.depth}L-{args.out_dim}out",
        config=vars(args)  
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_loader = get_dino_dataloader(
        batch_size=args.batch_size,
        local_crops_number=args.local_crops_number,
        num_workers=args.num_workers
    )
    
    
    student_model = DinoVIT(
        args.img_size, args.patch_size, args.in_channels,
        args.embedding_dim, args.num_heads, args.depth,
        args.mlp_dim, args.hidden_dim, args.bottleneck_dim,
        args.dropout, args.out_dim
    ).to(device)
    
    teacher_model = DinoVIT(
        args.img_size, args.patch_size, args.in_channels,
        args.embedding_dim, args.num_heads, args.depth,
        args.mlp_dim, args.hidden_dim, args.bottleneck_dim,
        0.0, args.out_dim
    ).to(device)
    
    student = MultiCropWrapper(student_model)
    teacher = MultiCropWrapper(teacher_model)
    
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
    
    criterion = DinoLoss(
        out_dim=args.out_dim,
        ncrops=args.ncrops,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        nepochs=args.epochs,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0, 
        betas=(0.9, 0.999)
    )

    lr_schedule = cosine_scheduler(
        args.lr * (args.batch_size / 256.),
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(train_loader),
    )

    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs,
        len(train_loader)
    )
    
    
    wandb.watch(student, log="gradients", log_freq=100)
    
    student.train()
    teacher.train()
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, crops in enumerate(pbar):
            it = len(train_loader) * epoch + batch_idx
            
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]
            
            crops = [crop.to(device) for crop in crops]
            global_crops = crops[:2]
            
            student_output = student(crops)
            with torch.no_grad():
                teacher_output = teacher(global_crops)
            
            loss = criterion(student_output, teacher_output, epoch)
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.clip_grad > 0:
                clip_gradients(student, args.clip_grad)
            
            optimizer.step()
            
            with torch.no_grad():
                m = momentum_schedule[it]
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr_schedule[it]:.6f}'})
            
            if batch_idx % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr_schedule[it],
                    "train/wd": wd_schedule[it],
                    "train/momentum": momentum_schedule[it],
                    "train/teacher_temp": criterion.teacher_temp_schedule[epoch],
                    "epoch": epoch,
                    "iteration": it
                })
        
        avg_loss = epoch_loss / num_batches
        print(f"\nepoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/number": epoch + 1
        })
        
        if (epoch + 1) % args.eval_freq == 0:
            train_acc, test_acc = evaluate_knn(student, device, k=args.knn_k, num_workers=args.num_workers)
            student.train()
            
            wandb.log({
                "eval/knn_train_acc": train_acc,
                "eval/knn_test_acc": test_acc,
                "epoch": epoch + 1
            })
            
                
        
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'center': criterion.center,
            }
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            wandb.save(checkpoint_path)
    
    
    train_acc, test_acc = evaluate_knn(student, device, k=args.knn_k, num_workers=args.num_workers)
    
    wandb.log({
        "final/knn_train_acc": train_acc,
        "final/knn_test_acc": test_acc,
    })
    
    
    print(f"best test accuracy: {best_acc:.2f}%")
    
    wandb.finish()
    
    return student, teacher, test_acc


def parse_args():

    parser = argparse.ArgumentParser(description='DINO training on cifar-10')
    
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--mlp_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--out_dim', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ncrops', type=int, default=10)
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--teacher_temp', type=float, default=0.07)
    parser.add_argument('--student_temp', type=float, default=0.1)
    parser.add_argument('--center_momentum', type=float, default=0.9)
    parser.add_argument('--momentum_teacher', type=float, default=0.996)
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04)
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30)    
    parser.add_argument('--knn_k', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--wandb_project', type=str, default='dino-cifar10')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dino(args)
