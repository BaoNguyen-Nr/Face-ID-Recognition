import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

        self.samples = []  # [(path, label)]
        self.label2idx = {}  # {'person1': 0, ...}
        self.idx2label = {}

        for idx, person in enumerate(os.listdir(root_dir)):
            person_folder = os.path.join(root_dir, person)
            if os.path.isdir(person_folder):
                self.label2idx[person] = idx
                self.idx2label[idx] = person
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    self.samples.append((img_path, idx))

        # Build dict: {label: [list các ảnh]}
        self.label_to_images = {}
        for path, label in self.samples:
            self.label_to_images.setdefault(label, []).append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = random.choice(list(self.label_to_images.keys()))
        anchor_path = random.choice(self.label_to_images[label])
        positive_path = random.choice([p for p in self.label_to_images[label] if p != anchor_path])
        negative_label = random.choice([l for l in self.label_to_images if l != label])
        negative_path = random.choice(self.label_to_images[negative_label])

        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(positive_path).convert('RGB'))
        negative = self.transform(Image.open(negative_path).convert('RGB'))

        return anchor, positive, negative
