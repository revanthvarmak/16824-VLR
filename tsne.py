import torch
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
from utils import get_data_loader
import matplotlib.pyplot as plt
from matplotlib import patches
from train_q2 import ResNet
import numpy as np

checkpoint_path = "/home/mmpug/revanth/hw1/q1_q2_classification/checkpoint-model-epoch50.pth"
num_classes = 20
model = ResNet(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
model.eval()  


test_loader = get_data_loader(
    name='voc',
    train=False,
    batch_size=100,
    split='test',
)

feature_layer = {'resnet': 'avgpool'}

feature_extractor = create_feature_extractor(model, return_nodes=feature_layer)


extracted_features = []
extracted_labels = []
processed_batches = 0
max_batches = 10 

for inputs, labels, _ in test_loader:
    with torch.no_grad():
        features = feature_extractor(inputs.to('cuda'))['avgpool'].flatten(start_dim=1)
        extracted_features.append(features.cpu().numpy())
        extracted_labels.append(labels.view(-1, num_classes).cpu().numpy())
    
    processed_batches += 1
    if processed_batches == max_batches:
        break 

all_features = np.vstack(extracted_features)
all_labels = np.vstack(extracted_labels).astype(int)

tsne_model = TSNE(random_state=42)
tsne_results = tsne_model.fit_transform(all_features)

class_color_palette = np.random.randint(0, 256, size=(num_classes, 3))

point_colors = []
for label in all_labels:
    active_classes = np.where(label == 1)[0]
    if len(active_classes) == 0:
        point_colors.append([0, 0, 0])
    else:
        mean_color = class_color_palette[active_classes].mean(axis=0)
        point_colors.append(mean_color / 255)  

plt.figure(figsize=(14, 12))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors, alpha=0.6, edgecolors='w', linewidth=0.5)

legend_elements = [
    patches.Patch(color=class_color_palette[i]/255, label=f"Class {i}") for i in range(num_classes)
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title("t-SNE Visualization of ResNet-18 Features")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig("tsne_feature_visualization.png")
plt.show()
