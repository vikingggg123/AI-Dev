import torch
from torchinfo import summary
from PIL import Image
import base64
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
load_model = torch.load("model/Lebin.pth")

class_names = ["plastic_bag","plastic bag","plasticbottle","plasticbottle","wrapping"]

# print(load_model)

image_path = "LEBINDSV2/test/snack_wrapping/Screenshot 2025-01-04 185514.jpg"
image = Image.open(image_path)

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()

load_model.eval()
with torch.inference_mode():
    transform_image = auto_transform(image)
    prediction_logits = load_model(transform_image.unsqueeze(dim=0))
prediction_pred_prob = torch.softmax(prediction_logits,dim=1)
prediction_label = torch.argmax(prediction_pred_prob, dim=1)

plt.figure()
plt.imshow(image)
plt.title(f"Pred: {class_names[prediction_label]} | Prob: {prediction_pred_prob.max():.3f}")
plt.axis(False)
plt.show()