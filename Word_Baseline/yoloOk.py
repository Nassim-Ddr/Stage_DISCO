import torch
from time import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images

imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
#imgs = ['https://domf5oio6qrcr.cloudfront.net/medialibrary/7349/7ae782c0-24f9-4128-97a9-8b64432bce76.jpg']


# Inference
results = model(imgs)

# Results
results.print()
results.show()#save()  # or .show()


print("Les predictions ")
results.xyxy[0]  # img1 predictions (tensor)
#print(results.xyxy[0])
results.pandas().xyxy[0]  # img1 predictions (pandas)
print(results.pandas().xyxy[0])