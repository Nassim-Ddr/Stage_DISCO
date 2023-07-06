import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
#imgs = ['https://manga-universe.fr/cdn/shop/articles/les-morts-de-goku-dragon-ball_1400x.progressive.jpg?v=1681138754']


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