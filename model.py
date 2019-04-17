import json
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import runway
from runway.data_types import image, text

labels = json.load(open('labels.json'))

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

@runway.setup
def setup():
  return models.squeezenet1_1(pretrained=True)

@runway.command('classify', inputs={'photo': image}, outputs={'label': text})
def classify(model, inputs):
    img = inputs['photo']
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    fc_out = model(img_variable)
    label = labels[str(fc_out.data.numpy().argmax())]
    return {'label': label}

if __name__ == '__main__':
    runway.run()
