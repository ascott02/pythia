import web
import re
import config
import base64

# web.py
urls = (
    '/', 'index',
    '/api', 'api',
    '/login', 'login',
    '/upload', 'upload',
    '/meme', 'meme',
)

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd


import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


from pythia.utils.configuration import ConfigNode
from pythia.tasks.processors import VocabProcessor, CaptionProcessor
from pythia.models.butd import BUTD
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList

# for meme
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
# font_path="/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"
font_path="Impact.ttf"
from io import BytesIO 

model_yaml = "content/model_data/butd.yaml"
model_pth = 'content/model_data/butd.pth'
#
# model_pth = "../save/ycII_ft_bs_64_lr_002.copy/captioning_youcookII_butd/butd_final.pth"
# model_yaml = "../save/ycII_ft_bs_64_lr_002.copy/captioning_youcookII_butd/config.yaml"
# model_pth = "../save/ycII_ft_bs_64_lr_005.copy/captioning_youcookII_butd/butd_final.pth"
# model_yaml = "../save/ycII_ft_bs_64_lr_005.copy/captioning_youcookII_butd/config.yaml"
# model_pth = "../save/ycII_training_bs_64_lr_01_no_preload.copy/captioning_youcookII_butd/butd_final.pth"
# model_yaml = "../save/ycII_training_bs_64_lr_01_no_preload.copy/captioning_youcookII_butd/config.yaml"

# best.ckpt  
# butd_final.pth  
# config.yaml



class PythiaDemo:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  def __init__(self):
    self._init_processors()
    self.pythia_model = self._build_pythia_model()
    self.detection_model = self._build_detection_model()

  def _init_processors(self):
    with open(model_yaml) as f:
      config = yaml.load(f)

    config = ConfigNode(config)
    # Remove warning
    config.training_parameters.evalai_inference = True
    registry.register("config", config)

    self.config = config

    captioning_config = config.task_attributes.captioning.dataset_attributes.coco
    # captioning_config = config.task_attributes.captioning.dataset_attributes.youcookII
    text_processor_config = captioning_config.processors.text_processor
    caption_processor_config = captioning_config.processors.caption_processor
    # print("DEBUG captioning_config:", captioning_config)
    # print("DEBUG text_processor_config:", text_processor_config)
    # print("DEBUG caption_processor_config:", caption_processor_config)

    text_processor_config.params.vocab.vocab_file = "content/model_data/vocabulary_captioning_thresh5.txt"
    caption_processor_config.params.vocab.vocab_file = "content/model_data/vocabulary_captioning_thresh5.txt"
    self.text_processor = VocabProcessor(text_processor_config.params)
    self.caption_processor = CaptionProcessor(caption_processor_config.params)
    # print("DEBUG text_processor:", self.text_processor)
    # print("DEBUG caption_processor:", self.caption_processor)

    registry.register("coco_text_processor", self.text_processor)
    registry.register("coco_caption_processor", self.caption_processor)
    # registry.register("youcookII_text_processor", self.text_processor)
    # registry.register("youcookII_caption_processor", self.caption_processor)

  def _build_pythia_model(self):
    state_dict = torch.load(model_pth)
    model_config = self.config.model_attributes.butd
    model_config.model_data_dir = "content/"
    model = BUTD(model_config)
    model.build()
    model.init_losses_and_metrics()

    if list(state_dict.keys())[0].startswith('module') and \
       not hasattr(model, 'module'):
      state_dict = self._multi_gpu_state_to_single(state_dict)

    model.load_state_dict(state_dict)
    # print("DEBUG Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print("DEBUG:", param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.to("cuda")
    model.eval()

    return model

  def _multi_gpu_state_to_single(self, state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd

  def predict(self, url):
    with torch.no_grad():
      detectron_features = self.get_detectron_features(url)

      sample = Sample()
      # sample.dataset_name = "coco"
      sample.dataset_name = "youcookII"
      sample.dataset_type = "test"
      sample.image_feature_0 = detectron_features
      sample.answers = torch.zeros((5, 10), dtype=torch.long)
      # print("DEBUG self.pythia_model:", self.pythia_model)

      sample_list = SampleList([sample])
      # print("DEBUG sample_list:", sample_list)
      sample_list = sample_list.to("cuda")
      # print("DEBUG self.pythia_model(sample_list):", self.pythia_model(sample_list))

      tokens = self.pythia_model(sample_list)["captions"]

    gc.collect()
    torch.cuda.empty_cache()

    return tokens


  def _build_detection_model(self):

      cfg.merge_from_file('content/model_data/detectron_model.yaml')
      cfg.freeze()

      model = build_detection_model(cfg)
      checkpoint = torch.load('content/model_data/detectron_model.pth',
                              map_location=torch.device("cpu"))

      load_state_dict(model, checkpoint.pop("model"))

      model.to("cuda")
      model.eval()
      return model

  def get_actual_image(self, image_path):
      if type(image_path) == str():
          if image_path.startswith('http'):
              # path = requests.get(image_path, stream=True).raw
              path = requests.get(image_path, stream=True)
          else:
              path = image_path
      else:
          path = image_path

      return path

  def _image_transform(self, image_path):
      path = self.get_actual_image(image_path)

      try:
          print("DEBUG in the try")
          img = Image.open(path)
      except:
          print("DEBUG in the exception")
          # stream = BytesIO(path.content)
          stream = BytesIO(path)
          img = Image.open(stream)

      im = np.array(img).astype(np.float32)
      im = im[:, :, ::-1]
      im -= np.array([102.9801, 115.9465, 122.7717])
      im_shape = im.shape
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
      im_scale = float(800) / float(im_size_min)
      # Prevent the biggest axis from being more than max_size
      if np.round(im_scale * im_size_max) > 1333:
           im_scale = float(1333) / float(im_size_max)
      im = cv2.resize(
           im,
           None,
           None,
           fx=im_scale,
           fy=im_scale,
           interpolation=cv2.INTER_LINEAR
       )
      img = torch.from_numpy(im).permute(2, 0, 1)
      return img, im_scale


  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list

  def masked_unk_softmax(self, x, dim, mask_idx):
      x1 = F.softmax(x, dim=dim)
      x1[:, mask_idx] = 0
      x1_sum = torch.sum(x1, dim=1, keepdim=True)
      y = x1 / x1_sum
      return y

  def get_detectron_features(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales,
                                                  'fc6', 0.2)
      return feat_list[0]

demo = PythiaDemo()

def init_widgets(url):
  image_text = widgets.Text(
    description="Image URL", layout=Layout(minwidth="70%")
  )

  image_text.value = url
  submit_button = widgets.Button(description="Caption the image!")

  display(image_text)
  display(submit_button)

  submit_button.on_click(lambda b: on_button_click(
      b, image_text
  ))

  return image_text

def on_button_click(b, image_text):
  clear_output()
  image_path = demo.get_actual_image(image_text.value)
  image = Image.open(image_path)

  tokens = demo.predict(image_text.value)
  answer = demo.caption_processor(tokens.tolist()[0])["caption"]
  init_widgets(image_text.value)
  display(image)

  display(HTML(answer))

def make_meme(filename, topString, bottomString):
    print("DEBUG filename:", filename)
    print("DEBUG topString:", topString)
    print("DEBUG bottomString:", bottomString)

    img = Image.open(filename)
    imageSize = img.size

    # find biggest font size that works
    fontSize = int(imageSize[1]/5)
    font = ImageFont.truetype(font_path, fontSize)
    topTextSize = font.getsize(topString)
    bottomTextSize = font.getsize(bottomString)
    while topTextSize[0] > imageSize[0]-20 or bottomTextSize[0] > imageSize[0]-20:
        fontSize = fontSize - 1
        font = ImageFont.truetype(font_path, fontSize)
        topTextSize = font.getsize(topString)
        bottomTextSize = font.getsize(bottomString)

    # find top centered position for top text
    topTextPositionX = (imageSize[0]/2) - (topTextSize[0]/2)
    topTextPositionY = 0
    topTextPosition = (topTextPositionX, topTextPositionY)

    # find bottom centered position for bottom text
    bottomTextPositionX = (imageSize[0]/2) - (bottomTextSize[0]/2)
    bottomTextPositionY = imageSize[1] - bottomTextSize[1]
    bottomTextPosition = (bottomTextPositionX, bottomTextPositionY)

    draw = ImageDraw.Draw(img)

    # draw outlines
    # there may be a better way
    outlineRange = int(fontSize/15)
    for x in range(-outlineRange, outlineRange+1):
        for y in range(-outlineRange, outlineRange+1):
            draw.text((topTextPosition[0]+x, topTextPosition[1]+y), topString, (0,0,0), font=font)
            draw.text((bottomTextPosition[0]+x, bottomTextPosition[1]+y), bottomString, (0,0,0), font=font)

    draw.text(topTextPosition, topString, (255,255,255), font=font)
    draw.text(bottomTextPosition, bottomString, (255,255,255), font=font)

    # img.save("temp.png")
    return img

def get_upper(somedata):
	'''
	Handle Python 2/3 differences in argv encoding
	'''
	result = ''
	try:
		result = somedata.decode("utf-8").upper()
	except:
		result = somedata.upper()
	return result

def get_lower(somedata):
	'''
	Handle Python 2/3 differences in argv encoding
	'''
	result = ''
	try:
		result = somedata.decode("utf-8").lower()
	except:
		result = somedata.lower()		

	return result

# image_text = init_widgets(
#     "http://images.cocodataset.org/train2017/000000505539.jpg"
# )


class index:

    def GET(self, *args):

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return """<html><head></head><body>
This form takes an image URL and returns a Pythia caption.<br/><br/>
<form method="POST" action="">
URL: <input type="input" name="image_url" /><br/><br/>
<input type="submit" />
</form>
</body></html>"""
        else:
            raise web.seeother('/login')

    def POST(self, *args):
        x = web.input()
        # demo = PythiaDemo()
        image_path = demo.get_actual_image(str(x['image_url']))

        tokens = demo.predict(str(x['image_url']))
        answer = demo.caption_processor(tokens.tolist()[0])["caption"]
    
        img_tag = '<img src="' + str(x['image_url']) + '">'

        page = """<html><head></head><body>
This form takes an image URL and returns a Pythia caption.<br/><br/>
<form method="POST" action="">
URL: <input type="input" name="image_url" /><br/><br/>
<input type="submit" />
</form>""" + img_tag + """<br/>Caption: """ + answer + """<br/>
</body></html>"""

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return page
        else:
            raise web.seeother('/login')

class meme:

    def GET(self, *args):

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return """<html><head></head><body>
This form takes an image URL and returns a Pythia captioned meme.<br/><br/>
<form method="POST" action="">
URL: <input type="input" name="image_url" /><br/><br/>
<input type="submit" />
</form>
</body></html>"""
        else:
            raise web.seeother('/login')

    def POST(self, *args):
        x=web.input()
        image_path = demo.get_actual_image(str(x['image_url']))

        tokens = demo.predict(str(x['image_url']))
        answer = demo.caption_processor(tokens.tolist()[0])["caption"]
        answer_list = answer.split(" ")

        topString = ' '.join(answer_list[:len(answer_list)//2])
        bottomString = ' '.join(answer_list[len(answer_list)//2:])

        image = make_meme(image_path, topString, bottomString)

        output = BytesIO()
        # im = Image.open("test.png") # Your image here!
        image.save(output, format='JPEG')
        # output.seek(0)
        # output_s = output.read()
        output_s = output.getvalue()
        b64 = base64.b64encode(output_s)

        img_tag = '<img src="data:image/jpeg;base64,{0}">'.format(b64.decode())

        page = """<html><head></head><body>
This form takes an image URL and returns a Pythia captioned meme.<br/><br/>
<form method="POST" action="">
URL: <input type="input" name="image_url" /><br/><br/>
<input type="submit" />
</form>""" + img_tag + """<br/>Caption: """ + answer + """<br/>
</body></html>"""

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return page
        else:
            raise web.seeother('/login')


class api:

    def POST(self, *args):
        x = web.input()
        web.debug(x)
        web.debug(x['image_url'])      # This is the filename
        web.debug(x['token'])      # This is the filename

        if not x['image_url']:
            return "No URL."

        if not x['token']:
            return "No token."

        if not str(x['token']) in config.tokens:
            return "Not in tokens."
    
        # demo = PythiaDemo()        
        image_path = demo.get_actual_image(str(x['image_url']))

        tokens = demo.predict(str(x['image_url']))
        answer = demo.caption_processor(tokens.tolist()[0])["caption"]
    
        return answer

class upload:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['img_file'].filename)      # This is the filename
        # web.debug(x['img_file'].value)       # This is the file contents
        # web.debug(x['img_file'].file.read()) # Or use a file(-like) object
        # data = web.data()

        if not x['img_file'].filename:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'].decode('utf-8') in config.tokens:
            return "Not in tokens."
    
        # caption = _get_score(x['img_file'].value, str(x['caption']))
        tokens = demo.predict(x['img_file'].value)
        caption = demo.caption_processor(tokens.tolist()[0])["caption"]

        return caption



class login:

    def GET(self):
        auth = web.ctx.env.get('HTTP_AUTHORIZATION')
        authreq = False
        if auth is None:
            authreq = True
        else:
            auth = re.sub('^Basic ','',auth)
            username,password = base64.b64decode(auth).decode().split(':')
            if (username,password) in config.allowed:
                raise web.seeother('/')
            else:
                authreq = True
        if authreq:
            web.header('WWW-Authenticate','Basic realm="Auth example"')
            web.ctx.status = '401 Unauthorized'
            return


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()

