import torch
# import torchvision.transforms as transforms
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='/workspace/fairouz/fairouz_conf/fairouz/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin')

# resize = [transforms.Resize((224, 224)), transforms.ToTensor()]
# transformation = transforms.Compose(resize)
# torch.hub.set_dir("/workspace/fairouz/fairouz_conf/fairouz/.hub/")
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitb14.to('cuda')

def load_image(image_file):
    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

api_keys = [
    "543c7086-c880-45de-8bce-6c9c906293bb"
]  

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post("/embed/", dependencies=[Depends(api_key_auth)])
async def embed_image(file: UploadFile):
    image = preprocess(load_image(file.file)).unsqueeze(0)
    # image = load_image(file.file)
    # image = transformation(image).to('cuda')
    # embedding = dinov2_vitb14.forward(torch.unsqueeze(image, 0)).numpy(force=True)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image, normalize=True).numpy(force=True)
    try:
        return json.dumps({"embedding": image_features.squeeze(0).tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e | image_features}")
    

