from studio.custom_block import *
import torch
import torch.nn as nn
import cv2, os
import numpy as np
from transformers import AutoImageProcessor, AutoModel

class SkinTypeClassifier(Block):
    op_code = 'SkinTypeClassifier'
    title = 'SkinTypeClassifier'
    tooltip = 'Classifies skin type using DINOv2-based PyTorch model'

    def init(self):
        self.width = 350
        self.height = 260

        self.input_sockets = [
            SocketTypes.ImageAny('Input Image'),
            SocketTypes.Boolean('Trigger')
        ]
        self.output_sockets = [
            SocketTypes.String('Skin Type'),
            SocketTypes.Number('Confidence')
        ]

        self.param['model_path'] = TextInput(
            text='C:/Users/kullanici/Desktop/best_dino_sweaty.pth',
            place_holder='Model .pth dosyasının yolu'
        )

        self.labels = ['kuru', 'normal', 'yağlı']
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        image = self.input['Input Image'].data
        trigger = self.input['Trigger'].data

        if not trigger:
            return

        # MODELi sadece bir kere yükle
        if self.model is None or self.processor is None:
            model_path = self.param['model_path'].value

            if not os.path.isfile(model_path):
                self.logError("Model yolu geçersiz veya dosya yok.")
                self.output['Skin Type'].data = "Error"
                self.output['Confidence'].data = 0.0
                return

            try:
                self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                base = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)

                class DinoClassifier(nn.Module):
                    def __init__(self, base, n_classes):
                        super().__init__()
                        self.base = base
                        self.fc = nn.Sequential(
                            nn.Linear(768, 256), nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, n_classes)
                        )
                    def forward(self, x):
                        feats = self.base(x).last_hidden_state[:, 0]
                        return self.fc(feats)

                model = DinoClassifier(base, len(self.labels)).to(self.device)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                self.model = model
                self.logInfo("Model başarıyla yüklendi.")

            except Exception as e:
                self.logError(f"Model yükleme hatası: {str(e)}")
                self.output['Skin Type'].data = "Error"
                self.output['Confidence'].data = 0.0
                return

        if image is None:
            self.logError("Görüntü verisi eksik.")
            self.output['Skin Type'].data = "Error"
            self.output['Confidence'].data = 0.0
            return

        try:
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=img_rgb, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)

            with torch.no_grad():
                logits = self.model(pixel_values)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            label = self.labels[pred.item()]
            self.output['Skin Type'].data = label
            self.output['Confidence'].data = round(conf.item(), 3)
            self.logInfo(f"Tahmin: {label} ({conf.item():.2f})")

        except Exception as e:
            self.logError(f"Inference hatası: {str(e)}")
            self.output['Skin Type'].data = "Error"
            self.output['Confidence'].data = 0.0

add_block(SkinTypeClassifier.op_code, SkinTypeClassifier)
