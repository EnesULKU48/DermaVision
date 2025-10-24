from studio.custom_block import *
import numpy as np

class SingleFrameCapture(Block):
    op_code = 'SingleFrameCapture'
    title = 'Single Frame Capture'
    tooltip = 'Captures a single frame when trigger is True'

    def init(self):
        self.width = 280
        self.height = 120

        self.input_sockets = [
            SocketTypes.ImageAny('Image'),
            SocketTypes.Boolean('Trigger')
        ]
        self.output_sockets = [SocketTypes.ImageAny('Captured Image')]
        self.memory = None  # Saklanan görüntü

    def run(self):
        img = self.input['Image'].data
        trig = self.input['Trigger'].data

        # Sadece trigger True ise yeni görüntü al
        if trig is True and img is not None:
            self.memory = img.copy()
            self.logInfo("Yeni görüntü kaydedildi.")

        # Hafızada bir görüntü yoksa boş bir görüntü gönder
        if self.memory is not None:
            self.output['Captured Image'].data = self.memory
        else:
            self.output['Captured Image'].data = np.zeros((1,1,3), dtype=np.uint8)

add_block(SingleFrameCapture.op_code, SingleFrameCapture)
