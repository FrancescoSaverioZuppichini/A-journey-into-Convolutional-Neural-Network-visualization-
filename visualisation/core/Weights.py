from .Base import Base

class Weights(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def __call__(self, inputs, layer, *args, **kwargs):
        layer.register_forward_hook(self.hook)
        self.module(inputs)
        b, c, h, w = self.outputs.shape
        # reshape to make an array of images 1-Channel
        outputs = self.outputs.view(c, b, h, w)

        return outputs, {}
