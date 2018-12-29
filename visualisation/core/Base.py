class Base:
    def __init__(self, module, device):
        self.module, self.device = module, device
        self.handles = []

    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, inputs, layer, *args, **kwargs):
        return inputs, {}