import ConvNext


model = ConvNext.load_convnext_base(6, pretrained=True)
print(model)
