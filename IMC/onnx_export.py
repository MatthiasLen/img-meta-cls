import torch 
from IMC.network04 import MRISequenceClassifier 
import os 

DEFAULT_LABEL_NAMES = {
    "label_SequenceType": [
        "T1", "T2", "DWI", "ADC", "SUB", "DIXON_F", 
        "DIXON_IN", "DIXON_OPP", "BOLUS", "OTHER", "na"
    ],
    "label_FatSat": ["yes", "no", "na"],
    "label_MRCP": ["yes", "no", "na"],
    "label_AcquisitionPlane": ["AX", "COR", "SAG", "ORTHO", "ROT", "na"],
    "label_ContrastPhase": ["pre", "art", "portven", "trans", "hepa", "na"],
    "label_Contrast": ["pre", "post", "na"],
    "label_Localizer": ["yes", "no", "na"],
}

num_classes_dict = {label_name: len(classes) for label_name, classes in DEFAULT_LABEL_NAMES.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRISequenceClassifier(metadata_input_dim=88, num_classes_dict=num_classes_dict)

ckpt_path = "/home/tuan.truong/codebase/IMC/logs/20251118_152627/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(model)

dummy_image_input = torch.randn(16, 3, 1, 224, 224).to(device)  # Example input shape
dummy_metadata_input = torch.randn(16, 88).to(device)  # Example metadata

onnx_program = torch.onnx.export(model, (dummy_image_input, dummy_metadata_input),  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=True)
onnx_program.save("/home/tuan.truong/codebase/IMC/logs/20251118_152627/net4_best.onnx")

