// Model configurations for SAM models
export const MODEL_CONFIG = {
  mobilesam_tiny: {
    id: "mobilesam_tiny",
    name: "Mobile SAM Tiny",
    description: "Mobile SAM Tiny (45 MB, TinyViT encoder)",
    encoderUrl: "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
    decoderUrl: "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx",
    imageSize: { w: 1024, h: 1024 },
    maskSize: { w: 256, h: 256 },
    modelType: "mobilesam",
    encoderInputName: "input_image",
    useBatchDimension: false,
    tensorFormat: "HWC", // Height, Width, Channels - [H, W, 3]
    // Acly ONNX export expects 0-255 range (includes normalization in graph)
    inputRange: 255, // Scale 0-1 back to 0-255
  },
  sam2_tiny: {
    id: "sam2_tiny",
    name: "Meta's SAM2 Tiny",
    description: "Meta's SAM2 Tiny (151 MB, Hiera encoder)",
    encoderUrl: "https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_encoder.with_runtime_opt.ort",
    decoderUrl: "https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_decoder_pr1.onnx",
    imageSize: { w: 1024, h: 1024 },
    maskSize: { w: 256, h: 256 },
    modelType: "sam2",
    encoderInputName: "image",
    useBatchDimension: true,
    tensorFormat: "CHW", // Channels, Height, Width - [1, 3, H, W]
  },
};

export const DEFAULT_MODEL = "sam2_tiny";
