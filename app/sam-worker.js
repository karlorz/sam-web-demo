/**
 * Worker for sam-web-demo
 * Uses local SAM2 class with onnxruntime-web
 */
import { SAM2 } from "./SAM2";
import { Tensor } from "onnxruntime-web";
import { MODEL_CONFIG, DEFAULT_MODEL } from "./modelConfig.js";

let sam = null;

const stats = {
  modelId: null,
  device: "unknown",
  downloadModelsTime: [],
  encodeImageTimes: [],
  decodeTimes: [],
};

// Convert CHW [C, H, W] to HWC [H, W, C] for MobileSAM
function chwToHwc(chwArray, channels, height, width) {
  const hwcArray = new Float32Array(chwArray.length);
  const channelSize = height * width;

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < channels; c++) {
        const chwIdx = c * channelSize + h * width + w;
        const hwcIdx = h * width * channels + w * channels + c;
        hwcArray[hwcIdx] = chwArray[chwIdx];
      }
    }
  }

  return hwcArray;
}

// Apply ImageNet normalization: (pixel * scale - mean) / std
function applyNormalization(data, channels, height, width, normalization) {
  const { mean, std, scale } = normalization;
  const channelSize = height * width;
  const normalizedData = new Float32Array(data.length);

  for (let c = 0; c < channels; c++) {
    for (let i = 0; i < channelSize; i++) {
      const idx = c * channelSize + i;
      normalizedData[idx] = (data[idx] * scale - mean[c]) / std[c];
    }
  }

  return normalizedData;
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "ping") {
    const modelId = data?.modelId || DEFAULT_MODEL;
    const modelConfig = MODEL_CONFIG[modelId];

    sam = new SAM2(modelConfig);
    stats.modelId = modelId;

    self.postMessage({ type: "downloadInProgress" });
    const startTime = performance.now();
    await sam.downloadModels();
    const durationMs = performance.now() - startTime;
    stats.downloadModelsTime.push(durationMs);

    self.postMessage({ type: "loadingInProgress" });
    const report = await sam.createSessions();

    stats.device = report.device;

    self.postMessage({ type: "pong", data: report });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "encodeImage") {
    const { float32Array, shape } = data;

    let tensorData = float32Array;
    let tensorShape = shape;

    // Apply normalization if model requires it
    if (sam.modelConfig.normalization) {
      const [, channels, height, width] = shape;
      tensorData = applyNormalization(
        tensorData,
        channels,
        height,
        width,
        sam.modelConfig.normalization
      );
    }

    // Scale input range if needed (e.g., MobileSAM expects 0-255)
    if (sam.modelConfig.inputRange) {
      const scaledData = new Float32Array(tensorData.length);
      for (let i = 0; i < tensorData.length; i++) {
        scaledData[i] = tensorData[i] * sam.modelConfig.inputRange;
      }
      tensorData = scaledData;
    }

    if (sam.modelConfig.tensorFormat === "HWC") {
      const actualShape = sam.modelConfig.useBatchDimension
        ? shape
        : shape.slice(1);
      const [channels, height, width] = actualShape;

      tensorData = chwToHwc(tensorData, channels, height, width);
      tensorShape = [height, width, channels];
    } else {
      tensorShape = sam.modelConfig.useBatchDimension ? shape : shape.slice(1);
    }

    const imgTensor = new Tensor("float32", tensorData, tensorShape);

    const startTime = performance.now();
    await sam.encodeImage(imgTensor);
    const durationMs = performance.now() - startTime;
    stats.encodeImageTimes.push(durationMs);

    self.postMessage({
      type: "encodeImageDone",
      data: { durationMs },
    });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "decodeMask") {
    const { points, maskArray, maskShape } = data;

    const startTime = performance.now();

    let decodingResults;
    if (maskArray) {
      const maskTensor = new Tensor("float32", maskArray, maskShape);
      decodingResults = await sam.decode(points, maskTensor);
    } else {
      decodingResults = await sam.decode(points);
    }

    const durationMs = performance.now() - startTime;
    stats.decodeTimes.push(durationMs);

    self.postMessage({ type: "decodeMaskResult", data: decodingResults });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "stats") {
    self.postMessage({ type: "stats", data: stats });
  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
};
