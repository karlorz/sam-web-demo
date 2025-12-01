"use client";

import React, { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

// sam-web package - using utilities and configs
import {
  MODEL_CONFIGS,
  DEFAULT_MODEL_ID,
  maskImageCanvas,
  resizeAndPadBox,
  resizeCanvas,
  canvasToFloat32Array,
  float32ArrayToCanvas,
  sliceTensor,
  maskCanvasToFloat32Array,
} from "sam-web";

// UI
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import InputDialog from "@/components/ui/inputdialog";
import { Button } from "@/components/ui/button";
import {
  LoaderCircle,
  ImageUp,
  ImageDown,
  Github,
  Fan,
} from "lucide-react";

export default function SAMDemo() {
  // Model selection
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL_ID);
  const modelConfig = MODEL_CONFIGS[selectedModel];

  // Resize+pad all images based on selected model config
  const imageSize = modelConfig.imageSize;
  const maskSize = modelConfig.maskSize;

  // State
  const [device, setDevice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageEncoded, setImageEncoded] = useState(false);
  const [status, setStatus] = useState("");

  // Web worker, image and mask
  const samWorker = useRef(null);
  const [image, setImage] = useState(null); // canvas
  const [mask, setMask] = useState(null); // canvas
  const [prevMaskArray, setPrevMaskArray] = useState(null); // Float32Array
  const [imageURL, setImageURL] = useState(
    "https://upload.wikimedia.org/wikipedia/commons/3/38/Flamingos_Laguna_Colorada.jpg"
  );
  const canvasEl = useRef(null);
  const fileInputEl = useRef(null);
  const pointsRef = useRef([]);

  const [stats, setStats] = useState(null);

  // Input dialog for custom URLs
  const [inputDialogOpen, setInputDialogOpen] = useState(false);
  const inputDialogDefaultURL =
    "https://upload.wikimedia.org/wikipedia/commons/9/96/Pro_Air_Martin_404_N255S.jpg";

  // Start encoding image
  const encodeImageClick = async () => {
    samWorker.current.postMessage({
      type: "encodeImage",
      data: canvasToFloat32Array(resizeCanvas(image, imageSize)),
    });

    setLoading(true);
    setStatus("Encoding");
  };

  // Start decoding, prompt with mouse coords
  const imageClick = (event) => {
    if (!imageEncoded) return;

    event.preventDefault();

    const canvas = canvasEl.current;
    const rect = event.target.getBoundingClientRect();

    // Input image will be resized to imageSize -> normalize mouse pos
    const point = {
      x: ((event.clientX - rect.left) / canvas.width) * imageSize.w,
      y: ((event.clientY - rect.top) / canvas.height) * imageSize.h,
      label: event.button === 0 ? 1 : 0,
    };
    pointsRef.current.push(point);

    // Do we have a mask already? ie. a refinement click?
    if (prevMaskArray) {
      const maskShape = [1, 1, maskSize.h, maskSize.w];

      samWorker.current.postMessage({
        type: "decodeMask",
        data: {
          points: pointsRef.current,
          maskArray: prevMaskArray,
          maskShape: maskShape,
        },
      });
    } else {
      samWorker.current.postMessage({
        type: "decodeMask",
        data: {
          points: pointsRef.current,
          maskArray: null,
          maskShape: null,
        },
      });
    }

    setLoading(true);
    setStatus("Decoding");
  };

  // Decoding finished -> parse result and update mask
  const handleDecodingResults = (decodingResults) => {
    const maskTensors = decodingResults.masks;
    const [, , width, height] = maskTensors.dims;
    const maskScores = decodingResults.iou_predictions.data;

    // Find best mask by IoU score (manual iteration to avoid stack overflow)
    let bestMaskIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < maskScores.length; i++) {
      if (maskScores[i] > bestScore) {
        bestScore = maskScores[i];
        bestMaskIdx = i;
      }
    }

    const bestMaskArray = sliceTensor(
      new Float32Array(maskTensors.data),
      maskTensors.dims,
      bestMaskIdx
    );
    let bestMaskCanvas = float32ArrayToCanvas(bestMaskArray, width, height);

    // Resize to image dimensions for display
    bestMaskCanvas = resizeCanvas(bestMaskCanvas, imageSize);
    setMask(bestMaskCanvas);

    // Store mask for refinement - resize to maskSize for decoder input
    const refinementMaskCanvas = resizeCanvas(
      float32ArrayToCanvas(bestMaskArray, width, height),
      maskSize
    );
    const refinementMaskArray = maskCanvasToFloat32Array(refinementMaskCanvas);
    setPrevMaskArray(refinementMaskArray);
  };

  // Handle web worker messages
  const onWorkerMessage = (event) => {
    const { type, data } = event.data;

    if (type === "pong") {
      const { success, device: deviceName } = data;

      if (success) {
        setLoading(false);
        setDevice(deviceName);
        setStatus("Encode image");
      } else {
        setStatus("Error (check JS console)");
      }
    } else if (type === "downloadInProgress" || type === "loadingInProgress") {
      setLoading(true);
      setStatus("Loading model");
    } else if (type === "encodeImageDone") {
      setImageEncoded(true);
      setLoading(false);
      setStatus("Ready. Click on image");
    } else if (type === "decodeMaskResult") {
      handleDecodingResults(data);
      setLoading(false);
      setStatus("Ready. Click on image");
    } else if (type === "stats") {
      setStats(data);
    } else if (type === "error") {
      console.error("Worker error:", data.message);
      setStatus("Error: " + data.message);
      setLoading(false);
    }
  };

  // Crop image with mask
  const cropClick = () => {
    const link = document.createElement("a");
    link.href = maskImageCanvas(image, mask).toDataURL();
    link.download = "crop.png";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Reset all the image-based state
  const resetState = () => {
    pointsRef.current = [];
    setImage(null);
    setMask(null);
    setPrevMaskArray(null);
    setImageEncoded(false);
  };

  // New image: From File
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    resetState();

    const dataURL = window.URL.createObjectURL(file);
    setImageURL(dataURL);
    setStatus("Encode image");

    e.target.value = null;
  };

  // New image: From URL
  const handleUrl = (urlText) => {
    resetState();
    setStatus("Encode image");
    setImageURL(urlText);
  };

  function handleRequestStats() {
    samWorker.current.postMessage({ type: "stats" });
  }

  // Handle model selection change
  const handleModelChange = (event) => {
    const newModelId = event.target.value;
    setSelectedModel(newModelId);

    // Reset encoding state but keep the loaded image
    pointsRef.current = [];
    setMask(null);
    setPrevMaskArray(null);
    setImageEncoded(false);
    setStatus("Encode image");
  };

  // Load web worker - recreate when model changes
  useEffect(() => {
    // Terminate existing worker if present
    if (samWorker.current) {
      samWorker.current.terminate();
      samWorker.current = null;
    }

    // Create new worker with selected model
    samWorker.current = new Worker(
      new URL("./sam-worker.js", import.meta.url),
      { type: "module" }
    );
    samWorker.current.addEventListener("message", onWorkerMessage);
    samWorker.current.postMessage({
      type: "ping",
      data: { modelId: selectedModel },
    });

    setLoading(true);

    // Cleanup on unmount or model change
    return () => {
      if (samWorker.current) {
        samWorker.current.terminate();
        samWorker.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  // Load image, pad to square and store in offscreen canvas
  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = imageURL;
      img.onload = function () {
        const largestDim = Math.max(img.naturalWidth, img.naturalHeight);
        const box = resizeAndPadBox(
          { h: img.naturalHeight, w: img.naturalWidth },
          { h: largestDim, w: largestDim }
        );

        const canvas = document.createElement("canvas");
        canvas.width = largestDim;
        canvas.height = largestDim;

        canvas
          .getContext("2d")
          .drawImage(
            img,
            0,
            0,
            img.naturalWidth,
            img.naturalHeight,
            box.x,
            box.y,
            box.w,
            box.h
          );
        setImage(canvas);
      };
    }

    // Cleanup: revoke blob URL to prevent memory leaks
    return () => {
      if (imageURL && imageURL.startsWith("blob:")) {
        URL.revokeObjectURL(imageURL);
      }
    };
  }, [imageURL]);

  // Offscreen canvas changed, draw it
  useEffect(() => {
    if (image) {
      const canvas = canvasEl.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        0,
        0,
        canvas.width,
        canvas.height
      );
    }
  }, [image]);

  // Mask changed, draw original image and mask on top with some alpha
  useEffect(() => {
    if (!image) return;

    const canvas = canvasEl.current;
    const ctx = canvas.getContext("2d");

    // Always redraw the base image first
    ctx.drawImage(
      image,
      0,
      0,
      image.width,
      image.height,
      0,
      0,
      canvas.width,
      canvas.height
    );

    // If mask exists, overlay it
    if (mask) {
      ctx.globalAlpha = 0.7;
      ctx.drawImage(
        mask,
        0,
        0,
        mask.width,
        mask.height,
        0,
        0,
        canvas.width,
        canvas.height
      );
      ctx.globalAlpha = 1;
    }
  }, [mask, image]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <div className="absolute top-4 right-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              window.open("https://github.com/karlorz/sam-web-demo", "_blank")
            }
          >
            <Github className="w-4 h-4 mr-2" />
            View on GitHub
          </Button>
        </div>
        <CardHeader>
          <CardTitle>
            <div className="flex flex-col gap-2">
              <p>
                Client-side Image Segmentation with{" "}
                <a
                  href="https://www.npmjs.com/package/sam-web"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  sam-web
                </a>
              </p>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <label htmlFor="model-select" className="text-sm font-normal">
                    Model:
                  </label>
                  <select
                    id="model-select"
                    value={selectedModel}
                    onChange={handleModelChange}
                    disabled={loading}
                    className="px-3 py-1 text-sm border rounded-md bg-white disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {Object.values(MODEL_CONFIGS).map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>
                <p
                  className={cn(
                    "flex gap-1 items-center",
                    device ? "visible" : "invisible"
                  )}
                >
                  <Fan
                    color="#000"
                    className="w-6 h-6 animate-[spin_2.5s_linear_infinite] direction-reverse"
                  />
                  Running on {device}
                </p>
              </div>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div className="flex justify-between gap-4">
              <Button
                onClick={encodeImageClick}
                disabled={loading || imageEncoded}
              >
                <p className="flex items-center gap-2">
                  {loading && <LoaderCircle className="animate-spin w-6 h-6" />}
                  {status}
                </p>
              </Button>
              <div className="flex gap-1">
                <Button
                  onClick={() => fileInputEl.current.click()}
                  variant="secondary"
                  disabled={loading}
                >
                  <ImageUp /> Upload
                </Button>
                <Button
                  onClick={() => setInputDialogOpen(true)}
                  variant="secondary"
                  disabled={loading}
                >
                  <ImageUp /> From URL
                </Button>
                <Button
                  onClick={cropClick}
                  disabled={mask == null}
                  variant="secondary"
                >
                  <ImageDown /> Crop
                </Button>
              </div>
            </div>
            <div className="flex justify-center">
              <canvas
                ref={canvasEl}
                width={512}
                height={512}
                onClick={imageClick}
                onContextMenu={(event) => {
                  event.preventDefault();
                  imageClick(event);
                }}
              />
            </div>
          </div>
        </CardContent>
        <div className="flex flex-col p-4 gap-2">
          <Button onClick={handleRequestStats} variant="secondary">
            Print stats
          </Button>
          <pre className="p-4 border-gray-600 bg-gray-100">
            {stats != null && JSON.stringify(stats, null, 2)}
          </pre>
        </div>
      </Card>
      <InputDialog
        open={inputDialogOpen}
        setOpen={setInputDialogOpen}
        submitCallback={handleUrl}
        defaultURL={inputDialogDefaultURL}
      />
      <input
        ref={fileInputEl}
        hidden="True"
        accept="image/*"
        type="file"
        onChange={handleFileUpload}
      />
    </div>
  );
}
