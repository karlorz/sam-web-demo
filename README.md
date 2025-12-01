# sam-web-demo

Demo for [sam-web](https://www.npmjs.com/package/sam-web) - Client-side image segmentation using Meta's SAM2 and MobileSAM.

**Live Demo:** [karlorz.github.io/sam-web-demo](https://karlorz.github.io/sam-web-demo)

## Features

- Uses the [sam-web](https://www.npmjs.com/package/sam-web) npm package for click-to-segment
- Multiple model support:
  - **SAM2 Tiny** (default) - Accurate segmentation (151 MB)
  - **MobileSAM Tiny** - Faster inference (45 MB)
- WebGPU acceleration with CPU fallback
- Model caching with OPFS
- Image upload or load from URL
- Mask refinement with iterative clicks (left=include, right=exclude)
- Crop and download segmented regions

## Quick Start

```bash
git clone https://github.com/karlorz/sam-web-demo
cd sam-web-demo
npm install
npm run dev
```

Open http://localhost:3000

## Usage

1. Select a model from the dropdown
2. Upload an image or load from URL
3. Click "Encode image" to prepare the image
4. Click on the image to segment (left=foreground, right=background)
5. Continue clicking to refine the mask
6. Click "Crop" to download the segmented region

## Using sam-web in Your Project

```bash
npm install sam-web onnxruntime-web
```

```typescript
import { SAMClient } from 'sam-web';

const sam = new SAMClient({ model: 'mobilesam' });
await sam.initialize(new URL('sam-web/worker', import.meta.url));
await sam.setImage(imageElement);

const mask = await sam.segment({
  points: [{ x: 0.5, y: 0.5, label: 1 }]
});

console.log(mask.bitmap);  // ImageBitmap
console.log(mask.score);   // IoU confidence
```

See [sam-web documentation](https://github.com/karlorz/sam-web) for full API reference.

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebGPU | 113+ | Flag | - | 113+ |
| CPU Fallback | All | All | All | All |
| OPFS Cache | All | All | Partial | All |

## Acknowledgements

- [sam-web](https://github.com/karlorz/sam-web) - The npm package powering this demo
- [Meta's SAM2](https://ai.meta.com/blog/segment-anything-2/)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [onnxruntime-web](https://github.com/microsoft/onnxruntime)
- [shadcn/ui](https://ui.shadcn.com/)

## License

MIT
