---
library_name: transformers.js
---

https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16 with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@xenova/transformers) using:
```bash
npm i @xenova/transformers
```

**Example:** Zero-shot image classification w/ `Xenova/chinese-clip-vit-base-patch16`.

```javascript
import { pipeline } from '@xenova/transformers';

// Create zero-shot image classification pipeline
const classifier = await pipeline('zero-shot-image-classification', 'Xenova/chinese-clip-vit-base-patch16');

// Set image url and candidate labels
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/pikachu.png';
const candidate_labels = ['杰尼龟', '妙蛙种子', '小火龙', '皮卡丘'] // Squirtle, Bulbasaur, Charmander, Pikachu in Chinese

// Classify image
const output = await classifier(url, candidate_labels);
console.log(output);
// [
//   { score: 0.9926728010177612, label: '皮卡丘' },        // Pikachu
//   { score: 0.003480620216578245, label: '妙蛙种子' },    // Bulbasaur
//   { score: 0.001942147733643651, label: '杰尼龟' },      // Squirtle
//   { score: 0.0019044597866013646, label: '小火龙' }      // Charmander
// ]
```


![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b253b7ac5ecaae3d1efe0c/bVOErVl5Zsz1dpstDfKpu.png)

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).