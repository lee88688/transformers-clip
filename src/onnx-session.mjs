import {InferenceSession} from 'onnxruntime-node'
import {AutoProcessor, AutoTokenizer, env, RawImage} from "@xenova/transformers";
import path from "path";
import {parentPort, isMainThread, workerData, Worker} from 'worker_threads'

// noinspection JSConstantReassignment
env.localModelPath = path.join(import.meta.dirname, "../models");
env.allowRemoteModels = false;
env.remotePathTemplate = "{model}/";
const modelName = "chinese-clip-vit-base-patch16";

export async function textEncoder() {
    const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
        quantized: false,
    });
    const token = tokenizer("皮卡丘", {padding: true, max_length: 52});

    const session = await InferenceSession.create('./models/chinese-clip-vit-base-patch16/onnx/txt.quant.onnx')

    const res = await session.run({text: token.input_ids})

    /**
     * @type {Float32Array}
     */
    const float32Array = res.unnorm_text_features.data;
    // await saveToText(float32Array, "text-f32.txt");

    return float32Array;
}

async function imageEncoder() {
    const rawImage = await RawImage.fromURL(path.resolve("src/pokemon.jpeg"));
    const processor = await AutoProcessor.from_pretrained(modelName, {
        quantized: false,
    });
    const image = await processor(rawImage);

    const session = await InferenceSession.create('./models/chinese-clip-vit-base-patch16/onnx/img.quant.onnx')

    const res = await session.run({image: image.pixel_values})

    const float32Array = res.unnorm_image_features.data;
    // await saveToText(float32Array, "image-f32.txt");

    return float32Array;
}

/**
 *
 * @param {Float32Array} floatArr
 * @return {number}
 */
function norm(floatArr) {
    let sum = 0;
    for (let i = 0; i < floatArr.length; i++) {
        sum += floatArr[i] * floatArr[i];
    }

    return Math.sqrt(sum);
}

if (isMainThread) {
    console.time('calculate')
    const [textVector, imageVector] = await Promise.all([
        new Promise(resolve => {
            const worker = new Worker(import.meta.filename, { workerData: {type: 'text'}})
            worker.on('message', e => {
                resolve(new Float32Array(e.data))
            })
        }),
        new Promise(resolve => {
            const worker = new Worker(import.meta.filename, {workerData: {type: 'image'}})
            worker.on('message', e => {
                resolve(new Float32Array(e.data))
            })
        })
    ])
    let sum = 0;
    for (let i = 0; i < textVector.length; i++) {
        sum += textVector[i] * imageVector[i];
    }
    console.log('sum', sum)
    console.timeEnd('calculate')
} else {
    if (workerData.type === 'text') {
        const textVector = await textEncoder();
        const textNorm = norm(textVector);
        for (let i = 0; i < textVector.length; i++) {
            textVector[i] = textVector[i] / textNorm
        }
        parentPort.postMessage({data: textVector.buffer}, [textVector.buffer])
    } else if (workerData.type === 'image') {
        const imageVector = await imageEncoder();
        const imageNorm = norm(imageVector);
        for (let i = 0; i < imageVector.length; i++) {
            imageVector[i] = imageVector[i] / imageNorm
        }
        parentPort.postMessage({data: imageVector.buffer}, [imageVector.buffer])
    }
}

// const textVector = await textEncoder();
// const imageVector = await imageEncoder();
//
// const textNorm = norm(textVector);
// const imageNorm = norm(imageVector);
// let sum = 0;
// for (let i = 0; i < textVector.length; i++) {
//   sum += (textVector[i] / textNorm) * (imageVector[i] / imageNorm);
// }
//
// console.log("vector product is: ", sum);
