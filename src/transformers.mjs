import {
  pipeline,
  env,
  RawImage,
  PreTrainedModel,
  AutoTokenizer,
  AutoProcessor,
} from "@xenova/transformers";
import * as path from "path";
import * as fsPromises from "fs/promises";

// noinspection JSConstantReassignment
env.localModelPath = path.join(import.meta.dirname, "../models");
env.allowRemoteModels = false;
env.remotePathTemplate = "{model}/";

const modelName = "chinese-clip-vit-base-patch16";

async function saveToText(float32Array, fileName) {
  const strArray = [];
  for (const item of float32Array) {
    strArray.push(item.toFixed(8));
  }
  await fsPromises.writeFile(fileName, strArray.join(" ") + "\n");
}

async function main() {
  // Create zero-shot image classification pipeline
  const classifier = await pipeline(
    "zero-shot-image-classification",
    modelName,
    { quantized: false }
  );

  const text_inputs = classifier.tokenizer(["树木"], {
    padding: true,
    truncation: true,
  });

  const images = await Promise.all(
    [path.join(import.meta.dirname, "pokemon.jpeg")].map((x) =>
      RawImage.read(x)
    )
  );

  const { pixel_values } = await classifier.processor(images);

  const modelOutput = await classifier.model({ ...text_inputs, pixel_values });
  // output.text_embeds and output.image_embeds are embeddings
  saveToText(modelOutput.text_embeds.data, "text-model.txt");
  saveToText(modelOutput.image_embeds.data, "image-model.txt");

  // Set image url and candidate labels
  const candidate_labels = ["树木"];

  // Classify image
  const output1 = await classifier(
    [path.join(import.meta.dirname, "pokemon.jpeg")],
    candidate_labels
  );
  console.log(output1);
}

async function textEncoder() {
  const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
    quantized: false,
  });
  const token = tokenizer("皮卡丘", { padding: true, max_length: 52 });

  const model = await PreTrainedModel.from_pretrained(modelName, {
    quantized: false,
    model_file_name: "txt.fp32",
  });

  const res = await model({ text: token.input_ids });
  /**
   * @type {Float32Array}
   */
  const float32Array = res.unnorm_text_features.data;
  await saveToText(float32Array, "text-f32.txt");

  return float32Array;
}

async function imageEncoder() {
  const rawImage = await RawImage.fromURL(path.resolve("src/pokemon.jpeg"));
  const processor = await AutoProcessor.from_pretrained(modelName, {
    quantized: false,
  });
  const image = await processor(rawImage);

  const model = await PreTrainedModel.from_pretrained(modelName, {
    quantized: false,
    model_file_name: "img.fp32",
  });
  const res = await model({ image: image.pixel_values });

  const float32Array = res.unnorm_image_features.data;
  await saveToText(float32Array, "image-f32.txt");

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

// await main()

const textVector = await textEncoder();
const imageVector = await imageEncoder();

const textNorm = norm(textVector);
const imageNorm = norm(imageVector);
let sum = 0;
for (let i = 0; i < textVector.length; i++) {
  sum += (textVector[i] / textNorm) * (imageVector[i] / imageNorm);
}

console.log("vector product is: ", sum);
