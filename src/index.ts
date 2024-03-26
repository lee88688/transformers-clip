import {pipeline, env, RawImage} from '@xenova/transformers';
import * as path from 'path'

// noinspection JSConstantReassignment
env.localModelPath = path.join(__dirname, '../models');
env.allowRemoteModels = false;
env.remotePathTemplate = '{model}/';

(async function main() {
  // Create zero-shot image classification pipeline
  const classifier = await pipeline('zero-shot-image-classification', 'chinese-clip-vit-base-patch16');

  const text_inputs = classifier.tokenizer(['太阳，树木', '黄昏，大海', '黄昏的树木在水中'], {
    padding: true,
    truncation: true,
  })

  const images = await Promise.all([path.join(__dirname, 'sea.jpg')].map(x => RawImage.read(x)))

  const { pixel_values } = await classifier.processor(images)

  const output = await classifier.model({ ...text_inputs, pixel_values })
  // output.text_embeds and output.image_embeds are embeddings

  // Set image url and candidate labels
  const candidate_labels = ['太阳，树木', '黄昏，大海', '黄昏的树木在水中']

  // Classify image
  const output1 = await classifier([path.join(__dirname, 'sea.jpg'), path.join(__dirname, 'pikachu.png')], candidate_labels);
  console.log(output1);
})()
