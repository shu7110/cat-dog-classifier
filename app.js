"use strict";

const byId = (id) => document.getElementById(id);
const loadButton = byId("load");
const statusText = byId("status");
const inputs = [byId("photo"), byId("camera")];
let session;
let config;
let busy = false;
const displayName = (name) => ({cat: "猫", dog: "犬"}[name] || name);

function setInputsDisabled(disabled) {
  inputs.forEach((input) => { input.disabled = disabled; });
}

loadButton.addEventListener("click", async () => {
  loadButton.disabled = true;
  statusText.textContent = "モデルを読み込んでいます。初回は少し時間がかかります…";
  try {
    if (!window.ort) throw new Error("実行ライブラリを取得できません。通信状態を確認し、ページを再読み込みしてください。");
    const response = await fetch("./models/model_config.json", {cache: "no-cache"});
    if (!response.ok) throw new Error("モデルが未配置です。ノートブックで書き出した models フォルダを配置してください。");
    config = await response.json();
    if (config.schema_version !== 1 || !Array.isArray(config.labels) ||
        !Number.isInteger(config.image_size) || config.image_size < 1 || config.image_size > 1024 ||
        config.mean?.length !== 3 || config.std?.length !== 3 ||
        !config.mean.every(Number.isFinite) || !config.std.every((v) => Number.isFinite(v) && v > 0)) {
      throw new Error("モデル設定の形式が正しくありません。ノートブックから再度書き出してください。");
    }
    // GitHub Pagesではcross-origin isolationヘッダーを設定できないため1スレッド。
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = true;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    session = await ort.InferenceSession.create("./models/cats_dogs_int8.onnx", {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
    if (session.inputNames[0] !== config.input_name || session.outputNames[0] !== config.output_name) {
      await session.release();
      session = undefined;
      throw new Error("モデルと設定ファイルが一致しません。同じ書き出し結果を配置してください。");
    }
    statusText.textContent = "準備できました。写真を選ぶか、カメラで撮ってください。";
    loadButton.hidden = true;
    setInputsDisabled(false);
  } catch (error) {
    console.error(error);
    statusText.textContent = `読み込みに失敗しました：${error.message}`;
    loadButton.disabled = false;
    loadButton.textContent = "もう一度読み込む";
  }
});

async function classify(file) {
  if (!file || !session || busy) return;
  busy = true;
  setInputsDisabled(true);
  byId("result").hidden = true;
  byId("preview").hidden = true;
  statusText.textContent = "画像を処理しています…";
  const url = URL.createObjectURL(file);
  let tensor;
  let outputs;
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const preview = byId("preview");
    preview.src = url;
    await preview.decode();
    preview.hidden = false;

    const size = config.image_size;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    if (!ctx) throw new Error("このブラウザでは画像を処理できません。");
    // 学習時のResize((224,224))と同じく、切り抜かずに正方形へ変換。
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, size, size);
    const pixels = ctx.getImageData(0, 0, size, size).data;
    const area = size * size;
    const data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
      for (let c = 0; c < 3; c++) {
        data[c * area + i] = (pixels[4 * i + c] / 255 - config.mean[c]) / config.std[c];
      }
    }
    tensor = new ort.Tensor("float32", data, [1, 3, size, size]);
    const start = performance.now();
    outputs = await session.run({[config.input_name]: tensor});
    const elapsed = performance.now() - start;
    const logits = Array.from(outputs[config.output_name].data);
    if (logits.length !== config.labels.length || !logits.every(Number.isFinite)) {
      throw new Error("予測結果の形式がモデル設定と一致しません。");
    }
    const max = Math.max(...logits);
    const exps = logits.map((value) => Math.exp(value - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probabilities = exps.map((value) => value / sum);
    const best = probabilities.indexOf(Math.max(...probabilities));
    byId("prediction").textContent = `${displayName(config.labels[best])} と予測しました`;
    byId("scores").replaceChildren();
    probabilities.forEach((value, index) => {
      const row = document.createElement("div");
      row.className = "score";
      const label = document.createElement("p");
      const name = document.createElement("span");
      name.textContent = displayName(config.labels[index]);
      const percentage = document.createElement("span");
      percentage.textContent = `${(value * 100).toFixed(1)}%`;
      label.append(name, percentage);
      const bar = document.createElement("progress");
      bar.max = 1;
      bar.value = value;
      bar.setAttribute("aria-label", name.textContent);
      row.append(label, bar);
      byId("scores").append(row);
    });
    byId("timing").textContent = `推論時間：${elapsed.toFixed(0)} ms（CPU）`;
    byId("result").hidden = false;
    statusText.textContent = "判定が完了しました。別の写真も試せます。";
  } catch (error) {
    console.error(error);
    statusText.textContent = `判定に失敗しました：画像の形式や端末の空きメモリを確認してください。${error.message}`;
  } finally {
    tensor?.dispose();
    if (outputs) Object.values(outputs).forEach((output) => output.dispose());
    URL.revokeObjectURL(url);
    inputs.forEach((input) => { input.value = ""; });
    setInputsDisabled(false);
    busy = false;
  }
}

inputs.forEach((input) => input.addEventListener("change", () => classify(input.files[0])));
