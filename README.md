# スマホで猫・犬を分類する

画像はアップロードせず、スマホ内のCPU（ONNX Runtime Web / WebAssembly）で推論します。
OpenVINO Runtimeはブラウザで実行できないため、Web用には同じ学習済みモデルをINT8 ONNXとして保存します。
OpenVINO用XML/BINは `model_exports/openvino` に別途保存します。

## モデルを作る

1. `notebooks/02_imageclassifier最終版.ipynb` で学習・評価まで実行します。
2. 追加した12〜13節を順番に実行します。
3. `cats_dogs_deploy.zip` を展開します。Colabでは自動ダウンロードします。
4. ZIP内の `github_pages/models` を、このフォルダの `models` に配置します。

必要なファイル：

```text
github_pages/
  index.html
  style.css
  app.js
  .nojekyll
  models/
    cats_dogs_int8.onnx
    model_config.json
```

モデルはまだ同梱していません。notebookの保存済み出力から学習済み重みは復元できません。
学習後に書き出してください。テスト画像での精度比較結果も追加セルで確認できます。
標準の比較は最大100枚のため、必要に応じて全件評価してください。

ローカルのuv環境では、書き出し用ライブラリをプロジェクト直下で追加します。
Colabではnotebook内の `%pip install` セルを使います。

```sh
uv pip install 'openvino>=2025.0,<2027' 'onnx>=1.17,<2' 'onnxruntime>=1.20,<2'
```

## GitHub Pagesで公開

このプロジェクトの構成で公開する場合：

1. `github_pages` と `.github/workflows/deploy-pages.yml` をGitHubにpushします。ONNXとJSONを含めてください。
2. リポジトリの **Settings → Pages → Source → GitHub Actions** を選びます。
3. **Actions → Deploy image classifier to GitHub Pages → Run workflow** を実行します。
4. 公開されたURLをスマホで開き、「モデルを読み込む」を押してから写真を選びます。

ワークフローは手動起動です。配置前は不足ファイルのチェックで停止します。
`model_exports` やZIPは公開対象に含めません。
ZIPにもワークフローを含めています。`.github` は隠しフォルダなので、コピー時に含めてください。

別の公開専用リポジトリを使う場合は、このフォルダの**中身**をリポジトリ直下へ置き、
Pagesの **Deploy from a branch → main / (root)** でも公開できます。
ブランチ公開の対象ディレクトリはrootまたはdocsで、github_pagesは選べません。

## ローカル確認

プロジェクト直下から：

```sh
python -m http.server 8000 --directory github_pages
```

`http://localhost:8000` を開きます。HTMLファイルの直接起動ではなくHTTPで開いてください。
同じWi-FiのスマホではPCのLAN内IPとポート8000にアクセスします。
カメラの選択肢や撮影動作はOS・ブラウザによって異なります。

## 実行について

- 初回はモデルとCDN上のONNX Runtime Web 1.22.0を取得するため通信が必要です。
- CPUのみを使用し、GPUは要求しません。GitHub Pagesで追加ヘッダーを使わず動くよう1スレッドに固定しています。
- 実行をWorkerに移して画面の停止を抑えます。速度は端末次第で、INT8化しても高速化は保証されません。
- 学習時と同じRGB・正方形リサイズ・ImageNet正規化を使用します。PillowとCanvasで補間や透過・色管理に差が出る場合があります。
- 猫・犬以外の判定や、確率の正確な校正は行っていません。

参考：[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)、
[スレッド数・Workerの設定](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)、
[INT8量子化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、
[OpenVINOのONNX変換](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-onnx.html)。
