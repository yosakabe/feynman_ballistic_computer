# Feynman's Ballistic Computer — interactive 5-PCS demo

Feynman の ballistic computer / Feynman clock をインタラクティブに可視化する Python + Dash デモ (ver3) です。


## v3 の変更点

- ページ冒頭にあったモデル説明文を削除し、ページ下部の折りたたみ **Model Note** に統合。
- PCS probability の可視化を上段に単独配置。
- 下段では **Bloch sphere** と **Answer bit conditioned on the observed PCS site** の表を横並びに配置。画面幅が狭い場合は自動的に折り返します。
- v2で導入した5 PCS、Periodic PCS measurement、Pause時のtimer停止、Dash `Patch` による部分更新はそのまま維持。

## v2 から継続している主な機能

- Program Counter Site (PCS) を **3 → 5** に拡張。
- 元の `√NOT × 2 = NOT` はそのまま保持し、PCS 2→3→4 は `Identity` を作用させる padding clock site とした。
- 5-site chain により、未観測の PCS 確率は v1 の3-site chainほど単純な完全周期往復に見えにくい。
- **Unobserved unitary evolution** と **Periodic PCS measurement** を切り替え可能。
- Periodic mode では指定した simulation-time 間隔ごとに PCS を射影測定し、観測されたカーソルの履歴を `0 → 1 → 0 → ...` のように表示。
- `Measure PCS now` による手動観測も維持。
- Pause 時には `dcc.Interval` 自体を停止し、不要な callback を発生させない。
- 描画周期を 120 ms に緩和。物理時間は wall time から計算するため、これはシミュレーション精度ではなく UI 負荷だけを下げる変更。
- Dash `Patch` による partial property update を採用し、Bloch sphere の3D surfaceやPCSの静的annotationを毎フレーム再送・再描画しない。

## モデル

v3 の物理モデルは v2 と同じく

```text
PCS 0 --√NOT-- PCS 1 --√NOT-- PCS 2 --I-- PCS 3 --I-- PCS 4
```

です。Answer bit を `|0>` から始めると、条件付き状態は理想的には

```text
PCS 0 : |0>
PCS 1 : √NOT |0>
PCS 2 : NOT |0> = |1>
PCS 3 : |1>   (padding)
PCS 4 : |1>   (padding)
```

となります。

全体は PCS 5 qubit + Answer 1 qubit = **6 qubit / 64-dimensional Hilbert space** として明示的に計算します。

Hamiltonian は

```text
H = J Σ_i [ q†_(i+1) q_i A_(i+1) + q†_i q_(i+1) A†_(i+1) ]
```

です。前進項と Hermitian conjugate の後退項を両方含むため、PCS カーソルは一方向には進みません。

## セットアップ（uv）

既に `uv` が使える環境なら、プロジェクトルートで次を実行します。

```bash
uv sync
```

## 動作確認

```bash
uv run python src/selfcheck.py
```

正常なら次のように表示されます。

```text
Self-check passed (v3).
Hilbert dimension: 64
```

## アプリ起動

```bash
uv run python src/app.py --config configs/rnot_5pcs.yaml
```

ブラウザで通常は次を開きます。

```text
http://127.0.0.1:8050
```

出力先を変更する場合:

```bash
uv run python src/app.py \
  --config configs/rnot_5pcs.yaml \
  --output-dir /path/to/output
```

起動ごとに timestamp 付きサブディレクトリを作成します。

```text
output/
└── feynman_ballistic_YYYYMMDD_HHMMSS/
    ├── config_resolved.yaml
    ├── derived_parameters.yaml
    ├── events.jsonl
    ├── diagnostics.csv
    ├── pcs_probability_evolution.html
    ├── answer_bloch_evolution.html
    └── conditional_answer_evolution.html
```

## Evolution mode

### Unobserved unitary evolution

射影測定を行わず、

```text
|ψ(t)> = exp(-iHt/ℏ)|ψ(0)>
```

をそのまま表示します。棒グラフは「実際にカーソルがそこにいた履歴」ではなく、各PCSで観測される確率です。

### Periodic PCS measurement

`Measurement interval` ごとに PCS を射影測定します。観測のたびに状態がそのPCS部分空間へ収縮し、その収縮状態を新しいanchorとしてunitary evolutionを再開します。

このため観測履歴は例えば

```text
0 → 1 → 2 → 1 → 2 → 3 → ...
```

のようになり、途中で戻るカーソル運動を直接見ることができます。

## UI応答性について

v1 は Pause 中も `dcc.Interval` が発火し、clock storeとPlotly figureを更新し続けていました。v2では Pause 時に `disabled=True` としてtimerそのものを止めます。

また、各animation tickではFigure全体を再生成せず、Dash `Patch` で次だけを更新します。

- PCS probability bar
- PCS marker size
- 最後に観測されたPCSのstar marker
- Bloch vector
- figure title

Bloch sphere surfaceは初回描画後は固定です。

## ファイル構成

```text
feynman_ballistic_demo_v3/
├── configs/
│   └── rnot_5pcs.yaml
├── src/
│   ├── app.py
│   ├── diagnostics.py
│   ├── hamiltonian.py
│   ├── io_utils.py
│   ├── plot_diagnostics.py
│   ├── selfcheck.py
│   ├── simulation.py
│   └── visualization.py
├── output/
│   └── .gitkeep
├── .gitignore
├── pyproject.toml
└── README.md
```
