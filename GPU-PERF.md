# Measuring the Vulkan path

The GPU toggle exists so that transcription can be faster, not so that it exists. This file is how
we find out which it is on a given phone, and it is the only thing that decides the defaults in
`ModelManager.useGpu` and `FLASH_ATTN_WITH_GPU`.

## Why a single run tells you nothing

Two costs land in the first run and in no later one:

- ggml-vulkan compiles pipelines **on first use, per shape**. The encoder and decoder shapes are
  only known once audio arrives, so the first `whisper_full` after a context is created pays the
  shader compile that the build-time `glslc` step cannot pay for it.
- the backend itself is ~50 MB and has to be mapped and initialised.

Neither is the steady-state cost of the GPU, so a cold comparison measures the wrong thing.
Transcribe the same file **twice**; only the second run counts.

## The comparison

Same phone, same recording, same active model. Let the phone cool between the two halves.

```
adb logcat -c
adb logcat -s Scrib ggml JNI:V LibWhisper:I
```

With **Use GPU acceleration** off, transcribe the same file twice. With it on, transcribe the same
file twice. Save both logs.

What the lines mean:

| Line | Tag | Meaning |
|---|---|---|
| `Context created, use_gpu=… flash_attn=… (context ok\|NULL)` | Scrib | what actually ran, after the CPU retry |
| `whisper_backend_init_gpu: found GPU device …` / `no GPU found` | ggml | whether Vulkan was engaged at all |
| `whisper_full total time: N ms` | Scrib | the number to compare, run 2 |
| `whisper_print_timings` block | ggml | load / mel / encode / decode / batchd split |
| `Transcribing: gpu=… flash_attn=… threads=…` | LibWhisper | the configuration the run used |

## Deciding

| Warm run-2 evidence | Diagnosis | What to do |
|---|---|---|
| `no GPU found`, or the backend failed to load | Vulkan never ran | fix the backend load/packaging, then re-measure |
| context load much slower on GPU, run times equal | load cost only | ignore it; the context is already cached per model |
| run 1 ≫ run 2 on GPU, equal on CPU | lazy pipeline compile | judge warm runs only |
| `encode` faster, `decode` slower, net slower | the batch-1 decoder, as expected | the real fix is an encoder-on-GPU/decoder-on-CPU split (upstream whisper.cpp PR #4089); do not expect app-side tuning to close this |
| `encode` and `decode` both slower | an op fell back to the CPU | set `FLASH_ATTN_WITH_GPU = false` and repeat, then compare thread counts |
| GPU loses on Tiny/Base, wins on Medium/Large | model-size dependent | make the default depend on the active model |

## Results

### The controlled comparison

The self-test clip (\`jfk.wav\`, 11 s) on a Pixel 10 — Tensor G5, PowerVR D-Series DXT-48-1536 MC1,
Vulkan0, model resident 573.40 MB — with \`large-v3-turbo-q5_0\`. Same clip, same shapes, one run each.

| Run | total (ms) | encode (ms) | decode | fallbacks | transcript |
|---|---|---|---|---|---|
| CPU, 6 threads | 20149 | 19894 | 200 ms / 28 runs | 0 / 0 | *"And so, my fellow Americans, ask not what your country can do for you…"* |
| GPU, flash attn off | 45783 | 22966 | 16288 ms / 212 runs | 1 p / 0 h | *"We're going to be a little bit more."* |
| GPU, flash attn on | 78852 | 73952 | 711 ms / 8 runs | 1 p / 0 h | not read |

### The original run, on a long recording

| Device | Model | GPU | warm run 2 (ms) | encode (ms) | decode (ms) |
|---|---|---|---|---|---|
| pixel 10 | large v3 turbo | off | 161527.72 ms | 154343.20 ms | 3980.56 ms |
| pixel 10 | large v3 turbo | on | 694673 ms, then `failed to decode` | — | — |

That run had flash attention on: it was hardcoded on in the JNI before this file existed, and the
switch was still on until the comparison above.

## Verdict for this phone

**Vulkan stays off by default, and the switch now says so.**

1. **It computes the wrong thing.** The CPU transcribed the sample clip verbatim with no fallback;
   the GPU came back with a different sentence and a logprob fallback in the same conditions. Your
   Italian recording coming back as English is the same failure seen through language detection.
   Flash attention is not the cause — the wrong answer survived turning it off, it only got 1.7x
   faster.
2. **It does not win where it should.** The encoder is 99% of this workload (19.9 s of a 20.1 s CPU
   run), and the GPU does not beat the CPU at it: 23 s with flash attention off, 74 s with it on.
   The batch-1 decoder is 10x slower per token (77 ms against 7.1 ms), but at 0.2 s of a 20 s run
   it is not what decides anything.
3. The likeliest explanation for (2) is that a quantized model this size is bound by streaming its
   weights out of LPDDR5 rather than by arithmetic, and both engines wait on the same memory — the
   GPU cannot accelerate a workload that is already memory-bound. **That is inference, not
   measurement.** Nothing here rules out a different quant, a different model size, or a future
   driver behaving differently.

## The knobs

- Flash attention is now `!useGpu` in `WhisperTranscriptionEngine.kt`, resolved by the table
  above: on the measured device it cost 74 s against 23 s for one encoder pass, against 20 s for
  the CPU.
- `GPU_THREAD_CAP` in `LibWhisper.kt` — how many CPU threads are used while the GPU is doing most
  of the work. Every big core left running competes with the GPU for the same memory bandwidth.
  Still a guess: it was never A/B'd, because the GPU path lost for reasons that had nothing to do
  with threads.

## Phase 0: can any accelerator beat the CPU at the encoder?

ggml-vulkan is not the only way to reach this GPU. LiteRT (Apache-2.0, no Play Services) does, and
its OpenCL accelerator does not share ggml's code — so the encoder was re-measured on it.

Harness: LiteRT **2.2.0** C++ `CompiledModel` API, `litert-community/whisper-large-v3-turbo`
(`whisper_large_v3_turbo_30s_i8.tflite`, 1.09 GB), the `encode` signature, input `[1,128,3000]`,
same 30 s window as everything above. Pixel 10. Timings are wall-clock per run **with a read of the
output inside the timed region** — an accelerated run otherwise only enqueues and reports 0.1 ms.

| Backend | encode (ms) | vs today's CPU path | notes |
|---|---:|---|---|
| whisper.cpp CPU (what ships now, q5_0) | 19,894 | — | the baseline from the section above |
| LiteRT CPU (XNNPACK, i8) | 9,829 | **2.0x faster** | 1462/1462 encode nodes |
| LiteRT GPU (OpenCL, i8) | **4,053** | **4.9x faster** | 1462/1462 encode nodes on the GPU |
| LiteRT NPU (Tensor G5) | 10,434 | ~1.0x | indistinguishable from CPU — the silent-fallback signature |

The NPU dispatcher `libLiteRtDispatch_GoogleTensor.so` loads and the vendor service resolves, but a
model that was not AOT-compiled for Tensor G5 lands on the CPU: the same outcome LiteRT #7787
reports, and consistent with the quantization wall. The NPU is not reachable from a public toolchain
today; the GPU is.

### The catch: the GPU encoder is fast and wrong

Comparing the encoder output against the LiteRT **CPU** output for the identical input and model:

    relative L2 difference  0.90  (90%)
    cosine similarity       0.59
    max absolute difference 10.8   (vector L2 norms: 569.9 CPU vs 568.5 GPU)

Two vectors that agree only to cosine 0.59 are not the same function, and no amount of fp16 rounding
explains it. Requesting `GpuOptions::Precision::kFp32` produced **byte-identical** output and the
same time, so that option did not take effect — the fp16 hypothesis is untested, not disproven.

This is the second independent GPU stack on this device to disagree with its own CPU reference, the
first being ggml-vulkan. The follow-up below shows what that common factor actually is — the
quantized weight path, not the GPU.

### What this means

1. **The headroom is real and worth chasing.** The encoder is 99% of this workload, and 4.05 s
   against 19.9 s is a 4.9x win on the stage that matters. A LiteRT encoder feeding whisper.cpp's
   decoder is the cheap way to collect it (phase 1 above), because the decoder, tokenizer, VAD,
   timestamps and streaming all stay where they are.
2. **Correctness is now the gate, not speed.** Do not start phase 1 until the GPU encoder's output
   is shown to decode into the same transcript. Cheap ways to settle it: feed a real mel, decode
   both embeddings through whisper.cpp, and compare transcripts; then repeat on an unquantized
   model to separate "the delegate mishandles int8 weights" from "this driver computes wrongly".
3. **There is a 2x win that needs no GPU at all.** LiteRT/XNNPACK does the same encoder in 9.8 s
   against whisper.cpp's 19.9 s on the same phone. That is a runtime swap, not a new accelerator,
   and it carries no precision risk from a GPU driver.

## Phase 0b: the correctness experiment

Four models, same harness, same synthetic input, encoder output dumped from the CPU run and the
GPU run and compared elementwise.

| Model | CPU encode | GPU encode | GPU vs CPU |
|---|---:|---:|---|
| whisper-base **f32** (80 mel) | 1,053 ms | **196 ms** (5.4x) | cosine **0.999025**, rel L2 4.44% |
| whisper-base **i8** | 532 ms | — | **fails to compile**: *"Unable to parse bc coord for BATCH axis… shape {bhwc, {1500,1,1,512}}"* |
| large-v3-turbo **i8** | 9,829 ms | 4,053 ms (4.9x) | cosine **0.59** — a different function |
| large-v3-turbo **i4** | 12,011 ms | — | **segfault** inside the OpenCL delegate |

### What this table does and does not show

Every failure in it — LiteRT's i8 and i4 — is on a quantized model, and the one unquantized case
agrees with the CPU to cosine 0.999. That makes "the quantized kernels are broken" the obvious
reading. **It is the right reading for LiteRT and the wrong one for ggml-vulkan**: probe A below
fails on an unquantized f16 model too. The two stacks have separate faults and should not be
described together.

### One option I could not make work

`GpuOptions::Precision::kFp32` produced **byte-identical** output and the same time as the default,
twice. The option never took effect through this harness. So the 4.44% residual between f32 CPU and
f32 GPU is *consistent with* fp16 rounding but is not proven to be; that needs the option to
actually apply, or a transcript-level comparison.

### What is now known

- **A correct GPU encoder is available today — for unquantized models.** whisper-base f32 is 290 MB
  and its encoder runs 5.4x faster on the GPU with the same output.
- **The large models cannot use it.** turbo f32 is ~3.2 GB (too big to ship), turbo i4 segfaults,
  turbo i8 computes the wrong function, base i8 does not compile.
- **The hybrid needs two model files**, because whisper.cpp's decoder still wants its own GGML
  weights: a LiteRT encoder plus a GGML model of the same size. That is fine at base
  (290 MB + 58 MB) and not fine at turbo f32 (3.2 GB + 574 MB).
- **Still unverified:** whether the 4.44% f32 difference survives into the transcript. Feed a real
  mel, decode both embeddings through whisper.cpp, compare the text. That is the last unknown
  before phase 1 is worth starting.

## Probes A and B: what closed the investigation

### A. ggml-vulkan with an unquantized model — it is not a quantization problem

The phase-0b result suggested the failures were all about quantized weights. That was true for
LiteRT and **false for ggml-vulkan**. `ggml-tiny.bin` (f16, unquantized, 78 MB), same clip:

| tiny f16 | total | encode | decode | fallbacks | transcript |
|---|---:|---:|---|---:|---|
| CPU (6 threads) | 440 ms | 323 ms | 65 ms / 25 runs | 0 | *"And so my fellow Americans…"* |
| GPU (Vulkan) | 11,891 ms | 1,132 ms | 7,636 ms / **218 runs** | 1 | *"I"* |

f16 fails exactly like q5_0 did: wrong text, a hallucination loop, and every stage slower than the
CPU. **ggml-vulkan is broken on this device regardless of quantization**, which closes the question
this fork was opened to answer — for this phone, that approach has no path.

The two stacks therefore fail for different reasons and should not be described together:

- **ggml-vulkan** — fails on quantized *and* unquantized models. Backend/driver, not precision.
- **LiteRT OpenCL** — correct at f32 (cosine 0.999, 5.4x), broken at i8 (cosine 0.59) and i4 (segfault).
  This one *is* a quantized-kernel problem.

### B. turbo with less-quantized weights, on the CPU — this is the actual fix

Same model, same clip, same configuration, only the weight format changed:

| large-v3-turbo | file | encode | total | transcript |
|---|---:|---:|---:|---|
| q5_0 (what the app ships) | 574 MB | 18,200 ms | 18,513 ms | correct |
| **q8_0** | 874 MB | **6,258 ms** | **6,511 ms** | correct |
| f16 | 1.62 GB | 12,704 ms | 13,103 ms | correct |

**q8_0 is 2.9x faster than q5_0 and stores more bits per weight.** f16 is twice the bytes and lands
between them, because it is bandwidth-bound. q8_0 is the sweet spot: ggml's 5-bit format does not
map onto the ARM int8 dot-product instruction while q8_0 does, so the cheaper-looking file costs
three times the compute.

### Correcting an earlier claim

Phase 0 said LiteRT/XNNPACK did the encoder in 9.8 s against whisper.cpp's 19.9 s and called that a
2x win for a runtime swap. That comparison was int8 against **q5_0** — it measured the weight
format, not the runtime. whisper.cpp with q8_0 does the same encoder in **6.3 s**, comfortably
faster than LiteRT's CPU path. **The LiteRT integration would buy nothing here.**

### Where this leaves the fork

| Approach | Result |
|---|---|
| ggml-vulkan (the original idea) | dead on this device, quantized or not |
| LiteRT GPU encoder | correct and 5.4x, but only for unquantized models, and turbo f32 is ~3.2 GB |
| LiteRT NPU | gated behind an internal toolchain |
| LiteRT as a CPU runtime swap | now pointless — slower than whisper.cpp on q8_0 |
| **q8_0 weights instead of q5_0** | **2.9x, better accuracy, one model file** |

The speedup was never in the accelerator. It was in a weight format that the CPU already handles
well, and the app ships the one format that chip is worst at.

## Probe C: a second device

One device cannot justify changing what every user downloads, so the same self-test ran on a
different SoC: **Samsung Galaxy Tab S10 FE (SM-X520), Exynos 1580, Android 16** — different vendor,
different core design, one Android version behind the Pixel.

| tier | device | q5_x | q8_0 | speedup | file delta |
|---|---|---:|---:|---:|---:|
| base | Pixel 10 — Tensor G5 | 784 ms | 395 ms | 1.98x | +22 MB |
| base | Tab S10 FE — Exynos 1580 | 1,419 ms | 673 ms | 2.11x | +22 MB |
| small | Pixel 10 | 3,048 ms | 1,219 ms | 2.50x | +74 MB |
| large-v3-turbo | Pixel 10 | 18,200 ms | 6,258 ms | **2.91x** | +300 MB |
| large-v3-turbo | Tab S10 FE | 29,478 ms | 10,850 ms | **2.72x** | +300 MB |

Same clip, same app build, CPU, GPU off, 0 fallbacks throughout. Every run transcribed to the same
text — on the Tab, turbo q8_0 reads back *"And so, my fellow Americans, ask not what your country can
do for you, ask what you can do for your country."*

The absolute numbers differ: the Tab is 1.6-1.9x slower than the Pixel at **both** formats. The
**ratio** is what reproduces — within 7% at base, within 7% at turbo. That is the property a change
to the default model depends on, and it is now measured on two chips instead of one.

Two gaps, neither blocking:

- **medium** was not measured; it interpolates between small (2.50x) and turbo (2.72-2.91x).
- **English-only variants** were not measured. They ship the same kernels, so the effect should be
  identical — but that is an assumption, not a measurement.

### Two traps found while measuring the second device

1. **Pushing a model into `Android/data/<pkg>/files/models` fails silently if the directory does not
   already exist.** `adb shell mkdir` creates it owned by `shell`, and an app cannot traverse a
   directory it does not own — the app reports "no active model" while the file sits right there.
   Launch the app once so it creates the directory, then push.
2. Scroll depth is device-dependent: the Pixel needed 3 swipes to reach the self-test on a
   1080x2424 screen, the Tab needed 7 on a 1440x2304 one.
