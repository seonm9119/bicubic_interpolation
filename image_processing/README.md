# Image Processing Lab

이 폴더는 **다양한 image processing 기법을 다룰 수 있고, 필요한 핵심 알고리즘은 직접 구현할 수 있다**는 점을 보여주기 위한 포트폴리오용 모듈이다. 단순히 라이브러리 필터를 호출하는 데서 끝내지 않고, Sobel, Gaussian, LoG, fuzzy stretching, texture density, local contrast, corner gate, critical feature fusion을 직접 계산하고, 이를 API로 노출해 웹 화면에서 바로 테스트할 수 있게 구성했다.

이 프로젝트에서 중요한 메시지는 두 가지다. 첫째, 이미지 decode / encode / pixel container처럼 안정성이 중요한 기반 처리는 검증된 라이브러리인 ImageSharp를 사용했다. 둘째, 실제 image processing 로직은 직접 kernel과 map 연산으로 구현했다. 즉 라이브러리를 쓸 줄도 알고, 라이브러리 뒤에 있는 핵심 계산을 직접 만들 줄도 있다는 점을 보여주는 모듈이다.

## Portfolio Position

이 README는 사용법만 적은 문서가 아니라 포트폴리오 어필 페이지다. 평가자가 이 폴더를 봤을 때 읽어야 하는 핵심은 다음과 같다.

- 다양한 image processing operation을 한 모듈에서 다룰 수 있다.
  Sobel edge, Gaussian blur, LoG sharpening, binary threshold, fuzzy stretching, texture density, local contrast, corner gate, critical feature fusion을 같은 pipeline에서 처리한다.

- 일부는 라이브러리를 쓰고, 핵심은 직접 구현했다.
  ImageSharp는 이미지 로딩, pixel access, PNG 저장에 사용하고, convolution, thresholding, feature map, normalization, fuzzy intensity transform은 C# 루프와 커널 연산으로 직접 작성했다.

- 구현을 API화해서 직접 테스트할 수 있게 만들었다.
  `ImageProcessingPipeline`을 ASP.NET Core Minimal API endpoint와 연결해, 포트폴리오 페이지에서 이미지를 업로드하고 operation과 파라미터를 바꿔가며 바로 결과를 확인할 수 있다.

면접에서 이 프로젝트는 다음처럼 설명하는 것이 가장 적절하다.

```text
Image processing library를 단순 호출한 프로젝트가 아니라,
이미지 입출력은 안정적인 라이브러리에 맡기고,
Sobel / LoG / fuzzy stretching / feature fusion 같은 핵심 연산은 직접 구현했다.
그리고 이 구현을 API로 감싸서 웹에서 다양한 이미지로 바로 테스트할 수 있게 만들었다.
```

## Library vs Custom Implementation

| Area | Choice | Why |
|---|---|---|
| Image decode / encode | `SixLabors.ImageSharp` | PNG/JPG/BMP/WEBP 계열 입력을 안전하게 읽고, 결과 PNG를 안정적으로 생성하기 위해 사용 |
| Pixel container | `Image<Rgba32>` | C#에서 channel 단위 접근과 output image 생성이 명확함 |
| Grayscale conversion | Custom | `0.299R + 0.587G + 0.114B` 변환을 직접 적용해 이후 모든 feature map의 입력을 통일 |
| Convolution | Custom | Gaussian, Sobel, LoG, local mean kernel을 직접 적용해 필터 동작을 제어 |
| Thresholding | Custom | fixed / mean / max-min threshold를 직접 계산 |
| Fuzzy stretching | Custom | min / max / mean intensity 기반 fuzzy map과 stretch/cut/midpoint 조정을 직접 구현 |
| Feature fusion | Custom | edge, LoG, texture, contrast, corner, inverse fuzzy response를 OR gate로 결합 |
| API boundary | ASP.NET Core Minimal API | 같은 구현을 업로드 이미지와 기본 샘플 이미지에 재사용하기 위해 endpoint로 노출 |

이 구조는 "라이브러리를 안 쓴다"가 아니라, 라이브러리와 직접 구현의 경계를 의도적으로 나눈 것이다. 안정적인 I/O는 라이브러리에 맡기고, 포트폴리오에서 보여줘야 하는 image processing 사고력은 직접 구현 영역에 둔다.

## Supported Operations

`ImageProcessingPipeline.ProcessToPng`는 `operation` 값에 따라 다음 처리를 수행한다.

| Operation | Implementation | Purpose |
|---|---|---|
| `sobel-edge` | vertical / horizontal Sobel 직접 convolution | 경계 응답 추출 |
| `gaussian-blur` | 5x5 Gaussian kernel 반복 적용 | noise 완화와 smoothing |
| `log-sharpening` | Gaussian smoothing 후 Laplacian detail 적용 | edge/detail 강조 |
| `fuzzy-stretching` | fuzzy intensity map, midpoint, cut, strength 직접 계산 | 저대비 구조 강조 |
| `binary-threshold` | fixed / mean / max-min threshold | foreground/background 분리 |
| `absolute-edge` | absolute Sobel magnitude | feature weight용 edge map |
| `absolute-log` | absolute LoG response | 세부 구조 map |
| `texture-density` | local mean 대비 intensity difference | 질감 밀도 map |
| `local-contrast` | 원본 gray와 Gaussian map 차이 | 국소 contrast map |
| `corner-gate` | vertical/horizontal Sobel 동시 응답 | corner-like 구조 강조 |
| `critical-feature-fusion` | multiple feature OR fusion | feature-weighted bicubic에 사용할 핵심 구조 weight map |

핵심은 operation 수가 많다는 것 자체가 아니라, 서로 다른 image processing 신호를 같은 입력 map 위에서 조합하고, 그 결과를 API 응답으로 재사용할 수 있게 만든 점이다.

## Feature Map Design

이 모듈의 중요한 역할은 Feature-weighted Bicubic interpolation에 들어갈 `critical feature map`을 만드는 것이다. 단일 edge detector만 사용하면 지문 ridge, 홍채 radial texture, PCB trace처럼 서로 다른 구조를 안정적으로 잡기 어렵다. 그래서 여러 신호를 OR gate로 결합한다.

```text
critical_feature =
    sobel_magnitude
    OR absolute_log
    OR texture_density
    OR local_contrast
    OR corner_gate
    OR inverse_fuzzy
```

이후 feature gate를 `CriticalFeatureScale`로 줄이고 `CriticalFeatureLimit`으로 제한해, 너무 강한 응답이 보간 결과를 과도하게 흔들지 않도록 했다. 결과 feature map은 `FeatureMapResult`에서 `uint8-row-major` base64로 반환되어 다른 interpolation 모듈에서 그대로 사용할 수 있다.

## API-Connected Test Flow

이 모듈은 단독 함수로 끝나지 않고, 상위 API 프로젝트의 `ImageProcessingEndpoints.cs`를 통해 직접 테스트 가능한 endpoint로 노출된다.

| Endpoint | Method | Role |
|---|---|---|
| `/api/image-processing/health` | `GET` | image-processing API 상태 확인 |
| `/api/image-processing/default-preview` | `GET` | 기본 `lenna-test.png` 샘플에 operation 적용 |
| `/api/image-processing/apply` | `POST` | 업로드 이미지에 선택한 operation 적용 |
| `/api/image-processing/feature-map` | `POST` | 업로드 이미지에서 feature-weighted bicubic용 feature map 생성 |
| `/api/image-processing/cached-images/{cacheId}.png` | `GET` | 처리 결과 PNG 반환 |

프론트 페이지:

```text
http://192.168.0.11:3000/projects/computer-vision/image-processing
```

사용 흐름은 다음과 같다.

1. 사용자가 기본 Lenna 샘플을 보거나 직접 이미지를 업로드한다.
2. Sobel, Gaussian, LoG, fuzzy stretching, feature fusion 중 하나를 선택한다.
3. threshold, gain, kernel size, blur radius, fuzzy strength 같은 파라미터를 조정한다.
4. frontend가 `/api/image-processing/apply` 또는 `/api/image-processing/default-preview`를 호출한다.
5. API가 `ImageProcessingPipeline.ProcessToPng`를 실행한다.
6. 결과 PNG는 memory cache에 저장되고, frontend는 cached image URL을 받아 화면에 표시한다.

이렇게 구성한 이유는 포트폴리오 화면에서 단순 스크린샷이 아니라, 실제 API 호출로 image processing 결과가 생성되는 흐름을 보여주기 위해서다.

## Request / Response Shape

업로드 이미지 처리 요청은 multipart form으로 들어온다.

```text
POST /api/image-processing/apply

FormData:
  image
  operation=sobel-edge | gaussian-blur | log-sharpening | fuzzy-stretching | ...
  edgeThresholdPercent=24
  sobelGainPercent=68
  sobelKernelSize=3
  gaussianRadius=2
  gaussianSigmaPercent=55
  logKernelStrengthPercent=70
  fuzzyStrengthPercent=72
  fuzzyCutPercent=50
  fuzzyMidpointPercent=54
```

응답은 처리 결과 이미지를 직접 base64로 싣지 않고, cache image URL을 반환한다.

```json
{
  "success": true,
  "cached": false,
  "input": {
    "fileName": "sample.png",
    "contentType": "image/png",
    "sha256": "..."
  },
  "request": {
    "operation": "absolute-edge",
    "edgeThresholdPercent": 24,
    "sobelGainPercent": 68,
    "sobelKernelSize": 3
  },
  "output": {
    "width": 512,
    "height": 512,
    "operationName": "Absolute Edge Map",
    "resultImage": "/api/image-processing/cached-images/{cacheId}.png"
  }
}
```

이미지 결과를 cache URL로 분리한 이유는 큰 PNG payload를 JSON에 반복해서 싣지 않고, 브라우저와 API cache를 활용해 포트폴리오 화면의 반응성을 유지하기 위해서다.

## Implementation Notes

- `ImageProcessingPipeline.cs`
  grayscale 변환, Sobel, Gaussian, LoG, fuzzy stretching, binary threshold, texture density, local contrast, corner gate, feature fusion을 담당한다.

- `ImageProcessingModels.cs`
  API와 pipeline 사이에서 사용하는 request / response record를 정의한다.

- `ImageProcessing.Api.csproj`
  .NET 8 class library 형태의 image-processing module이며, ImageSharp에만 의존한다.

- `assets/lenna-test.png`
  default preview API와 frontend 첫 화면에서 사용하는 샘플 이미지다.

- `../api/ImageProcessingEndpoints.cs`
  이 모듈을 실제 HTTP endpoint로 노출하고, request parsing, bounded parameter validation, memory cache, cached PNG response를 담당한다.

## Run

상위 bicubic interpolation API와 함께 실행한다.

```bash
cd /home/nami/repo/gpt_analysis/project/bicubic_interpolation
docker compose up -d --build sr-bicubic
```

API 상태 확인:

```bash
curl http://127.0.0.1:8020/api/image-processing/health
```

기본 샘플 preview:

```bash
curl "http://127.0.0.1:8020/api/image-processing/default-preview?operation=absolute-edge"
```

포트폴리오 웹 페이지:

```text
http://192.168.0.11:3000/projects/computer-vision/image-processing
```

## What I Wanted To Show

이 모듈에서 보여주고 싶었던 핵심은 "이미지 처리 버튼을 만들었다"가 아니다. 영상에서 어떤 구조를 잡아야 하는지에 따라 Sobel, LoG, texture density, local contrast, fuzzy stretching 같은 신호를 선택하고, 필요한 계산을 직접 구현한 뒤, 이를 API와 frontend로 연결해 누구나 이미지를 바꿔가며 테스트할 수 있게 만든 점이다.

포트폴리오 관점에서 이 README의 결론은 다음과 같다.

```text
나는 image processing 라이브러리를 사용할 줄 알고,
동시에 핵심 필터와 feature map 계산을 직접 구현할 수 있다.
그리고 그 구현을 API화해서 웹에서 바로 검증 가능한 도구로 만들 수 있다.
```
