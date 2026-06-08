using System.Security.Cryptography;
using BicubicInterpolation.Api;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 50 * 1024 * 1024;
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 50 * 1024 * 1024;
});

var srganApiBaseUrl = builder.Configuration["SRGAN_API_BASE_URL"] ?? "http://sr-benchmark:8080";
builder.Services.AddHttpClient("sr-benchmark-api", client =>
{
    client.BaseAddress = new Uri(srganApiBaseUrl);
    client.Timeout = TimeSpan.FromMinutes(5);
});

var app = builder.Build();
var sampleImageDirectory = builder.Configuration["BICUBIC_SAMPLE_DIR"] ?? ResolveDefaultSampleImageDirectory(app.Environment.ContentRootPath);
const int srganBenchmarkScaleFactor = 4;
const long srBenchmarkMaxInputPixels = 65_536;
var sampleImages = new[]
{
    new BicubicSampleImage(
        "set5-butterfly",
        "Set5 Butterfly",
        "SR benchmark",
        "set5-butterfly.png",
        "image/png",
        256,
        256,
        "Set5 HR dataset",
        "https://huggingface.co/datasets/eugenesiow/Set5/tree/main/data"),
    new BicubicSampleImage(
        "lenna",
        "Lenna",
        "Classic test image",
        "lenna-test.png",
        "image/png",
        512,
        512,
        "Lenna test image",
        "https://en.wikipedia.org/wiki/Lenna"),
    new BicubicSampleImage(
        "set5-woman",
        "Set5 Woman",
        "SR benchmark",
        "set5-woman.png",
        "image/png",
        228,
        344,
        "Set5 HR dataset",
        "https://huggingface.co/datasets/eugenesiow/Set5/tree/main/data"),
    new BicubicSampleImage(
        "fingerprint",
        "Fingerprint",
        "Ridge texture",
        "fingerprint.jpg",
        "image/jpeg",
        960,
        640,
        "Wikimedia Commons fingerprint",
        "https://commons.wikimedia.org/wiki/File:Fingerprint.jpg"),
    new BicubicSampleImage(
        "iris",
        "Iris",
        "Biometric texture",
        "iris.png",
        "image/png",
        960,
        848,
        "Wikimedia Commons iris",
        "https://commons.wikimedia.org/wiki/File:ColourIris.png"),
    new BicubicSampleImage(
        "feature-score-pcb",
        "PCB Dataset",
        "DeepPCB average benchmark",
        "feature-score-pcb.png",
        "image/png",
        512,
        512,
        "DeepPCB",
        "https://github.com/tangsanli5201/DeepPCB"),
    new BicubicSampleImage(
        "feature-score-pcb-proposed",
        "PCB Proposed Output",
        "Feature-weighted bicubic output",
        "feature-score-pcb-proposed.png",
        "image/png",
        512,
        512,
        "DeepPCB",
        "https://github.com/tangsanli5201/DeepPCB"),
    new BicubicSampleImage(
        "feature-score-fingerprint",
        "Fingerprint Dataset",
        "CrossMatch average benchmark",
        "feature-score-fingerprint.png",
        "image/png",
        512,
        512,
        "Neurotechnology CrossMatch Sample DB",
        "https://www.neurotechnology.com/download.html"),
    new BicubicSampleImage(
        "feature-score-fingerprint-proposed",
        "Fingerprint Proposed Output",
        "Feature-weighted bicubic output",
        "feature-score-fingerprint-proposed.png",
        "image/png",
        512,
        512,
        "Neurotechnology CrossMatch Sample DB",
        "https://www.neurotechnology.com/download.html"),
    new BicubicSampleImage(
        "feature-score-iris",
        "Iris Dataset",
        "UPOL average benchmark",
        "feature-score-iris.png",
        "image/png",
        512,
        512,
        "UPOL Iris Database",
        "https://phoenix.inf.upol.cz/iris/"),
    new BicubicSampleImage(
        "feature-score-iris-proposed",
        "Iris Proposed Output",
        "Feature-weighted bicubic output",
        "feature-score-iris-proposed.png",
        "image/png",
        512,
        512,
        "UPOL Iris Database",
        "https://phoenix.inf.upol.cz/iris/")
};

app.MapImageProcessingEndpoints();

app.MapGet("/api/bicubic/health", () =>
{
    return Results.Ok(new
    {
        success = true,
        status = "ok",
        service = "bicubic-interpolation-api",
        language = "C#",
        mode = "container"
    });
});

app.MapGet("/api/bicubic/samples", () =>
{
    return Results.Ok(new
    {
        success = true,
        samples = sampleImages.Select(CreateSampleImageResponse)
    });
});

app.MapGet("/api/bicubic/samples/{sampleFileName}", (string sampleFileName) =>
{
    var safeSampleFileName = Path.GetFileName(sampleFileName);
    var sampleImage = sampleImages.FirstOrDefault(candidateSampleImage =>
        string.Equals(candidateSampleImage.FileName, safeSampleFileName, StringComparison.OrdinalIgnoreCase));

    if (sampleImage is null)
    {
        return Results.NotFound(new
        {
            success = false,
            error = "요청한 샘플 이미지를 찾을 수 없습니다."
        });
    }

    var sampleImagePath = Path.Combine(sampleImageDirectory, sampleImage.FileName);

    if (!File.Exists(sampleImagePath))
    {
        return Results.NotFound(new
        {
            success = false,
            error = "샘플 이미지 파일이 컨테이너에 마운트되지 않았습니다."
        });
    }

    return Results.File(sampleImagePath, sampleImage.ContentType, enableRangeProcessing: true);
});

app.MapGet("/api/bicubic/benchmark-samples", () =>
{
    return Results.Ok(new
    {
        success = true,
        scaleFactor = srganBenchmarkScaleFactor,
        samples = sampleImages.Select(sampleImage => CreateBenchmarkSampleResponse(sampleImage, srganBenchmarkScaleFactor))
    });
});

app.MapGet("/api/bicubic/benchmark-samples/{sampleId}/low-resolution", async (string sampleId) =>
{
    var sampleImage = FindSampleImage(sampleImages, sampleId);

    if (sampleImage is null)
    {
        return Results.NotFound(new
        {
            success = false,
            error = "요청한 benchmark sample을 찾을 수 없습니다."
        });
    }

    try
    {
        var highResolutionImageBytes = await ReadSampleImageBytesAsync(sampleImageDirectory, sampleImage);
        var lowResolutionImage = SuperResolutionBenchmarkTools.CreateLowResolutionInput(
            highResolutionImageBytes,
            srganBenchmarkScaleFactor);

        return Results.File(
            lowResolutionImage.PngBytes,
            "image/png",
            $"{sampleImage.Id}-lr-x{srganBenchmarkScaleFactor}.png",
            enableRangeProcessing: true);
    }
    catch (InvalidOperationException exception)
    {
        return Results.NotFound(new
        {
            success = false,
            error = exception.Message
        });
    }
});

app.MapGet("/api/bicubic/benchmark", async (string? sampleId, int? featureWeightPercent, IHttpClientFactory httpClientFactory) =>
{
    var sampleImage = FindSampleImage(sampleImages, sampleId) ?? sampleImages[0];
    var boundedFeatureWeightPercent = Math.Clamp(featureWeightPercent ?? 100, 0, 100);

    if (boundedFeatureWeightPercent == 100 &&
        TryCreatePrecomputedBenchmarkResponse(sampleImage, srganBenchmarkScaleFactor, out var precomputedBenchmarkResponse))
    {
        return Results.Ok(precomputedBenchmarkResponse);
    }

    try
    {
        var highResolutionImageBytes = await ReadSampleImageBytesAsync(sampleImageDirectory, sampleImage);
        var lowResolutionImage = SuperResolutionBenchmarkTools.CreateLowResolutionInput(
            highResolutionImageBytes,
            srganBenchmarkScaleFactor);
        var classicBicubicResult = ClassicBicubicInterpolator.ResizeToPngDataUrl(
            lowResolutionImage.PngBytes,
            srganBenchmarkScaleFactor);
        var classicBicubicMetrics = SuperResolutionBenchmarkTools.CalculateMetrics(
            highResolutionImageBytes,
            classicBicubicResult.PngBytes);
        var featureWeightMap = FeatureWeightMapBuilder.BuildFeatureWeightMap(lowResolutionImage.PngBytes, boundedFeatureWeightPercent);
        var featureWeightedResult = FeatureWeightedBicubicInterpolator.ResizeToPngDataUrl(
            lowResolutionImage.PngBytes,
            srganBenchmarkScaleFactor,
            featureWeightMap);
        var featureWeightedMetrics = SuperResolutionBenchmarkTools.CalculateMetrics(
            highResolutionImageBytes,
            featureWeightedResult.PngBytes);
        var srganMethodResponse = await CreateSrganBenchmarkMethodResponse(
            httpClientFactory,
            highResolutionImageBytes,
            lowResolutionImage.PngBytes);
        var realEsrganMethodResponse = await CreateRealEsrganBenchmarkMethodResponse(
            httpClientFactory,
            highResolutionImageBytes,
            lowResolutionImage.PngBytes);

        return Results.Ok(new
        {
            success = true,
            sample = CreateBenchmarkSampleResponse(sampleImage, srganBenchmarkScaleFactor),
            request = new
            {
                scaleFactor = srganBenchmarkScaleFactor,
                featureWeightPercent = boundedFeatureWeightPercent
            },
            resolution = new
            {
                inputWidth = lowResolutionImage.Width,
                inputHeight = lowResolutionImage.Height,
                targetWidth = sampleImage.Width,
                targetHeight = sampleImage.Height,
                linearScale = srganBenchmarkScaleFactor,
                pixelScale = srganBenchmarkScaleFactor * srganBenchmarkScaleFactor,
                pixelIncreasePercent = (srganBenchmarkScaleFactor * srganBenchmarkScaleFactor - 1) * 100
            },
            methods = new object[]
            {
                CreateBenchmarkMethodResponse(
                    "classic-bicubic",
                    "Bicubic",
                    "complete",
                    classicBicubicResult,
                    classicBicubicMetrics,
                    "LR x4 input을 기존 bicubic interpolation으로 복원한 baseline입니다."),
                srganMethodResponse,
                realEsrganMethodResponse,
                CreateBenchmarkMethodResponse(
                    "feature-weighted",
                    "Feature-weighted Bicubic",
                    "complete",
                    featureWeightedResult,
                    featureWeightedMetrics,
                    "feature map 기반 weight를 적용한 제안 방법입니다.")
            },
            paperReference = CreateSrganPaperReferenceResponse()
        });
    }
    catch (InvalidOperationException exception)
    {
        return Results.BadRequest(new
        {
            success = false,
            error = exception.Message
        });
    }
});

app.MapGet("/api/bicubic/default-comparison", (string? sampleId, int? featureWeightPercent) =>
{
    var sampleImage = FindSampleImage(sampleImages, sampleId) ?? FindSampleImage(sampleImages, "iris");
    var boundedFeatureWeightPercent = Math.Clamp(featureWeightPercent ?? 100, 0, 100);

    if (sampleImage is null ||
        !string.Equals(sampleImage.Id, "iris", StringComparison.OrdinalIgnoreCase) ||
        boundedFeatureWeightPercent != 100)
    {
        return Results.NotFound(new
        {
            success = false,
            error = "요청한 기본 비교 cache를 찾을 수 없습니다."
        });
    }

    try
    {
        return Results.Ok(new
        {
            success = true,
            cached = true,
            cacheType = "default-iris-comparison",
            sample = CreateBenchmarkSampleResponse(sampleImage, srganBenchmarkScaleFactor),
            srgan = CreateCachedComparisonResponse(
                sampleImageDirectory,
                "iris-default-srgan.png",
                "srgan",
                "SRGAN",
                960,
                848,
                CreateCachedMetricResponse(35.5101, 0.998014, 18.2841, 4.2760, 3.7233),
                "Cached default iris SRGAN result."),
            realEsrgan = CreateCachedComparisonResponse(
                sampleImageDirectory,
                "iris-default-real-esrgan.png",
                "real-esrgan",
                "Real-ESRGAN",
                960,
                848,
                CreateCachedMetricResponse(36.7571, 0.997743, 13.7205, 3.7041, 2.7021),
                "Cached default iris Real-ESRGAN result."),
            improved = CreateCachedComparisonResponse(
                sampleImageDirectory,
                "iris-default-feature-weighted.png",
                "feature-weighted",
                "Feature-weighted Bicubic",
                960,
                848,
                CreateCachedMetricResponse(49.5545, 0.999873, 0.7205, 0.8488, 0.5025),
                "Cached default iris Feature-weighted Bicubic result.")
        });
    }
    catch (InvalidOperationException exception)
    {
        return Results.NotFound(new
        {
            success = false,
            error = exception.Message
        });
    }
});

app.MapPost("/api/bicubic/interpolate", async (HttpRequest request, IHttpClientFactory httpClientFactory) =>
{
    if (!request.HasFormContentType)
    {
        return Results.BadRequest(new
        {
            success = false,
            error = "multipart/form-data 요청이 필요합니다."
        });
    }

    var form = await request.ReadFormAsync();
    var imageFile = form.Files.GetFile("image");

    if (imageFile is null || imageFile.Length == 0)
    {
        return Results.BadRequest(new
        {
            success = false,
            error = "image 파일이 비어 있습니다."
        });
    }

    await using var imageStream = imageFile.OpenReadStream();
    using var memoryStream = new MemoryStream();
    await imageStream.CopyToAsync(memoryStream);

    var imageBytes = memoryStream.ToArray();
    var imageDimensions = ImageHeaderReader.Read(imageBytes);
    var scaleFactor = ReadBoundedInt(form["scaleFactor"], 2, 2, 4);
    var featureWeightPercent = ReadBoundedInt(form["featureWeightPercent"], 100, 0, 100);
    var interpolationMode = NormalizeInterpolationMode(form["mode"]);
    var targetWidth = imageDimensions.Width.HasValue ? imageDimensions.Width.Value * scaleFactor : 320;
    var targetHeight = imageDimensions.Height.HasValue ? imageDimensions.Height.Value * scaleFactor : 320;
    var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();
    BicubicImageResult? bicubicImageResult = null;
    SuperResolutionMetricResult? consistencyMetrics = null;
    var outputMessage = string.Empty;

    try
    {
        if (RequiresX4ModelScale(interpolationMode) && scaleFactor != srganBenchmarkScaleFactor)
        {
            return Results.BadRequest(new
            {
                success = false,
                error = "SRGAN과 Real-ESRGAN 비교는 x4 scale factor가 필요합니다."
            });
        }

        if (RequiresX4ModelScale(interpolationMode) && imageDimensions.Width.HasValue && imageDimensions.Height.HasValue)
        {
            var inputPixelCount = (long)imageDimensions.Width.Value * imageDimensions.Height.Value;

            if (inputPixelCount > srBenchmarkMaxInputPixels)
            {
                var maxSquareSide = (int)Math.Sqrt(srBenchmarkMaxInputPixels);

                return Results.BadRequest(new
                {
                    success = false,
                    error = $"SRGAN과 Real-ESRGAN 비교 입력이 너무 큽니다. {maxSquareSide}x{maxSquareSide} 이하 또는 {srBenchmarkMaxInputPixels} pixels 이하 이미지로 테스트해 주세요."
                });
            }
        }

        if (interpolationMode == "classic-bicubic")
        {
            bicubicImageResult = ClassicBicubicInterpolator.ResizeToPngDataUrl(imageBytes, scaleFactor);
            outputMessage = "Classic bicubic interpolation 결과 이미지를 생성했습니다.";
        }
        else if (interpolationMode == "feature-weighted")
        {
            var featureWeightMap = FeatureWeightMapBuilder.BuildFeatureWeightMap(imageBytes, featureWeightPercent);
            bicubicImageResult = FeatureWeightedBicubicInterpolator.ResizeToPngDataUrl(
                imageBytes,
                scaleFactor,
                featureWeightMap);
            outputMessage = "Feature weighted bicubic interpolation 결과 이미지를 생성했습니다.";
        }
        else
        {
            var srganClient = httpClientFactory.CreateClient("sr-benchmark-api");
            var srResult = interpolationMode == "real-esrgan"
                ? await SrganInferenceClient.RequestRealEsrganImage(srganClient, imageBytes)
                : await SrganInferenceClient.RequestSrganImage(srganClient, imageBytes);

            bicubicImageResult = new BicubicImageResult(srResult.Width, srResult.Height, srResult.PngBytes);
            outputMessage = srResult.Message;
        }

        targetWidth = bicubicImageResult.Width;
        targetHeight = bicubicImageResult.Height;
        var downsampledOutputImage = SuperResolutionBenchmarkTools.CreateLowResolutionInput(
            bicubicImageResult.PngBytes,
            scaleFactor);
        consistencyMetrics = SuperResolutionBenchmarkTools.CalculateMetrics(imageBytes, downsampledOutputImage.PngBytes);
    }
    catch (InvalidOperationException exception)
    {
        return Results.BadRequest(new
        {
            success = false,
            error = exception.Message
        });
    }

    return Results.Ok(new
    {
        success = true,
        status = "received",
        input = new
        {
            fileName = imageFile.FileName,
            contentType = imageFile.ContentType,
            size = imageFile.Length,
            width = imageDimensions.Width,
            height = imageDimensions.Height,
            format = imageDimensions.Format,
            sha256 = imageHash
        },
        request = new
        {
            scaleFactor,
            featureWeightPercent,
            mode = interpolationMode
        },
        output = new
        {
            targetWidth,
            targetHeight,
            resultImage = bicubicImageResult?.DataUrl,
            message = outputMessage,
            metrics = consistencyMetrics is null ? null : CreateMetricResponse(consistencyMetrics),
            metricReference = new
            {
                type = "downsample-consistency",
                label = "LR downsample consistency",
                referenceWidth = imageDimensions.Width,
                referenceHeight = imageDimensions.Height
            }
        },
        pipeline = new[]
        {
            "upload",
            "read image header",
            interpolationMode == "feature-weighted" ? "build image-processing feature map" : "classic bicubic interpolation",
            interpolationMode == "feature-weighted" ? "feature-weighted bicubic interpolation" : "encode png result",
            "return api response"
        }
    });
});

app.Run();

static int ReadBoundedInt(string? rawValue, int defaultValue, int minValue, int maxValue)
{
    if (!int.TryParse(rawValue, out var parsedValue))
    {
        return defaultValue;
    }

    return Math.Clamp(parsedValue, minValue, maxValue);
}

static string NormalizeInterpolationMode(string? rawMode)
{
    return rawMode switch
    {
        "classic-bicubic" => "classic-bicubic",
        "srgan" => "srgan",
        "real-esrgan" => "real-esrgan",
        _ => "feature-weighted"
    };
}

static bool RequiresX4ModelScale(string interpolationMode)
{
    return interpolationMode == "srgan" || interpolationMode == "real-esrgan";
}

static string ResolveDefaultSampleImageDirectory(string contentRootPath)
{
    var siblingSampleImageDirectory = Path.GetFullPath(Path.Combine(contentRootPath, "..", "sample"));

    if (Directory.Exists(siblingSampleImageDirectory))
    {
        return siblingSampleImageDirectory;
    }

    return Path.Combine(contentRootPath, "sample");
}

static object CreateSampleImageResponse(BicubicSampleImage sampleImage)
{
    return new
    {
        sampleImage.Id,
        sampleImage.Label,
        sampleImage.Detail,
        sampleImage.FileName,
        sampleImage.ContentType,
        sampleImage.Width,
        sampleImage.Height,
        sampleImage.SourceName,
        sampleImage.SourceUrl,
        imageUrl = $"/api/bicubic/samples/{Uri.EscapeDataString(sampleImage.FileName)}"
    };
}

static object CreateBenchmarkSampleResponse(BicubicSampleImage sampleImage, int scaleFactor)
{
    return new
    {
        sampleImage.Id,
        sampleImage.Label,
        sampleImage.Detail,
        sampleImage.SourceName,
        sampleImage.SourceUrl,
        scaleFactor,
        lowResolution = new
        {
            width = Math.Max(1, sampleImage.Width / scaleFactor),
            height = Math.Max(1, sampleImage.Height / scaleFactor),
            imageUrl = $"/api/bicubic/benchmark-samples/{Uri.EscapeDataString(sampleImage.Id)}/low-resolution"
        },
        highResolution = new
        {
            width = sampleImage.Width,
            height = sampleImage.Height,
            imageUrl = $"/api/bicubic/samples/{Uri.EscapeDataString(sampleImage.FileName)}"
        }
    };
}

static BicubicSampleImage? FindSampleImage(BicubicSampleImage[] sampleImages, string? sampleId)
{
    if (string.IsNullOrWhiteSpace(sampleId))
    {
        return null;
    }

    return sampleImages.FirstOrDefault(candidateSampleImage =>
        string.Equals(candidateSampleImage.Id, sampleId, StringComparison.OrdinalIgnoreCase));
}

static async Task<byte[]> ReadSampleImageBytesAsync(string sampleImageDirectory, BicubicSampleImage sampleImage)
{
    var sampleImagePath = Path.Combine(sampleImageDirectory, sampleImage.FileName);

    if (!File.Exists(sampleImagePath))
    {
        throw new InvalidOperationException("샘플 이미지 파일이 컨테이너에 마운트되지 않았습니다.");
    }

    return await File.ReadAllBytesAsync(sampleImagePath);
}

static object CreateBenchmarkMethodResponse(
    string id,
    string label,
    string status,
    BicubicImageResult imageResult,
    SuperResolutionMetricResult metricResult,
    string note)
{
    return new
    {
        id,
        label,
        status,
        output = new
        {
            width = imageResult.Width,
            height = imageResult.Height,
            resultImage = imageResult.DataUrl
        },
        metrics = CreateMetricResponse(metricResult),
        note
    };
}

static bool TryCreatePrecomputedBenchmarkResponse(
    BicubicSampleImage sampleImage,
    int scaleFactor,
    out object response)
{
    var methods = sampleImage.Id switch
    {
        "set5-butterfly" => new[]
        {
            CreateCachedBenchmarkMethodResponse("classic-bicubic", "Bicubic", 19.1929, 0.892718, 783.0466, 27.9830, 17.1146),
            CreateCachedBenchmarkMethodResponse("srgan", "SRGAN", 23.4497, 0.966190, 293.8427, 17.1418, 11.0174),
            CreateCachedBenchmarkMethodResponse("real-esrgan", "Real-ESRGAN", 20.5984, 0.945396, 566.5489, 23.8023, 16.3036),
            CreateCachedBenchmarkMethodResponse("feature-weighted", "Feature-weighted Bicubic", 20.7737, 0.931423, 544.1335, 23.3267, 15.0930)
        },
        "lenna" => new[]
        {
            CreateCachedBenchmarkMethodResponse("classic-bicubic", "Bicubic", 27.4548, 0.973705, 116.8428, 10.8094, 6.6036),
            CreateCachedBenchmarkMethodResponse("srgan", "SRGAN", 27.7916, 0.981384, 108.1244, 10.3983, 7.0775),
            CreateCachedBenchmarkMethodResponse("real-esrgan", "Real-ESRGAN", 27.1709, 0.978271, 124.7355, 11.1685, 7.2500),
            CreateCachedBenchmarkMethodResponse("feature-weighted", "Feature-weighted Bicubic", 28.8303, 0.982010, 85.1241, 9.2263, 5.6361)
        },
        "set5-woman" => new[]
        {
            CreateCachedBenchmarkMethodResponse("classic-bicubic", "Bicubic", 22.8415, 0.966941, 338.0089, 18.3850, 9.8941),
            CreateCachedBenchmarkMethodResponse("srgan", "SRGAN", 26.0613, 0.985449, 161.0464, 12.6904, 7.7212),
            CreateCachedBenchmarkMethodResponse("real-esrgan", "Real-ESRGAN", 24.4791, 0.980649, 231.8329, 15.2261, 9.1982),
            CreateCachedBenchmarkMethodResponse("feature-weighted", "Feature-weighted Bicubic", 24.4865, 0.977918, 231.4350, 15.2130, 8.1901)
        },
        "iris" => new[]
        {
            CreateCachedBenchmarkMethodResponse("classic-bicubic", "Bicubic", 37.2617, 0.997701, 12.2155, 3.4951, 2.2792),
            CreateCachedBenchmarkMethodResponse("srgan", "SRGAN", 33.6142, 0.996348, 28.2921, 5.3190, 4.2313),
            CreateCachedBenchmarkMethodResponse("real-esrgan", "Real-ESRGAN", 33.6725, 0.995072, 27.9143, 5.2834, 3.6372),
            CreateCachedBenchmarkMethodResponse("feature-weighted", "Feature-weighted Bicubic", 38.9628, 0.998463, 8.2567, 2.8734, 1.7543)
        },
        _ => null
    };

    if (methods is null)
    {
        response = new { };
        return false;
    }

    response = new
    {
        success = true,
        cached = true,
        cacheType = "precomputed-benchmark",
        sample = CreateBenchmarkSampleResponse(sampleImage, scaleFactor),
        request = new
        {
            scaleFactor,
            featureWeightPercent = 100
        },
        resolution = new
        {
            inputWidth = Math.Max(1, sampleImage.Width / scaleFactor),
            inputHeight = Math.Max(1, sampleImage.Height / scaleFactor),
            targetWidth = sampleImage.Width,
            targetHeight = sampleImage.Height,
            linearScale = scaleFactor,
            pixelScale = scaleFactor * scaleFactor,
            pixelIncreasePercent = (scaleFactor * scaleFactor - 1) * 100
        },
        methods,
        paperReference = CreateSrganPaperReferenceResponse()
    };

    return true;
}

static object CreateCachedBenchmarkMethodResponse(
    string id,
    string label,
    double psnrDb,
    double ssim,
    double mse,
    double rmse,
    double mae)
{
    return new
    {
        id,
        label,
        status = "complete",
        cached = true,
        output = (object?)null,
        metrics = CreateCachedMetricResponse(psnrDb, ssim, mse, rmse, mae),
        note = "Precomputed sample benchmark metric."
    };
}

static object CreateCachedComparisonResponse(
    string sampleImageDirectory,
    string fileName,
    string id,
    string label,
    int width,
    int height,
    object metrics,
    string message)
{
    var imagePath = Path.Combine(sampleImageDirectory, fileName);

    if (!File.Exists(imagePath))
    {
        throw new InvalidOperationException($"캐시 이미지 파일을 찾을 수 없습니다: {fileName}");
    }

    var imageBytes = File.ReadAllBytes(imagePath);

    return new
    {
        success = true,
        cached = true,
        id,
        label,
        output = new
        {
            targetWidth = width,
            targetHeight = height,
            resultImage = $"data:image/png;base64,{Convert.ToBase64String(imageBytes)}",
            metrics,
            message
        }
    };
}

static object CreateCachedMetricResponse(double psnrDb, double ssim, double mse, double rmse, double mae)
{
    return new
    {
        psnrDb,
        ssim,
        mse,
        rmse,
        mae
    };
}

static async Task<object> CreateSrganBenchmarkMethodResponse(
    IHttpClientFactory httpClientFactory,
    byte[] highResolutionImageBytes,
    byte[] lowResolutionImageBytes)
{
    try
    {
        var srganClient = httpClientFactory.CreateClient("sr-benchmark-api");
        var srganResult = await SrganInferenceClient.RequestSrganImage(srganClient, lowResolutionImageBytes);
        var srganMetrics = SuperResolutionBenchmarkTools.CalculateMetrics(
            highResolutionImageBytes,
            srganResult.PngBytes);

        return new
        {
            id = "srgan",
            label = "SRGAN",
            status = "complete",
            output = new
            {
                width = srganResult.Width,
                height = srganResult.Height,
                resultImage = srganResult.DataUrl,
                elapsedMs = srganResult.ElapsedMilliseconds,
                device = srganResult.Device
            },
            metrics = CreateMetricResponse(srganMetrics),
            note = $"SRGAN pretrained generator로 복원했습니다. device={srganResult.Device}, elapsed={srganResult.ElapsedMilliseconds}ms"
        };
    }
    catch (InvalidOperationException exception)
    {
        return new
        {
            id = "srgan",
            label = "SRGAN",
            status = "api-error",
            output = (object?)null,
            metrics = (object?)null,
            note = exception.Message
        };
    }
}

static async Task<object> CreateRealEsrganBenchmarkMethodResponse(
    IHttpClientFactory httpClientFactory,
    byte[] highResolutionImageBytes,
    byte[] lowResolutionImageBytes)
{
    try
    {
        var srganClient = httpClientFactory.CreateClient("sr-benchmark-api");
        var realEsrganResult = await SrganInferenceClient.RequestRealEsrganImage(srganClient, lowResolutionImageBytes);
        var realEsrganMetrics = SuperResolutionBenchmarkTools.CalculateMetrics(
            highResolutionImageBytes,
            realEsrganResult.PngBytes);

        return new
        {
            id = "real-esrgan",
            label = "Real-ESRGAN",
            status = "complete",
            output = new
            {
                width = realEsrganResult.Width,
                height = realEsrganResult.Height,
                resultImage = realEsrganResult.DataUrl,
                elapsedMs = realEsrganResult.ElapsedMilliseconds,
                device = realEsrganResult.Device
            },
            metrics = CreateMetricResponse(realEsrganMetrics),
            note = $"RealESRGAN_x4plus pretrained generator로 복원했습니다. device={realEsrganResult.Device}, elapsed={realEsrganResult.ElapsedMilliseconds}ms"
        };
    }
    catch (InvalidOperationException exception)
    {
        return new
        {
            id = "real-esrgan",
            label = "Real-ESRGAN",
            status = "api-error",
            output = (object?)null,
            metrics = (object?)null,
            note = exception.Message
        };
    }
}

static object CreateMetricResponse(SuperResolutionMetricResult metricResult)
{
    return new
    {
        psnrDb = metricResult.PsnrDb.HasValue ? Math.Round(metricResult.PsnrDb.Value, 4) : (double?)null,
        ssim = Math.Round(metricResult.Ssim, 6),
        mse = Math.Round(metricResult.Mse, 4),
        rmse = Math.Round(metricResult.Rmse, 4),
        mae = Math.Round(metricResult.Mae, 4)
    };
}

static object CreateSrganPaperReferenceResponse()
{
    return new
    {
        title = "SRGAN CVPR 2017 Table 2",
        scaleFactor = 4,
        sourceUrl = "https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf",
        rows = new[]
        {
            new { dataset = "Set5", method = "Bicubic", psnrDb = 28.43, ssim = 0.8211, mos = 1.97 },
            new { dataset = "Set5", method = "SRGAN-VGG54", psnrDb = 29.40, ssim = 0.8472, mos = 3.58 },
            new { dataset = "Set14", method = "Bicubic", psnrDb = 25.99, ssim = 0.7486, mos = 1.80 },
            new { dataset = "Set14", method = "SRGAN-VGG54", psnrDb = 26.02, ssim = 0.7397, mos = 3.72 },
            new { dataset = "BSD100", method = "Bicubic", psnrDb = 25.94, ssim = 0.6935, mos = 1.47 },
            new { dataset = "BSD100", method = "SRGAN-VGG54", psnrDb = 25.16, ssim = 0.6688, mos = 3.56 }
        }
    };
}

sealed record BicubicSampleImage(
    string Id,
    string Label,
    string Detail,
    string FileName,
    string ContentType,
    int Width,
    int Height,
    string SourceName,
    string SourceUrl);
