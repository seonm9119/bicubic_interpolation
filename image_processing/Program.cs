using System.Security.Cryptography;
using ImageProcessing.Api;
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

var app = builder.Build();

app.MapGet("/api/image-processing/health", () =>
{
    return Results.Ok(new
    {
        success = true,
        status = "ok",
        service = "image-processing-api",
        language = "C#",
        mode = "container"
    });
});

app.MapPost("/api/image-processing/apply", async (HttpRequest request) =>
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

    var imageBytes = await ReadImageBytes(imageFile);
    var processingRequest = new ImageProcessingRequest(
        NormalizeOperation(form["operation"]),
        ReadBoundedInt(form["edgeThresholdPercent"], 24, 0, 100),
        ReadBoundedInt(form["sobelGainPercent"], 68, 0, 100),
        NormalizeKernelSize(ReadBoundedInt(form["sobelKernelSize"], 3, 3, 7)),
        ReadBoundedInt(form["gaussianRadius"], 2, 1, 6),
        ReadBoundedInt(form["gaussianSigmaPercent"], 55, 10, 100),
        ReadBoundedInt(form["logKernelStrengthPercent"], 70, 0, 100),
        ReadBoundedInt(form["fuzzyStrengthPercent"], 72, 0, 100),
        ReadBoundedInt(form["fuzzyCutPercent"], 50, 0, 100),
        ReadBoundedInt(form["fuzzyMidpointPercent"], 54, 0, 100));

    try
    {
        var processedImageResult = ImageProcessingPipeline.ProcessToPngDataUrl(imageBytes, processingRequest);
        var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();

        return Results.Ok(new
        {
            success = true,
            input = new
            {
                fileName = imageFile.FileName,
                contentType = imageFile.ContentType,
                size = imageFile.Length,
                sha256 = imageHash
            },
            request = new
            {
                operation = processingRequest.Operation,
                edgeThresholdPercent = processingRequest.EdgeThresholdPercent,
                sobelGainPercent = processingRequest.SobelGainPercent,
                sobelKernelSize = processingRequest.SobelKernelSize,
                gaussianRadius = processingRequest.GaussianRadius,
                gaussianSigmaPercent = processingRequest.GaussianSigmaPercent,
                logKernelStrengthPercent = processingRequest.LogKernelStrengthPercent,
                fuzzyStrengthPercent = processingRequest.FuzzyStrengthPercent,
                fuzzyCutPercent = processingRequest.FuzzyCutPercent,
                fuzzyMidpointPercent = processingRequest.FuzzyMidpointPercent
            },
            output = new
            {
                width = processedImageResult.Width,
                height = processedImageResult.Height,
                operationName = processedImageResult.OperationName,
                resultImage = processedImageResult.DataUrl,
                message = $"{processedImageResult.OperationName} 결과 이미지를 생성했습니다."
            }
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

app.MapPost("/api/image-processing/feature-map", async (HttpRequest request) =>
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

    var imageBytes = await ReadImageBytes(imageFile);
    var featureWeightPercent = ReadBoundedInt(form["featureWeightPercent"], 100, 0, 100);

    try
    {
        var featureMapResult = ImageProcessingPipeline.BuildFeatureMap(imageBytes, featureWeightPercent);

        return Results.Ok(new
        {
            success = true,
            request = new
            {
                featureWeightPercent
            },
            output = new
            {
                width = featureMapResult.Width,
                height = featureMapResult.Height,
                encoding = featureMapResult.Encoding,
                featureMapBase64 = featureMapResult.FeatureMapBase64,
                message = "Feature weighted bicubic용 feature map을 생성했습니다."
            }
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

app.Run();

static async Task<byte[]> ReadImageBytes(IFormFile imageFile)
{
    await using var imageStream = imageFile.OpenReadStream();
    using var memoryStream = new MemoryStream();
    await imageStream.CopyToAsync(memoryStream);

    return memoryStream.ToArray();
}

static int ReadBoundedInt(string? rawValue, int defaultValue, int minValue, int maxValue)
{
    if (!int.TryParse(rawValue, out var parsedValue))
    {
        return defaultValue;
    }

    return Math.Clamp(parsedValue, minValue, maxValue);
}

static int NormalizeKernelSize(int kernelSize)
{
    return kernelSize % 2 == 1 ? kernelSize : kernelSize + 1;
}

static string NormalizeOperation(string? rawOperation)
{
    return rawOperation switch
    {
        "gaussian-blur" => "gaussian-blur",
        "log-sharpening" => "log-sharpening",
        "fuzzy-stretching" => "fuzzy-stretching",
        "absolute-edge" => "absolute-edge",
        "absolute-log" => "absolute-log",
        "texture-density" => "texture-density",
        "local-contrast" => "local-contrast",
        "corner-gate" => "corner-gate",
        "critical-feature-fusion" => "critical-feature-fusion",
        _ => "sobel-edge"
    };
}
