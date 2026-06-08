using System.Security.Cryptography;
using System.Text;
using ImageProcessing.Api;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Memory;

namespace BicubicInterpolation.Api;

public static class ImageProcessingEndpoints
{
    private const string DefaultProcessingSampleFileName = "lenna-test.png";
    private const string DefaultProcessingSampleContentType = "image/png";

    public static void MapImageProcessingEndpoints(this WebApplication app, string defaultImageDirectory)
    {
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

        app.MapPost("/api/image-processing/apply", ApplyImageProcessingAsync);
        app.MapPost("/api/image-processing/feature-map", CreateFeatureMapAsync);
        app.MapGet("/api/image-processing/default-preview", (
            HttpRequest request,
            IMemoryCache memoryCache,
            HttpResponse response) => CreateDefaultImageProcessingPreviewAsync(request, memoryCache, response, defaultImageDirectory));
        app.MapGet("/api/image-processing/cached-images/{cacheId}.png", GetCachedProcessedImage);
    }

    private static async Task<IResult> ApplyImageProcessingAsync(HttpRequest request, IMemoryCache memoryCache)
    {
        if (!request.HasFormContentType)
        {
            return CreateBadRequest("multipart/form-data 요청이 필요합니다.");
        }

        var formFields = await request.ReadFormAsync();
        var uploadedImage = formFields.Files.GetFile("image");

        if (uploadedImage is null || uploadedImage.Length == 0)
        {
            return CreateBadRequest("image 파일이 비어 있습니다.");
        }

        var imageBytes = await ReadImageBytesAsync(uploadedImage);
        var processingRequest = ReadImageProcessingRequest(formFields);
        var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();

        try
        {
            var processedImage = ReadOrCreateProcessedImage(
                imageBytes,
                processingRequest,
                imageHash,
                memoryCache,
                out var isCached,
                out var processedImageId);

            return Results.Ok(CreateImageProcessingResponse(
                uploadedImage.FileName,
                uploadedImage.ContentType,
                uploadedImage.Length,
                imageHash,
                processingRequest,
                processedImage,
                processedImageId,
                isCached));
        }
        catch (InvalidOperationException exception)
        {
            return CreateBadRequest(exception.Message);
        }
    }

    private static async Task<IResult> CreateDefaultImageProcessingPreviewAsync(
        HttpRequest request,
        IMemoryCache memoryCache,
        HttpResponse response,
        string defaultImageDirectory)
    {
        var defaultImagePath = Path.Combine(defaultImageDirectory, DefaultProcessingSampleFileName);

        if (!File.Exists(defaultImagePath))
        {
            return Results.NotFound(new
            {
                success = false,
                error = "기본 image-processing sample이 컨테이너에 마운트되지 않았습니다."
            });
        }

        var imageBytes = await File.ReadAllBytesAsync(defaultImagePath);
        var processingRequest = ReadImageProcessingRequest(request.Query);
        var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();

        try
        {
            var processedImage = ReadOrCreateProcessedImage(
                imageBytes,
                processingRequest,
                imageHash,
                memoryCache,
                out var isCached,
                out var processedImageId);

            ApiCacheTools.SetPublicCacheHeaders(response);

            return Results.Ok(CreateImageProcessingResponse(
                DefaultProcessingSampleFileName,
                DefaultProcessingSampleContentType,
                imageBytes.Length,
                imageHash,
                processingRequest,
                processedImage,
                processedImageId,
                isCached));
        }
        catch (InvalidOperationException exception)
        {
            return CreateBadRequest(exception.Message);
        }
    }

    private static async Task<IResult> CreateFeatureMapAsync(HttpRequest request, IMemoryCache memoryCache)
    {
        if (!request.HasFormContentType)
        {
            return CreateBadRequest("multipart/form-data 요청이 필요합니다.");
        }

        var formFields = await request.ReadFormAsync();
        var uploadedImage = formFields.Files.GetFile("image");

        if (uploadedImage is null || uploadedImage.Length == 0)
        {
            return CreateBadRequest("image 파일이 비어 있습니다.");
        }

        var imageBytes = await ReadImageBytesAsync(uploadedImage);
        var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();
        var featureWeightPercent = ReadBoundedInt(formFields["featureWeightPercent"], 100, 0, 100);
        var featureMapCacheKey = $"image-processing:feature-map:{imageHash}:{featureWeightPercent}";
        var isCached = true;

        try
        {
            if (!memoryCache.TryGetValue(featureMapCacheKey, out FeatureMapResult? featureMap) ||
                featureMap is null)
            {
                isCached = false;
                featureMap = ImageProcessingPipeline.BuildFeatureMap(imageBytes, featureWeightPercent);
                memoryCache.Set(featureMapCacheKey, featureMap, ApiCacheTools.CreateComputeCacheOptions());
            }

            return Results.Ok(new
            {
                success = true,
                cached = isCached,
                request = new
                {
                    featureWeightPercent
                },
                output = new
                {
                    width = featureMap.Width,
                    height = featureMap.Height,
                    encoding = featureMap.Encoding,
                    featureMapBase64 = featureMap.FeatureMapBase64,
                    message = "Feature weighted bicubic용 feature map을 생성했습니다."
                }
            });
        }
        catch (InvalidOperationException exception)
        {
            return CreateBadRequest(exception.Message);
        }
    }

    private static IResult GetCachedProcessedImage(string cacheId, IMemoryCache memoryCache, HttpResponse response)
    {
        if (!IsCacheId(cacheId))
        {
            return Results.NotFound(new
            {
                success = false,
                error = "요청한 image-processing cache를 찾을 수 없습니다."
            });
        }

        if (!memoryCache.TryGetValue(CreateProcessedImageCacheKey(cacheId), out ProcessedImageResult? processedImage) ||
            processedImage is null)
        {
            return Results.NotFound(new
            {
                success = false,
                error = "image-processing cache가 만료되었거나 생성되지 않았습니다."
            });
        }

        ApiCacheTools.SetPublicCacheHeaders(response);

        return Results.File(processedImage.PngBytes, "image/png", enableRangeProcessing: true);
    }

    private static ProcessedImageResult ReadOrCreateProcessedImage(
        byte[] imageBytes,
        ImageProcessingRequest processingRequest,
        string imageHash,
        IMemoryCache memoryCache,
        out bool isCached,
        out string processedImageId)
    {
        var processingCacheKey = CreateImageProcessingCacheKey(imageHash, processingRequest);
        processedImageId = CreateCacheId(processingCacheKey);
        isCached = true;

        if (!memoryCache.TryGetValue(processingCacheKey, out ProcessedImageResult? processedImage) ||
            processedImage is null)
        {
            isCached = false;
            processedImage = ImageProcessingPipeline.ProcessToPng(imageBytes, processingRequest);
            memoryCache.Set(processingCacheKey, processedImage, ApiCacheTools.CreateComputeCacheOptions());
        }

        memoryCache.Set(
            CreateProcessedImageCacheKey(processedImageId),
            processedImage,
            ApiCacheTools.CreateComputeCacheOptions());

        return processedImage;
    }

    private static object CreateImageProcessingResponse(
        string fileName,
        string contentType,
        long imageSize,
        string imageHash,
        ImageProcessingRequest processingRequest,
        ProcessedImageResult processedImage,
        string processedImageId,
        bool isCached)
    {
        return new
        {
            success = true,
            cached = isCached,
            input = new
            {
                fileName,
                contentType,
                size = imageSize,
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
                width = processedImage.Width,
                height = processedImage.Height,
                operationName = processedImage.OperationName,
                resultImage = $"/api/image-processing/cached-images/{processedImageId}.png",
                message = $"{processedImage.OperationName} 결과 이미지를 생성했습니다."
            }
        };
    }

    private static string CreateImageProcessingCacheKey(string imageHash, ImageProcessingRequest processingRequest)
    {
        return string.Join(
            ":",
            "image-processing",
            "apply",
            imageHash,
            processingRequest.Operation,
            processingRequest.EdgeThresholdPercent,
            processingRequest.SobelGainPercent,
            processingRequest.SobelKernelSize,
            processingRequest.GaussianRadius,
            processingRequest.GaussianSigmaPercent,
            processingRequest.LogKernelStrengthPercent,
            processingRequest.FuzzyStrengthPercent,
            processingRequest.FuzzyCutPercent,
            processingRequest.FuzzyMidpointPercent);
    }

    private static string CreateCacheId(string cacheKey)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(cacheKey))).ToLowerInvariant();
    }

    private static string CreateProcessedImageCacheKey(string cacheId)
    {
        return $"image-processing:cached-image:{cacheId}";
    }

    private static bool IsCacheId(string cacheId)
    {
        return cacheId.Length == 64 && cacheId.All(Uri.IsHexDigit);
    }

    private static ImageProcessingRequest ReadImageProcessingRequest(IFormCollection formFields)
    {
        return ReadImageProcessingRequest(fieldName => formFields[fieldName]);
    }

    private static ImageProcessingRequest ReadImageProcessingRequest(IQueryCollection queryFields)
    {
        return ReadImageProcessingRequest(fieldName => queryFields[fieldName]);
    }

    private static ImageProcessingRequest ReadImageProcessingRequest(Func<string, string?> readField)
    {
        return new ImageProcessingRequest(
            NormalizeOperation(readField("operation")),
            ReadBoundedInt(readField("edgeThresholdPercent"), 24, 0, 100),
            ReadBoundedInt(readField("sobelGainPercent"), 68, 0, 100),
            NormalizeKernelSize(ReadBoundedInt(readField("sobelKernelSize"), 3, 3, 7)),
            ReadBoundedInt(readField("gaussianRadius"), 2, 1, 6),
            ReadBoundedInt(readField("gaussianSigmaPercent"), 55, 10, 100),
            ReadBoundedInt(readField("logKernelStrengthPercent"), 70, 0, 100),
            ReadBoundedInt(readField("fuzzyStrengthPercent"), 72, 0, 100),
            ReadBoundedInt(readField("fuzzyCutPercent"), 50, 0, 100),
            ReadBoundedInt(readField("fuzzyMidpointPercent"), 54, 0, 100));
    }

    private static async Task<byte[]> ReadImageBytesAsync(IFormFile uploadedImage)
    {
        await using var imageStream = uploadedImage.OpenReadStream();
        using var memoryStream = new MemoryStream();
        await imageStream.CopyToAsync(memoryStream);

        return memoryStream.ToArray();
    }

    private static IResult CreateBadRequest(string errorMessage)
    {
        return Results.BadRequest(new
        {
            success = false,
            error = errorMessage
        });
    }

    private static int ReadBoundedInt(string? rawNumber, int defaultNumber, int minimumNumber, int maximumNumber)
    {
        if (!int.TryParse(rawNumber, out var parsedNumber))
        {
            return defaultNumber;
        }

        return Math.Clamp(parsedNumber, minimumNumber, maximumNumber);
    }

    private static int NormalizeKernelSize(int kernelSize)
    {
        return kernelSize % 2 == 1 ? kernelSize : kernelSize + 1;
    }

    private static string NormalizeOperation(string? rawOperation)
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
}
