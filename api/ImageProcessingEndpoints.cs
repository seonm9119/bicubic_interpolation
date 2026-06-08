using System.Security.Cryptography;
using ImageProcessing.Api;
using Microsoft.AspNetCore.Http;

namespace BicubicInterpolation.Api;

public static class ImageProcessingEndpoints
{
    public static void MapImageProcessingEndpoints(this WebApplication app)
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
    }

    private static async Task<IResult> ApplyImageProcessingAsync(HttpRequest request)
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

        try
        {
            var processedImage = ImageProcessingPipeline.ProcessToPngDataUrl(imageBytes, processingRequest);
            var imageHash = Convert.ToHexString(SHA256.HashData(imageBytes)).ToLowerInvariant();

            return Results.Ok(new
            {
                success = true,
                input = new
                {
                    fileName = uploadedImage.FileName,
                    contentType = uploadedImage.ContentType,
                    size = uploadedImage.Length,
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
                    resultImage = processedImage.DataUrl,
                    message = $"{processedImage.OperationName} 결과 이미지를 생성했습니다."
                }
            });
        }
        catch (InvalidOperationException exception)
        {
            return CreateBadRequest(exception.Message);
        }
    }

    private static async Task<IResult> CreateFeatureMapAsync(HttpRequest request)
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
        var featureWeightPercent = ReadBoundedInt(formFields["featureWeightPercent"], 100, 0, 100);

        try
        {
            var featureMap = ImageProcessingPipeline.BuildFeatureMap(imageBytes, featureWeightPercent);

            return Results.Ok(new
            {
                success = true,
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

    private static ImageProcessingRequest ReadImageProcessingRequest(IFormCollection formFields)
    {
        return new ImageProcessingRequest(
            NormalizeOperation(formFields["operation"]),
            ReadBoundedInt(formFields["edgeThresholdPercent"], 24, 0, 100),
            ReadBoundedInt(formFields["sobelGainPercent"], 68, 0, 100),
            NormalizeKernelSize(ReadBoundedInt(formFields["sobelKernelSize"], 3, 3, 7)),
            ReadBoundedInt(formFields["gaussianRadius"], 2, 1, 6),
            ReadBoundedInt(formFields["gaussianSigmaPercent"], 55, 10, 100),
            ReadBoundedInt(formFields["logKernelStrengthPercent"], 70, 0, 100),
            ReadBoundedInt(formFields["fuzzyStrengthPercent"], 72, 0, 100),
            ReadBoundedInt(formFields["fuzzyCutPercent"], 50, 0, 100),
            ReadBoundedInt(formFields["fuzzyMidpointPercent"], 54, 0, 100));
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
