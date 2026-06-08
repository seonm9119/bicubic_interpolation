using System.Net.Http.Headers;
using System.Text.Json;

namespace BicubicInterpolation.Api;

public sealed record SrganInferenceResult(
    int Width,
    int Height,
    byte[] PngBytes,
    string DataUrl,
    double ElapsedMilliseconds,
    string Device,
    string Message);

public static class SrganInferenceClient
{
    public static async Task<SrganInferenceResult> RequestSrganImage(HttpClient httpClient, byte[] lowResolutionImageBytes)
    {
        return await RequestUpscaledImage(
            httpClient,
            lowResolutionImageBytes,
            "/api/srgan/upscale",
            "SRGAN",
            "SRGAN x4 결과 이미지를 생성했습니다.");
    }

    public static async Task<SrganInferenceResult> RequestRealEsrganImage(HttpClient httpClient, byte[] lowResolutionImageBytes)
    {
        return await RequestUpscaledImage(
            httpClient,
            lowResolutionImageBytes,
            "/api/realesrgan/upscale",
            "Real-ESRGAN",
            "Real-ESRGAN x4 결과 이미지를 생성했습니다.");
    }

    private static async Task<SrganInferenceResult> RequestUpscaledImage(
        HttpClient httpClient,
        byte[] lowResolutionImageBytes,
        string endpointPath,
        string modelName,
        string defaultMessage)
    {
        using var formContent = new MultipartFormDataContent();
        using var imageContent = new ByteArrayContent(lowResolutionImageBytes);
        imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/png");

        formContent.Add(imageContent, "image", "low-resolution.png");

        using var response = await httpClient.PostAsync(endpointPath, formContent);
        var responseText = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(ReadApiErrorMessage(modelName, responseText));
        }

        using var jsonDocument = JsonDocument.Parse(responseText);
        var rootElement = jsonDocument.RootElement;

        if (rootElement.TryGetProperty("success", out var successElement) && !successElement.GetBoolean())
        {
            throw new InvalidOperationException($"{modelName} API 요청에 실패했습니다.");
        }

        var outputElement = rootElement.GetProperty("output");
        var modelElement = rootElement.GetProperty("model");
        var width = outputElement.GetProperty("targetWidth").GetInt32();
        var height = outputElement.GetProperty("targetHeight").GetInt32();
        var elapsedMilliseconds = outputElement.GetProperty("elapsedMs").GetDouble();
        var resultImageDataUrl = outputElement.GetProperty("resultImage").GetString() ?? string.Empty;
        var message = outputElement.GetProperty("message").GetString() ?? defaultMessage;
        var device = modelElement.GetProperty("device").GetString() ?? "unknown";
        var resultImageBytes = DecodePngDataUrl(resultImageDataUrl);

        return new SrganInferenceResult(
            width,
            height,
            resultImageBytes,
            resultImageDataUrl,
            elapsedMilliseconds,
            device,
            message);
    }

    private static byte[] DecodePngDataUrl(string dataUrl)
    {
        var base64StartIndex = dataUrl.IndexOf(',', StringComparison.Ordinal);

        if (base64StartIndex < 0 || base64StartIndex == dataUrl.Length - 1)
        {
            throw new InvalidOperationException("SRGAN API 결과 이미지 형식이 올바르지 않습니다.");
        }

        return Convert.FromBase64String(dataUrl[(base64StartIndex + 1)..]);
    }

    private static string ReadApiErrorMessage(string modelName, string responseText)
    {
        if (string.IsNullOrWhiteSpace(responseText))
        {
            return $"{modelName} API 요청에 실패했습니다.";
        }

        try
        {
            using var jsonDocument = JsonDocument.Parse(responseText);
            var rootElement = jsonDocument.RootElement;

            if (rootElement.TryGetProperty("detail", out var detailElement))
            {
                return detailElement.ValueKind == JsonValueKind.String
                    ? detailElement.GetString() ?? $"{modelName} API 요청에 실패했습니다."
                    : detailElement.ToString();
            }

            if (rootElement.TryGetProperty("error", out var errorElement))
            {
                return errorElement.ValueKind == JsonValueKind.String
                    ? errorElement.GetString() ?? $"{modelName} API 요청에 실패했습니다."
                    : errorElement.ToString();
            }
        }
        catch (JsonException)
        {
            return $"{modelName} API 요청에 실패했습니다. {responseText}";
        }

        return $"{modelName} API 요청에 실패했습니다. {responseText}";
    }
}
