using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;

namespace BicubicInterpolation.Api;

public sealed record FeatureWeightMap(int Width, int Height, byte[] Values);

public static class FeatureWeightMapClient
{
    public static async Task<FeatureWeightMap> RequestFeatureWeightMap(
        HttpClient httpClient,
        byte[] imageBytes,
        int featureWeightPercent)
    {
        using var formContent = new MultipartFormDataContent();
        using var imageContent = new ByteArrayContent(imageBytes);
        imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("application/octet-stream");

        formContent.Add(imageContent, "image", "source-image");
        formContent.Add(new StringContent(featureWeightPercent.ToString(CultureInfo.InvariantCulture)), "featureWeightPercent");

        using var response = await httpClient.PostAsync("/api/image-processing/feature-map", formContent);
        var responseText = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"Image Processing API feature map 요청에 실패했습니다. {responseText}");
        }

        using var jsonDocument = JsonDocument.Parse(responseText);
        var rootElement = jsonDocument.RootElement;

        if (rootElement.TryGetProperty("success", out var successElement) && !successElement.GetBoolean())
        {
            var errorMessage = rootElement.TryGetProperty("error", out var errorElement)
                ? errorElement.GetString()
                : "Image Processing API feature map 요청에 실패했습니다.";

            throw new InvalidOperationException(errorMessage);
        }

        var outputElement = rootElement.GetProperty("output");
        var width = outputElement.GetProperty("width").GetInt32();
        var height = outputElement.GetProperty("height").GetInt32();
        var featureMapBase64 = outputElement.GetProperty("featureMapBase64").GetString() ?? string.Empty;
        var featureMapValues = Convert.FromBase64String(featureMapBase64);

        if (featureMapValues.Length != width * height)
        {
            throw new InvalidOperationException("Image Processing API feature map 크기가 입력 이미지 크기와 맞지 않습니다.");
        }

        return new FeatureWeightMap(width, height, featureMapValues);
    }
}
