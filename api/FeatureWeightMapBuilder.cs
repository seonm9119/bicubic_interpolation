using ImageProcessing.Api;

namespace BicubicInterpolation.Api;

public sealed record FeatureWeightMap(int Width, int Height, byte[] Values);

public static class FeatureWeightMapBuilder
{
    public static FeatureWeightMap BuildFeatureWeightMap(byte[] imageBytes, int featureWeightPercent)
    {
        var featureMap = ImageProcessingPipeline.BuildFeatureMap(imageBytes, featureWeightPercent);
        var featureMapValues = Convert.FromBase64String(featureMap.FeatureMapBase64);

        if (featureMapValues.Length != featureMap.Width * featureMap.Height)
        {
            throw new InvalidOperationException("Image Processing feature map 크기가 입력 이미지 크기와 맞지 않습니다.");
        }

        return new FeatureWeightMap(featureMap.Width, featureMap.Height, featureMapValues);
    }
}
