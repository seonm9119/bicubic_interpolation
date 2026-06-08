using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BicubicInterpolation.Api;

public static class FeatureWeightedBicubicInterpolator
{
    private const int BackProjectionPassCount = 3;
    private const double MinimumCorrectionStrength = 0.16;
    private const double FeatureCorrectionStrength = 0.70;
    private const double ResidualMagnitudeCorrectionStrength = 0.20;
    private const double OrGateCorrectionStrength = 0.16;
    private const double ResidualCorrectionStrength = 1.05;
    private const double CorrectionPassDecay = 0.78;
    private const double BaseCorrectionLimit = 10.0;
    private const double ResidualCorrectionLimit = 0.70;

    public static BicubicImageResult ResizeToPngDataUrl(byte[] imageBytes, int scaleFactor, FeatureWeightMap featureWeightMap)
    {
        using var sourceImage = Image.Load<Rgba32>(imageBytes);
        var targetWidth = sourceImage.Width * scaleFactor;
        var targetHeight = sourceImage.Height * scaleFactor;

        BicubicResizeMath.ValidateOutputSize(targetWidth, targetHeight);
        ValidateFeatureWeightMap(sourceImage, featureWeightMap);

        using var targetImage = BuildClassicBicubicImage(sourceImage, targetWidth, targetHeight);
        ApplyFeatureAwareBackProjection(sourceImage, targetImage, featureWeightMap);

        using var outputStream = new MemoryStream();
        targetImage.SaveAsPng(outputStream);

        return new BicubicImageResult(targetWidth, targetHeight, outputStream.ToArray());
    }

    private static Image<Rgba32> BuildClassicBicubicImage(Image<Rgba32> sourceImage, int targetWidth, int targetHeight)
    {
        var targetImage = new Image<Rgba32>(targetWidth, targetHeight);

        for (var targetY = 0; targetY < targetHeight; targetY += 1)
        {
            var sourceY = BicubicResizeMath.GetSourcePosition(targetY, targetHeight, sourceImage.Height);
            var y2 = (int)sourceY;
            var y1 = BicubicResizeMath.ClampIndex(y2 - 1, sourceImage.Height);
            var y3 = BicubicResizeMath.ClampIndex(y2 + 1, sourceImage.Height);
            var y4 = BicubicResizeMath.ClampIndex(y2 + 2, sourceImage.Height);
            var yRatio = sourceY - y2;

            for (var targetX = 0; targetX < targetWidth; targetX += 1)
            {
                var sourceX = BicubicResizeMath.GetSourcePosition(targetX, targetWidth, sourceImage.Width);
                var x2 = (int)sourceX;
                var x1 = BicubicResizeMath.ClampIndex(x2 - 1, sourceImage.Width);
                var x3 = BicubicResizeMath.ClampIndex(x2 + 1, sourceImage.Width);
                var x4 = BicubicResizeMath.ClampIndex(x2 + 2, sourceImage.Width);
                var xRatio = sourceX - x2;
                var interpolatedPixel = BicubicPixelInterpolator.InterpolatePixelValues(
                    sourceImage,
                    x1,
                    x2,
                    x3,
                    x4,
                    y1,
                    y2,
                    y3,
                    y4,
                    xRatio,
                    yRatio);

                targetImage[targetX, targetY] = new Rgba32(
                    BicubicResizeMath.ClampChannel(interpolatedPixel.Red),
                    BicubicResizeMath.ClampChannel(interpolatedPixel.Green),
                    BicubicResizeMath.ClampChannel(interpolatedPixel.Blue),
                    BicubicResizeMath.ClampChannel(interpolatedPixel.Alpha));
            }
        }

        return targetImage;
    }

    private static void ApplyFeatureAwareBackProjection(
        Image<Rgba32> sourceImage,
        Image<Rgba32> targetImage,
        FeatureWeightMap featureWeightMap)
    {
        for (var passIndex = 0; passIndex < BackProjectionPassCount; passIndex += 1)
        {
            using var projectedSourceImage = targetImage.Clone(processingContext =>
                processingContext.Resize(new ResizeOptions
                {
                    Size = new Size(sourceImage.Width, sourceImage.Height),
                    Sampler = KnownResamplers.Bicubic
                }));
            var residualMap = BuildFeatureResidualMap(sourceImage, projectedSourceImage);

            ApplyResidualCorrection(targetImage, residualMap, featureWeightMap, passIndex);
        }
    }

    private static FeatureResidualMap BuildFeatureResidualMap(Image<Rgba32> sourceImage, Image<Rgba32> projectedSourceImage)
    {
        var residualValues = new BicubicInterpolatedPixel[sourceImage.Width * sourceImage.Height];
        var residualMagnitudes = new byte[sourceImage.Width * sourceImage.Height];

        for (var y = 0; y < sourceImage.Height; y += 1)
        {
            for (var x = 0; x < sourceImage.Width; x += 1)
            {
                var sourcePixel = sourceImage[x, y];
                var projectedPixel = projectedSourceImage[x, y];
                var redResidual = sourcePixel.R - projectedPixel.R;
                var greenResidual = sourcePixel.G - projectedPixel.G;
                var blueResidual = sourcePixel.B - projectedPixel.B;

                residualValues[y * sourceImage.Width + x] = new BicubicInterpolatedPixel(
                    redResidual,
                    greenResidual,
                    blueResidual,
                    sourcePixel.A - projectedPixel.A);
                residualMagnitudes[y * sourceImage.Width + x] = BicubicResizeMath.ClampChannel(
                    (Math.Abs(redResidual) + Math.Abs(greenResidual) + Math.Abs(blueResidual)) / 3.0);
            }
        }

        return new FeatureResidualMap(sourceImage.Width, sourceImage.Height, residualValues, residualMagnitudes);
    }

    private static void ApplyResidualCorrection(
        Image<Rgba32> targetImage,
        FeatureResidualMap residualMap,
        FeatureWeightMap featureWeightMap,
        int passIndex)
    {
        for (var targetY = 0; targetY < targetImage.Height; targetY += 1)
        {
            var sourceY = BicubicResizeMath.GetSourcePosition(targetY, targetImage.Height, featureWeightMap.Height);

            for (var targetX = 0; targetX < targetImage.Width; targetX += 1)
            {
                var sourceX = BicubicResizeMath.GetSourcePosition(targetX, targetImage.Width, featureWeightMap.Width);
                var residual = InterpolateResidual(residualMap, sourceX, sourceY);
                var featureWeight = InterpolateFeatureWeight(featureWeightMap, sourceX, sourceY);
                var residualMagnitude = BicubicResizeMath.ClampChannel(InterpolateResidualMagnitude(residualMap, sourceX, sourceY));
                var correctionStrength = CalculateCorrectionStrength(featureWeight, residualMagnitude, passIndex);
                var correctionLimit = CalculateCorrectionLimit(residualMagnitude);
                var targetPixel = targetImage[targetX, targetY];

                targetImage[targetX, targetY] = new Rgba32(
                    ApplyBoundedCorrection(targetPixel.R, residual.Red, correctionStrength, correctionLimit),
                    ApplyBoundedCorrection(targetPixel.G, residual.Green, correctionStrength, correctionLimit),
                    ApplyBoundedCorrection(targetPixel.B, residual.Blue, correctionStrength, correctionLimit),
                    targetPixel.A);
            }
        }
    }

    private static BicubicInterpolatedPixel InterpolateResidual(
        FeatureResidualMap residualMap,
        double sourceX,
        double sourceY)
    {
        var x1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceX), residualMap.Width);
        var y1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceY), residualMap.Height);
        var x2 = BicubicResizeMath.ClampIndex(x1 + 1, residualMap.Width);
        var y2 = BicubicResizeMath.ClampIndex(y1 + 1, residualMap.Height);
        var xRatio = sourceX - Math.Floor(sourceX);
        var yRatio = sourceY - Math.Floor(sourceY);
        var topLeft = residualMap.Values[y1 * residualMap.Width + x1];
        var topRight = residualMap.Values[y1 * residualMap.Width + x2];
        var bottomLeft = residualMap.Values[y2 * residualMap.Width + x1];
        var bottomRight = residualMap.Values[y2 * residualMap.Width + x2];

        return new BicubicInterpolatedPixel(
            BilinearInterpolate(topLeft.Red, topRight.Red, bottomLeft.Red, bottomRight.Red, xRatio, yRatio),
            BilinearInterpolate(topLeft.Green, topRight.Green, bottomLeft.Green, bottomRight.Green, xRatio, yRatio),
            BilinearInterpolate(topLeft.Blue, topRight.Blue, bottomLeft.Blue, bottomRight.Blue, xRatio, yRatio),
            BilinearInterpolate(topLeft.Alpha, topRight.Alpha, bottomLeft.Alpha, bottomRight.Alpha, xRatio, yRatio));
    }

    private static double InterpolateResidualMagnitude(FeatureResidualMap residualMap, double sourceX, double sourceY)
    {
        var x1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceX), residualMap.Width);
        var y1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceY), residualMap.Height);
        var x2 = BicubicResizeMath.ClampIndex(x1 + 1, residualMap.Width);
        var y2 = BicubicResizeMath.ClampIndex(y1 + 1, residualMap.Height);
        var xRatio = sourceX - Math.Floor(sourceX);
        var yRatio = sourceY - Math.Floor(sourceY);
        var topLeft = residualMap.Magnitudes[y1 * residualMap.Width + x1];
        var topRight = residualMap.Magnitudes[y1 * residualMap.Width + x2];
        var bottomLeft = residualMap.Magnitudes[y2 * residualMap.Width + x1];
        var bottomRight = residualMap.Magnitudes[y2 * residualMap.Width + x2];

        return BilinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, xRatio, yRatio);
    }

    private static double InterpolateFeatureWeight(FeatureWeightMap featureWeightMap, double sourceX, double sourceY)
    {
        var x1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceX), featureWeightMap.Width);
        var y1 = BicubicResizeMath.ClampIndex((int)Math.Floor(sourceY), featureWeightMap.Height);
        var x2 = BicubicResizeMath.ClampIndex(x1 + 1, featureWeightMap.Width);
        var y2 = BicubicResizeMath.ClampIndex(y1 + 1, featureWeightMap.Height);
        var xRatio = sourceX - Math.Floor(sourceX);
        var yRatio = sourceY - Math.Floor(sourceY);
        var topLeft = featureWeightMap.Values[y1 * featureWeightMap.Width + x1];
        var topRight = featureWeightMap.Values[y1 * featureWeightMap.Width + x2];
        var bottomLeft = featureWeightMap.Values[y2 * featureWeightMap.Width + x1];
        var bottomRight = featureWeightMap.Values[y2 * featureWeightMap.Width + x2];

        return BilinearInterpolate(topLeft, topRight, bottomLeft, bottomRight, xRatio, yRatio);
    }

    private static double CalculateCorrectionStrength(double featureWeight, byte residualMagnitude, int passIndex)
    {
        var combinedFeatureGate = BicubicResizeMath.ClampChannel(featureWeight) | residualMagnitude;
        var featureStrength = Math.Clamp(featureWeight / 64.0, 0, 1);
        var residualMagnitudeStrength = residualMagnitude / 255.0;
        var passStrength = Math.Pow(CorrectionPassDecay, passIndex);

        return (MinimumCorrectionStrength +
            FeatureCorrectionStrength * featureStrength +
            ResidualMagnitudeCorrectionStrength * residualMagnitudeStrength +
            OrGateCorrectionStrength * (combinedFeatureGate / 255.0)) *
            ResidualCorrectionStrength *
            passStrength;
    }

    private static double CalculateCorrectionLimit(byte residualMagnitude)
    {
        return BaseCorrectionLimit + ResidualCorrectionLimit * residualMagnitude;
    }

    private static byte ApplyBoundedCorrection(byte channelValue, double residual, double correctionStrength, double correctionLimit)
    {
        var boundedCorrection = Math.Clamp(residual * correctionStrength, -correctionLimit, correctionLimit);

        return BicubicResizeMath.ClampChannel(channelValue + boundedCorrection);
    }

    private static double BilinearInterpolate(double topLeft, double topRight, double bottomLeft, double bottomRight, double xRatio, double yRatio)
    {
        var topValue = topLeft + (topRight - topLeft) * xRatio;
        var bottomValue = bottomLeft + (bottomRight - bottomLeft) * xRatio;

        return topValue + (bottomValue - topValue) * yRatio;
    }

    private static void ValidateFeatureWeightMap(Image<Rgba32> sourceImage, FeatureWeightMap featureWeightMap)
    {
        if (featureWeightMap.Width != sourceImage.Width || featureWeightMap.Height != sourceImage.Height)
        {
            throw new InvalidOperationException("Feature map 크기가 입력 이미지 크기와 맞지 않습니다.");
        }
    }

    private sealed record FeatureResidualMap(
        int Width,
        int Height,
        BicubicInterpolatedPixel[] Values,
        byte[] Magnitudes);
}
