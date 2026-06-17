using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ImageProcessing.Api;

public static class ImageProcessingPipeline
{
    private const double CriticalFeatureScale = 3.5;
    private const int CriticalFeatureLimit = 96;

    public static ProcessedImageResult ProcessToPng(byte[] imageBytes, ImageProcessingRequest processingRequest)
    {
        using var sourceImage = Image.Load<Rgba32>(imageBytes);
        var grayMap = BuildGrayMap(sourceImage);

        using var outputImage = processingRequest.Operation switch
        {
            "gaussian-blur" => BuildGrayImage(BuildGaussianBlurMap(grayMap, processingRequest)),
            "log-sharpening" => BuildLogSharpenedImage(sourceImage, grayMap, processingRequest),
            "fuzzy-stretching" => BuildGrayImage(BuildFuzzyStretchingMap(grayMap, processingRequest)),
            "binary-threshold" => BuildGrayImage(BuildBinaryThresholdMap(grayMap, processingRequest)),
            "absolute-edge" => BuildGrayImage(BuildAbsoluteSobelEdgeMap(grayMap)),
            "absolute-log" => BuildGrayImage(BuildAbsoluteLogMap(grayMap)),
            "texture-density" => BuildGrayImage(BuildTextureDensityMap(grayMap)),
            "local-contrast" => BuildGrayImage(BuildLocalContrastMap(grayMap)),
            "corner-gate" => BuildGrayImage(BuildCornerGateMap(
                BuildAbsoluteVerticalSobelMap(grayMap),
                BuildAbsoluteHorizontalSobelMap(grayMap))),
            "critical-feature-fusion" => BuildGrayImage(NormalizeMap(BuildCriticalFeatureWeightMap(grayMap))),
            _ => BuildGrayImage(BuildSobelEdgeMap(grayMap, processingRequest))
        };

        using var outputStream = new MemoryStream();
        outputImage.SaveAsPng(outputStream);
        var resultImageBytes = outputStream.ToArray();
        var operationName = GetOperationName(processingRequest);

        return new ProcessedImageResult(sourceImage.Width, sourceImage.Height, resultImageBytes, operationName);
    }

    public static FeatureMapResult BuildFeatureMap(byte[] imageBytes, int featureWeightPercent)
    {
        using var sourceImage = Image.Load<Rgba32>(imageBytes);
        var featureWeightMap = BuildFeatureWeightMap(sourceImage, featureWeightPercent);
        var featureMapBytes = ConvertMapToRowMajorBytes(featureWeightMap, sourceImage.Width, sourceImage.Height);

        return new FeatureMapResult(
            sourceImage.Width,
            sourceImage.Height,
            Convert.ToBase64String(featureMapBytes),
            "uint8-row-major");
    }

    private static string GetOperationName(ImageProcessingRequest processingRequest)
    {
        var operationName = processingRequest.Operation switch
        {
            "gaussian-blur" => "Gaussian Blur",
            "log-sharpening" => "LoG Sharpening",
            "fuzzy-stretching" => "Fuzzy Stretching",
            "binary-threshold" => "Binary Threshold",
            "absolute-edge" => "Absolute Edge Map",
            "absolute-log" => "Absolute LoG Detail",
            "texture-density" => "Texture Density",
            "local-contrast" => "Local Contrast",
            "corner-gate" => "Corner Gate",
            "critical-feature-fusion" => "Critical Feature Fusion",
            _ => "Sobel Edge Detection"
        };
        return operationName;
    }

    private static int[,] BuildBinaryThresholdMap(int[,] grayMap, ImageProcessingRequest processingRequest)
    {
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var binaryMap = new int[width, height];
        var threshold = processingRequest.ThresholdMode switch
        {
            "mean" => CalculateMeanThreshold(grayMap),
            "max-min" => (FindMaximum(grayMap) + FindMinimum(grayMap)) / 2,
            _ => processingRequest.BinaryThresholdPercent * 255 / 100
        };

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                binaryMap[x, y] = grayMap[x, y] > threshold ? 255 : 0;
            }
        }

        return binaryMap;
    }

    private static int[,] BuildFeatureWeightMap(Image<Rgba32> sourceImage, int featureWeightPercent)
    {
        var grayMap = BuildGrayMap(sourceImage);
        var criticalFeatureWeightMap = BuildCriticalFeatureWeightMap(grayMap);
        var width = sourceImage.Width;
        var height = sourceImage.Height;
        var featureWeightMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                featureWeightMap[x, y] = criticalFeatureWeightMap[x, y] * featureWeightPercent / 100;
            }
        }

        return featureWeightMap;
    }

    private static int[,] BuildCriticalFeatureWeightMap(int[,] grayMap)
    {
        var absoluteVerticalSobelMap = BuildAbsoluteVerticalSobelMap(grayMap);
        var absoluteHorizontalSobelMap = BuildAbsoluteHorizontalSobelMap(grayMap);
        var sobelMagnitudeMap = BuildSobelMagnitudeMap(absoluteVerticalSobelMap, absoluteHorizontalSobelMap);
        var absoluteLogMap = BuildAbsoluteLogMap(grayMap);
        var textureDensityMap = BuildTextureDensityMap(grayMap);
        var localContrastMap = BuildLocalContrastMap(grayMap);
        var cornerGateMap = BuildCornerGateMap(absoluteVerticalSobelMap, absoluteHorizontalSobelMap);
        var fuzzyMap = BuildFuzzyMap(grayMap);
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var featureWeightMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var criticalFeatureGate =
                    sobelMagnitudeMap[x, y] |
                    absoluteLogMap[x, y] |
                    textureDensityMap[x, y] |
                    localContrastMap[x, y] |
                    cornerGateMap[x, y] |
                    (255 - fuzzyMap[x, y]);
                var criticalFeatureWeight = Math.Min(
                    CriticalFeatureLimit,
                    (int)(criticalFeatureGate / CriticalFeatureScale));

                featureWeightMap[x, y] = criticalFeatureWeight;
            }
        }

        return featureWeightMap;
    }

    private static int[,] BuildGrayMap(Image<Rgba32> sourceImage)
    {
        var grayMap = new int[sourceImage.Width, sourceImage.Height];

        for (var x = 0; x < sourceImage.Width; x += 1)
        {
            for (var y = 0; y < sourceImage.Height; y += 1)
            {
                var pixel = sourceImage[x, y];
                grayMap[x, y] = (int)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
            }
        }

        return grayMap;
    }

    private static int[,] BuildSobelEdgeMap(int[,] grayMap, ImageProcessingRequest processingRequest)
    {
        var sourceMap = processingRequest.SobelKernelSize > 3 ? BuildGaussianMap(grayMap) : grayMap;
        var verticalSobelMap = BuildVerticalSobelMap(sourceMap);
        var horizontalSobelMap = BuildHorizontalSobelMap(sourceMap);
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var sobelEdgeMap = new int[width, height];
        var threshold = processingRequest.EdgeThresholdPercent * 255 / 100;
        var gain = 1.0 + processingRequest.SobelGainPercent / 100.0;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var edgeValue = verticalSobelMap[x, y] | horizontalSobelMap[x, y];

                sobelEdgeMap[x, y] = edgeValue < threshold
                    ? 0
                    : ClampChannel(edgeValue * gain);
            }
        }

        return sobelEdgeMap;
    }

    private static int[,] BuildGaussianBlurMap(int[,] grayMap, ImageProcessingRequest processingRequest)
    {
        var gaussianMap = grayMap;
        var passCount = Math.Max(1, processingRequest.GaussianRadius);

        for (var passIndex = 0; passIndex < passCount; passIndex += 1)
        {
            gaussianMap = BuildGaussianMap(gaussianMap);
        }

        if (processingRequest.GaussianSigmaPercent == 55)
        {
            return gaussianMap;
        }

        var sigmaBlend = processingRequest.GaussianSigmaPercent / 100.0;
        return BlendMaps(grayMap, gaussianMap, sigmaBlend);
    }

    private static Image<Rgba32> BuildLogSharpenedImage(
        Image<Rgba32> sourceImage,
        int[,] grayMap,
        ImageProcessingRequest processingRequest)
    {
        var smoothedMap = grayMap;

        for (var passIndex = 0; passIndex < processingRequest.GaussianRadius; passIndex += 1)
        {
            smoothedMap = BuildGaussianMap(smoothedMap);
        }

        var logMap = BuildLogMap(smoothedMap);
        var outputImage = new Image<Rgba32>(sourceImage.Width, sourceImage.Height);
        var threshold = processingRequest.EdgeThresholdPercent * 255 / 100;
        var sharpenAmount = processingRequest.LogKernelStrengthPercent / 100.0;

        for (var x = 0; x < sourceImage.Width; x += 1)
        {
            for (var y = 0; y < sourceImage.Height; y += 1)
            {
                var sourcePixel = sourceImage[x, y];
                var detailValue = logMap[x, y] < threshold ? 0 : logMap[x, y] * sharpenAmount;

                outputImage[x, y] = new Rgba32(
                    ClampByte(sourcePixel.R + detailValue),
                    ClampByte(sourcePixel.G + detailValue),
                    ClampByte(sourcePixel.B + detailValue),
                    sourcePixel.A);
            }
        }

        return outputImage;
    }

    private static int[,] BuildFuzzyStretchingMap(int[,] grayMap, ImageProcessingRequest processingRequest)
    {
        var fuzzyMap = BuildFuzzyMap(grayMap);
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var stretchedMap = new int[width, height];
        var stretchAmount = 1.0 + processingRequest.FuzzyStrengthPercent / 50.0;
        var midpoint = processingRequest.FuzzyMidpointPercent * 255 / 100.0;
        var cutLimit = processingRequest.FuzzyCutPercent * 255 / 300.0;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var stretchedValue = (fuzzyMap[x, y] - midpoint) * stretchAmount + midpoint;

                if (stretchedValue < cutLimit)
                {
                    stretchedValue = 0;
                }
                else if (stretchedValue > 255 - cutLimit)
                {
                    stretchedValue = 255;
                }

                stretchedMap[x, y] = ClampChannel(stretchedValue);
            }
        }

        return stretchedMap;
    }

    private static int[,] BuildVerticalSobelMap(int[,] grayMap)
    {
        double[,] verticalSobelMask =
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        return Convolve(BuildGaussianMap(grayMap), verticalSobelMask, 1.0);
    }

    private static int[,] BuildHorizontalSobelMap(int[,] grayMap)
    {
        double[,] horizontalSobelMask =
        {
            { 1, 2, 1 },
            { 0, 0, 0 },
            { -1, -2, -1 }
        };

        return Convolve(BuildGaussianMap(grayMap), horizontalSobelMask, 1.0);
    }

    private static int[,] BuildLogMap(int[,] grayMap)
    {
        double[,] logMask =
        {
            { 1, 1, 1 },
            { 1, -8, 1 },
            { 1, 1, 1 }
        };

        return Convolve(BuildGaussianMap(grayMap), logMask, 1.0);
    }

    private static int[,] BuildAbsoluteVerticalSobelMap(int[,] grayMap)
    {
        double[,] verticalSobelMask =
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        return ConvolveAbsolute(BuildGaussianMap(grayMap), verticalSobelMask, 1.0);
    }

    private static int[,] BuildAbsoluteHorizontalSobelMap(int[,] grayMap)
    {
        double[,] horizontalSobelMask =
        {
            { 1, 2, 1 },
            { 0, 0, 0 },
            { -1, -2, -1 }
        };

        return ConvolveAbsolute(BuildGaussianMap(grayMap), horizontalSobelMask, 1.0);
    }

    private static int[,] BuildAbsoluteLogMap(int[,] grayMap)
    {
        double[,] logMask =
        {
            { 1, 1, 1 },
            { 1, -8, 1 },
            { 1, 1, 1 }
        };

        return ConvolveAbsolute(BuildGaussianMap(grayMap), logMask, 1.0);
    }

    private static int[,] BuildSobelMagnitudeMap(int[,] verticalSobelMap, int[,] horizontalSobelMap)
    {
        var width = verticalSobelMap.GetLength(0);
        var height = verticalSobelMap.GetLength(1);
        var sobelMagnitudeMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var verticalValue = verticalSobelMap[x, y];
                var horizontalValue = horizontalSobelMap[x, y];
                sobelMagnitudeMap[x, y] = ClampChannel(Math.Sqrt(verticalValue * verticalValue + horizontalValue * horizontalValue));
            }
        }

        return sobelMagnitudeMap;
    }

    private static int[,] BuildAbsoluteSobelEdgeMap(int[,] grayMap)
    {
        return BuildSobelMagnitudeMap(
            BuildAbsoluteVerticalSobelMap(grayMap),
            BuildAbsoluteHorizontalSobelMap(grayMap));
    }

    private static int[,] BuildTextureDensityMap(int[,] grayMap)
    {
        var localMeanMap = BuildLocalMeanMap(grayMap);
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var textureDifferenceMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                textureDifferenceMap[x, y] = Math.Abs(grayMap[x, y] - localMeanMap[x, y]);
            }
        }

        return NormalizeMap(BuildLocalMeanMap(textureDifferenceMap));
    }

    private static int[,] BuildLocalContrastMap(int[,] grayMap)
    {
        var gaussianMap = BuildGaussianMap(grayMap);
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var localContrastMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                localContrastMap[x, y] = Math.Abs(grayMap[x, y] - gaussianMap[x, y]);
            }
        }

        return NormalizeMap(localContrastMap);
    }

    private static int[,] BuildCornerGateMap(int[,] verticalSobelMap, int[,] horizontalSobelMap)
    {
        var width = verticalSobelMap.GetLength(0);
        var height = verticalSobelMap.GetLength(1);
        var cornerGateMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                cornerGateMap[x, y] = ClampChannel(Math.Sqrt(verticalSobelMap[x, y] * horizontalSobelMap[x, y]));
            }
        }

        return NormalizeMap(cornerGateMap);
    }

    private static int[,] BuildGaussianMap(int[,] grayMap)
    {
        double[,] gaussianMask =
        {
            { 1, 4, 6, 4, 1 },
            { 4, 16, 24, 16, 4 },
            { 6, 24, 36, 24, 6 },
            { 4, 16, 24, 16, 4 },
            { 1, 4, 6, 4, 1 }
        };

        return Convolve(grayMap, gaussianMask, 1.0 / 256.0);
    }

    private static int[,] BuildLocalMeanMap(int[,] grayMap)
    {
        double[,] localMeanMask =
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
        };

        return Convolve(grayMap, localMeanMask, 1.0 / 9.0);
    }

    private static int[,] BuildFuzzyMap(int[,] grayMap)
    {
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var fuzzyMap = new int[width, height];
        var minGray = 255.0;
        var maxGray = 0.0;
        var midGray = 0.0;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var gray = grayMap[x, y];
                midGray += gray;
                minGray = Math.Min(minGray, gray);
                maxGray = Math.Max(maxGray, gray);
            }
        }

        midGray /= width * height;

        var maxDistance = Math.Abs(maxGray - midGray);
        var minDistance = Math.Abs(midGray - minGray);
        var adjustment = GetFuzzyAdjustment(midGray, minDistance, maxDistance);
        var maxIntensity = midGray + adjustment;
        var minIntensity = midGray - adjustment;
        var midIntensity = (maxIntensity + minIntensity) / 2;
        var cut = minIntensity != 0 ? minIntensity / maxIntensity : 0.5;
        var alpha = (midIntensity - minIntensity) * cut + minIntensity;
        var beta = -1 * (maxIntensity - midIntensity) * cut + maxIntensity;
        var intensityRange = beta - alpha;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var fuzzyValue = Math.Abs(intensityRange) < double.Epsilon
                    ? 0
                    : ((grayMap[x, y] - alpha) / intensityRange) * 255;

                fuzzyMap[x, y] = ClampChannel(fuzzyValue);
            }
        }

        return fuzzyMap;
    }

    private static double GetFuzzyAdjustment(double midGray, double minDistance, double maxDistance)
    {
        if (midGray > 128)
        {
            return 255 - midGray;
        }

        if (midGray <= minDistance)
        {
            return minDistance;
        }

        if (midGray >= maxDistance)
        {
            return maxDistance;
        }

        return midGray;
    }

    private static int[,] Convolve(int[,] grayMap, double[,] mask, double factor)
    {
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var maskWidth = mask.GetLength(0);
        var maskHeight = mask.GetLength(1);
        var xPadding = maskWidth / 2;
        var yPadding = maskHeight / 2;
        var outputMap = new int[width, height];

        for (var x = 0; x < width - xPadding * 2; x += 1)
        {
            for (var y = 0; y < height - yPadding * 2; y += 1)
            {
                var sum = 0.0;

                for (var maskX = 0; maskX < maskWidth; maskX += 1)
                {
                    for (var maskY = 0; maskY < maskHeight; maskY += 1)
                    {
                        sum += grayMap[x + maskX, y + maskY] * mask[maskX, maskY];
                    }
                }

                outputMap[x + xPadding, y + yPadding] = ClampChannel(sum * factor);
            }
        }

        return outputMap;
    }

    private static int[,] ConvolveAbsolute(int[,] grayMap, double[,] mask, double factor)
    {
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var maskWidth = mask.GetLength(0);
        var maskHeight = mask.GetLength(1);
        var xPadding = maskWidth / 2;
        var yPadding = maskHeight / 2;
        var outputMap = new int[width, height];

        for (var x = 0; x < width - xPadding * 2; x += 1)
        {
            for (var y = 0; y < height - yPadding * 2; y += 1)
            {
                var sum = 0.0;

                for (var maskX = 0; maskX < maskWidth; maskX += 1)
                {
                    for (var maskY = 0; maskY < maskHeight; maskY += 1)
                    {
                        sum += grayMap[x + maskX, y + maskY] * mask[maskX, maskY];
                    }
                }

                outputMap[x + xPadding, y + yPadding] = ClampChannel(Math.Abs(sum * factor));
            }
        }

        return outputMap;
    }

    private static int[,] NormalizeMap(int[,] sourceMap)
    {
        var width = sourceMap.GetLength(0);
        var height = sourceMap.GetLength(1);
        var normalizedMap = new int[width, height];
        var maxValue = 0;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                maxValue = Math.Max(maxValue, sourceMap[x, y]);
            }
        }

        if (maxValue == 0)
        {
            return normalizedMap;
        }

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                normalizedMap[x, y] = ClampChannel(sourceMap[x, y] * 255.0 / maxValue);
            }
        }

        return normalizedMap;
    }

    private static int CalculateMeanThreshold(int[,] sourceMap)
    {
        var width = sourceMap.GetLength(0);
        var height = sourceMap.GetLength(1);
        var graySum = 0;

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                graySum += sourceMap[x, y];
            }
        }

        return graySum / (width * height);
    }

    private static int FindMaximum(int[,] sourceMap)
    {
        var maximum = 0;

        for (var x = 0; x < sourceMap.GetLength(0); x += 1)
        {
            for (var y = 0; y < sourceMap.GetLength(1); y += 1)
            {
                maximum = Math.Max(maximum, sourceMap[x, y]);
            }
        }

        return maximum;
    }

    private static int FindMinimum(int[,] sourceMap)
    {
        var minimum = 255;

        for (var x = 0; x < sourceMap.GetLength(0); x += 1)
        {
            for (var y = 0; y < sourceMap.GetLength(1); y += 1)
            {
                minimum = Math.Min(minimum, sourceMap[x, y]);
            }
        }

        return minimum;
    }

    private static int[,] BlendMaps(int[,] sourceMap, int[,] processedMap, double blendAmount)
    {
        var width = sourceMap.GetLength(0);
        var height = sourceMap.GetLength(1);
        var blendedMap = new int[width, height];

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                blendedMap[x, y] = ClampChannel(sourceMap[x, y] * (1 - blendAmount) + processedMap[x, y] * blendAmount);
            }
        }

        return blendedMap;
    }

    private static Image<Rgba32> BuildGrayImage(int[,] grayMap)
    {
        var width = grayMap.GetLength(0);
        var height = grayMap.GetLength(1);
        var outputImage = new Image<Rgba32>(width, height);

        for (var x = 0; x < width; x += 1)
        {
            for (var y = 0; y < height; y += 1)
            {
                var gray = ClampByte(grayMap[x, y]);
                outputImage[x, y] = new Rgba32(gray, gray, gray);
            }
        }

        return outputImage;
    }

    private static byte[] ConvertMapToRowMajorBytes(int[,] sourceMap, int width, int height)
    {
        var outputBytes = new byte[width * height];

        for (var y = 0; y < height; y += 1)
        {
            for (var x = 0; x < width; x += 1)
            {
                outputBytes[y * width + x] = ClampByte(sourceMap[x, y]);
            }
        }

        return outputBytes;
    }

    private static int ClampChannel(double value)
    {
        return Math.Clamp((int)value, 0, 255);
    }

    private static byte ClampByte(double value)
    {
        return (byte)Math.Clamp((int)value, 0, 255);
    }
}
