using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BicubicInterpolation.Api;

public sealed record SuperResolutionImageData(int Width, int Height, byte[] PngBytes)
{
    public string DataUrl => $"data:image/png;base64,{Convert.ToBase64String(PngBytes)}";
}

public sealed record SuperResolutionMetricResult(
    double? PsnrDb,
    double Ssim,
    double Mse,
    double Rmse,
    double Mae);

public static class SuperResolutionBenchmarkTools
{
    public static SuperResolutionImageData CreateLowResolutionInput(byte[] highResolutionImageBytes, int scaleFactor)
    {
        using var highResolutionImage = Image.Load<Rgba32>(highResolutionImageBytes);
        var lowResolutionWidth = Math.Max(1, highResolutionImage.Width / scaleFactor);
        var lowResolutionHeight = Math.Max(1, highResolutionImage.Height / scaleFactor);

        using var lowResolutionImage = highResolutionImage.Clone(processingContext =>
            processingContext.Resize(new ResizeOptions
            {
                Size = new Size(lowResolutionWidth, lowResolutionHeight),
                Sampler = KnownResamplers.Bicubic
            }));

        return new SuperResolutionImageData(lowResolutionWidth, lowResolutionHeight, EncodePng(lowResolutionImage));
    }

    public static SuperResolutionMetricResult CalculateMetrics(byte[] referenceImageBytes, byte[] reconstructedImageBytes)
    {
        using var referenceImage = Image.Load<Rgba32>(referenceImageBytes);
        using var reconstructedImage = Image.Load<Rgba32>(reconstructedImageBytes);

        if (referenceImage.Width != reconstructedImage.Width || referenceImage.Height != reconstructedImage.Height)
        {
            throw new InvalidOperationException("Metric 계산에는 GT와 복원 이미지의 해상도가 같아야 합니다.");
        }

        var pixelCount = referenceImage.Width * referenceImage.Height;
        var channelCount = pixelCount * 3.0;
        var squaredErrorTotal = 0.0;
        var absoluteErrorTotal = 0.0;
        var referenceLuminanceTotal = 0.0;
        var reconstructedLuminanceTotal = 0.0;
        var referenceLuminanceSquareTotal = 0.0;
        var reconstructedLuminanceSquareTotal = 0.0;
        var luminanceProductTotal = 0.0;

        for (var y = 0; y < referenceImage.Height; y += 1)
        {
            for (var x = 0; x < referenceImage.Width; x += 1)
            {
                var referencePixel = referenceImage[x, y];
                var reconstructedPixel = reconstructedImage[x, y];

                AccumulateChannelError(referencePixel.R, reconstructedPixel.R, ref squaredErrorTotal, ref absoluteErrorTotal);
                AccumulateChannelError(referencePixel.G, reconstructedPixel.G, ref squaredErrorTotal, ref absoluteErrorTotal);
                AccumulateChannelError(referencePixel.B, reconstructedPixel.B, ref squaredErrorTotal, ref absoluteErrorTotal);

                var referenceLuminance = CalculateLuminance(referencePixel);
                var reconstructedLuminance = CalculateLuminance(reconstructedPixel);

                referenceLuminanceTotal += referenceLuminance;
                reconstructedLuminanceTotal += reconstructedLuminance;
                referenceLuminanceSquareTotal += referenceLuminance * referenceLuminance;
                reconstructedLuminanceSquareTotal += reconstructedLuminance * reconstructedLuminance;
                luminanceProductTotal += referenceLuminance * reconstructedLuminance;
            }
        }

        var mse = squaredErrorTotal / channelCount;
        var rmse = Math.Sqrt(mse);
        var mae = absoluteErrorTotal / channelCount;
        var psnrDb = mse <= 0 ? (double?)null : 10 * Math.Log10(255 * 255 / mse);
        var ssim = CalculateGlobalSsim(
            pixelCount,
            referenceLuminanceTotal,
            reconstructedLuminanceTotal,
            referenceLuminanceSquareTotal,
            reconstructedLuminanceSquareTotal,
            luminanceProductTotal);

        return new SuperResolutionMetricResult(psnrDb, ssim, mse, rmse, mae);
    }

    public static byte[] EncodePng(Image<Rgba32> image)
    {
        using var outputStream = new MemoryStream();
        image.SaveAsPng(outputStream);
        return outputStream.ToArray();
    }

    private static void AccumulateChannelError(byte referenceChannel, byte reconstructedChannel, ref double squaredErrorTotal, ref double absoluteErrorTotal)
    {
        var channelError = referenceChannel - reconstructedChannel;
        squaredErrorTotal += channelError * channelError;
        absoluteErrorTotal += Math.Abs(channelError);
    }

    private static double CalculateLuminance(Rgba32 pixel)
    {
        return 0.2126 * pixel.R + 0.7152 * pixel.G + 0.0722 * pixel.B;
    }

    private static double CalculateGlobalSsim(
        int pixelCount,
        double referenceLuminanceTotal,
        double reconstructedLuminanceTotal,
        double referenceLuminanceSquareTotal,
        double reconstructedLuminanceSquareTotal,
        double luminanceProductTotal)
    {
        var referenceMean = referenceLuminanceTotal / pixelCount;
        var reconstructedMean = reconstructedLuminanceTotal / pixelCount;
        var referenceVariance = referenceLuminanceSquareTotal / pixelCount - referenceMean * referenceMean;
        var reconstructedVariance = reconstructedLuminanceSquareTotal / pixelCount - reconstructedMean * reconstructedMean;
        var covariance = luminanceProductTotal / pixelCount - referenceMean * reconstructedMean;
        var c1 = Math.Pow(0.01 * 255, 2);
        var c2 = Math.Pow(0.03 * 255, 2);
        var numerator = (2 * referenceMean * reconstructedMean + c1) * (2 * covariance + c2);
        var denominator = (referenceMean * referenceMean + reconstructedMean * reconstructedMean + c1) *
            (referenceVariance + reconstructedVariance + c2);

        return denominator == 0 ? 1 : numerator / denominator;
    }
}
