using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BicubicInterpolation.Api;

public sealed record BicubicImageResult(int Width, int Height, byte[] PngBytes)
{
    public string DataUrl => $"data:image/png;base64,{Convert.ToBase64String(PngBytes)}";
}

public static class ClassicBicubicInterpolator
{
    public static BicubicImageResult ResizeToPngDataUrl(byte[] imageBytes, int scaleFactor)
    {
        using var sourceImage = Image.Load<Rgba32>(imageBytes);
        var targetWidth = sourceImage.Width * scaleFactor;
        var targetHeight = sourceImage.Height * scaleFactor;

        BicubicResizeMath.ValidateOutputSize(targetWidth, targetHeight);

        using var targetImage = new Image<Rgba32>(targetWidth, targetHeight);

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

                targetImage[targetX, targetY] = BicubicPixelInterpolator.InterpolatePixel(
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
            }
        }

        using var outputStream = new MemoryStream();
        targetImage.SaveAsPng(outputStream);

        return new BicubicImageResult(targetWidth, targetHeight, outputStream.ToArray());
    }
}
