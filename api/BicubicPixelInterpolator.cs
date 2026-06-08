using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BicubicInterpolation.Api;

public readonly record struct BicubicInterpolatedPixel(double Red, double Green, double Blue, double Alpha);

public static class BicubicPixelInterpolator
{
    public static Rgba32 InterpolatePixel(
        Image<Rgba32> sourceImage,
        int x1,
        int x2,
        int x3,
        int x4,
        int y1,
        int y2,
        int y3,
        int y4,
        double xRatio,
        double yRatio)
    {
        var interpolatedPixel = InterpolatePixelValues(
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

        return new Rgba32(
            BicubicResizeMath.ClampChannel(interpolatedPixel.Red),
            BicubicResizeMath.ClampChannel(interpolatedPixel.Green),
            BicubicResizeMath.ClampChannel(interpolatedPixel.Blue),
            BicubicResizeMath.ClampChannel(interpolatedPixel.Alpha));
    }

    public static BicubicInterpolatedPixel InterpolatePixelValues(
        Image<Rgba32> sourceImage,
        int x1,
        int x2,
        int x3,
        int x4,
        int y1,
        int y2,
        int y3,
        int y4,
        double xRatio,
        double yRatio)
    {
        var row1 = InterpolateRow(sourceImage[x1, y1], sourceImage[x2, y1], sourceImage[x3, y1], sourceImage[x4, y1], xRatio);
        var row2 = InterpolateRow(sourceImage[x1, y2], sourceImage[x2, y2], sourceImage[x3, y2], sourceImage[x4, y2], xRatio);
        var row3 = InterpolateRow(sourceImage[x1, y3], sourceImage[x2, y3], sourceImage[x3, y3], sourceImage[x4, y3], xRatio);
        var row4 = InterpolateRow(sourceImage[x1, y4], sourceImage[x2, y4], sourceImage[x3, y4], sourceImage[x4, y4], xRatio);

        return new BicubicInterpolatedPixel(
            BicubicResizeMath.CubicInterpolate(row1.Red, row2.Red, row3.Red, row4.Red, yRatio),
            BicubicResizeMath.CubicInterpolate(row1.Green, row2.Green, row3.Green, row4.Green, yRatio),
            BicubicResizeMath.CubicInterpolate(row1.Blue, row2.Blue, row3.Blue, row4.Blue, yRatio),
            BicubicResizeMath.CubicInterpolate(row1.Alpha, row2.Alpha, row3.Alpha, row4.Alpha, yRatio));
    }

    private static BicubicInterpolatedPixel InterpolateRow(Rgba32 pixel1, Rgba32 pixel2, Rgba32 pixel3, Rgba32 pixel4, double ratio)
    {
        return new BicubicInterpolatedPixel(
            BicubicResizeMath.CubicInterpolate(pixel1.R, pixel2.R, pixel3.R, pixel4.R, ratio),
            BicubicResizeMath.CubicInterpolate(pixel1.G, pixel2.G, pixel3.G, pixel4.G, ratio),
            BicubicResizeMath.CubicInterpolate(pixel1.B, pixel2.B, pixel3.B, pixel4.B, ratio),
            BicubicResizeMath.CubicInterpolate(pixel1.A, pixel2.A, pixel3.A, pixel4.A, ratio));
    }
}
