namespace BicubicInterpolation.Api;

public static class BicubicResizeMath
{
    private const int MaxOutputMegapixels = 50;
    private const int MaxOutputPixels = MaxOutputMegapixels * 1_000_000;

    public static void ValidateOutputSize(int width, int height)
    {
        if ((long)width * height > MaxOutputPixels)
        {
            throw new InvalidOperationException($"출력 이미지는 최대 {MaxOutputMegapixels}MP까지 처리합니다. 더 작은 이미지나 낮은 scale factor를 사용해 주세요.");
        }
    }

    public static double GetSourcePosition(int targetPosition, int targetLength, int sourceLength)
    {
        if (targetLength <= 1 || sourceLength <= 1)
        {
            return 0;
        }

        return (double)(sourceLength - 1) * targetPosition / (targetLength - 1);
    }

    public static double CubicInterpolate(double value1, double value2, double value3, double value4, double distance)
    {
        var term1 = 2 * value2;
        var term2 = -value1 + value3;
        var term3 = 2 * value1 - 5 * value2 + 4 * value3 - value4;
        var term4 = -value1 + 3 * value2 - 3 * value3 + value4;

        return (term1 + distance * (term2 + distance * (term3 + distance * term4))) / 2;
    }

    public static int ClampIndex(int index, int length)
    {
        return Math.Clamp(index, 0, length - 1);
    }

    public static byte ClampChannel(double value)
    {
        return (byte)Math.Clamp((int)(value + 0.5), 0, 255);
    }
}
