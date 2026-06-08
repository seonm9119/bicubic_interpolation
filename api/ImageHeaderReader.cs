namespace BicubicInterpolation.Api;

public sealed record ImageDimensions(int? Width, int? Height, string Format);

public static class ImageHeaderReader
{
    public static ImageDimensions Read(byte[] imageBytes)
    {
        if (IsPng(imageBytes))
        {
            return new ImageDimensions(ReadBigEndianInt32(imageBytes, 16), ReadBigEndianInt32(imageBytes, 20), "PNG");
        }

        if (IsJpeg(imageBytes))
        {
            return ReadJpegDimensions(imageBytes);
        }

        if (IsBmp(imageBytes))
        {
            return new ImageDimensions(ReadLittleEndianInt32(imageBytes, 18), Math.Abs(ReadLittleEndianInt32(imageBytes, 22)), "BMP");
        }

        if (IsWebp(imageBytes))
        {
            return ReadWebpDimensions(imageBytes);
        }

        return new ImageDimensions(null, null, "Unknown");
    }

    private static bool IsPng(byte[] imageBytes)
    {
        return imageBytes.Length >= 24 &&
            imageBytes[0] == 0x89 &&
            imageBytes[1] == 0x50 &&
            imageBytes[2] == 0x4E &&
            imageBytes[3] == 0x47;
    }

    private static bool IsJpeg(byte[] imageBytes)
    {
        return imageBytes.Length >= 4 && imageBytes[0] == 0xFF && imageBytes[1] == 0xD8;
    }

    private static bool IsBmp(byte[] imageBytes)
    {
        return imageBytes.Length >= 26 && imageBytes[0] == 0x42 && imageBytes[1] == 0x4D;
    }

    private static bool IsWebp(byte[] imageBytes)
    {
        return imageBytes.Length >= 30 &&
            imageBytes[0] == 0x52 &&
            imageBytes[1] == 0x49 &&
            imageBytes[2] == 0x46 &&
            imageBytes[3] == 0x46 &&
            imageBytes[8] == 0x57 &&
            imageBytes[9] == 0x45 &&
            imageBytes[10] == 0x42 &&
            imageBytes[11] == 0x50;
    }

    private static ImageDimensions ReadJpegDimensions(byte[] imageBytes)
    {
        var offset = 2;

        while (offset + 9 < imageBytes.Length)
        {
            if (imageBytes[offset] != 0xFF)
            {
                offset += 1;
                continue;
            }

            var marker = imageBytes[offset + 1];
            var segmentLength = ReadBigEndianInt16(imageBytes, offset + 2);

            if (IsStartOfFrameMarker(marker))
            {
                var height = ReadBigEndianInt16(imageBytes, offset + 5);
                var width = ReadBigEndianInt16(imageBytes, offset + 7);

                return new ImageDimensions(width, height, "JPEG");
            }

            if (segmentLength <= 0)
            {
                break;
            }

            offset += segmentLength + 2;
        }

        return new ImageDimensions(null, null, "JPEG");
    }

    private static ImageDimensions ReadWebpDimensions(byte[] imageBytes)
    {
        if (imageBytes[12] == 0x56 && imageBytes[13] == 0x50 && imageBytes[14] == 0x38 && imageBytes[15] == 0x58)
        {
            var width = 1 + imageBytes[24] + (imageBytes[25] << 8) + (imageBytes[26] << 16);
            var height = 1 + imageBytes[27] + (imageBytes[28] << 8) + (imageBytes[29] << 16);

            return new ImageDimensions(width, height, "WEBP");
        }

        return new ImageDimensions(null, null, "WEBP");
    }

    private static bool IsStartOfFrameMarker(byte marker)
    {
        return marker is 0xC0 or 0xC1 or 0xC2 or 0xC3 or 0xC5 or 0xC6 or 0xC7 or 0xC9 or 0xCA or 0xCB or 0xCD or 0xCE or 0xCF;
    }

    private static int ReadBigEndianInt32(byte[] bytes, int offset)
    {
        return (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3];
    }

    private static int ReadBigEndianInt16(byte[] bytes, int offset)
    {
        return (bytes[offset] << 8) | bytes[offset + 1];
    }

    private static int ReadLittleEndianInt32(byte[] bytes, int offset)
    {
        return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24);
    }
}
