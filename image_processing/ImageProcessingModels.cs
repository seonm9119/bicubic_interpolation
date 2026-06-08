namespace ImageProcessing.Api;

public sealed record ImageProcessingRequest(
    string Operation,
    int EdgeThresholdPercent,
    int SobelGainPercent,
    int SobelKernelSize,
    int GaussianRadius,
    int GaussianSigmaPercent,
    int LogKernelStrengthPercent,
    int FuzzyStrengthPercent,
    int FuzzyCutPercent,
    int FuzzyMidpointPercent);

public sealed record ProcessedImageResult(int Width, int Height, byte[] PngBytes, string OperationName);

public sealed record FeatureMapResult(int Width, int Height, string FeatureMapBase64, string Encoding);
