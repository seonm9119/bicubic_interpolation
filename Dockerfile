FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

WORKDIR /src

COPY api/BicubicInterpolation.Api.csproj api/
COPY image_processing/ImageProcessing.Api.csproj image_processing/
RUN dotnet restore api/BicubicInterpolation.Api.csproj

COPY api/ api/
COPY image_processing/ image_processing/
RUN dotnet publish api/BicubicInterpolation.Api.csproj -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0

WORKDIR /app

ENV ASPNETCORE_URLS=http://+:8080
ENV IMAGE_PROCESSING_SAMPLE_DIR=/app/image_processing_assets
ENV SRGAN_API_BASE_URL=http://sr-benchmark:8080

EXPOSE 8080

COPY --from=build /app/publish ./
COPY image_processing/assets/ ./image_processing_assets/

ENTRYPOINT ["dotnet", "BicubicInterpolation.Api.dll"]
