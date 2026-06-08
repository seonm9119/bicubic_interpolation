using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Memory;

namespace BicubicInterpolation.Api;

public static class ApiCacheTools
{
    private const string PublicStaticCacheControl = "public, max-age=604800, immutable";

    public static void SetPublicCacheHeaders(HttpResponse response)
    {
        response.Headers["Cache-Control"] = PublicStaticCacheControl;
    }

    public static MemoryCacheEntryOptions CreateComputeCacheOptions()
    {
        return new MemoryCacheEntryOptions
        {
            SlidingExpiration = TimeSpan.FromHours(6),
            AbsoluteExpirationRelativeToNow = TimeSpan.FromDays(1)
        };
    }

    public static void ConfigureStaticMemoryCacheEntry(ICacheEntry cacheEntry)
    {
        cacheEntry.SlidingExpiration = TimeSpan.FromHours(12);
        cacheEntry.AbsoluteExpirationRelativeToNow = TimeSpan.FromDays(7);
    }
}
