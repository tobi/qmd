using System.Text.Json;

namespace Qmd.CSharp.Sidecar;

public static class Program
{
    public static async Task<int> Main()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        await using var input = Console.OpenStandardInput();
        var request = await JsonSerializer.DeserializeAsync<AnalysisRequest>(input, options);

        if (request is null)
        {
            await Console.Error.WriteLineAsync("Request payload was null.");
            return 1;
        }

        var service = new CSharpAnalysisService();
        var response = service.Analyze(request);

        await using var output = Console.OpenStandardOutput();
        await JsonSerializer.SerializeAsync(output, response, options);
        return 0;
    }
}
